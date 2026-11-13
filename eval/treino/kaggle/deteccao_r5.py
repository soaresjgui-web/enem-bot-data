#!/usr/bin/env python3
"""Etapa 0 da quinta execução (seção 14.3 de resultado_modelo_treinado.md): a caixa da refeição em cada foto de treino.

Um kernel curto, antes da execução principal. Para cada foto (Nutrition5k: só os pratos de treino_v2; bases externas:
todas as fotos que a quarta execução usava, inclusive as das partes de registro), grava a caixa da refeição, a janela do
guia (4:3, centrada na caixa, altura = maior lado da caixa ÷ 0,80) e o método:
  owlv2      OWLv2 base (google/owlv2-base-patch16-ensemble, Apache 2.0) em meia precisão, imagem a 960 px, sete
             consultas; caixas com nota >= 0,15 e área >= 2% da foto, NMS a 0,5; a caixa da refeição é a UNIÃO delas;
  elipse     sem caixa: a maior elipse ajustada às bordas (OpenCV: Canny + fitEllipse), área de 10% a 95% da foto e
             razão de eixos >= 0,5;
  sem_prato  sem elipse: a foto inteira (a janela é o menor quadro 4:3 que contém a foto).
No Nutrition5k sai também a concordância com a caixa da profundidade de 8.7 (zen_prato.py): a gravada naquela época
(listas/caixas_profundidade_n5k.csv, 1.874 dos 2.281 pratos) e a refeita aqui, com a mesma conta, para todos.
Nada aqui lê rótulo nenhum. As fotos e as folhas de contato ficam só na saída do Kaggle (e no scratchpad).

Uso (pelo notebook nutrelle_deteccao_r5.ipynb, ou direto):
  python3 deteccao_r5.py --saida <pasta> [--orcamento-h 1.5] [--smoke N --externo-imagens <pasta> --cpu] [--sem-detector]
Saída (raiz da pasta): caixas_r5_<base>.csv, folha_r5_<base>_<k>.jpg e folha_r5_<base>.csv (40 fotos sorteadas, semente
0, com a caixa e a janela desenhadas e o recorte ao lado), resumo_r5.json, manifesto_r5.json, eventos_r5.jsonl, log.
"""
import argparse, csv, hashlib, json, math, os, shutil, subprocess, sys, tempfile, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import treino_gpu as G  # noqa: E402  (listas, downloads e conferências da quarta execução)

FORMATO = "nutrelle-deteccao/5"
DETECTOR = "google/owlv2-base-patch16-ensemble"
REVISAO = "cfd3195ba4ea9592eec887ded089f4c08eff231d"        # commit do Hugging Face conferido em 13/11/2026
SHA_PESOS = "e1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7"      # model.safetensors dessa revisão
LICENCA_DETECTOR = "Apache 2.0 (pesos públicos do OWLv2; só prepara os dados de treino, não faz parte do modelo servido)"
CONSULTAS = ["a plate of food", "a bowl of food", "a food tray", "a lunch box", "a food container", "a plate", "a bowl"]
LADO = 960
NOTA_MIN, AREA_MIN, NMS_IOU = 0.15, 0.02, 0.5
ELIPSE_AREA, ELIPSE_RAZAO = (0.10, 0.95), 0.5
FRAC_GUIA, ASPECTO = 0.80, 4 / 3          # o guia do app: círculo de 80% do lado menor do quadro; entrada 4:3
SAIDA_RECORTE = (640, 480)                # o recorte vai para o mesmo formato do cache das execuções anteriores
MEDIA_OWL, DESVIO_OWL = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
N_FOLHA, POR_FOLHA = 40, 20
DOMINIOS = ["n5k"] + list(G.DOMINIOS4)
SO_PRATO = (0, 1, 5, 6)                   # consultas de prato/tigela (diagnóstico: a união sem bandeja, marmita e recipiente)
MAX_BRUTAS = 40
COLS = ["id", "dominio", "arquivo", "sha256", "largura", "altura", "girada", "metodo", "x0", "y0", "x1", "y1", "nota", "n_caixas",
        "janela_cx", "janela_cy", "janela_alt", "janela_larg", "versao_detector", "iou_uniao_so_prato", "caixa_so_prato", "caixa_elipse", "caixas_brutas"]


# ------------------------------------------------------------------ contas puras (testadas no --autoteste)
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return float(ix * iy / u) if u > 0 else 0.0


def nms(caixas, notas, limiar=NMS_IOU):
    """NMS sem classe: índices mantidos, da maior nota para a menor."""
    ordem, fica = list(np.argsort(-np.asarray(notas), kind="stable")), []
    while ordem:
        i = ordem.pop(0); fica.append(i)
        ordem = [j for j in ordem if iou(caixas[i], caixas[j]) <= limiar]
    return fica


def caixa_refeicao(caixas, notas, w, h):
    """Caixas do detector (em pixels da foto) -> (caixa da refeição ou None, maior nota, nº de caixas usadas):
    nota >= 0,15, área >= 2% da foto (depois de cortar na foto), NMS a 0,5, e a UNIÃO do que sobra."""
    c = np.clip(np.asarray(caixas, np.float64).reshape(-1, 4), 0, [w, h, w, h])
    n = np.asarray(notas, np.float64).reshape(-1)
    area = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0)
    ok = np.where((n >= NOTA_MIN) & (area >= AREA_MIN * w * h))[0]
    if not len(ok):
        return None, float(n.max()) if len(n) else 0.0, 0
    fica = ok[nms(c[ok], n[ok])]
    u = (float(c[fica, 0].min()), float(c[fica, 1].min()), float(c[fica, 2].max()), float(c[fica, 3].max()))
    return u, float(n[fica].max()), int(len(fica))


def brutas_de(caixas, notas, consultas, w, h):
    """As caixas do detector que passam na nota e na área, ANTES do NMS e da união (no máximo 40, da maior nota para a
    menor): [x0, y0, x1, y1, nota, consulta]. Gravadas para que outra regra possa ser refeita na CPU sem GPU."""
    c = np.clip(np.asarray(caixas, np.float64).reshape(-1, 4), 0, [w, h, w, h]); n = np.asarray(notas, np.float64).reshape(-1)
    area = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0)
    ok = [i for i in np.argsort(-n, kind="stable") if n[i] >= NOTA_MIN and area[i] >= AREA_MIN * w * h][:MAX_BRUTAS]
    return [[round(float(v), 1) for v in c[i]] + [round(float(n[i]), 4), int(consultas[i])] for i in ok]


def refaz_linha(r, consultas):
    """Uma linha de caixas_r5 refeita com outra regra, sem imagem nem GPU: a união das caixas brutas só das consultas
    dadas (com o mesmo NMS e os mesmos limiares); sem caixa, a elipse gravada; sem elipse, sem_prato."""
    w, h = float(r["largura"]), float(r["altura"])
    b = [x for x in json.loads(r["caixas_brutas"] or "[]") if int(x[5]) in consultas]
    caixa, nota, nc = caixa_refeicao([x[:4] for x in b], [x[4] for x in b], w, h) if b else (None, 0.0, 0)
    metodo = "owlv2"
    if caixa is None:
        caixa, metodo = (tuple(json.loads(r["caixa_elipse"])), "elipse") if r.get("caixa_elipse") else (None, "sem_prato")
    jan = janela(caixa, w, h)
    n = dict(r, metodo=metodo, nota=round(nota, 4), n_caixas=nc, janela_cx=round(jan[0], 1), janela_cy=round(jan[1], 1), janela_alt=round(jan[2], 1),
             janela_larg=round(jan[3], 1), versao_detector=r["versao_detector"] + f" | regra refeita: consultas {sorted(consultas)}")
    n.update({c: "" for c in ("x0", "y0", "x1", "y1")} if caixa is None else dict(zip(("x0", "y0", "x1", "y1"), [round(v, 1) for v in caixa])))
    return n


def janela(caixa, w, h):
    """Janela do guia (14.3): 4:3, centrada na caixa, altura = maior lado da caixa ÷ 0,80. Sem caixa: o menor quadro 4:3
    que contém a foto inteira (como o encaixe das fotos não 4:3 da quarta execução). -> (cx, cy, altura, largura)."""
    if caixa is None:
        alt = max(h, w / ASPECTO)
        return w / 2, h / 2, alt, alt * ASPECTO
    x0, y0, x1, y1 = caixa
    alt = max(x1 - x0, y1 - y0) / FRAC_GUIA
    return (x0 + x1) / 2, (y0 + y1) / 2, alt, alt * ASPECTO


def cor_borda(a):
    """Cor mediana da moldura de 3% da foto (a mesma conta do cache das execuções anteriores), em 0–255."""
    k = max(2, int(0.03 * min(a.shape[:2])))
    return np.median(np.concatenate([a[:k].reshape(-1, 3), a[-k:].reshape(-1, 3), a[:, :k].reshape(-1, 3), a[:, -k:].reshape(-1, 3)]), 0)


def recorta(a, jan, saida=SAIDA_RECORTE, cor=None):
    """Recorte da janela (cx, cy, alt, larg) numa foto uint8 [h, w, 3], com a cor mediana da borda onde a janela sai da
    foto; redimensionado (bicúbico) para `saida`. Nunca estica: a janela já é 4:3."""
    from PIL import Image
    cx, cy, alt, larg = jan
    h, w = a.shape[:2]
    x0, y0 = int(math.floor(cx - larg / 2)), int(math.floor(cy - alt / 2))
    x1, y1 = int(math.ceil(cx + larg / 2)), int(math.ceil(cy + alt / 2))
    tela = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
    tela[:] = (cor_borda(a) if cor is None else cor).astype(np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        tela[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return np.asarray(Image.fromarray(tela).resize(saida, Image.BICUBIC))


def elipse(a):
    """Reserva clássica (14.3, item 2): a maior elipse ajustada às bordas, com área de 10% a 95% da foto e razão de eixos
    >= 0,5. -> caixa (x0, y0, x1, y1) da elipse, cortada na foto, ou None."""
    import cv2
    H0, W0 = a.shape[:2]
    esc = min(1.0, 640 / max(H0, W0))            # bordas numa versão de até 640 px (a caixa volta para a escala da foto)
    if esc < 1:
        a = cv2.resize(a, (round(W0 * esc), round(H0 * esc)), interpolation=cv2.INTER_AREA)
    h, w = a.shape[:2]
    cinza = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    bordas = cv2.dilate(cv2.Canny(cinza, 50, 150), np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    melhor, maior = None, 0.0
    for c in contornos:
        if len(c) < 20:
            continue
        (cx, cy), (e1, e2), ang = cv2.fitEllipse(c)
        if min(e1, e2) <= 0 or min(e1, e2) / max(e1, e2) < ELIPSE_RAZAO:
            continue
        area = math.pi * e1 * e2 / 4
        if not (ELIPSE_AREA[0] * w * h <= area <= ELIPSE_AREA[1] * w * h) or area <= maior:
            continue
        t = math.radians(ang)       # caixa da elipse girada
        dx = math.sqrt((e1 / 2 * math.cos(t)) ** 2 + (e2 / 2 * math.sin(t)) ** 2)
        dy = math.sqrt((e1 / 2 * math.sin(t)) ** 2 + (e2 / 2 * math.cos(t)) ** 2)
        melhor, maior = (max(0.0, cx - dx), max(0.0, cy - dy), min(float(w), cx + dx), min(float(h), cy + dy)), area
    return None if melhor is None else tuple(v / esc for v in melhor)


def mapa_altura(caminho, rng):
    """Mapa de alturas 120×160 (mm acima da bancada) de um depth_raw.png do Nutrition5k: a conta de
    features_profundidade.atributos (plano de apoio por RANSAC na faixa da borda), só a parte do mapa."""
    from PIL import Image
    d = np.array(Image.open(caminho)).astype(np.float32)
    val = d > 0
    if np.median(d[val]) < 1000:
        d *= 10
    z = d / 10.0
    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W]
    borda = val & ((xx < 0.12 * W) | (xx > 0.88 * W) | (yy < 0.12 * H) | (yy > 0.88 * H))
    borda &= z >= np.median(z[borda]) - 10
    xyz = np.c_[xx[borda], yy[borda], z[borda]].astype(np.float64)
    sub = xyz[rng.choice(len(xyz), min(20000, len(xyz)), replace=False)]
    best, nbest = None, -1
    for _ in range(200):
        p = sub[rng.choice(len(sub), 3, replace=False)]
        nv = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(nv) < 1e-6:
            continue
        nv /= np.linalg.norm(nv)
        if abs(nv[2]) < 0.9:
            continue
        k = (np.abs((sub - p[0]) @ nv) < 4.0).sum()
        if k > nbest:
            best, nbest = (nv, p[0]), k
    nv, p0 = best
    inl = np.abs((sub - p0) @ nv) < 4.0
    A = np.c_[sub[inl, 0], sub[inl, 1], np.ones(inl.sum())]
    (a_, b_, c_), *_ = np.linalg.lstsq(A, sub[inl, 2], rcond=None)
    h = np.clip(np.where(val, a_ * xx + b_ * yy + c_ - z, 0.0), 0, 200)
    return np.array(Image.fromarray(h.astype(np.float32)).resize((160, 120), Image.BILINEAR)).astype(np.float32)


def caixa_profundidade(mapa, limiar=12.0):
    """A caixa do prato de 8.7 (zen_prato.caixa): maior região conexa acima de 12 mm, preferindo a do centro; em pixels
    da foto 640×480. None se não houver, ou se o menor lado tiver menos de 60 px."""
    from scipy import ndimage as ndi
    m = ndi.binary_opening(mapa > limiar, iterations=2)
    rot, n = ndi.label(m)
    if n == 0:
        return None
    tam = ndi.sum(m, rot, range(1, n + 1))
    yy, xx = np.mgrid[0:mapa.shape[0], 0:mapa.shape[1]]
    melhor, score = None, -1
    for k in range(1, n + 1):
        r = rot == k
        cx, cy = xx[r].mean() / mapa.shape[1], yy[r].mean() / mapa.shape[0]
        s = tam[k - 1] * (1 - 1.5 * max(abs(cx - 0.5), abs(cy - 0.5)))
        if s > score:
            melhor, score = r, s
    ys, xs = np.where(melhor)
    f = 640 / mapa.shape[1]
    b = (xs.min() * f, ys.min() * f, (xs.max() + 1) * f, (ys.max() + 1) * f)
    return None if min(b[2] - b[0], b[3] - b[1]) < 60 else b


def abre_canonica(caminho):
    """Foto como o modelo a vê: orientação do EXIF aplicada e retrato girado para paisagem (como na quarta execução).
    -> (uint8 [h, w, 3], girada)."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(caminho)).convert("RGB")
    girada = im.size[1] > im.size[0]
    if girada:
        im = im.transpose(Image.ROTATE_90)
    return np.asarray(im), girada


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ o detector
class Detector:
    """OWLv2 base em meia precisão (GPU). O pré-processamento é feito aqui, no dispositivo, igual ao do
    Owlv2ImageProcessor (completa a foto até um quadrado com cinza 0,5 embaixo e à direita, 960 px, normalização do
    CLIP): o processador do transformers é lento em CPU e muda de interface entre versões. As caixas saem em pixels da foto."""

    def __init__(self, dev, pasta_pesos):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Owlv2ForObjectDetection, AutoTokenizer
        self.t, self.dev = torch, dev
        local = snapshot_download(DETECTOR, revision=REVISAO, cache_dir=pasta_pesos)
        sp = os.path.join(local, "model.safetensors")
        self.sha_pesos = os.path.basename(os.path.realpath(sp))
        if self.sha_pesos != SHA_PESOS:         # o blob do cache se chama pelo sha256; se não, calcula
            self.sha_pesos = sha256(sp)
        assert self.sha_pesos == SHA_PESOS, f"pesos do detector diferentes dos conferidos ({self.sha_pesos})"
        meio = dev.type == "cuda"
        self.m = Owlv2ForObjectDetection.from_pretrained(local, torch_dtype=torch.float16 if meio else torch.float32).to(dev).eval()
        tok = AutoTokenizer.from_pretrained(local)
        e = tok(CONSULTAS, padding="max_length", max_length=16, return_tensors="pt")
        self.ids, self.mascara = e["input_ids"].to(dev), e["attention_mask"].to(dev)
        self.dtype = torch.float16 if meio else torch.float32
        self.media = torch.tensor(MEDIA_OWL, device=dev).view(1, 3, 1, 1)
        self.desvio = torch.tensor(DESVIO_OWL, device=dev).view(1, 3, 1, 1)
        import transformers
        self.versao = f"{DETECTOR}@{REVISAO[:12]} (transformers {transformers.__version__}, {'fp16' if meio else 'fp32'}, {LADO} px)"

    def prepara(self, a):
        F = self.t.nn.functional
        h, w = a.shape[:2]
        s = max(h, w)
        x = self.t.from_numpy(np.array(a, copy=True)).to(self.dev).permute(2, 0, 1).float().div(255)[None]
        x = F.pad(x, (0, s - w, 0, s - h), value=0.5)
        x = F.interpolate(x, size=(LADO, LADO), mode="bilinear", antialias=True, align_corners=False)
        return (x - self.media) / self.desvio, s

    def __call__(self, imagens):
        """imagens: lista de uint8 [h, w, 3] -> lista de (caixas [n, 4] em pixels, notas [n], consulta [n]) com nota >= NOTA_MIN."""
        t = self.t
        prep = [self.prepara(a) for a in imagens]
        x = t.cat([p[0] for p in prep]).to(self.dtype)
        B, Q = len(imagens), len(CONSULTAS)
        with t.inference_mode():
            o = self.m(input_ids=self.ids.repeat(B, 1), attention_mask=self.mascara.repeat(B, 1), pixel_values=x)
        notas, consulta = t.sigmoid(o.logits.float()).max(-1)       # [B, caixas]: a melhor das consultas e qual
        cx, cy, bw, bh = o.pred_boxes.float().unbind(-1)
        caixas = t.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
        saida = []
        for i, (_, s) in enumerate(prep):
            k = notas[i] >= NOTA_MIN
            saida.append(((caixas[i][k] * s).cpu().numpy(), notas[i][k].cpu().numpy(), consulta[i][k].cpu().numpy()))
        return saida


# ------------------------------------------------------------------ as fotos de cada base
def fotos(D, a):
    """{base: dict(ids, arquivos, raiz, corte)} das fotos a detectar: os pratos de treino_v2 e todas as fotos das bases
    externas que a quarta execução usava (treino e registro). As pastas são resolvidas por prontas()."""
    out = {"n5k": dict(ids=[d for d, u in zip(D.ids, D.usa_tr) if u], arquivos=[f"{d}/rgb.png" for d, u in zip(D.ids, D.usa_tr) if u])}
    for ch, d in D.dom.items():
        out[ch] = dict(ids=list(d["ids"]), arquivos=list(d["arquivos"]))
    return out


def caminho(F, i):
    corte = F.get("corte", 0)
    rel = "/".join(F["arquivos"][i].split("/")[corte:])
    return G.acha_foto(F["raiz"], rel) or os.path.join(F["raiz"], rel)


# ------------------------------------------------------------------ processo de detecção (um por GPU)
def trabalhador(a):
    """Processa a fatia k (índice % n == k) de cada base, na ordem, assim que a base fica pronta (pronto_<base>.json).
    Grava parte_<base>_<k>.csv (atômico): a retomada pula o que já existe."""
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    dev = torch.device(a.dispositivo)
    det = None if a.sem_detector else Detector(dev, os.path.join(a.cache, "hf"))
    versao = det.versao if det else "sem detector (só a reserva por elipse; teste)"
    k, n = a.trabalhador, a.n_trabalhadores
    feitas = set()
    while len(feitas) < len(DOMINIOS):
        for ch in DOMINIOS:
            if ch in feitas:
                continue
            arq = os.path.join(a.saida, f"parte_{ch}_{k}.csv")
            if os.path.exists(arq) or os.path.exists(os.path.join(a.saida, f"caixas_r5_{ch}.csv")) or os.path.exists(os.path.join(a.cache, f"falhou_{ch}.json")):
                feitas.add(ch); continue
            pronto = os.path.join(a.cache, f"pronto_{ch}.json")
            if not os.path.exists(pronto):
                continue
            F = json.load(open(pronto))
            idx = [i for i in range(len(F["ids"])) if i % n == k]
            t0, linhas, por_metodo = time.time(), [], {}
            with ThreadPoolExecutor(4) as ex:
                def abre(i):        # leitura, hash e elipse nas linhas de CPU (o OpenCV solta o GIL): a GPU não espera
                    img, girada = abre_canonica(caminho(F, i))
                    return i, img, girada, sha256(caminho(F, i)), elipse(img)
                for j in range(0, len(idx), a.lote):
                    if time.time() > a.prazo:
                        print(f"[{k}] {ch}: prazo esgotado", flush=True); sys.exit(3)
                    lote = list(ex.map(abre, idx[j:j + a.lote]))
                    res = det([x[1] for x in lote]) if det else [(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))] * len(lote)
                    for (i, img, girada, sha, el), (cx_, nt, cq) in zip(lote, res):
                        h, w = img.shape[:2]
                        cx_box, nota, nc = caixa_refeicao(cx_, nt, w, h)
                        brutas = brutas_de(cx_, nt, cq, w, h)
                        so_p = caixa_refeicao(cx_[np.isin(cq, SO_PRATO)], nt[np.isin(cq, SO_PRATO)], w, h)[0]
                        metodo = "owlv2"
                        if cx_box is None:
                            cx_box, metodo = el, "elipse"
                            if cx_box is None:
                                metodo = "sem_prato"
                        jan = janela(cx_box, w, h)
                        por_metodo[metodo] = por_metodo.get(metodo, 0) + 1
                        linhas.append(dict(id=F["ids"][i], dominio=ch, arquivo=F["arquivos"][i], sha256=sha, largura=w, altura=h, girada=int(girada), metodo=metodo,
                                           **({c: "" for c in ("x0", "y0", "x1", "y1")} if cx_box is None else dict(zip(("x0", "y0", "x1", "y1"), [round(v, 1) for v in cx_box]))),
                                           nota=round(nota, 4), n_caixas=nc, janela_cx=round(jan[0], 1), janela_cy=round(jan[1], 1), janela_alt=round(jan[2], 1),
                                           janela_larg=round(jan[3], 1), versao_detector=versao, _i=i,
                                           iou_uniao_so_prato="" if cx_box is None or so_p is None or metodo != "owlv2" else round(iou(cx_box, so_p), 4),
                                           caixa_so_prato="" if so_p is None else json.dumps([round(v, 1) for v in so_p]),
                                           caixa_elipse="" if el is None else json.dumps([round(v, 1) for v in el]),
                                           caixas_brutas=json.dumps(brutas, separators=(",", ":"))))
            linhas.sort(key=lambda r: r["_i"])
            with open(arq + ".tmp", "w", newline="") as f:
                wr = csv.DictWriter(f, COLS + ["_i"], lineterminator="\n"); wr.writeheader(); wr.writerows(linhas)
            os.replace(arq + ".tmp", arq)
            feitas.add(ch)
            print(f"[{k}] {ch}: {len(linhas)} fotos em {time.time() - t0:.0f}s ({len(linhas) / max(1e-9, time.time() - t0):.1f}/s) {por_metodo}", flush=True)
        time.sleep(0 if len(feitas) == len(DOMINIOS) else 2)


# ------------------------------------------------------------------ orquestrador
def diz(*p):
    G.diz(*p)


def prepara_base(a, D, ch, F):
    """Fotos de uma base em disco (as mesmas funções de download e conferência da quarta execução)."""
    if ch == "n5k":
        G.baixa(F["ids"], "rgb.png", a.cache)
        if not a.sem_profundidade:
            G.baixa(F["ids"], "depth_raw.png", a.cache)
        return os.path.join(a.cache, "img"), 0
    d = D.dom[ch]
    if d.get("raiz") and os.path.isdir(d["raiz"]):
        return d["raiz"], 0
    raiz = G.baixa_dominio4(a, d)
    return raiz, d.get("raiz_espelho", (None, 0))[1]


def profundidade(a, F, ref):
    """Caixa da profundidade de 8.7, refeita aqui para todos os pratos de treino_v2 (semente própria por prato)."""
    out = {}
    for d in F["ids"]:
        try:
            rng = np.random.default_rng(int(hashlib.sha1(d.encode()).hexdigest(), 16) % 2 ** 32)
            out[d] = caixa_profundidade(mapa_altura(os.path.join(a.cache, "img", d, "depth_raw.png"), rng))
        except Exception:
            out[d] = None
    return out


def folhas(a, ch, linhas, F, sufixo=""):
    """Folhas de contato (14.3): 40 fotos sorteadas (semente 0), em duas folhas de 20: a foto com a caixa (vermelho) e a
    janela (amarelo) desenhadas, e o recorte ao lado. Mais folha_r5_<base>.csv com a posição de cada uma."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(0)
    sel = sorted(rng.choice(len(linhas), min(N_FOLHA, len(linhas)), replace=False).tolist())
    W, H, colunas = 240, 180, 4
    tab = []
    for f0 in range(0, len(sel), POR_FOLHA):
        parte = sel[f0:f0 + POR_FOLHA]
        linhas_f = math.ceil(len(parte) / colunas)
        folha = Image.new("RGB", (colunas * (2 * W + 8), linhas_f * (H + 22)), (255, 255, 255))
        dr = ImageDraw.Draw(folha)
        for p, j in enumerate(parte):
            r = linhas[j]
            img, _ = abre_canonica(caminho(F, r["_i"]))
            h, w = img.shape[:2]
            jan = (float(r["janela_cx"]), float(r["janela_cy"]), float(r["janela_alt"]), float(r["janela_larg"]))
            esc = min(W / w, H / h)
            mini = Image.fromarray(img).resize((max(1, round(w * esc)), max(1, round(h * esc))))
            d2 = ImageDraw.Draw(mini)
            if r["x0"] != "":
                d2.rectangle([float(r[c]) * esc for c in ("x0", "y0", "x1", "y1")], outline=(255, 0, 0), width=3)
            d2.rectangle([(jan[0] - jan[3] / 2) * esc, (jan[1] - jan[2] / 2) * esc, (jan[0] + jan[3] / 2) * esc, (jan[1] + jan[2] / 2) * esc], outline=(255, 220, 0), width=2)
            rec = Image.fromarray(recorta(img, jan, (W, H)))
            x, y = (p % colunas) * (2 * W + 8), (p // colunas) * (H + 22)
            folha.paste(mini, (x, y)); folha.paste(rec, (x + W, y))
            num = f0 + p + 1
            dr.text((x + 2, y + H + 3), f"{num}. {r['metodo']} {float(r['nota']):.2f}  {r['id'][-26:]}", fill=(0, 0, 0))
            tab.append(dict(numero=num, folha=f0 // POR_FOLHA + 1, id=r["id"], arquivo=r["arquivo"], metodo=r["metodo"], nota=r["nota"]))
        folha.save(os.path.join(a.saida, f"folha_r5_{ch}{sufixo}_{f0 // POR_FOLHA + 1}.jpg"), quality=80)
    with open(os.path.join(a.saida, f"folha_r5_{ch}{sufixo}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, ["numero", "folha", "id", "arquivo", "metodo", "nota", "veredito"], lineterminator="\n")
        wr.writeheader(); wr.writerows(dict(t, veredito="") for t in tab)


def orquestra(a):
    t_ini = time.time()
    os.makedirs(a.saida, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
    G._arq_log = open(os.path.join(a.saida, "deteccao_r5.log"), "a")
    a.prazo = t_ini + a.orcamento_h * 3600 - 240
    a.externo_ate = t_ini + G.PRAZO_DOWNLOAD4
    G.PASTA_PESOS = os.path.join(a.cache, "pesos")
    D = G.Dados(a.listas, a.smoke, "4", externo_imagens=a.externo_imagens, dominios=a.dominios)
    FT = fotos(D, a)
    diz(f"{FORMATO} | etapa 0 da quinta execução (14.3) | detector {DETECTOR}@{REVISAO[:12]} | fotos: " + ", ".join(f"{ch} {len(F['ids'])}" for ch, F in FT.items())
        + f" | fora: {D.dom_fora or 'nenhuma'} | orçamento {a.orcamento_h} h")
    for t in os.listdir(a.cache):          # nenhum processo vivo daqui: as marcas de execuções anteriores saem (refazê-las é rápido)
        if t.startswith(("pronto_", "falhou_")):
            os.remove(os.path.join(a.cache, t))
    for ch in D.dom_fora:
        G.grava_json(os.path.join(a.cache, f"falhou_{ch}.json"), dict(motivo=D.dom_fora[ch]))
    info = dict(formato=FORMATO, inicio=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv[1:], smoke=bool(a.smoke), listas_hash=D.hash,
                detector=dict(modelo=DETECTOR, revisao=REVISAO, sha256_pesos=SHA_PESOS, licenca=LICENCA_DETECTOR, consultas=CONSULTAS, lado=LADO,
                              nota_min=NOTA_MIN, area_min=AREA_MIN, nms=NMS_IOU, elipse=dict(area=ELIPSE_AREA, razao_min=ELIPSE_RAZAO), usado=not a.sem_detector),
                janela=dict(frac_guia=FRAC_GUIA, aspecto="4:3", regra="centrada na caixa; altura = maior lado ÷ 0,80; cor mediana da borda fora da foto"),
                bases={ch: dict(situacao="aguardando as fotos", fotos=len(F["ids"])) for ch, F in FT.items()})
    info["bases"].update({ch: dict(situacao=f"FORA: {m}") for ch, m in D.dom_fora.items()})
    trava = threading.Lock()
    eventos = open(os.path.join(a.saida, "eventos_r5.jsonl"), "a")

    def evento(**e):
        with trava:
            eventos.write(json.dumps(dict(t=round(time.time() - t_ini, 1), **e), ensure_ascii=False) + "\n"); eventos.flush()
            G.grava_json(os.path.join(a.saida, "manifesto_r5.json"), info)

    def base(ch):
        F = FT[ch]
        try:
            F["raiz"], F["corte"] = prepara_base(a, D, ch, F)
            falta = [f for i, f in enumerate(F["arquivos"]) if not os.path.exists(caminho(F, i))]
            assert not falta, f"{len(falta)} fotos não estão em disco (ex.: {falta[:2]})"
            G.grava_json(os.path.join(a.cache, f"pronto_{ch}.json"), F)
            info["bases"][ch]["situacao"] = "fotos prontas"; info["bases"][ch]["seg_ate_as_fotos"] = round(time.time() - t_ini)
            diz(f"base {ch}: {len(F['ids'])} fotos prontas em {time.time() - t_ini:.0f}s")
            evento(base=ch, tipo="fotos_prontas")
        except BaseException as ex:
            G.grava_json(os.path.join(a.cache, f"falhou_{ch}.json"), dict(motivo=f"{type(ex).__name__}: {ex}"))
            info["bases"][ch]["situacao"] = f"FORA: as fotos não ficaram prontas ({type(ex).__name__}: {ex})"
            diz(f"ATENÇÃO: base {ch} fica de FORA da etapa 0: {type(ex).__name__}: {ex}")
            evento(base=ch, tipo="falhou", erro=f"{type(ex).__name__}: {ex}")
    linhas_b = [threading.Thread(target=base, args=(ch,), daemon=True) for ch in FT if ch not in D.dom_fora]
    [t.start() for t in linhas_b]
    import torch
    n_gpu = 0 if a.cpu else torch.cuda.device_count()
    disp = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu else ["cpu"] * a.workers
    if not n_gpu and not (a.smoke or a.cpu):
        raise SystemExit("Sem GPU. No Kaggle: Settings > Accelerator > GPU T4 x2. (Para testar em CPU: --smoke --cpu.)")
    diz(f"torch {torch.__version__} | {len(disp)} processo(s) de detecção: {disp}")

    def lanca(k, dev):
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 2) // len(disp))))
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--trabalhador", str(k), "--n-trabalhadores", str(len(disp)), "--saida", a.saida,
               "--cache", a.cache, "--listas", a.listas, "--lote", str(a.lote), "--prazo", str(a.prazo)] + (["--sem-detector"] if a.sem_detector else [])
        if dev.startswith("cuda"):
            env["CUDA_VISIBLE_DEVICES"] = dev.split(":")[1]; cmd += ["--dispositivo", "cuda:0"]
        else:
            cmd += ["--dispositivo", "cpu"]
        for tentativa in range(2):
            pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for l in pr.stdout:
                diz(f"[{dev}] {l.rstrip()}")
            rc = pr.wait()
            evento(trabalhador=k, tipo="terminou", codigo=rc, tentativa=tentativa + 1)
            if rc in (0, 3):
                return
            diz(f"[{dev}] processo de detecção {k} saiu com código {rc}; tentativa {tentativa + 2}")
    ts = [threading.Thread(target=lanca, args=(k, d)) for k, d in enumerate(disp)]
    prof = {}
    if "n5k" in FT and not a.sem_profundidade:       # caixa da profundidade (CPU), em paralelo com a detecção
        def faz_prof():
            while not os.path.exists(os.path.join(a.cache, "pronto_n5k.json")):
                time.sleep(2)
            t0 = time.time(); prof.update(profundidade(a, FT["n5k"], None)); diz(f"caixas da profundidade (8.7) refeitas: {sum(v is not None for v in prof.values())} de {len(prof)} em {time.time() - t0:.0f}s")
        ts.append(threading.Thread(target=faz_prof))
    [t.start() for t in ts]; [t.join() for t in ts]
    # junta as partes, mede a concordância no Nutrition5k, desenha as folhas
    ref = {}
    arq_ref = os.path.join(a.listas, "caixas_profundidade_n5k.csv")
    if os.path.exists(arq_ref):
        ref = {r["id"]: tuple(float(r[c]) for c in ("x0", "y0", "x1", "y1")) for r in csv.DictReader(open(arq_ref))}
    resumo = dict(formato=FORMATO, detector=info["detector"], janela=info["janela"], bases={})
    for ch, F in FT.items():
        partes = [os.path.join(a.saida, f"parte_{ch}_{k}.csv") for k in range(len(disp))]
        if os.path.exists(os.path.join(a.saida, f"caixas_r5_{ch}.csv")) and not any(os.path.exists(p) for p in partes):     # retomada: já feita
            anterior = json.load(open(os.path.join(a.saida, "resumo_r5.json"))) if os.path.exists(os.path.join(a.saida, "resumo_r5.json")) else {}
            resumo["bases"][ch] = anterior.get("bases", {}).get(ch, dict(situacao="feita numa execução anterior"))
            continue
        if not all(os.path.exists(p) for p in partes):
            resumo["bases"][ch] = dict(situacao=info["bases"][ch]["situacao"] if "FORA" in info["bases"][ch]["situacao"] else "detecção incompleta (prazo)")
            continue
        linhas = sorted((r for p in partes for r in csv.DictReader(open(p))), key=lambda r: int(r["_i"]))
        for r in linhas:
            r["_i"] = int(r["_i"])
        assert [r["id"] for r in linhas] == F["ids"], f"{ch}: fotos faltando ou fora de ordem"
        cols = COLS + (["iou_profundidade_gravada", "iou_profundidade_refeita"] if ch == "n5k" else [])
        r_ = dict(fotos=len(linhas), por_metodo={m: sum(r["metodo"] == m for r in linhas) for m in ("owlv2", "elipse", "sem_prato")})
        r_["taxa_sem_prato"] = round(r_["por_metodo"]["sem_prato"] / len(linhas), 4)
        v = [float(r["iou_uniao_so_prato"]) for r in linhas if r["iou_uniao_so_prato"] != ""]
        r_["diagnostico_uniao_so_prato"] = dict(nota="IoU entre a caixa da regra (união de todas as consultas) e a união só das consultas de prato e tigela; "
                                                     "não muda nada, só mostra onde bandeja, marmita ou recipiente aumentam a caixa", fotos=len(v),
                                                mediana=round(float(np.median(v)), 4) if v else None, pct_abaixo_de_0_7=round(100 * float(np.mean(np.array(v) < 0.7)), 1) if v else None)
        if ch == "n5k":
            for nome, fonte in (("gravada", ref), ("refeita", prof)):
                ious = []
                for r in linhas:
                    b = fonte.get(r["id"])
                    v = iou(tuple(float(r[c]) for c in ("x0", "y0", "x1", "y1")), b) if b is not None and r["x0"] != "" else (0.0 if b is not None else None)
                    r[f"iou_profundidade_{nome}"] = "" if v is None else round(v, 4)
                    if v is not None:
                        ious.append(v)
                r_[f"iou_profundidade_{nome}"] = dict(pratos=len(ious), mediana=round(float(np.median(ious)), 4) if ious else None,
                                                      pct_iou_0_7=round(100 * float(np.mean(np.array(ious) >= 0.7)), 1) if ious else None)
            comum = [d for d in F["ids"] if ref.get(d) is not None and prof.get(d) is not None]
            if comum:       # a conta refeita aqui repete a gravada em 8.7?
                v = [iou(ref[d], prof[d]) for d in comum]
                r_["profundidade_refeita_x_gravada"] = dict(pratos=len(v), mediana=round(float(np.median(v)), 4), pct_iou_0_9=round(100 * float(np.mean(np.array(v) >= 0.9)), 1))
        arq = os.path.join(a.saida, f"caixas_r5_{ch}.csv")
        with open(arq, "w", newline="") as f:
            wr = csv.DictWriter(f, cols, lineterminator="\n", extrasaction="ignore"); wr.writeheader(); wr.writerows(linhas)
        r_["arquivo"], r_["sha256"] = os.path.basename(arq), sha256(arq)
        r_["versao_detector"] = linhas[0]["versao_detector"] if linhas else None
        try:
            folhas(a, ch, linhas, F)
            # diagnóstico ao lado (não muda a regra): as mesmas 40 fotos com a caixa só das consultas de prato e tigela
            folhas(a, ch, [dict(refaz_linha(r, SO_PRATO), _i=r["_i"]) for r in linhas], F, "_so_prato")
            r_["folhas"] = sorted(x for x in os.listdir(a.saida) if x.startswith(f"folha_r5_{ch}_"))
        except Exception as ex:
            r_["folhas"] = f"falhou: {type(ex).__name__}: {ex}"
            traceback.print_exc()
        for p in partes:
            os.remove(p)
        resumo["bases"][ch] = r_
        diz(f"== {ch}: {r_['fotos']} fotos | {r_['por_metodo']} | sem_prato {100 * r_['taxa_sem_prato']:.1f}%"
            + f" | diagnóstico, união só de prato/tigela × regra: {r_['diagnostico_uniao_so_prato']['pct_abaixo_de_0_7']}% das fotos com IoU < 0,7"
            + (f" | IoU com a caixa da profundidade (8.7): gravada {r_['iou_profundidade_gravada']}, refeita {r_['iou_profundidade_refeita']}" if ch == "n5k" else ""))
    resumo["duracao_min"] = round((time.time() - t_ini) / 60, 1)
    G.grava_json(os.path.join(a.saida, "resumo_r5.json"), resumo)
    info.update(duracao_min=resumo["duracao_min"], resumo="resumo_r5.json")
    G.grava_json(os.path.join(a.saida, "manifesto_r5.json"), info)
    diz(f"fim: {resumo['duracao_min']} min. Tabelas caixas_r5_<base>.csv, folhas folha_r5_<base>_<k>.jpg e resumo_r5.json em {a.saida}.")


def refaz_tabelas(entrada, saida, consultas):
    """--refaz-regra: todas as caixas_r5_<base>.csv de uma saída da etapa 0, refeitas com outra regra (refaz_linha)."""
    os.makedirs(saida, exist_ok=True)
    for arq in sorted(f for f in os.listdir(entrada) if f.startswith("caixas_r5_") and f.endswith(".csv")):
        L = list(csv.DictReader(open(os.path.join(entrada, arq))))
        N = [refaz_linha(r, consultas) for r in L]
        with open(os.path.join(saida, arq), "w", newline="") as f:
            wr = csv.DictWriter(f, list(L[0]), lineterminator="\n"); wr.writeheader(); wr.writerows(N)
        cx_ = lambda r: None if r["x0"] in ("", None) else [float(r[c]) for c in ("x0", "y0", "x1", "y1")]
        mud = sum(cx_(r) is None or cx_(n) is None or max(abs(x - y) for x, y in zip(cx_(r), cx_(n))) > 1.0 if (cx_(r), cx_(n)) != (None, None) else False
                  for r, n in zip(L, N))      # mais de 1 px de diferença (as caixas brutas são gravadas com 0,1 px)
        print(f"{arq}: {len(N)} fotos, {mud} com a caixa mudada | {dict((m, sum(n['metodo'] == m for n in N)) for m in ('owlv2', 'elipse', 'sem_prato'))} | sha256 {sha256(os.path.join(saida, arq))[:16]}")


# ------------------------------------------------------------------ autoteste
def autoteste():
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-12 and iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    c = [(0, 0, 100, 100), (5, 5, 105, 105), (200, 200, 300, 300), (0, 0, 10, 10), (400, 0, 640, 480)]
    n = [0.9, 0.5, 0.3, 0.8, 0.1]
    assert nms(np.array(c[:3], float), n[:3]) == [0, 2]
    u, nota, k = caixa_refeicao(c, n, 640, 480)
    assert (u, nota, k) == ((0.0, 0.0, 300.0, 300.0), 0.9, 2)       # (0,0,10,10) é pequena demais, a de 0,1 tem nota baixa, a de 0,5 sai no NMS
    assert caixa_refeicao(c[3:], n[3:], 640, 480)[0] is None and caixa_refeicao(np.zeros((0, 4)), [], 640, 480)[0] is None
    assert [b[4:] for b in brutas_de(c, n, [6, 6, 2, 0, 1], 640, 480)] == [[0.9, 6], [0.5, 6], [0.3, 2]]       # antes do NMS, sem as pequenas e as de nota baixa
    r = dict(largura="640", altura="480", caixas_brutas=json.dumps(brutas_de(c, n, [6, 6, 2, 0, 1], 640, 480)), caixa_elipse="[10, 10, 300, 300]", versao_detector="v",
             x0="0", y0="0", x1="300", y1="300")
    assert refaz_linha(r, {6})["x1"] == 105.0 or refaz_linha(r, {6})["x1"] == 100.0       # só a consulta 6: a união das duas primeiras depois do NMS
    assert refaz_linha(r, {2})["metodo"] == "owlv2" and refaz_linha(r, {2})["x0"] == 200.0
    assert refaz_linha(r, {3})["metodo"] == "elipse" and refaz_linha(dict(r, caixa_elipse=""), {3})["metodo"] == "sem_prato"
    cx, cy, alt, larg = janela((100, 100, 300, 260), 640, 480)
    assert (cx, cy, alt, larg) == (200, 180, 250, 250 * 4 / 3)       # maior lado 200 ocupa 80% da altura
    assert janela(None, 640, 480) == (320, 240, 480, 640) and janela(None, 1024, 1024) == (512, 512, 1024, 1024 * 4 / 3)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (480, 640, 3)).astype(np.uint8)
    assert np.array_equal(recorta(a, (320, 240, 480, 640)), a)        # janela = a foto inteira 4:3: o recorte é a própria foto
    a[:] = 40; a[100:380, 160:480] = 200
    r = recorta(a, janela((160, 100, 480, 380), 640, 480))
    assert r.shape == (480, 640, 3) and r[240, 320].tolist() == [200, 200, 200] and r[5, 5].tolist() == [40, 40, 40]
    q = np.full((1024, 1024, 3), 30, np.uint8)
    r = recorta(q, janela(None, 1024, 1024))                           # foto quadrada: encaixada no 4:3 com a cor da borda
    assert r[240, 320].tolist() == [30, 30, 30] and r.shape == (480, 640, 3)
    try:
        import cv2  # noqa: F401
        im = np.full((480, 640, 3), 230, np.uint8)
        import cv2
        cv2.ellipse(im, (320, 240), (200, 150), 0, 0, 360, (60, 60, 60), 6)
        e = elipse(im)
        assert e is not None and abs(e[0] - 120) < 12 and abs(e[2] - 520) < 12 and abs(e[1] - 90) < 12 and abs(e[3] - 390) < 12, e
        assert elipse(np.full((480, 640, 3), 128, np.uint8)) is None
    except ImportError:
        print("autoteste: OpenCV ausente, elipse não testada")
    m = np.zeros((120, 160), np.float32); m[30:90, 40:120] = 30
    assert caixa_profundidade(m) == (160.0, 120.0, 480.0, 360.0) and caixa_profundidade(np.zeros((120, 160), np.float32)) is None
    print("autoteste da etapa 0: ok")


def main():
    kaggle = os.path.isdir("/kaggle/working")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--saida", default="/kaggle/working" if kaggle else os.path.abspath("nutrelle_deteccao"))
    ap.add_argument("--cache", default="/kaggle/temp/nutrelle_cache" if kaggle else os.path.join(tempfile.gettempdir(), "nutrelle_cache"))
    ap.add_argument("--listas", default=os.path.join(AQUI, "listas"))
    ap.add_argument("--orcamento-h", type=float, default=1.5)
    ap.add_argument("--dominios", help="bases externas a usar, separadas por vírgula (padrão: todas as que estão no pacote)")
    ap.add_argument("--externo-imagens", help="pasta com uma subpasta de fotos por base (teste; nada é baixado dessas bases)")
    ap.add_argument("--espelho", help="pasta onde procurar um espelho dos arquivos das bases externas (além de /kaggle/input)")
    ap.add_argument("--smoke", type=int, nargs="?", const=8, default=0)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--sem-detector", action="store_true", help="teste: só a reserva por elipse (sem baixar o OWLv2)")
    ap.add_argument("--sem-profundidade", action="store_true", help="não baixa depth_raw.png nem refaz a caixa da profundidade")
    ap.add_argument("--autoteste", action="store_true")
    ap.add_argument("--refaz-regra", metavar="PASTA", help="CPU, sem imagens: refaz caixas_r5_<base>.csv da PASTA (saída da etapa 0) só com as --consultas dadas e grava em --saida")
    ap.add_argument("--consultas", default=",".join(map(str, SO_PRATO)), help="índices de CONSULTAS para --refaz-regra (padrão: as de prato e tigela)")
    ap.add_argument("--trabalhador", type=int); ap.add_argument("--n-trabalhadores", type=int, default=1)
    ap.add_argument("--dispositivo", default="cpu"); ap.add_argument("--prazo", type=float, default=float("inf"))
    a = ap.parse_args()
    a.saida, a.cache, a.listas = (os.path.abspath(p) for p in (a.saida, a.cache, a.listas))
    if a.autoteste:
        return autoteste()
    if a.refaz_regra:
        return refaz_tabelas(a.refaz_regra, a.saida, {int(x) for x in a.consultas.split(",")})
    if a.trabalhador is not None:
        return trabalhador(a)
    a.dominios = None if a.dominios is None else [d for d in a.dominios.split(",") if d]
    a.externo_imagens = os.path.abspath(a.externo_imagens) if a.externo_imagens else None
    a.espelho = os.path.abspath(a.espelho) if a.espelho else None
    orquestra(a)


if __name__ == "__main__":
    main()
