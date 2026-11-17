#!/usr/bin/env python3
"""Nona execução (18.10 d): máscaras do pipeline geométrico. Nada é pontuado; nenhum rótulo é lido nem gravado.

    python3 geom_r9.py --saida <pasta> [--cache <pasta>] [--bases n5k,dfoodtj,fpb] [--espelho-fotos <pasta>]
                       [--smoke N] [--cpu] [--sem-sam] [--orcamento-h H] [--autoteste]

Bases (18.10 d): o treino do Nutrition5k (2.281 fotos de cima, `rgb.png` do bucket público, 640 × 480, alinhadas à
profundidade); as 711 fotos do DFood-TJ (do espelho privado, cada uma conferida pelo tamanho, CRC-32 e sha256 da tabela
das fotos, como na oitava; SEM máscara de visor; período e paciente copiados de `teste_r8_dfoodtj.csv`, que não tem
rótulo); a parte de registro do FPB (312 fotos, Hugging Face, foto a foto com tamanho e sha256 conferidos). As listas
do DFood-TJ e do FPB só vêm no notebook do push; sem elas, a base fica de fora e o manifesto diz por quê.

Em cada foto (orientação do EXIF e retrato girado, como na oitava):
 1. o OWLv2 da etapa 0 (mesmo checkpoint, 960 px), numa passada só com todas as consultas (a nota de cada caixa em cada
    consulta não depende das outras consultas): (a) o prato pela regra só de prato e tigela de 14.3 (nota 0,15, área 2%,
    NMS 0,5; elipse e `sem_prato` de reserva); (b) os itens com as 27 consultas de `vocabulario.csv` (cada caixa fica com
    a consulta de maior nota; nota >= 0,15; NMS 0,5 por termo);
 2. o SAM 2.1 (`facebook/sam2.1-hiera-large`, revisão e sha256 fixados) na resolução de trabalho (lado maior até 1.024 px;
    o Nutrition5k fica em 640 × 480), com a caixa do prato e cada caixa de item como prompt. A máscara do prato é o maior
    componente, com os buracos preenchidos; a elipse do prato sai de `cv2.fitEllipse` no contorno dela (eixos inteiros);
 3. cada item: a fração dentro do prato (antes do recorte), o recorte pelo prato, e o pixel disputado fica com a maior
    nota; `resto` = prato − itens − branco do prato (HSV: S < 0,15 e V > 0,75), sem componentes de menos de 0,5% do prato;
 4. no FPB, o xadrez: `cv2.findChessboardCorners` com alguns tamanhos de padrão; passo mediano entre cantos vizinhos;
 5. saídas: `fotos_geom.csv`, `itens_geom.csv.gz` (máscaras em RLE do COCO sem compressão, em ordem de coluna, começando
    pelos zeros), folhas de conferência (40 fotos por base, máscaras coloridas por grupo), `geom_r9.json` e `geom_r9.log`.
    Nenhuma máscara densa vai para o disco.
Sem o SAM (não baixou ou não carregou), a máscara de cada caixa é a elipse inscrita nela, recortada pelo prato
(`origem_mascara = caixa_elipse`), e o manifesto marca a reserva.

Nas folhas do DFood-TJ (só nelas, nunca nas tabelas), o mostrador da balança é pintado pela regra da 8b, para que quem
confere as máscaras não leia o peso."""
import argparse, csv, gzip, hashlib, io, json, math, os, sys, threading, time, traceback
from types import SimpleNamespace
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import treino_gpu as G  # noqa: E402
import deteccao_r5 as E  # noqa: E402

FORMATO = "nutrelle-geom/9"
SAM = "facebook/sam2.1-hiera-large"
REV_SAM = "665f8e2ad61cf5f53d65644ff27c8ee525124610"           # commit do Hugging Face conferido em 17/11/2026
SHA_SAM = "dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230"     # model.safetensors (formato do transformers)
ARQS_SAM = ["config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"]
NOTA_ITEM, NMS_ITEM = 0.15, 0.5
LADO_MAX = 1024
BRANCO_S, BRANCO_V = 0.15, 0.55      # 18.10 (d), emendado: V 0,75 -> 0,55; --branco-s / --branco-v mudam (e o manifesto registra)
RESTO_MIN = 0.005
PADROES_XADREZ = [(9, 6), (6, 9), (8, 6), (6, 8), (7, 7), (7, 5), (5, 7), (6, 6), (6, 4), (4, 6), (5, 5), (5, 4), (4, 5)]   # 3x3 e 4x4: falsos positivos
CONSULTAS_VISOR = ["a digital display", "an LCD screen"]       # só para pintar o visor nas folhas do DFood-TJ (8b)
PERIODO_FOLHAS = "2020-2023"        # as 40 do DFood-TJ saem do período da população da leitura 3 (18.7)
BASES = ["n5k", "dfoodtj", "fpb"]
COLS_FOTOS = ["base", "id", "arquivo", "sha256", "largura", "altura", "girada", "periodo", "paciente", "prato_metodo", "prato_caixa", "prato_elipse",
              "prato_rle", "mascara_largura", "mascara_altura", "xadrez_detectado", "xadrez_passo_px", "xadrez_padrao", "n_itens", "versoes",
              "livre_s_mediano", "livre_v_mediano", "resto_fracao_prato"]      # as três últimas: diagnóstico do limiar do branco (prato − itens)
COLS_ITENS = ["base", "id", "item", "termo", "consulta", "nota", "caixa", "rle", "mascara_largura", "mascara_altura", "area_px", "elipse",
              "origem_mascara", "dentro_prato"]
COLS_N5K, COLS_FPB = ["dish_id"], ["dish_id", "arquivo", "sha256", "bytes"]
CORES = dict(proteina_animal=(220, 40, 40), leguminosa=(150, 90, 30), laticinio=(250, 230, 120), amido=(250, 160, 30), vegetal=(40, 180, 60),
             gordura_molho=(160, 60, 200), misto=(40, 140, 230), outros=(130, 130, 130))
REGISTRO = []
LIMIAR = dict(s=BRANCO_S, v=BRANCO_V)


def diz(*p):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in p)
    print(s, flush=True)
    REGISTRO.append(s)


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ 1. listas
def le_csv(arq, cols):
    """Linhas de uma lista embutida; as colunas têm de ser exatamente as esperadas (nenhuma de rótulo)."""
    L = list(csv.DictReader(open(arq, newline="")))
    assert L and list(L[0]) == cols, f"{arq}: colunas {list(L[0]) if L else None}, esperado {cols}"
    assert len({r["dish_id"] for r in L}) == len(L), f"{arq}: ids repetidos"
    return L


def arq_vocabulario():
    """listas/vocabulario_r9.csv no pacote (cópia byte a byte de eval/geom/vocabulario.csv); no repositório, o original."""
    p = os.path.join(AQUI, "listas", "vocabulario_r9.csv")
    return p if os.path.exists(p) else os.path.join(AQUI, "..", "..", "geom", "vocabulario.csv")


def le_vocabulario(arq):
    """(termos, consultas, grupo por termo): as 27 consultas do OWLv2, na ordem do arquivo; `resto` não tem consulta."""
    V = list(csv.DictReader(open(arq, newline="")))
    it = [r for r in V if r["consulta_owlv2"]]
    assert len(it) == 27 and any(r["termo"] == "resto" and not r["consulta_owlv2"] for r in V), "vocabulário: 27 consultas e o resto"
    return [r["termo"] for r in it], [r["consulta_owlv2"] for r in it], {r["termo"]: r["grupo"] for r in V}


# ------------------------------------------------------------------ 2. máscaras: funções puras
def rle_codifica(m):
    """RLE do COCO sem compressão: contagens alternadas, em ordem de coluna, começando pelos zeros (= eval/geom/escala.py)."""
    v = np.asarray(m, bool).T.reshape(-1).astype(np.int8)
    d = np.flatnonzero(np.diff(np.concatenate([[0], v, [1 - v[-1]]])))
    c = np.diff(np.concatenate([[0], d]))
    return " ".join(str(int(x)) for x in c)


def rle_decodifica(s, h, w):
    c = np.array([int(x) for x in s.split()], dtype=np.int64)
    v = np.zeros(h * w, dtype=bool)
    pos = np.concatenate([[0], np.cumsum(c)])
    for k in range(1, len(c), 2):
        v[pos[k]:pos[k + 1]] = True
    return v.reshape(w, h).T


def elipse_de(pontos, fx=1.0, fy=1.0):
    """`cx cy eixo_maior eixo_menor angulo_graus` (eixos inteiros; o ângulo é o do eixo maior, em [0, 180)) de
    `cv2.fitEllipse` nos pontos de um contorno [n, 1, 2] da máscara de trabalho, levados à foto (× fx, fy). '' se < 5 pontos."""
    import cv2
    p = np.asarray(pontos, np.float32).reshape(-1, 2)
    if len(p) < 5:
        return ""
    p = (p + 0.5) * np.array([fx, fy], np.float32)
    (cx, cy), (a, b), ang = cv2.fitEllipse(p)
    if b > a:
        a, b, ang = b, a, ang + 90.0
    return f"{cx:.1f} {cy:.1f} {a:.1f} {b:.1f} {ang % 180.0:.1f}"


def maior_contorno(m):
    """Contorno externo do maior componente da máscara (ou None)."""
    import cv2
    cs, _ = cv2.findContours(np.asarray(m, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea) if cs else None


def prato_preenchido(m):
    """Máscara do prato: o maior componente, com os buracos preenchidos (a comida por cima do prato é prato). Função pura."""
    import cv2
    c = maior_contorno(m)
    out = np.zeros(np.shape(m), np.uint8)
    if c is not None:
        cv2.drawContours(out, [c], -1, 1, thickness=-1)
    return out.astype(bool), c


def elipse_inscrita(caixa, h, w):
    """Máscara (reserva sem o SAM): a elipse inscrita na caixa (pixels de trabalho)."""
    import cv2
    x0, y0, x1, y1 = caixa
    m = np.zeros((h, w), np.uint8)
    cx, cy, a, b = (x0 + x1) / 2, (y0 + y1) / 2, max(1.0, (x1 - x0) / 2), max(1.0, (y1 - y0) / 2)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(round(a)), int(round(b))), 0, 0, 360, 1, -1)
    return m.astype(bool)


def branco(img):
    """Branco do prato (limiar fixo de 18.10 d): S < 0,15 e V > 0,75 no HSV. Função pura."""
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] < LIMIAR["s"] * 255) & (hsv[..., 2] > LIMIAR["v"] * 255)


def sem_pequenos(m, minimo):
    """Tira os componentes (8-conexos) com menos de `minimo` pixels. Função pura."""
    import cv2
    n, rot, st, _ = cv2.connectedComponentsWithStats(np.asarray(m, np.uint8), connectivity=8)
    fica = np.zeros(n, bool)
    fica[1:] = st[1:, cv2.CC_STAT_AREA] >= minimo
    return fica[rot]


def resolve_itens(mascaras, notas, prato):
    """Cada máscara: fração dentro do prato (antes do recorte; None sem prato), recorte pelo prato e o pixel disputado
    para a maior nota. -> (máscaras finais na ordem de entrada, frações). Função pura."""
    fr, fin = [], [None] * len(mascaras)
    for m in mascaras:
        n = int(m.sum())
        fr.append(None if prato is None else (float((m & prato).sum()) / n if n else 0.0))
    tomado = np.zeros(prato.shape if prato is not None else (mascaras[0].shape if mascaras else (1, 1)), bool)
    for i in sorted(range(len(mascaras)), key=lambda i: -notas[i]):
        m = mascaras[i] & prato if prato is not None else mascaras[i].copy()
        m &= ~tomado
        tomado |= m
        fin[i] = m
    return fin, fr


def resto_de(prato, itens, img):
    """resto = prato − itens − branco do prato, sem componentes de menos de 0,5% do prato. Função pura."""
    r = prato.copy()
    for m in itens:
        r &= ~m
    r &= ~branco(img)
    return sem_pequenos(r, RESTO_MIN * prato.sum())


def nms_por_termo(caixas, notas, termos):
    """Índices mantidos: NMS a 0,5 dentro de cada termo. Função pura."""
    fica = []
    for t in sorted(set(termos)):
        idx = [i for i in range(len(termos)) if termos[i] == t]
        fica += [idx[j] for j in E.nms([caixas[i] for i in idx], [notas[i] for i in idx], NMS_ITEM)]
    return sorted(fica, key=lambda i: -notas[i])


def xadrez(cinza):
    """(detectado, passo mediano entre cantos vizinhos em pixels, padrão) pelo cv2.findChessboardCorners."""
    import cv2
    fl = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    for pc in PADROES_XADREZ:
        ok, c = cv2.findChessboardCorners(cinza, pc, flags=fl)
        if ok:
            c = cv2.cornerSubPix(cinza, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            g = c.reshape(pc[1], pc[0], 2)
            d = np.concatenate([np.linalg.norm(np.diff(g, axis=1), axis=-1).ravel(), np.linalg.norm(np.diff(g, axis=0), axis=-1).ravel()])
            return 1, float(np.median(d)), f"{pc[0]}x{pc[1]}"
    return 0, None, ""


def fmt_caixa(c):
    return " ".join(f"{float(v):.1f}" for v in c)


# ------------------------------------------------------------------ 3. modelos
class Modelos:
    """OWLv2 (todas as consultas numa passada) e SAM 2.1 num dispositivo."""

    def __init__(self, a, dev, consultas, usa_sam=True):
        self.dev = dev
        self.det = E.Detector(dev, a.hf).com_consultas(consultas)
        self.versao_det = self.det.versao
        self.sam, self.erro_sam = None, None
        if usa_sam:
            try:
                self.sam = carrega_sam(dev, a.hf)
            except Exception as ex:     # noqa: BLE001 — sem o SAM, a reserva por caixa e elipse (marcada)
                self.erro_sam = f"{type(ex).__name__}: {ex}"
                diz(f"SAM 2.1 não carregou em {dev} ({self.erro_sam}); reserva: elipse inscrita na caixa")

    def detecta(self, imgs):
        """-> por foto: (caixas [n, 4] em pixels da foto, notas [n, Q] de cada consulta), caixas com alguma nota >= 0,15."""
        d, t = self.det, self.det.t
        prep = [d.prepara(x) for x in imgs]
        x = t.cat([p[0] for p in prep]).to(d.dtype)
        B = len(imgs)
        with t.inference_mode():
            o = d.m(input_ids=d.ids.repeat(B, 1), attention_mask=d.mascara.repeat(B, 1), pixel_values=x)
        notas = t.sigmoid(o.logits.float())
        cx, cy, bw, bh = o.pred_boxes.float().unbind(-1)
        caixas = t.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
        out = []
        for i, (_, s) in enumerate(prep):
            k = notas[i].max(-1).values >= E.NOTA_MIN
            out.append(((caixas[i][k] * s).cpu().numpy().astype(np.float64), notas[i][k].cpu().numpy().astype(np.float64)))
        return out

    def mascaras(self, img, caixas):
        """Máscaras bool [k, h, w] na resolução de `img` (a de trabalho), uma por caixa (pixels de trabalho)."""
        h, w = img.shape[:2]
        if not len(caixas):
            return np.zeros((0, h, w), bool), "sam2"
        if self.sam is None:
            return np.stack([elipse_inscrita(c, h, w) for c in caixas]), "caixa_elipse"
        from PIL import Image
        import torch
        proc, m = self.sam
        inp = proc(images=Image.fromarray(img), input_boxes=[[[float(v) for v in c] for c in caixas]], return_tensors="pt")
        with torch.inference_mode():
            o = m(pixel_values=inp["pixel_values"].to(self.dev), input_boxes=inp["input_boxes"].to(self.dev), multimask_output=False)
        mk = proc.post_process_masks(o.pred_masks.float().cpu(), inp["original_sizes"])[0][:, 0]
        return mk.numpy().astype(bool), "sam2"


def carrega_sam(dev, pasta):
    """SAM 2.1 hiera-large do Hugging Face (formato do transformers), na revisão fixada e com o sha256 dos pesos conferido."""
    from huggingface_hub import snapshot_download
    from transformers import Sam2Model, Sam2Processor
    local = snapshot_download(SAM, revision=REV_SAM, cache_dir=pasta, allow_patterns=ARQS_SAM)
    sp = os.path.join(local, "model.safetensors")
    sha = os.path.basename(os.path.realpath(sp))
    if sha != SHA_SAM:
        sha = sha256(sp)
    assert sha == SHA_SAM, f"pesos do SAM 2.1 diferentes dos conferidos ({sha})"
    proc = Sam2Processor.from_pretrained(local)
    m = Sam2Model.from_pretrained(local).to(dev).eval()
    return proc, m


# ------------------------------------------------------------------ 4. uma foto
def processa_foto(md, f, img, girada, det_out, V, consultas):
    """Linha de fotos_geom, itens e (se a foto vai às folhas) a imagem de conferência. `f`: base, id, arquivo, sha256,
    periodo, paciente, i, folha."""
    import cv2
    termos, grupo = V
    n_pr, n_it = len(E.CONSULTAS), len(termos)
    h, w = img.shape[:2]
    s = min(1.0, LADO_MAX / max(h, w))
    mw, mh = (w, h) if s >= 1 else (int(round(w * s)), int(round(h * s)))
    fx, fy = w / mw, h / mh
    work = img if (mw, mh) == (w, h) else cv2.resize(img, (mw, mh), interpolation=cv2.INTER_AREA)
    cx_, nt = det_out
    # (a) o prato: a regra só de prato e tigela (14.3), com a mesma função da etapa 0
    npr = nt[:, :n_pr]
    l0 = E.linha_deteccao(f["id"], f["base"], f["arquivo"], img, girada, f["sha256"], E.elipse(img),
                          (cx_, npr.max(1) if len(npr) else np.zeros(0), npr.argmax(1) if len(npr) else np.zeros(0, int)), md.versao_det, f["i"])
    lr = E.refaz_linha(l0, set(E.SO_PRATO))
    cp = None if lr["x0"] == "" else tuple(float(lr[c]) for c in ("x0", "y0", "x1", "y1"))
    # (b) os itens: cada caixa com a consulta de maior nota; nota >= 0,15; NMS por termo
    nit = nt[:, n_pr:n_pr + n_it]
    k = np.where(nit.max(1) >= NOTA_ITEM)[0] if len(nit) else np.zeros(0, int)
    cx_i = np.clip(cx_[k], 0, [w, h, w, h]) if len(k) else np.zeros((0, 4))
    nota_i, termo_i = (nit[k].max(1), nit[k].argmax(1)) if len(k) else (np.zeros(0), np.zeros(0, int))
    fica = nms_por_termo(list(cx_i), list(nota_i), list(termo_i))
    fica = [i for i in fica if cx_i[i][2] > cx_i[i][0] and cx_i[i][3] > cx_i[i][1]]      # caixa vazia depois de cortada na foto: fora
    cx_i, nota_i, termo_i = cx_i[fica].reshape(-1, 4), nota_i[fica], termo_i[fica]
    # (c) máscaras (SAM 2.1; uma passada por foto, a caixa do prato primeiro)
    para_w = lambda c: [c[0] / fx, c[1] / fy, c[2] / fx, c[3] / fy]
    caixas_w = ([para_w(cp)] if cp else []) + [para_w(c) for c in cx_i]
    M, origem = md.mascaras(work, caixas_w)
    prato, cont = (None, None)
    if cp:
        bruto = M[0] if origem == "sam2" else elipse_inscrita(caixas_w[0], mh, mw)
        prato, cont = prato_preenchido(bruto)
        if not prato.any():
            prato, cont = prato_preenchido(elipse_inscrita(caixas_w[0], mh, mw))
        M = M[1:]
    fin, fr = resolve_itens(list(M), list(nota_i), prato)
    itens = []
    for j in range(len(fin)):
        m = fin[j]
        n = int(m.sum())
        if not n:
            continue
        c = maior_contorno(m)
        itens.append(dict(base=f["base"], id=f["id"], termo=termos[int(termo_i[j])], consulta=consultas[n_pr + int(termo_i[j])], nota=f"{float(nota_i[j]):.4f}",
                          caixa=fmt_caixa(cx_i[j]), rle=rle_codifica(m), mascara_largura=mw, mascara_altura=mh, area_px=f"{n * (w / mw) ** 2:.1f}",
                          elipse=elipse_de(c, fx, fy) if c is not None else "", origem_mascara=origem,
                          dentro_prato="" if fr[j] is None else f"{fr[j]:.4f}", _m=m))
    diag = dict(livre_s_mediano="", livre_v_mediano="", resto_fracao_prato="")
    if prato is not None:
        livre = prato.copy()
        for x in itens:
            livre &= ~x["_m"]
        if livre.any():
            hsv = cv2.cvtColor(np.ascontiguousarray(work), cv2.COLOR_RGB2HSV)[livre]
            diag.update(livre_s_mediano=f"{np.median(hsv[:, 1]) / 255:.3f}", livre_v_mediano=f"{np.median(hsv[:, 2]) / 255:.3f}")
        r = resto_de(prato, [x["_m"] for x in itens], work)
        diag["resto_fracao_prato"] = f"{r.sum() / max(1, prato.sum()):.4f}"
        if r.any():
            ys, xs = np.where(r)
            c = maior_contorno(r)
            itens.append(dict(base=f["base"], id=f["id"], termo="resto", consulta="", nota="0", caixa=fmt_caixa((xs.min() * fx, ys.min() * fy, (xs.max() + 1) * fx, (ys.max() + 1) * fy)),
                              rle=rle_codifica(r), mascara_largura=mw, mascara_altura=mh, area_px=f"{int(r.sum()) * (w / mw) ** 2:.1f}",
                              elipse=elipse_de(c, fx, fy) if c is not None else "", origem_mascara="derivada", dentro_prato="1.0000", _m=r))
    for j, x in enumerate(itens):
        x["item"] = j
    linha = dict(base=f["base"], id=f["id"], arquivo=f["arquivo"], sha256=f["sha256"], largura=w, altura=h, girada=int(girada), periodo=f.get("periodo", ""),
                 paciente=f.get("paciente", ""), prato_metodo=lr["metodo"], prato_caixa="" if cp is None else fmt_caixa(cp),
                 prato_elipse=elipse_de(cont, fx, fy) if cont is not None else "", prato_rle="" if prato is None else rle_codifica(prato),
                 mascara_largura=mw, mascara_altura=mh, xadrez_detectado="", xadrez_passo_px="", xadrez_padrao="", n_itens=len(itens), **diag,
                 _origem=origem if (cp or len(cx_i)) else "")
    if f["base"] == "fpb":
        ok, passo, pad = xadrez(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY))
        linha.update(xadrez_detectado=ok, xadrez_passo_px="" if passo is None else f"{passo:.2f}", xadrez_padrao=pad)
    tile = folha_tile(work, prato, itens, grupo, f, linha, cx_, nt, n_pr + n_it, (fx, fy), cp) if f.get("folha") else None
    for x in itens:
        del x["_m"]
    return linha, itens, tile


def folha_tile(work, prato, itens, grupo, f, linha, cx_, nt, q_visor, esc, cp, W=400):
    """Miniatura de conferência: máscaras coloridas por grupo, contorno do prato em branco, o termo de cada item. No
    DFood-TJ, o mostrador da balança pintado pela regra da 8b (só na folha)."""
    from PIL import Image, ImageDraw
    import cv2
    im = work.copy()
    if f["base"] == "dfoodtj" and len(nt):
        import teste_r8 as T8
        nv = nt[:, q_visor:q_visor + len(CONSULTAS_VISOR)]
        cv = T8.filtra_visor(cx_, nv.max(1), nv.argmax(1), work.shape[1] * esc[0], work.shape[0] * esc[1], cp)
        fica = [[c["caixa"][0] / esc[0], c["caixa"][1] / esc[1], c["caixa"][2] / esc[0], c["caixa"][3] / esc[1]] for c in cv if c["destino"] == "mascarada"]
        p_w = None if cp is None else (cp[0] / esc[0], cp[1] / esc[1], cp[2] / esc[0], cp[3] / esc[1])
        im = T8.mascara_fora_do_prato(im, fica, E.cor_borda(im).astype(np.uint8), p_w)[0]
    sob = im.astype(np.float32)
    for x in itens:
        cor = np.array(CORES[grupo[x["termo"]]], np.float32)
        sob[x["_m"]] = 0.45 * sob[x["_m"]] + 0.55 * cor
    out = sob.clip(0, 255).astype(np.uint8)
    if prato is not None:
        cs, _ = cv2.findContours(prato.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (255, 255, 255), max(1, out.shape[1] // 300))
    h, w = out.shape[:2]
    e = W / w
    pil = Image.fromarray(out).resize((W, max(1, round(h * e))), Image.BILINEAR)
    d = ImageDraw.Draw(pil)
    for x in itens:
        ys, xs = np.where(x["_m"])
        if len(xs):
            px, py = float(xs.mean()) * e, float(ys.mean()) * e
            d.text((px - 15, py - 5), x["termo"][:14], fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return pil


# ------------------------------------------------------------------ 5. fotos de cada base
def fotos_n5k(a, ids):
    G.baixa(ids, "rgb.png", a.cache)
    return [dict(base="n5k", id=d, arquivo=f"{d}/rgb.png", caminho=os.path.join(a.cache, "img", d, "rgb.png")) for d in ids]


def fotos_dfoodtj(a, L):
    import teste_r8 as T8
    a8 = SimpleNamespace(cache=a.cache, imagens=None, fotos=a.fotos, espelho_fotos=a.espelho_fotos, prazo=a.prazo)
    L, F, info = T8.baixa_pngs(a8, L)
    return [dict(base="dfoodtj", id=r["dish_id"], arquivo=r["arquivo"], caminho=F[r["dish_id"]]["caminho"], sha256=F[r["dish_id"]]["sha256"],
                 periodo=r["periodo"], paciente=r["paciente"]) for r in L], info


def fotos_fpb(a, L):
    raiz = os.path.join(a.cache, "fpb9")
    d = dict(chave="fpb", ids=[r["dish_id"] for r in L], arquivos=[r["arquivo"] for r in L], url=[""] * len(L),
             bytes=[int(r["bytes"]) for r in L], sha256=[r["sha256"] for r in L])
    if a.imagens_fpb:
        raiz = a.imagens_fpb
    else:
        G.baixa_fotos4(d, raiz, a.prazo)
    return [dict(base="fpb", id=r["dish_id"], arquivo=r["arquivo"], caminho=os.path.join(raiz, r["arquivo"]), sha256=r["sha256"]) for r in L]


def amostra_folhas(fotos, base, n=40):
    ids = [f["id"] for f in fotos if f["base"] == base and (base != "dfoodtj" or f.get("periodo") == PERIODO_FOLHAS)]
    rng = np.random.default_rng(9)
    return set(rng.choice(sorted(ids), min(n, len(ids)), replace=False).tolist()) if ids else set()


def grava_folhas(a, base, tiles):
    """Folhas de conferência (20 por folha) e o CSV com número → id, para a régua de 18.3."""
    from PIL import Image, ImageDraw
    ords = sorted(tiles)
    tab = []
    for k0 in range(0, len(ords), 20):
        parte = ords[k0:k0 + 20]
        th = max(t.size[1] for _, t, _ in (tiles[i] for i in parte))
        folha = Image.new("RGB", (4 * 404, math.ceil(len(parte) / 4) * (th + 18) + 22), (255, 255, 255))
        dr = ImageDraw.Draw(folha)
        x = 2
        for g, c in CORES.items():          # legenda
            dr.rectangle([x, 4, x + 12, 16], fill=c); dr.text((x + 16, 4), g, fill=(0, 0, 0)); x += 16 + 8 * len(g) + 14
        for p, i in enumerate(parte):
            id_, t, ln = tiles[i]
            num = k0 + p + 1
            xx, yy = (p % 4) * 404, 22 + (p // 4) * (th + 18)
            folha.paste(t, (xx, yy))
            dr.text((xx + 2, yy + t.size[1] + 2), f"{num}. {id_[:28]} {ln['prato_metodo']} {ln['n_itens']} itens", fill=(0, 0, 0))
            tab.append(dict(numero=num, folha=k0 // 20 + 1, id=id_, prato_metodo=ln["prato_metodo"], n_itens=ln["n_itens"], comida_coberta="", proteina_grupo_errado=""))
        folha.save(os.path.join(a.saida, f"folha_r9_{base}_{k0 // 20 + 1}.jpg"), quality=80)
    with open(os.path.join(a.saida, f"folha_r9_{base}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["numero", "folha", "id", "prato_metodo", "n_itens", "comida_coberta", "proteina_grupo_errado"], lineterminator="\n")
        w.writeheader(); w.writerows(tab)


# ------------------------------------------------------------------ 6. principal
def principal(a):
    t_ini = time.time()
    a.prazo = t_ini + a.orcamento_h * 3600
    os.makedirs(a.saida, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
    a.hf = a.hf or os.path.join(a.cache, "hf")
    LIMIAR.update(s=a.branco_s, v=a.branco_v)
    import torch
    import cv2
    import transformers
    devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() and not a.cpu else [torch.device("cpu")]
    man = dict(formato=FORMATO, inicio=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv[1:], smoke=a.smoke, torch=torch.__version__,
               transformers=transformers.__version__, opencv=cv2.__version__, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               nota="18.10 (d): só máscaras e geometria em pixels; nenhum rótulo lido, nada pontuado. A escala, os parâmetros e a leitura são da CPU (eval/geom).",
               regras=dict(nota_item=NOTA_ITEM, nms_item=NMS_ITEM, lado_max=LADO_MAX, branco=dict(S_menor=LIMIAR["s"], V_maior=LIMIAR["v"],
                                                                                  pre_registrado=LIMIAR == dict(s=BRANCO_S, v=BRANCO_V)), resto_min=RESTO_MIN,
                           padroes_xadrez=[f"{x}x{y}" for x, y in PADROES_XADREZ], sam=f"{SAM}@{REV_SAM[:12]}", sha256_sam=SHA_SAM,
                           visor="não mascarado nas tabelas; só nas folhas do DFood-TJ (regra 8b)"))
    tempos, bases = {}, {}

    def grava():
        man.update(tempos_seg=tempos, bases=bases, duracao_min=round((time.time() - t_ini) / 60, 1))
        with open(os.path.join(a.saida, "geom_r9.json"), "w") as f:
            json.dump(man, f, indent=1, ensure_ascii=False, default=str)
        open(os.path.join(a.saida, "geom_r9.log"), "w").write("\n".join(REGISTRO) + "\n")
    arq_v = arq_vocabulario()
    termos, consultas_it, grupo = le_vocabulario(arq_v)
    man["vocabulario"] = dict(arquivo="vocabulario_r9.csv", sha256=sha256(arq_v), termos=len(termos))
    consultas = list(E.CONSULTAS) + consultas_it + CONSULTAS_VISOR
    quer = [b for b in (a.bases.split(",") if a.bases else BASES)]
    listas = os.path.join(AQUI, "listas")
    fotos = []
    t = time.time()
    for b in quer:
        try:
            if b == "n5k":
                L = le_csv(os.path.join(listas, "n5k_r9.csv"), COLS_N5K)
                assert a.smoke or len(L) == 2281, f"n5k_r9.csv com {len(L)} pratos (esperado 2.281)"
                L = L[:a.smoke] if a.smoke else L
                fb = fotos_n5k(a, [r["dish_id"] for r in L])
            elif b == "dfoodtj":
                import teste_r8 as T8
                arq_t = a.teste or os.path.join(listas, "teste_r8_dfoodtj.csv")
                if not os.path.exists(arq_t):
                    bases[b] = dict(situacao="fora: sem teste_r8_dfoodtj.csv (só no notebook do push)"); continue
                L, sha_t = T8.le_teste(arq_t, a.smoke)
                if a.smoke and a.ids_dfoodtj:
                    L = [r for r in T8.le_teste(arq_t, 10 ** 6)[0] if r["dish_id"] in a.ids_dfoodtj.split(",")]
                fb, info = fotos_dfoodtj(a, L)
                bases[b] = dict(tabela_sha256=sha_t, confere_16_6=sha_t == T8.SHA_TESTE, download=info)
            elif b == "fpb":
                arq_f = os.path.join(listas, "fpb_r9.csv")
                if not os.path.exists(arq_f):
                    bases[b] = dict(situacao="fora: sem fpb_r9.csv (CC BY-NC 4.0; só no notebook do push)"); continue
                L = le_csv(arq_f, COLS_FPB)
                assert a.smoke or len(L) == 312, f"fpb_r9.csv com {len(L)} fotos (esperado 312)"
                L = L[:a.smoke] if a.smoke else L
                fb = fotos_fpb(a, L)
            else:
                raise SystemExit(f"base desconhecida: {b}")
        except SystemExit:
            raise
        except Exception as ex:     # noqa: BLE001 — uma base que não baixa fica de fora, registrada
            bases[b] = dict(situacao=f"fora: {type(ex).__name__}: {ex}", traceback=traceback.format_exc()[-1500:])
            diz(f"{b}: fora ({type(ex).__name__}: {ex})"); continue
        bases.setdefault(b, {}).update(situacao="ok", fotos=len(fb))
        folhas = amostra_folhas(fb, b)
        for f in fb:
            f["folha"] = f["id"] in folhas
        fotos += fb
        diz(f"{b}: {len(fb)} fotos prontas")
    tempos["download"] = round(time.time() - t, 1)
    for i, f in enumerate(fotos):
        f["i"] = i
    grava()
    t = time.time()
    mods = [Modelos(a, d, consultas, usa_sam=not a.sem_sam) for d in devs]
    v = mods[0]
    sam_ok = all(m.sam is not None for m in mods)
    versoes = (f"owlv2 {v.versao_det} | sam {SAM}@{REV_SAM[:12]}{'' if sam_ok else ' (RESERVA: elipse na caixa)'} | transformers {transformers.__version__} | "
               f"torch {torch.__version__} | opencv {cv2.__version__}")
    man.update(versoes=versoes, sam_carregado=sam_ok, erros_sam=[m.erro_sam for m in mods if m.erro_sam], consultas=consultas)
    tempos["modelos"] = round(time.time() - t, 1)
    diz(f"modelos em {len(devs)} dispositivo(s): {versoes}")
    linhas, itens, tiles, falhas = {}, {}, {}, {}
    trava = threading.Lock()
    t = time.time()

    def trabalha(k):
        md = mods[k]
        meus = fotos[k::len(mods)]
        for j in range(0, len(meus), 8):
            if time.time() > a.prazo:
                return
            lote = []
            for f in meus[j:j + 8]:
                try:
                    img, gir = E.abre_canonica(f["caminho"])
                    if "sha256" not in f:
                        f["sha256"] = sha256(f["caminho"])
                    lote.append((f, img, gir))
                except Exception as ex:     # noqa: BLE001
                    with trava:
                        falhas[f["i"]] = f"{type(ex).__name__}: {ex}"
            if not lote:
                continue
            dets = md.detecta([x[1] for x in lote])
            for (f, img, gir), dt in zip(lote, dets):
                try:
                    ln, its, tile = processa_foto(md, f, img, gir, dt, (termos, grupo), consultas)
                    ln["versoes"] = versoes
                    with trava:
                        linhas[f["i"]], itens[f["i"]] = ln, its
                        if tile is not None:
                            tiles.setdefault(f["base"], {})[f["i"]] = (f["id"], tile, ln)
                except Exception as ex:     # noqa: BLE001
                    with trava:
                        falhas[f["i"]] = f"{type(ex).__name__}: {ex} | {traceback.format_exc()[-600:]}"
            with trava:
                feitas = len(linhas)
            if feitas and feitas % 400 < 8:
                diz(f"máscaras: {feitas} de {len(fotos)} fotos ({time.time() - t:.0f}s)")
    ths = [threading.Thread(target=trabalha, args=(k,)) for k in range(len(mods))]
    [x.start() for x in ths]
    [x.join() for x in ths]
    tempos["mascaras"] = round(time.time() - t, 1)
    # tabelas, na ordem das bases e das listas
    with open(os.path.join(a.saida, "fotos_geom.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, COLS_FOTOS, lineterminator="\n", extrasaction="ignore"); w.writeheader()
        for i in sorted(linhas):
            w.writerow(linhas[i])
    with gzip.open(os.path.join(a.saida, "itens_geom.csv.gz"), "wt", newline="", compresslevel=6) as fo:
        w = csv.DictWriter(fo, COLS_ITENS, lineterminator="\n", extrasaction="ignore"); w.writeheader()
        for i in sorted(itens):
            w.writerows(itens[i])
    for b, tl in tiles.items():
        grava_folhas(a, b, tl)
    for b in bases:
        if bases[b].get("situacao") != "ok":
            continue
        ii = [f["i"] for f in fotos if f["base"] == b]
        ln = [linhas[i] for i in ii if i in linhas]
        its = [x for i in ii if i in itens for x in itens[i]]
        n_it = np.array([l["n_itens"] for l in ln]) if ln else np.zeros(0)
        cont = lambda chave, L: {k: sum(1 for x in L if x[chave] == k) for k in sorted({x[chave] for x in L})}
        bases[b].update(processadas=len(ln), falhas={fotos[i]["id"]: falhas[i] for i in ii if i in falhas}, sem_tempo=len(ii) - len(ln) - sum(1 for i in ii if i in falhas),
                        prato_metodo=cont("prato_metodo", ln), itens=len(its), termos=cont("termo", its), origem_mascara=cont("origem_mascara", its),
                        itens_por_foto=dict(media=round(float(n_it.mean()), 2), p50=float(np.median(n_it)), p90=float(np.percentile(n_it, 90))) if len(n_it) else None,
                        folhas=len(tiles.get(b, {})))
        if b == "fpb":
            bases[b]["xadrez_detectado"] = sum(1 for l in ln if l["xadrez_detectado"] == 1)
        rf = np.array([float(l["resto_fracao_prato"]) for l in ln if l["resto_fracao_prato"] != ""])
        vv = np.array([float(l["livre_v_mediano"]) for l in ln if l["livre_v_mediano"] != ""])
        if len(rf):
            bases[b]["resto_fracao_prato"] = dict(p50=round(float(np.median(rf)), 3), p90=round(float(np.percentile(rf, 90)), 3))
        if len(vv):
            bases[b]["livre_v_mediano"] = dict(p10=round(float(np.percentile(vv, 10)), 3), p50=round(float(np.median(vv)), 3), abaixo_do_limiar=int((vv <= LIMIAR["v"]).sum()), n=len(vv))
        diz(f"{b}: {len(ln)} fotos, {len(its)} itens, prato {bases[b]['prato_metodo']}, itens/foto {bases[b]['itens_por_foto']}, falhas {len(bases[b]['falhas'])}, "
            f"sem tempo {bases[b]['sem_tempo']}" + (f", xadrez em {bases[b]['xadrez_detectado']}" if b == "fpb" else "")
            + f" | resto/prato {bases[b].get('resto_fracao_prato')}, V mediano do prato livre {bases[b].get('livre_v_mediano')}")
    man["saidas"] = {n: dict(bytes=os.path.getsize(os.path.join(a.saida, n)), sha256=sha256(os.path.join(a.saida, n))) for n in ("fotos_geom.csv", "itens_geom.csv.gz")}
    man["completo"] = all(v.get("situacao") == "ok" and not v.get("falhas") and not v.get("sem_tempo") for v in bases.values())
    diz(f"fim: {(time.time() - t_ini) / 60:.1f} min; saídas {man['saidas']}; completo: {man['completo']}. Nada foi pontuado.")
    grava()


# ------------------------------------------------------------------ 7. autoteste (sem rede nem GPU)
def autoteste():
    import cv2
    rng = np.random.default_rng(0)
    for _ in range(20):         # RLE: ida e volta, começando pelos zeros, em ordem de coluna
        m = rng.random((rng.integers(1, 30), rng.integers(1, 30))) > 0.6
        assert (rle_decodifica(rle_codifica(m), *m.shape) == m).all()
    m = np.zeros((3, 4), bool); m[0, 0] = True; m[2, 1] = True
    assert rle_codifica(m) == "0 1 4 1 6" and rle_codifica(np.zeros((2, 2), bool)) == "4" and rle_codifica(np.ones((2, 2), bool)) == "0 4"
    pr = np.zeros((100, 120), bool); cv2.ellipse(pr.view(np.uint8), (60, 50), (40, 30), 0, 0, 360, 1, -1)
    pr2 = pr.copy(); pr2[45:55, 55:65] = False                     # um buraco (comida) é preenchido
    pp, c = prato_preenchido(pr2)
    assert pp.sum() == pr.sum()
    el = [float(x) for x in elipse_de(c).split()]
    assert abs(el[0] - 60.5) < 1 and abs(el[1] - 50.5) < 1 and abs(el[2] - 81) < 2 and abs(el[3] - 61) < 2 and el[2] >= el[3]
    el2 = [float(x) for x in elipse_de(c, 2.0, 2.0).split()]
    assert abs(el2[2] - 2 * el[2]) < 0.5
    rot = np.zeros((100, 100), np.uint8); cv2.ellipse(rot, (50, 50), (40, 20), 90, 0, 360, 1, -1)       # eixo maior na vertical
    elr = [float(x) for x in elipse_de(maior_contorno(rot)).split()]
    assert elr[2] > elr[3] and abs(elr[4] - 90) < 2, elr
    a1 = np.zeros((100, 120), bool); a1[30:60, 30:70] = True
    a2 = np.zeros((100, 120), bool); a2[40:90, 50:110] = True
    fin, fr = resolve_itens([a1, a2], [0.3, 0.6], pr)
    assert not (fin[0] & fin[1]).any() and (fin[1] == (a2 & pr)).all() and abs(fr[1] - (a2 & pr).sum() / a2.sum()) < 1e-9 and fr[0] == 1.0
    fin0, fr0 = resolve_itens([a1], [0.3], None)
    assert fr0 == [None] and (fin0[0] == a1).all()
    img = np.zeros((100, 120, 3), np.uint8); img[:] = (240, 240, 240); img[40:60, 40:60] = (180, 60, 40)      # prato branco, uma comida
    r = resto_de(pr, [], img)
    assert r[45, 45] and not r[10, 10] and r.sum() == 400
    img[20:22, 50:52] = (180, 60, 40)                               # 4 pixels: menos de 0,5% do prato -> fora
    assert resto_de(pr, [], img).sum() == 400
    caixas = [[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]]
    assert nms_por_termo(caixas, [0.5, 0.4, 0.3, 0.2], [0, 0, 1, 0]) == [0, 2, 3]
    tab = np.full((300, 400), 255, np.uint8)                      # xadrez sintético: 8 × 6 quadrados de 25 px (7 × 5 cantos internos)
    for i in range(6):
        for j in range(8):
            if (i + j) % 2 == 0:
                tab[50 + i * 25:75 + i * 25, 100 + j * 25:125 + j * 25] = 0
    ok, passo, pad = xadrez(tab)
    assert ok == 1 and abs(passo - 25) < 0.5 and pad == "7x5", (ok, passo, pad)
    assert xadrez(np.full((100, 100), 128, np.uint8))[0] == 0
    m = elipse_inscrita([10, 20, 50, 40], 60, 60)
    assert m[30, 30] and not m[0, 0]
    if os.path.exists(arq_vocabulario()):
        V = le_vocabulario(arq_vocabulario())
        assert set(V[2].values()) <= set(CORES) and "boiled_egg_in_shell" in V[0], "vocabulário: grupo sem cor ou sem o ovo da escala"
    print("autoteste da nona execução: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "saida_r9")
    ap.add_argument("--cache", default="/tmp/nutrelle_cache_r9")
    ap.add_argument("--hf", help="cache do Hugging Face (padrão: <cache>/hf)")
    ap.add_argument("--bases", help="n5k,dfoodtj,fpb (padrão: as três)")
    ap.add_argument("--teste", help="tabela do teste DFood-TJ (padrão: listas/teste_r8_dfoodtj.csv)")
    ap.add_argument("--fotos", help="tabela das fotos do DFood-TJ (padrão: listas/fotos_r8_dfoodtj.csv)")
    ap.add_argument("--espelho-fotos", help="pasta do espelho das fotos do DFood-TJ (padrão: procurada em /kaggle/input)")
    ap.add_argument("--imagens-fpb", help="pasta com as fotos do FPB já em disco (teste; o caminho da lista abaixo dela)")
    ap.add_argument("--ids-dfoodtj", help="com --smoke: estes dish_id do DFood-TJ, separados por vírgula")
    ap.add_argument("--orcamento-h", type=float, default=1.5)
    ap.add_argument("--smoke", type=int, default=0, help="só as N primeiras fotos de cada base (não confere os tamanhos das listas)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--sem-sam", action="store_true", help="sem o SAM (a reserva: elipse inscrita na caixa)")
    ap.add_argument("--branco-s", type=float, default=BRANCO_S, help="branco do prato: S abaixo disto (18.10 d: 0,15)")
    ap.add_argument("--branco-v", type=float, default=BRANCO_V, help="branco do prato: V acima disto (18.10 d emendado: 0,55)")
    ap.add_argument("--autoteste", action="store_true")
    a = ap.parse_args()
    if a.autoteste:
        return autoteste()
    principal(a)


if __name__ == "__main__":
    main()
