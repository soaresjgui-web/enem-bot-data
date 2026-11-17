#!/usr/bin/env python3
"""Décima execução (18.12): só segmentação, com um segmentador treinado em comida. Nada é pontuado; nenhum rótulo é lido.

    python3 geom_r10.py --saida <pasta> [--cache <pasta>] [--bases n5k,dfoodtj,fpb] [--espelho-fotos <pasta>]
                        [--smoke N] [--sintetico <pasta>] [--cpu] [--orcamento-h H] [--autoteste]

O segmentador (18.12 a, escolhido sem nenhuma saída nas nossas fotos): SETR-MLA ViT-B/16, o modelo oficial do
FoodSeg103-Benchmark-v1 (mIoU publicado 45,1), `iter_80000.pth` do espelho `PhilippMuellerR/FoodSAMWeights` (revisão e
sha256 fixados), carregado por uma reimplementação em PyTorch puro (timm ViT-B/16 a 768 + vizinhança e cabeça MLA), com a
conferência estrita das chaves. Inferência como a do mmseg 0.11 do benchmark: redimensiona mantendo a proporção para
(2049, 1025), janelas de 768 com passo 512, normalização da ImageNet; fp16 na GPU. O SAM fica desligado.

Bases, listas e downloads: os da nona (18.10 d), mais a população da leitura 2 do DFood-TJ (457 refeições, só ids;
só no notebook do push), que define a amostra nova das folhas.

Em cada foto (orientação do EXIF e retrato girado, como na oitava e na nona):
 1. uma passada do OWLv2 da etapa 0 com todas as consultas: o prato pela regra só de prato e tigela de 14.3 (elipse e
    `sem_prato` de reserva); no FPB, os recipientes ("a plate", "a bowl", "a food container", "a cup"; nota >= 0,15;
    NMS 0,5; caixa com mais de 60% da foto não é recipiente; sem recipiente, a regra do prato); "a checkerboard" (FPB) e
    o mostrador da balança (DFood-TJ, só para pintar nas folhas);
 2. a máscara de cada prato ou recipiente é a ELIPSE INSCRITA NA CAIXA (sem o SAM, desligado em 18.12 a): `prato_elipse`
    é essa elipse. Recipientes sobrepostos: o pixel fica com o menor;
 3. o SETR-MLA dá a classe de cada pixel (104 = fundo + 103 do FoodSeg103), só dentro do prato;
 4. o branco do prato, foto a foto (18.12 b): anel de 0,88 a 0,98 dos semieixos da elipse, menos o que o segmentador
    marcou como comida; mediana e MAD de cada canal em Lab; um pixel do prato que o segmentador NÃO chama de comida é prato
    se a distância dele à mediana for no máximo max(8, 3 × 1,4826 × |MAD|), com teto de ΔE 20 (emenda de 18.12 b; o que o
    segmentador chama de comida fica comida). Com menos de 500 pixels no anel, o limiar fixo (S < 0,15 e V > 0,55) e a foto
    marcada (`branco_reserva`);
 5. um item por componente conexo de cada classe (`termo = fs103:<classe>`), sem os de menos de 0,3% do prato; regra de
    sanidade: componente com mais de 60% do prato só vale se for a única classe de comida no prato, senão sai e a foto é
    marcada (`sanidade_falhou`); `resto` = prato − comida − pixels de prato, sem componentes de menos de 0,5% do prato;
 6. FPB, o xadrez (emenda de 18.12 c): `cv2.findChessboardCorners` comum (limiar adaptativo + normalização) na foto
    inteira, com 3×4 e 4×3 primeiro e depois os outros padrões; reserva: `findChessboardCornersSB` na foto ampliada 2
    vezes dentro da caixa de "a checkerboard", aceito só com dispersão entre cantos vizinhos de até 5%;
 7. saídas: `fotos_geom.csv` e `itens_geom.csv.gz` com as colunas de 18.10 (d) (mais as marcas da décima),
    `recipientes_geom.csv`, folhas de conferência das amostras NOVAS (semente 20261118, 18.12 d) e, como relato, das
    amostras da nona; `geom_r10.json` e `geom_r10.log`. Nenhuma máscara densa em disco."""
import argparse, csv, gzip, hashlib, json, math, os, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import treino_gpu as G  # noqa: E402
import deteccao_r5 as E  # noqa: E402
import geom_r9 as G9  # noqa: E402

FORMATO = "nutrelle-geom/10"
SETR_REPO, SETR_REV, SETR_ARQ = "PhilippMuellerR/FoodSAMWeights", "454015fde30211cdd6a897183a9e51b1748218b0", "iter_80000.pth"
SETR_SHA = "f72bf120eb00dca469f292cc2bde956883f3b64d22768f203c17aea8eaf5001a"
SETR_ESCALA, SETR_JANELA, SETR_PASSO = (2049, 1025), 768, 512
MEDIA, DESVIO = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375)
# FoodSeg103, ordem oficial (category_id.txt do benchmark = id2label do conjunto no Hugging Face): o índice é a saída do modelo
CLASSES = ["background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine",
           "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple",
           "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig",
           "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab",
           "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice",
           "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce",
           "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas", "cabbage",
           "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom",
           "white button mushroom", "salad", "other ingredients"]
ITEM_MIN, SANIDADE, RESTO_MIN = 0.003, 0.60, 0.005
ANEL, MAD_K, DE_MIN, ANEL_MIN = (0.88, 0.98), 3 * 1.4826, 8.0, 500
RESERVA_S, RESERVA_V = 0.15, 0.55
CONSULTAS_RECIP = ["a plate", "a bowl", "a food container", "a cup"]
RECIP_MAX = 0.60
RECIP_DENTRO = 0.90     # recipiente com 90% ou mais da caixa dentro da caixa de outro recipiente é a comida dele, não um recipiente (do agente; 18.12 c não diz)
DE_MAX_PADRAO = 20.0    # teto da tolerância do branco (emenda de 18.12 b, 18/11/2026); --de-max muda
DE_MAX = dict(v=DE_MAX_PADRAO)
CONSULTA_XADREZ = "a checkerboard"
PADROES = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (4, 5), (5, 4), (5, 6), (6, 5), (6, 7), (7, 6), (7, 8), (8, 7)]
SB_DISPERSAO_MAX = 0.05
SEMENTE = 20261118
BASES = ["n5k", "dfoodtj", "fpb"]
COLS_FOTOS = G9.COLS_FOTOS[:G9.COLS_FOTOS.index("versoes") + 1] + [
    "n_recipientes", "sanidade_falhou", "branco_reserva", "branco_tolerancia_de", "comida_cor_de_prato_fracao", "resto_fracao_prato",
    "xadrez_metodo", "xadrez_dispersao", "amostra"]
COLS_ITENS = G9.COLS_ITENS + ["recipiente", "classe"]
COLS_RECIP = ["base", "id", "recipiente", "consulta", "nota", "caixa", "elipse", "rle", "mascara_largura", "mascara_altura", "area_px",
              "branco_metodo", "branco_tolerancia_de", "anel_px", "comida_cor_de_prato_fracao", "sanidade_falhou", "n_itens"]
COLS_POP = ["dish_id"]
CORES = dict(G9.CORES, bebida=(0, 200, 200))      # as 6 bebidas do FoodSeg103 (fora da massa de comida, 18.12 a)
REGISTRO = []


def diz(*p):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in p)
    print(s, flush=True)
    REGISTRO.append(s)


# ------------------------------------------------------------------ 1. vocabulário e amostras
def arq_vocabulario():
    p = os.path.join(AQUI, "listas", "vocabulario_r10.csv")
    return p if os.path.exists(p) else os.path.join(AQUI, "..", "..", "geom", "vocabulario.csv")


def grupos_fs103(arq):
    """{classe: grupo} das 103 classes, pelos termos `fs103:<classe>` do vocabulário (o nome casa com CLASSES)."""
    V = {r["termo"]: r for r in csv.DictReader(open(arq, newline=""))}
    g = {c: V[f"fs103:{c}"]["grupo"] for c in CLASSES[1:]}
    assert len(g) == 103 and set(g.values()) <= set(CORES), "vocabulário: classe do FoodSeg103 sem termo ou grupo sem cor"
    return g


def amostra_nova(ids, n=40):
    """18.12 (d): numpy.random.default_rng(20261118).choice(n, 40, replace=False) sobre a lista ordenada por id."""
    s = sorted(ids)
    return [s[i] for i in np.random.default_rng(SEMENTE).choice(len(s), min(n, len(s)), replace=False)] if s else []


# ------------------------------------------------------------------ 2. SETR-MLA em PyTorch puro
def _modelo_setr(ncls=104):
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    class CBR(nn.Sequential):
        def __init__(self, ci, co, k):
            super().__init__(nn.Conv2d(ci, co, k, padding=k // 2, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    class SETR_MLA(nn.Module):
        """VIT_MLA + VIT_MLAHead do SETR (mmseg 0.11 do FoodSeg103-Benchmark-v1): ViT-B/16 a 768; saídas dos blocos 6, 8, 10
        e 12 (norm_0..3, sem o token de classe); vizinhança MLA de cima para baixo; quatro cabeças 3×3 ampliadas 4 vezes."""

        def __init__(self):
            super().__init__()
            self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, img_size=SETR_JANELA, num_classes=0)
            self.norms = nn.ModuleList([nn.LayerNorm(768, eps=1e-6) for _ in range(4)])
            self.p1x1 = nn.ModuleList([CBR(768, 256, 1) for _ in range(4)])
            self.p3x3 = nn.ModuleList([CBR(256, 256, 3) for _ in range(4)])
            self.heads = nn.ModuleList([nn.Sequential(CBR(256, 128, 3), CBR(128, 128, 3)) for _ in range(4)])
            self.cls = nn.Conv2d(512, ncls, 3, padding=1)
            self.idx = (5, 7, 9, 11)

        def forward(self, x):
            v = self.vit
            x = v._pos_embed(v.patch_embed(x))
            outs = []
            for i, b in enumerate(v.blocks):
                x = b(x)
                if i in self.idx:
                    outs.append(x)
            n = int(math.sqrt(outs[0].shape[1] - 1))
            c = [self.norms[i](o[:, 1:]).transpose(1, 2).reshape(o.shape[0], -1, n, n) for i, o in enumerate(outs)]
            l = [self.p1x1[i](c[i]) for i in range(4)]
            p5 = l[3]; p4 = p5 + l[2]; p3 = p4 + l[1]; p2 = p3 + l[0]
            m = [self.p3x3[0](p2), self.p3x3[1](p3), self.p3x3[2](p4), self.p3x3[3](p5)]
            h = [F.interpolate(self.heads[i](m[i]), scale_factor=4, mode="bilinear", align_corners=True) for i in range(4)]
            y = self.cls(__import__("torch").cat(h, 1))
            return F.interpolate(y, size=(SETR_JANELA, SETR_JANELA), mode="bilinear", align_corners=False)
    return SETR_MLA()


def mapeia_setr(sd):
    """state_dict do mmseg -> nomes do modelo acima (as cabeças auxiliares e o conv_seg do treino ficam de fora)."""
    m = {}
    for k, v in sd.items():
        if k.startswith(("backbone.blocks.", "backbone.patch_embed.")) or k in ("backbone.cls_token", "backbone.pos_embed"):
            m["vit." + k[len("backbone."):]] = v
        elif k.startswith("backbone.norm_"):
            m[f"norms.{int(k.split('.')[1][-1])}.{k.split('.')[-1]}"] = v
        elif k.startswith("backbone.mla.mla_p"):
            nome = k.split(".")[2]
            m[f"{'p1x1' if nome.endswith('_1x1') else 'p3x3'}.{int(nome[5]) - 2}." + ".".join(k.split(".")[3:])] = v
        elif k.startswith("decode_head.mlahead.head"):
            a, b = k.split(".")[3], k.split(".")[4]
            m[f"heads.{int(k.split('.')[2][-1]) - 2}.{dict(zip('0134', ('0.0', '0.1', '1.0', '1.1')))[a]}.{b}"] = v
        elif k.startswith("decode_head.cls."):
            m["cls." + k.split(".")[-1]] = v
        elif not k.startswith(("auxiliary_head.", "decode_head.conv_seg.")):
            raise KeyError(f"chave inesperada no checkpoint do SETR: {k}")
    return m


def carrega_setr(dev, pasta):
    """O SETR-MLA do espelho fixado, com o sha256 conferido e as chaves conferidas (nenhuma faltando, nenhuma sobrando)."""
    import torch
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(SETR_REPO, SETR_ARQ, revision=SETR_REV, cache_dir=pasta)
    sha = os.path.basename(os.path.realpath(p))
    if sha != SETR_SHA:
        sha = G9.sha256(p)
    assert sha == SETR_SHA, f"pesos do SETR-MLA diferentes dos escolhidos em 18.12 ({sha})"
    ck = torch.load(p, map_location="cpu", weights_only=True)
    assert len(ck["meta"]["CLASSES"]) == 104 and ck["meta"]["iter"] == 80000, "metadados do checkpoint"
    m = _modelo_setr()
    falta, sobra = m.load_state_dict(mapeia_setr(ck["state_dict"]), strict=False)
    falta = [k for k in falta if not k.startswith(("vit.norm.", "vit.fc_norm.", "vit.head"))]
    assert not falta and not sobra, f"SETR: faltam {falta[:5]}, sobram {sobra[:5]}"
    m = m.to(dev).eval()
    if dev.type == "cuda":
        m = m.half()
    return m, dict(repo=SETR_REPO, revisao=SETR_REV, arquivo=SETR_ARQ, sha256=sha, mmseg=ck["meta"].get("mmseg_version"), iter=ck["meta"].get("iter"))


def segmenta(m, img, mh, mw, lote=4):
    """(classe uint8 [mh, mw], probabilidade float32 [mh, mw]) pela inferência deslizante do benchmark, na resolução de
    trabalho. `img`: uint8 [h, w, 3] RGB (a foto orientada)."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    H, W = img.shape[:2]
    e = min(SETR_ESCALA[0] / max(W, H), SETR_ESCALA[1] / min(W, H))
    w, h = int(W * e + 0.5), int(H * e + 0.5)
    dev = next(m.parameters()).device
    dt = next(m.parameters()).dtype
    x = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR), np.float32)
    x = torch.from_numpy(((x - np.array(MEDIA, np.float32)) / np.array(DESVIO, np.float32)).transpose(2, 0, 1))[None].to(dev)
    cs, st = SETR_JANELA, SETR_PASSO
    nh, nw = max(h - cs + st - 1, 0) // st + 1, max(w - cs + st - 1, 0) // st + 1
    jan = []
    for a in range(nh):
        for b in range(nw):
            y2, x2 = min(a * st + cs, h), min(b * st + cs, w)
            jan.append((max(y2 - cs, 0), max(x2 - cs, 0), y2, x2))
    acc = torch.zeros(1, 104, h, w, device=dev)
    cnt = torch.zeros(1, 1, h, w, device=dev)
    with torch.inference_mode():
        for k in range(0, len(jan), lote):
            pj = jan[k:k + lote]
            xs = torch.cat([F.pad(x[:, :, y1:y2, x1:x2], (0, cs - (x2 - x1), 0, cs - (y2 - y1))) for y1, x1, y2, x2 in pj]).to(dt)
            o = m(xs).float()
            for (y1, x1, y2, x2), oo in zip(pj, o):
                acc[0, :, y1:y2, x1:x2] += oo[:, :y2 - y1, :x2 - x1]
                cnt[0, :, y1:y2, x1:x2] += 1
        lg = F.interpolate(acc / cnt, size=(mh, mw), mode="bilinear", align_corners=False)
        p, c = lg.softmax(1).max(1)
    return c[0].to(torch.uint8).cpu().numpy(), p[0].cpu().numpy()


# ------------------------------------------------------------------ 3. prato, branco e itens: funções puras
def elipse_da_caixa(caixa, h, w):
    """(máscara bool [h, w] da elipse inscrita na caixa, raio normalizado [h, w]). Pixels em coordenadas de centro (+0,5)."""
    x0, y0, x1, y1 = caixa
    cx, cy, a, b = (x0 + x1) / 2, (y0 + y1) / 2, max(0.5, (x1 - x0) / 2), max(0.5, (y1 - y0) / 2)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx + 0.5 - cx) / a) ** 2 + ((yy + 0.5 - cy) / b) ** 2)
    return r <= 1.0, r


def fmt_elipse_caixa(caixa):
    """`cx cy eixo_maior eixo_menor angulo` da elipse inscrita (eixos inteiros; 0° se o eixo maior é horizontal)."""
    x0, y0, x1, y1 = caixa
    a, b = x1 - x0, y1 - y0
    return f"{(x0 + x1) / 2:.1f} {(y0 + y1) / 2:.1f} {max(a, b):.1f} {min(a, b):.1f} {0.0 if a >= b else 90.0:.1f}"


def lab(img):
    import cv2
    return cv2.cvtColor(np.ascontiguousarray(img).astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def branco_do_prato(img, prato, raio, comida):
    """18.12 (b). -> (máscara dos pixels de prato dentro de `prato`, método, tolerância ΔE ou None, pixels no anel)."""
    import cv2
    anel = prato & (raio >= ANEL[0]) & (raio <= ANEL[1]) & ~comida
    n = int(anel.sum())
    if n < ANEL_MIN:
        hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV)
        return prato & (hsv[..., 1] < RESERVA_S * 255) & (hsv[..., 2] > RESERVA_V * 255), "reserva_hsv", None, n
    L = lab(img)
    v = L[anel]
    med = np.median(v, 0)
    mad = np.median(np.abs(v - med), 0)
    tol = max(DE_MIN, MAD_K * float(np.linalg.norm(mad)))
    if DE_MAX["v"] is not None:
        tol = min(tol, DE_MAX["v"])
    d = np.linalg.norm(L - med, axis=-1)
    return prato & (d <= tol), "anel_lab", tol, n


def itens_do_prato(classe, prob, prato):
    """Componentes de cada classe de comida dentro do prato (o que o segmentador chama de comida fica comida, mesmo com a
    cor do prato: emenda de 18.12 b), sem os de menos de 0,3% do prato;
    regra de sanidade de 60%. -> (lista de (classe, máscara, nota média), sanidade_falhou, máscara da comida mantida)."""
    import cv2
    area = max(1, int(prato.sum()))
    comida = prato & (classe > 0)
    comps = []
    for c in np.unique(classe[comida]):
        n, rot = cv2.connectedComponents((comida & (classe == c)).astype(np.uint8), connectivity=8)
        for k in range(1, n):
            mk = rot == k
            if mk.sum() >= ITEM_MIN * area:
                comps.append((int(c), mk, float(prob[mk].mean())))
    classes = {c for c, _, _ in comps}
    falhou = False
    fica = []
    for c, mk, p in comps:
        if mk.sum() > SANIDADE * area and len(classes) > 1:
            falhou = True
            continue
        fica.append((c, mk, p))
    mant = np.zeros_like(prato)
    for _, mk, _ in fica:
        mant |= mk
    return fica, falhou, mant


def recipientes(cx_, nt, q, w, h):
    """FPB, 18.12 (c): caixas de recipiente (nota >= 0,15 numa das consultas, NMS 0,5, até 60% da foto). -> [(caixa, nota, consulta)]."""
    if not len(cx_):
        return []
    s = nt[:, q:q + len(CONSULTAS_RECIP)]
    k = np.where(s.max(1) >= E.NOTA_MIN)[0]
    c = np.clip(cx_[k], 0, [w, h, w, h])
    area = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0)
    ok = [i for i in range(len(k)) if 0 < area[i] <= RECIP_MAX * w * h]
    if not ok:
        return []
    fica = E.nms(c[ok], s[k[ok]].max(1))
    cc = [c[ok][i] for i in fica]
    ar = [(b[2] - b[0]) * (b[3] - b[1]) for b in cc]

    def dentro(i, j):           # fração da caixa i dentro da caixa j
        a, b = cc[i], cc[j]
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / max(1e-9, ar[i])
    fica = [fica[i] for i in range(len(fica)) if not any(j != i and ar[j] > ar[i] and dentro(i, j) >= RECIP_DENTRO for j in range(len(fica)))]
    return [(tuple(float(v) for v in c[ok][i]), float(s[k[ok][i]].max()), CONSULTAS_RECIP[int(s[k[ok][i]].argmax())]) for i in fica]


def xadrez(cinza, caixa=None):
    """Emenda de 18.12 (c). -> (detectado, passo px, padrão 'CxL', método, dispersão)."""
    import cv2

    def grade(c, pc):
        g = c.reshape(pc[1], pc[0], 2)
        d = np.concatenate([np.linalg.norm(np.diff(g, axis=1), axis=-1).ravel(), np.linalg.norm(np.diff(g, axis=0), axis=-1).ravel()])
        return float(np.median(d)), float(d.std() / max(1e-9, np.median(d)))
    fl = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for pc in PADROES:
        ok, c = cv2.findChessboardCorners(cinza, pc, flags=fl)
        if ok:
            passo, disp = grade(c, pc)
            return 1, passo, f"{pc[0]}x{pc[1]}", "comum", disp
    if caixa is not None:           # reserva: SB, ampliada 2 vezes dentro da caixa de "a checkerboard"
        H, W = cinza.shape
        x0, y0, x1, y1 = caixa
        mx, my = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
        x0, y0, x1, y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            rec = cv2.resize(cinza[y0:y1, x0:x1], ((x1 - x0) * 2, (y1 - y0) * 2), interpolation=cv2.INTER_CUBIC)
            for pc in PADROES:
                ok, c = cv2.findChessboardCornersSB(rec, pc, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
                if ok:
                    passo, disp = grade(c, pc)
                    if disp <= SB_DISPERSAO_MAX:
                        return 1, passo / 2, f"{pc[0]}x{pc[1]}", "sb_caixa", disp
    return 0, None, "", "", None


# ------------------------------------------------------------------ 4. uma foto
def processa_foto(md, setr, f, img, girada, det_out, grupo, qs, tarefas_xadrez):
    import cv2
    h, w = img.shape[:2]
    s = min(1.0, G9.LADO_MAX / max(h, w))
    mw, mh = (w, h) if s >= 1 else (int(round(w * s)), int(round(h * s)))
    fx, fy = w / mw, h / mh
    work = img if (mw, mh) == (w, h) else cv2.resize(img, (mw, mh), interpolation=cv2.INTER_AREA)
    cx_, nt = det_out
    npr = nt[:, :len(E.CONSULTAS)]
    l0 = E.linha_deteccao(f["id"], f["base"], f["arquivo"], img, girada, f["sha256"], E.elipse(img),
                          (cx_, npr.max(1) if len(npr) else np.zeros(0), npr.argmax(1) if len(npr) else np.zeros(0, int)), md.versao_det, f["i"])
    lr = E.refaz_linha(l0, set(E.SO_PRATO))
    cp = None if lr["x0"] == "" else tuple(float(lr[c]) for c in ("x0", "y0", "x1", "y1"))
    metodo = lr["metodo"]
    recs = [(cp, float(lr["nota"]), metodo)] if cp else []
    if f["base"] == "fpb":
        rr = recipientes(cx_, nt, qs["recip"], w, h)
        if rr:
            recs, metodo = rr, "recipientes"
    classe, prob = segmenta(setr, img, mh, mw)
    para_w = lambda c: (c[0] / fx, c[1] / fy, c[2] / fx, c[3] / fy)
    tomado = np.zeros((mh, mw), bool)
    itens, linhas_r, folha_rec = [], [], []
    falhou_foto, reserva_foto, virou, resto_fr = False, False, [], []
    ordem = sorted(range(len(recs)), key=lambda k: (recs[k][0][2] - recs[k][0][0]) * (recs[k][0][3] - recs[k][0][1]))    # o menor primeiro
    prato_uniao = np.zeros((mh, mw), bool)
    for k in ordem:
        caixa, nota, consulta = recs[k]
        pm, raio = elipse_da_caixa(para_w(caixa), mh, mw)
        pm &= ~tomado
        tomado |= pm
        prato_uniao |= pm
        if not pm.any():
            continue
        comida_setr = pm & (classe > 0)
        eh_prato, bmet, tol, n_anel = branco_do_prato(work, pm, raio, comida_setr)
        fica, falhou, mant = itens_do_prato(classe, prob, pm)
        vir = float((comida_setr & eh_prato).sum()) / max(1, int(comida_setr.sum()))     # diagnóstico: comida com a cor do prato (fica comida)
        resto = G9.sem_pequenos(pm & ~mant & ~(eh_prato & ~comida_setr), RESTO_MIN * pm.sum())     # o teste de cor só vale fora da comida
        falhou_foto |= falhou
        reserva_foto |= bmet == "reserva_hsv"
        virou.append(vir)
        resto_fr.append(float(resto.sum()) / max(1, int(pm.sum())))
        n0 = len(itens)
        for c, mk, p in fica + ([("resto", resto, 0.0)] if resto.any() else []):
            ys, xs = np.where(mk)
            ct = G9.maior_contorno(mk)
            itens.append(dict(base=f["base"], id=f["id"], termo="resto" if c == "resto" else f"fs103:{CLASSES[c]}", consulta="",
                              nota=f"{p:.4f}", caixa=G9.fmt_caixa((xs.min() * fx, ys.min() * fy, (xs.max() + 1) * fx, (ys.max() + 1) * fy)),
                              rle=G9.rle_codifica(mk), mascara_largura=mw, mascara_altura=mh, area_px=f"{int(mk.sum()) * (w / mw) ** 2:.1f}",
                              elipse=G9.elipse_de(ct, fx, fy) if ct is not None else "", origem_mascara="derivada" if c == "resto" else "setr_mla",
                              dentro_prato="1.0000", recipiente=k, classe="" if c == "resto" else c, _m=mk))
        linhas_r.append(dict(base=f["base"], id=f["id"], recipiente=k, consulta=consulta, nota=f"{nota:.4f}", caixa=G9.fmt_caixa(caixa),
                             elipse=fmt_elipse_caixa(caixa), rle=G9.rle_codifica(pm), mascara_largura=mw, mascara_altura=mh,
                             area_px=f"{int(pm.sum()) * (w / mw) ** 2:.1f}", branco_metodo=bmet, branco_tolerancia_de="" if tol is None else f"{tol:.2f}",
                             anel_px=n_anel, comida_cor_de_prato_fracao=f"{vir:.4f}", sanidade_falhou=int(falhou), n_itens=len(itens) - n0))
        folha_rec.append((pm, eh_prato))
    for j, x in enumerate(itens):
        x["item"] = j
    principal_k = max(range(len(recs)), key=lambda k: (recs[k][0][2] - recs[k][0][0]) * (recs[k][0][3] - recs[k][0][1])) if recs else None
    main = next((r for r in linhas_r if r["recipiente"] == principal_k), None)
    linha = dict(base=f["base"], id=f["id"], arquivo=f["arquivo"], sha256=f["sha256"], largura=w, altura=h, girada=int(girada), periodo=f.get("periodo", ""),
                 paciente=f.get("paciente", ""), prato_metodo=metodo, prato_caixa="" if principal_k is None else G9.fmt_caixa(recs[principal_k][0]),
                 prato_elipse="" if principal_k is None else fmt_elipse_caixa(recs[principal_k][0]),
                 prato_rle="" if not prato_uniao.any() else G9.rle_codifica(prato_uniao), mascara_largura=mw, mascara_altura=mh,
                 xadrez_detectado="", xadrez_passo_px="", xadrez_padrao="", n_itens=len(itens), livre_s_mediano="", livre_v_mediano="",
                 resto_fracao_prato=f"{np.mean(resto_fr):.4f}" if resto_fr else "", n_recipientes=len(linhas_r), sanidade_falhou=int(falhou_foto),
                 branco_reserva=int(reserva_foto), branco_tolerancia_de="" if main is None else main["branco_tolerancia_de"],
                 comida_cor_de_prato_fracao=f"{np.mean(virou):.4f}" if virou else "", xadrez_metodo="", xadrez_dispersao="", amostra=f.get("amostra", ""))
    if f["base"] == "fpb":
        sx = nt[:, qs["xadrez"]] if len(nt) else np.zeros(0)
        cxz = tuple(float(v) for v in cx_[int(sx.argmax())]) if len(sx) and sx.max() >= E.NOTA_MIN else None
        tarefas_xadrez.append((f["i"], cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY), cxz))
    tile = None
    if f.get("amostra"):
        tile = folha_tile(work, folha_rec, itens, grupo, f, linha, cx_, nt, qs["visor"], (fx, fy), cp)
    for x in itens:
        del x["_m"]
    return linha, itens, linhas_r, tile


def folha_tile(work, recs, itens, grupo, f, linha, cx_, nt, q_visor, esc, cp, W=400):
    """Miniatura: comida colorida pelo grupo, `resto` em listras, contorno de cada prato em branco, o nome de cada item.
    No DFood-TJ, o mostrador pintado pela regra da 8b (só na folha)."""
    from PIL import Image, ImageDraw
    import cv2
    im = work.copy()
    if f["base"] == "dfoodtj" and len(nt):
        import teste_r8 as T8
        nv = nt[:, q_visor:q_visor + len(G9.CONSULTAS_VISOR)]
        cv = T8.filtra_visor(cx_, nv.max(1), nv.argmax(1), work.shape[1] * esc[0], work.shape[0] * esc[1], cp)
        fica = [[c["caixa"][0] / esc[0], c["caixa"][1] / esc[1], c["caixa"][2] / esc[0], c["caixa"][3] / esc[1]] for c in cv if c["destino"] == "mascarada"]
        p_w = None if cp is None else (cp[0] / esc[0], cp[1] / esc[1], cp[2] / esc[0], cp[3] / esc[1])
        im = T8.mascara_fora_do_prato(im, fica, E.cor_borda(im).astype(np.uint8), p_w)[0]
    sob = im.astype(np.float32)
    hh, ww = im.shape[:2]
    listras = ((np.add.outer(np.arange(hh), np.arange(ww)) // max(3, ww // 120)) % 2 == 0)
    for x in itens:
        m = x["_m"]
        if x["termo"] == "resto":
            m = m & listras
            cor = np.array(CORES["misto"], np.float32)
        else:
            cor = np.array(CORES[grupo[CLASSES[x["classe"]]]], np.float32)
        sob[m] = 0.45 * sob[m] + 0.55 * cor
    out = sob.clip(0, 255).astype(np.uint8)
    for pm, _ in recs:
        cs, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (255, 255, 255), max(1, ww // 300))
    e = W / ww
    pil = Image.fromarray(out).resize((W, max(1, round(hh * e))), Image.BILINEAR)
    d = ImageDraw.Draw(pil)
    for x in itens:
        if x["termo"] == "resto":
            continue
        ys, xs = np.where(x["_m"])
        d.text((float(xs.mean()) * e - 15, float(ys.mean()) * e - 5), x["termo"][6:20], fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return pil


def grava_folhas(a, nome, tiles, por=20):
    """Folhas (20 por folha) e o CSV para a régua de 18.3 (colunas de veredito vazias)."""
    from PIL import Image, ImageDraw
    ords = sorted(tiles, key=lambda t: t[0])
    tab = []
    for k0 in range(0, len(ords), por):
        parte = ords[k0:k0 + por]
        th = max(t.size[1] for _, t, _ in parte)
        folha = Image.new("RGB", (4 * 404, math.ceil(len(parte) / 4) * (th + 18) + 22), (255, 255, 255))
        dr = ImageDraw.Draw(folha)
        x = 2
        for g, c in CORES.items():
            dr.rectangle([x, 4, x + 12, 16], fill=c); dr.text((x + 16, 4), g, fill=(0, 0, 0)); x += 16 + 8 * len(g) + 14
        dr.text((x, 4), "listras = resto", fill=(0, 0, 0))
        for p, (id_, t, ln) in enumerate(parte):
            num = k0 + p + 1
            xx, yy = (p % 4) * 404, 22 + (p // 4) * (th + 18)
            folha.paste(t, (xx, yy))
            marca = ("SANIDADE " if ln["sanidade_falhou"] else "") + ("BRANCO_RESERVA" if ln["branco_reserva"] else "")
            dr.text((xx + 2, yy + t.size[1] + 2), f"{num}. {id_[:26]} {ln['prato_metodo']} {ln['n_itens']} it. {marca}", fill=(0, 0, 0))
            tab.append(dict(numero=num, folha=k0 // por + 1, id=id_, prato_metodo=ln["prato_metodo"], n_recipientes=ln["n_recipientes"], n_itens=ln["n_itens"],
                            sanidade_falhou=ln["sanidade_falhou"], branco_reserva=ln["branco_reserva"], comida_coberta="", proteina_grupo_errado="",
                            prato_vazio_como_comida=""))
        folha.save(os.path.join(a.saida, f"folha_r10_{nome}_{k0 // por + 1}.jpg"), quality=80)
    with open(os.path.join(a.saida, f"folha_r10_{nome}.csv"), "w", newline="") as fo:
        wr = csv.DictWriter(fo, ["numero", "folha", "id", "prato_metodo", "n_recipientes", "n_itens", "sanidade_falhou", "branco_reserva", "comida_coberta",
                                 "proteina_grupo_errado", "prato_vazio_como_comida"], lineterminator="\n")
        wr.writeheader(); wr.writerows(tab)


# ------------------------------------------------------------------ 5. fotos sintéticas (teste sem as nossas fotos)
def fotos_sinteticas(pasta):
    """--sintetico: arquivos n5k_*, dftj_* e fpb_* de uma pasta, como se fossem das três bases (só para o teste)."""
    out = []
    for n in sorted(os.listdir(pasta)):
        b = "n5k" if n.startswith("n5k_") else "dfoodtj" if n.startswith("dftj_") else "fpb" if n.startswith("fpb_") else None
        if b:
            out.append(dict(base=b, id=os.path.splitext(n)[0], arquivo=n, caminho=os.path.join(pasta, n), sha256=G9.sha256(os.path.join(pasta, n)),
                            **({"periodo": "2020-2023", "paciente": "p0"} if b == "dfoodtj" else {})))
    return out


# ------------------------------------------------------------------ 6. principal
def principal(a):
    t_ini = time.time()
    a.prazo = t_ini + a.orcamento_h * 3600
    os.makedirs(a.saida, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
    a.hf = a.hf or os.path.join(a.cache, "hf")
    DE_MAX["v"] = a.de_max
    import torch
    import cv2
    import transformers
    import timm
    devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() and not a.cpu else [torch.device("cpu")]
    man = dict(formato=FORMATO, inicio=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv[1:], smoke=a.smoke, sintetico=bool(a.sintetico), torch=torch.__version__,
               timm=timm.__version__, transformers=transformers.__version__, opencv=cv2.__version__,
               gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               nota="18.12: só máscaras; nenhum rótulo lido, nada pontuado. O SAM fica desligado; o prato é a elipse inscrita na caixa do OWLv2.",
               regras=dict(item_min=ITEM_MIN, sanidade=SANIDADE, resto_min=RESTO_MIN, anel=ANEL, mad_k=MAD_K, de_min=DE_MIN, anel_min=ANEL_MIN,
                           reserva_hsv=dict(S_menor=RESERVA_S, V_maior=RESERVA_V), recipientes=CONSULTAS_RECIP, recipiente_max=RECIP_MAX,
                           xadrez=dict(padroes=[f"{x}x{y}" for x, y in PADROES], sb_dispersao_max=SB_DISPERSAO_MAX), semente_amostras=SEMENTE,
                           prato="elipse inscrita na caixa do OWLv2 (regra só de prato e tigela; no FPB, cada recipiente)",
                           recipiente_dentro_de_outro=RECIP_DENTRO, de_max=DE_MAX["v"], de_max_da_emenda=DE_MAX["v"] == DE_MAX_PADRAO,
                           comida_fica_comida="o teste de cor do prato só vale fora do que o segmentador chama de comida (emenda de 18.12 b)"))
    tempos, bases = {}, {}

    def grava():
        man.update(tempos_seg=tempos, bases=bases, duracao_min=round((time.time() - t_ini) / 60, 1))
        with open(os.path.join(a.saida, "geom_r10.json"), "w") as fo:
            json.dump(man, fo, indent=1, ensure_ascii=False, default=str)
        open(os.path.join(a.saida, "geom_r10.log"), "w").write("\n".join(REGISTRO) + "\n")
    arq_v = arq_vocabulario()
    grupo = grupos_fs103(arq_v)
    man["vocabulario"] = dict(arquivo=os.path.basename(arq_v), sha256=G9.sha256(arq_v))
    listas = os.path.join(AQUI, "listas")
    arq_pop = os.path.join(listas, "dftj_populacao_r10.csv")
    pop = [r["dish_id"] for r in G9.le_csv(arq_pop, COLS_POP)] if os.path.exists(arq_pop) else None
    if pop is not None:
        man["populacao_dfoodtj"] = dict(arquivo="dftj_populacao_r10.csv", sha256=G9.sha256(arq_pop), n=len(pop))
    fotos = []
    t = time.time()
    if a.sintetico:
        fotos = fotos_sinteticas(a.sintetico)
        for b in BASES:
            bases[b] = dict(situacao="ok", fotos=sum(1 for f in fotos if f["base"] == b), sintetico=True)
    else:
        for b in (a.bases.split(",") if a.bases else BASES):
            try:
                if b == "n5k":
                    L = G9.le_csv(os.path.join(listas, "n5k_r9.csv"), G9.COLS_N5K)
                    assert a.smoke or len(L) == 2281
                    fb = G9.fotos_n5k(a, [r["dish_id"] for r in (L[:a.smoke] if a.smoke else L)])
                elif b == "dfoodtj":
                    import teste_r8 as T8
                    arq_t = os.path.join(listas, "teste_r8_dfoodtj.csv")
                    if not os.path.exists(arq_t):
                        bases[b] = dict(situacao="fora: sem teste_r8_dfoodtj.csv (só no notebook do push)"); continue
                    L, sha_t = T8.le_teste(arq_t, a.smoke)
                    fb, info = G9.fotos_dfoodtj(a, L)
                    bases[b] = dict(tabela_sha256=sha_t, confere_16_6=sha_t == T8.SHA_TESTE, download=info)
                elif b == "fpb":
                    arq_f = os.path.join(listas, "fpb_r9.csv")
                    if not os.path.exists(arq_f):
                        bases[b] = dict(situacao="fora: sem fpb_r9.csv (só no notebook do push)"); continue
                    L = G9.le_csv(arq_f, G9.COLS_FPB)
                    assert a.smoke or len(L) == 312
                    fb = G9.fotos_fpb(a, L[:a.smoke] if a.smoke else L)
                else:
                    raise SystemExit(f"base desconhecida: {b}")
            except SystemExit:
                raise
            except Exception as ex:     # noqa: BLE001
                bases[b] = dict(situacao=f"fora: {type(ex).__name__}: {ex}", traceback=traceback.format_exc()[-1500:])
                diz(f"{b}: fora ({type(ex).__name__}: {ex})"); continue
            bases.setdefault(b, {}).update(situacao="ok", fotos=len(fb))
            fotos += fb
            diz(f"{b}: {len(fb)} fotos prontas")
    tempos["download"] = round(time.time() - t, 1)
    # amostras das folhas: as novas (18.12 d) e, como relato, as da nona
    amostras = {}
    for b in BASES:
        ids = [f["id"] for f in fotos if f["base"] == b]
        if b == "dfoodtj" and pop is not None and not a.sintetico:
            fora = set(pop) - set(ids)
            assert not fora or a.smoke, f"{len(fora)} refeições da população fora das fotos baixadas"
            ids = [i for i in ids if i in set(pop)]
        amostras[(b, "nova")] = amostra_nova(ids)
        amostras[(b, "r9")] = sorted(G9.amostra_folhas([f for f in fotos if f["base"] == b], b))
    marca = {}
    for (b, tipo), ids in amostras.items():
        for i in ids:
            marca.setdefault((b, i), []).append(tipo)
    for f in fotos:
        f["amostra"] = "+".join(marca.get((f["base"], f["id"]), []))
    man["amostras"] = {f"{b}:{tipo}": ids for (b, tipo), ids in amostras.items()}
    fotos.sort(key=lambda f: (0 if "nova" in f["amostra"] else 1 if f["amostra"] else 2, BASES.index(f["base"])))      # as amostras primeiro
    for i, f in enumerate(fotos):
        f["i"] = i
    grava()
    t = time.time()
    consultas = list(E.CONSULTAS) + CONSULTAS_RECIP + [CONSULTA_XADREZ] + G9.CONSULTAS_VISOR
    qs = dict(recip=len(E.CONSULTAS), xadrez=len(E.CONSULTAS) + len(CONSULTAS_RECIP), visor=len(E.CONSULTAS) + len(CONSULTAS_RECIP) + 1)
    mods = [G9.Modelos(a, d, consultas, usa_sam=False) for d in devs]
    setrs = [carrega_setr(d, a.hf) for d in devs]
    man.update(setr=setrs[0][1], consultas=consultas,
               versoes=f"owlv2 {mods[0].versao_det} | setr-mla {SETR_REPO}@{SETR_REV[:12]} ({SETR_SHA[:12]}) | timm {timm.__version__} | torch {torch.__version__} | opencv {cv2.__version__}")
    tempos["modelos"] = round(time.time() - t, 1)
    diz(f"modelos em {len(devs)} dispositivo(s): {man['versoes']}")
    linhas, itens, recs, tiles, falhas = {}, {}, {}, {}, {}
    xq = []
    trava = threading.Lock()
    t = time.time()
    t_setr = []

    def trabalha(k):
        md, (setr, _) = mods[k], setrs[k]
        meus = fotos[k::len(mods)]
        for j in range(0, len(meus), 8):
            if time.time() > a.prazo:
                return
            lote = []
            for f in meus[j:j + 8]:
                try:
                    img, gir = E.abre_canonica(f["caminho"])
                    if "sha256" not in f:
                        f["sha256"] = G9.sha256(f["caminho"])
                    lote.append((f, img, gir))
                except Exception as ex:     # noqa: BLE001
                    with trava:
                        falhas[f["i"]] = f"{type(ex).__name__}: {ex}"
            if not lote:
                continue
            dets = md.detecta([x[1] for x in lote])
            for (f, img, gir), dt in zip(lote, dets):
                try:
                    t0 = time.time()
                    tx = []
                    ln, its, rr, tile = processa_foto(md, setr, f, img, gir, dt, grupo, qs, tx)
                    ln["versoes"] = man["versoes"]
                    with trava:
                        t_setr.append(time.time() - t0)
                        linhas[f["i"]], itens[f["i"]], recs[f["i"]] = ln, its, rr
                        xq.extend(tx)
                        if tile is not None:
                            for tipo in f["amostra"].split("+"):
                                tiles.setdefault((f["base"], tipo), []).append((f["id"], tile, ln))
                except Exception as ex:     # noqa: BLE001
                    with trava:
                        falhas[f["i"]] = f"{type(ex).__name__}: {ex} | {traceback.format_exc()[-800:]}"
            with trava:
                feitas = len(linhas)
            if feitas % 200 < 8:
                diz(f"segmentação: {feitas} de {len(fotos)} fotos ({time.time() - t:.0f}s; {np.mean(t_setr) if t_setr else 0:.2f} s/foto por dispositivo)")
    ths = [threading.Thread(target=trabalha, args=(k,)) for k in range(len(mods))]
    [x.start() for x in ths]
    [x.join() for x in ths]
    tempos["segmentacao"] = round(time.time() - t, 1)
    t = time.time()
    with ThreadPoolExecutor(4) as ex:          # o xadrez do FPB, na CPU
        res = list(ex.map(lambda q: (q[0], xadrez(q[1], q[2])), xq))
    for i, (ok, passo, pad, met, disp) in res:
        linhas[i].update(xadrez_detectado=ok, xadrez_passo_px="" if passo is None else f"{passo:.2f}", xadrez_padrao=pad, xadrez_metodo=met,
                         xadrez_dispersao="" if disp is None else f"{disp:.4f}")
    tempos["xadrez"] = round(time.time() - t, 1)
    ordem = sorted(linhas, key=lambda i: (BASES.index(fotos[i]["base"]), fotos[i]["id"]))
    with open(os.path.join(a.saida, "fotos_geom.csv"), "w", newline="") as fo:
        wr = csv.DictWriter(fo, COLS_FOTOS, lineterminator="\n", extrasaction="ignore"); wr.writeheader()
        for i in ordem:
            wr.writerow(linhas[i])
    with gzip.open(os.path.join(a.saida, "itens_geom.csv.gz"), "wt", newline="", compresslevel=6) as fo:
        wr = csv.DictWriter(fo, COLS_ITENS, lineterminator="\n", extrasaction="ignore"); wr.writeheader()
        for i in ordem:
            wr.writerows(itens[i])
    with open(os.path.join(a.saida, "recipientes_geom.csv"), "w", newline="") as fo:
        wr = csv.DictWriter(fo, COLS_RECIP, lineterminator="\n", extrasaction="ignore"); wr.writeheader()
        for i in ordem:
            wr.writerows(recs[i])
    for (b, tipo), tl in tiles.items():
        grava_folhas(a, f"{b}" if tipo == "nova" else f"{b}_amostra_r9", tl)
    for b in BASES:
        if bases.get(b, {}).get("situacao") != "ok":
            continue
        ii = [f["i"] for f in fotos if f["base"] == b]
        ln = [linhas[i] for i in ii if i in linhas]
        its = [x for i in ii if i in itens for x in itens[i]]
        n_it = np.array([l["n_itens"] for l in ln]) if ln else np.zeros(0)
        cont = lambda chave, L: {str(k): sum(1 for x in L if x[chave] == k) for k in sorted({x[chave] for x in L}, key=str)}
        vir = np.array([float(l["comida_cor_de_prato_fracao"]) for l in ln if l["comida_cor_de_prato_fracao"] != ""])
        rf = np.array([float(l["resto_fracao_prato"]) for l in ln if l["resto_fracao_prato"] != ""])
        tl = np.array([float(r["branco_tolerancia_de"]) for i in ii if i in recs for r in recs[i] if r["branco_tolerancia_de"] != ""])
        bases[b].update(processadas=len(ln), falhas={fotos[i]["id"]: falhas[i] for i in ii if i in falhas},
                        sem_tempo=len(ii) - len(ln) - sum(1 for i in ii if i in falhas), prato_metodo=cont("prato_metodo", ln), itens=len(its),
                        grupos={g: sum(1 for x in its if x["termo"] != "resto" and grupo[CLASSES[x["classe"]]] == g) for g in CORES},
                        resto=sum(1 for x in its if x["termo"] == "resto"), sanidade_falhou=sum(l["sanidade_falhou"] for l in ln),
                        branco_reserva=sum(l["branco_reserva"] for l in ln), n_recipientes=cont("n_recipientes", ln),
                        itens_por_foto=dict(media=round(float(n_it.mean()), 2), p50=float(np.median(n_it)), p90=float(np.percentile(n_it, 90))) if len(n_it) else None,
                        comida_cor_de_prato=dict(p50=round(float(np.median(vir)), 3), p90=round(float(np.percentile(vir, 90)), 3)) if len(vir) else None,
                        resto_fracao_prato=dict(p50=round(float(np.median(rf)), 3), p90=round(float(np.percentile(rf, 90)), 3)) if len(rf) else None,
                        tolerancia_de=dict(p50=round(float(np.median(tl)), 1), p90=round(float(np.percentile(tl, 90)), 1), acima_de_20=int((tl > 20).sum()), n=len(tl)) if len(tl) else None,
                        folhas={tipo: len(tiles.get((b, tipo), [])) for tipo in ("nova", "r9")},
                        amostra_nova_processada=sum(1 for i in amostras[(b, "nova")] if any(l["id"] == i for l in ln)))
        if b == "fpb":
            bases[b]["xadrez"] = dict(detectado=sum(1 for l in ln if l["xadrez_detectado"] == 1),
                                      por_padrao=cont("xadrez_padrao", [l for l in ln if l["xadrez_detectado"] == 1]),
                                      por_metodo=cont("xadrez_metodo", [l for l in ln if l["xadrez_detectado"] == 1]))
        diz(f"{b}: {len(ln)} fotos, {len(its)} itens, prato {bases[b]['prato_metodo']}, itens/foto {bases[b]['itens_por_foto']}, sanidade {bases[b]['sanidade_falhou']}, "
            f"branco reserva {bases[b]['branco_reserva']}, tolerância ΔE {bases[b]['tolerancia_de']}, comida com cor de prato {bases[b]['comida_cor_de_prato']}, resto/prato {bases[b]['resto_fracao_prato']}, "
            f"falhas {len(bases[b]['falhas'])}, sem tempo {bases[b]['sem_tempo']}" + (f", xadrez {bases[b]['xadrez']}" if b == "fpb" else ""))
    man["seg_por_foto_por_dispositivo"] = round(float(np.mean(t_setr)), 3) if t_setr else None
    man["saidas"] = {n: dict(bytes=os.path.getsize(os.path.join(a.saida, n)), sha256=G9.sha256(os.path.join(a.saida, n)))
                     for n in ("fotos_geom.csv", "itens_geom.csv.gz", "recipientes_geom.csv")}
    man["completo"] = all(v.get("situacao") == "ok" and not v.get("falhas") and not v.get("sem_tempo") for v in bases.values())
    diz(f"fim: {(time.time() - t_ini) / 60:.1f} min; {man['seg_por_foto_por_dispositivo']} s/foto por dispositivo; completo: {man['completo']}. Nada foi pontuado.")
    grava()


# ------------------------------------------------------------------ 7. autoteste (sem rede nem GPU)
def autoteste():
    import cv2
    assert len(CLASSES) == 104 and CLASSES[0] == "background" and CLASSES[66] == "rice" and CLASSES[24] == "egg"
    if os.path.exists(arq_vocabulario()):
        g = grupos_fs103(arq_vocabulario())
        assert g["rice"] == "amido" and g["pork"] == "proteina_animal"
    assert amostra_nova([f"x{i:03d}" for i in range(100)]) == amostra_nova([f"x{i:03d}" for i in reversed(range(100))]) and len(amostra_nova(["a"])) == 1
    pm, raio = elipse_da_caixa((10, 20, 110, 80), 100, 120)
    assert pm[50, 60] and not pm[0, 0] and abs(pm.sum() - math.pi * 50 * 30) < 60 and abs(raio[50, 109] - 0.99) < 0.02
    assert fmt_elipse_caixa((10, 20, 110, 80)) == "60.0 50.0 100.0 60.0 0.0" and fmt_elipse_caixa((0, 0, 10, 30)).endswith("90.0")
    # um prato branco com uma comida marrom: o anel acha o branco; comida que o segmentador não marcou vira resto
    img = np.full((100, 120, 3), 90, np.uint8)
    img[pm] = (235, 235, 230)
    img[40:60, 50:70] = (150, 80, 40)
    img[30:36, 30:36] = (20, 120, 30)
    classe = np.zeros((100, 120), np.uint8); classe[40:60, 50:70] = 66
    prob = np.full((100, 120), 0.9, np.float32)
    eh, met, tol, n = branco_do_prato(img, pm, raio, pm & (classe > 0))
    assert met == "anel_lab" and tol == DE_MIN and n >= ANEL_MIN and eh[25, 60] and not eh[50, 60] and not eh[33, 33]
    classe[20:26, 60:66] = 66                                   # arroz com a cor do prato: continua comida (emenda de 18.12 b)
    assert eh[22, 62]
    fica, falhou, mant = itens_do_prato(classe, prob, pm)
    assert sorted(int(m.sum()) for _, m, _ in fica) == [36, 400] and not falhou and mant[22, 62]
    resto = G9.sem_pequenos(pm & ~mant & ~(eh & ~(pm & (classe > 0))), RESTO_MIN * pm.sum())
    assert resto[33, 33] and resto.sum() == 36
    ruido = img.copy(); ruido[(raio >= ANEL[0]) & (raio <= ANEL[1]) & (np.indices(raio.shape).sum(0) % 2 == 0)] = (20, 20, 200)
    assert branco_do_prato(ruido, pm, raio, np.zeros_like(pm))[2] == DE_MAX_PADRAO      # anel misturado: a tolerância para no teto
    # sanidade: um componente com mais de 60% do prato e outra classe presente -> sai e marca
    cl2 = np.where(pm, 66, 0).astype(np.uint8); cl2[45:55, 20:30] = 47
    f2, falhou2, _ = itens_do_prato(cl2, prob, pm)
    assert falhou2 and [c for c, _, _ in f2] == [47]
    f3, falhou3, _ = itens_do_prato(np.where(pm, 66, 0).astype(np.uint8), prob, pm)
    assert not falhou3 and len(f3) == 1                         # a única classe: vale
    tiny = np.zeros_like(classe); tiny[50, 60] = 66
    assert itens_do_prato(tiny, prob, pm)[0] == []        # menos de 0,3% do prato
    # anel pequeno -> reserva HSV
    pp, rr = elipse_da_caixa((50, 40, 60, 50), 100, 120)
    assert branco_do_prato(img, pp, rr, np.zeros_like(pp))[1] == "reserva_hsv"
    # recipientes: NMS, limite de 60% da foto
    cx = np.array([[10, 10, 50, 50], [12, 12, 50, 50], [0, 0, 120, 100], [60, 10, 100, 40]], float)
    nt = np.zeros((4, 20)); nt[:, len(E.CONSULTAS) + 1] = [0.5, 0.4, 0.9, 0.3]
    r = recipientes(cx, nt, len(E.CONSULTAS), 120, 100)
    assert [x[0] for x in r] == [(10.0, 10.0, 50.0, 50.0), (60.0, 10.0, 100.0, 40.0)] and r[0][2] == "a bowl"
    cx2 = np.array([[10, 10, 90, 90], [30, 30, 60, 60]], float); nt2 = np.zeros((2, 20)); nt2[:, len(E.CONSULTAS) + 2] = [0.5, 0.6]
    assert [x[0] for x in recipientes(cx2, nt2, len(E.CONSULTAS), 120, 100)] == [(10.0, 10.0, 90.0, 90.0)]     # a caixa de dentro é comida
    # xadrez: 4 × 5 quadrados (3 × 4 cantos internos), como o do FPB
    tab = np.full((300, 300), 200, np.uint8)
    for i in range(5):
        for j in range(4):
            if (i + j) % 2 == 0:
                tab[60 + i * 30:90 + i * 30, 80 + j * 30:110 + j * 30] = 20
    ok, passo, pad, met, disp = xadrez(tab)
    assert ok == 1 and pad in ("3x4", "4x3") and abs(passo - 30) < 0.5 and met == "comum" and disp < 0.05, (ok, passo, pad, met, disp)
    assert xadrez(np.full((100, 100), 128, np.uint8))[0] == 0
    # SETR: o mapeamento das chaves cobre o modelo inteiro (com um state_dict falso, no formato do mmseg)
    import torch
    m = _modelo_setr()
    sd = {}
    for k, v in m.state_dict().items():
        if k.startswith("vit.") and not k.startswith(("vit.norm.", "vit.fc_norm.", "vit.head")):
            sd["backbone." + k[4:]] = v
        elif k.startswith("norms."):
            sd[f"backbone.norm_{k.split('.')[1]}.{k.split('.')[2]}"] = v
        elif k.startswith(("p1x1.", "p3x3.")):
            t, j = k.split(".")[:2]
            sd[f"backbone.mla.mla_p{int(j) + 2}{'_1x1' if t == 'p1x1' else ''}." + ".".join(k.split(".")[2:])] = v
        elif k.startswith("heads."):
            p = k.split(".")
            sd[f"decode_head.mlahead.head{int(p[1]) + 2}.{ {'0.0': '0', '0.1': '1', '1.0': '3', '1.1': '4'}[p[2] + '.' + p[3]]}.{p[4]}"] = v
        elif k.startswith("cls."):
            sd["decode_head." + k] = v
    sd["auxiliary_head.0.conv_seg.weight"] = torch.zeros(1)
    falta, sobra = m.load_state_dict(mapeia_setr(sd), strict=False)
    assert not [k for k in falta if not k.startswith(("vit.norm.", "vit.fc_norm.", "vit.head"))] and not sobra
    print("autoteste da décima execução: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "saida_r10")
    ap.add_argument("--cache", default="/tmp/nutrelle_cache_r10")
    ap.add_argument("--hf", help="cache do Hugging Face (padrão: <cache>/hf)")
    ap.add_argument("--bases", help="n5k,dfoodtj,fpb (padrão: as três)")
    ap.add_argument("--fotos", help="tabela das fotos do DFood-TJ (padrão: listas/fotos_r8_dfoodtj.csv)")
    ap.add_argument("--espelho-fotos", help="pasta do espelho das fotos do DFood-TJ (padrão: procurada em /kaggle/input)")
    ap.add_argument("--imagens-fpb", help="pasta com as fotos do FPB já em disco")
    ap.add_argument("--sintetico", help="teste: pasta com fotos sintéticas n5k_*, dftj_*, fpb_* (nenhuma foto nossa)")
    ap.add_argument("--orcamento-h", type=float, default=2.5)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--de-max", type=float, default=DE_MAX_PADRAO, help="teto da tolerância ΔE do branco do prato (emenda de 18.12 b: 20)")
    ap.add_argument("--autoteste", action="store_true")
    a = ap.parse_args()
    if a.autoteste:
        return autoteste()
    principal(a)


if __name__ == "__main__":
    main()
