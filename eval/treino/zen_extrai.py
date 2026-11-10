#!/usr/bin/env python3
"""Seção 8: extrai features congeladas das fotos ZENITAIS com orientações diedrais (D4),
resolução maior, tokens intermediários (para ajuste fino parcial) e variantes de escala.

Uso: python3 zen_extrai.py <dir_treino> <backbone> [--orient 8] [--tokens 9] [--variante base]
                           [--conj treino,aval,br] [--saida nome]
  backbones: siglip2_b16 (224×224, imagem inteira achatada, como nas seções 2 e 7),
             siglip2_b16_512 (384×512, proporção original), siglip_b16, vits16_in21k,
             tips_b14 / tips_s14 (336×448; TIPS do Google DeepMind, pesos em gs://tips_data).
  --orient N   nº de orientações D4 (ordem: original, 180°, espelho H, espelho V, 90°, 270°,
               transposta, antitransposta). As 4 primeiras preservam o formato do quadro.
  --tokens K   grava também os tokens na saída do bloco K (float16) das 4 primeiras orientações.
  --variante   base | zoom:<z> (z>1 aproxima: recorte central; z<1 afasta: mais mesa em volta)
               | escala:<semente> (z ~ U[0,7; 1,3] por imagem)
               | prato:<fração>[:<ruído>[:<semente>]] (recorte quadrado em que o prato ocupa a fração
                 dada do lado; ruído = erro relativo de enquadramento; usa <TD>/meta/zen_bbox.csv)
Grava <dir_treino>/feat/zen_<saida>.npz: ids, X[n, orient, dim] (float16) com
  dim = agregado ‖ média ‖ GeM(p=3) ‖ médias numa grade 2×2 dos tokens finais.
e, com --tokens, <dir_treino>/feat/zen_<saida>_tokL<K>.npy [n, 4, tokens, dim] + _ids.txt.
"""
import csv, glob, os, sys, time
import numpy as np, torch
from PIL import Image

SCR = "/tmp/claude-0/-home-user-enem-bot-data/59d3cb2b-a793-5c18-8c23-e9dab2604443/scratchpad"
PESOS = os.path.join(SCR, "treino", "pesos")
CFG = {  # nome: (arquitetura timm ou 'tips_*', arquivo, (H, W), média, desvio)
    "siglip2_b16": ("vit_base_patch16_siglip_224", "siglip2_b16_224.npz", (224, 224), 0.5, 0.5),
    "siglip2_b16_512": ("vit_base_patch16_siglip_512", "siglip2_b16_512.npz", (384, 512), 0.5, 0.5),
    "siglip_b16": ("vit_base_patch16_siglip_224", "webli_en_b16_224_63724782.npz", (224, 224), 0.5, 0.5),
    "vits16_in21k": ("vit_small_patch16_224", "S_16-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0.npz", (224, 288), 0.5, 0.5),
    "tips_b14": ("tips_base", "tips_b14.npz", (336, 448), 0.0, 1.0),
    "tips_s14": ("tips_small", "tips_s14.npz", (336, 448), 0.0, 1.0),
}


def orienta(x, k):
    """k-ésimo elemento do grupo diedral D4 aplicado a um tensor [..., H, W]."""
    if k == 0: return x
    if k == 1: return x.flip(-2, -1)
    if k == 2: return x.flip(-1)
    if k == 3: return x.flip(-2)
    t = x.transpose(-2, -1)
    return [t.flip(-2), t.flip(-1), t, t.flip(-2, -1)][k - 4]


def lista(td, conj):
    itens = []
    if "treino" in conj:
        with open(os.path.join(td, "meta", "zen_treino.csv")) as f:
            itens += [(r["dish_id"], os.path.join(td, "img", r["dish_id"], "rgb.png")) for r in csv.DictReader(f)]
    with open(os.path.join(td, "meta", "zen_aval.csv")) as f:
        for r in csv.DictReader(f):
            if "aval" in conj or r["split"] in conj:
                itens.append((r["dish_id"], os.path.join(td, "img", r["dish_id"], "rgb.png")))
    if "br" in conj:
        for p in sorted(glob.glob(os.path.join(SCR, "cego2", "lote*", "*"))):
            pid = os.path.splitext(os.path.basename(p))[0]
            if pid[0] in "bc":
                itens.append((pid, p))
    return itens


def _borda(a, m):
    """Aumenta a imagem com m pixels de 'mesa' em volta: cor mediana da moldura externa (3%) da foto.
    (Borda refletida criaria comida fantasma quando o prato encosta na margem.)"""
    h, w = a.shape[:2]
    k = max(2, int(0.03 * min(h, w)))
    cor = np.median(np.concatenate([a[:k].reshape(-1, 3), a[-k:].reshape(-1, 3), a[:, :k].reshape(-1, 3), a[:, -k:].reshape(-1, 3)]), 0)
    out = np.empty((h + 2 * m[0], w + 2 * m[1], 3), np.uint8); out[:] = cor.astype(np.uint8)
    out[m[0]:m[0] + h, m[1]:m[1] + w] = a
    return out


def zoom(im, z):
    """z > 1: câmera mais perto (recorte central de 1/z); z < 1: mais longe (mais 'mesa' em volta)."""
    w, h = im.size
    if abs(z - 1) < 1e-6:
        return im
    if z > 1:
        cw, ch = w / z, h / z
        return im.crop((round((w - cw) / 2), round((h - ch) / 2), round((w + cw) / 2), round((h + ch) / 2)))
    pw, ph = int(round((w / z - w) / 2)), int(round((h / z - h) / 2))
    return Image.fromarray(_borda(np.asarray(im), (ph, pw)))


def recorte_prato(im, bbox, frac, ruido, rng):
    """Recorte quadrado centrado no prato; o maior lado do prato ocupa `frac` do lado do recorte."""
    x0, y0, x1, y1 = bbox
    lado = max(x1 - x0, y1 - y0) / frac
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if ruido:
        lado *= np.exp(rng.uniform(np.log(1 - ruido), np.log(1 + ruido)))
        cx += rng.uniform(-ruido, ruido) * lado * 0.25; cy += rng.uniform(-ruido, ruido) * lado * 0.25
    a = np.asarray(im); h, w = a.shape[:2]
    m = int(np.ceil(max(0, lado / 2 - cx, lado / 2 - cy, cx + lado / 2 - w, cy + lado / 2 - h))) + 1
    im2 = Image.fromarray(_borda(a, (m, m)))
    return im2.crop((round(cx + m - lado / 2), round(cy + m - lado / 2), round(cx + m + lado / 2), round(cy + m + lado / 2)))


def modelo(nome, k_tok=None):
    arch, arq, hw, mean, std = CFG[nome]
    if arch.startswith("tips"):
        # TIPS (Google DeepMind) tem a arquitetura do DINOv2 com 1 token de registro. Montamos o ViT
        # equivalente do timm e carregamos o .npz como dados (sem executar código de terceiros):
        # o embedding de posição do CLS é somado ao próprio CLS e o registro entra sem posição.
        import timm
        m = timm.create_model("vit_base_patch14_reg4_dinov2" if arch == "tips_base" else "vit_small_patch14_reg4_dinov2",
                              pretrained=False, num_classes=0, img_size=448, reg_tokens=1, dynamic_img_size=True)
        w = {k: torch.from_numpy(v) for k, v in np.load(os.path.join(PESOS, arq), allow_pickle=False).items()}
        w["cls_token"] = w["cls_token"] + w["pos_embed"][:, :1]
        w["pos_embed"] = w["pos_embed"][:, 1:]
        w["reg_token"] = w.pop("register_tokens")
        w.pop("mask_token")
        falta = m.load_state_dict(w, strict=True)
        m.eval()

        def f(x):
            t = m.forward_features(x)
            return t[:, :2].flatten(1), t[:, 2:], None
        return f, hw, mean, std
    import timm
    from timm.models.vision_transformer import _load_weights
    m = timm.create_model(arch, pretrained=False, num_classes=0, dynamic_img_size=True).eval()
    _load_weights(m, os.path.join(PESOS, arq))

    def f(x):
        t = m.patch_embed(x); t = m._pos_embed(t); t = m.patch_drop(t); t = m.norm_pre(t)
        inter = None
        for i, b in enumerate(m.blocks):
            if i == k_tok:
                inter = t
            t = b(t)
        t = m.norm(t)
        return m.forward_head(t, pre_logits=True), t[:, m.num_prefix_tokens:], inter
    return f, hw, mean, std


def agrega(pool, pat, gh, gw):
    """agregado ‖ média ‖ GeM(p=3) ‖ grade 2×2 (médias por quadrante)."""
    B, N, D = pat.shape
    g = pat.reshape(B, gh, gw, D)
    gem = pat.clamp(min=0).pow(3).mean(1).pow(1 / 3)
    quad = [g[:, :gh // 2, :gw // 2], g[:, :gh // 2, gw // 2:], g[:, gh // 2:, :gw // 2], g[:, gh // 2:, gw // 2:]]
    return torch.cat([pool, pat.mean(1), gem] + [q.mean((1, 2)) for q in quad], 1)


def main():
    td, nome = sys.argv[1], sys.argv[2]
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    n_or, k_tok, var = int(arg("--orient", 8)), arg("--tokens", None), arg("--variante", "base")
    conj = arg("--conj", "treino,aval,br").split(",")
    saida = arg("--saida", nome)
    k_tok = int(k_tok) if k_tok else None
    torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
    itens = lista(td, conj)
    f, hw, mean, std = modelo(nome, k_tok)
    if hw[0] != hw[1]:
        n_or = min(n_or, 4)  # 90° mudaria o formato do quadro
    rng = np.random.default_rng(int(var.split(":")[-1]) if var.startswith("escala") or var.count(":") == 3 else 0)
    bbox = {}
    if var.startswith("prato"):
        with open(os.path.join(td, "meta", "zen_bbox.csv")) as fb:
            bbox = {r["id"]: tuple(float(r[c]) for c in ("x0", "y0", "x1", "y1")) for r in csv.DictReader(fb)}
        itens = [it for it in itens if it[0] in bbox]

    def carrega(i, p):
        im = Image.open(p).convert("RGB")
        if var.startswith("zoom"):
            im = zoom(im, float(var.split(":")[1]))
        elif var.startswith("escala"):
            im = zoom(im, rng.uniform(0.7, 1.3))
        elif var.startswith("prato"):
            ps = var.split(":")
            im = recorte_prato(im, bbox[i], float(ps[1]), float(ps[2]) if len(ps) > 2 else 0.0, rng)
        im = im.resize((hw[1], hw[0]), Image.BICUBIC)
        return (torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1) - mean) / std

    feats, t0 = [], time.time()
    tok = None
    bs = 16
    with torch.inference_mode():
        for i in range(0, len(itens), bs):
            x = torch.stack([carrega(d, p) for d, p in itens[i:i + bs]])
            por_or = []
            for k in range(n_or):
                xo = orienta(x, k)
                pool, pat, inter = f(xo)
                ph = xo.shape[-2] // (14 if nome.startswith("tips") else 16); pw = pat.shape[1] // ph
                por_or.append(agrega(pool, pat, ph, pw).numpy().astype(np.float16))
                if k_tok is not None and k < 4:
                    if tok is None:
                        tok = np.lib.format.open_memmap(os.path.join(td, "feat", f"zen_{saida}_tokL{k_tok}.npy"), mode="w+",
                                                        dtype=np.float16, shape=(len(itens), 4, *inter.shape[1:]))
                    tok[i:i + len(x), k] = inter.numpy().astype(np.float16)
            feats.append(np.stack(por_or, 1))
            if (i // bs) % 20 == 0:
                print(saida, i, len(itens), f"{time.time() - t0:.0f}s", flush=True)
    X = np.concatenate(feats)
    np.savez(os.path.join(td, "feat", f"zen_{saida}.npz"), ids=np.array([i for i, _ in itens]), X=X)
    if tok is not None:
        tok.flush()
        with open(os.path.join(td, "feat", f"zen_{saida}_tokL{k_tok}_ids.txt"), "w") as fo:
            fo.write("\n".join(i for i, _ in itens) + "\n")
    print(saida, X.shape, f"{(time.time() - t0) / 60:.1f} min, {(time.time() - t0) / len(itens) / n_or * 1000:.0f} ms/img·orientação", flush=True)


if __name__ == "__main__":
    main()
