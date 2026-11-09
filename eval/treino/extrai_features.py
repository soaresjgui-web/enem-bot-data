#!/usr/bin/env python3
"""Extrai embeddings congelados das fotos zenitais (treino + avaliação + itens brasileiros).

Uso: python3 extrai_features.py <dir_treino> <backbone> [<backbone> ...]
  backbones: dinov2_s, dinov2_b, dinov2_s448, clip_b16
Grava <dir_treino>/feat/<backbone>.npz com ids (dish_id ou prato_id) e X (CLS ‖ média dos patches).
A imagem inteira é redimensionada (sem recorte) para preservar a escala: no Nutrition5k a
câmera está a altura fixa, então área em pixels ~ área real.
"""
import glob, os, sys, time
import numpy as np, torch, timm
from PIL import Image

SCR = "/tmp/claude-0/-home-user-enem-bot-data/59d3cb2b-a793-5c18-8c23-e9dab2604443/scratchpad"
MEAN_IMNET, STD_IMNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MEAN_CLIP, STD_CLIP = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)


def lista_imagens(td):
    itens = []
    ids = [l.strip() for f in ("treino_ids.txt", "aval_ids.txt") for l in open(os.path.join(td, "meta", f)) if l.strip()]
    itens += [(d, os.path.join(td, "img", d, "rgb.png")) for d in ids]
    for p in sorted(glob.glob(os.path.join(SCR, "cego2", "lote*", "*"))):
        pid = os.path.splitext(os.path.basename(p))[0]
        if pid[0] in "bc":  # UFPR/Seara (b) e Liv Up (c)
            itens.append((pid, p))
    return itens


def carrega(path, hw, mean, std):
    im = Image.open(path).convert("RGB").resize((hw[1], hw[0]), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


def modelo(nome):
    if nome.startswith("dinov2"):
        arch = "vit_base_patch14_dinov2.lvd142m" if nome == "dinov2_b" else "vit_small_patch14_dinov2.lvd142m"
        hw = (448, 588) if nome.endswith("448") else (224, 294)
        m = timm.create_model(arch, pretrained=True, num_classes=0, dynamic_img_size=True).eval()
        def f(x):
            t = m.forward_features(x)
            return torch.cat([t[:, 0], t[:, m.num_prefix_tokens:].mean(1)], 1)
        return f, hw, MEAN_IMNET, STD_IMNET
    if nome == "clip_b16":
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
        m = m.eval()
        return (lambda x: m.encode_image(x)), (224, 224), MEAN_CLIP, STD_CLIP
    raise ValueError(nome)


def main(td, nomes):
    torch.set_num_threads(4)
    itens = lista_imagens(td)
    os.makedirs(os.path.join(td, "feat"), exist_ok=True)
    for nome in nomes:
        f, hw, mean, std = modelo(nome)
        feats, t0 = [], time.time()
        with torch.inference_mode():
            for i in range(0, len(itens), 32):
                x = torch.stack([carrega(p, hw, mean, std) for _, p in itens[i:i + 32]])
                feats.append(f(x).float().numpy())
                if i % 640 == 0:
                    print(nome, i, f"{time.time() - t0:.0f}s", flush=True)
        X = np.concatenate(feats)
        np.savez(os.path.join(td, "feat", f"{nome}.npz"), ids=np.array([i for i, _ in itens]), X=X)
        print(nome, X.shape, f"{(time.time() - t0) / len(itens) * 1000:.0f} ms/img (lote 32, 4 threads)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
