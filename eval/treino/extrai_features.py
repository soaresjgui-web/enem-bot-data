#!/usr/bin/env python3
"""Extrai embeddings congelados das fotos zenitais (treino + avaliação + itens brasileiros).

Uso: python3 extrai_features.py <dir_treino> <backbone> [<backbone> ...]
  backbones: siglip_b16, siglip2_b16, vits16_in21k, vits16_in21k_384, effb0
Grava <dir_treino>/feat/<backbone>.npz com ids (dish_id ou prato_id) e X (token agregado ‖ média dos patches).
A imagem inteira é redimensionada (sem recorte) para preservar a escala: no Nutrition5k a
câmera está a altura fixa, então área em pixels ~ área real.
"""
import glob, os, sys, time
import numpy as np, torch, timm
from PIL import Image

SCR = "/tmp/claude-0/-home-user-enem-bot-data/59d3cb2b-a793-5c18-8c23-e9dab2604443/scratchpad"
MEAN_IMNET, STD_IMNET = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


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


PESOS = os.path.join(SCR, "treino", "pesos")
# huggingface.co responde, mas o CDN dos pesos (cdn-lfs/xet) está bloqueado no proxy desta sessão:
# DINOv2 e CLIP (OpenAI/LAION) ficaram inacessíveis. Usamos pesos públicos no GCS do Google
# (SigLIP/SigLIP 2 — modelos imagem-texto tipo CLIP; ViT-S/16 AugReg ImageNet-21k) e no GitHub (EfficientNet-B0).
NPZ = {
    "siglip_b16": ("vit_base_patch16_siglip_224", "webli_en_b16_224_63724782.npz", (224, 224)),
    "siglip2_b16": ("vit_base_patch16_siglip_224", "siglip2_b16_224.npz", (224, 224)),
    "vits16_in21k": ("vit_small_patch16_224", "S_16-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0.npz", (224, 288)),
    "vits16_in21k_384": ("vit_small_patch16_224", "S_16-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0.npz", (384, 512)),
}


def modelo(nome):
    if nome in NPZ:
        from timm.models.vision_transformer import _load_weights
        arch, arq, hw = NPZ[nome]
        m = timm.create_model(arch, pretrained=False, num_classes=0, dynamic_img_size=True).eval()
        _load_weights(m, os.path.join(PESOS, arq))
        def f(x):
            t = m.forward_features(x)
            pool = m.forward_head(t, pre_logits=True)
            return torch.cat([pool, t[:, m.num_prefix_tokens:].mean(1)], 1)
        return f, hw, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if nome == "effb0":
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0).eval()
        m.load_state_dict({k: v for k, v in torch.load(os.path.join(PESOS, "efficientnet_b0_ra-3dd342df.pth")).items()
                           if not k.startswith("classifier")})
        return m, (360, 480), MEAN_IMNET, STD_IMNET
    raise ValueError(nome)


def main(td, nomes):
    torch.set_num_threads(int(os.environ.get("NTHREADS", "1")))  # nesta máquina compartilhada, 1 thread por processo foi o mais rápido
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
        print(nome, X.shape, f"{(time.time() - t0) / len(itens) * 1000:.0f} ms/img (lote 32)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
