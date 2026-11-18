#!/usr/bin/env python3
"""Lê o tar de imagens do FlavoriaFoodWeight1700 da entrada padrão, EM FLUXO (nada é gravado em disco além do CSV),
e grava a tabela que flavoria_monta.py usa: caminho, bytes e, para cada PNG, largura, altura,
arco_borda_oculta e brilho.

Uso (2,6 GB de tráfego, ~20 min, memória de uma imagem por vez):
  curl -sL https://zenodo.org/api/records/5850856/files/dataset_images_and_csv.tar/content \
    | python3 -I flavoria_quadros.py flavoria_arquivos.csv

arco_borda_oculta: o prato é branco, redondo e enche o quadro. No anel externo (84% a 97% do raio), dividido em
48 setores, um setor conta como "borda visível" se pelo menos 35% dos pixels são claros (acima de 60% do brilho
típico do anel) e pouco saturados. A medida é o maior arco contíguo de setores sem borda visível, em fração de
volta. Acima de 0,20 quase sempre é a cabeça ou o braço de quem carrega a bandeja (27 de 32 num sorteio);
não pega mão pequena na borda e erra quando a comida passa da borda ou quando a foto não é de um prato.
Dataset: Sarapisto et al. (2022), https://doi.org/10.5281/zenodo.5850856, CC BY 4.0.
"""
import csv, io, sys, tarfile

import numpy as np
from PIL import Image

SETORES = 48
ANEL = (0.84, 0.97)


def arco_borda_oculta(im):
    a = np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR)).astype(np.float32)
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx - w / 2 + .5) / (w / 2); dy = (yy - h / 2 + .5) / (h / 2)
    r = np.hypot(dx, dy)
    mx = a.max(2); sat = (mx - a.min(2)) / np.maximum(mx, 1)
    anel = (r > ANEL[0]) & (r < ANEL[1])
    borda = (mx > 0.6 * np.percentile(mx[anel], 75)) & (sat < 0.22)
    setor = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * SETORES).astype(int) % SETORES
    oculto = np.array([borda[anel & (setor == s)].mean() < 0.35 for s in range(SETORES)])
    if oculto.all():
        return 1.0
    maior = atual = 0
    for b in list(oculto) + list(oculto):           # duas voltas: o arco pode cruzar o ângulo zero
        atual = atual + 1 if b else 0
        maior = max(maior, atual)
    return min(maior, SETORES) / SETORES


def main(saida):
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["caminho", "bytes", "largura", "altura", "arco_borda_oculta", "brilho"])
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                if not m.name.endswith(".png"):
                    w.writerow([m.name, m.size, "", "", "", ""]); continue
                im = Image.open(io.BytesIO(tf.extractfile(m).read())); im.load()
                brilho = float(np.asarray(im.convert("L").resize((64, 64))).mean())
                w.writerow([m.name, m.size, im.size[0], im.size[1], f"{arco_borda_oculta(im):.3f}", f"{brilho:.1f}"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
