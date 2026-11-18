#!/usr/bin/env python3
"""Seção 8 (robustez de escala): caixa do prato/tigela em cada foto zenital do Nutrition5k.

Uso: python3 zen_prato.py <dir_treino>
Grava <dir_treino>/meta/zen_bbox.csv (id, x0, y0, x1, y1 em pixels da foto 640×480, lado_px).

A caixa vem do mapa de altura (profundidade): maior região conexa acima de 12 mm do plano da
bancada (o prato fica sobre a balança). A profundidade aqui NÃO é entrada do modelo: ela só
imita o que a usuária faz no app ao encaixar o prato no círculo-guia. O modelo recebe apenas
o recorte RGB resultante.
"""
import csv, os, sys
import numpy as np
from scipy import ndimage as ndi

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from features_profundidade import atributos  # noqa: E402


def caixa(mapa, limiar=12.0):
    """mapa: alturas em mm (120×160). Retorna (x0, y0, x1, y1) na escala 640×480 ou None."""
    m = ndi.binary_opening(mapa > limiar, iterations=2)
    rot, n = ndi.label(m)
    if n == 0:
        return None
    tam = ndi.sum(m, rot, range(1, n + 1))
    # prefere a maior região que não seja uma faixa colada na borda (bancada vizinha, outro prato)
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
    return xs.min() * f, ys.min() * f, (xs.max() + 1) * f, (ys.max() + 1) * f


def main(td):
    z = np.load(os.path.join(td, "feat", "depth_map.npz"))
    mapas = dict(zip(z["ids"], z["X"]))
    ids = []
    for arq in ("zen_treino.csv", "zen_aval.csv"):
        with open(os.path.join(td, "meta", arq)) as f:
            ids += [r["dish_id"] for r in csv.DictReader(f)]
    rng = np.random.default_rng(0)
    linhas, falhas = [], 0
    for d in ids:
        mp = mapas.get(d)
        if mp is None:
            try:
                mp = atributos(os.path.join(td, "img", d, "depth_raw.png"), rng)[1]
            except Exception:
                falhas += 1
                continue
        b = caixa(mp.astype(np.float32))
        if b is None or min(b[2] - b[0], b[3] - b[1]) < 60:
            falhas += 1
            continue
        linhas.append([d] + [round(v, 1) for v in b] + [round(max(b[2] - b[0], b[3] - b[1]), 1)])
    with open(os.path.join(td, "meta", "zen_bbox.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "x0", "y0", "x1", "y1", "lado_px"]); w.writerows(linhas)
    lado = np.array([l[-1] for l in linhas])
    print(f"{len(linhas)} caixas, {falhas} falhas | lado do prato (px de 640): p5 {np.percentile(lado, 5):.0f} "
          f"mediana {np.median(lado):.0f} p95 {np.percentile(lado, 95):.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
