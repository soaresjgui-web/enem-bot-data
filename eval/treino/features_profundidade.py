#!/usr/bin/env python3
"""Atributos de profundidade (RGB-D) a partir de depth_raw.png do Nutrition5k.

Uso: python3 features_profundidade.py <dir_treino>
Grava <dir_treino>/feat/depth_hand.npz (atributos de volume/altura) e
<dir_treino>/feat/depth_map.npz (mapa de alturas 120x160, para embedding).

- Unidade: 1e-4 m (0,1 mm). Os primeiros pratos (abril/2019) vêm em mm -> x10
  (detectado pela mediana < 1000).
- Plano da mesa: RANSAC (3 pontos, 200 iterações, tolerância 4 mm) nos pixels válidos;
  altura = distância ao plano (positiva em direção à câmera).
- Área por pixel = 5,957e-3 cm² a 35,9 cm (Thames et al. 2021), escalada por (z/35,9)².
- Atributos: volume (cm³) e área (cm²) acima de 5/10/15/20/30/50 mm, percentis de altura,
  fração sem leitura, e o mesmo relativo ao 'nível do prato' (moda das alturas 5–40 mm).
"""
import os, sys
import numpy as np
from PIL import Image

AREA_PX = 5.957e-3  # cm² a 35,9 cm
Z0 = 35.9


def plano_ransac(xyz, rng, it=200, tol=4.0):
    best, nbest = None, -1
    n = len(xyz)
    for _ in range(it):
        p = xyz[rng.choice(n, 3, replace=False)]
        nv = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(nv) < 1e-6:
            continue
        nv /= np.linalg.norm(nv)
        if abs(nv[2]) < 0.9:  # plano quase perpendicular ao eixo da câmera
            continue
        dist = np.abs((xyz - p[0]) @ nv)
        k = (dist < tol).sum()
        if k > nbest:
            best, nbest = (nv, p[0]), k
    nv, p0 = best
    inl = np.abs((xyz - p0) @ nv) < tol
    A = np.c_[xyz[inl, 0], xyz[inl, 1], np.ones(inl.sum())]  # refino: z = a x + b y + c
    coef, *_ = np.linalg.lstsq(A, xyz[inl, 2], rcond=None)
    return coef


def atributos(path, rng):
    d = np.array(Image.open(path)).astype(np.float32)
    val = d > 0
    if np.median(d[val]) < 1000:
        d *= 10
    z = d / 100.0  # mm
    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W]
    xyz = np.c_[xx[val], yy[val], z[val]].astype(np.float64)
    sub = xyz[rng.choice(len(xyz), min(20000, len(xyz)), replace=False)]
    a, b, c = plano_ransac(sub, rng)
    plano = a * xx + b * yy + c
    h = np.where(val, plano - z, 0.0)  # mm acima da mesa
    h = np.clip(h, 0, 200)
    area = AREA_PX * (np.where(val, z, plano) / 10 / Z0) ** 2  # cm² por pixel
    f = []
    for t in (5, 10, 15, 20, 30, 50):
        m = h > t
        f += [np.log1p((h * m * area).sum() / 10), np.log1p((m * area).sum())]  # volume cm³ (h em mm /10)
    hv = h[h > 5]
    f += list(np.percentile(hv, [50, 75, 90, 99]) if len(hv) > 50 else [0, 0, 0, 0])
    f += [1 - val.mean(), c / 10]
    hist, ed = np.histogram(h[(h > 5) & (h < 40)], bins=35, range=(5, 40))
    nivel = ed[hist.argmax()] + 0.5 if hist.sum() else 0
    h2 = np.clip(h - nivel, 0, None)
    for t in (3, 8, 15, 25):
        m = h2 > t
        f += [np.log1p((h2 * m * area).sum() / 10), np.log1p((m * area).sum())]
    f.append(nivel)
    mapa = np.array(Image.fromarray(h.astype(np.float32)).resize((160, 120), Image.BILINEAR))
    return np.array(f, np.float32), mapa.astype(np.float16)


def main(td):
    ids = [l.strip() for f in ("treino_ids.txt", "aval_ids.txt") for l in open(os.path.join(td, "meta", f)) if l.strip()]
    rng = np.random.default_rng(0)
    F, M, ok = [], [], []
    for i, d in enumerate(ids):
        p = os.path.join(td, "img", d, "depth_raw.png")
        try:
            f, m = atributos(p, rng)
            ok.append(True)
        except Exception as e:
            print("falhou", d, e)
            f, m = np.full(35, np.nan, np.float32), np.zeros((120, 160), np.float16)
            ok.append(False)
        F.append(f); M.append(m)
        if i % 500 == 0:
            print(i, flush=True)
    np.savez(os.path.join(td, "feat", "depth_hand.npz"), ids=np.array(ids), X=np.stack(F), ok=np.array(ok))
    np.savez(os.path.join(td, "feat", "depth_map.npz"), ids=np.array(ids), X=np.stack(M))


if __name__ == "__main__":
    main(sys.argv[1])
