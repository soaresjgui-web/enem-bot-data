#!/usr/bin/env python3
"""Amostra estratificada de 50 pratos do Nutrition5k (CC BY 4.0) para calibração.

Uso: python3 coleta_n5k.py <dir_trabalho>
  <dir_trabalho> deve conter dish_metadata_cafe1.csv, dish_metadata_cafe2.csv,
  rgb_test_ids.txt e overhead_ids.txt (lista de pastas em imagery/realsense_overhead,
  obtida com `gsutil ls`). Baixa rgb.png de cada prato com gsutil e grava
  n5k_amostra.csv (dish_id, massa, proteína, ingredientes).

Estratos de proteína (g): 0-5, 5-15, 15-25, 25-40, >40 — 10 pratos por estrato,
sorteados (seed fixa) entre os ids do split rgb_test que têm imagem overhead e
massa total >= 20 g.
"""
import csv
import os
import random
import subprocess
import sys

BUCKET = "gs://nutrition5k_dataset/nutrition5k_dataset"
ESTRATOS = [(0, 5), (5, 15), (15, 25), (25, 40), (40, 1e9)]
POR_ESTRATO = 10
SEED = 20261108


def ler_meta(paths):
    pratos = {}
    for p in paths:
        with open(p, newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].startswith("dish_"):
                    continue
                ingr = []
                for i in range(6, len(row) - 6, 7):
                    ingr.append((row[i + 1], float(row[i + 2])))
                pratos[row[0]] = {
                    "massa": float(row[2]),
                    "proteina": float(row[5]),
                    "kcal": float(row[1]),
                    "ingr": ingr,
                }
    return pratos


def main(wd):
    meta = ler_meta([os.path.join(wd, f) for f in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv")])
    test = {l.strip() for l in open(os.path.join(wd, "rgb_test_ids.txt")) if l.strip()}
    over = {l.strip() for l in open(os.path.join(wd, "overhead_ids.txt")) if l.strip()}
    cand = sorted(d for d in test & over if d in meta and meta[d]["massa"] >= 20)
    rnd = random.Random(SEED)
    escolhidos = []
    for lo, hi in ESTRATOS:
        pool = [d for d in cand if lo <= meta[d]["proteina"] < hi]
        escolhidos += rnd.sample(pool, POR_ESTRATO)
        print(f"estrato {lo}-{hi}: {len(pool)} candidatos")
    img_dir = os.path.join(wd, "rgb")
    os.makedirs(img_dir, exist_ok=True)
    with open(os.path.join(wd, "n5k_amostra.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dish_id", "massa_g", "proteina_g", "kcal", "ingredientes"])
        for d in escolhidos:
            m = meta[d]
            w.writerow([d, round(m["massa"], 1), round(m["proteina"], 2), round(m["kcal"], 1),
                        "; ".join(f"{n} {g:.0f} g" for n, g in m["ingr"])])
            dst = os.path.join(img_dir, f"{d}.png")
            if not os.path.exists(dst):
                subprocess.run(["gsutil", "-q", "cp", f"{BUCKET}/imagery/realsense_overhead/{d}/rgb.png", dst], check=True)
    print(f"{len(escolhidos)} pratos -> {img_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
