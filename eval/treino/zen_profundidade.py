#!/usr/bin/env python3
"""Seção 8 (profundidade privilegiada): atributos de volume/altura da profundidade zenital do
Nutrition5k para TODOS os pratos do treino zenital novo (zen_treino.csv) e da avaliação.

Uso: OMP_NUM_THREADS=1 python3 zen_profundidade.py <dir_treino>
Grava <dir_treino>/feat/zen_depth.npz (ids, X[n, 59], ok). Mesmos atributos de
features_profundidade.py. São usados só como ALVO auxiliar no treino (nunca como entrada).
"""
import csv, os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from features_profundidade import atributos  # noqa: E402


def main(td):
    ids = []
    for arq in ("zen_treino.csv", "zen_aval.csv"):
        with open(os.path.join(td, "meta", arq)) as f:
            ids += [r["dish_id"] for r in csv.DictReader(f)]
    velho = np.load(os.path.join(td, "feat", "depth_hand.npz"))
    cache = {k: v for k, v, o in zip(velho["ids"], velho["X"], velho["ok"]) if o}
    rng = np.random.default_rng(0)
    F, ok = [], []
    for d in ids:
        if d in cache:
            F.append(cache[d]); ok.append(True); continue
        try:
            F.append(atributos(os.path.join(td, "img", d, "depth_raw.png"), rng)[0]); ok.append(True)
        except Exception as e:
            print("falhou", d, e); F.append(np.full(59, np.nan, np.float32)); ok.append(False)
    np.savez(os.path.join(td, "feat", "zen_depth.npz"), ids=np.array(ids), X=np.stack(F), ok=np.array(ok))
    print(len(ids), "pratos;", int(np.sum(ok)), "com profundidade válida")


if __name__ == "__main__":
    main(sys.argv[1])
