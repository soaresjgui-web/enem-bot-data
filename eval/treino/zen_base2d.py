#!/usr/bin/env python3
"""Seção 8: refaz a linha de base "2d" da seção 7 (SigLIP 2 B/16 + ridge, zenital + quadros laterais)
e grava as previsões POR PRATO, para o bootstrap pareado contra as configurações novas.

Uso: OMP_NUM_THREADS=1 python3 zen_base2d.py
Grava <TD>/resultados/zen_base2d.npz: por split, prato_id, massa, densidade, proteína direta.
"""
import os
import numpy as np
from comum import TD
from experimentos import proteinas
from experimentos_lateral import Dados, configs, monta, ajusta

NOME = "2d zenital + lateral, + pratos test/sem split fora da avaliação"


def main():
    D = Dados("siglip2_b16")
    po, ps = configs(D)[NOME]
    X, Y, dish, _ = monta(D, po, ps)
    m = ajusta(D, X, Y, dish)
    out = {}
    for s in ("teste", "calib", "n5k_v1", "ufpr", "liv_up", "seara"):
        keys, pids, Ys = D.aval[s]
        pd, pm, massa, dens = proteinas(m.predict(np.stack([D.Xo[D.idx_o[k]] for k in keys])))
        out[f"{s}|prato_id"] = np.array(pids); out[f"{s}|massa"] = massa; out[f"{s}|dens"] = dens; out[f"{s}|direto"] = pd
        if s == "teste":
            from comum import ok
            print(f"2d teste: direto {100 * ok(Ys[:, 1], pd).mean():.1f}% m×d {100 * ok(Ys[:, 1], pm).mean():.1f}% "
                  f"MAPE massa {100 * np.mean(np.abs(massa - Ys[:, 0]) / Ys[:, 0]):.1f}%")
    np.savez(os.path.join(TD, "resultados", "zen_base2d.npz"), **out)


if __name__ == "__main__":
    main()
