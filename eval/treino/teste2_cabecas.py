#!/usr/bin/env python3
"""Seção 11: treina de novo as três cabeças de frações (CPU) do pipeline congelado da seção 10 e prevê o
teste2. Não lê nenhum gabarito de avaliação.

Uso: OMP_NUM_THREADS=2 python3 teste2_cabecas.py
Antes: features do teste2 (zen_extrai.py ... --conj teste2 --saida siglip2_b16_teste2 / siglip2_b16_512_teste2).

As cabeças da seção 9 só tinham as previsões guardadas, não os pesos. Aqui cada uma é treinada de novo em todo o
treino com o mesmo código, a mesma configuração (dens_experimentos.CFG_SOFT) e a mesma semente do modelo final
de dens_experimentos.etapa_soft, e a paridade é conferida SEM gabarito: as previsões novas nos 450 pratos antigos
(calib, teste, n5k_v1) são comparadas com as do banco.
Grava <TD>/resultados/dens/teste2/<cabeça>.npz: teste2_dens, teste2_ff (fração por fonte), ids_teste2 (dish_id, na
ordem de n5k_teste2_amostra.csv), as previsões refeitas nos 450 e as medidas de paridade.
"""
import csv, os, time
import numpy as np
from zen_comum import Feat
from dens_comum import BANCO_D, DENS, FONTES, PREC
import dens_experimentos as E

CABECAS = [("soft-mlp ingrediente-l2", "siglip2_b16_512"), ("soft-mlp fonte-l2d", "siglip2_b16_512"),
           ("soft-mlp fonte-l2 +perda-dens peso-faixa", "ambos")]
SAIDA = os.path.join(DENS, "teste2")


def ids_teste2():
    with open(os.path.join(PREC, "n5k_teste2_amostra.csv")) as f:
        return [r["dish_id"] for r in csv.DictReader(f)]


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(SAIDA, exist_ok=True)
    ids2 = ids_teste2()
    F2 = {"siglip2_b16": Feat("siglip2_b16_teste2"), "siglip2_b16_512": Feat("siglip2_b16_512_teste2")}
    assert all(set(ids2) == set(f.idx) for f in F2.values()), "features do teste2 incompletas"
    dads = {}
    for nome, bb in CABECAS:
        t0 = time.time()
        A = dads.get(bb) or dads.setdefault(bb, E.Dad(bb)); D = A.D
        assert not set(ids2) & set(D.ids)
        bbs = ["siglip2_b16", "siglip2_b16_512"] if bb == "ambos" else [bb]
        X2 = np.concatenate([F2[b].pega(ids2, A.o, E.BL) for b in bbs], -1)
        assert X2.shape[1:] == A.X.shape[1:], (X2.shape, A.X.shape)
        cls, kw = E.CFG_SOFT[nome]; kw = dict(kw)
        peso = kw.pop("peso", None)
        w = None if peso is None else D.w_cv ** peso
        Ft = {"fonte": A.Ff, "ingr": A.Fi, "fina": A.Fx}[cls]
        todos = np.ones(len(D.ids), bool)
        dc = A.dens_fonte(todos) if cls == "fonte" else (A.dens_fina(todos) if cls == "fina" else A.dens_i)
        prev = E.treina_soft(A.X, D.T, Ft, dc, [A.Xs[s] for s in E.SPLITS] + [X2], w, **kw)   # igual ao modelo final de etapa_soft
        f_cls = A.fonte_x if cls == "fina" else A.fonte_i
        ff = (lambda F: F) if cls == "fonte" else (lambda F: np.stack([F[:, f_cls == j].sum(1) for j in range(len(FONTES))], 1))
        banco = np.load(os.path.join(BANCO_D, f"{nome} {bb}.npz"), allow_pickle=False)
        out = {"ids_teste2": np.array(ids2), "teste2_dens": prev[-1][1] @ dc, "teste2_ff": ff(prev[-1][1])}
        par = {}
        for s, p in zip(E.SPLITS, prev):
            d = p[1] @ dc; b = banco[f"{s}_dens"]
            out[f"{s}_dens"] = d
            par[s] = (float(np.abs(d - b).max()), float(np.median(np.abs(d - b) / np.maximum(b, 1e-3))), float(np.corrcoef(d, b)[0, 1]))
        out["paridade"] = np.array([par[s] for s in E.SPLITS])
        np.savez(os.path.join(SAIDA, f"{nome} {bb}.npz"), **out)
        print(f"{nome} {bb}: teste2 {out['teste2_dens'].shape}, densidade mediana {100 * np.median(out['teste2_dens']):.1f} g/100 g"
              f" (nos 450 antigos: {100 * np.median(np.concatenate([out[f'{s}_dens'] for s in E.SPLITS])):.1f}) ({time.time() - t0:.0f}s)")
        for s in E.SPLITS:
            print(f"   paridade com o banco em {s:7s}: diferença máx {par[s][0]:.5f} g/g, diferença relativa mediana {100 * par[s][1]:.2f}%, correlação {par[s][2]:.5f}")


if __name__ == "__main__":
    main()
