#!/usr/bin/env python3
"""Seção 12.8: calorias, carboidratos e gordura na validação cruzada do treino_v2 (só previsões fora-do-fold; não lê o teste2).

    python3 macros_cv.py <pasta_kaggle_3>

Modelos: D1 (12.7: So400m congelado, cabeça ridge + cabeça de frações) e U0 = C1 (12.6: `gpu3 ft siglip2_b16_224 r2`).
Critério por nutriente, no estilo do de proteína: acerto se |erro| <= 20% do real OU <= piso absoluto; média das cinco
faixas do próprio nutriente com o mesmo peso. Faixas e pisos fixados antes de calcular qualquer acerto (12.8).
Rotas do D1 (escolhidas pela CV com 1 orientação, a do D1): direta da ridge; direta da cabeça de frações; massa da ridge ×
densidade implícita da cabeça de frações (nutriente ÷ massa da própria cabeça); média em log (log1p) das duas diretas;
média geométrica da direta da ridge com a rota massa × densidade. U0 só tem a direta (a densidade implícita dele dá o mesmo)."""
import csv, json, os, sys
import numpy as np
from teste2_avalia import boot_cadeias
from dens_comum import DENS
from um_modelo_avalia import arq, LISTA

I = dict(massa=0, proteina=1, kcal=2, gordura=4, carbo=5)
FAIXAS = dict(kcal=[100, 250, 400, 600], carbo=[5, 15, 30, 50], gordura=[2, 10, 20, 35], proteina=[5, 15, 25, 40])
PISO = dict(kcal=50.0, carbo=10.0, gordura=5.0, proteina=5.0)
PISO_ESTRITO = dict(kcal=25.0, carbo=5.0, gordura=3.0, proteina=2.5)
NOMES = dict(kcal="kcal", carbo="carboidrato (g)", gordura="gordura (g)", proteina="proteína (g)")
ORIENT = {1: "_o1", 8: ""}


def acerto(real, est, piso):
    e = np.abs(est - real)
    return (e <= piso) | (e <= 0.20 * real)


def crit(real, est, nut, idx=None, piso=None):
    if idx is not None:
        real, est = real[idx], est[idx]
    a = acerto(real, est, PISO[nut] if piso is None else piso)
    fx = np.digitize(real, FAIXAS[nut], right=True)
    return 100 * float(np.mean([a[fx == k].mean() for k in range(5) if (fx == k).any()]))


def rotas_d1(R, F, nut, s):
    j = I[nut]
    dr, df = np.expm1(R["oof" + s][:, j]).clip(0), np.expm1(F["oof" + s][:, j]).clip(0)
    mxd = np.exp(R["oof" + s][:, 0]) * df / np.exp(F["oof" + s][:, 0])
    return {"direta_ridge": dr, "direta_fracoes": df, "massa_ridge_x_dens_fracoes": mxd,
            "media_log_diretas": np.expm1((np.log1p(dr) + np.log1p(df)) / 2), "geometrica_ridge_mxd": np.sqrt(dr * mxd)}


def mede(real, est, nut, cad):
    fx = np.digitize(real, FAIXAS[nut], right=True)
    a = acerto(real, est, PISO[nut])
    pos = real > 0
    return dict(ponderado=boot_cadeias(cad, lambda i: crit(real, est, nut, i), n=2000), simples=100 * float(a.mean()),
                por_faixa=[(100 * float(a[fx == k].mean()), int((fx == k).sum())) for k in range(5)],
                estrito=crit(real, est, nut, piso=PISO_ESTRITO[nut]), so_20pct=crit(real, est, nut, piso=0.0),
                mape=float(100 * np.mean(np.abs(est[pos] - real[pos]) / real[pos])), mae=float(np.mean(np.abs(est - real))),
                vies_mediano=float(100 * np.median(est[pos] / real[pos] - 1)),
                vies_por_faixa=[float(100 * np.median(est[(fx == k) & pos] / real[(fx == k) & pos] - 1)) for k in range(5)])


def main():
    pasta = sys.argv[1]
    L = list(csv.DictReader(open(LISTA)))
    cad = np.array([int(r["cadeia"]) for r in L])
    real = {n: np.array([float(r[n]) for r in L]) for n in I}
    R = np.load(arq(pasta, "gpu3 ridge siglip2_so400m16_384x512 o8")); F = np.load(arq(pasta, "gpu3 fracoes so400m"))
    U = np.load(arq(pasta, "gpu3 ft siglip2_b16_224 r2"))
    ids = np.array([r["dish_id"] for r in L])
    assert all((z["ids_treino"] == ids).all() for z in (R, F, U))
    out = dict(faixas=FAIXAS, piso=PISO, piso_estrito=PISO_ESTRITO, n=len(ids), cadeias=int(len(np.unique(cad))), nutrientes={})
    for nut in ["kcal", "carbo", "gordura", "proteina"]:
        tab = {k: {r: crit(real[nut], e, nut) for r, e in rotas_d1(R, F, nut, s).items()} for k, s in ORIENT.items()}
        rota = max(tab[1], key=tab[1].get)
        e_d1 = rotas_d1(R, F, nut, "_o1")[rota]
        e_u0 = np.expm1(U["oof_o1"][:, I[nut]]).clip(0)
        D = dict(rotas_d1=tab, rota_d1=rota, D1=mede(real[nut], e_d1, nut, cad), U0=mede(real[nut], e_u0, nut, cad),
                 dif_D1_U0=boot_cadeias(cad, lambda i: crit(real[nut], e_d1, nut, i) - crit(real[nut], e_u0, nut, i), n=2000))
        out["nutrientes"][nut] = D
        print("\n== %s  faixas %s, piso %g" % (NOMES[nut], FAIXAS[nut], PISO[nut]))
        print("   rotas do D1, CV com 1 / 8 orientações: " + "; ".join("%s %.1f / %.1f" % (r, tab[1][r], tab[8][r]) for r in tab[1]) + "  -> %s" % rota)
        for m in ("D1", "U0"):
            x = D[m]
            print("   %s: ponderado %.1f%% (IC %.1f a %.1f) | simples %.1f | por faixa %s | piso estrito %.1f | só 20%% %.1f | MAPE %.1f%% | EMA %.1f | viés mediano %+.1f%% (por faixa %s)" %
                  (m, *x["ponderado"], x["simples"], " ".join("%.0f(n=%d)" % v for v in x["por_faixa"]), x["estrito"], x["so_20pct"], x["mape"], x["mae"], x["vies_mediano"], " ".join("%+.0f" % v for v in x["vies_por_faixa"])))
        print("   D1 − U0: %+.1f (IC %+.1f a %+.1f)" % D["dif_D1_U0"])
    # massa sozinha, para referência
    m = real["massa"]
    for nome, e in [("D1 (ridge)", np.exp(R["oof_o1"][:, 0])), ("U0", np.exp(U["oof_o1"][:, 0]))]:
        a = np.abs(e - m) <= 0.2 * m
        out.setdefault("massa", {})[nome] = dict(dentro_20pct=100 * float(a.mean()), mape=float(100 * np.mean(np.abs(e - m) / m)), vies_mediano=float(100 * np.median(e / m - 1)))
        print("massa %s: %.1f%% a ±20%%, MAPE %.1f%%, viés mediano %+.1f%%" % (nome, 100 * a.mean(), 100 * np.mean(np.abs(e - m) / m), 100 * np.median(e / m - 1)))
    # correlações entre os erros (o app mostraria os quatro juntos)
    e = {n: np.log1p(rotas_d1(R, F, n, "_o1")[out["nutrientes"][n]["rota_d1"]]) - np.log1p(real[n]) for n in ["kcal", "carbo", "gordura", "proteina"]}
    em = np.log(np.exp(R["oof_o1"][:, 0])) - np.log(m)
    print("correlação do erro (log) com o erro de massa, D1: " + " ".join("%s %.2f" % (n, np.corrcoef(v, em)[0, 1]) for n, v in e.items()))
    out["corr_erro_massa_D1"] = {n: float(np.corrcoef(v, em)[0, 1]) for n, v in e.items()}
    os.makedirs(os.path.join(DENS, "um_modelo"), exist_ok=True)
    json.dump(out, open(os.path.join(DENS, "um_modelo", "macros_cv.json"), "w"), indent=1, default=float, ensure_ascii=False)


if __name__ == "__main__":
    main()
