#!/usr/bin/env python3
"""Seção 9.7 (d): compara as rodadas cegas do LLM v3. Só relato; nenhuma escolha é feita aqui.

Uso: python3 dens_rodadas.py
Antes, rodar dens_avalia.py três vezes: DENS_V3_RODADA=1, DENS_V3_RODADA=2 e sem a variável (média das
rodadas). Cada execução escolhe a sua forma e os seus pesos no calib e grava as estimativas por prato em
<TD>/resultados/dens/v3_pratos_{r1,r2,media}.npz. Este script lê esses arquivos e o gabarito e relata:
por entrada (rodada 1, rodada 2, média): calib deixa-fora, teste com IC, diferenças pareadas contra o
híbrido da seção 8, contra o híbrido sem v3 (9.6) e contra a rodada 1, densidade v3 com a massa real,
tabela por faixa, n5k_v1; e a concordância entre as rodadas 1 e 2 (correlação da densidade por prato,
fonte principal declarada igual, pratos que entram e saem da tolerância).
Grava <TD>/resultados/dens/rodadas.json.
"""
import json, os
import numpy as np
from zen_comum import Dados, crit, bootstrap, ic_simples, ok, EPS_D
from dens_comum import DENS, FONTES, faixas, grava_json

f3 = lambda v: f"{v[0]:+.1f} ({v[1]:+.1f} a {v[2]:+.1f})"


def main():
    D = Dados()
    Z = {t: dict(np.load(os.path.join(DENS, f"v3_pratos_{t}.npz"), allow_pickle=False)) for t in ("r1", "r2", "media")}
    J = {t: json.load(open(os.path.join(DENS, "avalia.json" if t == "media" else f"avalia_v3_{t}.json"))) for t in Z}
    real = {s: D.aval[s]["proteina"] for s in ("calib", "teste", "n5k_v1")}; massa = {s: D.aval[s]["massa"] for s in real}
    rt, mt = real["teste"], massa["teste"]; fx = faixas(rt); a15, a25 = rt > 15, rt > 25
    res = {}
    a_s8 = ok(rt, Z["r1"]["teste|hib_s8"]); a_h9 = ok(rt, Z["r1"]["teste|hib_sem_v3"]); a_r1 = ok(rt, Z["r1"]["teste|hib_v3"])
    assert all(np.allclose(Z[t]["teste|hib_s8"], Z["r1"]["teste|hib_s8"]) and np.allclose(Z[t]["teste|hib_sem_v3"], Z["r1"]["teste|hib_sem_v3"]) for t in Z)
    print(f"referências no teste: híbrido s8 {100 * a_s8.mean():.1f}% | híbrido sem v3 (9.6) {100 * a_h9.mean():.1f}%")
    for t, z in Z.items():
        e = z["teste|hib_v3"]; a = ok(rt, e); d = z["teste|dens_v3"]; dc = z["calib|dens_v3"]
        r = dict(forma=str(z["forma"]), pesos=str(z["pesos"]), calib_loo=crit(real["calib"], z["calib|loo_v3"]), teste=100 * a.mean(), ic=ic_simples(a),
                 mae=float(np.abs(e - rt).mean()), vies=float((e - rt).mean()), vs_s8=bootstrap(a_s8, a), vs_sem_v3=bootstrap(a_h9, a), vs_r1=bootstrap(a_r1, a),
                 acima15=crit(rt[a15], e[a15]), acima15_ic=ic_simples(a[a15]), acima25=crit(rt[a25], e[a25]), por_faixa={k: crit(rt[f], e[f]) for k, f in fx.items()},
                 n5k_v1=crit(real["n5k_v1"], z["n5k_v1|hib_v3"]),
                 dens_massa_real=dict(calib=crit(real["calib"], massa["calib"] * dc), teste=crit(rt, mt * d), teste_acima15=crit(rt[a15], (mt * d)[a15]),
                                      n5k_v1=crit(real["n5k_v1"], massa["n5k_v1"] * z["n5k_v1|dens_v3"]), por_faixa={k: crit(rt[f], (mt * d)[f]) for k, f in fx.items()},
                                      vs_r1=bootstrap(ok(rt, mt * Z["r1"]["teste|dens_v3"]), ok(rt, mt * d))),
                 massa_modelo_x_dens=dict(teste=crit(rt, z["teste|massa_modelo"] * d), por_faixa={k: crit(rt[f], (z["teste|massa_modelo"] * d)[f]) for k, f in fx.items()},
                                          calib=crit(real["calib"], z["calib|massa_modelo"] * dc)),
                 formas_v3={n: dict(calib_loo=v["calib_loo"], teste=v["teste"], n5k_v1=v["n5k_v1"]) for n, v in J[t]["hib_teste"].items() if n.startswith("[v3]")},
                 fonte_calib=J[t]["v3_relato"]["fonte_principal_calib"], fracao_calib=J[t]["v3_relato"]["fracao_fontes_calib_acima15"])
        res[t] = r
        print(f"\n## {t}: forma '{r['forma']}', pesos {r['pesos']}")
        print(f"   calib deixa-fora {r['calib_loo']:.1f}% | TESTE {r['teste']:.1f}% (IC {r['ic'][0]:.1f} a {r['ic'][1]:.1f}) MAE {r['mae']:.1f} viés {r['vies']:+.1f} | vs híbrido s8 {f3(r['vs_s8'])}"
              f" | vs híbrido sem v3 {f3(r['vs_sem_v3'])} | vs rodada 1 {f3(r['vs_r1'])} | >15 g {r['acima15']:.1f}% (IC {r['acima15_ic'][0]:.1f} a {r['acima15_ic'][1]:.1f}) | >25 g {r['acima25']:.1f}% | n5k_v1 {r['n5k_v1']:.0f}%")
        print("   por faixa:", {k: round(v) for k, v in r["por_faixa"].items()})
        dm = r["dens_massa_real"]
        print(f"   massa real × dens v3: calib {dm['calib']:.1f}% | teste {dm['teste']:.1f}% (>15 g {dm['teste_acima15']:.1f}%) | vs rodada 1 {f3(dm['vs_r1'])} | n5k_v1 {dm['n5k_v1']:.0f}% | por faixa", {k: round(v) for k, v in dm["por_faixa"].items()})
        print(f"   massa do modelo × dens v3 (sem pesos): calib {r['massa_modelo_x_dens']['calib']:.1f}% | teste {r['massa_modelo_x_dens']['teste']:.1f}% | por faixa", {k: round(v) for k, v in r["massa_modelo_x_dens"]["por_faixa"].items()})
        print("   formas [v3] (calib deixa-fora / teste / n5k_v1):")
        for n, v in r["formas_v3"].items():
            print(f"      {v['calib_loo']:5.1f}% / {v['teste']:5.1f}% / {v['n5k_v1']:3.0f}%  {n}")
        fc = r["fonte_calib"]
        print(f"   calib: fonte principal certa {fc['v3_declarada']:.1f}% (derivada {fc['v3_derivada']:.1f}%) | fração das fontes (>15 g): r {r['fracao_calib']['r_v3']:.3f}, erro abs. mediano {100 * r['fracao_calib']['erro_abs_mediano_v3']:.1f} pontos")
    # concordância entre as rodadas 1 e 2
    print("\n## concordância entre as rodadas 1 e 2")
    lg = lambda d: np.log(d + EPS_D)
    con = {}
    for s in ("calib", "teste", "n5k_v1"):
        d1, d2 = Z["r1"][f"{s}|dens_v3"], Z["r2"][f"{s}|dens_v3"]; f1, f2 = Z["r1"][f"{s}|fonte_v3"], Z["r2"][f"{s}|fonte_v3"]
        alto = real[s] > 15
        rk = lambda x: np.argsort(np.argsort(x))
        con[s] = dict(n=len(d1), r_log_dens=float(np.corrcoef(lg(d1), lg(d2))[0, 1]), r_dens=float(np.corrcoef(d1, d2)[0, 1]), spearman=float(np.corrcoef(rk(d1), rk(d2))[0, 1]),
                      r_dens_acima15=float(np.corrcoef(d1[alto], d2[alto])[0, 1]), dif_abs_mediana_pct=float(100 * np.median(np.abs(d1 - d2) / np.maximum((d1 + d2) / 2, 1e-4))),
                      dif_abs_mediana_pct_acima15=float(100 * np.median((np.abs(d1 - d2) / np.maximum((d1 + d2) / 2, 1e-4))[alto])),
                      dentro_20pct=float(100 * np.mean(np.abs(d1 - d2) <= 0.2 * np.maximum(d1, d2))), fonte_igual=float(100 * (f1 == f2).mean()),
                      fonte_igual_acima15=float(100 * (f1 == f2)[alto].mean()))
        print(f"   {s} (n={con[s]['n']}): densidade por prato r = {con[s]['r_dens']:.3f} (log: {con[s]['r_log_dens']:.3f}; Spearman {con[s]['spearman']:.3f}; >15 g: {con[s]['r_dens_acima15']:.3f})"
              f" | diferença relativa mediana {con[s]['dif_abs_mediana_pct']:.0f}% (>15 g: {con[s]['dif_abs_mediana_pct_acima15']:.0f}%) | dentro de 20% uma da outra {con[s]['dentro_20pct']:.0f}%"
              f" | mesma fonte principal declarada {con[s]['fonte_igual']:.0f}% (>15 g: {con[s]['fonte_igual_acima15']:.0f}%)")

    def troca(a, b):
        return dict(ambos=int((a & b).sum()), so_1=int((a & ~b).sum()), so_2=int((~a & b).sum()), nenhum=int((~a & ~b).sum()), trocam_pct=float(100 * (a != b).mean()))
    con["troca_teste"] = {"híbrido escolhido": troca(a_r1, ok(rt, Z["r2"]["teste|hib_v3"])),
                          "massa real × dens v3": troca(ok(rt, mt * Z["r1"]["teste|dens_v3"]), ok(rt, mt * Z["r2"]["teste|dens_v3"])),
                          "massa do modelo × dens v3": troca(ok(rt, Z["r1"]["teste|massa_modelo"] * Z["r1"]["teste|dens_v3"]), ok(rt, Z["r2"]["teste|massa_modelo"] * Z["r2"]["teste|dens_v3"])),
                          "híbrido: rodada 1 × média": troca(a_r1, ok(rt, Z["media"]["teste|hib_v3"]))}
    for k, v in con["troca_teste"].items():
        print(f"   teste, {k}: dentro nas duas {v['ambos']}, só na 1ª {v['so_1']}, só na 2ª {v['so_2']}, em nenhuma {v['nenhum']} ({v['trocam_pct']:.1f}% dos pratos trocam)")
    cf = {FONTES[i]: {FONTES[j]: int(((Z['r1']['teste|fonte_v3'] == i) & (Z['r2']['teste|fonte_v3'] == j)).sum()) for j in range(len(FONTES)) if ((Z['r1']['teste|fonte_v3'] == i) & (Z['r2']['teste|fonte_v3'] == j)).sum()} for i in range(len(FONTES))}
    con["fonte_r1_x_r2_teste"] = cf
    print("   fonte declarada, rodada 1 -> rodada 2 (teste):", cf)
    res["concordancia"] = con
    grava_json("rodadas.json", res)


if __name__ == "__main__":
    main()
