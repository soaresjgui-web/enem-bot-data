#!/usr/bin/env python3
"""Seção 12.7: emenda a 12.1, aprovada pelo fundador em 12/11/2026. Leitura do teste2 de um candidato achado depois da
leitura de 12.6: um backbone congelado (SigLIP 2 So400m/16 a 384×512) com duas cabeças já treinadas na terceira execução.

    python3 um_modelo_emenda.py <pasta_kaggle_3> --so-cv     # definição congelada, só com previsões fora-do-fold
    python3 um_modelo_emenda.py <pasta_kaggle_3> --confere   # conferências sem gabarito
    python3 um_modelo_emenda.py <pasta_kaggle_3>             # leitura única (D1 e, só se passar no critério escrito, D2)

D1 = massa da cabeça ridge (`gpu3 ridge siglip2_so400m16_384x512 o8`) × densidade da cabeça de frações sem destilação
     (`gpu3 fracoes so400m`): média em log (piso 0,01) da densidade da regressão e da densidade das frações.
D2 = média em log (log1p) da proteína do D1 com a do C1 de 12.6 (U0, massa × densidade, 1 orientação).
A rota é fixa; o número de orientações do D1 sai da regra do item 2 de 12.3 (o menor entre 1, 4 e 8 a até 1,0 ponto do de
8). O D2 só é lido se passar no critério de 12.1 para um segundo modelo (2,0 pontos de CV sobre o D1 e limite inferior de
95% da diferença pareada acima de zero). O gabarito do teste2 só é aberto no modo de leitura, que grava `emenda_lido.json`
e se recusa a rodar de novo."""
import csv, json, os, sys, time
import numpy as np
from zen_comum import ok, EPS_D, I_M, I_D
from dens_comum import DENS, PREC, faixas
from teste2_avalia import crit_pond, boot_cadeias, resumo
from um_modelo_avalia import le_lista, arq, rotas, lg, AVAL2, SAIDA, ORIENT, MARGEM, MARGEM_C2

RIDGE, FRACOES, U0 = "gpu3 ridge siglip2_so400m16_384x512 o8", "gpu3 fracoes so400m", "gpu3 ft siglip2_b16_224 r2"
PROFESSOR = "gpu3 professor"
REF = {"C1 (12.6)": 66.0, "nove redes (11.5)": 69.7}


def d1(zr, zf, conj, suf):
    """Proteína (g) e massa (g) do D1 numa variante de orientações."""
    massa = np.exp(zr[conj + suf][:, I_M])
    d_reg = np.clip(np.exp(zf[conj + suf][:, I_D]) - EPS_D, 0, None)
    d_med = np.clip(np.exp((lg(d_reg) + lg(zf[conj + "_dens_ff" + suf])) / 2) - EPS_D, 0, None)
    return massa * d_med, massa


def c1(zu, conj):
    return rotas(zu, conj, "_o1")[0]["massa_x_dens"]


def par(a, b):
    return np.expm1((np.log1p(a) + np.log1p(b)) / 2)


def abre(pasta, L=None):
    Z = [np.load(arq(pasta, n), allow_pickle=False) for n in (RIDGE, FRACOES, U0)]
    if L is not None:
        for z in Z:
            assert (z["ids_treino"] == L["ids"]).all() and (z["fold"] == L["fold"]).all()
    return Z


def selecao(pasta):
    L = le_lista(); real, cad = L["proteina"], L["cadeia"]
    zr, zf, zu = abre(pasta, L)
    cv = {k: crit_pond(real, d1(zr, zf, "oof", s)[0]) for k, s in ORIENT.items()}
    k = min(k for k in ORIENT if cv[k] >= cv[8] - MARGEM)
    e1, m1 = d1(zr, zf, "oof", ORIENT[k]); ec = c1(zu, "oof"); e2 = par(e1, ec)
    ep = rotas(np.load(arq(pasta, PROFESSOR)), "oof", "")[0]["direta"]
    dif = lambda a, b: boot_cadeias(cad, lambda i: crit_pond(real, a, i) - crit_pond(real, b, i))
    S = dict(D1=dict(ridge=RIDGE, fracoes=FRACOES, rota="massa do ridge × média em log das duas densidades da cabeça de frações", cv_por_orient=cv, orient=k,
                     resumo=resumo(real, e1, cad), rmse_logm=float(np.sqrt(np.mean((np.log(m1) - np.log(L["massa"])) ** 2))), dif_c1=dif(e1, ec), dif_professor=dif(e1, ep)),
             D2=dict(resumo=resumo(real, e2, cad), dif_d1=dif(e2, e1), dif_c1=dif(e2, ec), dif_professor=dif(e2, ep)),
             professor=crit_pond(real, ep), c1=crit_pond(real, ec))
    d = S["D2"]["dif_d1"]
    S["D2"]["qualifica"] = bool(d[0] >= MARGEM_C2 and d[1] > 0)
    S["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    r = S["D1"]["resumo"]
    print("D1: CV com 8 / 4 / 1 orientações %.2f / %.2f / %.2f -> %d orientação(ões); CV %.2f (IC %.2f a %.2f); RMSE log-massa %.4f" % (cv[8], cv[4], cv[1], k, *r["ponderado"], S["D1"]["rmse_logm"]))
    print("    por faixa " + "  ".join("%s %.1f" % (f, v[0]) for f, v in r["por_faixa"].items()) + " | >25 g %.1f" % r["acima25"])
    print("    D1 − C1 %+.2f (IC %+.2f a %+.2f); D1 − professor (%.2f) %+.2f (IC %+.2f a %+.2f)" % (*S["D1"]["dif_c1"], S["professor"], *S["D1"]["dif_professor"]))
    r = S["D2"]["resumo"]
    print("D2 = média do D1 com o C1: CV %.2f (IC %.2f a %.2f); >25 g %.1f; D2 − D1 %+.2f (IC %+.2f a %+.2f) -> %s (exige >= %.1f e limite inferior > 0)" %
          (*r["ponderado"], r["acima25"], *d, "QUALIFICA" if S["D2"]["qualifica"] else "NÃO qualifica, não é lido", MARGEM_C2))
    json.dump(S, open(os.path.join(SAIDA, "emenda_selecao.json"), "w"), indent=1, default=float, ensure_ascii=False)


def confere(pasta, S):
    ids2 = np.array([r["dish_id"] for r in csv.DictReader(open(AVAL2)) if r["split"] == "teste2"])
    L = le_lista()
    assert len(ids2) == 829 and not set(ids2) & set(L["ids"])
    zr, zf, zu = abre(pasta, L)
    s = ORIENT[S["D1"]["orient"]]
    for z in (zr, zf):
        assert (z["ids_teste2"] == ids2).all() and z["teste2" + s].shape == (829, 22) and np.isfinite(z["teste2" + s]).all()
    assert zf["teste2_dens_ff" + s].shape == (829,) and np.isfinite(zf["teste2_dens_ff" + s]).all()
    c = json.load(open(os.path.join(pasta, "servico__" + RIDGE.replace(" ", "_") + ".json")))["conferencia"]
    assert c["conjunto"] == "teste2" and c["max_dif_log"] <= 0.02
    q = lambda v: [round(float(x), 2) for x in np.percentile(v, [10, 50, 90])]
    e, m = d1(zr, zf, "teste2", s); eo, mo = d1(zr, zf, "oof", s)
    R = dict(orient=S["D1"]["orient"], ridge_remontado_max_dif_log=c["max_dif_log"], fracoes_remontada="não conferida na execução (a cabeça de frações não teve medida de serviço)",
             proteina_p10_50_90=dict(teste2=q(e), cv=q(eo)), massa_p10_50_90=dict(teste2=q(m), cv=q(mo)))
    print("D1, %d orientação(ões): 829 previsões das duas cabeças na ordem de aval2_ids.csv, finitas; cabeça ridge remontada: dif. máx. em log %.4f; "
          "cabeça de frações: sem conferência de remontagem gravada" % (S["D1"]["orient"], c["max_dif_log"]))
    print("   proteína prevista p10/50/90: teste2 %s, fora-do-fold do treino %s; massa: %s, %s" % (q(e), q(eo), q(m), q(mo)))
    json.dump(R, open(os.path.join(SAIDA, "emenda_confere.json"), "w"), indent=1, ensure_ascii=False)
    return ids2


def boot_q(cadeia, estat, qs, n=4000, seed=0):
    """Mesmos sorteios de boot_cadeias, com vários pares de percentis."""
    grupos = [np.where(cadeia == c)[0] for c in np.unique(cadeia)]
    rng = np.random.default_rng(seed)
    v = np.array([estat(np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))])) for _ in range(n)])
    return {str(a): (float(np.percentile(v, (100 - a) / 2)), float(np.percentile(v, 100 - (100 - a) / 2))) for a in qs}


def leitura(pasta, S):
    marca = os.path.join(SAIDA, "emenda_lido.json")
    if os.path.exists(marca):
        sys.exit("O teste2 já foi lido para a seção 12.7 (%s). Não há outra leitura." % json.load(open(marca))["quando"])
    ids2 = confere(pasta, S)
    cs = ["D1"] + (["D2"] if S["D2"]["qualifica"] else [])
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), candidatos=cs), open(marca, "w"), indent=1, ensure_ascii=False)
    G = {r["dish_id"]: r for r in csv.DictReader(open(os.path.join(PREC, "n5k_teste2_amostra.csv")))}
    real = np.array([float(G[i]["proteina_g"]) for i in ids2]); mreal = np.array([float(G[i]["massa_g"]) for i in ids2])
    cad = np.array([int(G[i]["cadeia"]) for i in ids2]); sem_irmao = np.array(["teste" not in G[i]["cadeia_contem"].split("+") for i in ids2])
    fx = faixas(real)
    zr, zf, zu = abre(pasta)
    e1, m1 = d1(zr, zf, "teste2", ORIENT[S["D1"]["orient"]])
    E = {"D1": e1}
    if "D2" in cs:
        E["D2"] = par(e1, c1(zu, "teste2"))
    # pipelines já lidos (relato pareado): C1 de 12.6 e os da seção 11
    b = np.load(os.path.join(SAIDA, "pratos_teste2.npz")); a = np.load(os.path.join(DENS, "teste2", "pratos_teste2.npz"), allow_pickle=True)
    assert (b["ids"] == ids2).all() and np.allclose(a["real"], real) and (a["cadeia"] == cad).all() and sem_irmao.sum() == a["independente"].sum()
    lidos = {"C1 (12.6)": b["est_C1"], "nove redes (11.5)": a["est1"], "híbrido com LLM v3 (11.5)": a["est0"]}
    assert "só imagem" in str(a["nomes"][1])
    a25 = lambda e, j: 100 * np.mean([ok(real[j], e[j])[(real[j] > 25) & (real[j] <= 40)].mean(), ok(real[j], e[j])[real[j] > 40].mean()])
    R = dict(n=len(real), cadeias=int(len(np.unique(cad))), leituras=len(cs), candidatos={})
    for rot, est in E.items():
        r = resumo(real, est, cad)
        r["ic"] = boot_q(cad, lambda i: crit_pond(real, est, i), [95, 97.5, 98.33])
        r["massa"] = dict(mape=float(100 * np.mean(np.abs(m1 - mreal) / mreal)), vies_mediano=float(100 * np.median(m1 / mreal - 1)), rmse_log=float(np.sqrt(np.mean((np.log(m1) - np.log(mreal)) ** 2))),
                          por_faixa={k: dict(mape=float(100 * np.mean(np.abs(m1[f] - mreal[f]) / mreal[f])), vies_mediano=float(100 * np.median(m1[f] / mreal[f] - 1))) for k, f in fx.items()})
        r["vies_proteina_por_faixa"] = {k: float(100 * np.median(est[f] / np.maximum(real[f], 1e-9) - 1)) for k, f in fx.items() if k != "0-5"}
        r["sem_irmao_no_teste"] = dict(n=int(sem_irmao.sum()), ponderado=boot_cadeias(cad[sem_irmao], lambda j: crit_pond(real[sem_irmao], est[sem_irmao], j)))
        r["irmaos_do_teste"] = dict(n=int((~sem_irmao).sum()), ponderado=boot_cadeias(cad[~sem_irmao], lambda j: crit_pond(real[~sem_irmao], est[~sem_irmao], j)))
        r["contra"] = {n: dict(ref=crit_pond(real, e), dif=boot_cadeias(cad, lambda j: crit_pond(real, est, j) - crit_pond(real, e, j)),
                               dif_acima25=boot_cadeias(cad, lambda j: a25(est, j) - a25(e, j))) for n, e in lidos.items()}
        inf, sup = r["ponderado"][1:]
        r["veredito"] = "70% DEMONSTRADO" if inf >= 70 else "70% NÃO DEMONSTRADO" + (" (intervalo inteiro abaixo de 70%)" if sup < 70 else " (70% dentro do intervalo)")
        R["candidatos"][rot] = r
        print("\n%s" % rot)
        print("  teste2 ponderado %.1f%% (IC 95%% por cadeia %.1f a %.1f; 97,5%%: %.1f a %.1f; 98,3%%: %.1f a %.1f); simples %.1f%% (%.1f a %.1f)" %
              (*r["ponderado"], *r["ic"]["97.5"], *r["ic"]["98.33"], *r["simples"]))
        print("  por faixa: " + "  ".join("%s %.1f%% (n=%d)" % (k, v[0], v[1]) for k, v in r["por_faixa"].items()) + "  | >25 g %.1f%% | >15 g %.1f%%" % (r["acima25"], r["acima15"]))
        for n, c in r["contra"].items():
            print("  − %s (%.1f%%): %+.1f (IC %+.1f a %+.1f); acima de 25 g %+.1f (IC %+.1f a %+.1f)" % (n, c["ref"], *c["dif"], *c["dif_acima25"]))
        print("  massa: MAPE %.1f%%, viés mediano %+.1f%%, RMSE log %.4f; por faixa: " % (r["massa"]["mape"], r["massa"]["vies_mediano"], r["massa"]["rmse_log"]) +
              "  ".join("%s %.1f%% (%+.1f%%)" % (k, v["mape"], v["vies_mediano"]) for k, v in r["massa"]["por_faixa"].items()))
        print("  viés mediano da proteína por faixa: " + "  ".join("%s %+.1f%%" % kv for kv in r["vies_proteina_por_faixa"].items()))
        print("  sem irmão no teste antigo (n=%d): %.1f%% (IC %.1f a %.1f); irmãos (n=%d): %.1f%% (IC %.1f a %.1f)" %
              (r["sem_irmao_no_teste"]["n"], *r["sem_irmao_no_teste"]["ponderado"], r["irmaos_do_teste"]["n"], *r["irmaos_do_teste"]["ponderado"]))
        print("  VEREDITO PELA REGRA DE 12.1: " + r["veredito"])
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "emenda_avalia_teste2.json"), "w"), indent=1, default=float, ensure_ascii=False)
    np.savez(os.path.join(SAIDA, "emenda_pratos_teste2.npz"), ids=ids2, real=real, massa_real=mreal, cadeia=cad, massa_D1=m1, **{"est_" + k: v for k, v in E.items()})


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    if "--so-cv" in sys.argv:
        selecao(args[0])
        return
    sel = os.path.join(SAIDA, "emenda_selecao.json")
    if not os.path.exists(sel):
        sys.exit("Rode antes com --so-cv: a definição tem de estar gravada antes da leitura.")
    S = json.load(open(sel))
    S["D1"]["orient"] = int(S["D1"]["orient"])
    (confere if "--confere" in sys.argv else leitura)(args[0], S)


if __name__ == "__main__":
    main()
