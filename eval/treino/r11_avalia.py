#!/usr/bin/env python3
"""Seção 19: regra R11 (19.4), leitura dos dias de teste do FLIC e condições da leitura 3 do DFood-TJ (19.5), na CPU.

    python3 r11_avalia.py <pasta_kaggle_11> <pasta_divisoes> --so-cv          # R11: só validação cruzada (grava r11_selecao.json)
    python3 r11_avalia.py <pasta_kaggle_11> <pasta_divisoes> --flic-teste     # leitura única dos 2 dias de teste do FLIC (marca)
    python3 r11_avalia.py <pasta_kaggle_11> <pasta_kaggle_7> <pasta_kaggle_8b> <feat8__dfoodtj__recorte.npy> <conferencia_r9.json> --condicoes
                                                                              # 19.5: as três condições da leitura 3, sem rótulo
<pasta_divisoes>: onde estão as divisões das bases fechadas (divisao_r4_omnifood.csv, divisao_r4_fpb.csv); as outras vêm de
kaggle/listas/. Candidatos com a rota r4 e 4 orientações fixas (19.4); pesos da r4 pela validação cruzada (17.8 f):
4 orientações no Nutrition5k, 1 orientação nas bases externas (as previsões externas são de uma orientação)."""
import csv, json, os, sys, time
import numpy as np
from teste2_avalia import crit_pond, boot_cadeias
from zen_comum import ok, I_M, I_P
from dens_comum import DENS
from r4_avalia import Dominio, crit_o, dentro20, acima25, le_n5k, DOM_M, LISTAS
from r5_avalia import dens_med, b2
from r6_avalia import comps_n5k, comps_ext, rota, pesos_mistura

SAIDA = os.path.join(DENS, "r11")
NOMES = {"R4c11": "gpu11 ridge so400m prato +tudo", "R4c+F": "gpu11 ridge so400m prato +tudo +flic", "R1c11": "gpu11 ridge so400m prato +ok",
         "R1c+F": "gpu11 ridge so400m prato +ok +flic", "H-cap11": "gpu11 fracoes so400m prato +massa cap", "H-cap/F": "gpu11 fracoes so400m prato +massa cap mF",
         "H-cap+F": "gpu11 fracoes so400m prato +massa +flic cap mF"}
CAND = {"S7'": ("R4c11", "H-cap11"), "S11a": ("R4c+F", "H-cap/F"), "S11b": ("R4c+F", "H-cap+F")}
S7_REG = dict(N=71.36, N25=55.63, O=59.63, M=59.23)     # 17.8 b
GUARDA = dict(N=0.5, N25=1.5, O=1.0, M=1.0)              # 19.4 (1)
GANHO_FM, GANHO_FP, TOL_N_B = 5.0, 2.0, 0.3             # 19.4 (2) e (3)
F = "flic_curso"


def arq(p, nome):
    return os.path.join(p, "pred__" + "".join(c if c.isalnum() else "_" for c in nome) + ".npz")


class Flic:
    """As linhas do FLIC por curso, uma por foto × curso (sem média por prato): rótulos da lista, fold, dia, bandeja."""

    def __init__(self, z, pre):
        L = {r["dish_id"]: r for r in csv.DictReader(open(os.path.join(LISTAS, "externo_flic_curso.csv")))}
        self.ids = [str(x).split(":", 2)[2] for x in z[f"{pre}_ids__{F}"]]
        self.pre = pre
        rot = [L[i] for i in self.ids]
        self.fold = np.array(z[f"extcv_fold__{F}"]) if pre == "extcv" else np.full(len(self.ids), -1)
        self.dia = np.array([r["grupo"] for r in rot]); self.bandeja = np.array([r["bandeja"] for r in rot])
        self.fase = np.array([r["fase"] for r in rot]); self.curso = np.array([r["curso"] for r in rot])
        self.rot = rot

    def massa(self):
        return np.array([float(r["massa"]) for r in self.rot])

    def proteina_fraca(self):
        return np.array([float(r["proteina_fraca_g"]) for r in self.rot])

    def estimativas(self, zr, zh, W1):
        PR, PH = zr[f"{self.pre}__{F}"].astype(np.float64), zh[f"{self.pre}__{F}"].astype(np.float64)
        m = np.exp(PR[:, I_M]); dm = dens_med(PH, zh[f"{self.pre}_dens_ff__{F}"].astype(np.float64))
        c = (m * dm, np.expm1(PR[:, I_P]).clip(0), np.expm1(PH[:, I_P]).clip(0))
        return m, rota("r4", c, W1, self.fold)


def carrega(p11):
    Z = {k: np.load(arq(p11, n), allow_pickle=False) for k, n in NOMES.items()}
    N = le_n5k()
    for z in Z.values():
        assert list(z["ids_treino"]) == list(N["ids"]), "ordem do treino_v2"
    return Z, N


def selecao(p11, pdiv):
    Z, N = carrega(p11)
    real, cad, fold = N["proteina"], N["cadeia"], Z["R4c11"]["fold"]
    om = Dominio("omnifood", pdiv, Z["R4c11"]); tro = (om.papel == "treino") & np.isfinite(om.proteina)
    fe = np.where(om.papel == "treino", om.fold, -1)
    doms = {d: Dominio(d, pdiv, Z["R4c11"]) for d in DOM_M}
    M = {}
    for r in ("R4c11", "R4c+F", "R1c11", "R1c+F"):
        M[r] = float(np.mean([dentro20(dm.massa[dm.papel == "treino"], np.exp(dm.media(Z[r], "__")[:, I_M])[dm.papel == "treino"]) for dm in doms.values()]))
    fl = Flic(Z["R4c11"], "extcv")
    for z in Z.values():
        assert [str(x).split(":", 2)[2] for x in z[f"extcv_ids__{F}"]] == fl.ids
    mf, pf = fl.massa(), fl.proteina_fraca()
    res, est = {}, {}
    for k, (r, h) in CAND.items():
        c4 = comps_n5k(Z[r], Z[h], "_o4"); W4 = pesos_mistura(c4, real, fold)
        e = rota("r4", c4, W4, fold)
        W1 = pesos_mistura(comps_n5k(Z[r], Z[h], "_o1"), real, fold)
        eo = rota("r4", comps_ext(Z[r], Z[h], om), W1, fe)
        m_f, p_f = fl.estimativas(Z[r], Z[h], W1)
        res[k] = dict(massa=r, densidade=h, N=crit_pond(real, e), N25=acima25(real, e), O=crit_o(om.proteina[tro], eo[tro]), M=M[r],
                      Fm=dentro20(mf, m_f), Fp=crit_o(pf, p_f), Fm_vies=float(100 * np.median(m_f / mf - 1)), W4=[list(w) for w in W4], W1=[list(w) for w in W1],
                      Fm_fase={ph: dentro20(mf[fl.fase == ph], m_f[fl.fase == ph]) for ph in ("full", "left")},
                      Fm_curso={cu: dentro20(mf[fl.curso == cu], m_f[fl.curso == cu]) for cu in ("first", "second", "side")})
        est[k] = (e, m_f, p_f)
    ref = res["S7'"]
    for k in ("S11a", "S11b"):
        v, (e, m_f, p_f) = res[k], est[k]
        e7, m7, p7 = est["S7'"]
        v["dN"] = boot_cadeias(cad, lambda i: crit_pond(real, e, i) - crit_pond(real, e7, i), n=2000)
        v["dN25"] = boot_cadeias(cad, lambda i: acima25(real, e, i) - acima25(real, e7, i), n=2000)
        v["dFm"] = b2(fl.dia, lambda i: dentro20(mf[i], m_f[i]) - dentro20(mf[i], m7[i]))
        v["dFp"] = b2(fl.dia, lambda i: crit_o(pf[i], p_f[i]) - crit_o(pf[i], p7[i]))
        v["guarda"] = bool(v["N"] >= ref["N"] - GUARDA["N"] and v["N25"] >= ref["N25"] - GUARDA["N25"] and v["O"] >= ref["O"] - GUARDA["O"] and v["M"] >= ref["M"] - GUARDA["M"])
        v["ganho"] = bool(v["Fm"] - ref["Fm"] >= GANHO_FM and v["dFm"][1] > 0)
    a, b = res["S11a"], res["S11b"]
    ea, eb = est["S11a"], est["S11b"]
    b["dFp_vs_a"] = b2(fl.dia, lambda i: crit_o(pf[i], eb[2][i]) - crit_o(pf[i], ea[2][i]))
    adotado = None
    if a["guarda"] and a["ganho"]:
        adotado = "S11a"
    if b["guarda"] and b["ganho"] and b["Fp"] - a["Fp"] >= GANHO_FP and b["dFp_vs_a"][1] > 0 and b["N"] >= a["N"] - TOL_N_B:
        adotado = "S11b"
    # relatos: massa só com dados liberados; sementes
    lib = {}
    for r in ("R1c11", "R1c+F"):
        m_l = np.exp(Z[r][f"extcv__{F}"][:, I_M].astype(np.float64))
        lib[r] = dict(M=M[r], Fm=dentro20(mf, m_l), Fm_vies=float(100 * np.median(m_l / mf - 1)))
    sem = {}
    for h, r in (("H-cap11", "R4c11"), ("H-cap/F", "R4c+F"), ("H-cap+F", "R4c+F")):
        vals = []
        for s in range(3):
            zs = {**{c: Z[h][c] for c in Z[h].files}, **{c.replace("oof_s%d" % s, "oof"): Z[h][c] for c in Z[h].files if c.startswith("oof_s%d" % s)}}
            c4 = comps_n5k(Z[r], zs, "_o4")
            vals.append(crit_pond(real, rota("r4", c4, pesos_mistura(c4, real, fold), fold)))
        sem[h] = vals
    print("R11 (19.4). S7 registrado (17.8 b): N_c %.2f, > 25 g %.2f, O cru %.2f, M cru %.2f" % (S7_REG["N"], S7_REG["N25"], S7_REG["O"], S7_REG["M"]))
    print("%-6s %-16s %6s %6s %6s %6s | %6s %7s %6s | full/left  first/second/side" % ("", "massa × densidade", "N_c", ">25", "O cru", "M cru", "F_m", "viés", "F_p"))
    for k, v in res.items():
        print("%-6s %-16s %6.2f %6.2f %6.2f %6.2f | %6.1f %+6.0f%% %6.1f | %s  %s" % (k, v["massa"] + "×" + v["densidade"], v["N"], v["N25"], v["O"], v["M"], v["Fm"], v["Fm_vies"], v["Fp"],
              "/".join("%.0f" % x for x in v["Fm_fase"].values()), "/".join("%.0f" % x for x in v["Fm_curso"].values())))
    for k in ("S11a", "S11b"):
        v = res[k]
        print("%s − S7': N_c %+.2f (IC %+.2f a %+.2f) | > 25 g %+.2f (IC %+.2f a %+.2f) | F_m %+.1f (IC por dia %+.1f a %+.1f) | F_p fraca %+.1f (IC %+.1f a %+.1f) | guarda %s, ganho %s"
              % (k, *v["dN"], *v["dN25"], *v["dFm"][:3], *v["dFp"][:3], "sim" if v["guarda"] else "NÃO", "sim" if v["ganho"] else "NÃO"))
    print("S11b − S11a, F_p fraca: %+.1f (IC por dia %+.1f a %+.1f)" % b["dFp_vs_a"][:3])
    print("sementes, N_c (r4, 4 orientações): " + " | ".join("%s %s" % (h, "/".join("%.2f" % x for x in v)) for h, v in sem.items()))
    print("massa só com dados liberados: " + " | ".join("%s M cru %.2f, F_m %.1f (viés %+.0f%%)" % (r, v["M"], v["Fm"], v["Fm_vies"]) for r, v in lib.items()))
    print("\n-> adotado pela R11: %s" % (adotado or "nenhum (o app fica com o S7)"))
    sel = dict(res=res, adotado=adotado, sementes=sem, liberado=lib, n_flic_cv=len(fl.ids), dias_cv=int(len(np.unique(fl.dia))), quando=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(SAIDA, exist_ok=True)
    json.dump(sel, open(os.path.join(SAIDA, "r11_selecao.json"), "w"), indent=1, default=float, ensure_ascii=False)


def flic_teste(p11, pdiv):
    """19.5: leitura única dos 2 dias de teste do FLIC, para o S7' e o S11 adotado (ou o S11a, como relato). IC por bandeja."""
    marca = os.path.join(SAIDA, "r11_flic_teste_lido.json")
    if os.path.exists(marca):
        sys.exit("Os dias de teste do FLIC já foram lidos (%s)." % json.load(open(marca))["quando"])
    sel = json.load(open(os.path.join(SAIDA, "r11_selecao.json")))
    alvo = sel["adotado"] or "S11a"
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), modelos=["S7'", alvo]), open(marca, "w"), indent=1, ensure_ascii=False)
    Z, _ = carrega(p11)
    fl = Flic(Z["R4c11"], "extreg")
    mf, pf = fl.massa(), fl.proteina_fraca()
    R = dict(n=len(fl.ids), bandejas=int(len(np.unique(fl.bandeja))), dias=sorted(set(fl.dia)), alvo=alvo, adotado=sel["adotado"])
    est = {}
    for k in ("S7'", alvo):
        r, h = CAND[k]
        W1 = [np.array(w) for w in sel["res"][k]["W1"]]
        m, p = fl.estimativas(Z[r], Z[h], W1)
        est[k] = (m, p)
        R[k] = dict(massa20=b2(fl.bandeja, lambda i: dentro20(mf[i], m[i])), vies=float(100 * np.median(m / mf - 1)), Fp=b2(fl.bandeja, lambda i: crit_o(pf[i], p[i])),
                    por_fase={ph: dentro20(mf[fl.fase == ph], m[fl.fase == ph]) for ph in ("full", "left")})
        print("FLIC, dias de teste %s (%d linhas, %d bandejas), %s: massa a ±20%% %.1f%% (IC por bandeja %.1f a %.1f), viés mediano %+.0f%% | proteína FRACA %.1f%% (%.1f a %.1f) | antes/sobra %s"
              % (R["dias"], R["n"], R["bandejas"], k, *R[k]["massa20"][:3], R[k]["vies"], *R[k]["Fp"][:3], "/".join("%.0f" % x for x in R[k]["por_fase"].values())))
    m7, mA = est["S7'"][0], est[alvo][0]
    R["dif_massa20"] = b2(fl.bandeja, lambda i: dentro20(mf[i], mA[i]) - dentro20(mf[i], m7[i]))
    print("%s − S7', massa a ±20%%: %+.1f (IC por bandeja %+.1f a %+.1f; 2 dias: o IC não carrega a variação entre dias)" % (alvo, *R["dif_massa20"][:3]))
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "r11_flic_teste.json"), "w"), indent=1, default=float, ensure_ascii=False)


def preve_dfoodtj(feat, ridge, cab):
    """Previsões de uma R4c (ridge) e de uma cabeça de densidade com massa aninhada sobre as features [n, 8, 2304] da
    janela da 8b: (log-saídas da ridge [n, 8, 22], log-saídas da cabeça [n, 8, 22], densidade pelas frações [n, 8])."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle"))
    import treino_gpu as G
    X = feat.astype(np.float32)
    PR = ((((X - ridge["mu"]) / ridge["sd"]).astype(np.float64)) @ ridge["W"] + ridge["mt"])
    Xh = np.concatenate([X, PR[..., I_M:I_M + 1].astype(np.float32)], -1)
    PH, FF = G.aplica_cabeca4(cab, Xh)
    return PR, PH, FF @ cab["dens_fonte"]


def condicoes(p11, p7, p8, arq_feat, conf):
    """19.5: as três condições da leitura 3, todas sem rótulo. Grava r11_condicoes.json."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import r8_avalia as R8
    sel = json.load(open(os.path.join(SAIDA, "r11_selecao.json")))
    z8 = np.load(os.path.join(p8, "pred8__dfoodtj.npz"), allow_pickle=False)
    ids = [str(x) for x in z8["ids"]]
    feat = np.load(arq_feat)
    assert feat.shape == (len(ids), 8, 2304), feat.shape
    C = {}
    # 3. o caminho da CPU refaz as previsões gravadas do S7 (R4c e H-cap da sétima)
    r7 = np.load(os.path.join(p7, "cabeca__gpu7_ridge_so400m_prato__tudo.npz")); h7 = np.load(os.path.join(p7, "cabeca__gpu7_fracoes_so400m_prato__massa_cap.npz"))
    PR, PH, DF = preve_dfoodtj(feat, r7, h7)
    dif = dict(r4c=float(np.abs(PR - z8["recorte__r4c"]).max()), hcap=float(np.abs(PH - z8["recorte__hcap__s"]).max()),
               dens_ff=float(np.abs(np.log(DF + 0.01) - np.log(z8["recorte__hcap__dens_ff"].astype(np.float64) + 0.01)).max()))
    C["c3_remontagem"] = dict(dif_max_log=dif, passou=bool(max(dif.values()) <= 0.01))
    # população da leitura 2
    manter, entrada, reg = R8.conferencia(conf, z8)
    assert entrada["2020-2023"] == "recorte" and reg["sha256"].startswith("59916f69")
    C["populacao"] = dict(n=int(manter.sum()), conferencia_sha256=reg["sha256"])
    adot = sel["adotado"]
    C["c1_adotado"] = dict(adotado=adot, passou=adot is not None)
    s7 = R8.estimativas(z8, "recorte")
    alvo = adot or "S11a"
    r, h = CAND[alvo]
    nome_c = lambda n: os.path.join(p11, "cabeca__" + "".join(c if c.isalnum() else "_" for c in NOMES[n]) + ".npz")
    PRn, PHn, DFn = preve_dfoodtj(feat, np.load(nome_c(r)), np.load(nome_c(h)))
    o4 = lambda P: P[:, :4].mean(1)
    m = np.exp(o4(PRn)[:, I_M]); r1 = m * dens_med(o4(PHn), o4(DFn[..., None])[:, 0])
    W4 = np.array(sel["res"][alvo]["W4"][5])
    X = np.c_[np.log1p(r1), np.log1p(np.expm1(o4(PRn)[:, I_P]).clip(0)), np.log1p(np.expm1(o4(PHn)[:, I_P]).clip(0)), np.ones(len(r1))]
    p_novo = np.expm1(X @ W4).clip(0)
    k = manter
    dm = float(np.median(np.abs(np.log(m[k] / s7["massa"][k]))))
    dp = float(np.median(np.abs(np.log((p_novo[k] + 1e-6) / (s7["S7"][k] + 1e-6)))))
    C["c2_mudanca"] = dict(modelo=alvo, mediana_abs_log_massa=dm, mediana_abs_log_proteina=dp, razao_mediana_massa=float(np.median(m[k] / s7["massa"][k])),
                           razao_mediana_proteina=float(np.median(p_novo[k] / np.maximum(s7["S7"][k], 1e-6))), limiar=float(np.log(1.10)),
                           passou=bool(max(dm, dp) >= np.log(1.10)))
    C["passou"] = bool(C["c1_adotado"]["passou"] and C["c2_mudanca"]["passou"] and C["c3_remontagem"]["passou"])
    C["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print("Leitura 3 do DFood-TJ, condições (19.5), sem rótulo, em %d refeições da leitura 2:" % C["populacao"]["n"])
    print("  1. S11 adotado: %s" % (adot or "não (nenhum S11 passou na R11)"))
    print("  2. %s × S7: mediana |log razão| massa %.3f (razão mediana %.2f), proteína %.3f (razão mediana %.2f); limiar log(1,10) = %.3f -> %s"
          % (alvo, dm, C["c2_mudanca"]["razao_mediana_massa"], dp, C["c2_mudanca"]["razao_mediana_proteina"], np.log(1.10), "passa" if C["c2_mudanca"]["passou"] else "NÃO passa"))
    print("  3. remontagem do S7 na CPU: diferença máxima em log %s -> %s" % ({k_: round(v, 5) for k_, v in dif.items()}, "passa" if C["c3_remontagem"]["passou"] else "NÃO passa"))
    print("  -> leitura 3: %s" % ("ACONTECE" if C["passou"] else "não acontece"))
    np.savez(os.path.join(SAIDA, "r11_dfoodtj_previsoes.npz"), ids=np.array(ids), massa=m, proteina=p_novo, modelo=alvo)
    json.dump(C, open(os.path.join(SAIDA, "r11_condicoes.json"), "w"), indent=1, default=float, ensure_ascii=False)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--so-cv" in sys.argv and len(a) == 2:
        return selecao(*a)
    if "--flic-teste" in sys.argv and len(a) == 2:
        return flic_teste(*a)
    if "--condicoes" in sys.argv and len(a) == 5:
        return condicoes(*a)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
