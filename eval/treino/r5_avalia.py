#!/usr/bin/env python3
"""Seção 14.10: aplica a regra R14 (14.7) à quinta execução e faz a segunda leitura permitida das partes de registro externas.

    python3 r5_avalia.py <pasta_kaggle_4> <pasta_kaggle_5> <pasta_divisoes> --so-cv      # R14 só com validação cruzada
    python3 r5_avalia.py <pasta_kaggle_4> <pasta_kaggle_5> <pasta_divisoes> --registro   # segunda leitura (S5 novo, Ad5); marca

Candidatos a servidor (14.7): as 15 combinações da quarta execução (foto inteira), as 10 do recorte pelo prato (R0c–R4c ×
H0c, H1c), o P2 com as próprias cabeças e a massa do P2 × a densidade do H0. Quem falhou na aceitação (13.7) não concorre.
Medidas: N e > 25 g na CV do treino_v2; O (OmniFood8K) e M (Flavoria, FPB, Mensa) CRUS, sem constante por domínio, com os
números com constante ao lado. Guarda: N >= 71,29 − 1,0 e > 25 g >= 51,94 − 2,5. S5 = admissível de maior
(N + O cru + M cru) / 3; empate (< 0,5): recorte, depois menos bases não comerciais, depois menos bases.
Aparelho: F4, F4c, F5 aceitos, rota e orientações por 12.3 (itens 1 e 2); Ad5 se >= U0 (68,04) + 1,0."""
import csv, json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from teste2_avalia import crit_pond, boot_cadeias
from zen_comum import ok, EPS_D, I_M, I_P, I_D
from dens_comum import DENS, faixas
from um_modelo_avalia import lg
from macros_cv import crit as crit_macro
from r4_avalia import (Dominio, constante, crit_o, dentro20, acima25, macros_ext, le_n5k, FX_O, DOM_M, NAO_COMERCIAL, GUARDA_N, GUARDA_25, EMPATE, U0_CV)

SAIDA = os.path.join(DENS, "r5")
R4 = {"R0": "gpu4 ridge so400m", "R1": "gpu4 ridge so400m +ok", "R2": "gpu4 ridge so400m +ok+fpb", "R3": "gpu4 ridge so400m +omni", "R4": "gpu4 ridge so400m +tudo"}
H4 = {"H0": "gpu4 fracoes so400m", "H1": "gpu4 fracoes so400m +omni", "H2": "gpu4 fracoes so400m +omni+ff"}
R5 = {"R0c": "gpu5 ridge so400m prato", "R1c": "gpu5 ridge so400m prato +ok", "R2c": "gpu5 ridge so400m prato +ok+fpb", "R3c": "gpu5 ridge so400m prato +omni", "R4c": "gpu5 ridge so400m prato +tudo"}
H5 = {"H0c": "gpu5 fracoes so400m prato", "H1c": "gpu5 fracoes so400m prato +omni"}
FT5 = {"F4": "gpu5 ft siglip2_b16_224 r2 fracoes-sg", "F4c": "gpu5 ft siglip2_b16_224 r2 fracoes-sg prato", "F5": "gpu5 ft siglip2_b16_224 r2 fracoes-sg +tudo", "P2": "gpu5 ftparcial so400m rotulos"}
BASES = {"R0": set(), "R1": {"flavoria", "lefood", "mensa"}, "R2": {"flavoria", "lefood", "mensa", "fpb"}, "R3": {"omnifood"}, "R4": {"flavoria", "lefood", "mensa", "fpb", "omnifood"},
         "H0": set(), "H1": {"omnifood"}, "H2": {"omnifood", "flavoria", "fpb", "mensa", "lefood"}, "P2": set()}
for k in list(BASES):
    BASES[k + "c"] = BASES[k]
BASES["F5"] = {"flavoria", "lefood", "mensa", "fpb", "omnifood"}; BASES["F4"] = BASES["F4c"] = set()
A0, S13 = "R0×H0", "R3×H0"
ORI = {1: "_o1", 4: "_o4", 8: ""}


def arq(pasta, nome):
    return os.path.join(pasta, "pred__" + nome.replace(" ", "_").replace("+", "_").replace("-", "_") + ".npz")


def carrega(p4, p5):
    Z = {k: np.load(arq(p4, n)) for k, n in {**R4, **H4}.items()}
    Z.update({k: np.load(arq(p5, n)) for k, n in {**R5, **H5, **FT5}.items()})
    N = le_n5k()
    for z in Z.values():
        assert (z["ids_treino"] == N["ids"]).all() and (z["fold"] == Z["R0"]["fold"]).all()
    return Z, N


class Vista:
    """Uma 'cabeça' vista como npz: permite usar as saídas da cabeça sem gradiente (sg) do P2 como se fossem de uma cabeça
    de frações (a densidade da regressão vem das 22 saídas da cabeça sg; a das frações, de dens_ff)."""

    def __init__(self, z, sg):
        self.z, self.sg = z, sg
        self.files = z.files

    def __getitem__(self, k):
        if self.sg and (k.startswith("oof") or k.startswith("extcv__") or k.startswith("extreg__")) and "_ff" not in k and "dens" not in k:
            k2 = k.replace("oof", "oof_sg", 1) if k.startswith("oof") else k.replace("__", "_sg__", 1)
            return self.z[k2]
        return self.z[k]


def dens_med(PH, dff):
    d_reg = np.clip(np.exp(PH[:, I_D]) - EPS_D, 0, None)
    return np.clip(np.exp((lg(d_reg) + lg(dff)) / 2) - EPS_D, 0, None)


def prot_ext(zr, zh, dom, c=None):
    """Como r4_avalia.prot, aceitando Vista."""
    pega = lambda z, ch: np.array([z[("extcv" if f == "cv" else "extreg") + ch + dom.nome][ls].mean(0) for f, ls in zip(dom.fonte, dom.linhas)])
    P_R, P_H = pega(zr, "__"), pega(zh, "__")
    dff = pega(zh, "_dens_ff__")
    logm = P_R[:, I_M] + (0 if c is None else c)
    return np.exp(logm) * dens_med(P_H, dff), np.exp(logm), P_R, P_H


def b2(cl, f, n=4000):
    """Bootstrap por cluster: (valor, 2,5%, 97,5%, 1,25%, 98,75%)."""
    g = np.unique(cl, return_inverse=True)[1]
    grupos = [np.where(g == k)[0] for k in range(g.max() + 1)]
    rng = np.random.default_rng(0)
    v = np.array([f(np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))])) for _ in range(n)])
    return (float(f(np.arange(len(cl)))),) + tuple(float(np.percentile(v, q)) for q in (2.5, 97.5, 1.25, 98.75))


def rotas_ft(z, s, P=None, ff=None):
    """As cinco rotas de 12.3 para um ajuste fino com cabeça de frações sem gradiente: massa, direta e densidade da
    regressão vêm da cabeça principal; a densidade das frações, da cabeça sg."""
    P = z["oof" + s] if P is None else P
    ff = z["oof_dens_ff" + s] if ff is None else ff
    m, dr = np.exp(P[:, I_M]), np.expm1(P[:, I_P]).clip(0)
    d_reg = np.clip(np.exp(P[:, I_D]) - EPS_D, 0, None); dm = dens_med(P, ff)
    return dict(direta=dr, massa_x_dens=m * d_reg, massa_x_dens_fracoes=m * ff, massa_x_dens_media=m * dm, geometrica=np.sqrt(dr * m * dm))


def selecao(p4, p5, pdiv):
    Z, N = carrega(p4, p5)
    real, cad = N["proteina"], N["cadeia"]
    D = {d: Dominio(d, pdiv, Z["R0"]) for d in ["omnifood"] + DOM_M + ["lefood"]}
    om = D["omnifood"]; tro = (om.papel == "treino") & np.isfinite(om.proteina)
    log = json.load(open(os.path.join(p5, "log__" + FT5["P2"].replace(" ", "_").replace("+", "_").replace("-", "_") + ".json")))
    acc = {}
    for k, n in FT5.items():
        L = json.load(open(os.path.join(p5, "log__" + n.replace(" ", "_").replace("+", "_").replace("-", "_") + ".json")))
        v = [np.mean([[x for x in L["partes"]["fold%d" % i]["hist"] if "val_rmse_logm" in x][j]["val_rmse_logm"] for i in range(5)]) for j in (0, -1)]
        acc[k] = dict(epoca0=float(v[0]), ultima=float(v[1]), passou=bool(v[1] <= v[0]))
    print("aceitação (13.7): " + "; ".join("%s %.4f -> %.4f %s" % (k, v["epoca0"], v["ultima"], "passou" if v["passou"] else "NÃO passou") for k, v in acc.items()))
    cab = {}
    for r in list(R4) + list(R5):
        cab[r] = Z[r]
    for h in list(H4) + list(H5):
        cab[h] = Z[h]
    cab["P2m"] = Z["P2"]; cab["P2h"] = Vista(Z["P2"], True)
    combos = [(r, h) for r in R4 for h in H4] + [(r, h) for r in R5 for h in H5] + [("P2m", "P2h"), ("P2m", "H0")]
    M = {}
    for r in list(R4) + list(R5) + ["P2m"]:
        M[r] = {}
        for d in DOM_M + ["lefood"]:
            dom = D[d]; tr = dom.papel == "treino"
            lm = dom.media(cab[r], "__")[:, I_M]; c = constante(dom, lm, None)
            M[r][d] = dict(cru=dentro20(dom.massa[tr], np.exp(lm)[tr]), const=dentro20(dom.massa[tr], np.exp(lm + c)[tr]),
                           mape_cru=float(100 * np.mean(np.abs(np.exp(lm) - dom.massa)[tr] / dom.massa[tr])), vies_cru=float(100 * np.median(np.exp(lm)[tr] / dom.massa[tr] - 1)),
                           k=float(np.exp(np.median(c[tr]))), est_cru=np.exp(lm), tr=tr)
        M[r]["M_cru"] = float(np.mean([M[r][d]["cru"] for d in DOM_M])); M[r]["M_const"] = float(np.mean([M[r][d]["const"] for d in DOM_M]))
    res, est_n = {}, {}
    for r, h in combos:
        nome = ("P2" if r == "P2m" and h == "P2h" else "P2×H0" if r == "P2m" else r + "×" + h)
        zr, zh = cab[r], cab[h]
        e = np.exp(zr["oof_o1"][:, I_M]) * dens_med(zh["oof_o1"], zh["oof_dens_ff_o1"]); est_n[nome] = e
        lm = om.media(zr, "__")[:, I_M]; c = constante(om, lm, None)
        e_cru = prot_ext(zr, zh, om)[0]; e_c = prot_ext(zr, zh, om, c)[0]
        bases = sorted(BASES[r.replace("m", "") if r == "P2m" else r] | BASES[h.replace("h", "") if h == "P2h" else h])
        v = dict(N=crit_pond(real, e), N25=acima25(real, e), O_cru=crit_o(om.proteina[tro], e_cru[tro]), O_const=crit_o(om.proteina[tro], e_c[tro]),
                 M_cru=M[r]["M_cru"], M_const=M[r]["M_const"], recorte=r.endswith("c"), bases=bases, nao_comerciais=len(set(bases) & NAO_COMERCIAL),
                 concorre=not r.startswith("P2") or acc["P2"]["passou"], _eo=e_cru)
        v["S14"] = (v["N"] + v["O_cru"] + v["M_cru"]) / 3; v["S_const"] = (v["N"] + v["O_const"] + v["M_const"]) / 3
        res[nome] = v
    a0 = res[A0]
    for v in res.values():
        v["admissivel"] = bool(v["concorre"] and v["N"] >= a0["N"] - GUARDA_N and v["N25"] >= a0["N25"] - GUARDA_25)
    adm = [k for k, v in res.items() if v["admissivel"]]
    smax = max(res[k]["S14"] for k in adm)
    perto = [k for k in adm if res[k]["S14"] > smax - EMPATE]
    S5 = min(perto, key=lambda k: (not res[k]["recorte"], res[k]["nao_comerciais"], len(res[k]["bases"]), -res[k]["S14"]))
    lib = [k for k in adm if res[k]["nao_comerciais"] == 0]
    L = max(lib, key=lambda k: res[k]["S14"])
    print("\nR14 (1 orientação). A0: N %.2f, > 25 g %.2f\n" % (a0["N"], a0["N25"]))
    print("%-9s %6s %6s %7s %7s %7s %7s %6s  %s" % ("cand.", "N", ">25", "O cru", "O const", "M cru", "M const", "S14", "admissível / bases"))
    for k in sorted(res, key=lambda k: -res[k]["S14"]):
        v = res[k]
        print("%-9s %6.2f %6.2f %7.2f %7.2f %7.2f %7.2f %6.2f  %s %s" % (k, v["N"], v["N25"], v["O_cru"], v["O_const"], v["M_cru"], v["M_const"], v["S14"],
              "sim" if v["admissivel"] else ("NÃO (aceitação)" if not v["concorre"] else "NÃO"), ",".join(v["bases"]) or "-"))
    print("\nS5 = %s (S14 %.2f; a menos de %.1f: %s) | melhor só com dados liberados = %s (S14 %.2f)" % (S5, res[S5]["S14"], EMPATE, perto, L, res[L]["S14"]))
    om_cl = om.cluster[tro]
    dif = {}
    for k in sorted({S5, L, "R0c×H0c", "R3c×H0c", S13} - {A0}):
        dif[k] = dict(N=boot_cadeias(cad, lambda i: crit_pond(real, est_n[k], i) - crit_pond(real, est_n[A0], i), n=2000),
                      N25=boot_cadeias(cad, lambda i: acima25(real, est_n[k], i) - acima25(real, est_n[A0], i), n=2000),
                      O_cru=boot_cadeias(np.unique(om_cl, return_inverse=True)[1], lambda i: crit_o(om.proteina[tro], res[k]["_eo"][tro], i) - crit_o(om.proteina[tro], res[A0]["_eo"][tro], i), n=2000))
        print("   %s − A0: N %+.2f (IC %+.2f a %+.2f) | > 25 g %+.2f (IC %+.2f a %+.2f) | O cru %+.2f (IC %+.2f a %+.2f)" % (k, *dif[k]["N"], *dif[k]["N25"], *dif[k]["O_cru"]))
    print("   M por domínio (cru / com constante; constante): " + " | ".join("%s: %s" % (r, ", ".join("%s %.1f/%.1f (×%.2f)" % (d, M[r][d]["cru"], M[r][d]["const"], M[r][d]["k"]) for d in DOM_M + ["lefood"])) for r in ["R0", "R3", "R4", "R0c", "R3c", "R4c", "P2m"]))
    # aparelho
    real_, ad = real, {}
    for k in ("F4", "F4c", "F5"):
        z = Z[k]
        tab = {o: {r: crit_pond(real, e) for r, e in rotas_ft(z, s).items()} for o, s in ORI.items()}
        rota = max(tab[8], key=tab[8].get); o = min(o for o in (1, 4, 8) if tab[o][rota] >= tab[8][rota] - 1.0)
        ad[k] = dict(aceito=acc[k]["passou"], rota=rota, orient=o, cv=tab[o][rota], cv8=tab[8][rota], tabela=tab, est=rotas_ft(z, ORI[o])[rota],
                     rmse=float(np.sqrt(np.mean((z["oof_o1"][:, I_M] - np.log(N["massa"])) ** 2))))
        ad[k]["vira"] = bool(ad[k]["aceito"] and ad[k]["cv"] >= U0_CV + 1.0)
        mm = {}
        for d in DOM_M:
            dom = D[d]; tr = dom.papel == "treino"; lm = dom.media(z, "__")[:, I_M]
            mm[d] = dentro20(dom.massa[tr], np.exp(lm)[tr])
        ad[k]["M_cru"] = float(np.mean(list(mm.values()))); ad[k]["M_dom"] = mm
        print("%s: rotas com 8 orientações %s -> %s, %d orientação(ões): CV %.2f; − U0 %+.2f; RMSE log-massa %.4f; M cru %.1f %s -> %s" %
              (k, {a: round(b, 2) for a, b in tab[8].items()}, rota, o, ad[k]["cv"], ad[k]["cv"] - U0_CV, ad[k]["rmse"], ad[k]["M_cru"], {a: round(b, 1) for a, b in mm.items()}, "Ad5" if ad[k]["vira"] else "não vira Ad5"))
    Ad5 = max([k for k in ad if ad[k]["vira"]], key=lambda k: ad[k]["cv"], default=None)
    print("Ad5 = %s" % (Ad5 or "nenhum: o aparelho continua o C1 de 12.6"))
    # U0 (C1) de referência na mesma lista, e o par
    zc1 = np.load(os.path.join(os.path.dirname(p4.rstrip("/")), "kaggle_out3", "pred__gpu3_ft_siglip2_b16_224_r2.npz"))
    assert (zc1["ids_treino"] == N["ids"]).all()
    e_c1 = np.exp(zc1["oof_o1"][:, I_M]) * np.clip(np.exp(zc1["oof_o1"][:, I_D]) - EPS_D, 0, None)
    e_ap = ad[Ad5]["est"] if Ad5 else e_c1
    par = np.expm1((np.log1p(est_n[S5]) + np.log1p(e_ap)) / 2)
    dpar = boot_cadeias(cad, lambda i: crit_pond(real, par, i) - crit_pond(real, est_n[S5], i), n=2000)
    print("par S5 + %s: CV %.2f; − S5 %+.2f (IC %+.2f a %+.2f) -> %s" % (Ad5 or "C1", crit_pond(real, par), *dpar, "QUALIFICA" if dpar[0] >= 2.0 and dpar[1] > 0 else "não qualifica"))
    for k in ad:
        ad[k]["dif_C1"] = boot_cadeias(cad, lambda i: crit_pond(real, ad[k]["est"], i) - crit_pond(real, e_c1, i), n=2000)
        print("   %s − C1 (U0, m×d, 1 orientação; %.2f): %+.2f (IC %+.2f a %+.2f)" % (k, crit_pond(real, e_c1), *ad[k]["dif_C1"]))
    # 12.8 na CV
    mac = {}
    for k in sorted({A0, S5, L}):
        r, h = (k.split("×") if "×" in k else ("P2m", "P2h"))
        r = "P2m" if r == "P2" else r
        e = macros_ext(cab[r]["oof_o1"], cab[h]["oof_o1"], 0.0)
        mac[k] = {n: crit_macro(N[n], e[n], n) for n in ("kcal", "carbo", "gordura")}
    print("12.8 na CV (kcal / carboidrato / gordura): " + " | ".join("%s %.1f / %.1f / %.1f" % (k, v["kcal"], v["carbo"], v["gordura"]) for k, v in mac.items()))
    sel = dict(res={k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in res.items()}, A0=A0, S5=S5, liberado=L, empate=perto, dif_A0=dif,
               M={r: {d: {a: b for a, b in v.items() if a not in ("est_cru", "tr")} for d, v in M[r].items() if isinstance(v, dict)} for r in M},
               aceitacao=acc, ad={k: {a: b for a, b in v.items() if a != "est"} for k, v in ad.items()}, Ad5=Ad5, par=dict(com=Ad5 or "C1", cv=crit_pond(real, par), dif=dpar),
               macros_cv=mac, quando=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(SAIDA, exist_ok=True)
    json.dump(sel, open(os.path.join(SAIDA, "r5_selecao.json"), "w"), indent=1, default=float, ensure_ascii=False)


def registro(p4, p5, pdiv):
    marca = os.path.join(SAIDA, "r5_lido.json")
    if os.path.exists(marca):
        sys.exit("A segunda leitura das partes de registro já foi feita (%s)." % json.load(open(marca))["quando"])
    sel = json.load(open(os.path.join(SAIDA, "r5_selecao.json")))
    novos = {}
    if sel["S5"] not in (A0, S13):
        novos["S5"] = sel["S5"]
    if sel["Ad5"]:
        novos["Ad5"] = sel["Ad5"]
    if not novos:
        sys.exit("14.7: nenhum candidato novo (S5 é o A0 ou o S de 13.11, e não há Ad5); não há segunda leitura.")
    Z, N = carcega = carrega(p4, p5)
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), novos=novos), open(marca, "w"), indent=1, ensure_ascii=False)
    cab = dict(Z); cab["P2m"] = Z["P2"]; cab["P2h"] = Vista(Z["P2"], True)
    ref = {"A0": A0, "S13": S13}
    mods = {**novos, **ref}
    D = {d: Dominio(d, pdiv, Z["R0"]) for d in ["omnifood", "flavoria", "fpb", "mensa", "lefood", "snapme"]}

    def par_rh(comb):
        if comb in FT5:
            return comb, comb
        r, h = comb.split("×") if "×" in comb else ("P2m", "P2h")
        return ("P2m" if r == "P2" else r), h

    def est_dom(comb, dom, c=None):
        r, h = par_rh(comb)
        if comb in FT5:          # ajuste fino: a rota escolhida, com a densidade das frações da cabeça sg
            z = Z[comb]
            pega = lambda ch: np.array([z[("extcv" if f == "cv" else "extreg") + ch + dom.nome][ls].mean(0) for f, ls in zip(dom.fonte, dom.linhas)])
            P = pega("__").copy()
            if c is not None:
                P[:, I_M] += c
            e = rotas_ft(None, "", P=P, ff=pega("_dens_ff__"))[sel["ad"][comb]["rota"]]
            if c is not None and sel["ad"][comb]["rota"] in ("direta", "geometrica"):
                e = e * (np.exp(c) if sel["ad"][comb]["rota"] == "direta" else np.exp(c / 2))
            return e, np.exp(P[:, I_M]), P, P
        return prot_ext(cab[r], cab[h], dom, c)

    R = dict(novos=novos, omnifood={}, massa={}, snapme={})
    om = D["omnifood"]
    for papel in ("registro", "registro_misto"):
        for rot, comb in mods.items():
            r, _ = par_rh(comb)
            lm = om.media(cab[r], "__")[:, I_M]; c = constante(om, lm, None)
            e, m, PR, PH = est_dom(comb, om, c); e0 = est_dom(comb, om)[0]
            s_ = (om.papel == papel) & np.isfinite(om.proteina); so = s_ & ~om.osso
            x = dict(n=int(s_.sum()), cru=b2(om.cluster[s_], lambda i: crit_o(om.proteina[s_], e0[s_], i)), const=b2(om.cluster[s_], lambda i: crit_o(om.proteina[s_], e[s_], i)),
                     sem_osso_cru=b2(om.cluster[so], lambda i: crit_o(om.proteina[so], e0[so], i)),
                     por_faixa_cru=[100 * float(ok(om.proteina[s_], e0[s_])[np.digitize(om.proteina[s_], FX_O, right=True) == k].mean()) for k in range(4)],
                     massa_cru=dict(dentro20=dentro20(om.massa[s_], np.exp(lm)[s_]), mape=float(100 * np.mean(np.abs(np.exp(lm) - om.massa)[s_] / om.massa[s_])), vies=float(100 * np.median(np.exp(lm)[s_] / om.massa[s_] - 1))),
                     constante=float(np.exp(c[s_][0])))
            if papel == "registro" and comb not in FT5:
                h = par_rh(comb)[1]
                for nome_c, cc in (("cru", np.zeros(len(om.g))), ("const", c)):
                    mac = macros_ext(prot_ext(cab[r], cab[h], om)[2], prot_ext(cab[r], cab[h], om)[3], cc)
                    x["macros_" + nome_c] = {}
                    for nut in ("kcal", "carbo", "gordura"):
                        v = s_ & np.isfinite(getattr(om, nut))
                        x["macros_" + nome_c][nut] = dict(crit=b2(om.cluster[v], lambda i: crit_macro(getattr(om, nut)[v], mac[nut][v], nut, i)), vies=float(100 * np.median(mac[nut][v] / np.maximum(getattr(om, nut)[v], 1e-9) - 1)))
            R["omnifood"].setdefault(papel, {})[rot] = x
            print("OmniFood8K %s, %s (%s): %d cenas | cru %.1f%% (95%%: %.1f a %.1f; 97,5%%: %.1f a %.1f) | com constante %.1f%% (%.1f a %.1f) | sem osso, cru %.1f%% | faixas cru %s | massa crua ±20%% %.1f%%, MAPE %.1f%%, viés %+.1f%% (const ×%.3f)" %
                  (papel, rot, comb, x["n"], *x["cru"], *x["const"][:3], x["sem_osso_cru"][0], "/".join("%.0f" % y for y in x["por_faixa_cru"]), x["massa_cru"]["dentro20"], x["massa_cru"]["mape"], x["massa_cru"]["vies"], x["constante"]))
            for nc in ("macros_cru", "macros_const"):
                if nc in x:
                    print("      12.8 %s: " % nc[7:] + " | ".join("%s %.1f%% (%.1f a %.1f; viés %+.0f%%)" % (n, *v["crit"][:3], v["vies"]) for n, v in x[nc].items()))
    for d in ("flavoria", "fpb", "lefood", "mensa"):
        dom = D[d]; rg = dom.papel == "registro"; ncl = len(np.unique(dom.cluster[rg]))
        for rot, comb in mods.items():
            r, _ = par_rh(comb)
            lm = dom.media(cab[r], "__")[:, I_M]; c = constante(dom, lm, None)
            m0, m = np.exp(lm), np.exp(lm + c)
            g6 = rg & (dom.massa > 600)
            f = lambda est: b2(dom.cluster[rg], lambda i: dentro20(dom.massa[rg], est[rg], i)) if ncl > 1 else (dentro20(dom.massa[rg], est[rg]),) + (np.nan,) * 4
            x = dict(n=int(rg.sum()), cru=f(m0), const=f(m), mape_cru=float(100 * np.mean(np.abs(m0 - dom.massa)[rg] / dom.massa[rg])), vies_cru=float(100 * np.median(m0[rg] / dom.massa[rg] - 1)),
                     mape_const=float(100 * np.mean(np.abs(m - dom.massa)[rg] / dom.massa[rg])), acima600_mape_cru=float(100 * np.mean(np.abs(m0 - dom.massa)[g6] / dom.massa[g6])) if g6.any() else None,
                     constante=float(np.exp(c[rg][0])))
            R["massa"].setdefault(d, {})[rot] = x
            print("%s registro, %s (%s): n %d | cru ±20%% %.1f%% (95%%: %.1f a %.1f; 97,5%%: %.1f a %.1f), MAPE %.1f%%, viés %+.1f%% | com constante %.1f%%, MAPE %.1f%% (×%.2f) | > 600 g MAPE cru %s" %
                  (d, rot, comb, x["n"], *x["cru"], x["mape_cru"], x["vies_cru"], x["const"][0], x["mape_const"], x["constante"], "%.1f%%" % x["acima600_mape_cru"] if x["acima600_mape_cru"] is not None else "-"))
    sn = D["snapme"]
    for rot, comb in mods.items():
        e, m, PR, PH = est_dom(comb, sn)
        x = {}
        pares = [("proteina", e, "proteina_fraca_g"), ("massa", m, "massa_fraca_g")]
        if comb not in FT5:
            mac = macros_ext(PR, PH, np.zeros(len(sn.g)))
            pares += [("kcal", mac["kcal"], "kcal_fraca"), ("carbo", mac["carbo"], "carbo_fraca_g"), ("gordura", mac["gordura"], "gordura_fraca_g")]
        for nome, est, col in pares:
            decl = sn.fraco[col]; v = np.isfinite(decl)
            x[nome] = boot_cadeias(np.unique(sn.cluster[v], return_inverse=True)[1], lambda i: spearmanr(est[v][i], decl[v][i])[0], n=2000)
        R["snapme"][rot] = x
        print("SNAPMe (autorrelato; foto inteira), %s: " % rot + " | ".join("%s %.2f (%.2f a %.2f)" % (k, *v) for k, v in x.items()))
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "r5_registro.json"), "w"), indent=1, default=float, ensure_ascii=False)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 3:
        sys.exit(__doc__)
    if "--so-cv" in sys.argv:
        selecao(*a)
    elif "--registro" in sys.argv:
        registro(*a)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
