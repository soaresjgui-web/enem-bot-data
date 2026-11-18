#!/usr/bin/env python3
"""Seção 15.8: aplica a regra R15 (15.5) à sexta execução, no protocolo do recorte, e faz a segunda leitura permitida das
partes de registro externas (REF e o candidato final do app).

    python3 r6_avalia.py <pasta_kaggle_5> <pasta_kaggle_6> <pasta_divisoes> --so-cv
    python3 r6_avalia.py <pasta_kaggle_5> <pasta_kaggle_6> <pasta_divisoes> --registro

Rotas (15.2), com m = massa da cabeça de massa, d_med = média em log das duas densidades da cabeça de densidade,
dR = proteína direta da cabeça de massa, dH = proteína direta da cabeça de densidade:
  r1 = m × d_med;  r2 = sqrt(r1 · dR);  r3 = (r1 · dR · dH)^(1/3);
  r4 = mistura log-linear de log1p(r1), log1p(dR), log1p(dH) e intercepto, com os pesos de cada fold ajustados nos outros
       quatro folds do Nutrition5k (nas fotos externas de treino, os pesos do fold em que a foto valida; nas partes de
       registro, os pesos ajustados no Nutrition5k inteiro).
Orientações (só no Nutrition5k) pela regra de 12.3, item 2. REF = R4c×H0c, r1, 1 orientação."""
import json, os, sys, time
import numpy as np
from scipy.stats import spearmanr
from teste2_avalia import crit_pond, boot_cadeias
from zen_comum import ok, EPS_D, I_M, I_P, I_D
from dens_comum import DENS
from um_modelo_avalia import lg
from macros_cv import crit as crit_macro
from r4_avalia import Dominio, constante, crit_o, dentro20, acima25, macros_ext, le_n5k, FX_O, DOM_M
from r5_avalia import dens_med, rotas_ft, b2

SAIDA = os.path.join(DENS, "r6")
ORI = {1: "_o1", 4: "_o4", 8: ""}
MASSA = {"R4c": ("p5", "gpu5 ridge so400m prato +tudo"), "R1c": ("p5", "gpu5 ridge so400m prato +ok")}
DENSI = {"H0c": ("p5", "gpu5 fracoes so400m prato"), "H0c-m": ("p6", "gpu6 fracoes so400m prato +massa")}
DISP = {"F4c": ("p5", "gpu5 ft siglip2_b16_224 r2 fracoes-sg prato"), "F5c": ("p6", "gpu6 ft siglip2_b16_224 r2 fracoes-sg prato +tudo")}
P3 = ("p6", "gpu6 ftparcial so400m prato rotulos +tudo")
ROTAS = ["r1", "r2", "r3", "r4"]
REF = ("R4c", "H0c", "r1", 1)
NC = {"R4c": 2, "R1c": 0, "H0c": 0, "H0c-m": 0}       # bases não comerciais (FPB, OmniFood8K) em cada cabeça
NB = {"R4c": 5, "R1c": 3, "H0c": 0, "H0c-m": 0}
U = dict(GUARDA_N=1.0, GUARDA_25=2.5, EMPATE=0.5, AD=1.0, PAR=2.0)


def arq(p, nome):
    return os.path.join(p, "pred__" + nome.replace(" ", "_").replace("+", "_").replace("-", "_") + ".npz")


def carrega(p5, p6):
    P = dict(p5=p5, p6=p6)
    Z = {k: np.load(arq(P[a], n)) for k, (a, n) in {**MASSA, **DENSI, **DISP, "P3": P3}.items()}
    N = le_n5k()
    for z in Z.values():
        assert (z["ids_treino"] == N["ids"]).all()
    return Z, N


def comps_n5k(zr, zh, s):
    R, H = zr["oof" + s], zh["oof" + s]
    r1 = np.exp(R[:, I_M]) * dens_med(H, zh["oof_dens_ff" + s])
    return r1, np.expm1(R[:, I_P]).clip(0), np.expm1(H[:, I_P]).clip(0)


def comps_ext(zr, zh, dom, c=None):
    pega = lambda z, ch: np.array([z[("extcv" if f == "cv" else "extreg") + ch + dom.nome][ls].mean(0) for f, ls in zip(dom.fonte, dom.linhas)])
    PR, PH = pega(zr, "__"), pega(zh, "__")
    k = 1.0 if c is None else np.exp(c)
    r1 = np.exp(PR[:, I_M]) * dens_med(PH, pega(zh, "_dens_ff__"))
    return k * r1, k * np.expm1(PR[:, I_P]).clip(0), k * np.expm1(PH[:, I_P]).clip(0), PR, PH


def pesos_mistura(c, real, fold):
    """Pesos da r4 por fold (ajustados nos outros quatro) e com o Nutrition5k inteiro (índice 5)."""
    X = np.c_[np.log1p(c[0]), np.log1p(c[1]), np.log1p(c[2]), np.ones(len(real))]; y = np.log1p(real)
    W = [np.linalg.lstsq(X[fold != k], y[fold != k], rcond=None)[0] for k in range(5)]
    return W + [np.linalg.lstsq(X, y, rcond=None)[0]]


def rota(nome, c, W=None, fold=None):
    r1, dR, dH = c[:3]
    if nome == "r1":
        return r1
    if nome == "r2":
        return np.sqrt(r1 * dR)
    if nome == "r3":
        return np.cbrt(r1 * dR * dH)
    X = np.c_[np.log1p(r1), np.log1p(dR), np.log1p(dH), np.ones(len(r1))]
    out = np.array([X[i] @ W[5 if fold is None or fold[i] < 0 else fold[i]] for i in range(len(r1))])
    return np.expm1(out).clip(0)


def selecao(p5, p6, pdiv):
    Z, N = carrega(p5, p6)
    real, cad, fold = N["proteina"], N["cadeia"], Z["R4c"]["fold"]
    D = {d: Dominio(d, pdiv, Z["R4c"]) for d in ["omnifood"] + DOM_M + ["lefood"]}
    om = D["omnifood"]; tro = (om.papel == "treino") & np.isfinite(om.proteina)
    acc = {}
    for k, (a, n) in {**DISP, "P3": P3}.items():
        L = json.load(open(os.path.join(dict(p5=p5, p6=p6)[a], "log__" + n.replace(" ", "_").replace("+", "_").replace("-", "_") + ".json")))
        v = []
        for i in range(5):
            h = L["partes"]["fold%d" % i]["hist"]
            val = [x for x in h if "val_rmse_logm" in x and isinstance(x.get("epoca"), int)]
            esc = [x for x in h if x.get("epoca") == "escolhida"]
            ult = val[-1]["val_rmse_logm"]
            if esc:
                e = esc[0]["epoca_escolhida"]; ult = [x for x in val if x["epoca"] == e][0]["val_rmse_logm"] if any(x["epoca"] == e for x in val) else esc[0].get("val_rmse_logm", ult)
            v.append((val[0]["val_rmse_logm"], ult))
        acc[k] = dict(epoca0=float(np.mean([a for a, b in v])), ultima=float(np.mean([b for a, b in v])))
        acc[k]["passou"] = bool(acc[k]["ultima"] <= acc[k]["epoca0"])
    print("aceitação (13.7): " + "; ".join("%s %.4f -> %.4f %s" % (k, v["epoca0"], v["ultima"], "passou" if v["passou"] else "NÃO passou") for k, v in acc.items()))
    # M: só depende da cabeça de massa
    M = {}
    for r in MASSA:
        M[r] = {}
        for d in DOM_M + ["lefood"]:
            dom = D[d]; tr = dom.papel == "treino"
            lm = dom.media(Z[r], "__")[:, I_M]; c = constante(dom, lm, None)
            M[r][d] = dict(cru=dentro20(dom.massa[tr], np.exp(lm)[tr]), const=dentro20(dom.massa[tr], np.exp(lm + c)[tr]))
        M[r]["M_cru"] = float(np.mean([M[r][d]["cru"] for d in DOM_M])); M[r]["M_const"] = float(np.mean([M[r][d]["const"] for d in DOM_M]))
    res, est = {}, {}
    for r in MASSA:
        for h in DENSI:
            cs = {o: comps_n5k(Z[r], Z[h], s) for o, s in ORI.items()}
            W = pesos_mistura(cs[1], real, fold)
            lm = om.media(Z[r], "__")[:, I_M]; c = constante(om, lm, None)
            ce, cc = comps_ext(Z[r], Z[h], om), comps_ext(Z[r], Z[h], om, c)
            for ro in ROTAS:
                tab = {}
                for o in ORI:
                    Wo = pesos_mistura(cs[o], real, fold) if ro == "r4" else None
                    tab[o] = crit_pond(real, rota(ro, cs[o], Wo, fold))
                o = min(x for x in (1, 4, 8) if tab[x] >= tab[8] - 1.0)
                Wo = pesos_mistura(cs[o], real, fold) if ro == "r4" else None
                e = rota(ro, cs[o], Wo, fold)
                fe = np.where(om.papel == "treino", om.fold, -1)
                eo, eoc = rota(ro, ce, W, fe), rota(ro, cc, W, fe)
                nome = "%s×%s %s o%d" % (r, h, ro, o)
                v = dict(massa=r, densidade=h, rota=ro, orient=o, tabela=tab, N=tab[o], N25=acima25(real, e), O_cru=crit_o(om.proteina[tro], eo[tro]),
                         O_const=crit_o(om.proteina[tro], eoc[tro]), M_cru=M[r]["M_cru"], M_const=M[r]["M_const"], nc=NC[r] + NC[h], nb=NB[r] + NB[h], W=[list(w) for w in W])
                v["S15"] = (v["N"] + v["O_cru"] + v["M_cru"]) / 3
                res[nome] = v; est[nome] = e
                # a REF é a R4c×H0c r1 com 1 orientação, fixa
                if (r, h, ro) == REF[:3]:
                    nref = "REF R4c×H0c r1 o1"
                    e1 = rota("r1", cs[1])
                    res[nref] = dict(v, orient=1, N=tab[1], N25=acima25(real, e1)); res[nref]["S15"] = (res[nref]["N"] + v["O_cru"] + v["M_cru"]) / 3
                    est[nref] = e1
    ref = res["REF R4c×H0c r1 o1"]
    for k, v in res.items():
        v["admissivel"] = bool(v["N"] >= ref["N"] - U["GUARDA_N"] and v["N25"] >= ref["N25"] - U["GUARDA_25"])
    cand = {k: v for k, v in res.items() if v["massa"] == "R4c"}
    adm = [k for k in cand if cand[k]["admissivel"]]
    smax = max(res[k]["S15"] for k in adm)
    perto = [k for k in adm if res[k]["S15"] > smax - U["EMPATE"]]
    ordem_rota = {r: i for i, r in enumerate(ROTAS)}
    S6 = min(perto, key=lambda k: (res[k]["orient"], res[k]["nc"], ordem_rota[res[k]["rota"]], res[k]["nb"], -res[k]["S15"]))
    lib = [k for k in res if res[k]["massa"] == "R1c" and res[k]["densidade"] == "H0c" and res[k]["admissivel"]]
    L = max(lib, key=lambda k: res[k]["S15"]) if lib else None
    print("\nR15 (protocolo no recorte). REF: N_c %.2f, > 25 g %.2f, O cru %.2f, M cru %.2f, S15 %.2f\n" % (ref["N"], ref["N25"], ref["O_cru"], ref["M_cru"], ref["S15"]))
    print("%-24s %6s %6s %7s %7s %6s %6s  %s" % ("candidato", "N_c", ">25", "O cru", "O const", "M cru", "S15", "guarda | N_c com 1/4/8"))
    for k in sorted(res, key=lambda k: -res[k]["S15"]):
        v = res[k]
        print("%-24s %6.2f %6.2f %7.2f %7.2f %6.2f %6.2f  %s | %s" % (k, v["N"], v["N25"], v["O_cru"], v["O_const"], v["M_cru"], v["S15"], "sim" if v["admissivel"] else "não", " / ".join("%.2f" % v["tabela"][o] for o in (1, 4, 8))))
    print("\nS6 = %s (S15 %.2f; a menos de %.1f: %s) | melhor só com dados liberados = %s" % (S6, res[S6]["S15"], U["EMPATE"], perto, L))
    dif = {}
    for k in sorted({S6} | ({L} if L else set()) - {"REF R4c×H0c r1 o1"}):
        dif[k] = dict(N=boot_cadeias(cad, lambda i: crit_pond(real, est[k], i) - crit_pond(real, est["REF R4c×H0c r1 o1"], i), n=2000),
                      N25=boot_cadeias(cad, lambda i: acima25(real, est[k], i) - acima25(real, est["REF R4c×H0c r1 o1"], i), n=2000))
        print("   %s − REF: N_c %+.2f (IC %+.2f a %+.2f) | > 25 g %+.2f (IC %+.2f a %+.2f)" % (k, *dif[k]["N"], *dif[k]["N25"]))
    # aparelho no recorte
    ad, e_ad = {}, {}
    for k in DISP:
        z = Z[k]
        tab = {o: {r: crit_pond(real, e) for r, e in rotas_ft(z, s).items()} for o, s in ORI.items()}
        ro = max(tab[8], key=tab[8].get); o = min(x for x in (1, 4, 8) if tab[x][ro] >= tab[8][ro] - 1.0)
        e_ad[k] = rotas_ft(z, ORI[o])[ro]
        ad[k] = dict(aceito=acc.get(k, dict(passou=True))["passou"], rota=ro, orient=o, cv=tab[o][ro], tabela=tab)
        mm = {}
        for d in DOM_M:
            dom = D[d]; tr = dom.papel == "treino"; mm[d] = dentro20(dom.massa[tr], np.exp(dom.media(z, "__")[:, I_M])[tr])
        ad[k]["M_cru"] = float(np.mean(list(mm.values()))); ad[k]["M_dom"] = mm
        print("%s: %s, %d orientação(ões): N_c %.2f (rotas com 8: %s) | M cru %.1f %s" % (k, ro, o, ad[k]["cv"], {a: round(b, 2) for a, b in tab[8].items()}, ad[k]["M_cru"], {a: round(b, 1) for a, b in mm.items()}))
    ref_ad = ad["F4c"]["cv"]
    Ad6 = "F5c" if ad["F5c"]["aceito"] and ad["F5c"]["cv"] >= ref_ad + U["AD"] else "F4c"
    dad = boot_cadeias(cad, lambda i: crit_pond(real, e_ad["F5c"], i) - crit_pond(real, e_ad["F4c"], i), n=2000)
    print("F5c − F4c: %+.2f (IC %+.2f a %+.2f) -> Ad6 = %s" % (*dad, Ad6))
    par = np.expm1((np.log1p(est[S6]) + np.log1p(e_ad[Ad6])) / 2)
    dpar = boot_cadeias(cad, lambda i: crit_pond(real, par, i) - crit_pond(real, est[S6], i), n=2000)
    qual = bool(dpar[0] >= U["PAR"] and dpar[1] > 0)
    print("par S6 + %s: N_c %.2f (> 25 g %.2f); − S6 %+.2f (IC %+.2f a %+.2f) -> %s" % (Ad6, crit_pond(real, par), acima25(real, par), *dpar, "QUALIFICA" if qual else "não qualifica"))
    # o par também com a REF, só para registro
    par_ref = np.expm1((np.log1p(est["REF R4c×H0c r1 o1"]) + np.log1p(e_ad[Ad6])) / 2)
    print("   (registro) par REF + %s: N_c %.2f" % (Ad6, crit_pond(real, par_ref)))
    # P3, que não concorre: números para registro
    zp = Z["P3"]
    p3 = {}
    for h, zh in (("HP3", zp), ("H0c", Z["H0c"])):
        r1 = np.exp(zp["oof_o1"][:, I_M]) * dens_med(zh["oof_hp3_o1"] if h == "HP3" else zh["oof_o1"], zh["oof_dens_ff_o1"])
        p3[h] = crit_pond(real, r1)
    print("P3 (não concorre): N_c r1 1 orientação com HP3 %.2f, com H0c %.2f; RMSE log-massa 1 orientação %.4f (R4c %.4f)" %
          (p3["HP3"], p3["H0c"], np.sqrt(np.mean((zp["oof_o1"][:, I_M] - np.log(N["massa"])) ** 2)), np.sqrt(np.mean((Z["R4c"]["oof_o1"][:, I_M] - np.log(N["massa"])) ** 2))))
    final = ("par", S6, Ad6) if qual else ("S6", S6, None)
    print("candidato final do app: %s" % (final,))
    mac = {}
    for k in sorted({"REF R4c×H0c r1 o1", S6}):
        r, h = res[k]["massa"], res[k]["densidade"]
        e = macros_ext(Z[r]["oof_o1"], Z[h]["oof_o1"], 0.0)
        mac[k] = {n: crit_macro(N[n], e[n], n) for n in ("kcal", "carbo", "gordura")}
    print("12.8 na CV, 1 orientação (kcal / carboidrato / gordura): " + " | ".join("%s %.1f / %.1f / %.1f" % (k, v["kcal"], v["carbo"], v["gordura"]) for k, v in mac.items()))
    sel = dict(res=res, REF="REF R4c×H0c r1 o1", S6=S6, liberado=L, empate=perto, dif_REF=dif, aceitacao=acc, ad=ad, Ad6=Ad6, dif_ad=dad,
               par=dict(com=Ad6, N=crit_pond(real, par), dif=dpar, qualifica=qual), P3=p3, final=final, macros_cv=mac, quando=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(SAIDA, exist_ok=True)
    json.dump(sel, open(os.path.join(SAIDA, "r6_selecao.json"), "w"), indent=1, default=float, ensure_ascii=False)


def registro(p5, p6, pdiv):
    marca = os.path.join(SAIDA, "r6_lido.json")
    if os.path.exists(marca):
        sys.exit("A segunda leitura das partes de registro (15.5) já foi feita (%s)." % json.load(open(marca))["quando"])
    sel = json.load(open(os.path.join(SAIDA, "r6_selecao.json")))
    Z, N = carrega(p5, p6)
    real, fold = N["proteina"], Z["R4c"]["fold"]
    mods = {"REF": ("REF R4c×H0c r1 o1", None)}
    kind, s6, ad = sel["final"]
    if kind == "par":
        mods["final"] = (s6, ad)
    elif s6 != sel["REF"]:
        mods["final"] = (s6, None)
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), modelos=mods), open(marca, "w"), indent=1, ensure_ascii=False)
    D = {d: Dominio(d, pdiv, Z["R4c"]) for d in ["omnifood", "flavoria", "fpb", "mensa", "lefood", "snapme"]}

    def protein(nome, dom, c=None):
        v = sel["res"][nome]; r, h = v["massa"], v["densidade"]
        cs = comps_ext(Z[r], Z[h], dom, c)
        W = [np.array(w) for w in v["W"]]
        fe = np.where(dom.papel == "treino", dom.fold, -1)
        return rota(v["rota"], cs, W, fe), cs

    def estima(m, dom, c=None):
        nome, adk = m
        e, cs = protein(nome, dom, c)
        if adk:
            z = Z[adk]
            pega = lambda ch: np.array([z[("extcv" if f == "cv" else "extreg") + ch + dom.nome][ls].mean(0) for f, ls in zip(dom.fonte, dom.linhas)])
            P = pega("__").copy()
            if c is not None:
                P[:, I_M] += c
            ea = rotas_ft(None, "", P=P, ff=pega("_dens_ff__"))[sel["ad"][adk]["rota"]]
            if c is not None and sel["ad"][adk]["rota"] in ("direta", "geometrica"):
                ea = ea * (np.exp(c) if sel["ad"][adk]["rota"] == "direta" else np.exp(c / 2))
            e = np.expm1((np.log1p(e) + np.log1p(ea)) / 2)
        return e, cs

    R = dict(modelos=mods, omnifood={}, massa={}, snapme={})
    om = D["omnifood"]
    for papel in ("registro", "registro_misto"):
        for rot, m in mods.items():
            lm = om.media(Z["R4c"], "__")[:, I_M]; c = constante(om, lm, None)
            e0, cs0 = estima(m, om); e1, _ = estima(m, om, c)
            s_ = (om.papel == papel) & np.isfinite(om.proteina); so = s_ & ~om.osso
            x = dict(n=int(s_.sum()), cru=b2(om.cluster[s_], lambda i: crit_o(om.proteina[s_], e0[s_], i)), const=b2(om.cluster[s_], lambda i: crit_o(om.proteina[s_], e1[s_], i)),
                     sem_osso_cru=b2(om.cluster[so], lambda i: crit_o(om.proteina[so], e0[so], i)),
                     por_faixa_cru=[100 * float(ok(om.proteina[s_], e0[s_])[np.digitize(om.proteina[s_], FX_O, right=True) == k].mean()) for k in range(4)],
                     massa_cru=dict(dentro20=dentro20(om.massa[s_], np.exp(lm)[s_]), mape=float(100 * np.mean(np.abs(np.exp(lm) - om.massa)[s_] / om.massa[s_])), vies=float(100 * np.median(np.exp(lm)[s_] / om.massa[s_] - 1))))
            if papel == "registro":
                v = sel["res"][m[0]]
                mac = macros_ext(cs0[3], cs0[4], np.zeros(len(om.g)))
                x["macros_cru"] = {}
                for nut in ("kcal", "carbo", "gordura"):
                    vv = s_ & np.isfinite(getattr(om, nut))
                    x["macros_cru"][nut] = dict(crit=b2(om.cluster[vv], lambda i: crit_macro(getattr(om, nut)[vv], mac[nut][vv], nut, i)), vies=float(100 * np.median(mac[nut][vv] / np.maximum(getattr(om, nut)[vv], 1e-9) - 1)))
            R["omnifood"].setdefault(papel, {})[rot] = x
            print("OmniFood8K %s, %s %s: %d cenas | cru %.1f%% (95%%: %.1f a %.1f; 97,5%%: %.1f a %.1f) | com constante %.1f%% | sem osso, cru %.1f%% (%.1f a %.1f) | faixas cru %s | massa crua ±20%% %.1f%%, MAPE %.1f%%, viés %+.1f%%" %
                  (papel, rot, m, x["n"], *x["cru"], x["const"][0], *x["sem_osso_cru"][:3], "/".join("%.0f" % y for y in x["por_faixa_cru"]), x["massa_cru"]["dentro20"], x["massa_cru"]["mape"], x["massa_cru"]["vies"]))
            if "macros_cru" in x:
                print("      12.8, cru: " + " | ".join("%s %.1f%% (%.1f a %.1f; viés %+.0f%%)" % (n, *vv["crit"][:3], vv["vies"]) for n, vv in x["macros_cru"].items()))
    for d in ("flavoria", "fpb", "lefood", "mensa"):
        dom = D[d]; rg = dom.papel == "registro"; ncl = len(np.unique(dom.cluster[rg]))
        lm = dom.media(Z["R4c"], "__")[:, I_M]; c = constante(dom, lm, None)
        m0, m1 = np.exp(lm), np.exp(lm + c)
        g6 = rg & (dom.massa > 600)
        f = lambda est: b2(dom.cluster[rg], lambda i: dentro20(dom.massa[rg], est[rg], i)) if ncl > 1 else (dentro20(dom.massa[rg], est[rg]),) + (np.nan,) * 4
        x = dict(n=int(rg.sum()), cru=f(m0), const=f(m1), mape_cru=float(100 * np.mean(np.abs(m0 - dom.massa)[rg] / dom.massa[rg])), vies_cru=float(100 * np.median(m0[rg] / dom.massa[rg] - 1)),
                 acima600_mape_cru=float(100 * np.mean(np.abs(m0 - dom.massa)[g6] / dom.massa[g6])) if g6.any() else None, constante=float(np.exp(c[rg][0])))
        R["massa"][d] = x
        print("%s registro (massa da R4c, igual nos dois modelos): n %d | cru ±20%% %.1f%% (95%%: %.1f a %.1f; 97,5%%: %.1f a %.1f), MAPE %.1f%%, viés %+.1f%% | com constante %.1f%% (×%.2f) | > 600 g MAPE cru %s" %
              (d, x["n"], *x["cru"], x["mape_cru"], x["vies_cru"], x["const"][0], x["constante"], "%.1f%%" % x["acima600_mape_cru"] if x["acima600_mape_cru"] is not None else "-"))
    sn = D["snapme"]
    for rot, m in mods.items():
        e, cs = estima(m, sn)
        mac = macros_ext(cs[3], cs[4], np.zeros(len(sn.g)))
        x = {}
        for nome, est, col in (("proteina", e, "proteina_fraca_g"), ("massa", np.exp(cs[3][:, I_M]), "massa_fraca_g"), ("kcal", mac["kcal"], "kcal_fraca"), ("carbo", mac["carbo"], "carbo_fraca_g"), ("gordura", mac["gordura"], "gordura_fraca_g")):
            decl = sn.fraco[col]; v = np.isfinite(decl)
            x[nome] = boot_cadeias(np.unique(sn.cluster[v], return_inverse=True)[1], lambda i: spearmanr(est[v][i], decl[v][i])[0], n=2000)
        R["snapme"][rot] = x
        print("SNAPMe (autorrelato; foto inteira), %s: " % rot + " | ".join("%s %.2f (%.2f a %.2f)" % (k, *v) for k, v in x.items()))
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "r6_registro.json"), "w"), indent=1, default=float, ensure_ascii=False)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 3:
        sys.exit(__doc__)
    (selecao if "--so-cv" in sys.argv else registro if "--registro" in sys.argv else (lambda *x: sys.exit(__doc__)))(*a)


if __name__ == "__main__":
    main()
