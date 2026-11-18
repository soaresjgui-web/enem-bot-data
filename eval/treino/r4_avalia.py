#!/usr/bin/env python3
"""Seção 13.11: aplica a regra R13 (13.9) à quarta execução e faz a leitura única das partes de registro externas (13.5).

    python3 r4_avalia.py <pasta_kaggle_4> <pasta_divisoes> --so-cv     # R13 só com validação cruzada; grava r4_selecao.json
    python3 r4_avalia.py <pasta_kaggle_4> <pasta_divisoes> --registro  # leitura única das partes de registro (A0, S, X); marca r4_lido.json
    python3 r4_avalia.py <pasta_kaggle_4> <pasta_divisoes> --teste2    # última leitura do teste2, só do S e só se S != A0 (13.5)

<pasta_divisoes> tem os `divisao_r4_<dominio>.csv` de `externo_r4_divide.py` (os do OmniFood8K, FPB e SNAPMe derivam de
bases não comerciais e ficam fora do repositório). As listas `externo_*.csv` vêm de eval/treino/kaggle/listas/.

Medidas (13.2), sempre com 1 orientação e proteína = massa da ridge × média em log (piso 0,01) das duas densidades da
cabeça de frações:
  N  CV do treino_v2, critério ponderado nas cinco faixas (bootstrap por cadeia)
  O  OmniFood8K, CV por prato principal: critério nas faixas 0–5, 5–15, 15–25, > 25 g com o mesmo peso, só o quadro camera_0
     de cada cena; a massa leva a constante do domínio (mediana de log real − log previsto nos outros folds)
  M  % dos pratos com a massa a ±20%, média de Flavoria (quadro preferido), FPB de cima (média em log das fotos da bandeja)
     e Mensa, com a mesma constante
Regra R13: guarda (N cai no máximo 1,0 ponto e o critério acima de 25 g no máximo 2,5 em relação ao A0 = R0 × H0); A* = a
admissível de maior S = (N + O + M) / 3 (empate a menos de 0,5: menos bases não comerciais, depois menos bases); X = a de maior
S sem a guarda; e a melhor admissível só com dados liberados. O teste2 só é lido no modo --teste2."""
import csv, json, os, sys, time
import numpy as np
from teste2_avalia import crit_pond, boot_cadeias
from zen_comum import ok, EPS_D, I_M, I_P, I_D
from dens_comum import DENS, faixas
from um_modelo_avalia import LISTA, lg
from macros_cv import crit as crit_macro, PISO as PISO_MACRO, FAIXAS as FAIXAS_MACRO

AQUI = os.path.dirname(os.path.abspath(__file__))
LISTAS = os.path.join(AQUI, "kaggle", "listas")
SAIDA = os.path.join(DENS, "r4")
RS = {"R0": "gpu4 ridge so400m", "R1": "gpu4 ridge so400m +ok", "R2": "gpu4 ridge so400m +ok+fpb", "R3": "gpu4 ridge so400m +omni", "R4": "gpu4 ridge so400m +tudo"}
HS = {"H0": "gpu4 fracoes so400m", "H1": "gpu4 fracoes so400m +omni", "H2": "gpu4 fracoes so400m +omni+ff"}
BASES = {"R0": set(), "R1": {"flavoria", "lefood", "mensa"}, "R2": {"flavoria", "lefood", "mensa", "fpb"}, "R3": {"omnifood"},
         "R4": {"flavoria", "lefood", "mensa", "fpb", "omnifood"}, "H0": set(), "H1": {"omnifood"}, "H2": {"omnifood", "flavoria", "fpb", "mensa", "lefood"}}
NAO_COMERCIAL = {"fpb", "omnifood"}
FT = {"F1": "gpu4 ft siglip2_b16_224 r2 fracoes", "F2": "gpu4 ft siglip2_b16_224 r2 fracoes faixa", "F3": "gpu4 ft siglip2_b16_224 r2 +tudo"}
P1 = "gpu4 ftparcial so400m"
U0_CV = 68.04          # C1 de 12.6 (U0 da terceira execução), mesma lista e mesmos folds
GUARDA_N, GUARDA_25, EMPATE = 1.0, 2.5, 0.5
FX_O = [5, 15, 25]     # faixas de O: 0–5, 5–15, 15–25, > 25 g
DOM_M = ["flavoria", "fpb", "mensa"]


def arq(pasta, nome):
    return os.path.join(pasta, "pred__" + nome.replace(" ", "_").replace("+", "_") + ".npz")


# ------------------------------------------------------------------ dados
def le_n5k():
    L = list(csv.DictReader(open(LISTA)))
    f = lambda c: np.array([float(r[c]) for r in L])
    return dict(ids=np.array([r["dish_id"] for r in L]), cadeia=np.array([int(r["cadeia"]) for r in L]), proteina=f("proteina"), massa=f("massa"),
                kcal=f("kcal"), gordura=f("gordura"), carbo=f("carbo"))


class Dominio:
    """Unidades de métrica de um domínio externo: rótulos (da lista), papel, fold, cluster; e as linhas de previsão de cada uma."""

    def __init__(self, nome, pasta_div, z):
        self.nome = nome
        lista = {r["dish_id"]: r for r in csv.DictReader(open(os.path.join(LISTAS, "externo_%s.csv" % nome)))}
        div = list(csv.DictReader(open(os.path.join(pasta_div, "divisao_r4_%s.csv" % nome))))
        ids = {"cv": list(z["extcv_ids__" + nome]), "reg": list(z["extreg_ids__" + nome])}
        pos = {k: {s: i for i, s in enumerate(v)} for k, v in ids.items()}
        U = {}
        for d in div:
            if d["papel"] in ("fora",) or d["quadro_metrica"] != "1":
                continue
            fonte = "cv" if d["papel"] == "treino" else "reg"
            chave = "ext:%s:%s" % (nome, d["dish_id"]) + ("#" + d["quadro_metrica_nome"].replace(".jpg", "") if d["quadro_metrica_nome"] else "")
            if chave not in pos[fonte]:
                raise KeyError("%s: %s sem previsão" % (nome, chave))
            g = U.setdefault(d["grupo_metrica"], dict(linhas=[], fonte=fonte, papel=d["papel"], fold=int(d["fold_ext"]), cluster=d["cluster"], rot=[]))
            assert g["papel"] == d["papel"] and g["fold"] == int(d["fold_ext"])
            g["linhas"].append(pos[fonte][chave]); g["rot"].append(lista[d["dish_id"]])
        self.g = list(U)
        self.papel = np.array([U[k]["papel"] for k in self.g]); self.fonte = np.array([U[k]["fonte"] for k in self.g])
        self.fold = np.array([U[k]["fold"] for k in self.g]); self.cluster = np.array([U[k]["cluster"] for k in self.g])
        self.linhas = [U[k]["linhas"] for k in self.g]
        num = lambda c: np.array([np.mean([float(r[c]) for r in U[k]["rot"]]) if all(r.get(c, "") != "" for r in U[k]["rot"]) else np.nan for k in self.g])
        self.massa, self.proteina, self.kcal, self.gordura, self.carbo = (num(c) for c in ("massa", "proteina", "kcal", "gordura", "carbo"))
        self.osso = np.array(["osso_ou_casca" in U[k]["rot"][0].get("avisos", "") for k in self.g])
        self.fraco = {c: num(c) for c in ("massa_fraca_g", "proteina_fraca_g", "kcal_fraca", "gordura_fraca_g", "carbo_fraca_g") if c in next(iter(U.values()))["rot"][0]}

    def media(self, z, chave):
        """Média em log das linhas de cada unidade (saídas em log), por fonte (cv/reg)."""
        out = np.full((len(self.g),) + z["extcv__" + self.nome].shape[1:], np.nan)
        for i, (f, ls) in enumerate(zip(self.fonte, self.linhas)):
            out[i] = z[("extcv" if f == "cv" else "extreg") + chave + self.nome][ls].mean(0)
        return out


def constante(dom, logm_prev, idx_cv):
    """Constante de massa do domínio (log): por fold, a mediana de log real − log previsto nas unidades de treino dos outros
    folds; para as partes de registro, nas unidades de treino de todos os folds."""
    tr = (dom.papel == "treino") & np.isfinite(dom.massa)
    d = np.log(dom.massa) - logm_prev
    c = np.zeros(len(dom.g))
    for k in range(5):
        m = dom.fold == k
        c[m] = np.median(d[tr & (dom.fold != k)])
    c[dom.papel != "treino"] = np.median(d[tr])
    return c


def prot(zr, zh, dom, c=None):
    """Proteína (g) e massa (g) de uma combinação ridge × frações num domínio externo, com a constante c (log) na massa."""
    P_R, P_H = dom.media(zr, "__"), dom.media(zh, "__")
    dff = np.array([zh[("extcv" if f == "cv" else "extreg") + "_dens_ff__" + dom.nome][ls].mean() for f, ls in zip(dom.fonte, dom.linhas)])
    d_reg = np.clip(np.exp(P_H[:, I_D]) - EPS_D, 0, None)
    d_med = np.clip(np.exp((lg(d_reg) + lg(dff)) / 2) - EPS_D, 0, None)
    logm = P_R[:, I_M] + (0 if c is None else c)
    return np.exp(logm) * d_med, np.exp(logm), P_R, P_H


def macros_ext(P_R, P_H, c):
    """Rotas de 12.8 nos domínios externos; a constante de massa multiplica o nutriente inteiro."""
    k = np.exp(c)
    out = {}
    for nut, j in (("kcal", 2), ("carbo", 5)):
        dr, dh = np.expm1(P_R[:, j]).clip(0), np.expm1(P_H[:, j]).clip(0)
        out[nut] = k * np.expm1((np.log1p(dr) + np.log1p(dh)) / 2)
    dr = np.expm1(P_R[:, 4]).clip(0); mxd = np.exp(P_R[:, I_M]) * np.expm1(P_H[:, 4]).clip(0) / np.exp(P_H[:, I_M])
    out["gordura"] = k * np.sqrt(dr * mxd)
    return out


def crit_o(real, est, idx=None):
    if idx is not None:
        real, est = real[idx], est[idx]
    a = ok(real, est); fx = np.digitize(real, FX_O, right=True)
    return 100 * float(np.mean([a[fx == k].mean() for k in range(4) if (fx == k).any()]))


def dentro20(real, est, idx=None):
    if idx is not None:
        real, est = real[idx], est[idx]
    return 100 * float(np.mean(np.abs(est - real) <= 0.2 * real))


def acima25(real, est, idx=None):
    if idx is not None:
        real, est = real[idx], est[idx]
    a = ok(real, est); fx = faixas(real)
    return 100 * float(np.mean([a[fx["25-40"]].mean(), a[fx[">40"]].mean()]))


def boot(cl, f, n=2000, seed=0):
    return boot_cadeias(np.unique(cl, return_inverse=True)[1], f, n=n, seed=seed)


# ------------------------------------------------------------------ CV e regra
def carrega(pasta):
    Z = {k: np.load(arq(pasta, n), allow_pickle=False) for k, n in {**RS, **HS}.items()}
    N = le_n5k()
    for z in Z.values():
        assert (z["ids_treino"] == N["ids"]).all()
    return Z, N


def selecao(pasta, pasta_div):
    Z, N = carrega(pasta)
    D = {d: Dominio(d, pasta_div, Z["R0"]) for d in ["omnifood"] + DOM_M + ["lefood"]}
    for d, dom in D.items():
        for z in Z.values():
            assert list(z["extcv_ids__" + d]) == list(Z["R0"]["extcv_ids__" + d]) and list(z["extreg_ids__" + d]) == list(Z["R0"]["extreg_ids__" + d])
    real, cad = N["proteina"], N["cadeia"]
    est_n, res = {}, {}
    om = D["omnifood"]; tro = (om.papel == "treino") & np.isfinite(om.proteina)
    # M e massa externa: só dependem da ridge
    M = {}
    for r in RS:
        M[r] = {}
        for d in DOM_M + ["lefood"]:
            dom = D[d]; tr = dom.papel == "treino"
            lm = dom.media(Z[r], "__")[:, I_M]; c = constante(dom, lm, None)
            M[r][d] = dict(dentro20=dentro20(dom.massa[tr], np.exp(lm + c)[tr]), cru=dentro20(dom.massa[tr], np.exp(lm)[tr]),
                           mape=float(100 * np.mean(np.abs(np.exp(lm + c) - dom.massa)[tr] / dom.massa[tr])), n=int(tr.sum()), est=np.exp(lm + c), tr=tr)
        M[r]["M"] = float(np.mean([M[r][d]["dentro20"] for d in DOM_M]))
    for r in RS:
        for h in HS:
            nome = r + "×" + h
            m = np.exp(Z[r]["oof_o1"][:, I_M]); PH = Z[h]["oof_o1"]
            d_med = np.clip(np.exp((lg(np.clip(np.exp(PH[:, I_D]) - EPS_D, 0, None)) + lg(Z[h]["oof_dens_ff_o1"])) / 2) - EPS_D, 0, None)
            e = m * d_med; est_n[nome] = e
            lm = om.media(Z[r], "__")[:, I_M]; c = constante(om, lm, None)
            eo = prot(Z[r], Z[h], om, c)[0]
            res[nome] = dict(N=crit_pond(real, e), N25=acima25(real, e), O=crit_o(om.proteina[tro], eo[tro]), M=M[r]["M"],
                             O_cru=crit_o(om.proteina[tro], prot(Z[r], Z[h], om)[0][tro]), bases=sorted(BASES[r] | BASES[h]), _eo=eo)
            res[nome]["S"] = (res[nome]["N"] + res[nome]["O"] + res[nome]["M"]) / 3
            res[nome]["nao_comerciais"] = len(set(res[nome]["bases"]) & NAO_COMERCIAL)
    a0 = res["R0×H0"]
    for v in res.values():
        v["admissivel"] = bool(v["N"] >= a0["N"] - GUARDA_N and v["N25"] >= a0["N25"] - GUARDA_25)
    adm = [k for k, v in res.items() if v["admissivel"]]
    smax = max(res[k]["S"] for k in adm)
    perto = [k for k in adm if res[k]["S"] > smax - EMPATE]
    A = min(perto, key=lambda k: (res[k]["nao_comerciais"], len(res[k]["bases"]), -res[k]["S"]))
    X = max(res, key=lambda k: res[k]["S"])
    lib = [k for k in adm if k.split("×")[0] in ("R0", "R1") and k.split("×")[1] == "H0"]
    L = max(lib, key=lambda k: res[k]["S"])
    print("R13 sobre as 15 combinações (1 orientação). A0 = R0×H0: N %.2f, > 25 g %.2f\n" % (a0["N"], a0["N25"]))
    print("%-7s %6s %6s %6s %6s %6s %6s  %s" % ("comb.", "N", ">25", "O", "O cru", "M", "S", "admissível / bases"))
    for k in sorted(res, key=lambda k: -res[k]["S"]):
        v = res[k]
        print("%-7s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f  %s %s" % (k, v["N"], v["N25"], v["O"], v["O_cru"], v["M"], v["S"], "sim" if v["admissivel"] else "NÃO", ",".join(v["bases"]) or "-"))
    print("\nA* = %s (S %.2f; a menos de %.1f: %s) | X = %s (S %.2f) | melhor só com dados liberados = %s (S %.2f)" % (A, res[A]["S"], EMPATE, perto, X, res[X]["S"], L, res[L]["S"]))
    # intervalos pareados contra o A0
    dif = {}
    for k in sorted({A, X, L} - {"R0×H0"}):
        r, h = k.split("×")
        dif[k] = dict(N=boot(cad, lambda i: crit_pond(real, est_n[k], i) - crit_pond(real, est_n["R0×H0"], i)),
                      N25=boot(cad, lambda i: acima25(real, est_n[k], i) - acima25(real, est_n["R0×H0"], i)),
                      O=boot(om.cluster[tro], lambda i: crit_o(om.proteina[tro], res[k]["_eo"][tro], i) - crit_o(om.proteina[tro], res["R0×H0"]["_eo"][tro], i)))
        print("   %s − A0: N %+.2f (IC %+.2f a %+.2f) | > 25 g %+.2f (IC %+.2f a %+.2f) | O %+.2f (IC %+.2f a %+.2f)" % (k, *dif[k]["N"], *dif[k]["N25"], *dif[k]["O"]))
    print("   M por domínio (dentro de ±20%%, com constante; cru): " + " | ".join("%s: %s" % (r, ", ".join("%s %.1f (%.1f)" % (d, M[r][d]["dentro20"], M[r][d]["cru"]) for d in DOM_M + ["lefood"])) for r in RS))
    # transferência (não decide nada)
    T = {}
    for d in DOM_M:
        dom = D[d]; tr = M["R0"][d]["tr"]
        T["massa_" + d + "_R3−R0"] = boot(dom.cluster[tr], lambda i: dentro20(dom.massa[tr], M["R3"][d]["est"][tr], i) - dentro20(dom.massa[tr], M["R0"][d]["est"][tr], i))
    for r in ("R1", "R2"):
        T["O_%s×H0−A0" % r] = boot(om.cluster[tro], lambda i: crit_o(om.proteina[tro], res[r + "×H0"]["_eo"][tro], i) - crit_o(om.proteina[tro], res["R0×H0"]["_eo"][tro], i))
    for h in ("H1", "H2"):
        T["N_R0×%s−A0" % h] = boot(cad, lambda i: crit_pond(real, est_n["R0×" + h], i) - crit_pond(real, est_n["R0×H0"], i))
    print("   transferência (não decide): " + " | ".join("%s %+.2f (IC %+.2f a %+.2f)" % (k, *v) for k, v in T.items()))
    # S, Ad
    logs = {k: json.load(open(os.path.join(pasta, "log__" + n.replace(" ", "_").replace("+", "_") + ".json"))) for k, n in {**FT, "P1": P1}.items()}
    acc = {}
    for k, Lg in logs.items():
        v = [np.mean([[x for x in Lg["partes"]["fold%d" % i]["hist"] if "val_rmse_logm" in x][j]["val_rmse_logm"] for i in range(5)]) for j in (0, -1)]
        acc[k] = dict(epoca0=v[0], ultima=v[1], passou=bool(v[1] <= v[0]))
    S = "P1" if acc["P1"]["passou"] else A
    print("\naceitação (13.7): " + "; ".join("%s %.4f -> %.4f %s" % (k, v["epoca0"], v["ultima"], "passou" if v["passou"] else "NÃO passou") for k, v in acc.items()))
    print("S = %s" % S)
    ad = {}
    for k in FT:
        if not acc[k]["passou"]:
            continue
        z = np.load(arq(pasta, FT[k]))
        tab = {o: {} for o in (1, 4, 8)}
        for o, s in ((1, "_o1"), (4, "_o4"), (8, "")):
            P = z["oof" + s]; m = np.exp(P[:, I_M]); dr = np.expm1(P[:, I_P]).clip(0); dd = np.clip(np.exp(P[:, I_D]) - EPS_D, 0, None)
            tab[o] = dict(direta=crit_pond(real, dr), massa_x_dens=crit_pond(real, m * dd), geometrica=crit_pond(real, np.sqrt(dr * m * dd)))
        rota = max(tab[8], key=tab[8].get); o = min(o for o in (1, 4, 8) if tab[o][rota] >= tab[8][rota] - 1.0)
        ad[k] = dict(rota=rota, orient=o, cv=tab[o][rota], tabela=tab, dif_u0=tab[o][rota] - U0_CV, vira_ad=bool(tab[o][rota] >= U0_CV + 1.0))
        print("%s: rotas com 8 orientações %s -> %s, %d orientação(ões): CV %.2f; − U0 (%.2f) %+.2f -> %s" % (k, {a: round(b, 2) for a, b in tab[8].items()}, rota, o, tab[o][rota], U0_CV, tab[o][rota] - U0_CV, "novo modelo de aparelho" if ad[k]["vira_ad"] else "não vira Ad"))
    Ad = max([k for k in ad if ad[k]["vira_ad"]], key=lambda k: ad[k]["cv"], default=None)
    print("Ad = %s" % (Ad or "nenhum: o modelo de aparelho continua o C1 de 12.6"))
    # relato secundário de 12.8 na CV do treino_v2 (não decide)
    mac = {}
    for k in sorted({"R0×H0", A, X, L}):
        r, h = k.split("×")
        PR, PH = Z[r]["oof_o1"], Z[h]["oof_o1"]
        e = macros_ext(PR, PH, 0.0)
        mac[k] = {n: crit_macro(N[n], e[n], n) for n in ("kcal", "carbo", "gordura")}
    print("12.8 na CV (kcal / carboidrato / gordura): " + " | ".join("%s %.1f / %.1f / %.1f" % (k, v["kcal"], v["carbo"], v["gordura"]) for k, v in mac.items()))
    sel = dict(res={k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in res.items()}, A0="R0×H0", A=A, X=X, liberado=L, empate=perto, dif_A0=dif,
               M={r: {d: {a: b for a, b in v.items() if a not in ("est", "tr")} for d, v in M[r].items() if d != "M"} for r in RS}, transferencia=T,
               aceitacao=acc, S=S, ad=ad, Ad=Ad, macros_cv=mac, quando=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(SAIDA, exist_ok=True)
    json.dump(sel, open(os.path.join(SAIDA, "r4_selecao.json"), "w"), indent=1, default=float, ensure_ascii=False)


# ------------------------------------------------------------------ leitura única das partes de registro
def registro(pasta, pasta_div):
    marca = os.path.join(SAIDA, "r4_lido.json")
    if os.path.exists(marca):
        sys.exit("As partes de registro externas já foram lidas (%s). Não há outra leitura." % json.load(open(marca))["quando"])
    sel = json.load(open(os.path.join(SAIDA, "r4_selecao.json")))
    mods = {"A0": "R0×H0"}
    if sel["S"] != "P1":
        mods["S"] = sel["A"]
    if sel["X"] not in mods.values():
        mods["X"] = sel["X"]
    if sel["liberado"] not in mods.values():
        mods["liberado"] = sel["liberado"]       # 13.9 item 4: sai sempre ao lado
    Z, N = carrega(pasta)
    D = {d: Dominio(d, pasta_div, Z["R0"]) for d in ["omnifood", "flavoria", "fpb", "mensa", "lefood", "snapme"]}
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), modelos=mods), open(marca, "w"), indent=1, ensure_ascii=False)
    R = dict(modelos=mods, omnifood={}, massa={}, snapme={})
    om = D["omnifood"]
    for papel in ("registro", "registro_misto"):
        for rot, comb in mods.items():
            r, h = comb.split("×")
            lm = om.media(Z[r], "__")[:, I_M]; c = constante(om, lm, None)
            e, m, PR, PH = prot(Z[r], Z[h], om, c); e0 = prot(Z[r], Z[h], om)[0]
            sel_ = (om.papel == papel) & np.isfinite(om.proteina)
            so = sel_ & ~om.osso
            x = dict(n=int(sel_.sum()), pratos=int(len(np.unique(om.cluster[sel_]))), criterio=boot(om.cluster[sel_], lambda i: crit_o(om.proteina[sel_], e[sel_], i)),
                     cru=crit_o(om.proteina[sel_], e0[sel_]), sem_osso=dict(n=int(so.sum()), criterio=boot(om.cluster[so], lambda i: crit_o(om.proteina[so], e[so], i))),
                     por_faixa=[(100 * float(ok(om.proteina[sel_], e[sel_])[np.digitize(om.proteina[sel_], FX_O, right=True) == k].mean()), int((np.digitize(om.proteina[sel_], FX_O, right=True) == k).sum())) for k in range(4)],
                     vies_proteina=[float(100 * np.median((e / om.proteina)[sel_][np.digitize(om.proteina[sel_], FX_O, right=True) == k] - 1)) for k in range(1, 4)],
                     massa=dict(mape=float(100 * np.mean(np.abs(m - om.massa)[sel_] / om.massa[sel_])), dentro20=dentro20(om.massa[sel_], m[sel_]), vies=float(100 * np.median(m[sel_] / om.massa[sel_] - 1)), constante=float(np.exp(c[sel_][0]))))
            if papel == "registro":
                mac = macros_ext(PR, PH, c)
                x["macros"] = {}
                for nut in ("kcal", "carbo", "gordura"):
                    v = sel_ & np.isfinite(getattr(om, nut))
                    x["macros"][nut] = dict(criterio=boot(om.cluster[v], lambda i: crit_macro(getattr(om, nut)[v], mac[nut][v], nut, i)),
                                            vies=float(100 * np.median(mac[nut][v] / np.maximum(getattr(om, nut)[v], 1e-9) - 1)), n=int(v.sum()))
            R["omnifood"].setdefault(papel, {})[rot] = x
            print("OmniFood8K %s, %s (%s): %d cenas, %d pratos principais | critério %.1f%% (IC %.1f a %.1f), cru %.1f%% | sem osso/casca (n=%d) %.1f%% (IC %.1f a %.1f) | por faixa %s | massa: MAPE %.1f%%, ±20%% %.1f%%, viés %+.1f%% (constante ×%.3f)" %
                  (papel, rot, comb, x["n"], x["pratos"], *x["criterio"], x["cru"], x["sem_osso"]["n"], *x["sem_osso"]["criterio"], " ".join("%.0f(n=%d)" % t for t in x["por_faixa"]), x["massa"]["mape"], x["massa"]["dentro20"], x["massa"]["vies"], x["massa"]["constante"]))
            if "macros" in x:
                print("      kcal / carboidrato / gordura (12.8): " + " | ".join("%s %.1f%% (IC %.1f a %.1f; viés %+.0f%%)" % (n, *v["criterio"], v["vies"]) for n, v in x["macros"].items()))
    for d in ("flavoria", "fpb", "mensa", "lefood"):
        dom = D[d]; rg = dom.papel == "registro"
        for rot, comb in mods.items():
            r = comb.split("×")[0]
            lm = dom.media(Z[r], "__")[:, I_M]; c = constante(dom, lm, None)
            m, m0 = np.exp(lm + c), np.exp(lm)
            g6 = rg & (dom.massa > 600)
            x = dict(n=int(rg.sum()), clusters=int(len(np.unique(dom.cluster[rg]))), dentro20=boot(dom.cluster[rg], lambda i: dentro20(dom.massa[rg], m[rg], i)) if len(np.unique(dom.cluster[rg])) > 1 else (dentro20(dom.massa[rg], m[rg]), np.nan, np.nan),
                     mape=float(100 * np.mean(np.abs(m - dom.massa)[rg] / dom.massa[rg])), vies=float(100 * np.median(m[rg] / dom.massa[rg] - 1)),
                     cru=dict(dentro20=dentro20(dom.massa[rg], m0[rg]), mape=float(100 * np.mean(np.abs(m0 - dom.massa)[rg] / dom.massa[rg])), vies=float(100 * np.median(m0[rg] / dom.massa[rg] - 1))),
                     acima600=dict(n=int(g6.sum()), mape=float(100 * np.mean(np.abs(m - dom.massa)[g6] / dom.massa[g6])) if g6.any() else None, vies=float(100 * np.median(m[g6] / dom.massa[g6] - 1)) if g6.any() else None),
                     constante=float(np.exp(c[rg][0])))
            R["massa"].setdefault(d, {})[rot] = x
            print("%s registro, %s (%s): %d unidades, %d clusters | ±20%% %.1f%% (IC %.1f a %.1f) | MAPE %.1f%% | viés %+.1f%% | cru: ±20%% %.1f%%, MAPE %.1f%%, viés %+.1f%% | > 600 g (n=%d): MAPE %s, viés %s | constante ×%.3f" %
                  (d, rot, r, x["n"], x["clusters"], *x["dentro20"], x["mape"], x["vies"], x["cru"]["dentro20"], x["cru"]["mape"], x["cru"]["vies"], x["acima600"]["n"],
                   "%.1f%%" % x["acima600"]["mape"] if x["acima600"]["mape"] is not None else "-", "%+.1f%%" % x["acima600"]["vies"] if x["acima600"]["vies"] is not None else "-", x["constante"]))
    # SNAPMe: só correlação de postos com o autorrelato
    from scipy.stats import spearmanr
    sn = D["snapme"]
    for rot, comb in mods.items():
        r, h = comb.split("×")
        e, m, PR, PH = prot(Z[r], Z[h], sn)
        mac = macros_ext(PR, PH, np.zeros(len(sn.g)))
        x = {}
        for nome, est, decl in (("proteina", e, sn.fraco["proteina_fraca_g"]), ("massa", m, sn.fraco["massa_fraca_g"]), ("kcal", mac["kcal"], sn.fraco["kcal_fraca"]),
                                ("carbo", mac["carbo"], sn.fraco["carbo_fraca_g"]), ("gordura", mac["gordura"], sn.fraco["gordura_fraca_g"])):
            v = np.isfinite(decl)
            x[nome] = boot(sn.cluster[v], lambda i: spearmanr(est[v][i], decl[v][i])[0])
        R["snapme"][rot] = x
        print("SNAPMe (autorrelato, %d fotos, 95 pessoas), %s: correlação de postos " % (len(sn.g), rot) + " | ".join("%s %.2f (IC %.2f a %.2f)" % (k, *v) for k, v in x.items()))
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "r4_registro.json"), "w"), indent=1, default=float, ensure_ascii=False)


# ------------------------------------------------------------------ última leitura do teste2 (13.5), só do S
def teste2(pasta, pasta_div):
    from teste2_avalia import resumo
    from um_modelo_avalia import AVAL2
    from dens_comum import PREC
    marca = os.path.join(SAIDA, "r4_teste2_lido.json")
    if os.path.exists(marca):
        sys.exit("O teste2 já foi lido para a seção 13 (%s). Não há outra leitura." % json.load(open(marca))["quando"])
    sel = json.load(open(os.path.join(SAIDA, "r4_selecao.json")))
    if sel["S"] in ("P1",) or sel["A"] == "R0×H0":
        sys.exit("13.5: o S é o A0 (ou o P1, que não tem rota aqui); não há leitura nova do teste2.")
    r, h = sel["A"].split("×")
    Z, N = carrega(pasta)
    ids2 = np.array([x["dish_id"] for x in csv.DictReader(open(AVAL2)) if x["split"] == "teste2"])
    def est(rr, hh):
        PR, PH = Z[rr]["teste2_o1"], Z[hh]["teste2_o1"]
        d_med = np.clip(np.exp((lg(np.clip(np.exp(PH[:, I_D]) - EPS_D, 0, None)) + lg(Z[hh]["teste2_dens_ff_o1"])) / 2) - EPS_D, 0, None)
        return np.exp(PR[:, I_M]) * d_med, np.exp(PR[:, I_M]), PR, PH
    # conferências sem gabarito
    for k in (r, h, "R0", "H0"):
        assert (Z[k]["ids_teste2"] == ids2).all() and np.isfinite(Z[k]["teste2_o1"]).all()
    assert np.isfinite(Z[h]["teste2_dens_ff_o1"]).all()
    e_a0 = est("R0", "H0")[0]
    ant = np.load(os.path.join(DENS, "um_modelo", "emenda_pratos_teste2.npz"))
    assert (ant["ids"] == ids2).all()
    par_d1 = float(np.corrcoef(np.log1p(e_a0), np.log1p(ant["est_D1"]))[0, 1])
    sv = json.load(open(os.path.join(pasta, "servico__gpu4_ridge_so400m.json")))["conferencias_cabecas"]
    conf = {RS[r]: sv[RS[r]]["max_dif_log"], HS[h]: sv[HS[h]]["max_dif_log"]}
    e, m, PR, PH = est(r, h)
    q = lambda v: [round(float(x), 2) for x in np.percentile(v, [10, 50, 90])]
    print("conferências sem gabarito: 829 previsões finitas na ordem de aval2_ids.csv; A0 refeito × D1 gravado em 12.7: correlação %.6f; cabeças remontadas %s; "
          "proteína prevista p10/50/90 %s (A0: %s)" % (par_d1, conf, q(e), q(e_a0)))
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), modelo=sel["A"], paridade_A0_D1=par_d1, remontagem=conf), open(marca, "w"), indent=1, ensure_ascii=False)
    G = {x["dish_id"]: x for x in csv.DictReader(open(os.path.join(PREC, "n5k_teste2_amostra.csv")))}
    real = np.array([float(G[i]["proteina_g"]) for i in ids2]); mreal = np.array([float(G[i]["massa_g"]) for i in ids2])
    cad = np.array([int(G[i]["cadeia"]) for i in ids2]); sem_irmao = np.array(["teste" not in G[i]["cadeia_contem"].split("+") for i in ids2])
    from zen_dados import ler_meta_completo
    meta = ler_meta_completo()
    mac_real = {n: np.array([meta[i][n] for i in ids2]) for n in ("kcal", "gordura", "carbo")}
    assert np.allclose(mac_real["kcal"], [float(G[i]["kcal"]) for i in ids2], atol=0.6)
    fx = faixas(real)
    res = resumo(real, e, cad)
    grupos = [np.where(cad == c)[0] for c in np.unique(cad)]
    rng = np.random.default_rng(0)
    v = np.array([crit_pond(real, e, np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))])) for _ in range(4000)])
    res["ic"] = {"95": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))), "98.75": (float(np.percentile(v, 0.625)), float(np.percentile(v, 99.375)))}
    res["massa"] = dict(mape=float(100 * np.mean(np.abs(m - mreal) / mreal)), vies=float(100 * np.median(m / mreal - 1)),
                        por_faixa={k: (float(100 * np.mean(np.abs(m[f] - mreal[f]) / mreal[f])), float(100 * np.median(m[f] / mreal[f] - 1))) for k, f in fx.items()})
    res["vies_proteina"] = {k: float(100 * np.median(e[f] / np.maximum(real[f], 1e-9) - 1)) for k, f in fx.items() if k != "0-5"}
    b = np.load(os.path.join(DENS, "um_modelo", "pratos_teste2.npz")); a = np.load(os.path.join(DENS, "teste2", "pratos_teste2.npz"), allow_pickle=True)
    assert (b["ids"] == ids2).all() and np.allclose(a["real"], real)
    a25 = lambda x, j: 100 * np.mean([ok(real[j], x[j])[(real[j] > 25) & (real[j] <= 40)].mean(), ok(real[j], x[j])[real[j] > 40].mean()])
    res["contra"] = {n: dict(ref=crit_pond(real, x), dif=boot_cadeias(cad, lambda j: crit_pond(real, e, j) - crit_pond(real, x, j)), dif25=boot_cadeias(cad, lambda j: a25(e, j) - a25(x, j)))
                     for n, x in (("D1 (12.7)", ant["est_D1"]), ("C1 (12.6)", b["est_C1"]), ("nove redes (11.5)", a["est1"]))}
    res["sem_irmao"] = boot_cadeias(cad[sem_irmao], lambda j: crit_pond(real[sem_irmao], e[sem_irmao], j))
    mac = macros_ext(PR, PH, 0.0)
    res["macros"] = {n: dict(criterio=boot_cadeias(cad, lambda j: crit_macro(mac_real[n], mac[n], n, j)), vies=float(100 * np.median(mac[n][mac_real[n] > 0] / mac_real[n][mac_real[n] > 0] - 1)),
                             por_faixa=[float(100 * np.mean((np.abs(mac[n] - mac_real[n]) <= np.maximum(PISO_MACRO[n], 0.2 * mac_real[n]))[np.digitize(mac_real[n], FAIXAS_MACRO[n], right=True) == k])) for k in range(5)])
                     for n in ("kcal", "carbo", "gordura")}
    inf = res["ponderado"][1]
    res["veredito"] = ("70% DEMONSTRADO" + (" (também com a correção para quatro leituras)" if res["ic"]["98.75"][0] >= 70 else " no intervalo de 95%, mas não com a correção para quatro leituras")) if inf >= 70 else \
        "70% NÃO DEMONSTRADO" + (" (intervalo inteiro abaixo de 70%)" if res["ponderado"][2] < 70 else " (70% dentro do intervalo)")
    print("S = %s no teste2: ponderado %.1f%% (IC 95%% %.1f a %.1f; 98,75%% %.1f a %.1f); simples %.1f%%" % (sel["A"], *res["ponderado"], *res["ic"]["98.75"], res["simples"][0]))
    print("  por faixa: " + "  ".join("%s %.1f%% (n=%d)" % (k, x[0], x[1]) for k, x in res["por_faixa"].items()) + " | >25 g %.1f%% | >15 g %.1f%%" % (res["acima25"], res["acima15"]))
    for n, c in res["contra"].items():
        print("  − %s (%.1f%%): %+.1f (IC %+.1f a %+.1f); acima de 25 g %+.1f (IC %+.1f a %+.1f)" % (n, c["ref"], *c["dif"], *c["dif25"]))
    print("  massa: MAPE %.1f%%, viés %+.1f%%; por faixa " % (res["massa"]["mape"], res["massa"]["vies"]) + "  ".join("%s %.1f%% (%+.1f%%)" % (k, *x) for k, x in res["massa"]["por_faixa"].items()))
    print("  viés mediano da proteína: " + "  ".join("%s %+.1f%%" % kv for kv in res["vies_proteina"].items()))
    print("  sem irmão no teste antigo (n=%d): %.1f%% (IC %.1f a %.1f)" % (sem_irmao.sum(), *res["sem_irmao"]))
    print("  12.8 (secundário): " + " | ".join("%s %.1f%% (IC %.1f a %.1f; viés %+.0f%%; por faixa %s)" % (n, *x["criterio"], x["vies"], "/".join("%.0f" % y for y in x["por_faixa"])) for n, x in res["macros"].items()))
    print("  VEREDITO: " + res["veredito"])
    res.update(modelo=sel["A"], paridade_A0_D1=par_d1, quando=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(res, open(os.path.join(SAIDA, "r4_teste2.json"), "w"), indent=1, default=float, ensure_ascii=False)
    np.savez(os.path.join(SAIDA, "r4_pratos_teste2.npz"), ids=ids2, real=real, cadeia=cad, est_S=e, massa_S=m)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 2:
        sys.exit(__doc__)
    if "--so-cv" in sys.argv:
        selecao(*a)
    elif "--registro" in sys.argv:
        registro(*a)
    elif "--teste2" in sys.argv:
        teste2(*a)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
