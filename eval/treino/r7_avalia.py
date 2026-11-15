#!/usr/bin/env python3
"""Seção 17.8: regra R17 (17.4) na sétima execução, medidas pareadas de 17.5 e relatos das partes seguradas das bases novas.

    python3 r7_avalia.py <pasta_kaggle_7> <pasta_divisoes> --so-cv      # R17, sementes, 17.5 (só validação cruzada)
    python3 r7_avalia.py <pasta_kaggle_7> <pasta_divisoes> --relatos    # partes seguradas de ACETADA, receitas e FoodBD (uma vez; marca)
    python3 r7_avalia.py <pasta_kaggle_7> <pasta_divisoes> --omnifood   # 3ª e última leitura do OmniFood8K segurado, só do S7 (17.6)

Conferência do recorte do FoodBD (14.3): reprovada (35 certas em 40), então valem as variantes com a foto inteira (H-Bi,
H-ARBi); H-B e H-ARB não concorrem. Rotas r1–r4 e orientações como em 15.2 e 12.3; S17 = (N_c + > 25 g + O cru + M cru) / 4."""
import csv, json, os, sys, time
import numpy as np
from teste2_avalia import crit_pond, boot_cadeias
from zen_comum import ok, EPS_D, I_M, I_P, I_D
from dens_comum import DENS, faixas
from r4_avalia import Dominio, constante, crit_o, dentro20, acima25, le_n5k, FX_O, DOM_M, LISTAS
from r5_avalia import dens_med, b2
from r6_avalia import comps_n5k, comps_ext, rota, pesos_mistura

SAIDA = os.path.join(DENS, "r7")
ORI = {1: "_o1", 4: "_o4", 8: ""}
MASSA = {"R4c": "gpu7 ridge so400m prato +tudo", "R4c+A": "gpu7 ridge so400m prato +tudo +acetada", "R1c": "gpu7 ridge so400m prato +ok", "R0c": "gpu7 ridge so400m prato"}
DENSI = {"H0c-m": "gpu7 fracoes so400m prato +massa", "H-A": "gpu7 fracoes so400m prato +massa +acetada", "H-R": "gpu7 fracoes so400m prato +massa +receitas",
         "H-AR": "gpu7 fracoes so400m prato +massa +acetada+receitas", "H-Bi": "gpu7 fracoes so400m prato +massa +foodbd inteira",
         "H-ARBi": "gpu7 fracoes so400m prato +massa +acetada+receitas+foodbd inteira", "H-cap": "gpu7 fracoes so400m prato +massa cap",
         "H-B": "gpu7 fracoes so400m prato +massa +foodbd", "H-ARB": "gpu7 fracoes so400m prato +massa +acetada+receitas+foodbd",
         "H0c": "gpu7 fracoes so400m prato", "H0c+Bi": "gpu7 fracoes so400m prato +foodbd inteira"}
REPROVADOS = {"H-B", "H-ARB"}                       # conferência a olho do FoodBD reprovada (17.8 a)
BASES = {"R4c": {"flavoria", "lefood", "mensa", "fpb", "omnifood"}, "R4c+A": {"flavoria", "lefood", "mensa", "fpb", "omnifood", "acetada"}, "R1c": {"flavoria", "lefood", "mensa"}, "R0c": set(),
         "H0c-m": set(), "H-A": {"acetada"}, "H-R": {"receitas"}, "H-AR": {"acetada", "receitas"}, "H-Bi": {"foodbd"}, "H-ARBi": {"acetada", "receitas", "foodbd"}, "H-cap": set(),
         "H0c": set(), "H0c+Bi": {"foodbd"}, "H-B": {"foodbd"}, "H-ARB": {"acetada", "receitas", "foodbd"}}
NAO_COM = {"fpb", "omnifood", "acetada", "receitas"}
S6 = dict(N=70.73, N25=52.11, O=60.21, M=59.23)
S6["S17"] = (S6["N"] + S6["N25"] + S6["O"] + S6["M"]) / 4


def arq(p, nome):
    return os.path.join(p, "pred__" + nome.replace(" ", "_").replace("+", "_").replace("-", "_") + ".npz")


class Semente:
    """Uma semente de uma cabeça de densidade como npz (oof_s{k}*; nas bases externas só existe a média)."""

    def __init__(self, z, k):
        self.z, self.k = z, k

    def __getitem__(self, ch):
        return self.z[ch.replace("oof", "oof_s%d" % self.k, 1)] if ch.startswith("oof") else self.z[ch]


def selecao(p7, pdiv):
    Z = {k: np.load(arq(p7, n)) for k, n in {**MASSA, **DENSI}.items()}
    N = le_n5k()
    real, cad, fold, massa = N["proteina"], N["cadeia"], Z["R4c"]["fold"], N["massa"]
    for z in Z.values():
        assert (z["ids_treino"] == N["ids"]).all()
    D = {d: Dominio(d, pdiv, Z["R4c"]) for d in ["omnifood"] + DOM_M}
    om = D["omnifood"]; tro = (om.papel == "treino") & np.isfinite(om.proteina)
    M = {}
    for r in MASSA:
        vals = []
        for d in DOM_M:
            dom = D[d]; tr = dom.papel == "treino"
            vals.append(dentro20(dom.massa[tr], np.exp(dom.media(Z[r], "__")[:, I_M])[tr]))
        M[r] = float(np.mean(vals))
    res, est = {}, {}

    def avalia(r, h, zh, rotulo):
        cs = {o: comps_n5k(Z[r], zh, s) for o, s in ORI.items()}
        W1 = pesos_mistura(cs[1], real, fold)
        ce = comps_ext(Z[r], Z[h], om)
        out = {}
        for ro in ("r1", "r2", "r3", "r4"):
            tab = {o: crit_pond(real, rota(ro, cs[o], pesos_mistura(cs[o], real, fold) if ro == "r4" else None, fold)) for o in ORI}
            o = min(x for x in (1, 4, 8) if tab[x] >= tab[8] - 1.0)
            e = rota(ro, cs[o], pesos_mistura(cs[o], real, fold) if ro == "r4" else None, fold)
            eo = rota(ro, ce, W1, np.where(om.papel == "treino", om.fold, -1))
            v = dict(massa=r, densidade=h, rota=ro, orient=o, tabela=tab, N=tab[o], N25=acima25(real, e), O=crit_o(om.proteina[tro], eo[tro]),
                     O25=100 * float(ok(om.proteina[tro & (om.proteina > 25)], eo[tro & (om.proteina > 25)]).mean()),
                     M=M[r], nc=len((BASES[r] | BASES[h]) & NAO_COM), nb=len(BASES[r] | BASES[h]), W=[list(w) for w in W1],
                     dens_massa_real=crit_pond(real, massa * dens_med(zh["oof" + ORI[o]], zh["oof_dens_ff" + ORI[o]])))
            v["S17"] = (v["N"] + v["N25"] + v["O"] + v["M"]) / 4
            out[ro] = (v, e, eo)
        return out

    for r in ("R4c", "R4c+A", "R1c"):
        for h in DENSI:
            if r == "R1c" and h not in ("H0c", "H0c+Bi"):
                continue
            if r != "R1c" and h in ("H0c", "H0c+Bi"):
                continue
            for ro, (v, e, eo) in avalia(r, h, Z[h], "media").items():
                nome = "%s×%s %s o%d" % (r, h, ro, v["orient"])
                v["concorre"] = h not in REPROVADOS and r != "R1c"
                res[nome] = v; est[nome] = (e, eo)
    for k, v in res.items():
        v["admissivel"] = bool(v["N"] >= S6["N"] - 1.0 and v["N25"] >= S6["N25"] - 2.5)
    adm = [k for k, v in res.items() if v["admissivel"] and v["concorre"]]
    melhor = max(adm, key=lambda k: res[k]["S17"]) if adm else None
    if melhor is None or res[melhor]["S17"] < S6["S17"] + 0.5:
        S7 = "S6"
    else:
        perto = [k for k in adm if res[k]["S17"] > res[melhor]["S17"] - 0.5]
        ordem = {"r1": 0, "r2": 1, "r3": 2, "r4": 3}
        S7 = min(perto, key=lambda k: (res[k]["orient"], res[k]["nc"], ordem[res[k]["rota"]], res[k]["nb"], -res[k]["S17"]))
    lib = [k for k in res if res[k]["massa"] == "R1c" and res[k]["admissivel"]]
    L = max(lib, key=lambda k: res[k]["S17"]) if lib else None
    print("R17. S6 (registrado em 15.8): N_c %.2f, > 25 g %.2f, O cru %.2f, M cru %.2f, S17 %.2f\n" % (S6["N"], S6["N25"], S6["O"], S6["M"], S6["S17"]))
    print("%-30s %6s %6s %6s %7s %6s %6s %7s  %s" % ("candidato", "N_c", ">25", "O cru", "O >25", "M cru", "S17", "dens×m", "guarda / concorre | N_c 1/4/8"))
    for k in sorted(res, key=lambda k: -res[k]["S17"]):
        v = res[k]
        print("%-30s %6.2f %6.2f %6.2f %7.1f %6.2f %6.2f %7.2f  %s / %s | %s" % (k, v["N"], v["N25"], v["O"], v["O25"], v["M"], v["S17"], v["dens_massa_real"], "sim" if v["admissivel"] else "não",
              "sim" if v["concorre"] else "não", " / ".join("%.2f" % v["tabela"][o] for o in (1, 4, 8))))
    print("\nmelhor admissível que concorre: %s (S17 %.2f; S6 + 0,5 = %.2f) -> S7 = %s | melhor só com dados liberados: %s" % (melhor, res[melhor]["S17"] if melhor else float("nan"), S6["S17"] + 0.5, S7, L))
    # o S6 refeito com as cabeças da sétima execução (média de 3 sementes)
    s6r = "R4c×H0c-m r2 o4"
    print("S6 refeito (H0c-m da 7ª execução, média de 3 sementes, r2, 4 orientações): %s" % ({a: round(res[s6r][a], 2) for a in ("N", "N25", "O", "M", "S17")} if s6r in res else "rota/orientação diferente"))
    # sementes: dispersão de N_c (r2, 4 orientações) entre as 3 sementes de cada cabeça
    sem = {}
    for h in DENSI:
        if h in ("H0c", "H0c+Bi"):
            continue
        vals = []
        for k in range(3):
            zs = Semente(Z[h], k)
            vals.append(crit_pond(real, rota("r2", comps_n5k(Z["R4c"], zs, "_o4"))))
        sem[h] = vals
    print("sementes, N_c com R4c, r2, 4 orientações: " + " | ".join("%s %s (amplitude %.2f)" % (h, "/".join("%.2f" % x for x in v), max(v) - min(v)) for h, v in sem.items()))
    # 17.5: a faixa acima de 25 g com o ACETADA
    e0 = rota("r2", comps_n5k(Z["R4c"], Z["H0c-m"], "_o4")); e1 = rota("r2", comps_n5k(Z["R4c"], Z["H-A"], "_o4"))
    m1 = boot_cadeias(cad, lambda i: acima25(real, e1, i) - acima25(real, e0, i), n=2000)
    ce0, ce1 = comps_ext(Z["R4c"], Z["H0c-m"], om), comps_ext(Z["R4c"], Z["H-A"], om)
    W0, W1 = pesos_mistura(comps_n5k(Z["R4c"], Z["H0c-m"], "_o1"), real, fold), pesos_mistura(comps_n5k(Z["R4c"], Z["H-A"], "_o1"), real, fold)
    fe = np.where(om.papel == "treino", om.fold, -1)
    o0, o1 = rota("r2", ce0, W0, fe), rota("r2", ce1, W1, fe)
    g = tro & (om.proteina > 25)
    m2 = boot_cadeias(np.unique(om.cluster[g], return_inverse=True)[1], lambda i: 100 * (ok(om.proteina[g][i], o1[g][i]).mean() - ok(om.proteina[g][i], o0[g][i]).mean()), n=2000)
    d0 = massa * dens_med(Z["H0c-m"]["oof_o4"], Z["H0c-m"]["oof_dens_ff_o4"]); d1 = massa * dens_med(Z["H-A"]["oof_o4"], Z["H-A"]["oof_dens_ff_o4"])
    m3 = boot_cadeias(cad, lambda i: crit_pond(real, d1, i) - crit_pond(real, d0, i), n=2000)
    m3b = boot_cadeias(cad, lambda i: acima25(real, d1, i) - acima25(real, d0, i), n=2000)
    print("\n17.5 (H-A − H0c-m, R4c, r2, 4 orientações):")
    print("  1. > 25 g no Nutrition5k: %.2f -> %.2f; %+.2f (IC %+.2f a %+.2f)" % (acima25(real, e0), acima25(real, e1), *m1))
    print("  2. cenas > 25 g do OmniFood8K (n=%d, %d pratos principais): %.1f -> %.1f; %+.2f (IC %+.2f a %+.2f)" % (g.sum(), len(np.unique(om.cluster[g])), 100 * ok(om.proteina[g], o0[g]).mean(), 100 * ok(om.proteina[g], o1[g]).mean(), *m2))
    print("  3. densidade com a massa real: %.2f -> %.2f; %+.2f (IC %+.2f a %+.2f); só > 25 g: %.2f -> %.2f; %+.2f (IC %+.2f a %+.2f)" % (crit_pond(real, d0), crit_pond(real, d1), *m3, acima25(real, d0), acima25(real, d1), *m3b))
    # o mesmo para as receitas, para registro
    e2 = rota("r2", comps_n5k(Z["R4c"], Z["H-R"], "_o4"))
    r1 = boot_cadeias(cad, lambda i: crit_pond(real, e2, i) - crit_pond(real, e0, i), n=2000); r2 = boot_cadeias(cad, lambda i: acima25(real, e2, i) - acima25(real, e0, i), n=2000)
    print("  (registro) H-R − H0c-m: N_c %+.2f (IC %+.2f a %+.2f); > 25 g %+.2f (IC %+.2f a %+.2f)" % (*r1, *r2))
    sel = dict(res=res, S6=S6, S7=S7, melhor=melhor, liberado=L, sementes=sem, m175=dict(n5k_25=m1, omni_25=m2, dens_massa_real=m3, dens_massa_real_25=m3b),
               H_R=dict(N=r1, N25=r2), quando=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(SAIDA, exist_ok=True)
    json.dump(sel, open(os.path.join(SAIDA, "r7_selecao.json"), "w"), indent=1, default=float, ensure_ascii=False)


def externos(z_list, nome, chave_id="dish_id"):
    L = {r[chave_id]: r for r in csv.DictReader(open(os.path.join(LISTAS, "externo_%s.csv" % nome)))}
    return L


def relatos(p7, pdiv):
    marca = os.path.join(SAIDA, "r7_relatos_lido.json")
    if os.path.exists(marca):
        sys.exit("Os relatos das partes seguradas novas já foram lidos (%s)." % json.load(open(marca))["quando"])
    sel = json.load(open(os.path.join(SAIDA, "r7_selecao.json")))
    p6 = os.path.join(os.path.dirname(p7.rstrip("/")), "kaggle_out6")
    # S6 = R4c (sétima, idêntica à quinta) × H0c-m (a da sexta execução não tem as bases novas; a refeita da sétima, média de 3
    # sementes, é a que tem previsões nelas)
    mods = {"S6": ("R4c", "H0c-m", "r2", sel["res"]["R4c×H0c-m r2 o4"]["W"])}
    if sel["S7"] != "S6":
        v = sel["res"][sel["S7"]]; mods["S7"] = (v["massa"], v["densidade"], v["rota"], v["W"])
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), modelos=list(mods)), open(marca, "w"), indent=1, ensure_ascii=False)
    Z = {k: np.load(arq(p7, n)) for k, n in {**MASSA, **DENSI}.items()}
    R = {}
    for dom, ch in (("acetada", "acetada"), ("receitas_datahive", "receitas_datahive"), ("foodbd", "foodbd")):
        L = externos(None, dom)
        R[dom] = {}
        for rot, (r, h, ro, W) in mods.items():
            W = [np.array(w) for w in W]
            out = {}
            for parte, pre in (("treino_oof", "extcv"), ("segurada", "extreg")):
                ids = [s.split(":", 2)[2] for s in Z[r][pre + "_ids__" + ch]]
                PR, PH = Z[r][pre + "__" + ch], Z[h][pre + "__" + ch]
                dff = Z[h][pre + "_dens_ff__" + ch]
                dm = dens_med(PH, dff); m = np.exp(PR[:, I_M])
                r1 = m * dm; dR = np.expm1(PR[:, I_P]).clip(0); dH = np.expm1(PH[:, I_P]).clip(0)
                fo = Z[r][pre + "_fold__" + ch] if pre == "extcv" else np.full(len(ids), -1)
                e = rota(ro, (r1, dR, dH), W, fo)
                grp = np.array([L[i]["grupo"] for i in ids])
                out[parte] = dict(ids=ids, e=e, m=m, dm=dm, grp=grp, fo=fo)
            tr, sg = out["treino_oof"], out["segurada"]
            if dom == "acetada":
                pr = lambda P: np.array([float(L[i]["proteina"]) for i in P["ids"]]); ms = lambda P: np.array([float(L[i]["massa"]) for i in P["ids"]])
                real_tr, real_sg, mt, msg = pr(tr), pr(sg), ms(tr), ms(sg)
                c = float(np.median(np.log(real_tr) - np.log(tr["e"])))      # constante do domínio, nas refeições de treino (fora-do-fold)
                cm = float(np.median(np.log(mt) - np.log(tr["m"])))
                FX = [15, 25, 40]
                cr = lambda real, est, i=None: (lambda a, fx: 100 * float(np.mean([a[fx == k].mean() for k in range(4) if (fx == k).any()])))(ok(real if i is None else real[i], est if i is None else est[i]), np.digitize(real if i is None else real[i], FX, right=True))
                cl = np.unique(sg["grp"], return_inverse=True)[1]
                x = dict(n=len(real_sg), participantes=int(cl.max() + 1), cru=b2(sg["grp"], lambda i: cr(real_sg, sg["e"], i)), const=b2(sg["grp"], lambda i: cr(real_sg, np.exp(c) * sg["e"], i)),
                         massa_real_x_dens=b2(sg["grp"], lambda i: cr(real_sg, msg * sg["dm"], i)), constante=float(np.exp(c)), constante_massa=float(np.exp(cm)),
                         por_faixa_const=[100 * float(ok(real_sg, np.exp(c) * sg["e"])[np.digitize(real_sg, FX, right=True) == k].mean()) for k in range(4)],
                         por_faixa_massa_real=[100 * float(ok(real_sg, msg * sg["dm"])[np.digitize(real_sg, FX, right=True) == k].mean()) for k in range(4)],
                         vies_cru=float(100 * np.median(sg["e"] / real_sg - 1)))
                print("ACETADA segurado, %s (%d refeições, %d participantes; foto oblíqua da bandeja): cru %.1f%% (95%%: %.1f a %.1f) | com constante ×%.2f: %.1f%% (%.1f a %.1f), faixas %s | massa real × densidade: %.1f%% (%.1f a %.1f), faixas %s | viés cru %+.0f%%" %
                      (rot, x["n"], x["participantes"], *x["cru"][:3], x["constante"], *x["const"][:3], "/".join("%.0f" % y for y in x["por_faixa_const"]), *x["massa_real_x_dens"][:3], "/".join("%.0f" % y for y in x["por_faixa_massa_real"]), x["vies_cru"]))
            elif dom == "receitas_datahive":
                lab = lambda P: np.array([float(L[i]["densidade_proteina_fraca_g100"]) / 100 for i in P["ids"]])
                dl = lab(sg)
                err = np.log(sg["dm"] + EPS_D) - np.log(dl + EPS_D)
                x = dict(n=len(dl), dentro20=100 * float(np.mean(np.abs(sg["dm"] - dl) <= 0.2 * dl)), mae_log=float(np.mean(np.abs(err))), vies=float(100 * np.median(sg["dm"] / np.maximum(dl, 1e-6) - 1)),
                         spearman=float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(sg["dm"], dl)[0]))
                print("Receitas seguradas, %s (%d; rótulo fraco, ingredientes crus): densidade a ±20%% %.1f%% | erro absoluto médio em log %.3f | viés mediano %+.0f%% | correlação de postos %.2f" % (rot, x["n"], x["dentro20"], x["mae_log"], x["vies"], x["spearman"]))
            else:
                pr = lambda P: np.array([float(L[i]["proteina_fraca_g"]) for i in P["ids"]])
                real_tr, real_sg = pr(tr), pr(sg)
                c = float(np.median(np.log(real_tr) - np.log(np.maximum(tr["e"], 1e-3))))
                FX = [5, 15]
                cr = lambda real, est, i=None: (lambda a, fx: 100 * float(np.mean([a[fx == k].mean() for k in range(3) if (fx == k).any()])))(ok(real if i is None else real[i], est if i is None else est[i]), np.digitize(real if i is None else real[i], FX, right=True))
                x = dict(n=len(real_sg), cru=b2(sg["grp"], lambda i: cr(real_sg, sg["e"], i)), const=b2(sg["grp"], lambda i: cr(real_sg, np.exp(c) * sg["e"], i)), constante=float(np.exp(c)),
                         vies_cru=float(100 * np.median(sg["e"] / real_sg - 1)), spearman=float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(sg["e"], real_sg)[0]))
                print("FoodBD segurado, %s (%d pratos, rótulo de nutricionista, foto inteira na cabeça; recorte reprovado): cru %.1f%% (95%%: %.1f a %.1f) | com constante ×%.2f: %.1f%% | viés cru %+.0f%% | correlação de postos %.2f" %
                      (rot, x["n"], *x["cru"][:3], x["constante"], x["const"][0], x["vies_cru"], x["spearman"]))
            R[dom][rot] = x
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "r7_relatos.json"), "w"), indent=1, default=float, ensure_ascii=False)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) != 2:
        sys.exit(__doc__)
    (selecao if "--so-cv" in sys.argv else relatos if "--relatos" in sys.argv else registro_omnifood if "--omnifood" in sys.argv else (lambda *x: sys.exit(__doc__)))(*a)



def registro_omnifood(p7, pdiv):
    """17.6: terceira (e última) leitura da parte segurada do OmniFood8K, só para o S7; intervalo de Bonferroni de 98,3%.
    As partes só de massa não são lidas: a cabeça de massa do S7 é a do S6 (previsões idênticas)."""
    marca = os.path.join(SAIDA, "r7_omnifood_lido.json")
    if os.path.exists(marca):
        sys.exit("A terceira leitura do OmniFood8K já foi feita (%s)." % json.load(open(marca))["quando"])
    sel = json.load(open(os.path.join(SAIDA, "r7_selecao.json")))
    if sel["S7"] == "S6":
        sys.exit("17.6: S7 = S6; não há leitura.")
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), modelo=sel["S7"]), open(marca, "w"), indent=1, ensure_ascii=False)
    v = sel["res"][sel["S7"]]
    Z = {"R4c": np.load(arq(p7, MASSA["R4c"])), "H": np.load(arq(p7, DENSI[v["densidade"]]))}
    p6 = os.path.join(os.path.dirname(p7.rstrip("/")), "kaggle_out6")
    Z6 = np.load(os.path.join(p6, "pred__gpu6_fracoes_so400m_prato__massa.npz"))
    s6 = json.load(open(os.path.join(DENS, "r6", "r6_selecao.json")))["res"]["R4c×H0c-m r2 o4"]
    om = Dominio("omnifood", pdiv, Z["R4c"])
    fe = np.where(om.papel == "treino", om.fold, -1)
    e7 = rota(v["rota"], comps_ext(Z["R4c"], Z["H"], om), [np.array(w) for w in v["W"]], fe)
    e6 = rota("r2", comps_ext(Z["R4c"], Z6, om), [np.array(w) for w in s6["W"]], fe)
    s_ = (om.papel == "registro") & np.isfinite(om.proteina); so = s_ & ~om.osso
    g = np.unique(om.cluster[s_], return_inverse=True)[1]
    grupos = [np.where(g == k)[0] for k in range(g.max() + 1)]
    rng = np.random.default_rng(0)
    amostras = [np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))]) for _ in range(4000)]
    P, E7, E6 = om.proteina[s_], e7[s_], e6[s_]
    f = lambda e: crit_o(P, e)
    v7 = np.array([crit_o(P[i], E7[i]) for i in amostras]); vd = np.array([crit_o(P[i], E7[i]) - crit_o(P[i], E6[i]) for i in amostras])
    q = lambda v, a: (float(np.percentile(v, a)), float(np.percentile(v, 100 - a)))
    fx = np.digitize(P, FX_O, right=True)
    R = dict(n=int(s_.sum()), pratos=len(grupos), S7=f(E7), ic95=q(v7, 2.5), ic983=q(v7, 0.833), S6=f(E6), dif=crit_o(P, E7) - crit_o(P, E6), dif95=q(vd, 2.5), dif983=q(vd, 0.833),
             por_faixa_S7=[100 * float(ok(P, E7)[fx == k].mean()) for k in range(4)], por_faixa_S6=[100 * float(ok(P, E6)[fx == k].mean()) for k in range(4)],
             sem_osso_S7=crit_o(om.proteina[so], e7[so]), sem_osso_S6=crit_o(om.proteina[so], e6[so]))
    print("OmniFood8K segurado, 3ª leitura (%d cenas, %d pratos principais), cru: S7 %.1f%% (95%%: %.1f a %.1f; 98,3%%: %.1f a %.1f) | S6 (já lido em 15.8) %.1f%% | S7 − S6 %+.1f (95%%: %+.1f a %+.1f; 98,3%%: %+.1f a %+.1f) | faixas S7 %s, S6 %s | sem osso: S7 %.1f%%, S6 %.1f%%" %
          (R["n"], R["pratos"], R["S7"], *R["ic95"], *R["ic983"], R["S6"], R["dif"], *R["dif95"], *R["dif983"], "/".join("%.0f" % x for x in R["por_faixa_S7"]), "/".join("%.0f" % x for x in R["por_faixa_S6"]), R["sem_osso_S7"], R["sem_osso_S6"]))
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "r7_omnifood.json"), "w"), indent=1, default=float, ensure_ascii=False)


if __name__ == "__main__":
    main()
