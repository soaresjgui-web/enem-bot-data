#!/usr/bin/env python3
"""Seção 18.6–18.7: escolha na validação cruzada do Nutrition5k, condições da leitura 3 do DFood-TJ, relato do FPB e a
leitura 3 (com marca própria). Tabelas do kernel: seção 18.10.

    python3 -I avalia_geom.py --cv <dir_treino> <pasta_kaggle_7> <fotos_geom.csv> <itens_geom.csv.gz>
    python3 -I avalia_geom.py --condicoes <fotos_geom.csv> <itens_geom.csv.gz> <conferencia_mascaras.json>
    python3 -I avalia_geom.py --fpb <pasta_kaggle_7> <pasta_divisoes> <fotos_geom.csv> <itens_geom.csv.gz>       # última leitura do FPB; só ordem
    python3 -I avalia_geom.py --leitura3 <pasta_kaggle_8b> <gabarito> <conferencia_r9.json> <fotos_geom.csv> <itens_geom.csv.gz>
    python3 -I avalia_geom.py --ensaio <dir_treino> <pasta_kaggle_7> <pasta_kaggle_8b> <conferencia_r9.json>   # tabelas sintéticas e rótulos sorteados

A escala do Nutrition5k vem de <DENS>/geom/escala_n5k.csv (escala.py n5k); as alturas por grupo, de priors.csv
(calibra.py --alturas) ou, se estiverem vazias, calculadas na hora a partir das máscaras do treino (sem grama).
"""
import csv, hashlib, json, os, sys, time
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "treino"))
from escala import le_tabela, nums, dfoodtj as escala_dfoodtj, condicao_ovos, rle_codifica  # noqa: E402
from calibra import (GRUPOS, VOC, densidades, alturas_prior, le_n5k_grupos, le_escala_n5k, itens_por_foto, areas, alturas_n5k,  # noqa: E402
                     ajusta_k, ajusta_rho, g1, g2, gprof, mistura, aplica_mistura)
from dens_comum import DENS  # noqa: E402

SAIDA = os.path.join(DENS, "geom")
SELECAO = os.path.join(SAIDA, "geom_selecao.json")
CONDICOES = os.path.join(SAIDA, "geom_condicoes.json")
MARCA3 = os.path.join(SAIDA, "geom_leitura3_lida.json")
MARCA_FPB = os.path.join(SAIDA, "geom_fpb_lido.json")
ESCALA_N5K = os.path.join(SAIDA, "escala_n5k.csv")


def sha(caminho):
    return hashlib.sha256(open(caminho, "rb").read()).hexdigest()


def boot(cl, f, n=4000, q=(2.5, 97.5, 0.8333, 99.1667)):
    """Bootstrap por cluster: (valor, percentis de q)."""
    g = np.unique(cl, return_inverse=True)[1]
    grupos = [np.where(g == k)[0] for k in range(g.max() + 1)]
    rng = np.random.default_rng(0)
    v = np.array([f(np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))])) for _ in range(n)])
    return (float(f(np.arange(len(cl)))),) + tuple(float(np.nanpercentile(v, x)) for x in q)


# ------------------------------------------------------------------ Nutrition5k: S7 fora-do-fold e candidatos
def s7_n5k(p7, N):
    from r7_avalia import arq, MASSA, DENSI
    from r6_avalia import comps_n5k, pesos_mistura, rota
    from r5_avalia import dens_med
    from zen_comum import I_M
    R, H = np.load(arq(p7, MASSA["R4c"])), np.load(arq(p7, DENSI["H-cap"]))
    assert (R["ids_treino"] == N["ids"]).all()
    cs = comps_n5k(R, H, "_o4"); fold = R["fold"]
    W = pesos_mistura(cs, N["proteina"], fold)
    return dict(prot=rota("r4", cs, W, fold), massa=np.exp(R["oof_o4"][:, I_M]), dens=dens_med(H["oof_o4"], H["oof_dens_ff_o4"]), comps=cs, fold=fold)


def candidatos_n5k(N, s7, A, por_item, hg, vol, dens):
    n, fold = len(N["ids"]), s7["fold"]
    C = {c: dict(massa=np.full(n, np.nan), prot=np.full(n, np.nan)) for c in ("G1", "G2", "G3", "G-prof")}
    for f in range(5):
        tr, va = fold != f, np.where(fold == f)[0]
        k, rho = ajusta_k(A[tr], N["massa"][tr], N["Mg"][tr]), ajusta_rho(N["Mg"][tr], N["Pg"][tr])
        m1, p1 = g1(A[va], k, rho)
        C["G1"]["massa"][va], C["G1"]["prot"][va] = m1, p1
        C["G3"]["massa"][va], C["G3"]["prot"][va] = m1, m1 * s7["dens"][va]
        C["G2"]["massa"][va], C["G2"]["prot"][va] = g2([por_item[i] for i in va], hg, dens, rho)
        C["G-prof"]["massa"][va], C["G-prof"]["prot"][va] = gprof(N["ids"][va], [por_item[i] for i in va], vol, dens, rho)
    X = np.c_[C["G1"]["prot"], C["G3"]["prot"], s7["comps"][0], s7["comps"][1], s7["comps"][2]]
    W4 = mistura(X, N["proteina"], fold)
    C["G4"] = dict(massa=C["G1"]["massa"], prot=aplica_mistura(X, W4, fold))
    return C, W4


def tabelas(fotos, itens):
    return le_tabela(fotos), le_tabela(itens)


def prepara_n5k(td, fotos, itens):
    N = le_n5k_grupos()
    F = {f["id"]: f for f in fotos if f["base"] == "n5k"}
    cm = le_escala_n5k(ESCALA_N5K)
    itf = itens_por_foto(itens, "n5k")
    A, por_item = areas(list(N["ids"]), itf, cm)
    hp = alturas_prior()
    hg, vol = alturas_n5k(td, list(N["ids"]), itf, cm, F)
    if len(hp) == len(GRUPOS):
        hg = hp                                       # alturas gravadas em priors.csv têm prioridade (são as pré-registradas)
    return N, A, por_item, hg, vol


def cv(td, p7, fotos, itens, ensaio=False):
    from teste2_avalia import crit_pond, boot_cadeias
    from r4_avalia import acima25, dentro20
    fotos, itens = tabelas(fotos, itens) if isinstance(fotos, str) else (fotos, itens)
    N, A, por_item, hg, vol = prepara_n5k(td, fotos, itens)
    dens = densidades()
    s7 = s7_n5k(p7, N)
    C, W4 = candidatos_n5k(N, s7, A, por_item, hg, vol, dens)
    real, M, cad = N["proteina"], N["massa"], N["cadeia"]
    lin = {}
    ref = dict(N=crit_pond(real, s7["prot"]), N25=acima25(real, s7["prot"]), M20=dentro20(M, s7["massa"]), vies=100 * float(np.median(s7["massa"] / M - 1)))
    print("S7 (referência, 4 orientações): N_c %.2f | > 25 g %.2f | massa (R4c) a ±20%% %.1f%% (viés %+.0f%%)" % (ref["N"], ref["N25"], ref["M20"], ref["vies"]))
    for c, v in C.items():
        ok = np.isfinite(v["prot"])
        e = np.where(ok, v["prot"], 0.0)
        d = boot_cadeias(cad, lambda i: crit_pond(real[i], e[i]) - crit_pond(real[i], s7["prot"][i]))
        mm = np.where(np.isfinite(v["massa"]), v["massa"], 0.0)
        lin[c] = dict(N=crit_pond(real, e), N25=acima25(real, e), M20=dentro20(M, mm), vies=100 * float(np.nanmedian(v["massa"] / M - 1)),
                      dif=d[0], dif95=[d[1], d[2]], cobertura=float(ok.mean()))
        x = lin[c]
        print("%-7s N_c %5.2f | > 25 g %5.2f | massa a ±20%% %5.1f%% (viés %+4.0f%%) | contra o S7 %+5.2f (IC %+.2f a %+.2f) | pratos com previsão %.0f%%"
              % (c, x["N"], x["N25"], x["M20"], x["vies"], x["dif"], *x["dif95"], 100 * x["cobertura"]))
    troca = [c for c, x in lin.items() if x["N"] >= ref["N"] + 1.0 and x["dif95"][0] > 0 and x["N25"] >= ref["N25"] - 2.5]
    cand = max(("G1", "G3"), key=lambda c: lin[c]["N"])
    print("18.6 (1), troca do app: %s" % (", ".join(troca) if troca else "nenhum candidato passa"))
    print("18.6 (2), candidato do DFood-TJ: %s (N_c %.2f); 18.7 (3): massa a ±20%% %.1f contra %.1f da R4c -> %s"
          % (cand, lin[cand]["N"], lin[cand]["M20"], ref["M20"], "passa" if lin[cand]["M20"] >= ref["M20"] - 15 else "FALHA"))
    k_all, rho_all = ajusta_k(A, N["massa"], N["Mg"]), ajusta_rho(N["Mg"], N["Pg"])
    sel = dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), ensaio=ensaio, referencia=ref, linhas=lin, troca_app=troca, candidato=cand,
               condicao3=bool(lin[cand]["M20"] >= ref["M20"] - 15),
               congelado=dict(k=list(k_all), rho=list(rho_all), grupos=GRUPOS, alturas=hg, densidades=dens, W4=[list(w) for w in W4],
                              priors_sha256=sha(os.path.join(AQUI, "priors.csv")), vocabulario_sha256=sha(os.path.join(AQUI, "vocabulario.csv"))))
    if not ensaio:
        os.makedirs(SAIDA, exist_ok=True)
        json.dump(sel, open(SELECAO, "w"), indent=1, default=float, ensure_ascii=False)
        print("Congelado: %s (sha256 %s)" % (SELECAO, sha(SELECAO)))
    return sel


# ------------------------------------------------------------------ condições de 18.7
def condicoes(fotos, itens, conf_mascaras, sel=None, ensaio=False):
    fotos, itens = tabelas(fotos, itens) if isinstance(fotos, str) else (fotos, itens)
    cm_ = json.load(open(conf_mascaras)) if isinstance(conf_mascaras, str) else conf_mascaras
    d = cm_["dfoodtj"]
    c1 = bool(d["certas"] >= 36 and d["proteina_errada"] <= 2)
    res, cmpx = escala_dfoodtj(fotos, itens)
    c2 = condicao_ovos(res)
    if sel is None:
        sel = json.load(open(SELECAO)) if os.path.exists(SELECAO) else None
    c3 = bool(sel and sel["condicao3"])
    c4 = bool(sel and not sel.get("ensaio") and os.path.exists(SELECAO))
    R = dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), c1_mascaras=c1, c2_ovos=c2, c3_massa_n5k=c3, c4_congelado=c4, ovos=res,
             selecao_sha256=sha(SELECAO) if os.path.exists(SELECAO) else None, todas=bool(c1 and c2 and c3 and (c4 or ensaio)))
    print("18.7: máscaras %s | ovos %s %s | massa no Nutrition5k %s | congelado %s -> leitura 3 %s"
          % (c1, c2, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}), c3, c4, "LIBERADA" if R["todas"] else "NÃO acontece"))
    if not ensaio:
        os.makedirs(SAIDA, exist_ok=True)
        json.dump(R, open(CONDICOES, "w"), indent=1, default=float, ensure_ascii=False)
    return R, cmpx


# ------------------------------------------------------------------ DFood-TJ: geometria + S7 da leitura 2
def previsoes_dfoodtj(z, ids, itens, cmpx, sel):
    from r8_avalia import o4
    from r5_avalia import dens_med
    from zen_comum import I_M
    cg = sel["congelado"]; k, rho = np.array(cg["k"]), np.array(cg["rho"])
    A, _ = areas(ids, itens_por_foto(itens, "dfoodtj"), cmpx)
    m1, p1 = g1(A, k, rho)
    PH = o4(z["recorte__hcap__s"].astype(np.float64)); dff = z["recorte__hcap__dens_ff"].astype(np.float64)[:, :4].mean(1)
    dm = dens_med(PH, dff)
    return dict(G1=dict(massa=m1, prot=p1), G3=dict(massa=m1, prot=m1 * dm))


def leitura3(p8b, gab, conf_r9, fotos, itens, ensaio=False):
    from r8_avalia import conferencia, estimativas, crit3, por_faixa, MARCA, MARCA2, SHA_GABARITO
    from r4_avalia import dentro20
    fotos, itens = tabelas(fotos, itens) if isinstance(fotos, str) else (fotos, itens)
    npz = next(n for n in ("pred9__dfoodtj.npz", "pred8__dfoodtj.npz") if os.path.exists(os.path.join(p8b, n)))
    z = np.load(os.path.join(p8b, npz), allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    manter, entrada, reg = conferencia(conf_r9, z)
    if ensaio:
        sel = ensaio["sel"]; cmpx = ensaio["cmpx"]
        rng = np.random.default_rng(3)
        G = {i: dict(proteina_corrigida=rng.uniform(3, 45), massa=rng.uniform(150, 700), paciente=str(p)) for i, p in zip(ids, z["paciente"])}
    else:
        C = json.load(open(CONDICOES)) if os.path.exists(CONDICOES) else {}
        if not C.get("todas"):
            sys.exit("18.7: as condições não passaram (ou não foram avaliadas); a leitura 3 não acontece.")
        if C.get("selecao_sha256") != sha(SELECAO):
            sys.exit("O candidato congelado mudou depois das condições.")
        if not (os.path.exists(MARCA) and os.path.exists(MARCA2)):
            sys.exit("As leituras 1 e 2 não constam.")
        if os.path.exists(MARCA3):
            sys.exit("A leitura 3 do DFood-TJ já foi feita (%s)." % json.load(open(MARCA3))["quando"])
        if sha(gab) != SHA_GABARITO:
            sys.exit("sha256 do gabarito não confere com o de 16.6.")
        sel = json.load(open(SELECAO))
        _, cmpx = escala_dfoodtj(fotos, itens)
        os.makedirs(SAIDA, exist_ok=True)
        json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), candidato=sel["candidato"], selecao_sha256=sha(SELECAO), conferencia=reg, leitura=3),
                  open(MARCA3, "w"), indent=1, ensure_ascii=False)
        G = {r["dish_id"]: r for r in csv.DictReader(open(gab))}
    pv = previsoes_dfoodtj(z, ids, itens, cmpx, sel)
    s7 = estimativas(z, "recorte")
    idx = np.where(manter)[0]
    P = np.array([float(G[ids[i]]["proteina_corrigida"]) for i in idx]); M = np.array([float(G[ids[i]]["massa"]) for i in idx])
    pac = np.array([G[ids[i]]["paciente"] for i in idx])
    cand = sel["candidato"]
    R = dict(n=len(idx), pacientes=len(set(pac)), candidato=cand, ensaio=bool(ensaio))

    def linha(nome, e, m):
        e = np.where(np.isfinite(e), e, 0.0)
        v = boot(pac, lambda j: crit3(P[j], e[j]))
        pf, vies = por_faixa(P, e)
        R[nome] = dict(valor=v[0], ic95=v[1:3], ic983=v[3:5], por_faixa=pf, vies_por_faixa=vies, massa20=dentro20(M, np.where(np.isfinite(m), m, 0.0)),
                       vies_massa=100 * float(np.nanmedian(m / M - 1)))
        print("%-26s n=%d  %5.1f%%  IC95 %5.1f a %5.1f | IC98,3 %5.1f a %5.1f | faixas %s | massa a ±20%% %.1f%% (viés %+.0f%%)"
              % (nome, len(idx), v[0], v[1], v[2], v[3], v[4], "/".join("%.0f" % x for x in pf), R[nome]["massa20"], R[nome]["vies_massa"]))

    print("LEITURA 3 (18.7), regra: limite inferior do IC de 98,3% (Bonferroni, k = 3) de 70% ou mais")
    linha("%s (REGRA)" % cand, pv[cand]["prot"][idx], pv[cand]["massa"][idx])
    R["veredito"] = "sucesso" if R["%s (REGRA)" % cand]["ic983"][0] >= 70 else "não atingida"
    linha("S7 (leitura 2)", s7["S7"][idx], s7["massa"][idx])
    outro = "G3" if cand == "G1" else "G1"
    linha(outro + " (descritiva)", pv[outro]["prot"][idx], pv[outro]["massa"][idx])
    ec, es = np.nan_to_num(pv[cand]["prot"][idx]), s7["S7"][idx]
    d = boot(pac, lambda j: crit3(P[j], ec[j]) - crit3(P[j], es[j]))
    R["dif_S7"] = dict(valor=d[0], ic95=d[1:3])
    print("  -> meta %s | %s − S7: %+.1f (IC95 %+.1f a %+.1f)" % (R["veredito"], cand, d[0], d[1], d[2]))
    if not ensaio:
        json.dump(R, open(os.path.join(SAIDA, "geom_leitura3.json"), "w"), indent=1, default=float, ensure_ascii=False)
    return R


# ------------------------------------------------------------------ FPB: última leitura, só ordem
def fpb(p7, pdiv, fotos, itens, ensaio=False):
    from scipy.stats import spearmanr
    fotos, itens = tabelas(fotos, itens) if isinstance(fotos, str) else (fotos, itens)
    sel = ensaio["sel"] if ensaio else json.load(open(SELECAO))
    reg = [r["dish_id"] for r in csv.DictReader(open(os.path.join(pdiv, "divisao_r4_fpb.csv"))) if r["papel"] == "registro" and r["quadro_metrica"] == "1"]
    itf = itens_por_foto(itens, "fpb")
    A, _ = areas(reg, itf, {i: 1.0 for i in reg})                 # sem escala absoluta: área em pixels (só ordem)
    k = np.array(sel["congelado"]["k"])
    rel = A @ k
    if ensaio:
        M = np.random.default_rng(4).uniform(100, 900, len(reg))
    else:
        if os.path.exists(MARCA_FPB):
            sys.exit("A última leitura do FPB já foi feita (%s)." % json.load(open(MARCA_FPB))["quando"])
        json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), selecao_sha256=sha(SELECAO), leitura=3), open(MARCA_FPB, "w"), indent=1)
        L = {r["dish_id"]: r for r in csv.DictReader(open(os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_fpb.csv")))}
        M = np.array([float(L[i]["massa"]) for i in reg])
    ok = np.isfinite(rel) & (rel > 0)
    rho_ = spearmanr(rel[ok], M[ok])[0]
    print("FPB, parte segurada (%d fotos, %d com itens): correlação de postos massa geométrica relativa × massa pesada = %.3f (só ordem; 18.2)" % (len(reg), ok.sum(), rho_))
    if not ensaio:
        json.dump(dict(n=len(reg), com_itens=int(ok.sum()), spearman=rho_), open(os.path.join(SAIDA, "geom_fpb.json"), "w"), indent=1)


# ------------------------------------------------------------------ ensaio com tabelas sintéticas
def sinteticas(td, p8b, rng):
    """Tabelas no formato de 18.10, com máscaras elípticas sorteadas: Nutrition5k (treino), DFood-TJ (711) e FPB (registro)."""
    from escala import caixa_prato_n5k
    termos = [t for t in VOC if t != "resto"]
    fotos, itens = [], []

    def elipse(h, w, cx, cy, a, b):
        yy, xx = np.mgrid[0:h, 0:w]
        return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1

    def item(base, i, n, termo, m, f, nota=0.5):
        ys, xs = np.where(m)
        el = "%.1f %.1f %.1f %.1f 0" % (xs.mean() * f, ys.mean() * f, (np.ptp(xs) + 1) * f, (np.ptp(ys) + 1) * f)
        return dict(base=base, id=i, item=n, termo=termo, consulta=VOC[termo]["consulta_owlv2"], nota=nota,
                    caixa="%d %d %d %d" % (xs.min() * f, ys.min() * f, (xs.max() + 1) * f, (ys.max() + 1) * f), rle=rle_codifica(m),
                    mascara_largura=m.shape[1], mascara_altura=m.shape[0], area_px=int(m.sum() * f * f), elipse=el, origem_mascara="sintetica", dentro_prato=1.0)
    for d in le_n5k_grupos()["ids"]:
        cx = caixa_prato_n5k(td, d) or [120, 40, 520, 440]
        fotos.append(dict(base="n5k", id=d, largura=640, altura=480, prato_caixa="%.0f %.0f %.0f %.0f" % tuple(cx), periodo="", paciente=""))
        pcx, pcy, r = (cx[0] + cx[2]) / 2, (cx[1] + cx[3]) / 2, (cx[2] - cx[0]) / 2
        for n in range(rng.integers(2, 5)):
            a = r * rng.uniform(0.15, 0.35)
            itens.append(item("n5k", d, n, rng.choice(termos), elipse(480, 640, pcx + rng.uniform(-.4, .4) * r, pcy + rng.uniform(-.4, .4) * r, a, a * rng.uniform(.6, 1)), 1.0))
    z = np.load(os.path.join(p8b, "pred8__dfoodtj.npz"), allow_pickle=False)
    for d, per in zip(z["ids"], z["periodo"]):
        d, per = str(d), str(per)
        fotos.append(dict(base="dfoodtj", id=d, largura=1440, altura=1080, prato_elipse="720 540 900 860 0", periodo=per, paciente=""))
        for n in range(rng.integers(2, 5)):
            itens.append(item("dfoodtj", d, n, rng.choice(termos), elipse(540, 720, 360 + rng.uniform(-120, 120), 270 + rng.uniform(-100, 100), 60, 45), 2.0))
        if rng.random() < 0.3:
            ovo = elipse(540, 720, 300, 250, 900 / 6.0 / 2 * 1.3 / 2, 900 / 6.0 / 2 / 2)       # eixo menor ≈ prato/6 (em px originais)
            itens.append(item("dfoodtj", d, 9, "boiled_egg_in_shell", ovo, 2.0, 0.6))
    return fotos, itens


def ensaio(td, p7, p8b, conf_r9):
    rng = np.random.default_rng(0)
    print("ENSAIO: tabelas sintéticas, rótulos sorteados no DFood-TJ e no FPB; nada é gravado e nenhuma marca é criada.")
    fotos, itens = sinteticas(td, p8b, rng)
    sel = cv(td, p7, fotos, itens, ensaio=True)
    R, cmpx = condicoes(fotos, itens, {"dfoodtj": dict(certas=38, proteina_errada=1)}, sel=sel, ensaio=True)
    leitura3(p8b, None, conf_r9, fotos, itens, ensaio=dict(sel=sel, cmpx=cmpx))
    pdiv = os.path.join(os.path.dirname(td.rstrip("/")), "listas_r4")
    if os.path.exists(os.path.join(pdiv, "divisao_r4_fpb.csv")):
        reg = [r["dish_id"] for r in csv.DictReader(open(os.path.join(pdiv, "divisao_r4_fpb.csv"))) if r["papel"] == "registro" and r["quadro_metrica"] == "1"]
        for d in reg:
            m = np.zeros((480, 640), bool); m[100:100 + rng.integers(40, 200), 100:300] = True
            itens.append(dict(base="fpb", id=d, item=0, termo="meat", nota=0.5, area_px=int(m.sum()), rle=rle_codifica(m), mascara_largura=640, mascara_altura=480))
        fpb(p7, pdiv, fotos, itens, ensaio=dict(sel=sel))


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    m = [x for x in sys.argv[1:] if x.startswith("--")]
    if m == ["--cv"] and len(a) == 4:
        cv(*a)
    elif m == ["--condicoes"] and len(a) == 3:
        condicoes(*a)
    elif m == ["--fpb"] and len(a) == 4:
        fpb(*a)
    elif m == ["--leitura3"] and len(a) == 5:
        leitura3(*a)
    elif m == ["--ensaio"] and len(a) == 4:
        ensaio(*a)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
