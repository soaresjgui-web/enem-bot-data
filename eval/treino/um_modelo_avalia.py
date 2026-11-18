#!/usr/bin/env python3
"""Seção 12.6: aplica a regra pré-registrada de 12.3 à terceira execução em GPU e faz a leitura do teste2 permitida por 12.1.

    python3 um_modelo_avalia.py <pasta_kaggle_3> --so-cv     # regra de 12.3 só com as previsões fora-do-fold do treino_v2
    python3 um_modelo_avalia.py <pasta_kaggle_3> --confere   # conferências sem gabarito das previsões do teste2
    python3 um_modelo_avalia.py <pasta_kaggle_3>             # leitura única do teste2: C1 e, se a regra deixar, C2
    python3 um_modelo_avalia.py <pasta_kaggle_3> --relato    # depois da leitura: diferenças pareadas para a seção 11

Higiene: a escolha (rota, orientações, C1, C2) usa só `oof*` e os rótulos de `listas/treino_v2.csv`. O gabarito do teste2
só é aberto no modo de leitura, depois de `selecao.json` existir, e só para os candidatos escolhidos. A leitura grava
`lido.json` e o script se recusa a ler de novo. O professor e os candidatos que perderam não são pontuados no teste2."""
import csv, json, os, sys, time
import numpy as np
from zen_comum import ok, EPS_D, I_M, I_P, I_D
from dens_comum import DENS, PREC, faixas
from teste2_avalia import crit_pond, boot_cadeias, resumo

AQUI = os.path.dirname(os.path.abspath(__file__))
LISTA = os.path.join(AQUI, "kaggle", "listas", "treino_v2.csv")
AVAL2 = os.path.join(AQUI, "kaggle", "listas", "aval2_ids.csv")
SAIDA = os.path.join(DENS, "um_modelo")
UNICOS = {"U0": "gpu3 ft siglip2_b16_224 r2", "U1": "gpu3 unico so400m mlp destilado",
          "U3": "gpu3 ft siglip2_b16_384x512 r1 destilado", "U4": "gpu3 ft siglip2_b16_224 r2 destilado"}
PARES = [("U1", "U3"), ("U1", "U4"), ("U3", "U4")]          # os três pares declarados em 12.3
PROFESSOR = "gpu3 professor"
ORIENT = {1: "_o1", 4: "_o4", 8: ""}
ROTAS = ["direta", "massa_x_dens", "massa_x_dens_fracoes", "massa_x_dens_media", "geometrica"]   # ordem de 12.3
MARGEM = 1.0        # pontos: orientações (item 2) e empate de C1 (item 3)
MARGEM_C2 = 2.0     # pontos: C2 sobre C1 (12.1)
REF = {"nove redes (11.5)": 69.7, "r2 antigo (11.5)": 64.5}


def arq(pasta, nome):
    return os.path.join(pasta, "pred__" + nome.replace(" ", "_").replace("+", "_") + ".npz")


def le_lista():
    L = list(csv.DictReader(open(LISTA)))
    f = lambda c: np.array([float(r[c]) for r in L])
    return dict(ids=np.array([r["dish_id"] for r in L]), cadeia=np.array([int(r["cadeia"]) for r in L]),
                fold=np.array([int(r["fold"]) for r in L]), massa=f("massa"), proteina=f("proteina"))


def lg(d):
    return np.log(np.clip(d, 0, None) + EPS_D)


def rotas(z, conj, suf):
    """Estimativas de proteína (g) de cada rota de 12.3. Sem cabeça de frações (U0) só existem a direta, massa × densidade
    da regressão e a geométrica das duas (a 'anterior' se reduz à densidade da regressão)."""
    P = z[conj + suf]
    massa, direta = np.exp(P[:, I_M]), np.expm1(P[:, I_P]).clip(0)
    d_reg = np.clip(np.exp(P[:, I_D]) - EPS_D, 0, None)
    r = dict(direta=direta, massa_x_dens=massa * d_reg)
    if conj + "_dens_ff" + suf in z.files:
        d_ff = z[conj + "_dens_ff" + suf]
        d_med = np.clip(np.exp((lg(d_reg) + lg(d_ff)) / 2) - EPS_D, 0, None)
        r.update(massa_x_dens_fracoes=massa * d_ff, massa_x_dens_media=massa * d_med, geometrica=np.sqrt(direta * massa * d_med))
    else:
        r.update(geometrica=np.sqrt(direta * massa * d_reg))
    return r, massa


def abre(pasta, nome, L):
    z = np.load(arq(pasta, nome), allow_pickle=False)
    assert str(z["nome"]) == nome and (z["ids_treino"] == L["ids"]).all() and (z["fold"] == L["fold"]).all(), nome
    assert all(np.isfinite(z["oof" + s]).all() for s in ORIENT.values()), nome
    return z


def selecao(pasta):
    L = le_lista()
    real, cad = L["proteina"], L["cadeia"]
    serv = json.load(open(os.path.join(pasta, "servico.json")))["pipelines"]
    S = dict(n=len(real), cadeias=int(len(np.unique(cad))), faixas={k: int(v.sum()) for k, v in faixas(real).items()}, unicos={}, pares={})
    est = {}
    print("Regra de 12.3, só com previsões fora-do-fold (treino_v2: %d pratos, %d cadeias)\n" % (S["n"], S["cadeias"]))
    for u, nome in UNICOS.items():
        z = abre(pasta, nome, L)
        tab = {k: {r: crit_pond(real, e) for r, e in rotas(z, "oof", s)[0].items()} for k, s in ORIENT.items()}
        rota = max([r for r in ROTAS if r in tab[8]], key=lambda r: tab[8][r])          # item 1: rota com 8 orientações
        k = min(k for k in ORIENT if tab[k][rota] >= tab[8][rota] - MARGEM)             # item 2: menor nº de orientações
        est[u] = rotas(z, "oof", ORIENT[k])[0][rota]
        est[u + "@8"] = rotas(z, "oof", "")[0][rota]
        massa = rotas(z, "oof", ORIENT[k])[1]
        g = serv[u]["gpu_ms_por_imagem"]
        S["unicos"][u] = dict(nome=nome, tabela=tab, rota=rota, orient=k, crit=tab[k][rota], crit_o8=tab[8][rota],
                              t4_ms=g["lote1_o%d" % k], t4_ms_em_lote=g.get("lote1_o%d_em_lote" % k, g["lote1_o%d" % k]),
                              cpu_ms=serv[u]["cpu_ms_por_imagem"]["lote1_o%d" % k], mb_fp16=serv[u]["mb_fp16_em_memoria"],
                              rmse_logm=float(np.sqrt(np.mean((np.log(massa) - np.log(L["massa"])) ** 2))),
                              resumo=resumo(real, est[u], cad))
        print("%s  %s" % (u, nome))
        for k2 in ORIENT:
            print("   o%d  " % k2 + "  ".join("%s %.2f" % (r, tab[k2][r]) for r in ROTAS if r in tab[k2]))
        print("   -> rota %s, %d orientação(ões): CV %.2f (com 8: %.2f); T4 %.1f ms\n" % (rota, S["unicos"][u]["orient"], S["unicos"][u]["crit"], tab[8][rota], S["unicos"][u]["t4_ms"]))
    # item 3: C1 = maior critério; empate (< 1,0 ponto) -> o mais barato de servir. As duas leituras do critério
    # (com as orientações escolhidas e com 8) são calculadas; vale a das orientações escolhidas (12.6).
    c1 = {}
    for chave in ("crit", "crit_o8"):
        melhor = max(S["unicos"], key=lambda u: S["unicos"][u][chave])
        perto = [u for u in S["unicos"] if S["unicos"][u][chave] > S["unicos"][melhor][chave] - MARGEM]
        c1[chave] = dict(melhor=melhor, a_menos_de_1_ponto=perto, escolhido=min(perto, key=lambda u: S["unicos"][u]["t4_ms"]))
    S["c1_regra"] = c1
    C1 = c1["crit"]["escolhido"]
    S["C1"] = C1
    print("C1: maior critério = %s; a menos de 1,0 ponto: %s; mais barato na T4 = %s  (com o critério de 8 orientações: %s)" %
          (c1["crit"]["melhor"], c1["crit"]["a_menos_de_1_ponto"], C1, c1["crit_o8"]["escolhido"]))
    # item 4: pares declarados; média em log das estimativas de proteína de cada um na sua rota
    par = lambda a, b: np.expm1((np.log1p(est[a]) + np.log1p(est[b])) / 2)
    for a, b in PARES:
        e, e8 = par(a, b), par(a + "@8", b + "@8")
        dif = boot_cadeias(cad, lambda i: crit_pond(real, e, i) - crit_pond(real, est[C1], i))
        S["pares"][a + "+" + b] = dict(crit=crit_pond(real, e), crit_o8=crit_pond(real, e8), dif_c1=dif,
                                       t4_ms=S["unicos"][a]["t4_ms"] + S["unicos"][b]["t4_ms"])
        print("par %s+%s: CV %.2f (com 8 orientações: %.2f); diferença para C1 %+.2f (IC por cadeia %+.2f a %+.2f)" % (a, b, crit_pond(real, e), crit_pond(real, e8), *dif))
    C2 = max(S["pares"], key=lambda p: S["pares"][p]["crit"])
    d = S["pares"][C2]["dif_c1"]
    S["C2"] = dict(par=C2, qualifica=bool(d[0] >= MARGEM_C2 and d[1] > 0), dif=d)
    print("C2: melhor par = %s; %s (exige >= %.1f pontos sobre C1 e limite inferior > 0)" % (C2, "QUALIFICA" if S["C2"]["qualifica"] else "NÃO qualifica", MARGEM_C2))
    # referências que não concorrem: professor, e diferenças pareadas entre os únicos
    zp = abre(pasta, PROFESSOR, L)
    ep = rotas(zp, "oof", "")[0]["direta"]
    S["professor"] = dict(resumo=resumo(real, ep, cad), rmse_logm=float(np.sqrt(np.mean((zp["oof"][:, I_M] - np.log(L["massa"])) ** 2))),
                          dif_c1=boot_cadeias(cad, lambda i: crit_pond(real, est[C1], i) - crit_pond(real, ep, i)))
    S["dif_unicos"] = {u: boot_cadeias(cad, lambda i: crit_pond(real, est[u], i) - crit_pond(real, est[C1], i)) for u in UNICOS if u != C1}
    print("professor (não concorre): CV %.2f; C1 − professor %+.2f (IC %+.2f a %+.2f)" % (S["professor"]["resumo"]["ponderado"][0], *S["professor"]["dif_c1"]))
    for u, d in S["dif_unicos"].items():
        print("   %s − C1: %+.2f (IC %+.2f a %+.2f)" % (u, *d))
    for u in UNICOS:
        r = S["unicos"][u]["resumo"]
        print("   %s por faixa: " % u + "  ".join("%s %.1f" % (k, v[0]) for k, v in r["por_faixa"].items()) + "  | >25 g %.1f | RMSE log-massa %.4f" % (r["acima25"], S["unicos"][u]["rmse_logm"]))
    r = S["professor"]["resumo"]
    print("   prof por faixa: " + "  ".join("%s %.1f" % (k, v[0]) for k, v in r["por_faixa"].items()) + "  | >25 g %.1f | RMSE log-massa %.4f" % (r["acima25"], S["professor"]["rmse_logm"]))
    S["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(SAIDA, exist_ok=True)
    json.dump(S, open(os.path.join(SAIDA, "selecao.json"), "w"), indent=1, default=float, ensure_ascii=False)
    return S


def candidatos(S):
    c = [("C1", [S["C1"]])]
    if S["C2"]["qualifica"]:
        c.append(("C2", S["C2"]["par"].split("+")))
    return c


def confere(pasta, S):
    """Conferências sem gabarito (12.1, item 7): previsões completas, na ordem de aval2_ids.csv, finitas; o modelo
    remontado dos arquivos de servir reproduz as previsões gravadas (conferência feita no Kaggle, lida do registro)."""
    ids2 = np.array([r["dish_id"] for r in csv.DictReader(open(AVAL2)) if r["split"] == "teste2"])
    L = le_lista()
    assert len(ids2) == 829 and not set(ids2) & set(L["ids"])
    R = {}
    for rot, us in candidatos(S):
        for u in us:
            nome, k = S["unicos"][u]["nome"], S["unicos"][u]["orient"]
            z = np.load(arq(pasta, nome), allow_pickle=False)
            P = z["teste2" + ORIENT[k]]
            assert (z["ids_teste2"] == ids2).all() and P.shape == (829, 22) and np.isfinite(P).all(), nome
            c = json.load(open(os.path.join(pasta, "servico__" + nome.replace(" ", "_") + ".json")))["conferencia"]
            assert c["conjunto"] == "teste2" and c["max_dif_log"] <= 0.02, c
            est = rotas(z, "teste2", ORIENT[k])[0][S["unicos"][u]["rota"]]
            est_cv = rotas(z, "oof", ORIENT[k])[0][S["unicos"][u]["rota"]]
            q = lambda v: [round(float(x), 2) for x in np.percentile(v, [10, 50, 90])]
            R[u] = dict(nome=nome, orient=k, rota=S["unicos"][u]["rota"], remontado_max_dif_log=c["max_dif_log"], remontado_pratos=c["pratos"],
                        proteina_p10_50_90_teste2=q(est), proteina_p10_50_90_cv=q(est_cv))
            print("%s (%s) %s, %d orient., rota %s: 829 previsões na ordem de aval2_ids.csv, finitas; modelo remontado: dif. máx. em log %.4f (%d pratos); "
                  "proteína prevista p10/50/90 no teste2 %s (fora-do-fold do treino: %s)" % (rot, u, nome, k, R[u]["rota"], c["max_dif_log"], c["pratos"], q(est), q(est_cv)))
    json.dump(R, open(os.path.join(SAIDA, "confere.json"), "w"), indent=1, ensure_ascii=False)
    return ids2


def leitura(pasta, S):
    marca = os.path.join(SAIDA, "lido.json")
    if os.path.exists(marca):
        sys.exit("O teste2 já foi lido para a seção 12.6 (%s). Não há outra leitura." % json.load(open(marca))["quando"])
    ids2 = confere(pasta, S)
    cs = candidatos(S)
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), candidatos=[c[0] for c in cs], C1=S["C1"], C2=S["C2"]), open(marca, "w"), indent=1, ensure_ascii=False)
    G = {r["dish_id"]: r for r in csv.DictReader(open(os.path.join(PREC, "n5k_teste2_amostra.csv")))}
    real = np.array([float(G[i]["proteina_g"]) for i in ids2]); massa_real = np.array([float(G[i]["massa_g"]) for i in ids2])
    cad = np.array([int(G[i]["cadeia"]) for i in ids2])
    fx = faixas(real)
    R = dict(n=len(real), cadeias=int(len(np.unique(cad))), leituras=len(cs), candidatos={})
    por_prato = dict(ids=ids2, real=real, massa_real=massa_real, cadeia=cad)
    for rot, us in cs:
        es, ms = [], []
        for u in us:
            z = np.load(arq(pasta, S["unicos"][u]["nome"]), allow_pickle=False)
            e, m = rotas(z, "teste2", ORIENT[S["unicos"][u]["orient"]])
            es.append(e[S["unicos"][u]["rota"]]); ms.append(m)
        est = es[0] if len(es) == 1 else np.expm1(np.mean([np.log1p(e) for e in es], 0))
        massa = np.exp(np.mean([np.log(m) for m in ms], 0))
        r = resumo(real, est, cad)
        r["ponderado_975"] = boot_cadeias_q(cad, lambda i: crit_pond(real, est, i), 1.25, 98.75)
        r["massa"] = dict(mape=float(100 * np.mean(np.abs(massa - massa_real) / massa_real)), vies_mediano=float(100 * np.median(massa / massa_real - 1)),
                          rmse_log=float(np.sqrt(np.mean((np.log(massa) - np.log(massa_real)) ** 2))),
                          por_faixa={k: dict(mape=float(100 * np.mean(np.abs(massa[f] - massa_real[f]) / massa_real[f])), vies_mediano=float(100 * np.median(massa[f] / massa_real[f] - 1))) for k, f in fx.items()})
        r["vies_proteina_por_faixa"] = {k: float(100 * np.median(est[f] / np.maximum(real[f], 1e-9) - 1)) for k, f in fx.items() if k != "0-5"}
        r["dif_ref"] = {k: r["ponderado"][0] - v for k, v in REF.items()}
        inf = r["ponderado"][1]
        r["veredito"] = "70% DEMONSTRADO" if inf >= 70 else "70% NÃO DEMONSTRADO" + (" (intervalo inteiro abaixo de 70%)" if r["ponderado"][2] < 70 else " (70% dentro do intervalo)")
        R["candidatos"][rot] = dict(modelos=us, **r)
        por_prato["est_" + rot] = est; por_prato["massa_" + rot] = massa
        print("\n%s = %s" % (rot, " + ".join(S["unicos"][u]["nome"] for u in us)))
        print("  teste2 ponderado %.1f%% (IC 95%% por cadeia %.1f a %.1f; 97,5%%: %.1f a %.1f); simples %.1f%% (%.1f a %.1f)" % (*r["ponderado"], *r["ponderado_975"][1:], *r["simples"]))
        print("  por faixa: " + "  ".join("%s %.1f%% (n=%d)" % (k, v[0], v[1]) for k, v in r["por_faixa"].items()) + "  | >25 g %.1f%% | >15 g %.1f%%" % (r["acima25"], r["acima15"]))
        print("  contra as referências: " + "; ".join("%s %+.1f" % (k, v) for k, v in r["dif_ref"].items()))
        print("  massa: MAPE %.1f%%, viés mediano %+.1f%%, RMSE log %.4f; por faixa: " % (r["massa"]["mape"], r["massa"]["vies_mediano"], r["massa"]["rmse_log"]) +
              "  ".join("%s %.1f%% (%+.1f%%)" % (k, v["mape"], v["vies_mediano"]) for k, v in r["massa"]["por_faixa"].items()))
        print("  viés mediano da proteína por faixa: " + "  ".join("%s %+.1f%%" % kv for kv in r["vies_proteina_por_faixa"].items()))
        print("  VEREDITO PELA REGRA DE 12.1: " + r["veredito"])
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "avalia_teste2.json"), "w"), indent=1, default=float, ensure_ascii=False)
    np.savez(os.path.join(SAIDA, "pratos_teste2.npz"), **por_prato)


def relato():
    """Depois da leitura: relatos complementares com os resultados por prato já gravados (nenhum candidato novo é
    pontuado). Diferença pareada do C1 para os pipelines lidos na seção 11 e os 107 pratos sem irmão no `teste` antigo."""
    b = np.load(os.path.join(SAIDA, "pratos_teste2.npz"))
    a = np.load(os.path.join(DENS, "teste2", "pratos_teste2.npz"), allow_pickle=True)
    mp = {r["prato_id"]: r["dish_id"] for r in csv.DictReader(open(AVAL2))}
    assert (np.array([mp[i] for i in a["ids"]]) == b["ids"]).all() and np.allclose(a["real"], b["real"]) and (a["cadeia"] == b["cadeia"]).all()
    real, cad, ind = b["real"], b["cadeia"], a["independente"]
    a25 = lambda e, j: 100 * np.mean([ok(real[j], e[j])[(real[j] > 25) & (real[j] <= 40)].mean(), ok(real[j], e[j])[real[j] > 40].mean()])
    R = {}
    for rot in [k[4:] for k in b.files if k.startswith("est_")]:
        e = b["est_" + rot]
        R[rot] = dict(sem_irmao_no_teste=dict(n=int(ind.sum()), cadeias=int(len(np.unique(cad[ind]))), ponderado=boot_cadeias(cad[ind], lambda j: crit_pond(real[ind], e[ind], j))),
                      irmaos_do_teste=dict(n=int((~ind).sum()), ponderado=boot_cadeias(cad[~ind], lambda j: crit_pond(real[~ind], e[~ind], j))), contra={})
        print("%s: sem irmão no teste antigo (n=%d, %d cadeias) %.1f%% (IC %.1f a %.1f); irmãos (n=%d) %.1f%% (IC %.1f a %.1f)" %
              (rot, ind.sum(), R[rot]["sem_irmao_no_teste"]["cadeias"], *R[rot]["sem_irmao_no_teste"]["ponderado"], (~ind).sum(), *R[rot]["irmaos_do_teste"]["ponderado"]))
        for i, n in enumerate(a["nomes"]):
            r = a["est%d" % i]
            d = boot_cadeias(cad, lambda j: crit_pond(real, e, j) - crit_pond(real, r, j))
            d2 = boot_cadeias(cad, lambda j: a25(e, j) - a25(r, j))
            R[rot]["contra"][str(n)] = dict(ref=crit_pond(real, r), dif=d, dif_acima25=d2)
            print("   %s − %s (%.1f%%): %+.1f (IC %+.1f a %+.1f); acima de 25 g: %+.1f (IC %+.1f a %+.1f)" % (rot, n, crit_pond(real, r), *d, *d2))
    json.dump(R, open(os.path.join(SAIDA, "relato_teste2.json"), "w"), indent=1, default=float, ensure_ascii=False)


def boot_cadeias_q(cadeia, estat, q0, q1, n=4000, seed=0):
    """Como boot_cadeias (mesmas sementes e sorteios), com outros percentis."""
    grupos = [np.where(cadeia == c)[0] for c in np.unique(cadeia)]
    rng = np.random.default_rng(seed)
    v = np.array([estat(np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))])) for _ in range(n)])
    return float(estat(np.arange(len(cadeia)))), float(np.percentile(v, q0)), float(np.percentile(v, q1))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    pasta = args[0]
    if "--so-cv" in sys.argv:
        selecao(pasta)
        return
    if "--relato" in sys.argv:
        relato()
        return
    sel = os.path.join(SAIDA, "selecao.json")
    if not os.path.exists(sel):
        sys.exit("Rode antes com --so-cv: a escolha tem de estar gravada antes de qualquer leitura do teste2.")
    S = json.load(open(sel))
    if "--confere" in sys.argv:
        confere(pasta, S)
    else:
        leitura(pasta, S)


if __name__ == "__main__":
    main()
