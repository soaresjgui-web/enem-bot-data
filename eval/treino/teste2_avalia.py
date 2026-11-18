#!/usr/bin/env python3
"""Seção 11: leitura ÚNICA do teste2 com o pipeline congelado da seção 10 (regra em 11.2 e 11.3 do relatório).

Uso: python3 teste2_avalia.py <pasta_kaggle> [--confere | --ensaio]
  <pasta_kaggle>  saída da nova execução no Kaggle (pred__*.npz com oof, calib, teste, n5k_v1 e teste2/ids_teste2)
  --confere  só conferências SEM gabarito do teste2: previsões completas e na ordem certa, paridade dos modelos
             refeitos com os da seção 10 nos 450 pratos antigos, CV no treino, estimativas v3 completas.
             Pode rodar quantas vezes for preciso.
  --ensaio   roda a pontuação inteira com o `teste` ANTIGO no lugar do teste2 (serve com a pasta da primeira execução
             no Kaggle, que não tem teste2): confere que o código refaz os números da seção 10. Não lê o teste2.
  (sem opção) pontua o teste2. Roda UMA vez: grava <TD>/resultados/dens/teste2/lido.json e recusa rodar de novo.

Pipeline congelado (nada é escolhido nem ajustado aqui):
  massa     = exp(média dos logs de massa de MASSA)                                   [5 backbones congelados + ridge]
  dens      = média em log (piso 0,01) de: cabeça de densidade da média dos 4 ajustes finos (FT4); cabeça de densidade
              do ridge So400m; as três cabeças de frações de CPU (teste2_cabecas.py)
  só imagem = massa × dens
  híbrido   = massa × [(dens v3 com frango/peru na densidade média do treino)^0,4 × dens^0,6]
              LLM v3: uma rodada cega (docs/dados/precisao/estimativas_v3_teste2_partes.csv); no `teste` antigo, rodada 1.
Medida principal: critério do produto (±20% ou ±5 g) do híbrido, ponderado para as cinco faixas de proteína real terem o
mesmo peso; IC de 95% por bootstrap de CADEIAS (4.000 sorteios). Decisão: demonstrado se o limite inferior >= 70;
refutado se o limite superior < 70; senão, não demonstrado.
"""
import csv, glob, json, os, sys, time
import numpy as np
from zen_comum import Dados, proteinas, mape, ok, TD, EPS_D, I_M
from dens_comum import DENS, PREC, Ingr, faixas, llm_v3, V3_PARTES

MASSA = ["gpu ridge siglip2_b16_384x512 o8", "gpu ridge dinov2_b14_336x448 o8", "gpu ridge dinov2_l14_336x448 o8",
         "gpu ridge siglip2_l16_384x512 o8", "gpu ridge siglip2_so400m16_384x512 o8"]
FT4 = ["gpu ft siglip2_b16_224 r1", "gpu ft siglip2_b16_384x512 r1", "gpu ft dinov2_b14_336x448 r1", "gpu ft dinov2_l14_224x308 r1"]
SO400M = "gpu ridge siglip2_so400m16_384x512 o8"
CABECAS = ["soft-mlp ingrediente-l2 siglip2_b16_512", "soft-mlp fonte-l2d siglip2_b16_512", "soft-mlp fonte-l2 +perda-dens peso-faixa ambos"]
UNICO = "gpu ft siglip2_b16_224 r2"                # modelo único para o aparelho (só relato)
PESO_LLM = 0.4                                     # peso da densidade do LLM no híbrido, ajustado no calib na seção 10
DENS_FRANGO = 0.2825                               # densidade média de frango/peru no treino (conferida abaixo)
V3_TESTE2 = os.path.join(PREC, "estimativas_v3_teste2_partes.csv")
SAIDA = os.path.join(DENS, "teste2")
ANTIGOS = ["calib", "teste", "n5k_v1"]
lg = lambda d: np.log(np.maximum(d, 0) + EPS_D)
un = lambda x: np.clip(np.exp(x) - EPS_D, 0, None)
f3 = lambda v: f"{v[0]:.1f}% (IC {v[1]:.1f} a {v[2]:.1f})"


def le_kaggle(pasta, D, ids2=None):
    """nome -> dict(oof, calib, teste, n5k_v1[, teste2]) em log; confere ids e ordem."""
    K = {}
    for arq in sorted(glob.glob(os.path.join(pasta, "**", "pred__*.npz"), recursive=True)):
        z = np.load(arq, allow_pickle=False)
        nome = str(z["nome"])
        assert list(z["ids_treino"]) == D.ids, f"{nome}: ordem do treino"
        e = {"oof": z["oof"].astype(np.float64)}
        for s in ANTIGOS:
            assert list(z[f"ids_{s}"]) == D.aval[s]["chave"], f"{nome}: ordem de {s}"
            e[s] = z[s].astype(np.float64)
        if ids2 is not None and "teste2" in z.files:
            assert list(z["ids_teste2"]) == ids2, f"{nome}: ordem do teste2"
            e["teste2"] = z["teste2"].astype(np.float64)
        assert all(np.isfinite(v).all() for v in e.values()), f"{nome}: valores não finitos"
        K[nome] = e
    return K


def le_cabecas(alvo):
    H = {}
    for c in CABECAS:
        z = np.load(os.path.join(SAIDA, c + ".npz"), allow_pickle=False)
        H[c] = {s: z[f"{s}_dens"] for s in ANTIGOS + (["teste2"] if alvo == "teste2" else [])}
        H[c]["ids_teste2"] = list(z["ids_teste2"]); H[c]["paridade"] = z["paridade"]
    return H


def congelado(K, H, s):
    """Massa, densidade do modelo e proteína só de imagem do pipeline congelado, no split s."""
    massa = np.exp(np.mean([K[n][s][:, I_M] for n in MASSA], 0))
    ds = [proteinas(np.mean([K[n][s] for n in FT4], 0))["dens"], proteinas(K[SO400M][s])["dens"]] + [H[c][s] for c in CABECAS]
    dens = un(np.mean([lg(d) for d in ds], 0))
    return massa, dens


def hibrido(massa, dens, v3):
    """massa × [(dens v3 com frango/peru na média do treino)^0,4 × dens do modelo^0,6]."""
    fam = v3["dens"] - v3["FF"][:, 0] * np.nan_to_num(v3["DF"][:, 0]) + v3["FF"][:, 0] * DENS_FRANGO
    return massa * un(PESO_LLM * lg(fam) + (1 - PESO_LLM) * lg(dens))


def crit_pond(real, est, idx=None):
    """Critério do produto com as cinco faixas de proteína real com o mesmo peso (faixa vazia fica de fora)."""
    if idx is not None:
        real, est = real[idx], est[idx]
    a = ok(real, est)
    return 100 * float(np.mean([a[f].mean() for f in faixas(real).values() if f.any()]))


def boot_cadeias(cadeia, estat, n=4000, seed=0):
    """Bootstrap por cadeia: sorteia cadeias inteiras com reposição; estat(idx) -> número ou vetor. Devolve (valor, IC 2,5%, IC 97,5%)."""
    grupos = [np.where(cadeia == c)[0] for c in np.unique(cadeia)]
    rng = np.random.default_rng(seed)
    v = np.array([estat(np.concatenate([grupos[j] for j in rng.integers(0, len(grupos), len(grupos))])) for _ in range(n)])
    return float(estat(np.arange(len(cadeia)))), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def resumo(real, est, cadeia):
    """Medidas de um estimador: ponderado e simples com IC por cadeia, por faixa, acima de 15 e de 25 g."""
    a = ok(real, est)
    fx = faixas(real)
    return dict(ponderado=boot_cadeias(cadeia, lambda i: crit_pond(real, est, i)), simples=boot_cadeias(cadeia, lambda i: 100 * a[i].mean()),
                por_faixa={k: (100 * float(a[f].mean()), int(f.sum())) for k, f in fx.items()},
                acima15=100 * float(np.mean([a[fx[k]].mean() for k in ("15-25", "25-40", ">40")])), acima25=100 * float(np.mean([a[fx[k]].mean() for k in ("25-40", ">40")])),
                mae=float(np.abs(est - real).mean()), vies=float((est - real).mean()), n=int(len(real)), cadeias=int(len(np.unique(cadeia))))


def imprime(nome, r):
    print(f"   {nome:58s} ponderado {f3(r['ponderado'])} | simples {f3(r['simples'])} | por faixa " + " / ".join(f"{v[0]:.0f}" for v in r["por_faixa"].values())
          + f" | >15 g {r['acima15']:.1f}% | >25 g {r['acima25']:.1f}% | MAE {r['mae']:.1f} g, viés {r['vies']:+.1f} g | n={r['n']}, {r['cadeias']} cadeias")


def main():
    confere, ensaio = "--confere" in sys.argv, "--ensaio" in sys.argv
    pasta = sys.argv[1]
    alvo = "teste" if ensaio else "teste2"
    os.makedirs(SAIDA, exist_ok=True)
    marca = os.path.join(SAIDA, "lido.json")
    if not confere and not ensaio and os.path.exists(marca):
        sys.exit(f"O teste2 já foi lido ({json.load(open(marca))['quando']}). A regra da seção 11 é uma leitura só; os resultados estão em {SAIDA}/avalia_teste2.json.")
    D = Dados()
    dfm0 = Ingr(D.ids); dfm0 = float(dfm0.PF[:, 0].sum() / dfm0.MF[:, 0].sum())
    assert abs(dfm0 - DENS_FRANGO) < 5e-4, dfm0
    with open(os.path.join(PREC, "n5k_teste2_amostra.csv")) as f:
        am = list(csv.DictReader(f))
    ids2 = [r["dish_id"] for r in am]; pid2 = [r["prato_id"] for r in am]
    cad2 = np.array([int(r["cadeia"]) for r in am]); indep = np.array([r["cadeia_contem"] == "n5k_v1" for r in am])
    with open(os.path.join(TD, "meta", "zen_aval.csv")) as f:
        cad_antiga = {r["dish_id"]: int(r["cadeia"]) for r in csv.DictReader(f)}
    cad_te = np.array([cad_antiga[d] for d in D.aval["teste"]["chave"]])
    assert set(cad2[~indep]) <= set(cad_te), "cadeias do teste2 e do teste em numerações diferentes"
    K = le_kaggle(pasta, D, None if ensaio else ids2)
    falta = [n for n in MASSA + FT4 if n not in K]
    assert not falta, f"faltam configurações na pasta do Kaggle: {falta}"
    H = le_cabecas(alvo)
    assert all(H[c]["ids_teste2"] == ids2 for c in CABECAS)

    # ------------------------------------------------------------ conferências sem gabarito do teste2
    print("# Conferências sem gabarito do teste2")
    Z = {}
    for n in sorted(set(MASSA + FT4 + [UNICO]) & set(K)):          # paridade com o banco da seção 10 nos 450 antigos
        b = os.path.join(TD, "resultados", "zen_banco", n + ".npz")
        if os.path.exists(b):
            z = np.load(b, allow_pickle=False)
            dm = np.concatenate([K[n][s][:, I_M] - z[s][:, I_M] for s in ANTIGOS]); dt = np.concatenate([(K[n][s] - z[s]).ravel() for s in ANTIGOS])
            Z[n] = dict(dif_log_massa_mediana=float(np.median(np.abs(dm))), dif_log_massa_max=float(np.abs(dm).max()), dif_max_22_saidas=float(np.abs(dt).max()),
                        r_log_massa=float(np.corrcoef(np.concatenate([K[n][s][:, I_M] for s in ANTIGOS]), np.concatenate([z[s][:, I_M] for s in ANTIGOS]))[0, 1]),
                        cv_rmse_logm=float(np.sqrt(np.mean((K[n]["oof"][:, I_M] - D.T[:, I_M]) ** 2))), cv_rmse_logm_s10=float(np.sqrt(np.mean((z["oof"][:, I_M] - D.T[:, I_M]) ** 2))))
            print(f"   {n:42s} nos 450 antigos, novo × seção 10: |Δ log massa| mediana {Z[n]['dif_log_massa_mediana']:.4f}, máx {Z[n]['dif_log_massa_max']:.4f}, r {Z[n]['r_log_massa']:.4f}"
                  f" | CV RMSE log-massa {Z[n]['cv_rmse_logm']:.4f} (seção 10: {Z[n]['cv_rmse_logm_s10']:.4f})")
    ref = dict(np.load(os.path.join(DENS, "v3_pratos_media.npz"), allow_pickle=False))     # pipeline da seção 10, por prato
    par = {}
    for s in ANTIGOS:
        m, d = congelado(K, H, s)
        par[s] = dict(massa_r=float(np.corrcoef(np.log(m), np.log(ref[f"{s}|massa_modelo"]))[0, 1]), massa_dif_mediana=float(np.median(np.abs(np.log(m / ref[f"{s}|massa_modelo"])))),
                      img_r=float(np.corrcoef(np.log1p(m * d), np.log1p(ref[f"{s}|img"]))[0, 1]), img_dif_rel_mediana=float(np.median(np.abs(m * d - ref[f"{s}|img"]) / np.maximum(ref[f"{s}|img"], 1))),
                      img_dentro_5pct=float(100 * np.mean(np.abs(m * d - ref[f"{s}|img"]) <= 0.05 * np.maximum(ref[f"{s}|img"], 1) + 0.25)))
        print(f"   pipeline congelado refeito × seção 10 em {s:7s}: massa r {par[s]['massa_r']:.4f}, |Δ log| mediana {par[s]['massa_dif_mediana']:.4f}"
              f" | proteína só imagem r {par[s]['img_r']:.4f}, diferença relativa mediana {100 * par[s]['img_dif_rel_mediana']:.1f}%, a menos de 5% uma da outra em {par[s]['img_dentro_5pct']:.0f}% dos pratos")
    for c in CABECAS:
        p = H[c]["paridade"]
        print(f"   cabeça de CPU refeita × banco, {c[:48]:48s}: diferença máx {p[:, 0].max():.5f} g/g, relativa mediana {100 * p[:, 1].max():.2f}%, r mínima {p[:, 2].min():.5f}")
    mo = np.exp(np.mean([K[n]["oof"][:, I_M] for n in MASSA], 0))
    print(f"   CV no treino da massa congelada (5 backbones): RMSE log {np.sqrt(np.mean((np.log(mo) - D.T[:, I_M]) ** 2)):.4f} (seção 10: 0,2051), MAPE {mape(D.Y['massa'], mo):.1f}%")
    conf = dict(paridade_configs=Z, paridade_pipeline=par)
    if not ensaio:
        tem2 = [n for n in MASSA + FT4 if "teste2" in K[n]]
        print(f"   previsões para o teste2: {len(tem2)} de {len(MASSA + FT4)} configurações necessárias" + ("" if len(tem2) == len(MASSA + FT4) else f" (faltam {sorted(set(MASSA + FT4) - set(tem2))})"))
        v3 = llm_v3(pid2, arq=V3_TESTE2, rodada="") if os.path.exists(V3_TESTE2) else None
        if v3 is None:
            print(f"   estimativas v3 do teste2: arquivo ainda não existe ({V3_TESTE2})")
        else:
            with open(V3_TESTE2, newline="", encoding="utf-8") as f:
                lin = list(csv.DictReader(f))
            som = {}
            for l in lin:
                som[(l["prato_id"], l.get("rodada") or "1")] = som.get((l["prato_id"], l.get("rodada") or "1"), 0) + float(l["parte_pct"] or 0)
            rod = sorted({k[1] for k in som})
            conf["v3"] = dict(pratos=int(v3["tem"].sum()), faltam=int((~v3["tem"]).sum()), rodadas=rod, somas_fora_de_100=int(sum(abs(x - 100) > 0.5 for x in som.values())),
                              ids_estranhos=sorted({l["prato_id"] for l in lin} - set(pid2))[:10], componentes=len(lin))
            print("   estimativas v3 do teste2:", conf["v3"])
        if len(tem2) == len(MASSA + FT4):
            m2, d2 = congelado(K, H, "teste2"); mt, dt_ = congelado(K, H, "teste")
            conf["distribuicao"] = dict(massa_mediana_teste2=float(np.median(m2)), massa_mediana_teste=float(np.median(mt)), dens_mediana_teste2=float(np.median(d2)), dens_mediana_teste=float(np.median(dt_)))
            print("   distribuição das previsões (sem gabarito):", {k: round(v, 4) for k, v in conf["distribuicao"].items()})
        pronto = len(tem2) == len(MASSA + FT4) and v3 is not None and v3["tem"].all()
        print("   PRONTO para a leitura única do teste2." if pronto else "   AINDA NÃO está pronto para pontuar o teste2.")
        json.dump(conf, open(os.path.join(SAIDA, "confere.json"), "w"), indent=1, default=float, ensure_ascii=False)
        if confere:
            return
        assert pronto, "faltam entradas: rode com --confere"

    # ------------------------------------------------------------ pontuação
    res = dict(alvo=alvo, conferencias=conf)
    if alvo == "teste2":
        json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), pasta_kaggle=os.path.abspath(pasta), concluida=False), open(marca, "w"))
        real = np.array([float(r["proteina_g"]) for r in am]); mreal = np.array([float(r["massa_g"]) for r in am]); cad = cad2
        v3a = llm_v3(pid2, arq=V3_TESTE2, rodada="")
    else:
        real, mreal, cad = D.aval["teste"]["proteina"], D.aval["teste"]["massa"], cad_te
        v3a = llm_v3(D.aval["teste"]["prato"], arq=V3_PARTES, rodada="1")
    massa, dens = congelado(K, H, alvo)
    E = {"híbrido congelado (número principal)": hibrido(massa, dens, v3a), "só imagem congelado": massa * dens,
         "relato: massa × densidade v3, sem pesos": massa * v3a["dens"]}
    if UNICO in K and alvo in K[UNICO]:
        P = proteinas(K[UNICO][alvo]); E[f"relato: modelo único ({UNICO}, rota geo)"] = np.sqrt(P["direto"] * P["mxd"])
    print(f"\n# Pontuação no {alvo}: {len(real)} pratos, {len(np.unique(cad))} cadeias; faixas (0-5 / 5-15 / 15-25 / 25-40 / >40): " + " / ".join(str(int(f.sum())) for f in faixas(real).values()))
    res["estimadores"] = {}
    for n, e in E.items():
        res["estimadores"][n] = resumo(real, e, cad); imprime(n, res["estimadores"][n])
    hib, img = E["híbrido congelado (número principal)"], E["só imagem congelado"]
    res["hibrido_menos_imagem"] = boot_cadeias(cad, lambda i: crit_pond(real, hib, i) - crit_pond(real, img, i))
    print(f"   híbrido − só imagem (ponderado, por cadeia): {res['hibrido_menos_imagem'][0]:+.1f} ({res['hibrido_menos_imagem'][1]:+.1f} a {res['hibrido_menos_imagem'][2]:+.1f})")
    p = res["estimadores"]["híbrido congelado (número principal)"]["ponderado"]
    res["decisao"] = "70% DEMONSTRADO" if p[1] >= 70 else ("70% REFUTADO" if p[2] < 70 else "70% NÃO DEMONSTRADO (compatível com 70%, sem prova)")
    print(f"\n>>> Medida principal ({alvo}, híbrido congelado, ponderado por faixa, IC por cadeia): {f3(p)}  ->  {res['decisao']}")
    # massa e densidade (relato)
    fx = faixas(real)
    res["massa"] = dict(mape=mape(mreal, massa), por_faixa={k: (mape(mreal[f], massa[f]), float(100 * (np.exp(np.median(np.log(massa[f] / mreal[f]))) - 1))) for k, f in fx.items()})
    print(f"   massa: MAPE {res['massa']['mape']:.1f}%; por faixa, MAPE (viés mediano):", {k: f"{v[0]:.1f}% ({v[1]:+.0f}%)" for k, v in res["massa"]["por_faixa"].items()})
    res["decomposicao"] = {n: resumo(real, e, cad) for n, e in {"massa real × densidade do modelo": mreal * dens, "massa real × densidade v3": mreal * v3a["dens"],
                                                               "massa do modelo × densidade real": massa * real / mreal}.items()}
    for n, r in res["decomposicao"].items():
        imprime(n, r)
    # subconjuntos e conjunto antigo. No ensaio (alvo = teste antigo) entram substitutos só para exercitar o código:
    # "independentes" = cadeias de número par; "conjunto antigo" = calib.
    if alvo == "teste2":
        ind, antigo, rot = indep, "teste", "teste antigo"
    else:
        ind, antigo, rot = cad % 2 == 0, "calib", "calib (substituto no ensaio)"
    print(f"\n## Só os {ind.sum()} pratos sem irmão no teste antigo ({len(np.unique(cad[ind]))} cadeias)")
    res["independentes"] = {n: resumo(real[ind], e[ind], cad[ind]) for n, e in E.items()}
    for n, r in res["independentes"].items():
        imprime(n, r)
    print(f"## Os {(~ind).sum()} irmãos de pratos do teste antigo")
    res["irmaos_do_teste"] = {n: resumo(real[~ind], e[~ind], cad[~ind]) for n, e in E.items()}
    for n, r in res["irmaos_do_teste"].items():
        imprime(n, r)
    # conjunto antigo com o mesmo pipeline congelado (modelos refeitos, LLM v3 rodada 1) e os dois juntos
    rt, ct = D.aval[antigo]["proteina"], np.array([cad_antiga[d] for d in D.aval[antigo]["chave"]])
    mt, dt_ = congelado(K, H, antigo); v3t = llm_v3(D.aval[antigo]["prato"], arq=V3_PARTES, rodada="1")
    Et = {"híbrido congelado (número principal)": hibrido(mt, dt_, v3t), "só imagem congelado": mt * dt_, "relato: massa × densidade v3, sem pesos": mt * v3t["dens"]}
    print(f"\n## {rot} com o mesmo pipeline congelado (modelos refeitos; LLM v3 rodada 1) — releitura, só relato")
    res["teste_antigo"] = {n: resumo(rt, e, ct) for n, e in Et.items()}
    for n, r in res["teste_antigo"].items():
        imprime(n, r)
    print(f"## {antigo} + {alvo} juntos (por faixa com os pratos dos dois; IC por cadeia sobre a união)")
    rj, cj = np.r_[rt, real], np.r_[ct, cad]
    res["juntos"] = {n: resumo(rj, np.r_[Et[n], E[n]], cj) for n in Et}
    for n, r in res["juntos"].items():
        imprime(n, r)
    if alvo == "teste2":
        np.savez(os.path.join(SAIDA, "pratos_teste2.npz"), ids=np.array(pid2), real=real, massa_real=mreal, cadeia=cad, independente=indep, massa=massa, dens=dens,
                 dens_v3=v3a["dens"], fonte_v3=v3a["fonte"], **{f"est{i}": e for i, e in enumerate(E.values())}, nomes=np.array(list(E)))
    json.dump(res, open(os.path.join(SAIDA, f"avalia_{alvo}{'_ensaio' if ensaio else ''}.json"), "w"), indent=1, default=float, ensure_ascii=False)
    if alvo == "teste2":
        json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), pasta_kaggle=os.path.abspath(pasta), concluida=True), open(marca, "w"))


if __name__ == "__main__":
    main()
