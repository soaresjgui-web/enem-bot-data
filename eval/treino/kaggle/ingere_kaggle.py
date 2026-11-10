#!/usr/bin/env python3
"""Traz para o banco de previsões a saída de treino_gpu.py (Kaggle/Colab) e roda a avaliação da seção 8.

Uso: python3 eval/treino/kaggle/ingere_kaggle.py <pasta baixada com `kaggle kernels output`> [--ensaio DIR] [--so-confere]

O que faz:
  1. Lê manifesto.json e cada pred__*.npz da pasta. Confere que as listas usadas no Kaggle são as desta pasta
     (hash), que elas batem com <TD>/meta/zen_treino.csv, zen_aval.csv e com os folds de zen_comum.Dados, que cada
     previsão está na ordem certa (ids gravados no próprio arquivo) e que não falta nenhum prato.
  2. Copia cada configuração completa para <TD>/resultados/zen_banco/<nome>.npz, no esquema do banco
     (oof, calib, teste, n5k_v1 em log; as chaves ufpr/liv_up/seara não existem: o Kaggle não vê os itens brasileiros).
  3. Soma os conjuntos pré-declarados de conjuntos_gpu.json a zen_avalia.CONJUNTOS e roda zen_avalia.main(), sem
     mudar nenhuma regra: escolha por CV no treino, pesos e forma do híbrido no calib, teste só para relatar.
  4. Imprime o que mudou: linhas das configurações de GPU, escolhas antes e depois, números principais.

--ensaio DIR  não toca no banco nem em zen_avalia.json: monta uma cópia (links) do banco em DIR e avalia ali.
              É obrigatório para saídas de --smoke.
--so-confere  só as conferências do passo 1.
"""
import argparse, glob, json, os, shutil, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, ".."))
import treino_gpu as G  # noqa: E402  (só numpy no nível do módulo)
import zen_comum  # noqa: E402

SPLITS = G.SPLITS


def confere_listas(D):
    """As listas enviadas ao Kaggle são exatamente o treino, os folds e a avaliação da CPU."""
    L = G.Dados(os.path.join(AQUI, "listas"))
    assert L.ids == D.ids, "listas/treino.csv não bate com <TD>/meta/zen_treino.csv (rode monta_pacote.py listas)"
    assert np.array_equal(L.fold, D.fold), "folds das listas diferentes dos de zen_comum.Dados"
    assert np.allclose(L.T, D.T, atol=1e-9) and np.allclose(L.w_cv, D.w_cv), "alvos das listas diferentes dos da CPU"
    for s in SPLITS:
        assert L.aval[s] == D.aval[s]["chave"], f"ids de avaliação ({s}) diferentes"
    assert not set(L.ids) & {d for s in SPLITS for d in L.aval[s]}
    return L


def le_saida(pasta, L, permite_incompleto):
    man = sorted(glob.glob(os.path.join(pasta, "**", "manifesto.json"), recursive=True))
    if not man:
        raise SystemExit(f"manifesto.json não encontrado em {pasta}")
    raiz = os.path.dirname(man[0])
    M = json.load(open(man[0]))
    if M.get("formato") != G.FORMATO:
        raise SystemExit(f"formato {M.get('formato')} diferente do esperado ({G.FORMATO})")
    if M["listas_hash"] != L.hash:
        raise SystemExit("o Kaggle rodou com listas diferentes das desta pasta: as previsões não são comparáveis")
    entradas, problemas = {}, []
    for arq in sorted(glob.glob(os.path.join(raiz, "pred__*.npz"))):
        z = np.load(arq, allow_pickle=False)
        nome = str(z["nome"])
        try:
            assert str(z["listas_hash"]) == L.hash, "hash das listas"
            assert list(z["ids_treino"]) == L.ids and np.array_equal(z["fold"], L.fold), "ordem do treino ou folds"
            assert z["oof"].shape == (len(L.ids), L.N), "formato de oof"
            for s in SPLITS:
                assert list(z[f"ids_{s}"]) == L.aval[s], f"ordem de {s}"
                assert z[s].shape == (len(L.aval[s]), L.N), f"formato de {s}"
            falta = int(np.isnan(z["oof"]).any(1).sum()) + sum(int(np.isnan(z[s]).any(1).sum()) for s in SPLITS)
            assert permite_incompleto or falta == 0, f"{falta} pratos sem previsão"
            assert np.isfinite(z["oof"][~np.isnan(z["oof"][:, 0])]).all(), "valores não finitos"
            estado = M["configs"].get(nome, {}).get("estado")
            assert estado == "completa", f"estado no manifesto: {estado}"
        except AssertionError as e:
            problemas.append(f"{nome}: {e}")
            continue
        entradas[nome] = {k: z[k].astype(np.float64) for k in ["oof"] + SPLITS}
    return M, entradas, problemas


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pasta")
    ap.add_argument("--ensaio", help="pasta de trabalho isolada: não altera o banco nem zen_avalia.json")
    ap.add_argument("--so-confere", action="store_true")
    a = ap.parse_args()
    D = zen_comum.Dados()
    L = confere_listas(D)
    print(f"listas: {len(L.ids)} pratos de treino, 5 folds, {sum(len(v) for v in L.aval.values())} de avaliação; iguais às da CPU (hash {L.hash})")
    M, entradas, problemas = le_saida(a.pasta, L, permite_incompleto=bool(a.ensaio))
    print(f"saída do Kaggle: {M['inicio']}, {M.get('duracao_min')} min, GPUs {M['gpus'] or 'nenhuma'}, smoke={M['smoke']}")
    for nome, c in M["configs"].items():
        print(f"  {c['estado']:14s} {nome:46s} {'-> banco' if nome in entradas else ''}")
    for p in problemas:
        print("  RECUSADA:", p)
    if M["smoke"] and not a.ensaio:
        raise SystemExit("Saída de --smoke: use --ensaio DIR (não entra no banco de verdade).")
    if a.so_confere or not entradas:
        return print("nada a ingerir" if not entradas else "conferências ok")

    td_real = zen_comum.TD
    res_real = os.path.join(td_real, "resultados")
    if a.ensaio:   # cópia por links do banco + linha de base; a avaliação grava tudo aqui
        res = os.path.join(os.path.abspath(a.ensaio), "resultados"); banco = os.path.join(res, "zen_banco")
        os.makedirs(banco, exist_ok=True)
        for f in glob.glob(os.path.join(res_real, "zen_banco", "*.npz")):
            dest = os.path.join(banco, os.path.basename(f))
            if not os.path.lexists(dest):
                os.symlink(f, dest)
        if not os.path.lexists(os.path.join(res, "zen_base2d.npz")):
            os.symlink(os.path.join(res_real, "zen_base2d.npz"), os.path.join(res, "zen_base2d.npz"))
    else:
        res, banco = res_real, os.path.join(res_real, "zen_banco")
    antes = None
    if os.path.exists(os.path.join(res_real, "zen_avalia.json")):
        antes = json.load(open(os.path.join(res_real, "zen_avalia.json")))
        if not a.ensaio and not os.path.exists(os.path.join(res, "zen_avalia.antes_kaggle.json")):
            shutil.copy(os.path.join(res, "zen_avalia.json"), os.path.join(res, "zen_avalia.antes_kaggle.json"))
    for nome, e in entradas.items():
        np.savez(os.path.join(banco, nome + ".npz"), **e)
    conj = json.load(open(os.path.join(AQUI, "conjuntos_gpu.json")))["conjuntos"]
    json.dump(dict(pasta=os.path.abspath(a.pasta), manifesto={k: M[k] for k in ("formato", "inicio", "duracao_min", "gpus", "torch", "listas_hash", "melhor_modelo")},
                   ingeridas={n: M["configs"][n]["cv_treino"] for n in entradas}, recusadas=problemas, conjuntos_declarados=conj),
              open(os.path.join(res, "ingestao_kaggle.json"), "w"), indent=1, ensure_ascii=False)

    # avaliação existente, sem mudar regra: só entram mais configurações e os conjuntos pré-declarados
    import zen_avalia
    novos = {k: v for k, v in conj.items() if k not in zen_avalia.CONJUNTOS}
    zen_avalia.CONJUNTOS.update(novos)
    zen_avalia.TD = os.path.dirname(res)
    zen_avalia.BANCO = banco
    print("\n" + "=" * 30 + " zen_avalia.py " + "=" * 30, flush=True)
    zen_avalia.main()
    R = json.load(open(os.path.join(res, "zen_avalia.json")))
    tab = R["tabela"]

    print("\n" + "=" * 30 + " resumo da ingestão " + "=" * 30)
    print("Configurações de GPU e conjuntos que as usam (CV escolhe; teste só relata). * = CV incompleta, fora da escolha")
    print(f"{'configuração':78s} | CV RMSE logm | CV crit dir/mxd/geo | teste dir/mxd/geo  | MAPE m | m×dLLM")
    for n in sorted((n for n in tab if "gpu" in n or n.startswith("smoke")), key=lambda n: -tab[n]["cv"]["rmse_logm"]):
        c, t = tab[n]["cv"], tab[n]["teste"]
        print(f"{n[:78]:78s} | {c['rmse_logm']:.4f}{'*' if c['n_cv'] < len(D.ids) else ' '}      | {c['direto']:5.1f} {c['mxd']:5.1f} {c['geo']:5.1f}   | "
              f"{t['direto']:5.1f} {t['mxd']:5.1f} {t['geo']:5.1f}  | {t['mape']:5.1f}% | {t['hib']:5.1f}")
    # a CV calculada no Kaggle (só com as listas) tem de ser a mesma que zen_avalia calcula aqui
    difere = []
    for n in entradas:
        k, c = M["configs"][n]["cv_treino"], tab[n]["cv"]
        if max(abs(k["rmse_logm"] - c["rmse_logm"]), abs(k["crit_direto"] - c["direto"]) / 100, abs(k["crit_mxd"] - c["mxd"]) / 100) > 2e-3:
            difere.append(f"'{n}': Kaggle {k['rmse_logm']:.4f} / {k['crit_direto']:.1f} / {k['crit_mxd']:.1f} × aqui {c['rmse_logm']:.4f} / {c['direto']:.1f} / {c['mxd']:.1f}")
    print("ATENÇÃO, CV do Kaggle diferente da recalculada aqui: " + "; ".join(difere) if difere else
          f"CV recalculada aqui = CV registrada no Kaggle nas {len(entradas)} configurações ingeridas (alinhamento de pratos e folds conferido).")
    unicas = [n for n in entradas if tab[n]["cv"]["n_cv"] == len(D.ids)]
    if unicas:   # modelo único de GPU (um backbone, exportável): escolhido pela CV entre as configurações ingeridas
        rotas = ("direto", "mxd", "geo")
        m = max(unicas, key=lambda n: max(tab[n]["cv"][r] for r in rotas)); r = max(rotas, key=lambda r: tab[m]["cv"][r])
        print(f"Melhor modelo ÚNICO de GPU pela CV: '{m}', rota {r}: CV {tab[m]['cv'][r]:.1f}% -> teste {tab[m]['teste'][r]:.1f}% (MAPE de massa {tab[m]['teste']['mape']:.1f}%)")
    e = R["escolha"]
    if antes:
        ea = antes["escolha"]
        for k, rot in (("massa", "modelo de massa"), ("imagem", "só imagem"), ("hibrido", "forma do híbrido")):
            print(f"Escolha ({rot}): {'MUDOU' if ea[k] != e[k] else 'igual'}: '{ea[k]}'" + (f" -> '{e[k]}'" if ea[k] != e[k] else ""))
        print(f"Só imagem escolhido, teste: {antes['imagem']['teste']:.1f}% -> {R['imagem']['teste']:.1f}% (IC95% {R['imagem']['ic'][0]:.1f} a {R['imagem']['ic'][1]:.1f}); "
              f"vs 2d {R['imagem']['vs_2d_direto'][0]:+.1f} ({R['imagem']['vs_2d_direto'][1]:+.1f} a {R['imagem']['vs_2d_direto'][2]:+.1f})")
        h0, h1 = antes["hibrido"][ea["hibrido"]], R["hibrido"][e["hibrido"]]
        print(f"Híbrido escolhido, teste: {h0['teste']:.1f}% -> {h1['teste']:.1f}% (IC95% {h1['ic_abs'][0]:.1f} a {h1['ic_abs'][1]:.1f}); "
              f"vs híbrido 2d {h1['vs_2d_hibrido'][0]:+.1f} ({h1['vs_2d_hibrido'][1]:+.1f} a {h1['vs_2d_hibrido'][2]:+.1f})")
    print(f"Resultados em {os.path.join(res, 'zen_avalia.json')}" + ("" if a.ensaio else f" (anterior em zen_avalia.antes_kaggle.json); banco: {banco}"))


if __name__ == "__main__":
    main()
