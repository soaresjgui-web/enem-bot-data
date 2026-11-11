#!/usr/bin/env python3
"""Traz para o banco de previsões a saída de treino_gpu.py (Kaggle/Colab) e roda a avaliação da seção 8.

Uso: python3 eval/treino/kaggle/ingere_kaggle.py <pasta baixada com `kaggle kernels output`>
         [--ensaio DIR] [--so-confere] [--referencia PASTA] [--substitui]

Saída SEM teste2 (execução 1, seção 10):
  1. Lê manifesto.json e cada pred__*.npz da pasta. Confere que as listas usadas no Kaggle são as desta pasta
     (hash), que elas batem com <TD>/meta/zen_treino.csv, zen_aval.csv e com os folds de zen_comum.Dados, que cada
     previsão está na ordem certa (ids gravados no próprio arquivo) e que não falta nenhum prato.
  2. Copia cada configuração completa para <TD>/resultados/zen_banco/<nome>.npz, no esquema do banco
     (oof, calib, teste, n5k_v1 em log; as chaves ufpr/liv_up/seara não existem: o Kaggle não vê os itens brasileiros).
  3. Soma os conjuntos pré-declarados de conjuntos_gpu.json a zen_avalia.CONJUNTOS e roda zen_avalia.main(), sem
     mudar nenhuma regra: escolha por CV no treino, pesos e forma do híbrido no calib, teste só para relatar.
  4. Imprime o que mudou: linhas das configurações de GPU, escolhas antes e depois, números principais.

Saída COM teste2 (execução 2, seção 11) — nada aqui lê gabarito de avaliação (nem do teste, nem do teste2):
  1. As mesmas conferências, mais a lista do teste2 (hash, ordem, nenhum prato sem previsão, interseção vazia com o
     treino e com os 450).
  2. Paridade sem gabarito contra a execução 1 (por padrão, as entradas de mesmo nome em <TD>/resultados/zen_banco;
     --referencia aponta outra pasta, de banco ou de saída do Kaggle): por configuração, correlação e diferença
     absoluta mediana das previsões de log-massa e de log-proteína nos 450 pratos antigos e no fora-do-fold do
     treino, e a CV no treino (que só usa rótulos do treino). O mesmo para a massa média dos cinco backbones e para a
     densidade média dos quatro ajustes finos do pipeline congelado (11.2).
  3. Grava cada configuração em <TD>/resultados/zen_banco_r2/<nome>.npz: oof, calib, teste, n5k_v1, teste2 (em log)
     + ids_teste2 e pratos_teste2 (ordem de listas/aval2_ids.csv). O banco da seção 10 (zen_banco) NÃO é alterado e
     zen_avalia.py NÃO roda. Com --substitui, as entradas do zen_banco são trocadas pelas novas (as antigas vão para
     zen_banco_antes_r2/) e a avaliação da seção 8 roda como no caso sem teste2 (isso lê o `teste` antigo de novo).

--ensaio DIR  não toca em nada de <TD>/resultados: grava e avalia em DIR. É obrigatório para saídas de --smoke.
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
    for s in L.extras:      # teste2: fora do treino e fora dos 450 (Dados já confere; aqui também contra a CPU)
        assert not set(L.conj[s]) & (set(D.ids) | {d for x in SPLITS for d in D.aval[x]["chave"]}), f"{s} cruza com treino ou avaliação"
    return L


def le_saida(pasta, L, permite_incompleto):
    man = sorted(glob.glob(os.path.join(pasta, "**", "manifesto.json"), recursive=True))
    if not man:
        raise SystemExit(f"manifesto.json não encontrado em {pasta}")
    raiz = os.path.dirname(man[0])
    M = json.load(open(man[0]))
    if M.get("formato") not in G.FORMATOS_ACEITOS:
        raise SystemExit(f"formato {M.get('formato')} diferente dos aceitos ({G.FORMATOS_ACEITOS})")
    if M["listas_hash"] != L.hash:
        raise SystemExit("o Kaggle rodou com listas diferentes das desta pasta: as previsões não são comparáveis")
    if M.get("listas2_hash") and M["listas2_hash"] != L.hash2:
        raise SystemExit("o Kaggle rodou com outra lista do teste2 (aval2_ids.csv): as previsões não são comparáveis")
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
            extras = [s for s in L.extras if s in z.files]
            for s in extras:
                assert list(z[f"ids_{s}"]) == L.conj[s], f"ordem de {s}"
                assert z[s].shape == (len(L.conj[s]), L.N), f"formato de {s}"
            falta = int(np.isnan(z["oof"]).any(1).sum()) + sum(int(np.isnan(z[s]).any(1).sum()) for s in SPLITS + extras)
            assert permite_incompleto or falta == 0, f"{falta} pratos sem previsão"
            assert np.isfinite(z["oof"][~np.isnan(z["oof"][:, 0])]).all(), "valores não finitos"
            estado = M["configs"].get(nome, {}).get("estado")
            assert estado == "completa", f"estado no manifesto: {estado}"
        except AssertionError as e:
            problemas.append(f"{nome}: {e}")
            continue
        entradas[nome] = {k: z[k].astype(np.float64) for k in ["oof"] + SPLITS + extras}
    return M, entradas, problemas


def carrega_referencia(pasta, nomes):
    """Previsões da execução 1 para as mesmas configurações: pasta de banco (<nome>.npz) ou saída do Kaggle (pred__*.npz)."""
    ref = {}
    if glob.glob(os.path.join(pasta, "**", "manifesto.json"), recursive=True):
        for arq in glob.glob(os.path.join(pasta, "**", "pred__*.npz"), recursive=True):
            z = np.load(arq, allow_pickle=False)
            if str(z["nome"]) in nomes:
                ref[str(z["nome"])] = {k: z[k].astype(np.float64) for k in ["oof"] + SPLITS}
    else:
        for n in nomes:
            arq = os.path.join(pasta, n + ".npz")
            if os.path.exists(arq):
                z = np.load(arq, allow_pickle=False)
                ref[n] = {k: z[k].astype(np.float64) for k in ["oof"] + SPLITS}
    return ref


def _par(a, b):
    """Correlação e diferença absoluta (mediana e máxima) entre dois vetores de previsões em log; ignora NaN."""
    v = ~(np.isnan(a) | np.isnan(b))
    if v.sum() < 3:
        return dict(n=int(v.sum()), corr=None, dif_mediana=None, dif_max=None)
    return dict(n=int(v.sum()), corr=float(np.corrcoef(a[v], b[v])[0, 1]), dif_mediana=float(np.median(np.abs(a[v] - b[v]))), dif_max=float(np.abs(a[v] - b[v]).max()))


def rodada2(a, L, M, entradas, problemas, res_real):
    """Saída com teste2: paridade sem gabarito contra a execução 1 e gravação em zen_banco_r2 (não altera zen_banco)."""
    extras = L.extras
    ref_pasta = a.referencia or os.path.join(res_real, "zen_banco")
    ref = carrega_referencia(ref_pasta, set(entradas))
    junta450 = lambda e, col: np.concatenate([e[s][:, col] for s in SPLITS])
    fmt = lambda p: "   sem dados   " if p["corr"] is None else f"{p['corr']:.4f} {p['dif_mediana']:.4f}"
    print(f"\n{'=' * 30} paridade sem gabarito: execução 2 × execução 1 ({ref_pasta}) {'=' * 30}")
    print("Por configuração: correlação e |diferença| mediana, em log, entre as previsões novas e as antigas. CV = no treino (rótulos só do treino).")
    print(f"{'configuração':40s} | 450: massa      proteína    densidade   | fora-do-fold: massa  proteína  | CV RMSE log-massa ex.1 -> ex.2 | CV critério direto ex.1 -> ex.2")
    par, avisos = {}, []
    for n, e in entradas.items():
        cv2 = G.resumo_cv(L, e["oof"].astype(np.float32))
        if n not in ref:
            print(f"{n[:40]:40s} | sem referência da execução 1 | CV RMSE {cv2.get('rmse_logm', float('nan')):.4f}, critério direto {cv2.get('crit_direto', float('nan')):.1f}%")
            par[n] = dict(referencia=False, cv_r2=cv2)
            continue
        r = ref[n]
        cv1 = G.resumo_cv(L, r["oof"].astype(np.float32))
        p = dict(referencia=True, cv_r1=cv1, cv_r2=cv2,
                 massa_450=_par(junta450(e, G.I_M), junta450(r, G.I_M)), proteina_450=_par(junta450(e, G.I_P), junta450(r, G.I_P)),
                 densidade_450=_par(junta450(e, G.I_D), junta450(r, G.I_D)),
                 massa_oof=_par(e["oof"][:, G.I_M], r["oof"][:, G.I_M]), proteina_oof=_par(e["oof"][:, G.I_P], r["oof"][:, G.I_P]),
                 por_conjunto={s: dict(massa=_par(e[s][:, G.I_M], r[s][:, G.I_M]), proteina=_par(e[s][:, G.I_P], r[s][:, G.I_P])) for s in SPLITS})
        par[n] = p
        print(f"{n[:40]:40s} | {fmt(p['massa_450'])}  {fmt(p['proteina_450'])}  {fmt(p['densidade_450'])} | {fmt(p['massa_oof'])}  {fmt(p['proteina_oof'])} | "
              f"{cv1.get('rmse_logm', float('nan')):.4f} -> {cv2.get('rmse_logm', float('nan')):.4f}               | {cv1.get('crit_direto', float('nan')):.1f} -> {cv2.get('crit_direto', float('nan')):.1f}%")
        # ridge sobre backbone congelado é determinístico: no mesmo ambiente a diferença é ~0. Para comparar, trocar o
        # redimensionamento da GPU pelo da CPU (PIL) já dá |dif| mediana de 0,005 na massa.
        if " ridge " in n and p["massa_450"]["corr"] is not None and (p["massa_450"]["corr"] < 0.9995 or p["massa_450"]["dif_mediana"] > 0.002):
            avisos.append(f"{n}: ridge sobre backbone congelado deveria repetir a execução 1 quase exatamente (corr {p['massa_450']['corr']:.4f}, "
                          f"|dif| mediana {p['massa_450']['dif_mediana']:.4f}); o ambiente (torch, timm, pesos) pode ter mudado")
    # os dois agregados que o pipeline congelado (11.2) usa
    agregados = {"massa: média dos 5 backbones congelados (log-massa)": (G.PIPELINES["massa_5_backbones"], G.I_M),
                 "densidade: média dos 4 ajustes finos (log-densidade)": ([n for n in G.PIPELINES["imagem_9_modelos"] if " ft " in n], G.I_D)}
    par["agregados"] = {}
    for rot, (membros, col) in agregados.items():
        if all(m in entradas and m in ref for m in membros):
            m2 = np.mean([junta450(entradas[m], col) for m in membros], 0); m1 = np.mean([junta450(ref[m], col) for m in membros], 0)
            par["agregados"][rot] = _par(m2, m1)
            print(f"{rot[:60]:60s} nos 450: corr {par['agregados'][rot]['corr']:.4f}, |dif| mediana {par['agregados'][rot]['dif_mediana']:.4f}, máxima {par['agregados'][rot]['dif_max']:.3f}")
        else:
            print(f"{rot[:60]:60s} nos 450: faltam membros ({[m for m in membros if m not in entradas or m not in ref]})")
    for s in extras:      # só descrição das previsões (sem gabarito): pratos, valores não finitos, medianas previstas
        print(f"\n{s}: {len(L.conj[s])} pratos por configuração; mediana prevista de massa / proteína direta (g), ao lado da dos 450:")
        for n, e in entradas.items():
            if s in e:
                print(f"  {n[:44]:44s} {s}: {np.nanmedian(np.exp(e[s][:, G.I_M])):6.1f} / {np.nanmedian(np.expm1(e[s][:, G.I_P])):5.1f}   450: "
                      f"{np.nanmedian(np.exp(junta450(e, G.I_M))):6.1f} / {np.nanmedian(np.expm1(junta450(e, G.I_P))):5.1f}   sem previsão: {int(np.isnan(e[s]).any(1).sum())}")
    for x in avisos:
        print("ATENÇÃO:", x)
    if a.so_confere:
        return print("conferências ok (nada gravado)")
    res = os.path.join(os.path.abspath(a.ensaio), "resultados") if a.ensaio else res_real
    dest = os.path.join(res, "zen_banco_r2")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(AQUI, "listas", "aval2_ids.csv")) as f:
        import csv
        linhas = list(csv.DictReader(f))
    for n, e in entradas.items():
        ex = {}
        for s in extras:
            if s in e:
                ex[f"ids_{s}"] = np.array(L.conj[s]); ex[f"pratos_{s}"] = np.array([r["prato_id"] for r in linhas if r["split"] == s])
        np.savez(os.path.join(dest, n + ".npz"), **e, **ex)
    json.dump(dict(pasta=os.path.abspath(a.pasta), referencia=ref_pasta,
                   manifesto={k: M.get(k) for k in ("formato", "rodada", "inicio", "duracao_min", "gpus", "torch", "timm", "listas_hash", "listas2_hash", "tamanho_saida")},
                   ingeridas=sorted(entradas), recusadas=problemas, paridade=par, avisos=avisos),
              open(os.path.join(res, "ingestao_kaggle_r2.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n{len(entradas)} configurações gravadas em {dest} (oof, calib, teste, n5k_v1, {', '.join(extras)}; ids na ordem de aval2_ids.csv). "
          f"O banco da seção 10 não foi alterado e nenhum gabarito de avaliação foi lido. Paridade em {os.path.join(res, 'ingestao_kaggle_r2.json')}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pasta")
    ap.add_argument("--ensaio", help="pasta de trabalho isolada: não altera o banco nem zen_avalia.json")
    ap.add_argument("--so-confere", action="store_true")
    ap.add_argument("--referencia", help="execução 1 para a paridade: pasta de banco ou de saída do Kaggle (padrão: <TD>/resultados/zen_banco)")
    ap.add_argument("--substitui", action="store_true", help="saída com teste2: troca também as entradas do zen_banco e roda zen_avalia (lê o teste antigo)")
    a = ap.parse_args()
    D = zen_comum.Dados()
    L = confere_listas(D)
    print(f"listas: {len(L.ids)} pratos de treino, 5 folds, {sum(len(v) for v in L.aval.values())} de avaliação; iguais às da CPU (hash {L.hash})"
          + "".join(f"; {s}: {len(L.conj[s])} pratos, fora do treino e dos 450 (hash {L.hash2})" for s in L.extras))
    M, entradas, problemas = le_saida(a.pasta, L, permite_incompleto=bool(a.ensaio))
    print(f"saída do Kaggle: {M['formato']}, rodada {M.get('rodada', '1')}, {M['inicio']}, {M.get('duracao_min')} min, GPUs {M['gpus'] or 'nenhuma'}, "
          f"torch {M.get('torch')}, timm {M.get('timm')}, smoke={M['smoke']}")
    for nome, c in M["configs"].items():
        print(f"  {c['estado']:14s} {nome:46s} {'-> banco' if nome in entradas else ''}")
    for p in problemas:
        print("  RECUSADA:", p)
    if M["smoke"] and not a.ensaio:
        raise SystemExit("Saída de --smoke: use --ensaio DIR (não entra no banco de verdade).")
    td_real = zen_comum.TD
    res_real = os.path.join(td_real, "resultados")
    com_extra = bool(entradas) and all(any(s in e for s in L.extras) for e in entradas.values())
    if com_extra:       # execução 2: paridade sem gabarito + zen_banco_r2; o banco da seção 10 só muda com --substitui
        rodada2(a, L, M, entradas, problemas, res_real)
        if not a.substitui or a.so_confere:
            return
        print("\n--substitui: as entradas do zen_banco passam a ser as da execução 2 e a avaliação da seção 8 roda de novo (lê o `teste` antigo).")
    if a.so_confere or not entradas:
        return print("nada a ingerir" if not entradas else "conferências ok")

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
        velho = os.path.join(banco, nome + ".npz")
        if com_extra and not a.ensaio and os.path.exists(velho) and not os.path.islink(velho):   # guarda a entrada da execução 1
            os.makedirs(os.path.join(res, "zen_banco_antes_r2"), exist_ok=True)
            if not os.path.exists(os.path.join(res, "zen_banco_antes_r2", nome + ".npz")):
                shutil.copy(velho, os.path.join(res, "zen_banco_antes_r2", nome + ".npz"))
        if os.path.islink(velho):       # ensaio: o link aponta para o banco de verdade; não escrever através dele
            os.remove(velho)
        np.savez(velho, **e)
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
