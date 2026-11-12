#!/usr/bin/env python3
"""Traz para o banco de previsões a saída de treino_gpu.py (Kaggle/Colab) e roda a avaliação da seção 8.

Uso: python3 eval/treino/kaggle/ingere_kaggle.py <pasta baixada com `kaggle kernels output`>
         [--ensaio DIR] [--so-confere] [--referencia PASTA] [--substitui]

Saída da execução 4 (seção 13; treino_v2 + bases externas) — nada aqui lê rótulo de avaliação, nada é pontuado e nada é
escolhido (a regra de 13.9 roda depois, na CPU):
  1. Monta as listas exatamente como o pacote do Kaggle (monta_pacote.conteudo: versões enxutas, sem rótulo nas partes
     de registro) e confere hash de treino_v2 e do teste2, e por base o hash da lista e da tabela de divisão.
  2. Em cada pred__*.npz: ids e ordem do treino e do teste2, folds, as três variantes de orientações, frações; para cada
     base pronta, extcv__ (fotos de treino, na ordem da lista, com o fold em que validam) e extreg__ (registro, registro
     misto, avaliação fraca do SNAPMe), completas e finitas; a procedência (`so_comercial_ok` como em 13.8).
  3. Paridade com o D1 da execução 3 (13.7), refeita aqui a partir de R0 e H0 (ref_d1_r3.csv); se falhar, nada é gravado.
  4. Sem --so-confere: copia cada configuração completa para <TD>/resultados/zen_banco_r4/<nome>.npz e o resumo em
     ingestao_kaggle_r4.json (estado, aceitação dos ajustes finos de 13.7, procedência, o que faltou e por quê).
  As listas das bases fechadas (OmniFood8K, FPB, SNAPMe) e as tabelas de divisão delas são procuradas como no
  monta_pacote.py (listas/, --divisoes, NUTRELLE_SCRATCH); sem elas, essas bases não são conferidas e o script avisa.

Saída da execução 3 (seção 12.3; treino_v2, só teste2) — nada aqui lê gabarito de avaliação e nada é escolhido:
  1. Confere as listas (hash de treino_v2.csv e de aval2_ids.csv), e em cada pred__*.npz: ids e ordem do treino e do
     teste2, folds, formatos, nenhuma previsão faltando nem valor não finito, as três variantes de orientações, as
     frações (somam 1; a densidade pelas frações bate com frações × dens_fonte do fold), os alvos aninhados do professor
     (NaN exatamente nos pratos do próprio fold), nenhuma previsão para os 450 pratos da avaliação antiga.
  2. Paridade das features dos cinco backbones com a execução 2: cada `gpu3 ridge` traz as previsões no teste2 de uma
     réplica do ridge da execução 2 (lista e folds antigos, features desta execução); elas têm de repetir as do
     `gpu ridge` correspondente (--referencia = pasta da saída da execução 2; padrão: <TD>/resultados/zen_banco_r2).
  3. Base externa (12.5), se a execução usou: a lista externa do manifesto é a desta pasta (hash), as amostras
     externas não têm id do Nutrition5k, cada configuração `+<base>` foi treinada com essa lista; mostra a decisão de
     adoção gravada pelo kernel (decisao_<base>.json), quais versões concorrem em 12.3 (as adotadas) e o relato
     secundário de massa no teste_ext (relato_externo.json; não escolhe nada).
  4. Sem --so-confere: copia cada configuração completa para <TD>/resultados/zen_banco_r3/<nome>.npz (todas as chaves).
     A regra de 12.3 (rota, orientações, C1, C2) é aplicada pelo agente de avaliação a partir desses arquivos.

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
import argparse, csv, glob, json, os, shutil, sys
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


def rodada3(a, M, raiz):
    """Conferências da saída da execução 3 (ver o cabeçalho). Sem gabarito de avaliação."""
    L = G.Dados(os.path.join(AQUI, "listas"), rodada="3")
    if M.get("formato") not in G.FORMATOS_ACEITOS or M["listas_hash"] != L.hash or M.get("listas2_hash") != L.hash2:
        raise SystemExit(f"listas ou formato diferentes dos desta pasta (manifesto: {M.get('formato')}, {M['listas_hash']}, {M.get('listas2_hash')}; "
                         f"aqui: {G.FORMATO}, {L.hash}, {L.hash2})")
    import csv
    cad = {}
    for r in csv.DictReader(open(os.path.join(AQUI, "listas", "treino_v2.csv"))):
        assert cad.setdefault(r["cadeia"], r["fold"]) == r["fold"], "cadeia em dois folds"
    n, n2, N, K = len(L.ids), len(L.conj["teste2"]), L.N, len(L.fontes)
    print(f"listas: treino_v2 {n} pratos em {len(cad)} cadeias, 5 folds (hash {L.hash}); teste2 {n2} pratos, fora do treino (hash {L.hash2})")
    print(f"saída do Kaggle: {M['formato']}, rodada {M['rodada']}, {M['inicio']}, {M.get('duracao_min')} min, GPUs {M['gpus'] or 'nenhuma'}, "
          f"torch {M.get('torch')}, timm {M.get('timm')}, smoke={M['smoke']}, externos={M.get('externo')}")
    cfgs = {c["nome"]: c for c in G.todas_as_configs()}
    parcial = bool(M["smoke"])
    # base externa (12.5): a lista usada no Kaggle é a desta pasta? (no --smoke a lista é um recorte: só se mostra)
    ME = M.get("externo") or {}
    base, Lx = ME.get("nome"), None
    if base:
        arq_x = os.path.join(AQUI, "listas", G.EXTERNOS[base]["lista"])
        Lx = G.Dados(os.path.join(AQUI, "listas"), rodada="3", externo=arq_x, externo_pesos=G.EXTERNOS[base]["pesos"]).ext
        if not parcial and ME.get("lista") != Lx["hash"]:
            raise SystemExit(f"a lista externa do Kaggle ({ME.get('lista')}) não é a desta pasta ({Lx['hash']})")
        n5k = set(L.ids) | set(L.conj["teste2"]) | set(L.antigo)
        assert not [i for i in Lx["ids"] if i.split(":")[-1] in n5k or i.split(":")[-1].startswith("dish_")], "prato do Nutrition5k na lista externa"
        print(f"base externa {base}: {ME.get('situacao')} | lista {ME.get('lista')}{'' if parcial else ' (igual à desta pasta)'}, {ME.get('resumo')}, "
              f"teste_ext por {ME.get('regra_teste_ext')}, pesos {ME.get('pesos')}, teto {ME.get('teto')}")
    elif ME.get("situacao"):
        print(f"base externa: {ME['situacao']}")
    sem_suf = lambda nome: next((nome[:-len(f" +{x}")] for x in G.EXTERNOS if nome.endswith(f" +{x}")), nome)
    if parcial and not (a.ensaio or a.so_confere):
        raise SystemExit("Saída de --smoke: use --so-confere ou --ensaio DIR.")
    entradas, problemas = {}, []
    for arq in sorted(glob.glob(os.path.join(raiz, "pred__*.npz"))):
        z = dict(np.load(arq, allow_pickle=False))
        nome = str(z["nome"]); c = cfgs.get(sem_suf(nome), {})
        try:
            assert str(z["listas_hash"]) == L.hash and str(z["listas2_hash"]) == L.hash2, "hash das listas"
            assert list(z["ids_treino"]) == L.ids and np.array_equal(z["fold"], L.fold), "ordem do treino ou folds"
            assert list(z["ids_teste2"]) == L.conj["teste2"], "ordem do teste2"
            assert not any(k.split("_")[0] in ("calib", "teste", "n5k") and not k.startswith("teste2") for k in z), "previsão para a avaliação antiga"
            usa = (~np.isnan(z["oof"][:, 0]), ~np.isnan(z["teste2"][:, 0])) if parcial else (np.ones(n, bool), np.ones(n2, bool))
            tem_ff = "oof_ff" in z
            assert tem_ff == bool(c.get("tipo") in ("fracoes", "unico") or (c.get("tipo") == "ft" and c.get("fracoes"))), "cabeça de frações"
            for suf in G.ORIENT:
                for ch, m, u in (("oof", n, usa[0]), ("teste2", n2, usa[1])):
                    v = z[ch + suf]
                    assert v.shape == (m, N), f"formato de {ch + suf}"
                    assert np.isfinite(v[u]).all(), f"valor faltando ou não finito em {ch + suf}"
                    if tem_ff:
                        f, d = z[ch + "_ff" + suf], z[ch + "_dens_ff" + suf]
                        assert f.shape == (m, K) and d.shape == (m,), f"formato das frações de {ch + suf}"
                        assert np.isfinite(f[u]).all() and (f[u] >= -1e-6).all() and np.abs(f[u].sum(1) - 1).max() < 1e-3, f"frações de {ch + suf} não somam 1"
                        dc = z["dens_fonte"][L.fold[u]] if ch == "oof" else z["dens_fonte"][5][None]
                        assert np.abs(d[u] - (f[u] * dc).sum(1)).max() < 1e-4, f"densidade pelas frações de {ch + suf} não bate com frações × dens_fonte"
            if tem_ff:
                assert z["dens_fonte"].shape == (6, K) and (z["dens_fonte"] >= 0).all() and list(z["fontes"]) == L.fontes, "dens_fonte"
            for ch in ("oof_aninhado", "aninhado_dens_ff", "alvos_aninhados"):      # NaN exatamente no próprio fold
                if ch in z:
                    falta = np.isnan(z[ch].reshape(5, n, -1)).any(2)
                    for k in range(5):
                        assert falta[k][L.fold == k].all() and not falta[k][(L.fold != k) & usa[0]].any(), f"{ch}: NaN fora do lugar no fold {k}"
            if c.get("tipo") == "professor":
                assert np.allclose(z["alvos_oof"][usa[0]], z["oof"][usa[0]][:, [G.I_M, G.I_D, G.I_P]]), "alvos_oof do professor"
            if sem_suf(nome) != nome:       # treinada com a base externa: com a lista do manifesto
                assert base and "ext_lista" in z and str(z["ext_lista"]) == ME.get("lista"), "treinada com outra lista externa"
            if "ext" in z:                  # previsões nas fotos externas (relato): só ids externos, formato certo
                assert all(str(i).startswith("ext:") for i in z["ext_ids"]) and z["ext"].shape == (len(z["ext_ids"]), N), "previsões externas"
                assert parcial or Lx is None or [str(i) for i in z["ext_ids"]] == Lx["ids"], "ids externos diferentes dos da lista desta pasta"
            assert M["configs"].get(nome, {}).get("estado") == "completa", f"estado no manifesto: {M['configs'].get(nome, {}).get('estado')}"
        except AssertionError as e:
            problemas.append(f"{nome}: {e}")
            continue
        entradas[nome] = z
    print(f"\n{'configuração':46s} | estado        | previsões | CV: RMSE log-massa | melhor rota com 8 orientações (critério de CV) | serviço: conferência")
    serv = {}
    if os.path.exists(os.path.join(raiz, "servico.json")):
        serv = json.load(open(os.path.join(raiz, "servico.json")))["modelos"]
    for nome, ic in M["configs"].items():
        cv = ic.get("cv_treino") or {}
        rot = max(cv["rotas"]["o8"].items(), key=lambda kv: kv[1]) if "rotas" in cv else None
        sc = serv.get(nome, {}).get("conferencia")
        print(f"{nome[:46]:46s} | {ic['estado']:13s} | {'ok' if nome in entradas else '—':9s} | {cv.get('rmse_logm', float('nan')):.4f}             | "
              f"{(rot[0] + ' ' + format(rot[1], '.1f') + '%') if rot else '':46s} | "
              + (f"dif. máx. {sc['max_dif_log']:.4f} em log" + (f", {sc['max_dif_fracao']:.4f} nas frações" if 'max_dif_fracao' in sc else "") if sc else ic.get("servico", {}).get("estado", "")))
    for p in problemas:
        print("  RECUSADA:", p)
    # decisão sobre a base externa (tomada no kernel, pela regra de 12.5) e versões que concorrem em 12.3
    dec, concorrem = M.get("decisao_externo"), {}
    esperadas = list(G.RODADAS["3"])
    if base:
        adotado = bool((dec or {}).get("adotado"))
        fila = G.com_externo(G.configs_da_rodada("3", parcial), base)
        esperadas = [c["nome"] for c in fila if c["tipo"] not in ("externo", "decisao") and c.get("se_adotado", adotado) == adotado
                     and (adotado or not c.get("externo") or (dec or {}).get("motivo") is None)]
        for c in fila:
            if c["tipo"] in ("professor", "unico", "ft") and not c.get("externo"):
                tem_versao = any(q.get("plano") == c["nome"] for q in fila)
                concorrem[c["nome"]] = c["nome"] + (f" +{base}" if adotado and tem_versao else "")
        print(f"\ndecisão sobre a base externa {base} (regra de 12.5, aplicada no kernel às previsões fora-do-fold dos dois professores): "
              + ("SEM DECISÃO GRAVADA" if dec is None else ("ADOTADA" if adotado else "NÃO adotada") + " | " + (dec.get("motivo") or
                 f"critério de CV {dec['criterio_sem']:.2f} -> {dec['criterio_com']:.2f} ({dec['dif_criterio']:+.2f}); acima de 25 g "
                 f"{dec['acima_25g_sem']:.2f} -> {dec['acima_25g_com']:.2f} ({dec['dif_acima_25g']:+.2f}); n = {dec['n']}")))
        if dec and not dec.get("motivo") and not parcial and dec["sem"] in entradas and dec["com"] in entradas:      # a decisão refeita aqui tem de ser a mesma
            r = G.regra_adocao(L, entradas[dec["sem"]]["oof"], entradas[dec["com"]]["oof"])
            assert r["adotado"] == dec["adotado"] and abs(r["dif_criterio"] - dec["dif_criterio"]) < 1e-6, "a decisão gravada não se repete a partir das previsões"
            print("  refeita aqui a partir das previsões dos dois professores: igual")
        print("  concorrem em 12.3:", "; ".join(f"{k.replace('gpu3 ', '')} -> {'versão +' + base if v != k else 'versão sem a base'}" for k, v in concorrem.items()))
        arq_rel = os.path.join(raiz, "relato_externo.json")
        if os.path.exists(arq_rel):
            rel = json.load(open(arq_rel))
            print(f"\nrelato secundário, FORA do domínio (12.5): erro de massa no teste_ext do {base}, {rel['n_teste_ext']} pratos "
                  f"({rel.get('n_teste_ext_acima_600g')} acima de 600 g). Não escolhe nada, não é o critério do produto, não é leitura do teste2.")
            for nome, r in rel["modelos"].items():
                print(f"  {nome[:52]:52s} " + " | ".join(f"{k}: MAPE {v['total'].get('mape')}% viés {v['total'].get('vies_mediano_pct')}% "
                                                         f"(>600 g: {v['acima_600g'].get('mape')}% / {v['acima_600g'].get('vies_mediano_pct')}%)"
                                                         for k, v in r.items() if isinstance(v, dict)))
    faltam = [x for x in (esperadas if not M["smoke"] else []) if x not in entradas]
    if faltam:
        print("  não vieram (orçamento, falha ou dependência):", faltam)
    # paridade das features com a execução 2, pela réplica do ridge antigo
    ref_pasta = a.referencia or os.path.join(zen_comum.TD, "resultados", "zen_banco_r2")
    nomes_r2 = {str(z["r2_nome"]): nome for nome, z in entradas.items() if "r2_teste2" in z}
    ref = {}
    if glob.glob(os.path.join(ref_pasta, "**", "manifesto.json"), recursive=True):
        for arq in glob.glob(os.path.join(ref_pasta, "**", "pred__*.npz"), recursive=True):
            zr = np.load(arq, allow_pickle=False)
            if str(zr["nome"]) in nomes_r2 and "teste2" in zr.files:
                ref[str(zr["nome"])] = zr["teste2"].astype(np.float64)
    else:
        for x in nomes_r2:
            if os.path.exists(os.path.join(ref_pasta, x + ".npz")):
                zr = np.load(os.path.join(ref_pasta, x + ".npz"), allow_pickle=False)
                if "teste2" in zr.files:
                    ref[x] = zr["teste2"].astype(np.float64)
    print(f"\nparidade das features com a execução 2 ({ref_pasta}): réplica do ridge antigo sobre as features novas × previsões da execução 2 no teste2 (sem gabarito)")
    avisos = []
    for x, nome in nomes_r2.items():
        if x not in ref:
            print(f"  {nome[:44]:44s} sem referência da execução 2 ({x})")
            continue
        r2 = entradas[nome]["r2_teste2"]; u = ~np.isnan(r2[:, 0])
        pm, pt = _par(r2[u, G.I_M], ref[x][u, G.I_M]), float(np.abs(r2[u] - ref[x][u]).max())
        print(f"  {nome[:44]:44s} log-massa: corr {pm['corr']:.5f}, |dif| mediana {pm['dif_mediana']:.5f}, máxima {pm['dif_max']:.4f} | máxima nas 22 saídas {pt:.4f}"
              f" | CV da réplica: RMSE {float(entradas[nome]['r2_cv_rmse_logm']):.4f} em {int(entradas[nome]['r2_n'])} pratos")
        if pm["corr"] < 0.9995 or pm["dif_mediana"] > 0.002:
            avisos.append(f"{nome}: as features não repetem as da execução 2 (corr {pm['corr']:.4f}, |dif| mediana {pm['dif_mediana']:.4f}); o ambiente (torch, timm, pesos) pode ter mudado")
    if not nomes_r2:
        print("  nenhuma configuração trouxe a réplica (r2_teste2)")
    for x in avisos:
        print("ATENÇÃO:", x)
    if a.so_confere:
        return print(f"\nconferências feitas: {len(entradas)} configurações aceitas, {len(problemas)} recusadas (nada gravado)")
    res = os.path.join(os.path.abspath(a.ensaio), "resultados") if a.ensaio else os.path.join(zen_comum.TD, "resultados")
    dest = os.path.join(res, "zen_banco_r3")
    os.makedirs(dest, exist_ok=True)
    for nome, z in entradas.items():
        np.savez(os.path.join(dest, nome + ".npz"), **z)
    json.dump(dict(pasta=os.path.abspath(a.pasta), manifesto={k: M.get(k) for k in ("formato", "rodada", "inicio", "duracao_min", "gpus", "torch", "timm", "listas_hash", "listas2_hash", "externo", "tamanho_saida")},
                   ingeridas=sorted(entradas), recusadas=problemas, faltam=faltam, avisos=avisos, decisao_externo=dec, concorrem_em_12_3=concorrem,
                   cv_treino={x: M["configs"][x].get("cv_treino") for x in entradas}),
              open(os.path.join(res, "ingestao_kaggle_r3.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n{len(entradas)} configurações gravadas em {dest} (todas as chaves de pred__*.npz). Nenhum gabarito de avaliação foi lido e nada foi escolhido aqui.")


def rodada4(a, M, raiz):
    """Conferências da saída da execução 4 (ver o cabeçalho). Sem rótulo de avaliação: só lê os rótulos do treino."""
    import tempfile
    import monta_pacote as P
    if a.divisoes:
        sys.argv += ["--divisoes", a.divisoes]
    try:
        arqs, variante = P.conteudo("push"), "completo"
    except FileNotFoundError as e:
        arqs, variante = P.conteudo("repo"), f"só bases liberadas ({e})"
    t = tempfile.mkdtemp(prefix="ingere_r4_")
    os.makedirs(os.path.join(t, "listas"))
    for n, b in arqs.items():
        if n.startswith("listas/"):
            open(os.path.join(t, n), "wb").write(b)
    L = G.Dados(os.path.join(t, "listas"), rodada="4")
    shutil.rmtree(t)
    if M.get("formato") not in G.FORMATOS_ACEITOS or M["listas_hash"] != L.hash or M.get("listas2_hash") != L.hash2:
        raise SystemExit(f"listas ou formato diferentes dos desta pasta (manifesto: {M.get('formato')}, {M['listas_hash']}, {M.get('listas2_hash')}; aqui: {L.hash}, {L.hash2})")
    n, n2, N, K = len(L.ids), len(L.conj["teste2"]), L.N, len(L.fontes)
    print(f"listas ({variante}): treino_v2 {n} pratos (hash {L.hash}); teste2 {n2} pratos, fora do treino (hash {L.hash2})")
    print(f"saída do Kaggle: {M['formato']}, rodada 4, {M['inicio']}, {M.get('duracao_min')} min, GPUs {M['gpus'] or 'nenhuma'}, torch {M.get('torch')}, timm {M.get('timm')}, smoke={M['smoke']}")
    for ch, v in (M.get("dominios") or {}).items():
        aqui = L.dom.get(ch) if "sha256_lista" in v else None
        bate = aqui is not None and v.get("sha256_lista") == aqui["sha256_lista"] and v.get("sha256_divisao") == aqui["sha256_divisao"]
        print(f"  base {ch:9s} {v.get('situacao')}" + (f" | {v.get('licenca')}, {v.get('uso_declarado')} | lista e divisão " + (
            "iguais às desta máquina" if bate else "NÃO CONFERIDAS (sem a lista aqui)" if aqui is None else "DIFERENTES das desta máquina") if "sha256_lista" in v else ""))
        if aqui is not None and not bate:
            raise SystemExit(f"a lista ou a divisão de {ch} usada no Kaggle não é a desta máquina")
    parcial = bool(M["smoke"])
    if parcial and not (a.ensaio or a.so_confere):
        raise SystemExit("Saída de --smoke: use --so-confere ou --ensaio DIR.")
    cfgs = {c["nome"]: c for c in G.todas_as_configs()}
    comerciais = {"R0", "R1", "H0", "F1", "F2", "P1"}
    entradas, problemas = {}, []
    for arq in sorted(glob.glob(os.path.join(raiz, "pred__*.npz"))):
        z = dict(np.load(arq, allow_pickle=False))
        nome = str(z["nome"]); c = cfgs.get(nome, {})
        try:
            assert c.get("v4"), "configuração fora da rodada 4"
            assert str(z["listas_hash"]) == L.hash and str(z["listas2_hash"]) == L.hash2, "hash das listas"
            assert list(z["ids_treino"]) == L.ids and np.array_equal(z["fold"], L.fold) and list(z["ids_teste2"]) == L.conj["teste2"], "ids, ordem ou folds"
            usa = (~np.isnan(z["oof"][:, 0]), ~np.isnan(z["teste2"][:, 0])) if parcial else (np.ones(n, bool), np.ones(n2, bool))
            tem_ff = "oof_ff" in z
            for suf in G.ORIENT:
                for ch_, m, u in (("oof", n, usa[0]), ("teste2", n2, usa[1])):
                    v = z[ch_ + suf]
                    assert v.shape == (m, N) and np.isfinite(v[u]).all(), f"{ch_ + suf} faltando, fora de formato ou não finito"
                    if tem_ff:
                        f = z[ch_ + "_ff" + suf]
                        assert f.shape == (m, K) and np.abs(f[u].sum(1) - 1).max() < 1e-3, f"frações de {ch_ + suf}"
            feitos = [str(x) for x in z["dominios"]]
            listas_ext = json.loads(str(z["listas_ext"]))
            for ch in feitos:
                d = L.dom.get(ch)
                if d is None:
                    continue
                assert parcial or listas_ext[ch]["hash"] == d["hash"], f"{ch}: fotos de outra lista"
                pos = {x: i for i, x in enumerate(d["ids"])}
                for pre, sel in (("extcv", d["treina"]), ("extreg", d["reg"])):
                    ids = [str(x) for x in z[f"{pre}_ids__{ch}"]]
                    if parcial:     # --smoke: um recorte das fotos, na ordem da lista e do papel certo
                        assert all(x in pos and sel[pos[x]] for x in ids) and [pos[x] for x in ids] == sorted(pos[x] for x in ids), f"{pre} de {ch}: ids, ordem ou papel"
                    else:
                        assert ids == [d["ids"][i] for i in np.where(sel)[0]], f"{pre} de {ch}: ids ou ordem"
                    assert z[f"{pre}__{ch}"].shape == (len(ids), N) and np.isfinite(z[f"{pre}__{ch}"]).all(), f"{pre} de {ch}: faltando ou não finito"
                    if pre == "extcv":
                        assert list(z[f"extcv_fold__{ch}"]) == [d["fold_ext"][pos[x]] for x in ids], f"{ch}: fold em que cada foto valida"
                    else:
                        assert list(z[f"extreg_papel__{ch}"]) == [d["papel"][pos[x]] for x in ids], f"{ch}: papel das fotos de registro"
            proc = json.loads(str(z["procedencia"]))
            ap_ = c.get("apelido")
            assert proc["so_comercial_ok"] == (ap_ in comerciais), f"procedência: so_comercial_ok = {proc['so_comercial_ok']}"
            assert M["configs"].get(nome, {}).get("estado") == "completa", f"estado no manifesto: {M['configs'].get(nome, {}).get('estado')}"
        except (AssertionError, KeyError) as e:
            problemas.append(f"{nome}: {type(e).__name__}: {e}")
            continue
        entradas[nome] = z
    print(f"\n{'':4s}{'configuração':44s} | estado        | previsões | CV: RMSE log-massa | aceitação (13.7) | só dados comerciais | bases externas com previsão")
    for nome, ic in M["configs"].items():
        if not ic.get("apelido"):
            continue
        cv = ic.get("cv_treino") or {}
        ac = cv.get("aceitacao")
        print(f"{ic['apelido']:4s}{nome[:44]:44s} | {ic['estado']:13s} | {'ok' if nome in entradas else '—':9s} | {cv.get('rmse_logm', float('nan')):.4f}             | "
              f"{('passou' if ac['passou'] else 'NÃO passou') if ac else '':16s} | {str(ic.get('so_comercial_ok')):19s} | "
              + (", ".join(str(x) for x in entradas[nome]["dominios"]) if nome in entradas else ""))
    for p_ in problemas:
        print("  RECUSADA:", p_)
    faltam = [x for x in (G.RODADAS["4"] if not parcial else []) if cfgs[x]["tipo"] not in ("externo", "feat") and x not in entradas]
    if faltam:
        print("  não vieram (orçamento, falha, base ausente):", faltam)
    # paridade com o D1 da execução 3, refeita aqui
    r0, h0 = (next((nm for nm in entradas if cfgs[nm].get("apelido") == x), None) for x in ("R0", "H0"))
    par = dict(passou=False, motivo="R0 ou H0 não vieram")
    if r0 and h0:
        ref = {r["dish_id"]: [float(r[k]) for k in ("log_massa_o1", "log_massa_o8", "log_dens_regressao_o1", "log_dens_fracoes_o1")]
               for r in csv.DictReader(open(os.path.join(AQUI, "listas", G.REF_D1)))}
        par = G.paridade_d1(L, entradas[r0], entradas[h0], ref)
    print(f"\nparidade com o D1 da execução 3 (13.7), refeita aqui: {'OK' if par['passou'] else 'FALHOU: ' + str(par.get('motivo'))}"
          + (f" | correlações {', '.join(f'{k} {v:.5f}' for k, v in par['corr'].items())} | critério de CV do D1 {par['criterio_cv_d1']:.2f}% (71,3 ± 0,3)" if "corr" in par else "")
          + f" | no kernel: {(M.get('paridade_d1') or {}).get('passou')}")
    if a.so_confere:
        return print(f"\nconferências feitas: {len(entradas)} configurações aceitas, {len(problemas)} recusadas (nada gravado; nenhuma previsão de registro ou do teste2 foi pontuada)")
    if not par["passou"] and not parcial:
        raise SystemExit("A paridade com o D1 falhou: nada é gravado no banco (13.7). As previsões continuam na pasta de saída.")
    res = os.path.join(os.path.abspath(a.ensaio), "resultados") if a.ensaio else os.path.join(zen_comum.TD, "resultados")
    dest = os.path.join(res, "zen_banco_r4")
    os.makedirs(dest, exist_ok=True)
    for nome, z in entradas.items():
        np.savez(os.path.join(dest, nome + ".npz"), **z)
    json.dump(dict(pasta=os.path.abspath(a.pasta), manifesto={k: M.get(k) for k in ("formato", "rodada", "inicio", "duracao_min", "gpus", "torch", "timm", "listas_hash",
                                                                                     "listas2_hash", "dominios", "paridade_d1", "confianca", "tamanho_saida")},
                   paridade_refeita=par, ingeridas=sorted(entradas), recusadas=problemas, faltam=faltam,
                   configs={nm: {k: M["configs"][nm].get(k) for k in ("apelido", "estado", "cv_treino", "treinado_com", "so_comercial_ok", "confiavel")} for nm in M["configs"]}),
              open(os.path.join(res, "ingestao_kaggle_r4.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n{len(entradas)} configurações gravadas em {dest} (todas as chaves de pred__*.npz). Nada foi pontuado nem escolhido aqui (13.9 roda depois).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pasta")
    ap.add_argument("--ensaio", help="pasta de trabalho isolada: não altera o banco nem zen_avalia.json")
    ap.add_argument("--so-confere", action="store_true")
    ap.add_argument("--referencia", help="execução 1 para a paridade: pasta de banco ou de saída do Kaggle (padrão: <TD>/resultados/zen_banco)")
    ap.add_argument("--substitui", action="store_true", help="saída com teste2: troca também as entradas do zen_banco e roda zen_avalia (lê o teste antigo)")
    ap.add_argument("--divisoes", help="execução 4: pasta com as listas e tabelas de divisão das bases fechadas (fora do repositório)")
    a = ap.parse_args()
    man = sorted(glob.glob(os.path.join(a.pasta, "**", "manifesto.json"), recursive=True))
    if man and json.load(open(man[0])).get("rodada") == "4":       # execução 4: treino_v2 + bases externas
        return rodada4(a, json.load(open(man[0])), os.path.dirname(man[0]))
    if man and json.load(open(man[0])).get("rodada") == "3":       # execução 3: treino_v2, só teste2
        return rodada3(a, json.load(open(man[0])), os.path.dirname(man[0]))
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
