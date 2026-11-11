#!/usr/bin/env python3
"""Monta o pacote do Kaggle/Colab: listas de pratos e o notebook que embute o script e as listas.

Uso:
  python3 monta_pacote.py listas     regrava listas/treino.csv e listas/aval_ids.csv a partir de <TD>/meta
                                     (zen_treino.csv, zen_aval.csv, gravados por zen_dados.py) e dos folds de zen_comum.Dados
  python3 monta_pacote.py notebook   regrava nutrelle_treino_gpu.ipynb a partir de treino_gpu.py e das listas
  python3 monta_pacote.py confere    falha se o notebook não corresponde aos arquivos desta pasta (rode antes do push)

Por que embutir: `kaggle kernels push` envia um único arquivo de código. O notebook grava treino_gpu.py (em texto) e as
listas (comprimidas, byte a byte) em disco e chama o script; não depende de dataset anexado nem do repositório.

Listas (só ids e metadados públicos do Nutrition5k, CC BY 4.0; nenhuma imagem):
  treino.csv    os 1.684 pratos do treino zenital estrito, com cadeia, fold (GroupKFold por cadeia, o mesmo da CPU)
                e os alvos (massa, macros, massa e proteína por grupo de alimento)
  aval_ids.csv  prato_id, split, dish_id dos 450 pratos de avaliação, SEM gabarito
  aval2_ids.csv prato_id, split, dish_id dos 829 pratos do teste2 (seção 11), SEM gabarito; é gravado por
                eval/treino/teste2_monta.py, não por este script, que só o embute e confere
  treino_v2.csv os 2.281 pratos do treino ampliado da seção 12 (folds novos por cadeia, alvos e frações da massa por
                fonte de proteína); é gravado por eval/treino/treino_v2_monta.py; aqui só é embutido e conferido
  externo_flavoria.csv  base externa da seção 12.5 (FlavoriaFoodWeight1700, Zenodo 5850856, CC BY 4.0): uma linha por
                foto, só rótulos (massa pesada, massa por grupo, frações por fonte) e o caminho da foto dentro do .tar
                do Zenodo; é gravado por eval/dados_ext/flavoria_monta.py; aqui só é embutido e conferido. As fotos são
                baixadas pelo kernel. Se o arquivo não existir, o notebook sai sem a base (execução como em 12.3).
"""
import base64, csv, gzip, hashlib, json, os, re, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
LISTAS = os.path.join(AQUI, "listas")
NOTEBOOK = os.path.join(AQUI, "nutrelle_treino_gpu.ipynb")
EXTERNO = "flavoria"                 # base externa ligada no notebook quando a lista dela existe em listas/
LISTA_EXTERNA = "listas/externo_flavoria.csv"
TEM_EXTERNO = os.path.exists(os.path.join(AQUI, LISTA_EXTERNA))
EMBUTIDOS = ["treino_gpu.py", "listas/treino.csv", "listas/aval_ids.csv", "listas/aval2_ids.csv", "listas/treino_v2.csv"] + ([LISTA_EXTERNA] if TEM_EXTERNO else [])
HASH_TESTE2 = "4aad94d89dc2440e"     # conteúdo de aval2_ids.csv (treino_gpu.Dados.hash2) usado na execução 2: o teste2 não muda
N_TESTE2 = 829
DESTINO = "/tmp/nutrelle_codigo"     # fora da pasta de saída do Kaggle


def listas():
    sys.path.insert(0, os.path.join(AQUI, ".."))
    from zen_comum import TD, Dados
    D = Dados()
    os.makedirs(LISTAS, exist_ok=True)
    with open(os.path.join(TD, "meta", "zen_treino.csv")) as f:
        rows = list(csv.DictReader(f))
    assert [r["dish_id"] for r in rows] == D.ids
    cab = list(rows[0])
    cab.insert(cab.index("cadeia") + 1, "fold")
    with open(os.path.join(LISTAS, "treino.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cab, lineterminator="\n"); w.writeheader()
        for r, k in zip(rows, D.fold):
            w.writerow(dict(r, fold=int(k)))
    with open(os.path.join(TD, "meta", "zen_aval.csv")) as f:
        av = list(csv.DictReader(f))
    with open(os.path.join(LISTAS, "aval_ids.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["prato_id", "split", "dish_id"])
        for r in av:
            w.writerow([r["prato_id"], r["split"], r["dish_id"]])
    ev = {r["dish_id"] for r in av}
    assert len(ev) == len(av) == 450 and not ev & set(D.ids)
    for s in ("calib", "teste", "n5k_v1"):
        assert [r["dish_id"] for r in av if r["split"] == s] == D.aval[s]["chave"]
    # cada cadeia inteira num único fold
    fc = {}
    for c, k in zip(D.cadeia, D.fold):
        assert fc.setdefault(c, k) == k
    print(f"listas: {len(rows)} pratos de treino em {len(fc)} cadeias e 5 folds; {len(av)} de avaliação (sem gabarito); interseção 0")


def _hash():
    h = hashlib.sha256()
    for n in EMBUTIDOS:
        h.update(open(os.path.join(AQUI, n), "rb").read())
    return h.hexdigest()


def _celula_lista(n):
    """Célula que regrava uma lista byte a byte (gzip + base64)."""
    dados = base64.b64encode(gzip.compress(open(os.path.join(AQUI, n), "rb").read(), 9, mtime=0)).decode()
    linhas = "\n".join(dados[i:i + 120] for i in range(0, len(dados), 120))
    return (f"# {n} (gzip + base64; o arquivo sai idêntico ao do repositório)\nimport base64, gzip\n"
            f"open('{DESTINO}/{n}', 'wb').write(gzip.decompress(base64.b64decode(\'\'\'\n{linhas}\n\'\'\')))\nprint('gravado: {DESTINO}/{n}')")


def notebook():
    md = lambda t: dict(cell_type="markdown", metadata={}, source=t)
    code = lambda t: dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=t)
    cells = [
        md("# Nutrelle — treino em GPU (seções 8, 10, 11 e 12: uma única foto de cima)\n\n"
           "Notebook gerado por `eval/treino/kaggle/monta_pacote.py`; **não edite à mão** (edite `treino_gpu.py` e gere de novo). "
           "Roda sem interação: baixa as fotos zenitais do Nutrition5k do bucket público, treina as configurações da rodada "
           "(padrão: execução 3, as 13 configurações da seção 12.3, com o treino_v2 e previsões só no teste2"
           + (f"; com a base externa `{EXTERNO}` da seção 12.5, baixada do Zenodo, as configurações 1 a 10 rodam também na versão `+{EXTERNO}` "
              "e o kernel decide a adoção pela regra pré-registrada antes dos ajustes finos" if TEM_EXTERNO else "") + ") até o orçamento de tempo "
           "e grava previsões (fora-do-fold no treino e no teste2, sem ler gabarito), registros, os arquivos para servir cada modelo "
           "e `servico.json` (latências) na pasta de saída.\n\n"
           "Kaggle: *Settings → Accelerator → GPU T4 x2* e *Internet → On*. Colab: *Ambiente de execução → GPU*. "
           "Instruções completas em `eval/treino/kaggle/LEIA-ME.md`."),
        code("# Parâmetros (os padrões servem para o Kaggle)\n"
             "import json, os\n"
             "RODADA = '3'         # 3 = seção 12.3 (um modelo só, treino_v2); 2 = seção 11; 1 = seção 10\n"
             "ORCAMENTO_H = {}    # horas de relógio; o que não couber inteiro não começa (ordem de corte: U0, depois U3)\n".format(4.5 if TEM_EXTERNO else 3.5)
             + "EXTERNO = {!r}   # base externa de treino da seção 12.5 ('' = sem ela, como em 12.3); se o download falhar, a execução segue sem ela\n".format(EXTERNO if TEM_EXTERNO else "")
             +
             "SAIDA = '/kaggle/working' if os.path.isdir('/kaggle/working') else '/content/nutrelle_saida'   # no Colab, aponte para o Drive para retomar\n"
             "EXTRA = []           # ex.: ['--so', 'ridge'] para rodar só parte; ['--retoma', '/content/drive/MyDrive/nutrelle_saida']; dados externos: ver LEIA-ME\n"
             "# as variáveis de ambiente NUTRELLE_SAIDA e NUTRELLE_EXTRA (lista em JSON) substituem os valores acima sem editar o notebook\n"
             "SAIDA = os.environ.get('NUTRELLE_SAIDA') or SAIDA\n"
             "EXTRA = json.loads(os.environ.get('NUTRELLE_EXTRA') or 'null') or EXTRA\n"
             "EXTERNO = os.environ.get('NUTRELLE_EXTERNO', EXTERNO)\n"
             f"os.makedirs('{DESTINO}/listas', exist_ok=True)"),
    ]
    for n in EMBUTIDOS:      # o script vai em texto (dá para ler no Kaggle); as listas, comprimidas (o notebook fica abaixo de 1 MB)
        if n.endswith(".py"):
            cells.append(code(f"%%writefile {DESTINO}/{n}\n" + open(os.path.join(AQUI, n)).read()))
        else:
            cells.append(code(_celula_lista(n)))
    cells.append(code(
        "import subprocess, sys\n"
        f"cmd = [sys.executable, '-u', '{DESTINO}/treino_gpu.py', '--rodada', RODADA, '--saida', SAIDA, '--orcamento-h', str(ORCAMENTO_H)]\n"
        "cmd += (['--externo', EXTERNO] if EXTERNO and RODADA == '3' else []) + EXTRA\n"
        "p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "for linha in p.stdout:\n"
        "    print(linha, end='')\n"
        "print('código de saída:', p.wait())   # não levanta erro: o que ficou pronto continua na pasta de saída"))
    cells.append(code(
        "import json, os\n"
        "m = os.path.join(SAIDA, 'manifesto.json')\n"
        "if os.path.exists(m):\n"
        "    m = json.load(open(m))\n"
        "    for nome, c in m['configs'].items():\n"
        "        print(f\"{c['estado']:14s} {nome:46s} {c['cv_treino'] or ''}\")\n"
        "        print(' ' * 15 + 'serviço:', c.get('servico', {}).get('estado'))\n"
        "    print('melhor modelo:', m['melhor_modelo'])\n"
        "    print('base externa:', (m.get('externo') or {}).get('situacao'), '| decisão:', m.get('decisao_externo'))\n"
        "    print('tamanho da saída:', m.get('tamanho_saida'))\n"
        "print(sorted(os.listdir(SAIDA)))"))
    for i, c in enumerate(cells):
        c["id"] = f"celula-{i}"
    nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                            language_info=dict(name="python"), nutrelle=dict(sha256=_hash(), embutidos=EMBUTIDOS)))
    with open(NOTEBOOK, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"notebook: {os.path.getsize(NOTEBOOK) / 1e3:.0f} kB, sha256 dos embutidos {_hash()[:16]}")


def confere():
    nb = json.load(open(NOTEBOOK))
    assert nb["metadata"]["nutrelle"]["sha256"] == _hash(), "notebook desatualizado: rode `python3 monta_pacote.py notebook`"
    for n in EMBUTIDOS:
        if n.endswith(".py"):
            cab = f"%%writefile {DESTINO}/{n}\n"
            cel = [c for c in nb["cells"] if c["cell_type"] == "code" and c["source"].startswith(cab)]
            assert len(cel) == 1 and cel[0]["source"][len(cab):] == open(os.path.join(AQUI, n)).read(), n
        else:
            assert sum(c["source"] == _celula_lista(n) for c in nb["cells"]) == 1, n
    assert os.path.getsize(NOTEBOOK) < 1_000_000, "notebook com mais de 1 MB"
    meta = json.load(open(os.path.join(AQUI, "kernel-metadata.json")))
    assert meta["code_file"] == os.path.basename(NOTEBOOK) and meta["enable_gpu"] and meta["enable_internet"]
    # o teste2 não pode cruzar com o treino nem com os 450 (a mesma conferência que treino_gpu.Dados faz ao rodar)
    ids = lambda arq: [r["dish_id"] for r in csv.DictReader(open(os.path.join(LISTAS, arq)))]
    tr, av, av2 = ids("treino.csv"), ids("aval_ids.csv"), ids("aval2_ids.csv")
    assert len(set(av2)) == len(av2) and not set(av2) & (set(tr) | set(av)), "aval2_ids.csv cruza com o treino ou com a avaliação antiga"
    v2 = list(csv.DictReader(open(os.path.join(LISTAS, "treino_v2.csv"))))
    assert len({r["dish_id"] for r in v2}) == len(v2) and not {r["dish_id"] for r in v2} & set(av2), "treino_v2.csv cruza com o teste2"
    cad = {}
    for r in v2:        # cada cadeia inteira num fold só; frações por fonte somam 1
        assert cad.setdefault(r["cadeia"], r["fold"]) == r["fold"], "treino_v2: cadeia em dois folds"
        assert abs(sum(float(r[c]) for c in r if c.startswith("ff_")) - 1) < 2e-3, "treino_v2: frações por fonte não somam 1"
    assert set(tr) <= {r["dish_id"] for r in v2}, "treino antigo fora do treino_v2 (a réplica de paridade com a execução 2 precisa dele)"
    print(f"listas: treino {len(tr)}, avaliação {len(av)}, teste2 {len(av2)}, treino_v2 {len(v2)} em {len(cad)} cadeias; interseções com o teste2 vazias")
    # o teste2 é o mesmo da execução 2 (conteúdo fixado) e não aparece em nenhuma lista de treino
    sys.dont_write_bytecode = True
    sys.path.insert(0, AQUI)
    import treino_gpu as G
    D = G.Dados(LISTAS, rodada="3")
    assert D.hash2 == HASH_TESTE2 and len(av2) == N_TESTE2 and list(D.conj) == ["teste2"], "aval2_ids.csv mudou: o teste2 tem de ficar intocado"
    par = [c for c in nb["cells"] if c["cell_type"] == "code" and c["source"].startswith("# Parâmetros")][0]["source"]
    assert ("EXTERNO = %r " % (EXTERNO if TEM_EXTERNO else "")) in par, "parâmetro EXTERNO do notebook não corresponde à lista em listas/"
    if TEM_EXTERNO:
        confere_externo(G, D, set(tr) | set(av) | set(av2) | {r["dish_id"] for r in v2})
    else:
        print(f"base externa: {LISTA_EXTERNA} não existe; o notebook roda sem ela (como em 12.3)")
    if meta["id"].startswith("USUARIO_KAGGLE/"):
        print("atenção: troque USUARIO_KAGGLE em kernel-metadata.json pelo seu usuário do Kaggle antes do push")
    print("confere: notebook e kernel-metadata.json em dia")


def confere_externo(G, D, n5k):
    """A lista externa (12.5) nunca traz prato do Nutrition5k, não toca no teste2 e só entra com os alvos pré-registrados."""
    x = list(csv.DictReader(open(os.path.join(AQUI, LISTA_EXTERNA), newline="")))
    cab_v2 = next(csv.reader(open(os.path.join(LISTAS, "treino_v2.csv"))))
    assert list(x[0])[:len(cab_v2)] == cab_v2, "lista externa: as primeiras colunas têm de ser as de treino_v2.csv, na mesma ordem"
    assert all(c in x[0] for c in ("dominio", "grupo", "vista", "split_ext", "arquivo", "quadro_preferido", "oclusao")), "lista externa: coluna faltando"
    ids = [r["dish_id"] for r in x]
    assert len(set(ids)) == len(ids), "lista externa: id repetido"
    assert not set(ids) & n5k, "lista externa: id de prato do Nutrition5k (treino, treino_v2, avaliação antiga ou teste2)"
    assert not [i for i in ids if re.fullmatch(r"dish_\d+", i)], "lista externa: id no formato do Nutrition5k"
    assert not [r for r in x if "dish_" in r["arquivo"] or "nutrition5k" in r["arquivo"].lower()], "lista externa: foto do Nutrition5k"
    assert {r["dominio"] for r in x} == {EXTERNO} and {r["split_ext"] for r in x} == {"treino_ext", "teste_ext"}, "lista externa: domínio ou divisão"
    assert all(r["cadeia"] == "-1" and r["fold"] == "-1" for r in x), "lista externa: só treina (fold −1)"
    lado = {}
    for r in x:         # os quadros de um prato ficam do mesmo lado; um quadro preferido por prato
        assert lado.setdefault(r["grupo"], r["split_ext"]) == r["split_ext"], "lista externa: grupo dos dois lados da divisão"
    pref = {}
    for r in x:
        pref[r["grupo"]] = pref.get(r["grupo"], 0) + (r["quadro_preferido"] == "1")
    assert set(pref.values()) == {1}, "lista externa: cada prato tem de ter exatamente um quadro preferido"
    assert all(float(r["massa"]) > 0 and abs(sum(float(r[c]) for c in r if c.startswith("ff_")) - 1) < 2e-3 for r in x), "lista externa: massa ou frações"
    assert all(r[c] == "" for r in x for c in ["proteina", "kcal", "gordura", "carbo"] + [c for c in r if c.startswith("p_")]), \
        "lista externa: proteína, kcal, gordura, carboidrato e proteína por grupo têm de vir vazios (não são medidos)"
    assert len({os.path.basename(r["arquivo"]) for r in x}) == len(x) and all(re.fullmatch(r"images/(train|val|test)/tray\d+_\d+_[\d.]+_(enter|identification)_[\d.]+\.png", r["arquivo"]) for r in x), "lista externa: caminho da foto"
    # o que o kernel lê: mesma função, mesmos pesos; alvos fora do pré-registro ficam mascarados (NaN), proteína fraca nunca é lida
    E = G.Dados(LISTAS, rodada="3", externo=os.path.join(AQUI, LISTA_EXTERNA), externo_pesos=G.EXTERNOS[EXTERNO]["pesos"])
    e = E.ext
    usados = sorted(int(i) for i in np.where(np.isfinite(e["T"]).any(0))[0])
    assert usados == [G.I_M] + list(range(E.I_MG.start, E.I_MG.stop)), f"lista externa: alvos usados {usados}"
    assert E.hash == D.hash and E.hash2 == D.hash2 == HASH_TESTE2 and E.todos == D.todos and E.chave == D.chave, "a base externa mudou as listas do Nutrition5k ou o teste2"
    assert not [i for i in e["ids"] if not i.startswith(f"ext:{EXTERNO}:")], "lista externa: id fora do domínio"
    r = e["resumo"]
    assert r["n"] == r["n_lista"] - r["fora_por_oclusao"] - r["fora_por_nao_ser_o_quadro_preferido_do_teste_ext"] and r["teste_ext"] == r["grupos_teste_ext"], "lista externa: filtros do adendo de 12.5"
    assert 0.15 <= r["grupos_teste_ext"] / r["grupos"] <= 0.25, "lista externa: a parte segurada tem de ficar perto de 20% dos pratos"
    info = G.EXTERNOS[EXTERNO]
    assert info["url"].startswith("https://zenodo.org/api/records/5850856/") and info["lista"] == os.path.basename(LISTA_EXTERNA)
    fila = G.com_externo(G.configs_da_rodada("3"), EXTERNO)
    assert sum(1 for c in fila if c.get("externo") and c["tipo"] != "ft") == 10 and all(not c.get("externo") or EXTERNO in c["nome"] for c in fila)
    print(f"base externa {EXTERNO}: lista {e['hash']}, {len(x)} fotos de {len(lado)} pratos; sem id do Nutrition5k; teste2 e listas do Nutrition5k intocados "
          f"(hash {D.hash} / {D.hash2}); o kernel usa {r['treino']} fotos de treino ({r['grupos_treino']} pratos) e {r['teste_ext']} pratos de teste_ext "
          f"({r['fora_por_oclusao']} fotos fora por oclusão, {r['fora_por_nao_ser_o_quadro_preferido_do_teste_ext']} por não serem o quadro preferido); "
          f"alvos usados: massa total e massa por grupo (+ frações por fonte)")


if __name__ == "__main__":
    {"listas": listas, "notebook": notebook, "confere": confere}[sys.argv[1] if len(sys.argv) > 1 else "confere"]()
