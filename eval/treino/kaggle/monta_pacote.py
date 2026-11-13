#!/usr/bin/env python3
"""Monta o pacote do Kaggle/Colab: listas de pratos e o notebook que embute o script e as listas.

Uso:
  python3 monta_pacote.py listas     regrava listas/treino.csv e listas/aval_ids.csv a partir de <TD>/meta
                                     (zen_treino.csv, zen_aval.csv, gravados por zen_dados.py) e dos folds de zen_comum.Dados
  python3 monta_pacote.py notebook [--kernel K]
                                     regrava o notebook do REPOSITÓRIO do kernel K, só com as bases liberadas (Flavoria,
                                     LeFood, Mensa: uso comercial permitido). K = r5 (padrão: a execução principal da quinta
                                     execução, nutrelle_treino_gpu.ipynb), r5a (a etapa 0, deteccao_r5/) ou r4
  python3 monta_pacote.py push <pasta fora do repositório> [--kernel K] [--usuario U] [--espelho-omnifood U/slug] [--divisoes PASTA]
                                     monta o notebook COMPLETO (as seis bases) e o kernel-metadata.json numa pasta fora do
                                     repositório, pronta para `kaggle kernels push -p <pasta>`; no r5, `kernel_sources`
                                     aponta para U/nutrelle-deteccao-r5 (a saída da etapa 0)
  python3 monta_pacote.py confere [<pasta do push>] [--kernel K]
                                     falha se o notebook do repositório não corresponde aos arquivos desta pasta ou se
                                     embute qualquer coisa de uma base sem licença comercial (rode antes do commit e do
                                     push); com a pasta, confere também o notebook completo; no r5, também que nada do
                                     teste2 vai embutido

Listas e tabelas que ficam fora do repositório (as das bases fechadas) são procuradas em listas/ e, depois, nas pastas
dadas por --divisoes e pela variável NUTRELLE_SCRATCH (e nas subpastas listas_nc/ e listas_r4/ dela).

Por que embutir: `kaggle kernels push` envia um único arquivo de código. O notebook grava treino_gpu.py (em texto) e as
listas (comprimidas, byte a byte) em disco e chama o script; não depende de dataset anexado nem do repositório.

Duas versões, por causa da procedência (seção 13 e fontes_treino.md, 11): as listas do OmniFood8K, do Food Portion
Benchmark (CC BY-NC 4.0) e do SNAPMe (CC BY-SA 4.0, autores pedem para não treinar) e as tabelas de divisão delas ficam
FORA do repositório público (.gitignore). O notebook do repositório não embute nada delas e roda sozinho, como uma quarta
execução só com dados liberados; o completo é montado na hora do push, fora do repositório.

Listas (só ids, metadados e rótulos; nenhuma imagem):
  treino.csv, aval_ids.csv, aval2_ids.csv, treino_v2.csv   como nas execuções anteriores (aval2 = teste2, SEM gabarito)
  ref_d1_r3.csv          previsões fora-do-fold das duas cabeças do D1 na execução 3 (conferência de paridade de 13.7)
  externo_<base>.csv     uma base externa (eval/dados_ext/<base>_monta.py)
  divisao_r4_<base>.csv  a divisão dela para a quarta execução (eval/treino/externo_r4_divide.py)
No notebook vão versões ENXUTAS das duas últimas: sem as linhas `fora`, só as colunas que o kernel lê e com os rótulos das
linhas que não são de treino (partes de registro, avaliação fraca) já apagados; o sha256 dos arquivos inteiros vai em
procedencia_r4.json, para o manifesto de cada modelo (13.8).
"""
import base64, csv, hashlib, io, json, lzma, os, re, subprocess, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
LISTAS = os.path.join(AQUI, "listas")
REPO = os.path.abspath(os.path.join(AQUI, "..", "..", ".."))
# Um pacote por kernel. `pasta`: onde ficam, no repositório, o notebook só com bases liberadas e o kernel-metadata.json.
KERNELS = {
    "r4": dict(nome="quarta execução (seção 13); o notebook do repositório agora é o da quinta", pasta=AQUI, notebook="nutrelle_treino_gpu_r4.ipynb",
               slug="nutrelle-treino-gpu-r4", script="treino_gpu.py",
               basicos=["treino_gpu.py", "listas/aval2_ids.csv", "listas/treino_v2.csv", "listas/ref_d1_r3.csv"],     # não lê treino.csv nem aval_ids.csv
               args="['--rodada', '4']", orcamento=4.5, manifesto="manifesto.json"),
    "r5a": dict(nome="etapa 0 da quinta execução (seção 14.3: caixa da refeição em cada foto)", pasta=os.path.join(AQUI, "deteccao_r5"),
                notebook="nutrelle_deteccao_r5.ipynb", slug="nutrelle-deteccao-r5", script="deteccao_r5.py",
                basicos=["deteccao_r5.py", "treino_gpu.py", "listas/treino_v2.csv", "listas/caixas_profundidade_n5k.csv"],
                args="[]", orcamento=1.5, manifesto="manifesto_r5.json"),
    "r5": dict(nome="quinta execução (seção 14): recorte pelo prato e receitas corrigidas", pasta=AQUI, notebook="nutrelle_treino_gpu.ipynb", slug="nutrelle-treino-gpu-r5",
               script="treino_gpu.py", basicos=["treino_gpu.py", "deteccao_r5.py", "listas/treino_v2.csv", "listas/conferencia_r5.json"],     # sem o teste2 (14.1)
               args="['--rodada', '5']", orcamento=4.5, manifesto="manifesto.json", fontes_kernel=["nutrelle-deteccao-r5"]),
}
KERNEL = "r5"           # mudado por --kernel
NOTEBOOK = os.path.join(AQUI, "nutrelle_treino_gpu.ipynb")
SLUG = "nutrelle-treino-gpu-r5"
BASICOS = KERNELS["r5"]["basicos"]


def usa_kernel(k):
    """Escolhe o pacote com que as funções abaixo trabalham."""
    global KERNEL, NOTEBOOK, SLUG, BASICOS
    KERNEL, kk = k, KERNELS[k]
    NOTEBOOK, SLUG, BASICOS = os.path.join(kk["pasta"], kk["notebook"]), kk["slug"], kk["basicos"]
LIBERADAS = ["flavoria", "lefood", "mensa"]                  # uso_declarado = comercial_ok: podem ir para o repositório público
FECHADAS = ["omnifood", "fpb", "snapme"]                     # nao_comercial / autores_pedem_nao_treinar: só no notebook do push
PREFIXO_FECHADAS = ("omn_", "fpb_", "snp_")
DESTINO = "/tmp/nutrelle_codigo"     # fora da pasta de saída do Kaggle
HASH_TESTE2 = "4aad94d89dc2440e"     # conteúdo de aval2_ids.csv (treino_gpu.Dados.hash2) desde a execução 2: o teste2 não muda
N_TESTE2 = 829
COLS_LISTA = ["dominio", "grupo", "vista", "arquivo", "arquivo_base", "quadros", "url", "sha256", "bytes", "licenca", "uso_declarado"]
COLS_DIV = ["dish_id", "dominio", "grupo", "papel", "fold_ext", "folds_proibidos", "quadro_metrica", "quadro_metrica_nome", "quadros_treino",
            "grupo_metrica", "peso_nutricao"]
ASPAS3 = "'" * 3
RX_CELULA = re.compile(r"open\('" + re.escape(DESTINO) + r"/([^']+)', 'wb'\)\.write\(lzma\.decompress\(base64\.b85decode\(" + ASPAS3 + r"\n(.*?)\n" + ASPAS3
                       + re.escape(".replace('\\n', '')") + r"\)\)\)", re.S)


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


def _opcao(nome, padrao=None):
    return sys.argv[sys.argv.index(nome) + 1] if nome in sys.argv else padrao


def extras():
    """Pastas fora do repositório onde procurar as listas e tabelas das bases fechadas."""
    sc = os.environ.get("NUTRELLE_SCRATCH", "")
    return [x for x in [_opcao("--divisoes"), sc] + ([os.path.join(sc, "listas_nc"), os.path.join(sc, "listas_r4")] if sc else []) if x]


def acha(nome, pastas):
    for d in [LISTAS] + list(pastas):
        if os.path.exists(os.path.join(d, nome)):
            return os.path.join(d, nome)
    raise FileNotFoundError(f"{nome} não está em listas/ nem em {list(pastas)} (use --divisoes ou NUTRELLE_SCRATCH)")


def _bytes_csv(cab, linhas):
    f = io.StringIO()
    w = csv.DictWriter(f, cab, lineterminator="\n", extrasaction="ignore"); w.writeheader(); w.writerows(linhas)
    return f.getvalue().encode()


def _alvo(c):
    return c in ("massa", "proteina", "kcal", "gordura", "carbo") or c.startswith(("m_", "p_", "ff_"))


def enxuta(arq_lista, arq_div):
    """(lista enxuta, divisão enxuta, procedência) de uma base: sem as linhas `fora`, só as colunas que o kernel lê, e os
    rótulos das linhas que não são de treino apagados ANTES de sair desta máquina."""
    bl, bd = open(arq_lista, "rb").read(), open(arq_div, "rb").read()
    L = list(csv.DictReader(io.StringIO(bl.decode("utf-8"))))
    V = list(csv.DictReader(io.StringIO(bd.decode("utf-8"))))
    assert [r["dish_id"] for r in L] == [r["dish_id"] for r in V], f"{arq_lista}: lista e divisão desalinhadas"
    v2 = next(csv.reader(open(os.path.join(LISTAS, "treino_v2.csv"))))
    cab_l = [c for c in v2 if c in L[0]] + [c for c in COLS_LISTA if c in L[0]]
    out_l, out_d = [], []
    for r, v in zip(L, V):
        if v["papel"] == "fora":
            continue
        tr = v["papel"] == "treino"
        out_l.append({c: (r[c] if tr or not _alvo(c) else "") for c in cab_l})
        out_d.append({c: (v.get(c, "") if tr or c != "peso_nutricao" else "") for c in COLS_DIV})
    proc = dict(sha256_lista=hashlib.sha256(bl).hexdigest(), sha256_divisao=hashlib.sha256(bd).hexdigest(), linhas_lista=len(L),
                linhas_no_pacote=len(out_l), licenca=L[0]["licenca"], uso_declarado=L[0]["uso_declarado"])
    return _bytes_csv(cab_l, out_l), _bytes_csv(COLS_DIV, out_d), proc


def conteudo(variante):
    """{nome do arquivo no pacote: bytes} da variante 'repo' (só bases liberadas) ou 'push' (as seis)."""
    arqs = {n: open(os.path.join(AQUI, n), "rb").read() for n in BASICOS}
    proc, pastas = {}, ([] if variante == "repo" else extras())
    for ch in LIBERADAS + (FECHADAS if variante == "push" else []):
        lst, div, proc[ch] = enxuta(acha(f"externo_{ch}.csv", pastas), acha(f"divisao_r4_{ch}.csv", pastas))
        arqs[f"listas/externo_{ch}.csv"], arqs[f"listas/divisao_r4_{ch}.csv"] = lst, div
    arqs["listas/procedencia_r4.json"] = (json.dumps(proc, indent=1, ensure_ascii=False, sort_keys=True) + "\n").encode()
    return arqs


def _hash(arqs):
    h = hashlib.sha256()
    for n in sorted(arqs):
        h.update(n.encode()); h.update(arqs[n])
    return h.hexdigest()


def _celula_arquivo(n, dados):
    """Célula que regrava um arquivo byte a byte (lzma + base85: o notebook completo fica abaixo de 1 MB)."""
    b85 = base64.b85encode(lzma.compress(dados, preset=9 | lzma.PRESET_EXTREME)).decode()
    linhas = "\n".join(b85[i:i + 120] for i in range(0, len(b85), 120))
    return (f"# {n} (lzma + base85; o arquivo sai idêntico ao montado por monta_pacote.py)\nimport base64, lzma\n"
            f"open('{DESTINO}/{n}', 'wb').write(lzma.decompress(base64.b85decode({ASPAS3}\n{linhas}\n{ASPAS3}.replace('\\n', ''))))\nprint('gravado: {DESTINO}/{n}')")


def monta(variante):
    if KERNEL == "r5a":
        return monta_deteccao(variante)
    arqs = conteudo(variante)
    doms = LIBERADAS + (FECHADAS if variante == "push" else [])
    md = lambda t: dict(cell_type="markdown", metadata={}, source=t)
    code = lambda t: dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=t)
    aviso = ("**Versão do repositório: só bases liberadas** (Nutrition5k, Flavoria, LeFood-Set, GT Mensa). As configurações que usam "
             "OmniFood8K ou o Food Portion Benchmark ficam de fora e o manifesto diz por quê; o SNAPMe também. A versão completa é montada "
             "fora do repositório com `python3 monta_pacote.py push <pasta>`." if variante == "repo" else
             "**Versão completa, montada fora do repositório** (as seis bases). Contém listas derivadas de bases sem licença comercial "
             "(OmniFood8K e FPB, CC BY-NC 4.0) e do SNAPMe (CC BY-SA 4.0, só avaliação): **não commitar, não publicar**.")
    cells = [
        md(("# Nutrelle — treino em GPU, quarta execução (seção 13: mais bases de treino; uma única foto de cima)\n\n" if KERNEL == "r4" else
            "# Nutrelle — treino em GPU, quinta execução (seção 14: recorte pelo prato e receitas corrigidas; uma única foto de cima)\n\n")
           + "Notebook gerado por `eval/treino/kaggle/monta_pacote.py`; **não edite à mão** (edite `treino_gpu.py` e gere de novo). "
           + ("Roda sem interação: baixa as fotos zenitais do Nutrition5k e as das bases externas (em paralelo, desde o início), extrai as "
              "features do SigLIP 2 So400m, treina as configurações da seção 13.6 até o orçamento de tempo e grava previsões cruas "
              "(fora-do-fold no treino; no teste2 e nas partes de registro das bases externas, **sem pontuar nada**), registros, os arquivos "
              "para servir cada modelo e `servico.json`." if KERNEL == "r4" else
              "Lê a saída do kernel da etapa 0 (`nutrelle-deteccao-r5`, anexada em *Input*) e `listas/conferencia_r5.json` (regra e bases reprovadas na "
              "conferência a olho; sem `\"conferida\": true` o kernel não começa). Baixa as fotos, recorta cada uma na janela do guia, extrai as features do "
              "So400m sobre o recorte, treina as configurações de 14.5 até o orçamento e grava previsões cruas (fora-do-fold no treino e nas partes de "
              "registro das bases externas, **sem pontuar nada; nenhuma do teste2**), os arquivos para servir e `servico.json`.")
           + "\n\n" + aviso + "\n\n"
           "Kaggle: *Settings → Accelerator → GPU T4 x2* e *Internet → On*. Instruções completas em `eval/treino/kaggle/LEIA-ME.md`."),
        code("# Parâmetros (os padrões servem para o Kaggle)\n"
             "import json, os\n"
             f"RODADA = '{KERNEL[1:]}'         # este pacote só traz as listas desta execução\n"
             "ORCAMENTO_H = 4.5    # horas de relógio; o que não couber inteiro não começa (a ordem de prioridade decide o que fica de fora)\n"
             "SAIDA = '/kaggle/working' if os.path.isdir('/kaggle/working') else '/content/nutrelle_saida'   # no Colab, aponte para o Drive para retomar\n"
             "EXTRA = []           # ex.: ['--so', 'ridge']; ['--dominios', 'flavoria,mensa']; ['--retoma', '<pasta>']; ['--espelho', '<pasta com o zip do OmniFood8K>']\n"
             "# as variáveis de ambiente NUTRELLE_SAIDA e NUTRELLE_EXTRA (lista em JSON) substituem os valores acima sem editar o notebook\n"
             "SAIDA = os.environ.get('NUTRELLE_SAIDA') or SAIDA\n"
             "EXTRA = json.loads(os.environ.get('NUTRELLE_EXTRA') or 'null') or EXTRA\n"
             f"os.makedirs('{DESTINO}/listas', exist_ok=True)"),
    ]
    for n in sorted(arqs, key=lambda n: (n != "treino_gpu.py", n)):      # tudo comprimido (o script também): o notebook completo cabe em 1 MB
        cells.append(code(_celula_arquivo(n, arqs[n])))
    cells.append(code(
        "import subprocess, sys\n"
        f"cmd = [sys.executable, '-u', '{DESTINO}/treino_gpu.py', '--rodada', RODADA, '--saida', SAIDA, '--orcamento-h', str(ORCAMENTO_H)] + EXTRA\n"
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
        "        print(f\"{c['estado']:14s} {nome:46s} serviço: {c.get('servico', {}).get('estado')}\")\n"
        "    print('paridade com o D1:', m.get('paridade_d1'))\n"
        "    print('recorte pelo prato:', {k: v for k, v in (m.get('prato') or {}).items() if k in ('regra', 'consultas', 'reprovados', 'n5k_ok')})\n"
        "    print('confiança:', m.get('confianca'))\n"
        "    print('bases externas:', {k: v.get('situacao') for k, v in (m.get('dominios') or {}).items()})\n"
        "    print('tamanho da saída:', m.get('tamanho_saida'))\n"
        "print(sorted(os.listdir(SAIDA)))"))
    for i, c in enumerate(cells):
        c["id"] = f"celula-{i}"
    nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"), language_info=dict(name="python"),
                            nutrelle=dict(kernel=KERNEL, variante=variante, sha256=_hash(arqs), embutidos=sorted(arqs), dominios=doms)))
    return nb


def monta_deteccao(variante):
    """Notebook da etapa 0 (14.3): baixa as fotos, roda o OWLv2 e a reserva por elipse, grava as tabelas de caixas e as
    folhas de contato. Não treina nada e não lê rótulo."""
    arqs = conteudo(variante)
    doms = LIBERADAS + (FECHADAS if variante == "push" else [])
    md = lambda t: dict(cell_type="markdown", metadata={}, source=t)
    code = lambda t: dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=t)
    aviso = ("**Versão do repositório: só bases liberadas** (Nutrition5k, Flavoria, LeFood-Set, GT Mensa). A completa é montada fora do repositório "
             "com `python3 monta_pacote.py push <pasta> --kernel r5a`." if variante == "repo" else
             "**Versão completa, montada fora do repositório** (as seis bases): contém listas de bases sem licença comercial. **Não commitar, não publicar.**")
    cells = [
        md("# Nutrelle — etapa 0 da quinta execução: a caixa da refeição em cada foto (seção 14.3)\n\n"
           "Notebook gerado por `eval/treino/kaggle/monta_pacote.py --kernel r5a`; **não edite à mão**. Baixa as fotos dos pratos de "
           "treino_v2 e de todas as bases externas (inclusive as partes de registro), roda o detector OWLv2 base "
           "(`google/owlv2-base-patch16-ensemble`, Apache 2.0) nas duas GPUs, com a reserva por elipse do OpenCV, e grava por base "
           "`caixas_r5_<base>.csv` (caixa, nota, método, janela do guia, sha256 de cada foto), as folhas de contato de 40 fotos "
           "(`folha_r5_<base>_<k>.jpg`, com a regra; `folha_r5_<base>_so_prato_<k>.jpg`, diagnóstico) e `resumo_r5.json`. "
           "Nenhum rótulo é lido.\n\n" + aviso + "\n\nKaggle: *GPU T4 x2* e *Internet → On*."),
        code("# Parâmetros\n"
             "import json, os, subprocess, sys\n"
             f"ORCAMENTO_H = {KERNELS['r5a']['orcamento']}\n"
             "SAIDA = '/kaggle/working' if os.path.isdir('/kaggle/working') else '/content/nutrelle_deteccao'\n"
             "EXTRA = []           # ex.: ['--dominios', 'flavoria,mensa']; ['--espelho', '<pasta>']\n"
             "SAIDA = os.environ.get('NUTRELLE_SAIDA') or SAIDA\n"
             "EXTRA = json.loads(os.environ.get('NUTRELLE_EXTRA') or 'null') or EXTRA\n"
             "for mod, pacote in (('transformers', 'transformers'), ('cv2', 'opencv-python-headless'), ('huggingface_hub', 'huggingface_hub')):\n"
             "    try:\n"
             "        __import__(mod)\n"
             "    except ImportError:\n"
             "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pacote], check=False)\n"
             f"os.makedirs('{DESTINO}/listas', exist_ok=True)"),
    ]
    for n in sorted(arqs, key=lambda n: (not n.endswith(".py"), n)):
        cells.append(code(_celula_arquivo(n, arqs[n])))
    cells.append(code(
        "import subprocess, sys\n"
        f"cmd = [sys.executable, '-u', '{DESTINO}/deteccao_r5.py', '--saida', SAIDA, '--orcamento-h', str(ORCAMENTO_H)] + EXTRA\n"
        "p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "for linha in p.stdout:\n"
        "    print(linha, end='')\n"
        "print('código de saída:', p.wait())"))
    cells.append(code(
        "import json, os\n"
        "r = os.path.join(SAIDA, 'resumo_r5.json')\n"
        "if os.path.exists(r):\n"
        "    for b, v in json.load(open(r))['bases'].items():\n"
        "        print(b, {k: v.get(k) for k in ('fotos', 'por_metodo', 'taxa_sem_prato', 'iou_profundidade_gravada', 'iou_profundidade_refeita', 'diagnostico_uniao_so_prato', 'situacao')})\n"
        "print(sorted(os.listdir(SAIDA)))"))
    for i, c in enumerate(cells):
        c["id"] = f"celula-{i}"
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"), language_info=dict(name="python"),
                              nutrelle=dict(kernel=KERNEL, variante=variante, sha256=_hash(arqs), embutidos=sorted(arqs), dominios=doms)))


def grava_nb(nb, arq):
    with open(arq, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def notebook():
    nb = monta("repo")
    os.makedirs(os.path.dirname(NOTEBOOK), exist_ok=True)
    grava_nb(nb, NOTEBOOK)
    meta_arq = os.path.join(os.path.dirname(NOTEBOOK), "kernel-metadata.json")
    if not os.path.exists(meta_arq):        # kernel novo: o kernel-metadata.json do repositório, com o usuário de exemplo
        meta = json.load(open(os.path.join(AQUI, "kernel-metadata.json")))
        meta.update(id=f"USUARIO_KAGGLE/{SLUG}", title=SLUG, code_file=os.path.basename(NOTEBOOK), dataset_sources=[], kernel_sources=[])
        with open(meta_arq, "w") as f:
            json.dump(meta, f, indent=2); f.write("\n")
    print(f"notebook do repositório: {os.path.getsize(NOTEBOOK) / 1e3:.0f} kB, bases {nb['metadata']['nutrelle']['dominios']}, sha256 {nb['metadata']['nutrelle']['sha256'][:16]}")


def dentro_do_repo(pasta):
    return os.path.commonpath([os.path.realpath(pasta), os.path.realpath(REPO)]) == os.path.realpath(REPO)


def push():
    """Notebook completo + kernel-metadata.json numa pasta FORA do repositório."""
    pasta = os.path.abspath(sys.argv[2])
    if dentro_do_repo(pasta):
        raise SystemExit("a pasta do push tem de ficar FORA do repositório (o notebook completo embute listas de bases sem licença comercial)")
    os.makedirs(pasta, exist_ok=True)
    nb = monta("push")
    grava_nb(nb, os.path.join(pasta, os.path.basename(NOTEBOOK)))
    meta = json.load(open(os.path.join(os.path.dirname(NOTEBOOK), "kernel-metadata.json")))
    usuario = _opcao("--usuario", "USUARIO_KAGGLE")
    meta.update(id=f"{usuario}/{SLUG}", title=SLUG, kernel_sources=[f"{usuario}/{k}" for k in KERNELS[KERNEL].get("fontes_kernel", [])])
    if _opcao("--espelho-omnifood"):
        meta["dataset_sources"] = [_opcao("--espelho-omnifood")]
    with open(os.path.join(pasta, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2); f.write("\n")
    confere_nb(nb, "push")
    print(f"push: {pasta} | notebook {os.path.getsize(os.path.join(pasta, os.path.basename(NOTEBOOK))) / 1e3:.0f} kB, bases {nb['metadata']['nutrelle']['dominios']} | "
          f"id {meta['id']} | dataset_sources {meta['dataset_sources']}")


def _embutidos(nb):
    """{nome: bytes} decodificados das células de um notebook."""
    out = {}
    for c in nb["cells"]:
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        if c["cell_type"] != "code":
            continue
        for m in RX_CELULA.finditer(src):
            out[m.group(1)] = lzma.decompress(base64.b85decode(m.group(2).replace("\n", "")))
    return out


def ignorado_pelo_git(caminho):
    return subprocess.run(["git", "-C", REPO, "check-ignore", "-q", caminho], capture_output=True).returncode == 0


def vazamentos(emb, texto_nb):
    """O que, do notebook do repositório, vem de uma base fechada (lista vazia = nada)."""
    erros = []
    for n in emb:
        if ignorado_pelo_git(os.path.join(AQUI, n)):
            erros.append(f"{n} é ignorado pelo git (base sem licença comercial)")
        if any(f"_{ch}." in n for ch in FECHADAS):
            erros.append(f"{n}: arquivo de base fechada")
    texto = b"\n".join(v for n, v in emb.items() if n.startswith("listas/")) + texto_nb.encode()
    for pre in PREFIXO_FECHADAS:
        if re.search(rb"(?<![A-Za-z0-9])" + pre.encode() + rb"[A-Za-z0-9]", texto):
            erros.append(f"id de base fechada ({pre}…)")
    for lic in (b"CC BY-NC", b"CC BY-SA", b"nao_comercial", b"autores_pedem_nao_treinar"):
        if lic in texto:
            erros.append(f"'{lic.decode()}' nas listas embutidas")
    return erros


def confere_nb(nb, variante):
    """Um notebook (do repositório ou do push) corresponde aos arquivos desta pasta e não vaza nada que não devia."""
    meta = nb["metadata"]["nutrelle"]
    assert meta["variante"] == variante, f"variante {meta['variante']} no lugar de {variante}"
    emb = _embutidos(nb)
    if variante == "repo":      # antes de tudo: nada das bases sem licença comercial no notebook do repositório
        texto_nb = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb["cells"]
                             if not (c["cell_type"] == "code" and (c["source"] if isinstance(c["source"], str) else "").startswith(("# treino_gpu.py", "# listas/"))))
        erros = vazamentos(emb, texto_nb)
        assert not erros, "o notebook do repositório embute dado de base fechada: " + "; ".join(erros)
        assert meta["dominios"] == LIBERADAS and meta.get("kernel", "r4") == KERNEL
    else:
        assert meta["dominios"] == LIBERADAS + FECHADAS
    esperado = conteudo(variante)
    assert set(emb) == set(esperado), f"arquivos embutidos diferentes: {sorted(set(emb) ^ set(esperado))}"
    for n in esperado:
        assert emb[n] == esperado[n], f"{n}: o notebook está desatualizado (rode `python3 monta_pacote.py {'notebook' if variante == 'repo' else 'push <pasta>'}`)"
    assert meta["sha256"] == _hash(esperado)
    for n, b in emb.items():    # as linhas que não são de treino não levam rótulo nenhum, em nenhuma base
        if re.fullmatch(r"listas/externo_\w+\.csv", n):
            ch = n[len("listas/externo_"):-4]
            L = list(csv.DictReader(io.StringIO(b.decode())))
            V = list(csv.DictReader(io.StringIO(emb[f"listas/divisao_r4_{ch}.csv"].decode())))
            for r, v in zip(L, V):
                if v["papel"] != "treino":
                    assert not any(r[c] for c in r if _alvo(c)) and not v["peso_nutricao"], f"{n}: rótulo numa linha de {v['papel']}"
            assert not any("fraca" in c or c in ("componentes", "cluster", "prato_principal") for c in list(L[0]) + list(V[0])), f"{n}: coluna que o kernel não lê"
    if variante == "repo":
        proc = json.loads(emb["listas/procedencia_r4.json"])
        assert sorted(proc) == sorted(LIBERADAS) and all(v["uso_declarado"] == "comercial_ok" for v in proc.values())
    return emb


def confere():
    """Confere o notebook do repositório de cada kernel (ou só o de --kernel) e, se dada, a pasta do push dele."""
    pasta = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    ks = [_opcao("--kernel")] if _opcao("--kernel") else [k for k, v in KERNELS.items() if os.path.exists(os.path.join(v["pasta"], v["notebook"]))]
    assert not pasta or len(ks) == 1, "com a pasta do push, diga qual kernel (--kernel)"
    for k in ks:
        usa_kernel(k)
        print(f"== {k}: {KERNELS[k]['nome']}")
        confere_kernel(pasta)
    print("confere: ok")


def confere_kernel(pasta):
    import tempfile
    import numpy as np
    nb = json.load(open(NOTEBOOK))
    emb = confere_nb(nb, "repo")
    assert os.path.getsize(NOTEBOOK) < 1_000_000, "notebook do repositório com mais de 1 MB"
    meta = json.load(open(os.path.join(os.path.dirname(NOTEBOOK), "kernel-metadata.json")))
    assert meta["code_file"] == os.path.basename(NOTEBOOK) and meta["enable_gpu"] and meta["enable_internet"] and meta["id"].endswith("/" + SLUG)
    # o teste2 não pode cruzar com nenhuma lista de treino e tem de ser o mesmo da execução 2
    ids = lambda arq: [r["dish_id"] for r in csv.DictReader(open(os.path.join(LISTAS, arq)))]
    tr, av, av2 = ids("treino.csv"), ids("aval_ids.csv"), ids("aval2_ids.csv")      # as duas primeiras não vão no pacote, mas o teste2 não pode cruzar com elas
    assert len(set(av2)) == len(av2) and not set(av2) & (set(tr) | set(av)), "aval2_ids.csv cruza com o treino ou com a avaliação antiga"
    v2 = list(csv.DictReader(open(os.path.join(LISTAS, "treino_v2.csv"))))
    assert len({r["dish_id"] for r in v2}) == len(v2) and not {r["dish_id"] for r in v2} & set(av2), "treino_v2.csv cruza com o teste2"
    cad = {}
    for r in v2:        # cada cadeia inteira num fold só; frações por fonte somam 1
        assert cad.setdefault(r["cadeia"], r["fold"]) == r["fold"], "treino_v2: cadeia em dois folds"
        assert abs(sum(float(r[c]) for c in r if c.startswith("ff_")) - 1) < 2e-3, "treino_v2: frações por fonte não somam 1"
    assert set(tr) <= {r["dish_id"] for r in v2}
    sys.dont_write_bytecode = True
    sys.path.insert(0, AQUI)
    import treino_gpu as G
    D = G.Dados(LISTAS, rodada="4", dominios=[])
    assert D.hash2 == HASH_TESTE2 and len(av2) == N_TESTE2 and list(D.conj) == ["teste2"], "aval2_ids.csv mudou: o teste2 tem de ficar intocado"
    if "listas/ref_d1_r3.csv" in emb:
        ref = list(csv.DictReader(io.StringIO(emb["listas/ref_d1_r3.csv"].decode())))
        assert [r["dish_id"] for r in ref] == [r["dish_id"] for r in v2], "ref_d1_r3.csv fora da ordem do treino_v2"
    assert not any("aval2" in n or "teste2" in n for n in emb) or KERNEL == "r4", "a quinta execução não leva o teste2 (14.1)"

    def le_como_no_kaggle(emb, esperadas):
        """O kernel lê as listas embutidas como no Kaggle (as conferências de le_dominio4: nenhum prato do Nutrition5k,
        papéis, folds, licença) e o teste2 e as listas do Nutrition5k não mudam."""
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(os.path.join(t, "listas"))
            for n, b in emb.items():
                if n.startswith("listas/"):
                    open(os.path.join(t, n), "wb").write(b)
            E = G.Dados(os.path.join(t, "listas"), rodada="4")
            assert sorted(E.dom) == sorted(esperadas) and sorted(E.dom_fora) == sorted(set(G.DOMINIOS4) - set(esperadas)), E.dom_fora
            assert E.hash == D.hash and (E.hash2 == HASH_TESTE2 and E.todos == D.todos if KERNEL == "r4" else E.hash2 is None), "o pacote mudou as listas do Nutrition5k ou o teste2"
            for ch, d in E.dom.items():
                assert d["lista_enxuta"] and np.isnan(d["T"][d["reg"]]).all() and np.isnan(d["FF"][d["reg"]]).all()
                print(f"  {ch:9s} {d['resumo']} | {d['licenca']}, {d['uso_declarado']} | lista inteira {d['sha256_lista'][:16]}")
    le_como_no_kaggle(emb, LIBERADAS)
    print(f"listas: treino {len(tr)}, avaliação {len(av)}, teste2 {len(av2)} (hash {D.hash2}, intocado), treino_v2 {len(v2)} em {len(cad)} cadeias; interseções com o teste2 vazias")
    print(f"notebook do repositório: {os.path.getsize(NOTEBOOK) / 1e3:.0f} kB, só bases liberadas {LIBERADAS}; nenhum arquivo ignorado pelo git, nenhum id nem licença das bases fechadas")
    if pasta:       # notebook do push
        pasta = os.path.abspath(pasta)
        assert not dentro_do_repo(pasta), "a pasta do push está dentro do repositório"
        emb_p = confere_nb(json.load(open(os.path.join(pasta, os.path.basename(NOTEBOOK)))), "push")
        assert os.path.getsize(os.path.join(pasta, os.path.basename(NOTEBOOK))) < 1_000_000, "notebook do push com mais de 1 MB"
        le_como_no_kaggle(emb_p, LIBERADAS + FECHADAS)
        mp = json.load(open(os.path.join(pasta, "kernel-metadata.json")))
        assert mp["id"].endswith("/" + SLUG) and mp["code_file"] == os.path.basename(NOTEBOOK)
        print(f"notebook do push em {pasta}: em dia, seis bases, {os.path.getsize(os.path.join(pasta, os.path.basename(NOTEBOOK))) / 1e6:.2f} MB; "
              f"id {mp['id']}, dataset_sources {mp['dataset_sources']}")
    elif meta["id"].startswith("USUARIO_KAGGLE/"):
        print(f"o push usa a pasta montada por `monta_pacote.py push <pasta> --kernel {KERNEL} --usuario <seu usuário>` (fora do repositório), não esta")


if __name__ == "__main__":
    if _opcao("--kernel"):
        usa_kernel(_opcao("--kernel"))
    {"listas": listas, "notebook": notebook, "push": push, "confere": confere}[sys.argv[1] if len(sys.argv) > 1 else "confere"]()
