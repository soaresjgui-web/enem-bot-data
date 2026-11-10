#!/usr/bin/env python3
"""Monta o pacote do Kaggle/Colab: listas de pratos e o notebook que embute o script e as listas.

Uso:
  python3 monta_pacote.py listas     regrava listas/treino.csv e listas/aval_ids.csv a partir de <TD>/meta
                                     (zen_treino.csv, zen_aval.csv, gravados por zen_dados.py) e dos folds de zen_comum.Dados
  python3 monta_pacote.py notebook   regrava nutrelle_treino_gpu.ipynb a partir de treino_gpu.py e das listas
  python3 monta_pacote.py confere    falha se o notebook não corresponde aos arquivos desta pasta (rode antes do push)

Por que embutir: `kaggle kernels push` envia um único arquivo de código. O notebook grava treino_gpu.py e as
listas em disco e chama o script; não depende de dataset anexado nem do repositório.

Listas (só ids e metadados públicos do Nutrition5k, CC BY 4.0; nenhuma imagem):
  treino.csv    os 1.684 pratos do treino zenital estrito, com cadeia, fold (GroupKFold por cadeia, o mesmo da CPU)
                e os alvos (massa, macros, massa e proteína por grupo de alimento)
  aval_ids.csv  prato_id, split, dish_id dos 450 pratos de avaliação, SEM gabarito
"""
import csv, hashlib, json, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
LISTAS = os.path.join(AQUI, "listas")
NOTEBOOK = os.path.join(AQUI, "nutrelle_treino_gpu.ipynb")
EMBUTIDOS = ["treino_gpu.py", "listas/treino.csv", "listas/aval_ids.csv"]
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


def notebook():
    md = lambda t: dict(cell_type="markdown", metadata={}, source=t)
    code = lambda t: dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=t)
    cells = [
        md("# Nutrelle — treino em GPU (seção 8: uma única foto de cima)\n\n"
           "Notebook gerado por `eval/treino/kaggle/monta_pacote.py`; **não edite à mão** (edite `treino_gpu.py` e gere de novo). "
           "Roda sem interação: baixa as fotos zenitais do Nutrition5k do bucket público, treina as configurações declaradas "
           "em `treino_gpu.py` até o orçamento de tempo e grava previsões, registros e os pesos do melhor modelo na pasta de saída.\n\n"
           "Kaggle: *Settings → Accelerator → GPU T4 x2* e *Internet → On*. Colab: *Ambiente de execução → GPU*. "
           "Instruções completas em `eval/treino/kaggle/LEIA-ME.md`."),
        code("# Parâmetros (os padrões servem para o Kaggle)\n"
             "import json, os\n"
             "ORCAMENTO_H = 11.0   # horas de relógio; o que não couber fica marcado como 'sem_orcamento' e pode ser retomado\n"
             "SAIDA = '/kaggle/working' if os.path.isdir('/kaggle/working') else '/content/nutrelle_saida'   # no Colab, aponte para o Drive para retomar\n"
             "EXTRA = []           # ex.: ['--so', 'ridge'] para rodar só parte; ['--retoma', '/content/drive/MyDrive/nutrelle_saida']\n"
             "# as variáveis de ambiente NUTRELLE_SAIDA e NUTRELLE_EXTRA (lista em JSON) substituem os valores acima sem editar o notebook\n"
             "SAIDA = os.environ.get('NUTRELLE_SAIDA') or SAIDA\n"
             "EXTRA = json.loads(os.environ.get('NUTRELLE_EXTRA') or 'null') or EXTRA\n"
             f"os.makedirs('{DESTINO}/listas', exist_ok=True)"),
    ]
    for n in EMBUTIDOS:
        cells.append(code(f"%%writefile {DESTINO}/{n}\n" + open(os.path.join(AQUI, n)).read()))
    cells.append(code(
        "import subprocess, sys\n"
        f"cmd = [sys.executable, '-u', '{DESTINO}/treino_gpu.py', '--saida', SAIDA, '--orcamento-h', str(ORCAMENTO_H)] + EXTRA\n"
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
        "    print('melhor modelo:', m['melhor_modelo'])\n"
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
        cab = f"%%writefile {DESTINO}/{n}\n"
        cel = [c for c in nb["cells"] if c["cell_type"] == "code" and c["source"].startswith(cab)]
        assert len(cel) == 1 and cel[0]["source"][len(cab):] == open(os.path.join(AQUI, n)).read(), n
    meta = json.load(open(os.path.join(AQUI, "kernel-metadata.json")))
    assert meta["code_file"] == os.path.basename(NOTEBOOK) and meta["enable_gpu"] and meta["enable_internet"]
    if meta["id"].startswith("USUARIO_KAGGLE/"):
        print("atenção: troque USUARIO_KAGGLE em kernel-metadata.json pelo seu usuário do Kaggle antes do push")
    print("confere: notebook e kernel-metadata.json em dia")


if __name__ == "__main__":
    {"listas": listas, "notebook": notebook, "confere": confere}[sys.argv[1] if len(sys.argv) > 1 else "confere"]()
