#!/usr/bin/env python3
"""Monta a lista de treino externo do OmniFood8K (esquema de treino_v2 + colunas de 12.4 + proveniência).

Dataset: Yu, D.; Min, W.; Jiang, Q.; Lin, X.; Jin, X.; Jiang, S. OmniFood8K: Single-Image Nutrition Estimation via
Hierarchical Frequency-Aligned Fusion. CVPR 2026 (arXiv:2604.12356). https://github.com/yudongjian/OmniFood8K-food
Licença: CC BY-NC 4.0 (README dos autores): NÃO COMERCIAL. Entra por decisão da fundadora, com `uso_declarado`.

Onde: Google Drive, id 1tUIa1jXMLxRGLyex_jVkmqmHneIMU0wG, arquivo 0-OminiFood8k.zip (2.731.585.789 bytes).
  https://drive.usercontent.google.com/download?id=1tUIa1jXMLxRGLyex_jVkmqmHneIMU0wG&export=download&confirm=t
Dentro do zip: 0-OminiFood8k/8036/<cena>/camera_<0..5>.jpg (640x480), 8k-result.xlsx, ingred-nutrition.xlsx,
train_new333.txt, test_new333.txt.

O que é: 141 pratos chineses cozidos em laboratório (ingredientes crus pesados um a um), 24 frutas e arroz,
servidos em porções pesadas num prato branco sobre mesa branca, de 1 a 8 porções por prato: 8.037 cenas. Cada cena
tem 6 fotos, TODAS de cima ou quase (mudam a altura, a posição e a rotação do prato).

Uso:
  python3 -I omnifood_monta.py mapa <ingred-nutrition.xlsx>            # regrava omnifood_ingredientes.csv
  python3 -I omnifood_monta.py monta <pasta com os 4 arquivos de rótulo> [--saida <csv>]

UMA LINHA POR CENA (não por foto: seriam 48 mil linhas). `arquivo_base` é a pasta da cena dentro do zip e
`quadros` lista as fotos que existem (omnifood_arquivos.csv); todas levam os mesmos rótulos e o mesmo `grupo`.

ATENÇÃO, proteína dos autores: em 52 dos 141 pratos, a proteína publicada em 8k-result.xlsx difere por mais de
2 vezes (para mais ou para menos) da conta feita com os ingredientes pesados e a tabela dos próprios autores
(peixe frito: 0,8 g contra 76 g; fungo preto: 56 g contra 2,8 g). Por isso os alvos de nutrição desta lista são
RECALCULADOS aqui: gramas crus de cada ingrediente x ingred-nutrition.xlsx, divididos pelo peso do prato pronto,
vezes a porção servida. O valor dos autores fica em `proteina_autores`. `natureza_rotulo` = pesado_x_tabela_recalculado.
Limites do recálculo (avisos por cena):
  osso_ou_casca              o ingrediente foi pesado com osso, casca ou concha e a tabela é da parte comestível
                             (asa, pé e coxa de frango, costela, peixe inteiro, camarão, amêijoa, amendoim e milho na
                             casca). Os nutrientes desses ingredientes levam uma fração comestível SUPOSTA (COMESTIVEL);
                             a massa da cena continua sendo a pesada, com osso;
  ingredientes_inconsistentes a soma dos ingredientes crus passa de 2,5 vezes o peso do prato pronto, ou um ingrediente
                             está com peso zero (erro de anotação; 2 pratos): os alvos de nutrição da cena ficam
                             VAZIOS; a massa fica. (Soma bem menor que o prato pronto é normal: sopa e macarrão.)
Massa por grupo e frações por fonte saem da composição de cada prato em gramas crus (o óleo conta, como no
Nutrition5k), reescalada para o peso servido.
Parte segurada: o teste dos autores (test_new333.txt) vira teste_ext.
"""
import csv, os, re, sys
from collections import Counter, defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import FONTES, GRUPOS, confere_esquema  # noqa: E402

MAPA = os.path.join(AQUI, "omnifood_ingredientes.csv")
ARQUIVOS = os.path.join(AQUI, "omnifood_arquivos.csv")
SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_omnifood.csv")
LICENCA, USO = "CC BY-NC 4.0", "nao_comercial"
URL = "https://drive.usercontent.google.com/download?id=1tUIa1jXMLxRGLyex_jVkmqmHneIMU0wG&export=download&confirm=t"
RAZAO_MAX = 2.5
# Fração comestível dos ingredientes pesados com osso, casca ou concha. A tabela dos autores é por 100 g de parte
# comestível, e o peso anotado é o do ingrediente inteiro. Valores SUPOSTOS (de memória das tabelas de composição;
# não vêm do dataset): servem para não superestimar a proteína em 30 a 60%; conferir antes de confiar no alvo.
COMESTIVEL = {"chicken mid wing": 0.65, "chicken wing tip": 0.60, "chicken feet": 0.60, "chicken leg": 0.69, "pork ribs": 0.72,
              "crucian carp": 0.54, "white fish": 0.60, "prawn": 0.60, "clams in shell": 0.40, "peanuts in shell": 0.70,
              "corn cob": 0.46}

# (regex no nome em minúsculas, grupo, fonte); a primeira que casa vale
REGRAS = [
    (r"cooking oil|soy sauce|vinegar|cooking wine|doubanjiang|oyster sauce", "gordura_molho", "sem_proteina"),
    (r"vegetarian chicken|tofu|bean curd|edamame|white beans|^peas$", "leguminosa", "leguminosa_soja"),
    (r"\begg\b", "proteina_animal", "ovo"),
    (r"sausage|pork|^meat$|pig blood", "proteina_animal", "suina_embutido"),
    (r"chicken", "proteina_animal", "frango_peru"),
    (r"beef", "proteina_animal", "bovina"),
    (r"white fish|carp|prawn|shrimp|clams", "proteina_animal", "peixe_marisco"),
    (r"peanut", "outros", "nozes_sementes"),
    (r"rice|noodles|vermicelli|potato|taro|yam|lotus root|corn|lily bulb", "amido", "sem_proteina"),
    (r".", "vegetal", "sem_proteina"),
]
OSSO_OU_CASCA = re.compile(r"clams in shell|prawn|chicken mid wing|chicken feet|chicken leg|chicken wing tip|pork ribs|crucian carp|"
                           r"white fish|peanuts in shell|corn cob")


def num(v):
    try:
        return float(re.sub(r"[^0-9.]", "", str(v)) or 0)
    except ValueError:
        return 0.0


def chave(nome):
    return re.sub(r"\s+", " ", str(nome).strip().lower())


def ler_tabela(xlsx):
    """nome -> (kJ, proteína, gordura, carboidrato) por 100 g. A coluna de kJ que vale é a numérica (a de texto está
    deslocada uma linha)."""
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, read_only=True, data_only=True).worksheets[0]
    out = {}
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if r[0] is not None and r[1]:
            out.setdefault(chave(r[1]), (num(r[3]), num(r[4]), num(r[5]), num(r[6])))
    return out


def cmd_mapa(xlsx):
    tab = ler_tabela(xlsx); linhas = []
    for nome, (kj, p, g, c) in sorted(tab.items()):
        grupo, fonte = next((gr, fo) for rx, gr, fo in REGRAS if re.search(rx, nome))
        osso = bool(OSSO_OU_CASCA.search(nome))
        assert osso == (nome in COMESTIVEL), nome
        linhas.append([nome, grupo, fonte, f"{kj:.0f}", f"{p:.1f}", f"{g:.1f}", f"{c:.1f}", int(osso), COMESTIVEL.get(nome, 1.0)])
    with open(MAPA, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["ingrediente", "grupo", "fonte", "kj_100g", "proteina_100g", "gordura_100g", "carbo_100g", "osso_ou_casca", "comestivel_suposto"])
        w.writerows(linhas)
    print(MAPA, len(linhas), "ingredientes |", dict(Counter(l[2] for l in linhas)))


def componentes(linha):
    return [(linha[6 + 6 * i], num(linha[7 + 6 * i])) for i in range(8) if linha[6 + 6 * i] not in ("", None)]


def cmd_monta(pasta, saida):
    import openpyxl
    with open(MAPA, encoding="utf-8", newline="") as f:
        ing = {r["ingrediente"]: r for r in csv.DictReader(f)}
    tab = ler_tabela(os.path.join(pasta, "ingred-nutrition.xlsx"))
    for n, r in ing.items():                                # o mapa gravado tem de bater com a tabela dos autores
        assert abs(float(r["proteina_100g"]) - tab[n][1]) < 0.06, n
    wb = openpyxl.load_workbook(os.path.join(pasta, "8k-result.xlsx"), read_only=True, data_only=True)
    cenas = {int(x[0]): x for x in list(wb["Sheet1"].iter_rows(values_only=True))[1:] if isinstance(x[0], (int, float))}
    # pratos: Sheet3 dá o id do prato, o nome e a cena em que ele aparece sozinho, com os ingredientes crus
    pratos = {}
    for x in list(wb["Sheet3"].iter_rows(values_only=True))[1:]:
        if x[1] is None:
            continue
        pid = str(x[1]).strip()
        if pid.startswith("f") or pid == "0":               # fruta ou arroz: a linha já é por 100 g
            p100 = dict(kcal=num(x[5]) / 4.184, proteina=num(x[6]), gordura=num(x[7]), carbo=num(x[8]))
            pratos[pid] = dict(nome=x[3], dens={k: v / 100 for k, v in p100.items()},
                               mg={"vegetal" if pid.startswith("f") else "amido": 1.0}, ff={"sem_proteina": 1.0},
                               pg={"vegetal" if pid.startswith("f") else "amido": num(x[6]) / 100}, pff=0.0, osso=False, ruim=False)
            continue
        cozido = num(x[4]); comp = componentes(cenas[int(x[0])])
        cru = sum(g for _, g in comp)
        tot = dict(kcal=0.0, proteina=0.0, gordura=0.0, carbo=0.0); mg = defaultdict(float); ff = defaultdict(float)
        pg = defaultdict(float); pff = 0.0; osso = False
        for nome, g in comp:
            r = ing[chave(nome)]
            gc = g * float(r["comestivel_suposto"])          # gramas comestíveis, para os nutrientes
            p = gc * float(r["proteina_100g"]) / 100
            tot["kcal"] += gc * float(r["kj_100g"]) / 100 / 4.184; tot["proteina"] += p
            tot["gordura"] += gc * float(r["gordura_100g"]) / 100; tot["carbo"] += gc * float(r["carbo_100g"]) / 100
            mg[r["grupo"]] += g / cru; ff[r["fonte"]] += g / cru; pg[r["grupo"]] += p / cozido
            pff += p / cozido if r["fonte"] != "sem_proteina" else 0
            osso = osso or (r["osso_ou_casca"] == "1" and g / cru > 0.2)
        pratos[pid] = dict(nome=x[3], dens={k: v / cozido for k, v in tot.items()}, mg=dict(mg), ff=dict(ff), pg=dict(pg),
                           pff=pff, osso=osso, ruim=cru / cozido > RAZAO_MAX or any(g <= 0 for _, g in comp),
                           autores=num(x[6]) / cozido, proprio=int(x[0]))
    proprio = {p["proprio"]: pid for pid, p in pratos.items() if "proprio" in p}
    div = [pid for pid, p in pratos.items() if "autores" in p and not p["ruim"] and p["dens"]["proteina"] > 0
           and not (1 / 2 <= p["autores"] / p["dens"]["proteina"] <= 2)]         # com a fração comestível já aplicada
    split = {}
    for nome, lado in (("train_new333.txt", "train"), ("test_new333.txt", "test")):
        for l in open(os.path.join(pasta, nome), encoding="utf-8"):
            if l.split():
                split[int(float(l.split()[0]))] = lado
    with open(ARQUIVOS, encoding="utf-8", newline="") as f:
        quadros = {int(r["cena"]): r["quadros"] for r in csv.DictReader(f)}
    base = confere_esquema()
    cab = base + ["dominio", "grupo", "vista", "split_ext", "arquivo_base", "quadros", "n_quadros", "componentes", "n_componentes",
                  "natureza_rotulo", "proteina_autores", "proteina_fontes_g", "avisos", "licenca", "uso_declarado"]
    cont = Counter(cenas=len(cenas)); linhas = []
    for cid in sorted(cenas):
        x = cenas[cid]
        if cid not in quadros:
            cont["sem_foto"] += 1; continue
        if cid in proprio:                                   # cena de um prato só, descrita por ingredientes
            comp = [(proprio[cid], num(x[1]))]
        else:
            comp = [(str(c).strip() if isinstance(c, str) else str(int(c)), g) for c, g in componentes(x)]
        if any(c not in pratos for c, _ in comp):
            cont["componente_desconhecido"] += 1; continue
        massa = sum(g for _, g in comp)
        if massa <= 0:
            cont["massa_zero"] += 1; continue
        alvo = {k: sum(g * pratos[c]["dens"][k] for c, g in comp) for k in ("kcal", "proteina", "gordura", "carbo")}
        mg = {k: sum(g * pratos[c]["mg"].get(k, 0) for c, g in comp) for k in GRUPOS}
        pg = {k: sum(g * pratos[c]["pg"].get(k, 0) for c, g in comp) for k in GRUPOS}
        ff = {k: sum(g * pratos[c]["ff"].get(k, 0) for c, g in comp) / massa for k in FONTES}
        pff = sum(g * pratos[c]["pff"] for c, g in comp)
        ruim = any(pratos[c]["ruim"] for c, _ in comp)
        avisos = [a for a, s in (("ingredientes_inconsistentes", ruim), ("osso_ou_casca", any(pratos[c]["osso"] for c, _ in comp)),
                                 ("autores_divergem", any(c in div for c, _ in comp)),
                                 ("massa_difere_do_total", abs(massa - num(x[1])) > 1)) if s]
        cont["ingredientes_inconsistentes"] += ruim
        lado = split.get(cid, "")
        nq = quadros[cid].count(";") + 1
        nut = ["", "", "", ""] if ruim else [f"{alvo['proteina']:.2f}", f"{alvo['kcal']:.1f}", f"{alvo['gordura']:.2f}", f"{alvo['carbo']:.2f}"]
        linhas.append([f"omn_{cid:04d}", -1, -1, lado, f"{massa:.1f}"] + nut
                      + [f"{mg[k]:.1f}" for k in GRUPOS] + ([""] * len(GRUPOS) if ruim else [f"{pg[k]:.2f}" for k in GRUPOS])
                      + [f"{ff[k]:.4f}" for k in FONTES]
                      + ["omnifood8k", f"omn_{cid:04d}", "zenital", "teste_ext" if lado == "test" else "treino_ext",
                         f"0-OminiFood8k/8036/{cid}/", quadros[cid], nq,
                         ";".join(f"{pratos[c]['nome']}:{g:.0f}" for c, g in comp), len(comp),
                         "pesado_x_tabela_recalculado", f"{num(x[3]):.2f}", "" if ruim else f"{pff:.2f}", "|".join(avisos), LICENCA, USO])
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    i = {k: cab.index(k) for k in cab}
    faixa = lambda v: [sum(a < x <= b for x in v) for a, b in ((-1, 5), (5, 15), (15, 25), (25, 40), (40, 1e9))]
    p = [float(l[i["proteina"]]) for l in linhas if l[i["proteina"]] != ""]
    pa = [float(l[i["proteina_autores"]]) for l in linhas]
    m = sorted(float(l[i["massa"]]) for l in linhas); q = lambda a: m[int(a * (len(m) - 1))]
    print("contagens:", dict(sorted(cont.items())), "| pratos com proteína dos autores divergente:", len(div), "de", sum("autores" in v for v in pratos.values()))
    print(f"cenas (linhas): {len(linhas)} | fotos: {sum(l[i['n_quadros']] for l in linhas)} | split: {dict(Counter(l[i['split_ext']] for l in linhas))}")
    print(f"  massa (g): p10 {q(.1):.0f} mediana {q(.5):.0f} p90 {q(.9):.0f} máx {m[-1]:.0f}")
    print(f"  proteína recalculada, faixas 0-5/5-15/15-25/25-40/>40: {faixa(p)} | > 25 g: {sum(v > 25 for v in p)}"
          f" | > 25 g só das fontes: {sum(float(l[i['proteina_fontes_g']]) > 25 for l in linhas if l[i['proteina_fontes_g']] != '')}")
    print(f"  proteína dos autores, mesmas faixas: {faixa(pa)} | > 25 g: {sum(v > 25 for v in pa)}")
    print("  avisos (cenas):", dict(Counter(a for l in linhas for a in l[i['avisos']].split('|') if a)))
    semosso = [float(l[i["proteina"]]) for l in linhas if l[i["proteina"]] != "" and "osso_ou_casca" not in l[i["avisos"]]]
    print(f"  > 25 g sem o aviso de osso ou casca: {sum(v > 25 for v in semosso)} de {len(semosso)}")
    print("  cenas com a fonte (>= 20 g):", {k: sum(float(l[i[f'ff_{k}']]) * float(l[i['massa']]) >= 20 for l in linhas) for k in FONTES})
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "mapa":
        cmd_mapa(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
