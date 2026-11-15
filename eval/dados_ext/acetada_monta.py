#!/usr/bin/env python3
"""Monta a lista de treino externo do ACETADA (esquema de treino_v2 + colunas de 12.4 + procedência).

Dataset: Coburn, B.; He, J.; Rollo, M. E.; Dhaliwal, S. S.; Kerr, D. A.; Zhu, F. Evaluating Large Multimodal Models for
Nutrition Analysis: A New Benchmark Enriched with Contextual Metadata. IEEE BHI 2025 (arXiv:2507.07048).
https://skynet.ecn.purdue.edu/~coburn6/ACETADA/ — licença CC BY-NC 4.0 (página oficial): NÃO COMERCIAL.

O que é: estudo de alimentação controlada em Perth (Austrália), 152 adultos, 3 dias cada. As refeições foram preparadas
no laboratório e PESADAS item a item, antes e depois (0,1 g); as fotos são de celular (app mFR24), ANTES e DEPOIS.
Os nutrientes de cada item são o peso × tabela (densidade constante por item). 806 refeições, 1 a 15 itens cada.

VISTA: foto OBLÍQUA (uns 30 a 45 graus) da BANDEJA INTEIRA: dois ou três pratos, bebidas, sachês, cartão de cor e
número do participante. Não é foto de cima nem de um prato só (conferido em 20 fotos abertas).

Onde: https://skynet.ecn.purdue.edu/~coburn6/ACETADA/dataset/ACETADA-release.zip (4.941.836.247 bytes). Dentro dele:
ACETADA-release/ACETADA-HF-dataset.csv e ACETADA-release/ACETADA/<participante>/<data_hora>.jpg.
O servidor aceita Range: dá para tirar só o CSV e as fotos escolhidas (varredura2/scripts/zipremoto.py).

Uso:
  python3 -I acetada_monta.py monta <ACETADA-HF-dataset.csv> [--saida <csv>]       # grava também acetada_itens.csv

UMA LINHA POR REFEIÇÃO (foto "antes"). Os nutrientes do CSV são do CONSUMIDO; a foto "antes" mostra o SERVIDO.
Por isso cada item vira servido: proteína_servida = proteína_consumida × antes / consumido. Item com consumo zero
usa a densidade do mesmo item em outra refeição (é constante por item). natureza_rotulo = pesado_x_tabela.
`massa` e os alvos de nutrição contam só a COMIDA (sem bebidas, como em externo_mensa.csv); as bebidas vão em
bebida_g e proteina_bebidas_g. Regras:
  A1  item sem peso antes: a refeição sai;
  A2  item com consumo zero e sem densidade conhecida em nenhuma refeição: alvos de nutrição vazios (massa fica);
  A3  refeição só de bebida: sai;
  A4  a mesma foto "antes" em duas refeições (erro do CSV: café e jantar com a mesma foto; 2 casos): as duas saem.
m_* e ff_* saem do mapa de itens (ITENS, 192 nomes; acetada_itens.csv). Parte segurada: regra de reserva de 12.5
por participante (sha1(grupo) mod 5 == 0 -> teste_ext).
"""
import csv, hashlib, os, re, sys
from collections import Counter, defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import FONTES, GRUPOS, confere_esquema  # noqa: E402

SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_acetada.csv")
MAPA = os.path.join(AQUI, "acetada_itens.csv")
LICENCA, USO = "CC BY-NC 4.0", "nao_comercial"
URL = "https://skynet.ecn.purdue.edu/~coburn6/ACETADA/dataset/ACETADA-release.zip"
# (regex no nome em minúsculas, grupo, fonte, tipo); a primeira que casa vale. tipo: "" | bebida | condimento
ITENS = [
    (r"ham|cheese melt|toasted sandwich|toastie|wrap|stir fry|ndls|noodle", "misto", "prato_misto", ""),
    (r"muffin(?! plain)|lemon slice", "outros", "sem_proteina", ""),
    (r"potato", "amido", "sem_proteina", ""),
    (r"soy milk", "outros", "leguminosa_soja", "bebida"),
    (r"^milk|lactose|choc milk|hot choc|almond milk", "outros", "laticinio", "bebida"),
    (r"water|coffee|tea\b|tea \(|herbal|coke|juice|lemonade|squash|ginger beer", "outros", "sem_proteina", "bebida"),
    (r"soy sauce|wasabi|pickled ginger|sugar|sweetener|butter|margarine|^jam|honey|vegemite|marmalade|dressing|gravy|hoisin|^lemon$",
     "gordura_molho", "sem_proteina", "condimento"),
    (r"peanut paste", "gordura_molho", "nozes_sementes", "condimento"),
    (r"y[o]?u?gh?urt", "laticinio", "laticinio", ""),
    (r"cheddar", "laticinio", "laticinio", ""),
    (r"^roast chicken", "proteina_animal", "frango_peru", ""),
    (r"^roast beef", "proteina_animal", "bovina", ""),
    (r"eggs?- ?poached|poached eggs", "proteina_animal", "ovo", ""),
    (r"steamed veg|grilled mushroom|grilled tomato|green salad|garden salad|coleslaw|avocado",
     "vegetal", "sem_proteina", ""),
    (r"greek salad", "vegetal", "laticinio", ""),
    (r"fruit salad|peach|banana|blueberr|strawberr", "vegetal", "sem_proteina", ""),
    (r"toast|bread roll|muffin plain|eng muffin|english muffin plain|fruit bread", "amido", "sem_proteina", ""),
    (r"muesli|cornflakes|weet-bix|special k|porridge|oats|nutrigrain", "amido", "sem_proteina", ""),
    (r"muffin|brownie|cookie|cake|anzac|ice ?cream|zucchini slice$", "outros", "sem_proteina", ""),
    (r".", "misto", "prato_misto", ""),   # o resto é prato pronto com a fonte misturada (sanduíche, sushi, wrap, massa, curry...)
]


def classifica(nome):
    n = nome.lower().strip()
    for rx, g, f, t in ITENS:
        if re.search(rx, n):
            return g, f, t
    raise KeyError(nome)


def f(v):
    return float(v) if v not in ("", None) else None


def cmd_monta(origem, saida):
    with open(origem, encoding="utf-8-sig", newline="") as fh:
        r = list(csv.DictReader(fh))
    itens = lambda x: [(x[f"food_item_{i}_name"].strip(), f(x[f"food_item_{i}_before_portion_g"]), f(x[f"food_item_{i}_consumed_g"]),
                        f(x[f"food_item_{i}_protein_g"]), f(x[f"food_item_{i}_energy_kcal"]), f(x[f"food_item_{i}_fat_g"]),
                        f(x[f"food_item_{i}_carbs_g"])) for i in range(1, int(x["food_item_count"]) + 1)]
    dens = defaultdict(list)                         # g de nutriente por g consumido, por item
    for x in r:
        for n, b, c, p, k, g, cb in itens(x):
            if c and c > 0 and None not in (p, k, g, cb):
                dens[n.lower()].append((p / c, k / c, g / c, cb / c))
    dens = {n: tuple(sum(v[j] for v in vs) / len(vs) for j in range(4)) for n, vs in dens.items()}
    nomes = Counter(n for x in r for n, *_ in itens(x))
    with open(MAPA, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["item", "refeicoes", "grupo", "fonte", "tipo", "proteina_100g"])
        for n in sorted(nomes):
            g, fo, t = classifica(n)
            d = dens.get(n.lower())
            w.writerow([n, nomes[n], g, fo, t, f"{100 * d[0]:.2f}" if d else ""])
    base = confere_esquema()
    cab = base + ["dominio", "grupo", "vista", "split_ext", "arquivo", "arquivo_depois", "url_zip", "participante", "data",
                  "tipo_refeicao", "componentes", "n_componentes", "bebida_g", "proteina_bebidas_g", "condimento_g",
                  "proteina_consumida_g", "maior_fonte_g", "natureza_rotulo", "avisos", "licenca", "uso_declarado"]
    cont = Counter(refeicoes=len(r)); linhas = []
    usos = Counter(x["before_filepath"] for x in r)
    for x in r:
        if usos[x["before_filepath"]] > 1:
            cont["A4_foto_de_duas_refeicoes"] += 1; continue
        its = itens(x)
        if any(b is None for _, b, *_ in its):
            cont["A1_item_sem_peso"] += 1; continue
        mg = dict.fromkeys(GRUPOS, 0.0); pg = dict.fromkeys(GRUPOS, 0.0); ff = dict.fromkeys(FONTES, 0.0)
        tot = dict(p=0.0, k=0.0, g=0.0, c=0.0); massa = beb = pbeb = cond = maior = 0.0; falta = False; comp = []
        for n, b, c, p, k, g, cb in its:
            gr, fo, tipo = classifica(n)
            if c and c > 0 and None not in (p, k, g, cb):
                s = (p / c, k / c, g / c, cb / c)
            else:
                s = dens.get(n.lower())
            if s is None:
                falta = True; s = (0, 0, 0, 0)
            pi = b * s[0]
            comp.append(f"{n.replace(';', ',')}:{b:.0f}")
            if tipo == "bebida":
                beb += b; pbeb += pi; continue
            massa += b; mg[gr] += b; pg[gr] += pi; ff[fo] += b
            cond += b if tipo == "condimento" else 0
            tot["p"] += pi; tot["k"] += b * s[1]; tot["g"] += b * s[2]; tot["c"] += b * s[3]
            if fo != "sem_proteina":
                maior = max(maior, pi)
        if massa <= 0:
            cont["A3_so_bebida"] += 1; continue
        cont["A2_sem_densidade"] += falta
        grupo = "act_" + x["participant_id"]
        split = "teste_ext" if int(hashlib.sha1(grupo.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
        nut = [""] * 4 if falta else [f"{tot['p']:.2f}", f"{tot['k']:.1f}", f"{tot['g']:.2f}", f"{tot['c']:.2f}"]
        did = "act_" + x["participant_id"] + "_" + os.path.basename(x["before_filepath"]).rsplit(".", 1)[0]
        linhas.append([did, -1, -1, "", f"{massa:.1f}"] + nut + [f"{mg[k]:.1f}" for k in GRUPOS]
                      + ([""] * len(GRUPOS) if falta else [f"{pg[k]:.2f}" for k in GRUPOS]) + [f"{ff[k] / massa:.4f}" for k in FONTES]
                      + ["acetada", grupo, "obliqua_bandeja", split, "ACETADA-release/" + x["before_filepath"],
                         "ACETADA-release/" + x["after_filepath"] if x.get("after_filepath") else "", URL, x["participant_id"],
                         x["date"], x["meal_type"], ";".join(comp), len(comp), f"{beb:.0f}", f"{pbeb:.1f}", f"{cond:.0f}",
                         x["total_protein_g"], f"{maior:.1f}", "pesado_x_tabela", "sem_densidade" if falta else "", LICENCA, USO])
    assert len({l[0] for l in linhas}) == len(linhas)
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    i = {k: cab.index(k) for k in cab}
    p = [float(l[i["proteina"]]) for l in linhas if l[i["proteina"]]]
    faixa = [sum(a < v <= b for v in p) for a, b in ((-1, 5), (5, 15), (15, 25), (25, 40), (40, 1e9))]
    m = sorted(float(l[i["massa"]]) for l in linhas)
    print("regras:", dict(cont))
    print(f"refeições (linhas): {len(linhas)} | participantes: {len({l[i['grupo']] for l in linhas})} | split: {dict(Counter(l[i['split_ext']] for l in linhas))}")
    print(f"  massa servida sem bebidas (g): p10 {m[len(m) // 10]:.0f} mediana {m[len(m) // 2]:.0f} p90 {m[9 * len(m) // 10]:.0f}")
    print(f"  proteína servida sem bebidas: faixas 0-5/5-15/15-25/25-40/>40 {faixa} | > 25 g: {sum(v > 25 for v in p)}"
          f" | um item-fonte > 25 g: {sum(float(l[i['maior_fonte_g']]) > 25 for l in linhas)}")
    print("  por faixa e lado:", {s: [sum(a < float(l[i['proteina']]) <= b for l in linhas if l[i['proteina']] and l[i['split_ext']] == s)
                                       for a, b in ((-1, 5), (5, 15), (15, 25), (25, 40), (40, 1e9))] for s in ("treino_ext", "teste_ext")})
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas) | mapa:", os.path.relpath(MAPA), len(nomes), "itens")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
