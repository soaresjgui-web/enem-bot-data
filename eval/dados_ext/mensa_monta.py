#!/usr/bin/env python3
"""Monta a lista de treino externo do "VLM Food Benchmark" (GT Mensa 2025), no esquema de externo_flavoria.csv.

Dataset: Marchetti, T.; Capezzone, S.; Esposito, R.; De Spirito, M.; Maulucci, G. Benchmarking Open-Source
Vision-Language Models for Food Image Recognition and Portion Size Estimation. Nutrients 2026, 18(21), 3582.
https://doi.org/10.3390/nu18213582 — dados em https://github.com/Metabolic-Intelligence-Lab/vlm-food-benchmark,
licença CC BY 4.0 (arquivo LICENSE-DATA do repositório; atribuição obrigatória).

O que é: 200 fotos de celular (lado maior de até 748 px), quase todas de cima, da BANDEJA INTEIRA de um refeitório
universitário italiano: primeiro prato, segundo, acompanhamento, pão, iogurte, fruta, água. 36 fotos são de um
item só (pote de vidro, embalagem). Cada item foi pesado em balança de cozinha na hora, com a tara do prato
descontada: 870 itens, com nome em inglês. Seis pessoas fotografaram.

Uso:
  python3 -I mensa_monta.py mapa <labels_flat.csv>     # regrava mensa_componentes.csv a partir das regras abaixo
  python3 -I mensa_monta.py monta <labels_flat.csv> <image_participants.csv> [--saida <csv>]
Os dois CSV vêm de https://raw.githubusercontent.com/Metabolic-Intelligence-Lab/vlm-food-benchmark/main/
(labels_flat.csv e GT_Mensa_2025/image_participants.csv); as imagens, de .../GT_Mensa_2025/resized_images/NNN.jpg
(coluna `url` da lista). Nada de imagem é lido aqui.

UMA LINHA POR FOTO (uma bandeja). `massa` = soma dos itens pesados, SEM as bebidas (água, café, suco), que vão em
`bebida_g`. Itens em embalagem fechada (iogurte, biscoito, cracker) entram na massa e também em `embalado_g`.
`grupo` = participante (quem fotografou; são só seis). Proteína: só a estimativa FRACA (nome -> tabela), em colunas
à parte; `proteina` fica vazia.

O mapa de nomes (mensa_componentes.csv) sai de REGRAS, na ordem em que estão escritas; a primeira que casa vale.
Cada linha do mapa guarda a regra que casou, para revisão. Convenções iguais às do Flavoria: peça de proteína vai
para a fonte dela; proteína cozida em massa, arroz, torta ou sopa vai para prato_misto; massa só com molho de
tomate ou legumes é amido.
"""
import csv, hashlib, os, re, sys
from collections import Counter, defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import FONTES, GRUPOS, confere_esquema, densidade, tabelas  # noqa: E402

MAPA = os.path.join(AQUI, "mensa_componentes.csv")
SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_mensa.csv")
URL = "https://raw.githubusercontent.com/Metabolic-Intelligence-Lab/vlm-food-benchmark/main/GT_Mensa_2025/resized_images/"
PROTEINA_FRACA_ALTA_G = 25.0

SOPA_LEG = "0.6*n5k:mixed vegetables|0.4*DILUENTE"
SOPA_LEGUMINOSA = "0.4*n5k:lentils|0.6*DILUENTE"
VEG_BURGER = "0.3*n5k:veggie burgers|0.7*n5k:mixed vegetables"
MASSA_MOLHO = "0.8*n5k:pasta|0.2*n5k:pasta sauce"
MASSA_MOLHO_PARM = "0.77*n5k:pasta|0.2*n5k:pasta sauce|0.03*n5k:parmesan cheese"
EMPANADO_CARNE = "0.6*n5k:pork chops|0.25*n5k:bread crumbs|0.15*DILUENTE"

# (regra, regex sobre o nome em minúsculas, grupo, fonte, referência, confiança, tipo) — tipo: "" | bebida | embalado
REGRAS = [
    ("bebida", r"^water$|coffee|juice", "outros", "sem_proteina", "DILUENTE", "A", "bebida"),
    ("iogurte", r"yogurt", "laticinio", "laticinio", "T_IOGURTE", "B", "embalado"),
    ("biscoito_doce", r"cookies|sweet snacks|panettone", "outros", "sem_proteina", "n5k:cookies", "B", "embalado"),
    ("cracker", r"crackers|breadsticks", "amido", "sem_proteina", "n5k:crackers", "B", "embalado"),
    ("pao_recheado", r"crouton with", "misto", "prato_misto", "0.6*n5k:white bread|0.2*n5k:mozzarella cheese|0.2*n5k:pasta sauce", "C", ""),
    ("pao", r"^(wholemeal )?bread$", "amido", "sem_proteina", "taco:Pão, trigo, francês", "A", ""),
    ("maionese", r"mayonnaise", "gordura_molho", "sem_proteina", "n5k:mayonnaise", "A", ""),
    # sopas
    ("sopa_com_carne", r"soup.*sausage", "misto", "prato_misto", "0.3*n5k:bean(seed)|0.1*n5k:sausage|0.6*DILUENTE", "C", ""),
    ("sopa_leguminosa_parm", r"(legume|lentil|chickpea|bean).*soup.*parmesan|soup with chickpeas and parmesan", "misto", "prato_misto",
     "0.38*n5k:lentils|0.04*n5k:parmesan cheese|0.58*DILUENTE", "C", ""),
    ("sopa_leguminosa", r"(legume|lentil|chickpea|bean).*soup", "misto", "prato_misto", SOPA_LEGUMINOSA, "C", ""),
    ("sopa_legumes_parm", r"soup.*parmesan", "misto", "prato_misto", "0.57*n5k:mixed vegetables|0.04*n5k:parmesan cheese|0.39*DILUENTE", "C", ""),
    ("sopa_legumes", r"soup", "misto", "prato_misto", SOPA_LEG, "C", ""),
    # pratos prontos
    ("chili", r"chili con carne", "misto", "prato_misto", "0.3*n5k:ground beef|0.3*n5k:pinto beans|0.4*n5k:pasta sauce", "C", ""),
    ("quiche_torta", r"quiche|savoury pie|vegetable pie|^pie with", "misto", "prato_misto", "0.5*n5k:quiche|0.5*n5k:mixed vegetables", "C", ""),
    ("crepe", r"crepes?", "misto", "prato_misto", "0.5*n5k:crepes|0.25*n5k:mozzarella cheese|0.25*n5k:spinach (cooked)", "C", ""),
    ("pizza", r"pizza", "misto", "prato_misto", "n5k:cheese pizza", "C", ""),
    ("pao_chato_queijo", r"flatbread with cheese", "misto", "prato_misto", "0.7*n5k:flatbread|0.15*n5k:cheese|0.15*n5k:tomatoes", "C", ""),
    ("pao_chato_legumes", r"flat ?bread|wrap", "misto", "prato_misto", "0.7*n5k:flatbread|0.3*n5k:mixed vegetables", "C", ""),
    ("dumpling", r"dumplings", "misto", "prato_misto", "n5k:dumplings", "B", ""),
    ("pimentao_recheado", r"stuffed bell pepper", "misto", "prato_misto", "0.5*n5k:bell peppers|0.3*n5k:rice|0.2*n5k:ground beef", "C", ""),
    ("bolo_batata_ovo", r"potato cake with egg", "misto", "prato_misto", "0.7*n5k:potatoes|0.3*n5k:eggs", "C", ""),
    ("atum_feijao", r"tuna and white beans", "misto", "prato_misto", "0.4*n5k:tuna|0.6*n5k:white beans", "C", ""),
    ("cuscuz_peixe", r"cous ?cous with fish|mackerel with cous", "misto", "prato_misto", "0.5*n5k:couscous|0.5*n5k:fish", "C", ""),
    ("cuscuz_grao", r"cous ?cous", "misto", "prato_misto", "0.6*n5k:couscous|0.2*n5k:chickpeas|0.2*n5k:mixed vegetables", "C", ""),
    ("arroz_com_proteina", r"\brice\b.*(tuna|pork|shrimp|ham)", "misto", "prato_misto", "0.8*T_ARROZ|0.15*n5k:tuna|0.05*DILUENTE", "C", ""),
    ("salada_com_proteina", r"salad with.*(tuna|mozzarella|anchov)", "misto", "prato_misto", "0.7*n5k:lettuce|0.15*n5k:tuna|0.15*n5k:mozzarella cheese", "C", ""),
    # massa e arroz
    ("massa_com_proteina", r"(pasta|spaghetti|gnocchi).*(tuna|ham|pork|shrimp)", "misto", "prato_misto", "0.8*n5k:pasta|0.15*n5k:tuna|0.05*DILUENTE", "C", ""),
    ("massa_com_leguminosa", r"(pasta|rice).*(chickpea|(?<!green )beans)", "misto", "prato_misto", "0.65*n5k:pasta|0.25*n5k:chickpeas|0.1*DILUENTE", "C", ""),
    ("massa_forno_queijo", r"baked pasta.*(mozzarella|cheese)|pasta with (tomatoes and cheese|creamy sauce)", "misto", "prato_misto",
     "0.75*n5k:pasta|0.1*n5k:mozzarella cheese|0.15*n5k:pasta sauce", "C", ""),
    ("ravioli", r"ravioli", "misto", "prato_misto", "0.8*n5k:pasta|0.1*n5k:cheese|0.1*n5k:pasta sauce", "C", ""),
    ("massa_pesto", r"pesto", "amido", "sem_proteina", "0.9*n5k:pasta|0.1*n5k:pesto", "B", ""),
    ("nhoque", r"gnocchi", "amido", "sem_proteina", "0.85*n5k:gnocchi|0.15*n5k:pasta sauce", "B", ""),
    ("massa_parmesao", r"(pasta|spaghetti).*parmesan", "amido", "sem_proteina", MASSA_MOLHO_PARM, "B", ""),
    ("massa", r"pasta|spaghetti", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    ("arroz_parmesao", r"rice.*parmesan", "amido", "sem_proteina", "0.97*T_ARROZ|0.03*n5k:parmesan cheese", "B", ""),
    ("salada_batata", r"potato salad", "amido", "sem_proteina", "n5k:potato salad", "A", ""),
    ("arroz", r"\brice\b", "amido", "sem_proteina", "T_ARROZ", "B", ""),
    ("cevada", r"barley", "amido", "sem_proteina", "0.7*n5k:barley|0.3*n5k:peas", "B", ""),
    # proteína vegetal moldada
    ("falafel", r"falafel", "leguminosa", "leguminosa_soja", "n5k:falafel", "B", ""),
    ("burger_leguminosa", r"(legume|lentils).*burger|burger.*lentils|plant-based", "leguminosa", "leguminosa_soja", "0.5*n5k:veggie burgers|0.5*n5k:lentils", "C", ""),
    ("burger_legumes", r"vegetable (meatballs|burger)|vegetarian (meatballs|loaf)|veggie burger|potato meatballs|burger with courgettes", "misto", "prato_misto", VEG_BURGER, "C", ""),
    # peixe
    ("peixe_empanado", r"breaded (fish|cod)|fish cutlet", "proteina_animal", "peixe_marisco", "EMPANADO_PEIXE", "C", ""),
    ("cavala", r"mackerel", "proteina_animal", "peixe_marisco", "n5k:mackerel", "B", ""),
    ("peixe", r"fish|\bcod\b", "proteina_animal", "peixe_marisco", "n5k:fish", "B", ""),
    # carne
    ("vitela_atum", r"veal with tuna", "proteina_animal", "bovina", "0.7*n5k:veal|0.1*n5k:tuna|0.2*n5k:mayonnaise", "C", ""),
    ("linguica_pimentao", r"sausage and bell", "proteina_animal", "suina_embutido", "0.5*n5k:sausage|0.5*n5k:bell peppers", "C", ""),
    ("bolo_de_carne", r"meatloaf", "misto", "prato_misto", "n5k:meatloaf", "B", ""),
    ("almondega_molho", r"^meatballs (in|with) tomato", "proteina_animal", "bovina", "0.7*n5k:meatballs|0.3*n5k:pasta sauce", "C", ""),
    ("almondega", r"^meatballs$", "proteina_animal", "bovina", "n5k:meatballs", "B", ""),
    ("hamburguer", r"^(hamburger|burger)( with rd sauce)?$", "proteina_animal", "bovina", "n5k:hamburgers", "B", ""),
    ("porco_empanado", r"breaded pork", "proteina_animal", "suina_embutido", EMPANADO_CARNE, "C", ""),
    ("porco", r"pork", "proteina_animal", "suina_embutido", "n5k:roast pork", "B", ""),
    ("presunto", r"^cooked ham$", "proteina_animal", "suina_embutido", "n5k:ham", "B", ""),
    ("carne_assada", r"roast meat", "proteina_animal", "bovina", "n5k:roast beef", "C", ""),
    ("frango_asa", r"chicken wings", "proteina_animal", "frango_peru", "n5k:chicken wings", "B", ""),
    ("frango_empanado_fritas", r"chicken cutlet with potato", "proteina_animal", "frango_peru", "0.6*n5k:chicken nuggets|0.4*n5k:french fries", "C", ""),
    ("frango_empanado", r"chicken (nuggets|cutlet)", "proteina_animal", "frango_peru", "n5k:chicken nuggets", "C", ""),
    ("escalope", r"breaded cutlet|escalope", "proteina_animal", "frango_peru", "n5k:chicken nuggets", "C", ""),
    # ovo
    ("ovo_frito", r"fried egg", "proteina_animal", "ovo", "T_OVO_FRITO", "A", ""),
    ("omelete", r"omelette", "proteina_animal", "ovo", "n5k:omelets", "B", ""),
    # queijo
    ("mussarela_legumes", r"^mozzarella with", "laticinio", "laticinio", "0.4*n5k:mozzarella cheese|0.6*n5k:mixed vegetables", "C", ""),
    ("ricota_legumes", r"ricotta.*(eggplant|vegetables|courgette)", "laticinio", "laticinio", "0.6*taco:Queijo, ricota|0.4*n5k:eggplant", "C", ""),
    ("ricota", r"ricotta", "laticinio", "laticinio", "taco:Queijo, ricota", "B", ""),
    ("robiola", r"robiola", "laticinio", "laticinio", "taco:Queijo, minas, frescal", "C", ""),
    ("legumes_mussarela", r"vegetables with tomato and mozzarella", "misto", "prato_misto", "0.85*n5k:mixed vegetables|0.15*n5k:mozzarella cheese", "C", ""),
    # leguminosas
    ("ervilha_presunto", r"peas with ham", "misto", "prato_misto", "0.8*n5k:peas|0.2*n5k:ham", "C", ""),
    ("grao_abobora", r"chickpeas and pumpkin", "leguminosa", "leguminosa_soja", "0.5*n5k:chickpeas|0.5*n5k:pumpkins", "B", ""),
    ("grao_de_bico", r"chickpeas", "leguminosa", "leguminosa_soja", "n5k:chickpeas", "A", ""),
    ("lentilha", r"lentils", "leguminosa", "leguminosa_soja", "n5k:lentils", "B", ""),
    ("legumes_com_fava", r"carrots, peas, and broad beans", "leguminosa", "leguminosa_soja", "0.33*n5k:carrot|0.33*n5k:peas|0.34*n5k:bean(seed)", "B", ""),
    ("feijao", r"^(canned |)beans", "leguminosa", "leguminosa_soja", "n5k:bean(seed)", "B", ""),
    ("ervilha", r"^boiled peas$", "leguminosa", "leguminosa_soja", "n5k:peas", "A", ""),
    # frutas (peso com casca e caroço)
    ("fruta", r"^(apple|banana|plums?|pear|orange|peach|apricots|grapes|watermelon|mandarins)$", "vegetal", "sem_proteina", "n5k:fruit salad", "B", ""),
    ("fruta_cozida", r"apple puree|caramelized pear", "vegetal", "sem_proteina", "n5k:applesauce", "B", ""),
    # batata e legumes
    ("pure", r"mashed potatoes", "amido", "sem_proteina", "n5k:mashed potatoes", "A", ""),
    ("bolinho_batata", r"potato muffin", "amido", "sem_proteina", "0.8*n5k:potatoes|0.2*n5k:eggs", "C", ""),
    ("batata", r"^(baked|roasted|boiled) potatoes$", "amido", "sem_proteina", "n5k:roasted potatoes", "A", ""),
    ("legumes_com_batata", r"potato", "vegetal", "sem_proteina", "0.5*n5k:potatoes|0.5*n5k:mixed vegetables", "B", ""),
    ("salada_caesar", r"caesar", "vegetal", "sem_proteina", "n5k:caesar salad", "B", ""),
    ("espinafre_parmesao", r"spinach with parmesan", "vegetal", "sem_proteina", "0.95*n5k:spinach (cooked)|0.05*n5k:parmesan cheese", "B", ""),
    ("espinafre", r"spinach|chicory", "vegetal", "sem_proteina", "n5k:spinach (cooked)", "A", ""),
    ("brocolis", r"broccoli", "vegetal", "sem_proteina", "n5k:broccoli", "A", ""),
    ("couve_bruxelas", r"brussels? sprouts?", "vegetal", "sem_proteina", "n5k:brussels sprouts", "A", ""),
    ("vagem", r"gre+n beans", "vegetal", "sem_proteina", "n5k:green beans", "A", ""),
    ("ervilha_cenoura", r"peas", "vegetal", "sem_proteina", "0.5*n5k:peas|0.5*n5k:carrot", "B", ""),
    ("salada", r"salad|radicchio", "vegetal", "sem_proteina", "n5k:lettuce", "B", ""),
    ("legumes", r"cauliflower|carrot|pumpkin|fennel|courgette|zucchini|eggplant|vegetables|cabbage", "vegetal", "sem_proteina", "n5k:mixed vegetables", "B", ""),
]
_RX = [(nome, re.compile(rx), g, f, ref, c, t) for nome, rx, g, f, ref, c, t in REGRAS]


def norm(nome):
    return re.sub(r"\s+", " ", nome.strip().lower()).rstrip(":").strip()


def classifica(nome):
    n = norm(nome)
    for r in _RX:
        if r[1].search(n):
            return r
    raise KeyError(f"nome sem regra: {nome!r}")


def ler_itens(labels):
    with open(labels, encoding="utf-8", newline="") as f:
        return [(r["image"].strip(), r["food"], float(r["portion"])) for r in csv.DictReader(f)]


def cmd_mapa(labels):
    tab = tabelas(); stats = defaultdict(list)
    for _, nome, g in ler_itens(labels):
        stats[norm(nome)].append(g)
    linhas = []
    for n in sorted(stats):
        regra, _, grupo, fonte, ref, conf, tipo = classifica(n)
        v = sorted(stats[n])
        linhas.append([n, len(v), f"{v[len(v) // 2]:.0f}", regra, grupo, fonte, ref, f"{densidade(ref, tab):.2f}", conf, tipo])
    with open(MAPA, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["componente", "itens", "mediana_g", "regra", "grupo", "fonte", "ref", "proteina_100g", "confianca", "tipo"])
        w.writerows(linhas)
    print(MAPA, len(linhas), "nomes |", dict(Counter(l[8] for l in linhas)), "| por regra:", dict(Counter(l[3] for l in linhas).most_common(8)))


def cmd_monta(labels, participantes, saida):
    tab = tabelas()
    with open(MAPA, encoding="utf-8", newline="") as f:
        mapa = {r["componente"]: r for r in csv.DictReader(f)}
    for n, r in mapa.items():                              # o mapa gravado tem de ser o que as regras dão hoje
        c = classifica(n)
        assert (r["regra"], r["grupo"], r["fonte"], r["ref"], r["tipo"]) == (c[0], c[2], c[3], c[4], c[6]), n
        assert abs(float(r["proteina_100g"]) - densidade(r["ref"], tab)) < 0.006, n
    with open(participantes, encoding="utf-8", newline="") as f:
        quem = {r["resized_filename"]: r["participant_id"] for r in csv.DictReader(f)}
    base = confere_esquema()
    cab = base + ["dominio", "grupo", "vista", "split_ext", "arquivo", "url", "componentes", "n_componentes", "bebida_g",
                  "embalado_g", "proteina_fraca_g", "proteina_fraca_conf", "proteina_fraca_fontes_g", "maior_fonte_fraca_g", "avisos"]
    por_foto = defaultdict(list)
    for img, nome, g in ler_itens(labels):
        por_foto[img].append((nome, g))
    linhas = []
    for img in sorted(por_foto):
        mg = dict.fromkeys(GRUPOS, 0.0); ff = dict.fromkeys(FONTES, 0.0)
        massa = beb = emb = pf = pff = maior = 0.0; por_conf = Counter(); comp = []
        for nome, g in por_foto[img]:
            c = mapa[norm(nome)]
            if c["tipo"] == "bebida":
                beb += g; continue
            massa += g; mg[c["grupo"]] += g; ff[c["fonte"]] += g
            emb += g if c["tipo"] == "embalado" else 0
            p = g * float(c["proteina_100g"]) / 100
            pf += p; por_conf[c["confianca"]] += p
            if c["fonte"] != "sem_proteina":
                pff += p; maior = max(maior, p)
            comp.append((nome, g))
        if massa <= 0:
            continue                                         # foto só de bebida
        grupo = "men_" + quem[img]
        split = "teste_ext" if int(hashlib.sha1(grupo.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
        avisos = [a for a, s in (("um_item_so", len(comp) == 1), ("tem_embalado", emb > 0)) if s]
        linhas.append([f"men_{img[:3]}", -1, -1, "", f"{massa:.1f}", "", "", "", ""]
                      + [f"{mg[g]:.1f}" for g in GRUPOS] + [""] * len(GRUPOS) + [f"{ff[k] / massa:.4f}" for k in FONTES]
                      + ["mensa", grupo, "zenital_bandeja", split, "GT_Mensa_2025/resized_images/" + img, URL + img,
                         ";".join(f"{n.replace(';', ',')}:{g:.0f}" for n, g in comp), len(comp), f"{beb:.0f}", f"{emb:.0f}",
                         f"{pf:.1f}", max(por_conf, key=por_conf.get), f"{pff:.1f}", f"{maior:.1f}", "|".join(avisos)])
    assert len({l[0] for l in linhas}) == len(linhas)
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    i = {k: cab.index(k) for k in cab}
    m = sorted(float(l[i["massa"]]) for l in linhas); q = lambda p: m[int(p * (len(m) - 1))]
    pf = [float(l[i["proteina_fraca_g"]]) for l in linhas]
    print(f"fotos (linhas): {len(linhas)} | participantes: {dict(Counter(l[i['grupo']] for l in linhas))}")
    print(f"  split: {dict(Counter(l[i['split_ext']] for l in linhas))} | um item só: {sum('um_item_so' in l[-1] for l in linhas)}")
    print(f"  massa sem bebida (g): p10 {q(.1):.0f} mediana {q(.5):.0f} p90 {q(.9):.0f} máx {m[-1]:.0f} | > 600 g: {sum(v > 600 for v in m)}")
    print(f"  proteína FRACA (g): mediana {sorted(pf)[len(pf) // 2]:.1f} | > {PROTEINA_FRACA_ALTA_G:.0f} g: {sum(v > PROTEINA_FRACA_ALTA_G for v in pf)}"
          f" | só das fontes > 25 g: {sum(float(l[i['proteina_fraca_fontes_g']]) > 25 for l in linhas)}"
          f" | um componente-fonte > 25 g: {sum(float(l[i['maior_fonte_fraca_g']]) > 25 for l in linhas)}"
          f" | um componente-fonte > 15 g: {sum(float(l[i['maior_fonte_fraca_g']]) > 15 for l in linhas)}")
    tot = sum(m)
    print("  massa por grupo %:", {g: round(100 * sum(float(l[i[f'm_{g}']]) for l in linhas) / tot, 1) for g in GRUPOS})
    print("  fotos com a fonte (>= 20 g):", {k: sum(float(l[i[f'ff_{k}']]) * float(l[i['massa']]) >= 20 for l in linhas) for k in FONTES})
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "mapa":
        cmd_mapa(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[3], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
