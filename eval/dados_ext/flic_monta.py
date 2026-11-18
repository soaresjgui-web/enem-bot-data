#!/usr/bin/env python3
"""Monta a lista de treino externo do FLIC, no esquema de externo_mensa.csv.

Dataset: Piccoli, F.; Callegaro, D.; Marelli, D.; Buzzelli, M.; Franchini, C.; Stella, L.; Bianco, S.; Ciocca, G.;
Schettini, R.; Scazzina, F. FLIC: A Real-World Dataset for Visual Estimation of Food Leftovers in Canteens.
Applied Sciences 2026, 16(11), 5465. https://doi.org/10.3390/app16115465 — dados na pasta do Google Drive
1IGCjr7y-_zsZfHTsL4CyxdkspTp8Uhnt (link da seção de disponibilidade do artigo), **licença CC BY 4.0**
(LICENSE.txt da pasta, versão 1.1 de 05/10/2026: "Licensing and data available"). Atribuição obrigatória.

O que é: refeitório de colônia de férias na Itália, 22 dias de junho e julho de 2024. Câmera fixa a cerca de 65 cm,
**foto de cima** da bandeja inteira (640×480), antes (`full`) e depois da refeição (`left`). Cada bandeja tem até
quatro cursos: primeiro (massa, arroz ou pizza), segundo (a fonte de proteína), acompanhamento (legume, sem nome no
cardápio) e pão (50 g, em saquinho). **Cada curso foi pesado antes e depois**; FLIC_grams.csv traz os gramas e a
contagem de pixels de cada curso na máscara. O nome dos pratos vem do cardápio do dia (FLIC_menu.csv).

Cuidado com os cabeçalhos do FLIC: em FLIC_menu.csv as colunas 3 e 4 são o primeiro e o segundo prato em INGLÊS
(o cabeçalho diz "First_ita,Second_eng"); em FLIC_grams.csv o segundo prato se chama "Second_gull_*". Aqui se lê por
posição e por nome exato, e o script confere os dois.

Uso:
  python3 -I flic_monta.py monta <FLIC_grams.csv> <FLIC_menu.csv> [--saida <csv>]
  python3 -I flic_monta.py cursos <FLIC_grams.csv> <FLIC_menu.csv> [--saida <csv>]    # seção 19: uma linha por curso
Os arquivos vêm da pasta do Drive; as imagens são baixadas pelo identificador em flic_arquivos.csv
(https://drive.usercontent.google.com/download?id=<id>&export=download). Nada de imagem é lido aqui.

UMA LINHA POR FOTO: a bandeja cheia e a bandeja com sobra viram duas linhas, com os gramas da própria foto.
Fotos de sobra com menos de MASSA_MIN_G no total ficam de fora (bandeja praticamente vazia). `grupo` = dia
(todas as bandejas do dia têm o mesmo cardápio, então a divisão treino/teste é por dia). Proteína: só a estimativa
FRACA (prato do cardápio -> tabela), em colunas à parte; `proteina` fica vazia. A massa é medida.
"""
import csv, hashlib, os, sys
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import FONTES, GRUPOS, confere_esquema, densidade, tabelas  # noqa: E402

ARQUIVOS = os.path.join(AQUI, "flic_arquivos.csv")
FOTOS = os.path.join(AQUI, "flic_fotos.csv")          # tamanho e sha256 de cada PNG baixado (flic_cursos.py confere)
SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_flic.csv")
SAIDA_CURSOS = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_flic_curso.csv")
URL = "https://drive.usercontent.google.com/download?export=download&id="
LICENCA, USO = "CC BY 4.0", "comercial_ok"
MASSA_MIN_G = 10.0
PROTEINA_FRACA_ALTA_G = 25.0
PIXELS_MIN_CURSO = 200      # 19.2 b: curso pesado e invisível na máscara não entra
# cor de cada curso nas máscaras (conferida: a contagem de pixels de cada cor é a coluna *_pixels de FLIC_grams.csv)
CORES = {"First": (50, 183, 250), "Second": (184, 61, 245), "Side": (170, 240, 209), "Bread": (250, 250, 55)}
CURSOS_PRATO = ("First", "Second", "Side")     # 19.2 a: o pão vem num saquinho e não entra

MASSA_MOLHO = "0.8*n5k:pasta|0.2*n5k:pasta sauce"
GRATINADO = "0.85*{}|0.15*n5k:bread crumbs"

# prato do cardápio (inglês, como em FLIC_menu.csv) -> (regra, grupo, fonte, referência, confiança, tipo)
# Mesmas convenções do Mensa/Flavoria: peça de proteína vai para a fonte dela; proteína cozida dentro de massa,
# arroz, torta ou pizza vai para prato_misto; massa com molho de tomate ou legumes é amido.
PRATOS = {
    # primeiros
    "Pasta Bolognese style": ("massa_ragu", "misto", "prato_misto", "0.8*n5k:pasta|0.12*n5k:ground beef|0.08*n5k:pasta sauce", "C", ""),
    "Pasta with tomato and tuna": ("massa_atum", "misto", "prato_misto", "0.8*n5k:pasta|0.12*n5k:tuna|0.08*n5k:pasta sauce", "C", ""),
    "Ricotta and spinach tortelli": ("tortelli", "misto", "prato_misto", "0.85*n5k:tortellini|0.15*n5k:pasta sauce", "C", ""),
    "Three-color cold pasta salad": ("massa_fria_mussarela", "misto", "prato_misto",
                                     "0.7*n5k:pasta|0.15*n5k:mozzarella cheese|0.15*n5k:tomatoes", "C", ""),
    "Pizza margherita": ("pizza", "misto", "prato_misto", "n5k:cheese pizza", "B", ""),
    "Pasta with olive oil and Parmesan": ("massa_parmesao", "amido", "sem_proteina", "0.95*n5k:pasta|0.05*n5k:parmesan cheese", "B", ""),
    "Pasta with pesto sauce": ("massa_pesto", "amido", "sem_proteina", "0.9*n5k:pasta|0.1*n5k:pesto", "B", ""),
    "Pasta with eggplant and tomatoes": ("massa", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    "Pasta with tomato and basil": ("massa", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    "Pasta with tomato and peas": ("massa", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    "Pasta with vegetables": ("massa", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    "Pasta with zucchini": ("massa", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    "Whole pasta with eggplant": ("massa", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    "Whole pasta with tomato and basil": ("massa", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    "Whole pasta with zucchini": ("massa", "amido", "sem_proteina", MASSA_MOLHO, "B", ""),
    "Rice with oil and Parmesan": ("arroz_parmesao", "amido", "sem_proteina", "0.97*T_ARROZ|0.03*n5k:parmesan cheese", "B", ""),
    "Rice with tomato and basil": ("arroz_tomate", "amido", "sem_proteina", "0.85*T_ARROZ|0.15*n5k:pasta sauce", "B", ""),
    # segundos
    "Roast pork": ("porco", "proteina_animal", "suina_embutido", "n5k:roast pork", "A", ""),
    "Cooked ham": ("presunto", "proteina_animal", "suina_embutido", "n5k:ham", "A", ""),
    # sovracoscia costuma ir com osso: a densidade por grama pesado fica superestimada
    "Chicken upper thigh": ("frango_sobrecoxa", "proteina_animal", "frango_peru", "n5k:chicken thighs", "C", ""),
    "Gratin-style chicken bites": ("frango_gratinado", "proteina_animal", "frango_peru", GRATINADO.format("n5k:chicken breast"), "B", ""),
    "Turkey bites with lemon": ("peru", "proteina_animal", "frango_peru", "n5k:turkey", "B", ""),
    "Cod with tomato and olives": ("bacalhau_molho", "proteina_animal", "peixe_marisco", "0.85*n5k:cod|0.15*n5k:pasta sauce", "B", ""),
    "Cod fillets with potatoes": ("bacalhau_batata", "proteina_animal", "peixe_marisco", "0.6*n5k:cod|0.4*n5k:potatoes", "C", ""),
    "Gratin-style plaice fillets": ("peixe_gratinado", "proteina_animal", "peixe_marisco", GRATINADO.format("n5k:fish"), "C", ""),
    "Gratin-style cuttlefish": ("sepia_gratinada", "proteina_animal", "peixe_marisco", GRATINADO.format("n5k:squid"), "C", ""),
    "Zucchini omelet": ("omelete_legume", "proteina_animal", "ovo", "0.8*n5k:omelets|0.2*n5k:zucchini", "B", ""),
    "Spinach omelet": ("omelete_legume", "proteina_animal", "ovo", "0.8*n5k:omelets|0.2*n5k:spinach (cooked)", "B", ""),
    "Swiss chard and ricotta omelet": ("omelete_ricota", "proteina_animal", "ovo",
                                       "0.7*n5k:omelets|0.2*n5k:chard|0.1*taco:Queijo, ricota", "B", ""),
    "Parmesan cheese": ("parmesao", "laticinio", "laticinio", "n5k:parmesan cheese", "A", ""),
    "Chickpea and potato pattie": ("bolinho_grao", "leguminosa", "leguminosa_soja", "0.5*n5k:chickpeas|0.5*n5k:potatoes", "C", ""),
    "Legume pattie": ("burger_leguminosa", "leguminosa", "leguminosa_soja", "0.5*n5k:veggie burgers|0.5*n5k:lentils", "C", ""),
    "Lentil flan": ("flan_lentilha", "leguminosa", "leguminosa_soja", "0.6*n5k:lentils|0.2*n5k:eggs|0.2*n5k:potatoes", "C", ""),
    "Savory chard and spinach pie": ("erbazzone", "misto", "prato_misto",
                                     "0.5*n5k:chard|0.35*n5k:pizza dough|0.15*n5k:parmesan cheese", "C", ""),
    # cursos sem nome no cardápio
    "Side (unnamed vegetable)": ("acompanhamento", "vegetal", "sem_proteina", "n5k:mixed vegetables", "C", ""),
    "Bread": ("pao", "amido", "sem_proteina", "taco:Pão, trigo, francês", "A", "embalado"),
}
CURSOS = [("First", 2), ("Second", 3), ("Side", None), ("Bread", None)]   # (prefixo, coluna do nome em inglês no cardápio)


def col(curso, fase, medida):
    """Nome da coluna em FLIC_grams.csv; o segundo prato cheio se chama "Second_gull_*" no original."""
    return "Second_gull_" + medida if (curso, fase) == ("Second", "full") else f"{curso}_{fase}_{medida}"


def ler_menu(caminho):
    with open(caminho, encoding="utf-8", newline="") as f:
        r = list(csv.reader(f))
    assert r[0][:4] == ["Date", "Weekday", "First_ita", "Second_eng"], r[0]   # cabeçalho trocado do original
    menu = {}
    for x in r[1:]:
        d, m, a = x[0].split("-")
        menu[f"{a}-{m}-{d}"] = {"First": x[2], "Second": x[3]}
    return menu


def ler_ids():
    with open(ARQUIVOS, encoding="utf-8", newline="") as f:
        return {r["arquivo"]: r["drive_id"] for r in csv.DictReader(f)}


def cmd_monta(grams, menu_csv, saida):
    tab = tabelas()
    dens = {p: densidade(v[3], tab) for p, v in PRATOS.items()}
    menu = ler_menu(menu_csv); ids = ler_ids()
    base = confere_esquema()
    cab = base + ["dominio", "grupo", "vista", "split_ext", "arquivo", "url", "componentes", "n_componentes", "bebida_g",
                  "embalado_g", "proteina_fraca_g", "proteina_fraca_conf", "proteina_fraca_fontes_g", "maior_fonte_fraca_g", "avisos",
                  "pixels_cursos", "mascara", "par", "licenca", "uso_declarado"]
    with open(grams, encoding="utf-8", newline="") as f:
        linhas_g = list(csv.DictReader(f))
    assert "Second_gull_grams" in linhas_g[0] and "Filename_full" in linhas_g[0], list(linhas_g[0])
    linhas = []; fora = Counter(); usados = Counter()
    for r in linhas_g:
        dia = r["Date"]
        assert dia in menu, dia
        nomes = {"First": menu[dia]["First"], "Second": menu[dia]["Second"], "Side": "Side (unnamed vegetable)", "Bread": "Bread"}
        for fase, pasta_img, pasta_msk, col_arq in (("full", "full_images", "full_masks", "Filename_full"),
                                                 ("left", "left_images", "left_masks", "Filename_left")):
            mg = dict.fromkeys(GRUPOS, 0.0); ff = dict.fromkeys(FONTES, 0.0)
            massa = emb = pf = pff = maior = 0.0; por_conf = Counter(); comp = []; pix = []
            for curso, _ in CURSOS:
                g = float(r[col(curso, fase, "grams")]); px = int(r[col(curso, fase, "pixels")])
                pix.append(f"{curso.lower()}:{px}")
                if g <= 0:
                    continue
                nome = nomes[curso]; regra, grupo, fonte, ref, conf, tipo = PRATOS[nome]; usados[nome] += 1
                massa += g; mg[grupo] += g; ff[fonte] += g
                emb += g if tipo == "embalado" else 0
                p = g * dens[nome] / 100
                pf += p; por_conf[conf] += p
                if fonte != "sem_proteina":
                    pff += p; maior = max(maior, p)
                comp.append((nome, g))
            if massa < MASSA_MIN_G:
                fora[fase] += 1; continue
            img = f"{pasta_img}/{r[col_arq]}"; msk = f"{pasta_msk}/{r[col_arq]}"
            assert img in ids and msk in ids, img
            grupo_dia = "flic_" + dia
            split = split_dia(grupo_dia)
            avisos = [a for a, s in (("sobra", fase == "left"), ("um_item_so", len(comp) == 1), ("tem_embalado", emb > 0)) if s]
            linhas.append([f"flic_{dia}_{r['QR_code']}_{fase}", -1, -1, "", f"{massa:.1f}", "", "", "", ""]
                          + [f"{mg[g]:.1f}" for g in GRUPOS] + [""] * len(GRUPOS) + [f"{ff[k] / massa:.4f}" for k in FONTES]
                          + ["flic", grupo_dia, "zenital_bandeja", split, img, URL + ids[img],
                             ";".join(f"{n.replace(';', ',')}:{g:.0f}" for n, g in comp), len(comp), "0", f"{emb:.0f}",
                             f"{pf:.1f}", max(por_conf, key=por_conf.get), f"{pff:.1f}", f"{maior:.1f}", "|".join(avisos),
                             ";".join(pix), msk, f"flic_{dia}_{r['QR_code']}", LICENCA, USO])
    assert len({l[0] for l in linhas}) == len(linhas)
    sem_uso = sorted(set(PRATOS) - set(usados))
    assert not sem_uso, f"pratos do mapa que não aparecem: {sem_uso}"
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    i = {k: cab.index(k) for k in cab}
    m = sorted(float(l[i["massa"]]) for l in linhas); q = lambda p: m[int(p * (len(m) - 1))]
    pf = [float(l[i["proteina_fraca_g"]]) for l in linhas]
    print(f"bandejas: {len(linhas_g)} | fotos (linhas): {len(linhas)} | fora (< {MASSA_MIN_G:.0f} g): {dict(fora)}")
    print(f"  dias: {len({l[i['grupo']] for l in linhas})} | split: {dict(Counter(l[i['split_ext']] for l in linhas))}"
          f" | dias de teste: {sorted({l[i['grupo']][5:] for l in linhas if l[i['split_ext']] == 'teste_ext'})}")
    print(f"  massa (g): p10 {q(.1):.0f} mediana {q(.5):.0f} p90 {q(.9):.0f} máx {m[-1]:.0f}")
    print(f"  proteína FRACA (g): mediana {sorted(pf)[len(pf) // 2]:.1f} | > {PROTEINA_FRACA_ALTA_G:.0f} g: {sum(v > PROTEINA_FRACA_ALTA_G for v in pf)}"
          f" | um componente-fonte > 15 g: {sum(float(l[i['maior_fonte_fraca_g']]) > 15 for l in linhas)}")
    print("  densidade (g proteína/100 g) por prato:", {p: round(d, 1) for p, d in sorted(dens.items(), key=lambda x: -x[1])[:8]}, "...")
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


def espelho(arquivo):
    """Nome do arquivo na pasta plana do espelho do Kaggle: 'full_images/x.png' -> 'full_images__x.png'."""
    return arquivo.replace("/", "__")


def split_dia(grupo_dia):
    """A divisão por dia da lista por bandeja (hash do dia): 2 dias de teste."""
    return "teste_ext" if int(hashlib.sha1(grupo_dia.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"


def cmd_cursos(grams, menu_csv, saida):
    """Seção 19.2 b: uma linha por (foto, curso) — primeiro, segundo e acompanhamento, antes e depois da refeição —
    com a massa pesada do curso naquela foto. Entra o curso com MASSA_MIN_G ou mais e PIXELS_MIN_CURSO ou mais na
    máscara. `arquivo` é o nome na pasta plana do espelho; `url`, o do Google Drive; `sha256` e `bytes`, os da
    conferência (flic_fotos.csv). A proteína é só a FRACA (prato do cardápio -> tabela)."""
    tab = tabelas()
    dens = {p: densidade(v[3], tab) for p, v in PRATOS.items()}
    menu = ler_menu(menu_csv); ids = ler_ids()
    with open(FOTOS, encoding="utf-8", newline="") as f:
        fotos = {r["arquivo"]: r for r in csv.DictReader(f)}
    base = confere_esquema()
    cab = base + ["dominio", "grupo", "vista", "split_ext", "arquivo", "url", "sha256", "bytes", "arquivo_original", "curso", "cor_mascara",
                  "mascara", "pixels_curso", "prato", "proteina_fraca_g", "densidade_proteina_fraca_g100", "proteina_fraca_conf", "par",
                  "bandeja", "fase", "licenca", "uso_declarado"]
    with open(grams, encoding="utf-8", newline="") as f:
        linhas_g = list(csv.DictReader(f))
    linhas, fora = [], Counter()
    for r in linhas_g:
        dia = r["Date"]
        nomes = {"First": menu[dia]["First"], "Second": menu[dia]["Second"], "Side": "Side (unnamed vegetable)"}
        for fase, pasta_img, pasta_msk, col_arq in (("full", "full_images", "full_masks", "Filename_full"),
                                                 ("left", "left_images", "left_masks", "Filename_left")):
            img, msk = f"{pasta_img}/{r[col_arq]}", f"{pasta_msk}/{r[col_arq]}"
            assert img in ids and msk in ids and img in fotos and msk in fotos, img
            for curso in CURSOS_PRATO:
                g = float(r[col(curso, fase, "grams")]); px = int(r[col(curso, fase, "pixels")])
                if g < MASSA_MIN_G:
                    fora[f"{fase}_menos_de_{MASSA_MIN_G:.0f}g"] += 1; continue
                if px < PIXELS_MIN_CURSO:
                    fora[f"{fase}_sem_mascara"] += 1; continue
                nome = nomes[curso]; regra, grupo, fonte, ref, conf, tipo = PRATOS[nome]
                mg = dict.fromkeys(GRUPOS, 0.0); mg[grupo] = g
                ff = dict.fromkeys(FONTES, 0.0); ff[fonte] = 1.0
                grupo_dia = "flic_" + dia
                linhas.append([f"flic_{dia}_{r['QR_code']}_{fase}_{curso.lower()}", -1, -1, "", f"{g:.1f}", "", "", "", ""]
                              + [f"{mg[k]:.1f}" for k in GRUPOS] + [""] * len(GRUPOS) + [f"{ff[k]:.4f}" for k in FONTES]
                              + ["flic", grupo_dia, "zenital_curso", split_dia(grupo_dia), espelho(img), URL + ids[img], fotos[img]["sha256"],
                                 fotos[img]["bytes"], img, curso.lower(), "%d,%d,%d" % CORES[curso], espelho(msk), px, nome,
                                 f"{g * dens[nome] / 100:.2f}", f"{dens[nome]:.3f}", conf, f"flic_{dia}_{r['QR_code']}_{curso.lower()}",
                                 f"flic_{dia}_{r['QR_code']}", fase, LICENCA, USO])
    assert len({l[0] for l in linhas}) == len(linhas)
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    i = {k: cab.index(k) for k in cab}
    print(f"linhas (foto x curso): {len(linhas)} | fora: {dict(fora)}")
    print("  por fase e curso:", dict(Counter((l[i['fase']], l[i['curso']]) for l in linhas)))
    print("  split:", dict(Counter(l[i['split_ext']] for l in linhas)), "| pratos:", len({l[i['par']] for l in linhas}),
          "| bandejas:", len({l[i['bandeja']] for l in linhas}), "| dias:", len({l[i['grupo']] for l in linhas}))
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[3], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    elif len(sys.argv) >= 4 and sys.argv[1] == "cursos":
        cmd_cursos(sys.argv[2], sys.argv[3], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA_CURSOS)
    else:
        sys.exit(__doc__)
