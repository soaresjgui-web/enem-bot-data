#!/usr/bin/env python3
"""Monta a lista de treino externo do SNAPMe, com RÓTULO FRACO (autorrelato), no esquema de externo_flavoria.csv.

Dataset: Chin, E. L.; Larke, J. A.; Bouzid, Y. Y.; Nguyen, T.; Vainberg, Y.; Smilowitz, J. T.; Lemay, D. G. (2022).
SNAPMe: A Benchmark Dataset of Food Photos with Food Records for Evaluation of Computer Vision Algorithms in the
Context of Dietary Assessment. Ag Data Commons. https://doi.org/10.15482/USDA.ADC/1528346
Licença no registro: CC BY-SA 4.0 (atribuição e compartilha-igual).
ATENÇÃO: o registro traz um campo "Use limitations" que diz: "This data set should not be used to train models as
it is a small curated data set." Não é cláusula da licença, mas é o pedido expresso dos autores. Usar esta lista no
treino do produto é uma decisão que não foi tomada: ver docs/dados/precisao/fontes_treino.md, seção 10.

O que é: 95 pessoas nos EUA fotografaram com o celular tudo o que comeram em até 3 dias, com um marcador
quadriculado de tamanho conhecido na cena, e preencheram no mesmo dia o recordatório ASA24. Cada item do
recordatório está ligado à foto "antes" em que aparece. A quantidade é a que a pessoa DECLAROU ter comido, em
porções padrão do ASA24; os nutrientes são os do FNDDS para essa quantidade. Nada foi pesado.

Uso:
  curl -sL https://ndownloader.figshare.com/files/44532971 \
    | tar -xzO --occurrence=1 snapme_db_09Dec2022/snapme_cs_db/master_SNAPME_linkfile.csv > master.csv
  python3 -I snapme_monta.py monta master.csv [--saida <csv>]
As imagens não são lidas aqui; `arquivo` é o caminho exato dentro de snapme_db_09Dec2022.tar.gz (2,03 GB), cuja
listagem (caminho, bytes, largura, altura) está em snapme_arquivos.csv.

UMA LINHA POR FOTO "antes". TODOS os alvos do esquema (massa, proteina, kcal, gordura, carbo, m_*, p_*, ff_*) ficam
VAZIOS: não há nada medido. O autorrelato vai em colunas à parte, com o sufixo ou prefixo "fraca":
massa_fraca_g, proteina_fraca_g, kcal_fraca, gordura_fraca_g, carbo_fraca_g, m_fraca_<grupo>, ff_fraca_<fonte>.
Grupo e fonte de cada item saem das mesmas regras do Nutrition5k (grupos_alimentos.grupo_de, dens_comum.fonte_de),
aplicadas à descrição do FNDDS, com três ajustes (ver fonte_snapme).

Regras:
  S1  item sem foto (nome NA) ou cuja foto não está no tar: sai o item;
  S2  foto com algum item embalado (packaged_food = 1; a foto é da embalagem): sai a foto;
  S3  foto só de bebida: sai a foto;
  S4  foto com massa declarada abaixo de 20 g ou acima de 2.000 g: sai a foto.
`grupo` = participante (todas as fotos de uma pessoa ficam do mesmo lado). Parte segurada, regra de reserva de
12.5: sha1(grupo) mod 5 == 0 -> teste_ext.
"""
import csv, hashlib, os, re, sys
from collections import Counter, defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, "..")); sys.path.insert(0, os.path.join(AQUI, "..", "treino"))
from flavoria_monta import FONTES, GRUPOS, confere_esquema  # noqa: E402
from grupos_alimentos import grupo_de  # noqa: E402
from dens_comum import fonte_de  # noqa: E402

ARQUIVOS = os.path.join(AQUI, "snapme_arquivos.csv")
SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_snapme.csv")
MIN_G, MAX_G = 20.0, 2000.0
BEBIDA = re.compile(r"coffee|\btea\b|water|juice|soda|soft drink|\bmilk\b|beer|wine|latte|smoothie|drink|cola|lemonade|kombucha|"
                    r"shake|cocktail|espresso|cappuccino|cocoa|hot chocolate|liquor|whiskey|vodka|sports drink|energy drink", re.I)


_GORDURA_DE_PREPARO = re.compile(r",?\s*(made|cooked|fried|prepared)? ?with (butter|margarine|oil|cooking spray|animal fat[^,]*|cream)\b[^,]*"
                                r"|,?\s*(and|with) sour cream\b|,?\s*fat (not )?added in cooking[^,]*|,?\s*NS as to (type of )?fat[^,]*", re.I)


def fonte_snapme(desc):
    """fonte_de com três ajustes para as descrições do FNDDS, que as regras do Nutrition5k não previam:
    tira a gordura de preparo do nome ("Salmon, baked or broiled, made with butter" cairia em sem_proteina por causa de
    "butter"); "chili con carne" é prato pronto (no Nutrition5k "chili" é a pimenta); pó e shake de proteína vão para
    laticínio, como "protein powder"."""
    d = desc.lower()
    if "chili con carne" in d:
        return "prato_misto"
    if re.search(r"protein, nfs|high protein|protein powder|protein supplement", d) and re.search(r"powder|shake|drink", d):
        return "laticinio"
    return fonte_de(_GORDURA_DE_PREPARO.sub("", desc))


def num(v):
    return float(v) if v not in ("", "NA", None) else 0.0


def ler_fotos():
    """nome do arquivo -> linhas da listagem. O caminho é .../snapme_nut_db/<9001>_QC/<data>_day<N>[_revision|_redo]/<nome>;
    só 2 nomes aparecem em duas pastas."""
    por_nome = defaultdict(list)
    with open(ARQUIVOS, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            m = re.search(r"snapme_nut_db/(\d+)_QC/\d+_day(\d)[^/]*/([^/]+)$", r["caminho"])
            if m and r["largura"]:
                r["sujeito"], r["dia"] = "SNAPME" + m.group(1), m.group(2)
                por_nome[m.group(3)].append(r)
    return por_nome


def acha(por_nome, sujeito, dia, nome):
    """A foto do item: mesmo nome e mesmo participante; se houver duas, a do mesmo dia de estudo."""
    c = [r for r in por_nome.get(nome, []) if r["sujeito"] == sujeito]
    if len(c) > 1:
        c = [r for r in c if r["dia"] == dia] or c
    return sorted(c, key=lambda r: r["caminho"])[0] if c else None


def cmd_monta(master, saida):
    por_nome = ler_fotos()
    base = confere_esquema()
    with open(master, encoding="utf-8", errors="replace", newline="") as f:
        itens = list(csv.DictReader(f))
    cont = Counter(itens=len(itens))
    fotos = defaultdict(list)
    for x in itens:
        r = acha(por_nome, x["subject_id"], x["snapme_study_day"], x["filename"])
        if r is None:
            cont["S1_item_sem_foto"] += 1; continue
        fotos[(x["subject_id"], x["snapme_study_day"], r["caminho"])].append(x)
    cont["fotos_antes_com_item"] = len(fotos)
    cab = (base + ["dominio", "grupo", "vista", "split_ext", "arquivo", "arquivo_depois", "largura", "altura", "dia_estudo",
                   "massa_fraca_g", "proteina_fraca_g", "kcal_fraca", "gordura_fraca_g", "carbo_fraca_g", "bebida_g"]
           + [f"m_fraca_{g}" for g in GRUPOS] + [f"ff_fraca_{k}" for k in FONTES]
           + ["proteina_fraca_fontes_g", "componentes", "n_componentes", "avisos"])
    linhas = []
    for (suj, dia, caminho), its in sorted(fotos.items()):
        if any(x["packaged_food"] == "1" for x in its):
            cont["S2_foto_de_embalagem"] += 1; continue
        if all(BEBIDA.search(x["Food_Description"]) for x in its):
            cont["S3_so_bebida"] += 1; continue
        g = sum(num(x["FoodAmt"]) for x in its)
        if not (MIN_G <= g <= MAX_G):
            cont["S4_massa_fora_de_20_a_2000g"] += 1; continue
        mg = dict.fromkeys(GRUPOS, 0.0); ff = dict.fromkeys(FONTES, 0.0); pff = 0.0
        for x in its:
            fo = fonte_snapme(x["Food_Description"])
            mg[grupo_de(x["Food_Description"])] += num(x["FoodAmt"]); ff[fo] += num(x["FoodAmt"])
            if fo != "sem_proteina":
                pff += num(x["PROT"])
        r = [v for v in por_nome[os.path.basename(caminho)] if v["caminho"] == caminho][0]
        dep = caminho.rsplit(".", 1)[0] + "_after." + caminho.rsplit(".", 1)[1]
        dep = dep if any(v["caminho"] == dep for v in por_nome.get(os.path.basename(dep), [])) else ""
        grupo = "snp_" + suj
        split = "teste_ext" if int(hashlib.sha1(grupo.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
        did = "snp_" + suj[6:] + "_" + dia + "_" + os.path.basename(caminho).rsplit(".", 1)[0][:16]
        linhas.append([did, -1, -1, ""] + [""] * (len(base) - 4)
                      + ["snapme", grupo, "celular_nao_rotulada", split, caminho, dep, r["largura"], r["altura"], dia,
                         f"{g:.0f}", f"{sum(num(x['PROT']) for x in its):.1f}", f"{sum(num(x['KCAL']) for x in its):.0f}",
                         f"{sum(num(x['TFAT']) for x in its):.1f}", f"{sum(num(x['CARB']) for x in its):.1f}",
                         f"{sum(num(x['FoodAmt']) for x in its if BEBIDA.search(x['Food_Description'])):.0f}"]
                      + [f"{mg[k]:.0f}" for k in GRUPOS] + [f"{ff[k] / g:.4f}" for k in FONTES]
                      + [f"{pff:.1f}", ";".join(f"{x['Food_Description'].replace(';', ',')}:{num(x['FoodAmt']):.0f}" for x in its),
                         len(its), "autorrelato"])
    assert len({l[0] for l in linhas}) == len(linhas), "dish_id repetido"
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    i = {k: cab.index(k) for k in cab}
    p = sorted(float(l[i["proteina_fraca_g"]]) for l in linhas); m = sorted(float(l[i["massa_fraca_g"]]) for l in linhas)
    q = lambda v, a: v[int(a * (len(v) - 1))]
    print("contagens:", dict(sorted(cont.items())))
    print(f"fotos (linhas): {len(linhas)} | participantes: {len({l[i['grupo']] for l in linhas})}"
          f" | split (fotos): {dict(Counter(l[i['split_ext']] for l in linhas))}"
          f" | com foto 'depois': {sum(bool(l[i['arquivo_depois']]) for l in linhas)}")
    print(f"  tamanho: {dict(Counter((l[i['largura']], l[i['altura']]) for l in linhas).most_common(3))}")
    print(f"  massa declarada (g): p10 {q(m, .1):.0f} mediana {q(m, .5):.0f} p90 {q(m, .9):.0f}")
    print(f"  proteína declarada (g): mediana {q(p, .5):.1f} p90 {q(p, .9):.1f} | faixas 0-5/5-15/15-25/25-40/>40: "
          f"{[sum(a < v <= b for v in p) for a, b in ((-1, 5), (5, 15), (15, 25), (25, 40), (40, 1e9))]}"
          f" | > 25 g: {sum(v > 25 for v in p)} | > 25 g só das fontes: {sum(float(l[i['proteina_fraca_fontes_g']]) > 25 for l in linhas)}")
    print("  fotos com a fonte (fração >= 10%):", {k: sum(float(l[i[f'ff_fraca_{k}']]) >= 0.1 for l in linhas) for k in FONTES})
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
