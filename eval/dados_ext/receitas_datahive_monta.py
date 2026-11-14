#!/usr/bin/env python3
"""Monta a lista de DENSIDADE (proteína por grama) de receitas com foto do prato pronto: DataHive "Recipes Dataset".

Dataset: datahiveai/recipes-with-nutrition, Hugging Face (https://huggingface.co/datasets/datahiveai/recipes-with-nutrition),
licença CC BY-NC 4.0 no cartão: NÃO COMERCIAL. As receitas e as fotos vêm de sites de receita (Food Network, food.com,
Allrecipes, Food52, BBC Good Food...); os direitos das fotos são desses sites. A análise nutricional segue o formato da
API Edamam: cada linha de ingrediente vira gramas (`ingredients[].weight`), e a receita traz `total_weight_g` e
`total_nutrients` (PROCNT, FAT, CHOCDF, ENERC_KCAL) para a receita inteira.

Para que serve: alvo FRACO da cabeça de DENSIDADE (g de proteína por 100 g de comida), em escala. Não serve para
massa: a foto não mostra uma porção pesada. `densidade_proteina_fraca_g100` = 100 x proteína total / soma dos gramas
CRUS dos ingredientes (natureza_rotulo = receita_ingredientes_x_tabela). O peso pronto não é conhecido: cozinhar tira
água (carne, assado) ou põe água (arroz, massa), então a densidade crua erra para baixo na carne assada e para cima no
arroz e na massa cozidos. `massa_porcao_fraca_g` e `proteina_porcao_fraca_g` dividem a receita pelo número de porções
declarado, só para referência.

Uso:
  curl -sL https://huggingface.co/datasets/datahiveai/recipes-with-nutrition/resolve/main/recipes-with-nutrition.csv \
    | python3 -I receitas_datahive_monta.py monta - [--saida <csv>]
As fotos baixam de `url_imagem` (bucket S3 do DataHive, sem login; 300x300 ou 600x600).

UMA LINHA POR RECEITA. Colunas de alvo de treino_v2 VAZIAS. Regras, com contagem impressa:
  H1  sem peso total, sem proteína ou sem porções: sai;
  H2  tipo de prato fora de prato de comida (sobremesa, bebida, drinque, molho, biscoito, pão sozinho, conserva,
      cereal) ou sem tipo: sai. Ficam main course, salad, starter, soup, sandwiches, egg e pancake;
  H3  mais de 20% das linhas de ingrediente sem peso: sai (a densidade não fecha);
  H3b alguma fonte de proteína sem peso: sai (a densidade cairia);
  H4  água, gelo ou caldo acima de 30% do peso cru, fora de sopa: sai (água de cozimento escorrida ou evaporada
      entra no peso e derruba a densidade); de 10% a 30%, fica com o aviso `agua_na_receita`;
  H5  kcal fora de 25% de 4P + 4C + 9G: sai (análise incoerente);
  H6  densidade de proteína acima de 40 g/100 g: sai (ingrediente seco ou erro de peso);
  H7  a mesma foto em duas receitas: fica a primeira;
  H8  porção (peso cru / porções) fora de 50 a 1.500 g: as colunas por porção ficam vazias; a densidade fica.
`ff_fraca_<fonte>` = fração do peso cru de cada fonte de proteína (fonte_de de dens_comum.py sobre o nome do
ingrediente). `grupo` = nome normalizado da receita (receitas homônimas ficam do mesmo lado); regra de reserva de 12.5.
"""
import csv, hashlib, json, os, re, sys
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, "..")); sys.path.insert(0, os.path.join(AQUI, "..", "treino"))
from flavoria_monta import FONTES, confere_esquema  # noqa: E402
from dens_comum import fonte_de  # noqa: E402

SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_receitas_datahive.csv")
LICENCA, USO = "CC BY-NC 4.0 (cartão do dataset); fotos e receitas dos sites de origem", "nao_comercial"
FICA = {"main course", "salad", "starter", "soup", "sandwiches", "egg", "pancake"}
SAI = {"desserts", "drinks", "alcohol cocktail", "condiments and sauces", "biscuits and cookies", "preserve", "preps",
       "cereals", "bread"}
AGUA = re.compile(r"^(water|ice|ice water|ice cubes|boiling water|cold water|hot water|warm water|tap water)$", re.I)
CALDO = re.compile(r"broth|stock|bouillon", re.I)
PROTEINA_ALTA_G = 25.0


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cmd_monta(origem, saida):
    csv.field_size_limit(10 ** 9)
    entrada = sys.stdin if origem == "-" else open(origem, encoding="utf-8", newline="")
    base = confere_esquema()
    cab = (base + ["dominio", "grupo", "vista", "split_ext", "url_imagem", "url_receita", "site", "tipo_prato",
                   "densidade_proteina_fraca_g100", "kcal_fraca_100g", "gordura_fraca_g100", "carbo_fraca_g100",
                   "massa_receita_crua_g", "porcoes", "massa_porcao_fraca_g", "proteina_porcao_fraca_g"]
           + [f"ff_fraca_{k}" for k in FONTES] + ["natureza_rotulo", "componentes", "n_componentes", "avisos", "licenca", "uso_declarado"])
    cont = Counter(); linhas = []; fotos = set()
    for i, x in enumerate(csv.DictReader(entrada)):
        cont["receitas_lidas"] += 1
        try:
            tn = json.loads(x["total_nutrients"] or "{}"); ing = json.loads(x["ingredients"] or "[]")
            tipos = set(json.loads(x["dish_type"] or "[]"))
        except ValueError:
            cont["H1_sem_dado"] += 1; continue
        q = lambda k: f((tn.get(k) or {}).get("quantity"))
        peso, p, kcal, g, c, porc = f(x["total_weight_g"]), q("PROCNT"), q("ENERC_KCAL"), q("FAT"), q("CHOCDF"), f(x["servings"])
        if not peso or p is None or not porc or kcal is None or g is None or c is None:
            cont["H1_sem_dado"] += 1; continue
        if not tipos or tipos & SAI or not tipos & FICA:
            cont["H2_tipo_de_prato"] += 1; continue
        if ing and sum(1 for a in ing if not a.get("weight")) > 0.2 * len(ing):
            cont["H3_ingrediente_sem_peso"] += 1; continue
        if any(not a.get("weight") and fonte_de(a.get("food") or "") != "sem_proteina" for a in ing):
            cont["H3b_fonte_de_proteina_sem_peso"] += 1; continue
        agua = sum(a.get("weight") or 0 for a in ing if AGUA.match((a.get("food") or "").strip())
                   or CALDO.search(a.get("food") or ""))
        sopa = "soup" in tipos
        if agua > 0.3 * peso and not sopa:
            cont["H4_agua_acima_de_30pc"] += 1; continue
        if kcal <= 0 or abs(4 * p + 4 * c + 9 * g - kcal) > 0.25 * kcal:
            cont["H5_kcal_incoerente"] += 1; continue
        dens = 100 * p / peso
        if dens > 40:
            cont["H6_densidade_acima_de_40"] += 1; continue
        if x["image_url"] in fotos:
            cont["H7_foto_repetida"] += 1; continue
        fotos.add(x["image_url"])
        ff = dict.fromkeys(FONTES, 0.0)
        for a in ing:
            ff[fonte_de(a.get("food") or "")] += a.get("weight") or 0
        tot = sum(ff.values()) or 1.0
        mp = peso / porc
        avisos = [a for a, s in (("agua_na_receita", agua > 0.1 * peso), ("sopa", sopa)) if s]
        if not 50 <= mp <= 1500:
            avisos.append("porcao_fora_de_50_a_1500g"); cont["H8_porcao_fora"] += 1
        nome = re.sub(r"[^a-z0-9]+", " ", x["recipe_name"].lower()).strip()
        grupo = "dhv_" + hashlib.sha1(nome.encode()).hexdigest()[:10]
        split = "teste_ext" if int(hashlib.sha1(grupo.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
        porcao_ok = 50 <= mp <= 1500
        linhas.append([f"dhv_{i:05d}", -1, -1, ""] + [""] * (len(base) - 4)
                      + ["receitas_datahive", grupo, "receita_web", split, x["image_url"], x["url"], x["source"],
                         "|".join(sorted(tipos)), f"{dens:.2f}", f"{100 * kcal / peso:.1f}", f"{100 * g / peso:.2f}",
                         f"{100 * c / peso:.2f}", f"{peso:.0f}", f"{porc:g}", f"{mp:.0f}" if porcao_ok else "",
                         f"{p / porc:.1f}" if porcao_ok else ""]
                      + [f"{ff[k] / tot:.4f}" for k in FONTES]
                      + ["receita_ingredientes_x_tabela",
                         ";".join(f"{(a.get('food') or '').replace(';', ',')}:{(a.get('weight') or 0):.0f}" for a in ing)[:1500],
                         len(ing), "|".join(avisos), LICENCA, USO])
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    ix = {k: cab.index(k) for k in cab}
    d = sorted(float(l[ix["densidade_proteina_fraca_g100"]]) for l in linhas)
    pp = [float(l[ix["proteina_porcao_fraca_g"]]) for l in linhas if l[ix["proteina_porcao_fraca_g"]]]
    print("regras:", dict(cont))
    print(f"receitas (linhas): {len(linhas)} | split: {dict(Counter(l[ix['split_ext']] for l in linhas))}")
    print(f"  densidade de proteína (g/100 g cru): p10 {d[len(d) // 10]:.1f} mediana {d[len(d) // 2]:.1f} p90 {d[9 * len(d) // 10]:.1f}"
          f" | faixas 0-3/3-6/6-10/10-15/>15: {[sum(a < v <= b for v in d) for a, b in ((-1, 3), (3, 6), (6, 10), (10, 15), (15, 99))]}")
    print(f"  proteína por porção > {PROTEINA_ALTA_G:.0f} g: {sum(v > PROTEINA_ALTA_G for v in pp)} de {len(pp)}")
    print("  avisos:", dict(Counter(a for l in linhas for a in l[ix["avisos"]].split("|") if a)))
    print("  fonte com >= 20% do peso:", {k: sum(float(l[ix[f'ff_fraca_{k}']]) >= 0.2 for l in linhas) for k in FONTES})
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
