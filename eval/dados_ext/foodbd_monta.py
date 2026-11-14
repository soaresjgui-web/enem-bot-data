#!/usr/bin/env python3
"""Monta a lista de treino externo do FoodBD (esquema de treino_v2 + colunas de 12.4 + procedência).

Dataset: FoodBD: A Polygon-Annotated Meal Image Dataset of Bangladeshi Cuisine with Visual and Nutritional Labels.
Mendeley Data, V2 (publicado em 18/08/2025). https://doi.org/10.17632/xh3ghf3jbg.2 — licença CC BY 4.0.
(As cópias do Kaggle dizem CC BY-SA 4.0; vale a do registro original.)

O que é: 3.523 fotos de celular de refeições caseiras de Bangladesh (arroz, curry de peixe ou frango, dal, ovo,
ruti, legumes), quase todas de cima ou perto disso, num prato só, em casa. Polígonos de 67 classes em todas.
Em 1.837 fotos, uma nutricionista estimou carboidrato, PROTEÍNA, gordura, fibra, kcal e carga glicêmica do prato
(MealNutrition1837_All.csv). NÃO há peso: nada foi pesado, e a estimativa é por prato inteiro.

Uso:
  curl -sS https://data.mendeley.com/public-api/datasets/xh3ghf3jbg -o ds.json      # lista de arquivos e URLs
  (baixar MealNutrition1837_All.csv e FoodBD_Meta_data.csv pelas URLs de ds.json, na mesma pasta)
  python3 -I foodbd_monta.py monta <pasta com ds.json e os dois csv> [--saida <csv>]
Cada imagem baixa da sua `url` (API pública do Mendeley Data, sem login).

UMA LINHA POR FOTO. Todas as colunas de alvo de treino_v2 ficam VAZIAS (fora dish_id, cadeia = -1, fold = -1): a
estimativa da nutricionista vai em colunas à parte, como o autorrelato do SNAPMe:
  proteina_fraca_g, kcal_fraca, gordura_fraca_g, carbo_fraca_g, fibra_fraca_g  e  natureza_rotulo =
  estimado_por_nutricionista. `massa_fraca_g` fica vazia (não há massa).
Regras:
  D1  foto da tabela de nutrição sem a foto no registro: sai;
  D2  proteína zero ou kcal abaixo de 50 (prato de fruta ou chá): sai;
  D3  kcal fora de 25% de 4P + 4C + 9G: fica, com o aviso `kcal_incoerente`.
`componentes` = classes dos polígonos (FoodBD_Meta_data.csv). `fontes_presentes` = fontes de proteína entre elas
(mapa FONTE_DA_CLASSE). Os valores vêm em dois estilos, inteiros e com decimais, em blocos alternados de ids
(provavelmente duas rodadas de estimativa); fica em `estilo_valor`.
Parte segurada: não há id de pessoa nem de data. Fotos de ids próximos são da mesma casa (mesma toalha, mesmo
prato), então `grupo` = bloco de 25 ids seguidos e vale a regra de reserva de 12.5: sha1(grupo) mod 5 == 0 ->
teste_ext. A divisão dos autores fica em `split_autores`.
"""
import csv, hashlib, json, os, sys
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import confere_esquema  # noqa: E402

SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_foodbd.csv")
LICENCA, USO = "CC BY 4.0", "comercial_ok"
BLOCO = 25
PROTEINA_ALTA_G = 25.0
FONTE_DA_CLASSE = {
    "chicken": "frango_peru", "beef": "bovina", "kabab": "bovina", "fish": "peixe_marisco",
    "egg": "ovo", "egg-boiled": "ovo", "egg-poached": "ovo",
    "daal": "leguminosa_soja", "chola": "leguminosa_soja", "green-pea": "leguminosa_soja", "shim": "leguminosa_soja",
    "peyaju": "leguminosa_soja", "piyaju": "leguminosa_soja",
    "milk": "laticinio", "payesh": "laticinio", "badam": "nozes_sementes",
    "biriyani": "prato_misto", "khichuri": "prato_misto", "roll": "prato_misto", "noodles": "prato_misto",
}


def num(x):
    return float(str(x).strip())


def cmd_monta(pasta, saida):
    with open(os.path.join(pasta, "ds.json"), encoding="utf-8") as f:
        ds = json.load(f)
    assert ds["doi"]["id"] == "10.17632/xh3ghf3jbg.2", ds["doi"]
    url = {x["filename"]: x["content_details"]["download_url"] for x in ds["files"]}
    with open(os.path.join(pasta, "FoodBD_Meta_data.csv"), encoding="utf-8-sig", newline="") as f:
        meta = {r["filename"]: r for r in csv.DictReader(f)}
    with open(os.path.join(pasta, "MealNutrition1837_All.csv"), encoding="utf-8-sig", newline="") as f:
        nut = list(csv.DictReader(f))
    base = confere_esquema()
    cab = base + ["dominio", "grupo", "vista", "split_ext", "arquivo", "url", "split_autores", "massa_fraca_g",
                  "proteina_fraca_g", "kcal_fraca", "gordura_fraca_g", "carbo_fraca_g", "fibra_fraca_g", "carga_glicemica",
                  "natureza_rotulo", "estilo_valor", "componentes", "n_componentes", "fontes_presentes", "avisos",
                  "licenca", "uso_declarado"]
    cont = Counter(linhas_tabela=len(nut)); linhas = []
    for r in sorted(nut, key=lambda r: r["FoodImage"]):
        arq = r["FoodImage"].strip()
        if arq not in url:
            cont["D1_sem_foto"] += 1; continue
        p, kcal, g, c = num(r["Protein"]), num(r["Calorie"]), num(r["Fat"]), num(r["CHO"])
        if p <= 0 or kcal < 50:
            cont["D2_proteina_zero_ou_kcal_baixa"] += 1; continue
        m = meta.get(arq, {})
        comp = [x for x in m.get("instances", "").split(",") if x]
        fontes = sorted({FONTE_DA_CLASSE[x] for x in comp if x in FONTE_DA_CLASSE})
        avisos = []
        if abs(4 * p + 4 * c + 9 * g - kcal) > 0.25 * kcal:
            avisos.append("kcal_incoerente"); cont["D3_kcal_incoerente"] += 1
        if not m:
            avisos.append("sem_metadado")
        n = int(arq[7:11])
        grupo = f"fbd_b{n // BLOCO:03d}"
        split = "teste_ext" if int(hashlib.sha1(grupo.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
        estilo = "inteiro" if all(float(r[k]) == int(float(r[k])) for k in ("Protein", "CHO", "Fat", "Calorie")) else "decimal"
        linhas.append([f"fbd_{n:04d}", -1, -1, ""] + [""] * (len(base) - 4)
                      + ["foodbd", grupo, "celular_de_cima", split, arq, url[arq], m.get("types", ""), "",
                         f"{p:.2f}", f"{kcal:.1f}", f"{g:.2f}", f"{c:.2f}", f"{num(r['Fiber']):.2f}", r["GL"].strip(),
                         "estimado_por_nutricionista", estilo, ";".join(comp), len(comp), ";".join(fontes), "|".join(avisos),
                         LICENCA, USO])
    assert len({l[0] for l in linhas}) == len(linhas)
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    i = {k: cab.index(k) for k in cab}
    p = sorted(float(l[i["proteina_fraca_g"]]) for l in linhas)
    faixa = [sum(a < x <= b for x in p) for a, b in ((-1, 5), (5, 15), (15, 25), (25, 40), (40, 1e9))]
    print("regras:", dict(cont))
    print(f"fotos (linhas): {len(linhas)} | grupos: {len({l[i['grupo']] for l in linhas})} | split: {dict(Counter(l[i['split_ext']] for l in linhas))}")
    print(f"  proteína estimada (g): p10 {p[len(p) // 10]:.1f} mediana {p[len(p) // 2]:.1f} p90 {p[9 * len(p) // 10]:.1f} máx {p[-1]:.1f}"
          f" | faixas 0-5/5-15/15-25/25-40/>40: {faixa} | > {PROTEINA_ALTA_G:.0f} g: {sum(x > PROTEINA_ALTA_G for x in p)}")
    print("  fotos com a fonte:", dict(Counter(k for l in linhas for k in l[i["fontes_presentes"]].split(";") if k)))
    print("  estilo:", dict(Counter(l[i["estilo_valor"]] for l in linhas)), "| avisos:", dict(Counter(a for l in linhas for a in l[i["avisos"]].split("|") if a)))
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
