#!/usr/bin/env python3
"""Manifestos (mesmo formato de manifestos.py) para datasets que entram como "Input" de um notebook do Kaggle.

Hoje só o NutritionVerse-Real (`nutritionverse/nutritionverse-real`). A licença dele é CC BY-NC-SA 4.0:
serve para AVALIAR, nunca para treinar o modelo do produto. Por isso todas as linhas saem com eval_holdout=1.
Os demais datasets de `fontes_treino.json` não precisam de leitor aqui (ver o campo "como_ler" de cada um).

Uso (no notebook, com o dataset anexado):
  python3 -I manifestos_kaggle.py nvreal /kaggle/input/nutritionverse-real <saida.csv>

Rótulos conferidos em 11/11/2026 no arquivo original: 251 pratos no CSV, 889 fotos, 225 pratos com foto
(26 pratos do CSV não têm foto). A soma dos itens confere com o total em todos os pratos.

ATENÇÃO, achado da conferência: o peso NÃO foi medido prato a prato. Cada um dos 45 tipos de item tem um único
peso no CSV inteiro (toda "chicken-wing" pesa 43 g, toda "french-fry" 5 g, todo "meatloaf" 701 g), e o total do
prato é a soma desses pesos nominais. Em itens contados (asinha, sushi, ovo) isso é plausível; em itens cortados
não é: o prato 100 mostra duas fatias finas de bolo de carne e está rotulado com 2 × 701 g = 1.402 g.
Os tipos em NVREAL_PESO_SUSPEITO levam a nota `peso_nominal_suspeito`; filtre-os antes de pontuar.
"""
import csv, glob, os, re, sys

from manifestos import COLS, fmt, write  # noqa: F401  (COLS documenta o formato)

NVREAL_LICENCA = "CC BY-NC-SA 4.0 (NAO comercial: so avaliacao)"
NVREAL_MAX_ITENS = 7
# Itens servidos em pedaços de tamanho variável, mas rotulados com o peso de uma peça inteira (ver o aviso acima).
# Lista feita pelos nomes e por uma foto conferida (prato 100); os demais tipos não foram conferidos foto a foto.
NVREAL_PESO_SUSPEITO = {"meatloaf", "near-whole-chicken", "lasagna"}


def nvreal_pratos(csv_path):
    """dish_id -> dict(mass, prot, kcal, ing=[(nome, gramas)]). Os totais já vêm em gramas e kcal."""
    pratos = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        ing = []
        for i in range(1, NVREAL_MAX_ITENS + 1):
            nome = r.get(f"food_item_type_{i}")
            if nome:
                ing.append((nome, float(r[f"food_weight_g_{i}"])))
        pratos[r["dish_id"]] = dict(mass=float(r["total_food_weight"]), prot=float(r["total_protein"]),
                                    kcal=float(r["total_calories"]), ing=ing)
    return pratos


def do_nvreal(raiz, saida):
    pratos = nvreal_pratos(os.path.join(raiz, "nutritionverse_dish_metadata3.csv"))
    pasta = os.path.join(raiz, "nutritionverse-manual", "nutritionverse-manual")
    split = {}
    sp_csv = os.path.join(pasta, "updated-manual-dataset-splits.csv")
    if os.path.exists(sp_csv):
        for r in csv.DictReader(open(sp_csv)):
            split[r["file_name"]] = r["category"].lower()
    rows, sem_rotulo = [], 0
    for img in sorted(glob.glob(os.path.join(pasta, "images", "*.jpg"))):
        nome = os.path.basename(img)
        m = re.match(r"dish_(\d+)_", nome)
        p = pratos.get(m.group(1)) if m else None
        if p is None:
            sem_rotulo += 1
            continue
        rows.append(dict(dataset="nutritionverse_real", image_path=img, dish_id="nv_" + m.group(1),
                         split=split.get(nome, ""), mass_g=fmt(p["mass"], 1), protein_g=fmt(p["prot"]),
                         kcal=fmt(p["kcal"], 1), ingredients=";".join(f"{n}:{g:.1f}" for n, g in p["ing"]),
                         camera="celular (de cima e inclinada; angulo nao rotulado)", has_depth=0,
                         license=NVREAL_LICENCA, eval_holdout=1,
                         notes="|".join(["roboflow_export", "so_avaliacao", "peso_nominal_por_tipo"]
                                        + (["peso_nominal_suspeito"]
                                           if any(n in NVREAL_PESO_SUSPEITO for n, _ in p["ing"]) else []))))
    print("nvreal: fotos sem rótulo:", sem_rotulo, "| pratos com foto:", len({r["dish_id"] for r in rows}))
    write(saida, rows)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "nvreal":
        sys.exit(__doc__)
    do_nvreal(sys.argv[2], sys.argv[3])
