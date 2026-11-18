#!/usr/bin/env python3
"""v2: monta os gabaritos (prato e componente) e os lotes cegos do teste ampliado.

Uso: python3 monta_gabarito_v2.py <dir_n5k> <dir_img_n5k_v2> <dir_cego_v1> <dir_cego2>
  <dir_n5k>         dish_metadata_cafe1/2.csv + ingredients_metadata.csv (Nutrition5k)
  <dir_img_n5k_v2>  imagens d0001.png… baixadas por coleta_n5k_v2.py
  <dir_cego_v1>     lotes cegos do v1 (lote1..lote5/<prato_id>.<ext>)
  <dir_cego2>       destino dos lotes v2 (lote01, lote02…; SÓ imagens, 40 por pasta)

Lê (desta pasta): gabarito_pratos.csv (v1, 194 itens), n5k_v2_amostra.csv (400 novos).
Grava (desta pasta):
  gabarito_componentes.csv  prato_id,componente,gramas,proteina_g,grupo,origem,confianca
                            (UFPR alimento único, Nutrition5k v1 e v2; Liv Up/Seara não têm componentes)
  gabarito_v2.csv           formato do eval/score_precisao.py (componentes quando houver)
  gabarito_pratos_v2.csv    1 linha por prato: origem, split, categoria, peso, proteína, confiança, lote
Splits: "calib" (120 Nutrition5k v2, para ajustar correções), "teste" (280 Nutrition5k v2) e
"v1" (os 194 itens do v1, também fora do ajuste — servem para comparar v1 × v2).
"""
import csv
import os
import random
import re
import shutil
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "..", "..", "eval"))
from coleta_n5k_v2 import ler_meta  # noqa: E402
from grupos_alimentos import grupo_de  # noqa: E402

SEED_LOTES = 20261112
POR_LOTE = 40


def ler(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def taco_do_obs(obs):
    m = re.search(r"TACO #\d+ '([^']+)'", obs)
    return m.group(1) if m else ""


def main(dir_n5k, dir_img, dir_cego1, dir_cego2):
    meta = ler_meta([os.path.join(dir_n5k, f) for f in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv")],
                    os.path.join(dir_n5k, "ingredients_metadata.csv"))
    taco_grupo = {r["nome"]: r["grupo"] for r in ler(os.path.join(AQUI, "ref", "taco.csv"))}
    v1 = ler(os.path.join(AQUI, "gabarito_pratos.csv"))
    v2 = ler(os.path.join(AQUI, "n5k_v2_amostra.csv"))

    comp, score, pratos = [], [], []

    def add_n5k(pid, dish, origem, conf):
        for i in meta[dish]["ingr"]:
            if i["gramas"] <= 0 and i["proteina"] <= 0:
                continue
            comp.append({"prato_id": pid, "componente": i["nome"], "gramas": round(i["gramas"], 1),
                         "proteina_g": round(i["proteina"], 3), "grupo": grupo_de(i["nome"]),
                         "origem": origem, "confianca": conf})
            if i["gramas"] > 0:
                score.append({"prato_id": pid, "categoria": "n5k", "componente": i["nome"],
                              "gramas": round(i["gramas"], 3),
                              "proteina_100g": round(100 * i["proteina"] / i["gramas"], 4)})

    for r in v1:
        pid = r["prato_id"]
        peso, prot = float(r["peso_porcao_g"]), float(r["proteina_g"])
        if r["origem"] == "n5k":
            dish = r["descricao"].split(":")[0].strip()
            add_n5k(pid, dish, "n5k_v1", "A")
        else:
            if r["fonte"].startswith("UFPR"):
                nome = r["descricao"].split(" — ")[0]
                tn = taco_do_obs(r["observacoes"])
                comp.append({"prato_id": pid, "componente": nome, "gramas": peso, "proteina_g": prot,
                             "grupo": grupo_de(tn or nome, taco_grupo.get(tn)), "origem": "ufpr",
                             "confianca": r["confianca"]})
            score.append({"prato_id": pid, "categoria": r["categoria"], "componente": "prato inteiro",
                          "gramas": peso, "proteina_100g": round(100 * prot / peso, 4)})
        origem = "n5k_v1" if r["origem"] == "n5k" else ("ufpr" if r["fonte"].startswith("UFPR") else
                                                        "liv_up" if r["categoria"] == "marmita" else "seara")
        pratos.append({"prato_id": pid, "origem": origem, "split": "v1", "categoria": r["categoria"],
                       "peso_porcao_g": peso, "proteina_g": prot, "confianca": r["confianca"]})

    for r in v2:
        add_n5k(r["prato_id"], r["dish_id"], "n5k_v2", r["confianca"])
        pratos.append({"prato_id": r["prato_id"], "origem": "n5k_v2", "split": r["split"], "categoria": "n5k",
                       "peso_porcao_g": r["massa_g"], "proteina_g": r["proteina_g"], "confianca": r["confianca"]})

    # lotes cegos: todos os itens embaralhados juntos, 40 por pasta
    imgs = {}
    for lote in sorted(os.listdir(dir_cego1)):
        d = os.path.join(dir_cego1, lote)
        if os.path.isdir(d) and lote.startswith("lote"):
            for fn in os.listdir(d):
                imgs[os.path.splitext(fn)[0]] = os.path.join(d, fn)
    for fn in os.listdir(dir_img):
        imgs[os.path.splitext(fn)[0]] = os.path.join(dir_img, fn)
    ids = sorted(p["prato_id"] for p in pratos)
    faltam = [i for i in ids if i not in imgs]
    if faltam:
        sys.exit(f"sem imagem: {faltam[:10]}")
    random.Random(SEED_LOTES).shuffle(ids)
    lote_de = {}
    for k, pid in enumerate(ids):
        lote = f"lote{k // POR_LOTE + 1:02d}"
        lote_de[pid] = lote
        dst = os.path.join(dir_cego2, lote)
        os.makedirs(dst, exist_ok=True)
        src = imgs[pid]
        shutil.copyfile(src, os.path.join(dst, pid + os.path.splitext(src)[1].lower()))
    for p in pratos:
        p["lote_v2"] = lote_de[p["prato_id"]]

    def grava(nome, linhas):
        with open(os.path.join(AQUI, nome), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(sorted(linhas, key=lambda x: x["prato_id"]) if nome != "gabarito_v2.csv" else linhas)

    grava("gabarito_componentes.csv", comp)
    grava("gabarito_v2.csv", score)
    grava("gabarito_pratos_v2.csv", pratos)
    n_lotes = len(set(lote_de.values()))
    print(f"{len(pratos)} pratos, {len(comp)} componentes, {n_lotes} lotes em {dir_cego2}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
