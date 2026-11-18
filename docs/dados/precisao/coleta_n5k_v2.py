#!/usr/bin/env python3
"""v2: amostra de ~400 pratos NOVOS do Nutrition5k (CC BY 4.0), com gabarito por ingrediente.

Uso: python3 coleta_n5k_v2.py <dir_n5k> <dir_saida_imagens> [--sem-download]
  <dir_n5k> contém dish_metadata_cafe1.csv, dish_metadata_cafe2.csv, ingredients_metadata.csv e overhead_ids.txt
  (pastas de imagery/realsense_overhead, via `gsutil ls`).
  Grava no repositório (ao lado deste script):
    n5k_v2_amostra.csv  — 1 linha por prato (id neutro dNNNN, dish_id, split, cafe, estrato, massa, proteína)
  e baixa rgb.png de cada prato para <dir_saida_imagens>/<id>.png (fora do repositório).

Regras:
  - exclui os 50 dish_ids já usados no v1 (gabarito_pratos.csv, origem n5k);
  - exige pasta overhead, massa >= 20 g, nenhum ingrediente "deprecated"/"plate only",
    soma das proteínas dos ingredientes = total_protein (±5% + 0,5 g) e soma das gramas = total_mass (±5% + 2 g);
  - estratos: proteína (0-5, 5-15, 15-25, 25-40, >40 g) × nº de ingredientes (1, 2-4, 5+);
    80 pratos por faixa de proteína, divididos igualmente entre as 3 faixas de ingredientes
    (o que faltar numa célula é completado pelas outras da mesma faixa de proteína);
    em cada célula ~20% vêm do café 2 (se houver);
  - split determinístico por faixa de proteína: 24 "calib" + 56 "teste" (120/280 no total);
  - ids neutros d0001… atribuídos em ordem aleatória (seed fixa), sem relação com conteúdo.
"""
import csv
import os
import random
import subprocess
import sys

BUCKET = "gs://nutrition5k_dataset/nutrition5k_dataset"
SEED = 20261110
EST_PROT = [(0, 5), (5, 15), (15, 25), (25, 40), (40, 1e9)]
EST_ING = [(1, 1, "1"), (2, 4, "2-4"), (5, 999, "5+")]
POR_FAIXA = 80
CALIB_POR_FAIXA = 24
FRAC_CAFE2 = 0.2
CAFE2_INICIO = 1571931457
AQUI = os.path.dirname(os.path.abspath(__file__))


def ler_nomes_ingr(path):
    """ingredients_metadata.csv -> {ingr_id: nome}; corrige nomes vazios no CSV do café 2."""
    nomes = {}
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                nomes[f"ingr_{int(r['id']):010d}"] = r["ingr"].strip()
    return nomes


def ler_meta(paths, ingr_meta=None):
    """Lê os CSVs de pratos. No café 2 os totais do prato vêm zerados (só a massa total é
    preenchida); nesse caso proteína/kcal do prato = soma dos ingredientes."""
    nomes = ler_nomes_ingr(ingr_meta)
    pratos = {}
    for p in paths:
        with open(p, newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].startswith("dish_"):
                    continue
                ingr = []
                for i in range(6, len(row) - 6, 7):
                    nome = row[i + 1].strip() or nomes.get(row[i], "")
                    ingr.append({"ingr_id": row[i], "nome": nome, "gramas": float(row[i + 2]),
                                 "kcal": float(row[i + 3]), "proteina": float(row[i + 6])})
                prot, kcal, tot_zerado = float(row[5]), float(row[1]), False
                if prot == 0 and kcal == 0 and ingr:
                    prot = sum(x["proteina"] for x in ingr)
                    kcal = sum(x["kcal"] for x in ingr)
                    tot_zerado = True
                pratos[row[0]] = {"massa": float(row[2]), "proteina": prot, "kcal": kcal,
                                  "ingr": ingr, "totais_da_soma": tot_zerado}
    return pratos


def valido(m):
    if m["massa"] < 20:
        return False
    if any(i["nome"].lower() in ("", "deprecated", "plate only") for i in m["ingr"]):
        return False
    sp = sum(i["proteina"] for i in m["ingr"])
    sg = sum(i["gramas"] for i in m["ingr"])
    return abs(sp - m["proteina"]) <= 0.05 * m["proteina"] + 0.5 and abs(sg - m["massa"]) <= 0.05 * m["massa"] + 2


def faixa_ing(n):
    return next(lbl for lo, hi, lbl in EST_ING if lo <= n <= hi)


def cafe(d):
    return "cafe2" if int(d.split("_")[1]) >= CAFE2_INICIO else "cafe1"


def ja_usados():
    usados = set()
    with open(os.path.join(AQUI, "gabarito_pratos.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["origem"] == "n5k":
                usados.add(r["descricao"].split(":")[0].strip())
    return usados


def amostrar(meta, over, usados, rnd):
    cand = sorted(d for d in over if d in meta and d not in usados and valido(meta[d]))
    escolhidos = []
    for lo, hi in EST_PROT:
        faixa = [d for d in cand if lo <= meta[d]["proteina"] < hi]
        celulas = {lbl: [d for d in faixa if faixa_ing(len(meta[d]["ingr"])) == lbl] for _, _, lbl in EST_ING}
        alvo = {lbl: POR_FAIXA // 3 for lbl in celulas}
        alvo["5+"] += POR_FAIXA - sum(alvo.values())
        # redistribui falta de células pequenas
        falta = sum(max(0, alvo[l] - len(celulas[l])) for l in celulas)
        for l in celulas:
            alvo[l] = min(alvo[l], len(celulas[l]))
        for l in sorted(celulas, key=lambda l: -len(celulas[l])):
            extra = min(falta, len(celulas[l]) - alvo[l])
            alvo[l] += extra
            falta -= extra
        sel_faixa = []
        for l, pool in celulas.items():
            c2 = [d for d in pool if cafe(d) == "cafe2"]
            c1 = [d for d in pool if cafe(d) == "cafe1"]
            n2 = min(len(c2), round(alvo[l] * FRAC_CAFE2))
            n1 = min(len(c1), alvo[l] - n2)
            n2 = alvo[l] - n1  # completa com café 2 se faltar café 1
            sel = rnd.sample(c1, n1) + rnd.sample(c2, n2)
            sel_faixa += [(d, f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}", l) for d in sel]
            print(f"proteína {lo:g}-{hi:g} ing {l}: " if hi < 1e8 else f"proteína >{lo:g} ing {l}: ", end="")
            print(f"{len(pool)} candidatos ({len(c2)} café 2) -> {n1}+{n2}")
        # split determinístico dentro da faixa
        rnd.shuffle(sel_faixa)
        for k, (d, fp, fi) in enumerate(sel_faixa):
            escolhidos.append((d, fp, fi, "calib" if k < CALIB_POR_FAIXA else "teste"))
    return escolhidos


def main(wd, img_dir, baixar=True):
    meta = ler_meta([os.path.join(wd, f) for f in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv")],
                    os.path.join(wd, "ingredients_metadata.csv"))
    over = {l.strip() for l in open(os.path.join(wd, "overhead_ids.txt")) if l.strip()}
    rnd = random.Random(SEED)
    escolhidos = amostrar(meta, over, ja_usados(), rnd)
    ordem = list(range(len(escolhidos)))
    rnd.shuffle(ordem)
    linhas = []
    for novo, k in enumerate(ordem, 1):
        d, fp, fi, split = escolhidos[k]
        m = meta[d]
        linhas.append({"prato_id": f"d{novo:04d}", "dish_id": d, "split": split, "cafe": cafe(d),
                       "faixa_proteina": fp, "faixa_ingredientes": fi, "n_ingredientes": len(m["ingr"]),
                       "massa_g": round(m["massa"], 1), "proteina_g": round(m["proteina"], 2),
                       "kcal": round(m["kcal"], 1),
                       "confianca": "B" if m["totais_da_soma"] else "A",
                       "ingredientes": "; ".join(f"{i['nome']} {i['gramas']:.0f} g" for i in m["ingr"])})
    linhas.sort(key=lambda r: r["prato_id"])
    with open(os.path.join(AQUI, "n5k_v2_amostra.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    if baixar:
        os.makedirs(img_dir, exist_ok=True)
        for r in linhas:
            dst = os.path.join(img_dir, f"{r['prato_id']}.png")
            if not os.path.exists(dst):
                subprocess.run(["gsutil", "-q", "cp",
                                f"{BUCKET}/imagery/realsense_overhead/{r['dish_id']}/rgb.png", dst], check=True)
    print(f"{len(linhas)} pratos; calib={sum(r['split']=='calib' for r in linhas)}; "
          f"café 2={sum(r['cafe']=='cafe2' for r in linhas)}")
    return meta, linhas


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], "--sem-download" not in sys.argv)
