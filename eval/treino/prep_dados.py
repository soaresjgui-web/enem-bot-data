#!/usr/bin/env python3
"""Monta a lista de pratos de TREINO do Nutrition5k e verifica que nenhum prato de avaliação vaza.

Uso: python3 prep_dados.py <dir_treino>
  <dir_treino>/meta deve ter splits/rgb_train_ids.txt, rgb_test_ids.txt (bucket oficial).
Lê do repositório: n5k_v2_amostra.csv (400 calib/teste) e gabarito_pratos.csv (50 n5k v1).
Grava <dir_treino>/meta/treino.csv (dish_id, massa, proteina, kcal, cafe) e
<dir_treino>/meta/avaliacao.csv (prato_id, dish_id, split, massa, proteina, kcal).

Treino = split oficial rgb_train ∩ pasta realsense_overhead, MENOS todos os 450 pratos
de avaliação (que foram sorteados de train e test oficiais). Filtros como no v2:
massa >= 20 g, sem ingrediente 'deprecated'/'plate only'/sem nome, soma dos ingredientes
coerente com o total (café 1). Café 2: totais vêm zerados -> soma dos ingredientes.
"""
import csv, os, re, sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PREC = os.path.join(REPO, "docs", "dados", "precisao")
N5K = "/tmp/claude-0/-home-user-enem-bot-data/59d3cb2b-a793-5c18-8c23-e9dab2604443/scratchpad/n5k"


def ler_meta():
    pratos = {}
    ingr_meta = {}
    with open(os.path.join(N5K, "ingredients_metadata.csv")) as f:
        for r in csv.DictReader(f):
            ingr_meta[r["id"].strip()] = r
    for cafe in ("cafe1", "cafe2"):
        with open(os.path.join(N5K, f"dish_metadata_{cafe}.csv")) as f:
            for row in csv.reader(f):
                if not row or not row[0].startswith("dish_"):
                    continue
                ingr = [row[i:i + 7] for i in range(6, len(row) - 6, 7)]
                nomes = [(x[1].strip() or ingr_meta.get(x[0].strip(), {}).get("ingr", "")) for x in ingr]
                ok = all(n and "deprecated" not in n and "plate only" not in n for n in nomes)
                sp = sum(float(x[6]) for x in ingr)
                sk = sum(float(x[3]) for x in ingr)
                sm = sum(float(x[2]) for x in ingr)
                massa, kcal, prot = float(row[2]), float(row[1]), float(row[5])
                if cafe == "cafe2":
                    prot, kcal = sp, sk
                    ok = ok and abs(sm - massa) <= 2 + 0.05 * massa
                else:
                    ok = ok and abs(sp - prot) <= 0.5 + 0.05 * prot
                pratos[row[0]] = dict(massa=massa, proteina=prot, kcal=kcal, cafe=cafe, ok=ok)
    return pratos


def main(td):
    meta = ler_meta()
    sp = os.path.join(td, "meta", "splits")
    train = {l.strip() for l in open(os.path.join(sp, "rgb_train_ids.txt")) if l.strip()}
    test = {l.strip() for l in open(os.path.join(sp, "rgb_test_ids.txt")) if l.strip()}
    over = {l.strip().rstrip("/").split("/")[-1] for l in open(os.path.join(N5K, "overhead_ids.txt")) if l.strip()}
    aval = []
    with open(os.path.join(PREC, "n5k_v2_amostra.csv")) as f:
        for r in csv.DictReader(f):
            aval.append((r["prato_id"], r["dish_id"], r["split"]))
    with open(os.path.join(PREC, "gabarito_pratos.csv")) as f:
        for r in csv.DictReader(f):
            if r["origem"] == "n5k":
                aval.append((r["prato_id"], re.match(r"(dish_\d+)", r["descricao"]).group(1), "n5k_v1"))
    aval_ids = {d for _, d, _ in aval}
    print(f"avaliação: {len(aval)} pratos, {len(aval_ids)} dish_ids distintos")
    for s in ("calib", "teste", "n5k_v1"):
        ids = {d for _, d, x in aval if x == s}
        print(f"  {s:6}: {len(ids)}  no rgb_train oficial: {len(ids & train)}  no rgb_test: {len(ids & test)}")
    cand = sorted(d for d in (train & over) if d in meta and meta[d]["ok"] and meta[d]["massa"] >= 20)
    treino = [d for d in cand if d not in aval_ids]
    print(f"rgb_train ∩ overhead ∩ filtros: {len(cand)}; removidos por estarem na avaliação: {len(cand) - len(treino)}")
    print(f"TREINO final: {len(treino)}; interseção treino × avaliação = {len(set(treino) & aval_ids)}")
    with open(os.path.join(td, "meta", "treino.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dish_id", "massa", "proteina", "kcal", "cafe"])
        for d in treino:
            m = meta[d]; w.writerow([d, m["massa"], round(m["proteina"], 3), round(m["kcal"], 2), m["cafe"]])
    with open(os.path.join(td, "meta", "avaliacao.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prato_id", "dish_id", "split", "massa", "proteina", "kcal"])
        for p, d, s in aval:
            m = meta[d]; w.writerow([p, d, s, m["massa"], round(m["proteina"], 3), round(m["kcal"], 2)])


if __name__ == "__main__":
    main(sys.argv[1])
