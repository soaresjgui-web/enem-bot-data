#!/usr/bin/env python3
"""Seção 8 (uma única foto de cima): monta o universo de treino ZENITAL e os alvos multitarefa.

Uso: python3 zen_dados.py <dir_treino>
Lê os metadados do Nutrition5k (cafés 1 e 2), a lista de pratos com foto zenital
(<dir_treino>/img/<dish_id>/rgb.png) e as listas de avaliação do repositório.
Grava <dir_treino>/meta/zen_treino.csv e <dir_treino>/meta/zen_aval.csv.

Higiene (a mesma da seção 7, regra "universal"):
  - os 450 dish_id de avaliação (calib/teste/n5k_v1) nunca entram no treino;
  - cadeia: pratos com < 120 s entre timestamps consecutivos, calculada sobre TODOS os pratos
    com metadado; sai do treino toda cadeia que contém um prato de avaliação;
  - filtros do v2: massa >= 20 g, sem ingrediente 'deprecated'/'plate only'/sem nome, soma coerente.
Diferença para as seções 2 e 7: entram também os pratos zenitais do split oficial de teste e os
sem split que não estão na avaliação nem em cadeia de avaliação (antes só rgb_train ∩ overhead).

Alvos por prato: massa, kcal, gordura, carboidrato, proteína e a massa por grupo de alimento
(eval/grupos_alimentos.py aplicado ao nome de cada ingrediente do Nutrition5k) e a proteína por grupo.
"""
import csv, os, re, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, ".."))
from comum import PREC, grupos_sessao  # noqa: E402
from grupos_alimentos import GRUPOS, grupo_de  # noqa: E402
from prep_dados import N5K  # noqa: E402

COLS = ["massa", "proteina", "kcal", "gordura", "carbo"]


def ler_meta_completo():
    """dish_id -> dict(massa, proteina, kcal, gordura, carbo, ok, cafe, mg{grupo: g}, pg{grupo: g de proteína})."""
    nomes_meta = {}
    with open(os.path.join(N5K, "ingredients_metadata.csv")) as f:
        for r in csv.DictReader(f):
            nomes_meta[r["id"].strip()] = r["ingr"]
    pratos = {}
    for cafe in ("cafe1", "cafe2"):
        with open(os.path.join(N5K, f"dish_metadata_{cafe}.csv")) as f:
            for row in csv.reader(f):
                if not row or not row[0].startswith("dish_"):
                    continue
                ingr = [row[i:i + 7] for i in range(6, len(row) - 6, 7)]
                nomes = [(x[1].strip() or nomes_meta.get(x[0].strip(), "")) for x in ingr]
                ok = all(n and "deprecated" not in n and "plate only" not in n for n in nomes)
                sm = sum(float(x[2]) for x in ingr); sk = sum(float(x[3]) for x in ingr)
                sf = sum(float(x[4]) for x in ingr); sc = sum(float(x[5]) for x in ingr)
                sp = sum(float(x[6]) for x in ingr)
                kcal, massa, gord, carb, prot = (float(v) for v in row[1:6])
                if cafe == "cafe2":  # totais zerados -> soma dos ingredientes
                    prot, kcal, gord, carb = sp, sk, sf, sc
                    ok = ok and abs(sm - massa) <= 2 + 0.05 * massa
                else:
                    ok = ok and abs(sp - prot) <= 0.5 + 0.05 * prot
                mg = dict.fromkeys(GRUPOS, 0.0); pg = dict.fromkeys(GRUPOS, 0.0)
                for x, n in zip(ingr, nomes):
                    g = grupo_de(n) if n else "outros"
                    mg[g] += float(x[2]); pg[g] += float(x[6])
                pratos[row[0]] = dict(massa=massa, proteina=prot, kcal=kcal, gordura=gord, carbo=carb, ok=ok,
                                      cafe=cafe, mg=mg, pg=pg, n_ingr=len(ingr))
    return pratos


def avaliacao():
    aval = []
    with open(os.path.join(PREC, "n5k_v2_amostra.csv")) as f:
        for r in csv.DictReader(f):
            aval.append((r["prato_id"], r["dish_id"], r["split"]))
    with open(os.path.join(PREC, "gabarito_pratos.csv")) as f:
        for r in csv.DictReader(f):
            if r["origem"] == "n5k":
                aval.append((r["prato_id"], re.match(r"(dish_\d+)", r["descricao"]).group(1), "n5k_v1"))
    return aval


def main(td):
    meta = ler_meta_completo()
    aval = avaliacao()
    ev = {d for _, d, _ in aval}
    todos = sorted(meta)
    cadeia = dict(zip(todos, grupos_sessao(todos, 120)))
    proib = {cadeia[d] for d in ev}
    com_foto = {d for d in os.listdir(os.path.join(td, "img"))
                if os.path.exists(os.path.join(td, "img", d, "rgb.png"))}
    sp = os.path.join(td, "meta", "splits")
    fonte = {}
    for nome in ("rgb_train_ids.txt", "rgb_test_ids.txt"):
        for l in open(os.path.join(sp, nome)):
            fonte[l.strip()] = nome.split("_")[1]
    cand = [d for d in sorted(com_foto) if d in meta and meta[d]["ok"] and meta[d]["massa"] >= 20 and d not in ev]
    treino = [d for d in cand if cadeia[d] not in proib]
    assert not set(treino) & ev
    assert not {cadeia[d] for d in treino} & proib
    print(f"pratos com foto zenital: {len(com_foto)} | passam nos filtros e fora da avaliação: {len(cand)}")
    print(f"TREINO zenital estrito (regra universal): {len(treino)} pratos, {len({cadeia[d] for d in treino})} cadeias")
    print("  por split oficial:", {k: sum(fonte.get(d, 'sem_split') == k for d in treino) for k in ('train', 'test', 'sem_split')})
    print("  por café:", {c: sum(meta[d]['cafe'] == c for d in treino) for c in ('cafe1', 'cafe2')})
    cab = ["dish_id", "cadeia", "split_oficial"] + COLS + [f"m_{g}" for g in GRUPOS] + [f"p_{g}" for g in GRUPOS]

    def linha(d):
        m = meta[d]
        return ([d, cadeia[d], fonte.get(d, "sem_split")] + [round(m[c], 3) for c in COLS]
                + [round(m["mg"][g], 2) for g in GRUPOS] + [round(m["pg"][g], 3) for g in GRUPOS])
    with open(os.path.join(td, "meta", "zen_treino.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cab)
        for d in treino:
            w.writerow(linha(d))
    with open(os.path.join(td, "meta", "zen_aval.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prato_id", "split"] + cab)
        for p, d, s in aval:
            w.writerow([p, s] + linha(d))
    print("avaliação:", {s: sum(x == s for _, _, x in aval) for s in ("calib", "teste", "n5k_v1")})


if __name__ == "__main__":
    main(sys.argv[1])
