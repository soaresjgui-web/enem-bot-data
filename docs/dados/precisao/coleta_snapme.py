#!/usr/bin/env python3
"""v2: seleciona ~150 fotos do SNAPMe (USDA; Larke et al. 2023, Nutrients 15:4972) com gabarito por foto.

Uso: python3 coleta_snapme.py <master_SNAPME_linkfile.csv>
  O linkfile vem do repositório github.com/JulesLarke-USDA/SNAPMe (input/, CC BY-SA 4.0).
  Grava snapme_selecao.csv ao lado deste script (1 linha por foto).

GABARITO DE QUALIDADE BAIXA (confiança C): cada foto está ligada a alimentos do recordatório
ASA24 que a própria participante preencheu. A porção (FoodAmt, g) é AUTORRELATADA, escolhida
nas medidas caseiras/imagens do ASA24, e não pesada. Proteína = soma de PROT (FNDDS 2018) das
linhas ligadas à foto. Use o SNAPMe para medir identificação e robustez a "foto do mundo real",
não para calibrar o peso das porções.

As imagens ficam no Ag Data Commons (agdatacommons.nal.usda.gov), que em 08/11/2026 respondia
com desafio JavaScript do AWS WAF (HTTP 202, x-amzn-waf-action: challenge) e não puderam ser
baixadas. A seleção fica pronta para quando as imagens forem obtidas (o `filename` é o nome
do arquivo no pacote de imagens).

Regras: exclui fotos com qualquer alimento embalado (packaged_food != 0) e fotos com mais de
6 alimentos; estratifica por proteína (0-5, 5-15, 15-25, 25-40, >40 g), 30 por faixa
(completa com outras faixas se faltar), no máximo 3 fotos por participante em cada faixa.
"""
import csv
import os
import random
import sys
from collections import defaultdict

SEED = 20261111
FAIXAS = [(0, 5), (5, 15), (15, 25), (25, 40), (40, 1e9)]
POR_FAIXA = 30
MAX_POR_PESSOA = 3
AQUI = os.path.dirname(os.path.abspath(__file__))


def main(linkfile):
    fotos = defaultdict(list)
    with open(linkfile, encoding="latin-1", newline="") as f:
        for r in csv.DictReader(f):
            fotos[r["filename"]].append(r)
    cand = []
    for fn, rows in fotos.items():
        if any(r["packaged_food"] != "0" for r in rows) or len(rows) > 6:
            continue
        cand.append({
            "filename": fn, "subject_id": rows[0]["subject_id"], "dia": rows[0]["snapme_study_day"],
            "n_alimentos": len(rows),
            "gramas_total": round(sum(float(r["FoodAmt"]) for r in rows), 1),
            "proteina_g": round(sum(float(r["PROT"]) for r in rows), 2),
            "itens": "; ".join(f"{r['Food_Description']} ({r['FoodCode']}): {float(r['FoodAmt']):.0f} g / "
                               f"{float(r['PROT']):.1f} g prot" for r in rows),
        })
    cand.sort(key=lambda c: c["filename"])
    rnd = random.Random(SEED)
    sel = []
    for lo, hi in FAIXAS:
        pool = [c for c in cand if lo <= c["proteina_g"] < hi]
        rnd.shuffle(pool)
        por_pessoa = defaultdict(int)
        n = 0
        for c in pool:
            if n >= POR_FAIXA:
                break
            if por_pessoa[c["subject_id"]] >= MAX_POR_PESSOA:
                continue
            por_pessoa[c["subject_id"]] += 1
            c["faixa_proteina"] = f"{lo}-{hi:g}" if hi < 1e8 else f">{lo}"
            sel.append(c)
            n += 1
        print(f"faixa {lo}-{hi:g}: " if hi < 1e8 else f"faixa >{lo}: ", f"{len(pool)} candidatas -> {n}")
    for k, c in enumerate(sel, 1):
        c["prato_id"] = f"s{k:03d}"
        c["confianca"] = "C"
    campos = ["prato_id", "filename", "subject_id", "dia", "faixa_proteina", "n_alimentos",
              "gramas_total", "proteina_g", "confianca", "itens"]
    with open(os.path.join(AQUI, "snapme_selecao.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(sel)
    print(f"{len(cand)} fotos candidatas de {len(fotos)}; {len(sel)} selecionadas")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
