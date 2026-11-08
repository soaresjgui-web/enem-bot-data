#!/usr/bin/env python3
"""Pontua a precisão da estimativa de proteína por foto (protótipo Nutrelle).

Uso:
    python3 score_precisao.py gabarito.csv estimativas.csv

gabarito.csv  — um componente pesado por linha:
    prato_id,categoria,componente,gramas,proteina_100g
    (proteina_100g vem da TACO/TBCA ou do rótulo)

estimativas.csv — uma estimativa por prato e por sistema:
    prato_id,sistema,proteina_estimada_g

Saída: por sistema (e por categoria), erro absoluto médio em gramas,
erro percentual médio, % de pratos dentro de ±20% e % dentro de ±5 g.
"""
import csv
import sys
from collections import defaultdict
from statistics import mean, median

TOL_PCT = 0.20
TOL_G = 5.0


def load_truth(path):
    proteina = defaultdict(float)
    categoria = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["prato_id"].strip()
            proteina[pid] += float(row["gramas"]) * float(row["proteina_100g"]) / 100
            categoria[pid] = row["categoria"].strip()
    return proteina, categoria


def load_estimates(path):
    est = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            est[row["sistema"].strip()][row["prato_id"].strip()] = float(row["proteina_estimada_g"])
    return est


def metrics(pairs):
    erros = [abs(e - r) for r, e in pairs]
    pct = [abs(e - r) / r for r, e in pairs if r > 0]
    vies = [e - r for r, e in pairs]
    return {
        "n": len(pairs),
        "erro_medio_g": mean(erros),
        "erro_mediano_g": median(erros),
        "erro_medio_pct": mean(pct) * 100 if pct else float("nan"),
        "vies_g": mean(vies),
        "dentro_20pct": 100 * sum(p <= TOL_PCT for p in pct) / len(pct) if pct else float("nan"),
        "dentro_5g": 100 * sum(e <= TOL_G for e in erros) / len(erros),
    }


def fmt(m):
    return (f"n={m['n']:>3}  erro médio {m['erro_medio_g']:5.1f} g (mediana {m['erro_mediano_g']:4.1f})  "
            f"erro {m['erro_medio_pct']:5.1f}%  viés {m['vies_g']:+5.1f} g  "
            f"±20%: {m['dentro_20pct']:5.1f}%  ±5 g: {m['dentro_5g']:5.1f}%")


def main(truth_path, est_path):
    real, categoria = load_truth(truth_path)
    est = load_estimates(est_path)
    for sistema in sorted(est):
        faltando = sorted(set(real) - set(est[sistema]))
        pares = [(real[p], e) for p, e in est[sistema].items() if p in real]
        print(f"\n== {sistema} ==")
        if faltando:
            print(f"  aviso: {len(faltando)} prato(s) sem estimativa: {', '.join(faltando[:10])}")
        if not pares:
            continue
        print("  geral      ", fmt(metrics(pares)))
        por_cat = defaultdict(list)
        for p, e in est[sistema].items():
            if p in real:
                por_cat[categoria[p]].append((real[p], e))
        for cat in sorted(por_cat):
            print(f"  {cat:<11}", fmt(metrics(por_cat[cat])))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
