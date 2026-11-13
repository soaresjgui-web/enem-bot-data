#!/usr/bin/env python3
"""Simula o seletor de porção (6 tamanhos: PP a GG) sobre as estimativas da foto.

A foto dá os alimentos e uma porção estimada E (g). O app mostra 4 tamanhos
de referência ao redor de E: E × MULT (escala de 6 tamanhos). A usuária escolhe o tamanho que parece
mais com o prato. Proteína final = densidade estimada pela foto × peso escolhido.

Cenários:
  foto_sozinha  — sem seletor (baseline)
  seletor_ideal — usuária sempre escolhe o tamanho mais próximo do real
  seletor_real  — erra para um tamanho vizinho com prob. P_ERRO (média de N_SIM sorteios)

Uso: python3 simula_seletor.py gabarito_pratos.csv estimativas.csv
"""
import csv, random, sys
from collections import defaultdict

MULT = [0.5, 0.75, 1.0, 1.35, 1.8, 2.5]
P_ERRO = 0.30
N_SIM = 200


def ok(real, est):
    return abs(est - real) <= 5 or (real > 0 and abs(est - real) / real <= 0.20)


def main(gab, est):
    g = {r["prato_id"]: r for r in csv.DictReader(open(gab, encoding="utf-8"))}
    e = {r["prato_id"]: r for r in csv.DictReader(open(est, encoding="utf-8"))}
    rng = random.Random(42)
    grupos = defaultdict(lambda: defaultdict(list))
    for pid, r in g.items():
        if pid not in e:
            continue
        real_p, real_w = float(r["proteina_g"]), float(r["peso_porcao_g"])
        est_p, est_w = float(e[pid]["proteina_estimada_g"]), float(e[pid]["peso_total_estimado_g"])
        dens = est_p / est_w if est_w else 0
        opcoes = [est_w * m for m in MULT]
        ideal = min(range(len(opcoes)), key=lambda k: abs(opcoes[k] - real_w))
        res = {
            "foto_sozinha": ok(real_p, est_p),
            "seletor_ideal": ok(real_p, dens * opcoes[ideal]),
        }
        hits = 0
        for _ in range(N_SIM):
            k = ideal
            if rng.random() < P_ERRO:
                viz = [j for j in (ideal - 1, ideal + 1) if 0 <= j < len(opcoes)]
                k = rng.choice(viz)
            hits += ok(real_p, dens * opcoes[k])
        res["seletor_real"] = hits / N_SIM
        rec = "n5k" if r["origem"] == "n5k" else ("br_" + ("marmita" if r["categoria"] in ("congelado", "marmita") else "alimento_unico"))
        for grupo in ("todos", rec):
            for k, v in res.items():
                grupos[grupo][k].append(float(v))
    print(f"Tamanhos: E × {MULT}; erro de escolha: {int(P_ERRO*100)}% para tamanho vizinho\n")
    print(f"{'recorte':<20}{'n':>4}  {'foto sozinha':>13}  {'seletor ideal':>14}  {'seletor real':>13}")
    for grupo in sorted(grupos, key=lambda x: (x != "todos", x)):
        d = grupos[grupo]
        n = len(d["foto_sozinha"])
        f = lambda k: f"{100*sum(d[k])/n:.0f}%"
        print(f"{grupo:<20}{n:>4}  {f('foto_sozinha'):>13}  {f('seletor_ideal'):>14}  {f('seletor_real'):>13}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
