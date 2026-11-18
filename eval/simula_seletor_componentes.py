#!/usr/bin/env python3
"""v2: simula o seletor de porção POR COMPONENTE (6 tamanhos) sobre estimativas por componente.

Tela imaginada: lista de itens detectados; o item que mais pesa na proteína (carne/peixe/ovo/
feijão…) aparece em destaque com um seletor de 6 tamanhos ao redor da porção estimada E
(E × MULT). A usuária escolhe o tamanho mais parecido; os outros itens ficam como a foto estimou.
Proteína final = soma(gramas × proteína/100 g da tabela).

O "componente" ajustado é o GRUPO (grupos_alimentos.py) do item com mais proteína estimada;
se a foto separou o mesmo grupo em vários itens, todos são escalados juntos. O tamanho "certo" é
o mais próximo do total REAL desse grupo no gabarito por componente (gabarito_componentes.csv);
se o grupo não existe no gabarito (a foto inventou o item), o certo é o menor tamanho.

Cenários (mesma regra de acerto do v1: ±20% ou ±5 g):
  foto_sozinha    — sem ajuste
  seletor_prato   — seletor do v1: escala o prato inteiro (peso total real)
  comp_ideal      — seletor no componente proteico, sempre o tamanho certo
  comp_real       — idem, erra para o vizinho com prob. P_ERRO (média de N_SIM sorteios)
  comp2_real      — seletor nos 2 componentes de mais proteína (se forem grupos diferentes)

Uso: python3 simula_seletor_componentes.py componentes_estimados.csv \
        [--gab-comp ...gabarito_componentes.csv] [--pratos ...gabarito_pratos_v2.csv] [--p-erro 0.3]
Só entram pratos com gabarito por componente (UFPR, Nutrition5k); Liv Up/Seara ficam de fora.
"""
import argparse
import csv
import os
import random
import sys
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.join(AQUI, "..", "docs", "dados", "precisao")
MULT = [0.5, 0.75, 1.0, 1.35, 1.8, 2.5]
N_SIM = 200


def ok(real, est):
    return abs(est - real) <= 5 or (real > 0 and abs(est - real) / real <= 0.20)


def ler(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def escolhe(opcoes, alvo, rng=None, p_erro=0.0):
    ideal = min(range(len(opcoes)), key=lambda k: abs(opcoes[k] - alvo))
    if rng is not None and rng.random() < p_erro:
        viz = [j for j in (ideal - 1, ideal + 1) if 0 <= j < len(opcoes)]
        return opcoes[rng.choice(viz)]
    return opcoes[ideal]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("componentes")
    ap.add_argument("--gab-comp", default=os.path.join(DADOS, "gabarito_componentes.csv"))
    ap.add_argument("--pratos", default=os.path.join(DADOS, "gabarito_pratos_v2.csv"))
    ap.add_argument("--p-erro", type=float, default=0.30)
    a = ap.parse_args()

    est = defaultdict(list)
    for r in ler(a.componentes):
        est[r["prato_id"]].append({"grupo": r["grupo"], "g": float(r["gramas_estimadas"]),
                                   "p100": float(r["proteina_100g"])})
    real_g = defaultdict(lambda: defaultdict(float))
    for r in ler(a.gab_comp):
        real_g[r["prato_id"]][r["grupo"]] += float(r["gramas"])
    pratos = {r["prato_id"]: r for r in ler(a.pratos)}
    rng = random.Random(42)
    res = defaultdict(lambda: defaultdict(list))
    sem_gab = 0

    for pid, comps in est.items():
        if pid not in pratos:
            continue
        if pid not in real_g:
            sem_gab += 1
            continue
        real_p = float(pratos[pid]["proteina_g"])
        real_w = float(pratos[pid]["peso_porcao_g"])
        prot_g = defaultdict(float)
        gram_g = defaultdict(float)
        for c in comps:
            prot_g[c["grupo"]] += c["g"] * c["p100"] / 100
            gram_g[c["grupo"]] += c["g"]
        total_p = sum(prot_g.values())
        total_w = sum(gram_g.values())
        ordem = sorted(prot_g, key=prot_g.get, reverse=True)

        def com_ajuste(grupos_escolha):
            p = total_p
            for g, novo in grupos_escolha.items():
                if gram_g[g] > 0:
                    p += prot_g[g] * (novo / gram_g[g] - 1)
            return p

        def alvo(g):
            return real_g[pid].get(g, 0.0)

        r = {"foto_sozinha": ok(real_p, total_p)}
        dens = total_p / total_w if total_w else 0
        r["seletor_prato"] = ok(real_p, dens * escolhe([total_w * m for m in MULT], real_w))
        g1 = ordem[0] if ordem else None
        if g1 is None:
            continue
        op1 = [gram_g[g1] * m for m in MULT]
        r["comp_ideal"] = ok(real_p, com_ajuste({g1: escolhe(op1, alvo(g1))}))
        hits = hits2 = 0
        g2 = ordem[1] if len(ordem) > 1 and prot_g[ordem[1]] >= 2 else None
        for _ in range(N_SIM):
            e1 = escolhe(op1, alvo(g1), rng, a.p_erro)
            hits += ok(real_p, com_ajuste({g1: e1}))
            esc = {g1: e1}
            if g2:
                esc[g2] = escolhe([gram_g[g2] * m for m in MULT], alvo(g2), rng, a.p_erro)
            hits2 += ok(real_p, com_ajuste(esc))
        r["comp_real"] = hits / N_SIM
        r["comp2_real"] = hits2 / N_SIM
        origem = pratos[pid]["origem"]
        split = pratos[pid]["split"]
        for grupo in ("todos", f"{origem} ({split})", f"grupo ajustado: {g1}"):
            for k, v in r.items():
                res[grupo][k].append(float(v))

    cen = ["foto_sozinha", "seletor_prato", "comp_ideal", "comp_real", "comp2_real"]
    print(f"Tamanhos: E × {MULT}; erro de escolha {int(a.p_erro * 100)}% para o vizinho. "
          f"{sem_gab} prato(s) sem gabarito por componente ignorados.\n")
    print(f"{'recorte':<36}{'n':>4}" + "".join(f"{c:>15}" for c in cen))
    for g in sorted(res, key=lambda x: (x != "todos", x.startswith("grupo"), x)):
        d = res[g]
        n = len(d["foto_sozinha"])
        print(f"{g:<36}{n:>4}" + "".join(f"{100 * sum(d[c]) / n:>14.0f}%" for c in cen))


if __name__ == "__main__":
    main()
