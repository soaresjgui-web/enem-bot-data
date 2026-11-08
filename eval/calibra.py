#!/usr/bin/env python3
"""v2: ajusta correções de porção SÓ no split "calib" e mede o efeito no split "teste".

Uso:
  python3 calibra.py componentes_estimados.csv \
      [--gab-comp docs/dados/precisao/gabarito_componentes.csv] \
      [--pratos docs/dados/precisao/gabarito_pratos_v2.csv] \
      [--medidas docs/dados/precisao/ref/medidas_ibge.csv] \
      [--saida estimativas_corrigidas.csv] [--parametros parametros.json]

componentes_estimados.csv = saída --componentes do protein_from_components.py
(prato_id,sistema,grupo,gramas_estimadas,proteina_100g,...).

Modelos (todos corrigem as GRAMAS de cada componente; a proteína/100 g da tabela não muda):
  sem_correcao      — baseline.
  mult_grupo        — multiplicador por (grupo, tipo de prato): exp(mediana de log(real/estimado))
                      nos pares prato×grupo do calib, encolhido para 1 com peso n/(n+5).
                      tipo = "unico" (1 componente relevante) ou "composto"; com < 8 pares cai
                      para o multiplicador do grupo e depois para o global. Ataca a superestimação
                      de carne em prato composto (v1: +16% de densidade).
  loglinear_grupo   — log(real) = a + b·log(estimado) por grupo (b > 1 corrige a subestimação de
                      porções grandes vista no v1). Com < 12 pares usa o ajuste global.
  prior_ibge        — log(g') = w·log(g) + (1−w)·log(porção típica do grupo), porção típica = mediana
                      das "PORCAO" da Tabela de Medidas Referidas (IBGE POF 2008-09) dos alimentos
                      do grupo; w por grupo escolhido em grade 0…1,5 no calib (w > 1 = afasta da
                      porção típica, isto é, corrige compressão para o meio).
Saída: métricas (as mesmas do score_precisao.py) por modelo no split "teste" e, separadamente,
nos itens do v1 (também fora do ajuste); --saida grava as estimativas corrigidas do melhor modelo
(no calib) para todos os pratos, no formato do score_precisao.py.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from statistics import median

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from grupos_alimentos import grupo_de  # noqa: E402
from score_precisao import fmt, metrics  # noqa: E402

DADOS = os.path.join(AQUI, "..", "docs", "dados", "precisao")
MIN_G = 10.0  # pares prato×grupo com menos que isso (real ou estimado) não entram no ajuste
GRADE_W = [round(0.1 * k, 1) for k in range(0, 16)]


def ler(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tipo_prato(comps):
    relevantes = [c for c in comps if c["g"] >= 0.15 * sum(x["g"] for x in comps)]
    return "unico" if len(relevantes) <= 1 else "composto"


def carrega(path_est, path_gc, path_pratos):
    est = defaultdict(list)
    sistemas = set()
    for r in ler(path_est):
        sistemas.add(r["sistema"])
        est[r["prato_id"]].append({"grupo": r["grupo"], "g": float(r["gramas_estimadas"]),
                                   "p100": float(r["proteina_100g"])})
    if len(sistemas) > 1:
        sys.exit(f"um sistema por vez, achei {sorted(sistemas)}")
    real_grupo = defaultdict(lambda: defaultdict(float))
    for r in ler(path_gc):
        real_grupo[r["prato_id"]][r["grupo"]] += float(r["gramas"])
    pratos = {r["prato_id"]: r for r in ler(path_pratos)}
    return est, real_grupo, pratos, (sistemas.pop() if sistemas else "v2")


def pares(est, real_grupo, ids):
    """[(grupo, tipo, g_est, g_real)] por prato×grupo, só pratos com gabarito por componente."""
    out = []
    for pid in ids:
        if pid not in real_grupo or pid not in est:
            continue
        tipo = tipo_prato(est[pid])
        eg = defaultdict(float)
        for c in est[pid]:
            eg[c["grupo"]] += c["g"]
        for g, ge in eg.items():
            gr = real_grupo[pid].get(g, 0.0)
            if ge >= MIN_G and gr >= MIN_G:
                out.append((g, tipo, ge, gr))
    return out


def ajusta_mult(ps):
    def m(sel):
        if not sel:
            return 1.0, 0
        med = median(math.log(gr / ge) for *_, ge, gr in sel)
        n = len(sel)
        return math.exp(med * n / (n + 5)), n
    glob, _ = m(ps)
    tab = {"_global": glob}
    for g in {p[0] for p in ps}:
        mg, ng = m([p for p in ps if p[0] == g])
        tab[g] = mg if ng >= 8 else glob
        for t in ("unico", "composto"):
            mt, nt = m([p for p in ps if p[0] == g and p[1] == t])
            if nt >= 8:
                tab[f"{g}|{t}"] = mt
    return tab


def aplica_mult(tab, comps):
    t = tipo_prato(comps)
    return [c["g"] * tab.get(f"{c['grupo']}|{t}", tab.get(c["grupo"], tab["_global"])) for c in comps]


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return my - mx, 1.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    b = min(max(b, 0.5), 1.6)  # limita extrapolação
    return my - b * mx, b


def ajusta_loglin(ps):
    tab = {"_global": ols([math.log(p[2]) for p in ps], [math.log(p[3]) for p in ps]) if ps else (0.0, 1.0)}
    for g in {p[0] for p in ps}:
        sel = [p for p in ps if p[0] == g]
        if len(sel) >= 12:
            tab[g] = ols([math.log(p[2]) for p in sel], [math.log(p[3]) for p in sel])
    return tab


def aplica_loglin(tab, comps):
    out = []
    for c in comps:
        a, b = tab.get(c["grupo"], tab["_global"])
        out.append(math.exp(a + b * math.log(c["g"])) if c["g"] > 0 else 0.0)
    return out


def priors_ibge(path):
    por = defaultdict(list)
    if path and os.path.exists(path):
        for r in ler(path):
            if r["medida"] == "PORCAO" and r["gramas"]:
                por[grupo_de(r["alimento"])].append(float(r["gramas"]))
    tab = {g: median(v) for g, v in por.items() if len(v) >= 5}
    todos = [x for v in por.values() for x in v]
    tab["_global"] = median(todos) if todos else 100.0
    return tab


def ajusta_prior(ps, pri):
    tab = {}
    for g in {p[0] for p in ps} | {"_global"}:
        sel = ps if g == "_global" else [p for p in ps if p[0] == g]
        if g != "_global" and len(sel) < 8:
            continue
        best = None
        for w in GRADE_W:
            err = 0.0
            for gg, _, ge, gr in sel:
                lp = math.log(pri.get(gg, pri["_global"]))
                err += (w * math.log(ge) + (1 - w) * lp - math.log(gr)) ** 2
            if best is None or err < best[0]:
                best = (err, w)
        tab[g] = best[1]
    return tab


def aplica_prior(tab, pri, comps):
    out = []
    for c in comps:
        w = tab.get(c["grupo"], tab["_global"])
        lp = math.log(pri.get(c["grupo"], pri["_global"]))
        out.append(math.exp(w * math.log(c["g"]) + (1 - w) * lp) if c["g"] > 0 else 0.0)
    return out


def proteina(comps, gramas):
    return sum(g * c["p100"] / 100 for c, g in zip(comps, gramas))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("componentes")
    ap.add_argument("--gab-comp", default=os.path.join(DADOS, "gabarito_componentes.csv"))
    ap.add_argument("--pratos", default=os.path.join(DADOS, "gabarito_pratos_v2.csv"))
    ap.add_argument("--medidas", default=os.path.join(DADOS, "ref", "medidas_ibge.csv"))
    ap.add_argument("--saida", default="")
    ap.add_argument("--parametros", default="")
    a = ap.parse_args()

    est, real_grupo, pratos, sistema = carrega(a.componentes, a.gab_comp, a.pratos)
    calib = [p for p, r in pratos.items() if r["split"] == "calib" and p in est]
    ps = pares(est, real_grupo, calib)
    if len(ps) < 10:
        sys.exit(f"poucos pares prato×grupo no calib ({len(ps)}); faltam estimativas do split calib?")
    pri = priors_ibge(a.medidas)
    m_tab, l_tab = ajusta_mult(ps), ajusta_loglin(ps)
    w_tab = ajusta_prior(ps, pri)
    modelos = {
        "sem_correcao": lambda c: [x["g"] for x in c],
        "mult_grupo": lambda c: aplica_mult(m_tab, c),
        "loglinear_grupo": lambda c: aplica_loglin(l_tab, c),
        "prior_ibge": lambda c: aplica_prior(w_tab, pri, c),
    }
    params = {"n_pratos_calib": len(calib), "n_pares": len(ps), "mult_grupo": m_tab,
              "loglinear_grupo": {k: {"a": v[0], "b": v[1]} for k, v in l_tab.items()},
              "prior_ibge": {"w": w_tab, "porcao_tipica_g": pri}}
    print(f"Sistema: {sistema}. Ajuste no calib: {len(calib)} pratos, {len(ps)} pares prato×grupo.")
    print("mult_grupo:", ", ".join(f"{k}={v:.2f}" for k, v in sorted(m_tab.items())))
    print("loglinear_grupo (a, b):", ", ".join(f"{k}=({v[0]:.2f}, {v[1]:.2f})" for k, v in sorted(l_tab.items())))
    print("prior_ibge w:", ", ".join(f"{k}={v}" for k, v in sorted(w_tab.items())),
          "| porção típica (g):", ", ".join(f"{k}={v:.0f}" for k, v in sorted(pri.items())))

    def avalia(ids, nome_modelo):
        f = modelos[nome_modelo]
        return [(float(pratos[p]["proteina_g"]), proteina(est[p], f(est[p]))) for p in ids if p in est]

    # escolhe o melhor modelo pelo calib (erro absoluto médio), sem olhar o teste
    erro_calib = {m: metrics(avalia(calib, m))["erro_medio_g"] for m in modelos}
    melhor = min(erro_calib, key=erro_calib.get)
    print("\nErro médio no calib (só para escolher o modelo):",
          ", ".join(f"{m} {e:.1f} g" for m, e in erro_calib.items()), f"-> {melhor}")

    recortes = [("teste (Nutrition5k v2)", [p for p, r in pratos.items() if r["split"] == "teste"])]
    v1 = [p for p, r in pratos.items() if r["split"] == "v1"]
    for origem in sorted({pratos[p]["origem"] for p in v1}):
        recortes.append((f"v1 — {origem}", [p for p in v1 if pratos[p]["origem"] == origem]))
    for nome, ids in recortes:
        ids = [p for p in ids if p in est]
        if not ids:
            continue
        print(f"\n== {nome} (fora do ajuste) ==")
        for m in modelos:
            print(f"  {m:<16}", fmt(metrics(avalia(ids, m))))
    params["melhor_no_calib"] = melhor
    if a.parametros:
        with open(a.parametros, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
    if a.saida:
        with open(a.saida, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prato_id", "sistema", "proteina_estimada_g", "peso_total_estimado_g"])
            for p in sorted(est):
                g = modelos[melhor](est[p])
                w.writerow([p, f"{sistema}+{melhor}", round(proteina(est[p], g), 1), round(sum(g))])


if __name__ == "__main__":
    main()
