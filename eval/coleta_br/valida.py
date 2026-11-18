#!/usr/bin/env python3
"""Confere as planilhas da coleta brasileira e calcula a proteína por prato.

Uso: python3 valida.py pratos.csv componentes.csv [--gabarito saida.csv] [--taco caminho/taco.csv]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
TACO_PADRAO = os.path.join(AQUI, "..", "..", "docs", "dados", "precisao", "ref", "taco.csv")
COLS_PRATOS = ["prato_id", "refeicao_id", "data", "categoria", "recipiente", "diametro_cm",
               "foto_principal", "fotos_extras", "aparelho", "tem_profundidade", "quem_coletou", "observacoes"]
COLS_COMP = ["prato_id", "componente", "alimento_taco", "preparo", "gramas", "proteina_100g_rotulo"]
FAIXAS = [(0, 5, "0-5 g"), (5, 15, "5-15 g"), (15, 25, "15-25 g"), (25, 40, "25-40 g"), (40, 1e9, ">40 g")]


def le(caminho, colunas):
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
    faltam = [c for c in colunas if not linhas or c not in linhas[0]]
    if faltam:
        sys.exit(f"{caminho}: faltam as colunas {faltam}")
    return linhas


def num(texto):
    return float(str(texto).strip().replace(",", "."))


def confere(pratos, comps, taco):
    """Função pura: devolve (erros, avisos, {prato_id: (massa, proteina)})."""
    erros, avisos, por_prato = [], [], {}
    ids = [p["prato_id"].strip() for p in pratos]
    repetidos = {i for i in ids if ids.count(i) > 1}
    if repetidos:
        erros.append(f"prato_id repetido: {sorted(repetidos)}")
    itens = defaultdict(list)
    for n, c in enumerate(comps, start=2):
        pid = c["prato_id"].strip()
        if pid not in ids:
            erros.append(f"componentes linha {n}: prato_id {pid!r} não está em pratos")
            continue
        try:
            g = num(c["gramas"])
        except ValueError:
            erros.append(f"componentes linha {n}: gramas inválido {c['gramas']!r}")
            continue
        if not 1 <= g <= 1500:
            avisos.append(f"componentes linha {n}: {g:g} g parece fora do comum")
        rotulo = c["proteina_100g_rotulo"].strip()
        nome = c["alimento_taco"].strip()
        if rotulo:
            try:
                dens = num(rotulo)
            except ValueError:
                erros.append(f"componentes linha {n}: proteina_100g_rotulo inválido {rotulo!r}")
                continue
            if nome in taco and taco[nome] > 0 and abs(dens - taco[nome]) / taco[nome] > 0.5:
                avisos.append(f"componentes linha {n}: rótulo {dens:g} g/100 g difere mais de 50% da TACO ({taco[nome]:g})")
        elif nome in taco:
            dens = taco[nome]
        else:
            erros.append(f"componentes linha {n}: {nome!r} não está na TACO e não há rótulo")
            continue
        if not 0 <= dens <= 90:
            erros.append(f"componentes linha {n}: proteína {dens:g} g/100 g fora da faixa possível")
            continue
        itens[pid].append((g, g * dens / 100))
    for p in pratos:
        pid = p["prato_id"].strip()
        if not itens[pid]:
            erros.append(f"prato {pid}: nenhum componente válido")
            continue
        if not p["refeicao_id"].strip():
            erros.append(f"prato {pid}: refeicao_id vazio")
        if not p["foto_principal"].strip():
            erros.append(f"prato {pid}: foto_principal vazia")
        por_prato[pid] = (sum(g for g, _ in itens[pid]), sum(pr for _, pr in itens[pid]))
    return erros, avisos, por_prato


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pratos")
    ap.add_argument("componentes")
    ap.add_argument("--gabarito")
    ap.add_argument("--taco", default=TACO_PADRAO)
    a = ap.parse_args()
    with open(a.taco, encoding="utf-8", newline="") as f:
        taco = {l["nome"]: float(l["proteina_100g"]) for l in csv.DictReader(f) if l["proteina_100g"] not in ("", "NA")}
    pratos, comps = le(a.pratos, COLS_PRATOS), le(a.componentes, COLS_COMP)
    erros, avisos, por_prato = confere(pratos, comps, taco)
    for m in avisos:
        print("aviso:", m)
    for m in erros:
        print("ERRO:", m)
    print(f"{len(pratos)} pratos, {len({p['refeicao_id'] for p in pratos})} refeições, {len(comps)} componentes; "
          f"{len(erros)} erro(s), {len(avisos)} aviso(s)")
    for lo, hi, nome in FAIXAS:
        n = sum(lo <= pr < hi for _, pr in por_prato.values())
        print(f"  proteína {nome:8s} {n:4d} pratos")
    if por_prato:
        ricos = sum(pr >= 25 for _, pr in por_prato.values()) / len(por_prato)
        print(f"  pratos com 25 g ou mais: {100 * ricos:.0f}% (alvo: pelo menos 40%)")
    if a.gabarito and not erros:
        cat = {p["prato_id"].strip(): p["categoria"].strip() for p in pratos}
        with open(a.gabarito, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prato_id", "origem", "split", "categoria", "peso_porcao_g", "proteina_g", "confianca", "lote_v2"])
            for pid, (m, pr) in por_prato.items():
                w.writerow([pid, "coleta_br", "", cat[pid], f"{m:.1f}", f"{pr:.2f}", "A", ""])
        print("gabarito gravado em", a.gabarito)
    sys.exit(1 if erros else 0)


if __name__ == "__main__":
    main()
