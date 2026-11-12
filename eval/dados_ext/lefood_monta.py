#!/usr/bin/env python3
"""Monta a lista de treino externo do LeFood-Set (mesmo esquema de externo_flavoria.csv; seção 12.4).

Dataset: Sari, Yuita Arum; Wani, Yudi Arimba; Nakazawa, Atsushi (2025). LeFood-Set: Leftovers Food Dataset.
Mendeley Data, V1. https://doi.org/10.17632/cchsk79jkt.1 — licença CC BY 4.0 (atribuição obrigatória).

O que é: comida de hospital da Indonésia, UM item por prato (arroz, frango, tempeh, tofu, ovo, peixe...),
fotografado de cima num prato raso branco sobre uma toalha quadriculada, ANTES e DEPOIS de comer. Cada foto tem
o peso da comida em balança digital. 749x548 px.

Uso:
  python3 -I lefood_monta.py mapa
  python3 -I lefood_monta.py monta <data_original.xlsx> [--saida <csv>]
O xlsx (59 KB) e as imagens (52 MB no total) vêm das URLs de lefood_arquivos.csv (coluna `url`, API pública do
Mendeley Data). As imagens não são lidas aqui.

UMA LINHA POR FOTO. A foto "antes" leva o peso antes; a "depois" leva o peso da sobra. As duas têm o mesmo
`grupo` (a mesma porção servida). Regras:
  L1  linha do xlsx cuja foto não existe em lefood_arquivos.csv: sai (as preparações de legumes e sopas do xlsx não
      têm foto publicada);
  L2  foto "depois" com sobra abaixo de 5 g: sai (prato vazio ou quase);
  L3  nome de comida sem linha em lefood_componentes.csv: erro.
A sobra maior que a porção inicial (erro de anotação) fica, com o aviso `sobra_maior_que_antes`.
Parte segurada: sem data de coleta, vale a regra de reserva de 12.5: sha1(grupo) mod 5 == 0 -> teste_ext.
"""
import csv, hashlib, os, sys
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import ALVOS, FONTES, GRUPOS, confere_esquema, densidade, tabelas  # noqa: E402

MAPA = os.path.join(AQUI, "lefood_componentes.csv")
ARQUIVOS = os.path.join(AQUI, "lefood_arquivos.csv")
SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_lefood.csv")
MIN_SOBRA_G = 5.0


def cmd_mapa():
    tab = tabelas()
    with open(MAPA, encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f)); campos = list(linhas[0].keys())
    for r in linhas:
        assert r["grupo"] in GRUPOS and r["fonte"] in FONTES and r["confianca"] in "ABC", r
        r["proteina_100g"] = f"{densidade(r['ref'], tab):.2f}"
    with open(MAPA, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, lineterminator="\n"); w.writeheader(); w.writerows(linhas)
    print(MAPA, len(linhas), "comidas |", dict(Counter(r["confianca"] for r in linhas)))


def ler_xlsx(caminho):
    import openpyxl
    ws = openpyxl.load_workbook(caminho, read_only=True, data_only=True).worksheets[0]
    linhas = list(ws.iter_rows(values_only=True))
    cab = [str(c).strip() for c in linhas[0]]
    assert cab[:7] == ["ID", "Name of the food", "Image Before Eaten", "Weight Before Eaten (g)", "Image After Eaten",
                       "Weight After Eaten (g)", "Visual Estimation by Observer (1-7)"], cab
    return [dict(id=int(r[0]), nome=str(r[1]).strip(), antes=str(r[2]).strip(), g_antes=float(r[3]),
                 depois=str(r[4]).strip(), g_depois=float(r[5]), nivel=r[6]) for r in linhas[1:] if r[0] is not None]


def cmd_monta(xlsx, saida):
    tab = tabelas()
    with open(MAPA, encoding="utf-8", newline="") as f:
        mapa = {r["componente"]: r for r in csv.DictReader(f)}
    for r in mapa.values():
        assert abs(float(r["proteina_100g"]) - densidade(r["ref"], tab)) < 0.006, r["componente"]
    with open(ARQUIVOS, encoding="utf-8", newline="") as f:
        arq = {r["arquivo"]: r for r in csv.DictReader(f)}
    base = confere_esquema()
    cab = base + ["dominio", "grupo", "vista", "split_ext", "arquivo", "url", "quadro", "componentes", "n_componentes",
                  "proteina_fraca_g", "proteina_fraca_conf", "proteina_fraca_fontes_g", "nivel_sobra", "avisos"]
    cont = Counter(); linhas = []
    for x in ler_xlsx(xlsx):
        cont["linhas_xlsx"] += 1
        if x["antes"] not in arq and x["depois"] not in arq:
            cont["L1_sem_foto"] += 1; continue
        c = mapa[x["nome"]]                                   # L3: KeyError se faltar
        grupo = f"lef_{x['id']:03d}"
        split = "teste_ext" if int(hashlib.sha1(grupo.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
        for quadro, nome, g in (("antes", x["antes"], x["g_antes"]), ("depois", x["depois"], x["g_depois"])):
            if nome not in arq:
                cont["L1_sem_foto_" + quadro] += 1; continue
            if quadro == "depois" and g < MIN_SOBRA_G:
                cont["L2_sobra_abaixo_de_5g"] += 1; continue
            pf = g * float(c["proteina_100g"]) / 100
            avisos = ["sobra_maior_que_antes"] if quadro == "depois" and g > x["g_antes"] else []
            linhas.append([f"lef_{x['id']:03d}_{quadro[0]}", -1, -1, "", f"{g:.1f}", "", "", "", ""]
                          + [f"{g if gr == c['grupo'] else 0:.1f}" for gr in GRUPOS] + [""] * len(GRUPOS)
                          + [f"{1 if fo == c['fonte'] else 0:.4f}" for fo in FONTES]
                          + ["lefood", grupo, "zenital", split, nome, arq[nome]["url"], quadro,
                             f"{x['nome']}:{g:.0f}", 1, f"{pf:.1f}", c["confianca"],
                             f"{pf if c['fonte'] != 'sem_proteina' else 0:.1f}", x["nivel"], "|".join(avisos)])
            cont["fotos_" + quadro] += 1
    assert len({l[0] for l in linhas}) == len(linhas)
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    i = {k: cab.index(k) for k in cab}
    m = sorted(float(l[i["massa"]]) for l in linhas); q = lambda p: m[int(p * (len(m) - 1))]
    print("contagens:", dict(sorted(cont.items())))
    print(f"fotos (linhas): {len(linhas)} | porções (grupos): {len({l[i['grupo']] for l in linhas})}"
          f" | split: {dict(Counter(l[i['split_ext']] for l in linhas))}")
    print(f"  massa (g): p10 {q(.1):.0f} mediana {q(.5):.0f} p90 {q(.9):.0f} máx {m[-1]:.0f}")
    fonte = Counter(); grupo = Counter()
    for l in linhas:
        fonte[[fo for fo in FONTES if l[i[f'ff_{fo}']] == "1.0000"][0]] += 1
    print("  fotos por fonte de proteína:", dict(fonte))
    pf = [float(l[i["proteina_fraca_fontes_g"]]) for l in linhas]
    print(f"  proteína FRACA das fontes: máx {max(pf):.1f} g | > 25 g: {sum(v > 25 for v in pf)} | > 10 g: {sum(v > 10 for v in pf)}")
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "mapa":
        cmd_mapa()
    elif len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
