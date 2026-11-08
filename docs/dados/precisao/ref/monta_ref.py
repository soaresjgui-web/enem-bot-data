#!/usr/bin/env python3
"""Monta as tabelas de referência brasileiras usadas no pipeline v2 (proteína e porções).

Uso: python3 monta_ref.py <dir_downloads>
  <dir_downloads> deve conter:
    taco_food.csv, taco_nutrients.csv, taco_categories.csv
        <- https://raw.githubusercontent.com/raulfdm/taco-api/main/references/csv/{food,nutrients,categories}.csv
           (TACO 4ª ed., NEPA/UNICAMP 2011, convertida pelo projeto taco-api, licença MIT do código;
            a TACO é de livre uso com citação: "NEPA-UNICAMP. Tabela brasileira de composição
            de alimentos. 4. ed. Campinas: NEPA-UNICAMP, 2011.")
    ibge/tabelamedidas_bd.xls
        <- https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2008_2009/
           Tabela_de_Medidas_Referidas_para_os_Alimentos_Consumidos_no_Brasil/tabelamedidas_bd.zip
    ibge/tabelacompleta.xls
        <- .../Tabelas_de_Composicao_Nutricional_dos_Alimentos_Consumidos_no_Brasil/tabelacompleta.zip
           (IBGE, POF 2008-2009; dados públicos do IBGE, reprodução permitida com citação da fonte)
  Requer pandas + xlrd para ler os .xls.

Grava, na pasta deste script:
  taco.csv            id,nome,grupo,proteina_100g,kcal_100g,lipidios_100g,carboidrato_100g,fibra_100g,umidade_100g
  medidas_ibge.csv    cod_alimento,alimento,cod_preparacao,preparacao,medida,gramas,fonte_ref,descricao_ref
  composicao_ibge.csv cod_alimento,alimento,cod_preparacao,preparacao,descricao_ref,proteina_100g,kcal_100g
E confere 10 valores da TACO 4ª ed. (proteína g/100 g) — sai com erro se algum divergir > 0,15 g.
"""
import csv
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))

# (trecho do nome na TACO, proteína esperada g/100 g — TACO 4ª ed. impressa)
CONFERENCIA = [
    ("Arroz, tipo 1, cozido", 2.5),
    ("Feijão, carioca, cozido", 4.8),
    ("Frango, peito, sem pele, grelhado", 32.0),
    ("Ovo, de galinha, inteiro, cozido/10minutos", 13.3),
    ("Pão, trigo, francês", 8.0),
    ("Carne, bovina, patinho, sem gordura, grelhado", 35.9),
    ("Feijão, preto, cozido", 4.5),
    ("Queijo, minas, frescal", 17.4),
    ("Batata, inglesa, cozida", 1.2),
    ("Banana, prata, crua", 1.3),
]


def num(x):
    try:
        v = float(str(x).replace(",", "."))
    except ValueError:
        return ""
    return round(v, 2)


def monta_taco(dl):
    cats = {r["id"]: r["name"] for r in csv.DictReader(open(os.path.join(dl, "taco_categories.csv"), encoding="utf-8"))}
    nut = {r["foodId"]: r for r in csv.DictReader(open(os.path.join(dl, "taco_nutrients.csv"), encoding="utf-8"))}
    linhas = []
    for r in csv.DictReader(open(os.path.join(dl, "taco_food.csv"), encoding="utf-8")):
        n = nut.get(r["id"], {})
        linhas.append({"id": int(r["id"]), "nome": r["name"].strip(), "grupo": cats.get(r["categoryId"], ""),
                       "proteina_100g": num(n.get("protein", "")), "kcal_100g": num(n.get("kcal", "")),
                       "lipidios_100g": num(n.get("lipids", "")), "carboidrato_100g": num(n.get("carbohydrates", "")),
                       "fibra_100g": num(n.get("dietaryFiber", "")), "umidade_100g": num(n.get("moisture", ""))})
    with open(os.path.join(AQUI, "taco.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    return linhas


def confere(taco):
    ok = True
    print("Conferência TACO 4ª ed. (proteína g/100 g):")
    for trecho, esperado in CONFERENCIA:
        achados = [t for t in taco if t["nome"].lower().startswith(trecho.lower())]
        if not achados:
            print(f"  ?? não achado: {trecho}")
            ok = False
            continue
        t = achados[0]
        dif = abs(float(t["proteina_100g"]) - esperado)
        st = "ok" if dif <= 0.15 else "DIVERGE"
        ok &= dif <= 0.15
        print(f"  {st:7} #{t['id']:<3} {t['nome'][:55]:<55} {t['proteina_100g']:>6} (esperado {esperado})")
    return ok


def monta_ibge(dl):
    import pandas as pd
    m = pd.read_excel(os.path.join(dl, "ibge", "tabelamedidas_bd.xls"), sheet_name=0, header=None, skiprows=5)
    m = m[pd.to_numeric(m[0], errors="coerce").notna()]
    with open(os.path.join(AQUI, "medidas_ibge.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cod_alimento", "alimento", "cod_preparacao", "preparacao", "medida", "gramas",
                    "fonte_ref", "descricao_ref"])
        for r in m.itertuples(index=False):
            w.writerow([int(r[0]), str(r[1]).strip(), int(r[2]), str(r[3]).strip(), str(r[5]).strip(),
                        num(r[8]), int(r[9]) if str(r[9]) != "nan" else "", str(r[10]).strip()])
    c = pd.read_excel(os.path.join(dl, "ibge", "tabelacompleta.xls"), sheet_name=0, header=None, skiprows=4)
    c = c[pd.to_numeric(c[0], errors="coerce").notna()]
    with open(os.path.join(AQUI, "composicao_ibge.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cod_alimento", "alimento", "cod_preparacao", "preparacao", "descricao_ref",
                    "proteina_100g", "kcal_100g"])
        for r in c.itertuples(index=False):
            w.writerow([int(r[0]), str(r[1]).strip(), int(r[2]), str(r[3]).strip(), str(r[5]).strip(),
                        num(r[7]), num(r[6])])
    return len(m), len(c)


def main(dl):
    taco = monta_taco(dl)
    print(f"taco.csv: {len(taco)} alimentos")
    ok = confere(taco)
    nm, nc = monta_ibge(dl)
    print(f"medidas_ibge.csv: {nm} linhas; composicao_ibge.csv: {nc} linhas")
    if not ok:
        sys.exit("conferência da TACO falhou")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
