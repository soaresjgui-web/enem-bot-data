#!/usr/bin/env python3
"""Seção 16.6: população do teste de registro DFood-TJ, fixada antes de qualquer modelo prever estas fotos.

    python3 dfoodtj_teste_divide.py <pasta_de_saida>

Lê `eval/treino/kaggle/listas/externo_dfoodtj.csv` (CC0) e grava:
  teste_r8_dfoodtj.csv          uma linha por foto do teste, SEM rótulo (para o kernel): dish_id, arquivo, periodo, paciente
  gabarito_r8_dfoodtj.csv       o mesmo com o rótulo corrigido (só para a leitura na CPU)
e imprime as contagens e os sha256 das duas tabelas.

Regras (16.6): foto `zenital` com proteína; vista conferida sem dúvida (sai `*_incerta`); sem `nome_difere_do_id`
(alvos vazios), `pesos_redondos` e `rotulo_difere_no_par`; `periodo` = 2020–2023 ou 2024–2025 (no segundo o visor da
balança aparece aceso e entra o mascaramento de 16.6). Ovo cozido (food_id 6) pesado com casca: a proteína dele é
multiplicada pela fração comestível 0,88 (casca = 12% do ovo inteiro, valor de tabela do USDA para ovo de galinha)."""
import csv, hashlib, os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
LISTA = os.path.join(AQUI, "kaggle", "listas", "externo_dfoodtj.csv")
OVO_ID, OVO_PROT, COMESTIVEL = "6", 0.131, 0.88
FAIXAS = [15, 25]


def corrige(r):
    ovo = sum(float(c.split(":", 2)[2]) for c in r["componentes"].split(";") if c.split(":", 1)[0] == OVO_ID)
    return float(r["proteina"]) - ovo * OVO_PROT * (1 - COMESTIVEL), ovo


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    R = [r for r in csv.DictReader(open(LISTA)) if r["vista"] == "zenital" and r["proteina"] != ""
         and "incerta" not in r["conferencia_vista"] and "pesos_redondos" not in r["avisos"] and "rotulo_difere_no_par" not in r["avisos"]]
    assert len(set(r["refeicao"] for r in R)) == len(R), "uma foto por refeição"
    T, G = [], []
    for r in sorted(R, key=lambda r: r["dish_id"]):
        p, ovo = corrige(r)
        per = "2024-2025" if r["data"][:4] in ("2024", "2025") else "2020-2023"
        T.append(dict(dish_id=r["dish_id"], arquivo=r["arquivo"], periodo=per, paciente=r["grupo"]))
        G.append(dict(T[-1], proteina_corrigida=round(p, 3), proteina_lista=r["proteina"], ovo_g=ovo, massa=r["massa"], kcal=r["kcal"], gordura=r["gordura"], carbo=r["carbo"]))
    for nome, L in (("teste_r8_dfoodtj.csv", T), ("gabarito_r8_dfoodtj.csv", G)):
        with open(os.path.join(out, nome), "w", newline="") as f:
            w = csv.DictWriter(f, list(L[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(L)
        print(nome, hashlib.sha256(open(os.path.join(out, nome), "rb").read()).hexdigest())
    P = np.array([g["proteina_corrigida"] for g in G]); per = np.array([g["periodo"] for g in G]); pac = np.array([g["paciente"] for g in G])
    for nome, m in (("todos", np.ones(len(G), bool)), ("2020-2023", per == "2020-2023"), ("2024-2025", per == "2024-2025")):
        fx = np.digitize(P[m], FAIXAS, right=True)
        print("%-10s %4d refeições, %2d pacientes | faixas <=15 / 15-25 / >25: %s | cinco faixas: %s | com ovo: %d" %
              (nome, m.sum(), len(set(pac[m])), np.bincount(fx, minlength=3), np.bincount(np.digitize(P[m], [5, 15, 25, 40], right=True), minlength=5), int(sum(g["ovo_g"] > 0 for g, x in zip(G, m) if x))))


if __name__ == "__main__":
    main()
