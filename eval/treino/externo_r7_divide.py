#!/usr/bin/env python3
"""Seção 17: divisões das três bases novas da sétima execução, fixadas antes de qualquer resultado.

    python3 externo_r7_divide.py [pasta_das_listas]

Lê `externo_acetada.csv`, `externo_receitas_datahive.csv` e `externo_foodbd.csv` (padrão: eval/treino/kaggle/listas/) e
grava, na mesma pasta, `divisao_r4_<base>.csv` no formato de externo_r4_divide.py (o kernel lê as duas da mesma forma):
  papel            fora (linha sem foto) | treino (split_ext = treino_ext) | registro (split_ext = teste_ext: a parte segurada de 17.2, fixada pelo
                   construtor da lista; nunca treina, o kernel apaga os rótulos e só grava as previsões)
  fold_ext         0–4 nas linhas de treino, por `grupo` (participante no ACETADA, nome da receita, bloco de 25 fotos no
                   FoodBD), com folds balanceados como na seção 13; −1 nas outras
  folds_proibidos  o próprio fold_ext
  grupo_metrica    a própria linha (a refeição, a receita, a foto): o peso da base se reparte igual entre as linhas de
                   treino, e a soma é a fração de 17.2 (0,28 / 0,15 / 0,15 do Nutrition5k), fixada no kernel
  cluster          o `grupo` (unidade do bootstrap)

Sobre o fold: 17.2 diz que as bases novas entram em todos os folds de treino ("fold −1"). Aqui cada linha de treino
fica fora do modelo de UM fold (o do grupo dela), como nas bases da seção 13. É o que permite a previsão fora-do-fold
nas partes de treino que 17.4 pede ("acerto acima de 25 g por base nas partes de treino do ... ACETADA"); o Nutrition5k
continua sem nenhuma linha delas na validação, e cada modelo de fold ainda treina com 4/5 de cada base.
Só lê ids, `grupo` e `split_ext`: nenhum rótulo, nenhum resultado de modelo. As listas não são alteradas."""
import collections, csv, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from externo_r4_divide import COLS, folds_balanceados  # noqa: E402

BASES = ["acetada", "receitas_datahive", "foodbd"]


def divide(R):
    """Linhas da divisão de uma base (função pura): ver o cabeçalho."""
    tr = [r for r in R if r["split_ext"] == "treino_ext" and (r.get("arquivo") or r.get("url_imagem") or r.get("url"))]
    assert all(r["split_ext"] in ("treino_ext", "teste_ext") for r in R), "split_ext desconhecido"
    fold = folds_balanceados(collections.Counter(r["grupo"] for r in tr))
    out = []
    for r in R:
        papel = "treino" if r["split_ext"] == "treino_ext" else "registro"
        if not (r.get("arquivo") or r.get("url_imagem") or r.get("url")):      # receita sem foto (dhv_00247): fica fora
            papel = "fora"
        k = fold[r["grupo"]] if papel == "treino" else -1
        out.append(dict(dish_id=r["dish_id"], dominio=r["dominio"], grupo=r["grupo"], papel=papel, fold_ext=k, folds_proibidos=str(k) if papel == "treino" else "",
                        quadro_metrica=1, grupo_metrica=r["dish_id"], cluster=r["grupo"]))
    return out


def main():
    pasta = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "kaggle", "listas")
    for b in BASES:
        arq = os.path.join(pasta, f"externo_{b}.csv")
        if not os.path.exists(arq):
            print(f"{b}: sem lista em {pasta}"); continue
        R = list(csv.DictReader(open(arq, newline="")))
        L = divide(R)
        with open(os.path.join(pasta, f"divisao_r4_{b}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, COLS, restval="", lineterminator="\n"); w.writeheader(); w.writerows(L)
        c = collections.Counter(l["papel"] for l in L)
        print("%-18s %6d linhas: %s | folds do treino %s" % (b, len(L), dict(c), [sum(1 for l in L if l["papel"] == "treino" and l["fold_ext"] == k) for k in range(5)]))


if __name__ == "__main__":
    main()
