#!/usr/bin/env python3
"""Seção 19.2 e: divisão do FLIC por curso para a décima primeira execução, fixada antes de qualquer resultado.

    python3 externo_r11_divide.py [pasta_das_listas]

Lê `externo_flic_curso.csv` e `caixas_r5_flic_curso.csv` (padrão: eval/treino/kaggle/listas/) e grava, na mesma pasta,
`divisao_r4_flic_curso.csv` no formato de externo_r4_divide.py:
  papel            fora (sem janela pela regra de 19.2 c: `sem_prato` na tabela) | treino (split_ext = treino_ext) |
                   registro (split_ext = teste_ext: os 2 dias de teste da lista por bandeja; nunca treina, o kernel
                   apaga os rótulos e só grava as previsões)
  fold_ext         0–4 nas linhas de treino, por DIA (folds_balanceados sobre os dias): a validação cruzada do FLIC
                   sempre prevê um cardápio nunca visto; −1 nas outras
  folds_proibidos  o próprio fold_ext
  grupo_metrica    o prato (dia, bandeja, curso): a foto antes e a foto depois do mesmo prato dividem o peso
  cluster          o dia (unidade do bootstrap da validação cruzada)
Só lê ids, dia, prato, `split_ext` e o método da tabela de janelas: nenhuma massa, proteína ou previsão."""
import collections, csv, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from externo_r4_divide import COLS, folds_balanceados  # noqa: E402

CHAVE = "flic_curso"


def divide(R, metodo):
    """Linhas da divisão (função pura). R: linhas da lista; metodo: {id da foto no kernel: método da tabela de janelas}."""
    assert all(r["split_ext"] in ("treino_ext", "teste_ext") for r in R), "split_ext desconhecido"
    sem = lambda r: metodo.get(f"ext:{CHAVE}:{r['dish_id']}", "sem_prato") == "sem_prato"
    tr = [r for r in R if r["split_ext"] == "treino_ext" and not sem(r)]
    fold = folds_balanceados(collections.Counter(r["grupo"] for r in tr))
    out = []
    for r in R:
        papel = "fora" if sem(r) else "treino" if r["split_ext"] == "treino_ext" else "registro"
        k = fold[r["grupo"]] if papel == "treino" else -1
        out.append(dict(dish_id=r["dish_id"], dominio=r["dominio"], grupo=r["grupo"], papel=papel, fold_ext=k, folds_proibidos=str(k) if papel == "treino" else "",
                        quadro_metrica=1, grupo_metrica=r["par"], cluster=r["grupo"]))
    return out


def main():
    pasta = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "kaggle", "listas")
    R = list(csv.DictReader(open(os.path.join(pasta, f"externo_{CHAVE}.csv"), newline="")))
    metodo = {r["id"]: r["metodo"] for r in csv.DictReader(open(os.path.join(pasta, f"caixas_r5_{CHAVE}.csv"), newline=""))}
    L = divide(R, metodo)
    with open(os.path.join(pasta, f"divisao_r4_{CHAVE}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLS, restval="", lineterminator="\n"); w.writeheader(); w.writerows(L)
    c = collections.Counter(l["papel"] for l in L)
    tr = [l for l in L if l["papel"] == "treino"]
    print("%s: %d linhas: %s | folds do treino (linhas) %s | dias por fold %s | dias de teste %s" % (
        CHAVE, len(L), dict(c), [sum(1 for l in tr if l["fold_ext"] == k) for k in range(5)], [len({l["grupo"] for l in tr if l["fold_ext"] == k}) for k in range(5)],
        sorted({l["grupo"] for l in L if l["papel"] == "registro"})))


if __name__ == "__main__":
    main()
