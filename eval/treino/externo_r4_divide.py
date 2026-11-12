#!/usr/bin/env python3
"""Seção 13: divisões das bases externas para a quarta execução em GPU, fixadas antes de qualquer resultado.

    python3 externo_r4_divide.py <pasta_de_saida> [pasta_das_listas]

Lê as listas `externo_*.csv` (padrão: eval/treino/kaggle/listas/) e grava `divisao_r4_<dominio>.csv`, uma linha por linha
da lista, com:
  papel         treino | registro | registro_misto | fora | avaliacao_fraca
                (registro = parte segurada, nunca treina nem entra em escolha; só é pontuada na leitura final)
  fold_ext      0–4 para as linhas de treino: o fold em que o grupo é validado na validação cruzada conjunta; −1 nas outras
  folds_proibidos  folds de treino em que a linha NÃO pode entrar (sempre inclui fold_ext; no OmniFood8K, também os folds
                de todos os pratos da cena, para que o prato principal validado nunca tenha sido visto)
  quadro_metrica  1 se a linha (ou o quadro nomeado em quadro_metrica_nome) é a que conta na métrica por grupo
  grupo_metrica  a unidade que conta uma vez na métrica (cena, prato, bandeja ou foto)
  cluster       a unidade do bootstrap (prato principal, dia, grupo de porção, participante)
Só usa rótulos das bases externas e nenhum resultado de modelo. As listas não são alteradas.

Regras (13.3):
  OmniFood8K  densidade de proteína de cada prato por mínimos quadrados nas cenas (proteína ≈ Σ gramas × densidade);
              prato principal da cena = o que mais contribui com proteína. Pratos de registro: sha1(nome) mod 5 == 3.
              registro = cenas cujo principal é de registro; registro_misto = cenas com prato de registro que não é o
              principal; treino = as que não têm nenhum prato de registro. Folds do treino: por prato principal.
  Flavoria    registro = split_ext teste_ext (os 7 últimos dias); folds do treino por dia de coleta.
  FPB         só as fotos de cima com massa; registro = split_ext teste_ext; folds por `grupo` (porções repetidas).
  Mensa       registro = split_ext teste_ext (um participante); folds = um participante cada.
  LeFood      registro = split_ext teste_ext; folds por `grupo` (porção).
  SNAPMe      avaliacao_fraca em todas as linhas: nunca treina (pedido dos autores; rótulo de autorrelato)."""
import collections, csv, hashlib, os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
COLS = ["dish_id", "dominio", "grupo", "papel", "fold_ext", "folds_proibidos", "quadro_metrica", "quadro_metrica_nome", "quadros_treino",
        "grupo_metrica", "cluster", "peso_nutricao", "prato_principal"]
QUADROS_TREINO = ["camera_0.jpg", "camera_2.jpg", "camera_4.jpg"]      # 3 dos 6 quadros do OmniFood8K (todos de cima), por custo
QUADRO_METRICA = "camera_0.jpg"


def sha(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def folds_balanceados(tamanhos):
    """Grupos -> fold 0–4: do maior para o menor (empate pelo nome), cada um no fold mais vazio (empate: o de menor número)."""
    carga, f = [0] * 5, {}
    for g in sorted(tamanhos, key=lambda g: (-tamanhos[g], g)):
        k = min(range(5), key=lambda j: (carga[j], j))
        f[g] = k; carga[k] += tamanhos[g]
    return f


def grava(pasta, dominio, linhas):
    with open(os.path.join(pasta, "divisao_r4_%s.csv" % dominio), "w", newline="") as f:
        w = csv.DictWriter(f, COLS, restval="", lineterminator="\n"); w.writeheader(); w.writerows(linhas)
    c = collections.Counter(l["papel"] for l in linhas)
    print("%-10s %5d linhas: %s | folds do treino %s" % (dominio, len(linhas), dict(c), [sum(1 for l in linhas if l["papel"] == "treino" and l["fold_ext"] == k) for k in range(5)]))


def omnifood(R):
    comp = [[(c.rsplit(":", 1)[0], float(c.rsplit(":", 1)[1])) for c in r["componentes"].split(";")] for r in R]
    nomes = sorted({n for c in comp for n, _ in c}); ix = {n: i for i, n in enumerate(nomes)}
    A = np.zeros((len(R), len(nomes)))
    for i, c in enumerate(comp):
        for n, g in c:
            A[i, ix[n]] += g
    tem = np.array([r["proteina"] != "" for r in R])
    p = np.array([float(r["proteina"]) if r["proteina"] else np.nan for r in R])
    dens = np.linalg.lstsq(A[tem], p[tem], rcond=None)[0].clip(0)
    contrib = A * dens
    principal = np.array([nomes[j] for j in contrib.argmax(1)])
    registro = {n for n in nomes if sha(n) % 5 == 3}
    tem_reg = np.array([any(n in registro for n, _ in c) for c in comp])
    papel = np.where(np.isin(principal, sorted(registro)), "registro", np.where(tem_reg, "registro_misto", "treino"))
    tr = papel == "treino"
    fold_prato = folds_balanceados(collections.Counter(principal[tr]))
    for n in nomes:             # prato do treino que nunca é principal: fold pelo nome (só conta para folds_proibidos)
        fold_prato.setdefault(n, sha(n) % 5)
    L = []
    for i, r in enumerate(R):
        l = dict(dish_id=r["dish_id"], dominio=r["dominio"], grupo=r["grupo"], papel=papel[i], fold_ext=-1, quadro_metrica=1, quadro_metrica_nome=QUADRO_METRICA,
                 quadros_treino=";".join(QUADROS_TREINO) if tr[i] else "", grupo_metrica=r["grupo"], cluster=principal[i], prato_principal=principal[i],
                 peso_nutricao=0 if r["proteina"] == "" else 0.5 if "osso_ou_casca" in r["avisos"] else 1)
        if tr[i]:
            l["fold_ext"] = fold_prato[principal[i]]
            l["folds_proibidos"] = ";".join(str(k) for k in sorted({fold_prato[n] for n, _ in comp[i]}))
        L.append(l)
    f = lambda m: "%d cenas (>25 g: %d, sem osso/casca %d; 15–25: %d; 5–15: %d; 0–5: %d; sem proteína %d)" % (
        m.sum(), (m & (p > 25)).sum(), (m & (p > 25) & np.array(["osso_ou_casca" not in r["avisos"] for r in R])).sum(), (m & (p > 15) & (p <= 25)).sum(),
        (m & (p > 5) & (p <= 15)).sum(), (m & (p <= 5)).sum(), (m & ~tem).sum())
    print("OmniFood8K: %d pratos, %d de registro; fração da proteína que vem do prato principal: mediana %.2f" % (len(nomes), len(registro), np.median(contrib.max(1) / np.maximum(contrib.sum(1), 1e-9))))
    print("   registro: %s, %d pratos principais" % (f(papel == "registro"), len(set(principal[papel == "registro"]))))
    print("   treino:   %s, %d pratos principais" % (f(tr), len(set(principal[tr]))))
    print("   registro_misto: %d cenas" % (papel == "registro_misto").sum())
    nf = np.array([len(l["folds_proibidos"].split(";")) for l in L if l["papel"] == "treino"])
    print("   treino por fold (cenas que podem treinar quando o fold k valida): %s" % [int(sum(1 for l in L if l["papel"] == "treino" and str(k) not in l["folds_proibidos"].split(";"))) for k in range(5)])
    for k in range(5):
        m = tr & np.array([l["fold_ext"] == k for l in L])
        print("   fold %d valida %s" % (k, f(m)))
    return L


def por_grupo(R, usa, chave_fold, grupo_metrica, quadro_metrica, cluster):
    """Listas de uma linha por foto. usa(r): a linha entra; chave_fold(r): unidade dos folds; quadro_metrica(r): conta na métrica."""
    tr = [r for r in R if usa(r) and r["split_ext"] == "treino_ext"]
    fold = folds_balanceados(collections.Counter(chave_fold(r) for r in tr))
    L = []
    for r in R:
        papel = "fora" if not usa(r) else "treino" if r["split_ext"] == "treino_ext" else "registro"
        k = fold[chave_fold(r)] if papel == "treino" else -1
        L.append(dict(dish_id=r["dish_id"], dominio=r["dominio"], grupo=r["grupo"], papel=papel, fold_ext=k, folds_proibidos=str(k) if papel == "treino" else "",
                      quadro_metrica=int(papel != "fora" and quadro_metrica(r)), grupo_metrica=grupo_metrica(r), cluster=cluster(r)))
    return L


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    saida = sys.argv[1]; listas = sys.argv[2] if len(sys.argv) > 2 else os.path.join(AQUI, "kaggle", "listas")
    os.makedirs(saida, exist_ok=True)
    le = lambda n: list(csv.DictReader(open(os.path.join(listas, "externo_%s.csv" % n))))
    grava(saida, "omnifood", omnifood(le("omnifood")))
    grava(saida, "flavoria", por_grupo(le("flavoria"), lambda r: r["oclusao"] == "0", lambda r: r["data"], lambda r: r["grupo"],
                                       lambda r: r["quadro_preferido"] == "1", lambda r: r["data"]))
    grava(saida, "fpb", por_grupo(le("fpb"), lambda r: r["vista"] == "zenital" and r["massa"] != "", lambda r: r["grupo"], lambda r: r["prato"],
                                  lambda r: True, lambda r: r["grupo"]))
    grava(saida, "mensa", por_grupo(le("mensa"), lambda r: True, lambda r: r["grupo"], lambda r: r["dish_id"], lambda r: True, lambda r: r["grupo"]))
    grava(saida, "lefood", por_grupo(le("lefood"), lambda r: True, lambda r: r["grupo"], lambda r: r["grupo"], lambda r: r["quadro"] == "antes", lambda r: r["grupo"]))
    grava(saida, "snapme", [dict(dish_id=r["dish_id"], dominio=r["dominio"], grupo=r["grupo"], papel="avaliacao_fraca", fold_ext=-1, quadro_metrica=1,
                                 grupo_metrica=r["dish_id"], cluster=r["grupo"]) for r in le("snapme")])


if __name__ == "__main__":
    main()
