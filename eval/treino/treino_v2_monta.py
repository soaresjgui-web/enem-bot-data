#!/usr/bin/env python3
"""Seção 12: monta o treino ampliado do Nutrition5k (treino_v2), sem tocar na higiene do teste2.

Uso: python3 treino_v2_monta.py <dir_treino> [--so-conta]

Com o LLM fora do pipeline (decisão de 11/11/2026), o `calib` não precisa mais ficar de fora, e o `teste`
antigo deixa de ser o teste de registro (é o teste2). Entra no treino_v2 TODO prato que:
  (a) tem foto zenital e passa no filtro do treino (zen_dados: massa >= 20 g, sem ingrediente vazio/'deprecated'/
      'plate only', soma coerente);
  (b) não é do teste2;
  (c) não está na cadeia (intervalos < 120 s, sobre os 5.006 pratos com metadado) de nenhum prato do teste2.
Isso dá: o treino antigo (1.684), os pratos do calib, os que tinham ficado de fora só por estarem em cadeia do
calib, e os pratos do `teste` antigo e do `n5k_v1` cujas cadeias não tocam o teste2. Os pratos do `teste` antigo e
do `n5k_v1` que estão em cadeia do teste2 continuam fora de tudo (são irmãos do teste2).

Grava:
  eval/treino/kaggle/listas/treino_v2.csv   mesmo esquema de listas/treino.csv (dish_id, cadeia, fold por cadeia,
        split_oficial, alvos, massa e proteína por grupo) + 10 colunas ff_<fonte> no fim (fração da massa do prato
        em cada fonte de proteína, alvo das cabeças de frações; quem lê por nome de coluna não é afetado)
  <dir_treino>/meta/zen_treino_v2.csv       o mesmo, sem a coluna fold (como zen_treino.csv), para a CPU
  docs/dados/precisao/n5k_treino_v2_origem.csv   dish_id, origem (treino_v1 | calib | cadeia_do_calib | teste_antigo |
        n5k_v1), cadeia
Folds: GroupKFold(5) por cadeia sobre a lista ordenada por dish_id (a mesma conta de zen_comum.Dados).
Conferência (assert): nenhum prato do teste2; nenhuma cadeia em comum com o teste2; toda cadeia num fold só.
"""
import csv, os, sys
from collections import Counter
import numpy as np
from sklearn.model_selection import GroupKFold

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, ".."))
from comum import PREC, grupos_sessao  # noqa: E402
from zen_dados import ler_meta_completo, avaliacao, COLS  # noqa: E402
from grupos_alimentos import GRUPOS  # noqa: E402
from prep_dados import N5K  # noqa: E402
from dens_comum import FONTES, fonte_de  # noqa: E402
sys.path.insert(0, PREC)
import coleta_n5k_v2 as V2  # noqa: E402


def main():
    td = sys.argv[1]
    meta = ler_meta_completo()
    ev = {d: s for _, d, s in avaliacao()}
    todos = sorted(meta); cadeia = dict(zip(todos, grupos_sessao(todos, 120)))
    com_foto = {d for d in os.listdir(os.path.join(td, "img")) if os.path.exists(os.path.join(td, "img", d, "rgb.png"))}
    with open(os.path.join(td, "meta", "zen_treino.csv")) as f:
        v1 = {r["dish_id"] for r in csv.DictReader(f)}
    with open(os.path.join(PREC, "n5k_teste2_amostra.csv")) as f:
        t2 = {r["dish_id"] for r in csv.DictReader(f)}
    with open(os.path.join(AQUI, "kaggle", "listas", "aval2_ids.csv")) as f:
        assert {r["dish_id"] for r in csv.DictReader(f)} == t2
    cad_t2 = {cadeia[d] for d in t2}
    cad_calib = {cadeia[d] for d, s in ev.items() if s == "calib"}
    ok = [d for d in sorted(com_foto) if d in meta and meta[d]["ok"] and meta[d]["massa"] >= 20]
    v2 = [d for d in ok if d not in t2 and cadeia[d] not in cad_t2]
    assert v1 <= set(v2) and not set(v2) & t2 and not {cadeia[d] for d in v2} & cad_t2
    origem = lambda d: "treino_v1" if d in v1 else {"calib": "calib", "teste": "teste_antigo", "n5k_v1": "n5k_v1"}.get(ev.get(d), "cadeia_do_calib")
    assert all(cadeia[d] in cad_calib for d in v2 if origem(d) == "cadeia_do_calib")
    fold = np.zeros(len(v2), int)
    for k, (_, va) in enumerate(GroupKFold(5).split(v2, groups=[cadeia[d] for d in v2])):
        fold[va] = k
    fc = {}
    for d, k in zip(v2, fold):
        assert fc.setdefault(cadeia[d], k) == k
    p = np.array([meta[d]["proteina"] for d in v2]); p1 = np.array([meta[d]["proteina"] for d in sorted(v1)])
    fx = lambda x: np.bincount(np.digitize(x, [5, 15, 25, 40]), minlength=5).tolist()
    print(f"treino_v2: {len(v2)} pratos em {len(fc)} cadeias (treino antigo: {len(v1)} em {len({cadeia[d] for d in v1})})")
    print("  origem:", dict(Counter(origem(d) for d in v2)))
    print("  por faixa de proteína (até 5 / 5-15 / 15-25 / 25-40 / > 40 g): antes", fx(p1), "-> agora", fx(p),
          f"| > 25 g: {int((p1 > 25).sum())} -> {int((p > 25).sum())} | > 15 g: {int((p1 > 15).sum())} -> {int((p > 15).sum())}")
    print("  por fold:", np.bincount(fold).tolist(), "| > 25 g por fold:", [int((p[fold == k] > 25).sum()) for k in range(5)],
          "| por café:", dict(Counter(meta[d]["cafe"] for d in v2)))
    fora = {s: [d for d, x in ev.items() if x == s and d not in set(v2)] for s in ("calib", "teste", "n5k_v1")}
    print("  avaliação antiga que continua fora:", {s: f"{len(v)} ({sum(cadeia[d] in cad_t2 for d in v)} em cadeia do teste2, {sum(cadeia[d] not in cad_t2 for d in v)} pelo filtro)" for s, v in fora.items()})
    print("  conferência: 0 pratos do teste2, 0 cadeias em comum com o teste2, cada cadeia num fold só")
    if "--so-conta" in sys.argv:
        return
    # fração da massa por fonte de proteína (alvo das cabeças de frações), dos ingredientes do Nutrition5k
    m2 = V2.ler_meta([os.path.join(N5K, f"dish_metadata_{c}.csv") for c in ("cafe1", "cafe2")], os.path.join(N5K, "ingredients_metadata.csv"))
    fonte = {}
    for nome in ("rgb_train_ids.txt", "rgb_test_ids.txt"):
        for l in open(os.path.join(td, "meta", "splits", nome)):
            fonte[l.strip()] = nome.split("_")[1]

    def linha(d):
        m = meta[d]
        ff = dict.fromkeys(FONTES, 0.0)
        for i in m2[d]["ingr"]:
            ff[fonte_de(i["nome"])] += i["gramas"]
        tot = max(sum(ff.values()), 1e-9)
        return ([d, cadeia[d], fonte.get(d, "sem_split")] + [round(m[c], 3) for c in COLS] + [round(m["mg"][g], 2) for g in GRUPOS]
                + [round(m["pg"][g], 3) for g in GRUPOS] + [round(ff[k] / tot, 4) for k in FONTES])
    cab = ["dish_id", "cadeia", "split_oficial"] + COLS + [f"m_{g}" for g in GRUPOS] + [f"p_{g}" for g in GRUPOS] + [f"ff_{k}" for k in FONTES]
    with open(os.path.join(td, "meta", "zen_treino_v2.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cab)
        for d in v2:
            w.writerow(linha(d))
    cab2 = list(cab); cab2.insert(2, "fold")
    with open(os.path.join(AQUI, "kaggle", "listas", "treino_v2.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cab2)
        for d, k in zip(v2, fold):
            l = linha(d); l.insert(2, int(k)); w.writerow(l)
    with open(os.path.join(PREC, "n5k_treino_v2_origem.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dish_id", "origem", "cadeia"])
        for d in v2:
            w.writerow([d, origem(d), cadeia[d]])
    print("gravados: kaggle/listas/treino_v2.csv, meta/zen_treino_v2.csv, docs/dados/precisao/n5k_treino_v2_origem.csv")


if __name__ == "__main__":
    main()
