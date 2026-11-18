#!/usr/bin/env python3
"""Seção 11: monta o conjunto de teste novo (teste2) do Nutrition5k, sem olhar nenhuma previsão.

Uso: python3 teste2_monta.py <dir_treino> <dir_cego4> [--so-conta]
  <dir_treino>  scratchpad/treino (metadados em ../n5k, fotos zenitais em img/<dish_id>/rgb.png, meta/zen_treino.csv)
  <dir_cego4>   pasta dos lotes cegos (fora do repositório): loteNN/eNNNN.png + INSTRUCOES.md
  --so-conta    só imprime as contagens; não grava nada

Quem entra (regra fixa; nenhuma amostragem por rótulo):
  (a) prato com foto zenital e fora dos 450 ids de avaliação (calib, teste, n5k_v1);
  (b) nunca usado para treinar nada: fora do treino zenital estrito (1.684 pratos, o único treino de todos os
      componentes das seções 8 a 10, CPU e GPU) e fora dos pratos cujos quadros laterais treinaram o ridge da seção 8;
  (c) a mais de 120 s de todo prato usado em treino OU em ajuste: a cadeia do prato (intervalos < 120 s, calculada
      sobre os 5.006 pratos com metadado) não contém prato de treino nem prato do calib (o calib ajustou os pesos e
      a forma do híbrido);
  (d) filtros de qualidade do v2 (coleta_n5k_v2.valido): massa >= 20 g, nenhum ingrediente vazio/'deprecated'/
      'plate only', soma das proteínas e das gramas dos ingredientes coerente com o total; e o filtro do treino
      (zen_dados: o mesmo, por café).
  Entram TODOS os que passam. Todo prato que passa está na cadeia de algum prato do teste antigo ou do n5k_v1
  (é por isso que nunca entrou no treino); a coluna `cadeia_contem` registra qual.

Grava:
  docs/dados/precisao/n5k_teste2_amostra.csv     mesmo esquema de n5k_v2_amostra.csv + cadeia, cadeia_contem, split_oficial, lote
  docs/dados/precisao/gabarito_pratos_teste2.csv mesmo esquema de gabarito_pratos_v2.csv (origem n5k_teste2, split teste2)
  eval/treino/kaggle/listas/aval2_ids.csv        prato_id,split,dish_id (sem rótulo), como aval_ids.csv
  <dir_cego4>/loteNN/eNNNN.png                   fotos com id neutro, embaralhadas, ~30 por lote; INSTRUCOES.md = texto v3
Os ids neutros e0001… são atribuídos em ordem aleatória (semente fixa), sem relação com conteúdo, cadeia ou horário.
"""
import csv, os, random, shutil, sys
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, ".."))
from comum import PREC, grupos_sessao  # noqa: E402
from zen_dados import ler_meta_completo, avaliacao  # noqa: E402
from prep_dados import N5K  # noqa: E402
sys.path.insert(0, PREC)
import coleta_n5k_v2 as V2  # noqa: E402

SEED = 20261111
POR_LOTE = 30
FAIXAS = [(0, 5, "0-5"), (5, 15, "5-15"), (15, 25, "15-25"), (25, 40, "25-40"), (40, 1e9, ">40")]   # como no v2: lo <= p < hi


def candidatos(td):
    meta = ler_meta_completo()
    m2 = V2.ler_meta([os.path.join(N5K, f"dish_metadata_{c}.csv") for c in ("cafe1", "cafe2")], os.path.join(N5K, "ingredients_metadata.csv"))
    ev = {d: s for _, d, s in avaliacao()}
    todos = sorted(meta); cadeia = dict(zip(todos, grupos_sessao(todos, 120)))
    com_foto = {d for d in os.listdir(os.path.join(td, "img")) if os.path.exists(os.path.join(td, "img", d, "rgb.png"))}
    with open(os.path.join(td, "meta", "zen_treino.csv")) as f:
        treino = {r["dish_id"] for r in csv.DictReader(f)}
    # pratos cujos quadros laterais treinaram o ridge "com laterais" (mesma regra de zen_experimentos.dados_laterais)
    import numpy as np
    z = np.load(os.path.join(td, "feat", "siglip2_b16_lateral.npz"))
    proib = {cadeia[d] for d in ev}
    lat = {k.split("|")[0] for k in z["ids"] if not k.startswith("ecu|")}
    lateral = {d for d in lat if d in meta and meta[d]["ok"] and meta[d]["massa"] >= 20 and d not in ev and cadeia[d] not in proib}
    usados = treino | lateral
    cad_usada = {cadeia[d] for d in usados} | {cadeia[d] for d, s in ev.items() if s == "calib"}
    tipos = {}
    for d, s in ev.items():
        tipos.setdefault(cadeia[d], set()).add(s)
    fonte = {}
    for nome in ("rgb_train_ids.txt", "rgb_test_ids.txt"):
        for l in open(os.path.join(td, "meta", "splits", nome)):
            fonte[l.strip()] = nome.split("_")[1]
    passos = dict(com_foto=len(com_foto))
    c = [d for d in sorted(com_foto) if d in meta and d in m2 and meta[d]["ok"] and meta[d]["massa"] >= 20 and V2.valido(m2[d])]
    passos["passam_nos_filtros"] = len(c)
    c = [d for d in c if d not in ev]; passos["fora_da_avaliacao"] = len(c)
    c = [d for d in c if d not in usados]; passos["nunca_treinados"] = len(c)
    passos["em_cadeia_de_treino"] = sum(cadeia[d] in {cadeia[x] for x in usados} for d in c)
    passos["em_cadeia_do_calib"] = sum("calib" in tipos.get(cadeia[d], ()) for d in c)
    c = [d for d in c if cadeia[d] not in cad_usada]; passos["teste2"] = len(c)
    assert not set(c) & (set(ev) | usados) and not {cadeia[d] for d in c} & cad_usada
    return c, meta, m2, cadeia, tipos, fonte, passos


def main():
    td, cego = sys.argv[1], sys.argv[2]
    c, meta, m2, cadeia, tipos, fonte, passos = candidatos(td)
    print("contagens:", passos)
    contem = lambda d: "+".join(sorted(tipos.get(cadeia[d], {"nenhum"})))
    print("por tipo de cadeia:", dict(Counter(contem(d) for d in c)), "| cadeias:", len({cadeia[d] for d in c}))
    print("por split oficial:", dict(Counter(fonte.get(d, "sem_split") for d in c)), "| por café:", dict(Counter(meta[d]["cafe"] for d in c)))
    faixa = lambda p: next(lbl for lo, hi, lbl in FAIXAS if lo <= p < hi)
    print("por faixa de proteína (só contagem, para o desenho da leitura):", dict(Counter(faixa(m2[d]["proteina"]) for d in c)))
    print("por faixa de nº de ingredientes:", dict(Counter(V2.faixa_ing(len(m2[d]["ingr"])) for d in c)))
    if "--so-conta" in sys.argv:
        return
    rnd = random.Random(SEED)
    ordem = sorted(c); rnd.shuffle(ordem)                       # id neutro em ordem aleatória
    pid = {d: f"e{i + 1:04d}" for i, d in enumerate(ordem)}
    emb = list(ordem); rnd.shuffle(emb)                         # outra ordem aleatória para repartir em lotes
    n_lotes = -(-len(emb) // POR_LOTE)
    lote = {d: f"lote{i % n_lotes + 1:02d}" for i, d in enumerate(emb)}
    linhas = []
    for d in ordem:
        m = m2[d]
        linhas.append({"prato_id": pid[d], "dish_id": d, "split": "teste2", "cafe": meta[d]["cafe"], "faixa_proteina": faixa(m["proteina"]),
                       "faixa_ingredientes": V2.faixa_ing(len(m["ingr"])), "n_ingredientes": len(m["ingr"]), "massa_g": round(m["massa"], 1),
                       "proteina_g": round(m["proteina"], 2), "kcal": round(m["kcal"], 1), "confianca": "B" if m["totais_da_soma"] else "A",
                       "ingredientes": "; ".join(f"{i['nome']} {i['gramas']:.0f} g" for i in m["ingr"]),
                       "cadeia": cadeia[d], "cadeia_contem": contem(d), "split_oficial": fonte.get(d, "sem_split"), "lote": lote[d]})
    with open(os.path.join(PREC, "n5k_teste2_amostra.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    with open(os.path.join(PREC, "gabarito_pratos_teste2.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["prato_id", "origem", "split", "categoria", "peso_porcao_g", "proteina_g", "confianca", "lote_v2"])
        for r in linhas:
            w.writerow([r["prato_id"], "n5k_teste2", "teste2", "n5k", r["massa_g"], r["proteina_g"], r["confianca"], r["lote"]])
    with open(os.path.join(AQUI, "kaggle", "listas", "aval2_ids.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prato_id", "split", "dish_id"])
        for r in linhas:
            w.writerow([r["prato_id"], "teste2", r["dish_id"]])
    # lotes cegos: só imagens com id neutro (regravadas, sem metadado) + o texto do estimador v3
    from PIL import Image
    if os.path.isdir(cego):
        shutil.rmtree(cego)
    for d in emb:
        pasta = os.path.join(cego, lote[d]); os.makedirs(pasta, exist_ok=True)
        Image.open(os.path.join(td, "img", d, "rgb.png")).convert("RGB").save(os.path.join(pasta, pid[d] + ".png"))
    t = open(os.path.join(AQUI, "..", "prompt_v3.md"), encoding="utf-8").read()
    a = t.index("\n---\n") + 5; b = t.index("\n---\n", a)
    with open(os.path.join(cego, "INSTRUCOES.md"), "w", encoding="utf-8") as f:
        f.write(t[a:b].strip() + "\n")
    print(f"{len(linhas)} pratos em {n_lotes} lotes ({dict(Counter(Counter(lote.values()).values()))} imagens por lote) -> {cego}")


if __name__ == "__main__":
    main()
