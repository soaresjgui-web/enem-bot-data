#!/usr/bin/env python3
"""Extrai embeddings congelados dos quadros LATERAIS do Nutrition5k (side_angles) e do ECUSTFD.

Uso: python3 extrai_features_lateral.py <dir_treino> <backbone> [--quadros 4] [--shards 4]
  Lê os manifestos em <scratchpad>/dados_ext/{n5k_side,ecustfd}/manifest.csv.
  Por prato do Nutrition5k escolhe até --quadros quadros, uma câmera por quadro, alternando
  o momento (A_ini, B_fim, C_ini, D_fim; se faltar câmera, completa com o outro momento).
  ECUSTFD: todas as imagens (só massa; licença não declarada -> uso só de pesquisa).
Grava <dir_treino>/feat/<backbone>_lateral.npz com ids "dish_x|A_ini" / "ecu|<arquivo>".
Pré-processamento idêntico a extrai_features.py (imagem inteira redimensionada, sem recorte).
Paraleliza em --shards processos de 1 thread (mais rápido que 1 processo de 4 threads nesta máquina).
"""
import csv, os, subprocess, sys, time
import numpy as np

SCR = "/tmp/claude-0/-home-user-enem-bot-data/59d3cb2b-a793-5c18-8c23-e9dab2604443/scratchpad"
PREF = ["A_ini", "B_fim", "C_ini", "D_fim", "A_fim", "B_ini", "C_fim", "D_ini"]


def lista(quadros):
    por_prato = {}
    with open(os.path.join(SCR, "dados_ext", "n5k_side", "manifest.csv")) as f:
        for r in csv.DictReader(f):
            q = r["notes"].split("frame=")[1].split("|")[0]
            por_prato.setdefault(r["dish_id"], {})[q] = r["image_path"]
    itens = []
    for d in sorted(por_prato):
        disp = por_prato[d]
        escolhidos, cams = [], set()
        for q in PREF:  # primeiro uma por câmera
            if q in disp and q[0] not in cams and len(escolhidos) < quadros:
                escolhidos.append(q); cams.add(q[0])
        for q in PREF:  # completa se faltam câmeras
            if q in disp and q not in escolhidos and len(escolhidos) < quadros:
                escolhidos.append(q)
        itens += [(f"{d}|{q}", disp[q]) for q in escolhidos]
    with open(os.path.join(SCR, "dados_ext", "ecustfd", "manifest.csv")) as f:
        for r in csv.DictReader(f):
            itens.append((f"ecu|{os.path.basename(r['image_path'])}", r["image_path"]))
    return itens


def roda_shard(td, nome, k, n, quadros):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extrai_features import modelo, carrega
    itens = lista(quadros)[k::n]
    f, hw, mean, std = modelo(nome)
    feats, t0 = [], time.time()
    with torch.inference_mode():
        for i in range(0, len(itens), 32):
            x = torch.stack([carrega(p, hw, mean, std) for _, p in itens[i:i + 32]])
            feats.append(f(x).float().numpy())
            if i % 640 == 0:
                print(f"shard {k}: {i}/{len(itens)} {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(td, "feat", f"_{nome}_lateral_{k}.npz"), ids=np.array([i for i, _ in itens]),
             X=np.concatenate(feats), seg=time.time() - t0)


def main():
    td, nome = sys.argv[1], sys.argv[2]
    quadros = int(sys.argv[sys.argv.index("--quadros") + 1]) if "--quadros" in sys.argv else 4
    n = int(sys.argv[sys.argv.index("--shards") + 1]) if "--shards" in sys.argv else 4
    if "--shard" in sys.argv:
        roda_shard(td, nome, int(sys.argv[sys.argv.index("--shard") + 1]), n, quadros)
        return
    t0 = time.time()
    print(f"{len(lista(quadros))} imagens, {n} processos")
    ps = [subprocess.Popen([sys.executable, __file__, td, nome, "--quadros", str(quadros), "--shards", str(n),
                            "--shard", str(k)], env={**os.environ, "OMP_NUM_THREADS": "1"}) for k in range(n)]
    assert all(p.wait() == 0 for p in ps)
    partes = [np.load(os.path.join(td, "feat", f"_{nome}_lateral_{k}.npz")) for k in range(n)]
    ids = np.concatenate([p["ids"] for p in partes]); X = np.concatenate([p["X"] for p in partes])
    np.savez(os.path.join(td, "feat", f"{nome}_lateral.npz"), ids=ids, X=X)
    for k in range(n):
        os.remove(os.path.join(td, "feat", f"_{nome}_lateral_{k}.npz"))
    dt = time.time() - t0
    print(f"{nome}: {X.shape} em {dt / 60:.1f} min = {dt / len(ids) * 1000:.0f} ms/img de parede ({n} processos)")


if __name__ == "__main__":
    main()
