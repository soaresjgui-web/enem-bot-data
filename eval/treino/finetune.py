#!/usr/bin/env python3
"""D — fine-tune leve de EfficientNet-B0 (ImageNet, pesos timm/GitHub), multi-tarefa
(log massa, log proteína, log kcal, log densidade), treino 'estrito', CPU.

Uso: python3 finetune.py [épocas=10]
Grava <TD>/pesos/effb0_ft.pt e <TD>/resultados/finetune.json (+ previsões por split).
Aumentos: espelho H/V, rotação 180°, brilho/contraste/saturação leves. Sem mudança de
escala (a escala carrega a informação de porção).
"""
import json, os, sys, time
import numpy as np, torch, timm
from PIL import Image
from comum import TD, ler_treino, ler_avaliacao, metricas, fmt
from experimentos import alvos, proteinas, treino_estrito, SPLITS_AVAL
from extrai_features import lista_imagens

H, W = 240, 320
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def cache_imagens():
    p = os.path.join(TD, "feat", f"img_{H}x{W}.npz")
    if not os.path.exists(p):
        itens = lista_imagens(TD)
        A = np.stack([np.asarray(Image.open(f).convert("RGB").resize((W, H), Image.BILINEAR)) for _, f in itens])
        np.savez(p, ids=np.array([i for i, _ in itens]), X=A)
    z = np.load(p)
    return {k: i for i, k in enumerate(z["ids"])}, z["X"]


def prep(a, aug, rng):
    x = torch.from_numpy(a).permute(0, 3, 1, 2).float() / 255
    if aug:
        for i in range(len(x)):
            if rng.random() < 0.5: x[i] = x[i].flip(2)
            if rng.random() < 0.5: x[i] = x[i].flip(1)
            b, c, s = 1 + rng.uniform(-.15, .15), 1 + rng.uniform(-.15, .15), 1 + rng.uniform(-.2, .2)
            g = x[i].mean(0, keepdim=True)
            x[i] = ((x[i] * b - x[i].mean()) * c + x[i].mean())
            x[i] = (g + (x[i] - g) * s).clamp(0, 1)
    return (x - MEAN) / STD


def main(epocas=10):
    torch.set_num_threads(int(os.environ.get("NTHREADS", "1")))  # nesta máquina compartilhada, 1 thread por processo foi o mais rápido
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    idx, A = cache_imagens()
    ids, Y = ler_treino()
    aval = ler_avaliacao()
    m = treino_estrito(ids, aval)
    tr_ids = [d for d, k in zip(ids, m) if k]
    T = alvos(Y[m]).astype(np.float32)
    mu, sd = T.mean(0), T.std(0)
    Tn = torch.from_numpy((T - mu) / sd)
    Xtr = A[[idx[d] for d in tr_ids]]
    net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4, drop_rate=0.2)
    sdict = torch.load(os.path.join(TD, "pesos", "efficientnet_b0_ra-3dd342df.pth"))
    net.load_state_dict({k: v for k, v in sdict.items() if not k.startswith("classifier")}, strict=False)
    head = [p for n, p in net.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in net.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": body, "lr": 4e-4}, {"params": head, "lr": 2e-3}], weight_decay=1e-4)
    bs, n = 32, len(Xtr)
    passos = epocas * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[4e-4, 2e-3], total_steps=passos, pct_start=0.15)
    pesos = torch.tensor([1.0, 1.0, 0.5, 1.0])
    t0 = time.time()
    for ep in range(epocas):
        net.train()
        perm = rng.permutation(n)
        perdas = []
        for b in range(n // bs):
            j = perm[b * bs:(b + 1) * bs]
            x = prep(Xtr[j], True, rng)
            loss = (torch.nn.functional.smooth_l1_loss(net(x), Tn[j], reduction="none") * pesos).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            perdas.append(loss.item())
        print(f"época {ep + 1}/{epocas} perda {np.mean(perdas):.4f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval()
    torch.save(net.state_dict(), os.path.join(TD, "pesos", "effb0_ft.pt"))
    res, prevs = {}, {}
    with torch.inference_mode():
        for s in SPLITS_AVAL:
            Xs = A[[idx[k] for k in aval[s][0]]]
            out = []
            for i in range(0, len(Xs), 64):
                x = prep(Xs[i:i + 64], False, rng)
                out.append(((net(x) + net(x.flip(3))) / 2).numpy())  # TTA espelho
            Tp = np.concatenate(out) * sd + mu
            pd, pm, massa, _ = proteinas(Tp)
            Ys = aval[s][2]
            res[s] = {"direto": metricas(Ys[:, 1], pd), "massa_x_dens": metricas(Ys[:, 1], pm),
                      "massa_mape": float(np.mean(np.abs(massa - Ys[:, 0]) / Ys[:, 0]) * 100)}
            prevs[s] = dict(zip(aval[s][1], pd.tolist()))
            print(f"{s:7s} direto {fmt(res[s]['direto'])} || m×d {fmt(res[s]['massa_x_dens'])} || massa MAPE {res[s]['massa_mape']:.0f}%")
    res["norm"] = {"mu": mu.tolist(), "sd": sd.tolist()}
    res["tempo_treino_s"] = time.time() - t0
    with open(os.path.join(TD, "resultados", "finetune.json"), "w") as f:
        json.dump({"metricas": res, "prev": prevs}, f, indent=1, default=float)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
