#!/usr/bin/env python3
"""Seção 8, ideia 5 (diagnóstico): por que o ajuste fino da EfficientNet-B0 (finetune.py, seção 2-D) foi mal?

Uso: OMP_NUM_THREADS=4 python3 zen_diag_effb0.py [receita=original|suave] [épocas=8]
Repete a receita original (lr 4e-4 no corpo, BatchNorm em modo treino, só espelhos) com um conjunto de
validação separado por cadeia (fold 0 do treino) e mede a validação A CADA ÉPOCA, coisa que o
experimento original não fazia. 'suave' = lr 1e-4, BatchNorm congelada, 4 orientações.
Compara com o ridge sobre a mesma EfficientNet-B0 congelada e sobre o SigLIP 2 no mesmo fold.
Usa o cache de imagens 240×320 de finetune.py (só os pratos de treino que já estavam nele).
"""
import json, os, sys, time
import numpy as np, torch, timm
from comum import TD, carrega_feat
from finetune import cache_imagens, prep, MEAN, STD
from zen_comum import Dados, ridge_cv, proteinas, crit, mape, I_M


def main(receita="original", epocas=8):
    torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
    torch.manual_seed(0); rng = np.random.default_rng(0)
    D = Dados()
    idx, A = cache_imagens()
    tem = np.array([d in idx for d in D.ids])
    tr = np.where(tem & (D.fold != 0))[0]; va = np.where(tem & (D.fold == 0))[0]
    cols = [0, 1, 2, 3]  # log massa, log1p proteína, log1p kcal, log densidade (como no original)
    T = D.T[:, cols].astype(np.float32)
    mu, sd = T[tr].mean(0), T[tr].std(0)
    Xtr = A[[idx[D.ids[i]] for i in tr]]; Xva = A[[idx[D.ids[i]] for i in va]]
    Tn = torch.from_numpy((T[tr] - mu) / sd)
    # referência: ridge sobre features congeladas no mesmo corte
    for bb in ("effb0", "siglip2_b16"):
        ix, X = carrega_feat(bb)
        Xr = np.stack([X[ix[D.ids[i]]] for i in np.r_[tr, va]])[:, None]
        f = np.r_[np.ones(len(tr), int), np.zeros(len(va), int)]
        oof, _, _ = ridge_cv(Xr, D.T[np.r_[tr, va]], f)
        o = oof[len(tr):]; P = proteinas(o)
        print(f"ridge {bb} congelado, validação (n={len(va)}): RMSE log-massa {np.sqrt(np.mean((o[:, I_M] - D.T[va, I_M]) ** 2)):.4f} | "
              f"MAPE massa {mape(D.Y['massa'][va], P['massa']):.1f}% | critério ponderado direto {crit(D.Y['proteina'][va], P['direto'], D.w_cv[va]):.1f}%", flush=True)
    net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4, drop_rate=0.2)
    sdict = torch.load(os.path.join(TD, "pesos", "efficientnet_b0_ra-3dd342df.pth"))
    net.load_state_dict({k: v for k, v in sdict.items() if not k.startswith("classifier")}, strict=False)
    head = [p for n, p in net.named_parameters() if n.startswith("classifier")]
    body = [p for n, p in net.named_parameters() if not n.startswith("classifier")]
    lr_c = 4e-4 if receita == "original" else 1e-4
    opt = torch.optim.AdamW([{"params": body, "lr": lr_c}, {"params": head, "lr": 2e-3}], weight_decay=1e-4 if receita == "original" else 0.05)
    bs, n = 32, len(Xtr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_c, 2e-3], total_steps=epocas * (n // bs), pct_start=0.15)
    pesos = torch.tensor([1.0, 1.0, 0.5, 1.0])
    hist, t0 = [], time.time()

    def valida():
        net.eval(); out = []
        with torch.inference_mode():
            for i in range(0, len(Xva), 64):
                x = prep(Xva[i:i + 64], False, rng)
                out.append(((net(x) + net(x.flip(3))) / 2).numpy())
        o = np.concatenate(out) * sd + mu
        massa = np.exp(o[:, 0]); pd = np.expm1(o[:, 1]).clip(0)
        return dict(rmse_logm=float(np.sqrt(np.mean((o[:, 0] - D.T[va, I_M]) ** 2))), mape_massa=mape(D.Y["massa"][va], massa),
                    crit_direto=crit(D.Y["proteina"][va], pd, D.w_cv[va]),
                    perda_val=float(torch.nn.functional.smooth_l1_loss(torch.from_numpy((o - mu) / sd), torch.from_numpy((T[va] - mu) / sd)).item()))

    for ep in range(epocas):
        net.train()
        if receita != "original":
            for m in net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        perm = rng.permutation(n); ls = []
        for b in range(n // bs):
            j = perm[b * bs:(b + 1) * bs]
            x = prep(Xtr[j], True, rng)
            if receita != "original":
                x = torch.stack([xi.flip(1, 2) if rng.random() < 0.5 else xi for xi in x])
            loss = (torch.nn.functional.smooth_l1_loss(net(x), Tn[j], reduction="none") * pesos).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); ls.append(loss.item())
        h = dict(epoca=ep + 1, perda_treino=float(np.mean(ls)), seg=time.time() - t0, **valida())
        hist.append(h); print(h, flush=True)
    with open(os.path.join(TD, "resultados", f"zen_diag_effb0_{receita}.json"), "w") as f:
        json.dump(hist, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "original", int(sys.argv[2]) if len(sys.argv) > 2 else 8)
