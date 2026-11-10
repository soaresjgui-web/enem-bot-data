#!/usr/bin/env python3
"""Seção 8, ideia 5: ajuste fino PARCIAL do SigLIP 2 B/16 em CPU, multitarefa.

Uso: OMP_NUM_THREADS=4 python3 zen_finetune.py <modo> [--fold k] [--lr 2e-5] [--epocas 8] [--blocos 3]
                                                [--cabeca linear|aditiva] [--aux] [--nome ft]
  modo = holdout  treina sem o fold k (padrão 0) e valida nele a cada época (escolha de lr/épocas, só treino)
  modo = fold     igual, mas só prevê o fold k no fim (para completar as previsões fora-do-fold dos 5 folds)
  modo = final    treina em todos os pratos de treino e prevê os conjuntos de avaliação
  modo = junta    reúne os 5 folds + final numa entrada "<nome> cv" do banco (zen_avalia.py)

Por que parcial: os 9 primeiros blocos ficam congelados e seus tokens de saída já estão em disco
(zen_extrai.py --tokens 9, 4 orientações), então cada passo só percorre os 3 últimos blocos +
agregação por atenção (MAP) + cabeça. Custa ~1/4 do ajuste fino completo.
Aumento de dados: as 4 orientações que preservam o quadro (original, 180°, espelhos), sorteadas por
época, + drop-path. Previsão: média das 4 orientações.
Cabeça 'linear': camada linear sobre [agregado ‖ média dos tokens], INICIALIZADA com a solução do ridge
(o ponto de partida é o modelo congelado; o ajuste fino só pode sair dali). Cabeça 'aditiva': cada token
contribui com uma quantidade não negativa de massa/kcal/macros/massa por grupo e as contribuições são
somadas (a massa é aditiva no plano da foto: câmera a altura fixa).
Perda: Huber nos alvos em log padronizados (massa, proteína, kcal, densidade, gordura, carboidrato,
massa e proteína por grupo); --aux soma atributos de profundidade como alvos auxiliares.
"""
import copy, json, os, sys, time
import numpy as np, torch
from zen_comum import TD, Dados, Feat, ridge_cv, ridge_final, proteinas, crit, mape, I_M, I_P, I_K, I_D, I_F, I_C, I_MG, I_PG, N_ALVOS
from zen_experimentos import grava, SPLITS

PESOS = os.path.join(TD, "pesos")
K0 = 9


def carrega_tokens():
    ids = [l.strip() for l in open(os.path.join(TD, "feat", f"zen_siglip2_b16_tokL{K0}_ids.txt")) if l.strip()]
    return {k: i for i, k in enumerate(ids)}, np.load(os.path.join(TD, "feat", f"zen_siglip2_b16_tokL{K0}.npy"), mmap_mode="r")


class Topo(torch.nn.Module):
    """Últimos blocos do ViT + norma + agregação MAP + cabeça."""

    def __init__(self, n_saidas, cabeca="linear", drop_path=0.1):
        super().__init__()
        import timm
        from timm.models.vision_transformer import _load_weights
        m = timm.create_model("vit_base_patch16_siglip_224", pretrained=False, num_classes=0, drop_path_rate=drop_path)
        _load_weights(m, os.path.join(PESOS, "siglip2_b16_224.npz"))
        self.blocks = copy.deepcopy(m.blocks[K0:])
        self.norm, self.attn_pool, self.fc_norm = copy.deepcopy(m.norm), copy.deepcopy(m.attn_pool), copy.deepcopy(m.fc_norm)
        self.cabeca = cabeca
        if cabeca == "linear":
            self.head = torch.nn.Linear(2 * 768, n_saidas)
        else:
            self.tok = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.GELU(), torch.nn.Linear(256, n_saidas))
            self.glob = torch.nn.Linear(2 * 768, n_saidas)

    def feats(self, t):
        t = self.norm(self.blocks(t))
        return torch.cat([self.fc_norm(self.attn_pool(t)), t.mean(1)], 1), t

    def forward(self, t):
        f, tk = self.feats(t)
        if self.cabeca == "linear":
            return self.head(f)
        return self.glob(f), torch.nn.functional.softplus(self.tok(tk)).sum(1)


def treina(D, tr_idx, va_idx, tok_idx, TOK, a):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T = D.T
    tm, ts = T[tr_idx].mean(0), T[tr_idx].std(0) + 1e-6
    aux = None
    if a["aux"]:
        z = np.load(os.path.join(TD, "feat", "zen_depth.npz")); ix = {k: i for i, k in enumerate(z["ids"])}
        aux = z["X"][[ix[d] for d in D.ids]]
        aux = np.clip(np.nan_to_num((aux - np.nanmean(aux[tr_idx], 0)) / (np.nanstd(aux[tr_idx], 0) + 1e-6)), -5, 5).astype(np.float32)
    n_out = N_ALVOS + (aux.shape[1] if aux is not None else 0)
    net = Topo(n_out, a["cabeca"])
    # pesos da perda por alvo
    w = np.ones(n_out, np.float32)
    w[I_K] = 0.5; w[I_F] = 0.3; w[I_C] = 0.3; w[I_MG] = 0.15; w[I_PG] = 0.15; w[N_ALVOS:] = 0.3 / max(1, n_out - N_ALVOS) * 8
    w = torch.from_numpy(w)
    linhas = np.array([tok_idx[D.ids[i]] for i in tr_idx])
    Xtok = np.ascontiguousarray(TOK[np.sort(linhas)])           # [n, 4, 196, 768] float16 em RAM
    pos = np.argsort(np.argsort(linhas))
    Yn = ((T[tr_idx] - tm) / ts).astype(np.float32)
    if aux is not None:
        Yn = np.c_[Yn, aux[tr_idx]]
    Yn = torch.from_numpy(Yn)
    if a["cabeca"] == "linear":  # inicializa a cabeça com o ridge sobre as features congeladas (mesmos pratos)
        F = Feat("siglip2_b16")
        Xr = F.pega([D.ids[i] for i in tr_idx], (0, 1, 2, 3), ("agregado", "media"))
        f = np.zeros(len(D.ids), int); f[tr_idx] = D.fold[tr_idx]
        _, alfa, _ = ridge_cv(Xr, T[tr_idx], D.fold[tr_idx])
        _, m = ridge_final(Xr, T[tr_idx], alfa)
        Wn = (m.W / m.sd[:, None]) / ts            # saída padronizada
        bn = (m.mt - (m.mu / m.sd) @ m.W - tm) / ts
        with torch.no_grad():
            net.head.weight.zero_(); net.head.bias.zero_()
            net.head.weight[:N_ALVOS] = torch.from_numpy(Wn.T.astype(np.float32))
            net.head.bias[:N_ALVOS] = torch.from_numpy(bn.astype(np.float32))
    corpo = [p for n, p in net.named_parameters() if not n.startswith(("head", "tok", "glob"))]
    cab = [p for n, p in net.named_parameters() if n.startswith(("head", "tok", "glob"))]
    if a["blocos"] < 3:  # congela os blocos mais baixos
        for b in net.blocks[:3 - a["blocos"]]:
            for p in b.parameters():
                p.requires_grad_(False)
    lr_cab = a["lr"] * (1 if a["cabeca"] == "linear" else 30)
    opt = torch.optim.AdamW([{"params": [p for p in corpo if p.requires_grad], "lr": a["lr"]},
                             {"params": cab, "lr": lr_cab}], weight_decay=0.05)
    bs = 32
    npb = len(tr_idx) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a["lr"], lr_cab], total_steps=a["epocas"] * npb, pct_start=0.15)
    huber = torch.nn.functional.smooth_l1_loss
    tm_t, ts_t = torch.from_numpy(tm.astype(np.float32)), torch.from_numpy(ts.astype(np.float32))

    def perda(out, y):
        if a["cabeca"] == "linear":
            return (huber(out, y, reduction="none") * w).mean()
        g, soma = out
        # rota aditiva: soma das contribuições por token -> log -> padroniza (massa, kcal, gordura, carbo, proteína, grupos)
        la = torch.log(soma + 1e-3)
        cols = [I_M, I_K, I_F, I_C, I_P] + list(range(I_MG.start, I_PG.stop))
        alvo_log = y[:, cols] * ts_t[cols] + tm_t[cols]            # volta para log / log1p
        # massa é log(m); os demais são log1p(x): compara em log1p para os demais
        pred_log = torch.cat([la[:, :1], torch.log1p(soma[:, 1:len(cols)])], 1)
        l_ad = (huber((pred_log - tm_t[cols]) / ts_t[cols], y[:, cols], reduction="none") * w[cols]).mean()
        return (huber(g, y, reduction="none") * w).mean() + l_ad

    def preve(indices):
        net.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(indices), 16):
                lin = [tok_idx[k] for k in indices[i:i + 16]]
                t = torch.from_numpy(np.stack([TOK[j] for j in lin]).astype(np.float32))   # [b, 4, 196, 768]
                o = net(t.flatten(0, 1))
                if a["cabeca"] == "linear":
                    p = o[:, :N_ALVOS] * ts_t + tm_t
                else:
                    g, soma = o
                    p = g[:, :N_ALVOS] * ts_t + tm_t
                    # massa e proteína: média (em log) entre a rota global e a aditiva
                    p[:, I_M] = 0.5 * (p[:, I_M] + torch.log(soma[:, 0] + 1e-3))
                    p[:, I_P] = 0.5 * (p[:, I_P] + torch.log1p(soma[:, 4]))
                out.append(p.reshape(len(lin), 4, -1).mean(1).numpy())
        return np.concatenate(out)

    hist = []
    t0 = time.time()
    if va_idx is not None:
        pv = preve([D.ids[i] for i in va_idx]); P = proteinas(pv)
        hist.append(dict(epoca=0, rmse_logm=float(np.sqrt(np.mean((pv[:, I_M] - T[va_idx, I_M]) ** 2))),
                         crit_direto=crit(D.Y["proteina"][va_idx], P["direto"], D.w_cv[va_idx]),
                         crit_mxd=crit(D.Y["proteina"][va_idx], P["mxd"], D.w_cv[va_idx])))
        print("época 0 (ponto de partida):", hist[-1], flush=True)
    for ep in range(a["epocas"]):
        net.train()
        perm = rng.permutation(len(tr_idx)); ls = []
        for b in range(npb):
            j = perm[b * bs:(b + 1) * bs]
            o = rng.integers(0, 4, len(j))
            x = torch.from_numpy(Xtok[pos[j], o].astype(np.float32))
            loss = perda(net(x), Yn[j])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); ls.append(loss.item())
        h = dict(epoca=ep + 1, perda=float(np.mean(ls)), seg=time.time() - t0)
        if va_idx is not None:
            pv = preve([D.ids[i] for i in va_idx]); P = proteinas(pv)
            h.update(rmse_logm=float(np.sqrt(np.mean((pv[:, I_M] - T[va_idx, I_M]) ** 2))),
                     mape_massa=mape(D.Y["massa"][va_idx], P["massa"]),
                     crit_direto=crit(D.Y["proteina"][va_idx], P["direto"], D.w_cv[va_idx]),
                     crit_mxd=crit(D.Y["proteina"][va_idx], P["mxd"], D.w_cv[va_idx]))
        hist.append(h); print(h, flush=True)
    return net, preve, hist, (tm, ts)


def main():
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    a = dict(lr=float(arg("--lr", 2e-5)), epocas=int(arg("--epocas", 8)), blocos=int(arg("--blocos", 3)),
             cabeca=arg("--cabeca", "linear"), aux="--aux" in sys.argv)
    nome = arg("--nome", "ft")
    modo = sys.argv[1]
    torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
    D = Dados()
    partes = os.path.join(TD, "resultados", "zen_banco", "partes")
    os.makedirs(partes, exist_ok=True)
    if modo == "junta":   # previsões fora-do-fold dos 5 folds + previsões finais -> uma entrada do banco
        oof = np.full((len(D.ids), N_ALVOS), np.nan)
        for k in range(5):
            z = np.load(os.path.join(partes, f"{nome} fold{k}.npz"))
            oof[D.fold == k] = z["oof"][D.fold == k]
        assert not np.isnan(oof).any()
        z = np.load(os.path.join(partes, f"{nome} final.npz"))
        grava(f"{nome} cv", oof, {s: z[s] for s in SPLITS})
        return
    tok_idx, TOK = carrega_tokens()
    if modo in ("holdout", "fold"):
        k = int(arg("--fold", 0))
        tr, va = np.where(D.fold != k)[0], np.where(D.fold == k)[0]
        net, preve, hist, _ = treina(D, tr, va if modo == "holdout" else None, tok_idx, TOK, a)
        oof = np.full((len(D.ids), N_ALVOS), np.nan); oof[va] = preve([D.ids[i] for i in va])
        np.savez(os.path.join(partes, f"{nome} fold{k}.npz"), oof=oof, hist=json.dumps(hist), cfg=json.dumps(a))
        P = proteinas(oof[va])
        print(f"fold {k}: RMSE log-massa {np.sqrt(np.mean((oof[va, I_M] - D.T[va, I_M]) ** 2)):.4f} | critério ponderado direto "
              f"{crit(D.Y['proteina'][va], P['direto'], D.w_cv[va]):.1f}% m×d {crit(D.Y['proteina'][va], P['mxd'], D.w_cv[va]):.1f}%", flush=True)
    else:
        net, preve, hist, (tm, ts) = treina(D, np.arange(len(D.ids)), None, tok_idx, TOK, a)
        prev = {s: preve(D.aval[s]["chave"]) for s in SPLITS}
        np.savez(os.path.join(partes, f"{nome} final.npz"), hist=json.dumps(hist), cfg=json.dumps(a), **prev)
        torch.save(dict(estado=net.state_dict(), tm=tm, ts=ts, cfg=a), os.path.join(PESOS, f"zen_{nome}.pt"))


if __name__ == "__main__":
    main()
