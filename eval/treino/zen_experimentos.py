#!/usr/bin/env python3
"""Seção 8 (uma única foto de cima): cabeças sobre features congeladas, com orientações D4,
features espaciais, alvos multitarefa, profundidade privilegiada e vários backbones.

Uso: OMP_NUM_THREADS=4 python3 zen_experimentos.py <etapa> [...]
  ridge <backbone> [...]      ridge por configuração de orientação/blocos para cada backbone
  mlp <backbone>              cabeça MLP multitarefa (com e sem alvos auxiliares de profundidade)
  escala                      robustez de escala (zoom, escala aleatória, recorte pelo prato)
  lateral                     quadros laterais do Nutrition5k como sinal extra só de treino
  escala_d4                   8 orientações + escala aleatória no treino (entra no banco e na tabela de zoom)
Cada configuração grava <TD>/resultados/zen_banco/<nome>.npz com as previsões em log:
  oof [n_treino, alvos] (fora-do-fold, GroupKFold por cadeia) e uma matriz por split de avaliação.
Nada aqui olha o gabarito do teste: a seleção e as tabelas ficam em zen_avalia.py.
"""
import os, sys, time
import numpy as np
from zen_comum import TD, Dados, Feat, ridge_cv, ridge_final, proteinas, crit, mape, I_M, I_P, N_ALVOS

BANCO = os.path.join(TD, "resultados", "zen_banco")
SPLITS = ["calib", "teste", "n5k_v1", "ufpr", "liv_up", "seara"]
O8, O4, O1 = tuple(range(8)), tuple(range(4)), (0,)


def grava(nome, oof, prev, extra=None):
    os.makedirs(BANCO, exist_ok=True)
    np.savez(os.path.join(BANCO, nome + ".npz"), oof=oof, **prev, **(extra or {}))


def resumo_cv(D, nome, oof, t0=None):
    P = proteinas(oof)
    s = (f"{nome:46s} CV: log-massa RMSE {np.sqrt(np.mean((oof[:, I_M] - D.T[:, I_M]) ** 2)):.4f} | MAPE massa {mape(D.Y['massa'], P['massa']):4.1f}%"
         f" | critério ponderado direto {crit(D.Y['proteina'], P['direto'], D.w_cv):4.1f}% m×d {crit(D.Y['proteina'], P['mxd'], D.w_cv):4.1f}%"
         f" grupos {crit(D.Y['proteina'], P['grupos'], D.w_cv):4.1f}%")
    print(s + (f" ({time.time() - t0:.0f}s)" if t0 else ""), flush=True)


def roda_ridge(D, nome, F, o_tr, o_te, blocos, splits=SPLITS):
    """F: lista de Feat (concatena os blocos). Treina com as orientações o_tr e prevê com a média em o_te."""
    t0 = time.time()
    cat = lambda chaves, o: np.concatenate([f.pega(chaves, o, blocos) for f in F], -1)
    Xtr, Xte = cat(D.ids, o_tr), cat(D.ids, o_te)
    oof, alfa, mse = ridge_cv(Xtr, D.T, D.fold, Xte)
    pred, _ = ridge_final(Xtr, D.T, alfa)
    prev = {s: pred(cat(D.aval[s]["chave"], o_te)) for s in splits if all(k in F[0].idx for k in D.aval[s]["chave"])}
    grava(nome, oof, prev, dict(alfa=alfa, cv_mse=mse))
    resumo_cv(D, nome, oof, t0)


def etapa_ridge(D, bbs):
    for bb in bbs:
        F = Feat(bb)
        no = F.X.shape[1]
        cfgs = [("o1", O1, O1, ("agregado", "media"))]
        if no >= 2:
            cfgs.append(("o2", (0, 1), (0, 1), ("agregado", "media")))
        if no >= 4:
            cfgs.append(("o4", O4, O4, ("agregado", "media")))
        if no >= 8:
            cfgs += [("o8", O8, O8, ("agregado", "media")), ("o1-tta8", O1, O8, ("agregado", "media")),
                     ("o8+gem", O8, O8, ("agregado", "media", "gem")),
                     ("o8+gem+grade", O8, O8, ("agregado", "media", "gem", "grade"))]
        else:
            om = tuple(range(no))
            cfgs.append((f"o{no}+gem+grade", om, om, ("agregado", "media", "gem", "grade")))
        for suf, otr, ote, bl in cfgs:
            roda_ridge(D, f"ridge {bb} {suf}", [F], otr, ote, bl)


def dados_laterais(D, bb="siglip2_b16"):
    """Quadros laterais do Nutrition5k permitidos pela regra de cadeia: (X, alvos em log, fold de cada quadro)."""
    from zen_comum import alvos
    from zen_dados import ler_meta_completo
    from comum import grupos_sessao
    z = np.load(os.path.join(TD, "feat", f"{bb}_lateral.npz"))
    meta = ler_meta_completo()
    todos = sorted(meta); cadeia = dict(zip(todos, grupos_sessao(todos, 120)))
    ev = {d for s in N5K_AVAL for d in D.aval[s]["chave"]}
    proib = {cadeia[d] for d in ev}
    fold_cadeia = {c: f for c, f in zip(D.cadeia, D.fold)}
    Xl, Tl, fl = [], [], []
    Xlat = z["X"]   # (um acesso só: cada z["X"] relê o arquivo inteiro)
    for i, k in enumerate(z["ids"]):
        if k.startswith("ecu|"):
            continue
        d = k.split("|")[0]
        m = meta.get(d)
        if m is None or not m["ok"] or m["massa"] < 20 or d in ev or cadeia[d] in proib:
            continue
        Y = {c: np.array([m[c]]) for c in ("massa", "proteina", "kcal", "gordura", "carbo")}
        Y["mg"] = np.array([[m["mg"][g] for g in m["mg"]]]); Y["pg"] = np.array([[m["pg"][g] for g in m["pg"]]])
        Xl.append(Xlat[i]); Tl.append(alvos(Y)[0]); fl.append(fold_cadeia.get(cadeia[d], cadeia[d] % 5))
    return np.array(Xl, np.float32), np.array(Tl), np.array(fl)


def ridge_lateral(D, F, Xl, Tl, fl):
    """Ridge com 8 orientações zenitais + quadros laterais como amostras; CV só no zenital.
    Devolve (previsões fora-do-fold, alfas, MSE de CV, modelo final)."""
    from zen_comum import Ridge
    bl = ("agregado", "media")
    Xz = F.pega(D.ids, O8, bl); n, o, Dm = Xz.shape
    r = Ridge()
    P = np.zeros((len(r.ALFAS), n, D.T.shape[1]))
    for k in range(5):
        tr, va = D.fold != k, D.fold == k
        Xt = np.r_[Xz[tr].reshape(-1, Dm), Xl[fl != k]]; Tt = np.r_[np.repeat(D.T[tr], o, 0), Tl[fl != k]]
        P[:, va] = r.caminho(Xt, Tt, Xz[va].reshape(-1, Dm)).reshape(len(r.ALFAS), va.sum(), o, -1).mean(2)
    mse = ((P - D.T) ** 2).mean(1); best = mse.argmin(0)
    oof = np.stack([P[best[j], :, j] for j in range(D.T.shape[1])], 1)
    m = Ridge().fit(np.r_[Xz.reshape(-1, Dm), Xl], np.r_[np.repeat(D.T, o, 0), Tl], r.ALFAS[best])
    return oof, r.ALFAS[best], mse.min(0), m


def etapa_lateral(D, bb="siglip2_b16"):
    """Sinal só de treino: soma ao treino zenital (8 orientações) os quadros laterais do Nutrition5k
    (features da seção 7, 1 orientação) dos pratos permitidos pela mesma regra de cadeia. Valida só no zenital."""
    t0 = time.time()
    F = Feat(bb); bl = ("agregado", "media")
    Xl, Tl, fl = dados_laterais(D, bb)
    print(f"quadros laterais permitidos: {len(Xl)}")
    oof, alfa, mse, m = ridge_lateral(D, F, Xl, Tl, fl)
    Dm = Xl.shape[1]
    prev = {s: m.predict(F.pega(D.aval[s]["chave"], O8, bl).reshape(-1, Dm)).reshape(len(D.aval[s]["chave"]), 8, -1).mean(1) for s in SPLITS}
    grava(f"ridge {bb} o8 + quadros laterais no treino", oof, prev, dict(alfa=alfa, cv_mse=mse))
    resumo_cv(D, f"ridge {bb} o8 + quadros laterais no treino", oof, t0)


N5K_AVAL = ("calib", "teste", "n5k_v1")


# ---------------------------------------------------------------- MLP multitarefa
def treina_mlp(Xtr, Ttr, Xva_list, aux=None, oculto=512, wd=1e-2, epocas=60, drop=0.3, seed=0, peso_aux=0.3, devolve_modelo=False):
    """MLP de 1 camada oculta sobre features padronizadas; alvos principais (log) + auxiliares (padronizados).
    Xtr[n, o, D]: cada orientação é uma amostra. Devolve previsões (média nas orientações) para cada X em Xva_list."""
    import torch
    torch.manual_seed(seed)
    n, o, Dm = Xtr.shape
    mu, sd = Xtr.reshape(-1, Dm).mean(0), Xtr.reshape(-1, Dm).std(0) + 1e-6
    tm, ts = Ttr.mean(0), Ttr.std(0) + 1e-6
    alvo = (Ttr - tm) / ts
    na = 0
    if aux is not None:
        am, as_ = np.nanmean(aux, 0), np.nanstd(aux, 0) + 1e-6
        A = np.clip(np.nan_to_num((aux - am) / as_), -5, 5)
        alvo = np.c_[alvo, A]; na = A.shape[1]
    X = torch.from_numpy(((Xtr - mu) / sd).reshape(-1, Dm).astype(np.float32))
    Y = torch.from_numpy(np.repeat(alvo, o, 0).astype(np.float32))
    net = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(Dm, oculto), torch.nn.GELU(), torch.nn.Dropout(drop),
                              torch.nn.Linear(oculto, alvo.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    passos = epocas * (len(X) // 256 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=passos, pct_start=0.1)
    w = torch.ones(alvo.shape[1]); w[Ttr.shape[1]:] = peso_aux
    g = torch.Generator().manual_seed(seed)
    for _ in range(epocas):
        net.train()
        for j in torch.randperm(len(X), generator=g).split(256):
            loss = (torch.nn.functional.smooth_l1_loss(net(X[j]), Y[j], reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    outs = []
    with torch.inference_mode():
        for Xv in Xva_list:
            z = torch.from_numpy(((Xv - mu) / sd).reshape(-1, Dm).astype(np.float32))
            p = net(z).numpy()[:, :Ttr.shape[1]].reshape(Xv.shape[0], Xv.shape[1], -1).mean(1)
            outs.append(p * ts + tm)
    if devolve_modelo:
        return outs, dict(net=net, mu=mu, sd=sd, tm=tm, ts=ts)
    return outs


def etapa_mlp(D, bb):
    F = Feat(bb)
    no = min(8, F.X.shape[1]); o = tuple(range(no))
    bl = ("agregado", "media")
    Xtr = F.pega(D.ids, o, bl)
    z = np.load(os.path.join(TD, "feat", "zen_depth.npz"))
    ix = {k: i for i, k in enumerate(z["ids"])}
    aux = z["X"][[ix[d] for d in D.ids]]
    Xs = {s: F.pega(D.aval[s]["chave"], o, bl) for s in SPLITS}
    for nome, a, kw in [("mlp", None, {}), ("mlp+aux-profundidade", aux, {}),
                        ("mlp wd0.1", None, dict(wd=0.1)), ("mlp oculto1024 drop0.5", None, dict(oculto=1024, drop=0.5))]:
        t0 = time.time()
        oof = np.zeros((len(D.ids), N_ALVOS))
        for k in range(5):
            tr, va = D.fold != k, D.fold == k
            oof[va] = treina_mlp(Xtr[tr], D.T[tr], [Xtr[va]], None if a is None else a[tr], **kw)[0]
        prev = treina_mlp(Xtr, D.T, [Xs[s] for s in SPLITS], a, **kw)
        grava(f"{nome} {bb} o{no}", oof, dict(zip(SPLITS, prev)))
        resumo_cv(D, f"{nome} {bb} o{no}", oof, t0)


# ---------------------------------------------------------------- robustez de escala
def etapa_escala(D, bb="siglip2_b16"):
    """Treinos: (a) quadro inteiro (escala fixa do N5k); (b) + cópias com escala aleatória U[0,7;1,3];
    (c) recorte pelo prato (prato ocupa 90% do lado). Avalia cada um no teste com zoom artificial e,
    no caso (c), com erro de enquadramento. Só orientação original; blocos agregado+média."""
    bl = ("agregado", "media")
    base = Feat(bb)
    Xb = base.pega(D.ids, O1, bl)
    esc = [Feat(f"{bb}_escala{s}") for s in (1, 2)]
    Xe = np.concatenate([Xb] + [f.pega(D.ids, O1, bl) for f in esc], 1)   # original + 2 escalas como amostras
    te = D.aval["teste"]["chave"]
    zooms = [0.7, 0.85, 1.0, 1.15, 1.3]
    Xz = {z: (base if z == 1.0 else Feat(f"{bb}_zoom{z}")).pega(te, O1, bl) for z in zooms}
    for nome, Xtr in (("escala (a) quadro inteiro", Xb), ("escala (b) + escala aleatória", Xe)):
        oof, alfa, mse = ridge_cv(Xtr, D.T, D.fold, Xb)
        pred, _ = ridge_final(Xtr, D.T, alfa)
        grava(nome, oof, {f"teste_zoom{z}": pred(Xz[z]) for z in zooms}, dict(alfa=alfa, cv_mse=mse))
        resumo_cv(D, nome, oof)
    pr = Feat(f"{bb}_prato0.9")
    com = np.array([d in pr.idx for d in D.ids])
    print(f"recorte pelo prato: {com.sum()} de {len(com)} pratos de treino com caixa")
    Xp = pr.pega([d for d, c in zip(D.ids, com) if c], O1, bl)
    oof_c, alfa, mse = ridge_cv(Xp, D.T[com], D.fold[com])
    pred, _ = ridge_final(Xp, D.T[com], alfa)
    oof = np.full((len(D.ids), N_ALVOS), np.nan); oof[com] = oof_c
    tem = np.array([d in pr.idx for d in te])
    prev = {"teste_prato0.9": pred(pr.pega([d for d, c in zip(te, tem) if c], O1, bl)), "teste_tem_caixa": tem}
    for r in (0.1, 0.3):
        f = Feat(f"{bb}_prato0.9_ruido{r}")
        prev[f"teste_prato0.9_ruido{r}"] = pred(f.pega([d for d, c in zip(te, tem) if c], O1, bl))
    # (zoom no quadro inteiro seguido do recorte pelo prato dá o mesmo recorte: invariante por construção;
    #  o que varia é o erro de enquadramento, medido com 'ruido')
    grava("escala (c) recorte pelo prato", oof, prev, dict(alfa=alfa, cv_mse=mse, com=com))
    print(f"{'escala (c) recorte pelo prato':46s} CV: log-massa RMSE {np.sqrt(np.mean((oof_c[:, I_M] - D.T[com, I_M]) ** 2)):.4f}"
          f" | critério ponderado direto {crit(D.Y['proteina'][com], proteinas(oof_c)['direto'], D.w_cv[com]):.1f}%")
    # (d) treino com recorte pelo prato + erro de enquadramento de ±10% como amostras extras
    fr = Feat(f"{bb}_prato0.9_ruido0.1_treino")
    Xd = np.concatenate([Xp, fr.pega([d for d, c in zip(D.ids, com) if c], O1, bl)], 1)
    oof_d, alfa, mse = ridge_cv(Xd, D.T[com], D.fold[com], Xp)
    pred, _ = ridge_final(Xd, D.T[com], alfa)
    oof = np.full((len(D.ids), N_ALVOS), np.nan); oof[com] = oof_d
    prev = {k: pred(Feat(f"{bb}_{k[6:]}").pega([d for d, c in zip(te, tem) if c], O1, bl))
            for k in ("teste_prato0.9", "teste_prato0.9_ruido0.1", "teste_prato0.9_ruido0.3")}
    prev["teste_tem_caixa"] = tem
    grava("escala (d) recorte pelo prato + ruído no treino", oof, prev, dict(alfa=alfa, cv_mse=mse, com=com))


def etapa_escala_d4(D, bb="siglip2_b16"):
    """(e) Escala aleatória + D4: treino = 8 orientações + 2 cópias com escala aleatória (orientação original).
    Entra no banco como configuração normal (previsão = média das 8 orientações) e também é avaliada nos zooms."""
    t0 = time.time()
    bl = ("agregado", "media")
    F = Feat(bb)
    Xtr = np.concatenate([F.pega(D.ids, O8, bl)] + [Feat(f"{bb}_escala{s}").pega(D.ids, O1, bl) for s in (1, 2)], 1)
    oof, alfa, mse = ridge_cv(Xtr, D.T, D.fold, F.pega(D.ids, O8, bl))
    pred, _ = ridge_final(Xtr, D.T, alfa)
    prev = {s: pred(F.pega(D.aval[s]["chave"], O8, bl)) for s in SPLITS}
    nome = f"ridge {bb} o8 + escala aleatória no treino"
    grava(nome, oof, prev, dict(alfa=alfa, cv_mse=mse))
    resumo_cv(D, nome, oof, t0)
    te = D.aval["teste"]["chave"]
    zooms = [0.7, 0.85, 1.0, 1.15, 1.3]
    grava("escala (e) D4 + escala aleatória (zoom com 1 orientação)", oof,
          {f"teste_zoom{z}": pred((F if z == 1.0 else Feat(f"{bb}_zoom{z}")).pega(te, O1, bl)) for z in zooms})


if __name__ == "__main__":
    D = Dados()
    etapa = sys.argv[1]
    if etapa == "ridge":
        etapa_ridge(D, sys.argv[2:])
    elif etapa == "mlp":
        etapa_mlp(D, sys.argv[2])
    elif etapa == "escala":
        etapa_escala(D)
    elif etapa == "lateral":
        etapa_lateral(D)
    elif etapa == "escala_d4":
        etapa_escala_d4(D)
