#!/usr/bin/env python3
"""Seção 9, passo 2: cabeças de DENSIDADE de proteína sobre as features congeladas da seção 8.

Uso: OMP_NUM_THREADS=2 python3 dens_experimentos.py <etapa> [backbone ...]
  ridge   ridge com peso por faixa; ridge de fração de massa por fonte; ridge de contribuição por ingrediente
  knn     recuperação: k vizinhos mais próximos no embedding (densidade e fração por fonte dos vizinhos);
          'a|b' concatena os embeddings de dois backbones
  soft    cabeças softmax (linear e MLP) de fração de massa por fonte / por ingrediente / por classe fina, multitarefa,
          com e sem perda direta de densidade, com e sem peso por faixa
Backbones: siglip2_b16 (224 px, 8 orientações), siglip2_b16_512 (384×512, 2 orientações), tips_b14 (336×448, 1 orientação),
  'ambos' (siglip2_b16 + siglip2_b16_512 concatenados, 2 orientações);
  o sufixo '+esp' soma os blocos espaciais (GeM e grade 2×2 dos tokens) ao agregado e à média.

Cada configuração grava <TD>/resultados/dens/banco/<nome>.npz com
  oof_dens [n_treino] (fora-do-fold, mesmos folds por cadeia da seção 8) e <split>_dens para calib/teste/n5k_v1;
  quando há: oof (matriz de alvos em log, como no zen_banco), oof_ff / <split>_ff (fração de massa por fonte).
Nada aqui lê o gabarito do teste: as previsões do teste só são avaliadas em dens_avalia.py.
Alvos por ingrediente: só dos pratos de treino (dens_comum.Ingr recusa ids do teste).
"""
import os, sys, time
import numpy as np
from zen_comum import Dados, Feat, Ridge, ridge_cv, ridge_final, proteinas, crit, EPS_D, I_M, I_D, N_ALVOS
from dens_comum import Ingr, FONTES, BANCO_D

SPLITS = ["calib", "teste", "n5k_v1"]
BL = ("agregado", "media")


def grava(nome, **k):
    os.makedirs(BANCO_D, exist_ok=True)
    np.savez(os.path.join(BANCO_D, nome + ".npz"), **k)


def resumo(D, nome, dens, t0=None, extra=""):
    yp, ym = D.Y["proteina"], D.Y["massa"]
    rd = yp / ym
    e = np.log(dens + EPS_D) - np.log(rd + EPS_D)
    a = yp > 15
    print(f"{nome:58s} CV: RMSE log-dens {np.sqrt((e ** 2).mean()):.4f} (>15 g: {np.sqrt((e[a] ** 2).mean()):.4f}) | "
          f"critério ponderado massa real × dens {crit(yp, ym * dens, D.w_cv):4.1f}% (>15 g: {crit(yp[a], (ym * dens)[a], D.w_cv[a]):4.1f}%)"
          f" {extra}" + (f" ({time.time() - t0:.0f}s)" if t0 else ""), flush=True)


class Dad:
    """Dados + alvos por ingrediente + features de um backbone."""

    def __init__(self, bb):
        self.D = D = Dados()
        self.I = I = Ingr(D.ids)
        # vocabulário de ingredientes: os com >= 5 pratos de treino; o resto vira a classe 'outros'
        n_pratos = (I.M > 0).sum(0)
        self.keep = keep = n_pratos >= 5
        resto = I.M[:, ~keep].sum(1)
        self.Fi = np.c_[I.M[:, keep], resto]; self.Fi /= np.maximum(self.Fi.sum(1, keepdims=True), 1e-9)
        d_resto = (I.M[:, ~keep] @ I.dens_ingr[~keep]).sum() / max(resto.sum(), 1e-9)
        self.dens_i = np.r_[I.dens_ingr[keep], d_resto]
        self.fonte_i = np.r_[I.fonte_ingr[keep], len(FONTES) - 1]
        self.Ff = I.frac_fonte
        # classes "finas": cada ingrediente com >= 0,8% da proteína do treino é uma classe (chicken, chicken thighs, pork,
        # salmon, fish, bacon, scrambled eggs...); os demais ficam agrupados pela fonte
        prot_i = (I.M * I.dens_ingr).sum(0)
        proprio = np.where(prot_i / prot_i.sum() >= 0.008)[0]
        cls = np.full(len(I.vocab), -1); cls[proprio] = np.arange(len(proprio))
        cls[cls < 0] = len(proprio) + I.fonte_ingr[cls < 0]
        K = len(proprio) + len(FONTES)
        self.MX = np.stack([I.M[:, cls == k].sum(1) for k in range(K)], 1)
        self.PX = np.stack([(I.M[:, cls == k] * I.dens_ingr[cls == k]).sum(1) for k in range(K)], 1)
        self.Fx = self.MX / np.maximum(self.MX.sum(1, keepdims=True), 1e-9)
        self.fonte_x = np.r_[I.fonte_ingr[proprio], np.arange(len(FONTES))]
        self.nomes_x = [I.vocab[i] for i in proprio] + [f"outros:{f}" for f in FONTES]
        self.bb = bb
        bl = BL + ("gem", "grade") if bb.endswith("+esp") else BL
        base = bb[:-4] if bb.endswith("+esp") else bb
        if base == "ambos" or "|" in base:
            fs = [Feat(b) for b in (["siglip2_b16", "siglip2_b16_512"] if base == "ambos" else base.split("|"))]
            self.o = tuple(range(min(2, min(f.X.shape[1] for f in fs))))
            self.pega = lambda ch: np.concatenate([f.pega(ch, self.o, bl) for f in fs], -1)
        else:
            f = Feat(base); self.o = tuple(range(f.X.shape[1]))
            self.pega = lambda ch: f.pega(ch, self.o, bl)
        self.X = self.pega(D.ids)
        self.Xs = {s: self.pega(D.aval[s]["chave"]) for s in SPLITS}
        self.rd = D.Y["proteina"] / D.Y["massa"]

    def dens_fina(self, idx):
        return self.PX[idx].sum(0) / np.maximum(self.MX[idx].sum(0), 1e-9)

    def dens_fonte(self, idx):
        """Densidade média de cada fonte (g/g) nos pratos idx do treino."""
        return self.I.PF[idx].sum(0) / np.maximum(self.I.MF[idx].sum(0), 1e-9)


# ---------------------------------------------------------------- ridge
def ridge_alvos(A, T, w=None):
    """Ridge multi-alvo com CV agrupada e peso por amostra opcional. Devolve (oof, previsões por split)."""
    D, X = A.D, A.X
    n, o, Dm = X.shape
    if w is None:
        oof, alfa, _ = ridge_cv(X, T, D.fold)
        pred, _ = ridge_final(X, T, alfa)
        return oof, {s: pred(A.Xs[s]) for s in SPLITS}
    r = Ridge()

    def prep(tr):
        ww = np.repeat(w[tr] / w[tr].mean(), o)
        Xt = X[tr].reshape(-1, Dm); Tt = np.repeat(T[tr], o, 0)
        mu = (Xt * ww[:, None]).sum(0) / ww.sum(); sd = np.sqrt((((Xt - mu) ** 2) * ww[:, None]).sum(0) / ww.sum()) + 1e-6
        mt = (Tt * ww[:, None]).sum(0) / ww.sum()
        Z = (Xt - mu) / sd * np.sqrt(ww)[:, None]
        s2, V = np.linalg.eigh((Z.T @ Z).astype(np.float64))
        return mu, sd, mt, s2.clip(0), V, V.T @ (Z.T @ ((Tt - mt) * np.sqrt(ww)[:, None]))

    def aplica(par, Xv, alfas):
        mu, sd, mt, s2, V, VtZtT = par
        ZvV = ((Xv.reshape(-1, Dm) - mu) / sd) @ V
        return np.stack([(ZvV @ (VtZtT / (s2 + a)[:, None]) + mt).reshape(Xv.shape[0], Xv.shape[1], -1).mean(1) for a in alfas])

    P = np.zeros((len(r.ALFAS), n, T.shape[1]))
    for k in range(5):
        tr, va = D.fold != k, D.fold == k
        P[:, va] = aplica(prep(tr), X[va], r.ALFAS)
    mse = (((P - T) ** 2) * w[None, :, None]).sum(1) / w.sum(); best = mse.argmin(0)
    oof = np.stack([P[best[j], :, j] for j in range(T.shape[1])], 1)
    par = prep(np.ones(n, bool))
    prev = {}
    for s in SPLITS:
        Pa = aplica(par, A.Xs[s], r.ALFAS)
        prev[s] = np.stack([Pa[best[j], :, j] for j in range(T.shape[1])], 1)
    return oof, prev


def etapa_ridge(bb):
    A = Dad(bb); D = A.D
    ld = np.log(A.rd + EPS_D)[:, None]
    un = lambda x: np.clip(np.exp(x[:, 0]) - EPS_D, 0, None)
    # (0) referência: ridge de log-densidade sem peso (igual ao zen_banco)
    t0 = time.time(); oof, prev = ridge_alvos(A, ld)
    grava(f"ridge logdens {bb}", oof_dens=un(oof), **{f"{s}_dens": un(prev[s]) for s in SPLITS}); resumo(D, f"ridge logdens {bb}", un(oof), t0)
    # (1) peso por faixa de proteína (gama = 1: as 5 faixas com o mesmo peso; 0,5: meio caminho)
    for gama in (0.5, 1.0):
        t0 = time.time(); w = D.w_cv ** gama
        oof, prev = ridge_alvos(A, ld, w)
        nome = f"ridge logdens peso-faixa{gama} {bb}"
        grava(nome, oof_dens=un(oof), **{f"{s}_dens": un(prev[s]) for s in SPLITS}); resumo(D, nome, un(oof), t0)
    # (2) fração de massa por fonte (linear) -> Σ fração × densidade média da fonte (do treino do fold)
    t0 = time.time(); oof, prev = ridge_alvos(A, A.Ff)
    norm = lambda F: np.clip(F, 0, None) / np.maximum(np.clip(F, 0, None).sum(1, keepdims=True), 1e-9)
    d_oof = np.zeros(len(D.ids))
    for k in range(5):
        d_oof[D.fold == k] = norm(oof[D.fold == k]) @ A.dens_fonte(D.fold != k)
    df = A.dens_fonte(np.ones(len(D.ids), bool))
    grava(f"ridge fração por fonte {bb}", oof_dens=d_oof, oof_ff=norm(oof), **{f"{s}_dens": norm(prev[s]) @ df for s in SPLITS},
          **{f"{s}_ff": norm(prev[s]) for s in SPLITS})
    resumo(D, f"ridge fração por fonte {bb}", d_oof, t0)
    # (3) contribuição de proteína por ingrediente (fração × densidade do ingrediente; soma = densidade do prato)
    t0 = time.time(); C = A.Fi * A.dens_i
    oof, prev = ridge_alvos(A, C)
    s_ = lambda P: np.clip(P, 0, None).sum(1)
    grava(f"ridge contribuição por ingrediente {bb}", oof_dens=s_(oof), **{f"{s}_dens": s_(prev[s]) for s in SPLITS})
    resumo(D, f"ridge contribuição por ingrediente {bb}", s_(oof), t0)


# ---------------------------------------------------------------- k vizinhos
def etapa_knn(bb):
    """bb pode ser 'a|b': concatena os embeddings (centrados e normalizados) de dois backbones."""
    partes = bb.split("|")
    As = [Dad(p) for p in partes]
    A = As[0]; D = A.D
    d = 768
    nrm = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)

    def emb(pega):      # embedding agregado (média das orientações), centrado no treino e normalizado, por backbone
        E = []
        for a in As:
            e = nrm(a.X[:, :, :d].mean(1).astype(np.float64)); mu = e.mean(0)
            E.append(nrm(nrm(pega(a)[:, :, :d].mean(1).astype(np.float64)) - mu))
        return nrm(np.concatenate(E, 1))
    E = emb(lambda a: a.X); Es = {s: emb(lambda a, s=s: a.Xs[s]) for s in SPLITS}
    ld = np.log(A.rd + EPS_D)

    def prev(Et, ytr, Fftr, Ev, k, tau):
        S = Ev @ Et.T
        ix = np.argsort(-S, 1)[:, :k]
        w = np.exp((np.take_along_axis(S, ix, 1) - S.max(1, keepdims=True)) / tau); w /= w.sum(1, keepdims=True)
        return (w * ytr[ix]).sum(1), (w[:, :, None] * Fftr[ix]).sum(1)

    melhor = None
    for k in (5, 10, 20, 40, 80):
        for tau in (0.02, 0.05, 0.1, 0.2):
            t0 = time.time()
            o = np.zeros(len(ld)); of = np.zeros_like(A.Ff)
            for f in range(5):
                tr, va = D.fold != f, D.fold == f
                o[va], of[va] = prev(E[tr], ld[tr], A.Ff[tr], E[va], k, tau)
            rm = np.sqrt((((o - ld) ** 2) * D.w_cv).sum() / D.w_cv.sum())
            if melhor is None or rm < melhor[0]:
                melhor = (rm, k, tau, o, of)
    rm, k, tau, o, of = melhor
    un = lambda x: np.clip(np.exp(x) - EPS_D, 0, None)
    ps = {s: prev(E, ld, A.Ff, Es[s], k, tau) for s in SPLITS}
    nome = f"knn densidade {bb}"
    grava(nome, oof_dens=un(o), oof_ff=of, k=k, tau=tau, **{f"{s}_dens": un(ps[s][0]) for s in SPLITS}, **{f"{s}_ff": ps[s][1] for s in SPLITS})
    resumo(D, nome, un(o), extra=f"[k={k}, tau={tau}]")
    # fração por fonte dos vizinhos × densidade média da fonte
    d_oof = np.zeros(len(ld))
    for f in range(5):
        d_oof[D.fold == f] = of[D.fold == f] @ A.dens_fonte(D.fold != f)
    df = A.dens_fonte(np.ones(len(ld), bool))
    grava(f"knn fração por fonte {bb}", oof_dens=d_oof, **{f"{s}_dens": ps[s][1] @ df for s in SPLITS})
    resumo(D, f"knn fração por fonte {bb}", d_oof)


# ---------------------------------------------------------------- cabeças softmax
def treina_soft(Xtr, Ttr, Ftr, dens_cls, Xva_list, w=None, oculto=512, wd=1e-2, epocas=60, drop=0.3, l_ce=1.0, l_d=0.0, l_reg=1.0, seed=0,
                perda_frac="ce"):
    """Cabeça (linear se oculto=0, senão MLP de 1 camada) com duas saídas: alvos em log padronizados (multitarefa,
    como no MLP da seção 8) e logits de fração de massa por classe (entropia cruzada com alvo contínuo).
    l_d: peso de uma perda direta em log da densidade composta Σ p_i × dens_i.
    Devolve, para cada X em Xva_list: (alvos em log [n, N_ALVOS], frações [n, K]) — média nas orientações."""
    import torch
    torch.manual_seed(seed)
    n, o, Dm = Xtr.shape
    K = Ftr.shape[1]
    mu, sd = Xtr.reshape(-1, Dm).mean(0), Xtr.reshape(-1, Dm).std(0) + 1e-6
    tm, ts = Ttr.mean(0), Ttr.std(0) + 1e-6
    X = torch.from_numpy(((Xtr - mu) / sd).reshape(-1, Dm).astype(np.float32))
    Y = torch.from_numpy(np.repeat((Ttr - tm) / ts, o, 0).astype(np.float32))
    Fr = torch.from_numpy(np.repeat(Ftr, o, 0).astype(np.float32))
    W = torch.from_numpy(np.repeat((np.ones(n) if w is None else w / w.mean()), o).astype(np.float32))
    dc = torch.from_numpy(dens_cls.astype(np.float32))
    pc = 0.25 + dc / dc.mean()
    n_out = Ttr.shape[1] + K
    if oculto:
        net = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(Dm, oculto), torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(oculto, n_out))
    else:
        net = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(Dm, n_out))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    passos = epocas * (len(X) // 256 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=passos, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    nt = Ttr.shape[1]
    sl1 = torch.nn.functional.smooth_l1_loss
    for _ in range(epocas):
        net.train()
        for j in torch.randperm(len(X), generator=g).split(256):
            out = net(X[j]); wj = W[j]
            loss = l_reg * (sl1(out[:, :nt], Y[j], reduction="none").mean(1) * wj).mean()
            lp = torch.log_softmax(out[:, nt:], 1)
            if perda_frac == "ce":
                loss = loss + l_ce * (-(Fr[j] * lp).sum(1) * wj).mean()
            elif perda_frac == "l2":   # erro quadrático nas frações (escala ×10 para ficar na ordem das outras perdas)
                loss = loss + l_ce * 10 * (((lp.exp() - Fr[j]) ** 2).sum(1) * wj).mean()
            elif perda_frac == "l1":
                loss = loss + l_ce * 3 * (((lp.exp() - Fr[j]).abs()).sum(1) * wj).mean()
            else:   # "l2d": erro quadrático ponderado pela densidade da classe (erro numa fonte rica em proteína pesa mais)
                loss = loss + l_ce * 10 * ((((lp.exp() - Fr[j]) ** 2) * pc).sum(1) * wj).mean()
            if l_d:
                d = lp.exp() @ dc
                alvo = Y[j][:, I_D] * float(ts[I_D]) + float(tm[I_D])
                loss = loss + l_d * (sl1((torch.log(d + EPS_D) - alvo) / float(ts[I_D]), torch.zeros_like(alvo), reduction="none") * wj).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    outs = []
    with torch.inference_mode():
        for Xv in Xva_list:
            z = net(torch.from_numpy(((Xv - mu) / sd).reshape(-1, Dm).astype(np.float32)))
            T = (z[:, :nt].numpy().reshape(Xv.shape[0], Xv.shape[1], -1).mean(1)) * ts + tm
            F = torch.softmax(z[:, nt:], 1).numpy().reshape(Xv.shape[0], Xv.shape[1], -1).mean(1)
            outs.append((T, F))
    return outs


CFG_SOFT = {
    # nome: (classes, argumentos)
    "soft-linear fonte": ("fonte", dict(oculto=0, wd=1e-1, drop=0.0)),
    "soft-linear ingrediente": ("ingr", dict(oculto=0, wd=1e-1, drop=0.0)),
    "soft-mlp fonte": ("fonte", {}),
    "soft-mlp ingrediente": ("ingr", {}),
    "soft-mlp ingrediente +perda-dens": ("ingr", dict(l_d=1.0)),
    "soft-mlp ingrediente +perda-dens peso-faixa": ("ingr", dict(l_d=1.0, peso=1.0)),
    "soft-mlp ingrediente +perda-dens peso-faixa0.5": ("ingr", dict(l_d=1.0, peso=0.5)),
    "soft-mlp fonte +perda-dens": ("fonte", dict(l_d=1.0)),
    "soft-mlp ingrediente +perda-dens oculto1024 drop0.5": ("ingr", dict(l_d=1.0, oculto=1024, drop=0.5)),
    "soft-mlp ingrediente só-frações +perda-dens": ("ingr", dict(l_d=1.0, l_reg=0.0)),
    "soft-mlp fonte-l2": ("fonte", dict(perda_frac="l2")),
    "soft-mlp fonte-l2 +perda-dens peso-faixa": ("fonte", dict(perda_frac="l2", l_d=1.0, peso=1.0)),
    "soft-mlp fonte +perda-dens peso-faixa": ("fonte", dict(l_d=1.0, peso=1.0)),
    "soft-mlp fonte peso-faixa": ("fonte", dict(peso=1.0)),
    "soft-mlp fonte-l2 peso-faixa": ("fonte", dict(perda_frac="l2", peso=1.0)),
    "soft-mlp fonte-l2d": ("fonte", dict(perda_frac="l2d")),
    "soft-mlp fonte-l1": ("fonte", dict(perda_frac="l1")),
    "soft-mlp ingrediente-l2": ("ingr", dict(perda_frac="l2")),
    "soft-mlp fonte-l2 oculto1024 drop0.5": ("fonte", dict(perda_frac="l2", oculto=1024, drop=0.5)),
    "soft-mlp fonte-l2 semente1": ("fonte", dict(perda_frac="l2", seed=1)),
    "soft-mlp fonte-l2 semente2": ("fonte", dict(perda_frac="l2", seed=2)),
    "soft-mlp fonte-l2 só-frações": ("fonte", dict(perda_frac="l2", l_reg=0.0)),
    "soft-mlp fina-l2 +perda-dens peso-faixa": ("fina", dict(perda_frac="l2", l_d=1.0, peso=1.0)),
    "soft-mlp fina-l2": ("fina", dict(perda_frac="l2")),
    "soft-mlp fonte-l2 +perda-dens peso-faixa semente1": ("fonte", dict(perda_frac="l2", l_d=1.0, peso=1.0, seed=1)),
    "soft-mlp fonte-l2 +perda-dens peso-faixa semente2": ("fonte", dict(perda_frac="l2", l_d=1.0, peso=1.0, seed=2)),
    "soft-mlp fonte-l2 +perda-dens2 peso-faixa": ("fonte", dict(perda_frac="l2", l_d=2.0, peso=1.0)),
    "soft-mlp fonte-l2 +perda-dens peso-faixa 100 épocas": ("fonte", dict(perda_frac="l2", l_d=1.0, peso=1.0, epocas=100)),
}


def etapa_soft(bb, quais=None):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    A = Dad(bb); D = A.D
    for nome, (cls, kw) in CFG_SOFT.items():
        if quais and not any(q == nome for q in quais):
            continue
        nome_b = f"{nome} {bb}"
        if os.path.exists(os.path.join(BANCO_D, nome_b + ".npz")):
            continue
        t0 = time.time(); kw = dict(kw)
        peso = kw.pop("peso", None)
        w = None if peso is None else D.w_cv ** peso
        Ft = {"fonte": A.Ff, "ingr": A.Fi, "fina": A.Fx}[cls]
        n = len(D.ids)
        oof = np.zeros((n, N_ALVOS)); oof_f = np.zeros((n, Ft.shape[1])); oof_d = np.zeros(n)
        todos = np.ones(n, bool)
        dens_de = lambda tr: A.dens_fonte(tr) if cls == "fonte" else (A.dens_fina(tr) if cls == "fina" else A.dens_i)
        for k in range(5):
            tr, va = D.fold != k, D.fold == k
            dc = dens_de(tr)
            T, F = treina_soft(A.X[tr], D.T[tr], Ft[tr], dc, [A.X[va]], None if w is None else w[tr], **kw)[0]
            oof[va], oof_f[va], oof_d[va] = T, F, F @ dc
        dc = dens_de(todos)
        prev = treina_soft(A.X, D.T, Ft, dc, [A.Xs[s] for s in SPLITS], w, **kw)
        f_cls = A.fonte_x if cls == "fina" else A.fonte_i
        ff = (lambda F: F) if cls == "fonte" else (lambda F: np.stack([F[:, f_cls == j].sum(1) for j in range(len(FONTES))], 1))
        grava(nome_b, oof=oof, oof_dens=oof_d, oof_ff=ff(oof_f), **{s: p[0] for s, p in zip(SPLITS, prev)},
              **{f"{s}_dens": p[1] @ dc for s, p in zip(SPLITS, prev)}, **{f"{s}_ff": ff(p[1]) for s, p in zip(SPLITS, prev)})
        P = proteinas(oof)
        # acerto da fonte principal prevista (a de maior fração entre as fontes de verdade), em pratos com fonte dominante
        tem = A.I.principal < len(FONTES) - 1
        ac = (ff(oof_f)[tem][:, :-1].argmax(1) == A.I.principal[tem]).mean()
        resumo(D, nome_b, oof_d, t0, extra=f"| cabeça log-dens: {crit(D.Y['proteina'], D.Y['massa'] * P['dens'], D.w_cv):4.1f}% | fonte principal certa {100 * ac:.0f}%")


if __name__ == "__main__":
    etapa = sys.argv[1]
    for bb in sys.argv[2:] or ["siglip2_b16"]:
        if etapa == "ridge":
            etapa_ridge(bb)
        elif etapa == "knn":
            etapa_knn(bb)
        elif etapa == "soft":
            etapa_soft(bb, os.environ.get("DENS_CFG", "").split(";") if os.environ.get("DENS_CFG") else None)
