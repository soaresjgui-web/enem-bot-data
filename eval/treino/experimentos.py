#!/usr/bin/env python3
"""Experimentos A/B/C + curva de aprendizado + transferência para comida brasileira.

Uso: python3 experimentos.py [--rapido]
Lê features de <TD>/feat (extrai_features.py, features_profundidade.py) e grava
<TD>/resultados/*.json e um resumo em texto na saída padrão.

Alvos (log): massa, proteína, kcal e densidade de proteína (g/g). Proteína é prevista
(1) direto e (2) como massa prevista × densidade prevista. Hiperparâmetros escolhidos por
validação cruzada agrupada (GroupKFold por 'cadeia' de pratos montados em sequência) só no treino.
Treino 'oficial' = rgb_train oficial − 450 pratos de avaliação. Treino 'estrito' = também sem
os pratos de treino da mesma cadeia (< 120 s entre pratos consecutivos) de qualquer prato de avaliação.
"""
import csv, json, os, sys, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from comum import TD, PREC, ler_treino, ler_avaliacao, matriz, grupos_sessao, metricas, fmt

EPS_D = 0.01
SPLITS_AVAL = ["teste", "n5k_v1", "calib", "ufpr", "liv_up", "seara"]


def alvos(Y):
    m, p, k = Y[:, 0], Y[:, 1], Y[:, 2]
    return np.c_[np.log(m), np.log1p(p), np.log1p(k), np.log(p / m + EPS_D)]


def proteinas(T):
    """T: previsões em log -> (proteína direta, proteína via massa×densidade, massa, densidade)."""
    massa, dens = np.exp(T[:, 0]), np.clip(np.exp(T[:, 3]) - EPS_D, 0, None)
    return np.expm1(T[:, 1]).clip(0), massa * dens, massa, dens


def ajusta(metodo, Xtr, Ttr, grupos, seed=0, n_raw=0):
    cv = list(GroupKFold(5).split(Xtr, groups=grupos))
    if metodo == "ridge":
        return RidgeMulti(cv).fit(Xtr, Ttr)
    if metodo == "mlp":
        best = None
        for alpha in (1e-2, 1e-1, 1.0, 10.0):
            errs = []
            for tr, va in cv[:3]:
                mm = _mlp(alpha, seed).fit(Xtr[tr], Ttr[tr])
                errs.append(np.mean((mm.predict(Xtr[va]) - Ttr[va]) ** 2))
            if best is None or np.mean(errs) < best[0]:
                best = (np.mean(errs), alpha)
        return _mlp(best[1], seed).fit(Xtr, Ttr)
    if metodo == "lgbm":
        return LGBMMulti(cv, seed, n_raw).fit(Xtr, Ttr)
    raise ValueError(metodo)


def _mlp(alpha, seed):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(256, random_state=seed),
                         MLPRegressor(hidden_layer_sizes=(256,), alpha=alpha, early_stopping=True, max_iter=400,
                                      learning_rate_init=1e-3, random_state=seed))


class RidgeMulti:
    """Ridge multi-alvo; alfa por alvo escolhido por CV agrupada (SVD por fold: todos os alfas de uma vez)."""
    ALFAS = np.logspace(-1, 5, 19)

    def __init__(self, cv):
        self.cv = cv

    @staticmethod
    def _svd_fit(X, T):
        mx, mt = X.mean(0), T.mean(0)
        U, S, Vt = np.linalg.svd(X - mx, full_matrices=False)
        return mx, mt, U, S, Vt

    def fit(self, X, T):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
        Z = self.pre.transform(X)
        err = np.zeros((len(self.ALFAS), T.shape[1]))
        for tr, va in self.cv:
            mx, mt, U, S, Vt = self._svd_fit(Z[tr], T[tr])
            UtT = U.T @ (T[tr] - mt)
            Zv = (Z[va] - mx) @ Vt.T
            for i, a in enumerate(self.ALFAS):
                P = Zv @ ((S / (S ** 2 + a))[:, None] * UtT) + mt
                err[i] += ((P - T[va]) ** 2).sum(0)
        self.alfa = self.ALFAS[err.argmin(0)]
        self.cv_mse = err.min(0) / len(T)  # erro de CV (log) por alvo, usado para escolher o backbone
        mx, mt, U, S, Vt = self._svd_fit(Z, T)
        UtT = U.T @ (T - mt)
        self.W = np.stack([Vt.T @ ((S / (S ** 2 + a)) * UtT[:, j]) for j, a in enumerate(self.alfa)], 1)
        self.mx, self.mt = mx, mt
        return self

    def predict(self, X):
        return (self.pre.transform(X) - self.mx) @ self.W + self.mt


class LGBMMulti:
    """PCA(64) das features de imagem + atributos de profundidade crus; um LightGBM por alvo,
    nº de árvores escolhido por CV agrupada."""
    def __init__(self, cv, seed, n_raw=0):
        self.cv, self.seed, self.n_raw = cv, seed, n_raw

    def _z(self, X, fit=False):
        k = X.shape[1] - self.n_raw
        if k == 0:
            return X
        if fit:
            self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(64, random_state=self.seed)).fit(X[:, :k])
        return np.c_[self.pre.transform(X[:, :k]), X[:, k:]]

    def fit(self, X, T):
        import lightgbm as lgb
        Z = self._z(X, fit=True)
        self.ms = []
        for j in range(T.shape[1]):
            par = dict(learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, random_state=self.seed, n_jobs=1)
            best = None
            for n in (300, 800):
                e = np.mean([np.mean((lgb.LGBMRegressor(n_estimators=n, **par).fit(Z[tr], T[tr, j]).predict(Z[va]) - T[va, j]) ** 2)
                             for tr, va in self.cv[:3]])
                if best is None or e < best[0]:
                    best = (e, n)
            self.ms.append(lgb.LGBMRegressor(n_estimators=best[1], **par).fit(Z, T[:, j]))
        return self

    def predict(self, X):
        Z = self._z(X)
        return np.c_[[m.predict(Z) for m in self.ms]].T


def treino_estrito(ids, aval):
    ev = [d for s in ("calib", "teste", "n5k_v1") for d in aval[s][0]]
    g = grupos_sessao(ids + ev, 120)
    ge = set(g[len(ids):])
    return np.array([x not in ge for x in g[:len(ids)]])


def avalia(nome, prev_por_split, aval, saida):
    linhas = {}
    for s, T in prev_por_split.items():
        Y = aval[s][2]
        pd, pm, massa, _ = proteinas(T)
        linhas[s] = {"direto": metricas(Y[:, 1], pd), "massa_x_dens": metricas(Y[:, 1], pm),
                     "massa": metricas(Y[:, 0], massa)}
        linhas[s]["massa"]["mape"] = float(np.mean(np.abs(massa - Y[:, 0]) / Y[:, 0]) * 100)
    saida[nome] = linhas
    t = linhas.get("teste")
    if t:
        print(f"{nome:42s} teste: direto {fmt(t['direto'])} || m×d {fmt(t['massa_x_dens'])} || massa MAPE {t['massa']['mape']:.0f}%"
              + (f" || v1 {linhas['n5k_v1']['direto']['crit']:.0f}/{linhas['n5k_v1']['massa_x_dens']['crit']:.0f}%" if "n5k_v1" in linhas else ""),
              flush=True)
    return linhas


def llm_por_prato():
    """Massa e densidade de proteína implícitas nas estimativas por componente do LLM."""
    g, p = {}, {}
    with open(os.path.join(PREC, "estimativas_v2_comp_casados.csv")) as f:
        for r in csv.DictReader(f):
            g[r["prato_id"]] = g.get(r["prato_id"], 0) + float(r["gramas_estimadas"] or 0)
            p[r["prato_id"]] = p.get(r["prato_id"], 0) + float(r["proteina_g"] or 0)
    return {k: (g[k], p[k]) for k in g}


def mapa_reduzido():
    """'Embedding' simples do mapa de alturas: média em blocos 5x5 (120x160 -> 24x32), recorte 0–80 mm."""
    z = np.load(os.path.join(TD, "feat", "depth_map.npz"))
    M = np.clip(z["X"].astype(np.float32), 0, 80).reshape(len(z["X"]), 24, 5, 32, 5).mean((2, 4))
    np.savez(os.path.join(TD, "feat", "depth_emb.npz"), ids=z["ids"], X=M.reshape(len(M), -1))


def main(rapido=False):
    os.makedirs(os.path.join(TD, "resultados"), exist_ok=True)
    mapa_reduzido()
    ids, Y = ler_treino()
    T = alvos(Y)
    aval = ler_avaliacao()
    estrito = treino_estrito(ids, aval)
    g_all = grupos_sessao(ids, 120)
    print(f"treino oficial {len(ids)} | estrito {estrito.sum()}")
    res = {}
    img_feats = ["vits16_in21k", "siglip_b16", "siglip2_b16", "effb0", "vits16_in21k_384"]
    img_feats = [f for f in img_feats if os.path.exists(os.path.join(TD, "feat", f + ".npz"))]

    cv_mse = {}

    def roda(nome, feats, metodo, mask=None, splits=SPLITS_AVAL, seed=0):
        mask = np.ones(len(ids), bool) if mask is None else mask
        Xtr = matriz(feats, ids)[mask]
        n_raw = matriz(["depth_hand"], ids[:1]).shape[1] if feats[-1] == "depth_hand" else 0
        m = ajusta(metodo, Xtr, T[mask], g_all[mask], seed, n_raw)
        if hasattr(m, "cv_mse"):
            cv_mse[nome] = m.cv_mse
            res.setdefault("cv_mse_treino", {})[nome] = m.cv_mse.tolist()
        prev = {s: m.predict(matriz(feats, aval[s][0])) for s in splits}
        return avalia(nome, prev, aval, res), prev

    # --- A: embeddings congelados, ridge, treino oficial vs estrito
    print("\n# A — embeddings congelados (RGB)")
    pular_a = os.environ.get("PULAR_A")  # retomar a partir do combo (resultados de A já no log)
    for f in ([] if pular_a else img_feats):
        roda(f"A ridge {f} [oficial]", [f], "ridge")
        roda(f"A ridge {f} [estrito]", [f], "ridge", estrito)
    # escolha do backbone SÓ pelo erro de CV no treino (log proteína), nunca pelo teste
    rng = np.random.default_rng(0)
    sub = np.zeros(len(ids), bool); sub[rng.choice(len(ids), estrito.sum(), replace=False)] = True
    for f in ([] if pular_a else img_feats):
        roda(f"A ridge {f} [oficial subamostrado n={estrito.sum()}]", [f], "ridge", sub)
    if pular_a:
        melhor_img = pular_a
    else:
        melhor_img = min(img_feats, key=lambda f: cv_mse[f"A ridge {f} [estrito]"][1])
        print("CV (MSE log proteína) no treino:", {f: round(float(cv_mse[f"A ridge {f} [estrito]"][1]), 4) for f in img_feats})
    combo = [f for f in ("siglip_b16", "siglip2_b16", "vits16_in21k") if f in img_feats]
    print("melhor backbone:", melhor_img, "| combo:", combo)
    if not pular_a:
        roda(f"A ridge {'+'.join(combo)} [estrito]", combo, "ridge", estrito)
    for met in ([] if pular_a else ("lgbm", "mlp")):
        roda(f"A {met} {melhor_img} [estrito]", [melhor_img], met, estrito)

    # --- B: RGB-D
    print("\n# B — RGB + profundidade")
    nd = ["teste", "n5k_v1", "calib"]
    roda("B ridge só profundidade (atributos)", ["depth_hand"], "ridge", estrito, nd)
    roda("B lgbm só profundidade (atributos)", ["depth_hand"], "lgbm", estrito, nd)
    if os.path.exists(os.path.join(TD, "feat", "depth_emb.npz")):
        roda("B ridge mapa de altura reduzido (24x32)", ["depth_emb"], "ridge", estrito, nd)
        roda(f"B ridge {melhor_img}+mapa+atributos", [melhor_img, "depth_emb", "depth_hand"], "ridge", estrito, nd)
    roda(f"B ridge {melhor_img}+atributos", [melhor_img, "depth_hand"], "ridge", estrito, nd)
    roda(f"B lgbm {melhor_img}+atributos", [melhor_img, "depth_hand"], "lgbm", estrito, nd)
    roda(f"B mlp {melhor_img}+atributos", [melhor_img, "depth_hand"], "mlp", estrito, nd)

    # --- C: híbrido LLM (identificação/densidade) + massa do modelo treinado
    print("\n# C — híbrido: densidade do LLM × massa do modelo")
    llm = llm_por_prato()
    _, prev_img = roda(f"C base: ridge {melhor_img} [estrito]", [melhor_img], "ridge", estrito)
    hib = {}
    for s in SPLITS_AVAL:
        pids, Ys = aval[s][1], aval[s][2]
        pd, pm, massa, dens = proteinas(prev_img[s])
        mg = np.array([llm.get(p, (np.nan, np.nan))[0] for p in pids])
        pl = np.array([llm.get(p, (np.nan, np.nan))[1] for p in pids])
        dl = pl / np.maximum(mg, 1)
        hib[s] = dict(real=Ys[:, 1], llm=pl, mass_mod=massa, mass_llm=mg, dens_llm=dl, dens_mod=dens, direto=pd)
    # pesos do blend em log: ajustados SÓ no calib (120), aplicados no teste/v1/BR
    c = hib["calib"]
    F = lambda h: np.c_[np.log(h["mass_mod"]), np.log(np.maximum(h["mass_llm"], 1)), np.log(h["dens_llm"] + EPS_D),
                        np.log(h["dens_mod"] + EPS_D), np.ones(len(h["real"]))]
    w, *_ = np.linalg.lstsq(F(c), np.log1p(c["real"]), rcond=None)
    res["C pesos do blend (calib)"] = w.tolist()
    for s, h in hib.items():
        variantes = {
            "LLM sozinho (v2, componentes+tabela)": h["llm"],
            "massa modelo × densidade LLM": h["mass_mod"] * h["dens_llm"],
            "massa √(modelo·LLM) × densidade LLM": np.sqrt(h["mass_mod"] * h["mass_llm"]) * h["dens_llm"],
            "massa modelo × √(dens modelo·dens LLM)": h["mass_mod"] * np.sqrt((h["dens_mod"] + EPS_D) * (h["dens_llm"] + EPS_D)),
            "blend log-linear (ajustado no calib)": np.expm1(F(h) @ w).clip(0),
        }
        for k, v in variantes.items():
            ok_ = ~np.isnan(v)
            res.setdefault(f"C {k}", {})[s] = {"direto": metricas(h["real"][ok_], v[ok_])}
    for k in [k for k in res if k.startswith("C ") and "pesos" not in k and "base" not in k]:
        r = res[k]
        print(f"{k:42s} " + " | ".join(f"{s} {r[s]['direto']['crit']:.0f}% (MAE {r[s]['direto']['mae']:.1f})" for s in SPLITS_AVAL if s in r))

    # --- curva de aprendizado (melhor imagem, ridge, estrito; 3 sementes)
    print("\n# Curva de aprendizado")
    idx = np.where(estrito)[0]
    curva = {}
    for frac in (0.1, 0.25, 0.5, 1.0):
        vals = []
        for seed in range(1 if frac == 1.0 else 3):
            rng = np.random.default_rng(seed)
            sub = np.zeros(len(ids), bool)
            sub[rng.choice(idx, int(len(idx) * frac), replace=False)] = True
            mm = ajusta("ridge", matriz([melhor_img], ids)[sub], T[sub], g_all[sub])
            Tp = mm.predict(matriz([melhor_img], aval["teste"][0]))
            pd, pm, _, _ = proteinas(Tp)
            yt = aval["teste"][2][:, 1]
            vals.append([metricas(yt, pd)["crit"], metricas(yt, pm)["crit"], metricas(yt, pd)["mae"]])
        v = np.mean(vals, 0)
        curva[f"{frac:.2f}"] = dict(n=int(len(idx) * frac), direto=v[0], massa_x_dens=v[1], mae_direto=v[2])
        print(f"  {frac:4.0%} n={int(len(idx) * frac):4d}: direto {v[0]:.1f}% | m×d {v[1]:.1f}% | MAE {v[2]:.1f} g")
    res["curva"] = curva
    res["melhor_img"] = melhor_img
    with open(os.path.join(TD, "resultados", "experimentos" + ("_BC" if pular_a else "") + ".json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main("--rapido" in sys.argv)
