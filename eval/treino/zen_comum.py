"""Seção 8 (uma única foto de cima): dados, alvos multitarefa, ridge com orientações D4,
validação cruzada agrupada por cadeia, critério ponderado e bootstrap pareado."""
import csv, os, sys
import numpy as np
from sklearn.model_selection import GroupKFold

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, ".."))
from comum import TD, PREC, ok  # noqa: E402
from grupos_alimentos import GRUPOS  # noqa: E402

EPS_D = 0.01
FAIXAS = np.array([5, 15, 25, 40])  # faixas de proteína do sorteio do v2 (0-5, 5-15, 15-25, 25-40, >40)
BASE = ["massa", "proteina", "kcal", "gordura", "carbo"]
# colunas do alvo em log
I_M, I_P, I_K, I_D, I_F, I_C = 0, 1, 2, 3, 4, 5
I_MG = slice(6, 6 + len(GRUPOS))                       # log1p(massa por grupo)
I_PG = slice(6 + len(GRUPOS), 6 + 2 * len(GRUPOS))     # log1p(proteína por grupo)
N_ALVOS = 6 + 2 * len(GRUPOS)


def alvos(Y):
    """Y: dict de colunas -> matriz de alvos em log [n, N_ALVOS]."""
    m, p = Y["massa"], Y["proteina"]
    return np.c_[np.log(m), np.log1p(p), np.log1p(Y["kcal"]), np.log(p / m + EPS_D), np.log1p(Y["gordura"]),
                 np.log1p(Y["carbo"]), np.log1p(Y["mg"]), np.log1p(Y["pg"])]


def proteinas(T):
    """Previsões em log -> dict com as três rotas de proteína, massa e densidade."""
    massa = np.exp(T[:, I_M]); dens = np.clip(np.exp(T[:, I_D]) - EPS_D, 0, None)
    return dict(direto=np.expm1(T[:, I_P]).clip(0), mxd=massa * dens, grupos=np.expm1(T[:, I_PG]).clip(0).sum(1),
                massa=massa, dens=dens)


def _le(arq):
    with open(arq) as f:
        rows = list(csv.DictReader(f))
    Y = {c: np.array([float(r[c]) for r in rows]) for c in BASE}
    Y["mg"] = np.array([[float(r[f"m_{g}"]) for g in GRUPOS] for r in rows])
    Y["pg"] = np.array([[float(r[f"p_{g}"]) for g in GRUPOS] for r in rows])
    return rows, Y


class Dados:
    """Treino zenital estrito + avaliação (calib, teste, n5k_v1) + itens brasileiros."""

    def __init__(self, td=TD, n_folds=5):
        rows, self.Y = _le(os.path.join(td, "meta", "zen_treino.csv"))
        self.ids = [r["dish_id"] for r in rows]
        self.cadeia = np.array([int(r["cadeia"]) for r in rows])
        self.T = alvos(self.Y)
        self.fold = np.zeros(len(self.ids), int)
        for k, (_, va) in enumerate(GroupKFold(n_folds).split(self.ids, groups=self.cadeia)):
            self.fold[va] = k
        # peso de CV: as 5 faixas de proteína com o mesmo peso total, como no sorteio do calib/teste
        fx = np.digitize(self.Y["proteina"], FAIXAS)
        self.w_cv = (1 / np.bincount(fx, minlength=5))[fx]; self.w_cv /= self.w_cv.sum()
        rows, Y = _le(os.path.join(td, "meta", "zen_aval.csv"))
        self.aval = {}
        for s in ("calib", "teste", "n5k_v1"):
            j = [i for i, r in enumerate(rows) if r["split"] == s]
            self.aval[s] = dict(chave=[rows[i]["dish_id"] for i in j], prato=[rows[i]["prato_id"] for i in j],
                                massa=Y["massa"][j], proteina=Y["proteina"][j],
                                Y={k: v[j] for k, v in Y.items()})
        assert not set(self.ids) & {d for s in self.aval.values() for d in s["chave"]}
        with open(os.path.join(PREC, "gabarito_pratos_v2.csv")) as f:
            for r in csv.DictReader(f):
                if r["origem"] in ("ufpr", "liv_up", "seara"):
                    s = self.aval.setdefault(r["origem"], dict(chave=[], prato=[], massa=[], proteina=[]))
                    s["chave"].append(r["prato_id"]); s["prato"].append(r["prato_id"])
                    s["massa"].append(float(r["peso_porcao_g"])); s["proteina"].append(float(r["proteina_g"]))
        for s in ("ufpr", "liv_up", "seara"):
            self.aval[s]["massa"] = np.array(self.aval[s]["massa"]); self.aval[s]["proteina"] = np.array(self.aval[s]["proteina"])


class Feat:
    """Features zen_<nome>.npz: X[n, orientações, dim]. Blocos: agregado ‖ média ‖ GeM ‖ grade 2×2."""

    def __init__(self, nome, td=TD):
        z = np.load(os.path.join(td, "feat", f"zen_{nome}.npz"))
        self.idx = {k: i for i, k in enumerate(z["ids"])}
        self.X = z["X"]
        self.nome = nome

    def blocos(self, quais):
        D = self.X.shape[-1]
        d = 384 if D in (384 * 7, 384 * 8) else 768
        p = D - 6 * d
        fat = {"agregado": slice(0, p), "media": slice(p, p + d), "gem": slice(p + d, p + 2 * d), "grade": slice(p + 2 * d, D)}
        return np.concatenate([np.arange(D)[fat[q]] for q in quais])

    def pega(self, chaves, orient=(0,), blocos=("agregado", "media")):
        """-> X[n, len(orient), dim] float32."""
        j = [self.idx[k] for k in chaves]
        return self.X[j][:, list(orient)][:, :, self.blocos(blocos)].astype(np.float32)


class Ridge:
    """Ridge multi-alvo com alfa por alvo, resolvido pela decomposição da matriz de Gram
    (serve para N >> D, caso das orientações D4 como amostras). Padroniza as colunas."""
    ALFAS = np.logspace(-1, 5, 19)

    def _decomp(self, X, T):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        mt = T.mean(0)
        s2, V = np.linalg.eigh((Z.T @ Z).astype(np.float64))
        return mu, sd, mt, s2.clip(0), V, V.T @ (Z.T @ (T - mt))

    def caminho(self, X, T, Xv):
        """Previsões em Xv para todos os alfas: [n_alfas, n_v, n_alvos]."""
        mu, sd, mt, s2, V, VtZtT = self._decomp(X, T)
        ZvV = ((Xv - mu) / sd) @ V
        return np.stack([ZvV @ (VtZtT / (s2 + a)[:, None]) + mt for a in self.ALFAS])

    def fit(self, X, T, alfa):
        mu, sd, mt, s2, V, VtZtT = self._decomp(X, T)
        self.mu, self.sd, self.mt = mu, sd, mt
        self.W = np.stack([V @ (VtZtT[:, j] / (s2 + a)) for j, a in enumerate(alfa)], 1)
        self.alfa = alfa
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.W + self.mt


def ridge_cv(Xtr, T, fold, Xte=None):
    """Xtr[n, o_treino, D] (cada orientação é uma amostra de treino); Xte[n, o_teste, D] (média das
    previsões nas orientações; padrão = Xtr). Devolve (previsões fora-do-fold [n, alvos], alfa por alvo, MSE de CV)."""
    Xte = Xtr if Xte is None else Xte
    n, o, D = Xtr.shape
    ot = Xte.shape[1]
    r = Ridge()
    P = np.zeros((len(r.ALFAS), n, T.shape[1]))
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        cam = r.caminho(Xtr[tr].reshape(-1, D), np.repeat(T[tr], o, 0), Xte[va].reshape(-1, D))
        P[:, va] = cam.reshape(len(r.ALFAS), va.sum(), ot, -1).mean(2)
    mse = ((P - T) ** 2).mean(1)            # [alfas, alvos]
    best = mse.argmin(0)
    oof = np.stack([P[best[j], :, j] for j in range(T.shape[1])], 1)
    return oof, r.ALFAS[best], mse.min(0)


def ridge_final(Xtr, T, alfa):
    n, o, D = Xtr.shape
    m = Ridge().fit(Xtr.reshape(-1, D), np.repeat(T, o, 0), alfa)
    return lambda X: m.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], X.shape[1], -1).mean(1), m


def crit(real, est, w=None):
    a = ok(real, est).astype(float)
    return 100 * (a.mean() if w is None else (a * w).sum() / w.sum())


def mape(real, est):
    return 100 * float(np.mean(np.abs(est - real) / real))


def bootstrap(a, b, n=4000, seed=0):
    """a, b: vetores de acerto (bool) por prato. Diferença b−a em pontos e IC 95% pareado."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(a), (n, len(a)))
    d = 100 * (b[ix].mean(1) - a[ix].mean(1))
    return 100 * (b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ic_simples(a, n=4000, seed=0):
    a = np.asarray(a, float)
    rng = np.random.default_rng(seed)
    d = 100 * a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def llm_componentes():
    """prato_id -> lista de (grupo, gramas, proteína) estimados pelo LLM v2 (por componente, casados com tabela)."""
    out = {}
    with open(os.path.join(PREC, "estimativas_v2_comp_casados.csv")) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["prato_id"], []).append((r["grupo"], float(r["gramas_estimadas"] or 0), float(r["proteina_g"] or 0),
                                                      r["confianca"]))
    return out
