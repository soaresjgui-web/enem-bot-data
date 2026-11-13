"""Funções comuns: dados, features, critério de precisão e grupos para validação cruzada."""
import csv, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PREC = os.path.join(REPO, "docs", "dados", "precisao")
TD = "/tmp/claude-0/-home-user-enem-bot-data/59d3cb2b-a793-5c18-8c23-e9dab2604443/scratchpad/treino"


def ok(real, est):
    """Critério do produto: dentro de ±5 g OU ±20%."""
    real, est = np.asarray(real, float), np.asarray(est, float)
    err = np.abs(est - real)
    return (err <= 5) | (err <= 0.20 * np.maximum(real, 1e-9))


def metricas(real, est):
    real, est = np.asarray(real, float), np.asarray(est, float)
    return dict(n=len(real), crit=100 * ok(real, est).mean(), mae=np.abs(est - real).mean(),
                vies=(est - real).mean(), vies_pct=100 * (np.median(est / np.maximum(real, 0.5)) - 1))


def fmt(m):
    return f"{m['crit']:5.1f}% | MAE {m['mae']:5.1f} g | viés {m['vies']:+5.1f} g"


def ler_treino():
    with open(os.path.join(TD, "meta", "treino.csv")) as f:
        rows = list(csv.DictReader(f))
    ids = [r["dish_id"] for r in rows]
    Y = np.array([[float(r["massa"]), float(r["proteina"]), float(r["kcal"])] for r in rows])
    return ids, Y


def ler_avaliacao():
    """Retorna dict split -> (chaves_de_feature, prato_ids, Y[massa, proteína])."""
    out = {}
    with open(os.path.join(TD, "meta", "avaliacao.csv")) as f:
        for r in csv.DictReader(f):
            s = out.setdefault(r["split"], ([], [], []))
            s[0].append(r["dish_id"]); s[1].append(r["prato_id"]); s[2].append([float(r["massa"]), float(r["proteina"])])
    with open(os.path.join(PREC, "gabarito_pratos_v2.csv")) as f:
        for r in csv.DictReader(f):
            if r["origem"] in ("ufpr", "liv_up", "seara"):
                s = out.setdefault(r["origem"], ([], [], []))
                s[0].append(r["prato_id"]); s[1].append(r["prato_id"])
                s[2].append([float(r["peso_porcao_g"]), float(r["proteina_g"])])
    return {k: (a, b, np.array(c)) for k, (a, b, c) in out.items()}


def carrega_feat(nome):
    z = np.load(os.path.join(TD, "feat", f"{nome}.npz"))
    X = z["X"].astype(np.float32)
    if X.ndim > 2:
        X = X.reshape(len(X), -1)
    return {k: i for i, k in enumerate(z["ids"])}, X


def matriz(nomes, chaves):
    """Concatena features de vários arquivos para as chaves pedidas (NaN se ausente)."""
    blocos = []
    for n in nomes:
        idx, X = carrega_feat(n)
        B = np.full((len(chaves), X.shape[1]), np.nan, np.float32)
        for i, k in enumerate(chaves):
            if k in idx:
                B[i] = X[idx[k]]
        blocos.append(B)
    return np.concatenate(blocos, 1)


def grupos_sessao(dish_ids, gap_s=900):
    """Pratos do Nutrition5k são montados em sequência (ingrediente a ingrediente, mesmo prato).
    dish_id = timestamp; agrupa pratos separados por < gap_s segundos -> folds sem vazamento."""
    ts = np.array([int(d.split("_")[1]) for d in dish_ids])
    ordem = np.argsort(ts)
    g = np.zeros(len(ts), int)
    cur = 0
    for a, b in zip(ordem[:-1], ordem[1:]):
        g[b] = cur = cur + (ts[b] - ts[a] > gap_s)
    return g
