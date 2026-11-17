#!/usr/bin/env python3
"""Seção 18.5–18.6: parâmetros e candidatos do pipeline geométrico (G1, G2, G3, G4, G-prof). Só o treino do Nutrition5k
calibra; nenhum rótulo de teste entra. É biblioteca de avalia_geom.py e também roda sozinho:

    python3 -I calibra.py --alturas <dir_treino> <fotos_geom.csv> <itens_geom.csv.gz> <escala_n5k.csv>   # h por grupo (sem grama)

Unidades: área em cm², altura em cm, densidade aparente em g/cm³, massa em g, proteína em g.
"""
import csv, os, sys
import numpy as np
from scipy.optimize import nnls

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "treino"))
from escala import le_tabela, rle_decodifica, nums, profundidade_cm, PRIORS, LISTA_N5K  # noqa: E402

GRUPOS = ["proteina_animal", "leguminosa", "laticinio", "amido", "vegetal", "gordura_molho", "misto", "outros"]
VOC = {r["termo"]: r for r in csv.DictReader(open(os.path.join(AQUI, "vocabulario.csv")))}


def densidades():
    """Densidade aparente por termo (priors.csv); 'mediana' = mediana das densidades da tabela."""
    d = {r["chave"]: float(r["valor"]) for r in csv.DictReader(open(PRIORS)) if r["tipo"] == "densidade"}
    med = float(np.median(list(d.values())))
    return {t: (med if v["chave_densidade"] == "mediana" else d[v["chave_densidade"]]) for t, v in VOC.items()}


def alturas_prior():
    return {r["chave"]: float(r["valor"]) for r in csv.DictReader(open(PRIORS)) if r["tipo"] == "altura" and r["valor"] != ""}


def le_n5k_grupos():
    """Rótulos do treino do Nutrition5k: massa, proteína e, por grupo, massa (m_*) e proteína (p_*)."""
    L = list(csv.DictReader(open(LISTA_N5K)))
    f = lambda c: np.array([float(r[c]) for r in L])
    return dict(ids=np.array([r["dish_id"] for r in L]), cadeia=np.array([int(r["cadeia"]) for r in L]), massa=f("massa"), proteina=f("proteina"),
                Mg=np.c_[tuple(f("m_" + g) for g in GRUPOS)], Pg=np.c_[tuple(f("p_" + g) for g in GRUPOS)])


def le_escala_n5k(caminho):
    return {r["dish_id"]: float(r["cm_por_px"]) for r in csv.DictReader(open(caminho))}


def itens_por_foto(itens, base):
    out = {}
    for it in itens:
        if it["base"] == base and it["termo"] in VOC:
            out.setdefault(it["id"], []).append(it)
    return out


def areas(ids, itens_foto, cm_px):
    """A[i, g] = área (cm²) do grupo g na foto i; e a lista de itens (termo, grupo, área em cm²) por foto. Foto sem
    escala fica com NaN (não entra na conta)."""
    A = np.zeros((len(ids), len(GRUPOS)))
    por_item = []
    for i, d in enumerate(ids):
        its = []
        c = cm_px.get(d)
        for it in itens_foto.get(d, []):
            g = VOC[it["termo"]]["grupo"]
            a = float(it["area_px"]) * (c * c if c else np.nan)
            A[i, GRUPOS.index(g)] += a
            its.append((it["termo"], g, a, it))
        if c is None:
            A[i] = np.nan
        por_item.append(its)
    return A, por_item


def alturas_n5k(td, ids, itens_foto, cm_px, fotos=None):
    """Por item do Nutrition5k: altura mediana da comida acima do plano do prato (cm) e volume integrado (cm³), pela
    profundidade local. Por grupo: a mediana das alturas dos itens (pesada pela área). Não usa grama nenhuma."""
    from escala import caixa_prato_n5k, plano_prato
    H, vol = {g: [] for g in GRUPOS}, {}
    for d in ids:
        its = itens_foto.get(d, [])
        if not its or d not in cm_px:
            continue
        dep = profundidade_cm(td, d)
        cx = caixa_prato_n5k(td, d, fotos)
        if cx is None:
            continue
        plano = plano_prato(dep, cx)
        alt = np.clip(plano - dep, 0, 15)                      # cm acima do plano do prato; NaN onde a profundidade falta
        px = cm_px[d] ** 2
        for it in its:
            m = rle_decodifica(it["rle"], int(it["mascara_altura"]), int(it["mascara_largura"]))
            if m.shape != dep.shape:
                continue
            v = alt[m]
            v = v[np.isfinite(v)]
            if len(v) < 20:
                continue
            g = VOC[it["termo"]]["grupo"]
            H[g].append((float(np.median(v)), float(it["area_px"])))
            vol[(d, it["item"])] = float(np.mean(v)) * float(it["area_px"]) * px      # média × área = volume (falhas pela média)
    hg = {}
    for g, l in H.items():
        if l:
            h, w = np.array(l).T
            o = np.argsort(h); c = np.cumsum(w[o]) / w.sum()
            hg[g] = float(h[o][np.searchsorted(c, 0.5)])
    return hg, vol


def ajusta_k(A, M, Mg):
    """k por grupo (g/cm²) por mínimos quadrados não negativos, em erro relativo à massa do prato: linhas do prato
    (Σ k_g A_g = massa) e, com peso 0,5, linhas por grupo (k_g A_g = m_g, onde há área ou massa do grupo). As linhas por
    grupo usam a massa do prato como escala, para que um grupo identificado errado não puxe k para zero."""
    ok = np.all(np.isfinite(A), 1) & (M > 0)
    A, M, Mg = A[ok], M[ok], Mg[ok]
    linhas, alvo = [A / M[:, None]], [np.ones(len(M))]
    for g in range(len(GRUPOS)):
        s = (A[:, g] > 0) | (Mg[:, g] > 0)
        if s.any():
            E = np.zeros((s.sum(), len(GRUPOS))); E[:, g] = A[s, g]
            esc = M[s] / 0.5
            linhas.append(E / esc[:, None]); alvo.append(Mg[s, g] / esc)
    k, _ = nnls(np.vstack(linhas), np.concatenate(alvo))
    return k


def ajusta_rho(Mg, Pg):
    """Proteína por grama de cada grupo no treino (Σ p_g / Σ m_g)."""
    m, p = Mg.sum(0), Pg.sum(0)
    return np.where(m > 0, p / np.maximum(m, 1e-9), 0.0)


def g1(A, k, rho):
    return A @ k, (A * k) @ rho


def g2(por_item, hg, dens, rho):
    """Massa = Σ área × h_grupo × d_termo; proteína = Σ massa_item × ρ_grupo. Grupo sem altura: a mediana das alturas."""
    hmed = float(np.median(list(hg.values()))) if hg else np.nan
    m, p = np.zeros(len(por_item)), np.zeros(len(por_item))
    for i, its in enumerate(por_item):
        for termo, g, a, _ in its:
            x = a * hg.get(g, hmed) * dens[termo]
            m[i] += x; p[i] += x * rho[GRUPOS.index(g)]
    return m, p


def gprof(ids, por_item, vol, dens, rho):
    m, p = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
    for i, (d, its) in enumerate(zip(ids, por_item)):
        vs = [(vol.get((d, it["item"])), termo, g) for termo, g, a, it in its]
        if its and all(v is not None for v, _, _ in vs):
            m[i] = sum(v * dens[t] for v, t, _ in vs); p[i] = sum(v * dens[t] * rho[GRUPOS.index(g)] for v, t, g in vs)
    return m, p


def mistura(X, y, fold):
    """Pesos da G4 por fold (ajustados nas previsões fora-do-fold dos outros folds) e com tudo (índice 5)."""
    Xl = np.c_[np.log1p(X), np.ones(len(y))]; yl = np.log1p(y)
    ok = np.all(np.isfinite(Xl), 1)
    return [np.linalg.lstsq(Xl[ok & (fold != f)], yl[ok & (fold != f)], rcond=None)[0] for f in range(5)] + [np.linalg.lstsq(Xl[ok], yl[ok], rcond=None)[0]]


def aplica_mistura(X, W, fold=None):
    Xl = np.c_[np.log1p(X), np.ones(len(X))]
    return np.expm1(np.array([Xl[i] @ W[5 if fold is None or fold[i] < 0 else fold[i]] for i in range(len(X))])).clip(0)


def main():
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "--alturas":
        td, fotos, itens, esc = a[1:]
        N = le_n5k_grupos()
        F = {f["id"]: f for f in le_tabela(fotos) if f["base"] == "n5k"}
        hg, _ = alturas_n5k(td, list(N["ids"]), itens_por_foto(le_tabela(itens), "n5k"), le_escala_n5k(esc), F)
        for g in GRUPOS:
            print("%-16s %s" % (g, "%.2f cm" % hg[g] if g in hg else "sem itens"))
        print("Copie para priors.csv (tipo=altura) antes de qualquer leitura; a fonte é a profundidade do treino, sem grama.")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
