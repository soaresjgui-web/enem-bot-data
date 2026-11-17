#!/usr/bin/env python3
"""Seção 16.6 (k) e 16.7: leitura única do teste de registro DFood-TJ (oitava execução), na CPU, com marca.

    python3 -I r8_avalia.py <pasta_kaggle_8> <gabarito_r8_dfoodtj.csv>
    python3 -I r8_avalia.py <pasta_kaggle_8> --ensaio     # ensaio do código com rótulos sorteados; não abre o gabarito nem grava marca

Regra (16.6 b, c, j, k): S7 congelado (R4c × H-cap, rota r4 com os pesos de 17.8 g, 4 primeiras orientações) na foto
inteira mascarada (a conferência do recorte reprovou os dois períodos); critério do produto em três faixas de proteína
corrigida (até 15 / 15–25 / acima de 25 g) com o mesmo peso, sem constante de domínio; IC de 95% por bootstrap de
paciente (4.000 sorteios). Sucesso se o limite inferior for de 70% ou mais (leitura 1 de 3).
Linhas descritivas (não reclamam a meta): S6, F5c, por período, sem correção do ovo, massa, kcal/carboidrato/gordura,
o S7 na janela do guia mascarada e por cobertura de cinza na janela (até 10% / mais de 25%)."""
import csv, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from comum import ok
from zen_comum import EPS_D, I_M, I_P, I_D
from dens_comum import DENS
from r5_avalia import dens_med, b2
from r4_avalia import dentro20
from macros_cv import crit as crit_macro, I as I_MACRO

SHA_GABARITO = "fdf2e17238f9222fab3946acedd96c5605251f79de5d93930a8fec63fc07fde6"
W_R4 = np.array([0.4903, 0.2609, 0.3018, -0.1558])     # 17.8 g: Nutrition5k inteiro, 4 orientações
FX = [15, 25]                                          # 16.6 a: três faixas com o mesmo peso
SAIDA = os.path.join(DENS, "r8")
MARCA = os.path.join(SAIDA, "r8_leitura_lida.json")


def crit3(real, est, idx=None):
    if idx is not None:
        real, est = real[idx], est[idx]
    a, fx = ok(real, est), np.digitize(real, FX, right=True)
    return 100 * float(np.mean([a[fx == k].mean() for k in range(3) if (fx == k).any()]))


def por_faixa(real, est):
    a, fx = ok(real, est), np.digitize(real, FX, right=True)
    return [100 * float(a[fx == k].mean()) for k in range(3)], [100 * float(np.median(est[fx == k] / real[fx == k] - 1)) for k in range(3)]


def o4(P):
    """Média nas 4 primeiras orientações (as que preservam o quadro), como `media_or` do treino."""
    return P[:, :4].mean(1)


def estimativas(z, var):
    """S7, S6, componentes, massa e macros do S7 numa variante ('inteira' ou 'recorte')."""
    PR = o4(z[var + "__r4c"].astype(np.float64))
    out = {}
    for nome, h in (("S7", "hcap"), ("S6", "hm")):
        PH = o4(z[var + "__" + h + "__s"].astype(np.float64))
        dff = o4(z[var + "__" + h + "__dens_ff"].astype(np.float64)[..., None])[:, 0]
        m = np.exp(PR[:, I_M])
        r1 = m * dens_med(PH, dff); dR = np.expm1(PR[:, I_P]).clip(0); dH = np.expm1(PH[:, I_P]).clip(0)
        if nome == "S7":
            X = np.c_[np.log1p(r1), np.log1p(dR), np.log1p(dH), np.ones(len(r1))]
            out["S7"] = np.expm1(X @ W_R4).clip(0)
            out["massa"] = m
            dk = lambda P, j: np.expm1(P[:, j]).clip(0)
            lm = lambda a, b: np.expm1((np.log1p(a) + np.log1p(b)) / 2)
            out["kcal"] = lm(dk(PR, I_MACRO["kcal"]), dk(PH, I_MACRO["kcal"]))
            out["carbo"] = lm(dk(PR, I_MACRO["carbo"]), dk(PH, I_MACRO["carbo"]))
            out["gordura"] = np.sqrt(dk(PR, I_MACRO["gordura"]) * m * dk(PH, I_MACRO["gordura"]) / np.exp(PH[:, I_M]))
        else:
            out["S6"] = np.sqrt(r1 * dR)
    P5, ff5 = z[var + "__f5c__s"].astype(np.float64), z[var + "__f5c__dens_ff"].astype(np.float64)
    out["F5c"] = np.exp(P5[:, I_M]) * dens_med(P5, ff5)          # rota massa × densidade média (15.8)
    return out


def cobertura_janela(caixas, ids):
    """Fração da janela do guia (dentro da foto) coberta pelas caixas de visor, pela tabela de caixas (sem rótulo)."""
    L = {r["id"]: r for r in csv.DictReader(open(caixas))}
    cob = []
    for i in ids:
        r = L[i]; W, H = int(r["largura"]), int(r["altura"]); s = 8
        g = np.zeros((H // s + 1, W // s + 1), bool)
        for v in json.loads(r["caixas_visor"]):
            x0, y0, x1, y1 = v["caixa"]; g[int(y0) // s:int(y1) // s + 1, int(x0) // s:int(x1) // s + 1] = True
        cx, cy, a, l = (float(r[k]) for k in ("janela_cx", "janela_cy", "janela_alt", "janela_larg"))
        x0, x1, y0, y1 = max(0, cx - l / 2), min(W, cx + l / 2), max(0, cy - a / 2), min(H, cy + a / 2)
        cob.append(g[int(y0) // s:int(y1) // s + 1, int(x0) // s:int(x1) // s + 1].mean())
    return np.array(cob)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--ensaio" in sys.argv and len(a) == 1:
        return ensaio(a[0])
    if len(a) != 2:
        sys.exit(__doc__)
    p8, gab = a
    if os.path.exists(MARCA):
        sys.exit("A leitura do DFood-TJ já foi feita (%s). Não há segunda leitura deste candidato." % json.load(open(MARCA))["quando"])
    if hashlib.sha256(open(gab, "rb").read()).hexdigest() != SHA_GABARITO:
        sys.exit("sha256 do gabarito não confere com o gravado em 16.6.")
    z = np.load(os.path.join(p8, "pred8__dfoodtj.npz"), allow_pickle=False)
    G = {r["dish_id"]: r for r in csv.DictReader(open(gab))}
    ids = [str(x) for x in z["ids"]]
    assert len(ids) == 711 and set(ids) == set(G), "ids das previsões e do gabarito não casam"
    os.makedirs(SAIDA, exist_ok=True)
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), modelo="S7 = R4c×H-cap r4 o4 (17.8 g), foto inteira mascarada", leitura=1), open(MARCA, "w"), indent=1, ensure_ascii=False)
    le(p8, z, G, ids, os.path.join(SAIDA, "r8_leitura.json"))


def ensaio(p8):
    """Os mesmos cálculos com rótulos sorteados (sem gabarito, sem marca), para não perder a leitura única por um erro de código."""
    z = np.load(os.path.join(p8, "pred8__dfoodtj.npz"), allow_pickle=False)
    ids = [str(x) for x in z["ids"]]; rng = np.random.default_rng(1)
    G = {i: dict(proteina_corrigida=rng.uniform(3, 45), proteina_lista=rng.uniform(3, 45), massa=rng.uniform(150, 700), kcal=rng.uniform(150, 900),
                 carbo=rng.uniform(10, 120), gordura=rng.uniform(2, 40), paciente=str(p), periodo=str(q))
         for i, p, q in zip(ids, z["paciente"], z["periodo"])}
    le(p8, z, G, ids, None)


def le(p8, z, G, ids, destino):
    col = lambda k: np.array([float(G[i][k]) for i in ids])
    P, P_lista, massa = col("proteina_corrigida"), col("proteina_lista"), col("massa")
    pac = np.array([G[i]["paciente"] for i in ids]); per = np.array([G[i]["periodo"] for i in ids])
    assert (per == np.array([str(x) for x in z["periodo"]])).all()
    E = {v: estimativas(z, v) for v in ("inteira", "recorte")}
    cob = cobertura_janela(os.path.join(p8, "caixas_r8_dfoodtj.csv"), ids)
    R = dict(n=len(ids), pacientes=len(set(pac)), faixas_n=np.bincount(np.digitize(P, FX, right=True), minlength=3).tolist())

    def linha(rot, real, est, idx=None):
        i = np.arange(len(real)) if idx is None else np.where(idx)[0]
        v = b2(pac[i], lambda j: crit3(real[i], est[i], j))
        pf, vies = por_faixa(real[i], est[i])
        R[rot] = dict(n=len(i), pacientes=len(set(pac[i])), valor=v[0], ic95=v[1:3], ic975=v[3:5], por_faixa=pf, vies_por_faixa=vies,
                      vies=100 * float(np.median(est[i] / real[i] - 1)))
        print("%-58s n=%3d (%2d pac.)  %5.1f%%  IC95 %5.1f a %5.1f | faixas %s | viés por faixa %s" %
              (rot, len(i), len(set(pac[i])), v[0], v[1], v[2], " / ".join("%.0f" % x for x in pf), " / ".join("%+.0f%%" % x for x in vies)))

    print("REGRA (16.6): S7 na foto inteira mascarada, 4 orientações, sem constante")
    linha("S7 inteira (REGRA)", P, E["inteira"]["S7"])
    v = R["S7 inteira (REGRA)"]
    R["veredito"] = "sucesso" if v["ic95"][0] >= 70 else "não atingida"
    print("  -> limite inferior de 95%%: %.1f; meta %s\n" % (v["ic95"][0], R["veredito"]))
    print("Linhas descritivas declaradas (não reclamam a meta):")
    linha("S6 inteira", P, E["inteira"]["S6"])
    linha("F5c inteira (1 orientação)", P, E["inteira"]["F5c"])
    for p in ("2020-2023", "2024-2025"):
        linha("S7 inteira, %s" % p, P, E["inteira"]["S7"], per == p)
    linha("S7 inteira, proteína da lista (sem correção do ovo)", P_lista, E["inteira"]["S7"])
    linha("S7 janela do guia mascarada", P, E["recorte"]["S7"])
    for var in ("inteira", "recorte"):
        linha("S7 %s, janela com até 10%% de cinza" % var, P, E[var]["S7"], cob <= 0.10)
        linha("S7 %s, janela com mais de 25%% de cinza" % var, P, E[var]["S7"], cob > 0.25)
    for var in ("inteira", "recorte"):
        m = E[var]["massa"]
        R["massa_" + var] = dict(dentro20=dentro20(massa, m), vies=100 * float(np.median(m / massa - 1)),
                                 por_periodo={p: dentro20(massa[per == p], m[per == p]) for p in ("2020-2023", "2024-2025")})
        print("Massa (R4c) %-8s a ±20%%: %.1f%% | viés mediano %+.0f%% | por período %s" % (var, R["massa_" + var]["dentro20"], R["massa_" + var]["vies"],
              " / ".join("%.1f" % x for x in R["massa_" + var]["por_periodo"].values())))
    for nut in ("kcal", "carbo", "gordura"):
        real, est = col(nut), E["inteira"][nut]
        v = b2(pac, lambda j: crit_macro(real, est, nut, j))
        R["macro_" + nut] = dict(valor=v[0], ic95=v[1:3], vies=100 * float(np.median(est / np.maximum(real, 0.5) - 1)))
        print("%-8s (S7, foto inteira; critério de 12.8): %.1f%% (IC95 %.1f a %.1f) | viés mediano %+.0f%%" % (nut, v[0], v[1], v[2], R["macro_" + nut]["vies"]))
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if destino:
        json.dump(R, open(destino, "w"), indent=1, default=float, ensure_ascii=False)


if __name__ == "__main__":
    main()
