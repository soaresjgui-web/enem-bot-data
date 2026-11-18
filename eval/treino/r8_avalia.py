#!/usr/bin/env python3
"""Seção 16.6 (k) e 16.7: leitura única do teste de registro DFood-TJ (oitava execução), na CPU, com marca.

    python3 -I r8_avalia.py <pasta_kaggle_8> <gabarito_r8_dfoodtj.csv>
    python3 -I r8_avalia.py <pasta_kaggle_8> --ensaio     # ensaio do código com rótulos sorteados; não abre o gabarito nem grava marca
    python3 -I r8_avalia.py <pasta_kaggle_9> <gabarito> <conferencia_r9.json> <pasta_kaggle_8> --leitura2   # 16.8: leitura 2 de 3 (marca própria)
    python3 -I r8_avalia.py <pasta_kaggle_9> <conferencia_r9.json> <pasta_kaggle_8> --ensaio2            # ensaio da leitura 2, rótulos sorteados
    python3 -I r8_avalia.py <pasta_kaggle_9> <gabarito> <conferencia_r9.json> <pasta_kaggle_8> --pos-hoc    # 16.8 (h): diagnósticos post hoc, só depois da leitura 2

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
MARCA2 = os.path.join(SAIDA, "r8_leitura2_lida.json")
PERIODOS = ("2020-2023", "2024-2025")


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
        r = L[i]
        if r.get("fracao_cinza_janela", "") != "":          # 9ª execução: o kernel grava a fração pintada de fato (a máscara poupa o prato)
            cob.append(float(r["fracao_cinza_janela"])); continue
        W, H = int(r["largura"]), int(r["altura"]); s = 8
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
    if "--ensaio2" in sys.argv and len(a) == 3:
        return leitura2(a[0], None, a[1], a[2])
    if "--leitura2" in sys.argv and len(a) == 4:
        return leitura2(*a)
    if "--pos-hoc" in sys.argv and len(a) == 4:
        return pos_hoc(*a)
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


def le(p8, z, G, ids, destino, caixas="caixas_r8_dfoodtj.csv", entrada=None, manter=None, nivel="ic95", grupos_r8=None):
    """entrada: período -> variante da regra ('inteira' ou 'recorte'; leitura 1: foto inteira nos dois); manter: máscara de
    refeições que ficam (exclusões por dígito e queda de período, 16.6 a); nivel: intervalo da regra (ic95 na leitura 1,
    ic975 = Bonferroni de 97,5% na leitura 2); grupos_r8: cobertura de cinza da oitava execução, para os mesmos pratos."""
    ids_todos = ids
    manter = np.ones(len(ids), bool) if manter is None else manter
    col = lambda k: np.array([float(G[i][k]) for i, m_ in zip(ids_todos, manter) if m_])
    P, P_lista, massa = col("proteina_corrigida"), col("proteina_lista"), col("massa")
    pac = np.array([G[i]["paciente"] for i, m_ in zip(ids_todos, manter) if m_]); per = np.array([G[i]["periodo"] for i, m_ in zip(ids_todos, manter) if m_])
    assert (per == np.array([str(x) for x in z["periodo"]])[manter]).all()
    E = {v: {k: x[manter] for k, x in estimativas(z, v).items()} for v in ("inteira", "recorte")}
    cob = cobertura_janela(os.path.join(p8, caixas), ids_todos)[manter]
    entrada = entrada or {p: "inteira" for p in PERIODOS}
    regra = np.where(np.isin(per, [p for p in PERIODOS if entrada[p] == "recorte"]), E["recorte"]["S7"], E["inteira"]["S7"])
    var_macro = "recorte" if all(entrada[p] == "recorte" for p in set(per)) else "inteira"
    R = dict(n=int(manter.sum()), pacientes=len(set(pac)), faixas_n=np.bincount(np.digitize(P, FX, right=True), minlength=3).tolist(), entrada=entrada, nivel=nivel)

    def linha(rot, real, est, idx=None):
        i = np.arange(len(real)) if idx is None else np.where(idx)[0]
        if len(i) == 0:                                  # grupo vazio (ex.: sem janela com cinza na máscara nova)
            R[rot] = dict(n=0); print("%-58s n=  0 (grupo vazio)" % rot); return
        v = b2(pac[i], lambda j: crit3(real[i], est[i], j))
        pf, vies = por_faixa(real[i], est[i])
        R[rot] = dict(n=len(i), pacientes=len(set(pac[i])), valor=v[0], ic95=v[1:3], ic975=v[3:5], por_faixa=pf, vies_por_faixa=vies,
                      vies=100 * float(np.median(est[i] / real[i] - 1)))
        print("%-58s n=%3d (%2d pac.)  %5.1f%%  IC95 %5.1f a %5.1f | IC97,5 %5.1f a %5.1f | faixas %s | viés por faixa %s" %
              (rot, len(i), len(set(pac[i])), v[0], v[1], v[2], v[3], v[4], " / ".join("%.0f" % x for x in pf), " / ".join("%+.0f%%" % x for x in vies)))

    nome_regra = "S7 inteira (REGRA)" if nivel == "ic95" else "S7 (REGRA; entrada por período: %s)" % ", ".join("%s %s" % (p, entrada[p]) for p in sorted(set(per)))
    print("REGRA: S7, 4 orientações, sem constante; intervalo %s" % ("de 95%" if nivel == "ic95" else "de 97,5% (Bonferroni, leitura 2)"))
    linha(nome_regra, P, regra)
    v = R[nome_regra]
    R["veredito"] = "sucesso" if v[nivel][0] >= 70 else "não atingida"
    print("  -> limite inferior (%s): %.1f; meta %s\n" % (nivel, v[nivel][0], R["veredito"]))
    print("Linhas descritivas declaradas (não reclamam a meta):")
    rec = np.isin(per, [p for p in PERIODOS if entrada[p] == "recorte"])
    mix = lambda k: np.where(rec, E["recorte"][k], E["inteira"][k])        # a entrada da regra, período a período
    tag = "inteira" if nivel == "ic95" else "entrada da regra"
    linha("S6 %s" % tag, P, mix("S6"))
    linha("F5c %s (1 orientação)" % tag, P, mix("F5c"))
    for p in PERIODOS:
        if (per == p).any():
            linha("S7 %s, %s" % (tag, p), P, regra, per == p)
    linha("S7 %s, proteína da lista (sem correção do ovo)" % tag, P_lista, regra)
    linha("S7 janela do guia mascarada", P, E["recorte"]["S7"])
    if nivel != "ic95":
        linha("S7 foto inteira", P, E["inteira"]["S7"])
    for var in ("inteira", "recorte"):
        linha("S7 %s, janela com até 10%% de cinza" % var, P, E[var]["S7"], cob <= 0.10)
        linha("S7 %s, janela com mais de 25%% de cinza" % var, P, E[var]["S7"], cob > 0.25)
        if grupos_r8 is not None:
            g8 = grupos_r8[manter]
            linha("S7 %s, pratos com até 10%% de cinza na 8ª" % var, P, E[var]["S7"], g8 <= 0.10)
            linha("S7 %s, pratos com mais de 25%% de cinza na 8ª" % var, P, E[var]["S7"], g8 > 0.25)
    for var in ("inteira", "recorte"):
        m = E[var]["massa"]
        R["massa_" + var] = dict(dentro20=dentro20(massa, m), vies=100 * float(np.median(m / massa - 1)),
                                 por_periodo={p: dentro20(massa[per == p], m[per == p]) for p in PERIODOS if (per == p).any()})
        print("Massa (R4c) %-8s a ±20%%: %.1f%% | viés mediano %+.0f%% | por período %s" % (var, R["massa_" + var]["dentro20"], R["massa_" + var]["vies"],
              " / ".join("%.1f" % x for x in R["massa_" + var]["por_periodo"].values())))
    for nut in ("kcal", "carbo", "gordura"):
        vm = var_macro if nivel != "ic95" else "inteira"
        real, est = col(nut), E[vm][nut]
        v = b2(pac, lambda j: crit_macro(real, est, nut, j))
        R["macro_" + nut] = dict(valor=v[0], ic95=v[1:3], vies=100 * float(np.median(est / np.maximum(real, 0.5) - 1)))
        print("%-8s (S7, %s; critério de 12.8): %.1f%% (IC95 %.1f a %.1f) | viés mediano %+.0f%%" % (nut, vm, v[0], v[1], v[2], R["macro_" + nut]["vies"]))
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if destino:
        json.dump(R, open(destino, "w"), indent=1, default=float, ensure_ascii=False)


def conferencia(conf, z):
    """16.8 (c): a conferência a olho gravada antes da leitura. Devolve (manter, entrada por período, registro)."""
    C = json.load(open(conf))
    ids = [str(x) for x in z["ids"]]; per = np.array([str(x) for x in z["periodo"]])
    dig = set(C["digitos_legiveis"]); assert dig <= set(ids)
    manter = np.array([i not in dig for i in ids])
    p25 = per == "2024-2025"
    cai = len(dig & set(np.array(ids)[p25])) > 0.10 * p25.sum()          # 16.6 a (3): mais de 10% de 2024–2025 com dígito
    if cai:
        manter &= ~p25
    entrada = {p: ("inteira" if p in C["recorte_reprovado"] else "recorte") for p in PERIODOS}
    reg = dict(sha256=hashlib.sha256(open(conf, "rb").read()).hexdigest(), excluidas=len(dig), periodo_2024_cai=bool(cai), entrada=entrada)
    print("Conferência (16.8 c): %d excluídas por dígito; 2024–2025 %s; entrada %s; sha256 %s" % (len(dig), "SAI" if cai else "fica", entrada, reg["sha256"][:16]))
    return manter, entrada, reg


def leitura2(p9, gab, conf, p8):
    """16.8: leitura 2 de 3 do S7 congelado com a máscara só do visor; Bonferroni de 97,5%. gab=None: ensaio com rótulos sorteados."""
    assert os.path.realpath(p9) != os.path.realpath(p8), "a pasta da nona execução não pode ser a da oitava"
    acha = lambda *ns: next((n for n in ns if os.path.exists(os.path.join(p9, n))), None)
    npz, cx = acha("pred9__dfoodtj.npz", "pred8__dfoodtj.npz"), acha("caixas_r9_dfoodtj.csv", "caixas_r8_dfoodtj.csv")
    assert npz and cx, "previsões ou caixas da nona execução não encontradas"
    assert "fracao_cinza_janela" in next(csv.reader(open(os.path.join(p9, cx)))), "a tabela de caixas não é a da máscara nova (16.8 a)"
    z = np.load(os.path.join(p9, npz), allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    manter, entrada, reg = conferencia(conf, z)
    z8 = np.load(os.path.join(p8, "pred8__dfoodtj.npz"), allow_pickle=False)
    assert [str(x) for x in z8["ids"]] == ids, "ids da 8ª e da 9ª não casam"
    g8 = cobertura_janela(os.path.join(p8, "caixas_r8_dfoodtj.csv"), ids)
    if gab is None:
        rng = np.random.default_rng(2)
        G = {i: dict(proteina_corrigida=rng.uniform(3, 45), proteina_lista=rng.uniform(3, 45), massa=rng.uniform(150, 700), kcal=rng.uniform(150, 900),
                     carbo=rng.uniform(10, 120), gordura=rng.uniform(2, 40), paciente=str(p), periodo=str(q)) for i, p, q in zip(ids, z["paciente"], z["periodo"])}
        return le(p9, z, G, ids, None, cx, entrada, manter, "ic975", g8)
    if not os.path.exists(MARCA):
        sys.exit("A leitura 1 não consta; a leitura 2 não vem antes dela.")
    if os.path.exists(MARCA2):
        sys.exit("A leitura 2 do DFood-TJ já foi feita (%s)." % json.load(open(MARCA2))["quando"])
    if hashlib.sha256(open(gab, "rb").read()).hexdigest() != SHA_GABARITO:
        sys.exit("sha256 do gabarito não confere com o gravado em 16.6.")
    G = {r["dish_id"]: r for r in csv.DictReader(open(gab))}
    assert len(ids) == 711 and set(ids) == set(G)
    json.dump(dict(quando=time.strftime("%Y-%m-%d %H:%M:%S"), modelo="S7 = R4c×H-cap r4 o4 (17.8 g), máscara só do visor (16.8)", leitura=2, conferencia=reg),
              open(MARCA2, "w"), indent=1, ensure_ascii=False)
    le(p9, z, G, ids, os.path.join(SAIDA, "r8_leitura2.json"), cx, entrada, manter, "ic975", g8)


def pos_hoc(p9, gab, conf, p8):
    """16.8 (h): diagnósticos POST HOC, só depois da leitura 2. Explicam; não escolhem nada e não reclamam a meta."""
    if not os.path.exists(MARCA2):
        sys.exit("Os diagnósticos post hoc só rodam depois da leitura 2 (16.8 h).")
    from macros_cv import FAIXAS
    G = {r["dish_id"]: r for r in csv.DictReader(open(gab))}
    R = {}
    acha = lambda *ns: next(n for n in ns if os.path.exists(os.path.join(p9, n)))
    for rot, p, npz, cx in (("8ª", p8, "pred8__dfoodtj.npz", "caixas_r8_dfoodtj.csv"),
                            ("9ª", p9, acha("pred9__dfoodtj.npz", "pred8__dfoodtj.npz"), acha("caixas_r9_dfoodtj.csv", "caixas_r8_dfoodtj.csv"))):
        z = np.load(os.path.join(p, npz), allow_pickle=False); ids = [str(x) for x in z["ids"]]
        col = lambda k: np.array([float(G[i][k]) for i in ids])
        P, M = col("proteina_corrigida"), col("massa"); per = np.array([G[i]["periodo"] for i in ids])
        cob = cobertura_janela(os.path.join(p, cx), ids)
        PH = o4(z["recorte__hcap__s"].astype(np.float64)); dff = z["recorte__hcap__dens_ff"].astype(np.float64)[:, :4].mean(1)
        dm = dens_med(PH, dff); m = np.exp(o4(z["recorte__r4c"].astype(np.float64))[:, I_M]); dreal = P / M
        for nome, sel in (("todos", np.ones(len(ids), bool)), ("até 10% de cinza", cob <= 0.10)):
            for pp in (None,) + PERIODOS:
                i = sel & (per == pp if pp else True)
                if not i.any():
                    continue
                x = dict(n=int(i.sum()), massa_real_x_dens_prevista=crit3(P[i], (M * dm)[i]), massa_prevista_x_dens_real=crit3(P[i], (m * dreal)[i]),
                         massa_20=dentro20(M[i], m[i]), vies_massa=100 * float(np.median(m[i] / M[i] - 1)),
                         dens_20=100 * float(np.mean(np.abs(dm[i] - dreal[i]) <= 0.2 * dreal[i])), vies_dens=100 * float(np.median(dm[i] / dreal[i] - 1)),
                         massa_real_mediana=float(np.median(M[i])), massa_prevista_mediana=float(np.median(m[i])))
                R["%s | %s | %s" % (rot, nome, pp or "os dois")] = x
                print("POST HOC %s, janela do guia, %-17s %-10s n=%3d: massa real × dens. prevista %.1f%% | massa prevista × dens. real %.1f%% | massa ±20%% %.1f%% (viés %+.0f%%; real %.0f g, prevista %.0f g) | dens. ±20%% %.1f%% (viés %+.0f%%)" %
                      (rot, nome, pp or "os dois", x["n"], x["massa_real_x_dens_prevista"], x["massa_prevista_x_dens_real"], x["massa_20"], x["vies_massa"],
                       x["massa_real_mediana"], x["massa_prevista_mediana"], x["dens_20"], x["vies_dens"]))
    for nut in ("kcal", "carbo", "gordura"):
        real = np.array([float(G[i][nut]) for i in G])
        R["faixas_" + nut] = np.bincount(np.digitize(real, FAIXAS[nut], right=True), minlength=5).tolist()
        print("POST HOC refeições por faixa de 12.8, %-8s %s: %s" % (nut, FAIXAS[nut], R["faixas_" + nut]))
    R["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(R, open(os.path.join(SAIDA, "r8_pos_hoc.json"), "w"), indent=1, default=float, ensure_ascii=False)


if __name__ == "__main__":
    main()
