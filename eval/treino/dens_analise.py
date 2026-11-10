#!/usr/bin/env python3
"""Seção 9, passo 1: análise de erro da DENSIDADE de proteína. Só usa as previsões fora-do-fold do
treino (CV agrupada por cadeia) e o `calib` (120 pratos). Nenhum prato do `teste` é lido aqui.

Uso: python3 dens_analise.py
Lê o banco da seção 8 (zen_banco), os ingredientes do Nutrition5k dos pratos de treino/calib e as
estimativas do LLM v2 por componente. Grava <TD>/resultados/dens/analise.json e imprime as tabelas.
"""
import csv, os, sys
import numpy as np
from zen_comum import Dados, proteinas, crit, ok, I_M, I_D, EPS_D, GRUPOS
from dens_comum import Ingr, FONTES, fonte_de, tabela_n5k, faixas, grava_json, PREC, pt_en as _pt_en
import zen_avalia as ZA
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from protein_from_components import Tabela, TACO_PADRAO, N5K_PADRAO  # noqa: E402

SEL_IMG = "média: ajuste fino + siglip2-512"
SEL_MASSA = "média: ajuste fino + mlp + ridge o8 com laterais + siglip2-512 (sem TIPS)"


def decomp(nome, real_p, real_m, est_m, est_d, w=None, quem=None):
    """Por faixa: critério, 'se a massa fosse a real', 'se a densidade fosse a real', viés mediano e
    dispersão (desvio absoluto mediano) do erro em log de massa e de densidade."""
    real_d = real_p / real_m
    out = {}
    grupos = quem if quem is not None else faixas(real_p)
    print(f"\n## {nome}")
    print(f"{'grupo':22s} {'n':>4s} | crit m×d | massa real×dens est | massa est×dens real | viés massa | viés dens | |erro log| massa / dens (mediana) | % dens fora de ±20%")
    for k, f in grupos.items():
        if f.sum() < 5:
            continue
        ww = None if w is None else w[f]
        em = np.log(est_m[f] / real_m[f]); ed = np.log((est_d[f] + EPS_D) / (real_d[f] + EPS_D))
        r = dict(n=int(f.sum()), crit=crit(real_p[f], est_m[f] * est_d[f], ww), so_dens=crit(real_p[f], real_m[f] * est_d[f], ww),
                 so_massa=crit(real_p[f], est_m[f] * real_d[f], ww), vies_massa=float(100 * (np.exp(np.median(em)) - 1)),
                 vies_dens=float(100 * (np.exp(np.median(ed)) - 1)), abs_massa=float(np.median(np.abs(em))), abs_dens=float(np.median(np.abs(ed))),
                 dens_fora20=float(100 * np.mean(np.abs(est_d[f] / np.maximum(real_d[f], 1e-6) - 1) > 0.2)))
        out[k] = r
        print(f"{k:22s} {r['n']:4d} | {r['crit']:5.1f}%   | {r['so_dens']:5.1f}%              | {r['so_massa']:5.1f}%              | {r['vies_massa']:+5.0f}%     | {r['vies_dens']:+5.0f}%    |"
              f" {r['abs_massa']:.3f} / {r['abs_dens']:.3f}                  | {r['dens_fora20']:.0f}%")
    return out


def encolhimento(real_d, est_d, quem=None):
    """Inclinação de log(dens prevista) contra log(dens real): 1 = sem regressão à média."""
    x, y = np.log(real_d + EPS_D), np.log(est_d + EPS_D)
    if quem is not None:
        x, y = x[quem], y[quem]
    b = np.polyfit(x, y, 1)
    return dict(inclinacao=float(b[0]), r=float(np.corrcoef(x, y)[0, 1]), dp_real=float(x.std()), dp_prev=float(y.std()))


def main():
    D = Dados()
    B = ZA.carrega()
    res = {}
    yp, ym = D.Y["proteina"], D.Y["massa"]
    oof = B[SEL_IMG]["oof"]; P = proteinas(oof)
    m_sel = np.exp(B[SEL_MASSA]["oof"][:, I_M])
    I = Ingr(D.ids)
    print(f"treino: {len(D.ids)} pratos, {len(I.vocab)} ingredientes distintos")

    # ---------- A. treino, fora-do-fold
    print("\n# A. TREINO (fora-do-fold): modelo só de imagem da seção 8")
    res["treino_por_faixa"] = decomp("por faixa de proteína real (sem ponderar dentro da faixa)", yp, ym, m_sel, P["dens"])
    princ = {FONTES[j]: I.principal == j for j in range(len(FONTES))}
    princ["sem_fonte_dominante(≤5g ou <50%)"] = princ.pop("sem_proteina")
    res["treino_por_fonte"] = decomp("por fonte principal de proteína (≥ 50% da proteína do prato; todas as faixas)", yp, ym, m_sel, P["dens"], quem=princ)
    alto = yp > 15
    res["treino_por_fonte_15g"] = decomp("por fonte principal, só pratos com > 15 g", yp, ym, m_sel, P["dens"], quem={k: v & alto for k, v in princ.items()})
    res["treino_por_n_ingr_15g"] = decomp("pratos com > 15 g, por número de ingredientes", yp, ym, m_sel, P["dens"],
                                         quem={"1-2": alto & (I.n_ingr <= 2), "3-5": alto & (I.n_ingr > 2) & (I.n_ingr <= 5),
                                               "6-10": alto & (I.n_ingr > 5) & (I.n_ingr <= 10), "11+": alto & (I.n_ingr > 10)})
    fr = I.frac_fonte[:, :-1].sum(1)        # fração da massa que é fonte de proteína
    res["treino_por_fracao_fonte_15g"] = decomp("pratos com > 15 g, por fração da massa que é fonte de proteína", yp, ym, m_sel, P["dens"],
                                               quem={"<25%": alto & (fr < .25), "25-50%": alto & (fr >= .25) & (fr < .5),
                                                     "50-75%": alto & (fr >= .5) & (fr < .75), "≥75%": alto & (fr >= .75)})
    real_d = yp / ym
    res["encolhimento_treino"] = dict(todos=encolhimento(real_d, P["dens"]), acima15=encolhimento(real_d, P["dens"], alto),
                                      massa=dict(inclinacao=float(np.polyfit(np.log(ym), np.log(m_sel), 1)[0])))
    print("\nregressão à média (inclinação de log dens prevista × real):", res["encolhimento_treino"])
    q = np.quantile(real_d, [0, .2, .4, .6, .8, 1]); print("razão mediana dens prevista/real por quintil de densidade real:")
    res["razao_por_quintil"] = []
    for a, b in zip(q[:-1], q[1:]):
        f = (real_d >= a) & (real_d <= b)
        r = float(np.median(P["dens"][f] / np.maximum(real_d[f], 1e-4)))
        res["razao_por_quintil"].append((float(a), float(b), r))
        print(f"   dens real {100 * a:4.1f}–{100 * b:4.1f} g/100 g: razão {r:.2f}")
    # densidade prevista × densidade da fonte de proteína: o modelo erra a fonte ou a fração?
    # densidade real = Σ fração_i × dens_i. Contrafactuais com a FRAÇÃO de massa real das fontes:
    tab = tabela_n5k()
    dens_fonte_media = I.PF.sum(0) / np.maximum(I.MF.sum(0), 1e-9)   # g prot / g por fonte, no treino
    d_cf = I.frac_fonte @ dens_fonte_media                           # fração real de cada fonte × densidade média da fonte
    print("\ndensidade média por fonte no treino (g/100 g):", {f: round(100 * d, 1) for f, d in zip(FONTES, dens_fonte_media)})
    res["dens_fonte_media"] = dict(zip(FONTES, dens_fonte_media))
    res["oraculo_fracao_fonte"] = {k: crit(yp[f], ym[f] * d_cf[f]) for k, f in faixas(yp).items()}
    res["oraculo_fracao_fonte"]["ponderado"] = crit(yp, ym * d_cf, D.w_cv)
    res["oraculo_fracao_fonte_massa_modelo"] = crit(yp, m_sel * d_cf, D.w_cv)
    print("oráculo no treino — massa real × (fração real de cada uma das 10 fontes × densidade média da fonte):",
          {k: round(v, 1) for k, v in res["oraculo_fracao_fonte"].items()}, "| com a massa do modelo (ponderado):", round(res["oraculo_fracao_fonte_massa_modelo"], 1))
    # variação da densidade DENTRO da fonte (frango 0,31 × coxa 0,24 × grelhado 0,23; peixe 0,18 × salmão 0,24)
    print("\ningredientes de cada fonte no treino (n pratos, dens g/100 g):")
    from collections import Counter
    cnt = Counter(n for l in I.listas for n, g, _ in l if g > 0)
    res["ingr_por_fonte"] = {}
    for f in FONTES[:-1]:
        its = sorted(((cnt[n], n, 100 * tab.get(n, 0)) for n in I.vocab if fonte_de(n) == f), reverse=True)[:8]
        res["ingr_por_fonte"][f] = its
        print(f"   {f:16s}", "; ".join(f"{n} ({c}, {d:.0f})" for c, n, d in its))

    # ---------- B. calib: modelo e LLM v2
    print("\n# B. CALIB (120 pratos): modelo só de imagem × LLM v2")
    L = ZA.llm_pratos(D)["calib"]
    c = D.aval["calib"]; cp, cm = c["proteina"], c["massa"]
    Pc = proteinas(B[SEL_IMG]["calib"]); mc = np.exp(B[SEL_MASSA]["calib"][:, I_M])
    Ic = Ingr(c["chave"], vocab=I.vocab)
    res["calib_modelo"] = decomp("modelo (massa do conjunto × densidade do só-imagem)", cp, cm, mc, Pc["dens"])
    res["calib_llm"] = decomp("LLM v2 (massa × densidade do próprio LLM)", cp, cm, np.maximum(L["massa"], 1), L["dens"])
    res["calib_hib"] = decomp("massa do modelo × densidade do LLM", cp, cm, mc, L["dens"])
    res["encolhimento_calib"] = dict(modelo=encolhimento(cp / cm, Pc["dens"]), llm=encolhimento(cp / cm, L["dens"]))
    print("regressão à média no calib:", res["encolhimento_calib"])
    princ_c = {FONTES[j]: Ic.principal == j for j in range(len(FONTES))}
    res["calib_llm_por_fonte"] = decomp("LLM v2 por fonte principal real", cp, cm, np.maximum(L["massa"], 1), L["dens"], quem=princ_c)
    res["calib_modelo_por_fonte"] = decomp("modelo por fonte principal real", cp, cm, mc, Pc["dens"], quem=princ_c)

    # identificação da fonte principal pelo LLM, e tabela (TACO/USDA) × tabela do Nutrition5k
    tabref = Tabela(TACO_PADRAO, N5K_PADRAO)
    comp = {}
    with open(os.path.join(PREC, "estimativas_v2_comp_casados.csv")) as f:
        for r in csv.DictReader(f):
            comp.setdefault(r["prato_id"], []).append(r)
    n = len(cp)
    MFl = np.zeros((n, len(FONTES))); PFl = np.zeros((n, len(FONTES))); PFl_n5k = np.zeros((n, len(FONTES)))
    usa_taco = np.zeros(n)
    pares = []      # (fonte, dens usada pelo LLM, dens do N5K para o nome mais próximo)
    for i, pid in enumerate(c["prato"]):
        for r in comp.get(pid, []):
            g = float(r["gramas_estimadas"] or 0); p = float(r["proteina_g"] or 0)
            ref = r["ref_casado"] or r["alimento_ref"]
            fo = fonte_de(_pt_en(ref))
            j = FONTES.index(fo)
            MFl[i, j] += g; PFl[i, j] += p
            # o mesmo componente na tabela do Nutrition5k (força a tabela USDA = n5k_ingredients)
            nome5, _, p100, _, sim = tabref.casa("USDA: " + _pt_en(ref))
            d5 = p100 / 100 if sim >= 0.5 else float(r["proteina_100g"] or 0) / 100
            PFl_n5k[i, j] += g * d5
            if r["tabela"] == "TACO":
                usa_taco[i] += g
            if fo != "sem_proteina" and g > 0:
                pares.append((fo, float(r["proteina_100g"] or 0), 100 * d5, r["ref_casado"], nome5, g))
    usa_taco /= np.maximum(MFl.sum(1), 1)
    print(f"\ncomponentes do LLM casados com a TACO (fração da massa, média no calib): {100 * usa_taco.mean():.0f}%")
    res["llm_fracao_taco"] = float(usa_taco.mean())
    pl = PFl[:, :-1]; plt_ = PFl.sum(1)
    princ_l = np.where((pl.max(1) >= 0.5 * np.maximum(plt_, 1e-9)) & (plt_ > 5), pl.argmax(1), len(FONTES) - 1)
    tem = Ic.principal < len(FONTES) - 1
    acerta = princ_l[tem] == Ic.principal[tem]
    print(f"fonte principal: LLM acerta em {100 * acerta.mean():.0f}% dos {tem.sum()} pratos do calib com fonte dominante real")
    res["llm_fonte_principal"] = dict(n=int(tem.sum()), acerto=float(100 * acerta.mean()))
    conf = {}
    for a, b in zip(Ic.principal[tem], princ_l[tem]):
        conf.setdefault(FONTES[a], {}).setdefault(FONTES[b], 0); conf[FONTES[a]][FONTES[b]] += 1
    res["llm_confusao"] = conf
    for a, d in conf.items():
        print(f"   real {a:16s} -> LLM:", d)
    # critério com a massa real: acertou × errou a fonte
    okd = ok(cp, cm * L["dens"])
    print(f"   massa real × dens LLM: {100 * okd[tem][acerta].mean():.0f}% quando acerta a fonte (n={acerta.sum()}), "
          f"{100 * okd[tem][~acerta].mean():.0f}% quando erra (n={(~acerta).sum()})")
    res["llm_fonte_principal"].update(crit_acerta=float(100 * okd[tem][acerta].mean()), crit_erra=float(100 * okd[tem][~acerta].mean()))

    # contrafactuais de densidade no calib, sempre com a MASSA REAL (isola a densidade)
    d_real_f = Ic.PF / np.maximum(Ic.MF, 1e-9)                       # dens real por fonte no prato
    d_llm_f = PFl / np.maximum(MFl, 1e-9)                            # dens do LLM por fonte no prato
    fr_real = Ic.frac_fonte; fr_llm = MFl / np.maximum(MFl.sum(1, keepdims=True), 1e-9)
    cf = {
        "dens do LLM (como está)": L["dens"],
        "dens do LLM com a tabela do Nutrition5k": PFl_n5k.sum(1) / np.maximum(MFl.sum(1), 1),
        "frações do LLM × dens REAL de cada fonte no prato (fonte ausente: média do treino)":
            (fr_llm * np.where(Ic.MF > 0, d_real_f, dens_fonte_media)).sum(1),
        "frações REAIS × dens do LLM por fonte (fonte que o LLM não viu: média do treino)":
            (fr_real * np.where(MFl > 0, d_llm_f, dens_fonte_media)).sum(1),
        "frações REAIS × dens média da fonte no treino": fr_real @ dens_fonte_media,
        "frações do LLM × dens média da fonte no treino": fr_llm @ dens_fonte_media,
        "dens do modelo só de imagem": Pc["dens"],
    }
    print("\ncontrafactuais no calib (massa REAL × densidade…): critério total | por faixa 0-5 / 5-15 / 15-25 / 25-40 / >40 | >15 g")
    res["contrafactuais_calib"] = {}
    fx = faixas(cp)
    for k, d in cf.items():
        r = dict(total=crit(cp, cm * d), **{f: crit(cp[m], cm[m] * d[m]) for f, m in fx.items()}, acima15=crit(cp[cp > 15], (cm * d)[cp > 15]))
        res["contrafactuais_calib"][k] = r
        print(f"   {k[:84]:84s} {r['total']:5.1f}% | " + " / ".join(f"{r[f]:3.0f}" for f in fx) + f" | {r['acima15']:.0f}%")
    # erro de fração × erro de densidade por fonte, em pratos com > 15 g
    print("\nfração de massa da(s) fonte(s) de proteína: LLM × real (pratos do calib com > 15 g)")
    a15 = cp > 15
    fr_r = fr_real[:, :-1].sum(1); fr_l = fr_llm[:, :-1].sum(1)
    res["fracao_fontes"] = dict(real=float(fr_r[a15].mean()), llm=float(fr_l[a15].mean()), razao_mediana=float(np.median(fr_l[a15] / np.maximum(fr_r[a15], 1e-3))),
                                r=float(np.corrcoef(fr_r[a15], fr_l[a15])[0, 1]),
                                modelo_r=float(np.corrcoef(fr_r[a15], Pc["dens"][a15])[0, 1]))
    print("  ", res["fracao_fontes"])
    dd = (PFl[:, :-1].sum(1) / np.maximum(MFl[:, :-1].sum(1), 1e-9)); dr = (Ic.PF[:, :-1].sum(1) / np.maximum(Ic.MF[:, :-1].sum(1), 1e-9))
    v = a15 & (MFl[:, :-1].sum(1) > 0) & (Ic.MF[:, :-1].sum(1) > 0)
    res["dens_das_fontes"] = dict(real=float(np.median(dr[v])), llm=float(np.median(dd[v])), razao_mediana=float(np.median(dd[v] / dr[v])),
                                  n5k=float(np.median((PFl_n5k[:, :-1].sum(1) / np.maximum(MFl[:, :-1].sum(1), 1e-9))[v] / dr[v])))
    print("   densidade só da parte 'fonte de proteína' (LLM/real, mediana):", res["dens_das_fontes"])
    # tabela: densidade que o LLM usou × a do Nutrition5k para o mesmo alimento, por fonte (ponderado por gramas)
    print("\ndensidade usada pelo LLM (TACO/USDA) × Nutrition5k para o mesmo nome, por fonte (g/100 g, média ponderada por gramas):")
    res["tabela_por_fonte"] = {}
    for fo in FONTES[:-1]:
        q = [(a, b, g) for f_, a, b, _, _, g in pares if f_ == fo]
        if len(q) < 3:
            continue
        g = np.array([x[2] for x in q]); a = np.array([x[0] for x in q]); b = np.array([x[1] for x in q])
        res["tabela_por_fonte"][fo] = dict(n=len(q), llm=float((a * g).sum() / g.sum()), n5k=float((b * g).sum() / g.sum()))
        print(f"   {fo:16s} n={len(q):3d}  LLM {res['tabela_por_fonte'][fo]['llm']:5.1f}  N5k {res['tabela_por_fonte'][fo]['n5k']:5.1f}")
    from collections import Counter
    top = Counter((a, b) for _, _, _, a, b, _ in pares).most_common(25)
    res["casamentos"] = [(a, b, k) for (a, b), k in top]
    print("\nnomes mais usados pelo LLM nas fontes de proteína -> nome do Nutrition5k casado:")
    for (a, b), k in top:
        print(f"   {k:3d}× {a!r:55s} -> {b!r}")
    grava_json("analise.json", res)


if __name__ == "__main__":
    main()
