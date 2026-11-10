#!/usr/bin/env python3
"""Seção 8 (uma única foto de cima): seleção por CV no treino, avaliação única no teste,
híbridos com o LLM (pesos só no calib), bootstrap pareado, confiança e itens brasileiros.

Uso: python3 zen_avalia.py
Lê o banco de previsões (<TD>/resultados/zen_banco/*.npz, gravado por zen_experimentos.py e
zen_finetune.py), a linha de base 2d (zen_base2d.npz) e as estimativas do LLM v2 do repositório.
Grava <TD>/resultados/zen_avalia.json e imprime as tabelas.

Regras (declaradas antes de olhar o teste):
  - modelo de MASSA (entrada do híbrido): menor RMSE de log(massa) na CV agrupada do treino;
  - pipeline só de imagem (on-device): maior critério do produto na CV do treino, ponderado para
    as 5 faixas de proteína terem o mesmo peso (é como calib/teste foram sorteados);
  - forma do híbrido: maior critério no calib com validação deixa-um-fora (pesos só no calib);
  - limiar de confiança: escolhido no calib; teste só para relatar.
  Todas as configurações são avaliadas no teste e todas aparecem na tabela.
"""
import glob, json, os
import numpy as np
from zen_comum import (TD, Dados, proteinas, crit, mape, bootstrap, ic_simples, llm_componentes, ok, GRUPOS,
                       I_M, I_P, I_D, I_MG, I_PG, EPS_D, N_ALVOS)

BANCO = os.path.join(TD, "resultados", "zen_banco")
N5K = ["calib", "teste", "n5k_v1"]
BR = ["ufpr", "liv_up", "seara"]
ROTAS = ["direto", "mxd", "grupos", "gmd", "geo"]
# conjuntos (média em log das previsões); só entram os membros que existirem no banco
CONJUNTOS = {
    "média: siglip2 o8 + siglip2-512": ["ridge siglip2_b16 o8", "ridge siglip2_b16_512 o2"],
    "média: siglip2 o8 + tips": ["ridge siglip2_b16 o8", "ridge tips_b14 o1"],
    "média: siglip2 o8 + siglip2-512 + tips": ["ridge siglip2_b16 o8", "ridge siglip2_b16_512 o2", "ridge tips_b14 o1"],
    "média: siglip2 o8 + mlp": ["ridge siglip2_b16 o8", "mlp siglip2_b16 o8"],
    "média: ajuste fino + mlp": ["ft cv", "mlp siglip2_b16 o8"],
    "média: ajuste fino + mlp + ridge o8 com laterais": ["ft cv", "mlp siglip2_b16 o8", "ridge siglip2_b16 o8 + quadros laterais no treino"],
    "média: ajuste fino + mlp + ridge o8 com laterais + siglip2-512 + tips": ["ft cv", "mlp siglip2_b16 o8", "ridge siglip2_b16 o8 + quadros laterais no treino",
                                                                              "ridge siglip2_b16_512 o2", "ridge tips_b14 o1"],
    "média: ajuste fino + ridge siglip2 o8": ["ft cv", "ridge siglip2_b16 o8"],
    "média: ajuste fino + ridge o8 com laterais": ["ft cv", "ridge siglip2_b16 o8 + quadros laterais no treino"],
    "média: ajuste fino + siglip2-512": ["ft cv", "ridge siglip2_b16_512 o2"],
    # os conjuntos abaixo foram declarados com um terceiro backbone (TIPS B/14), que não chegou a ser extraído
    # por falta de tempo; as versões "sem TIPS" são os mesmos conjuntos sem esse membro
    "média: ajuste fino + ridge o8 + siglip2-512 (sem TIPS)": ["ft cv", "ridge siglip2_b16 o8", "ridge siglip2_b16_512 o2"],
    "média: ajuste fino + ridge o8 com laterais + siglip2-512 (sem TIPS)": ["ft cv", "ridge siglip2_b16 o8 + quadros laterais no treino", "ridge siglip2_b16_512 o2"],
    "média: ajuste fino + mlp + ridge o8 com laterais + siglip2-512 (sem TIPS)": ["ft cv", "mlp siglip2_b16 o8", "ridge siglip2_b16 o8 + quadros laterais no treino",
                                                                                  "ridge siglip2_b16_512 o2"],
    "média: ajuste fino + siglip2-512 + tips": ["ft cv", "ridge siglip2_b16_512 o2", "ridge tips_b14 o1"],
    "média: ajuste fino + ridge o8 + siglip2-512 + tips": ["ft cv", "ridge siglip2_b16 o8", "ridge siglip2_b16_512 o2", "ridge tips_b14 o1"],
    "média: ajuste fino + ridge o8 com laterais + siglip2-512 + tips": ["ft cv", "ridge siglip2_b16 o8 + quadros laterais no treino",
                                                                        "ridge siglip2_b16_512 o2", "ridge tips_b14 o1"],
}


DENS_GRUPO = None  # proteína por grama de cada grupo no TREINO (razão das somas); definido em main()


def rotas(T):
    """direto | mxd (massa × densidade) | grupos (soma da proteína prevista por grupo) |
    gmd (soma de massa prevista por grupo × densidade média do grupo no treino) | geo (média geométrica direto·mxd)."""
    P = proteinas(T)
    P["geo"] = np.sqrt(P["direto"] * P["mxd"])
    P["gmd"] = np.expm1(T[:, I_MG]).clip(0) @ DENS_GRUPO
    return P


def carrega():
    B = {}
    for f in sorted(glob.glob(os.path.join(BANCO, "*.npz"))):
        nome = os.path.splitext(os.path.basename(f))[0]
        if nome.startswith("escala") or nome.endswith("holdout"):
            continue
        B[nome] = dict(np.load(f, allow_pickle=False))
    for nome, membros in CONJUNTOS.items():
        if all(m in B for m in membros):
            B[nome] = {k: np.mean([B[m][k] for m in membros], 0) for k in ["oof"] + N5K + BR if all(k in B[m] for m in membros)}
    return B


def llm_pratos(D):
    comp = llm_componentes()
    nivel = {"alta": 2, "media": 1, "média": 1, "baixa": 0}
    out = {}
    for s, a in D.aval.items():
        g = np.zeros((len(a["prato"]), len(GRUPOS))); p = np.zeros_like(g); conf = np.full(len(a["prato"]), np.nan); tem = np.zeros(len(a["prato"]), bool)
        for i, pid in enumerate(a["prato"]):
            for grupo, gr, pr, c in comp.get(pid, []):
                j = GRUPOS.index(grupo) if grupo in GRUPOS else len(GRUPOS) - 1
                g[i, j] += gr; p[i, j] += pr; tem[i] = True
                conf[i] = np.nanmin([conf[i], nivel.get(c.strip().lower(), 1)])
        out[s] = dict(g=g, p=p, massa=g.sum(1), prot=p.sum(1), dens=p.sum(1) / np.maximum(g.sum(1), 1), conf=conf, tem=tem)
    return out


# ------------------------------------------------------------------ híbridos (pesos só no calib)
def hibridos(Pm, L, real_calib, dens_grupo):
    """Pm[s]: rotas do modelo (com 'mg' = massa por grupo); L[s]: LLM. Devolve {nome: {split: estimativa}} e
    as estimativas deixa-um-fora no calib para a escolha da forma."""
    lg = lambda x: np.log(np.maximum(x, 1e-3))

    def feats(s):
        P, l = Pm[s], L[s]
        return dict(lm=lg(P["massa"]), ll=lg(np.maximum(l["massa"], 1)), dl=np.log(l["dens"] + EPS_D), dm=np.log(P["dens"] + EPS_D),
                    pd=np.log1p(P["direto"]), pl=np.log1p(l["prot"]), um=np.ones(len(P["massa"])))

    def por_grupo(s, a):
        P, l = Pm[s], L[s]
        d = np.where(l["g"] > 0, l["p"] / np.maximum(l["g"], 1e-6), 0)
        m = np.where(l["g"] > 0, np.maximum(l["g"], 1) ** (1 - a) * np.maximum(P["mg"], 1) ** a, 0)
        return (m * d).sum(1)

    def so_animal(s, a):
        """Só as gramas do grupo 'proteína animal' são misturadas com as do modelo; o resto fica como o LLM estimou."""
        P, l = Pm[s], L[s]
        tem = l["g"][:, 0] > 0
        d0 = np.where(tem, l["p"][:, 0] / np.maximum(l["g"][:, 0], 1e-6), 0)
        m0 = np.where(tem, np.maximum(l["g"][:, 0], 1) ** (1 - a) * np.maximum(P["mg"][:, 0], 1) ** a, 0)
        return l["prot"] - l["p"][:, 0] + m0 * d0

    formas = {
        "massa modelo × dens LLM": (None, lambda s, w: Pm[s]["massa"] * L[s]["dens"]),
        "massa √(modelo·LLM) × dens LLM": (None, lambda s, w: np.sqrt(Pm[s]["massa"] * np.maximum(L[s]["massa"], 1)) * L[s]["dens"]),
        "massa modelo^a·LLM^(1−a) × dens LLM": ("grade1", lambda s, w: np.exp(w * feats(s)["lm"] + (1 - w) * feats(s)["ll"]) * L[s]["dens"]),
        "massa^a × dens (LLM^b·modelo^(1−b))": ("grade2", lambda s, w: np.exp(w[0] * feats(s)["lm"] + (1 - w[0]) * feats(s)["ll"])
                                                 * (np.exp(w[1] * feats(s)["dl"] + (1 - w[1]) * feats(s)["dm"]) - EPS_D).clip(0)),
        "blend log-linear (5 pesos)": (["lm", "ll", "dl", "dm", "um"], None),
        "blend log-linear + proteína direta e do LLM (7 pesos)": (["lm", "ll", "dl", "dm", "pd", "pl", "um"], None),
        "por grupo: gramas LLM^(1−a)·modelo^a × dens LLM do grupo": ("grupo", por_grupo),
        "só proteína animal: gramas LLM^(1−a)·modelo^a × dens LLM": ("grupo", so_animal),
        "média geométrica: proteína direta do modelo e do LLM": (None, lambda s, w: np.sqrt(Pm[s]["direto"] * L[s]["prot"])),
    }
    y = np.log1p(real_calib)
    grade = np.linspace(0, 1.2, 13)

    def ajusta(tipo, f, idx):
        """Ajusta os pesos nos pratos idx do calib; devolve função split -> estimativa."""
        if tipo is None:
            return lambda s: f(s, None)
        if isinstance(tipo, list):
            F = np.stack([feats("calib")[k] for k in tipo], 1)
            w = np.linalg.solve(F[idx].T @ F[idx] + 1e-3 * np.eye(F.shape[1]), F[idx].T @ y[idx])
            return lambda s: np.expm1(np.stack([feats(s)[k] for k in tipo], 1) @ w).clip(0)
        if tipo in ("grade1", "grupo"):
            e = [np.mean((np.log1p(f("calib", a))[idx] - y[idx]) ** 2) for a in grade]
            a = grade[int(np.argmin(e))]
            return lambda s: f(s, a)
        e = {(a, b): np.mean((np.log1p(f("calib", (a, b)))[idx] - y[idx]) ** 2) for a in grade for b in grade[:11]}
        ab = min(e, key=e.get)
        return lambda s: f(s, ab)

    n = len(y)
    est, loo = {}, {}
    for nome, (tipo, f) in formas.items():
        g = ajusta(tipo, f, np.arange(n))
        est[nome] = {s: g(s) for s in Pm}
        if tipo is None:
            loo[nome] = est[nome]["calib"]
        else:  # deixa-um-fora em blocos de 10 (12 ajustes): barato e sem vazar o próprio prato
            v = np.zeros(n)
            for b in range(12):
                fora = np.arange(n) % 12 == b
                v[fora] = ajusta(tipo, f, np.where(~fora)[0])("calib")[fora]
            loo[nome] = v
    return est, loo


# ------------------------------------------------------------------ confiança
def logistica(X, y, w=None, l2=1.0, it=200):
    """Regressão logística simples (Newton) com intercepto; devolve função X -> P(acerto)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.ones(len(y)) / len(y) if w is None else w / w.sum()
    b = np.zeros(Z.shape[1])
    R = l2 / len(y) * np.eye(Z.shape[1]); R[-1, -1] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ b))
        g = Z.T @ (w * (p - y)) + R @ b
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + R + 1e-9 * np.eye(Z.shape[1])
        d = np.linalg.solve(H, g); b -= d
        if np.abs(d).max() < 1e-8:
            break
    return lambda Xn: 1 / (1 + np.exp(-np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ b))


def curva_cobertura(p, acerto):
    """Acurácia entre os k pratos mais confiantes, para cada cobertura."""
    o = np.argsort(-p)
    ac = np.cumsum(acerto[o]) / np.arange(1, len(o) + 1)
    return {f"{c:.0%}": float(100 * ac[max(1, int(round(c * len(o)))) - 1]) for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}


def limiar_para(p, acerto, alvo=0.70, minimo=10):
    """Menor limiar de confiança cujo conjunto aceito tem acurácia >= alvo (maior cobertura possível)."""
    o = np.argsort(-p)
    ac = np.cumsum(acerto[o]) / np.arange(1, len(o) + 1)
    ks = [k for k in range(minimo, len(o) + 1) if ac[k - 1] >= alvo]
    return float(p[o][max(ks) - 1]) if ks else float("inf")


def main():
    global DENS_GRUPO
    D = Dados()
    DENS_GRUPO = D.Y["pg"].sum(0) / np.maximum(D.Y["mg"].sum(0), 1)
    B = carrega()
    L = llm_pratos(D)
    b2 = np.load(os.path.join(TD, "resultados", "zen_base2d.npz"))
    res = {"n_treino": len(D.ids)}
    real = {s: D.aval[s]["proteina"] for s in D.aval}
    massa_real = {s: D.aval[s]["massa"] for s in D.aval}
    te = "teste"
    base_dir = ok(real[te], b2["teste|direto"])
    base_hib = ok(real[te], b2["teste|massa"] * L[te]["dens"])
    llm_ok = ok(real[te], L[te]["prot"])
    print(f"Linhas de base no teste: LLM {100 * llm_ok.mean():.1f}% | 2d direto {100 * base_dir.mean():.1f}% | 2d híbrido (massa × dens LLM) {100 * base_hib.mean():.1f}%"
          f" | oráculo massa real × dens LLM {crit(real[te], massa_real[te] * L[te]['dens']):.1f}%"
          f" | oráculo massa LLM × dens real {crit(real[te], L[te]['massa'] * real[te] / massa_real[te]):.1f}%")
    res["base"] = dict(llm=100 * llm_ok.mean(), d2_direto=100 * base_dir.mean(), d2_hibrido=100 * base_hib.mean(),
                       mape_massa_2d=mape(massa_real[te], b2["teste|massa"]), mape_massa_llm=mape(massa_real[te], np.maximum(L[te]["massa"], 1)))

    # ---------------- 1. CV no treino (seleção) e teste (relato), todas as configurações
    print("\n# 1. Todas as configurações: CV no treino e teste")
    print(f"{'configuração':52s} | CV logm RMSE | CV MAPE m | CV crit dir/mxd/grp/gmd/geo        | TESTE MAPE m | dir / mxd / grp / gmd / geo         | m×dLLM  Δ vs 2d híbrido (IC95%)")
    tab = {}
    for nome, b in B.items():
        if te not in b:
            continue
        oof = b["oof"]; v = ~np.isnan(oof[:, 0])
        Po = rotas(oof[v]); Pt = rotas(b[te])
        cv = dict(rmse_logm=float(np.sqrt(np.mean((oof[v, I_M] - D.T[v, I_M]) ** 2))), mape=mape(D.Y["massa"][v], Po["massa"]),
                  n_cv=int(v.sum()), **{r: crit(D.Y["proteina"][v], Po[r], D.w_cv[v]) for r in ROTAS})
        t = dict(mape=mape(massa_real[te], Pt["massa"]), **{r: crit(real[te], Pt[r]) for r in ROTAS})
        h = ok(real[te], Pt["massa"] * L[te]["dens"])
        t["hib"] = 100 * h.mean(); t["hib_ic"] = bootstrap(base_hib, h)
        t["dir_ic"] = bootstrap(base_dir, ok(real[te], Pt["direto"]))
        t["mae_direto"] = float(np.abs(Pt["direto"] - real[te]).mean()); t["vies_direto"] = float((Pt["direto"] - real[te]).mean())
        tab[nome] = dict(cv=cv, teste=t)
        print(f"{nome[:52]:52s} | {cv['rmse_logm']:.4f}{'*' if cv['n_cv'] < len(D.ids) else ' '}     | {cv['mape']:5.1f}%    | "
              f"{cv['direto']:5.1f} {cv['mxd']:5.1f} {cv['grupos']:5.1f} {cv['gmd']:5.1f} {cv['geo']:5.1f}      | {t['mape']:5.1f}%       | "
              f"{t['direto']:5.1f} {t['mxd']:5.1f} {t['grupos']:5.1f} {t['gmd']:5.1f} {t['geo']:5.1f}      | {t['hib']:5.1f}  {t['hib_ic'][0]:+5.1f} ({t['hib_ic'][1]:+.1f} a {t['hib_ic'][2]:+.1f})")
    res["tabela"] = tab
    completos = [n for n in tab if tab[n]["cv"]["n_cv"] == len(D.ids)]
    sel_massa = min(completos, key=lambda n: tab[n]["cv"]["rmse_logm"])
    sel_img, sel_rota = max(((n, r) for n in completos for r in ROTAS), key=lambda nr: tab[nr[0]]["cv"][nr[1]])
    print(f"\nEscolha pela CV — modelo de massa: '{sel_massa}' (RMSE log {tab[sel_massa]['cv']['rmse_logm']:.4f})")
    print(f"Escolha pela CV — só imagem: '{sel_img}', rota {sel_rota} (critério ponderado {tab[sel_img]['cv'][sel_rota]:.1f}%)")
    res["escolha"] = dict(massa=sel_massa, imagem=sel_img, rota=sel_rota)

    # fator multiplicativo c escolhido para maximizar o critério ponderado fora-do-fold; só é adotado se
    # melhorar a CV com ajuste cruzado (c escolhido em 4 folds, aplicado no 5º)
    grade_c = np.round(np.arange(0.80, 1.31, 0.02), 2)
    po = rotas(B[sel_img]["oof"])[sel_rota]; yp = D.Y["proteina"]
    cv_c = lambda idx: grade_c[int(np.argmax([crit(yp[idx], c * po[idx], D.w_cv[idx]) for c in grade_c]))]
    cruz = np.zeros(len(yp))
    for k in range(5):
        cruz[D.fold == k] = cv_c(D.fold != k) * po[D.fold == k]
    c_all = cv_c(np.arange(len(yp)))
    usa_c = crit(yp, cruz, D.w_cv) > tab[sel_img]["cv"][sel_rota] + 0.5
    res["fator_c"] = dict(c=float(c_all), cv_sem=tab[sel_img]["cv"][sel_rota], cv_com_ajuste_cruzado=crit(yp, cruz, D.w_cv), adotado=bool(usa_c),
                          teste_sem=crit(real[te], rotas(B[sel_img][te])[sel_rota]), teste_com=crit(real[te], c_all * rotas(B[sel_img][te])[sel_rota]))
    print(f"Fator de escala c={c_all:.2f}: CV {res['fator_c']['cv_sem']:.1f}% -> {res['fator_c']['cv_com_ajuste_cruzado']:.1f}% (ajuste cruzado); "
          f"{'ADOTADO' if usa_c else 'não adotado'}; teste sem/com: {res['fator_c']['teste_sem']:.1f}% / {res['fator_c']['teste_com']:.1f}%")
    if usa_c:
        for k in ["oof"] + N5K + BR:
            if k in B[sel_img]:
                B[sel_img] = dict(B[sel_img]); T2 = B[sel_img][k].copy(); T2[:, I_P] = np.log1p(c_all * np.expm1(T2[:, I_P]).clip(0)); T2[:, I_D] = np.log(c_all * (np.exp(T2[:, I_D]) - EPS_D).clip(0) + EPS_D)
                T2[:, I_PG] = np.log1p(c_all * np.expm1(T2[:, I_PG]).clip(0)); B[sel_img][k] = T2

    # ---------------- 2. Só imagem (on-device) escolhido: teste, IC
    Pt = rotas(B[sel_img][te])
    a = ok(real[te], Pt[sel_rota])
    d = bootstrap(base_dir, a); dl = bootstrap(llm_ok, a); dh = bootstrap(base_hib, a)
    print(f"\n# 2. Só imagem escolhido no teste: {100 * a.mean():.1f}% (IC95% {ic_simples(a)[0]:.1f} a {ic_simples(a)[1]:.1f}) | MAE {np.abs(Pt[sel_rota] - real[te]).mean():.1f} g"
          f" | viés {(Pt[sel_rota] - real[te]).mean():+.1f} g | vs 2d direto {d[0]:+.1f} ({d[1]:+.1f} a {d[2]:+.1f}) | vs LLM {dl[0]:+.1f} ({dl[1]:+.1f} a {dl[2]:+.1f})"
          f" | vs 2d híbrido {dh[0]:+.1f} ({dh[1]:+.1f} a {dh[2]:+.1f})")
    res["imagem"] = dict(teste=100 * a.mean(), ic=ic_simples(a), mae=float(np.abs(Pt[sel_rota] - real[te]).mean()),
                         vies=float((Pt[sel_rota] - real[te]).mean()), vs_2d_direto=d, vs_llm=dl, vs_2d_hibrido=dh,
                         mape_massa=mape(massa_real[te], Pt["massa"]),
                         por_faixa={f: crit(real[te][fx], Pt[sel_rota][fx]) for f, fx in faixas(real[te]).items()},
                         n5k_v1=crit(real["n5k_v1"], rotas(B[sel_img]["n5k_v1"])[sel_rota]))
    res["imagem"]["oraculo_massa_real_x_dens_modelo"] = crit(real[te], massa_real[te] * Pt["dens"])
    res["imagem"]["oraculo_massa_modelo_x_dens_real"] = crit(real[te], Pt["massa"] * real[te] / massa_real[te])
    print("   por faixa de proteína:", {k: round(v, 1) for k, v in res["imagem"]["por_faixa"].items()}, "| n5k_v1:", round(res["imagem"]["n5k_v1"], 1),
          f"| oráculos: massa real × dens do modelo {res['imagem']['oraculo_massa_real_x_dens_modelo']:.1f}%, massa do modelo × dens real {res['imagem']['oraculo_massa_modelo_x_dens_real']:.1f}%")

    # 2b. opção prática on-device: um único backbone (o ajuste fino parcial), mesmo que a CV prefira um conjunto
    res["um_backbone"] = {}
    um = [n for n in completos if "512" not in n and "tips" not in n]          # só o SigLIP 2 B/16 a 224 px
    melhor_um = max(um, key=lambda n: max(tab[n]["cv"][r] for r in ROTAS))
    res["escolha"]["um_backbone"] = melhor_um
    print(f"   melhor com um único backbone pela CV: '{melhor_um}'")
    for nome in dict.fromkeys([melhor_um, "ft cv", "mlp siglip2_b16 o8", "ridge siglip2_b16 o8"]):
        if nome not in tab:
            continue
        rota = max(ROTAS, key=lambda r: tab[nome]["cv"][r])
        P1 = rotas(B[nome][te]); a1 = ok(real[te], P1[rota])
        res["um_backbone"][nome] = dict(rota=rota, cv=tab[nome]["cv"][rota], teste=100 * a1.mean(), ic=ic_simples(a1), mae=float(np.abs(P1[rota] - real[te]).mean()),
                                        vies=float((P1[rota] - real[te]).mean()), vs_2d_direto=bootstrap(base_dir, a1), vs_llm=bootstrap(llm_ok, a1),
                                        vs_2d_hibrido=bootstrap(base_hib, a1), mape_massa=mape(massa_real[te], P1["massa"]),
                                        por_faixa={f: crit(real[te][fx], P1[rota][fx]) for f, fx in faixas(real[te]).items()},
                                        n5k_v1=crit(real["n5k_v1"], rotas(B[nome]["n5k_v1"])[rota]),
                                        oraculo_massa_real_x_dens_modelo=crit(real[te], massa_real[te] * P1["dens"]))
        u = res["um_backbone"][nome]
        print(f"   um backbone — {nome} (rota {rota}, CV {u['cv']:.1f}%): teste {u['teste']:.1f}% (IC95% {u['ic'][0]:.1f} a {u['ic'][1]:.1f}) MAE {u['mae']:.1f} viés {u['vies']:+.1f}"
              f" | vs 2d direto {u['vs_2d_direto'][0]:+.1f} ({u['vs_2d_direto'][1]:+.1f} a {u['vs_2d_direto'][2]:+.1f}) | vs LLM {u['vs_llm'][0]:+.1f} ({u['vs_llm'][1]:+.1f} a {u['vs_llm'][2]:+.1f})"
              f" | vs 2d híbrido {u['vs_2d_hibrido'][0]:+.1f} ({u['vs_2d_hibrido'][1]:+.1f} a {u['vs_2d_hibrido'][2]:+.1f}) | v1 {u['n5k_v1']:.0f}% | faixas {({k: round(v) for k, v in u['por_faixa'].items()})}")

    # ---------------- 3. Híbridos com o modelo de massa escolhido (densidade/direto do modelo só de imagem escolhido)
    print(f"\n# 3. Híbridos (massa: '{sel_massa}'; densidade/proteína direta do modelo: '{sel_img}'); pesos e escolha da forma só no calib")
    Pm = {}
    for s in N5K + BR:
        if s in B[sel_massa] and s in B[sel_img]:
            Pi = rotas(B[sel_img][s]); Pm[s] = dict(massa=np.exp(B[sel_massa][s][:, I_M]), dens=Pi["dens"], direto=Pi[sel_rota],
                                                   mg=np.expm1(B[sel_massa][s][:, I_MG]).clip(0))
    est, loo = hibridos(Pm, L, real["calib"], None)
    hib = {}
    for nome in est:
        h = ok(real[te], est[nome][te]); ic = bootstrap(base_hib, h); il = bootstrap(llm_ok, h)
        hib[nome] = dict(calib_loo=crit(real["calib"], loo[nome]), teste=100 * h.mean(), ic_abs=ic_simples(h), vs_2d_hibrido=ic, vs_llm=il,
                         mae=float(np.abs(est[nome][te] - real[te]).mean()), vies=float((est[nome][te] - real[te]).mean()),
                         n5k_v1=crit(real["n5k_v1"], est[nome]["n5k_v1"]),
                         br={s: crit(real[s], est[nome][s]) for s in BR if s in est[nome]})
        print(f"  {nome[:58]:58s} calib(LOO) {hib[nome]['calib_loo']:5.1f}% | TESTE {hib[nome]['teste']:5.1f}% MAE {hib[nome]['mae']:.1f} viés {hib[nome]['vies']:+.1f}"
              f" | Δ vs 2d híbrido {ic[0]:+5.1f} ({ic[1]:+.1f} a {ic[2]:+.1f}) | Δ vs LLM {il[0]:+5.1f} ({il[1]:+.1f} a {il[2]:+.1f}) | v1 {hib[nome]['n5k_v1']:.0f}%")
    sel_h = max(hib, key=lambda n: hib[n]["calib_loo"])
    print(f"Forma escolhida no calib: '{sel_h}' -> teste {hib[sel_h]['teste']:.1f}% (IC95% {hib[sel_h]['ic_abs'][0]:.1f} a {hib[sel_h]['ic_abs'][1]:.1f})")
    res["hibrido"] = hib; res["escolha"]["hibrido"] = sel_h
    res["hibrido_por_faixa"] = {f: crit(real[te][fx], est[sel_h][te][fx]) for f, fx in faixas(real[te]).items()}
    res["llm_por_faixa"] = {f: crit(real[te][fx], L[te]["prot"][fx]) for f, fx in faixas(real[te]).items()}
    res["oraculo"] = dict(massa_real_x_dens_llm=crit(real[te], massa_real[te] * L[te]["dens"]),
                          massa_modelo_x_dens_real=crit(real[te], Pm[te]["massa"] * real[te] / massa_real[te]),
                          massa_llm_x_dens_real=crit(real[te], L[te]["massa"] * real[te] / massa_real[te]))
    print("  oráculos no teste:", {k: round(v, 1) for k, v in res["oraculo"].items()})

    # ---------------- 4. Confiança
    print("\n# 4. Confiança: P(acerto) calibrada e curva acurácia × cobertura")
    # (a) só imagem: ajustada nas previsões fora-do-fold do TREINO; limiar para 70% escolhido no calib
    membros = [n for n in ("ridge siglip2_b16 o8", "ridge siglip2_b16_512 o2", "ridge tips_b14 o1", "mlp siglip2_b16 o8", "ft cv") if n in B]

    def x_img(chave):
        T = B[sel_img][chave]; P = rotas(T)
        lp = np.stack([B[m][chave][:, I_P] for m in membros], 1)
        lm = np.stack([B[m][chave][:, I_M] for m in membros], 1)
        return np.c_[np.log1p(P[sel_rota]), np.abs(np.log1p(P["direto"]) - np.log1p(P["mxd"])), np.abs(np.log1p(P["grupos"]) - np.log1p(P["direto"])),
                     lp.std(1), lm.std(1), T[:, I_D]]
    acerto_tr = ok(D.Y["proteina"], rotas(B[sel_img]["oof"])[sel_rota]).astype(float)
    # P(acerto) fora-do-fold no treino, para medir a calibração sem otimismo
    f_conf = logistica(x_img("oof"), acerto_tr, D.w_cv)
    conf = {}
    for s in ("calib", te):
        p = f_conf(x_img(s)); a = ok(real[s], rotas(B[sel_img][s])[sel_rota]).astype(float)
        conf[s] = (p, a)
    lim = limiar_para(*conf["calib"])
    p, a = conf[te]; ac = p >= lim
    res["confianca_imagem"] = dict(curva=curva_cobertura(p, a), limiar_calib=lim, cobertura_teste=float(100 * ac.mean()),
                                   acuracia_aceitos=float(100 * a[ac].mean()) if ac.any() else None,
                                   acuracia_recusados=float(100 * a[~ac].mean()) if (~ac).any() else None,
                                   calibracao=calibracao(p, a), cobertura_oraculo_70=cobertura_max(p, a))
    res["confianca_imagem"]["quem_e_aceito"] = dict(proteina_real_media_aceitos=float(real[te][ac].mean()), proteina_real_media_recusados=float(real[te][~ac].mean()),
                                                    fracao_aceitos_ate_15g=float(100 * (real[te][ac] <= 15).mean()), acuracia_aceitos_acima_15g=float(100 * a[ac & (real[te] > 15)].mean()),
                                                    n_aceitos_acima_15g=int((ac & (real[te] > 15)).sum()))
    res["confianca_imagem"]["regra_sem_rotulo"] = regra_sem_rotulo(p, a)
    print("  só imagem — regra sem rótulo (média prevista >= 70%):", res["confianca_imagem"]["regra_sem_rotulo"], "| cobertura máxima a 70% (descritivo):", res["confianca_imagem"]["cobertura_oraculo_70"])
    print("  só imagem — curva (cobertura: acurácia):", {k: round(v, 1) for k, v in res["confianca_imagem"]["curva"].items()})
    print(f"  só imagem — limiar do calib {lim:.2f}: cobre {100 * ac.mean():.0f}% do teste com acurácia {100 * a[ac].mean() if ac.any() else float('nan'):.1f}%"
          f" (recusados: {100 * a[~ac].mean() if (~ac).any() else float('nan'):.1f}%) | calibração: {res['confianca_imagem']['calibracao']}")
    # (b) híbrido: ajustada no calib (deixa-um-fora em blocos para escolher o limiar), aplicada no teste
    def x_hib(s):
        P, l = Pm[s], L[s]
        return np.c_[np.log1p(est[sel_h][s]), np.abs(np.log(P["massa"]) - np.log(np.maximum(l["massa"], 1))),
                     np.abs(np.log(P["dens"] + EPS_D) - np.log(l["dens"] + EPS_D)), np.nan_to_num(l["conf"], nan=1.0),
                     np.abs(np.log1p(P["direto"]) - np.log1p(est[sel_h][s]))]
    a_c = ok(real["calib"], loo[sel_h]).astype(float)
    Xc = x_hib("calib")
    p_loo = np.zeros(len(a_c))
    for b in range(12):
        fora = np.arange(len(a_c)) % 12 == b
        p_loo[fora] = logistica(Xc[~fora], a_c[~fora], l2=3.0)(Xc[fora])
    f_h = logistica(Xc, a_c, l2=3.0)
    lim_h = limiar_para(p_loo, a_c)
    p = f_h(x_hib(te)); a = ok(real[te], est[sel_h][te]).astype(float); ac = p >= lim_h
    res["confianca_hibrido"] = dict(curva=curva_cobertura(p, a), limiar_calib=lim_h, cobertura_teste=float(100 * ac.mean()),
                                    acuracia_aceitos=float(100 * a[ac].mean()) if ac.any() else None,
                                    acuracia_recusados=float(100 * a[~ac].mean()) if (~ac).any() else None,
                                    calibracao=calibracao(p, a), cobertura_oraculo_70=cobertura_max(p, a))
    res["confianca_hibrido"]["quem_e_aceito"] = dict(proteina_real_media_aceitos=float(real[te][ac].mean()), proteina_real_media_recusados=float(real[te][~ac].mean()),
                                                     fracao_aceitos_ate_15g=float(100 * (real[te][ac] <= 15).mean()), acuracia_aceitos_acima_15g=float(100 * a[ac & (real[te] > 15)].mean()),
                                                     n_aceitos_acima_15g=int((ac & (real[te] > 15)).sum()))
    res["confianca_hibrido"]["regra_sem_rotulo"] = regra_sem_rotulo(p, a)
    print("  híbrido — regra sem rótulo (média prevista >= 70%):", res["confianca_hibrido"]["regra_sem_rotulo"], "| cobertura máxima a 70% (descritivo):", res["confianca_hibrido"]["cobertura_oraculo_70"])
    print("  híbrido — curva (cobertura: acurácia):", {k: round(v, 1) for k, v in res["confianca_hibrido"]["curva"].items()})
    print(f"  híbrido — limiar do calib {lim_h:.2f}: cobre {100 * ac.mean():.0f}% do teste com acurácia {100 * a[ac].mean() if ac.any() else float('nan'):.1f}%"
          f" (recusados: {100 * a[~ac].mean() if (~ac).any() else float('nan'):.1f}%) | calibração: {res['confianca_hibrido']['calibracao']}")

    # ---------------- 5. Itens brasileiros (secundários: quase nenhum é zenital)
    print("\n# 5. Itens brasileiros (só Seara é foto de cima)")
    br = {}
    for s in BR:
        Pi = rotas(B[sel_img][s]); l = L[s]
        br[s] = dict(n=len(real[s]), imagem=crit(real[s], Pi[sel_rota]), mape_massa=mape(massa_real[s], np.exp(B[sel_massa][s][:, I_M])),
                     vies_massa=float(np.mean(np.exp(B[sel_massa][s][:, I_M]) - massa_real[s])),
                     llm=crit(real[s], l["prot"]), hibrido=crit(real[s], est[sel_h][s]),
                     hib_raiz=crit(real[s], est["massa √(modelo·LLM) × dens LLM"][s]),
                     base2d_direto=crit(real[s], b2[f"{s}|direto"]), base2d_hib=crit(real[s], b2[f"{s}|massa"] * l["dens"]),
                     base2d_mape=mape(massa_real[s], b2[f"{s}|massa"]))
        print(f"  {s:7s} n={br[s]['n']:3d} | só imagem {br[s]['imagem']:5.1f}% (2d: {br[s]['base2d_direto']:.1f}%) | MAPE massa {br[s]['mape_massa']:.0f}% (2d: {br[s]['base2d_mape']:.0f}%) viés {br[s]['vies_massa']:+.0f} g"
              f" | LLM {br[s]['llm']:.1f}% | híbrido escolhido {br[s]['hibrido']:.1f}% | √ {br[s]['hib_raiz']:.1f}% (2d m×dLLM: {br[s]['base2d_hib']:.1f}%)")
    tot = lambda f: crit(np.concatenate([real[s] for s in BR]), np.concatenate([f(s) for s in BR]))
    br["todos"] = dict(imagem=tot(lambda s: rotas(B[sel_img][s])[sel_rota]), llm=tot(lambda s: L[s]["prot"]), hibrido=tot(lambda s: est[sel_h][s]),
                       hib_raiz=tot(lambda s: est["massa √(modelo·LLM) × dens LLM"][s]), base2d_direto=tot(lambda s: b2[f"{s}|direto"]),
                       base2d_hib=tot(lambda s: b2[f"{s}|massa"] * L[s]["dens"]))
    print("  todos (144):", {k: round(v, 1) for k, v in br["todos"].items()})
    res["brasil"] = br

    # ---------------- 6. Robustez de escala
    res["escala"] = escala(D, real[te], massa_real[te], L[te])
    with open(os.path.join(TD, "resultados", "zen_avalia.json"), "w") as f:
        json.dump(res, f, indent=1, default=float, ensure_ascii=False)


def faixas(p):
    return {"0-5": p <= 5, "5-15": (p > 5) & (p <= 15), "15-25": (p > 15) & (p <= 25), "25-40": (p > 25) & (p <= 40), ">40": p > 40}


def calibracao(p, a, n=4):
    """P(acerto) prevista × observada em n faixas de mesma contagem."""
    o = np.argsort(p); out = []
    for q in np.array_split(o, n):
        out.append(f"{100 * p[q].mean():.0f}→{100 * a[q].mean():.0f}")
    return " | ".join(out)


def regra_sem_rotulo(p, a, alvo=0.70):
    """Aceita os pratos mais confiantes enquanto a MÉDIA da P(acerto) prevista entre os aceitos for >= alvo.
    Não usa nenhum rótulo do conjunto avaliado; depende só de a probabilidade estar calibrada."""
    o = np.argsort(-p); med = np.cumsum(p[o]) / np.arange(1, len(o) + 1)
    k = int((med >= alvo).sum())
    return dict(cobertura=float(100 * k / len(o)), acuracia=float(100 * a[o][:k].mean()) if k else None,
                acuracia_recusados=float(100 * a[o][k:].mean()) if k < len(o) else None)


def cobertura_max(p, a, alvo=0.70):
    """Maior cobertura no teste com acurácia >= alvo ordenando pela confiança (limiar ideal, só descritivo)."""
    o = np.argsort(-p); ac = np.cumsum(a[o]) / np.arange(1, len(o) + 1)
    ks = [k for k in range(10, len(o) + 1) if ac[k - 1] >= alvo]
    return float(100 * max(ks) / len(o)) if ks else 0.0


def escala(D, real, mreal, Lt):
    out = {}
    arqs = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(BANCO, "escala*.npz"))}
    if not arqs:
        return out
    print("\n# 6. Robustez de escala (SigLIP 2 B/16, 1 orientação, ridge) — teste")
    for nome in sorted(arqs):
        b = dict(np.load(arqs[nome], allow_pickle=False))
        tem = b.get("teste_tem_caixa", np.ones(len(real), bool)).astype(bool)
        out[nome] = {}
        for k in sorted(k for k in b if k.startswith("teste_") and k != "teste_tem_caixa"):
            P = rotas(b[k])
            out[nome][k[6:]] = dict(n=int(tem.sum()), direto=crit(real[tem], P["direto"]), mape_massa=mape(mreal[tem], P["massa"]),
                                    vies_massa_pct=float(100 * (np.median(P["massa"] / mreal[tem]) - 1)),
                                    hibrido=crit(real[tem], P["massa"] * Lt["dens"][tem]))
            r = out[nome][k[6:]]
            print(f"  {nome[:48]:48s} {k[6:]:22s} n={r['n']} direto {r['direto']:5.1f}% | MAPE massa {r['mape_massa']:5.1f}% (viés mediano {r['vies_massa_pct']:+.0f}%) | m×dLLM {r['hibrido']:5.1f}%")
    return out


if __name__ == "__main__":
    main()
