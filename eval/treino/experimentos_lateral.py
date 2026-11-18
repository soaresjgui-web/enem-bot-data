#!/usr/bin/env python3
"""Ângulos laterais: o treino com quadros side_angles do Nutrition5k melhora o modelo de peso/proteína?

Uso: OMP_NUM_THREADS=1 python3 experimentos_lateral.py [backbone=siglip2_b16]
Pré-requisitos: extrai_features.py (zenital + itens brasileiros) e extrai_features_lateral.py
(quadros laterais + ECUSTFD) com o mesmo backbone.

Higiene (igual a experimentos.py, mas aplicada ao universo de pratos com qualquer imagem):
  - os 450 dish_id de avaliação (calib/teste/n5k_v1) nunca entram no treino, em nenhuma fonte;
  - regra estrita de cadeia: agrupa pratos com < 120 s entre timestamps consecutivos e remove do treino
    qualquer cadeia que contenha prato de avaliação. A cadeia é calculada sobre TODOS os pratos com
    metadado (cafés 1 e 2), então a mesma lista de pratos vale para a fonte zenital e a lateral;
  - split por dish_id: um prato está no treino com todas as suas imagens ou fora com todas;
  - hiperparâmetros (alfa do ridge) e a escolha da configuração por GroupKFold (cadeias) só no treino;
    calib só para os pesos do blend do híbrido; resultados em teste, n5k_v1 e itens brasileiros.
Grava <TD>/resultados/lateral_<backbone>.json e imprime as tabelas.
"""
import csv, json, os, sys, time, zlib
import numpy as np
from sklearn.model_selection import GroupKFold
from comum import TD, PREC, ler_avaliacao, carrega_feat, grupos_sessao, metricas, ok
from experimentos import RidgeMulti, alvos, proteinas, llm_por_prato, EPS_D
from prep_dados import ler_meta

SCR = os.path.dirname(TD)
BR = ["ufpr", "liv_up", "seara"]


def le_manifesto():
    split = {}
    with open(os.path.join(SCR, "dados_ext", "n5k_side", "manifest.csv")) as f:
        for r in csv.DictReader(f):
            split[r["dish_id"]] = r["split"] or "sem_split"
    return split


def ecustfd():
    m = {}
    with open(os.path.join(SCR, "dados_ext", "ecustfd", "manifest.csv")) as f:
        for r in csv.DictReader(f):
            m[f"ecu|{os.path.basename(r['image_path'])}"] = (r["dish_id"], float(r["mass_g"]))
    return m


class Dados:
    def __init__(self, bb):
        self.meta = ler_meta()
        self.aval = ler_avaliacao()
        self.ev = {d for s in ("calib", "teste", "n5k_v1") for d in self.aval[s][0]}
        self.idx_o, self.Xo = carrega_feat(bb)
        z = np.load(os.path.join(TD, "feat", f"{bb}_lateral.npz"))
        self.Xs = z["X"].astype(np.float32)
        self.lat = {}
        self.ecu_rows = []
        for i, k in enumerate(z["ids"]):
            if k.startswith("ecu|"):
                self.ecu_rows.append((i, k))
            else:
                d, q = k.split("|")
                self.lat.setdefault(d, []).append(i)
        self.side_split = le_manifesto()
        # treino zenital = o mesmo de experimentos.py (rgb_train ∩ overhead ∩ filtros − avaliação)
        with open(os.path.join(TD, "meta", "treino.csv")) as f:
            self.over_ids = [r["dish_id"] for r in csv.DictReader(f)]
        # cadeias sobre todos os pratos com metadado
        todos = sorted(self.meta)
        g = grupos_sessao(todos, 120)
        self.cadeia = dict(zip(todos, g))
        proib = {self.cadeia[d] for d in self.ev if d in self.cadeia}
        self.estrito_univ = {d for d in todos if self.cadeia[d] not in proib}
        # regra original (experimentos.py): cadeias só sobre treino zenital + avaliação
        evl = [d for s in ("calib", "teste", "n5k_v1") for d in self.aval[s][0]]
        g0 = grupos_sessao(self.over_ids + evl, 120)
        ge = set(g0[len(self.over_ids):])
        self.estrito_orig = {d for d, x in zip(self.over_ids, g0) if x not in ge}

    def usavel(self, d):
        m = self.meta.get(d)
        return m is not None and m["ok"] and m["massa"] >= 20 and d not in self.ev

    def y(self, d):
        m = self.meta[d]
        return [m["massa"], m["proteina"], m["kcal"]]


def monta(D, pratos_o, pratos_s):
    """Linhas de treino: 1 linha zenital por prato em pratos_o + 1 linha por quadro lateral em pratos_s."""
    X, Y, dish, fonte = [], [], [], []
    for d in pratos_o:
        X.append(D.Xo[D.idx_o[d]]); Y.append(D.y(d)); dish.append(d); fonte.append(0)
    for d in pratos_s:
        for i in D.lat.get(d, []):
            X.append(D.Xs[i]); Y.append(D.y(d)); dish.append(d); fonte.append(1)
    return np.array(X), np.array(Y), np.array(dish), np.array(fonte)


def ajusta(D, X, Y, dish):
    g = np.array([D.cadeia[d] for d in dish])
    cv = list(GroupKFold(5).split(X, groups=g))
    return RidgeMulti(cv).fit(X, alvos(Y))


def configs(D):
    so = sorted(D.estrito_orig)                       # 1.506 pratos (regra original)
    ou = sorted(d for d in D.over_ids if d in D.estrito_univ)   # zenital, regra universal
    lat_train = sorted(d for d in D.lat if D.usavel(d) and d in D.estrito_univ and D.side_split.get(d) == "train")
    lat_todos = sorted(d for d in D.lat if D.usavel(d) and d in D.estrito_univ)
    ou_com_lat = [d for d in ou if d in D.lat]
    return {
        "1 zenital (regra original, 1.506)": (so, []),
        "1' zenital (regra universal)": (ou, []),
        "2a lateral, mesmos pratos": ([], ou_com_lat),
        "2b zenital + lateral, mesmos pratos": (ou, ou_com_lat),
        "2c zenital + lateral, + pratos só com vídeo (split train)": (ou, lat_train),
        "2d zenital + lateral, + pratos test/sem split fora da avaliação": (ou, lat_todos),
        "2e lateral, todos os pratos (split train)": ([], lat_train),
    }


def cv_selecao(D, cfgs):
    """CV agrupada SÓ no treino: folds por cadeia sobre a união de pratos; valida em pratos zenitais
    comuns a todas as configurações (zenital e quadros laterais deles). Retorna MSE log por alvo."""
    base = cfgs["1' zenital (regra universal)"][0]
    todos = sorted(set().union(*[set(o) | set(s) for o, s in cfgs.values()]))
    g = np.array([D.cadeia[d] for d in todos])
    fold = dict(zip(todos, np.zeros(len(todos), int)))
    for k, (_, va) in enumerate(GroupKFold(5).split(todos, groups=g)):
        for j in va:
            fold[todos[j]] = k
    out = {}
    for nome, (po, ps) in cfgs.items():
        if nome.startswith("1 "):
            continue
        err = {"zenital": [], "lateral": []}
        for k in range(5):
            m = ajusta(D, *monta(D, [d for d in po if fold[d] != k], [d for d in ps if fold[d] != k])[:3])
            va = [d for d in base if fold[d] == k]
            for fonte, (Xv, Yv, _, _) in (("zenital", monta(D, va, [])), ("lateral", monta(D, [], va))):
                err[fonte].append((m.predict(Xv) - alvos(Yv)) ** 2)
        out[nome] = {f: np.concatenate(e).mean(0).round(4).tolist() for f, e in err.items()}
        print(f"  CV {nome:62s} zenital log(massa) {out[nome]['zenital'][0]:.4f} log(prot) {out[nome]['zenital'][1]:.4f}"
              f" | lateral log(massa) {out[nome]['lateral'][0]:.4f} log(prot) {out[nome]['lateral'][1]:.4f}", flush=True)
    return out


def resumo(real, pd, pm, massa, mreal):
    r = {"direto": metricas(real, pd), "m×d": metricas(real, pm)}
    r["mape_massa"] = float(np.mean(np.abs(massa - mreal) / mreal) * 100)
    r["vies_massa_g"] = float(np.mean(massa - mreal))
    return r


def avalia(D, m, m_massa=None):
    """Predições em todas as avaliações. Lateral: média em log das predições dos quadros do prato
    ('prato') e também cada quadro como se fosse uma foto isolada ('quadro')."""
    out, prev = {}, {}
    for s in ["teste", "n5k_v1", "calib"] + BR:
        keys, _, Y = D.aval[s]
        Xz = np.stack([D.Xo[D.idx_o[k]] for k in keys])
        T = m.predict(Xz)
        if m_massa is not None:
            T[:, 0] = m_massa.predict(Xz)
        pd, pm, massa, dens = proteinas(T)
        out[s] = resumo(Y[:, 1], pd, pm, massa, Y[:, 0])
        prev[s] = dict(massa=massa, dens=dens, direto=pd, real=Y[:, 1], mreal=Y[:, 0])
        if s in BR:
            continue
        # quadros laterais dos mesmos pratos
        tem = [j for j, k in enumerate(keys) if k in D.lat]
        Tq, jq, Tp = [], [], []
        for j in tem:
            Xl = D.Xs[D.lat[keys[j]]]
            Tl = m.predict(Xl)
            if m_massa is not None:
                Tl[:, 0] = m_massa.predict(Xl)
            Tq.append(Tl); jq += [j] * len(Tl); Tp.append(Tl.mean(0))
        Tq, Tp, jq, tem = np.concatenate(Tq), np.stack(Tp), np.array(jq), np.array(tem)
        a = proteinas(Tq); b = proteinas(Tp)
        out[s + " lateral/quadro"] = resumo(Y[jq, 1], a[0], a[1], a[2], Y[jq, 0])
        out[s + " lateral/prato"] = resumo(Y[tem, 1], b[0], b[1], b[2], Y[tem, 0])
        out[s + " zenital (mesmos pratos)"] = resumo(Y[tem, 1], pd[tem], pm[tem], massa[tem], Y[tem, 0])
        prev[s + " lateral/prato"] = dict(massa=b[2], idx=tem, direto=b[0])
        prev[s + " lateral/quadro"] = dict(massa=a[2], idx=jq, direto=a[0])
    br_real = np.concatenate([prev[s]["real"] for s in BR])
    for k in ("direto", "m×d"):
        est = np.concatenate([prev[s]["direto" if k == "direto" else "massa"] * (1 if k == "direto" else prev[s]["dens"]) for s in BR])
        out.setdefault("brasileiros", {})[k] = metricas(br_real, est)
    mr = np.concatenate([prev[s]["mreal"] for s in BR]); me = np.concatenate([prev[s]["massa"] for s in BR])
    out["brasileiros"]["mape_massa"] = float(np.mean(np.abs(me - mr) / mr) * 100)
    return out, prev


def hibrido(D, prev, llm):
    """C: massa do modelo × densidade do LLM (o LLM viu a foto zenital/da fonte). Blend em log com pesos do calib."""
    H = {}
    for s in ["teste", "n5k_v1", "calib"] + BR:
        pids = D.aval[s][1]
        mg = np.array([llm.get(p, (np.nan, np.nan))[0] for p in pids])
        pl = np.array([llm.get(p, (np.nan, np.nan))[1] for p in pids])
        H[s] = dict(real=prev[s]["real"], llm=pl, mm=prev[s]["massa"], ml=mg, dl=pl / np.maximum(mg, 1), dm=prev[s]["dens"])
    F = lambda h: np.c_[np.log(h["mm"]), np.log(np.maximum(h["ml"], 1)), np.log(h["dl"] + EPS_D),
                        np.log(h["dm"] + EPS_D), np.ones(len(h["real"]))]
    c = H["calib"]; v = ~np.isnan(c["llm"])
    w, *_ = np.linalg.lstsq(F(c)[v], np.log1p(c["real"][v]), rcond=None)
    out = {}
    for s, h in H.items():
        var = {"LLM sozinho": h["llm"], "massa modelo × dens LLM": h["mm"] * h["dl"],
               "massa √(modelo·LLM) × dens LLM": np.sqrt(h["mm"] * h["ml"]) * h["dl"],
               "blend log-linear (calib)": np.expm1(F(h) @ w).clip(0)}
        for k, est in var.items():
            v = ~np.isnan(est)
            out.setdefault(k, {})[s] = metricas(h["real"][v], est[v])
    for k in out:
        v = np.concatenate([np.c_[H[s]["real"], {"LLM sozinho": H[s]["llm"], "massa modelo × dens LLM": H[s]["mm"] * H[s]["dl"],
                            "massa √(modelo·LLM) × dens LLM": np.sqrt(H[s]["mm"] * H[s]["ml"]) * H[s]["dl"],
                            "blend log-linear (calib)": np.expm1(F(H[s]) @ w).clip(0)}[k]] for s in BR])
        v = v[~np.isnan(v[:, 1])]
        out[k]["brasileiros"] = metricas(v[:, 0], v[:, 1])
    # massa prevista a partir dos quadros LATERAIS × densidade do LLM (o LLM viu a foto zenital;
    # a densidade é identificação do alimento, pouco dependente do ângulo)
    for s in ("teste", "n5k_v1"):
        L = prev[s + " lateral/prato"]; h = H[s]; j = L["idx"]
        est = L["massa"] * h["dl"][j]; v = ~np.isnan(est)
        out.setdefault("massa lateral × dens LLM", {})[s] = metricas(h["real"][j][v], est[v])
        est = np.sqrt(L["massa"] * h["ml"][j]) * h["dl"][j]; v = ~np.isnan(est)
        out.setdefault("massa √(lateral·LLM) × dens LLM", {})[s] = metricas(h["real"][j][v], est[v])
        est = h["llm"][j]; v = ~np.isnan(est)
        out.setdefault("LLM sozinho (pratos com vídeo)", {})[s] = metricas(h["real"][j][v], est[v])
        # um único quadro lateral por vez (o caso do celular: uma foto só)
        Q = prev[s + " lateral/quadro"]; j = Q["idx"]
        est = Q["massa"] * h["dl"][j]; v = ~np.isnan(est)
        out.setdefault("massa lateral (1 quadro) × dens LLM", {})[s] = metricas(h["real"][j][v], est[v])
        est = np.sqrt(Q["massa"] * h["ml"][j]) * h["dl"][j]; v = ~np.isnan(est)
        out.setdefault("massa √(lateral 1 quadro·LLM) × dens LLM", {})[s] = metricas(h["real"][j][v], est[v])
    return out


def massa_ecustfd(D, X, Y, dish):
    """Ridge só de log(massa) com as linhas do ECUSTFD somadas (alimento único, celular, com moeda).
    Grupos: cadeia (N5k) e objeto (ECUSTFD), para o alfa não vazar."""
    ecu = ecustfd()
    Xe = np.stack([D.Xs[i] for i, _ in D.ecu_rows])
    ye = np.array([ecu[k][1] for _, k in D.ecu_rows])
    ge = [10 ** 7 + zlib.crc32(ecu[k][0].encode()) % 10 ** 6 for _, k in D.ecu_rows]
    g = np.r_[[D.cadeia[d] for d in dish], ge]
    XX = np.r_[X, Xe]; TT = np.r_[np.log(Y[:, 0]), np.log(ye)][:, None]
    cv = list(GroupKFold(5).split(XX, groups=g))
    r = RidgeMulti(cv).fit(XX, TT)

    class M:
        def predict(self, Z):
            return r.predict(Z)[:, 0]
    return M()


def acertos(D, prev, llm):
    """Vetores de acerto por item (para bootstrap pareado entre configurações)."""
    o = {}
    o["teste zenital direto"] = ok(prev["teste"]["real"], prev["teste"]["direto"])
    Q = prev["teste lateral/quadro"]
    o["teste lateral 1 quadro direto"] = ok(prev["teste"]["real"][Q["idx"]], Q["direto"])
    o["_grupo teste lateral 1 quadro"] = Q["idx"]
    br_r = np.concatenate([prev[s]["real"] for s in BR]); br_d = np.concatenate([prev[s]["direto"] for s in BR])
    o["brasileiros direto"] = ok(br_r, br_d)
    for s, nome in (("teste", "teste C √(modelo·LLM) zenital"), ("liv_up", "Liv Up C √(modelo·LLM)"), ("ufpr", "UFPR C √(modelo·LLM)")):
        pids = D.aval[s][1]
        mg = np.array([llm[p][0] for p in pids]); pl = np.array([llm[p][1] for p in pids])
        o[nome] = ok(prev[s]["real"], np.sqrt(prev[s]["massa"] * mg) * pl / np.maximum(mg, 1))
    pids = D.aval["teste"][1]
    mg = np.array([llm[p][0] for p in pids]); pl = np.array([llm[p][1] for p in pids])
    o["teste C √(lateral 1 quadro·LLM)"] = ok(prev["teste"]["real"][Q["idx"]], np.sqrt(Q["massa"] * mg[Q["idx"]]) * pl[Q["idx"]] / np.maximum(mg[Q["idx"]], 1))
    o["_grupo teste C √(lateral 1 quadro·LLM)"] = Q["idx"]
    return o


def bootstrap(a, b, n=4000, seed=0):
    """Diferença b−a (pontos) com IC 95% por bootstrap pareado, reamostrando PRATOS (grupo)."""
    out = {}
    rng = np.random.default_rng(seed)
    for k in a:
        if k.startswith("_"):
            continue
        g = a.get("_grupo " + k, np.arange(len(a[k])))
        ug = np.unique(g); pos = {u: np.where(g == u)[0] for u in ug}
        d = []
        for _ in range(n):
            ix = np.concatenate([pos[u] for u in rng.choice(ug, len(ug))])
            d.append(100 * (b[k][ix].mean() - a[k][ix].mean()))
        out[k] = [100 * a[k].mean(), 100 * b[k].mean(), 100 * (b[k].mean() - a[k].mean()), *np.percentile(d, [2.5, 97.5])]
        print(f"  {k:40s} {out[k][0]:5.1f}% → {out[k][1]:5.1f}%  Δ {out[k][2]:+5.1f} (IC95% {out[k][3]:+.1f} a {out[k][4]:+.1f})")
    return out


def linha(nome, r, h=None):
    t = r["teste"]; v1 = r["n5k_v1"]
    tl = r["teste lateral/prato"]; tq = r["teste lateral/quadro"]
    s = (f"{nome[:60]:60s} | teste zen {t['direto']['crit']:5.1f}/{t['m×d']['crit']:5.1f}% MAPE {t['mape_massa']:3.0f}%"
         f" | teste lat/prato {tl['direto']['crit']:5.1f}/{tl['m×d']['crit']:5.1f}% MAPE {tl['mape_massa']:3.0f}%"
         f" | lat/quadro {tq['direto']['crit']:5.1f}% MAPE {tq['mape_massa']:3.0f}%"
         f" | v1 {v1['direto']['crit']:3.0f}/{v1['m×d']['crit']:3.0f}%"
         f" | UFPR {r['ufpr']['direto']['crit']:4.1f}% ({r['ufpr']['mape_massa']:.0f}%) LivUp {r['liv_up']['direto']['crit']:4.1f}% ({r['liv_up']['mape_massa']:.0f}%, viés {r['liv_up']['vies_massa_g']:+.0f} g)"
         f" Seara {r['seara']['direto']['crit']:4.1f}% BR {r['brasileiros']['direto']['crit']:4.1f}/{r['brasileiros']['m×d']['crit']:4.1f}%")
    if h:
        s += "\n      C: " + " | ".join(f"{k}: teste {v['teste']['crit']:.1f}% v1 {v['n5k_v1']['crit']:.0f}% BR {v['brasileiros']['crit']:.1f}% (UFPR {v['ufpr']['crit']:.0f}, LivUp {v['liv_up']['crit']:.0f})"
                                for k, v in h.items() if "brasileiros" in v and k != "LLM sozinho")
        s += "\n      C lateral (pratos com vídeo): " + " | ".join(f"{k}: teste {h[k]['teste']['crit']:.1f}% v1 {h[k]['n5k_v1']['crit']:.0f}%"
                                for k in ("massa lateral × dens LLM", "massa √(lateral·LLM) × dens LLM", "LLM sozinho (pratos com vídeo)",
                  "massa lateral (1 quadro) × dens LLM", "massa √(lateral 1 quadro·LLM) × dens LLM"))
    print(s, flush=True)


def main(bb="siglip2_b16"):
    t0 = time.time()
    D = Dados(bb)
    cfgs = configs(D)
    print(f"universo com metadado: {len(D.meta)} | avaliação: {len(D.ev)} | estrito (regra original): {len(D.estrito_orig)}")
    for k, (po, ps) in cfgs.items():
        nq = sum(len(D.lat.get(d, [])) for d in ps)
        print(f"  {k:62s} pratos zenitais {len(po):5d} | pratos laterais {len(ps):5d} ({nq} quadros) | pratos distintos {len(set(po) | set(ps))}")
        assert not (set(po) | set(ps)) & D.ev
    res = {"bb": bb, "n": {k: [len(po), len(ps), sum(len(D.lat.get(d, [])) for d in ps)] for k, (po, ps) in cfgs.items()}}
    print("\n# Seleção por CV agrupada no treino (MSE em log; menor é melhor)")
    if os.environ.get("SEM_CV"):  # reaproveita a CV já gravada (mesma semente/folds)
        with open(os.path.join(TD, "resultados", f"lateral_{bb}.json")) as f:
            res["cv"] = json.load(f)["cv"]
    else:
        res["cv"] = cv_selecao(D, cfgs)
    escolha = min(res["cv"], key=lambda k: np.mean([res["cv"][k]["zenital"][1], res["cv"][k]["lateral"][1]]))
    res["escolha_cv"] = escolha
    print("escolhida pela CV (média log proteína zenital+lateral):", escolha)

    print("\n# Avaliação (direto / m×d)")
    llm = llm_por_prato()
    res["aval"], res["hibrido"] = {}, {}
    for nome, (po, ps) in cfgs.items():
        X, Y, dish, _ = monta(D, po, ps)
        m = ajusta(D, X, Y, dish)
        r, prev = avalia(D, m)
        h = hibrido(D, prev, llm)
        if nome.startswith("1 ") or nome == escolha:
            res.setdefault("_acertos", {})[nome] = acertos(D, prev, llm)
        res["aval"][nome], res["hibrido"][nome] = r, h
        linha(nome, r, h)
        if nome in ("1' zenital (regra universal)", escolha, "2d zenital + lateral, + pratos test/sem split fora da avaliação"):
            me = massa_ecustfd(D, X, Y, dish)
            r2, prev2 = avalia(D, m, me)
            h2 = hibrido(D, prev2, llm)
            k2 = nome + " + ECUSTFD (só massa)"
            res["aval"][k2], res["hibrido"][k2] = r2, h2
            linha(k2, r2, h2)
    print(f"\n# Bootstrap pareado: '1 zenital (regra original)' → '{escolha}'")
    ac = res.pop("_acertos")
    res["bootstrap"] = bootstrap(ac["1 zenital (regra original, 1.506)"], ac[escolha])
    res["llm"] = {s: res["hibrido"][escolha]["LLM sozinho"][s] for s in ["teste", "n5k_v1", "calib", "brasileiros"] + BR}
    print("LLM sozinho:", {s: round(v["crit"], 1) for s, v in res["llm"].items()})
    res["seg"] = time.time() - t0
    os.makedirs(os.path.join(TD, "resultados"), exist_ok=True)
    with open(os.path.join(TD, "resultados", f"lateral_{bb}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float, ensure_ascii=False)
    print(f"{res['seg'] / 60:.1f} min")


if __name__ == "__main__":
    main(*sys.argv[1:])
