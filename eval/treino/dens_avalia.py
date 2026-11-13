#!/usr/bin/env python3
"""Seção 9, passos 2 e 4: escolha da densidade por CV no treino, pipeline só de imagem, híbridos com o
LLM (pesos e forma só no calib), avaliação única no teste e bootstrap pareado contra a seção 8.

Uso: [DENS_V3_RODADA=1|2] python3 dens_avalia.py [--so-cv | --ensaio] [--sem-gpu]
  --sem-gpu  deixa de fora as configurações e os conjuntos de GPU (seção 10): refaz o estado da seção 9
  DENS_V3_RODADA  usa só essa rodada do LLM v3 (padrão: média de todas as rodadas do arquivo); grava avalia_v3_r<N>.json
  --so-cv   para antes de ler o gabarito do teste (usado enquanto as configurações eram montadas)
  --ensaio  roda a parte de relato no n5k_v1 (50 pratos) em vez do teste, só para conferir que o código roda
Lê o banco da seção 8 (zen_banco), o banco de densidade (resultados/dens/banco, de dens_experimentos.py),
as estimativas do LLM v2 e, se existir, docs/dados/precisao/estimativas_v3_partes.csv (prompt_v3.md).
Grava <TD>/resultados/dens/avalia.json e imprime as tabelas.

Regras, declaradas antes de olhar o teste (as da seção 8, mais as de densidade):
  R1  modelo de MASSA: a regra da seção 8 (menor RMSE de log-massa na CV), aplicada a todas as configurações e
      conjuntos do banco, inclusive os de GPU (seção 10; conjuntos de kaggle/conjuntos_gpu.json). Sem os modelos
      de GPU, a regra escolhe o conjunto de 4 membros da seção 8;
  R2  modelo de DENSIDADE: maior critério do produto na CV do treino, ponderado por faixa, calculado com a
      massa REAL × densidade prevista (isola a densidade). Candidatos: todas as cabeças de densidade da seção 8,
      todas as do banco de densidade (rota por frações e cabeça de log-densidade), as médias declaradas em
      MEDIAS e a seleção gulosa de conjunto (avaliada com ajuste cruzado: escolhida em 4 folds, aplicada no 5º);
  R3  desencolhimento (esticar em log, em torno da média, a densidade escolhida e, depois, a proteína final do
      pipeline só de imagem): cada um só é adotado se melhorar a CV com ajuste cruzado em mais de 0,5 ponto;
  R4  pipeline SÓ DE IMAGEM: maior critério ponderado na CV entre (a) todas as configurações e rotas do banco,
      (b) massa escolhida em R1 × a densidade escolhida em R2/R3 e (c) a média geométrica de (b) com a rota direta
      da configuração que a regra da seção 8 escolhe no banco (maior critério de CV entre configurações e rotas). Só a densidade escolhida concorre: com dezenas de densidades quase empatadas, escolher também entre
      elas pela CV do produto seria escolher ruído. As outras densidades aparecem na tabela, só como relato;
  R5  HÍBRIDO: pesos só no calib; forma escolhida pelo critério no calib com validação deixa-fora em 12 blocos;
      empate -> a forma com menos pesos. As formas com a densidade do LLM v3 já estão declaradas em formas_v3()
      e só entram quando o arquivo de estimativas existir.
  Todas as configurações são avaliadas no teste e todas aparecem nas tabelas; nenhuma escolha usa o teste.
  O bloco "4d" (LLM v3) é só relato, acrescentado depois de as estimativas v3 chegarem: não muda nenhuma escolha.
"""
import glob, json, os, sys
import numpy as np
from zen_comum import Dados, proteinas, crit, mape, bootstrap, ic_simples, ok, EPS_D, I_M, I_D, I_MG
from dens_comum import Ingr, FONTES, BANCO_D, faixas, grava_json, llm_v2_fontes, llm_v3
import zen_avalia as ZA

SPL = ["calib", "teste", "n5k_v1"]
TAG = ("r" + os.environ["DENS_V3_RODADA"]) if os.environ.get("DENS_V3_RODADA") else "media"   # que rodadas do LLM v3 entram
SEM_GPU = "--sem-gpu" in sys.argv        # refaz o estado da seção 9 (sem os modelos de GPU da seção 10)
SUF = "_sem_gpu" if SEM_GPU else ""      # sufixo dos arquivos de saída
SEL8_IMG, SEL8_ROTA = "média: ajuste fino + siglip2-512", "geo"
SEL8_MASSA = "média: ajuste fino + mlp + ridge o8 com laterais + siglip2-512 (sem TIPS)"
SEL8_HIB = "massa^a × dens (LLM^b·modelo^(1−b))"
CONJ_GPU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle", "conjuntos_gpu.json")   # conjuntos de GPU, declarados antes dos resultados
FL2 = "soft-mlp fonte-l2 +perda-dens peso-faixa"
# médias (em log) declaradas; só entram as que tiverem todos os membros no banco
MEDIAS = {
    "média: frações-l2 512 + sua cabeça log-dens": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} siglip2_b16_512 [log-dens]"],
    "média: frações-l2 512 + knn 512": [f"{FL2} siglip2_b16_512 [frações]", "knn densidade siglip2_b16_512"],
    "média: frações-l2 512 + knn 512 + mlp 224 (s8)": [f"{FL2} siglip2_b16_512 [frações]", "knn densidade siglip2_b16_512", "s8: mlp siglip2_b16 o8"],
    "média: frações-l2 512 + frações-l2 224": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} siglip2_b16 [frações]"],
    "média: frações-l2 512 + 224 + ambos": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} siglip2_b16 [frações]", f"{FL2} ambos [frações]"],
    "média: frações-l2 512 + ambos": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} ambos [frações]"],
    "média: frações-l2 512, 3 sementes": ["soft-mlp fonte-l2 siglip2_b16_512 [frações]", "soft-mlp fonte-l2 semente1 siglip2_b16_512 [frações]",
                                          "soft-mlp fonte-l2 semente2 siglip2_b16_512 [frações]"],
    "média: frações-l2 (512, 224, ambos) + knn 512 + ridge fração por fonte 512": [
        f"{FL2} siglip2_b16_512 [frações]", f"{FL2} siglip2_b16 [frações]", f"{FL2} ambos [frações]", "knn densidade siglip2_b16_512",
        "ridge fração por fonte siglip2_b16_512"],
    "média: frações-l2 512, 3 sementes (+perda-dens peso-faixa)": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} semente1 siglip2_b16_512 [frações]",
                                                                    f"{FL2} semente2 siglip2_b16_512 [frações]"],
    "média: frações-l2 512, 3 sementes + knn 512 + mlp 224 (s8)": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} semente1 siglip2_b16_512 [frações]",
                                                                   f"{FL2} semente2 siglip2_b16_512 [frações]", "knn densidade siglip2_b16_512",
                                                                   "s8: mlp siglip2_b16 o8"],
    "média: frações-l2 512 + classes finas-l2 512": [f"{FL2} siglip2_b16_512 [frações]", "soft-mlp fina-l2 +perda-dens peso-faixa siglip2_b16_512 [frações]"],
    "média: frações-l2 512 + classes finas-l2 512 + knn 512 + mlp 224 (s8)": [f"{FL2} siglip2_b16_512 [frações]", "soft-mlp fina-l2 +perda-dens peso-faixa siglip2_b16_512 [frações]",
                                                                              "knn densidade siglip2_b16_512", "s8: mlp siglip2_b16 o8"],
    "média: frações-l2 512 + frações-l2 TIPS": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} tips_b14 [frações]"],
    "média: frações-l2 512 + frações-l2 TIPS + knn 512": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} tips_b14 [frações]", "knn densidade siglip2_b16_512"],
    "média: frações-l2 512 + frações-l2 TIPS + knn 512 + knn TIPS + mlp 224 (s8)": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} tips_b14 [frações]", "knn densidade siglip2_b16_512",
                                                                                   "knn densidade tips_b14", "s8: mlp siglip2_b16 o8"],
    "média: frações-l2 512 + knn TIPS": [f"{FL2} siglip2_b16_512 [frações]", "knn densidade tips_b14"],
    "média: frações-l2 512 + knn TIPS + mlp 224 (s8)": [f"{FL2} siglip2_b16_512 [frações]", "knn densidade tips_b14", "s8: mlp siglip2_b16 o8"],
    "média: frações-l2 512 + classes finas-l2 512 + knn TIPS + mlp 224 (s8)": [f"{FL2} siglip2_b16_512 [frações]", "soft-mlp fina-l2 +perda-dens peso-faixa siglip2_b16_512 [frações]",
                                                                               "knn densidade tips_b14", "s8: mlp siglip2_b16 o8"],
    "média: frações-l2 512 + frações-l2 TIPS + knn TIPS": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} tips_b14 [frações]", "knn densidade tips_b14"],
    "média: frações-l2 (512, TIPS, TIPS|512) + knn TIPS + mlp 224 (s8)": [f"{FL2} siglip2_b16_512 [frações]", f"{FL2} tips_b14 [frações]", f"{FL2} tips_b14|siglip2_b16_512 [frações]",
                                                                          "knn densidade tips_b14", "s8: mlp siglip2_b16 o8"],
    "média: frações-l2 512 + s8 escolhido (ajuste fino + 512)": [f"{FL2} siglip2_b16_512 [frações]", f"s8: {SEL8_IMG}"],
    "média: frações-l2 512 + ajuste fino (s8)": [f"{FL2} siglip2_b16_512 [frações]", "s8: ft cv"],
}
lg = lambda d: np.log(np.maximum(d, 0) + EPS_D)
un = lambda x: np.clip(np.exp(x) - EPS_D, 0, None)


def candidatos(B):
    """nome -> dict(oof, calib, teste, n5k_v1): densidade (g/g) de cada candidato."""
    C = {}
    for nome, b in B.items():
        if "teste" in b and not np.isnan(b["oof"]).any():
            C[f"s8: {nome}"] = {k: proteinas(b[k])["dens"] for k in ["oof"] + SPL}
    for f in sorted(glob.glob(os.path.join(BANCO_D, "*.npz"))):
        nome = os.path.splitext(os.path.basename(f))[0]
        z = dict(np.load(f, allow_pickle=False))
        C[f"{nome} [frações]" if nome.startswith("soft-") else nome] = dict(oof=z["oof_dens"], **{s: z[f"{s}_dens"] for s in SPL})
        if "oof" in z and "só-frações" not in nome:
            C[f"{nome} [log-dens]"] = {k: proteinas(z[k])["dens"] for k in ["oof"] + SPL}
    base = list(C)
    for nome, membros in MEDIAS.items():
        if all(m in C for m in membros):
            C[nome] = {k: un(np.mean([lg(C[m][k]) for m in membros], 0)) for k in ["oof"] + SPL}
    return C, base


def guloso(C, base, yp, ym, w, idx, kmax=8):
    """Seleção gulosa de conjunto (com reposição) nos pratos idx: média em log que maximiza o critério de densidade
    (massa real × densidade); desempate pelo menor erro quadrático ponderado em log. Devolve a lista de membros."""
    L = np.stack([lg(C[m]["oof"])[idx] for m in base])
    y, m_, ww = yp[idx], ym[idx], w[idx]
    alvo = lg(y / m_)
    esc, soma, melhor = [], np.zeros(len(idx)), (-1, 0)
    for k in range(kmax):
        nota = []
        for j in range(len(base)):
            v = (soma + L[j]) / (k + 1)
            nota.append((crit(y, m_ * un(v), ww), -float((((v - alvo) ** 2) * ww).sum() / ww.sum())))
        j = max(range(len(base)), key=lambda i: nota[i])
        if nota[j] <= melhor:
            break
        melhor = nota[j]; esc.append(j); soma += L[j]
    return [base[j] for j in esc]


def estica(d, mu, b, c):
    return un(mu + b * (lg(d) - mu) + c)


def desencolhe(d_oof, yp, ym, w, fold):
    """Grade (b, c) que maximiza o critério de densidade; devolve (b, c, mu, CV com ajuste cruzado)."""
    gb, gc = np.round(np.arange(1.0, 1.41, 0.05), 2), np.round(np.arange(-0.10, 0.11, 0.05), 2)

    def melhor(idx):
        mu = float(np.average(lg(d_oof[idx]), weights=w[idx]))
        return max(((crit(yp[idx], ym[idx] * estica(d_oof[idx], mu, b, c), w[idx]), -abs(b - 1) - abs(c), b, c, mu) for b in gb for c in gc))[2:]
    cruz = np.zeros(len(yp))
    for k in range(5):
        b, c, mu = melhor(np.where(fold != k)[0])
        cruz[fold == k] = estica(d_oof[fold == k], mu, b, c)
    b, c, mu = melhor(np.arange(len(yp)))
    return b, c, mu, crit(yp, ym * cruz, w)


def forma_grade(f, grade, y, splits):
    """f(split, peso) -> estimativa. Peso escolhido no calib por menor erro quadrático de log1p(proteína).
    Devolve (estimativas por split, deixa-fora no calib em 12 blocos, peso)."""
    n = len(y)

    def ajusta(idx):
        if grade is None:
            return None
        e = {w: float(np.mean((np.log1p(f("calib", w))[idx] - y[idx]) ** 2)) for w in grade}
        return min(e, key=e.get)
    w = ajusta(np.arange(n))
    est = {s: f(s, w) for s in splits}
    if grade is None:
        return est, est["calib"], None
    loo = np.zeros(n)
    for b in range(12):
        fora = np.arange(n) % 12 == b
        loo[fora] = f("calib", ajusta(np.where(~fora)[0]))[fora]
    return est, loo, w


G1 = [round(x, 1) for x in np.linspace(0, 1.2, 13)]
G2 = [(a, b) for a in G1 for b in G1[:11]]
G3 = [(a, b) for a in G1[:11] for b in G1[:11] if a + b <= 1.0 + 1e-9]


def main():
    so_cv = "--so-cv" in sys.argv
    D = Dados()
    ZA.DENS_GRUPO = D.Y["pg"].sum(0) / np.maximum(D.Y["mg"].sum(0), 1)
    if not SEM_GPU and os.path.exists(CONJ_GPU):
        ZA.CONJUNTOS.update({k: v for k, v in json.load(open(CONJ_GPU))["conjuntos"].items() if k not in ZA.CONJUNTOS})
    B = ZA.carrega()
    if SEM_GPU:
        B = {n: b for n, b in B.items() if "gpu" not in n}
    yp, ym, w = D.Y["proteina"], D.Y["massa"], D.w_cv
    alto = yp > 15
    res = {}

    # ------------------------------------------------------------ 0. massa: CV no treino (R1)
    completos = [n for n, b in B.items() if "teste" in b and not np.isnan(b["oof"]).any()]
    cv_m = {n: dict(rmse=float(np.sqrt(np.mean((B[n]["oof"][:, I_M] - D.T[:, I_M]) ** 2))), mape=mape(ym, np.exp(B[n]["oof"][:, I_M]))) for n in completos}
    sel_m = min(completos, key=lambda n: cv_m[n]["rmse"])
    res["massa_cv"] = cv_m; res["escolha"] = dict(massa=sel_m)
    print(f"# 0. Massa: CV no treino. Escolhida (R1): '{sel_m}' (RMSE log {cv_m[sel_m]['rmse']:.4f}, MAPE {cv_m[sel_m]['mape']:.1f}%); seção 8: '{SEL8_MASSA}' ({cv_m[SEL8_MASSA]['rmse']:.4f}, {cv_m[SEL8_MASSA]['mape']:.1f}%)")
    for n in sorted(completos, key=lambda n: -cv_m[n]["rmse"]):
        print(f"   {cv_m[n]['rmse']:.4f}  MAPE {cv_m[n]['mape']:5.1f}%  {n}")

    def massa_por_faixa(real_p, real_m, est_m):
        return {k: dict(mape=mape(real_m[f], est_m[f]), vies=float(100 * (np.exp(np.median(np.log(est_m[f] / real_m[f]))) - 1))) for k, f in faixas(real_p).items()}
    res["massa_cv_por_faixa"] = dict(s8=massa_por_faixa(yp, ym, np.exp(B[SEL8_MASSA]["oof"][:, I_M])), escolhida=massa_por_faixa(yp, ym, np.exp(B[sel_m]["oof"][:, I_M])))
    print("   CV por faixa de proteína, MAPE (viés mediano): seção 8", {k: f"{v['mape']:.1f}% ({v['vies']:+.0f}%)" for k, v in res["massa_cv_por_faixa"]["s8"].items()})
    print("                                                  escolhida", {k: f"{v['mape']:.1f}% ({v['vies']:+.0f}%)" for k, v in res["massa_cv_por_faixa"]["escolhida"].items()})
    print(f"   inclinação de log massa prevista × real: seção 8 {np.polyfit(D.T[:, I_M], B[SEL8_MASSA]['oof'][:, I_M], 1)[0]:.3f} | escolhida {np.polyfit(D.T[:, I_M], B[sel_m]['oof'][:, I_M], 1)[0]:.3f}\n")

    # ------------------------------------------------------------ 1. densidade: CV no treino
    C, base = candidatos(B)
    cv = {}
    for nome, c in C.items():
        e = lg(c["oof"]) - lg(yp / ym)
        cv[nome] = dict(rmse=float(np.sqrt((e ** 2).mean())), rmse15=float(np.sqrt((e[alto] ** 2).mean())),
                        crit=crit(yp, ym * c["oof"], w), crit15=crit(yp[alto], (ym * c["oof"])[alto], w[alto]))
    # seleção gulosa de conjunto, com ajuste cruzado
    cruz = np.zeros(len(yp))
    for k in range(5):
        mem = guloso(C, base, yp, ym, w, np.where(D.fold != k)[0])
        cruz[D.fold == k] = un(np.mean([lg(C[m]["oof"][D.fold == k]) for m in mem], 0))
    mem = guloso(C, base, yp, ym, w, np.arange(len(yp)))
    NG = "seleção gulosa de conjunto"
    C[NG] = {k: un(np.mean([lg(C[m][k]) for m in mem], 0)) for k in SPL}
    C[NG]["oof"] = cruz        # fora-do-fold honesto: membros escolhidos sem o fold
    e = lg(cruz) - lg(yp / ym)
    cv[NG] = dict(rmse=float(np.sqrt((e ** 2).mean())), rmse15=float(np.sqrt((e[alto] ** 2).mean())), crit=crit(yp, ym * cruz, w),
                  crit15=crit(yp[alto], (ym * cruz)[alto], w[alto]), membros=mem)
    sel_d = max(cv, key=lambda n: (cv[n]["crit"], -cv[n]["rmse"]))
    print("# 1. Densidade: CV no treino (critério = massa REAL × densidade prevista, ponderado por faixa)")
    print(f"{'candidato':78s} | RMSE log-dens (todos / >15 g) | critério (todos / >15 g)")
    for nome in sorted(cv, key=lambda n: cv[n]["crit"]):
        print(f"{nome[:78]:78s} | {cv[nome]['rmse']:.4f} / {cv[nome]['rmse15']:.4f}            | {cv[nome]['crit']:5.1f}% / {cv[nome]['crit15']:5.1f}%"
              + ("   <- escolhido (R2)" if nome == sel_d else ""))
    print(f"seleção gulosa (todos os folds): {mem}")
    res["dens_cv"] = cv; res["escolha"]["densidade"] = sel_d
    # R3: desencolhimento
    b, c, mu, cv_est = desencolhe(C[sel_d]["oof"], yp, ym, w, D.fold)
    adota = cv_est > cv[sel_d]["crit"] + 0.5
    res["desencolhimento"] = dict(b=b, c=c, cv_sem=cv[sel_d]["crit"], cv_com_ajuste_cruzado=cv_est, adotado=bool(adota))
    print(f"\ndesencolhimento da densidade escolhida: b={b:.2f}, c={c:+.2f}; CV {cv[sel_d]['crit']:.1f}% -> {cv_est:.1f}% (ajuste cruzado): {'ADOTADO' if adota else 'não adotado'}")
    Dsel = C[sel_d]
    if adota:   # só vira candidata (e entra nas tabelas) se a regra R3 adotar
        NE = sel_d + " + desencolhimento"
        C[NE] = {k: estica(C[sel_d][k], mu, b, c) for k in ["oof"] + SPL}
        e = lg(C[NE]["oof"]) - lg(yp / ym)
        cv[NE] = dict(rmse=float(np.sqrt((e ** 2).mean())), rmse15=float(np.sqrt((e[alto] ** 2).mean())), crit=cv_est, crit15=float("nan"))
        Dsel = C[NE]; res["escolha"]["densidade"] = NE
    # o que mudou na CV, por faixa e por fonte principal (seção 8 escolhida × densidade escolhida)
    I = Ingr(D.ids)
    d8 = C[f"s8: {SEL8_IMG}"]["oof"]
    quem = dict(faixas(yp))
    quem.update({f"fonte {FONTES[j]} (>15 g)": (I.principal == j) & alto for j in range(len(FONTES) - 1)})
    print("\nCV, massa real × densidade: seção 8 escolhida -> densidade escolhida (viés mediano da densidade)")
    res["cv_antes_depois"] = {}
    for k, f in quem.items():
        if f.sum() < 10:
            continue
        vb = lambda d: 100 * (np.exp(np.median(lg(d[f]) - lg((yp / ym)[f]))) - 1)
        r = dict(n=int(f.sum()), antes=crit(yp[f], (ym * d8)[f]), depois=crit(yp[f], (ym * Dsel["oof"])[f]), vies_antes=float(vb(d8)), vies_depois=float(vb(Dsel["oof"])))
        res["cv_antes_depois"][k] = r
        print(f"   {k:34s} n={r['n']:4d}  {r['antes']:5.1f}% -> {r['depois']:5.1f}%   (viés {r['vies_antes']:+.0f}% -> {r['vies_depois']:+.0f}%)")
    x, y = lg(yp / ym), lg(Dsel["oof"])
    res["encolhimento_depois"] = dict(todos=float(np.polyfit(x, y, 1)[0]), acima15=float(np.polyfit(x[alto], y[alto], 1)[0]),
                                      r_acima15=float(np.corrcoef(x[alto], y[alto])[0, 1]))
    print("   inclinação log dens prevista × real:", res["encolhimento_depois"])

    # ------------------------------------------------------------ 2. só imagem: CV
    m8 = {k: np.exp(B[SEL8_MASSA][k][:, I_M]) for k in ["oof"] + SPL}        # massa da seção 8 (só referência)
    mS = {k: np.exp(B[sel_m][k][:, I_M]) for k in ["oof"] + SPL}             # massa escolhida (R1)
    R8 = {k: ZA.rotas(B[SEL8_IMG][k]) for k in ["oof"] + SPL}
    IMG = {}        # nome -> dict(oof, calib, teste, n5k_v1) proteína
    for nome in completos:
        for r in ZA.ROTAS:
            IMG[f"s8: {nome} | {r}"] = {k: ZA.rotas(B[nome][k])[r] for k in ["oof"] + SPL}
    # configuração que a regra da seção 8 escolhe no banco (config, rota): dela sai a rota direta da média geométrica
    zen_img = max(IMG, key=lambda n: crit(yp, IMG[n]["oof"], w)).split(" | ")[0][len("s8: "):]
    Rdir = {k: ZA.rotas(B[zen_img][k])["direto"] for k in ["oof"] + SPL}
    res["escolha"]["direta_de"] = zen_img
    for nome, c in C.items():
        IMG[f"massa × {nome}"] = {k: mS[k] * c[k] for k in ["oof"] + SPL}
        IMG[f"geo(direta, massa × {nome})"] = {k: np.sqrt(Rdir[k] * mS[k] * c[k]) for k in ["oof"] + SPL}
    cv_img = {n: crit(yp, v["oof"], w) for n, v in IMG.items()}
    nd = res["escolha"]["densidade"]
    concorrem = [n for n in IMG if n.startswith("s8: ")] + [f"massa × {nd}", f"geo(direta, massa × {nd})"]
    sel_i = max(concorrem, key=cv_img.get)
    ref_i = f"s8: {SEL8_IMG} | {SEL8_ROTA}"
    # R3 também na saída final: esticar a proteína prevista em torno da média (mesma regra de adoção)
    um = np.ones(len(yp))
    bp, cp_, mup, cv_p = desencolhe(IMG[sel_i]["oof"], yp, um, w, D.fold)
    adota_p = cv_p > cv_img[sel_i] + 0.5
    res["esticar_proteina"] = dict(b=bp, c=cp_, cv_sem=cv_img[sel_i], cv_com_ajuste_cruzado=cv_p, adotado=bool(adota_p))
    print(f"\nesticar a proteína do só-imagem escolhido: b={bp:.2f}, c={cp_:+.2f}; CV {cv_img[sel_i]:.1f}% -> {cv_p:.1f}% (ajuste cruzado): {'ADOTADO' if adota_p else 'não adotado'}")
    if adota_p:
        ne = sel_i + " + proteína esticada"
        IMG[ne] = {k: estica(IMG[sel_i][k], mup, bp, cp_) for k in ["oof"] + SPL}
        cv_img[ne] = cv_p; sel_i = ne
    res["img_cv"] = cv_img; res["escolha"]["imagem"] = sel_i
    print(f"\n# 2. Só imagem: CV ponderada. Seção 8 escolhida: {cv_img[ref_i]:.1f}% | escolhido agora (R4): '{sel_i}' {cv_img[sel_i]:.1f}%")
    print("   por faixa na CV (seção 8 -> agora):", {k: f"{crit(yp[f], IMG[ref_i]['oof'][f]):.0f} -> {crit(yp[f], IMG[sel_i]['oof'][f]):.0f}" for k, f in faixas(yp).items()})
    print(f"   rota direta usada na média geométrica: a de '{zen_img}'")
    for n in concorrem[-2:]:
        print(f"   concorre: {cv_img[n]:5.1f}%  {n}")
    # modelo ÚNICO (um backbone), para o aparelho: maior critério de CV entre as rotas direta, m×d e geo
    unicos = [n for n in completos if not n.startswith("média")]
    grande = lambda n: any(t in n for t in ("_l16", "_l14", "so400m"))
    rota_de = lambda n: max(("direto", "mxd", "geo"), key=lambda r: cv_img[f"s8: {n} | {r}"])
    cvu = {n: cv_img[f"s8: {n} | {rota_de(n)}"] for n in unicos}
    sel_u = max(unicos, key=cvu.get); sel_ub = max((n for n in unicos if not grande(n)), key=cvu.get)
    res["escolha"].update(unico=sel_u, unico_rota=rota_de(sel_u), unico_vitb=sel_ub, unico_vitb_rota=rota_de(sel_ub))
    res["unico_cv"] = {n: dict(rota=rota_de(n), cv=cvu[n], vitb=not grande(n), rmse_massa=cv_m[n]["rmse"]) for n in unicos}
    print(f"   modelo único pela CV: qualquer tamanho '{sel_u}' ({rota_de(sel_u)}, {cvu[sel_u]:.1f}%) | tamanho ViT-B '{sel_ub}' ({rota_de(sel_ub)}, {cvu[sel_ub]:.1f}%)")
    print("   (só relato) as 8 melhores entre todas as densidades:")
    for n in sorted(cv_img, key=cv_img.get, reverse=True)[:8]:
        print(f"   {cv_img[n]:5.1f}%  {n}")

    # ------------------------------------------------------------ 3. híbridos: pesos e forma no calib
    L = ZA.llm_pratos(D)
    real = {s: D.aval[s]["proteina"] for s in SPL}; mreal = {s: D.aval[s]["massa"] for s in SPL}
    yc = np.log1p(real["calib"])
    Pm8 = {s: dict(massa=m8[s], dens=R8[s]["dens"], direto=R8[s][SEL8_ROTA], mg=np.expm1(B[SEL8_MASSA][s][:, I_MG]).clip(0)) for s in SPL}
    est8, loo8 = ZA.hibridos(Pm8, L, real["calib"], None)
    Pm9 = {s: dict(massa=mS[s], dens=Dsel[s], direto=IMG[sel_i][s], mg=np.expm1(B[sel_m][s][:, I_MG]).clip(0)) for s in SPL}
    est9, loo9 = ZA.hibridos(Pm9, L, real["calib"], None)
    n_pesos = {"massa modelo × dens LLM": 0, "massa √(modelo·LLM) × dens LLM": 0, "massa modelo^a·LLM^(1−a) × dens LLM": 1, SEL8_HIB: 2,
               "blend log-linear (5 pesos)": 5, "blend log-linear + proteína direta e do LLM (7 pesos)": 7,
               "média geométrica: proteína direta do modelo e do LLM": 0}
    H = {}          # nome -> dict(est, loo, pesos, n_pesos)
    for nome in n_pesos:
        H[f"[dens s9] {nome}"] = dict(est=est9[nome], loo=loo9[nome], n=n_pesos[nome])
    # formas novas com a densidade escolhida
    dl = {s: L[s]["dens"] for s in SPL}; ll = {s: np.maximum(L[s]["massa"], 1) for s in SPL}
    mix = lambda a, b, w_: un(w_ * lg(a) + (1 - w_) * lg(b))

    def add(nome, f, grade, n):
        est, loo, w_ = forma_grade(f, grade, yc, SPL)
        H[nome] = dict(est=est, loo=loo, n=n, pesos=w_)
    add("[dens s9] massa modelo × dens (LLM^b · modelo^(1−b))", lambda s, b_: mS[s] * mix(dl[s], Dsel[s], b_), G1[:11], 1)
    # por fonte: frações de massa por fonte do LLM e do modelo misturadas × densidade por fonte dita pelo LLM
    ff_nome = max((n for n in base if n.endswith("[frações]") and n.startswith("soft-mlp fonte")), key=lambda n: cv[n]["crit"])
    zf = dict(np.load(os.path.join(BANCO_D, ff_nome[:-len(" [frações]")] + ".npz"), allow_pickle=False))
    res["escolha"]["fracoes_do_modelo"] = ff_nome
    dfm = I.PF.sum(0) / np.maximum(I.MF.sum(0), 1e-9)          # densidade média por fonte no treino
    V2 = {}
    for s in SPL:
        MF, PF = llm_v2_fontes(D.aval[s]["prato"])
        tot = np.maximum(MF.sum(1, keepdims=True), 1)
        V2[s] = dict(FF=MF / tot, DF=np.where(MF > 0, PF / np.maximum(MF, 1e-9), dfm), MF=MF, PF=PF,
                     dens_fam=(PF.sum(1) - PF[:, 0] + MF[:, 0] * dfm[0]) / tot[:, 0])
    add("[dens s9] massa modelo × Σ fonte (fração LLM^w ⊕ modelo) × dens por fonte do LLM",
        lambda s, w_: mS[s] * ((w_ * V2[s]["FF"] + (1 - w_) * zf[f"{s}_ff"]) * V2[s]["DF"]).sum(1), G1[:11], 1)
    add("[dens s9] massa^a × dens (LLM com frango pela média do treino ^b · modelo^(1−b))",
        lambda s, ab: np.exp(ab[0] * np.log(mS[s]) + (1 - ab[0]) * np.log(ll[s])) * mix(V2[s]["dens_fam"], Dsel[s], ab[1]), G2, 2)
    add("[dens s9] massa modelo × dens (LLM com frango pela média do treino ^b · modelo^(1−b))",
        lambda s, b_: mS[s] * mix(V2[s]["dens_fam"], Dsel[s], b_), G1[:11], 1)
    add("[dens s9] massa modelo × dens do modelo (sem LLM)", lambda s, _: mS[s] * Dsel[s], None, 0)

    # "qual é a fonte" do LLM como prior sobre "quanto é fonte" do modelo: a fração de massa que o modelo dá às fontes de
    # proteína é redistribuída entre elas com peso g para a fonte principal dita pelo LLM (g = 1: não muda)
    def com_prior(s, fonte, g):
        ff = zf[f"{s}_ff"].copy()
        pr = np.ones_like(ff[:, :-1]); tem = fonte < len(FONTES) - 1
        pr[np.where(tem)[0], fonte[tem]] = g
        t = ff[:, :-1].sum(1, keepdims=True)
        ff[:, :-1] = t * ff[:, :-1] * pr / np.maximum((ff[:, :-1] * pr).sum(1, keepdims=True), 1e-9)
        return ff @ dfm
    for s in SPL:
        pp, pt = V2[s]["PF"][:, :-1], V2[s]["PF"].sum(1)
        V2[s]["fonte"] = np.where((pp.max(1) >= 0.5 * np.maximum(pt, 1e-9)) & (pt > 5), pp.argmax(1), len(FONTES) - 1)
    GP = [(g, b_) for g in (1.0, 2.0, 4.0, 8.0, 1e6) for b_ in G1[:11]]
    add("[dens s9] massa modelo × dens (LLM^b · frações do modelo com a fonte do LLM como prior^(1−b))",
        lambda s, gb: mS[s] * mix(dl[s], com_prior(s, V2[s]["fonte"], gb[0]), gb[1]), GP, 2)
    # LLM v3 (identificação + partes): formas declaradas antes de os dados existirem
    V3 = {s: llm_v3(D.aval[s]["prato"]) for s in SPL}
    tem_v3 = V3["calib"] is not None and V3["calib"]["tem"].all() and V3["teste"]["tem"].all()
    res["v3_disponivel"] = bool(tem_v3)
    if V3["calib"] is not None and not tem_v3:
        print(f"AVISO: estimativas v3 incompletas (faltam {(~V3['calib']['tem']).sum()} pratos do calib e {(~V3['teste']['tem']).sum()} do teste); as formas [v3] não entram")
    if tem_v3:
        for s in SPL:       # pratos sem estimativa v3 (só no n5k_v1, se não foi estimado) caem na densidade v2
            V3[s]["dens"] = np.where(V3[s]["tem"], V3[s]["dens"], dl[s])
            V3[s]["FF"] = np.where(V3[s]["tem"][:, None], V3[s]["FF"], V2[s]["FF"])
            V3[s]["DFc"] = np.where(V3[s]["tem"][:, None] & ~np.isnan(V3[s]["DF"]), np.nan_to_num(V3[s]["DF"]), np.where(V3[s]["tem"][:, None], dfm, V2[s]["DF"]))
            V3[s]["dens_fam"] = V3[s]["dens"] - V3[s]["FF"][:, 0] * V3[s]["DFc"][:, 0] + V3[s]["FF"][:, 0] * dfm[0]
        d3 = {s: V3[s]["dens"] for s in SPL}
        add("[v3] massa modelo × dens LLM v3", lambda s, _: mS[s] * d3[s], None, 0)
        add("[v3] massa modelo × dens (v3^b · modelo^(1−b))", lambda s, b_: mS[s] * mix(d3[s], Dsel[s], b_), G1[:11], 1)
        add("[v3] massa^a (modelo, LLM v2) × dens (v3^b · modelo^(1−b))",
            lambda s, ab: np.exp(ab[0] * np.log(mS[s]) + (1 - ab[0]) * np.log(ll[s])) * mix(d3[s], Dsel[s], ab[1]), G2, 2)
        add("[v3] massa modelo × dens (v2^b1 · v3^b2 · modelo^(1−b1−b2))",
            lambda s, bb: mS[s] * un(bb[0] * lg(dl[s]) + bb[1] * lg(d3[s]) + (1 - bb[0] - bb[1]) * lg(Dsel[s])), G3, 2)
        add("[v3] massa modelo × Σ fonte (fração v3^w ⊕ modelo) × dens por fonte do v3",
            lambda s, w_: mS[s] * ((w_ * V3[s]["FF"] + (1 - w_) * zf[f"{s}_ff"]) * V3[s]["DFc"]).sum(1), G1[:11], 1)
        add("[v3] massa modelo × dens (v3 com frango pela média do treino ^b · modelo^(1−b))",
            lambda s, b_: mS[s] * mix(V3[s]["dens_fam"], Dsel[s], b_), G1[:11], 1)
        for s in SPL:
            V3[s]["fonte"] = np.where(V3[s]["tem"], V3[s]["fonte"], V2[s]["fonte"])
        add("[v3] massa modelo × dens (v3^b · frações do modelo com a fonte declarada no v3 como prior^(1−b))",
            lambda s, gb: mS[s] * mix(d3[s], com_prior(s, V3[s]["fonte"], gb[0]), gb[1]), GP, 2)
        add("[v3] massa modelo × frações do modelo com a fonte declarada no v3 como prior (sem densidade do LLM)",
            lambda s, g: mS[s] * com_prior(s, V3[s]["fonte"], g), [1.0, 2.0, 4.0, 8.0, 1e6], 1)
    for nome in H:
        H[nome]["calib_loo"] = crit(real["calib"], H[nome]["loo"])
    sel_h = max(H, key=lambda n: (H[n]["calib_loo"], -H[n]["n"]))
    sel_h_v2 = max((n for n in H if not n.startswith("[v3]")), key=lambda n: (H[n]["calib_loo"], -H[n]["n"]))
    # melhor forma COM o v3 (número principal da seção 10 quando o calib prefere uma forma sem v3; até a seção 9, era a própria sel_h)
    sel_h_v3 = max((n for n in H if n.startswith("[v3]")), key=lambda n: (H[n]["calib_loo"], -H[n]["n"])) if tem_v3 else sel_h
    res["escolha"].update(hibrido=sel_h, hibrido_sem_v3=sel_h_v2, hibrido_com_v3=sel_h_v3)
    print(f"\n(LLM v3: {'rodada ' + TAG[1:] if TAG != 'media' else 'média de todas as rodadas do arquivo'})")
    print(f"\n# 3. Híbridos no calib (deixa-fora). Seção 8 escolhida: {crit(real['calib'], loo8[SEL8_HIB]):.1f}%")
    for nome in H:
        print(f"   {H[nome]['calib_loo']:5.1f}%  pesos={H[nome].get('pesos')}  {nome}" + ("   <- escolhida (R5)" if nome == sel_h else "")
              + ("   <- melhor com v3" if nome == sel_h_v3 and sel_h_v3 != sel_h else "") + ("   <- melhor sem v3" if nome == sel_h_v2 and sel_h_v2 != sel_h else ""))
    cc = real["calib"]
    print("   calib, massa real × densidade: LLM v2", round(crit(cc, mreal["calib"] * dl["calib"]), 1), "| modelo s8", round(crit(cc, mreal["calib"] * R8["calib"]["dens"]), 1),
          "| modelo s9", round(crit(cc, mreal["calib"] * Dsel["calib"]), 1), "| LLM v2 com frango pela média", round(crit(cc, mreal["calib"] * V2["calib"]["dens_fam"]), 1),
          ("| LLM v3 " + str(round(crit(cc, mreal["calib"] * V3["calib"]["dens"]), 1))) if tem_v3 else "")
    res["calib"] = dict(s8_hibrido_loo=crit(cc, loo8[SEL8_HIB]), dens_llm_v2=crit(cc, mreal["calib"] * dl["calib"]), dens_s8=crit(cc, mreal["calib"] * R8["calib"]["dens"]),
                        dens_s9=crit(cc, mreal["calib"] * Dsel["calib"]), img_s8=crit(cc, IMG[ref_i]["calib"]), img_s9=crit(cc, IMG[sel_i]["calib"]),
                        por_faixa_dens_s8={k: crit(cc[f], (mreal["calib"] * R8["calib"]["dens"])[f]) for k, f in faixas(cc).items()},
                        por_faixa_dens_s9={k: crit(cc[f], (mreal["calib"] * Dsel["calib"])[f]) for k, f in faixas(cc).items()})
    print("   calib, só imagem: seção 8", round(res["calib"]["img_s8"], 1), "| agora", round(res["calib"]["img_s9"], 1))
    res["hibridos_calib"] = {n: dict(calib_loo=H[n]["calib_loo"], n_pesos=H[n]["n"], pesos=H[n].get("pesos")) for n in H}
    grava_json("avalia_so_cv.json" if so_cv or "--ensaio" in sys.argv else f"avalia{'' if TAG == 'media' else '_v3_' + TAG}{SUF}.json", res)
    if so_cv:
        print("\n(--so-cv: parou antes do teste)")
        return

    # ------------------------------------------------------------ 4. TESTE (só relato)
    te = "n5k_v1" if "--ensaio" in sys.argv else "teste"; rt, mt = real[te], mreal[te]
    fx = faixas(rt)
    pf = lambda est: {k: crit(rt[f], est[f]) for k, f in fx.items()}
    a8_img = ok(rt, IMG[ref_i][te]); a8_hib = ok(rt, est8[SEL8_HIB][te]); a_llm = ok(rt, L[te]["prot"])
    print(f"\n# 4. TESTE. Referências reproduzidas: só imagem s8 {100 * a8_img.mean():.1f}% | híbrido s8 {100 * a8_hib.mean():.1f}% | LLM {100 * a_llm.mean():.1f}%")
    assert te != "teste" or (abs(100 * a8_img.mean() - 61.07) < 0.1 and abs(100 * a8_hib.mean() - 62.86) < 0.1), "referências da seção 8 não reproduzidas"
    res["teste_ref"] = dict(img_s8=100 * a8_img.mean(), hib_s8=100 * a8_hib.mean(), llm=100 * a_llm.mean(), por_faixa_img_s8=pf(IMG[ref_i][te]),
                            por_faixa_hib_s8=pf(est8[SEL8_HIB][te]), por_faixa_llm=pf(L[te]["prot"]),
                            oraculo_massa_real_x_dens_s8=crit(rt, mt * R8[te]["dens"]), oraculo_massa_real_x_dens_llm=crit(rt, mt * dl[te]),
                            oraculo_massa_modelo_x_dens_real=crit(rt, mS[te] * rt / mt), oraculo_massa_s8_x_dens_real=crit(rt, m8[te] * rt / mt))
    # pipelines escolhidos até a seção 9 (antes dos modelos de GPU): previsões por prato gravadas antes da ingestão
    ref = dict(np.load(os.path.join(os.path.dirname(BANCO_D), "ref_pre_gpu.npz"), allow_pickle=False))
    a9_img, a9_hv2, a9_hv3 = ok(rt, ref[f"{te}|img_s9"]), ok(rt, ref[f"{te}|hib_v2"]), ok(rt, ref[f"{te}|hib_v3_{TAG}"])
    print(f"   pipelines da seção 9 (sem GPU): só imagem {100 * a9_img.mean():.1f}% | híbrido v2 {100 * a9_hv2.mean():.1f}% | híbrido v3 ({TAG}) {100 * a9_hv3.mean():.1f}%")
    assert te != "teste" or (abs(100 * a9_img.mean() - 64.64) < 0.1 and abs(100 * a9_hv2.mean() - 67.14) < 0.1
                             and abs(100 * a9_hv3.mean() - dict(media=71.43, r1=70.36, r2=68.57)[TAG]) < 0.1), "referências da seção 9 não reproduzidas"
    res["teste_ref"].update(img_s9=100 * a9_img.mean(), hib_v2_s9=100 * a9_hv2.mean(), hib_v3_s9=100 * a9_hv3.mean(), por_faixa_img_s9=pf(ref[f"{te}|img_s9"]),
                            por_faixa_hib_v2_s9=pf(ref[f"{te}|hib_v2"]), por_faixa_hib_v3_s9=pf(ref[f"{te}|hib_v3_{TAG}"]))
    # massa no teste: seção 8 × escolhida
    res["massa_teste"] = dict(escolhida=sel_m, mape_s8=mape(mt, m8[te]), mape_escolhida=mape(mt, mS[te]), por_faixa_s8=massa_por_faixa(rt, mt, m8[te]),
                              por_faixa_escolhida=massa_por_faixa(rt, mt, mS[te]), massa_x_dens_real_s8=dict(todos=crit(rt, m8[te] * rt / mt), **pf(m8[te] * rt / mt)),
                              massa_x_dens_real_escolhida=dict(todos=crit(rt, mS[te] * rt / mt), acima15=crit(rt[rt > 15], (mS[te] * rt / mt)[rt > 15]), **pf(mS[te] * rt / mt)),
                              vs_s8_com_dens_real=bootstrap(ok(rt, m8[te] * rt / mt), ok(rt, mS[te] * rt / mt)),
                              rmse_log_s8=float(np.sqrt(np.mean(np.log(m8[te] / mt) ** 2))), rmse_log_escolhida=float(np.sqrt(np.mean(np.log(mS[te] / mt) ** 2))),
                              n5k_v1_mape_s8=mape(mreal["n5k_v1"], m8["n5k_v1"]), n5k_v1_mape_escolhida=mape(mreal["n5k_v1"], mS["n5k_v1"]))
    mte = res["massa_teste"]
    print(f"\n## 4.0 Massa no teste: '{sel_m}'. MAPE {mte['mape_s8']:.1f}% (s8) -> {mte['mape_escolhida']:.1f}% | RMSE log {mte['rmse_log_s8']:.4f} -> {mte['rmse_log_escolhida']:.4f}"
          f" | massa × dens real: {mte['massa_x_dens_real_s8']['todos']:.1f}% -> {mte['massa_x_dens_real_escolhida']['todos']:.1f}%"
          f" (Δ {mte['vs_s8_com_dens_real'][0]:+.1f}, {mte['vs_s8_com_dens_real'][1]:+.1f} a {mte['vs_s8_com_dens_real'][2]:+.1f})")
    print("   por faixa, MAPE (viés mediano): seção 8", {k: f"{v['mape']:.1f}% ({v['vies']:+.0f}%)" for k, v in mte["por_faixa_s8"].items()})
    print("                                   escolhida", {k: f"{v['mape']:.1f}% ({v['vies']:+.0f}%)" for k, v in mte["por_faixa_escolhida"].items()})
    print("   massa × densidade real por faixa: seção 8", {k: round(mte["massa_x_dens_real_s8"][k]) for k in fx}, "| escolhida", {k: round(mte["massa_x_dens_real_escolhida"][k]) for k in fx})

    def linha(est, ref):
        a = ok(rt, est)
        return dict(teste=100 * a.mean(), ic=ic_simples(a), vs=bootstrap(ref, a), mae=float(np.abs(est - rt).mean()), vies=float((est - rt).mean()),
                    por_faixa=pf(est), acima15=crit(rt[rt > 15], est[rt > 15]))
    print("\n## 4a. Densidade: todas as candidatas no teste (massa REAL × densidade | massa escolhida × densidade | geo com a rota direta)")
    print(f"{'candidata':78s} | CV crit | TESTE massa real×dens (>15 g) | massa × dens: Δ vs só imagem da seção 9 (IC95%) | geo")
    res["dens_teste"] = {}
    for nome in sorted(cv, key=lambda n: cv[n]["crit"]):
        d = C[nome][te]
        r = dict(cv=cv[nome]["crit"], dens=crit(rt, mt * d), dens15=crit(rt[rt > 15], (mt * d)[rt > 15]), mxd=linha(mS[te] * d, a9_img),
                 geo=linha(np.sqrt(Rdir[te] * mS[te] * d), a9_img), n5k_v1=crit(real["n5k_v1"], mS["n5k_v1"] * C[nome]["n5k_v1"]))
        res["dens_teste"][nome] = r
        print(f"{nome[:78]:78s} | {r['cv']:5.1f}% | {r['dens']:5.1f}% ({r['dens15']:5.1f}%)               | {r['mxd']['teste']:5.1f}%  {r['mxd']['vs'][0]:+5.1f} ({r['mxd']['vs'][1]:+.1f} a {r['mxd']['vs'][2]:+.1f})"
              f"      | {r['geo']['teste']:5.1f}%" + ("   <- densidade escolhida" if nome == sel_d else ""))
    print("\n## 4b. Só imagem escolhido (R4) no teste")
    r = linha(IMG[sel_i][te], a9_img); r["vs_llm"] = bootstrap(a_llm, ok(rt, IMG[sel_i][te])); r["vs_hib_s8"] = bootstrap(a8_hib, ok(rt, IMG[sel_i][te]))
    r["vs_img_s8"] = bootstrap(a8_img, ok(rt, IMG[sel_i][te])); r["vs_hib_v2_s9"] = bootstrap(a9_hv2, ok(rt, IMG[sel_i][te]))
    r["n5k_v1"] = crit(real["n5k_v1"], IMG[sel_i]["n5k_v1"]); r["cv"] = cv_img[sel_i]
    r["oraculo_massa_real_x_dens"] = crit(rt, mt * Dsel[te])
    res["img_teste"] = r
    print(f"   '{sel_i}': {r['teste']:.1f}% (IC95% {r['ic'][0]:.1f} a {r['ic'][1]:.1f}) | MAE {r['mae']:.1f} g, viés {r['vies']:+.1f} g | Δ vs só imagem s9 {r['vs'][0]:+.1f} ({r['vs'][1]:+.1f} a {r['vs'][2]:+.1f})"
          f" | Δ vs só imagem s8 {r['vs_img_s8'][0]:+.1f} ({r['vs_img_s8'][1]:+.1f} a {r['vs_img_s8'][2]:+.1f}) | Δ vs híbrido v2 s9 {r['vs_hib_v2_s9'][0]:+.1f} ({r['vs_hib_v2_s9'][1]:+.1f} a {r['vs_hib_v2_s9'][2]:+.1f})"
          f" | Δ vs LLM {r['vs_llm'][0]:+.1f} ({r['vs_llm'][1]:+.1f} a {r['vs_llm'][2]:+.1f}) | Δ vs híbrido s8 {r['vs_hib_s8'][0]:+.1f} ({r['vs_hib_s8'][1]:+.1f} a {r['vs_hib_s8'][2]:+.1f}) | n5k_v1 {r['n5k_v1']:.0f}%")
    print("   por faixa (s8 -> s9 -> agora):", {k: f"{res['teste_ref']['por_faixa_img_s8'][k]:.0f} -> {res['teste_ref']['por_faixa_img_s9'][k]:.0f} -> {r['por_faixa'][k]:.0f}" for k in fx},
          f"| >15 g: {crit(rt[rt > 15], IMG[ref_i][te][rt > 15]):.1f} -> {crit(rt[rt > 15], ref[f'{te}|img_s9'][rt > 15]):.1f} -> {r['acima15']:.1f}"
          f" | >25 g: {crit(rt[rt > 25], ref[f'{te}|img_s9'][rt > 25]):.1f} -> {crit(rt[rt > 25], IMG[sel_i][te][rt > 25]):.1f}")
    r["acima25"] = crit(rt[rt > 25], IMG[sel_i][te][rt > 25]); r["acima25_s9"] = crit(rt[rt > 25], ref[f"{te}|img_s9"][rt > 25])
    print(f"   oráculos: massa real × dens escolhida {r['oraculo_massa_real_x_dens']:.1f}% (s8: {res['teste_ref']['oraculo_massa_real_x_dens_s8']:.1f}%; LLM: {res['teste_ref']['oraculo_massa_real_x_dens_llm']:.1f}%)"
          f" | massa do modelo × dens real {res['teste_ref']['oraculo_massa_modelo_x_dens_real']:.1f}%")
    # decomposição por faixa (só relato): onde está o erro que sobra
    dec = {"massa real × dens s8": mt * R8[te]["dens"], "massa real × dens escolhida": mt * Dsel[te], "massa real × dens LLM v2": mt * dl[te],
           "massa s8 × dens real": m8[te] * rt / mt, "massa escolhida × dens real": mS[te] * rt / mt, "massa escolhida × dens escolhida": mS[te] * Dsel[te]}
    res["teste_decomposicao"] = {k: dict(todos=crit(rt, v), acima15=crit(rt[rt > 15], v[rt > 15]), **pf(v)) for k, v in dec.items()}
    res["teste_decomposicao"]["vies_massa_por_faixa"] = {k: float(100 * (np.exp(np.median(np.log(mS[te][f] / mt[f]))) - 1)) for k, f in fx.items()}
    res["teste_decomposicao"]["vies_dens_por_faixa"] = {k: float(100 * (np.exp(np.median(lg(Dsel[te][f]) - lg((rt / mt)[f]))) - 1)) for k, f in fx.items()}
    res["teste_decomposicao"]["dens_escolhida_vs_s8"] = bootstrap(ok(rt, dec["massa real × dens s8"]), ok(rt, dec["massa real × dens escolhida"]))
    res["teste_decomposicao"]["dens_escolhida_vs_llm"] = bootstrap(ok(rt, dec["massa real × dens LLM v2"]), ok(rt, dec["massa real × dens escolhida"]))
    print("   com a massa real, densidade escolhida vs s8: %+.1f (%+.1f a %+.1f) | vs LLM v2: %+.1f (%+.1f a %+.1f)"
          % (*res["teste_decomposicao"]["dens_escolhida_vs_s8"], *res["teste_decomposicao"]["dens_escolhida_vs_llm"]))
    print("   decomposição por faixa (0-5 / 5-15 / 15-25 / 25-40 / >40 | todos | >15 g):")
    for k, v in res["teste_decomposicao"].items():
        if not isinstance(v, dict):
            continue
        if "todos" in v:
            print(f"      {k:30s} " + " / ".join(f"{v[f]:3.0f}" for f in fx) + f" | {v['todos']:.1f}% | {v['acima15']:.1f}%")
        else:
            print(f"      {k:30s} " + " / ".join(f"{v[f]:+3.0f}%" for f in fx))
    # modelos únicos (um backbone): relato
    print("\n## 4b2. Modelos únicos (um backbone), rota escolhida pela CV; * = maior que ViT-B")
    a8_um = ok(rt, IMG["s8: média: ajuste fino + mlp + ridge o8 com laterais | direto"][te])
    res["unico_teste"] = {}
    for n in sorted(unicos, key=cvu.get):
        e = IMG[f"s8: {n} | {rota_de(n)}"]; r = linha(e[te], a8_um); r.update(cv=cvu[n], rota=rota_de(n), vitb=not grande(n), mape_massa=mape(mt, np.exp(B[n][te][:, I_M])),
                                                                         n5k_v1=crit(real["n5k_v1"], e["n5k_v1"]), vs_img_s9=bootstrap(a9_img, ok(rt, e[te])))
        res["unico_teste"][n] = r
        print(f"   {'*' if grande(n) else ' '} {n[:52]:52s} {r['rota']:6s} CV {r['cv']:5.1f}% | TESTE {r['teste']:5.1f}% (IC {r['ic'][0]:.1f} a {r['ic'][1]:.1f}) MAE {r['mae']:.1f} viés {r['vies']:+.1f} | MAPE massa {r['mape_massa']:.1f}%"
              f" | Δ vs modelo exportado em 8.9 {r['vs'][0]:+5.1f} ({r['vs'][1]:+.1f} a {r['vs'][2]:+.1f}) | >15 g {r['acima15']:.1f}% | v1 {r['n5k_v1']:.0f}%"
              + ("   <- único escolhido" if n == sel_u else "") + ("   <- único ViT-B escolhido" if n == sel_ub else ""))
    print("\n## 4c. Híbridos no teste (Δ contra o híbrido com v3 da seção 9 nesta leitura; e contra o híbrido com v2 da seção 9, 67,1%)")
    res["hib_teste"] = {}
    todos = [(f"[s8] {n}", est8[n], loo8[n]) for n in n_pesos] + [(n, H[n]["est"], H[n]["loo"]) for n in H]
    for nome, est, loo in todos:
        r = linha(est[te], a9_hv3); r["calib_loo"] = crit(real["calib"], loo); r["vs_llm"] = bootstrap(a_llm, ok(rt, est[te]))
        r["vs_hib_v2_s9"] = bootstrap(a9_hv2, ok(rt, est[te])); r["vs_hib_s8"] = bootstrap(a8_hib, ok(rt, est[te])); r["vs_img_s9"] = bootstrap(a9_img, ok(rt, est[te]))
        r["acima25"] = crit(rt[rt > 25], est[te][rt > 25])
        r["vs_img_s8"] = bootstrap(a8_img, ok(rt, est[te])); r["n5k_v1"] = crit(real["n5k_v1"], est["n5k_v1"])
        res["hib_teste"][nome] = r
        print(f"   {nome[:86]:86s} calib {r['calib_loo']:5.1f}% | TESTE {r['teste']:5.1f}% (IC {r['ic'][0]:.1f} a {r['ic'][1]:.1f}) MAE {r['mae']:.1f} viés {r['vies']:+.1f}"
              f" | Δ v3s9 {r['vs'][0]:+5.1f} ({r['vs'][1]:+.1f} a {r['vs'][2]:+.1f}) | Δ v2s9 {r['vs_hib_v2_s9'][0]:+5.1f} ({r['vs_hib_v2_s9'][1]:+.1f} a {r['vs_hib_v2_s9'][2]:+.1f}) | >15 g {r['acima15']:.1f}% | >25 g {r['acima25']:.1f}% | v1 {r['n5k_v1']:.0f}%"
              + ("   <- escolhida (R5)" if nome == sel_h else "") + ("   <- melhor sem v3" if nome == sel_h_v2 and sel_h_v2 != sel_h else "")
              + ("   <- melhor com v3" if nome == sel_h_v3 and sel_h_v3 != sel_h else ""))
    r = res["hib_teste"][sel_h]
    print(f"   híbrido escolhido por faixa (s8 -> v3 s9 -> agora):", {k: f"{res['teste_ref']['por_faixa_hib_s8'][k]:.0f} -> {res['teste_ref']['por_faixa_hib_v3_s9'][k]:.0f} -> {r['por_faixa'][k]:.0f}" for k in fx},
          f"| >15 g: {crit(rt[rt > 15], est8[SEL8_HIB][te][rt > 15]):.1f} -> {crit(rt[rt > 15], ref[f'{te}|hib_v3_{TAG}'][rt > 15]):.1f} -> {r['acima15']:.1f}")
    r2_ = res["hib_teste"][sel_h_v2]
    print(f"   híbrido sem v3 ('{sel_h_v2}'): {r2_['teste']:.1f}% (IC {r2_['ic'][0]:.1f} a {r2_['ic'][1]:.1f}); por faixa (v2 s9 -> agora):",
          {k: f"{res['teste_ref']['por_faixa_hib_v2_s9'][k]:.0f} -> {r2_['por_faixa'][k]:.0f}" for k in fx})
    r3_ = res["hib_teste"][sel_h_v3]
    print(f"   >>> forma escolhida pelo calib entre todas ({TAG}): '{sel_h}': {r['teste']:.1f}%, IC 95% {r['ic'][0]:.1f} a {r['ic'][1]:.1f}; limite inferior {'>=' if r['ic'][0] >= 70 else '<'} 70%")
    print(f"   >>> melhor forma COM v3 pelo calib ({TAG}): '{sel_h_v3}': {r3_['teste']:.1f}%, IC 95% {r3_['ic'][0]:.1f} a {r3_['ic'][1]:.1f}; limite inferior {'>=' if r3_['ic'][0] >= 70 else '<'} 70%")
    print(f"       por faixa (v3 s9 -> agora):", {k: f"{res['teste_ref']['por_faixa_hib_v3_s9'][k]:.0f} -> {r3_['por_faixa'][k]:.0f}" for k in fx}, f"| >15 g {r3_['acima15']:.1f}% | >25 g {r3_['acima25']:.1f}%")
    res["teste_ref"]["acima15"] = dict(img_s8=crit(rt[rt > 15], IMG[ref_i][te][rt > 15]), hib_s8=crit(rt[rt > 15], est8[SEL8_HIB][te][rt > 15]),
                                       llm=crit(rt[rt > 15], L[te]["prot"][rt > 15]))
    # ------------------------------------------------------------ 4d. LLM v3: só relato (nenhuma escolha depende disto)
    if tem_v3:
        print("\n## 4d. LLM v3 (identificação + partes da massa): relato")
        d3t = V3[te]["dens"]
        f3 = lambda v: f"{v[0]:+.1f} ({v[1]:+.1f} a {v[2]:+.1f})"
        v3 = dict(dens_massa_real=dict(todos=crit(rt, mt * d3t), acima15=crit(rt[rt > 15], (mt * d3t)[rt > 15]), **pf(mt * d3t)),
                  dens_vs_v2=bootstrap(ok(rt, mt * dl[te]), ok(rt, mt * d3t)), dens_vs_modelo_s9=bootstrap(ok(rt, mt * Dsel[te]), ok(rt, mt * d3t)),
                  dens_vs_modelo_s8=bootstrap(ok(rt, mt * R8[te]["dens"]), ok(rt, mt * d3t)),
                  massa_modelo_x_dens_v3=dict(todos=crit(rt, mS[te] * d3t), **pf(mS[te] * d3t)), vies_dens_v3_por_faixa={k: float(100 * (np.exp(np.median(lg(d3t[f]) - lg((rt / mt)[f]))) - 1)) for k, f in fx.items()},
                  vies_dens_v2_por_faixa={k: float(100 * (np.exp(np.median(lg(dl[te][f]) - lg((rt / mt)[f]))) - 1)) for k, f in fx.items()})
        print(f"   {te}, massa real × densidade: v3 {v3['dens_massa_real']['todos']:.1f}% (>15 g {v3['dens_massa_real']['acima15']:.1f}%) | v2 {crit(rt, mt * dl[te]):.1f}% | modelo s9 {crit(rt, mt * Dsel[te]):.1f}%"
              f" | v3 vs v2: {f3(v3['dens_vs_v2'])} | v3 vs modelo s9: {f3(v3['dens_vs_modelo_s9'])} | v3 vs modelo s8: {f3(v3['dens_vs_modelo_s8'])}")
        print("   por faixa, massa real × dens v3:", {k: round(v3["dens_massa_real"][k]) for k in fx}, "| massa do modelo × dens v3:", {k: round(v, 1) for k, v in v3["massa_modelo_x_dens_v3"].items()})
        print("   viés mediano da densidade por faixa: v3", {k: f"{v:+.0f}%" for k, v in v3["vies_dens_v3_por_faixa"].items()}, "| v2", {k: f"{v:+.0f}%" for k, v in v3["vies_dens_v2_por_faixa"].items()})
        # híbrido escolhido com v3 contra o híbrido escolhido sem v3 (seção 9.6), o só imagem e o LLM
        a_v3, a_h9, a_i9 = ok(rt, H[sel_h_v3]["est"][te]), ok(rt, H[sel_h_v2]["est"][te]), ok(rt, IMG[sel_i][te])
        v3["hibrido"] = dict(nome=sel_h_v3, pesos=H[sel_h_v3].get("pesos"), teste=100 * a_v3.mean(), vs_hibrido_sem_v3=bootstrap(a_h9, a_v3), vs_so_imagem_s9=bootstrap(a_i9, a_v3),
                             vs_llm_v2=bootstrap(a_llm, a_v3), por_faixa=pf(H[sel_h_v3]["est"][te]), por_faixa_sem_v3=pf(H[sel_h_v2]["est"][te]),
                             acima15_ic=ic_simples(a_v3[rt > 15]), acima25=crit(rt[rt > 25], H[sel_h_v3]["est"][te][rt > 25]),
                             acima25_sem_v3=crit(rt[rt > 25], H[sel_h_v2]["est"][te][rt > 25]),
                             vs_hibrido_sem_v3_acima15=bootstrap(a_h9[rt > 15], a_v3[rt > 15]),
                             vs_v3_s9=bootstrap(a9_hv3, a_v3), vs_v2_s9=bootstrap(a9_hv2, a_v3), vs_img_s9=bootstrap(a9_img, a_v3))
        h = v3["hibrido"]
        print(f"   híbrido escolhido ('{sel_h_v3}', pesos {h['pesos']}): {h['teste']:.1f}% | vs híbrido sem v3 ('{sel_h_v2}'): {f3(h['vs_hibrido_sem_v3'])} | vs só imagem s9: {f3(h['vs_so_imagem_s9'])}"
              f" | vs LLM v2: {f3(h['vs_llm_v2'])}")
        print("   por faixa (sem v3 -> com v3):", {k: f"{h['por_faixa_sem_v3'][k]:.0f} -> {h['por_faixa'][k]:.0f}" for k in fx},
              f"| >15 g: IC {h['acima15_ic'][0]:.1f} a {h['acima15_ic'][1]:.1f}; Δ {f3(h['vs_hibrido_sem_v3_acima15'])} | >25 g: {h['acima25_sem_v3']:.1f} -> {h['acima25']:.1f}")
        v3["formas_vs_hibrido_sem_v3"] = {n: bootstrap(a_h9, ok(rt, H[n]["est"][te])) for n in H if n.startswith("[v3]")}
        for n, v in v3["formas_vs_hibrido_sem_v3"].items():
            print(f"      {n[:100]:100s} vs híbrido sem v3: {v[0]:+5.1f} ({v[1]:+.1f} a {v[2]:+.1f})")
        # calib (único conjunto de avaliação cujos ingredientes podem ser lidos): fonte principal e fração das fontes
        Ic = Ingr(D.aval["calib"]["chave"], vocab=I.vocab)
        nf = len(FONTES) - 1
        tem_f = Ic.principal < nf
        c3 = V3["calib"]["FF"] * V3["calib"]["DFc"]            # contribuição de proteína por fonte (g/g)
        f3_comp = np.where(c3[:, :-1].max(1) >= 0.5 * np.maximum(c3.sum(1), 1e-9), c3[:, :-1].argmax(1), nf)
        f_mod = zf["calib_ff"][:, :-1].argmax(1)
        ac = lambda f: float(100 * (f[tem_f] == Ic.principal[tem_f]).mean())
        v3["fonte_principal_calib"] = dict(n=int(tem_f.sum()), v2_derivada=ac(V2["calib"]["fonte"]), v3_declarada=ac(V3["calib"]["fonte"]), v3_derivada=ac(f3_comp),
                                           modelo_fracoes=ac(f_mod), v3_ou_modelo=float(100 * ((V3["calib"]["fonte"] == Ic.principal) | (f_mod == Ic.principal))[tem_f].mean()),
                                           v3_e_modelo_concordam=float(100 * (V3["calib"]["fonte"] == f_mod)[tem_f].mean()),
                                           falso_positivo_v3=float(100 * (V3["calib"]["fonte"][~tem_f] < nf).mean()), n_sem_fonte=int((~tem_f).sum()))
        conf = {}
        for a_, b_ in zip(Ic.principal[tem_f], V3["calib"]["fonte"][tem_f]):
            conf.setdefault(FONTES[a_], {}).setdefault(FONTES[b_], 0); conf[FONTES[a_]][FONTES[b_]] += 1
        v3["fonte_principal_calib"]["confusao_v3"] = conf
        print("   calib, fonte principal certa nos pratos com fonte dominante real:", {k: (round(v, 1) if not isinstance(v, dict) else v) for k, v in v3["fonte_principal_calib"].items()})
        okc = ok(cc, mreal["calib"] * V3["calib"]["dens"]); certo = (V3["calib"]["fonte"] == Ic.principal)[tem_f]
        v3["fonte_principal_calib"].update(crit_acerta=float(100 * okc[tem_f][certo].mean()), crit_erra=float(100 * okc[tem_f][~certo].mean()) if (~certo).any() else None)
        a15 = cc > 15
        fr_r = Ic.frac_fonte[:, :-1].sum(1); fr3 = V3["calib"]["FF"][:, :-1].sum(1); fr2 = V2["calib"]["FF"][:, :-1].sum(1)
        v3["fracao_fontes_calib_acima15"] = dict(real=float(fr_r[a15].mean()), v3=float(fr3[a15].mean()), v2=float(fr2[a15].mean()), r_v3=float(np.corrcoef(fr_r[a15], fr3[a15])[0, 1]),
                                                 r_v2=float(np.corrcoef(fr_r[a15], fr2[a15])[0, 1]), erro_abs_mediano_v3=float(np.median(np.abs(fr3 - fr_r)[a15])),
                                                 erro_abs_mediano_v2=float(np.median(np.abs(fr2 - fr_r)[a15])))
        print("   calib, fração da massa que é fonte de proteína (> 15 g):", {k: round(v, 3) for k, v in v3["fracao_fontes_calib_acima15"].items()},
              f"| massa real × dens v3: {v3['fonte_principal_calib']['crit_acerta']:.0f}% quando acerta a fonte, {v3['fonte_principal_calib']['crit_erra']:.0f}% quando erra")
        fxc = faixas(cc)
        v3["dens_massa_real_calib"] = dict(v3={k: crit(cc[f], (mreal["calib"] * V3["calib"]["dens"])[f]) for k, f in fxc.items()}, v2={k: crit(cc[f], (mreal["calib"] * dl["calib"])[f]) for k, f in fxc.items()},
                                           v3_acima15=crit(cc[a15], (mreal["calib"] * V3["calib"]["dens"])[a15]), v2_acima15=crit(cc[a15], (mreal["calib"] * dl["calib"])[a15]),
                                           v3_todos=crit(cc, mreal["calib"] * V3["calib"]["dens"]), v2_todos=crit(cc, mreal["calib"] * dl["calib"]))
        print("   calib, massa real × densidade por faixa: v3", {k: round(v) for k, v in v3["dens_massa_real_calib"]["v3"].items()}, "| v2", {k: round(v) for k, v in v3["dens_massa_real_calib"]["v2"].items()},
              f"| >15 g: v3 {v3['dens_massa_real_calib']['v3_acima15']:.1f}% v2 {v3['dens_massa_real_calib']['v2_acima15']:.1f}%")
        res["v3_relato"] = v3
        # estimativas por prato desta execução, para comparar rodadas (dens_rodadas.py)
        if te == "teste":
            np.savez(os.path.join(os.path.dirname(BANCO_D), f"v3_pratos_{TAG}{SUF}.npz"),
                     **{f"{s}|hib_v3": H[sel_h_v3]["est"][s] for s in SPL}, **{f"{s}|hib_calib": H[sel_h]["est"][s] for s in SPL}, **{f"{s}|hib_sem_v3": H[sel_h_v2]["est"][s] for s in SPL},
                     **{f"{s}|hib_s8": est8[SEL8_HIB][s] for s in SPL}, **{f"{s}|dens_v3": V3[s]["dens"] for s in SPL},
                     **{f"{s}|fonte_v3": V3[s]["fonte"] for s in SPL}, **{f"{s}|massa_modelo": mS[s] for s in SPL}, **{f"{s}|img": IMG[sel_i][s] for s in SPL},
                     **{f"{s}|loo_v3": H[sel_h_v3]["loo"] for s in ("calib",)}, forma=np.array(sel_h_v3), pesos=np.array(str(H[sel_h_v3].get("pesos"))),
                     forma_calib=np.array(sel_h))
    sufixo = ("" if TAG == "media" else f"_v3_{TAG}") + SUF
    grava_json("avalia_ensaio.json" if te != "teste" else f"avalia{sufixo}.json", res)


if __name__ == "__main__":
    main()
