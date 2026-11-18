#!/usr/bin/env python3
"""Seção 18.4: escala absoluta (cm por pixel) para o pipeline geométrico. Não usa rótulo nenhum.

    python3 -I escala.py n5k <dir_treino> [saida.csv]                 # Nutrition5k: profundidade + campo de visão
    python3 -I escala.py dfoodtj <fotos_geom.csv> <itens_geom.csv.gz>  # DFood-TJ: diâmetro do prato por período, pelos ovos
    python3 -I escala.py fpb <fotos_geom.csv> [--quadrado-mm X]        # FPB: xadrez (sem o tamanho do quadrado, só ordem)

Formato das tabelas do kernel (fotos_geom.csv, itens_geom.csv.gz): seção 18.10. As máscaras vêm em RLE do COCO sem
compressão (contagens, em ordem de coluna, começando pelos zeros), na resolução de trabalho (mascara_largura × mascara_altura).
"""
import csv, gzip, json, math, os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
PRIORS = os.path.join(AQUI, "priors.csv")
LISTA_N5K = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "treino_v2.csv")


def prior(chave, tipo="escala"):
    for r in csv.DictReader(open(PRIORS)):
        if r["tipo"] == tipo and r["chave"] == chave and r["valor"] != "":
            return float(r["valor"])
    return None


def le_tabela(caminho):
    abre = gzip.open if caminho.endswith(".gz") else open
    with abre(caminho, "rt", newline="") as f:
        return list(csv.DictReader(f))


def rle_decodifica(s, h, w):
    """RLE do COCO sem compressão: contagens alternadas (zeros, uns, ...) em ordem de coluna."""
    c = np.array([int(x) for x in s.split()], dtype=np.int64)
    v = np.zeros(h * w, dtype=bool)
    pos = np.concatenate([[0], np.cumsum(c)])
    for k in range(1, len(c), 2):
        v[pos[k]:pos[k + 1]] = True
    return v.reshape(w, h).T


def rle_codifica(m):
    """Inverso de rle_decodifica (usado no ensaio e nos testes)."""
    v = m.T.reshape(-1).astype(np.int8)
    d = np.flatnonzero(np.diff(np.concatenate([[0], v, [1 - v[-1]]])))
    c = np.diff(np.concatenate([[0], d]))
    return " ".join(str(int(x)) for x in c)


def nums(s):
    return [float(x) for x in s.split()] if s else []


# ------------------------------------------------------------------ Nutrition5k
def profundidade_cm(td, dish):
    from PIL import Image
    a = np.asarray(Image.open(os.path.join(td, "img", dish, "depth_raw.png"))).astype(np.float64)
    v = a[a > 0]
    f = 0.1 if np.median(v) < 1000 else 0.01          # quase todas em 1e-4 m; umas poucas em mm
    a = a * f
    a[a <= 0] = np.nan
    return a                                             # cm, 480 × 640, alinhada à foto de cor


_BB = {}


def caixa_prato_n5k(td, dish, fotos=None):
    """Caixa do prato: a do kernel (fotos_geom) se houver; senão a de zen_prato.py (profundidade); None se não achar."""
    if fotos and dish in fotos and fotos[dish].get("prato_caixa"):
        return nums(fotos[dish]["prato_caixa"])
    bb = os.path.join(td, "meta", "zen_bbox.csv")
    if not _BB and os.path.exists(bb):
        _BB.update({r["id"]: [float(r[k]) for k in ("x0", "y0", "x1", "y1")] for r in csv.DictReader(open(bb))})
    if dish in _BB:
        return _BB[dish]
    sys.path.insert(0, os.path.join(AQUI, "..", "treino"))
    from zen_prato import caixa
    from features_profundidade import atributos
    c = caixa(atributos(os.path.join(td, "img", dish, "depth_raw.png"), np.random.default_rng(0))[1].astype(np.float32))
    return None if c is None else list(c)


def plano_prato(dep, cx):
    """Distância (cm) da câmera ao plano do prato: mediana da profundidade no anel da borda (0,80 a 0,97 dos semieixos
    da elipse inscrita na caixa), onde quase nunca há comida."""
    x0, y0, x1, y1 = cx
    yy, xx = np.mgrid[0:dep.shape[0], 0:dep.shape[1]]
    r = np.sqrt(((xx - (x0 + x1) / 2) / max((x1 - x0) / 2, 1)) ** 2 + ((yy - (y0 + y1) / 2) / max((y1 - y0) / 2, 1)) ** 2)
    v = dep[(r > 0.80) & (r < 0.97)]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) > 50 else float(np.nanmedian(dep))


def n5k(td, ids=None, campo=None, fotos=None):
    campo = campo or prior("n5k_campo_horizontal")
    ids = ids if ids is not None else [r["dish_id"] for r in csv.DictReader(open(LISTA_N5K))]
    out, sem = {}, []
    for d in ids:
        dep = profundidade_cm(td, d)
        cx = caixa_prato_n5k(td, d, fotos)
        if cx is None:
            sem.append(d); continue
        dist = plano_prato(dep, cx)
        cmpx = 2 * dist * math.tan(math.radians(campo) / 2) / dep.shape[1]
        out[d] = dict(dist_cm=dist, cm_por_px=cmpx, diametro_prato_cm=0.5 * ((cx[2] - cx[0]) + (cx[3] - cx[1])) * cmpx, caixa=cx, caixa_achada=1)
    med = float(np.median([r["dist_cm"] for r in out.values()]))
    for d in sem:                                    # câmera fixa: sem caixa, vale a distância mediana (marcado)
        cmpx = 2 * med * math.tan(math.radians(campo) / 2) / 640
        out[d] = dict(dist_cm=med, cm_por_px=cmpx, diametro_prato_cm=float("nan"), caixa=None, caixa_achada=0)
    return out


# ------------------------------------------------------------------ DFood-TJ
OVO = ("egg", "boiled_egg_in_shell", "fs103:egg")      # fs103:egg: a classe de ovo do SETR-MLA (décima execução, 18.12)


def dfoodtj(fotos, itens, ovo_cm=None, nota_min=0.30, razao=(1.15, 1.50)):
    """Diâmetro físico do prato de cada período = mediana(eixo maior do prato / eixo menor do ovo) × eixo menor do ovo
    (prior). Ovo aceito: termo de ovo, nota >= 0,30, elipse com razão de eixos entre 1,15 e 1,50 (ovo inteiro, deitado)
    e pelo menos 90% dentro do prato. Devolve, por período: diâmetro, n de ovos, dispersão robusta (1,4826·MAD/mediana);
    e cm/px por foto."""
    ovo_cm = ovo_cm or prior("ovo_eixo_menor")
    F = {f["id"]: f for f in fotos if f["base"] == "dfoodtj"}
    rz = {}
    for it in itens:
        if it["base"] != "dfoodtj" or it["termo"] not in OVO or float(it["nota"]) < nota_min:
            continue
        f = F.get(it["id"])
        el, pe = nums(it.get("elipse", "")), nums(f.get("prato_elipse", "")) if f else []
        if len(el) < 4 or len(pe) < 4 or el[3] <= 0 or not (razao[0] <= el[2] / el[3] <= razao[1]):
            continue
        if float(it.get("dentro_prato") or 0) < 0.9:
            continue
        rz.setdefault(f["periodo"], {})[it["id"]] = pe[2] / el[3]            # um ovo por foto (o primeiro aceito)
    res = {}
    for per, d in rz.items():
        v = np.array(list(d.values()))
        med = float(np.median(v))
        res[per] = dict(n_ovos=len(v), diametro_cm=med * ovo_cm, dispersao=float(1.4826 * np.median(np.abs(v - med)) / med))
    cm = {}
    for i, f in F.items():
        pe = nums(f.get("prato_elipse", ""))
        if f["periodo"] in res and len(pe) >= 4 and pe[2] > 0:
            cm[i] = res[f["periodo"]]["diametro_cm"] / pe[2]
    return res, cm


def condicao_ovos(res, periodo="2020-2023"):
    """18.7 (2): pelo menos 30 ovos, dispersão de até 8% e diâmetro entre 22 e 32 cm."""
    r = res.get(periodo)
    return bool(r and r["n_ovos"] >= 30 and r["dispersao"] <= 0.08 and 22 <= r["diametro_cm"] <= 32)


# ------------------------------------------------------------------ FPB
def fpb(fotos, quadrado_mm=None):
    """cm/px pelo passo mediano do xadrez. Sem o tamanho do quadrado documentado (18.2), devolve None: o FPB entra só
    como relato de ordem."""
    if not quadrado_mm:
        return None
    out = {}
    for f in fotos:
        if f["base"] == "fpb" and f.get("xadrez_detectado") == "1" and float(f.get("xadrez_passo_px") or 0) > 0:
            out[f["id"]] = quadrado_mm / 10 / float(f["xadrez_passo_px"])
    return out


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "n5k":
        R = n5k(a[1])
        ok_ = [r for r in R.values() if r["caixa_achada"]]
        print("pratos sem caixa (distância mediana): %d" % (len(R) - len(ok_)))
        dia = np.array([r["diametro_prato_cm"] for r in ok_]); dist = np.array([r["dist_cm"] for r in ok_])
        print("Nutrition5k (%d pratos do treino): distância ao plano do prato %.1f cm (p10 %.1f, p90 %.1f); diâmetro do prato %.1f cm (p10 %.1f, p90 %.1f); cm/px mediano %.4f"
              % (len(R), np.median(dist), *np.percentile(dist, [10, 90]), np.median(dia), *np.percentile(dia, [10, 90]), np.median([r["cm_por_px"] for r in ok_])))
        print("Conferência sem rótulo (18.4): diâmetro mediano entre 18 e 30 cm -> %s" % ("passa" if 18 <= np.median(dia) <= 30 else "FALHA"))
        if len(a) > 2:
            with open(a[2], "w", newline="") as f:
                w = csv.writer(f, lineterminator="\n"); w.writerow(["dish_id", "dist_cm", "cm_por_px", "diametro_prato_cm"])
                for d, r in R.items():
                    w.writerow([d, "%.3f" % r["dist_cm"], "%.6f" % r["cm_por_px"], "%.2f" % r["diametro_prato_cm"]])
    elif a[0] == "dfoodtj":
        res, cm = dfoodtj(le_tabela(a[1]), le_tabela(a[2]))
        print(json.dumps(res, indent=1, ensure_ascii=False))
        print("18.7 (2), 2020–2023: %s; %d fotos com escala" % ("passa" if condicao_ovos(res) else "FALHA", len(cm)))
    elif a[0] == "fpb":
        q = float(a[a.index("--quadrado-mm") + 1]) if "--quadrado-mm" in a else None
        r = fpb(le_tabela(a[1]), q)
        print("FPB: só ordem (tamanho do quadrado não documentado)" if r is None else "FPB: %d fotos com escala" % len(r))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
