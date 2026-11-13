#!/usr/bin/env python3
"""Monta o gabarito do teste de precisão (proteína por foto) — Nutrelle.

Etapas (cada fonte tem seu script; este junta tudo):
  1. coleta_ufpr.py  -> <ufpr_dir>/ufpr_itens.csv + recortes das fotos do Manual UFPR
  2. coleta_n5k.py   -> <n5k_dir>/n5k_amostra.csv + rgb/<dish_id>.png
  3. Seara (este arquivo, `seara`): baixa a arte da embalagem das 7 "Marmitas Seara"
     300 g, recorta só a bandeja aberta (vista de cima) e pinta de branco todo o
     fundo fora da bandeja (padrão com a palavra "Marmita", nome e selo de proteína
     ficam fora do recorte). Proteína = proteína/100 g da tabela oficial × 300 g.
  4. `montar`: sorteia ids neutros (b001…, n001…), copia as imagens para
     <scratch>/precisao/imagens/<id>.<ext> e grava gabarito_pratos.csv,
     gabarito.csv (formato do eval/score_precisao.py) e lista_imagens.txt.

Uso:
  python3 coleta.py seara <dir_trabalho_seara>
  python3 coleta.py montar <ufpr_dir> <seara_dir> <n5k_dir> <scratch_precisao_dir> <repo_precisao_dir>

As imagens NUNCA vão para o repositório (direitos de terceiros; repo público).
"""
import csv
import os
import random
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

SEED = 20261108
SEARA_BASE = "https://www.seara.com.br"
# (arquivo da arte, slug da página, descrição, proteína/100 g da tabela no site, caixa de recorte)
SEARA = [
    ("seara-product-116567-500.webp", "marmita-carne-moida-300g-seara",
     "Marmita: carne moída com legumes, arroz e feijão", 5.7, (237, 138, 432, 368)),
    ("seara-product-116571-500.webp", "marmita-estrogonofe-300g-seara",
     "Marmita: estrogonofe de frango, arroz e batatas em cubos", 6.3, (237, 138, 432, 368)),
    ("seara-product-116575-500.webp", "marmita-feijoada-300g-seara",
     "Marmita: feijoada, arroz e couve", 6.7, (250, 135, 430, 361)),
    ("seara-product-116579-500.webp", "marmita-frango-pure-300g-seara",
     "Marmita: frango desfiado, arroz com brócolis e purê de batata", 4.6, (237, 138, 432, 368)),
    ("seara-product-116583-500.webp", "marmita-frango-milho-300g-seara",
     "Marmita: frango desfiado, arroz e creme de milho", 5.8, (237, 138, 432, 368)),
    ("seara-product-116587-500.webp", "marmita-calabresa-300g-seara",
     "Marmita: calabresa acebolada, arroz e feijão", 6.2, (237, 138, 432, 368)),
    ("seara-product-116592-500.webp", "marmita-ragu-suino-300g-seara",
     "Marmita: ragu suíno, arroz e purê de batata", 6.0, (237, 138, 432, 368)),
]
SEARA_PESO = 300


def _keep_largest(bg):
    lab, n = ndimage.label(~bg)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        bg |= lab != (1 + int(np.argmax(sizes)))
    return bg


def mask_tray(src, box, dst, modo="cinza"):
    """Recorta `box` e pinta de branco o fundo conectado às bordas.
    modo 'cinza': o contorno cinza claro da bandeja separa fundo e comida.
    modo 'laranja': o fundo é o padrão pêssego/laranja (usado na feijoada, cuja
    borda superior encosta no feijão preto)."""
    a = np.array(Image.open(src).convert("RGB").crop(box)).astype(int)
    if modo == "cinza":
        sep = (a.min(2) > 160) & ((a.max(2) - a.min(2)) < 40)
        sep = ndimage.binary_dilation(sep, iterations=2)
        lab, _ = ndimage.label(~sep)
    else:
        R, G, B = a[..., 0], a[..., 1], a[..., 2]
        lab, _ = ndimage.label((R > 170) & (R - B > 40) & (G > 70))
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge))
    if modo != "cinza":
        bg = ndimage.binary_closing(bg, iterations=2)
    bg = _keep_largest(bg)
    a[bg] = 255
    Image.fromarray(a.astype("uint8")).save(dst)


def seara(wd):
    """Baixa cada página de marmita, acha a URL da arte e gera o recorte."""
    import re
    os.makedirs(wd, exist_ok=True)
    for fn, slug, desc, p100, box in SEARA:
        html = subprocess.run(["curl", "-sS", f"{SEARA_BASE}/produto/{slug}/"],
                              capture_output=True, text=True).stdout
        urls = sorted(set(re.findall(r"https://www\.seara\.com\.br/wp-content/uploads/[^\"' ]*" + re.escape(fn), html)))
        src = os.path.join(wd, fn)
        if urls:
            with open(os.path.join(wd, "urls.tsv"), "a", encoding="utf-8") as f:
                f.write(f"{fn}\t{urls[0]}\n")
        if not os.path.exists(src):
            # a arte pode estar listada só na página de outra marmita (galeria "relacionados")
            if not urls:
                print("arte não encontrada na página:", slug, fn)
                continue
            subprocess.run(["curl", "-sS", "-o", src, urls[0]], check=True)
        modo = "laranja" if "116575" in fn else "cinza"
        mask_tray(src, box, os.path.join(wd, fn.replace(".webp", "_bandeja.png")), modo)


def montar(ufpr_dir, seara_dir, n5k_dir, scratch, repo):
    img_dir = os.path.join(scratch, "imagens")
    os.makedirs(img_dir, exist_ok=True)
    br, n5k = [], []

    for r in csv.DictReader(open(os.path.join(ufpr_dir, "ufpr_itens.csv"), encoding="utf-8")):
        g, p100 = float(r["gramas"]), float(r["proteina_100g"])
        conf = "C" if r["codigo"] in ("P718", "P146", "P145") else ("B" if r["nota"] else "A")
        br.append(dict(
            origem="br", fonte="UFPR Manual Fotográfico de Quantificação Alimentar (2017) + TACO 4ª ed.",
            categoria=r["categoria"], descricao=f"{r['descricao']} — foto {r['codigo']}-{r['foto']}",
            peso_porcao_g=g, proteina_g=round(g * p100 / 100, 2), confianca=conf,
            url_pagina="https://nutritotal.com.br/pro/wp-content/uploads/2019/02/manual_quantificacao_alimentar-compactado-1.pdf",
            url_imagem=f"PDF p. {r['pagina']} (impressa), foto {r['codigo']}–{r['foto']}",
            observacoes=(f"TACO #{r['taco_id']} '{r['taco_nome']}' = {p100} g/100 g; gramatura da tabela p. 143-144 do manual"
                         + (f"; {r['nota']}" if r["nota"] else "") + "; recorte da foto sem o rótulo do código"),
            _src=os.path.join(ufpr_dir, os.path.basename(r["arquivo"]))))

    urls = {}
    if os.path.exists(os.path.join(seara_dir, "urls.tsv")):
        urls = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(seara_dir, "urls.tsv"), encoding="utf-8"))
    for fn, slug, desc, p100, box in SEARA:
        src = os.path.join(seara_dir, fn.replace(".webp", "_bandeja.png"))
        if not os.path.exists(src):
            continue
        br.append(dict(
            origem="br", fonte="Seara (site oficial) — Marmita 300 g", categoria="congelado",
            descricao=desc, peso_porcao_g=SEARA_PESO, proteina_g=round(p100 * SEARA_PESO / 100, 2), confianca="B",
            url_pagina=f"{SEARA_BASE}/produto/{slug}/", url_imagem=urls.get(fn, fn),
            observacoes=(f"tabela oficial {p100} g prot/100 g × 300 g (embalagem inteira = 1 porção). "
                         "Imagem = recorte da bandeja na arte da embalagem (ilustração da porção, não foto do produto "
                         "aquecido); fundo/textos pintados de branco"),
            _src=src))

    for r in csv.DictReader(open(os.path.join(n5k_dir, "n5k_amostra.csv"), encoding="utf-8")):
        d = r["dish_id"]
        n5k.append(dict(
            origem="n5k", fonte="Nutrition5k (Google, CC BY 4.0)", categoria="n5k",
            descricao=f"{d}: {r['ingredientes']}", peso_porcao_g=float(r["massa_g"]),
            proteina_g=float(r["proteina_g"]), confianca="A",
            url_pagina="https://github.com/google-research-datasets/Nutrition5k",
            url_imagem=f"gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/{d}/rgb.png",
            observacoes="pesagem incremental por ingrediente + USDA; foto zenital realsense",
            _src=os.path.join(n5k_dir, "rgb", f"{d}.png")))

    rnd = random.Random(SEED)
    rnd.shuffle(br)
    rnd.shuffle(n5k)
    linhas = []
    for pref, grupo in (("b", br), ("n", n5k)):
        for i, it in enumerate(grupo, 1):
            it["prato_id"] = f"{pref}{i:03d}"
            ext = os.path.splitext(it["_src"])[1]
            dst = os.path.join(img_dir, it["prato_id"] + ext)
            Image.open(it["_src"]).convert("RGB").save(dst) if ext == ".png" else shutil.copy(it["_src"], dst)
            it["_dst"] = dst
            linhas.append(it)

    cols = ["prato_id", "origem", "fonte", "categoria", "descricao", "peso_porcao_g", "proteina_g", "confianca",
            "url_pagina", "url_imagem", "observacoes"]
    with open(os.path.join(repo, "gabarito_pratos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)
    with open(os.path.join(repo, "gabarito.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prato_id", "categoria", "componente", "gramas", "proteina_100g"])
        for it in linhas:
            w.writerow([it["prato_id"], it["categoria"], "prato inteiro", it["peso_porcao_g"],
                        round(it["proteina_g"] / it["peso_porcao_g"] * 100, 2)])
    with open(os.path.join(scratch, "lista_imagens.txt"), "w", encoding="utf-8") as f:
        for it in linhas:
            f.write(f"{it['prato_id']}\t{it['_dst']}\n")
    print(f"{len(br)} brasileiros + {len(n5k)} Nutrition5k")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "seara":
        seara(sys.argv[2])
    elif cmd == "montar":
        montar(*sys.argv[2:7])
