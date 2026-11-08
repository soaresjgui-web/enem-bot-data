#!/usr/bin/env python3
"""Coleta marmitas Liv Up (www.livup.com.br) para o gabarito do teste de precisão.

Uso:
  python3 coleta_livup.py raspar <lista_urls.txt> <dir_trabalho>
      Para cada página de produto: nome, peso da porção (texto "NNNg" ao lado do preço),
      tabela nutricional ("Porção de 100g" -> Proteínas) e as URLs das fotos
      (livup.imgix.net/website/images/photos-original/...). Baixa as fotos (w=1200)
      em <dir_trabalho>/raw/ e grava livup.json.
  python3 coleta_livup.py montar <dir_trabalho> <aceitos.csv> <dir_cego> <repo_precisao_dir>
      aceitos.csv (revisão visual manual): slug,foto,confianca,obs — uma linha por prato aceito.
      Sorteia ids c001.. (seed fixa), recorta a foto na caixa e grava <dir_cego>/<id>.jpg e acrescenta
      linhas em gabarito_pratos.csv e gabarito.csv.
Proteína da porção = proteína/100 g (tabela do site) × peso da embalagem (= 1 porção).
"""
import csv, html, json, os, random, re, subprocess, sys, shutil

BASE = "https://www.livup.com.br"


def get(url):
    return subprocess.run(["curl", "-sS", "-L", url], capture_output=True, text=True).stdout


def texto(h):
    t = re.sub(r"<script.*?</script>|<style.*?</style>|<svg.*?</svg>", "", h, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", "\n", t))
    return [l.strip() for l in t.split("\n") if l.strip()]


def raspar(lista, wd):
    os.makedirs(os.path.join(wd, "raw"), exist_ok=True)
    out = []
    for path in open(lista).read().split():
        url = BASE + path
        h = get(url)
        L = texto(h)
        try:
            i = L.index("Adicionar")
        except ValueError:
            print("sem produto:", path); continue
        # bloco: nome, R$ de, R$ por, 300g, Adicionar ; nome = <title> sem " | Liv Up"
        peso = next((l for l in L[i - 4:i] if re.fullmatch(r"\d+\s*g", l)), None)
        nome = html.unescape(re.search(r"<title>(.*?)</title>", h, re.S).group(1)).split(" | ")[0].strip()
        prot = None
        if "Tabela Nutricional" in L:
            j = L.index("Tabela Nutricional")
            base = L[j + 1]
            k = L.index("Proteínas", j)
            prot = L[k + 1]
        fotos = sorted(set(re.findall(r"https://livup\.imgix\.net/website/images/photos-original/[^?\"'\s]+", h)))
        arquivos = []
        for n, f in enumerate(fotos):
            dst = os.path.join(wd, "raw", f"{path.strip('/').split('/')[0][:60]}__{n}.jpg")
            if not os.path.exists(dst):
                subprocess.run(["curl", "-sS", "-o", dst, f + "?fm=jpg&w=1200&fit=max"], check=True)
            arquivos.append([f, dst])
        out.append(dict(slug=path, url=url, nome=nome, peso=peso, base_tabela=base if prot else None,
                        proteinas=prot, fotos=arquivos))
        print(nome, "|", peso, "|", prot, "|", len(fotos), flush=True)
    json.dump(out, open(os.path.join(wd, "livup.json"), "w"), ensure_ascii=False, indent=1)


def recorte_caixa(path, m=12, mn=150, diff=25):
    """Recorta a foto na caixa aberta (maior região clara pouco saturada) para tirar da cena
    o prato de salada e o copo de decoração, que não fazem parte da porção."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    im = Image.open(path).convert("RGB"); a = np.array(im).astype(int)
    white = (a.min(2) > mn) & ((a.max(2) - a.min(2)) < diff)
    white = ndimage.binary_closing(ndimage.binary_opening(white, iterations=2), iterations=6)
    lab, n = ndimage.label(white)
    k = 1 + int(np.argmax(ndimage.sum(white, lab, range(1, n + 1))))
    ys, xs = np.where(lab == k)
    H, W = white.shape
    return im.crop((max(0, xs.min() - m), max(0, ys.min() - m), min(W, xs.max() + m), min(H, ys.max() + m)))


def montar(wd, aceitos, cego, repo):
    dados = {d["slug"]: d for d in json.load(open(os.path.join(wd, "livup.json")))}
    itens = []
    for r in csv.DictReader(open(aceitos, encoding="utf-8")):
        d = dados[r["slug"]]
        f_url, f_loc = d["fotos"][int(r["foto"])]
        peso = float(re.sub(r"\D", "", d["peso"]))
        p100 = float(d["proteinas"].replace("g", "").replace(",", ".").strip())
        itens.append(dict(origem="br", fonte="Liv Up (site oficial www.livup.com.br)", categoria="marmita",
                          descricao=d["nome"], peso_porcao_g=peso, proteina_g=round(p100 * peso / 100, 2),
                          confianca=r["confianca"], url_pagina=d["url"], url_imagem=f_url,
                          observacoes=(f"tabela do site: {p100} g prot/{d['base_tabela'].replace('Porção de ', '')}"
                                       f" × {peso:.0f} g (embalagem = 1 porção); foto de estúdio da marmita aberta, recortada na caixa (sem salada/copo de cena)" + (f"; {r['obs']}" if r["obs"] else "")),
                          _src=f_loc))
    random.Random(20261109).shuffle(itens)
    os.makedirs(cego, exist_ok=True)
    cols = ["prato_id", "origem", "fonte", "categoria", "descricao", "peso_porcao_g", "proteina_g", "confianca",
            "url_pagina", "url_imagem", "observacoes"]
    gp = os.path.join(repo, "gabarito_pratos.csv")
    existentes = {r["prato_id"] for r in csv.DictReader(open(gp, encoding="utf-8"))}
    with open(gp, "a", newline="", encoding="utf-8") as f1, \
         open(os.path.join(repo, "gabarito.csv"), "a", newline="", encoding="utf-8") as f2:
        w1 = csv.DictWriter(f1, fieldnames=cols, extrasaction="ignore"); w2 = csv.writer(f2)
        for n, it in enumerate(itens, 1):
            it["prato_id"] = f"c{n:03d}"
            assert it["prato_id"] not in existentes
            recorte_caixa(it["_src"]).save(os.path.join(cego, it["prato_id"] + ".jpg"), quality=92)
            w1.writerow(it)
            w2.writerow([it["prato_id"], it["categoria"], "prato inteiro", it["peso_porcao_g"],
                         round(it["proteina_g"] / it["peso_porcao_g"] * 100, 2)])
    print(len(itens), "pratos Liv Up")


if __name__ == "__main__":
    {"raspar": raspar, "montar": montar}[sys.argv[1]](*sys.argv[2:])
