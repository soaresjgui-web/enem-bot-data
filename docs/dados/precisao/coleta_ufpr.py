#!/usr/bin/env python3
"""Extrai fotos de porções pesadas do Manual Fotográfico de Quantificação Alimentar
(Crispim et al., UFPR 2017) e calcula a proteína da porção pela TACO (4ª ed.).

Uso: python3 coleta_ufpr.py <manual.pdf> <taco_food.csv> <taco_nutrients.csv> <dir_saida>

- manual.pdf: https://nutritotal.com.br/pro/wp-content/uploads/2019/02/manual_quantificacao_alimentar-compactado-1.pdf
- TACO: https://raw.githubusercontent.com/raulfdm/taco-api/main/references/csv/{food,nutrients}.csv
- Gramas por foto: tabela "Gramatura das porções alimentares" (pp. 143-144 do manual).
- Cada foto é recortada pela caixa da imagem no PDF (pdfplumber), sem o rótulo
  "Pxxx – n" que fica abaixo dela. Saída: <dir_saida>/<codigo>_<n>.png + ufpr_itens.csv.
"""
import csv
import os
import subprocess
import sys

import pdfplumber
from PIL import Image

# Excluídos: P302 (omelete e ovos mexidos na mesma foto), P147 (tábua com vários queijos),
# P047 (almôndega grande + pequenas lado a lado) e P086 (dois tipos de fígado) — a foto mostra
# alternativas lado a lado e não fica claro a que parte se refere a gramatura.
# código: (descrição, página impressa, gramas por foto 1..6, id TACO, categoria, nota, fotos escolhidas)
ITENS = {
    "P036": ("Arroz branco cozido", 85, [60, 112, 171, 239, 318, 403], 3, "almoco_pf", "", [1, 2, 3, 4, 5, 6]),
    "P021": ("Feijão (sem caldo)", 68, [80, 140, 210, 280, 360], 561, "almoco_pf", "", [1, 2, 3, 4, 5]),
    "P124": ("Feijão (com caldo)", 69, [61, 105, 146, 174, 354, 454], 567, "almoco_pf",
             "foto é de feijão preto; TACO 'Feijão, preto, cozido'; caldo extra pode diluir", [2, 4]),
    "P145": ("Feijoada", 71, [70, 236, 402, 568, 734, 900], 540, "almoco_pf",
             "manual informa volume (ml); assumido 1 g/ml", [1, 2, 3, 4, 5]),
    "P143": ("Baião de dois", 72, [37, 68, 115, 163, 252, 380], 527, "almoco_pf", "", [1, 2, 3, 4, 5, 6]),
    "P760": ("Lentilha cozida", 70, [30, 48, 75, 119, 189, 300], 577, "jantar", "", [3, 5]),
    "P773": ("Bife bovino", 99, [55, 93, 153, 180, 235, 310], 344, "almoco_pf",
             "corte não especificado; usado contra-filé com gordura grelhado", [1, 2, 3, 4, 5, 6]),
    "P755": ("Filé de frango", 101, [22, 34, 49, 76, 102, 158], 410, "almoco_pf",
             "usado peito sem pele grelhado", [1, 2, 3, 4, 5, 6]),
    "P826": ("Filé mignon em pedaços", 104, [48, 82, 137, 163, 216, 285], 358, "jantar", "", [1, 2, 3, 4, 5, 6]),
    "P718": ("Estrogonofe", 111, [15, 50, 100, 180, 260, 340], 538, "almoco_pf",
             "foto parece ragu de frango com legumes (álbum IARC); TACO 'Estrogonofe de frango' — confiança C", [3, 5]),
    "P140": ("Vaca atolada", 110, [89, 136, 182, 288, 329, 527], 553, "jantar", "", [2, 4, 6]),
    "P043": ("Carne de porco em fatias", 103, [53, 89, 111, 172, 225], 435, "almoco_pf",
             "usado pernil assado", [1, 2, 3, 4, 5]),
    "P148": ("Carne seca", 100, [22, 37, 63, 102, 150, 229], 384, "almoco_pf", "", [1, 2, 3, 4, 5, 6]),
    "P146": ("Torresmo", 107, [33, 48, 62, 83, 106, 126], 445, "lanche",
             "TACO sem torresmo; usado 'Toucinho, frito'", [1, 2, 3, 4, 5]),
    "P832": ("Presunto em fatias", 105, [18, 34, 59, 97, 158, 200], 439, "cafe", "", [1, 2, 3, 4, 5, 6]),
    "P834": ("Salame", 106, [23, 35, 56, 92, 118, 149], 443, "lanche", "", [1, 2, 3, 4, 5, 6]),
    "P051": ("Filé de peixe", 115, [64, 97, 151, 199, 285], 301, "jantar",
             "espécie não informada; usado merluza filé assado", [1, 2, 3, 4, 5]),
    "P120": ("Camarão pequeno", 114, [91, 157, 223, 289], 284, "restaurante", "", [1, 2, 3, 4]),
    "P139": ("Camarão médio sem casca", 113, [59, 86, 120, 187, 273, 366], 284, "restaurante", "", [1, 2, 3, 4, 5, 6]),
    "P404": ("Macarronada", 96, [70, 140, 220, 310, 400], 542, "jantar", "TACO 'Macarrão, molho bolognesa'", [1, 2, 3, 4, 5]),
    "P003": ("Batata frita", 48, [62, 97, 142, 184, 265], 93, "lanche", "", [2, 4]),
    "P125": ("Batata-doce cozida", 49, [68, 123, 198, 257, 310, 434], 88, "almoco_pf", "", [3]),
    "P001": ("Batata inglesa cozida, pedaços grandes", 46, [70, 141, 228, 304, 405, 507], 91, "almoco_pf", "", [3]),
    "P126": ("Mandioca cozida", 50, [118, 149, 238, 327, 415], 129, "almoco_pf", "", [2]),
    "P017": ("Brócolis", 57, [78, 131, 182, 238, 305], 100, "jantar", "", [3]),
}


def taco(food_csv, nutr_csv):
    nomes = {r["id"]: r["name"] for r in csv.DictReader(open(food_csv, encoding="utf-8"))}
    prot = {r["foodId"]: r["protein"] for r in csv.DictReader(open(nutr_csv, encoding="utf-8"))}
    return nomes, prot


def main(pdf_path, food_csv, nutr_csv, out):
    os.makedirs(out, exist_ok=True)
    nomes, prot = taco(food_csv, nutr_csv)
    pdf = pdfplumber.open(pdf_path)
    rows = []
    for cod, (desc, pag, gramas, tid, cat, nota, escolha) in ITENS.items():
        idx = pag + 2  # página do PDF (1-based) = página impressa + 2
        page = pdf.pages[idx - 1]
        words = page.extract_words()
        png = os.path.join(out, f"_page{idx}.png")
        dpi = 150
        if not os.path.exists(png):
            subprocess.run(["pdftoppm", "-f", str(idx), "-l", str(idx), "-r", str(dpi), "-png", "-singlefile",
                            pdf_path, png[:-4]], check=True)
        img = Image.open(png).convert("RGB")
        s = dpi / 72
        # associa cada imagem ao rótulo "Pxxx – n" logo abaixo dela
        for im in page.images:
            lab = [w for w in words if w["top"] > im["bottom"] and w["top"] < im["bottom"] + 25
                   and w["x0"] >= im["x0"] - 5 and w["x1"] <= im["x1"] + 5 and w["text"].isdigit()]
            if not lab:
                continue
            n = int(lab[0]["text"])
            if n not in escolha:
                continue
            box = tuple(int(round(v * s)) for v in (im["x0"], im["top"], im["x1"], im["bottom"]))
            dst = os.path.join(out, f"{cod}_{n}.png")
            img.crop(box).save(dst)
            g = gramas[n - 1]
            p100 = float(prot[str(tid)])
            rows.append(dict(codigo=cod, foto=n, descricao=desc, pagina=pag, gramas=g, taco_id=tid,
                             taco_nome=nomes[str(tid)], proteina_100g=p100,
                             proteina_g=round(g * p100 / 100, 1), categoria=cat, nota=nota, arquivo=dst))
    with open(os.path.join(out, "ufpr_itens.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "fotos")


if __name__ == "__main__":
    main(*sys.argv[1:5])
