#!/usr/bin/env python3
"""Monta a lista de DENSIDADE (proteína por grama) das receitas do Povarenok.ru com foto do prato pronto.

Fonte: d0rj/povarenok_recipes_detail, Hugging Face (https://huggingface.co/datasets/d0rj/povarenok_recipes_detail),
raspagem de https://www.povarenok.ru. O cartão diz Apache-2.0, mas isso cobre só a raspagem: pelas regras do site,
os direitos das receitas e das fotos são do Povarenok e dos autores. Por isso `uso_declarado = sem_licenca`.

O que tem: 154 mil receitas russas, cada uma com a foto principal do prato pronto (`main_image_url`, no próprio site)
e a tabela nutricional que O SITE calcula a partir das quantidades dos ingredientes: por 100 g do prato, do prato
inteiro e por porção (`nae_value`). A densidade vem pronta, por 100 g; o peso do prato inteiro sai da razão
kcal do prato / kcal por 100 g. Ingrediente sem quantidade ("a gosto") não entra na conta do site.

Uso (precisa de pyarrow):
  baixar os dois parquet de data/ do repositório acima (200 MB no total)
  python3 -I receitas_povarenok_monta.py monta <parquet> [<parquet> ...] [--saida <csv>]
As fotos baixam de `url_imagem` (www.povarenok.ru). ESTE HOST ESTAVA BLOQUEADO na sessão que montou a lista:
nenhuma foto foi aberta, e a vista das fotos não foi conferida.

UMA LINHA POR RECEITA. Colunas de alvo de treino_v2 VAZIAS; densidade e o resto em colunas `*_fraca`
(natureza_rotulo = receita_ingredientes_x_tabela_do_site). Regras, com contagem impressa:
  V1  sem tabela nutricional, ou kcal por 100 g zero: sai;
  V2  categoria fora de pratos de comida: ficam Горячие блюда (pratos quentes), Салаты (saladas),
      Бульоны и супы (sopas) e Закуски (entradas); saem doces, bolos, pães, bebidas, conservas e o resto;
  V3  mais de 25% dos ingredientes que pesam (fora sal, pimenta, ervas, temperos e água) sem quantidade: sai;
  V3b alguma fonte de proteína (carne, peixe, ovo, queijo, leguminosa) sem quantidade: sai (a conta do site a ignora,
      e a densidade cai: "frango a gosto" deu 0,7 g/100 g);
  V4  kcal fora de 25% de 4P + 4C + 9G (por 100 g): sai;
  V5  densidade de proteína acima de 40 g/100 g: sai;
  V6  a mesma foto em duas receitas: fica a primeira;
  V7  porção (peso do prato / porções) fora de 50 a 1.500 g: colunas por porção vazias; a densidade fica.
`fontes_presentes` = fontes de proteína pelos nomes dos ingredientes (regras FONTES_RU; sem peso por ingrediente).
`grupo` = título normalizado; regra de reserva de 12.5.
"""
import ast, csv, hashlib, os, re, sys
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import confere_esquema  # noqa: E402

SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_receitas_povarenok.csv")
LICENCA, USO = "sem licença do titular (raspagem do povarenok.ru; cartão do HF diz Apache-2.0 só para a raspagem)", "sem_licenca"
FICA = {"Горячие блюда", "Салаты", "Бульоны и супы", "Закуски"}
SEM_PESO_OK = re.compile(r"соль|перец|специ|зелен|приправ|лавров|вода|укроп|петрушк|кинз|базилик|чеснок|сахар|уксус|паприк|карри|орегано|тмин|корица", re.I)
FONTES_RU = [
    ("frango_peru", r"(?<!яйцо )курин|куриц|куры|окорочк|цыпл|индей|утк[аи]|утин|гусь|гуся"),
    ("bovina", r"говя|телят|баранин"),
    ("suina_embutido", r"свин|бекон|колбас|сосиск|сардел|ветчин|грудинк|сало|фарш"),
    ("peixe_marisco", r"рыб|лосос|сёмг|семг|форел|треск|тунец|тунц|кревет|кальмар|минтай|скумбри|сельд(?!ерей)|горбуш|морепрод|краб|миди|икра|судак|щук|хек"),
    ("ovo", r"яйц|яичн"),
    ("leguminosa_soja", r"фасол|\bнут\b|чечевиц|горох|тофу|соев"),
    ("laticinio", r"сыр(?!о[йего])|творог|сметан|молок|йогурт|кефир|сливк|брынз|моцарел"),
    ("nozes_sementes", r"орех|миндал|семечк|кунжут|фисташ"),
]
PROTEINA_ALTA_G = 25.0


def cmd_monta(arqs, saida):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("precisa de pyarrow")
    base = confere_esquema()
    cab = (base + ["dominio", "grupo", "vista", "split_ext", "url_imagem", "url_receita", "categoria",
                   "densidade_proteina_fraca_g100", "kcal_fraca_100g", "gordura_fraca_g100", "carbo_fraca_g100",
                   "massa_prato_fraca_g", "porcoes", "massa_porcao_fraca_g", "proteina_porcao_fraca_g", "fontes_presentes",
                   "natureza_rotulo", "componentes", "n_componentes", "avisos", "licenca", "uso_declarado"])
    cont = Counter(); linhas = []; fotos = set()
    for arq in arqs:
        cols = ["title", "page_url", "main_image_url", "ingredients", "portions_count", "nae_value", "kroshki"]
        for r in pq.read_table(arq, columns=cols).to_pylist():
            cont["receitas_lidas"] += 1
            nv = r["nae_value"]
            nv = ast.literal_eval(nv) if isinstance(nv, str) else (nv or {})
            c100, tot = nv.get("100 г блюда") or {}, nv.get("Готового блюда") or {}
            p, k, g, c = (c100.get(x) for x in ("protein", "kcal", "fats", "carb"))
            if None in (p, k, g, c) or not k or not tot.get("kcal"):
                cont["V1_sem_tabela"] += 1; continue
            cat = (r["kroshki"] or [""])[0]
            if cat not in FICA:
                cont["V2_categoria"] += 1; continue
            ing = r["ingredients"] or []
            pesam = [a for a in ing if not SEM_PESO_OK.search(a.get("name") or "")]
            if pesam and sum(1 for a in pesam if not a.get("count")) > 0.25 * len(pesam):
                cont["V3_ingrediente_sem_quantidade"] += 1; continue
            if any(not a.get("count") and any(re.search(rx, (a.get("name") or "").lower()) for _, rx in FONTES_RU)
                   for a in ing):
                cont["V3b_fonte_de_proteina_sem_quantidade"] += 1; continue
            if abs(4 * p + 4 * c + 9 * g - k) > 0.25 * k:
                cont["V4_kcal_incoerente"] += 1; continue
            if p > 40:
                cont["V5_densidade_acima_de_40"] += 1; continue
            if not r["main_image_url"] or r["main_image_url"] in fotos:
                cont["V6_foto_repetida_ou_ausente"] += 1; continue
            fotos.add(r["main_image_url"])
            massa = 100 * tot["kcal"] / k
            try:
                porc = float(re.sub(r"[^\d.]", "", r["portions_count"] or "") or 0)
            except ValueError:
                porc = 0
            mp = massa / porc if porc else 0
            ok = 50 <= mp <= 1500
            cont["V7_porcao_fora"] += not ok
            nomes = [(a.get("name") or "").lower() for a in ing]
            fontes = [f for f, rx in FONTES_RU if any(re.search(rx, n) for n in nomes)]
            avisos = [a for a, s in (("porcao_fora_de_50_a_1500g", not ok), ("sopa", cat == "Бульоны и супы")) if s]
            nome = re.sub(r"\W+", " ", (r["title"] or "").lower()).strip()
            grupo = "pvk_" + hashlib.sha1(nome.encode()).hexdigest()[:10]
            split = "teste_ext" if int(hashlib.sha1(grupo.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
            rid = re.search(r"/show/(\d+)", r["page_url"] or "")
            linhas.append([f"pvk_{rid.group(1) if rid else cont['receitas_lidas']}", -1, -1, ""] + [""] * (len(base) - 4)
                          + ["receitas_povarenok", grupo, "receita_web", split, r["main_image_url"], r["page_url"], "/".join((r["kroshki"] or [])[:3]),
                             f"{p:.2f}", f"{k:.1f}", f"{g:.2f}", f"{c:.2f}", f"{massa:.0f}", f"{porc:g}" if porc else "",
                             f"{mp:.0f}" if ok else "", f"{p * mp / 100:.1f}" if ok else "", ";".join(fontes),
                             "receita_ingredientes_x_tabela_do_site",
                             ";".join(f"{(a.get('name') or '').replace(';', ',')}:{a.get('count') or '?'}" for a in ing)[:1200],
                             len(ing), "|".join(avisos), LICENCA, USO])
    vistos, unicas = set(), []                 # a mesma receita nos dois arquivos: fica a primeira
    for l in linhas:
        if l[0] in vistos:
            cont["V6_receita_repetida"] += 1; continue
        vistos.add(l[0]); unicas.append(l)
    linhas = unicas
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    ix = {k: cab.index(k) for k in cab}
    d = sorted(float(l[ix["densidade_proteina_fraca_g100"]]) for l in linhas)
    pp = [float(l[ix["proteina_porcao_fraca_g"]]) for l in linhas if l[ix["proteina_porcao_fraca_g"]]]
    print("regras:", dict(cont))
    print(f"receitas (linhas): {len(linhas)} | split: {dict(Counter(l[ix['split_ext']] for l in linhas))}")
    print("  por categoria:", dict(Counter(l[ix["categoria"]].split("/")[0] for l in linhas)))
    print(f"  densidade de proteína (g/100 g): p10 {d[len(d) // 10]:.1f} mediana {d[len(d) // 2]:.1f} p90 {d[9 * len(d) // 10]:.1f}"
          f" | faixas 0-3/3-6/6-10/10-15/>15: {[sum(a < v <= b for v in d) for a, b in ((-1, 3), (3, 6), (6, 10), (10, 15), (15, 99))]}")
    print(f"  proteína por porção > {PROTEINA_ALTA_G:.0f} g: {sum(v > PROTEINA_ALTA_G for v in pp)} de {len(pp)}")
    print("  receitas com a fonte:", dict(Counter(f for l in linhas for f in l[ix["fontes_presentes"]].split(";") if f)))
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "monta":
        a = sys.argv[2:]
        saida = SAIDA
        if "--saida" in a:
            saida = a[a.index("--saida") + 1]; a = a[:a.index("--saida")]
        cmd_monta(a, saida)
    else:
        sys.exit(__doc__)
