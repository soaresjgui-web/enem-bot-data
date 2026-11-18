#!/usr/bin/env python3
"""Monta a lista de treino externo do DFood-TJ (esquema de treino_v2 + colunas de 12.4 + procedência).

Dataset: "Chinese Food Images for Full-cycle Nutrition Analysis Towards Diabetes Management", Scientific Data 2026
(PMC13333986). Dados: Science Data Bank, doi 10.57760/sciencedb.28012 (versão V3), licença do depósito CC0 1.0
(metadado da página; o artigo é CC BY-NC-ND 4.0, mas vale a do depósito). uso_declarado = comercial_ok.

O que é: refeições de 91 pacientes diabéticos internados num hospital de Tianjin (2020 a 2025). Cada alimento da
refeição foi PESADO (balança de cozinha, 0,1 g; a planilha arredonda para 1 g) e todos vão juntos num prato branco
grande, fotografado com celular duas vezes: uma de cima (ou quase) e outra oblíqua. A partir de 2024, quase sempre há
também uma foto de cada alimento sozinho num pratinho de inox em cima da balança, com o visor aceso.

Onde: https://china.scidb.cn/download?fileId=cb2751a22a516c057913e15b7d3cc5c4 (DFoodTJ-F-Nutr.zip, 4.247.942.139 bytes,
md5 b67cc4116e2c734296fbd6cd99cd50bc). Dentro: DFoodTJ-F-Nutr/img/<6 dígitos>.png (3.280) e DFoodTJ-F-Nutr/truth/
{truth.xls, truth_raw.xlsx}. A tabela vem de https://china.scidb.cn/download?fileId=c308bb046a736249b8223b8baa0de335
(DFoodTJ-Nutr.zip, 50.522 bytes; DFoodTJ-Nutr/density.xlsx, 351 alimentos). O servidor aceita Range: dá para tirar só
as planilhas e as fotos escolhidas (varredura2/scripts/zipremoto.py), mas é lento (uns 36 kB/s por conexão).

Uso:
  python3 -I dfoodtj_monta.py monta <pasta com truth.xls, truth_raw.xlsx e density.xlsx> [--saida <csv>]

Entradas versionadas ao lado deste script:
  dfoodtj_alimentos.csv  mapa food_id -> grupo, fonte, fonte_principal, nomes PT/EN e nota (101 alimentos usados);
  dfoodtj_vistas.csv     vista de cada foto de prato, conferida olhando miniaturas de 384 px em folhas de pares
                         (conferencia: olhada_em_par, olhada_em_par_incerta, olhada_sozinha, olhada_sozinha_incerta).
                         Sem o arquivo, a vista fica `nao_conferida`.

UMA LINHA POR FOTO (3.280). `truth_raw.xlsx` traz, por foto, uma linha por alimento: paciente, data+refeição, nome,
food_id e peso. Alvos: massa = soma dos pesos; proteína, kcal, gordura e carboidrato = peso × tabela (density.xlsx,
"mass density"), recalculados aqui; natureza_rotulo = pesado_x_tabela. O valor dos autores (truth.xls) fica em
proteina_autores_g. Tipos de foto (`vista`):
  zenital       foto do prato inteiro, a mais de cima do par;
  obliqua_45    foto do prato inteiro, a mais inclinada do par (na prática de 30 a 60 graus);
  item_balanca  um alimento sozinho, no pratinho em cima da balança, com o peso no visor (rótulo visível na foto:
                não usar para treinar massa sem tapar o visor);
  nao_conferida foto de prato sem vista conferida.
Refeição (`refeicao`): fotos de prato do mesmo paciente e mesma data+refeição; se forem mais de 2, separa pelo
conjunto de food_id. Foto de um alimento só cujo (food_id, peso) aparece numa foto de prato da mesma refeição é
item_balanca; se não aparece, é a própria refeição (refeição de um alimento, só no início, paciente 0).
Regras e avisos:
  R1  nome do alimento diferente do nome da tabela para aquele food_id (7 food_id, 29 linhas): a identidade é incerta;
      alvos de nutrição, p_*, m_* e ff_* ficam vazios (massa fica); aviso nome_difere_do_id;
  R2  as duas fotos de prato da mesma refeição com pesos ou itens diferentes: cada uma fica com o seu rótulo; aviso
      rotulo_difere_no_par;
  R3  proteína recalculada difere da dos autores em mais de 0,1 g (só fígado de porco, food_id 222: a planilha usa
      10,73% e a tabela 18,2%): aviso proteina_autores_difere;
  R4  kcal da tabela fora de 25% de 4P + 4C + 9G na refeição: aviso kcal_incoerente;
  R5  ovo cozido (food_id 6) pesado com casca, e a tabela é da parte comestível: aviso ovo_com_casca (proteína do ovo
      superestimada em uns 12%);
  R6  foto de prato com todos os pesos múltiplos de 10 g (19 fotos, pacientes 0 e 6, 2020 e 2021; no resto só 10% dos
      pesos são redondos, o esperado ao acaso): aviso pesos_redondos (peso provavelmente estimado ou arredondado);
  R7  o mesmo food_id duas vezes na mesma foto: aviso food_id_repetido. A foto 1688 (item na balança, food_id 42,
      140 g) corresponde ao segundo "18" das fotos 1684/1685: fica como item_balanca (ITEM_BALANCA_FORCADO).
`grupo` = paciente (unidade de independência); split_ext pela regra de reserva de 12.5:
sha1(grupo) mod 5 == 0 -> teste_ext.
"""
import csv, hashlib, os, re, sys
from collections import Counter, defaultdict
from datetime import date

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import FONTES, GRUPOS, confere_esquema  # noqa: E402

SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_dfoodtj.csv")
MAPA = os.path.join(AQUI, "dfoodtj_alimentos.csv")
VISTAS = os.path.join(AQUI, "dfoodtj_vistas.csv")
LICENCA, USO = "CC0 1.0", "comercial_ok"
URL = "https://china.scidb.cn/download?fileId=cb2751a22a516c057913e15b7d3cc5c4"
FAIXAS = ((-1, 5), (5, 15), (15, 25), (25, 40), (40, 1e9))
OVO = 6
# Foto de um alimento só que é foto de item na balança (visor visível), mas cujo food_id não bate com a foto do prato:
# a foto do prato 1684/1685 lista o food_id 18 duas vezes, e o item de 140 g é o 42 (conferido na foto).
ITEM_BALANCA_FORCADO = {1688: 1685}


def le_planilhas(pasta):
    import pandas as pd   # só aqui: o resto do repositório não depende de pandas
    raw = pd.read_excel(os.path.join(pasta, "truth_raw.xlsx"), sheet_name="Sheet1")
    aut = pd.read_excel(os.path.join(pasta, "truth.xls"))
    den = pd.read_excel(os.path.join(pasta, "density.xlsx"), sheet_name="Sheet1")
    assert list(raw.columns[:6]) == ["Image ID", "Patient ID", "Date", "Food Name", "Food ID", "Weight (g)"], raw.columns
    tab = {int(r["Food ID"]): dict(zh=str(r["Food Chinese Name"]).strip(), en=str(r["Food English Name"]).strip(),
                                   p=float(r["Protein mass density (g/g)"]), k=float(r["Energy mass density (kcal/g)"]),
                                   g=float(r["Fat mass density (g/g)"]), c=float(r["Carbohydrate mass density (g/g)"]))
           for _, r in den.iterrows()}
    fotos, cur = [], None
    for _, r in raw.iterrows():
        if r["Image ID"] == r["Image ID"]:                     # não NaN: começa uma foto nova
            cur = dict(img=int(r["Image ID"]), pac=int(r["Patient ID"]), data=str(r["Date"]).strip(), itens=[])
            fotos.append(cur)
        assert int(r["Patient ID"]) == cur["pac"]
        cur["itens"].append((int(r["Food ID"]), str(r["Food Name"]).strip(), float(r["Weight (g)"]), float(r["Protein (g)"])))
    autores = {int(r["Image ID"]): dict(ids=[int(x) for x in str(r["Food ID"]).strip("[]").split(",")], p=float(r["Protein (g)"]))
               for _, r in aut.iterrows()}
    return fotos, autores, tab


def le_mapa(tab):
    with open(MAPA, encoding="utf-8", newline="") as fh:
        m = {int(r["food_id"]): r for r in csv.DictReader(fh)}
    for fid, r in m.items():
        assert r["grupo"] in GRUPOS and r["fonte"] in FONTES and r["fonte_principal"] in FONTES + [""], r
        assert r["nome_zh"] == tab[fid]["zh"] and abs(float(r["proteina_100g"]) - 100 * tab[fid]["p"]) < 0.006, (fid, r)
    return m


def le_vistas():
    if not os.path.exists(VISTAS):
        return {}
    with open(VISTAS, encoding="utf-8", newline="") as fh:
        v = {int(r["imagem"]): r for r in csv.DictReader(fh)}
    assert all(r["vista"] in ("zenital", "obliqua_45") for r in v.values())
    return v


def refeicoes(fotos):
    """Atribui a cada foto uma refeição e o papel (prato ou item_balanca). Determinístico pela ordem das fotos."""
    por_rd = defaultdict(list)
    for f in fotos:
        por_rd[(f["pac"], f["data"])].append(f)
    ref, papel = {}, {}
    for (pac, data), fs in por_rd.items():
        pratos = [f for f in fs if len(f["itens"]) > 1]
        pesados = {(i[0], i[2]) for f in pratos for i in f["itens"]}
        soltos = [f for f in fs if len(f["itens"]) == 1]
        for f in soltos:
            if (f["itens"][0][0], f["itens"][0][2]) in pesados or f["img"] in ITEM_BALANCA_FORCADO:
                papel[f["img"]] = "item_balanca"
            else:
                pratos.append(f)                                # refeição de um alimento (ou item fora da foto do prato)
        if len(pratos) <= 2:
            grupos = [pratos]
        else:
            g = defaultdict(list)
            for f in pratos:
                g[tuple(sorted({i[0] for i in f["itens"]}))].append(f)
            grupos = list(g.values())
        for k, gr in enumerate(sorted(grupos, key=lambda x: min(f["img"] for f in x))):
            rid = f"dftj_p{pac:02d}_{data_iso(data)}_{tipo(data)}" + (f"_{k + 1}" if len(grupos) > 1 else "")
            for f in gr:
                ref[f["img"]] = rid; papel.setdefault(f["img"], "prato")
        for f in fs:
            if papel[f["img"]] == "item_balanca":
                dono = [x for x in fs if papel.get(x["img"]) == "prato" and len(x["itens"]) > 1
                        and ((f["itens"][0][0], f["itens"][0][2]) in {(i[0], i[2]) for i in x["itens"]}
                             or ITEM_BALANCA_FORCADO.get(f["img"]) == x["img"])]
                ref[f["img"]] = ref[min(dono, key=lambda x: x["img"])["img"]]
    return ref, papel


def data_iso(s):
    a, m, d = re.match(r"(\d{3,4})/(\d{1,2})/(\d{1,2})", s).groups()
    if len(a) == 3:                                             # uma linha diz "021/6/20Lunch"
        a = "2" + a
    return date(int(a), int(m), int(d)).isoformat()


def tipo(s):
    return re.sub(r"[\d/]", "", s).strip()


def cmd_monta(pasta, saida):
    fotos, autores, tab = le_planilhas(pasta)
    mapa, vistas = le_mapa(tab), le_vistas()
    assert sorted(f["img"] for f in fotos) == list(range(1, len(fotos) + 1))
    ref, papel = refeicoes(fotos)
    base = confere_esquema()
    cab = base + ["dominio", "grupo", "vista", "split_ext", "arquivo", "url_zip", "paciente", "data", "tipo_refeicao",
                  "refeicao", "fotos_prato_na_refeicao", "componentes", "n_componentes", "fontes_presentes", "maior_fonte_g",
                  "proteina_autores_g", "conferencia_vista", "natureza_rotulo", "avisos", "licenca", "uso_declarado"]
    n_prato = Counter(ref[f["img"]] for f in fotos if papel[f["img"]] == "prato")
    rot_ref = defaultdict(set)
    for f in fotos:
        if papel[f["img"]] == "prato":
            rot_ref[ref[f["img"]]].add(tuple(sorted((i[0], i[2]) for i in f["itens"])))
    cont = Counter(); linhas = []
    for f in sorted(fotos, key=lambda x: x["img"]):
        img = f["img"]
        assert autores[img]["ids"] == [i[0] for i in f["itens"]], img
        mg = dict.fromkeys(GRUPOS, 0.0); pg = dict.fromkeys(GRUPOS, 0.0); ff = dict.fromkeys(FONTES, 0.0)
        tot = dict(p=0.0, k=0.0, g=0.0, c=0.0); massa = maior = 0.0; avisos = []; comp = []; presentes = set()
        for fid, nome, w, _ in f["itens"]:
            t, m = tab[fid], mapa[fid]
            if nome.lower() != t["en"].lower() and "nome_difere_do_id" not in avisos:
                avisos.append("nome_difere_do_id")
            massa += w; mg[m["grupo"]] += w; ff[m["fonte"]] += w
            p = w * t["p"]; pg[m["grupo"]] += p
            tot["p"] += p; tot["k"] += w * t["k"]; tot["g"] += w * t["g"]; tot["c"] += w * t["c"]
            if m["fonte_principal"]:
                presentes.add(m["fonte_principal"])
            if m["fonte"] != "sem_proteina":
                maior = max(maior, p)
            comp.append(f"{fid}:{t['en'].replace(';', ',').replace(':', ' ')}:{w:.0f}")
        if abs(tot["p"] - autores[img]["p"]) > 0.1:
            avisos.append("proteina_autores_difere")
        if abs(4 * tot["p"] + 4 * tot["c"] + 9 * tot["g"] - tot["k"]) > 0.25 * max(tot["k"], 1):
            avisos.append("kcal_incoerente")
        if len(f["itens"]) > 1 and all(i[2] % 10 == 0 for i in f["itens"]):
            avisos.append("pesos_redondos")
        if len({i[0] for i in f["itens"]}) < len(f["itens"]):
            avisos.append("food_id_repetido")
        if img in ITEM_BALANCA_FORCADO:
            avisos.append("id_difere_da_foto_do_prato")
        if any(i[0] == OVO for i in f["itens"]):
            avisos.append("ovo_com_casca")
        if papel[img] == "prato" and len(rot_ref[ref[img]]) > 1:
            avisos.append("rotulo_difere_no_par")
        vi = vistas.get(img, {})
        if papel[img] == "item_balanca":
            vista, conf = "item_balanca", "pelo_rotulo"
        else:
            vista, conf = vi.get("vista", "nao_conferida"), vi.get("conferencia", "")
        incerto = "nome_difere_do_id" in avisos
        for a in avisos:
            cont[a] += 1
        grupo = f"dftj_p{f['pac']:02d}"
        split = "teste_ext" if int(hashlib.sha1(grupo.encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
        nut = [""] * 4 if incerto else [f"{tot['p']:.2f}", f"{tot['k']:.1f}", f"{tot['g']:.2f}", f"{tot['c']:.2f}"]
        linhas.append([f"dftj_{img:06d}", -1, -1, "", f"{massa:.1f}"] + nut
                      + ([""] * len(GRUPOS) if incerto else [f"{mg[k]:.1f}" for k in GRUPOS])
                      + ([""] * len(GRUPOS) if incerto else [f"{pg[k]:.2f}" for k in GRUPOS])
                      + ([""] * len(FONTES) if incerto else [f"{ff[k] / massa:.4f}" for k in FONTES])
                      + ["dfoodtj", grupo, vista, split, f"DFoodTJ-F-Nutr/img/{img:06d}.png", URL, f["pac"], data_iso(f["data"]),
                         tipo(f["data"]), ref[img], n_prato[ref[img]], ";".join(comp), len(comp), ";".join(sorted(presentes)),
                         f"{maior:.1f}", f"{autores[img]['p']:.2f}", conf, "pesado_x_tabela", "|".join(avisos), LICENCA, USO])
    assert len({l[0] for l in linhas}) == len(linhas)
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    resumo(linhas, cab, cont)
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas, {len(linhas)} linhas)")


def faixas(vals):
    return [sum(a < v <= b for v in vals) for a, b in FAIXAS]


def resumo(linhas, cab, cont):
    i = {k: cab.index(k) for k in cab}
    col = lambda l, k: l[i[k]]
    print("avisos (fotos):", dict(cont))
    print("fotos por vista:", dict(Counter(col(l, "vista") for l in linhas)))
    pratos = [l for l in linhas if col(l, "vista") != "item_balanca"]
    refs = {col(l, "refeicao") for l in pratos}
    print(f"fotos de prato: {len(pratos)} | refeições: {len(refs)} | pacientes: {len({col(l, 'grupo') for l in linhas})}"
          f" | fotos de prato por refeição: {dict(Counter(Counter(col(l, 'refeicao') for l in pratos).values()))}")
    print("split (pacientes):", dict(Counter(s for _, s in {(col(l, 'grupo'), col(l, 'split_ext')) for l in linhas})),
          "| split (fotos de prato):", dict(Counter(col(l, "split_ext") for l in pratos)))
    # uma proteína por refeição: a da foto zenital se houver, senão a primeira foto de prato com rótulo
    por_ref = {}
    for l in sorted(pratos, key=lambda l: (col(l, "vista") != "zenital", l[0])):
        if col(l, "proteina") and col(l, "refeicao") not in por_ref:
            por_ref[col(l, "refeicao")] = (float(col(l, "proteina")), col(l, "split_ext"), col(l, "grupo"))
    p = sorted(v[0] for v in por_ref.values())
    m = sorted(float(col(l, "massa")) for l in pratos)
    print(f"refeições com proteína: {len(p)} | proteína (g): p10 {p[len(p) // 10]:.1f} mediana {p[len(p) // 2]:.1f}"
          f" p90 {p[9 * len(p) // 10]:.1f} máx {p[-1]:.1f} | massa da foto de prato (g): mediana {m[len(m) // 2]:.0f}")
    print("  refeições por faixa 0-5/5-15/15-25/25-40/>40:", faixas(p),
          "| por lado:", {s: faixas([v[0] for v in por_ref.values() if v[1] == s]) for s in ("treino_ext", "teste_ext")})
    print("  pacientes com refeição por faixa:", [len({v[2] for v in por_ref.values() if a < v[0] <= b}) for a, b in FAIXAS])
    zen = [float(col(l, "proteina")) for l in pratos if col(l, "vista") == "zenital" and col(l, "proteina")]
    print("  fotos zenitais com proteína, por faixa:", faixas(zen), "total", len(zen))
    print("  fotos de prato com a fonte:", dict(Counter(k for l in pratos for k in col(l, "fontes_presentes").split(";") if k)))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
