#!/usr/bin/env python3
"""Monta a lista de treino externo do FlavoriaFoodWeight1700 (seções 12.4 e 12.5 de resultado_modelo_treinado.md).

Dataset: Sarapisto, T., Koivunen, L., Mäkilä, T., Klami, A., Ojansivu, P. (2022). FlavoriaFoodWeight1700:
Automated Lunch Line Meal Pictures with Automatic Measurement of Weight and Contents (1.0.0). Zenodo.
https://doi.org/10.5281/zenodo.5850856 — licença CC BY 4.0 (atribuição obrigatória).

Uso:
  python3 -I flavoria_monta.py mapa                      # recalcula proteina_100g de flavoria_componentes.csv
  python3 -I flavoria_monta.py monta <dataset_csv.tar | pasta com os 11 CSVs> [--saida <csv>]
O dataset_csv.tar (1,3 MB) vem de https://zenodo.org/api/records/5850856/files/dataset_csv.tar/content.
As imagens NÃO são lidas aqui: a lista de arquivos do tar de imagens (2,6 GB) está em flavoria_arquivos.csv.

UMA LINHA POR FOTO (não por prato). Cada bandeja tem de 1 a 4 quadros ("enter" e "identification"); todos os
quadros da mesma bandeja repetem os mesmos rótulos e o mesmo `grupo`. `quadro_preferido` = 1 marca um quadro por
prato (o mais limpo). `arquivo` é o caminho exato dentro de dataset_images_and_csv.tar (ex.: images/train/...png).

Entradas versionadas (são nossas):
  flavoria_componentes.csv  os 246 nomes de componente -> grupo (8), fonte de proteína (10), referência de tabela
                            para a densidade de proteína (rótulo FRACO), confiança, se fica no prato, e a razão
  flavoria_arquivos.csv     listagem do tar de imagens (caminho, bytes) + medidas por quadro (largura, altura,
                            arco_borda_oculta, brilho), gravada por flavoria_quadros.py lendo o tar em fluxo;
                            as medidas podem estar vazias (aí `oclusao` sai vazia)

Regras de limpeza, nesta ordem (as contagens saem na tela):
  E1  linhas de pesagem repetidas: fica uma. Duas sessões aparecem no arquivo `registered` e no `non_registered`
      da mesma semana, com a hora de identificação alguns segundos diferente; vale a versão `registered`;
  E2  pesagens abaixo de 5 g: descartadas (ruído de balança ao tocar o talher; o histograma tem um pico em 3 g);
  P1  bandeja com sopa ou sobremesa (no_prato = 0 no mapa): sai, porque parte da comida fica fora da foto;
  P2  bandeja com pesagem implausível: mais de 300 g numa cuba de salada ou mais de 650 g numa cuba quente
      (vale dos histogramas; são trocas ou reabastecimento de cuba). Sai a bandeja inteira: a porção real
      daquela cuba não é recuperável;
  P3  massa total fora de 30 a 1.200 g depois de E1 e E2: sai;
  F1  foto sem sessão de pesagem da mesma bandeja a até 90 s: sai a foto; sessão sem foto não entra.
Pesagens repetidas na mesma cuba (a pessoa se serviu duas vezes) são somadas e levam o aviso `repeticao`.

Horários: `registered_2020_10_05-2020_10_18.csv` está em UTC; os outros arquivos, em hora local da Finlândia
(UTC+3 até 24/10/2020, UTC+2 depois). O nome da foto traz a época UTC; a foto "identification" coincide com
`user_identification_time` ao segundo.

Parte segurada (12.5): os grupos (bandejas) dos últimos dias de coleta. Dias inteiros, do último para trás; o
corte é o número de dias que deixa a fração de grupos mais perto de 20%. `split_ext` = teste_ext; o resto, treino_ext.
"""
import csv, datetime, io, os, re, sys, tarfile
from collections import Counter, defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(AQUI, "..", "..", "docs", "dados", "precisao", "ref")
MAPA = os.path.join(AQUI, "flavoria_componentes.csv")
ARQUIVOS = os.path.join(AQUI, "flavoria_arquivos.csv")
SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_flavoria.csv")

GRUPOS = ["proteina_animal", "leguminosa", "laticinio", "amido", "vegetal", "gordura_molho", "misto", "outros"]
FONTES = ["frango_peru", "bovina", "suina_embutido", "peixe_marisco", "ovo", "leguminosa_soja", "laticinio",
          "prato_misto", "nozes_sementes", "sem_proteina"]
ALVOS = ["massa", "proteina", "kcal", "gordura", "carbo"]

MIN_EVENTO_G = 5.0
MAX_SALADA_G, MAX_QUENTE_G = 300.0, 650.0
MIN_PRATO_G, MAX_PRATO_G = 30.0, 1200.0
TOL_FOTO_S = 90
FRACAO_TESTE = 0.20
ARCO_OCLUSAO = 0.20            # maior arco da borda do prato encoberto, em fração de volta
PROTEINA_FRACA_ALTA_G = 25.0
# procedência em cada linha (decisão de 2026-11-12: toda lista externa leva `licenca` e `uso_declarado`)
LICENCA, USO = "CC BY 4.0", "comercial_ok"
ARQ_UTC = "registered_2020_10_05-2020_10_18"
FIM_HORARIO_VERAO = datetime.date(2020, 10, 25)

APELIDOS = {
    "T_ARROZ": "taco:Arroz, tipo 1, cozido", "T_BATATA": "taco:Batata, inglesa, cozida",
    "T_CENOURA": "taco:Cenoura, crua", "T_CENOURA_COZ": "taco:Cenoura, cozida",
    "T_TOMATE": "taco:Tomate, com semente, cru", "T_ALFACE": "taco:Alface, americana, crua",
    "T_REPOLHO": "taco:Repolho, branco, cru", "T_REPOLHO_ROXO": "taco:Repolho, roxo, cru",
    "T_PEPINO": "taco:Pepino, cru", "T_BROCOLIS": "taco:Brócolis, cozido", "T_COUVEFLOR": "taco:Couve-flor, cozida",
    "T_BETERRABA": "taco:Beterraba, cozida", "T_MILHO": "taco:Milho, verde, enlatado, drenado",
    "T_ERVILHA": "taco:Ervilha, enlatada, drenada", "T_FEIJAO": "taco:Feijão, carioca, cozido",
    "T_RABANETE": "taco:Rabanete, cru", "T_NABO": "taco:Nabo, cru", "T_ABOBRINHA": "taco:Abobrinha, italiana, cozida",
    "T_PIMENTAO": "taco:Pimentão, vermelho, cru", "T_ABACAXI": "taco:Abacaxi, cru",
    "T_TANGERINA": "taco:Tangerina, Poncã, crua", "T_LARANJA": "taco:Laranja, pêra, crua",
    "T_MELANCIA": "taco:Melancia, crua", "T_MELAO": "taco:Melão, cru", "T_PESSEGO": "taco:Pêssego, enlatado, em calda",
    "T_AZEITONA": "taco:Azeitona, verde, conserva", "T_OVO_FRITO": "taco:Ovo, de galinha, inteiro, frito",
    "T_OVO_COZ": "taco:Ovo, de galinha, inteiro, cozido/10minutos", "T_IOGURTE": "taco:Iogurte, natural",
    "T_BERINJELA": "taco:Berinjela, cozida",
    # peixe empanado de cozinha industrial: 55% peixe, 25% farinha de rosca, 20% água e óleo
    "EMPANADO_PEIXE": "0.55*n5k:fish|0.25*n5k:bread crumbs|0.2*DILUENTE",
}


# ------------------------------------------------------------------ tabelas de referência e mapa
def tabelas():
    t = {}
    with open(os.path.join(REF, "taco.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["proteina_100g"]:
                t["taco:" + r["nome"]] = float(r["proteina_100g"])
    with open(os.path.join(REF, "n5k_ingredients.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t.setdefault("n5k:" + r["ingr"].strip(), float(r["protein(g)"]) * 100)   # 1ª ocorrência ("deprecated" repete)
    t["DILUENTE"] = 0.0     # água, óleo, osso: massa sem proteína
    return t


def _termos(ref, peso=1.0):
    """Expande apelidos: '0.6*EMPANADO_PEIXE|0.4*T_ARROZ' -> [(peso, 'n5k:fish'), ...]."""
    for p in ref.split("|"):
        w, nome = (p.split("*", 1) if "*" in p else ("1", p))
        w = peso * float(w)
        if nome in APELIDOS:
            yield from _termos(APELIDOS[nome], w)
        else:
            yield w, nome


def densidade(ref, tab):
    """Referência -> g de proteína por 100 g. Os pesos têm de somar 1."""
    tot = soma = 0.0
    for w, nome in _termos(ref):
        if nome not in tab:
            raise KeyError(f"referência sem linha na tabela: {nome!r}")
        tot += w * tab[nome]; soma += w
    if abs(soma - 1) > 0.011:
        raise ValueError(f"pesos de {ref!r} somam {soma}")
    return tot


def ler_mapa(confere=True):
    tab = tabelas()
    with open(MAPA, encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))
    for r in linhas:
        assert r["grupo"] in GRUPOS and r["fonte"] in FONTES and r["confianca"] in "ABC" and r["no_prato"] in "01", r
        d = round(densidade(r["ref"], tab), 2)
        if confere:
            assert abs(float(r["proteina_100g"]) - d) < 0.006, (r["componente"], r["proteina_100g"], d)
        r["dens"] = d
    return linhas


def componente_de(nome, mapa, _cache={}):
    """Linha do mapa para um nome do CSV. Nome exato; senão, o único nome do mapa que é prefixo dele
    (um nome do CSV tem apóstrofo corrompido)."""
    if nome not in _cache:
        ex = [r for r in mapa if r["componente"] == nome] or [r for r in mapa if nome.startswith(r["componente"])]
        assert len(ex) == 1, f"componente sem linha única no mapa: {nome!r}"
        _cache[nome] = ex[0]
    return _cache[nome]


def cmd_mapa():
    tab = tabelas()
    with open(MAPA, encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f)); campos = list(linhas[0].keys())
    for r in linhas:
        r["proteina_100g"] = f"{densidade(r['ref'], tab):.2f}"
    with open(MAPA, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, lineterminator="\n"); w.writeheader(); w.writerows(linhas)
    print(MAPA, len(linhas), "componentes |", dict(Counter(r["confianca"] for r in linhas)))


# ------------------------------------------------------------------ pesagens
def _hora(s):
    return datetime.datetime.strptime(s.strip().replace(".", "-"), "%Y-%m-%d %H:%M:%S")


def _utc(dt, arquivo):
    """Hora do CSV -> (época UTC, data local da Finlândia)."""
    desloc = 3 if dt.date() < FIM_HORARIO_VERAO else 2
    if arquivo == ARQ_UTC:
        return int((dt - datetime.datetime(1970, 1, 1)).total_seconds()), (dt + datetime.timedelta(hours=desloc)).date()
    return int((dt - datetime.timedelta(hours=desloc) - datetime.datetime(1970, 1, 1)).total_seconds()), dt.date()


def ler_pesagens(origem):
    """-> lista de dicts (sessao, arquivo, registrado, bandeja, balanca, componente, gramas, t_evento, t_ident, data)."""
    textos = {}
    if os.path.isdir(origem):
        for n in sorted(os.listdir(origem)):
            if n.endswith(".csv"):
                textos[n[:-4]] = open(os.path.join(origem, n), "rb").read()
    else:
        with tarfile.open(origem) as tf:
            for m in sorted(tf.getmembers(), key=lambda m: m.name):
                if m.isfile() and m.name.endswith(".csv"):
                    textos[os.path.basename(m.name)[:-4]] = tf.extractfile(m).read()
    assert len(textos) == 11, f"esperava 11 CSVs, achei {len(textos)}"
    ev = []
    for arq, bruto in textos.items():
        r = csv.reader(io.StringIO(bruto.decode("utf-8-sig")))
        cab = [h.strip() for h in next(r) if h.strip()]
        for linha in r:
            if not any(c.strip() for c in linha):
                continue
            x = dict(zip(cab, [c.strip() for c in linha[:len(cab)]]))
            t_id, data = _utc(_hora(x["user_identification_time"]), arq)
            ev.append(dict(sessao=x["session_id"], arquivo=arq, registrado=int(arq.startswith("registered")),
                           bandeja=x["tray_id"], balanca=x["scale_identifier"], componente=x["component_name"],
                           gramas=float(x["weight_of_a_component"]),
                           t_evento=_utc(_hora(x["weighing_event_time"]), arq)[0], t_ident=t_id, data=data))
    return ev


def sessoes(ev, mapa):
    """Aplica E1, E2, P1, P2, P3. -> (dict chave -> sessão limpa, contagens)."""
    cont = Counter(pesagens=len(ev))
    em_reg = {x["sessao"] for x in ev if x["registrado"]}
    dupla = {x["sessao"] for x in ev if not x["registrado"]} & em_reg
    cont["E1_sessoes_em_dois_arquivos"] = len(dupla)
    vistos, unicos = set(), []
    for x in ev:
        if x["sessao"] in dupla and not x["registrado"]:   # E1: a mesma sessão exportada duas vezes; vale a registrada
            cont["E1_linhas_repetidas"] += 1; continue
        k = (x["sessao"], x["bandeja"], x["t_ident"], x["t_evento"], x["balanca"], x["gramas"], x["componente"])
        if k in vistos:
            cont["E1_linhas_repetidas"] += 1; continue
        vistos.add(k); unicos.append(x)
    por = defaultdict(list)
    for x in unicos:
        por[(x["sessao"], x["bandeja"], x["t_ident"])].append(x)
    assert len({k[0] for k in por}) == len(por), "session_id repetido com bandeja ou hora diferente"
    cont["sessoes"] = len(por)
    out = {}
    for k, l in por.items():
        s = dict(sessao=k[0], bandeja=k[1], t_ident=k[2], data=l[0]["data"],
                 registrado=max(x["registrado"] for x in l), motivo="")
        bons = [x for x in l if x["gramas"] >= MIN_EVENTO_G]
        cont["E2_pesagens_abaixo_de_5g"] += len(l) - len(bons)
        fora = any(componente_de(x["componente"], mapa)["no_prato"] == "0" for x in l)
        falha = any(x["gramas"] > (MAX_SALADA_G if "salaatti" in x["balanca"] else MAX_QUENTE_G) for x in bons)
        massa = sum(x["gramas"] for x in bons)
        if fora:
            s["motivo"] = "P1_sopa_ou_sobremesa"
        elif falha:
            s["motivo"] = "P2_pesagem_implausivel"
        elif not (MIN_PRATO_G <= massa <= MAX_PRATO_G):
            s["motivo"] = "P3_massa_fora_de_30_a_1200g"
        comp = defaultdict(float)
        for x in bons:
            comp[x["componente"]] += x["gramas"]
        s["comp"] = sorted(comp.items(), key=lambda t: (-t[1], t[0]))
        s["massa"] = massa
        s["repeticao"] = len({(x["balanca"]) for x in bons}) < len(bons)
        out[k] = s
    return out, cont


# ------------------------------------------------------------------ fotos
_RX_FOTO = re.compile(r"^images/(train|val|test)/(tray\d+)_(\d+)_([\d.]+)_(enter|identification)_[\d.]+\.png$")


def ler_fotos():
    fotos = []
    with open(ARQUIVOS, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            m = _RX_FOTO.match(r["caminho"])
            if not m:
                assert not r["caminho"].endswith(".png"), r["caminho"]
                continue
            fotos.append(dict(arquivo=r["caminho"], pasta=m.group(1), bandeja=m.group(2), epoca=int(m.group(3)),
                              tipo=m.group(5), bytes=int(r["bytes"]), arco=r.get("arco_borda_oculta", ""),
                              largura=r.get("largura", ""), altura=r.get("altura", "")))
    return fotos


def casa_fotos(fotos, sess):
    """F1: cada foto vai para a sessão da mesma bandeja com identificação mais próxima, a até TOL_FOTO_S."""
    por_bandeja = defaultdict(list)
    for k, s in sess.items():
        por_bandeja[s["bandeja"]].append(k)
    cont = Counter(fotos=len(fotos))
    casadas = defaultdict(list)
    for f in fotos:
        cand = sorted((abs(f["epoca"] - k[2]), k) for k in por_bandeja.get(f["bandeja"], []))
        if not cand or cand[0][0] > TOL_FOTO_S:
            cont["F1_foto_sem_sessao"] += 1; continue
        assert len(cand) == 1 or cand[1][0] > TOL_FOTO_S, ("foto ambígua", f["arquivo"])
        f["dif_s"] = f["epoca"] - cand[0][1][2]
        casadas[cand[0][1]].append(f)
    return casadas, cont


def preferido(quadros):
    """Índice do quadro mais limpo: o 'identification' (o 'enter' pega mais mãos e cabeças), a não ser que a medida
    de borda encoberta diga que ele está pior que um 'enter'; empate: o mais tardio."""
    def nota(f):
        a = float(f["arco"]) if f["arco"] else 0.0
        return (a >= ARCO_OCLUSAO, f["tipo"] != "identification", round(a, 3), -f["epoca"])
    return min(range(len(quadros)), key=lambda i: nota(quadros[i]))


# ------------------------------------------------------------------ montagem
def confere_esquema():
    """As colunas de alvo têm de ser as de listas/treino_v2.csv, na mesma ordem."""
    with open(os.path.join(os.path.dirname(SAIDA), "treino_v2.csv"), encoding="utf-8") as f:
        v2 = next(csv.reader(f))
    nosso = (["dish_id", "cadeia", "fold", "split_oficial"] + ALVOS + [f"m_{g}" for g in GRUPOS]
             + [f"p_{g}" for g in GRUPOS] + [f"ff_{k}" for k in FONTES])
    assert v2 == nosso, "o esquema de treino_v2.csv mudou"
    return nosso


def confere_saida(saida, cab):
    """Conferências da lista gravada (assert): ids únicos, somas, grupos de um lado só, arquivos que existem no tar."""
    with open(saida, encoding="utf-8", newline="") as f:
        r = list(csv.DictReader(f))
    assert list(r[0].keys()) == cab and len({x["dish_id"] for x in r}) == len(r)
    no_tar = {f["arquivo"] for f in ler_fotos()}
    lado, rot = {}, {}
    for x in r:
        assert x["arquivo"] in no_tar and x["dominio"] == "flavoria" and x["fold"] == "-1"
        assert all(x[c] == "" for c in ["proteina", "kcal", "gordura", "carbo"] + [f"p_{g}" for g in GRUPOS])
        m = float(x["massa"])
        assert MIN_PRATO_G <= m <= MAX_PRATO_G and abs(sum(float(x[f"m_{g}"]) for g in GRUPOS) - m) < 0.6
        assert abs(sum(float(x[f"ff_{k}"]) for k in FONTES) - 1) < 0.002
        assert lado.setdefault(x["grupo"], (x["split_ext"], x["data"])) == (x["split_ext"], x["data"])
        chave = tuple(x[c] for c in cab[4:35] + ["componentes"])
        assert rot.setdefault(x["grupo"], chave) == chave, "quadros do mesmo prato com rótulos diferentes"
    for g in lado:
        assert sum(x["quadro_preferido"] == "1" for x in r if x["grupo"] == g) == 1
    teste = {d for s_, d in lado.values() if s_ == "teste_ext"}; treino = {d for s_, d in lado.values() if s_ == "treino_ext"}
    assert teste and treino and min(teste) > max(treino), "o teste_ext tem de ser os últimos dias"


def cmd_monta(origem, saida):
    mapa = ler_mapa()
    base = confere_esquema()
    sess, cont = sessoes(ler_pesagens(origem), mapa)
    casadas, c2 = casa_fotos(ler_fotos(), sess)
    cont.update(c2)
    cont["sessoes_com_foto"] = len(casadas)
    cont["sessoes_sem_foto"] = len(sess) - len(casadas)
    for k in casadas:
        cont["com_foto__" + (sess[k]["motivo"] or "fica")] += 1
    ficam = sorted((k for k in casadas if not sess[k]["motivo"]), key=lambda k: (sess[k]["data"], k[2], k[0]))
    # parte segurada: últimos dias inteiros até 20% dos grupos
    por_dia = Counter(sess[k]["data"] for k in ficam)
    dias = sorted(por_dia, reverse=True)
    acum = [sum(por_dia[d] for d in dias[:i]) / len(ficam) for i in range(len(dias) + 1)]
    n_dias = min(range(1, len(dias)), key=lambda i: (abs(acum[i] - FRACAO_TESTE), i))   # dias inteiros, o mais perto de 20%
    dias_teste = dias[:n_dias]
    cab = (base + ["dominio", "grupo", "vista", "split_ext", "data", "arquivo", "quadro",
              "quadro_preferido", "n_quadros", "sessao", "bandeja", "registrado", "dif_s", "componentes", "n_componentes",
              "proteina_fraca_g", "proteina_fraca_conf", "proteina_fraca_fontes_g", "arco_borda_oculta", "oclusao", "avisos",
              "licenca", "uso_declarado"])
    linhas, ids = [], set()
    resumo = dict(massa=[], pf=[], pff=[], fotos=Counter(), split=Counter(), fotos_split=Counter())
    for k in ficam:
        s = sess[k]; quadros = sorted(casadas[k], key=lambda f: (f["epoca"], f["tipo"]))
        mg = dict.fromkeys(GRUPOS, 0.0); ff = dict.fromkeys(FONTES, 0.0)
        pf = pff = 0.0; por_conf = Counter()
        for nome, g in s["comp"]:
            c = componente_de(nome, mapa)
            mg[c["grupo"]] += g; ff[c["fonte"]] += g
            p = g * c["dens"] / 100
            pf += p; por_conf[c["confianca"]] += p
            if c["fonte"] != "sem_proteina":
                pff += p
        avisos = []
        if s["repeticao"]:
            avisos.append("repeticao")
        if any(abs(f["dif_s"]) > 30 for f in quadros):
            avisos.append("foto_a_mais_de_30s")
        split = "teste_ext" if s["data"] in dias_teste else "treino_ext"
        grupo = "flv_" + s["sessao"]
        ip = preferido(quadros)
        resumo["massa"].append(s["massa"]); resumo["pf"].append(pf); resumo["pff"].append(pff)
        resumo["split"][split] += 1
        for i, f in enumerate(quadros):
            did = f"flv_{f['bandeja']}_{f['epoca']}_{f['tipo'][0]}"
            assert did not in ids, did
            ids.add(did)
            resumo["fotos"][f["tipo"]] += 1; resumo["fotos_split"][split] += 1
            ocl = "" if not f["arco"] else int(float(f["arco"]) >= ARCO_OCLUSAO)
            linhas.append([did, -1, -1, f["pasta"], f"{s['massa']:.1f}", "", "", "", ""]
                          + [f"{mg[g]:.1f}" for g in GRUPOS] + [""] * len(GRUPOS)
                          + [f"{ff[x] / s['massa']:.4f}" for x in FONTES]
                          + ["flavoria", grupo, "zenital_mascarada", split, s["data"].isoformat(), f["arquivo"], f["tipo"],
                             int(i == ip), len(quadros), s["sessao"], s["bandeja"], s["registrado"], f["dif_s"],
                             ";".join(f"{n.replace(';', ',')}:{g:.0f}" for n, g in s["comp"]), len(s["comp"]),
                             f"{pf:.1f}", max(por_conf, key=por_conf.get), f"{pff:.1f}", f["arco"], ocl, "|".join(avisos), LICENCA, USO])
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)

    confere_saida(saida, cab)
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    m = resumo["massa"]
    print("contagens:")
    for k_, v in sorted(cont.items()):
        print(f"  {k_}: {v}")
    print(f"pratos na lista: {len(ficam)} | fotos (linhas): {len(linhas)} {dict(resumo['fotos'])}")
    print(f"  quadros por prato: {sorted(Counter(len(casadas[k]) for k in ficam).items())}")
    print(f"  pasta do tar (fotos): {dict(Counter(l[3] for l in linhas))}")
    print(f"  registrado (pratos): {dict(Counter(sess[k]['registrado'] for k in ficam))}")
    print(f"  massa (g): p5 {q(m, .05):.0f} p25 {q(m, .25):.0f} mediana {q(m, .5):.0f} p75 {q(m, .75):.0f} p95 {q(m, .95):.0f} "
          f"máx {max(m):.0f} | > 600 g: {sum(v > 600 for v in m)}")
    pf = resumo["pf"]
    print(f"  proteína FRACA (g): mediana {q(pf, .5):.1f} p90 {q(pf, .9):.1f} | > {PROTEINA_FRACA_ALTA_G:.0f} g: {sum(v > PROTEINA_FRACA_ALTA_G for v in pf)}"
          f" | só das fontes de proteína > {PROTEINA_FRACA_ALTA_G:.0f} g: {sum(v > PROTEINA_FRACA_ALTA_G for v in resumo['pff'])}"
          f" | faixas 0-5/5-15/15-25/25-40/>40: {[sum(a < v <= b for v in pf) for a, b in ((-1, 5), (5, 15), (15, 25), (25, 40), (40, 1e9))]}")
    print(f"  dias de coleta: {len(por_dia)} ({min(por_dia)} a {max(por_dia)}) | teste_ext: {[d.isoformat() for d in sorted(dias_teste)]}"
          f" -> {resumo['split']['teste_ext']} pratos ({resumo['split']['teste_ext'] / len(ficam):.1%}), {resumo['fotos_split']['teste_ext']} fotos"
          f" | treino_ext: {resumo['split']['treino_ext']} pratos, {resumo['fotos_split']['treino_ext']} fotos")
    print(f"  fração acumulada de grupos, do último dia para trás: {[(d.isoformat()[5:], round(a, 3)) for d, a in zip(dias, acum[1:])][:10]}")
    print(f"  avisos (pratos): {dict(Counter(a for l in linhas if l[cab.index('quadro_preferido')] == 1 for a in l[cab.index('avisos')].split('|') if a))}")
    oc = [l[cab.index('oclusao')] for l in linhas]
    print(f"  oclusao (fotos): {dict(Counter(oc))} | preferido por tipo: {dict(Counter(l[cab.index('quadro')] for l in linhas if l[cab.index('quadro_preferido')] == 1))}")
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "mapa":
        cmd_mapa()
    elif len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
