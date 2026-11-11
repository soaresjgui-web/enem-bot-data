"""Seção 9 (densidade de proteína): ingredientes por prato, fonte de proteína de cada ingrediente,
alvos de fração de massa e utilitários comuns aos scripts dens_*.py.

Higiene: os ingredientes por prato só são lidos para os pratos de TREINO (alvos) e, na análise de
erro, para o `calib`. `ingredientes()` recusa qualquer dish_id do `teste`/`n5k_v1`.
"""
import csv, os, re, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI); sys.path.insert(0, os.path.join(AQUI, ".."))
from comum import TD, PREC  # noqa: E402
from prep_dados import N5K  # noqa: E402
from grupos_alimentos import normaliza  # noqa: E402

DENS = os.path.join(TD, "resultados", "dens")          # saídas desta seção (fora do repositório)
BANCO_D = os.path.join(DENS, "banco")                   # previsões por configuração (como o zen_banco)
FONTES = ["frango_peru", "bovina", "suina_embutido", "peixe_marisco", "ovo", "leguminosa_soja", "laticinio",
          "prato_misto", "nozes_sementes", "sem_proteina"]

# fonte de proteína de cada ingrediente do Nutrition5k, por palavra-chave no nome (ordem = precedência)
_RX = [
    ("prato_misto", r"pizza|lasagna|sandwich|burrito|dumpling|taco|quesadilla|enchilada|calzone|pot pie|casserole|"
                    r"fried rice|chow mein|lo mein|macaroni and cheese|stew|curr|chili|gumbo|jambalaya|paella|sushi|"
                    r"wrap|gyro|fajita|nacho|quiche|ravioli|tortellini|soup|chowder|goulash|meatloaf|"
                    r"chicken salad|tuna salad|egg salad"),
    ("ovo", r"\begg|omelet|frittata"),
    ("frango_peru", r"chicken|turkey|duck|poultry|hen\b|wings"),
    ("suina_embutido", r"pork|bacon|ham\b|sausage|salami|pepperoni|prosciutto|chorizo|hot dog|bologna|pastrami|ribs|carnitas"),
    ("bovina", r"beef|steak|brisket|veal|bison|lamb|meatball|burger|patties|jerky|corned|roast\b|goat|venison"),
    ("peixe_marisco", r"fish|salmon|tuna|cod\b|tilapia|halibut|trout|shrimp|crab|lobster|scallop|clam|mussel|oyster|"
                      r"squid|calamari|octopus|\beel\b|herring|sardine|anchov|mackerel|snapper|swordfish|flounder|haddock|"
                      r"carp\b|mahi|char\b|seafood|crawfish|bass\b|caviar|roe\b"),
    ("leguminosa_soja", r"tofu|tempeh|edamame|bean|lentil|chickpea|hummus|falafel|\bpeas\b|soy|seitan"),
    ("laticinio", r"cheese|yogurt|milk|cottage|kefir|ricotta|mozzarella|parmesan|feta|paneer|whey|casein|protein powder"),
    ("nozes_sementes", r"almond|walnut|peanut|cashew|pecan|pistachio|hazelnut|\bnuts?\b|seeds?\b|tahini|nut butter"),
]
_RX = [(f, re.compile(r)) for f, r in _RX]


_NAO = re.compile(r"soy sauce|soy milk|almond milk|coconut milk|oat milk|rice milk|eggplant|green beans|snow peas|snap peas|"
                  r"\bchili\b|chilies|caesar salad|garden salad|\bsalads\b|with peel|jalapeno|bean sprouts|vanilla bean|"
                  r"coffee|cocoa|butter\b|cream\b|ice cream|milk chocolate|peanut oil|sesame oil|seed oil")
_LAT = re.compile(r"cheese|yogurt")


def fonte_de(nome):
    n = normaliza(nome)
    if _NAO.search(n) and not re.search(r"peanut butter|almond butter|nut butter", n):
        return "sem_proteina"
    if _LAT.search(n) and not re.search(r"pizza|sandwich|burger|macaroni|cheesecake", n):
        return "laticinio"
    for f, rx in _RX:
        if rx.search(n):
            return f
    return "sem_proteina"


def tabela_n5k():
    """nome do ingrediente -> proteína por grama (tabela do Nutrition5k, cópia em ref/n5k_ingredients.csv)."""
    out = {}
    with open(os.path.join(PREC, "ref", "n5k_ingredients.csv")) as f:
        for r in csv.DictReader(f):
            out[r["ingr"].strip()] = float(r["protein(g)"])
    return out


def _proibidos():
    """dish_id do teste e do n5k_v1: seus ingredientes nunca são lidos por estes scripts."""
    p = set()
    with open(os.path.join(TD, "meta", "zen_aval.csv")) as f:
        for r in csv.DictReader(f):
            if r["split"] != "calib":
                p.add(r["dish_id"])
    return p


def ingredientes(dish_ids):
    """dish_id -> lista de (nome, gramas, proteína_g). Só treino e calib."""
    quer = set(dish_ids)
    assert not quer & _proibidos(), "ingredientes do teste/n5k_v1 não podem ser lidos"
    nomes_meta = {}
    with open(os.path.join(N5K, "ingredients_metadata.csv")) as f:
        for r in csv.DictReader(f):
            nomes_meta[r["id"].strip()] = r["ingr"]
    out = {}
    for cafe in ("cafe1", "cafe2"):
        with open(os.path.join(N5K, f"dish_metadata_{cafe}.csv")) as f:
            for row in csv.reader(f):
                if not row or row[0] not in quer:
                    continue
                ingr = [row[i:i + 7] for i in range(6, len(row) - 6, 7)]
                out[row[0]] = [((x[1].strip() or nomes_meta.get(x[0].strip(), "")), float(x[2]), float(x[6])) for x in ingr]
    return out


class Ingr:
    """Alvos por ingrediente para uma lista de pratos: fração de massa por ingrediente (vocabulário do treino),
    fração de massa e de proteína por fonte, fonte principal."""

    def __init__(self, ids, vocab=None, min_pratos=1):
        ing = ingredientes(ids)
        self.listas = [ing[d] for d in ids]
        if vocab is None:
            from collections import Counter
            c = Counter(n for l in self.listas for n, g, _ in l if g > 0)
            vocab = sorted(n for n, k in c.items() if k >= min_pratos)
        self.vocab = vocab
        ix = {n: i for i, n in enumerate(vocab)}
        tab = tabela_n5k()
        self.dens_ingr = np.array([tab.get(n, 0.0) for n in vocab])
        self.fonte_ingr = np.array([FONTES.index(fonte_de(n)) for n in vocab])
        n = len(ids)
        self.M = np.zeros((n, len(vocab)))              # gramas por ingrediente
        self.MF = np.zeros((n, len(FONTES))); self.PF = np.zeros((n, len(FONTES)))
        for i, l in enumerate(self.listas):
            for nome, g, p in l:
                if nome in ix:
                    self.M[i, ix[nome]] += g
                j = FONTES.index(fonte_de(nome))
                self.MF[i, j] += g; self.PF[i, j] += p
        self.frac = self.M / np.maximum(self.M.sum(1, keepdims=True), 1e-9)
        self.frac_fonte = self.MF / np.maximum(self.MF.sum(1, keepdims=True), 1e-9)
        pt = self.PF.sum(1)
        # fonte principal: a de maior proteína entre as fontes "de verdade", se der >= 50% da proteína do prato
        pp = self.PF[:, :-1]
        self.principal = np.where((pp.max(1) >= 0.5 * np.maximum(pt, 1e-9)) & (pt > 5), pp.argmax(1), len(FONTES) - 1)
        self.n_ingr = np.array([sum(g > 0 for _, g, _ in l) for l in self.listas])


def faixas(p):
    return {"0-5": p <= 5, "5-15": (p > 5) & (p <= 15), "15-25": (p > 15) & (p <= 25), "25-40": (p > 25) & (p <= 40), ">40": p > 40}


def grava_json(nome, obj):
    import json
    os.makedirs(DENS, exist_ok=True)
    with open(os.path.join(DENS, nome), "w") as f:
        json.dump(obj, f, indent=1, default=float, ensure_ascii=False)


# ---------------------------------------------------------------- LLM: fração e densidade por fonte
_PT_EN = [("frango", "chicken"), ("peru", "turkey"), ("carne, bovina", "beef"), ("bovina", "beef"), ("bife", "steak"), ("porco", "pork"),
          ("lingui", "sausage"), ("presunto", "ham"), ("toucinho", "bacon"), ("salmao", "salmon"), ("atum", "tuna"), ("camarao", "shrimp"),
          ("merluza", "fish"), ("pescada", "fish"), ("peixe", "fish"), ("sardinha", "sardines"), ("ovo", "eggs"), ("feijao", "black beans"),
          ("lentilha", "lentils"), ("grao-de-bico", "chickpeas"), ("queijo", "cheese"), ("iogurte", "yogurt"), ("leite", "milk")]


def pt_en(ref):
    """Nome TACO -> palavra-chave em inglês do mesmo alimento (para achar a fonte e o nome do Nutrition5k)."""
    n = normaliza(ref)
    for pt, en in _PT_EN:
        if normaliza(pt) in n:
            return en
    return ref


def llm_v2_fontes(prato_ids):
    """LLM v2 (por componente, em gramas): gramas e proteína por fonte, por prato. -> (MF [n, fontes], PF [n, fontes])."""
    comp = {}
    with open(os.path.join(PREC, "estimativas_v2_comp_casados.csv")) as f:
        for r in csv.DictReader(f):
            comp.setdefault(r["prato_id"], []).append(r)
    MF = np.zeros((len(prato_ids), len(FONTES))); PF = np.zeros_like(MF)
    for i, pid in enumerate(prato_ids):
        for r in comp.get(pid, []):
            j = FONTES.index(fonte_de(pt_en(r["ref_casado"] or r["alimento_ref"])))
            MF[i, j] += float(r["gramas_estimadas"] or 0); PF[i, j] += float(r["proteina_g"] or 0)
    return MF, PF


V3_PARTES = os.environ.get("DENS_V3") or os.path.join(PREC, "estimativas_v3_partes.csv")   # DENS_V3: outro arquivo (ensaio)


def llm_v3(prato_ids, arq=V3_PARTES, pesos_alt=(0.0, 0.25, 0.40), rodada=None):
    """LLM v3 (identificação + partes da massa, prompt_v3.md). Devolve None se o arquivo ainda não existe; senão
    dict(dens [n], tem [n] bool, FF [n, fontes] fração de massa por fonte, DF [n, fontes] densidade por fonte (NaN se ausente),
    fonte [n] índice da fonte principal declarada pelo estimador)."""
    if not os.path.exists(arq):
        return None
    from protein_from_shares import Casador, densidades, TACO_PADRAO, N5K_PADRAO
    with open(arq, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    rod = os.environ.get("DENS_V3_RODADA", "") if rodada is None else str(rodada)   # "1", "2"...: só essa rodada; vazio: todas (média)
    if rod:
        linhas = [l for l in linhas if ((l.get("rodada") or "1").strip() or "1") == rod]
    pratos, comps, _ = densidades(linhas, Casador(TACO_PADRAO, N5K_PADRAO, 0.5), list(pesos_alt))
    ix = {p: i for i, p in enumerate(prato_ids)}
    n = len(prato_ids)
    dens = np.full(n, np.nan); FF = np.zeros((n, len(FONTES))); C = np.zeros((n, len(FONTES))); nr = np.zeros(n)
    fonte = np.full(n, len(FONTES) - 1)          # fonte principal declarada (coluna fonte_principal); 'nenhuma' = última
    for p, d in pratos.items():
        if p in ix:
            dens[ix[p]] = d["dens"]; nr[ix[p]] = d["n_rod"]
            if d["fonte"] in FONTES[:-1]:
                fonte[ix[p]] = FONTES.index(d["fonte"])
    for c in comps:
        if c["prato_id"] in ix:
            i = ix[c["prato_id"]]; j = FONTES.index(fonte_de(pt_en(c["ref_casado"] or c["alimento_ref"])))
            FF[i, j] += c["parte_norm"] / max(nr[i], 1); C[i, j] += c["contrib_dens"] / max(nr[i], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        DF = np.where(FF > 0, C / FF, np.nan)
    return dict(dens=dens, tem=~np.isnan(dens), FF=FF, DF=DF, fonte=fonte)
