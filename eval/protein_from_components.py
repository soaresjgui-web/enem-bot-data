#!/usr/bin/env python3
"""v2: converte estimativas POR COMPONENTE (gramas) em proteína por prato, usando tabelas.

O estimador cego (prompt_v2.md) nunca calcula proteína: ele devolve, por componente,
`alimento_ref` + `gramas_estimadas`. Este script casa `alimento_ref` com:
  - TACO 4ª ed. (docs/dados/precisao/ref/taco.csv) — nomes brasileiros;
  - Nutrition5k ingredients_metadata.csv (proteína por g, USDA) — nomes genéricos em inglês;
usando difflib (stdlib), e soma gramas × proteína/100 g.

Uso:
  python3 protein_from_components.py estimativas_v2.csv saida_pratos.csv \
      [--taco ref/taco.csv] [--n5k ref/n5k_ingredients.csv] \
      [--componentes saida_componentes.csv] [--sistema nome] [--limiar 0.72]

Entrada (uma linha por componente):
  prato_id,componente,alimento_ref,gramas_estimadas,confianca,pista_escala[,sistema]
  `alimento_ref` pode vir com prefixo "TACO:" ou "USDA:" para forçar a tabela.
Saída por prato (compatível com score_precisao.py e simula_seletor.py):
  prato_id,sistema,proteina_estimada_g,peso_total_estimado_g,itens,observacoes
Saída por componente (--componentes; usada por calibra.py e simula_seletor_componentes.py):
  prato_id,sistema,componente,alimento_ref,ref_casado,tabela,similaridade,proteina_100g,grupo,
  gramas_estimadas,proteina_g,confianca,pista_escala
Nomes sem casamento (similaridade < limiar) entram com proteína 0, são listados no stderr e em
<saida_pratos>.nao_casados.txt para revisão (acrescentar sinônimo em SINONIMOS abaixo).
"""
import argparse
import csv
import difflib
import os
import re
import sys
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from grupos_alimentos import grupo_de, normaliza  # noqa: E402

REF = os.path.join(AQUI, "..", "docs", "dados", "precisao", "ref")
TACO_PADRAO = os.path.join(REF, "taco.csv")
N5K_PADRAO = os.path.join(REF, "n5k_ingredients.csv")

# sinônimos frequentes -> nome exato da tabela (chave já normalizada)
SINONIMOS = {
    "arroz branco cozido": "Arroz, tipo 1, cozido",
    "arroz branco": "Arroz, tipo 1, cozido",
    "arroz": "Arroz, tipo 1, cozido",
    "arroz integral": "Arroz, integral, cozido",
    "feijao": "Feijão, carioca, cozido",
    "feijao carioca": "Feijão, carioca, cozido",
    "feijao preto": "Feijão, preto, cozido",
    "frango grelhado": "Frango, peito, sem pele, grelhado",
    "peito de frango grelhado": "Frango, peito, sem pele, grelhado",
    "ovo cozido": "Ovo, de galinha, inteiro, cozido/10minutos",
    "ovo frito": "Ovo, de galinha, inteiro, frito",
    "pao frances": "Pão, trigo, francês",
    "bife": "Carne, bovina, contra-filé, sem gordura, grelhado",
    "bife grelhado": "Carne, bovina, contra-filé, sem gordura, grelhado",
    "bife de alcatra": "Carne, bovina, miolo de alcatra, sem gordura, grelhado",
    "carne moida": "Carne, bovina, acém, moído, cozido",
    "peixe grelhado": "Merluza, filé, assado",
    "file de peixe": "Merluza, filé, assado",
    "pure de batata": "mashed potatoes",
    "salada verde": "Alface, crespa, crua",
    "green salad": "mixed greens",
    "salada": "Alface, crespa, crua",
    "farofa": "Mandioca, farofa, temperada",
    "camarao": "Camarão, Rio Grande, grande, cozido",
    "frango desfiado": "Frango, peito, sem pele, cozido",
    "peito de frango": "Frango, peito, sem pele, grelhado",
    "carne bovina": "Carne, bovina, contra-filé, sem gordura, grelhado",
    "bife bovino": "Carne, bovina, contra-filé, sem gordura, grelhado",
    "file mignon": "Carne, bovina, filé mingnon, sem gordura, grelhado",
    "presunto": "Presunto, com capa de gordura",
    "carne seca": "Carne, bovina, charque, cozido",
    "charque": "Carne, bovina, charque, cozido",
    "macarronada": "Macarrão, trigo, cozido (USDA 20121)",
    "torresmo": "Toucinho, frito",
    "carne de porco": "Porco, pernil, assado",
    "camarao pequeno": "Camarão, Rio Grande, grande, cozido",
    "estrogonofe": "Estrogonofe de frango",
    "strogonoff": "Estrogonofe de frango",
    "baiao de dois": "Baião de dois, arroz e feijão-de-corda",
    "feijoada": "Feijoada",
    "lasanha": "lasagna",
    "file de frango": "Frango, peito, sem pele, grelhado",
    "coxa de frango": "Frango, coxa, com pele, assada",
    "sobrecoxa de frango": "Frango, sobrecoxa, com pele, assada",
    "frango frito": "Frango, coxa, com pele, assada",
    "ovos mexidos": "scrambled eggs",
    "ovo mexido": "scrambled eggs",
    "omelete": "omelets",
    "peixe frito": "Pescada, filé, frito",
    "peixe assado": "Merluza, filé, assado",
    "linguica": "Lingüiça, porco, grelhada",
    "costela": "Carne, bovina, costela, assada",
    "picanha": "Carne, bovina, picanha, com gordura, grelhada",
    "alcatra": "Carne, bovina, miolo de alcatra, sem gordura, grelhado",
    "tilapia": "tilapia",
    "salmao": "Salmão, sem pele, fresco, grelhado",
    "atum": "Atum, conserva em óleo",
    "sardinha": "Sardinha, assada",
    "queijo": "Queijo, minas, frescal",
    "mussarela": "Queijo, mozarela",
    "mucarela": "Queijo, mozarela",
}
# itens que a TACO não tem cozidos: proteína/100 g de fontes alternativas (g/100 g)
EXTRA = {
    "Macarrão, trigo, cozido (USDA 20121)": 5.8,  # = IBGE POF 2008-09, cód. 6503401 prep. 2
}
SINONIMOS["macarrao cozido"] = "Macarrão, trigo, cozido (USDA 20121)"
SINONIMOS["macarrao"] = "Macarrão, trigo, cozido (USDA 20121)"


STOP = {"de", "da", "do", "das", "dos", "com", "sem", "e", "a", "o", "em", "ao", "na", "no", "tipo", "and",
        "with", "of", "the", "in"}


def toks(s):
    return {t for t in re.split(r"[\s,/()\-]+", s) if t and t not in STOP}


def tok_igual(a, b):
    if a == b:
        return True
    return min(len(a), len(b)) >= 4 and a[:3] == b[:3] and difflib.SequenceMatcher(None, a, b).ratio() > 0.8


def pontua(q, qt, c, ct, en):
    """0..1: F1 de tokens (com igualdade aproximada) + palavra-núcleo + similaridade de string."""
    if not qt or not ct:
        return 0.0
    hit_q = sum(any(tok_igual(a, b) for b in ct) for a in qt)
    hit_c = sum(any(tok_igual(b, a) for a in qt) for b in ct)
    prec, rec = hit_c / len(ct), hit_q / len(qt)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # núcleo: 1ª palavra nos nomes da TACO ("Frango, peito, ..."), última nos nomes em inglês ("grilled chicken")
    pc = [t for t in re.split(r"[\s,/()\-]+", c) if t]
    pq = [t for t in re.split(r"[\s,/()\-]+", q) if t]
    nucleo_c = pc[-1] if en else pc[0]
    nucleo_q = pq[-1] if en else pq[0]
    nucleo = 1.0 if tok_igual(nucleo_c, nucleo_q) else (0.5 if any(tok_igual(nucleo_c, a) for a in qt) else 0.0)
    seq = difflib.SequenceMatcher(None, q, c).ratio()
    score = 0.45 * f1 + 0.35 * nucleo + 0.20 * seq
    # sem "cru" na consulta, penaliza itens crus (a comida no prato quase sempre está pronta)
    if ("cru" in ct or "crua" in ct or "raw" in ct) and not ({"cru", "crua", "raw"} & qt):
        score -= 0.1
    return score


class Tabela:
    def __init__(self, taco_path, n5k_path):
        self.itens = {}  # nome_normalizado -> (nome, tabela, prot100, grupo_taco)
        if taco_path and os.path.exists(taco_path):
            for r in csv.DictReader(open(taco_path, encoding="utf-8")):
                if r["proteina_100g"] == "":
                    continue
                self.itens[normaliza(r["nome"])] = (r["nome"], "TACO", float(r["proteina_100g"]), r["grupo"])
        for nome, p in EXTRA.items():
            self.itens[normaliza(nome)] = (nome, "EXTRA", p, None)
        if n5k_path and os.path.exists(n5k_path):
            for r in csv.DictReader(open(n5k_path, encoding="utf-8")):
                nome = r["ingr"].strip()
                if nome in ("deprecated", "plate only"):
                    continue
                k = normaliza(nome)
                if k not in self.itens:
                    self.itens[k] = (nome, "USDA", 100 * float(r["protein(g)"]), None)
        self.chaves = list(self.itens)
        self.por_tabela = defaultdict(list)
        for k, v in self.itens.items():
            self.por_tabela[v[1]].append(k)
        self.cache = {}
        self._toks = {}

    def toks(self, c):
        if c not in self._toks:
            self._toks[c] = toks(c)
        return self._toks[c]

    def casa(self, ref):
        """-> (nome, tabela, prot100, grupo_taco, similaridade)"""
        bruto = ref.strip()
        forca = None
        m = re.match(r"^(taco|usda)\s*:\s*(.*)$", bruto, re.I)
        if m:
            forca, bruto = m.group(1).upper(), m.group(2)
        q = normaliza(bruto)
        if (q, forca) in self.cache:
            return self.cache[(q, forca)]
        res = None
        if q in SINONIMOS:
            alvo = normaliza(SINONIMOS[q])
            if alvo in self.itens:
                res = (*self.itens[alvo], 1.0)
        if res is None and q in self.itens and (forca is None or self.itens[q][1] == forca):
            res = (*self.itens[q], 1.0)
        if res is None and forca is None:
            # sinônimo contido na consulta como palavras inteiras ("carne moida refogada" -> "carne moida")
            dentro = [k for k in SINONIMOS if re.search(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", q)]
            if dentro:
                k = max(dentro, key=len)
                res = (*self.itens[normaliza(SINONIMOS[k])], 0.9)
        if res is None:
            pool = self.por_tabela[forca] if forca else self.chaves
            if forca == "USDA":
                pool = pool + self.por_tabela.get("EXTRA", [])
            qt = toks(q)
            cand = set(difflib.get_close_matches(q, pool, n=10, cutoff=0.5))
            for c in pool:
                if qt & self.toks(c) or any(tok_igual(a, b) for a in qt for b in self.toks(c)):
                    cand.add(c)
            melhor, best = None, -1.0
            for c in cand:
                score = pontua(q, qt, c, self.toks(c), self.itens[c][1] == "USDA")
                if score > best or (score == best and len(c) < len(melhor)):
                    melhor, best = c, score
            if melhor is not None:
                res = (*self.itens[melhor], round(best, 3))
        if res is None:
            res = ("", "", 0.0, None, 0.0)
        self.cache[(q, forca)] = res
        return res


def num(x):
    try:
        return float(str(x).replace(",", ".").strip())
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--taco", default=TACO_PADRAO)
    ap.add_argument("--n5k", default=N5K_PADRAO,
                    help="ingredients_metadata.csv do Nutrition5k (cópia em ref/n5k_ingredients.csv)")
    ap.add_argument("--componentes", default="")
    ap.add_argument("--sistema", default="")
    ap.add_argument("--limiar", type=float, default=0.5)
    a = ap.parse_args()

    tab = Tabela(a.taco, a.n5k)
    if not tab.itens:
        sys.exit("nenhuma tabela carregada")
    pratos = defaultdict(lambda: {"prot": 0.0, "peso": 0.0, "itens": [], "obs": []})
    comps, nao = [], defaultdict(int)
    with open(a.entrada, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = r["prato_id"].strip()
            sistema = a.sistema or r.get("sistema", "").strip() or "v2"
            g = num(r.get("gramas_estimadas", ""))
            if g is None:
                print(f"aviso: {pid}: gramas inválidas {r.get('gramas_estimadas')!r}", file=sys.stderr)
                continue
            ref = (r.get("alimento_ref") or r.get("componente") or "").strip()
            nome, tabela, p100, gtaco, sim = tab.casa(ref)
            if sim < a.limiar:
                nao[ref] += 1
                p100_usado = 0.0
            else:
                p100_usado = p100
            prot = g * p100_usado / 100
            grupo = grupo_de(nome or ref, gtaco) if sim >= a.limiar else grupo_de(ref)
            d = pratos[(pid, sistema)]
            d["prot"] += prot
            d["peso"] += g
            d["itens"].append(f"{r.get('componente', ref).strip()}:{g:.0f}")
            if sim < a.limiar:
                d["obs"].append(f"sem casamento: {ref}")
            comps.append({"prato_id": pid, "sistema": sistema, "componente": r.get("componente", "").strip(),
                          "alimento_ref": ref, "ref_casado": nome if sim >= a.limiar else "",
                          "tabela": tabela if sim >= a.limiar else "", "similaridade": sim,
                          "proteina_100g": round(p100_usado, 2), "grupo": grupo, "gramas_estimadas": g,
                          "proteina_g": round(prot, 2), "confianca": r.get("confianca", "").strip(),
                          "pista_escala": r.get("pista_escala", "").strip()})

    with open(a.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prato_id", "sistema", "proteina_estimada_g", "peso_total_estimado_g", "itens", "observacoes"])
        for (pid, sis), d in sorted(pratos.items()):
            w.writerow([pid, sis, round(d["prot"], 1), round(d["peso"]), "; ".join(d["itens"]), "; ".join(d["obs"])])
    if a.componentes:
        with open(a.componentes, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(comps[0]))
            w.writeheader()
            w.writerows(comps)
    with open(a.saida + ".nao_casados.txt", "w", encoding="utf-8") as f:
        for ref, n in sorted(nao.items(), key=lambda x: -x[1]):
            f.write(f"{n}\t{ref}\n")
    print(f"{len(pratos)} pratos, {len(comps)} componentes; {sum(nao.values())} componentes sem casamento "
          f"({len(nao)} nomes distintos) -> {a.saida}.nao_casados.txt", file=sys.stderr)
    for ref, n in sorted(nao.items(), key=lambda x: -x[1])[:15]:
        print(f"  sem casamento ({n}x): {ref}", file=sys.stderr)


if __name__ == "__main__":
    main()
