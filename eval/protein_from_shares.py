#!/usr/bin/env python3
"""v3: converte IDENTIFICAÇÃO + PARTES DA MASSA (prompt_v3.md) em densidade de proteína e, com uma
massa total, em proteína por prato.

O estimador cego v3 não estima gramas: devolve, por componente, o alimento de referência, uma
segunda hipótese, a certeza, o preparo, a parte da massa (%) e a principal fonte de proteína do prato.
Este script casa os nomes com as tabelas (Nutrition5k/USDA em ref/n5k_ingredients.csv, TACO em
ref/taco.csv), calcula a densidade do prato

    densidade (g de proteína por g de comida) = Σ parte_i × dens_i,
    dens_i = (1 − w) × dens(alimento_ref) + w × dens(alimento_alt),  w = 0 / 0,25 / 0,40 para certeza alta / media / baixa

e, se receber uma massa por prato (--massa), devolve proteína = massa × densidade.
Com mais de uma rodada (coluna `rodada`), a densidade do prato é a média geométrica das rodadas.

Uso:
  python3 protein_from_shares.py estimativas_v3_partes.csv saida_pratos.csv \
      [--massa massas.csv [--massa-col massa_g]] [--componentes saida_componentes.csv] \
      [--sistema nome] [--peso-alt 0,0.25,0.40] [--limiar 0.5] [--taco ...] [--n5k ...]

Entrada (uma linha por componente; cabeçalho exato do prompt_v3.md, `rodada` opcional):
  prato_id,componente,alimento_ref,alimento_alt,certeza,preparo,parte_pct,fonte_principal[,rodada]
--massa: CSV com prato_id e a coluna de massa em gramas (padrão `massa_g`; para usar o peso do LLM v2,
  --massa estimativas_v2_pratos.csv --massa-col peso_total_estimado_g).
Saída por prato (compatível com score_precisao.py quando há --massa):
  prato_id,sistema,proteina_estimada_g,peso_total_estimado_g,densidade_g_por_g,fonte_principal,
  parte_fontes_pct,n_componentes,n_rodadas,soma_partes,itens,observacoes
Saída por componente (--componentes):
  prato_id,rodada,sistema,componente,alimento_ref,ref_casado,tabela,similaridade,proteina_100g,alimento_alt,
  alt_casado,proteina_100g_alt,certeza,peso_alt,proteina_100g_usada,grupo,preparo,parte_pct,parte_norm,contrib_dens,fonte_principal
Nomes sem casamento entram com a densidade da segunda hipótese (se houver) ou 0, e são listados em
<saida>.nao_casados.txt. Partes que não somam 100 são normalizadas (aviso se a soma sair de 95–105).
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from grupos_alimentos import grupo_de, normaliza  # noqa: E402
from protein_from_components import Tabela, TACO_PADRAO, N5K_PADRAO, num  # noqa: E402

FONTES = ["frango_peru", "bovina", "suina_embutido", "peixe_marisco", "ovo", "leguminosa_soja", "laticinio",
          "prato_misto", "nozes_sementes", "nenhuma"]
CERTEZA = {"alta": 0, "media": 1, "média": 1, "baixa": 2}
_GRUPO_FONTE = {"proteina_animal", "leguminosa", "laticinio", "misto"}


class Casador:
    """Casa um nome com as tabelas. Nome que é exatamente um ingrediente do Nutrition5k usa essa tabela
    (os nomes em inglês do prompt v3 são essa lista); o resto segue a regra da v2 (TACO, sinônimos, aproximação)."""

    def __init__(self, taco, n5k, limiar):
        self.tab = Tabela(taco, n5k)
        self.limiar = limiar
        self.n5k = {}
        if n5k and os.path.exists(n5k):
            for r in csv.DictReader(open(n5k, encoding="utf-8")):
                nome = r["ingr"].strip()
                if nome and nome not in ("deprecated", "plate only"):
                    self.n5k[normaliza(nome)] = (nome, 100 * float(r["protein(g)"]))

    def casa(self, ref):
        """-> (nome casado, tabela, proteína/100 g, grupo, similaridade) ou None se não casar."""
        ref = (ref or "").strip()
        if not ref:
            return None
        q = normaliza(ref.split(":", 1)[1] if ref.lower().startswith(("usda:", "taco:")) else ref)
        if not ref.lower().startswith("taco:") and q in self.n5k:
            nome, p100 = self.n5k[q]
            return nome, "USDA", p100, grupo_de(nome), 1.0
        nome, tabela, p100, gtaco, sim = self.tab.casa(ref)
        if sim < self.limiar:
            return None
        return nome, tabela, p100, grupo_de(nome, gtaco), sim


def le_massa(arq, col):
    out = {}
    with open(arq, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = num(r.get(col, ""))
            if v is not None and v > 0:
                out[r["prato_id"].strip()] = v
    return out


def densidades(linhas, casador, pesos_alt, sistema=""):
    """linhas: dicts do CSV v3. Devolve (por_prato, componentes, nao_casados).
    por_prato[prato_id] = dict(dens, fonte, parte_fontes, n_comp, n_rod, soma, itens, obs)."""
    rod = defaultdict(lambda: defaultdict(list))        # prato -> rodada -> linhas
    for r in linhas:
        pid = r["prato_id"].strip()
        if pid:
            rod[pid][(r.get("rodada") or "1").strip() or "1"].append(r)
    comps, nao = [], defaultdict(int)
    pratos = {}
    for pid, rs in rod.items():
        dens_r, fontes, partes_fonte, obs, itens, soma_r, ncomp = [], [], [], [], [], [], []
        for rd, ls in sorted(rs.items()):
            partes = [max(num(l.get("parte_pct", "")) or 0.0, 0.0) for l in ls]
            soma = sum(partes)
            if soma <= 0:
                obs.append(f"rodada {rd}: partes zeradas")
                continue
            if not 95 <= soma <= 105:
                obs.append(f"rodada {rd}: partes somam {soma:.0f}")
            d = 0.0; pf = 0.0
            for l, p in zip(ls, partes):
                c = casador.casa(l.get("alimento_ref", "") or l.get("componente", ""))
                a = casador.casa(l.get("alimento_alt", ""))
                nivel = CERTEZA.get((l.get("certeza") or "alta").strip().lower(), 1)
                w = pesos_alt[nivel] if a else 0.0
                if c is None:
                    nao[(l.get("alimento_ref") or l.get("componente") or "").strip()] += 1
                    if a:
                        w = 1.0
                    else:
                        obs.append(f"sem casamento: {l.get('alimento_ref', '')}")
                p_ref = c[2] if c else 0.0
                p_alt = a[2] if a else 0.0
                p_usada = (1 - w) * p_ref + w * p_alt
                grupo = (c or a or (None, None, None, grupo_de(l.get("componente", ""))))[3]
                pn = p / soma
                d += pn * p_usada / 100
                if grupo in _GRUPO_FONTE or p_usada >= 8:
                    pf += pn
                comps.append({"prato_id": pid, "rodada": rd, "sistema": sistema, "componente": (l.get("componente") or "").strip(),
                              "alimento_ref": (l.get("alimento_ref") or "").strip(), "ref_casado": c[0] if c else "",
                              "tabela": c[1] if c else "", "similaridade": c[4] if c else 0.0, "proteina_100g": round(p_ref, 2),
                              "alimento_alt": (l.get("alimento_alt") or "").strip(), "alt_casado": a[0] if a else "",
                              "proteina_100g_alt": round(p_alt, 2) if a else "", "certeza": (l.get("certeza") or "").strip(),
                              "peso_alt": w, "proteina_100g_usada": round(p_usada, 2), "grupo": grupo,
                              "preparo": (l.get("preparo") or "").strip(), "parte_pct": p, "parte_norm": round(pn, 4),
                              "contrib_dens": round(pn * p_usada / 100, 5), "fonte_principal": (l.get("fonte_principal") or "").strip()})
            dens_r.append(d); partes_fonte.append(pf); soma_r.append(soma); ncomp.append(len(ls))
            f = [(l.get("fonte_principal") or "").strip().lower() for l in ls if (l.get("fonte_principal") or "").strip()]
            if f:
                fontes.append(max(set(f), key=f.count))
                if fontes[-1] not in FONTES:
                    obs.append(f"fonte_principal fora da lista: {fontes[-1]}")
            if rd == sorted(rs)[0]:
                itens = [f"{(l.get('componente') or '').strip()}:{p:.0f}%" for l, p in zip(ls, partes)]
        if not dens_r:
            continue
        # média geométrica das rodadas (com piso para pratos sem proteína)
        dg = math.exp(sum(math.log(x + 0.01) for x in dens_r) / len(dens_r)) - 0.01
        pratos[pid] = dict(dens=max(dg, 0.0), fonte=max(set(fontes), key=fontes.count) if fontes else "",
                           parte_fontes=100 * sum(partes_fonte) / len(partes_fonte), n_comp=sum(ncomp) / len(ncomp), n_rod=len(dens_r),
                           soma=sum(soma_r) / len(soma_r), itens="; ".join(itens), obs="; ".join(dict.fromkeys(obs)),
                           dens_rodadas=dens_r)
    return pratos, comps, nao


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--massa", default="", help="CSV com prato_id e a massa total em gramas")
    ap.add_argument("--massa-col", default="massa_g")
    ap.add_argument("--componentes", default="")
    ap.add_argument("--sistema", default="nuvem_llm_v3")
    ap.add_argument("--peso-alt", default="0,0.25,0.40", help="peso da segunda hipótese para certeza alta,media,baixa")
    ap.add_argument("--limiar", type=float, default=0.5)
    ap.add_argument("--taco", default=TACO_PADRAO)
    ap.add_argument("--n5k", default=N5K_PADRAO)
    a = ap.parse_args()

    pesos = [float(x) for x in a.peso_alt.split(",")]
    assert len(pesos) == 3, "--peso-alt precisa de 3 valores"
    with open(a.entrada, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    falta = {"prato_id", "alimento_ref", "parte_pct"} - set(linhas[0] if linhas else [])
    if falta:
        sys.exit(f"colunas ausentes na entrada: {sorted(falta)}")
    pratos, comps, nao = densidades(linhas, Casador(a.taco, a.n5k, a.limiar), pesos, a.sistema)
    massa = le_massa(a.massa, a.massa_col) if a.massa else {}
    sem_massa = 0
    with open(a.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prato_id", "sistema", "proteina_estimada_g", "peso_total_estimado_g", "densidade_g_por_g", "fonte_principal",
                    "parte_fontes_pct", "n_componentes", "n_rodadas", "soma_partes", "itens", "observacoes"])
        for pid in sorted(pratos):
            d = pratos[pid]
            m = massa.get(pid)
            if a.massa and m is None:
                sem_massa += 1
            w.writerow([pid, a.sistema, "" if m is None else round(m * d["dens"], 1), "" if m is None else round(m),
                        round(d["dens"], 4), d["fonte"], round(d["parte_fontes"]), round(d["n_comp"], 1), d["n_rod"],
                        round(d["soma"]), d["itens"], d["obs"]])
    if a.componentes and comps:
        with open(a.componentes, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(comps[0]))
            w.writeheader()
            w.writerows(comps)
    with open(a.saida + ".nao_casados.txt", "w", encoding="utf-8") as f:
        for ref, n in sorted(nao.items(), key=lambda x: -x[1]):
            f.write(f"{n}\t{ref}\n")
    print(f"{len(pratos)} pratos, {len(comps)} componentes; {sum(nao.values())} sem casamento ({len(nao)} nomes distintos)"
          + (f"; {sem_massa} pratos sem massa" if a.massa else "; sem --massa: só densidade"), file=sys.stderr)
    for ref, n in sorted(nao.items(), key=lambda x: -x[1])[:15]:
        print(f"  sem casamento ({n}x): {ref}", file=sys.stderr)


if __name__ == "__main__":
    main()
