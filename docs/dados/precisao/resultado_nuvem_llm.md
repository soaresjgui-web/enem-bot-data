# Resultado — teste cego, sistema `nuvem_llm` (08/11/2026)

**Sistema:** modelo de visão em nuvem (LLM multimodal), 4 instâncias independentes, cada uma vendo só as imagens de seu lote (sem acesso ao gabarito nem à web). Uma foto por prato, sem correção.
**Base:** 160 itens — 103 fotos do Manual Fotográfico UFPR 2017 (alimento único, porção pesada, proteína via TACO), 7 marmitas Seara 300 g (ilustração da bandeja), 50 pratos Nutrition5k (pesados).
**Reproduzir:** `python3 eval/score_precisao.py docs/dados/precisao/gabarito.csv docs/dados/precisao/estimativas_nuvem_llm.csv`

## Números

| Recorte | n | Dentro de ±20% ou ±5 g (meta ≥ 70%) | Erro médio | Erro mediano | Viés |
|---|---|---|---|---|---|
| **Todos** | 160 | **59%** | 7,8 g | 4,9 g | −0,1 g |
| Brasil (todos) | 110 | 55% | — | — | — |
| Brasil — alimento único (UFPR) | 103 | 59% | — | — | — |
| Brasil — marmita Seara | 7 | 0% | 12,0 g | 14,9 g | +10,4 g |
| Nutrition5k | 50 | 66% | 8,2 g | 2,0 g | −2,0 g |

Por categoria (script): `almoco_pf` 7,7 g de erro médio; `jantar` 5,7 g; `lanche` 8,5 g; `restaurante` 9,4 g; `cafe` 2,8 g.

## Diagnóstico

- **Identificar o alimento não é o problema:** erro mediano da *densidade* de proteína (g de proteína por g de comida) = **7%** nos itens brasileiros de alimento único.
- **O problema é a porção:** erro mediano do *peso* estimado = **19–22%**, com viés de +11% em porções pequenas e **subestimação forte em porções grandes** (bife de 310 g estimado em 180 g; feijoada de 734 g estimada em 350 g; camarão 366 g → 240 g). Os 8 maiores erros são todos porções grandes de carne/peixe.
- **Marmitas Seara:** o modelo superestima a proteína (+10 g) — as ilustrações de embalagem sugerem mais carne do que o produto tem. Amostra pequena (7) e de ilustração; não generalizar.

## Decisão (critério de `docs/prototipo-precisao.md` §4)

**A meta de 70% não foi atingida (59%) nem com um modelo de nuvem de ponta.** Um modelo on-device tende a ser igual ou pior. Pela regra do protocolo, a foto **não pode ser a única fonte do número**; o caminho é:

1. Foto **identifica** os alimentos (ponto forte, ~7% de erro de densidade).
2. A usuária **confirma a porção** com um seletor visual de tamanhos (P/M/G/GG com fotos de referência, no estilo do próprio manual fotográfico) ou com referência (mão/prato) — 1 toque.
3. A proteína é calculada pela TACO com a porção confirmada.

Próximos testes: (a) medir o ganho do seletor de porção (simular "porção escolhida entre 4 tamanhos de referência"); (b) adicionar ~30 refeições completas reais da Liv Up (requer liberar `livup.imgix.net`); (c) repetir com o modelo on-device da Apple em um iPhone.

---

# Atualização — refeições completas Liv Up + simulação do seletor de porção

**Base ampliada:** 194 itens (+34 refeições completas Liv Up, fotos reais da marmita aberta, 300–370 g, proteína 10,8–44,4 g).

## Refeições completas (Liv Up, n = 34)

| Métrica | Valor |
|---|---|
| Dentro de ±20% ou ±5 g | **53%** |
| Erro médio / mediano | 7,2 g / 5,4 g |
| Viés | **+5,3 g** (superestima) |
| Erro mediano do **peso** | **7%** (bom: o recipiente dá escala) |
| Erro mediano da **densidade de proteína** | **21%**, viés +16% |

Os maiores erros superestimam a proteína (frango assado com arroz integral: real 30 g, estimado 49 g; picadinho com arroz e feijão: 23,7 → 40 g). **Em prato composto, o problema se inverte:** o peso total sai certo, mas o modelo supõe mais carne (e menos arroz/legumes) do que o prato tem.

## Simulação do seletor de porção (`eval/simula_seletor.py`, 6 tamanhos, 30% de erro de escolha)

| Recorte | n | Foto sozinha | Seletor ideal | Seletor realista |
|---|---|---|---|---|
| Todos | 194 | 58% | 72% | 65% |
| Brasil — alimento único | 103 | 59% | 86% | **75%** |
| Brasil — marmitas (Liv Up + Seara) | 41 | 44% | 44% | 43% |
| Nutrition5k (pratos mistos) | 50 | 66% | 64% | 60% |

Granularidade (alimento único, erro de escolha 30%): 4 tamanhos 68% · 6 tamanhos 75% · 8 tamanhos 85%.

## Conclusão para o produto

- **Alimento único:** foto + seletor de porção fino (6–8 tamanhos ou slider) supera a meta de 70%.
- **Prato composto:** seletor do prato inteiro **não ajuda** (o peso já está certo). O ajuste precisa ser **por componente** — a foto separa "arroz / feijão / carne / legumes" e a usuária ajusta cada um (principalmente a proteína animal, que concentra o erro).
- **Desenho da tela de registro:** lista de itens detectados → cada item com seletor de porção visual → proteína recalculada na hora. A carne/peixe/ovo vem primeiro e em destaque.
- Próximo teste necessário: simular ou medir o ajuste **por componente** (exige gabarito por componente — o manual UFPR e o Nutrition5k têm; Liv Up não).
