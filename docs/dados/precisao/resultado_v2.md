# Resultado v2 — base ampliada (594 fotos) e estimativa por componente (08/11/2026)

**Base:** 594 fotos reais com proteína medida — 400 Nutrition5k novos (120 `calib` / 280 `teste`, fora de qualquer ajuste), 50 Nutrition5k v1, 103 UFPR, 34 Liv Up, 7 Seara. Fontes rejeitadas e motivos em `coleta.md` (MM-Food-100K: rótulos estimados por GPT-4o; SNAPMe: porções autorrelatadas; Food Portion Benchmark: sem proteína, licença NC).
**Sistema:** modelo de visão em nuvem, 15 instâncias cegas (só imagens + `INSTRUCOES.md`), estimando **gramas por componente**; proteína calculada por `eval/protein_from_components.py` (TACO 4ª ed. + ingredientes Nutrition5k). 1.450 componentes; 22 sem casamento com a tabela (14 deles espinafre), contados como 0 g de proteína.

**Reproduzir:**
```
python3 eval/protein_from_components.py docs/dados/precisao/estimativas_v2_componentes.csv docs/dados/precisao/estimativas_v2_pratos.csv --taco docs/dados/precisao/ref/taco.csv --n5k docs/dados/precisao/ref/n5k_ingredients.csv --componentes docs/dados/precisao/estimativas_v2_comp_casados.csv --sistema nuvem_llm_v2
python3 eval/score_precisao.py docs/dados/precisao/gabarito_v2.csv docs/dados/precisao/estimativas_v2_pratos.csv
python3 eval/calibra.py docs/dados/precisao/estimativas_v2_comp_casados.csv
python3 eval/simula_seletor_componentes.py docs/dados/precisao/estimativas_v2_comp_casados.csv
```

## 1. O método v2 (por componente + tabela) ficou PIOR que o v1 na foto sozinha

Mesmas 194 fotos, critério ±20% ou ±5 g:

| Fonte | v1 (estimativa livre) | v2 (componentes + TACO) | Peso: viés v1 → v2 | Densidade de proteína: viés v1 → v2 |
|---|---|---|---|---|
| UFPR (alimento único) | **59%** | 47% | +11% → **+23%** | 0% → 0% (erro 7% → 11%) |
| Liv Up (marmita) | **53%** | 24% | +7% → +8% | +16% → **+28%** |
| Seara | 0% | 14% | +13% → +8% | +62% → +103% |
| Nutrition5k v1 | **66%** | 52% | +7% → −5% | −3% → 0% |

Por quê:
- **A dica "não subestime carne grande" virou superestimação geral** em alimento único (viés de peso dobrou para +23%). Instrução textual corrige um viés criando outro.
- **Tabela por componente erra marmita:** a TACO tem carne *grelhada* (~30 g/100 g); a carne de marmita vem em molho, desfiada ou empanada, com menos proteína por grama. A densidade piorou de +16% para +28%.
- Há também variação natural entre execuções do modelo (mesma foto, outra estimativa).

## 2. Calibração estatística: ganho pequeno

Ajustada só no `calib` (120 pratos), medida no `teste` (280, nunca visto): melhor modelo `mult_grupo` (multiplicador por grupo de alimento). Dentro de ±20%: 31% → 37%; erro médio 8,5 → 8,3 g. **Não muda o quadro** e não transfere para comida brasileira (Liv Up/UFPR pioram levemente).

## 3. O que funciona: seletor de porção no componente proteico

`simula_seletor_componentes.py` — a usuária ajusta o item que mais pesa na proteína (6 tamanhos ao redor da estimativa), erra para o tamanho vizinho em 30% (ou 15%) das vezes:

| Recorte | n | Foto sozinha | Seletor no prato inteiro (ideal) | **Seletor no componente (30% erro)** | Seletor no componente (15% erro) |
|---|---|---|---|---|---|
| **Nutrition5k teste (held-out)** | 280 | 55% | 70% | **71%** | **76%** |
| Nutrition5k calib | 120 | 54% | 72% | 71% | 74% |
| Nutrition5k v1 | 50 | 52% | 60% | 66% | 69% |
| UFPR (alimento único) | 103 | 47% | 74% | 64% | 67% |
| **Todos com gabarito por componente** | 553 | 53% | 70% | **69%** | **73%** |

Por grupo ajustado: proteína animal (339 pratos) 43% → 62–67%; mistos 53% → 79–83%.

## Conclusões para o produto

1. **A foto sozinha estabiliza em ~50–60%** dentro da margem, com dois métodos e quase 600 fotos. Não há truque de instrução que leve a 70%. **A foto não pode ser o número final.**
2. **Um toque no seletor do item proteico leva a ~70–76%** no conjunto de teste nunca visto — atinge a meta. Esse é o fluxo da v1 do app.
3. **Tabela de densidade precisa de preparações reais**, não só cortes grelhados: "frango desfiado ao molho", "carne de panela", "estrogonofe", "carne de marmita". Antes do lançamento, montar uma tabela de pratos prontos brasileiros (TBCA tem preparações; exige termo de uso) ou deixar a usuária escolher a preparação.
4. Manter a estimativa **livre do v1 como valor inicial** do seletor (foi mais precisa) e usar a tabela só para converter o tamanho escolhido em proteína.
5. Ainda falta: teste com usuárias reais (o erro humano de 15–30% é suposição) e com o modelo on-device da Apple.
