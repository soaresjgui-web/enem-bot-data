# Protótipo de precisão — proteína por foto em comida brasileira

**Pergunta a responder antes de construir o app:** a estimativa de proteína por foto *on-device* acerta comida brasileira bem o suficiente — e melhor que o Cal AI — para ser o núcleo da Nutrelle?

## 1. Amostra (~100 pratos reais)

| Categoria (`categoria` no CSV) | Pratos | Exemplos |
|---|---|---|
| `almoco_pf` | 30 | arroz + feijão + proteína + salada; marmita; estrogonofe; feijoada |
| `cafe` | 20 | pão francês com ovo, cuscuz com ovo, tapioca com queijo, iogurte com granola |
| `lanche` | 20 | açaí, sanduíche natural, pão de queijo, whey com fruta |
| `jantar` | 15 | sopa, omelete, macarrão com carne moída, peixe com legumes |
| `restaurante` | 15 | self-service por quilo, japonês, lanche de rede, prato de padaria |

Variar: luz (dia/noite), ângulo (de cima / 45°), prato (raso/fundo/marmita), porção (pequena/grande), comida misturada vs. separada.

## 2. Gabarito (verdade)

1. Pesar **cada componente** em balança de cozinha (resolução de 1 g) antes de montar o prato.
2. Proteína por 100 g: **TACO/TBCA** (preparação mais próxima, ex.: "Frango, peito, sem pele, grelhado") ou o rótulo do produto.
3. Registrar uma linha por componente em `gabarito.csv` (`prato_id,categoria,componente,gramas,proteina_100g`).
4. Fotografar **antes** de comer: 1 foto de cima + 1 a 45°, com o celular que será usado no teste.

## 3. Sistemas comparados

| `sistema` | O quê | Como coletar |
|---|---|---|
| `on_device` | Modelo da Apple no aparelho (Foundation Models + Vision) → itens e porções → proteína pela TACO | App de teste mínimo em SwiftUI (precisa de Mac com Xcode e iPhone compatível com Apple Intelligence) |
| `cal_ai` | Referência de mercado | Fotografar o mesmo prato no Cal AI e anotar a proteína |
| `concorrente_br` | Monju ou OzemPro (registro de comida) | Idem |
| `nuvem` | Fallback com modelo de visão em nuvem | Script com a mesma foto; mede custo por foto |

Registrar em `estimativas.csv` (`prato_id,sistema,proteina_estimada_g`). A pessoa que registra **não corrige** a estimativa — medimos a primeira resposta. Anotar à parte quantos toques foram precisos para corrigir.

## 4. Métricas e critério de decisão

Rodar: `python3 eval/score_precisao.py gabarito.csv estimativas.csv` (exemplo fictício em `eval/exemplo/`).

| Métrica | Meta para seguir com on-device |
|---|---|
| % de pratos dentro de ±20% **ou** ±5 g de proteína | **≥ 70%** |
| Erro médio absoluto | **≤ 8 g por prato** e menor que o do Cal AI |
| Viés | entre −3 g e +3 g (sem subestimar sistematicamente) |
| `almoco_pf` (categoria principal) | não pior que a média geral |

**Decisão:**
- On-device bate as metas → segue como núcleo; fotos ilimitadas em todos os planos.
- On-device fica abaixo, mas a nuvem bate → nuvem como padrão com limite diário e on-device como modo offline; recalcular a margem com o custo por foto medido.
- Nenhum bate → foto vira "atalho" para busca na TACO (sugere itens, usuária confirma porção) e o diferencial passa a ser a base brasileira + "quanto falta hoje".

## 5. Prazo estimado

1 semana de coleta (≈ 15 pratos/dia, contando refeições próprias, de amigas e de restaurante) + 2–3 dias para o app de teste on-device.
