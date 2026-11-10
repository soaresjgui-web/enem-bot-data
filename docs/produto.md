# Nutrelle — definição de produto (v1)

> Documento de decisão. Resume o que foi decidido a partir de `pesquisa-mercado.md` e `concorrencia-precificacao.md`. Onde este documento diverge dos relatórios, **vale este documento**.

## 1. Nome e posicionamento

- **Nome:** Nutrelle — "nutrir" + "elle" (ela). Pronúncia igual em PT/ES/EN: *nu-TRÉL*. Livre na App Store BR/US/MX em 08/11/2026 (busca por nome). **Pendente:** busca de marca no INPI e USPTO, domínio e @ no Instagram.
- **Subtítulo App Store:** "Proteína por foto" (PT) · "Protein from a photo" (EN) · "Proteína con una foto" (ES).
- **Promessa:** *Fotografe o prato e veja a proteína. O app de proteína da mulher que está mudando o corpo.*
- **Público:** mulheres de 30–55 anos em emagrecimento/recomposição corporal. **Canal de entrada:** quem usa caneta GLP-1 (Mounjaro, Ozempic, Ozivy) — o tema feminino que mais cresce no Google Trends BR.
- **Voz:** acolhedora, sem culpa, sem "dieta". Celebra conquistas ("Meta de proteína batida. Amei!").

## 2. Por que dá para ganhar (lacunas validadas)

| Lacuna | Evidência (ver `concorrencia-precificacao.md` §5) |
|---|---|
| **Registro de comida brasileira fraco, sem foco em proteína** | "falta comidas básicas" (MeuGLP), "registro nunca es exacto" (OzemPro MX), precisão = 41% das queixas do Cal AI, MacroFactor/Cronometer sem PT-BR |
| **Nenhum app de caneta é feminino** | Zero companions GLP-1 com foco em mulheres; rising queries femininas ("mounjaro amamentando", "anticoncepcional") |
| **Apple Health mal integrado** | Queixas em Pep, Balance, Mingo |
| **Estabilidade** | Bugs recorrentes em MeuGLP, Tecnonutri, Dietbox, Lifesum |

Privacidade e "cobrança honesta" **não** são diferenciais de posicionamento (decisão da fundadora; os líderes coletam dados e têm nota 4,8–4,9). Boas práticas de privacidade (LGPD, dado de saúde sensível) seguem obrigatórias, mas não são a mensagem principal.

**Onde não brigar:** rastreador de aplicação puro (Monju, OzemPro), base gigante de alimentos (MyFitnessPal), mídia paga em massa.

## 3. Funcionalidades da v1

1. **Proteína por foto, afinada para comida brasileira** (núcleo do produto) — PF, marmita, cuscuz, tapioca, açaí, feijoada; base TACO/TBCA embarcada; o app já abre com a proteína sugerida (modelo híbrido: identificação do alimento + modelo treinado de peso), com faixa provável e confiança; **sempre uma única foto, tirada de cima**, com círculo-guia e aviso de nível para padronizar a escala; ajuste de porção opcional em 1 toque; favoritos ("meu café de sempre"); código de barras.
2. **"Quanto falta hoje"** — sugestões com comida de verdade para fechar a meta ("faltam 30 g: 2 ovos + 1 iogurte natural").
3. **Modo caneta ligado à proteína** — dia da aplicação, apetite, náusea; em dias de pouco apetite, sugere opções pequenas e densas em proteína. Sem recomendação de dose.
4. **Check-in feminino** — sintomas do dia; respostas com fontes às dúvidas em alta (amamentação, anticoncepcional). Base para o modo 40+.
5. **Evolução + Apple Health** — peso, proteína e água lidos/escritos no Health; gráficos; widget "proteína de hoje".

**Plano do dia:** tela inicial com meta de proteína, quanto falta e check-in rápido.

**Papel da IA on-device:** objetivo de custo marginal zero por foto → **fotos ilimitadas** em todos os planos; resposta < 1 s; funciona offline. **Estado em 09/11/2026:** **Estado em 10/11/2026:** um modelo só de imagem (SigLIP 2 B/16, convertido para Core ML: 231 MB em FP16, 116 MB em 8 bits) chega a 61% sem nenhum LLM, empatado com o LLM em nuvem (55%) dentro da margem de erro. O híbrido com LLM em nuvem chega a ~63%. Latência no iPhone e identificação com Foundation Models **ainda não foram medidas**.

**Fora da v1:** treino guiado por câmera (removido), contador de calorias como foco, nota de "forma perfeita", comunidade, chat com IA.

**v1.5:** modo 40+ (perimenopausa: fogachos, sono, humor × proteína); rotina simples de força sem câmera (avaliar); Apple Watch. **v2:** Android nativo; parcerias com nutricionistas/telemedicina.

## 4. Preço

Assinatura com **trial de 7 dias** e **hard paywall após o onboarding**; sem plano semanal nem vitalício.

| Mercado | Mensal | Anual | Exibição do anual |
|---|---|---|---|
| Brasil | R$ 39,90 | R$ 149,90 | "R$ 12,49/mês" |
| EUA | US$ 9,99 | US$ 49,99 | "US$ 4,17/mês" |
| Portugal | € 7,99 | € 39,99 | "€ 3,33/mês" |
| México | MX$ 149 | MX$ 699 | "MX$ 58/mês" |
| Chile (loja em US$) | US$ 7,99 | US$ 34,99 | — |
| Colômbia | COP 29.900 | COP 129.900 | — |
| Peru | S/ 24,90 | S/ 99,90 | — |
| Argentina (loja em US$) | US$ 5,99 | US$ 24,99 | — |

Brasil: mensal ajustado de R$ 29,90 para **R$ 39,90** para reforçar a âncora do anual, que segue no meio do cluster (R$ 99,90–199,90) e abaixo do mensal do Monju (R$ 59,90). Preços fixos por país (equalização automática da Apple desligada).

## 5. Lançamento

1. **Brasil** (iOS, PT-BR) → 2. **Portugal + EUA** (mesmo binário, inglês) → 3. **México** (es-MX completo, inclusive base de alimentos) → 4. Chile e Colômbia → 5. Peru → 6. Argentina.

## 6. Stack

iOS nativo: Swift 6, SwiftUI, SwiftData, Foundation Models + Vision (on-device), HealthKit, StoreKit 2 + RevenueCat, WidgetKit. Requer Mac com Xcode e conta Apple Developer.

## 7. Precisão medida, maior risco e próximo passo

Critério: proteína dentro de ±20% ou ±5 g; meta ≥ 70%. Conjunto de teste fora do treino (280 pratos); detalhes em [`dados/precisao/resultado_v2.md`](dados/precisao/resultado_v2.md) e [`dados/precisao/resultado_modelo_treinado.md`](dados/precisao/resultado_modelo_treinado.md).

| Abordagem | Acerto |
|---|---|
| LLM em nuvem sozinho | 55% |
| Modelo treinado só com a imagem, 1ª versão | 48–52% |
| Modelo só com a imagem, um modelo exportável para o aparelho | 61% (IC 55–67) |
| Modelo só com a imagem, três modelos + busca de pratos parecidos (servidor, sem LLM) | 64,6% (IC 59–70) |
| Híbrido com o roteiro antigo do LLM (v2) | 67,1% (IC 61–73) |
| **Híbrido com o roteiro novo do LLM (v3: identifica e dá a parte da massa de cada alimento), média de duas rodadas cegas** | **71,4%** (IC 66–76); rodada 1 sozinha 70,4%, rodada 2 sozinha 68,6% |
| Híbrido na comida brasileira (144 itens, fotos em ângulo, versão anterior) | 46–49% |
| Com ajuste de porção pela usuária (**simulado**, 15–30% de erro de escolha) | 71–76% |

- **O híbrido está em torno de 70%, mas a meta não está demonstrada:** as duas rodadas cegas independentes deram 70,4% e 68,6% (média 71,4%), e o limite inferior do intervalo fica abaixo de 70% nas três leituras. O ganho sobre o roteiro antigo é real (+4,3; IC +1,4 a +7,5). As duas rodadas erram nos mesmos pratos, então repetir rodadas não melhora. É um refeitório americano só (280 pratos); no conjunto secundário (50 pratos) o acerto foi 62%.
- **Depende de LLM em nuvem** para identificar. Sem nuvem: 64,6% em servidor, 61% no modelo exportável para o aparelho.
- **Pratos com muita proteína seguem fracos:** 73% na faixa de 15–25 g, 45% em 25–40 g e 55% acima de 40 g. Acima de 25 g, o limite voltou a ser o peso, que o modelo subestima em 8–12%.
- **Sugestão automática com confiança:** só com a imagem, cerca de metade dos pratos pode ser aceita com ~69% de acerto; os aceitos são, na maioria, pratos de pouca proteína.
- **Escala:** sem padronizar a distância, um zoom de ±30% derruba o híbrido de 60% para 46–54%; treinar com escala aleatória ou recortar pelo prato elimina a perda. Daí o círculo-guia na câmera.
- **Mais dados do mesmo tipo não resolvem:** a curva de aprendizado estabiliza a partir de ~400 pratos da Nutrition5k.
- **Fotos laterais no treino** removem o viés de ângulo (comida brasileira só com imagem: 22% → 36%) e ficam no conjunto de treino.
- **Não existe base pública latino-americana com peso medido** ([`dados/precisao/datasets_internacionais.md`](dados/precisao/datasets_internacionais.md)); os dados brasileiros terão de ser coletados.

**Decisão para a v1:** **uma única foto de cima** (decisão da fundadora, 09/11/2026), com círculo-guia e aviso de nível; sugestão automática pelo híbrido (modelo treinado para o peso + LLM em nuvem com o roteiro v3 para a identificação); em confiança baixa, faixa mais larga e destaque para conferir as porções; ajuste de porção opcional. Sem segunda foto. Sem ajuste, o híbrido fica em torno de 70% (71,4% na média de duas rodadas; IC 66–76), sem a meta demonstrada; os 71–76% com ajuste da versão anterior são simulação, não teste com usuárias. O ajuste de porção continua necessário, principalmente acima de 25 g de proteína.

**Maiores riscos:** (1) nenhuma foto brasileira tirada de cima no conjunto de teste (UFPR e Liv Up são em ângulo), então o desempenho no Brasil sob a nova premissa é desconhecido; (2) peso e densidade em pratos com mais de 25 g de proteína; custo e latência do LLM em nuvem por foto, contra a promessa de fotos ilimitadas; (3) identificação on-device (Foundation Models) não testada; (4) o modelo exportado tem 116–231 MB e a latência no iPhone não foi medida.

**Próximos passos de modelo:** treino em GPU (pacote em `eval/treino/kaggle/`, primeira execução no Kaggle iniciada em 10/11/2026) para corrigir o peso dos pratos grandes. **Próximo passo de dados:** piloto de 50 pratos brasileiros no iPhone (uma foto de cima com o círculo-guia, profundidade do aparelho como sinal de treino, peso por componente em balança) e teste da identificação on-device; depois, coleta de 400–800 pratos brasileiros pesados. Protocolo original em [`prototipo-precisao.md`](prototipo-precisao.md).
