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

**Plano do dia:** tela inicial com o anel da meta e **uma única ação** (quanto falta e o que registrar agora). No dia da aplicação da caneta e nos dois seguintes, um cartão fixo no topo.

**Jornada (mock em `docs/mock/jornada.html`, 21 telas):** o onboarding termina com a **primeira foto e o primeiro resultado antes do paywall**; na sugestão de proteína, a faixa provável tem o mesmo destaque do número, o selo de confiança segue o acerto medido por faixa e, em pratos com muita proteína, o seletor de tamanho do item de proteína já vem aberto; a Evolução destaca a constância na meta, com o peso opcional; o check-in devolve na hora uma ligação entre sintoma e proteína.

**Papel da IA on-device:** objetivo de custo marginal zero por foto → **fotos ilimitadas** em todos os planos; resposta < 1 s; funciona offline. **Estado em 09/11/2026:** **Estado em 11/11/2026:** um modelo único exportável para o aparelho (SigLIP 2 B/16; a versão anterior foi convertida para Core ML com 231 MB em FP16 e 116 MB em 8 bits) chega a 61–64% sem nenhum LLM. Em servidor, nove redes de imagem chegam a cerca de 70% sem LLM e a cerca de 71% com uma chamada de LLM em nuvem. A diferença entre aparelho e servidor é de 5 a 9 pontos, e a promessa de custo zero por foto só vale para a versão do aparelho. Latência no iPhone e identificação com Foundation Models **ainda não foram medidas**.

**Fora da v1:** treino guiado por câmera (removido), contador de calorias como foco, nota de "forma perfeita", comunidade, chat com IA.

**v1.5:** modo 40+ (perimenopausa: fogachos, sono, humor × proteína); rotina simples de força sem câmera (avaliar); Apple Watch. **v2:** Android nativo; parcerias com nutricionistas/telemedicina.

## 4. Preço

Assinatura com **trial de 7 dias** e **hard paywall após o onboarding** (que inclui a primeira foto e o primeiro resultado); sem plano semanal nem vitalício.

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

Critério: proteína dentro de ±20% ou ±5 g. **70% é a meta mínima, não o teto** (decisão da fundadora, 11/11/2026: é a nutrição das pessoas): o produto precisa passar de 70% com folga, isto é, com o limite inferior do intervalo em 70% ou mais. Conjunto de teste fora do treino (280 pratos); detalhes em [`dados/precisao/resultado_v2.md`](dados/precisao/resultado_v2.md) e [`dados/precisao/resultado_modelo_treinado.md`](dados/precisao/resultado_modelo_treinado.md).

| Abordagem | Acerto |
|---|---|
| LLM em nuvem sozinho | 55% |
| Modelo treinado só com a imagem, 1ª versão | 48–52% |
| Modelo único (93 M de parâmetros, 186 MB, 8 ms por foto numa GPU T4), treinado com 2.281 pratos | **66,0%** no teste novo (IC 62–70); versão anterior 64,5% |
| **Modelo único de servidor** (um backbone So400m congelado com duas cabeças; 430 M de parâmetros, 859 MB, 61 ms por foto numa GPU T4) | **69,7%** no teste novo (IC 65,0–74,3): igual às nove redes |
| Modelo só com a imagem, nove redes em servidor, sem LLM | **~70%** (69,6% no teste; 69,7% no teste novo, IC 65–74) |
| **Híbrido: redes de imagem em servidor + LLM em nuvem com o roteiro v3** | **~71%** (72,9% no teste; **70,8% no teste novo, IC 66–75**; os dois juntos 71,5%, IC 68–75) |
| Híbrido na comida brasileira (144 itens, fotos em ângulo, versão anterior) | 46–49% |
| Com ajuste de porção pela usuária (**simulado**, 15–30% de erro de escolha) | 71–76% |

- **Teste novo, lido uma única vez com o modelo congelado (829 pratos, 11/11/2026):** híbrido 70,8% (IC 66,3–75,2) e só imagem 69,7% (IC 65,2–74,2), com as cinco faixas de proteína pesando igual. Pela regra registrada antes da leitura, **a meta de 70% não está demonstrada nem refutada**: o nível se repete em torno de 70%, mas o limite inferior fica em 66%.
- **O ganho do LLM não se repetiu:** no teste novo o híbrido fica só 1,1 ponto acima do modelo de imagem (IC −2,1 a +4,3), e a densidade do LLM em pratos com mais de 25 g foi pior que a do modelo treinado. O modelo só de imagem é o número mais estável do estudo (69,6%, 69,3% e 69,7% em três leituras).
- **Pratos com muita proteína seguem sendo o problema:** no teste novo, 65% na faixa de 15–25 g e 52% acima de 25 g, nos dois pipelines.
- **Custo de servir (medido no Kaggle):** as nove redes somam 1,8 bilhão de parâmetros e 3,6 GB em memória; uma foto leva 0,24 s numa GPU T4 com uma orientação e 1,7–2,0 s com as oito orientações usadas na avaliação; em CPU, de 13 s a quase 2 min. O modelo único para o aparelho tem 186 MB e leva 8 ms na T4 e 0,2 s em CPU com uma orientação. Não medidos: iPhone, a chamada do LLM, custo em dinheiro.
- **Terceira rodada de treino (12/11/2026): reduzir a um modelo custa cerca de 4 pontos.** O modelo único escolhido pela regra registrada fez 66,0% no teste novo (IC 61,6–70,1), contra 69,7% das nove redes: **a meta mínima de 70% não foi demonstrada**, e o intervalo fica quase todo abaixo dela. Acima de 25 g de proteína ele acerta 48%.
- **Mais dados da Nutrition5k ajudaram** (+3 pontos na validação cruzada com 2.281 pratos em vez de 1.684), mas a Nutrition5k agora está esgotada.
- **Treinar o modelo pequeno para imitar o conjunto piorou** (61–64% na validação cruzada, contra 68% sem imitação), por um problema de receita de treino a corrigir na próxima rodada.
- **A base Flavoria (1.264 pratos pesados, CC BY 4.0) não foi adotada pela regra** (−1 ponto na validação cruzada da Nutrition5k), mas fora da Nutrition5k ela reduz o erro de peso (21,8% → 18,7%; em pratos acima de 600 g, 29,7% → 18,4%). Sem ela, o modelo subestima pratos grandes de outro refeitório em cerca de 29%.
- **Leitura extra aprovada pela fundadora (12/11/2026): um único modelo grande congelado (So400m) com duas cabeças faz 69,7% no teste novo (IC 65,0–74,3)**, igual às nove redes, com um backbone só: 30 vezes mais rápido e 4 vezes menor que elas. **A meta mínima continua não demonstrada.** Ele é um modelo de servidor (859 MB); o de aparelho fica em 66,0%. Os 3,7 pontos entre os dois são hoje o preço de rodar no celular.
- **O teste novo está perto de se esgotar:** já foi lido três vezes (69,7%, 66,0% e 69,7%), e nenhuma leitura tem limite inferior acima de 66%. Demonstrar a meta exige um conjunto de teste novo e maior, além de mais dados de treino.
- **Só Nutrition5k:** um refeitório americano, câmera a altura fixa; o teste novo não é independente do antigo (são pratos "irmãos"). Nenhum prato brasileiro foi avaliado com esses modelos.
- **Sugestão automática com confiança:** só com a imagem, cerca de metade dos pratos pode ser aceita com ~69% de acerto; os aceitos são, na maioria, pratos de pouca proteína.
- **Escala:** sem padronizar a distância, um zoom de ±30% derruba o híbrido de 60% para 46–54%; treinar com escala aleatória ou recortar pelo prato elimina a perda. Daí o círculo-guia na câmera.
- **Mais dados do mesmo tipo não resolvem:** a curva de aprendizado estabiliza a partir de ~400 pratos da Nutrition5k.
- **Fotos laterais no treino** removem o viés de ângulo (comida brasileira só com imagem: 22% → 36%) e ficam no conjunto de treino.
- **Não existe base pública latino-americana com peso medido** ([`dados/precisao/datasets_internacionais.md`](dados/precisao/datasets_internacionais.md)); os dados brasileiros terão de ser coletados.

**Decisão para a v1 (atualizada em 12/11/2026):** **uma única foto de cima**, com círculo-guia e aviso de nível; a proteína sai **de um modelo de imagem só**, sem LLM em nuvem. Há dois candidatos medidos: o **modelo de servidor** (So400m com duas cabeças, 69,7%) e o **modelo de aparelho** (SigLIP 2 B/16 com ajuste fino, 66,0%). Recomendação: servidor como caminho principal e aparelho como modo offline, porque o acerto é o maior risco do produto. Em confiança baixa, faixa mais larga e destaque para conferir as porções; em pratos com muita proteína, o seletor de tamanho do item de proteína já vem aberto. **Nenhum dos dois atinge a meta mínima de 70% com folga.** Frentes abertas: mais dados de treino com licença comercial (pedidos em `dados/precisao/pedidos_licenca.md`), coleta brasileira (`coleta-brasil.md`), pratos com mais de 25 g de proteína e um conjunto de teste independente maior.

**Dados de treino (decisão da fundadora, 12/11/2026):** o treino não se restringe a bases com licença comercial; o objetivo é enriquecer o modelo, qualquer que seja a base. Fica registrado o risco: usar bases de licença não comercial no modelo de um app pago pode violar a licença delas. Para manter a saída aberta, **toda base externa carrega a licença e o uso declarado em cada lista de treino, e cada modelo registra quais bases usou**, de modo que uma versão treinada só com dados liberados possa ser refeita. Os pedidos de licença (`dados/precisao/pedidos_licenca.md`) continuam valendo. Bases com acesso fechado por formulário ou pedido não são contornadas: o pedido é da fundadora.

**Autonomia no treino (decisão da fundadora, 12/11/2026):** as decisões sobre o treinamento do modelo (quais bases entram, receitas, rodadas de GPU, leituras de teste dentro das regras registradas) são tomadas sem consulta prévia e relatadas depois, com o que foi decidido e por quê. Continuam dependendo da fundadora: pedidos de licença e de acesso a bases fechadas, decisões de produto e qualquer gasto. Primeira decisão tomada assim: a SNAPMe **não entra no treino** (os autores pedem isso) e serve só como medida fraca em fotos de celular.

**Maiores riscos:** (1) nenhuma foto brasileira tirada de cima no conjunto de teste (UFPR e Liv Up são em ângulo), então o desempenho no Brasil sob a nova premissa é desconhecido; (2) peso e densidade em pratos com mais de 25 g de proteína; custo e latência do LLM em nuvem por foto, contra a promessa de fotos ilimitadas; (3) identificação on-device (Foundation Models) não testada; (4) o modelo exportado tem 116–231 MB e a latência no iPhone não foi medida.

**Próximos passos de modelo:** decidir se o LLM em nuvem entra na v1 (o teste novo não mostrou ganho seguro sobre o modelo de imagem); reduzir o modelo de servidor (nove redes) para algo que caiba em custo; atacar os pratos com mais de 25 g de proteína para corrigir o peso dos pratos grandes. **Próximo passo de dados:** piloto de 50 pratos brasileiros no iPhone (uma foto de cima com o círculo-guia, profundidade do aparelho como sinal de treino, peso por componente em balança) e teste da identificação on-device; depois, coleta de 400–800 pratos brasileiros pesados. Protocolo original em [`prototipo-precisao.md`](prototipo-precisao.md).
