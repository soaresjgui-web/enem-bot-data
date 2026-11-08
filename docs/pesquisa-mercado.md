# Pesquisa de mercado: app pago de rotina diária para mulheres com IA on-device (edge AI)

**Data da pesquisa:** 08/11/2026
**Para quem:** desenvolvedor(a) solo no Brasil decidindo qual app pago construir
**Ideias avaliadas:** (A) dieta por foto, (B) coach de execução na academia, (C) força + proteína para mulheres em uso de GLP-1, (D) treino/nutrição sincronizados ao ciclo, (E) perimenopausa/menopausa, (F) gestação/pós-parto + assoalho pélvico, (G) análise de pele por câmera, (H) avaliação de postura/glúteo

---

## 0. Como ler este documento e limitações

- Preferência por dados de 2025–2026. Dados mais antigos ou de fontes fracas (blogs de concorrentes, relatórios de "market research" genéricos) estão **sinalizados**.
- **Google Trends não pôde ser consultado diretamente** nesta sessão (acesso de rede bloqueado para `trends.google.com`, `revenuecat.com`, `healio.com` etc.). Os sinais de busca abaixo vêm de reportagens e relatórios que citam o Trends; recomendo você mesmo abrir o Google Trends (filtro Brasil, 12 meses e 5 anos) para os termos listados na seção 2.4 antes de decidir.
- Onde não encontrei número confiável, digo "não encontrado" em vez de inventar.
- Preços em R$ de concorrentes foram obtidos das fichas da App Store brasileira ou de sites oficiais; podem variar por testes A/B.

---

## 1. Resumo executivo

**Recomendação ranqueada (detalhe na seção 6):**

| # | Ideia | Nota geral (1–5) | Veredito em uma linha |
|---|-------|------------------|-----------------------|
| 1 | **(C) Força + proteína para mulheres em uso de caneta GLP-1** | **4,2** | Onda real e recente no Brasil (R$ 13,3 bi em canetas, genérico desde jun/2026), dor concreta (perda de massa magra), público já paga caro pelo remédio, quase nenhum concorrente em PT-BR, baixa dificuldade técnica |
| 2 | **(E) Rotina para perimenopausa/menopausa (35–60 anos)** | **4,0** | Mercado bilionário em formação, apps brasileiros ainda incipientes (Raquel Menopausa grátis; Namu B2B), público com renda; exige cuidado com alegações médicas. **Combina naturalmente com a #1** (mesma faixa etária, mesmas necessidades: força, proteína, sono) |
| 3 | (D) Treino/nutrição sincronizados ao ciclo | 3,4 | Demanda em alta no TikTok (buscas +62%/+124%), privacidade de dados menstruais é argumento forte para edge AI, mas base científica fraca e Flo já oferece |
| 4 | (A) Dieta por foto (calorias/macros) | 3,0 | Maior demanda de todas, mas commoditizado (Cal AI comprado pelo MyFitnessPal; dezenas de clones em PT-BR a R$ 49,90/ano); precisão ruim (~-33%). **Faz sentido como *recurso* dentro da #1/#2, não como produto** |
| 5 | (F) Gestação/pós-parto + assoalho pélvico | 3,0 | Demanda estável e previsível, mas janela de uso curta (LTV baixo), sensibilidade regulatória alta |
| 6 | (B) Coach de execução com pose estimation | 2,6 | Tecnicamente sedutor, mas histórico ruim: Kemtai migrou para fisioterapia B2B, Zenia foi absorvida, Zing Vision é criticado por falsos alertas; erro angular ~10° no melhor caso; monetização fraca |
| 7 | (G) Análise de pele por câmera | 2,4 | Dominado por L'Oréal/Perfect Corp (US$ 69 M de receita) e marcas; difícil cobrar assinatura por algo que marcas dão de graça |
| 8 | (H) Postura/glúteo por foto | 2,0 | Micro-nicho recém-inundado de apps (abr/2026), baixa retenção |

**Sugestão de produto final:** um único app — *rotina diária de força + proteína para mulheres 35+* — lançado primeiro para o nicho GLP-1 (aquisição mais fácil, dor mais aguda) e expandido para perimenopausa. Edge AI usada em: (1) estimativa de proteína por foto no aparelho, (2) contagem de repetições/tempo sob tensão com a câmera, (3) todos os dados sensíveis (medicação, sintomas, ciclo) **nunca saem do celular** — argumento de marketing forte depois do veredito Flo × Meta (ago/2025).

---

## 2. Panorama de mercado e sinais de demanda

### 2.1 Tamanho e dinâmica da categoria Saúde & Fitness

| Métrica | Valor | Fonte / observação |
|---|---|---|
| Receita IAP global Saúde & Fitness 2025 | **US$ 4,5 bi (+13% a/a)**, recorde | Sensor Tower, fev/2026 |
| Downloads globais 2025 | 3,96 bi (+0,8%) | Sensor Tower — ou seja, crescimento vem de monetização, não de volume |
| Categorias que mais cresceram em receita | Corrida/ciclismo (Strava, Runna) e **nutrição com IA (Cal AI, MacroFactor, YAZIO)** | Sensor Tower |
| Mercado de apps fitness (estimativa ampla) | US$ 12,1 bi em 2025 → US$ 33,6 bi em 2033 | Grand View Research — relatório genérico, usar com cautela |
| MyFitnessPal | ~900 mil downloads e **US$ 13 M/mês** | Sensor Tower via Fitt Insider, mar/2026 |

### 2.2 Brasil

- Android domina: **~75–79% Android vs. 21–25% iOS** (StatCounter, jun–ago/2026). Implicação: iOS é minoria, mas concentra renda e conversão; ver seção 8.
- Ranking de receita (Google Play BR, Similarweb, snapshots mai e ago/2026): **Strava** em 1º; Befit, Foodvisor e Yazio entre o 3º e 5º. Flo e Yazio aparecem nos tops da Appfigures BR. Não encontrei ranking de receita iOS BR confiável e atual.
- **Queima Diária** (Smart Fit): 438 mil clientes no 4T25 (+19% a/a), R$ 500 M faturados desde a fundação, público majoritariamente feminino; plano anual a R$ 39,90/mês. Prova de que mulher brasileira paga assinatura fitness recorrente.
- Mercado fitness BR: ABA diz que **~55% das mensalidades de academia são de mulheres**; IBOPE Repucom: **mulheres são 60% dos novos praticantes**; musculação é 16% da oferta mas 39% das matrículas (ABC EVO via Pacto, 2026). Dados de gênero divergem entre fontes (38% a 49,7% de mulheres praticantes).
- Think with Google BR: buscas por corrida +122%, academias +56%, **whey/creatina +205%** (comparação de dois anos; data exata não indicada).
- Estudo citado pelo Futuro da Saúde: **75% dos usuários de apps fitness no Brasil são mulheres**, faixa mais engajada 25–34 anos — **dado antigo (~2023), metodologia não clara**.
- 47% dos brasileiros pesquisam saúde/bem-estar no Google (Agência Qcom, data não verificada).

### 2.3 Mulheres: uso de apps e disposição a pagar

- Flo: ~**80 M MAU** (ago/2026), ~5 M assinantes pagos (jun/2024), receita estimada entre US$ 157 M e US$ 275 M em 2025 (estimativas de terceiros divergem; a empresa não publica), valuation > US$ 1 bi (General Atlantic, jul/2024). É a prova máxima de que mulheres pagam por apps de saúde.
- Sweat (Kayla Itsines): ~US$ 65 M/ano, 1 M+ usuárias ativas/mês, mas receita caiu de US$ 100 M (2021) → US$ 71 M → US$ 65 M após corte de marketing. Lição: **categoria de treino genérico para mulheres depende de mídia paga pesada**.
- Calorie trackers: ~58% dos usuários são mulheres (2025), caindo de 62% em 2020 porque foto-IA atraiu homens (Statista via blog do Nutrola — **não verificado na fonte primária**). Cal AI: base principal **homens de 15–25 anos** (Forbes).
- Estudo PLOS ONE 2025 (n=259 mulheres, China): intenção de continuar usando prediz comportamento de pagamento; valor "de saúde" e utilitário pesam mais que hedônico (Healthcare 2024, n=395). Transferível com cautela.
- McKinsey Future of Wellness 2026 (prévia): 84% dos consumidores dizem que bem-estar é prioridade máxima; Gen Z e millennials lideram. Sem recorte de gênero disponível.
- **Não encontrei** pesquisa 2025–2026 com disposição a pagar separada por gênero. Lacuna real.

### 2.4 Tendências de busca e redes sociais (proxies para Google Trends)

| Termo/tendência | Sinal | Fonte | Confiabilidade |
|---|---|---|---|
| Pilates (Brasil) | **Maior patamar anual da série histórica do Google em 2025**; destaque Sudeste e Sul | Band/Sala Digital, 09/12/2025 | Boa |
| Pilates (global) | Modalidade mais reservada no ClassPass pelo 3º ano, **+66% a/a**; força +27%; low-impact +112% | ClassPass Look Back 2025 | Boa (jan–out/2025) |
| Pilates (ACSM 2026) | "Balance, Flow and Core Strength" em 5º nas tendências 2026 | ACSM via ABPilates | Boa |
| Treino de força feminino | Strava: treino de força **+21% entre mulheres** em 2025 | Strava Year in Sport 2025 (via blogs) | Média |
| "Cycle syncing" | Buscas **+124% a/a em 2024**; "cycle syncing workout" **+62% a/a em 2025**; "cycle syncing diet" dobrou desde 2022 | Living360, WorldHealth.net citando Google Trends | Média-baixa (não verificado no Trends) |
| Ciclo menstrual + treino (BR) | Imprensa BR (Band fev/2026, Terra, Brasil em Folhas set/2026) cobre o tema e **alerta que a ciência não comprova** benefício | Imprensa | Boa |
| Perimenopausa/menopausa | Mercado global US$ 17,8 bi (2024) → US$ 24,4 bi (2030); imprensa BR (Estado de Minas set/2026) fala em "mercado bilionário" e alerta para *menowashing* | Grand View via EM | Média |
| #gymtok / #pilates / #menopause (TikTok) | ~259 bi / ~7,5 bi / ~6 bi de views acumuladas | Agregadores de hashtags, sem data | Baixa (ordem de grandeza) |
| GLP-1 (Brasil) | **R$ 13,3 bi** em Mounjaro+Wegovy+Ozempic (mai/2025–mai/2026); Mounjaro de 208 mil → **4,5 M unidades/mês (+2.070%)**; genérico Ozivy (EMS) lançado 15/06/2026; UBS projeta ~R$ 20 bi em 2026 | ICTQ/Close-Up, InvestNews, NeoFeed/IQVIA | Boa |
| GLP-1 (EUA) | 11% dos adultos usam para emagrecer (Gallup 2026); apps: Shotsy 250 mil MAU, MeAgain 372 mil cadastros | Medscape 2026 | Boa |
| Whey/creatina (BR) | Buscas +205% | Think with Google BR | Média |

**Termos sugeridos para você checar no Google Trends (BR, 5 anos):** "contador de calorias", "calorias por foto", "treino de glúteo", "pilates", "treino para mulheres", "menopausa sintomas", "perimenopausa", "Mounjaro", "proteína por dia", "treinar de acordo com o ciclo", "fortalecer assoalho pélvico".

---

## 3. Benchmarks de monetização (RevenueCat State of Subscription Apps 2025/2026 e outros)

> Observação: os números abaixo vieram de resumos de terceiros do relatório RevenueCat 2026 (o site da RevenueCat estava bloqueado nesta sessão). Confirme no relatório original.

| Métrica (Saúde & Fitness) | Mediana | Top quartil / top 10% | Fonte |
|---|---|---|---|
| Trial → pago | **37,7%** (2026); 39,9% (2025) | > 51,4% (top 25%); 68,3% (top 10%, 2025) | RevenueCat via Sahha, Singular |
| Download → pago (D35) | **2,9%** | > 6,2% | RevenueCat 2026 via Sahha |
| Download → trial | 6,9% | > 23% | idem |
| Receita por instalação | **US$ 0,48 (D14), US$ 0,66 (D60)** — maior de todas as categorias | — | RevenueCat 2026 via Tasu |
| Renovação | mensal 57%; **anual 25%** | — | RevenueCat blog 2026 |
| Mix de receita | **~59–61% vem de planos anuais** (exceção à tendência de semanal) | — | RevenueCat 2026 (um resumo diz 68% mensal — conflito; verificar) |
| Preço mediano | **US$ 9,99/mês; US$ 4,99/semana** | — | RevenueCat 2026 |
| Duração de trial | 5–9 dias converte melhor em H&F (46,8%) que 10–16 dias (40,4%); trials ≥17 dias: 42,5% geral | — | RevenueCat blog |
| Hard paywall vs freemium | Hard paywall converte **~5× mais** downloads em pagantes, com retenção anual semelhante | — | RevenueCat 2026 via BuildMVPFast |
| 82% dos trials começam no dia da instalação | — | — | Sahha |

**América Latina vs. América do Norte (RevenueCat 2026):**
- Download → pago D35: **LATAM 1,5% vs. NA 2,6%**; só a NA chega a dois dígitos no top 10%.
- LATAM tem os **menores preços de todas as regiões** (abaixo até de MEA) e a menor receita por instalação (US$ 0,06–0,09 em 2025).
- Mas LATAM é a região de **crescimento mais rápido de MRR (+17,2%)**.
- LTV por pagante: NA US$ 32 vs. mediana global US$ 23.
- Traduzir o paywall para os 5 principais idiomas gera +37% de LTV — mais que mexer no preço. Um app de meditação dobrou assinaturas na LATAM em 14 dias só com preço regional.
- Taxas de reembolso atípicas na LATAM chegam à casa dos 30%.

**Preços praticados no Brasil (App Store BR / sites oficiais, nov/2026):**

| App | Preço BR | Observação |
|---|---|---|
| MyFitnessPal Premium | **R$ 31,90/mês; R$ 162,90/ano** (também R$ 82,90 e R$ 329,90 listados — Premium+?) | Ficha App Store BR; EUA: US$ 79,99/ano Premium, US$ 99,99 Premium+ |
| Queima Diária | R$ 39,90/mês no plano anual | Site oficial |
| Olho no Prato (foto-IA, BR) | R$ 9,90/mês; **R$ 49,90/ano** | App Store BR |
| Vitaliz AI (foto-IA, BR) | R$ 119,90/ano | Site |
| Cal AI (EUA) | US$ 2,99–3,99/sem; US$ 9,99/mês; **US$ 29,99/ano** (testes A/B de US$ 19,99 a 49,99); trial 3 dias; hard paywall para foto | Blogs de concorrentes — aproximado |
| MacroFactor (EUA) | US$ 11,99/mês; US$ 71,99/ano; sem versão grátis | Blogs |
| Yazio PRO (EUA) | ~US$ 40–48/ano | Blogs |
| Lose It! (EUA) | subiu de US$ 39,99 → US$ 79,99/ano e removeu mensal | Blogs |
| Balance+ (menopausa, EUA) | ~US$ 4–11/mês; US$ 50–99/ano | Blog out/2026 |
| Apps de postura (EUA, 2026) | US$ 9,99/mês; US$ 39,99–99,99/ano | App Store |
| Apps gestação (mercado) | US$ 4,99–19,99/mês; US$ 39,99–99,99/ano; 21,8% dos usuários pagam e geram 68% da receita | Dataintelo/Mordor — relatórios genéricos |

**Faixa recomendada para o seu app (Brasil):** R$ 24,90–29,90/mês e **R$ 149–189/ano** (ancorar como "R$ 12–16/mês"), trial de 7 dias, hard paywall após onboarding. Para iOS EUA/Europa: US$ 9,99/mês e US$ 49,99–59,99/ano.

---

## 4. Avaliação detalhada por ideia

### 4.1 Ideia A — Dieta por foto (calorias, proteína, carbo, gordura)

**Demanda**
- Maior demanda bruta do estudo: nutrição com IA foi um dos segmentos de maior crescimento de receita em 2025 (Sensor Tower). Cal AI: **15 M+ downloads, > US$ 30 M/ano** (MyFitnessPal/TechCrunch, mar/2026); fundador alega US$ 40 M+; estimativas de terceiros US$ 2 M/mês (jul/2026). Gastava > US$ 1 M/mês em anúncios (não verificado).
- Mas: público do Cal AI é **masculino, 15–25 anos** (Forbes). Mulheres são ~58% dos calorie trackers em geral (não verificado).

**Concorrência (saturação altíssima)**
- Cal AI **adquirido pelo MyFitnessPal** (fechado dez/2025, anunciado 02/03/2026) — o incumbente de 200 M de contas agora tem a melhor marca de foto-IA.
- MyFitnessPal Meal Scan, Yazio, Lose It!, MacroFactor, SnapCalorie (LiDAR em iPhone Pro), Foodvisor (top 5 receita Play BR), Fitia.
- **Em PT-BR já há dezenas de clones**: ContaCal (WhatsApp), CalorIA, Olho no Prato (R$ 49,90/ano), Vitaliz AI (R$ 119,90/ano), NutriPic, Nutrion (usa Gemini 2.5) — competição em preço já começou.

**Reclamações recorrentes (lacunas)**
- Porções e pratos mistos: variação de 25–50%; mesma foto re-escaneada dá valores diferentes; estimativa "pela metade" vs. balança; Cal AI afirma "90% de precisão" sem citar fonte.
- Estudo independente (NIH, congresso NUTRITION 2026, cozinha metabólica controlada, 4 apps): **subestimam calorias em ~1/3; Cal AI errou ~345 kcal em média; gordura subestimada em ~30 g** (via Healio, ago/2026). LLMs genéricos: erro médio ~36% (GPT-4o/Claude 3.5) e ~64% (Gemini 1.5 Pro) em 52 fotos; GPT-5 ~30% só com foto, ~14% com ingredientes informados.
- Lacunas exploráveis: comida brasileira (arroz-feijão-farofa, marmita, PF), foco em **proteína** (não só calorias), correção rápida por voz/texto, consistência (mesmo prato = mesmo número), transparência de incerteza ("entre 28 e 40 g de proteína").

**Viabilidade técnica on-device (solo)**
- **iOS 27 (jun/2026): Foundation Models aceita imagem + texto no aparelho**, com ferramentas de Vision (OCR, código de barras) chamáveis pelo modelo — isso torna "foto → alimentos → porção aproximada" viável 100% on-device em iPhones compatíveis com Apple Intelligence (iPhone 15 Pro+). Sem custo de API. Precisão do modelo pequeno (~3B) para porções **não foi medida publicamente** — teste com 50 pratos brasileiros antes de prometer.
- **Android: ML Kit GenAI (Gemini Nano) tem "image description"** via AICore em Pixel 9/10/11, Galaxy S25/S26 e alguns MediaTek/Snapdragon — cobertura pequena na base brasileira (Android de entrada). Alternativa: Gemma 3n/4 via LiteRT (modelo de ~3,6 GB — pesado) ou classificador Core ML/TFLite fine-tunado (só reconhece classes, não porção).
- **Fallback em nuvem é barato**: Gemini 2.5 Flash US$ 0,30/M tokens de entrada; uma foto custa ~258 tokens (imagem ≤384px; imagens maiores são fatiadas em blocos de 258 tokens — valor da documentação do Google segundo meu conhecimento, confirme) → **~US$ 0,0003–0,002 por foto com prompt e saída; ~US$ 0,05–0,30/usuária/mês a 5 fotos/dia**. Totalmente absorvível em R$ 149/ano.
- Bases nutricionais: **TACO (Unicamp)** — sem API oficial; APIs de terceiros (REST com 1.000 req/mês grátis; GraphQL open source) — verificar licença/manutenção. **TBCA (USP/FoRC)**: ~1.900–3.500 alimentos, app Android, sem API oficial encontrada. **Open Food Facts**: códigos de barras, API aberta, cobertura BR razoável. USDA FoodData Central para fallback.
- Porção continua sendo o problema não resolvido: LiDAR de celular tem resolução espacial insuficiente para avaliação dietética (revisão 2026, arXiv).

**Monetização**: comprovada (Cal AI US$ 29,99/ano; MFP R$ 162,90/ano no BR), mas compressão de preço em PT-BR (R$ 49,90/ano). Como produto standalone, você competiria em preço contra clones; como **recurso** dentro de um app de nicho, vira diferencial sem precisar ser o melhor scanner do mundo.

**Riscos**: Apple 1.4.1 exige disclose de metodologia para "medições de saúde" — apps de calorias passam com disclaimer e fontes (TACO/USDA). Risco reputacional se mulher confiar em -33% kcal. Saturação → CAC alto.

**Notas**: Demanda 5 · Concorrência (quanto menor, melhor) 1 · Diferenciação 2 · Facilidade técnica 3 · Monetização 4 · **Geral 3,0**

---

### 4.2 Ideia B — Coach de execução (pose estimation + ângulos articulares)

**Demanda**
- Nicho de "form check" existe (Gymscore, CueForm AI, AiKYNETIX, FormLens, Zing Vision) mas **não há nenhum caso público de receita relevante** em B2C. Mulheres em musculação crescem (+21% Strava; "muscle mommy" 44 M views), mas a dor delas é "não sei montar o treino/não vejo resultado", mais que "quero ângulo do joelho".

**Concorrência e histórico (sinal de alerta)**
- **Kemtai** (Israel) começou em fitness, admitiu que precisava de mais precisão e **migrou para fisioterapia B2B** ("mercado difícil").
- **Zenia** (yoga com câmera) foi **absorvida pela Zing Coach** (fev/2024).
- **Tempo** (hardware 3D, US$ 220 M Series C) segue operando (218 funcionários, set/2026) — mas é hardware.
- **Zing Vision**: avaliações dizem que "detecta um arqueamento leve nas costas", mas também "alerta erro quando a execução estava certa", "feedback chega depois do movimento", problemas em pouca luz. Nota Trustpilot 3,5; 32% de 1 estrela (maioria por cobrança). Gymscore: travamentos no upload de vídeo.
- Guia 2026: "AI form checkers identificam erros óbvios, mas ainda não substituem um coach".

**Viabilidade técnica (solo)**
- Modelos: **Apple Vision** (`VNDetectHumanBodyPoseRequest` 2D 19 pontos; `VNDetectHumanBodyPose3DRequest` 3D 17 articulações em metros, iOS 17+), **MediaPipe Pose Landmarker/BlazePose** (33 pontos, 3D "world landmarks", iOS/Android), **MoveNet** (17 pontos 2D, mais rápido: 34 fps vs 11–12 fps BlazePose em Pixel 5, dado de fornecedor).
- Precisão: preprint 2025 — pipeline pós-processado sobre BlazePose atinge **erro angular médio de 10,6°** (meta de 12° para coaching); BlazePose cru é ~17% pior. Comparativos: MoveNet mais robusto no geral; MediaPipe melhor em 7 de 14 ações. **Nenhum benchmark público do 3D da Apple.**
- Problemas práticos: oclusão (barra, banco, roupa larga), ângulo de câmera (precisa tripé lateral), profundidade monocular ruim para agachamento visto de frente, iluminação de academia, 1 pessoa por vez. Regras de "forma correta" dependem de antropometria (fêmur longo = tronco mais inclinado) → falsos positivos que irritam.
- Esforço de MVP realista para 5 exercícios (agachamento, hip thrust, afundo, RDL, remada) com regras + overlay: 8–12 semanas só nisso, sem conteúdo.
- Flutter/React Native: plugins MediaPipe existem mas são novos, pequenos, Android-first (flutter_mp_pose_landmarker 0.1.8 abr/2026, "iOS requer mais testes"; RN: `@gymbrosinc/react-native-mediapipe-pose`, VisionCamera v4; Expo Go não roda). Nativo é o caminho seguro.

**Monetização**: incerta; público disposto a pagar por isso é pequeno e majoritariamente masculino (powerlifting). Pode ser **recurso** ("conte minhas repetições e avise se eu não desci o suficiente") dentro de um app de rotina.

**Riscos**: Apple 1.4.1 se prometer "evitar lesão"; frustração com falsos alertas → churn; vídeo do corpo é dado sensível (edge AI resolve, mas a percepção "a câmera me filma na academia" é barreira para mulheres).

**Notas**: Demanda 2 · Concorrência 3 · Diferenciação 3 · Facilidade técnica 1 · Monetização 2 · **Geral 2,6**

---

### 4.3 Ideia C — Força + proteína para mulheres em uso de GLP-1 (companion de "caneta")

**Demanda**
- Brasil: **R$ 13,3 bi** em 12 meses; **Mounjaro 4,5 M unidades/mês** (mai/2026, +2.070% a/a); patente da semaglutida caiu em 20/03/2026 e **Ozivy (EMS) lançado em 15/06/2026** → preço cai, base explode em 2026–2027. UBS: ~R$ 20 bi em 2026.
- Dor clínica concreta: **26–40% do peso perdido é massa magra** (SURMOUNT-1/STEP 1); abstract ENDO 2025 sugere que **mulheres perdem massa magra desproporcionalmente** e que mais proteína na 12ª semana reduziu a perda (preliminar). Consenso 2025 de 4 sociedades: **1,2–1,6 g proteína/kg/dia + treino de força**. Medscape 2026: "nichos como mulheres na menopausa em GLP-1" estão sendo atacados por founders.
- EUA: Shotsy 250 mil MAU, MeAgain 372 mil cadastros, MyNetDiary lançou "GLP-1 Companion"; Gallup: 11% dos adultos americanos usam GLP-1. Categoria "GLP-1 companion" reconhecida (Trends.vc lista 20 empresas).
- **Não encontrei** recorte por sexo dos usuários de GLP-1 no Brasil — lacuna; em pesquisas internacionais mulheres são maioria dos usuários para emagrecimento (não verificado aqui).

**Concorrência**
- Em PT-BR: praticamente **nenhum app dedicado** encontrado. Apps de calorias genéricos não endereçam náusea, saciedade precoce, sarcopenia, hidratação, constipação, "food noise".
- Risco de entrada: MyFitnessPal/Cal AI podem lançar modo GLP-1 (MyNetDiary já lançou). Diferencial defensável: PT-BR, comida brasileira, treino em casa com halteres, acolhimento feminino, privacidade.

**Viabilidade técnica**: **a mais simples do estudo**. Núcleo é produto/conteúdo: meta proteica, registro rápido (foto on-device/nuvem barata, código de barras OFF, favoritos), treino de força 3×/semana com vídeos, lembretes de dose (sem recomendar dose), diário de efeitos colaterais, medidas corporais. Edge AI opcional: estimativa de proteína por foto (Foundation Models iOS 27) e contador de repetições (Vision 2D, bem mais fácil que "forma correta").

**Monetização**: público já gasta **R$ 1.000–2.000+/mês** no remédio; R$ 29,90/mês é <3% disso — ancoragem perfeita ("proteja o músculo que você está pagando para perder peso"). Planos anuais fazem sentido (tratamento dura 12+ meses). Possível B2B2C com farmácias/clínicas/telemedicina (Ozivy/EMS, redes de farmácia).

**Riscos**
- **Apple 1.4.1**: diário de medicação é permitido (Apple Health faz isso); NÃO sugerir dose, titulação ou "pule a dose". Incluir fontes (consenso 2025) e disclaimer.
- **ANVISA RDC 657/2022**: software "para bem-estar" (exercício, hábitos) está **fora** da regulação de SaMD, desde que não prometa tratar/diagnosticar. Não falar em "tratamento de obesidade".
- **LGPD**: uso de medicamento = dado sensível de saúde (art. 5º, II; art. 11). Armazenar on-device, sem analytics de eventos de saúde para terceiros (lição Flo × Meta). Criptografia + opção de exportar/apagar.
- Publicidade de medicamento controlado: não usar nomes comerciais em anúncios de forma que pareça promover o remédio (termos como "caneta", "GLP-1", "em tratamento para emagrecimento" são seguros).
- Dependência de ciclo de hype: se preço do genérico cair muito, base cresce; se houver restrição regulatória, encolhe.

**Notas**: Demanda 4 · Concorrência 5 · Diferenciação 4 · Facilidade técnica 5 · Monetização 4 · **Geral 4,2**

---

### 4.4 Ideia D — Treino e nutrição sincronizados ao ciclo menstrual

**Demanda**: buscas "cycle syncing" +124% (2024) e "cycle syncing workout" +62% (2025); tendência TikTok reconhecida pela imprensa brasileira (Band, Terra, Brasil em Folhas 2026). Público 18–35, muito ativo no TikTok/Instagram → aquisição orgânica mais barata.

**Concorrência**: **Flo** (80 M MAU) já vende "cycle syncing" de treino/comida/trabalho no Premium; 28 (vídeos por fase), Wild.AI e FitrWoman (atletas), Clue (sem treino), Lively, Svelta. No Brasil: nenhum dedicado encontrado.

**Diferenciação possível**: privacidade radical (dados do ciclo só no aparelho — depois do júri da Califórnia condenar a Meta por coletar dados de menstruação do Flo em 01/08/2025, e 84% dos apps de ciclo compartilharem dados com terceiros segundo ORCHA), treino de força periodizado em PT-BR, integração com Apple Health/Health Connect.

**Viabilidade técnica**: fácil (regras por fase + conteúdo). Edge AI real é pequena: previsão de ciclo on-device e, opcionalmente, foto de alimentos.

**Monetização**: moderada — público mais jovem, menor renda, Flo oferece quase de graça no plano premium. Preço BR provavelmente R$ 14,90–19,90/mês.

**Riscos**: **ciência fraca** (revisões sistemáticas não acham benefício; imprensa BR já faz essa crítica) → risco reputacional e de rejeição 1.4.1 se alegar performance. Previsão de ciclo + "anticoncepção" entra na RDC 657 — não prometer isso. Contrainte: muita usuária usa anticoncepcional hormonal e não tem ciclo natural.

**Notas**: Demanda 4 · Concorrência 3 · Diferenciação 3 · Facilidade técnica 4 · Monetização 3 · **Geral 3,4**

---

### 4.5 Ideia E — Rotina para perimenopausa/menopausa (35–60 anos)

**Demanda**
- Mercado global de menopausa US$ 17,8 bi (2024) → US$ 24,4 bi (2030) (Grand View via Estado de Minas, set/2026). Apps de "hormone balancing": estimativas de **US$ 0,6 bi a US$ 1,8 bi em 2026** (relatórios divergem 3×; baixa confiabilidade).
- UVA: 55% das mulheres de 30–35 anos relatam sintomas moderados/severos de perimenopausa (citado por Living360). Varejo brasileiro (Vichy/Neovadiol) está "educando" o público via Instagram (Times Brasil). Imprensa BR (set/2026) trata como mercado bilionário em formação.
- ClassPass: Pilates e low-impact (+112%) crescem — modalidades preferidas desta faixa. Mulheres 40+ têm renda e tempo; "Silver fitness" nas redes Smart Fit/Bodytech.
- **Lacuna**: não encontrei volume de busca BR para "menopausa"/"perimenopausa" — checar no Trends.

**Concorrência**: EUA/UK — Balance (Editors' Choice, ORCHA; balance+ US$ 50–99/ano), Midday (Lisa Health, clínica), Peppy (B2B), Health & Her, Clue. **Brasil**: Raquel Menopausa (gratuito, pesquisa com 300 usuárias out/2026), Namu Mulher (B2B, nov/2026). Ou seja, **nenhum app pago B2C em PT-BR consolidado**.

**Diferenciação**: rotina diária prática (força 3×/semana para massa óssea/muscular, proteína 1,2–1,6 g/kg, sono, registro de fogachos/humor com padrões detectados no aparelho), linguagem brasileira, sem venda de suplemento (anti-menowashing). Edge AI: detecção de padrões de sintomas × sono × treino on-device; foto de refeição para proteína; contador de repetições.

**Viabilidade técnica**: fácil-média (conteúdo + regras + ML leve). Integração com Apple Watch/Health (sono, FC) é diferencial.

**Monetização**: alta — público com renda, problema crônico (LTV longo: 5–10 anos de climatério), já paga por suplementos e consultas. Preço BR R$ 29,90/mês ou R$ 189/ano é plausível. Possível B2B (RH/benefícios, como Peppy/Namu).

**Riscos**: **linha tênue com medicina** (TRH, sintomas severos) → manter como bem-estar, encaminhar para médico, citar fontes (SOBRAC, NAMS). Apple 1.4.1 exige fontes/disclaimer. Aquisição: público 40+ é menos TikTok, mais Instagram/Facebook/YouTube e indicação médica — CAC pode ser maior, mas concorrência em anúncios é menor que em "calorias".

**Notas**: Demanda 4 · Concorrência 4 · Diferenciação 4 · Facilidade técnica 4 · Monetização 4 · **Geral 4,0**

---

### 4.6 Ideia F — Gestação/pós-parto + assoalho pélvico

**Demanda**: mercado de apps de gestação/pós-parto 2026 estimado entre **US$ 0,41 bi e US$ 2,1 bi** (quatro relatórios divergem 5×). Incontinência urinária: prevalência global ~27% das mulheres (Rostami 2025, variação 3–60%); estudo BR 2025 em praticantes de atividade física: muitas conhecem fisioterapia pélvica, poucas fizeram. Demanda constante (nascimentos) mas **janela de uso curta** (~12–18 meses), e cada usuária precisa ser reconquistada.

**Concorrência**: Baby2Body, Juna, Expectful, Every Mother (EUA); trainers com hardware Elvie (US$ 149–249) e Perifit (US$ 99–149), Squeezy (NHS, US$ 3). Brasil: apps de gestação de grandes players (BabyCenter etc.) são de conteúdo, não de treino. Fisioterapia pélvica cresce como profissão no Brasil.

**Viabilidade técnica**: conteúdo + lembretes; Kegel guiado por áudio/háptico; detecção de diástase por foto é **arriscada** (alegação diagnóstica). Edge AI limitada.

**Monetização**: 21,8% pagam e geram 68% da receita; US$ 5–15/mês em iOS (relatórios genéricos). LTV curto é o problema central.

**Riscos**: gestação é zona médica — Apple e ANVISA mais exigentes; responsabilidade em caso de evento adverso; LGPD sensível.

**Notas**: Demanda 3 · Concorrência 3 · Diferenciação 3 · Facilidade técnica 4 · Monetização 2 · **Geral 3,0**

---

### 4.7 Ideia G — Análise de pele por câmera

**Demanda**: alta entre mulheres, mas o mercado de "AI skin analysis" (US$ 2,1–3,8 bi em 2026, relatórios divergentes) é **B2B/varejo**: Perfect Corp (YouCam, Skincare Pro) faturou **US$ 69,2 M em 2025**; Haut.AI vende para NAOS/Bioderma; L'Oréal Skin Genius é gratuito para vender creme.

**Concorrência/diferenciação**: marcas oferecem análise grátis; apps independentes (TroveSkin, Skin Bliss) — não encontrei receita. Difícil cobrar assinatura pelo que a Vichy dá de graça.

**Viabilidade técnica**: Vision face landmarks + modelo de classificação de acne/manchas on-device é factível, mas **alegações de "medir" pele** caem na 1.4.1 (validação de metodologia) e iluminação/câmera variam demais.

**Monetização**: fraca em assinatura; melhor em afiliação de produtos (modelo diferente do pedido).

**Notas**: Demanda 3 · Concorrência 2 · Diferenciação 2 · Facilidade técnica 3 · Monetização 2 · **Geral 2,4**

---

### 4.8 Ideia H — Avaliação de postura/glúteo por foto

**Demanda**: baixa e pontual (uma avaliação, depois esquece). Entre fev e abr/2026 surgiram vários apps idênticos na App Store (Posture Scan, PosturePal US$ 89,99–99,99/ano, StatueMax on-device US$ 39,99/ano, FixMyPosture US$ 180,99/ano, PhysioAI) — a maioria **sem avaliações** → sinal de mercado pequeno saturado por indies.

**Viabilidade**: fácil (Vision 2D + ângulos), por isso saturou. "Avaliação de glúteo" por foto é tecnicamente frágil (volume ≠ silhueta) e esbarra em body shaming.

**Notas**: Demanda 2 · Concorrência 2 · Diferenciação 2 · Facilidade técnica 4 · Monetização 2 · **Geral 2,0**

---

## 5. Tabela comparativa

Escala 1–5 (5 = melhor para você). "Concorrência" = quanto menos saturado, maior a nota. "Facilidade técnica" = 5 significa fácil para dev solo em 6–10 semanas.

| Ideia | Demanda (mulheres) | Concorrência | Diferenciação | Facilidade técnica | Monetização | **Geral** |
|---|---|---|---|---|---|---|
| **C. Força + proteína p/ mulheres em GLP-1** | 4 | 5 | 4 | 5 | 4 | **4,2** |
| **E. Perimenopausa/menopausa** | 4 | 4 | 4 | 4 | 4 | **4,0** |
| D. Treino sincronizado ao ciclo | 4 | 3 | 3 | 4 | 3 | 3,4 |
| A. Dieta por foto | 5 | 1 | 2 | 3 | 4 | 3,0 |
| F. Gestação/pós-parto + pélvico | 3 | 3 | 3 | 4 | 2 | 3,0 |
| B. Coach de execução (pose) | 2 | 3 | 3 | 1 | 2 | 2,6 |
| G. Pele por câmera | 3 | 2 | 2 | 3 | 2 | 2,4 |
| H. Postura/glúteo por foto | 2 | 2 | 2 | 4 | 2 | 2,0 |

Pesos usados no "Geral": demanda 25%, concorrência 15%, diferenciação 20%, facilidade técnica 20%, monetização 20%.

---

## 6. Recomendação ranqueada e raciocínio

### 1º — Construir **C** agora, com **E** como expansão natural (mesmo app, mesma persona 35+)

Por quê:
1. **Timing único no Brasil**: genérico de semaglutida (jun/2026) + Mounjaro em 4,5 M unidades/mês criam milhões de novas usuárias em 2026–2027 com uma dor que nenhum app PT-BR resolve: perder peso sem perder músculo. Não há "Cal AI" dessa categoria aqui ainda.
2. **Disposição a pagar comprovada por comportamento**: quem gasta R$ 1.000+/mês no remédio e já compra whey (+205% buscas) paga R$ 29,90/mês sem fricção. Isso compensa a conversão LATAM mais baixa (1,5% D35).
3. **Baixo risco técnico**: o núcleo é rotina + conteúdo + registro. A edge AI entra como diferencial (foto → proteína on-device no iOS 27; contador de repetições com Vision), não como pré-requisito. Você consegue lançar em 6–10 semanas.
4. **Privacidade como argumento de venda**: "seus dados de medicação, peso e ciclo nunca saem do seu celular" — concreto e verificável, ao contrário do que o Flo prometia. Reduz risco LGPD e evita o erro do Flo (SDKs de analytics com eventos de saúde).
5. **Caminho de expansão**: a mesma usuária 35–55 entra na perimenopausa; força + proteína + sono + registro de sintomas são o mesmo motor. A ideia E dobra o mercado endereçável sem reescrever o app.

### 2º — E (menopausa) sozinha, se você preferir evitar qualquer proximidade com medicamento
Mercado maior e mais duradouro que GLP-1, concorrência BR quase zero, mas aquisição mais lenta e tom mais "médico". Nota 4,0.

### 3º — D (ciclo), como módulo opcional, não como produto
Boa para aquisição orgânica no TikTok, mas ciência frágil e Flo já faz. Ofereça "ajuste pelo ciclo" como toggle dentro do app, com linguagem honesta ("algumas mulheres preferem…").

### 4º — A (dieta por foto) como recurso, nunca como produto
Você não vai vencer MyFitnessPal+Cal AI nem os clones a R$ 49,90/ano. Mas um scanner de **proteína** on-device, calibrado para comida brasileira (TACO/TBCA), dentro do app C/E, é um diferencial que ninguém pequeno terá.

### 5º–8º — F, B, G, H
B é a mais tentadora tecnicamente e a que mais consumiria seu tempo com menor retorno: todos os players consumer de "form check" por câmera ou pivotaram (Kemtai), foram absorvidos (Zenia) ou colecionam reclamações de falsos alertas (Zing). Use pose estimation só para **contar repetições e tempo sob tensão** (problema 10× mais fácil e útil para quem treina em casa).

---

## 7. MVP sugerido (v1 em 6–10 semanas) — "Rotina de força + proteína para mulheres 35+ (modo caneta GLP-1)"

**Princípios**: iOS-first nativo, hard paywall após onboarding (trial 7 dias), tudo on-device por padrão, zero SDK de analytics com dados de saúde.

| Semana | Entrega |
|---|---|
| 1–2 | Onboarding (peso, altura, objetivo, nível, "uso caneta? qual dia da aplicação?" opcional, fase da vida: ciclo regular / perimenopausa / menopausa) → meta de proteína (1,2–1,6 g/kg com fontes) e plano de 3 treinos/semana. Paywall (RevenueCat ou StoreKit 2). Modelo de dados em SwiftData com criptografia. |
| 3–4 | **Registro de proteína**: busca em base TACO/TBCA embarcada (SQLite), favoritos, código de barras (Vision + Open Food Facts), **foto → lista de alimentos + estimativa de porção com Foundation Models (iOS 27)**, com fallback de busca manual em aparelhos sem Apple Intelligence (ou Gemini Flash opcional com consentimento). Sempre mostrar faixa de incerteza e botão "ajustar porção". |
| 5–6 | **Treino de força em casa/academia**: 2 programas (halteres/elástico em casa; máquinas básicas) com vídeos curtos de 20–30 exercícios (pode gravar você mesma/o ou licenciar), timer, registro de carga, progressão automática. **Contador de repetições com Vision 2D** para 4–5 exercícios simples (agachamento, afundo, elevação pélvica, remada curvada, flexão apoiada) — só contagem + "amplitude baixa", sem julgar "forma". |
| 7 | **Diário de bem-estar** (opcional): náusea, apetite, constipação, energia, fogachos, sono, hidratação; lembretes gentis (água, proteína no café, dia de aplicação sem citar dose). Gráficos semanais on-device (peso, proteína, treinos, sintomas). |
| 8 | Integração Apple Health (peso, passos, sono, treinos) · Widgets e Live Activity para "proteína de hoje" · Exportar/apagar dados (LGPD). |
| 9–10 | Polimento, beta no TestFlight com 30–50 mulheres de grupos de Facebook/Telegram de "caneta" e de perimenopausa, ajuste de paywall e preço, textos de 1.4.1 (fontes, disclaimers), lançamento. |

**Fora da v1**: Android, modo ciclo completo, chat com IA, comunidade, receitas elaboradas, integração com Apple Watch nativa (fazer na v1.1).

**Métricas de validação (60 dias)**: trial-start ≥ 10% dos downloads; trial→pago ≥ 35% (mediana H&F 37,7%); D30 retention de pagantes ≥ 40%; ≥ 3 registros de proteína/dia nas pagantes.

---

## 8. Posicionamento e nicho

- **Nome/claim (exemplos)**: "Perca gordura, não músculo." / "A rotina de força e proteína para mulheres em tratamento com caneta — e para a fase 40+."
- **Promessas verificáveis**: (1) meta de proteína baseada em consenso médico 2025; (2) treino de força de 25 min, 3×/semana; (3) seus dados nunca saem do seu iPhone — sem rastreadores.
- **Não prometer**: emagrecimento, tratamento de obesidade/menopausa, ajuste de dose, diagnóstico, "forma perfeita".
- **Canais**: grupos de Facebook/Telegram/WhatsApp de usuárias de Mounjaro/Ozempic/Ozivy (enormes no Brasil), creators de nutrição e musculação feminina no Instagram/TikTok (micro-influenciadoras 10–100 mil), parcerias com nutricionistas e fisios pélvicas (código de indicação), SEO em PT-BR ("quanto de proteína tomando Mounjaro", "perda de massa magra Ozempic treino").
- **Preço**: R$ 29,90/mês · **R$ 169,90/ano** (ancorar "R$ 14/mês"); trial 7 dias; sem plano semanal (evita reembolso/churn LATAM). Internacional: US$ 9,99/mês · US$ 59,99/ano.
- **Expansão**: v1.5 modo perimenopausa com registro de fogachos/sono e padrões on-device; v2 Android; v2.5 B2B2C (farmácias, telemedicina, RH).

---

## 9. Stack técnica recomendada

| Camada | Recomendação | Por quê |
|---|---|---|
| Plataforma v1 | **iOS nativo (Swift 6 / SwiftUI / SwiftData)**, iOS 17+ com recursos de IA em iOS 26/27 | Toda a edge AI que você quer é nativa Apple: Foundation Models (imagem+texto on-device, iOS 27), Vision (pose 2D/3D, OCR, barcode), Core ML, HealthKit, StoreKit 2. Zero ponte, zero plugin abandonado. iOS = 22% do Brasil, mas é onde está a renda e a conversão; RevenueCat mostra que H&F monetiza melhor em iOS. |
| Pose/repetições | `VNDetectHumanBodyPoseRequest` (2D, 19 pontos) para contagem; 3D só se precisar de ângulo no plano sagital | 2D é suficiente para contar e detectar amplitude; mais estável e rápido |
| Foto → alimento | Foundation Models com `@Generable` (lista de itens + porção estimada + confiança) → lookup na base TACO/TBCA local; fallback Gemini 2.5 Flash via backend minúsculo (Cloudflare Workers/Supabase Edge) só com opt-in | Sem custo em aparelhos compatíveis; custo residual ~US$ 0,05–0,30/usuária/mês no fallback |
| Base nutricional | SQLite embarcado com TACO (Unicamp) + TBCA (USP) + USDA para fallback; Open Food Facts para barcodes | Dados brasileiros; verificar licenças (TACO é pública; TBCA pedir autorização de uso) |
| Pagamentos | StoreKit 2 + RevenueCat (grátis até US$ 2,5 mil MTR; 1% depois) | Paywalls remotos, A/B, métricas de trial |
| Backend | Quase nenhum: iCloud/CloudKit para backup criptografado opcional; sem contas obrigatórias | Reduz LGPD, custo e superfície de ataque |
| Analytics | TelemetryDeck ou Apple App Analytics; **nunca** enviar eventos com peso/medicação/sintomas | Evita o erro Flo × Meta (SDKs com "custom app events") |
| Android (fase 2) | **Kotlin nativo** com ML Kit Pose + ML Kit GenAI (Gemini Nano, onde disponível) + CameraX; ou Kotlin Multiplatform para compartilhar regras de negócio | Plugins Flutter/RN de MediaPipe são imaturos (0.x, Android-first, "iOS requer testes"); Expo Go não roda nativo |
| Por que não React Native/Flutter agora | Perde Foundation Models e Vision nativos, depende de pontes para câmera em tempo real, e a economia de "um código, duas lojas" não vale para um app cujo diferencial é IA on-device | — |

---

## 10. Riscos transversais e como mitigar

| Risco | Impacto | Mitigação |
|---|---|---|
| **Apple 1.4.1** (saúde/dano físico): exige fontes para recomendações, disclaimers, metodologia para "medições" | Rejeição/atraso | Tela "Fontes" (consenso 2025, SOBRAC, OMS), disclaimer "não substitui orientação médica", nunca usar "diagnóstico", "tratamento", "dose"; estimativa de porção rotulada como "aproximada" |
| **ANVISA RDC 657/2022** (SaMD) | Enquadramento como dispositivo médico | Manter finalidade declarada como bem-estar (exercício/hábitos) — explicitamente excluída da norma; não prometer prevenir/tratar/contracepção |
| **LGPD** (dados de saúde = sensíveis) | Multas, perda de confiança | Processamento on-device, base legal consentimento explícito, exportar/apagar, DPO nomeado, sem compartilhamento com ad-tech |
| Privacidade percebida (câmera/vídeo do corpo, ciclo, medicação) | Barreira de adoção | Edge AI real + comunicação clara ("nada sai do aparelho"); vídeos nunca salvos, só keypoints |
| Saturação (ideias A/B/H) | CAC alto, compressão de preço | Escolher C/E; usar foto-IA como recurso |
| Conversão LATAM baixa (1,5%) e reembolsos | Receita abaixo do esperado | Preço regional, plano anual ancorado, trial 7 dias, paywall em PT-BR, também vender em EUA/Portugal |
| Dependência do hype GLP-1 | Base encolhe se regulação/escassez mudar | Expandir para perimenopausa (crônico) desde a v1.5 |
| Precisão da foto-IA (-33% kcal) | Reclamações, 1 estrela | Focar proteína (menos sensível a óleo/porção que calorias), mostrar faixa, permitir correção em 1 toque, aprender porções frequentes da usuária |
| Hardware Apple Intelligence (iPhone 15 Pro+) limita quem usa foto on-device | Recurso indisponível em parte da base | Fallback nuvem com opt-in ou busca manual; comunicar claramente |

---

## 11. Fontes

**Monetização / benchmarks**
- RevenueCat, State of Subscription Apps 2026: https://www.revenuecat.com/state-of-subscription-apps (não acessível nesta sessão; dados via resumos abaixo)
- RevenueCat 2025: https://www.revenuecat.com/state-of-subscription-apps-2025
- Sahha, Health & Fitness conversion benchmarks 2026: https://sahha.ai/blog/health-app-monetization-first-month/
- Singular, 17 learnings RevenueCat: https://www.singular.net/blog/subscription-apps/
- RevenueCat, renovação por categoria: https://www.revenuecat.com/blog/growth/average-subscription-renewal-rates-by-app-category
- RevenueCat, duração de trial: https://www.revenuecat.com/blog/growth/free-trial-length
- Tasu, RPI por categoria: https://tasu.ai/library/app-category-revenue-per-install-benchmark
- BuildMVPFast, hard paywall vs trial: https://www.buildmvpfast.com/blog/hard-paywall-vs-free-trial-revenuecat-indie-app-2026
- Medium (Chuvak), RevenueCat + Adapty 2026: https://chuvak-pavel.medium.com/state-of-app-monetization-2026-key-trends-from-revenuecat-and-adapty-8bdc23a4bb6f
- SaaStr, top 10 learnings: https://www.saastr.com/the-top-10-learnings-from-revenuecats-state-of-subscription-apps-how-115000-mobile-apps-deliver-16b-in-revenue-whats-working-whats-quietly-killing-growth/
- RevenueCat, preços globais: https://www.revenuecat.com/blog/growth/5-quick-fixes-for-bad-default-global-pricing-on-app-stores

**Mercado Saúde & Fitness**
- Sensor Tower, H&F revenue fueled by AI (fev/2026): https://sensortower.com/blog/health-and-fitness-apps-ai
- Sensor Tower, Top 5 H&F US Q2 2025: https://sensortower.com/blog/2025-q2-unified-top-5-health%20and%20fitness-units-us-600af518241bc16eb8dce802
- Fitt Insider, MyFitnessPal acquires Cal AI: https://insider.fitt.co/myfitnesspal-acquires-rival-food-tracker-cal-ai/
- Grand View Research, fitness app market: https://www.grandviewresearch.com/industry-analysis/fitness-app-market
- Similarweb, top grossing H&F Google Play Brasil: https://www.similarweb.com/pt/top-apps/google/brazil/health-fitness/top-grossing/
- Appfigures, top H&F iOS Brasil: https://app.appfigures.com/top-apps/ios-app-store/brazil/iphone/health-and-fitness
- StatCounter, OS mobile Brasil: https://gs.statcounter.com/os-market-share/mobile/brazil

**Brasil: fitness, mulheres, GLP-1, menopausa**
- Smart Fit, Resultados 4T25 (Queima Diária 438 mil clientes): https://api.mziq.com/mzfilemanager/v2/d/757d3905-6a2d-496a-bbb0-32ddd17de6a0/544a3617-557a-963c-e276-3ea9cc4f7243?origin=2
- Diário do Comércio, Queima Diária: https://diariodocomercio.com.br/negocios/queima-diaria-revolucionou-mercado-fitness/
- Queima Diária, preços: https://www.queimadiaria.com/assinar/
- Pacto, mercado fitness BR 2026: https://blog.sistemapacto.com.br/mercado-fitness-brasil-2026-panorama-setorial/
- Bling, mercado fitness BR 2025: https://blog.bling.com.br/mercado-fitness-no-brasil/
- IBOPE Repucom, mulheres 60% dos novos praticantes: https://www.iboperepucom.com/br/noticias/mulheres-sao-maioria-entre-os-novos-praticantes-de-atividades-fisicas-no-brasil/
- Futuro da Saúde, apps de atividade física (75% mulheres, dado antigo): https://futurodasaude.com.br/aplicativos-para-atividades-fisicas/
- Think with Google BR, bem-estar: https://business.google.com/br/think/consumer-insights/paradoxo-bem-estar-brasil/
- Band, pilates recorde no Google 2025: https://www.band.com.br/noticias/tendencias-esporte-202512091653
- ABPilates, ACSM 2026: https://abpilates.com.br/atualizacao-2026-pilates-top-5-tendencias-fitness-acsm/
- Vitacore Pilates (Strava +21% força mulheres): https://www.vitacorepilates.com.br/post/7-tend%C3%AAncias-de-bem-estar-em-2025-que-explicam-o-boom-do-pilates-no-brasil
- ICTQ, canetas R$ 13,3 bi: https://ictq.com.br/varejo-farmaceutico/5308-canetas-emagrecedoras-movimentam-r-13-3-bilhoes-em-um-ano-no-brasil
- InvestNews, mercado pode dobrar em 2026: https://investnews.com.br/negocios/ozempic-mounjaro-2026-brasil-genericos/
- NeoFeed/IQVIA: https://neofeed.com.br/negocios/o-efeito-das-canetas-emagrecedoras-ozempic-e-mounjaro-no-varejo-farmaceutico-brasileiro/
- Estado de Minas, menopausa mercado bilionário (set/2026): http://www.em.com.br/bem-viver/2026/09/7502813-menopausa-vira-mercado-bilionario-mas-preocupa-medicos.html
- Times Brasil, menopausa e varejo: https://timesbrasil.com.br/colunas/economia-prateada/menopausa-o-varejo-educa-o-que-o-sistema-nao-ensinou-e-agora/
- Raquel Menopausa (app BR): https://www.raquelmenopausa.com/
- Namu Mulher (nov/2026): https://www.onortao.com.br/2026/11/da-primeira-menstruacao-menopausa-namu.html
- Band, treinar conforme o ciclo (fev/2026): https://www.band.com.br/saude/noticias/treinar-conforme-o-ciclo-menstrual-funciona-o-que-diz-a-ciencia-202602161337
- Brasil em Folhas, sincronizar ao ciclo sem comprovação (set/2026): https://www.brasilemfolhas.com.br/2026/09/sincronizar-dieta-e-treino-ao-ciclo-menstrual-nao-tem-comprovacao-cientifica/

**Dieta por foto / concorrentes**
- TechCrunch, MyFitnessPal adquire Cal AI: https://techcrunch.com/2026/03/02/myfitnesspal-has-acquired-cal-ai-the-viral-calorie-app-built-by-teens/
- Forbes, Cal AI (público 15–25, masculino): https://www.forbes.com/sites/josipamajic/2024/11/02/meet-the-17-year-old-ceo-behind-a-12m-ai-powered-nutrition-app/
- Superframeworks, Cal AI case study: https://superframeworks.com/case-study/cal-ai
- Healio, apps subestimam calorias em 33% (ago/2026): https://www.healio.com/news/primary-care/20260804/ai-photobased-calorietracking-tools-underestimate-them-by-33
- Callie, revisão de estudos 2023–2026: https://www.mycallie.app/en/blog/ai-calorie-accuracy-research
- Kcalm, ChatGPT accuracy review: https://kcalm.app/blog/chatgpt-calorie-counting-accuracy/
- Fitia, precisão apps por foto: https://fitia.app/learn/article/ai-calorie-photo-apps-accuracy-2026/
- JustUseApp, reviews Cal AI: https://justuseapp.com/en/app/6480417616/cal-ai-calorie-tracking/reviews
- Eesel, Cal AI pricing: https://www.eesel.ai/blog/cal-ai-pricing
- Nutrola, por que Cal AI é caro: https://nutrola.app/en/blog/why-is-cal-ai-so-expensive
- Nutrola, market share calorie apps (58% mulheres, não verificado): https://nutrola.app/en/blog/calorie-tracking-app-market-share-2026-most-users
- MyFitnessPal App Store BR (preços R$): https://apps.apple.com/br/app/myfitnesspal-di%C3%A1rio-alimentar/id341232718
- Olho no Prato (BR): https://apps.apple.com/br/app/id6757802677
- Vitaliz AI (BR): https://www.vitaliz.app/
- ContaCal (BR): https://contacal.com/en
- SnapCalorie App Store: https://apps.apple.com/us/app/-/id1574239307
- arXiv, revisão de estimativa de porção 2026: https://arxiv.org/pdf/2602.05078
- Fitia, melhores macro trackers 2026: https://fitia.app/learn/article/best-macro-tracker-apps-2026/

**Pose estimation / form check**
- Zing adquire Zenia (fev/2024): https://www.prweb.com/releases/zing-coach-acquires-interactive-workout-pioneer-zenia-to-increase-personalization-and-responsivity-within-its-ai-driven-fitness-app-302061746.html
- Kemtai, pivot para fisioterapia (podcast Mindbowser): https://www.mindbowser.com/podcast/building-the-future-of-physiotherapy-kemtais-journey-with-mike-telem/
- Kemtai site: https://kemtai.com/?p=2424
- Tracxn, Tempo: https://tracxn.com/d/companies/tempo/__MprfUYMasaeKMrJrA3D7B_IcBGSl1q5FDwEEDAO6SVc
- Techpoint, Zing Coach 45 dias: https://techpoint.africa/guide/zing-coach-ai-review/
- Trustpilot Zing Coach: https://www.trustpilot.com/review/zing.coach
- Sensai, best form check apps 2026: https://www.sensai.fit/blog/best-ai-workout-form-check-apps-2026
- Kensoforge, form checkers 2026: https://kensoforge.com/insights/ai-form-checker-computer-vision-lifting-apps-2026
- arXiv, physics-informed posture (10,6° erro): https://arxiv.org/pdf/2512.06783
- MDPI, comparativo skeleton-based (MoveNet vs MediaPipe): https://www.mdpi.com/1999-5903/14/12/380
- PoseTracker, best pose model 2026: https://www.posetracker.com/news/best-pose-estimation-model-in-2026-the-real-time-mobile-guide
- Flutter plugin flutter_mp_pose_landmarker: https://pub.dev/packages/flutter_mp_pose_landmarker
- RN MediaPipe pose: https://www.npmjs.com/package/@gymbrosinc/react-native-mediapipe-pose

**Edge AI (Apple/Google)**
- Apple, What's new iOS (Foundation Models multimodal): https://developer.apple.com/ios/whats-new/
- MacRumors, Xcode/Foundation Models (jun/2026): https://www.macrumors.com/2026/06/08/apple-unveils-xcode-and-models-improvements/
- Blake Crosley, Foundation Models image input iOS 27: https://blakecrosley.com/blog/foundation-models-image-input-ios-27
- Android Developers, Gemini Nano: https://developer.android.com/ai/gemini-nano
- Android Developers Blog, ML Kit GenAI (ago/2025): https://android-developers.googleblog.com/2025/08/the-latest-gemini-nano-with-on-device-ml-kit-genai-apis.html
- Forasoft, on-device AI Android 2026: https://www.forasoft.com/blog/article/neural-networks-on-android-369
- Google Cloud, preços Gemini: https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing

**Bases nutricionais BR**
- TabNews, TACO API REST: https://www.tabnews.com.br/dnts0/transformei-a-tabela-taco-unicamp-em-uma-api-rest-moderna-com-ia-e-gerador-de-rotulos
- TACO GraphQL: https://www.blog.brightcoding.dev/2026/07/06/taco-api-the-essential-graphql-tool-for-nutrition-data
- Agência FAPESP, TBCA ecossistema digital: https://agencia.fapesp.br/tabela-brasileira-de-composicao-de-alimentos-se-transforma-em-ecossistema-digital-de-pesquisa/59305
- Nutritotal, TBCA app 3.500 alimentos: https://nutritotal.com.br/pro/material/tabela-brasileira-de-composicao-dos-alimentos/

**Mulheres / femtech / menopausa / GLP-1**
- Business of Apps, Flo statistics: https://www.businessofapps.com/data/flo-statistics/
- General Atlantic, Flo unicórnio: https://www.generalatlantic.com/media-article/flo-health-secures-more-than-200m-investment-from-general-atlantic-to-revolutionize-womens-health-first-purely-digital-consumer-womens-health-app-to-achieve-unicorn-status/
- Flo, Anonymous Mode FAQ: https://amp.flo.health/privacy-portal/anonymous-mode-faq
- Forbes Australia, Sweat (US$ 65 M): https://www.forbes.com.au/covers/entrepreneurs/im-the-boss-now-kayla-itsines-on-being-back-at-sweats-helm/
- PLOS ONE 2025, She-economy pagamento fitness: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0319246
- Phase, best cycle syncing apps 2026: https://cyclesyncedlifting.com/blog/best-cycle-syncing-apps
- Living360, trends 2026 (cycle syncing +124%): https://living360.uk/5-fitness-trends-2026-ai-personal-training/
- WorldHealth.net, cycle syncing hype: https://worldhealth.net/news/cycle-syncing-social-media-hyp-hormone-health/
- Wellphases, menopause apps 2026 (Balance preços): https://wellphases.com/best-menopause-apps/
- Research and Markets, menopause app market: https://www.researchandmarkets.com/reports/6178247/menopause-digital-hormone-balancing-app-market
- Medscape, GLP-1 apps (2026): https://www.medscape.com/viewarticle/glp-1-apps-helpful-companions-or-false-sense-security-2026a1000i9n
- Trends.vc, GLP-1 companion apps: https://trends.vc/opportunities/glp-1-companion-apps
- ClinicalTrials.gov LEAN-PREP: https://clinicaltrials.gov/study/NCT06885736
- Second Nature, GLP-1 muscle/protein planner (1,2–1,6 g/kg): https://www.secondnature.io/guides/weight-loss-medications/glp1-muscle-protein-planner
- ClassPass Look Back 2025: https://classpass.com/blog/2025-classpass-look-back-report/
- Market.us, pregnancy/postpartum apps: https://market.us/report/pregnancy-tracking-and-postpartum-care-apps-market/
- Markwide, pelvic floor exerciser market: https://markwideresearch.com/pelvic-floor-exerciser-market
- Perfect Corp (receita 2025): https://en.wikipedia.org/wiki/Perfect_Corp
- Apps de postura 2026: https://apps.apple.com/app/id6758010343 · https://apps.apple.com/app/id6790813674

**Regulatório / privacidade**
- ANVISA, Perguntas & Respostas RDC 657/2022: https://bibliotecadigital.anvisa.gov.br/jspui/bitstream/anvisa/16043/1/perguntas-respostas-rdc-657-de-2022-v1-01-09-2022.pdf
- Mattos Filho, novo marco SaMD: https://www.mattosfilho.com.br/unico/anvisa-softwares-dispositivos/
- Apple Developer Forums, 1.4.1 wellness vs medical: https://developer.apple.com/forums/thread/807508
- ConductAtlas, resumo 1.4.1: https://conductatlas.com/platform/apple/apple-app-store-review-guidelines/health-and-medical-app-requirements/
- TechCrunch, júri condena Meta no caso Flo (ago/2025): https://techcrunch.com/2025/08/05/jury-rules-meta-violated-california-privacy-laws-by-collecting-menstrual-health-data-from-flo
- HIPAA Journal, Flo × Meta: https://www.hipaajournal.com/jury-trial-meta-flo-health-consumer-privacy/
- Campaign, 84% dos apps de ciclo compartilham dados (ORCHA): https://campaignlive.com/article/period-tracker-app-flo-turns-anonymous-mode-will-ease-user-privacy-concerns/1799360
