# Pesquisa de mercado: em qual setor construir um app pago, nativo e com IA on-device para mulheres

> **Nota (08/11/2026):** as decisões finais de produto, nome (Nutrelle), posicionamento e preço estão em [`produto.md`](produto.md) e prevalecem sobre as recomendações deste relatório.

**Data da pesquisa:** 08/11/2026 (dados coletados entre 15:45 e 16:40 UTC)
**Base de evidência primária:** Google Trends (API `trends.google.com`, Brasil e mundo; janelas de 5 anos, 12 meses e 3 meses) e RevenueCat *State of Subscription Apps* 2026 e 2025 (texto completo das páginas públicas + 3 posts do blog).
**Dados brutos auditáveis:** `docs/dados/trends/` (CSVs das séries, JSON das consultas relacionadas, `_summary.json`, scripts `trends.py`/`analyze.py`, planos de consulta e log da coleta).
**Para quem:** desenvolvedor(a) solo no Brasil escolhendo setor + ideia de app pago para mulheres com IA de borda (edge AI).

> **Aviso sobre gênero:** o Google Trends **não segmenta por gênero**. Usei termos com forte viés feminino (menopausa, ciclo menstrual, treino de glúteo, skincare, cronograma capilar, coloração pessoal, amamentação etc.) como proxy. Tudo que não vem do Trends ou da RevenueCat está marcado como **[secundário]**.

---

## 1. Resumo executivo

### 1.1 Ranking de setores (Brasil, termos com viés feminino)

| # | Setor | Últimos 12 meses | Últimos 3 meses (ago–nov/2026) | Monetização (RevenueCat 2026) | Veredito |
|---|---|---|---|---|---|
| **1** | **Saúde & fitness feminino: "corpo em transformação" (GLP-1 + força/glúteo + pilates em casa)** | `mounjaro` **+61%** a/a e 6,9× o volume de `skincare`; `caneta emagrecedora` **+275%**; `treino de glúteo` **+30%**; `hip thrust` **+157%**; `app de treino` **+20%**; `pilates` +6% (recorde da série em 2025–26); `sarcopenia` +54% | `caneta emagrecedora` **+34%** vs. 3 m anteriores (pico 13/09/2026); `ozivy` +26%; `pilates` **+11%** (pico 20/09/2026, "pilates na parede" +400%); `treino de glúteo` **+15%**; `treino em casa` +12%; `app de treino` +8% (+32% dentro da janela) | Health & Fitness lidera **todas** as métricas: trial→pago 37,7%, download→pago 2,9%, RPI D14 US$ 0,48, RLTV Y1 US$ 35,64, renovação ativa anual 86,4% | **Construir aqui** |
| **2** | **Saúde da mulher 40+ (perimenopausa/menopausa)** | `perimenopausa` **+184%** a/a (5 anos: +1.472%); `menopausa` +6% e 2,3× `skincare` (volume alto e estável); `reposição hormonal` +20%; rising: `veoza` +2.250%, "anvisa aprova medicamento para menopausa" +1.250% | `perimenopausa` **+28%** vs. 3 m anteriores (pico ago/2026); `menopausa` −5%, mas "dia mundial da menopausa" +4.250% e "sintomas da menopausa aos 48 anos" +2.850% | Mesma categoria RevenueCat (Health & Fitness inclui Medical) | **Construir aqui** (combina com o #1: mesma mulher, mesmo motor) |
| 3 | Beleza/skincare | `skincare` +56% a/a (5 anos +134%), mas `análise de pele` = 0 e `cronograma capilar` −22% | `skincare` **−19%** (sazonal: pico dez/2025); `maquiagem` +5% (Halloween) | Vira Photo & Video / Lifestyle: trial→pago 22,2% (pior), renovação anual 23–25% | Alto interesse, **baixo encaixe** com app pago (rising = marcas de produto, não apps) |
| 4 | Produtividade/rotina feminina | `rotina` +4% a/a (+26% dentro da janela); `planner` sazonal | `planner` **+37%** ("planner 2027" +800%); "apps para organizar rotina" **breakout**; `rotina` +9% | Productivity: 77% mensal, renovação anual **23%** (pior), reembolso 4,7% (maior), 41% já são apps de IA | Médio; IA de borda pouco diferenciadora |
| 5 | Finanças pessoais | `app de finanças` +411% (base ínfima); `controle de gastos` −8%; `planilha de gastos` −34% | "app para controle de gastos pessoais **grátis**" +250%; `organizar finanças` −25% | Utilities (inclui Finance): renovação anual 35%, download→trial 6,5% | Volume baixo (0,09× `skincare`) e intenção "grátis" |
| 6 | Saúde mental/sono | `sono` 8,4× `skincare` mas flat (+3%); `ansiedade` −2%; `meditação` **−18%** (5 anos −53%); `journaling` micro-pico em jul/2026 | `ansiedade` 0%; `meditação` −7%; `journaling` −29% | Health & Fitness (boa), mas demanda estagnada e incumbentes globais | Maduro/declinante |
| 7 | Saúde reprodutiva/ciclo | `ciclo menstrual` −12% a/a (5 anos −28%); `gravidez` **−19%** (5 anos −43%); `SOP` +6%; `app menstruação` +25% | `ciclo` −1%, `gravidez` −6%, `SOP` −14%; rising "monitor de fertilidade" +350%, "calculadora fertilidade feminina" breakout, "melhor app ciclo menstrual **grátis**" breakout | Health & Fitness | Estrutural em queda (natalidade) + Flo domina + intenção "grátis" |
| 8 | Maternidade | `bebê` +13% a/a (5 anos −20%); `amamentação` −16% (5 anos −34%) | `bebê` −7%; `amamentação` −5% | Sem categoria própria (Lifestyle/Health) | Declínio estrutural, LTV curto |
| 9 | Moda/estilo | `provador virtual` +16% (puxado por "zara provador virtual" +700%); `coloração pessoal` **−39%** (5 anos −68%); rising "prompt coloração pessoal chatgpt" +400% | `look do dia` −15%; `coloração pessoal` −10% | Lifestyle/Photo & Video (fracas) | ChatGPT já canibaliza a análise de coloração |
| 10 | Relacionamentos | `relacionamento` +6%; `tinder` −11% (5 anos −47%); `namoro` 5 anos −44% | tudo negativo (−1% a −17%) | Social & Lifestyle: **piores renovações** (anual 25%, mensal 42%), 43,6% sem trial | Evitar |
| 11 | Astrologia/espiritualidade | `signo` −14% (5 anos −55%), `mapa astral` −9% (5 anos −61%), `horóscopo` 5 anos −70%; `tarot` +16% | `tarot` −14%, `mapa astral` −11%; picos só em dez/jan ("mapa astral 2026 gratuito" +1.900%) | Lifestyle | Declínio + "gratuito" |
| 12 | Educação/carreira | `inglês` −10% (5 anos −18%); `currículo` −17% | tudo negativo; rising "curso de inglês online gratuito com certificado" | Education: preço anual mais alto (US$ 44,99) mas renovação anual 24% e reembolso 4,86% (2025) | Fora do perfil |
| 13 | Segurança pessoal | `medida protetiva` +40% (notícias de celebridades); `app de segurança para mulheres` = 0 | `medida protetiva` +33% (idem) | — | Sem intenção de app |

### 1.2 Recomendação em uma frase

**Construir um app nativo iOS-first de rotina diária de força + proteína para mulheres em transformação corporal, lançado para quem usa caneta GLP-1 (o tema feminino que mais cresce no Brasil em 12 e 3 meses), com treino de glúteo/pilates em casa guiado por câmera on-device, estimativa de proteína por foto on-device e um "modo 40+" (perimenopausa) na v1.5.** Dieta por foto e coach de execução, as ideias originais, entram como *recursos* desse app, não como produto: `contador de calorias` caiu 45% a/a (−19% nos últimos 3 m) e os termos de "execução correta" têm volume quase nulo.

---

## 2. Metodologia

### 2.1 Google Trends

- **Como foi puxado:** script Python (`docs/dados/trends/trends.py`) contra a API não documentada do Trends: (1) `GET https://trends.google.com/trends/?geo=BR` para cookie; (2) `GET /trends/api/explore?hl=pt-BR&tz=180&req={"comparisonItem":[{"keyword":k,"geo":"BR","time":janela}...],"category":0,"property":""}`; (3) com os `token`/`request` de cada widget, `GET /trends/api/widgetdata/multiline` (interesse ao longo do tempo) e `/widgetdata/relatedsearches` (consultas relacionadas *top* e *rising*). Prefixo `)]}'` removido antes do parse. Pausas aleatórias de 2,5–4,5 s entre chamadas; 1 único HTTP 429 em ~800 chamadas, resolvido com retry.
- **Parâmetros:** `hl=pt-BR`, `tz=180` (UTC−3), `geo=BR` (ou vazio = mundo), `category=0`, `property=""` (busca web). Janelas: `today 5-y` (semanal, 261 pontos), `today 12-m` (semanal, 53 pontos) e `today 3-m` (diário, 92 pontos). Último ponto parcial descartado.
- **Grupos de comparação:** até 5 termos por consulta (limite do Trends; todos ficam na mesma escala 0–100, em que 100 = pico do termo mais buscado do grupo no período). Para comparar setores **entre grupos**, incluí o termo-âncora `skincare` em todas as comparações cruzadas (`X1`–`X8`) e expressei cada termo como múltiplo da média de `skincare` no mesmo grupo/janela (tabela 3.1). Isso introduz erro de arredondamento (valores baixos ficam em 0–1 na escala) — termos com média < 1 devem ser lidos como "ordem de grandeza", não como medida precisa.
- **Métricas derivadas (`analyze.py`):** "12 m vs. 12 m anteriores" = média das últimas 52 semanas / 52 anteriores (série de 5 anos); "5 anos" = últimas 52 semanas / primeiras 52; "3 m vs. 3 m anteriores" = média das últimas 13 semanas / 13 anteriores (série de 12 meses); "dentro da janela de 3 m" = 2ª metade / 1ª metade da série diária. Pico = data do valor máximo.
- **Rising queries:** lista "Em ascensão" do Trends; "Aumento repentino" (*breakout*) significa crescimento > 5.000% ou termo novo. Descartei manualmente ruído óbvio (letras de música em `postura`, notícias de celebridades em `medida protetiva`, questões escolares em `rotina`).
- **Limitações:** índice relativo, não volume absoluto; sem recorte de gênero; termos de nicho aparecem como 0 quando comparados a termos de massa (ex.: `contar calorias` vs. `mounjaro`) — por isso há consultas "solo" e grupos temáticos para cada ideia.
- **Arquivos:** `<grupo>__<BR|WORLD>__<janela>.csv` (colunas: data, termos, isPartial), `<grupo>__...__related.json` (top/rising por termo), `_summary.json` (métricas), `_coleta_log.txt`.

### 2.2 RevenueCat

- Baixei o HTML público de `https://www.revenuecat.com/state-of-subscription-apps` (SOSA 2026; 115 mil+ apps, US$ 16 bi em receita, >1 bi de transações, período-alvo 2025) e `/state-of-subscription-apps-2025`, extraí o texto (sem scripts) e li as seções por nome. Também usei os posts do blog "Average subscription renewal rates by app category (2026 benchmarks)", "How long should your free trial be? Data from 17,000+ apps" e "5 quick fixes for bad default global pricing on app stores".
- Categorias da RevenueCat são "baldes": **Health & Fitness inclui Medical**; **Utilities inclui Finance**; **Social & Lifestyle inclui Lifestyle, Social Networking e Dating**; **Productivity inclui Graphics & Design**; **Photo & Video inclui Photography**. "Latin America" = Brasil, México, Argentina, Chile, Colômbia, Peru.
- Uma inconsistência do próprio relatório: a seção "Which plan durations do subscribers choose by category?" diz que Health & Fitness "lidera adoção anual com 68%", enquanto "Where do weekly vs. yearly plans dominate" diz "Health & Fitness is monthly-heavy (68%)". A seção de receita é inequívoca: **59% da receita de H&F vem de planos anuais**.

---

## 3. Google Trends: o que as mulheres (proxy) estão buscando

### 3.1 Volume relativo entre setores (Brasil, âncora `skincare` = 1,00)

Médias do índice em cada janela, encadeadas pelos grupos `X1`–`X8`. Leia como "quantas vezes o volume de `skincare`".

| Termo (setor) | 5 anos | 12 meses | 3 meses | Direção 12 m vs. anteriores | Direção 3 m vs. anteriores |
|---|---|---|---|---|---|
| inglês (educação) | 64,3 | 39,0 | 37,0 | −10% | −5% |
| bebê (maternidade) | 31,3 | 19,9 | 18,9 | +13% (5 a: −20%) | −7% |
| gravidez (saúde fem.) | 28,8 | 13,5 | 12,7 | −19% (5 a: −43%) | −6% |
| sono (mental) | 12,7 | 8,4 | 9,1 | +3% | 0% |
| **mounjaro (GLP-1)** | 3,6 | **6,9** | 5,8 | **+61%** | −12% (pico jan/2026) |
| ansiedade (mental) | 12,1 | 6,8 | 7,1 | −2% (5 a: −24%) | 0% |
| maquiagem (beleza) | 10,7 | 6,0 | 6,9 | +1% (5 a: −33%) | +5% |
| relacionamento | 6,4 | 3,8 | 4,0 | +6% (5 a: −22%) | −1% |
| tarot (astro) | 4,7 | 3,2 | 3,4 | +16% | −14% |
| **pilates (fitness)** | 3,6 | 3,0 | 3,4 | +6% (5 a: **+55%**) | **+11%** |
| **menopausa** | 3,0 | 2,3 | 2,4 | +6% (5 a: **+34%**) | −5% |
| rotina (produtividade) | 2,7 | 1,9 | 2,1 | +4% (5 a: +17%) | +9% |
| planner | 2,2 | 1,35 | 1,6 | −5% | **+37%** |
| amamentação | 2,1 | 1,1 | 1,0 | −16% (5 a: −34%) | −5% |
| **skincare (beleza)** | **1,00** | **1,00** | **1,00** | **+56%** (5 a: +134%) | **−19%** |
| meditação | 1,5 | 0,64 | 0,60 | −18% (5 a: −53%) | −7% |
| mapa astral | 1,6 | 0,60 | 0,59 | −9% (5 a: −61%) | −11% |
| **caneta emagrecedora** | — | 0,47 | **0,75** | **+275%** | **+34%** |
| ciclo menstrual | 0,70 | 0,37 | 0,39 | −12% (5 a: −28%) | −1% |
| cronograma capilar | 0,90 | 0,35 | 0,30 | −22% (5 a: −56%) | −20% |
| treino feminino | 0,43 | 0,27 | 0,26 | −5% | −6% |
| **perimenopausa** | — | 0,25 | **0,35** | **+184%** (5 a: +1.472%) | **+28%** |
| journaling | — | 0,15 | 0,29 | pico isolado jul/2026 | −29% |
| controle de gastos (finanças) | 0,17 | 0,09 | 0,07 | −8% | −3% |
| app de treino | 0,06 | 0,05 | 0,02 | **+20%** (5 a: +104%) | +8% |
| treino de glúteo | — | 0,04 | 0,02 | **+30%** (5 a: +90%) | **+15%** |
| coloração pessoal (moda) | 0,10 | 0,03 | 0,00 | −39% (5 a: −68%) | −10% |
| contador de calorias (dieta) | 0,04 | 0,02 | 0,00 | **−45%** (5 a: −23%) | −19% |
| app menstruação | 0,03 | 0,02 | 0,00 | +25% | 0% |
| provador virtual / segurança pessoal / análise de pele | ~0 | ~0 | ~0 | — | — |

Fontes: `X1`–`X8__BR__*.csv`, `*__BR__today_5_y.csv`, `_summary.json`. Direções vêm dos grupos temáticos (seções 3.2–3.5), onde a escala é mais fina.

**Intenção explícita de app, mesma escala (grupo `apps_fem_12m`, BR, 12 m):** `app de treino` média **48,7** (+8% no trimestre) · `app menstruação` 16,1 (−7%) · `app de finanças` 8,1 (−1%) · `app de skincare` 5,5 (zerou nos últimos 3 m) · `app de meditação` ≈ 0. Ou seja, entre as mulheres (proxy), a busca ativa por *aplicativo* se concentra em treino — 3× ciclo, 6× finanças, 9× skincare.

**Leitura:** os termos de grande volume (inglês, bebê, gravidez, sono, ansiedade, maquiagem) são estáveis ou em queda e pertencem a setores de app com monetização fraca ou dominados por incumbentes. Os únicos temas femininos com volume relevante **e** crescimento forte em 12 m e em 3 m são **GLP-1/caneta**, **pilates/glúteo/treino em casa** e **perimenopausa**. `skincare` cresce no ano mas cai no trimestre e não gera intenção de app.

### 3.2 Setor 1 — Saúde & fitness feminino (grupos `glp1`, `musculo_proteina`, `fitness_fem`, `fem_12m`, `execucao`, `modalidades`, `apps`, `dieta`)

| Termo | 5 anos | 12 m vs. ant. | 3 m vs. ant. | Pico | Rising queries relevantes (janela) |
|---|---|---|---|---|---|
| mounjaro | novo (0 → 65) | **+61%** | −12% | jan/2026 | 12 m: `ozivy` breakout, `poviztra` +4.100%, `combo mounjaro` +3.450%, `mounjaro pancreatite aguda` +1.500%. 3 m: "mounjaro 5mg emagrece quantos quilos em 1 mês" +250%, **"pode tomar mounjaro amamentando" +60%, "mounjaro corta o efeito do anticoncepcional" +60%**, "mounjaro sus" +50% |
| caneta emagrecedora | novo | **+275%** | **+34%** | 13/09/2026 | breakout: `ozivy`, `retatrutida`, `mantecorp caneta emagrecedora`, `semavy`, "caneta emagrecedora lula" +3.200%, `poviztra` +2.200% |
| ozivy (genérico EMS) | novo (jun/2026) | — | +26% | 14/06/2026 | "ozivy preço", "ozivy 1mg", "semaglutida ozivy" (todos breakout) |
| ozempic | −2% | −50% | −20% | mar/2024 | migração para `mounjaro` ("mounjaro ou ozempic" breakout) |
| dieta mounjaro / o que comer tomando mounjaro | — | — | −13% / −12% | jan/2026 | "dieta para quem toma mounjaro" +700% (dentro de `dieta`) |
| perda de massa muscular | — | — | +12% | pico 21/12/2025 | — |
| sarcopenia | **+200%** | **+54%** | — | dez/2025 | — |
| massa magra / flacidez | −13% / −9% | +14% / +12% | — | — | — |
| proteína | — | +21% (dentro de 12 m) | +5% | mar/2026 | "quantos gramas de proteína por kg hipertrofia" **+2.750%** |
| creatina / whey protein | +30% / +4% | −20% / −3% | — / −3% | dez/2024 | marcas (Dark Lab, Growth) |
| **pilates** | **+55%** | +6% | **+11%** | 20/09/2026 (12 m) | 12 m: "pilates de parede funciona" +4.950%, "pilates na parede" +2.400%, "kit pilates em casa" +500%, `wellhub`; 3 m: **"pilates na parede" +400%**, "pilates em casa gratuito" +60%, "hot pilates" +40%, "professor de pilates" +250% |
| pilates em casa | — | — | **+35%** | jan/2026 | — |
| **treino de glúteo** | **+90%** | **+30%** | **+15%** | jan/2026 | "ponte de glúteo" breakout, "elevação pélvica" +180%, "cadeira abdutora" +160%, "treino de posterior e glúteo feminino" +120% |
| hip thrust | +1.196% | **+157%** | — | ago/2026 | "hip thrust máquina" +1.350%, "afundo" +700%, "dead bug" +550% |
| treino em casa | +32% | +8% | **+12%** | — | "treino funcional para fazer em casa" breakout, "elástico para treino em casa" +80% |
| treino abdominal | — | — | +13% | — | — |
| **app de treino** | **+104%** | **+20%** | +8% (+32% dentro da janela) | jan/2026 | "melhores app para treino de academia **grátis**" +2.000%, `hevy app` |
| treino feminino / treino para mulheres | +9% / ~0 | −5% | −6% | — | rising = tênis (Olympikus, Under Armour): termo de **produto**, não de treino |
| academia / corrida | +27% / **+89%** | +3% / +11% | — | jan/2026 / abr/2026 | — |
| execução do exercício | +127% | +1% | — | set/2025 | nenhuma rising |
| agachamento correto | −36% | +129% (base ~0) | — | — | nenhuma rising |
| **contador de calorias** | −23% | **−45%** | **−19%** | jan/2025 | "contador de calorias online **grátis**" +1.750%, "melhor app contador de calorias **gratuito**" +1.200% |
| contar calorias (solo) | −15% | −30% | −21% | jan/2026 | "pare de contar calorias" +250% |
| calorias por foto | ~0 | — | — | — | nenhuma |
| cal ai | novo | +947% (base 0) | **−50%** | nov/2025 | "cal ai price" +250% |
| myfitnesspal / yazio | +41% / +225% | +29% / −12% | −65% / −18% | jul/2026 / jan/2025 | "yazio vs myfitnesspal", `cronometer` +1.300% |
| aplicativo de dieta / app contar calorias | −47% / −11% | −47% / −44% | — | — | "aplicativo de dieta para emagrecer **grátis**" +300% |
| postura (solo) | −3% | +4% | +1% | — | rising = letras de música; "corretor postural" −21% |
| assoalho pélvico (solo) | +21% | +14% | — | ago/2026 | "enfraquecimento do assoalho pélvico" +120%, "como fortalecer" +60% |
| diástase | +38% | +18% | — | jan/2026 | "qual o melhor exercício para fechar diástase" breakout, "quem tem diástase pode fazer prancha" +300% |
| pós-parto / exercício na gravidez | −8% / −94% | +9% / −67% | — | — | — |

**Mundo (grupos `world_*`, 5 anos):** `mounjaro` +3.365% (12 m +28%), `protein intake` **+613%** (12 m **+194%**, pico mai/2026; rising "protein intake per kg" +450%), `pilates` +167% (12 m +14%), `pelvic floor exercises` +122% (12 m +45%), `posture corrector` +111% (12 m +37%), `ai calorie counter` 12 m +186% (base ínfima; rising `cal ai`, `snapcalorie`, "photo calorie counter app **free**"), `cal ai` pico mar/2026 (aquisição) e `calorie counter` flat (+2%). `cycle syncing` e `cycle syncing workout` = 0 na escala de `pilates`; rising "cycle syncing meal plan" breakout.

**Síntese do setor:** a demanda migrou de **"contar calorias"** (em queda e com intenção "grátis") para **"caneta + proteína + não perder músculo + glúteo/pilates em casa"**. As rising queries de `mounjaro` nos últimos 3 meses são explicitamente femininas (amamentação, anticoncepcional). `app de treino` é o único termo de "app" do estudo com crescimento consistente em 5 anos, 12 m e 3 m.

### 3.3 Setor 2 — Saúde da mulher 40+ (grupo `menopausa`, `solo_perimenopausa`, `S_saude_fem`)

| Termo | 5 anos | 12 m vs. ant. | 3 m vs. ant. | Pico | Rising |
|---|---|---|---|---|---|
| menopausa (solo) | **+34%** | +6% | −5% | mar/2025 | 5 a: `neovadiol menopausa` +2.150%, `vichy menopausa` +1.100%, "menopausa cancelada" +700%, "como eu sei que estou entrando na menopausa" +450%, `utrogestan` +350%, "peri menopausa" +300%. 12 m: `veoza` +2.250%, "anvisa aprova medicamento para menopausa" +1.250%, "qual hormônio é reduzido…" +4.700%. 3 m: "dia mundial da menopausa" +4.250%, "sintomas da menopausa aos 48 anos" +2.850% |
| **perimenopausa (solo)** | **+1.472%** | **+184%** | **+28–30%** | ago/2026 | `progesterona` +80%, `estradiol` +60% |
| climatério | +41% | +1% | +2% | set/2025 | "climatério sintomas físicos" +800%, "climatério sintomas iniciais" +400%, `sobrac` +60% |
| reposição hormonal | +70% | +20% | +1% | ago/2025 | "qual o melhor medicamento para reposição hormonal feminina", `vagifem` |
| fogachos | +31% | +14% | −28% | jan/2024 | "o que é fogachos" +130% |
| endometriose / SOP | +4% / +71% | +4% / +6% | +2% / −14% | — | "endometriose cerebral" +350%; `SOP` sem rising |

**Mundo:** `perimenopause` 5 a **+322%**, 12 m +28%, pico mai/2026 (rising "what are the 34 symptoms of perimenopause" +1.200%, `magnesium glycinate` +850%, "hrt for perimenopause" +250%); `menopause` 5 a +74%, 12 m +27% (rising "frozen shoulder menopause" +850%, "genitourinary syndrome of menopause" +550%); `menopause app` 12 m +289% (base ínfima).

**Síntese:** `menopausa` já é um termo de massa estável (2,3× `skincare`) e `perimenopausa` é o termo de saúde feminina que mais cresce no Brasil nas duas janelas. As rising queries mostram uma mulher **procurando entender sintomas iniciais (40–48 anos)** e sendo educada por marcas (Vichy/Neovadiol) e por notícias de medicamentos (Veoza/ANVISA) — não há nenhuma rising query de "app", o que indica **espaço vazio**, não ausência de demanda.

### 3.4 Setores de interesse alto mas encaixe fraco

**Beleza (`S_beleza`, `pele`):** `skincare` 5 a +134%, 12 m +56%, mas 3 m **−19%** (pico 28/12/2025 — sazonal); `maquiagem` 5 a −33%; `cronograma capilar` 5 a −56%; `análise de pele` e `rotina de skincare` ≈ 0; `tipo de pele` +24%. Rising (12 m e 3 m): `medicube`, `celimax`, `principia`, `kokeshi`, "kit skincare completo", "kit skincare masculino" — **100% produtos/marcas**. Mundo: `skincare` 3 m −32%. → Interesse enorme, mas em **comprar produto**, não em pagar app.

**Moda (`S_moda`):** `coloração pessoal` 5 a −68%, 12 m −39%; rising "prompt para análise de coloração pessoal" (breakout) e "prompt coloração pessoal chatgpt" +400%, "teste de coloração pessoal online **grátis**" +250% → o LLM genérico já substitui o app. `provador virtual` +16% (puxado por "zara provador virtual" +700% — recurso de varejista). `armário cápsula` ≈ 0.

**Saúde mental (`S_mental`):** `sono` e `ansiedade` são termos de massa estáveis (8,4× e 6,8× `skincare`), `meditação` −18% (5 a −53%), `journaling` praticamente inexistente em PT-BR (pico isolado em jul/2026; rising em inglês). Curiosidade útil: "mounjaro dá sono" +180% dentro de `sono` — mais um sinal da onda GLP-1.

**Produtividade/rotina (`S_produtividade`):** `rotina` +17% em 5 a, +26% dentro de 12 m, +9% no trimestre; "apps para organizar rotina" **breakout** (3 m); `planner` +37% (sazonal: "planner 2027" +800%); `hábitos` −8% (rising = livro "365 hábitos poderosos"). Demanda real, porém pequena (1,9× `skincare`) e com monetização fraca (seção 4).

**Finanças (`S_financas`):** `controle de gastos` 0,09× `skincare`; `app de finanças` +411% a/a sobre base ínfima; rising "app para controle de gastos pessoais **grátis**" +250%, "planilha … **grátis**". Intenção de pagar baixa.

**Saúde reprodutiva/ciclo (`S_saude_fem`, `S_saude_fem2`, `ciclo`):** `gravidez` 5 a −43%, `engravidar` −23%, `ovulação` −34%, `ciclo menstrual` −28%; `app menstruação` +25% a/a mas rising "app menstruação **grátis**" +160%, "melhor app ciclo menstrual **grátis**" breakout, "app ciclo menstrual gratuito" +700%. "fases do ciclo menstrual e humor" +1.500% e "as 4 fases do ciclo menstrual" +900% mostram curiosidade sobre fases, mas `treinar menstruada` e `cycle syncing` têm volume ≈ 0 no Brasil.

**Relacionamentos, astrologia, educação, maternidade, segurança:** todos com tendência de 5 anos negativa (−20% a −70%) ou sem intenção de app (`app de segurança para mulheres` = 0). Detalhes nos CSVs `S_*`.

### 3.5 O que os últimos 3 meses dizem (ago–nov/2026) — ranking de momentum

Ordenado por crescimento vs. os 3 meses anteriores, considerando só termos com volume mínimo e rising queries com intenção real:

1. **Caneta emagrecedora +34%** (e `ozivy` +26%; `mounjaro` ainda 5,8× `skincare`): genérico brasileiro, promessa política ("caneta emagrecedora lula"), novas marcas (Poviztra, TG, Semavy, retatrutida) e perguntas femininas (amamentação, anticoncepcional).
2. **Perimenopausa +28%** (pico 09/08/2026) + Dia Mundial da Menopausa (18/10) gerando "sintomas da menopausa aos 48 anos" +2.850%.
3. **Pilates +11%** (pico 20/09/2026; "pilates na parede" +400%, "pilates em casa gratuito" +60%), **pilates em casa +35%**, **treino de glúteo +15%**, treino abdominal +13%, treino em casa +12%, app de treino +8%.
4. **Planner +37% / rotina +9%** ("apps para organizar rotina" breakout) — sazonal de fim de ano.
5. Finanças: "app para controle de gastos pessoais grátis" +250% — intenção gratuita.
6. Beleza: `skincare` −19%, `maquiagem` +5% (Halloween).
7. Saúde reprodutiva: `gravidez` −6%, `SOP` −14%; "monitor de fertilidade" +350%.
8. Mental, moda, astro, relacionamentos, educação, maternidade: estáveis ou negativos.

---

## 4. RevenueCat: monetização por categoria (SOSA 2026, com 2025 onde indicado)

### 4.1 Comparativo entre categorias

| Métrica (mediana, salvo indicação) | Health & Fitness (incl. Medical) | Productivity | Education | Photo & Video | Social & Lifestyle (incl. Dating) | Utilities (incl. Finance) | Business | Todas |
|---|---|---|---|---|---|---|---|---|
| Download → trial (D30) | **6,9%** (top: >23%) | — | 6,5% | — | — | 6,5% | 9,1% | ~5–6% |
| Trial → pago | **37,7%** (Q3 >51,4%) | — | — | **22,2%** (pior) | — | — | — | Travel 43,5% é a maior |
| Download → pago (D35) | **2,9%** (Q3 >6,2%) | — | "lag" (2025) | — | — | — | 2,6% | Gaming 1,0% |
| RPI D14 / D60 | **US$ 0,48 / 0,66** | — | 0,30 / — | — | — | — | 0,31 / 0,50 | 0,23 / 0,34 |
| RLTV por pagante mês 1 / ano 1 | **US$ 24,23 / 35,64** (top >60) | — / 24,95 | — / 22,82 | — | — | — | 18,76 / 35,48 | Gaming 8,41 / 11,22 |
| Preço mediano mensal / anual | **US$ 9,99 / 39,94** (semanal 4,99) | — | 9,99 / **44,99** | — | 9,99 / — | — | semanal 6,89 | 9,99 / 34,80 |
| Mix de assinaturas | anual 68% (seção de planos) | **77% mensal** | anual 59–66% | semanal 33% nos paywalls | semanal 47% (maior) | — | — | 42% mensal, 34% anual |
| Share da receita por plano | **59% anual** | 91% mensal | anual majoritário | — | — | — | — | — |
| 1ª renovação anual (Q1 / med / Q3) | 16% / **25%** / 37% | 14% / 23% / 38% | 17% / 24% / 32% | 15% / 23% / 35% | 13% / 25% / 41% | 21% / 35% / 52% | 21% / 40% / 53% | 23–40% |
| 1ª renovação mensal | 46% / **57%** / 68% | 40% / 54% / 66% | 48% / 56% / 66% | 35% / 48% / 59% | 27% / **42%** / 61% (pior) | 43% / 57% / 70% | 49% / 61% / 74% | 42–61% |
| Renovação ativa anual | **86,4%** (líder) | 85,6% (mensal 28,2%, pior) | — | — | — | — | — | 83,4% |
| Retenção 6 m (semanal / mensal) | — | — | — | — | 11% / 24% (pior) | — | 26% / 40% | 3–6% / 14–26% |
| Reembolso | 3–4% (2025: 4,71%) | **4,7%** (maior) | 4,86% (2025) | alto | — | — | 2,7–2,9% | 3–4% |
| Estratégia de trial | 59% mista; só 18,3% sem trial; **54% trials de 5–9 dias** | — | 50% de 5–9 d | 68% ≤4 d | **43,6% sem trial** | — | — | 46,5% ≤4 d |
| Share de apps de IA | baixo | **41,1%** | — | **61,4%** | — | — | 19,1% | 27,1% |
| Tempo até US$ 1k MRR / % que chega a US$ 10k | >100 dias p/ US$ 10k | — | — | 21,4% chegam a 1k, **7,3% a 10k** (maior) | 45 dias p/ 1k | — | 113 dias p/ 1k | 4,6% chegam a 10k em 2 anos |
| Churn "custo" / "pouco uso" (Google Play) | — | **44% custo** | — | — | — | — | — | 25–45% / 26–40% |

Fontes: SOSA 2026 — seções "How does D30 download-to-trial conversion vary by app category?", "How does trial-to-paid conversion vary by app category?", "Which categories best convert downloads into paid users?", "What's the D14 revenue benchmark?", "How does revenue grow from D14 to D60?", "What's monthly revenue per paying customer?", "What's yearly revenue per payer?", "What do apps charge by category?", "Which plan durations drive revenue by category?", "How do active renewal rates compare by category and plan duration?", "What are typical refund rates by category?", "What trial strategy do different app categories favour?", "What category footprint do AI apps have?", "How long does it take apps to reach monthly revenue milestones?", "Why do users cancel subscriptions?"; blog "Average subscription renewal rates by app category (2026)"; SOSA 2025 seções "Refund rate by app category", "Average revenue per install".

**Leitura:** Health & Fitness é a única categoria que lidera simultaneamente conversão, receita por instalação, valor por pagante e renovação anual — e é onde caem os setores #1 e #2 do ranking do Trends (fitness, GLP-1, menopausa, pelve). Produtividade (setor "rotina") tem a pior renovação anual, o maior reembolso e já é 41% IA. Photo & Video (onde cairia um app de análise de pele/coloração) tem a pior conversão trial→pago e é 61% IA. Social & Lifestyle (relacionamentos, astrologia, lifestyle) tem as piores renovações.

### 4.2 Benchmarks transversais que definem o desenho do produto

| Decisão | Dado RevenueCat 2026 | Implicação |
|---|---|---|
| Hard paywall vs. freemium | D35 **10,7% vs. 2,1%** (5×); top 10% de hard paywall 38,7%; RPI D14 US$ 2,32 vs. 0,27 (9×); retenção Y1 igual (27% vs. 28%); reembolso igual (2,5% vs. 2,9%) | Paywall logo após o onboarding; freemium só se precisar de boca a boca |
| Duração do trial | trial→pago: ≤4 d 25,5%; 5–9 d 37,4%; 10–16 d 35,4%; 17–32 d 42,5%. H&F prefere 5–9 d (54%). Anual: conversão 24% → 44,6% e 1ª renovação 18,3% → 47,5% ao ir de ≤4 d para 17–32 d (blog) | **7 dias** na v1; testar 14 dias para o plano anual |
| Plano | Y1 retenção: anual 20–40%, mensal 6–14%, semanal 1–2%. Renovação ativa anual 83,4% (H&F 86,4%). H&F: 59% da receita anual | Anual em destaque, mensal como alternativa, **sem semanal** |
| Preço | H&F US$ 9,99/mês e ~US$ 39,99/ano; anual mediano geral subiu de US$ 31,60 → 34,80. Baixo preço retém melhor no anual (36% vs. 23% alto) mas RLTV Y1 alto preço US$ 62,19 vs. 10,69; reembolso 2,7% → 4,5% | Faixa média: R$ 24,90–29,90/mês, R$ 149–179/ano no Brasil; US$ 9,99 / 49,99–59,99 fora |
| Apps de IA | +52% trial start (8,5% vs. 5,6%), +20% download→pago (2,4% vs. 2,0%), RLTV Y1 US$ 30,16 vs. 21,37 (+41%), **mas** retenção 12 m anual 21,1% vs. 30,7%, mensal 6,1% vs. 9,5%; reembolso 4,2% vs. 3,5%; 60% das assinaturas de IA são mensais | Vender a **rotina**, não "a IA"; IA on-device evita custo marginal por usuária (RevenueCat alerta para custo de LLM em apps de IA) |
| iOS vs. Android | D35 iOS 2,6% vs. Play 0,9% (2,9×); H&F iOS 3,5%; trial→pago praticamente igual (32,6% vs. 32,5%); erros de cobrança = 32,2% dos cancelamentos no Play vs. 15,2% na App Store | iOS primeiro; no Android, tratar billing errors (grace period) |
| Framework | D35: React Native 2,5%, Native 2,0%, Flutter 1,8%; RPI D60 RN US$ 0,51 / Native 0,31 / Flutter 0,29; variação dentro de cada stack 10–30× maior que entre stacks ("the framework doesn't matter — until it does") | Sem penalidade por nativo; escolha pelo produto (IA on-device = nativo) |
| Primeiro mês | 35% dos cancelamentos anuais acontecem no mês 1; D0 concentra 50,6% das conversões; 82,1% dos trials de H&F começam no D0 | Onboarding → valor em minutos; lembrete de renovação honesto |
| Ofertas | só 9,3% dos apps usam promoções (H&F 14%); desconto mediano −50%; apps "hobby" dependem 65–99% de intro offers, top performers 0–10% | Intro offer só como experimento, não como muleta |
| Metas realistas | mediana de receita 1 ano após lançamento ≈ US$ 72/mês; top quartil US$ 429; 4,6% chegam a US$ 10k MRR em 2 anos; H&F leva >100 dias para US$ 10k | Planejar 12–18 meses de runway |

### 4.3 América Latina vs. América do Norte

| Métrica | América Latina | América do Norte | Fonte |
|---|---|---|---|
| Download → pago D35 | **1,5%** | 2,6% (P90 10,4%) | 2026, "D35 conversion by developer HQ" |
| Download → trial | 3,0–3,7% | 7,1% (Q3 15%) | 2026 |
| Trial → pago | ~20–23% (2025: 25,0%, top 53,3%) | 34,2% (Q3 47,9%) | 2026/2025 |
| RPI D14 / D60 | ~US$ 0,10 / "mais perto da Europa que de IN/SEA" (2025: 0,06–0,09) | US$ 0,38 / 0,55 | 2026/2025 |
| RLTV Y1 por pagante | abaixo da mediana global (US$ 23) | US$ 32 | 2026 |
| Preços | **abaixo de MEA em todas as durações** ("unusual given GDP") | mais altos: US$ 9,99 mensal, 39,99 anual | 2026 |
| Mix de planos | **29% semanal (maior do mundo)** | 40% anual (maior) | 2026 |
| Crescimento MRR mediano | **+17,2% (maior do mundo)** | — | 2026 |
| Reembolsos | outliers na casa dos 30%; P25–P75 2–8% | 3,4% (mais previsível) | 2026 |
| 1ª renovação semanal | 44% | 55% | 2026 |
| Conversões tardias | mais conversões na semana 6+ | — | 2025 |
| Caso | ancorar o anual como "só R$ X/mês" subiu trial start +30% e take rate anual +10% sem mudar preço (Latin America) | — | 2026, ensaio "How do monetization strategies differ across geographies?" |
| Caso | app de meditação dobrou assinaturas na LatAm em 14 dias com preço regional; "em Brasil você pode estar cobrando 5× mais que o concorrente local" | — | blog "5 quick fixes…" |

**Implicação:** no Brasil, esperar ~metade da conversão dos EUA e ~1/4 da receita por instalação; compensar com preço regional, plano anual ancorado em "R$/mês", paywall em PT-BR, e **vender também em EUA/Portugal** desde o dia 1 (mesmo binário, preço por loja).

---

## 5. Ideias de app nos setores vencedores, com IA de borda como diferencial real

Escala 1–5 (5 = melhor). **Demanda** e **Momentum 3 m** vêm do Trends; **Monetização** da RevenueCat; **IA de borda** = quanto a IA no aparelho é diferencial defensável; **Facilidade** = viabilidade para dev solo em 6–10 semanas. Concorrência vem da pesquisa secundária (seção 7) e pesa menos.

| Ideia | Setor | Demanda 12 m | Momentum 3 m | Monetização | IA de borda | Facilidade | **Geral** |
|---|---|---|---|---|---|---|---|
| **C. Força + proteína para mulheres em uso de caneta GLP-1** (rotina diária; foto→proteína on-device; contador de repetições; diário de efeitos) | 1 | 5 (`mounjaro` +61%, `caneta` +275%, `sarcopenia` +54%) | 5 (`caneta` +34%, `ozivy` +26%, perguntas femininas) | 5 (H&F) | 4 | 5 | **4,8** |
| **I. Pilates em casa / glúteo com câmera on-device** (pilates na parede, elevação pélvica; alinhamento e contagem de repetições sem vídeo sair do aparelho) | 1 | 4 (`pilates` recorde, `glúteo` +30%, `app de treino` +20%) | 5 (`pilates` +11%, `pilates em casa` +35%, `glúteo` +15%) | 5 (H&F) | 4 | 3 | **4,3** |
| **E. Rotina para perimenopausa/menopausa** (sintomas × sono × treino × proteína com padrões detectados no aparelho) | 2 | 4 (`perimenopausa` +184%, `menopausa` estável alto) | 4 (+28%) | 5 (H&F) | 3 | 4 | **4,1** |
| **F'. Assoalho pélvico + diástase** (Kegel guiado, programa pós-parto; sem hardware) | 1/2 | 3 (`assoalho pélvico` +14%, `diástase` +18%) | 3 | 4 | 2 | 4 | 3,3 |
| **A. Dieta por foto (calorias/macros)** | 1 | 2 (`contador de calorias` −45%; rising "grátis") | 1 (−19%; `cal ai` −50%) | 4 (H&F, mas apps de IA churnam 30% mais) | 3 (Foundation Models iOS 27 roda imagem on-device) | 3 | 2,7 |
| **J. Rotina/hábitos feminina com planner inteligente** | 4 | 3 (`rotina` +26% na janela; "apps para organizar rotina" breakout) | 4 (`planner` +37%) | 2 (Productivity: renovação anual 23%, reembolso 4,7%, 41% IA) | 2 | 5 | 3,0 |
| **D. Treino sincronizado ao ciclo** | 7 | 2 (`ciclo` −12%; `cycle syncing` ≈ 0 no BR) | 2 | 4 | 3 (dados do ciclo só no aparelho) | 4 | 2,9 |
| **L. Fertilidade/ciclo privacy-first** | 7 | 2 (`gravidez` −19%; "monitor de fertilidade" +350%) | 3 | 4 | 3 | 4 | 2,9 |
| **B. Coach de execução (ângulos articulares, "forma correta")** | 1 | 1 (`execução do exercício` flat; `agachamento correto` ≈ 0) | 1 | 3 | 4 | 1 | 2,0 |
| **G. Análise de pele por câmera** | 3 | 1 (`análise de pele` = 0; `skincare` −19% no trimestre) | 1 | 1 (Photo & Video: trial→pago 22%) | 3 | 3 | 1,8 |
| **M. Coloração pessoal / provador virtual** | 9 | 1 (`coloração pessoal` −39%; ChatGPT canibaliza) | 2 (pico isolado) | 1 | 3 | 3 | 1,8 |
| **H. Postura/glúteo por foto (avaliação)** | 1 | 1 (`postura` flat; `corretor postural` −21%) | 1 | 3 | 3 | 4 | 2,0 |
| **K. Finanças pessoais femininas** | 5 | 1 (0,09× `skincare`; "grátis") | 2 | 3 (Utilities: renovação anual 35%) | 1 | 4 | 2,1 |

**Por que C e I lideram:** são as únicas ideias em que **demanda em alta nas duas janelas**, **melhor categoria de monetização** e **IA de borda com função clara** coincidem. Em C a IA de borda resolve privacidade (medicação e peso nunca saem do aparelho — argumento forte após o veredito Flo × Meta [secundário]) e custo marginal zero (RevenueCat alerta que apps de IA com LLM em nuvem têm custo por assinante). Em I a IA de borda resolve o que mata apps de "wall pilates" genéricos (conteúdo passivo): a câmera conta repetições, tempo sob tensão e alinhamento básico em exercícios de solo/parede — poses simples, câmera fixa, sem barra ocluindo o corpo, ao contrário do agachamento com barra da ideia B.

**Por que A e B caem:** o Trends mostra o termo de dieta mais buscado caindo 45% a/a e 19% no trimestre, com rising queries dominadas por "grátis"; `cal ai` caiu 50% no trimestre no Brasil; e a RevenueCat mostra que apps de IA convertem melhor mas retêm pior (anual 21,1% vs. 30,7%). Para B, não existe demanda mensurável por "execução correta" em PT-BR, e os termos de forma estão zerados.

---

## 6. Recomendação final e MVP

### 6.1 O produto

**Nome de trabalho:** "Rotina de força e proteína para mulheres em transformação" — v1 posicionada para quem usa caneta (GLP-1), com treino de glúteo/pilates em casa e proteína por foto; v1.5 adiciona o "modo 40+" (perimenopausa).

**Promessa:** "Perca gordura, não músculo — 25 minutos por dia, sem academia, com seus dados só no seu celular."

**Por que juntar C + I (+E):** a mesma usuária aparece nas três rising lists — "pode tomar mounjaro amamentando", "mounjaro corta o efeito do anticoncepcional", "pilates na parede", "elevação pélvica", "quantos gramas de proteína por kg", "sintomas da menopausa aos 48 anos". É uma mulher de 30–55 anos mudando o corpo e procurando rotina em casa. Um único motor (rotina diária + proteína + treino em casa + registro de sintomas) atende às três com *modos* diferentes.

### 6.2 Escopo da v1 (6–10 semanas, iOS nativo)

| Semana | Entrega | Dado que sustenta |
|---|---|---|
| 1–2 | Onboarding em PT-BR (peso, altura, objetivo, nível, "usa caneta? qual dia aplica?" opcional, fase da vida) → meta de proteína 1,2–1,6 g/kg [secundário: consenso 2025] e plano de 3 treinos/semana. **Hard paywall** com trial de **7 dias**, plano **anual ancorado em "R$ 14/mês"** + mensal R$ 29,90. SwiftData criptografado, sem conta obrigatória. | RevenueCat: hard paywall 5×; H&F 54% trials 5–9 d; LatAm +30% trial start com âncora mensal; 59% da receita de H&F é anual |
| 3–4 | **Registro de proteína:** base TACO/TBCA embarcada, favoritos, código de barras (Vision + Open Food Facts), **foto → itens + porção aproximada com Foundation Models (iOS 27, on-device)**, sempre com faixa de incerteza e correção em 1 toque; fallback: busca manual (ou Gemini Flash opt-in). | Trends: "quantos gramas de proteína por kg" +2.750%; RevenueCat: custo marginal de IA em nuvem pressiona margem |
| 5–6 | **Treino em casa guiado por câmera:** 2 programas (halteres/elástico; pilates de solo e parede) com 20–30 exercícios em vídeo curto; **Vision 2D (`VNDetectHumanBodyPoseRequest`) conta repetições, tempo sob tensão e amplitude** em 6–8 exercícios de solo (elevação pélvica, ponte, afundo, agachamento livre, prancha, dead bug, wall pilates). Vídeo nunca é gravado — só keypoints. | Trends: `pilates na parede` +400%, `elevação pélvica` +180%, `ponte de glúteo` breakout, `treino funcional para fazer em casa` breakout |
| 7 | **Diário de bem-estar (modo caneta):** náusea, apetite, constipação, energia, sono, hidratação, dia da aplicação (sem recomendar dose); gráficos semanais on-device; lembretes gentis. | Trends: "efeitos colaterais mounjaro" é o termo-satélite mais buscado (55 de média); "mounjaro dá sono" +180% |
| 8 | Apple Health (peso, passos, sono), widgets/Live Activity "proteína de hoje", exportar/apagar dados (LGPD), tela "Fontes" e disclaimers (Apple 1.4.1). | RevenueCat: 35% dos cancelamentos anuais no mês 1 → valor visível no dia 0 |
| 9–10 | Beta TestFlight com 30–50 mulheres de grupos de "caneta" e de pilates em casa; ajustar paywall; publicar em BR + EUA + PT. | RevenueCat: D0 = 50% das conversões; traduzir paywall vale +37% LTV [relatório 2026, via resumo — verificar] |

**v1.5 (semanas 11–16):** modo 40+ (fogachos, sono, humor, padrões detectados on-device entre sintomas × treino × proteína), conteúdo SOBRAC-based, integração Apple Watch. **v2:** Android nativo (Kotlin + ML Kit Pose + ML Kit GenAI/Gemini Nano onde houver), B2B2C com farmácias/telemedicina.

**Fora da v1:** calorias totais como foco (Trends em queda), "forma perfeita" com ângulos (sem demanda, alto risco de falso positivo), coloração/pele (encaixe fraco), comunidade, chat com IA.

### 6.3 Métricas-alvo (benchmarks RevenueCat 2026, mediana → top quartil de Health & Fitness)

| Métrica | Mínimo aceitável (mediana H&F) | Meta (top quartil) |
|---|---|---|
| Download → trial (D30) | 6,9% | >15% (NA Q3) / H&F top >23% |
| Trial → pago | 37,7% | >51,4% |
| Download → pago D35 (global; Brasil esperar ~metade) | 2,9% | >6,2% |
| RPI D60 | US$ 0,66 | >US$ 1,39 (NA Q3) |
| 1ª renovação anual / mensal | 25% / 57% | 37% / 68% |
| Retenção Y1 do plano anual | 28% | 32–59% |
| Reembolso | <4% | <2,7% |
| Receita 12 meses após lançamento | US$ 72/mês (mediana geral) | >US$ 429/mês (top quartil) |

### 6.4 Preço e posicionamento

- **Brasil:** R$ 29,90/mês · **R$ 169,90/ano** (mostrar "R$ 14,16/mês"); trial 7 dias; sem semanal (LatAm já é 29% semanal e o semanal retém 1–2% em Y1).
- **EUA/Europa/Portugal:** US$ 9,99/mês · US$ 59,99/ano (H&F medianas: US$ 9,99 / 39,94; anual mediano geral subindo).
- **Canais (dos próprios rising queries):** grupos e creators de "caneta" (perguntas sobre amamentação/anticoncepcional mostram público feminino), creators de pilates na parede e glúteo, nutricionistas e fisios pélvicas (código de indicação), SEO em PT-BR para "o que comer tomando mounjaro", "quantos gramas de proteína por kg", "pilates na parede funciona", "exercício para fechar diástase".
- **Não prometer:** emagrecimento, tratamento, dose, "forma perfeita", diagnóstico.

### 6.5 Stack

| Camada | Escolha | Justificativa |
|---|---|---|
| Plataforma | **iOS nativo (Swift 6 / SwiftUI / SwiftData)**, iOS 17+, recursos de IA em iOS 26/27 | RevenueCat: iOS converte 2,9× o Play e H&F iOS 3,5%; sem penalidade de monetização para nativo (D35 2,0% vs. RN 2,5%, dispersão interna 10–30× maior); Foundation Models (imagem+texto on-device, iOS 27), Vision, HealthKit, StoreKit 2 são nativos |
| Pose/repetições | Vision 2D (19 pontos); 3D só se precisar de ângulo sagital | Exercícios de solo/parede com câmera fixa são o caso fácil |
| Foto → alimento | Foundation Models `@Generable` → lookup TACO/TBCA local; fallback opcional Gemini 2.5 Flash via backend mínimo | Zero custo marginal na maioria dos iPhones compatíveis; RevenueCat alerta sobre custo de IA em nuvem |
| Pagamentos | StoreKit 2 + RevenueCat (paywalls remotos, A/B, métricas) | Benchmarks desta pesquisa vêm da mesma base |
| Backend/analytics | CloudKit opcional; TelemetryDeck/App Analytics; **nenhum evento com peso, medicação ou sintomas** | LGPD (dado de saúde = sensível) e lição Flo × Meta [secundário] |
| Android (fase 2) | Kotlin nativo + ML Kit Pose + ML Kit GenAI (Gemini Nano) | Plugins Flutter/RN de MediaPipe ainda 0.x e Android-first [secundário]; Play tem 32% de cancelamentos por erro de cobrança → implementar grace period |

---

## 7. Evidência secundária (fora de Trends/RevenueCat) — marcada e resumida

Mantida apenas como contexto; nenhuma decisão acima depende dela.

- **[secundário] Mercado GLP-1 no Brasil:** R$ 13,3 bi em Mounjaro+Wegovy+Ozempic (mai/2025–mai/2026, Close-Up via ICTQ); Mounjaro de 208 mil → 4,5 M unidades/mês (+2.070%); Ozivy (EMS) lançado 15/06/2026; UBS projeta ~R$ 20 bi em 2026 (InvestNews). Coerente com o Trends (`caneta emagrecedora` pico set/2026, `ozivy` jun/2026).
- **[secundário] Perda de massa magra em GLP-1:** 26–40% do peso perdido (SURMOUNT-1/STEP 1, via resumos); ENDO 2025 sugere perda desproporcional em mulheres; consenso 2025 de 4 sociedades: 1,2–1,6 g proteína/kg + treino de força.
- **[secundário] Concorrência dieta por foto:** Cal AI (15 M+ downloads, >US$ 30 M/ano) comprado pelo MyFitnessPal (anunciado 02/03/2026, TechCrunch); clones PT-BR a R$ 49,90/ano (Olho no Prato), R$ 119,90/ano (Vitaliz); estudo NIH (NUTRITION 2026, via Healio) — apps subestimam calorias em ~33%, Cal AI ~345 kcal. Coerente com a queda de `contador de calorias` e `cal ai` no Trends.
- **[secundário] Coach de execução:** Kemtai migrou para fisioterapia B2B; Zenia absorvida pela Zing (2024); Zing Vision criticado por falsos alertas; erro angular ~10,6° no melhor pipeline (preprint 2025).
- **[secundário] Menopausa:** mercado global US$ 17,8 bi (2024) → 24,4 bi (2030) (Grand View via Estado de Minas, set/2026); apps BR: Raquel Menopausa (grátis), Namu Mulher (B2B, nov/2026); Balance+ US$ 50–99/ano.
- **[secundário] Mulheres e pagamento:** Flo ~80 M MAU, ~5 M pagantes (2024), valuation >US$ 1 bi; Queima Diária (Smart Fit) 438 mil clientes no 4T25, R$ 39,90/mês; ABA: 55% das mensalidades de academia são de mulheres.
- **[secundário] Privacidade:** júri federal (Califórnia, 01/08/2025) condenou a Meta por coletar dados menstruais de usuárias do Flo via SDK; ORCHA: 84% dos apps de ciclo compartilham dados com terceiros.
- **[secundário] Regulatório:** ANVISA RDC 657/2022 exclui "softwares para bem-estar" (exercício, hábitos) da regulação de SaMD; Apple 1.4.1 exige fontes/disclaimers para recomendações de saúde; LGPD trata dados de saúde como sensíveis (art. 5º, II; art. 11).
- **[secundário] Edge AI:** Apple Foundation Models aceita imagem + texto on-device no iOS 27 (anúncio 08/06/2026, developer.apple.com); ML Kit GenAI image description via Gemini Nano em Pixel 9/10/11 e Galaxy S25/S26; Gemini 2.5 Flash US$ 0,30/M tokens de entrada (~US$ 0,05–0,30/usuária/mês a 5 fotos/dia, estimativa).
- **[secundário] Android no Brasil:** 75–79% Android vs. 21–25% iOS (StatCounter, jun–ago/2026).

---

## 8. Riscos e mitigação

| Risco | Evidência | Mitigação |
|---|---|---|
| Ciclo de hype do GLP-1 | `mounjaro` −44% do pico de jan/2026 aos últimos 3 m (mas `caneta` +34% e `ozivy` +26% — a onda muda de marca, não de tamanho) | Posicionar por **problema** (manter músculo, rotina em casa), não por remédio; modo 40+ como segunda perna |
| Apps de IA retêm pior | RevenueCat: retenção anual 21,1% vs. 30,7%; reembolso 4,2% vs. 3,5% | IA como recurso; a assinatura vende rotina + treino + registro; on-device evita custo por usuária |
| Conversão LatAm baixa | D35 1,5% vs. 2,6%; RPI ~US$ 0,10; preços abaixo de MEA | Preço regional, âncora mensal, vender em NA/EU desde o dia 1 |
| Semanal/churn no Brasil | LatAm 29% semanal; semanal retém 1–2% em Y1 | Não oferecer semanal |
| Intenção "grátis" em PT-BR | rising queries "app … grátis" em calorias, ciclo, treino, finanças | Hard paywall; comunicar valor premium (IA on-device, privacidade, PT-BR, comida brasileira) |
| Apple 1.4.1 / ANVISA / LGPD | [secundário] | Fontes e disclaimers; finalidade "bem-estar"; dados on-device; sem SDK de ads/analytics com saúde |
| Falsos positivos da câmera | [secundário] Zing/Kemtai | Só contagem, tempo e amplitude; nunca "sua forma está errada" |

---

## 9. Fontes primárias

**Google Trends (coleta própria, 08/11/2026):** 118 séries (CSV) e 62 arquivos de consultas relacionadas em `docs/dados/trends/` — grupos `ideias_cross`, `dieta`, `glp1`, `glp1_long`, `musculo_proteina`, `massa`, `menopausa`, `ciclo`, `execucao`, `fitness_fem`, `fem_12m`, `modalidades`, `gestacao_pelvico`, `pele`, `postura`, `apps`, `apps_fem_*`, `amplo`, `cross_12m`, `S_beleza`, `S_moda`, `S_saude_fem`, `S_saude_fem2`, `S_mental`, `S_financas`, `S_maternidade`, `S_relacionamento`, `S_seguranca`, `S_astro`, `S_produtividade`, `S_educacao`, `S_fitness`, `X1`–`X8`, `W1`–`W2`, `world_cross`, `world_apps`, `world_niche`, `solo_*`. Endpoint: `https://trends.google.com/trends/api/explore`, `/widgetdata/multiline`, `/widgetdata/relatedsearches`.

**RevenueCat:**
- State of Subscription Apps 2026 — https://www.revenuecat.com/state-of-subscription-apps (seções citadas na tabela 4.1 e 4.2; "Methodology", "Regional definitions", "Category bucketing", "Where are subscription apps growing fastest…", "What annual value does each payer generate?", "What's the D35 download-to-paid conversion rate by developer HQ geography?", "How do monetization strategies differ across geographies?", "Which geographies generate the highest D14 revenue?", "How does D60 revenue vary by geography?", "Does trial length affect trial-to-paid conversion rate?", "Does access method impact download-to-paid conversion within 35 days?", "Does freemium vs. hard paywall affect Y1 retention?", "Do pricing tiers affect Y1 retention?", "How does subscription duration mix vary across geographies?", "Median price per plan duration by geography", "How do renewal rates vary by geography and subscription duration?", "How much variance exists in geographic refund rates?", "Why do subscriptions end on each platform?", "When do annual subscribers cancel during their first year?", "How do D35 conversion rates compare between categories and stores?", "Does development framework affect download-to-paid conversion?", "How does D60 RPI compare by framework?", "Do AI apps…" (todas), "How widely are promotional offers used across categories?", "What share of apps hit monthly revenue milestones?").
- State of Subscription Apps 2025 — https://www.revenuecat.com/state-of-subscription-apps-2025 ("Trial to paid, by category/by geography", "Average revenue per install", "Realized LTV per payer", "Refund rate by app category", "Active Renewal Rate, By Category").
- Blog: https://www.revenuecat.com/blog/growth/average-subscription-renewal-rates-by-app-category · https://www.revenuecat.com/blog/growth/free-trial-length · https://www.revenuecat.com/blog/growth/5-quick-fixes-for-bad-default-global-pricing-on-app-stores

**Secundárias (seção 7):** TechCrunch (02/03/2026, Cal AI), Healio (04/08/2026, estudo NIH), ICTQ/Close-Up, InvestNews, NeoFeed/IQVIA, Estado de Minas (set/2026), Smart Fit Resultados 4T25, TechCrunch/HIPAA Journal (Flo × Meta, ago/2025), ANVISA (Perguntas & Respostas RDC 657/2022), Apple Developer ("What's new in iOS"), Android Developers (ML Kit GenAI), StatCounter, Grand View Research, ClassPass Look Back 2025, PRWeb (Zing/Zenia), Mindbowser (Kemtai), arXiv 2512.06783.
