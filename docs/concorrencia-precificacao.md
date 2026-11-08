# Concorrência e precificação — app de rotina para mulheres em uso de caneta GLP-1 (América do Sul + EUA)

> **Nota (08/11/2026):** as decisões finais de produto, nome (Nutrelle), posicionamento e preço estão em [`produto.md`](produto.md) e prevalecem sobre as recomendações deste relatório.

**Data da coleta:** 08/11/2026 (16:40–17:40 UTC)
**Produto:** app para mulheres 30–55 em transformação corporal; público inicial = usuárias de caneta GLP-1 (Mounjaro/Ozempic/Ozivy). Promessa: "perca gordura, não músculo", dados ficam no celular. v1: plano diário em uma tela (meta de proteína + check-in), proteína por foto on-device (TACO, faixa de incerteza, código de barras, favoritos), diário GLP-1 (dia da aplicação, náusea, apetite, intestino, energia, sono, água; correlações on-device; sem orientação de dose), progresso (Apple Health, fotos privadas, streaks, widget), privacidade como marca (sem conta, exportar/apagar). v1.5: modo perimenopausa. **Sem treinos guiados por câmera.**

**Fontes primárias desta pesquisa (dados brutos em `docs/dados/concorrencia/`):**
- iTunes Search API (`itunes.apple.com/search`, 39 termos × 4 storefronts → `itunes_search_results.csv`, 823 linhas).
- Páginas da App Store por storefront (`apps.apple.com/<cc>/app/id…`): 60 apps; 32 em **us, br, mx, ar, cl, co, pe, pt** e 28 em us+br → `appstore_storefronts.csv` (preços locais de cada compra no app, nota, nº de avaliações, rótulos de privacidade) e `compiled_prices.json`.
- Feed RSS de avaliações (`itunes.apple.com/<cc>/rss/customerreviews`, 50 mais recentes por app em us/br/mx) → `appstore_reviews.json` (≈2.300 avaliações, classificadas por tema em `compiled_prices_summary.txt`).
- Google Play (`play.google.com/store/apps/details`, gl=BR/US/MX/AR): 47 pacotes → `play_storefronts.csv` (faixa de instalações, nota, nº de avaliações, faixa de preço das compras no app).
- Google Trends (API `explore`/`multiline`, mesmo termo comparado entre países, 12 m e 3 m) → `trends_geo__*.csv`, `trends_geo_summary.csv`.
- RevenueCat *State of Subscription Apps 2026/2025* (texto extraído em 08/11/2026, já usado em `pesquisa-mercado.md`).
- Tudo que vem de fora dessas fontes está marcado **[secundário]**.

**Câmbio:** não consegui cotação PTAX do dia. Para converter, usei a **razão implícita da própria App Store** — mediana da relação preço local ÷ preço nos EUA do *mesmo* plano do *mesmo* app (63 pares BR, 54 MX, 46 CO, 51 PE, 56 PT): **BR 4,28 R$/US$ · MX 17,0 · CO 4.444 · PE 3,90 · PT 1,11 €/US$**. Isso mede o que os desenvolvedores **de fato cobram** em cada país relativo aos EUA (BR e MX ~15–20% abaixo do câmbio; PT acima por IVA). Para referência, USD/BRL estava em R$ 5,01 (07/10/2026, Suno) e R$ 5,22 (28/09/2026, Renova) **[secundário]**. **Argentina e Chile têm App Store em dólar** (preços observados em US$).

---

## 1. Resumo executivo

- **A categoria "companion de GLP-1" já existe e está lotada no Brasil** — e é brasileira: **Monju** (lançado mai/2026, 23.357 avaliações iOS BR, 100 mil+ instalações no Play, nota 4,9) e **OzemPro** (out/2025, 18.137 avaliações iOS BR, 100 mil+ Play) superam o líder americano **Shotsy** no Brasil (2.547 iOS BR; 100 mil+ Play). Há mais 10 apps BR com 100–2.500 avaliações (MeuGLP, GlipOne, MounjaPRO, Mingo, MounHealth, Canetinha…). Nos EUA lideram **MeAgain** (40.259) e **Shotsy** (34.646).
- **Preço de mercado da categoria:** EUA **US$ 9,99/mês e US$ 39,99/ano** (Shotsy, Glapp, OzemPro, Pep); Brasil **R$ 59,90/mês e R$ 99,90–199,90/ano** (Monju, OzemPro, GlipOne, Glapp); MX **MX$ 199–249/mês, MX$ 399–899/ano**; CO **COP 49.900/mês, 89.900–199.900/ano**; CL/AR (em US$) **US$ 9,99/mês, 22,99–49,99/ano**; PE **S/ 44,90/mês, 89,90–179,90/ano**; PT **€9,99–14,99/mês, €34,99–49,99/ano**.
- **As 3 maiores lacunas (vêm das 2.300 avaliações recentes):**
  1. **Privacidade/sem conta/dados que não somem.** Nenhum concorrente de GLP-1 tem rótulo "Dados não coletados" (só o WebDiet, app de nutricionista); OzemPro, Monju, MeAgain, Pep, GLP AI e Glapp declaram "dados vinculados" e/ou "rastreamento". Reclamações recorrentes de 1★: "all my data is gone", "zera no dia da aplicação", loops de login (Shotsy, GLP AI, Glippy, MeuGLP, Dietbox, OzemPro). **Clearblue Menopause** virou 34/50 avaliações 1★ só por exigir conta ("data mining").
  2. **Registro de comida fraco e sem foco em proteína/comida brasileira.** "Food tracking leaves a lot to be desired" (Mingo), "nutrition data not accurate" (Pep), "falta muitos dados de comidas básicas" (MeuGLP), "registro de comida nunca es exacto" (OzemPro MX), 13 de 32 avaliações ≤3★ do Cal AI são sobre precisão, MacroFactor e Cronometer **não têm português**, MyFitnessPal cobra pelo código de barras, Alimente-se dá meta de 136 g de proteína a uma mulher (reclamação). Nenhum app do cluster é "proteína primeiro".
  3. **Cobrança honesta.** O tema nº 1 das avaliações negativas de apps BR/MX é preço/cobrança: "trial cobrou na hora" (Shotsy US$ 65, Pep US$ 40, Sweat, Foodvisor, Lila), "promoção 9,99 e debitou 119,90" (BitePal), cupom que não desconta (OzemPro), "tudo só funciona se pagar" (Monju, GlipOne, Fitia, Protein Pal), lifetime R$ 1.999,90 ao remover o anual (Lose It!). Nenhum concorrente de GLP-1 é **feminino**: o único app "peri + emagrecimento" (Lila) tem 17/43 avaliações recentes 1★ por paywall/trial, e "Her Pep" tem 0 avaliações.
- **Preços recomendados (resumo):** **Brasil R$ 29,90/mês · R$ 149,90/ano** (≈ R$ 12,49/mês) · **EUA US$ 9,99 · US$ 49,99** · **México MX$ 149 · MX$ 699** · **Chile US$ 7,99 · US$ 34,99** · **Colômbia COP 29.900 · COP 129.900** · **Peru S/ 24,90 · S/ 99,90** · **Portugal €7,99 · €39,99** · **Argentina US$ 5,99 · US$ 24,99** (último). Trial de 7 dias, hard paywall após o onboarding, **sem plano semanal, sem lifetime no lançamento**.
- **Ordem de entrada:** 1) Brasil (iOS) → 2) Portugal + EUA no mesmo binário (PT-BR cobre PT; EN) → 3) México (exige UI em espanhol; `mounjaro` MX +61% no trimestre) → 4) Chile e Colômbia → 5) Peru → 6) Argentina.

---

## 2. Interesse por país (Google Trends, mesmo termo comparado entre países)

Índice relativo: 100 = pico do país mais interessado no período; valores são **comparáveis entre países** (normalizados pelo total de buscas de cada país). Arquivos: `trends_geo__*.csv`.

| Termo | Janela | BR | MX | CL | CO | PE | AR | US | PT | ES |
|---|---|---|---|---|---|---|---|---|---|---|
| **mounjaro** | 12 m | **65,6** | 24,6 | 9,6 | 11,7 | 6,1 | 4,1 | 14,2 | 30,1 | 17,5 |
| mounjaro | 3 m | **73,7** | **39,6** | **18,3** | 12,6 | 5,5 | 4,6 | 15,8 | 22,8 | 17,6 |
| ozempic | 12 m | 27,7–30,7 | 27,6 | **53,4** | 24,5 | 13,2 | 23,0 | **53,8** | 34,8 | 36,9 |
| ozempic | 3 m | 9,9–12,2 | 12,7 | 28,6 | 12,1 | 4,6 | 11,6 | 20,3 | 14,1 | 23,4 |
| glp-1 | 12 m | 6,7 | 8,6 | — | 7,1 | — | 6,0 | **64,6** | — | — |
| proteína | 12 m | 73,2 | **78,2** | 63,2 | 56,1 | — | 40,0 | — | — | — |
| menopausia / menopausa | 12 m | (BR: ver pesquisa-mercado) | 77,3 | **79,5** | 63,3 | 55,4 | 59,1 | — | — | — |
| perimenopausia / perimenopausa | 12 m | 50,8 | 59,2 | **67,7** | 37,2 | — | 47,1 | — | 57,2 | 65,6 |

**Leitura:**
- O Brasil é, de longe, o país onde "mounjaro" mais pesa nas buscas (2,7× o México, 4,6× os EUA) e continua subindo no trimestre. **México** é o segundo mercado e o que mais acelera (24,6 → 39,6); **Chile** dobrou (9,6 → 18,3). Argentina e Peru têm interesse baixo.
- "Ozempic" é o termo dominante em **Chile, EUA, Espanha e Portugal** e está em queda em toda parte (migração para tirzepatida/genéricos). Nos países hispânicos, a marca é o termo de busca; "glp-1" como palavra só funciona nos EUA (64,6 vs. 6–9 nos demais) → **ASO em espanhol deve usar "Mounjaro/Ozempic", não "GLP-1"**.
- "Proteína" é forte em todos os países hispânicos (MX 78, CL 63, CO 56) — a proposta "proteína primeiro" viaja bem.
- Perimenopausa (v1.5) é mais forte no Chile, Espanha e México que no Brasil.
- **Todos os mercados hispânicos exigem UI em espanhol** (ver reclamações: "está en inglés y no entiendo" — SnapCalorie MX; "lo ponen en brasileño" — Tecnonutri MX, 9/14 avaliações recentes 1★).

---

## 3. Matriz de concorrentes

Legenda: **Mulheres?** = posicionamento explícito para mulheres; **GLP-1?** = recursos de caneta (dose/dia da aplicação, efeitos colaterais); **Privacidade** = rótulos da App Store US ("Vinculados" = dados vinculados à usuária; "Rastreio" = usados para rastrear); **n** = nº de avaliações iOS no storefront. Recursos marcados (inf.) são inferidos do título/subtítulo da listagem e das avaliações.

### 3.1 Companions de GLP-1

| App (empresa, país, lançamento) | Posicionamento | Mulheres? | GLP-1? | Privacidade (US) | iOS US (nota/n) | iOS BR | iOS MX | Play BR (instal./avaliações) | Principais reclamações (avaliações recentes) |
|---|---|---|---|---|---|---|---|---|---|
| **Shotsy** (Shotsy Co., EUA, mai/2024) | Rastreador de doses/locais de aplicação, peso, efeitos; localizado PT/ES | Não | Sim | Nenhum rótulo "vinculado/rastreio" detectado, mas exige login | 4,8 / 34.646 | 4,9 / 2.547 | 4,8 / 613 | 100 mil+ / 13,2 mil (4,9) | US: 7/50 1★ — "perdi todos os dados 4 vezes", "trial cobrou US$ 65 na hora"; BR: 48/50 5★ |
| **MeAgain** (Dots Future, EUA, abr/2025) | Tracker + comida + IA; vende "Unlimited" (vitalício em tiers) | Não | Sim | Vinculados + Rastreio | 4,8 / 40.259 | 4,5 / 19 | 4,9 / 108 | 50 mil+ / — | "navegação bagunçada", "queria orientação específica de proteína nos dias de aplicação" |
| **Glapp** (Glapp Inc., mar/2026) | Tracker "inteligente"; semanal/mensal/anual | Não | Sim | Vinculados | 4,8 / 1.075 | 4,6 / 14 | 4,8 / 19 | 10 mil+ / 494 | "recursos grátis viraram pagos", não abre no iOS 27 |
| **Pep AI** (Zode) | Tracker + IA | Não | Sim | Vinculados + Rastreio | 4,7 / 4.017 | — / 3 | — / 6 | — | **32/50 recentes 1★**: "removeram recursos, adicionaram IA que ninguém pediu, bugs" |
| **Pep** (Shred Apps) | Tracker + "Pep Strength" (app de treino separado, 13 aval.) | Não | Sim | Vinculados | 4,7 / 967 | 4,7 / 32 | 4,9 / 18 | — | "trial de 7 dias anunciado, cobrou US$ 40 na hora", "peso não importa do Apple Health", scan de comida fraco |
| **Zepbound & Mounjaro Tracker** (Manhattan App Studio) | Tracker simples; **semanal US$ 13,99** | Não | Sim | Nenhum | 4,8 / 3.399 | 4,9 / 83 | 4,9 / 49 | — | 46/50 recentes 5★ |
| **GLP AI** (OutliveAI, dez/2025) | Tracker + dieta por IA; mensal até US$ 99,99 | Não | Sim | Vinculados | 4,7 / 830 | 4,9 / 969 | 4,9 / 468 | 10 mil+ / 3 mil | "deleta dados", "só funciona pagando", trava (MX) |
| **OzemPro** (FITCAL LTDA, **Brasil**, out/2025) | "Emagrecer com GLP-1": doses, dieta, água; em PT/ES/EN | Não | Sim | Vinculados + Rastreio | 4,8 / 459 | **4,8 / 18.137** | 4,9 / 527 | **100 mil+ / 21,2 mil (4,9)** | **BR: 25/50 recentes 1★** — cupom que não desconta, "zera no dia da aplicação", preço; MX: trava, lento |
| **Monju** (MeuGuru/MG Soluções, **Brasil**, mai/2026) | "Emagrecer com GLP-1"; semanal/mensal/anual; o que mais cresce | Não | Sim | Vinculados + Rastreio | 4,9 / 230 | **4,9 / 23.357** | 4,9 / 1.090 | **100 mil+ / 20,9 mil (4,9)** | 9/50 1★: "propaganda enganosa", "tudo só funciona se pagar", anúncios; MX: "muy cara, deberían permitir prueba" |
| **MeuGLP** (BI Treinamentos, **Brasil**, fev/2026) | Tracker + dieta | Não | Sim | Vinculados + Rastreio | — | 4,9 / 2.486 | — | — | **16/47 1★**: sem suporte, "falta muitos dados de comidas básicas", não abre após update |
| **GlipOne** (Genio Tech, fev/2026) | "Meu tratamento GLP-1"; semanal/mensal/anual | Não | Sim | Rastreio | 5,0 / 5 | 4,9 / 1.089 | — | — | 6/17 1★: reembolso negado, não edita aplicação, não suporta dose quinzenal, "valor alto vs. benefício" |
| **MounjaPRO** (W12, dez/2025) | Tracker barato (semanal R$ 2,90) | Não | Sim | Nenhum | 4,6 / 13 | 4,8 / 1.060 | — | 10+ / — | — |
| **Mingo** (Agora Labs, **Brasil**, jan/2026) | Tracker; vende vitalício | Não | Sim | Rastreio | 4,7 / 68 | 4,9 / 214 | 4,7 / 468 | 10 mil+ / 1,04 mil | MX: "ofrecen mensual y cobran el año", "50 USD mensuales"; US: comida sem Apple Health |
| MounHealth, Canetinha/DosePen, DreamMe (pet), Glippy, Weightly, GLProgress, GLP1 Health Control (R$ 19,99 único; 50 mil+ Play), Dosie, Remi | Trackers simples | Não | Sim | — | ≤3,4 mil | ≤171 | — | 1–50 mil+ | DreamMe: "versão grátis corta após 2 itens"; Glippy: "um ano de registros sumiu" |
| **Lilly Health** (Eli Lilly, EUA) | Oficial Zepbound/Mounjaro; grátis | Não | Sim | Vinculados | 4,8 / 5.363 | — | — | — | 19/50 recentes 1★: trava, não funciona no iPad |
| **Healthi** (EUA) | Weight loss com programa GLP-1 | Não | Parcial | Vinculados + Rastreio | 4,8 / 89.968 | 5,0 / 9 | — | — | **25/50 1★**: busca ruim, "pede cartão fora da Apple", bugs |
| Found, Mochi (telemedicina EUA) | Prescrição + app | Não | Sim | Vinculados | 4,7 / 8.878; 4,9 / 18.186 | — | — | — | Found: **41/50 1★** (taxas ocultas) |

### 3.2 Comida por foto / macros / dieta

| App | Posicionamento | Mulheres? | GLP-1? | Privacidade (US) | iOS US | iOS BR | iOS MX | Play BR | Reclamações recentes |
|---|---|---|---|---|---|---|---|---|---|
| **Cal AI** (Viral Dev./MyFitnessPal) | Calorias por foto; vende "Unlimited" | Não | Não | Rastreio | 4,8 / 374.535 | 4,9 / 6.053 | 4,8 / 28.158 | 100 mil+ / 6,25 mil (4,2) | **US 32/50 ≤3★**: precisão (13), paywall (10), bugs; MX 21/50 1★: "roba dinero", "cobran por recuperar racha" |
| **MyFitnessPal** | Padrão da categoria; anúncios | Não | Não | Vinculados + Rastreio | 4,7 / 2,39 M | 4,9 / 99.104 | 4,8 / 32.460 | 100 mi+ / 2,92 mi | precisão (16/32), anúncios, código de barras pago; BR: "cobra por fora do Gympass" |
| **Lose It!** | Calorias; removeu anual no BR | Não | Não | Vinculados + Rastreio | 4,8 / 783.867 | 4,8 / 1.932 | 4,8 / 1.641 | 10 mi+ / 185 mil | BR: "só lifetime R$ 1.999,90", anúncios na versão paga, crash |
| **Yazio** | Calorias + jejum; anúncios | Não | Não | Vinculados + Rastreio | 4,7 / 51.726 | 4,8 / 70.977 | 4,8 / 18.258 | 50 mi+ / 880 mil | **BR 33/50 ≤3★**: formulário enorme e paywall antes de entrar, precisão |
| **MacroFactor** | Macros adaptativos; sem grátis | Não | Não | Vinculados | 4,8 / 23.891 | 4,8 / 105 | 4,8 / 256 | 1 mi+ / 17,7 mil (4,2) | "não tem português" (BR), pagamento falha no MX, bugs |
| **Fitia** (Nutrition Technologies, **Peru**) | Líder hispânico de calorias | Não | Não | Vinculados + Rastreio | 4,9 / 24.139 | 4,8 / 200 | **4,9 / 72.842** (AR 21,7k, CL 32k, CO 34k, PE 17k) | 10 mi+ / 506 mil | precisão ("200 kcal virou 800"), "todo premium" |
| **FatSecret** | Calorias barato; sem anúncios no iOS | Não | Não | Vinculados | 4,7 / 15.479 | **4,9 / 22.562** | 4,8 / 4.253 | 50 mi+ / 549 mil | poucas (8/50): falta busca em refeições salvas, linguagem neutra |
| Lifesum, Fastic, Simple, Foodvisor, SnapCalorie, BitePal, Noom, Cronometer, MyNetDiary | Globais | Não | Não | Vinculados + Rastreio (maioria) | — | Fastic 21,8k; Lifesum 14,3k; BitePal 17k; Foodvisor 11,4k; Simple 6,9k | — | Lifesum BR 33/50 ≤3★ (bugs); Fastic US 27/50 1★ (anúncios); **Simple BR 25/50 1★ e MX 16/50 1★ ("fraude", cobrança)**; Foodvisor BR 14/50 1★ (cobrança); BitePal "promo 9,99 debitou 119,90"; Noom BR 33/50 1★ (preço); Cronometer "queremos em português" |
| **Tecnonutri** (Vitat, **Brasil**) | Dietas prontas | Não | Não | Vinculados + Rastreio | 4,7 / 266 | 4,6 / 26.031 (Play **3,4** / 91,5 mil) | 3,8 / 41 | 10 mi+ | **BR 33/50 ≤3★**: busca quebrada, termos não abrem; MX: "en brasileño, roba datos" |
| **WebDiet** / **Dietbox** / **Alimente-se** (BR) | Apps de nutricionista / dieta | Não | Não | **WebDiet: "Dados não coletados"** | — | 202 mil / 71 mil / 5,9 mil | — | 1 mi+ / 1 mi+ / 5 mi+ | WebDiet 33/50 ≤3★ (IA de macros imprecisa); Dietbox 39/50 (login toda hora); Alimente-se: meta 136 g de proteína p/ mulher, fecha sozinho |
| **Protein Pal** (Ten Labs) / Protein Tracker (Oelfke) | Só proteína | Não | Não | Nenhum | 4,7 / 10.882; 4,8 / 1.833 | 17; 26 | — | — | Protein Pal **19/50 1★**: "NOT FREE — não registra nada sem assinar" |

### 3.3 Apps focados em mulheres

| App | Posicionamento | GLP-1? | Privacidade (US) | iOS US | iOS BR | Play BR | Reclamações recentes |
|---|---|---|---|---|---|---|---|
| **Flo** | Ciclo/gravidez; cycle-syncing no Premium | Não | Vinculados + Rastreio | 4,7 / 2,02 M | 4,9 / 912.065 (MX 451 mil) | 100 mi+ / 5,18 mi | paywall agressivo, anúncios, redesign "infantilizado", "não cobre pós-parto" (MX) |
| **Balance** (UK) | Menopausa; grátis + balance+ | Não | Vinculados | 4,7 / 1.162 | 5,0 / 11 | **100 mil+ / 7,27 mil** | não integra Apple Health, calendário confuso após update |
| **Lila – Peri Weight Loss Coach** (Tidalflow) | **Único "peri + emagrecimento"**; vitalício | Não | Vinculados + Rastreio | 4,7 / 1.730 | 5,0 / 3 | — | **17/43 1★**: trial virou cobrança anual, "check-in inútil", IA repetitiva |
| **Raquel Menopausa** (BR) | Menopausa, **grátis sem IAP** | Não | Vinculados | 5,0 / 21 | 5,0 / 408 | 50 mil+ / 1,15 mil | 50/50 5★ (sem reclamações; sem monetização) |
| Stardust, Clue, Wild.AI, Caria, Health & Her, Clearblue Menopause, Menelle | Ciclo/menopausa | Não | Vinculados (maioria) | Wild.AI 4,3 (25/50 1★: quebrou após venda); Clearblue **34/50 1★** (conta obrigatória = "data mining"); Caria erros 500/503 | — | — | — |
| **Sweat**, Workout for Women, **Perda de Peso para Mulheres** (Leap) | Treino feminino | Não | Vinculados + Rastreio | Sweat 4,6 / 43.342 | Sweat 874; Leap 747 | — | Sweat BR: "teste grátis é falso, cobra na hora"; Leap: "poderia incluir nutrição e pilates" |

**Linha única sobre treino sem câmera:** nenhum companion de GLP-1 relevante inclui treino (Pep lançou um app separado com 13 avaliações); treino é *table stakes* só nos apps de emagrecimento feminino (Leap, Sweat). Para a v1 não é obrigatório; uma rotina simples de força em texto/vídeo (sem câmera) é um bom candidato para v1.5 junto com o modo perimenopausa.

---

## 4. Tabelas de preço por país (compras no app observadas na App Store em 08/11/2026)

Valores = menor e maior preço de cada tipo de plano listado na ficha (os apps testam vários tiers ao mesmo tempo). "US$ equiv." usa a razão implícita da App Store (seção 0); AR e CL já estão em US$.

### 4.1 Brasil (R$) — equiv. US$ a 4,28

| App | Semanal | Mensal | Anual | Vitalício/outro | Anual em US$ equiv. |
|---|---|---|---|---|---|
| Monju | 39,90 | 59,90 | 99,90 · 129,90 · 149,90 · 199,90 | — | 23–47 |
| OzemPro | — | 59,90–79,90 | 99,90–249,90 | — | 23–58 |
| Shotsy | — | 25,90 · 49,90 · 79,90 · 129,90 | 127,90 · 139,00 | — | 30–32 |
| GlipOne | 19,90–39,90 | 49,90–69,90 | 199,90–299,90 | — | 47–70 |
| MeuGLP | — | 39,90 (?) | 149,90 | — | 35 |
| MounjaPRO | 2,90 | 14,90–39,90 | 79,90–99,90 | 299,90 vital. | 19–23 |
| Glapp | 29,90 | 59,90 | 249,90 | — | 58 |
| GLP AI | — | 99,90–459,90 | 179,90–334,90 | 59,90–499,90 | 42–78 |
| Mingo | — | 9,90–79,90 | 129,90–229,90 | 159,90–599,90 vital. | 30–54 |
| MeAgain | — | — | — | "Unlimited" 99,90–799,90 | — |
| Zepbound & Mounjaro Tracker | 99,90 | — | 149,90 · 499,90 | 1.299,90 vital. | 35–117 |
| MyFitnessPal | — | 31,90 · 82,90 | 162,90 · 329,90 | — | 38–77 |
| Yazio | — | — | 89,90 · 214,90 | trimestral 89,99–150 | 21–50 |
| FatSecret | — | 37,90–53,90 | 82,90–234,90 | — | 19–55 |
| Lose It! | — | — | (removido) | 1.999,90 vital. (reclamação) | — |
| MacroFactor | — | 49,90–66,90 | 264,90–599,90 | — | 62–140 |
| Fitia | — | 23,90–85,90 | 143,90–249,00 | — | 34–58 |
| Lifesum | — | 21,99–69,90 | 149,90–499,90 | — | 35–117 |
| Cal AI | — | — | — | "Unlimited" 14,90–249,90 | — |
| Balance | — | 19,90–61,90 | 249,90–579,90 | — | 58–135 |
| Sweat | — | 44,90–64,90 | 300,00 | — | 70 |
| Perda de Peso p/ Mulheres (Leap) | — | 40,90–79,90 | 119,90–199,90 | — | 28–47 |
| Google Play BR (faixa total) | Monju R$ 39,99–199,90 · OzemPro 59,99–249,99 · Shotsy 66,99–204,99 · MeAgain 74,99–629,99 · Glapp 25,99–204,99 · GLP1 Health Control 19,99 (único) | | | | |

### 4.2 EUA (US$)

| App | Semanal | Mensal | Anual | Vitalício/outro |
|---|---|---|---|---|
| Shotsy | — | 9,99 · 14,99 | 39,99 · 49,99 · 59,99 | — |
| MeAgain | — | — | — | "Unlimited" 14,99–119,99 |
| Glapp | 4,99 | 9,99 | 39,99 | — |
| Pep AI | — | 9,99 | 29,99–49,99 | 29,99 |
| Pep (Shred) | — | — | 39,99 | 9,99–59,99 |
| Zepbound & Mounjaro Tracker | 13,99 | — | 29,99 | 9,99–69,99 |
| GLP AI | — | 19,99 · 24,99 · 99,99 | 39,99–79,99 | 19,99–119,99 |
| OzemPro | — | 9,99 · 19,99 | 19,99–49,99 | — |
| Monju | 5,99 | 14,90 | 25,90–49,99 | — |
| MounjaPRO | 1,99 | 7,99–12,99 | 34,99 · 49,99 | 99,99 vital. |
| MyFitnessPal | — | 9,99 · 19,99 | 49,99 · 79,99 | — |
| MacroFactor | — | 9,99 · 11,99 | 47,99–89,99 | — |
| Yazio | — | 23,99 | 23,90–47,90 | — |
| Cal AI | — | — | — | 2,99–29,99 ("Unlimited", A/B) |
| Lose It! | — | — | — | 9,99–79,99; vital. 49,99–59,99 |
| Protein Pal | — | 3,99–4,99 | 23,99–35,99 | — |
| Balance | — | 3,99–10,99 | 49,99 · 98,99 | — |
| Lila (peri) | — | — | — | 5,99–79,99 (+464,99) |
| Healthi | — | 35,99 | 29,99 · 44,99 | 11,99–35,99 |
| RevenueCat H&F mediana 2026 | 4,99 | **9,99** | **39,94** | — |

### 4.3 México (MX$) — equiv. US$ a 17,0

| App | Semanal | Mensal | Anual | Anual em US$ equiv. |
|---|---|---|---|---|
| Shotsy | — | 139 · 179 · 299 · 399 | 599 · 669 · 899 | 35–53 |
| Monju | 129 | 249 | 399 · 599 | 23–35 |
| OzemPro | — | 199–399 | 399–999 | 23–59 |
| Glapp | 99 | 199 | 899 | 53 |
| GLP AI | — | 299–1.199 | 599–1.199 | 35–71 |
| Mingo | — | 249–1.299 | 799–1.799 | 47–106 |
| MeAgain | — | — | "Unlimited" 299–2.499 | — |
| Fitia | — | 115 · 149 · 219 | 609–1.029 | 36–61 |
| MyFitnessPal | — | 169 · 399 | 899 · 1.599 | 53–94 |
| Yazio | — | — | 299–699 | 18–41 |
| FatSecret | — | 135 · 194 | 290–830 | 17–49 |
| Balance | — | 89–279 | 999 · 2.399 | 59–141 |
| Play MX (faixa) | Shotsy 255–789 · Monju 159–779 · OzemPro 239–989 · MeAgain 299–2.499 | | | |

### 4.4 Chile e Argentina (App Store em US$)

| App | Chile semanal / mensal / anual | Argentina semanal / mensal / anual |
|---|---|---|
| Shotsy | — / 5,39–22,99 / 26,99–49,99 | — / 4,90–19,99 / 22,90–39,99 |
| Monju | 6,99 / 9,99 / 22,99–34,99 | 5,99 / 14,99 / 19,99–34,99 |
| OzemPro | — / 9,99–17,99 / 22,99–59,99 | — / 9,99–19,99 / 19,99–49,99 |
| Glapp | 5,99 / 9,99 / 49,99 | 4,99 / 9,99 / 39,99 |
| GLP AI | — / 17,99–79,99 / 34,99–65,00 | — / 14,99–69,99 / 26,99–52,99 |
| Fitia | — / 4,20–7,99 / 22,49–37,90 | — / 5,99–8,99 / 24,99–41,99 |
| MyFitnessPal | — / 6,50–14,50 / 32,90–57,90 | — / 9,99–19,99 / 49,99–79,99 |
| Yazio | — / — / 11,50–22,90 | — / 19,99 / 19,99–109,99 |
| FatSecret | — / 6,29–8,99 / 13,99–39,99 | — / 7,00–9,99 / 21,49–42,99 |
| Balance | — / 3,99–10,70 / 49,99–104,90 | — / 3,99–10,99 / 49,99–98,99 |

### 4.5 Colômbia (COP), Peru (S/) e Portugal (€)

| App | Colômbia mensal / anual | Peru mensal / anual | Portugal mensal / anual |
|---|---|---|---|
| Shotsy | 16.500–99.900 / 82.500–199.900 | 22,90–89,90 / 113,90–179,90 | 7,09–22,99 / 34,99–49,99 |
| Monju | 49.900 (sem. 29.900) / 89.900–129.900 | 44,90 (sem. 24,90) / 89,90–129,90 | 14,99 (sem. 6,99) / 22,99–34,99 |
| OzemPro | 69.900 / 99.900–229.900 | 69,90–89,90 / 89,90–249,90 | 9,99–22,99 / 22,99–59,99 |
| Glapp | 49.900 (sem. 24.900) / 199.900 | 44,90 (sem. 22,90) / 179,90 | 9,99 (sem. 5,99) / 44,99 |
| Fitia | 15.900–31.900 / 86.900–145.900 | 19,90–31,90 / 86,90–145,90 | 9,99–35,99 / 35,99–59,99 |
| MyFitnessPal | 27.900–82.900 / 142.900–334.900 | 31,90–89,90 / 162,90–269,90 | 9,99–21,99 / 49,99–87,99 |
| FatSecret | 29.000–41.500 / 62.500–179.000 | 25,90–36,90 / 112,90–160,90 | 9,39–13,49 / 35,00–53,90 |
| Balance | 19.900–52.900 / 299.900–469.900 | 14,90–44,90 / 199,90–399,90 | 4,99–11,99 / 59,99–105,99 |

**Trials:** a ficha da App Store não expõe a duração; pelas avaliações, Shotsy, Pep, Sweat, Foodvisor, Lila e Noom anunciam **7 dias** (e são acusados de cobrar na hora); Monju no México é criticado por **não ter trial**. RevenueCat 2026: 54% dos apps de H&F usam trials de 5–9 dias; 17+ dias convertem 42,5% vs. 25,5% (≤4 dias).

---

## 5. Mapa de posicionamento e lacunas

```
                      Foco em COMIDA/PROTEÍNA ──────────────▶
   GENÉRICO    MyFitnessPal · Yazio · Lose It! · Fitia · FatSecret · Cal AI (foto) · MacroFactor
      │        Protein Pal (só proteína, paywall duro)        WebDiet (nutricionista, "dados não coletados")
      │
   GLP-1       Shotsy · MeAgain · Glapp · Pep · Zepbound Tracker (EUA)         GLP AI (dieta IA, caro)
   TRACKER     Monju · OzemPro · MeuGLP · GlipOne · MounjaPRO · Mingo (BR)   ← todos: conta + rastreio,
      │        Lilly Health (oficial, grátis)                                   comida fraca, "tudo pago"
      │
   MULHERES    Flo · Clue · Stardust (ciclo)   Balance · Raquel (menopausa)   Lila (peri + emagrecimento: bugs/paywall)
      ▼        Sweat · Leap (treino)           ─── NINGUÉM: mulher + GLP-1 + proteína + privacidade ───
```

**Lacunas exploráveis (ordenadas por evidência nas avaliações):**

| # | Lacuna | Evidência | Como o produto ocupa |
|---|---|---|---|
| 1 | **Sem conta, dados no aparelho, nunca "somem"** | Nenhum GLP-1 app com "Dados não coletados"; OzemPro/Monju/MeAgain/Pep/GLP AI = vinculados + rastreio; 1★ recorrentes: "all my data is gone" (Shotsy, GLP AI, Glippy), "zera no dia da aplicação" (OzemPro), login toda hora (Dietbox), conta obrigatória = 34/50 1★ (Clearblue) | Sem login; SwiftData local + iCloud opcional; exportar/apagar em 1 toque; rótulo "Dados não coletados" na ficha (é raro e visível) |
| 2 | **Proteína primeiro, comida brasileira, porção honesta** | "falta comidas básicas" (MeuGLP), "registro nunca es exacto" (OzemPro MX), precisão = 41% das reclamações do Cal AI, MacroFactor/Cronometer sem PT-BR, MFP cobra código de barras, Alimente-se: 136 g de proteína p/ mulher | TACO embarcada, meta g/kg individual, foto on-device com **faixa** de incerteza + correção em 1 toque, barcode grátis |
| 3 | **Cobrança transparente e preço médio** | tema nº 1 das 1★ em BR/MX: trial que cobra na hora (Shotsy, Pep, Sweat, Foodvisor, Lila), promo que vira anual (BitePal, Mingo, Simple), cupom falso (OzemPro), lifetime R$ 1.999,90 (Lose It!), "tudo só funciona se pagar" (Monju, GlipOne, Fitia, Protein Pal) | Paywall com linha do tempo do trial, lembrete D5, sem semanal, sem lifetime, preço anual abaixo de Monju/OzemPro, versão grátis mínima mas útil (diário + 3 fotos/dia) |
| 4 | **Mulher como público explícito** | Zero companions de GLP-1 femininos; perguntas femininas em alta no Trends ("mounjaro amamentando", "mounjaro corta o efeito do anticoncepcional"); Lila (peri) é o único e está quebrado | Check-in de bem-estar com sintomas femininos, conteúdo com fontes, modo perimenopausa na v1.5 |
| 5 | **Estabilidade** | MeuGLP, Tecnonutri (bugs 16/33), Lifesum (17/33), Dietbox (15/39), Wild.AI, Caria, Lilly Health | App nativo pequeno, sem backend crítico |
| 6 | **Apple Health de verdade** | Pep "peso não importa do Health", Balance "não integra Apple Health", Mingo "não puxa nutrição" | HealthKit leitura/escrita de peso, proteína, água |

**Onde os concorrentes são fortes (não brigar):** base de alimentos gigante e comunidade (MFP), marca/UA em massa (Cal AI, Monju com 23 mil avaliações em 5 meses — sinal de mídia paga pesada), preço de entrada ultrabaixo (MounjaPRO R$ 2,90/semana, GLP1 Health Control R$ 19,99 único), programa oficial (Lilly Health).

---

## 6. Recomendação de preços por mercado

Princípios (RevenueCat 2026, seções "Pricing and packaging", "Funnel performance", "How do monetization strategies differ across geographies?", blog "5 quick fixes for bad default global pricing"):
- **Faixa média de preço**, não baixa: apps de preço alto têm RLTV Y1 US$ 62,19 vs. US$ 28,75 (médio) e US$ 10,69 (baixo), e convertem download→trial a 8,9% vs. 4,4%; mas retenção anual cai (36% baixo → 23% alto) e reembolso sobe (2,7% → 4,5%). Médio é o equilíbrio.
- **Anual em destaque, mensal como alternativa, sem semanal**: H&F tira 59% da receita do anual; renovação ativa anual 86,4%; semanal retém 1–2% em Y1 e a LatAm já é 29% semanal (pior mix do mundo).
- **Hard paywall após o onboarding + trial de 7 dias**: 10,7% vs. 2,1% de download→pago; H&F prefere trials de 5–9 dias.
- **Âncora "R$ X/mês" no anual**: caso RevenueCat na LatAm: +30% trial start, +10% take rate anual, sem mudar preço.
- **Localizar**: "no Brasil você pode estar cobrando 5× mais que o concorrente local"; app de meditação dobrou assinaturas na LatAm em 14 dias com preço regional.
- **Ficar abaixo do cluster BR de GLP-1 no mensal** (R$ 59,90) para tornar o anual óbvio, e **no meio do cluster no anual** (R$ 99,90–199,90).

| Mercado (moeda da loja) | Mensal | Anual | Anual ÷ 12 | ≈ US$ (anual) | Trial | Semanal / Vitalício | Referências do cluster |
|---|---|---|---|---|---|---|---|
| **Brasil (R$)** | **29,90** | **149,90** | 12,49 | 35 (Apple impl.) / 29 (câmbio ~5,1) | 7 dias | Não / Não (talvez 299,90 na v2) | Monju 59,90 / 99,90–199,90 · OzemPro 59,90–79,90 / 99,90–249,90 · Shotsy 25,90+ / 127,90–139 · FatSecret anual 82,90–234,90 · MFP 162,90 |
| **EUA (US$)** | **9,99** | **49,99** | 4,17 | 49,99 | 7 dias | Não / Não | Shotsy 9,99 / 39,99–59,99 · Glapp 9,99 / 39,99 · MeAgain "Unlimited" 59,99–69,99 · RevenueCat H&F 9,99 / 39,94 · MFP 49,99–79,99 |
| **Portugal (€)** | **7,99** | **39,99** | 3,33 | ~36 | 7 dias | Não / Não | Shotsy 7,09–22,99 / 34,99–49,99 · Monju 14,99 / 22,99–34,99 · Balance 4,99–11,99 / 59,99–105,99 |
| **México (MX$)** | **149** | **699** | 58 | 41 | 7 dias | Não / Não | Shotsy 139–399 / 599–899 · Monju 249 / 399–599 · OzemPro 199–399 / 399–999 · Fitia 115–219 / 609–1.029 |
| **Chile (US$ na loja)** | **7,99** | **34,99** | 2,92 | 34,99 | 7 dias | Não / Não | Shotsy 5,39–22,99 / 26,99–49,99 · Monju 9,99 / 22,99–34,99 · Fitia 4,20–7,99 / 22,49–37,90 · MFP 6,50–14,50 / 32,90–57,90 |
| **Colômbia (COP)** | **29.900** | **129.900** | 10.825 | 29 | 7 dias | Não / Não | Monju 49.900 / 89.900–129.900 · Shotsy 16.500–99.900 / 82.500–199.900 · Fitia 15.900–31.900 / 86.900–145.900 · OzemPro 69.900 / 99.900–229.900 |
| **Peru (S/)** | **24,90** | **99,90** | 8,33 | 26 | 7 dias | Não / Não | Monju 44,90 / 89,90–129,90 · Fitia 19,90–31,90 / 86,90–145,90 · Shotsy 22,90–89,90 / 113,90–179,90 |
| **Argentina (US$ na loja)** | **5,99** | **24,99** | 2,08 | 24,99 | 7 dias | Não / Não | Fitia 5,99–8,99 / 24,99–41,99 · Monju 14,99 / 19,99–34,99 · Shotsy 4,90–19,99 / 22,90–39,99 |

**Pontos de preço na App Store (sugestão de configuração):** definir o preço-base nos EUA (US$ 9,99 e US$ 49,99) e **desligar a equalização automática** nos 7 países acima, fixando manualmente os valores da tabela — eles seguem as convenções locais observadas nas fichas (R$ x,90; MX$ x9; COP x.900; S/ x,90; € x,99; US$ x,99), todas existentes como pontos de preço nas listagens dos concorrentes. Não consegui acessar a tabela de tiers do App Store Connect nesta sessão; os valores foram confirmados apenas por aparecerem em fichas publicadas.

**Versão grátis (para não cair em "tudo só funciona se pagar"):** diário GLP-1 completo + meta de proteína + 3 fotos/dia + barcode; pagos: fotos ilimitadas, correlações/insights, fotos de progresso, widget avançado, exportação. Isso responde à reclamação mais frequente dos concorrentes sem virar freemium puro (RevenueCat: freemium converte 2,1% vs. 10,7%). Testar com paywall "soft após o 3º dia de uso" vs. hard no onboarding.

**Promoções:** só 9,3% dos apps usam ofertas (H&F 14%); desconto mediano −50%. Usar apenas oferta de reconquista (winback) e um "primeiro ano −30%" sazonal (janeiro, quando `mounjaro`, `app de treino` e `dieta` têm pico no Trends BR). Nunca cupom no checkout (vide bug do OzemPro).

---

## 7. Ordem de lançamento

| Fase | Mercado | Por quê (evidência) | Pré-requisitos |
|---|---|---|---|
| 1 | **Brasil (iOS)** | `mounjaro` 65,6 (maior do mundo, +12% no trimestre); 2 apps BR com 100 mil+ instalações e 20 mil+ avaliações provam disposição a pagar R$ 99,90–199,90/ano; reclamações claras para explorar | PT-BR; TACO; LGPD (dados no aparelho) |
| 2 | **Portugal + EUA** (mesmo binário, semanas depois) | PT: `mounjaro` 30,1 (2º maior), português já pronto, Shotsy €34,99–49,99. EUA: maior conversão do mundo (D35 2,6%, H&F iOS 3,5%), preço US$ 49,99; concorrentes com reclamações de dados perdidos e trials enganosos | Inglês; base USDA como fallback; rótulo "Data Not Collected" |
| 3 | **México** | 2º maior interesse na AL e o que mais acelera (24,6 → 39,6); Fitia prova que o público hispânico paga (72,8 mil avaliações MX); Monju/OzemPro já têm 500–1.100 avaliações MX com queixas de preço/trial | **UI em espanhol (es-MX)**; tabela de alimentos mexicana (SMAE) ou fallback USDA; MX$ 149/699 |
| 4 | **Chile e Colômbia** | CL: `ozempic` 53,4 (empata com EUA), `menopausia` 79,5 e `perimenopausia` 67,7 (maiores da AL) — bom para a v1.5; loja em US$ simplifica. CO: Monju 733 e OzemPro 408 avaliações, Fitia 34 mil | Espanhol neutro; CL em US$ 7,99/34,99; CO em COP |
| 5 | Peru | Interesse baixo (`mounjaro` 6,1) mas Fitia nasceu lá (17 mil); S/ 24,90/99,90 | — |
| 6 | Argentina | Menor interesse (4,1), loja em US$ com poder de compra baixo, Fitia domina a US$ 24,99–41,99 | Preço US$ 5,99/24,99 |

Android (Play) fica para a fase 2 do produto: no Brasil os GLP-1 apps têm 100 mil+ instalações no Play e o Play tem 32,2% de cancelamentos por erro de cobrança (RevenueCat) — relevante para volume, não para a validação inicial.

---

## 8. Riscos competitivos e de preço

| Risco | Evidência | Mitigação |
|---|---|---|
| **Saturação e velocidade dos incumbentes BR** | Monju: 23 mil avaliações iOS em 5 meses; OzemPro 18 mil; ambos 100 mil+ no Play; 12+ apps BR no nicho | Não competir em UA de massa; posicionar por privacidade + proteína + mulher; CAC via creators/grupos; ASO em "mounjaro", "ozempic", "caneta" (não "glp-1") |
| **Notas infladas vs. avaliações recentes** | OzemPro 4,8 mas 25/50 recentes 1★; MeuGLP 16/47; Pep AI 32/50 — prompts de avaliação agressivos | Pedir avaliação só após 3 registros em dias distintos; responder reviews |
| **Guerra de preço no mensal** | MounjaPRO R$ 2,90/semana; GLP1 Health Control R$ 19,99 único; Mingo R$ 9,90 mensal (tier de teste) | Não disputar o piso; vender anual; comunicar privacidade e precisão |
| **Expectativa de "grátis"** | Rising queries "app … grátis"; "NOT FREE" é a 1★ mais comum em Protein Pal, MacroFactor MX, Fitia | Versão grátis útil (seção 6) e paywall claro |
| **Câmbio e poder de compra na AL** | RevenueCat: LatAm RPI ~US$ 0,10, D35 1,5%, preços abaixo de MEA, reembolsos até 30% em outliers; MacroFactor "pagamento não funciona no MX" | Preços fixos locais (não equalizados), revisar trimestralmente; aceitar que LatAm é volume/growth e NA/PT é margem |
| **Apple Store em US$ (AR/CL)** | Preços em dólar com renda local → churn | Preço baixo nesses dois (US$ 24,99–34,99/ano), lançar por último |
| **Apple 1.4.1 / ANVISA / LGPD** [secundário] | Diário de medicação sem dose; dados sensíveis | Sem recomendação de dose; fontes; dados on-device |
| **Idioma** | "está en inglés" (SnapCalorie MX), "en brasileño" (Tecnonutri MX 9/14 1★), "não tem português" (MacroFactor, Cronometer BR) | Lançar hispânicos só com es-MX/es-419 completo, inclusive paywall e base de alimentos |

---

## 9. Arquivos de dados

`docs/dados/concorrencia/`: `itunes_search_results.csv` · `appstore_storefronts.csv` · `appstore_reviews.json` · `compiled_prices.json` · `compiled_prices_summary.txt` · `play_storefronts.csv` · `trends_geo_summary.csv` · `trends_geo__<termo>__<países>__<janela>.csv` (18) · scripts `conc_search.py`, `appstore_scrape.py`, `play_scrape.py`, `trends_geo.py`, `compile_conc.py`.
