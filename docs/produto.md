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

1. **Proteína por foto, afinada para comida brasileira** (núcleo do produto) — PF, marmita, cuscuz, tapioca, açaí, feijoada; base TACO/TBCA embarcada; porção estimada com referência (prato/mão), faixa de incerteza e correção em 1 toque; favoritos ("meu café de sempre"); código de barras.
2. **"Quanto falta hoje"** — sugestões com comida de verdade para fechar a meta ("faltam 30 g: 2 ovos + 1 iogurte natural").
3. **Modo caneta ligado à proteína** — dia da aplicação, apetite, náusea; em dias de pouco apetite, sugere opções pequenas e densas em proteína. Sem recomendação de dose.
4. **Check-in feminino** — sintomas do dia; respostas com fontes às dúvidas em alta (amamentação, anticoncepcional). Base para o modo 40+.
5. **Evolução + Apple Health** — peso, proteína e água lidos/escritos no Health; gráficos; widget "proteína de hoje".

**Plano do dia:** tela inicial com meta de proteína, quanto falta e check-in rápido.

**Papel da IA on-device:** custo marginal zero por foto → **fotos ilimitadas** em todos os planos; resposta < 1 s; funciona offline. Fallback em nuvem só se a precisão on-device não atingir a meta (ver `prototipo-precisao.md`).

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

## 7. Próximo passo e maior risco

O produto depende de a foto acertar a proteína em comida brasileira **visivelmente melhor** que Cal AI e os apps de caneta. Antes de construir o resto: **protótipo de precisão com ~100 pratos brasileiros reais** — protocolo e critérios em [`prototipo-precisao.md`](prototipo-precisao.md).
