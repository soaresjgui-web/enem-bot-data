# Máquinas de estado da Nutrelle

Catálogo vivo das máquinas de estado explícitas do app (regra em `CLAUDE.md`). Cada máquina: estados, eventos, transições e o que a UI mostra em cada estado. Rascunho inicial derivado do mock `docs/mock/registro-foto.html`; atualizar junto com o código.

## 1. Registro de refeição por foto (`MealCaptureMachine`)

```swift
enum MealCaptureState: Codable, Equatable {
    case idle
    case requestingCameraPermission
    case cameraDenied                                   // UX: explicar + botão "Abrir Ajustes" + opção galeria/busca manual
    case framing(liveLabels: [DetectedLabel], alignment: FramingAlignment) // câmera aberta, rótulos ao vivo, guia de enquadramento
    case analyzing(photo: PhotoRef, attempt: Int)       // análise em andamento; repete sozinha até 3 vezes antes de falhar
    case suggested(Estimate)                            // proteína pronta, faixa + confiança (.high/.medium/.low)
    case editing(Estimate, focusedItem: ItemID?)        // ajuste opcional
    case saving(Estimate)
    case saved(MealID)
    case failed(MealCaptureError, recoverFrom: RecoveryPoint)          // UX: mensagem + ação de recuperação
}

enum MealCaptureEvent: Codable {
    case openCamera, permissionGranted, permissionDenied
    case labelsUpdated([DetectedLabel]), alignmentChanged(FramingAlignment)
    case shutterTapped(PhotoRef), galleryPicked(PhotoRef), barcodeScanned(String)
    case estimateReady(Estimate), estimateFailed(MealCaptureError)
    case retakeTapped
    case adjustTapped(ItemID?), itemChanged(ItemID, grams: Double, preparation: Preparation?), doneEditing
    case saveTapped, saveSucceeded(MealID), saveFailed(MealCaptureError)
    case retry, cancel
}

enum FramingAlignment: Codable, Equatable {
    case searching                       // nenhum prato encontrado
    case tilted(degrees: Double)         // celular inclinado (CoreMotion): "Segure reto, acima do prato"
    case tooFar, tooClose                // prato fora do círculo-guia
    case aligned                         // de cima, prato no círculo
}

enum CaptureQuality: Codable, Equatable { case aligned, misaligned(FramingAlignment), unverified } // unverified = galeria
```

**Premissa de produto (09/11/2026): sempre uma única foto, tirada de cima.** Não existe estado de "segunda foto". O círculo-guia e o aviso de nível padronizam ângulo e escala, que é o que o modelo de peso precisa. O disparo nunca fica bloqueado: `shutterTapped` fora de `aligned` (ou `galleryPicked`) segue para `analyzing` com `CaptureQuality` registrada na `PhotoRef`, e a `Estimate` sai com confiança rebaixada. **Confiança calibrada por faixa de proteína:** o selo usa o acerto medido do modelo na faixa do prato (cerca de 90% até 15 g; 54–59% acima de 25 g, ver `docs/produto.md` §7). Quando o prato tem mais de ~25 g de proteína, a tela `suggested` não mostra "confiança alta": mostra "Confira o tamanho" com o seletor do item de proteína já aberto. Confiança baixa não é um estado separado: é `suggested(Estimate)` com `confidence == .low`, e a UI mostra a faixa mais larga, o destaque "Confira as porções" e a opção "Tirar outra foto".

| De | Evento | Para |
|---|---|---|
| idle | openCamera | requestingCameraPermission (ou framing se já autorizado) |
| requestingCameraPermission | permissionDenied | cameraDenied |
| framing | shutterTapped / galleryPicked | analyzing |
| framing | barcodeScanned | analyzing (lookup de produto) |
| analyzing | estimateReady | suggested |
| analyzing(attempt < 3) | estimateFailed (erro recuperável: rede, tempo esgotado) | analyzing(attempt + 1), sem mostrar erro |
| analyzing(attempt = 3) | estimateFailed | failed(recoverFrom:) com três saídas: tentar de novo, estimativa do aparelho, registro manual |
| suggested | itemChanged (seletor de tamanho do item de proteína, aberto na própria tela) | suggested (estimativa recalculada) |
| framing | alignmentChanged / labelsUpdated | framing (dados atualizados) |
| suggested | retakeTapped | framing |
| suggested | adjustTapped | editing |
| editing | doneEditing | suggested (estimativa recalculada) |
| suggested / editing | saveTapped | saving |
| saving | saveSucceeded | saved |
| qualquer | *Failed | failed(recoverFrom:) |
| failed | retry | estado de `recoverFrom` |
| qualquer (exceto saving) | cancel | idle |

`Estimate` = itens (alimento, preparo, gramas, proteína), total, faixa provável, confiança, qualidade da captura, versão do modelo — tudo `Codable` para log e reprodução (AX).

## 2. Treino em GPU por partes (`eval/treino/kaggle/treino_gpu.py`)

Ferramenta de treino, fora do app. Cada parte (configuração × `fold0`…`fold4`/`final`, `tudo` nas cabeças sobre features congeladas, `ext` nas previsões das fotos externas da quarta execução, `servico` na fase de serviço) tem um estado explícito, mudado só pela função pura `transicao(estado, evento)`, testada por `python3 treino_gpu.py --autoteste` e registrada em `eventos.jsonl` e `manifesto.json`.

| De | Evento | Para |
|---|---|---|
| pendente | `retomou` (arquivo já existe e foi feito com as mesmas listas) | feita |
| pendente | `iniciou` | rodando |
| pendente | `sem_orcamento` | sem_orcamento |
| pendente | `reduziu_lote` (outra parte estourou a memória) | pendente, com lote menor |
| pendente | `cancelou` (outra parte falhou ou ficou sem orçamento; dependência que não vai ficar pronta; base externa ausente do pacote) | cancelada |
| rodando | `concluiu` | feita |
| rodando | `estourou_memoria` | pendente com lote ÷ 2 (até ÷ 4); depois, falhou |
| rodando | `falhou` | pendente (1ª vez); falhou (2ª vez) |
| rodando | `sem_orcamento` (a projeção de tempo não cabe) | sem_orcamento |

A configuração é `completa` quando todas as partes estão `feita`; só então saem as previsões. Uma falha derruba só a configuração dela e as que dependem dela.

**Escalonador (o que cada processo faz a seguir).** Também funções puras, testadas sem GPU:

| Função | Entrada → saída |
|---|---|
| `proxima_acao(configs, estado, cabe, gpu)` | → `roda` (primeira parte pendente, na ordem de prioridade, com as dependências prontas) \| `espera` \| `cancela` (dependência morta) \| `sem_orcamento` (decidido só quando chega a vez da configuração) \| `fim` |
| `simula_orcamento(configs, estado, livres, seg_parte, limite)` | → quais configurações cabem inteiras no tempo que resta, em ordem de prioridade |
| `seg_parte(config, configs, estado)` | → segundos por parte: mediana do que já foi medido, ou a estimativa corrigida pela razão medido/estimado do mesmo tipo |

Mudanças da quarta execução (seção 13 de `docs/dados/precisao/resultado_modelo_treinado.md`):

- **GPU preferida** (`gpu` na configuração): cada processo pega primeiro o trabalho da sua GPU (GPU A: features e cabeças do So400m, ajuste fino parcial; GPU B: os três ajustes finos pequenos) e só leva o da outra quando não tem nada seu para rodar. Na simulação do orçamento, as partes de uma configuração com GPU preferida contam só nela.
- **Parte tardia** (`tardia`): a parte `ext` (previsões cruas nas fotos das bases externas) só roda depois das outras partes da configuração e quando todas as etapas de que ela depende já terminaram, bem ou mal. Uma dependência conta como pronta sem a sua parte tardia (R1 a R4 não esperam o R0 terminar as previsões externas).
- **Etapas sem modelo na mesma máquina de estados:** `externo <base>` (baixar, conferir e preparar as fotos de uma base, numa linha à parte, desde o minuto zero; prazo de 40 min; `falhou` derruba por dependência só as configurações que usam aquela base) e `<prefixo> feat so400m <base>` (features congeladas de uma base). Base sem lista no pacote: a etapa dela é `cancelada` no início, com o motivo no manifesto.
- **Conferência de paridade** (`paridade_d1`, função pura): assim que R0 e H0 terminam, compara com o D1 da execução 3; se falhar, o manifesto marca todas as configurações como `confiavel = false`. Não muda o estado de nenhuma parte.
- **Nenhuma decisão dentro do kernel:** nada cancela uma configuração por resultado de outra.

Mudanças da quinta execução (seção 14), só no escalonador; a máquina de estados das partes não muda:

- **GPU preferida flexível** (`gpu_flexivel`, nos ajustes finos F4, P2, F4c e F5): `simula_orcamento` devolve, além de `True`/`False`, `"qualquer_gpu"` quando a configuração não cabe na GPU dela mas cabe se as partes forem para o processo que fica livre primeiro; `proxima_acao` deixa então a outra GPU pegar as partes dela antes do próprio trabalho de prioridade menor. Assim a ordem de corte de 14.5 vale entre as duas GPUs: se o P2 (GPU A) atrasar, a GPU B ajuda nele e quem sai é o F5, depois o F4c. Sem `gpu_flexivel` (quarta execução), nada muda.
- **Portão da conferência a olho:** sem `"conferida": true` em `listas/conferencia_r5.json`, a execução nem começa (`--sem-conferencia` ou `--smoke` passam). Não é um estado de parte: é uma pré-condição.
- **Base sem recorte:** se o Nutrition5k foi reprovado na conferência ou não tem tabela da etapa 0, todas as configurações sobre o recorte (features, R0c–R4c, H0c, H1c, F4c) são `cancelada` no início, com o motivo; o F4, o P2 e o F5 (foto inteira) seguem. Base externa reprovada ou sem tabela entra com a foto inteira, e o manifesto diz por quê.
- A etapa `externo <base>` também prepara os recortes da base (com o sha256 de cada foto conferido contra a tabela da etapa 0); a parte `ext` dos ajustes finos espera os recortes, como esperava as fotos.
- A conferência de paridade com o D1 não se aplica (o R0 e o H0 não são refeitos).

Mudanças da sexta execução (seção 15); a máquina de estados das partes e as funções do escalonador não mudam:

- **Sem GPU preferida:** nenhuma configuração declara `gpu`, então cada processo pega a primeira parte pronta na ordem de prioridade (as features do recorte e as 7 partes do P3 se repartem entre as duas GPUs). A ordem de corte (H0c-m, depois F5c) é a da prioridade, como antes.
- **Parte `final` do P3 sem modelo:** ela treina todas as épocas e grava a média móvel de cada uma no cache; o modelo final (época = mediana das escolhidas nos folds), a HP3 final, os conjuntos de previsão e o arquivo de servir saem na parte tardia `ext`, que já roda depois de todas as outras partes. Por isso a `final` do P3 conta como pronta com as épocas gravadas, e a `ext` só com o arquivo de servir (`parte_pronta4`).
- **Paridade:** a mesma função pura (`paridade_d1`) compara a R4c6 e a H0c6 com a quinta execução (`listas/ref_r5_prato.csv`, critério 68,34 ± 0,3); se falhar, o manifesto marca tudo como não confiável. Não muda o estado de nenhuma parte.

Detalhes em `eval/treino/kaggle/LEIA-ME.md`.

## 3. Próximas máquinas a modelar

- **Onboarding e meta de proteína** (`OnboardingMachine`): boas-vindas → medidas → objetivo e caneta → meta calculada → primeira foto (ou prato de exemplo) → primeiro resultado → paywall. A primeira foto reutiliza a `MealCaptureMachine`; o paywall só aparece depois de `suggested`. Sem login na v1 (a conta é a do aparelho; "Restaurar compra" fica no paywall).
- **Assinatura** (`EntitlementMachine`): unknown → trial(ativo, dias) → active → gracePeriod → billingRetry → expired → restored. Fonte da verdade: StoreKit 2/RevenueCat; UI nunca lê flags soltas.
- **Diário da caneta** (`InjectionScheduleMachine`): sem aplicação configurada → agendada → dia da aplicação → registrada → atrasada. Dirige o cartão fixo no topo da tela Hoje (no dia da aplicação e nos dois dias seguintes).
- **Sincronização com Apple Health** (`HealthSyncMachine`): notAuthorized → authorizing → denied / syncing → upToDate → failed(retryAt).
- **Meta do dia** (`DailyGoalMachine`): abaixo da meta → perto (≥ 80%) → batida → excedida; dirige o anel, **a única ação da tela Hoje** (a sugestão "quanto falta"; com apetite baixo no modo caneta, opções pequenas e densas) e as comemorações.
