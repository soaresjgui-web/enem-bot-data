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

Ferramenta de treino, fora do app. Cada parte (configuração × `fold0`…`fold4`/`final`, ou `tudo` no ridge) tem um estado explícito, mudado só pela função pura `transicao(estado, evento)`, testada por `python3 treino_gpu.py --autoteste` e registrada em `eventos.jsonl` e `manifesto.json`.

| De | Evento | Para |
|---|---|---|
| pendente | `retomou` (arquivo já existe) | feita |
| pendente | `iniciou` | rodando |
| pendente | `sem_orcamento` | sem_orcamento |
| pendente | `reduziu_lote` (outra parte estourou a memória) | pendente, com lote menor |
| pendente | `cancelou` (outra parte falhou ou ficou sem orçamento) | cancelada |
| rodando | `concluiu` | feita |
| rodando | `estourou_memoria` | pendente com lote ÷ 2 (até ÷ 4); depois, falhou |
| rodando | `falhou` | pendente (1ª vez); falhou (2ª vez) |
| rodando | `sem_orcamento` (a projeção de tempo não cabe) | sem_orcamento |

A configuração é `completa` quando todas as partes estão `feita`; só então saem as previsões. Uma falha derruba só a configuração dela. Detalhes em `eval/treino/kaggle/LEIA-ME.md`.

## 3. Próximas máquinas a modelar

- **Onboarding e meta de proteína** (`OnboardingMachine`): boas-vindas → medidas → objetivo e caneta → meta calculada → primeira foto (ou prato de exemplo) → primeiro resultado → paywall. A primeira foto reutiliza a `MealCaptureMachine`; o paywall só aparece depois de `suggested`. Sem login na v1 (a conta é a do aparelho; "Restaurar compra" fica no paywall).
- **Assinatura** (`EntitlementMachine`): unknown → trial(ativo, dias) → active → gracePeriod → billingRetry → expired → restored. Fonte da verdade: StoreKit 2/RevenueCat; UI nunca lê flags soltas.
- **Diário da caneta** (`InjectionScheduleMachine`): sem aplicação configurada → agendada → dia da aplicação → registrada → atrasada. Dirige o cartão fixo no topo da tela Hoje (no dia da aplicação e nos dois dias seguintes).
- **Sincronização com Apple Health** (`HealthSyncMachine`): notAuthorized → authorizing → denied / syncing → upToDate → failed(retryAt).
- **Meta do dia** (`DailyGoalMachine`): abaixo da meta → perto (≥ 80%) → batida → excedida; dirige o anel, **a única ação da tela Hoje** (a sugestão "quanto falta"; com apetite baixo no modo caneta, opções pequenas e densas) e as comemorações.
