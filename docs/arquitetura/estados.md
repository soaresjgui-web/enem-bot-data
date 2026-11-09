# Máquinas de estado da Nutrelle

Catálogo vivo das máquinas de estado explícitas do app (regra em `CLAUDE.md`). Cada máquina: estados, eventos, transições e o que a UI mostra em cada estado. Rascunho inicial derivado do mock `docs/mock/registro-foto.html`; atualizar junto com o código.

## 1. Registro de refeição por foto (`MealCaptureMachine`)

```swift
enum MealCaptureState: Codable, Equatable {
    case idle
    case requestingCameraPermission
    case cameraDenied                                   // UX: explicar + botão "Abrir Ajustes" + opção galeria/busca manual
    case framing(liveLabels: [DetectedLabel], alignment: FramingAlignment) // câmera aberta, rótulos ao vivo, guia de enquadramento
    case analyzing(photo: PhotoRef)                     // modelo on-device rodando
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

**Premissa de produto (09/11/2026): sempre uma única foto, tirada de cima.** Não existe estado de "segunda foto". O círculo-guia e o aviso de nível padronizam ângulo e escala, que é o que o modelo de peso precisa. O disparo nunca fica bloqueado: `shutterTapped` fora de `aligned` (ou `galleryPicked`) segue para `analyzing` com `CaptureQuality` registrada na `PhotoRef`, e a `Estimate` sai com confiança rebaixada. Confiança baixa não é um estado separado: é `suggested(Estimate)` com `confidence == .low`, e a UI mostra a faixa mais larga, o destaque "Confira as porções" e a opção "Tirar outra foto".

| De | Evento | Para |
|---|---|---|
| idle | openCamera | requestingCameraPermission (ou framing se já autorizado) |
| requestingCameraPermission | permissionDenied | cameraDenied |
| framing | shutterTapped / galleryPicked | analyzing |
| framing | barcodeScanned | analyzing (lookup de produto) |
| analyzing | estimateReady | suggested |
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

## 2. Próximas máquinas a modelar

- **Onboarding e meta de proteína** (`OnboardingMachine`): perfil → objetivo → modo caneta (opcional) → meta calculada → paywall.
- **Assinatura** (`EntitlementMachine`): unknown → trial(ativo, dias) → active → gracePeriod → billingRetry → expired → restored. Fonte da verdade: StoreKit 2/RevenueCat; UI nunca lê flags soltas.
- **Diário da caneta** (`InjectionScheduleMachine`): sem aplicação configurada → agendada → dia da aplicação → registrada → atrasada.
- **Sincronização com Apple Health** (`HealthSyncMachine`): notAuthorized → authorizing → denied / syncing → upToDate → failed(retryAt).
- **Meta do dia** (`DailyGoalMachine`): abaixo da meta → perto (≥ 80%) → batida → excedida; dirige o anel, as sugestões "quanto falta" e as comemorações.
