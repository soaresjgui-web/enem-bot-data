# Nutrelle — instruções do projeto

App iOS nativo (Swift/SwiftUI) de proteína para mulheres que estão mudando o corpo, com modo caneta (GLP-1) e IA on-device. Decisões de produto em `docs/produto.md`; fluxo de registro em `docs/mock/registro-foto.html`; evidências de precisão em `docs/dados/precisao/`.

## Regra permanente: máquinas de estado explícitas (UX + DX + AX)

Durante toda a construção do app, **sempre**:

1. **Encontrar todas as máquinas de estado implícitas** — combinações de `Bool`/opcionais/flags (`isLoading`, `hasError`, `result != nil`, `didShowPaywall`...), telas que dependem de "se X e não Y", fluxos assíncronos, retries, permissões, assinatura, sincronização.
2. **Torná-las explícitas**: um `enum` de estados com dados associados + transições num único lugar (reducer/função pura `(estado, evento) -> estado`). Estados impossíveis não devem ser representáveis.
3. **Decidir pelo que eleva os três ao mesmo tempo**:
   - **UX** — todo estado tem tela/feedback definido (carregando, vazio, erro com ação, offline, permissão negada, confiança baixa); nada de telas "presas" ou combinações incoerentes.
   - **DX** — estados e transições legíveis, testáveis em testes unitários sem UI, com previews do SwiftUI para cada estado.
   - **AX (experiência de agentes)** — estados e eventos nomeados, serializáveis (`Codable`) e logáveis, para que agentes (e humanos) consigam inspecionar, reproduzir e testar qualquer fluxo; documentar cada máquina em `docs/arquitetura/estados.md`.
4. **Sem quebrar nada**: refatorar com testes cobrindo o comportamento atual antes da mudança; mudanças incrementais; build e testes verdes a cada passo.

Ao revisar ou criar qualquer código, verifique esta regra e atualize `docs/arquitetura/estados.md`.

## Outras convenções

- Idioma do produto e da documentação: PT-BR. Código e nomes técnicos em inglês.
- Repositório público: nunca commitar imagens de terceiros, pesos de modelos, chaves ou dados pessoais.
- Precisão da foto: qualquer mudança no pipeline de estimativa deve ser medida com `eval/` contra o conjunto de teste fora do treino (`split = teste`), nunca contra dados usados para ajustar.
