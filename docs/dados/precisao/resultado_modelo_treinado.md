# Modelo treinado: proteína só pela foto, sem informar porção (09/11/2026)

**Pergunta:** um modelo **treinado** (e não um LLM em nuvem) consegue sugerir a proteína só pela foto e atingir a meta de **≥ 70% dos pratos dentro de ±20% ou ±5 g**?
**Resposta curta: não, com os dados que existem hoje.** Com higiene de dados rigorosa, o melhor modelo só de imagem fica em **~48–52%** no `teste` (280 pratos), abaixo do LLM (55%). Profundidade (RGB-D) não ajudou. O melhor resultado foi o **híbrido** "LLM identifica e dá a densidade, modelo treinado dá o peso", com **~60% no teste (59,6%) e 60–62% no n5k v1**: melhora o LLM em 5 pontos, mas ainda fica 10 pontos abaixo da meta. A comida brasileira em foto que não é zenital **quebra** o modelo de peso (Liv Up 3%). **Atualização (seção 7):** treinar também com os quadros laterais do Nutrition5k deixa o modelo robusto ao ângulo (foto lateral 44% → 49%) e leva o Brasil de 22% para 36% (Liv Up 3% → 26%), mas o híbrido continua em ~59–60%, longe de 70%. **Atualização (seção 8, premissa de uma única foto de cima):** com 8 orientações da foto, resolução maior, ajuste fino parcial e média de modelos, o modelo **só de imagem sobe de 50% para 61%** no teste (IC 55 a 67; +11 pontos sobre o anterior, IC +5 a +17) e alcança o LLM (55%). O erro de massa cai de 22% para 17%. O **híbrido escolhido fica em 63%** (IC 57 a 69; formas equivalentes dão de 62% a 66%), sem diferença segura para os 59,6% de antes. **A meta de 70% continua não atingida**, e o erro que sobra está mais na densidade (que alimento é) do que no peso. Cerca de metade dos pratos pode ser sugerida sozinha com ~70% de acerto, mas essa metade é, na maior parte, de pratos com pouca proteína. Fora da distância fixa do Nutrition5k o modelo erra a escala, a menos que seja treinado com escala aleatória ou com o prato enquadrado num guia.

Scripts em `eval/treino/`. Dados, features, pesos e o `.mlpackage` ficam só no scratchpad (`…/scratchpad/treino/`), fora do repositório.

## 1. Dados e higiene

- **Treino:** split oficial `rgb_train_ids.txt` do Nutrition5k ∩ pasta `realsense_overhead`, com os filtros do v2 (massa ≥ 20 g; sem ingrediente `deprecated`/`plate only`/sem nome; soma dos ingredientes coerente com o total). Sobraram 2.691 pratos, **todos do café 1**: os pratos do café 2 não aparecem nos arquivos de split.
- **Avaliação excluída por `dish_id`** (`prep_dados.py`). São 450 ids distintos: 280 `teste`, 120 `calib` e 50 `n5k_v1`. Os pratos do v2 tinham sido sorteados entre **todos** os pratos com imagem, então 199 do `teste` e 83 do `calib` estavam no split oficial de **treino**. Os 282 foram removidos, e **a interseção treino × avaliação é 0**. Os 50 do `n5k_v1` estão todos no split oficial de teste.
- **Achado — vazamento por pratos "irmãos":** o Nutrition5k fotografa o **mesmo prato várias vezes**, à medida que cada ingrediente é adicionado, com cerca de 30 s entre fotos. **68% dos pratos do `teste` têm um prato de treino a menos de 2 min** (provavelmente o mesmo prato com um ingrediente a mais ou a menos). Tirar só os `dish_id` iguais não basta. Por isso há dois treinos:
  - **oficial:** 2.409 pratos (split oficial menos os 450 de avaliação);
  - **estrito:** 1.506 pratos. Além dos 450, sai toda a "cadeia" de pratos de treino ligados a um prato de avaliação por intervalos < 120 s. **Este é o número honesto**, e é o que vale nas conclusões.
  - Controle com o mesmo tamanho: o treino oficial subamostrado para 1.506 pratos dá **55,4%** (SigLIP) contra **48,9%** no estrito. **~6 pontos do resultado "oficial" vêm de ver o prato irmão no treino**, não de generalização.
- Hiperparâmetros (alfa do ridge, nº de árvores do LightGBM, regularização do MLP) e **a escolha do backbone** saíram de validação cruzada **agrupada por cadeia** (GroupKFold), feita só no treino. O `calib` (120) foi usado só para ajustar os pesos do blend do híbrido C. `teste` e `n5k_v1` não entraram em nenhuma escolha.
- Incerteza: com n = 280, um resultado em torno de 50% tem erro-padrão de **±3 pontos**; com n = 50 (`n5k_v1`), **±7 pontos**.

## 2. Resultados

Critério do produto: ±5 g **ou** ±20%. "Direto" = regressão de log(proteína). "m×d" = massa prevista × densidade de proteína prevista. Viés em gramas (negativo = subestima).

| Experimento | Treino | Teste 280: direto | Teste: m×d | MAE / viés (direto) | Erro de massa (MAPE) | n5k v1 (50): direto / m×d |
|---|---|---|---|---|---|---|
| **LLM em nuvem v2 (linha de base)** | — | **55,0%** | — | 8,6 g / −3,6 g | 30% | 52% |
| **Meta** | | **70%** | | | | |
| A · ViT-S/16 IN-21k + ridge | oficial | 50,0% | 50,0% | 9,9 / −3,6 | 26% | 46 / 46% |
| A · ViT-S/16 IN-21k + ridge | estrito | 49,6% | 49,6% | 10,5 / −4,5 | 26% | 42 / 36% |
| A · SigLIP B/16 + ridge | oficial | 55,7% | 52,9% | 8,1 / −2,9 | 20% | 54 / 50% |
| A · SigLIP B/16 + ridge | estrito | 48,9% | 46,8% | 9,2 / −4,2 | 22% | 44 / 38% |
| A · SigLIP 2 B/16 + ridge (melhor na CV) | estrito | 47,9% | 48,9% | 8,8 / −4,3 | 22% | 46 / 44% |
| A · EfficientNet-B0 + ridge | estrito | 46,1% | 45,7% | 11,3 / −2,8 | 27% | 42 / 38% |
| A · SigLIP + SigLIP 2 + ViT-S + ridge | estrito | 51,4% | 51,4% | 8,5 / −4,4 | 20% | 48 / 50% |
| A · SigLIP 2 + LightGBM | estrito | 48,6% | 46,1% | 9,4 / −6,3 | 25% | 50 / 50% |
| A · SigLIP 2 + MLP | estrito | 49,6% | 50,0% | 8,7 / −4,3 | 31% | 42 / 42% |
| B · só profundidade (atributos de volume) + ridge | estrito | 27,5% | 27,1% | 15,2 / −11,0 | 45% | 32 / 28% |
| B · só profundidade + LightGBM | estrito | 30,7% | 30,7% | 14,1 / −9,8 | 34% | 30 / 30% |
| B · só mapa de altura 24×32 + ridge | estrito | 28,9% | 28,9% | 15,9 / −13,1 | 54% | 32 / 38% |
| B · SigLIP 2 + atributos de profundidade + ridge | estrito | 51,1% | 51,4% | 8,7 / −4,2 | 21% | 42 / 40% |
| B · SigLIP 2 + mapa + atributos + ridge | estrito | 51,8% | 50,4% | 8,9 / −4,0 | 22% | 34 / 40% |
| B · SigLIP 2 + atributos + LightGBM / MLP | estrito | 48,2 / 48,6% | 50,0 / 51,8% | 9,5 / 8,7 | 22 / 32% | 42 / 42–46% |
| D · fine-tune EfficientNet-B0 multitarefa, 8 épocas | estrito | 43,9% | 47,1% | 10,5 / −4,2 | 31% | 44 / 44% |
| **C · massa do modelo × densidade do LLM** | estrito | **59,6%** | | 7,2 / −2,2 | 22% | 52% |
| **C · massa √(modelo·LLM) × densidade do LLM** | estrito | **58,9%** | | 7,4 / −3,2 | | **60%** |
| **C · blend log-linear (pesos ajustados no calib)** | estrito | **59,6%** | | **7,0 / −2,6** | | **62%** |
| C · massa do modelo × √(densidade do modelo · do LLM) | estrito | 55,4% | | 7,4 / −1,1 | | 54% |

O que os números mostram:

1. **Só imagem, com treino honesto: 46–52%.** Nenhum backbone ou regressor passa do LLM. Os embeddings de modelos imagem-texto (SigLIP) estimam massa melhor (MAPE 20–22% contra 30% do LLM), mas a densidade de proteína sai pior, e o critério empata ou piora. "Direto" e "massa × densidade" dão praticamente o mesmo resultado (diferença ≤ 3 pontos, dentro do ruído).
2. **B (RGB-D) não ajudou.** Profundidade sozinha dá 28–31%, e somada à imagem não sai do ruído (+0–3 pontos). O plano de apoio por RANSAC na borda e o volume acima da mesa e do nível do prato correlacionam pouco com a massa (|r| ≤ 0,5, e a maior parte vem da *altura do prato*, ou seja, tigela × prato raso, não da comida). Sem intrínsecos, sem segmentação de comida e com ~14% de pixels sem leitura, o sensor do Nutrition5k não serve para volume. No iPhone (LiDAR com intrínsecos e malha) seria diferente, mas teria de ser **medido**, não suposto.
3. **D (fine-tune leve) piorou:** com 1.506 pratos o modelo decora o treino (perda caiu de 1,32 para 0,11) e fica em 44–47%. Com o dataset tão pequeno, features congeladas de um modelo grande ganham do ajuste fino de um modelo pequeno.
4. **C (híbrido) é o único que passa o LLM:** 59–60% no teste (+5 pontos, MAE 8,6 → 7,0 g) e 60–62% no n5k v1. Funciona porque cada lado faz o que faz melhor: o modelo treinado erra menos o peso, e o LLM identifica o alimento (densidade).
5. **Onde está o gargalo** (análise de oráculo no teste): com o **peso real** × a densidade do LLM, o resultado já seria **71%** (n5k v1 66%, UFPR 78%). Com o peso do LLM × a densidade real, seria 63%. **O erro dominante é o peso.** O seletor de porção do v2 (71–76%) ataca exatamente esse erro.

## 3. Curva de aprendizado (SigLIP 2 + ridge, treino estrito, média de 3 sorteios)

| Fração do treino | Pratos | Direto | m×d | MAE |
|---|---|---|---|---|
| 10% | 150 | 42,6% | 42,4% | 10,9 g |
| 25% | 376 | 48,2% | 46,3% | 9,5 g |
| 50% | 753 | 46,8% | 46,9% | 9,3 g |
| 100% | 1.506 | 47,9% | 48,9% | 8,8 g |

- **O critério estabiliza em ~47–49% a partir de ~400 pratos.** O MAE ainda cai, mas devagar: dobrar os dados tira ~0,3–0,5 g. Uma lei de potência ajustada (MAE ∝ n^−0,09) só levaria o MAE a ~5 g, nível compatível com 70%, com **centenas de milhares de pratos**. Mais dados do mesmo tipo, com features congeladas, não chegam à meta.
- O treino oficial com 2.409 pratos chega a 52–56%, mas o controle de mesmo tamanho mostra que o ganho vem de pratos irmãos, não do volume de dados.

## 4. Transferência para comida brasileira (E)

Melhor modelo só de imagem pela CV (SigLIP 2 + ridge, estrito), aplicado às fotos dos lotes cegos:

| Fonte | n | Modelo treinado (direto) | MAE / viés | Erro de massa | LLM v2 | Híbrido C (massa √(modelo·LLM) × densidade LLM) |
|---|---|---|---|---|---|---|
| Nutrition5k teste (referência) | 280 | 47,9% | 8,8 g / −4,3 | 22% | 55% | 58,9% |
| UFPR (alimento único, foto a 45°) | 103 | **29,1%** | 14,6 g / −2,9 | 51% | 47% | 54% |
| Liv Up (marmita, foto de estúdio) | 34 | **2,9%** | 57,6 g / **+57,6** | 66% (viés de massa **+194 g**) | 24% | 3% |
| Seara (arte da bandeja) | 7 | 14,3% | 41,9 g / +40,9 | 65% | 14% | 14% |
| **Brasileiros (144)** | 144 | **22%** | | | 40% | 42% |

- **Queda de 48% para ~22%** (−26 pontos) na comida brasileira, contra 40% do LLM nas mesmas fotos. O modelo aprendeu a escala do Nutrition5k: câmera zenital a altura fixa, mesmo prato, mesma bancada. Num recorte de estúdio, a marmita ocupa a imagem inteira e o modelo "vê" um prato de ~3× o peso. O fine-tune (D) sofre menos com Liv Up (26%), mas cai em UFPR (33%).
- No UFPR (fundo e prato padronizados) o híbrido ainda ajuda (47% → 54%). Em marmita não ajuda, porque o peso que vem da imagem está errado.
- **Conclusão:** um modelo treinado só no Nutrition5k **não pode** ir para o app brasileiro sem dados locais, e nem mesmo para fotos do Nutrition5k tiradas de outro ângulo.

## 5. On-device (F)

| Modelo | Parâmetros | Tamanho FP16 | Latência CPU x86, 1 thread, lote 1* | Core ML |
|---|---|---|---|---|
| EfficientNet-B0 (fine-tune, D) | 4,0 M | 7,7 MB | 59 ms | **convertido** (`mlprogram` FP16, iOS 16+, 7,7 MB) |
| ViT-S/16 + cabeça linear | 21,7 M | 41 MB | 169 ms | falhou no `torch.jit.trace` → `ct.convert` (torch 2.15 é mais novo que o testado pelo coremltools 9; resolve com torch ≤ 2.7 ou `torch.export`) |
| SigLIP B/16 + cabeça linear | 92,9 M | 177 MB (~90 MB em int8) | 379 ms | não tentado (mesma arquitetura ViT; mesmo caminho) |

\* Medido numa máquina compartilhada e sobrecarregada. Com 4 threads ficou *mais lento* por disputa de CPU. Na Neural Engine de um iPhone recente, modelos desse porte costumam rodar em ~5–30 ms. Como o .mlpackage não pode ser executado no Linux, isso não foi medido aqui.
- **Viável no aparelho:** os ridges são uma camada linear (~6 mil pesos), e o backbone cabe. O pacote de 7,7 MB da EfficientNet-B0 é trivial, e um ViT-B quantizado (~90 MB) é aceitável. **O problema não é o aparelho, é a precisão.**
- Pesos: o CDN do Hugging Face (`cdn-lfs`/`xet`/`us.aws.cdn.hf.co`) e `dl.fbaipublicfiles.com` estão **bloqueados** no proxy desta sessão. Por isso **DINOv2 e CLIP (OpenAI/LAION) não puderam ser baixados**. Usei os equivalentes públicos que estavam liberados: SigLIP e SigLIP 2 (`gs://big_vision`, modelos imagem-texto do tipo CLIP), ViT-S/16 AugReg IN-21k (`gs://vit_models`) e EfficientNet-B0 (GitHub do timm). Um teste com DINOv2 continua pendente. A literatura sugere que ele é melhor para geometria e porção do que os modelos imagem-texto, mas, pela curva de aprendizado acima, não espero que feche os 20 pontos que faltam.

## 6. Recomendação: que dados brasileiros seriam necessários

1. **Não usar regressão de proteína direta como número final.** Mesmo com dados ideais, o teto com features congeladas está perto de 50%, e o erro dominante é o **peso**. O caminho que atinge a meta continua sendo o **seletor de porção do componente proteico** (71–76% no v2). O modelo treinado pode servir para **sugerir o valor inicial do seletor**, no lugar do LLM: o híbrido C erra menos o peso.
2. **Se for treinar um estimador de peso para o Brasil, o tipo de dado vale mais que a quantidade:**
   - fotos **reais de celular**, no ângulo e na distância em que a usuária fotografa (não zenital fixo), com **peso por componente** medido em balança: arroz, feijão, carne/frango/ovo, guarnição, salada;
   - variação de prato e recipiente: prato raso, fundo, marmita de alumínio, pote, bandeja de self-service, com **referência de escala conhecida** (diâmetro do prato) ou profundidade do iPhone com intrínsecos (`AVDepthData`), gravada junto;
   - preparações reais (carne de panela, frango desfiado ao molho, estrogonofe, feijoada, marmitas), já que a densidade de proteína das preparações é o segundo erro (ver `resultado_v2.md`).
3. **Quantidade:** pela curva, ~400 pratos já levam o modelo congelado ao platô no domínio de origem. Para **adaptar** ao domínio brasileiro, o plano é **400–800 pratos pesados por componente**, cada um fotografado de 2 a 3 ângulos (≈1.500–2.000 imagens, com split **por prato**, nunca por foto). Esse volume trocaria a queda de −26 pontos por um desempenho próximo do teste do Nutrition5k (~50% só de imagem, ~60% no híbrido). **Não** chega a 70%. Passar disso exigiria escala de dezenas a centenas de milhares de pratos, ou uma referência métrica (profundidade/LiDAR calibrada) que precisaria ser validada antes.
4. Antes de coletar, vale **um piloto de 50 pratos** com iPhone (RGB + `AVDepthData` + peso por componente) para medir se a profundidade *com intrínsecos* reduz o erro de massa abaixo de ~10%. Pela análise de oráculo, é o que falta para chegar perto de 70% junto com a densidade do LLM.

## 7. Ângulos laterais (Nutrition5k `side_angles`, 09/11/2026)

**Pergunta:** treinar também com quadros laterais do Nutrition5k (câmeras inclinadas, mais parecidas com a foto de celular) melhora o modelo de peso e proteína, principalmente fora da vista zenital e na comida brasileira?
**Resposta curta: sim para robustez de ângulo e para o Brasil, não para a meta.** No zenital o ganho fica dentro do ruído (+2 pontos). Numa foto lateral isolada o modelo sobe de 44% para 49% e o erro de massa cai de 45% para 26%. Na comida brasileira o modelo só de imagem sobe de **22% para 36%**, e a Liv Up sai de 3% para 26% (viés de massa de +194 g para −27 g). O híbrido C continua em **~59–60%** no teste. **Nada disso chega perto de 70%.**

### 7.1 Dados e higiene

- **Quadros:** 4 por prato, um de cada câmera (A, B, C, D), alternando o início e o fim da rotação (`A_ini`, `B_fim`, `C_ini`, `D_fim`; quando falta câmera, entra o outro momento). São 19.172 quadros de 4.793 pratos, mais 2.968 imagens do ECUSTFD. Mesmo backbone e pré-processamento do zenital: SigLIP 2 B/16, imagem inteira redimensionada para 224×224, sem recorte.
- **Tempo de CPU:** 22.140 imagens em **30,5 min**, ou 83 ms/imagem de parede, com 4 processos de 1 thread (~330 ms/imagem por processo). Os experimentos levam 17 min com a CV e 4,5 min sem ela.
- **Mesma higiene da seção 1, com uma lista única de pratos para as duas fontes.** Os 450 `dish_id` de avaliação nunca entram no treino, nem como foto zenital, nem como quadro lateral. A regra estrita de cadeia (< 120 s) passou a ser calculada sobre **todos os 5.006 pratos com metadado** (cafés 1 e 2), não só sobre o treino zenital. É mais conservadora: o treino zenital cai de 1.506 para 1.435 pratos. Um prato entra no treino com todas as suas imagens, ou fica fora com todas. Os quadros laterais dos pratos de avaliação servem só para avaliar.
- **Configurações** (pratos distintos no treino):

| Config. | Zenital | Lateral | Pratos distintos |
|---|---|---|---|
| 1 zenital (regra original, a da seção 2) | 1.506 | — | 1.506 |
| 1' zenital (regra universal) | 1.435 | — | 1.435 |
| 2a só lateral, mesmos pratos | — | 1.381 (5.524 quadros) | 1.381 |
| 2b zenital + lateral, mesmos pratos | 1.435 | 1.381 | 1.435 |
| 2c 2b + pratos que só têm vídeo (split `train`) | 1.435 | 2.591 (10.364) | 2.645 |
| 2d 2c + pratos `test`/sem split fora da avaliação e das cadeias dela | 1.435 | 2.999 (11.996) | 3.053 |
| 2e só lateral, todos os pratos do split `train` | — | 2.591 | 2.591 |

- **Escolha da configuração só pela CV no treino:** GroupKFold por cadeia, validando nos mesmos pratos zenitais em todas as configurações, tanto na foto zenital quanto nos quadros laterais. O critério foi o MSE de log(proteína), na média entre zenital e lateral. A **2d** venceu (0,2317 / 0,2617, contra 0,2256 / 0,3901 da 1'). A diferença entre 2b, 2c e 2d na CV é pequena. O zenital puro é o melhor só na vista zenital, e o lateral puro (2a, 2e) piora muito o zenital (log massa 0,08 → 0,31–0,38). O alfa do ridge também saiu da CV; o `calib` foi usado só no blend do híbrido.
- Sem LLM nos quadros laterais: o LLM v2 viu apenas a foto zenital. Nos híbridos com massa lateral, a densidade vem do LLM sobre a foto zenital. Isso supõe que o LLM identifica o alimento igualmente bem de lado, o que **não foi medido**.

### 7.2 Resultados

"Lateral, 1 quadro" trata cada quadro como uma foto isolada, que é o caso do celular (1.020 quadros de 255 pratos do `teste`; 25 pratos do teste não têm vídeo). "Lateral, média 4" usa a média em log dos 4 quadros do prato. O critério é o da seção 2 (±5 g ou ±20%), e o resultado aparece como direto / m×d.

| Config. | Teste zenital (280) | MAPE massa zen. | Teste lateral, 1 quadro | MAPE massa lat. | Teste lateral, média 4 | n5k v1 zen. / lat. 1 quadro |
|---|---|---|---|---|---|---|
| LLM v2 (referência) | 55,0% | 30% | — | — | — | 52% / — |
| 1 zenital (regra original) | 47,9 / 48,9% | 22% | 44,1 / 42,8% | 45% (viés +52 g) | 50,2 / 51,0% | 46% / 38% |
| 1' zenital (regra universal) | 48,9 / 47,9% | 22% | 42,8 / 42,5% | 46% | 50,6 / 51,0% | 48% / 40% |
| 2a só lateral, mesmos pratos | 43,2 / 43,2% | 40% | **49,8** / 48,9% | **25%** | 52,9 / 51,4% | 32% / 45% |
| 2b zenital + lateral, mesmos pratos | 47,9 / 47,5% | 21% | 49,2 / 49,3% | 26% | 53,7 / 51,4% | 44% / 45% |
| 2c + pratos só com vídeo | **50,7** / 48,9% | 22% | 48,9 / 48,9% | 27% | 56,1 / 54,9% | 46% / 47% |
| **2d + test/sem split (escolhida pela CV)** | **50,0 / 50,7%** | 22% | **48,6 / 49,0%** | **26%** (viés −8 g) | **56,9 / 54,1%** | 48% / 46% |
| 2e só lateral, todos os pratos | 40,7 / 36,4% | 44% | 48,5 / 50,1% | 26% | 56,5 / 54,5% | 34% / 46% |
| 2d + ECUSTFD (só na massa) | 50,0 / 52,1% | 23% | 48,6 / 49,6% | 27% | 56,9 / 53,3% | 48% / 46% |

**Híbrido C** (massa do modelo × densidade do LLM):

| Config. | Teste zenital: m×dLLM / √(m·LLM)×dLLM / blend | Teste lateral, 1 quadro: √(lat·LLM)×dLLM | Teste lateral, média 4: lat×dLLM | n5k v1: √ / blend |
|---|---|---|---|---|
| LLM sozinho (mesmos pratos) | 55,0% | 55,7% | 55,7% | 52% |
| 1 zenital (regra original) | 59,6 / 58,9 / 59,6% | 56,1% | 55,7% | 60 / 62% |
| 2b zenital + lateral, mesmos pratos | 57,5 / 59,3 / 58,2% | 58,1% | 60,8% | 56 / 58% |
| 2c + pratos só com vídeo | **60,7** / 60,4 / 59,6% | 57,7% | **63,5%** | 56 / 58% |
| **2d (escolhida pela CV)** | 59,6 / 58,9 / 58,9% | **59,1%** | 63,1% | 54 / 62% |

**Transferência para comida brasileira** (direto; erro e viés de massa entre parênteses):

| Config. | UFPR (103) | Liv Up (34) | Seara (7) | Brasileiros (144): direto / m×d | Híbrido √(m·LLM)×dLLM: BR / UFPR / Liv Up |
|---|---|---|---|---|---|
| LLM v2 | 47% | 24% | 14% | 40% | — |
| 1 zenital (regra original) | 29,1% (51%, −12 g) | 2,9% (66%, **+194 g**) | 14% (65%) | 22,2 / 25,0% | 40,3 / 54 / 3% |
| 1' + ECUSTFD (só na massa) | 28,2% (50%) | 2,9% (**28%, +47 g**) | 0% (31%) | 20,8 / 26,4% | 43,8 / 53 / 21% |
| 2a só lateral, mesmos pratos | 34,0% (46%, −56 g) | **29,4%** (14%, −16 g) | 14% (23%) | 31,9 / 31,2% | 47,2 / 51 / 38% |
| 2b zenital + lateral, mesmos pratos | 34,0% (44%) | 26,5% (16%) | 14% (29%) | 31,2 / 32,6% | **49,3** / 54 / 38% |
| 2c + pratos só com vídeo | **41,7%** (47%) | 20,6% (18%, 0 g) | 14% (29%) | 35,4 / 34,7% | 46,5 / 54 / 26% |
| **2d (escolhida pela CV)** | **41,7%** (46%, −64 g) | 26,5% (21%, −27 g) | 0% (20%) | **36,1 / 36,1%** | 46,5 / 51 / 35% |
| 2d + ECUSTFD (só na massa) | 41,7% (47%) | 26,5% (23%) | 0% (23%) | 36,1 / 38,2% | 47,9 / 52 / 38% |

**Bootstrap pareado**, reamostrando pratos, 4.000 sorteios, da configuração 1 (seção 2) para a 2d (escolhida pela CV):

| Medida | 1 → 2d | Δ (IC 95%) |
|---|---|---|
| Teste zenital, direto | 47,9 → 50,0% | +2,1 (−2,9 a +6,8): **ruído** |
| Teste lateral, 1 quadro, direto | 44,1 → 48,6% | **+4,5 (+1,5 a +7,7)** |
| Teste híbrido √(m·LLM), zenital | 58,9 → 58,9% | 0,0 (−3,2 a +3,2) |
| Teste híbrido √(lateral 1 quadro·LLM) | 56,1 → 59,1% | +3,0 (−0,2 a +6,4) |
| Brasileiros, direto | 22,2 → 36,1% | **+13,9 (+4,9 a +22,9)** |
| Liv Up, híbrido √ | 2,9 → 35,3% | **+32,4 (+17,6 a +47,1)** |
| UFPR, híbrido √ | 54,4 → 51,5% | −2,9 (−11,7 a +5,8) |

### 7.3 O que os números mostram

1. **O ângulo era parte do problema de escala, e os quadros laterais corrigem isso.** O modelo só zenital, aplicado a um quadro lateral, superestima a massa (MAPE 45%, viés +52 g) e cai para 44%. Com quadros laterais no treino, o erro de massa lateral cai para 25–27% e o critério sobe para ~49%, praticamente o nível do zenital. No zenital, somar quadros laterais **não atrapalha** (MAPE 21–22% em todas as configurações com zenital). Treinar **só** com lateral, porém, estraga o zenital (MAPE 40–44%). O certo é misturar as duas fontes.
2. **O ganho no zenital é ruído.** De 47,9% para 50,0% (IC −3 a +7), e o híbrido C no zenital fica igual (~59–60%). Duplicar os pratos de treino com os que só têm vídeo (1.435 → 3.053 pratos distintos) quase não muda nada, o que confirma o platô da curva de aprendizado (seção 3).
3. **Comida brasileira: o maior ganho, mas ainda abaixo do LLM.** O modelo só de imagem vai de 22% para 36% (+14, IC +5 a +23). A Liv Up é a que mais muda: o modelo deixa de "ver" uma marmita de estúdio como um prato de três vezes o peso (viés de +194 g para −27 g; erro de massa de 66% para 21%). No UFPR (+13 pontos) o viés vira **−64 g**: o modelo agora subestima porções de alimento único fotografadas a 45°. **Mesmo assim, 36% fica abaixo dos 40% do LLM nas mesmas fotos.**
4. **Híbrido no Brasil:** com a massa nova, o híbrido √(m·LLM) vai de 40% para 46–49% nos brasileiros (Liv Up 3% → 35–38%), acima do LLM sozinho (40%). No UFPR fica igual (51–54%). Os itens brasileiros são 144, então o IC passa de ±8 pontos; Seara (n = 7) é anedótico.
5. **Foto de celular de lado + LLM:** com um único quadro lateral, o híbrido chega a **59,1%** (LLM sozinho nos mesmos pratos: 55,7%). Isso é igual ao híbrido zenital, ou seja, a vantagem do híbrido **sobrevive à mudança de ângulo**, o que não acontecia com o modelo só zenital (56,1%). Ressalva: a densidade veio do LLM olhando a foto zenital. Com 4 vistas do mesmo prato (média), o híbrido chega a 63%. Seria uma opção de produto ("fotografe de dois lados"), mas é **ganho de múltiplas vistas**, não de treino, e não fecha a distância até 70%.
6. **ECUSTFD (só massa, pesquisa):** corrige o viés de massa do modelo zenital na Liv Up (+211 g → +47 g; erro de 71% para 28%), mas não melhora o critério de proteína em nenhuma fonte, e com a 2d não acrescenta nada. Como a licença não foi declarada, **não serve para o produto**, e o efeito também não justifica pedir permissão aos autores.
7. **Isso nos aproxima de 70%? Pouco.** O melhor número honesto continua no híbrido, com ~59–60% no teste zenital e ~59% numa foto lateral isolada. Os quadros laterais tornam o modelo de peso **robusto ao ângulo** e tiram o pior modo de falha no Brasil (Liv Up), mas não reduzem o erro de massa abaixo de ~20%, que é o que a análise de oráculo (seção 2, item 5) pede para chegar a ~70%. A recomendação da seção 6 continua valendo: seletor de porção para o número final, e o modelo (agora treinado com zenital + lateral) só como **valor inicial** do seletor. Se houver coleta brasileira, ela deve incluir **vários ângulos por prato**: o resultado acima mostra que a variedade de ângulos no treino é o que tira o viés de escala.

## 8. Uma única foto de cima (10/11/2026)

**Premissa do produto:** o app usa sempre **uma** foto tirada bem de cima. **Pergunta:** quanto dá para subir a precisão da proteína sob essa premissa?
**Resposta curta: sobe de verdade no modelo só de imagem, pouco no híbrido, e a meta de 70% não foi atingida.** Todos os números são do `teste` (280 pratos), com a configuração escolhida só por validação cruzada no treino.

| Medida no `teste` | Linha de base (2d, seção 7) | Seção 8, escolhido pela CV | Diferença (IC 95%, bootstrap pareado) |
|---|---|---|---|
| Só imagem, sem LLM (dois backbones) | 50,0% | **61,1%** (IC 55,4 a 66,8) | **+11,1 (+4,6 a +17,2)** |
| Só imagem, sem LLM (um backbone, o modelo exportado) | 50,0% | **61,1%** (IC 55,4 a 66,8) | **+11,1 (+5,4 a +16,8)** |
| Híbrido: massa do modelo + densidade do LLM | 59,6% | **62,9%** (IC 57,1 a 68,6) | +3,2 (−2,1 a +8,6): não distinguível |
| Erro de massa (MAPE) | 22,1% | 16,9% | |
| LLM sozinho (referência) | 55,0% | | |
| **Meta** | **70%** | | |

- O modelo **só de imagem** passou de 50% para 61%, um ganho que o intervalo sustenta. Na estimativa pontual ele fica acima do LLM em nuvem (55%), mas essa diferença não é segura: +6,1 (−1,1 a +13,2). É o primeiro pipeline **totalmente no aparelho** deste estudo que não perde para o LLM.
- O **híbrido** escolhido ficou em 62,9%. Outras formas e outros modelos de massa quase empatados na escolha deram de 62% a 66% no mesmo teste. O valor honesto é "entre 62% e 66%, com IC de 57% a 72%". Contra o LLM sozinho o ganho é claro: +7,9 (+1,8 a +14,3).
- **Faltam de 4 a 9 pontos para 70%**, e o erro que sobra mudou de lugar: com o novo modelo de massa, 81% dos pratos já teriam a proteína certa se a densidade fosse a real. Com a massa real e a densidade do LLM seriam 71%. **O gargalo agora é mais a densidade (saber que alimento é e quanta proteína ele tem por grama) do que o peso.**

### 8.1 Dados, higiene e regras de escolha

- **Entrada do modelo:** uma única foto RGB zenital. Profundidade e quadros laterais só entram como sinal de treino (8.2, itens 4 e 10), nunca na inferência.
- **Treino zenital maior, mesma regra estrita.** Baixei as 631 fotos zenitais que faltavam (o Nutrition5k tem 3.490 pratos com foto de cima). Passam nos filtros do v2 e estão fora da avaliação 2.918 pratos. Com a regra de cadeia da seção 7 (cadeias de < 120 s calculadas sobre os 5.006 pratos com metadado; sai toda cadeia que contém prato de avaliação) sobram **1.684 pratos em 511 cadeias**: 1.435 do split oficial de treino, 166 do split oficial de teste e 83 sem split (café 2). Os 450 `dish_id` de avaliação continuam fora, e a interseção treino × avaliação é 0 (o script verifica com `assert`).
- **Validação cruzada:** 5 folds por cadeia (`GroupKFold`), só no treino. Como `calib` e `teste` foram sorteados com o mesmo número de pratos em cada faixa de proteína (0–5, 5–15, 15–25, 25–40, > 40 g) e o treino tem muito mais pratos com pouca proteína (38% com ≤ 5 g), o critério de CV é **ponderado para as cinco faixas terem o mesmo peso**. Sem isso, a CV premiaria o modelo que acerta pratos fáceis.
- **Regras de escolha, fixadas antes de olhar o teste:**
  1. modelo de **massa** (entrada do híbrido): menor RMSE de log(massa) na CV;
  2. pipeline **só de imagem**: maior critério ponderado na CV, entre todas as configurações e rotas (direta, massa × densidade, soma por grupo);
  3. **forma do híbrido** e seus pesos: só no `calib` (120 pratos), com validação deixa-fora em 12 blocos;
  4. limiar de confiança: só no `calib`.
- **Teste:** cada configuração foi avaliada no `teste` com o mesmo script (`zen_avalia.py`), e **todas** aparecem na tabela de 8.3, inclusive as que perderam. O script roda de ponta a ponta, então as configurações que ficaram prontas primeiro foram reavaliadas (com o mesmo resultado) quando as outras entraram. Nenhuma escolha usou o teste. O ajuste fino usou uma única receita (lr 2e-5, 8 épocas), decidida antes de qualquer resultado; o fold 0 serviu só para decidir se valia rodar os outros quatro.
- **Uma mudança depois da primeira rodada completa, registrada aqui:** três conjuntos tinham sido declarados com um terceiro backbone (TIPS) que não foi extraído por falta de tempo. Na primeira rodada eles ficaram de fora e a CV escolheu como modelo de massa a média "ajuste fino + 384×512" (híbrido escolhido: 66,1% no teste). Incluí depois os mesmos três conjuntos sem o membro TIPS, sem mudar nenhuma regra; a CV passou a escolher o de 4 membros, e o híbrido escolhido caiu para 62,9%. O número que vale é o da regra aplicada a todas as configurações: **62,9%**.
- **Linha de base para os intervalos:** configuração 2d da seção 7 (SigLIP 2 B/16 + ridge, zenital + quadros laterais), refeita por `zen_base2d.py` com previsões por prato: 50,0% só imagem, **59,6% no híbrido** massa × densidade do LLM, erro de massa (MAPE) de 22,1%. Intervalos por bootstrap pareado, 4.000 sorteios de pratos.
- **Incerteza:** com 280 pratos, a meia-largura do IC de 95% de um resultado isolado é de ~6 pontos, e a de uma diferença pareada, ~5 a 6 pontos. Diferenças de 2 ou 3 pontos entre configurações **não são distinguíveis** aqui.

### 8.2 O que foi testado e o que cada coisa deu

Critério de CV = critério do produto ponderado por faixa de proteína (8.1). "RMSE" = erro de log(massa) na CV; menor é melhor. O ponto de partida é o SigLIP 2 B/16 a 224 px com ridge e uma orientação, a mesma receita das seções 2 e 7 no treino novo (RMSE 0,2775; critério 54,6%; no teste, 50,0% e MAPE de massa de 21,1%).

| # | Ideia | O que foi feito | Resultado na CV | Veredito |
|---|---|---|---|---|
| 1 | Invariância a rotação e espelho | As 8 orientações do quadro (grupo diedral) como amostras de treino e média das 8 na previsão | RMSE 0,2775 → 0,2551; critério 54,6 → 56,6%. Só a média na previsão, sem treinar com elas: 0,2688 | **Ajudou** (massa). Teste: MAPE 21,1 → 18,6% |
| 2a | Features espaciais agregadas | GeM e médias numa grade 2×2 dos tokens, somadas ao embedding | RMSE 0,2533 e 0,2544 contra 0,2551 | Não ajudou (ruído) |
| 2b | Resolução maior | SigLIP 2 B/16 a 384×512 (proporção original, pesos de 512 px), 2 orientações | Com 1 orientação: RMSE 0,2775 → 0,2576, critério 54,6 → 59,4%. Com 2: 0,2471 e 59,7% | **Ajudou** (massa e identificação) |
| 3a | Multitarefa com cabeça não linear | MLP de uma camada oculta prevendo massa, kcal, gordura, carboidrato, proteína, densidade e massa e proteína por grupo | Critério 56,6 → 60,5%; RMSE de massa 0,2591 (igual ou pior) | **Ajudou** a proteína, não a massa |
| 3b | Proteína a partir da massa por grupo | Σ massa prevista do grupo × densidade média do grupo; e Σ proteína prevista por grupo | 40–44% e 49–56%, contra 55–62% da regressão direta | **Piorou**. Direta e massa × densidade empatam |
| 4 | Profundidade privilegiada | Atributos de volume e altura da profundidade zenital como alvos auxiliares do MLP | Critério 60,5 → 59,5%; RMSE 0,2591 → 0,2642 | Não ajudou (como na seção 2-B) |
| 5 | Ajuste fino parcial | 3 últimos blocos do SigLIP 2 + agregação por atenção, cabeça iniciada no ridge, lr 2e-5, 8 épocas, multitarefa, 4 orientações | Critério 56,6 → 59,4%; RMSE 0,2548 (igual). Erro de log-densidade 0,377 → 0,354 | **Ajudou** a densidade, não a massa |
| 6 | Conjuntos | Média em log das previsões de cabeças e backbones diferentes | RMSE 0,2551 → 0,2329 (4 membros); critério até 62,2% | **Ajudou**: foi o que mais baixou o erro de massa |
| 7 | Híbrido melhor | 9 formas de combinar o modelo com a saída por componente do LLM, pesos no `calib` | Ver 8.5 | Mistura simples ajuda pouco; por grupo **piorou** |
| 8 | Escala (celular) | Zoom artificial, escala aleatória no treino, recorte pelo prato | Ver 8.7 | Problema real, com solução barata dentro do Nutrition5k |
| 9 | Confiança | P(acerto) por regressão logística sobre a discordância entre rotas e cabeças | Ver 8.6 | Funciona, mas separa sobretudo pratos com pouca proteína |
| 10 | Quadros laterais como sinal de treino | Ridge com 8 orientações zenitais + 11.996 quadros laterais permitidos | RMSE 0,2551 → 0,2518; critério 56,6 → 57,3% | Ganho pequeno na CV, nenhum no teste (51,8%) |
| 11 | Fator de escala global | Multiplicar a previsão por c escolhido para maximizar o critério | CV com ajuste cruzado 62,2 → 61,7% | Não adotado |

Não rodados por falta de tempo de CPU, e por isso sem conclusão: um terceiro backbone (TIPS B/14, do tipo DINOv2; os pesos estão acessíveis em `gs://tips_data` e o carregador está em `zen_extrai.py`), a cabeça aditiva por token (`zen_finetune.py --cabeca aditiva`), ajuste fino na resolução maior, 8 orientações na resolução maior e destilação de um professor RGB-D. O ajuste fino completo do backbone não cabe em CPU: cada época do ajuste parcial já leva ~2 min.

**Por que o ajuste fino da EfficientNet-B0 (seção 2-D) tinha ido mal.** Repeti a receita antiga com validação a cada época num fold separado por cadeia (`zen_diag_effb0.py`). A perda de validação **cai o tempo todo** (0,92 → 0,15 em 8 épocas) e nunca sobe: não era o modelo decorando o treino. Era treino insuficiente a partir de uma cabeça aleatória com passo alto. Ao fim das 8 épocas o erro de log-massa na validação é 0,39, pior que o ridge sobre a **mesma** EfficientNet congelada (0,30) e que o ridge sobre o SigLIP 2 (0,27); o critério fica em 43% contra 51% e 54%. Daí a receita desta seção: backbone mais forte, só os blocos de cima, passo baixo e cabeça **iniciada na solução do ridge**, de modo que o treino parte do melhor modelo congelado. No fold de validação o critério saiu de 57,4% e chegou a 65,4%; nos outros quatro folds o ganho foi menor (+1,7, +5,8, +3,1 e −2,4 pontos contra o ridge de 8 orientações), e a média dos cinco é a que vale: 56,6 → 59,4%.

### 8.3 Todas as configurações avaliadas

Uma linha por configuração, em ordem de erro de massa na CV (do pior para o melhor). "direto" = regressão de log(proteína); "m×d" = massa × densidade previstas; "geo" = média geométrica das duas. As diferenças são contra a linha de base 2d: 50,0% só imagem e 59,6% no híbrido. **[massa]**, **[só imagem]** e **[1 backbone]** marcam o que a CV escolheu. "384×512" é o SigLIP 2 B/16 na resolução maior, com ridge e 2 orientações. As quatro variantes de MLP e as médias são todas as que foram rodadas; os três últimos conjuntos tinham sido declarados com um terceiro backbone (TIPS) que não chegou a ser extraído, e entram sem esse membro.

| Configuração | CV: RMSE log-massa | CV: critério (direto / m×d / geo) | Teste, só imagem (direto / m×d / geo) | Δ direto vs 2d só imagem (IC 95%) | Teste: MAPE massa | Teste: massa × dens. LLM | Δ vs 2d híbrido (IC 95%) |
|---|---|---|---|---|---|---|---|
| SigLIP 2 224 + ridge, 1 orientação (como nas seções 2 e 7) | 0,2775 | 54,6 / 53,1 / 53,7% | 50,0 / 51,4 / 50,7% | +0,0 (−5,0 a +5,0) | 21,1% | 60,0% | +0,4 (−4,3 a +5,0) |
| SigLIP 2 224 + ridge, treino com 1 orientação, previsão com 8 | 0,2688 | 56,0 / 55,6 / 55,7% | 55,0 / 51,8 / 53,6% | +5,0 (−1,1 a +11,1) | 20,0% | 62,9% | +3,2 (−1,8 a +8,2) |
| SigLIP 2 224 + ridge, 2 orientações | 0,2662 | 57,0 / 55,2 / 55,9% | 51,4 / 51,8 / 51,1% | +1,4 (−4,3 a +6,8) | 20,4% | 60,7% | +1,1 (−3,9 a +6,1) |
| SigLIP 2 224 + MLP + alvos auxiliares de profundidade, 8 orientações | 0,2642 | 59,5 / 58,5 / 59,0% | 60,0 / 61,4 / 61,1% | +10,0 (+3,9 a +16,1) | 20,0% | 60,0% | +0,4 (−5,0 a +5,7) |
| SigLIP 2 224 + MLP (1.024 unidades, dropout 0,5), 8 orientações | 0,2640 | 60,1 / 60,1 / 60,5% | 59,6 / 60,4 / 59,6% | +9,6 (+3,2 a +15,7) | 19,9% | 60,0% | +0,4 (−4,3 a +5,0) |
| SigLIP 2 224 + ridge, 4 orientações | 0,2626 | 56,6 / 55,3 / 55,7% | 51,8 / 53,6 / 52,9% | +1,8 (−3,9 a +7,5) | 20,1% | 61,4% | +1,8 (−3,2 a +6,8) |
| SigLIP 2 224 + MLP multitarefa, 8 orientações | 0,2591 | 60,5 / 60,5 / 60,6% | 63,6 / 62,5 / 62,1% | +13,6 (+7,1 a +19,6) | 19,4% | 59,3% | −0,4 (−5,4 a +5,0) |
| SigLIP 2 224 + MLP (decaimento 0,1), 8 orientações | 0,2588 | 60,5 / 60,6 / 60,6% | 62,9 / 62,1 / 62,5% | +12,9 (+6,4 a +19,3) | 19,4% | 59,3% | −0,4 (−5,4 a +5,0) |
| SigLIP 2 384×512 + ridge, 1 orientação | 0,2576 | 59,4 / 57,8 / 59,2% | 58,2 / 59,3 / 58,6% | +8,2 (+2,1 a +14,3) | 19,3% | 64,3% | +4,6 (−0,4 a +10,0) |
| SigLIP 2 224 + ridge, 8 orientações | 0,2551 | 56,6 / 54,2 / 54,9% | 55,4 / 57,5 / 58,2% | +5,4 (−0,4 a +11,1) | 18,6% | 62,9% | +3,2 (−1,8 a +8,2) |
| SigLIP 2 224, ajuste fino parcial (3 blocos), 4 orientações | 0,2548 | 59,4 / 58,3 / 59,1% | 57,5 / 56,8 / 57,5% | +7,5 (+1,4 a +13,6) | 18,9% | 61,1% | +1,4 (−3,9 a +6,8) |
| SigLIP 2 224 + ridge, 8 orientações + escala aleatória no treino | 0,2545 | 55,9 / 54,4 / 55,5% | 56,4 / 57,1 / 57,9% | +6,4 (+0,7 a +12,1) | 18,5% | 62,1% | +2,5 (−2,5 a +7,1) |
| SigLIP 2 224 + ridge, 8 orientações + GeM + grade 2×2 | 0,2544 | 57,3 / 55,1 / 56,6% | 57,1 / 56,8 / 58,2% | +7,1 (+1,1 a +12,9) | 18,9% | 64,6% | +5,0 (−0,4 a +10,4) |
| SigLIP 2 224 + ridge, 8 orientações + GeM | 0,2533 | 56,9 / 54,8 / 55,8% | 56,4 / 57,1 / 57,1% | +6,4 (+0,7 a +12,5) | 19,1% | 63,9% | +4,3 (−1,1 a +9,6) |
| SigLIP 2 224 + ridge, 8 orientações + quadros laterais no treino | 0,2518 | 57,3 / 56,5 / 56,9% | 51,8 / 52,9 / 52,5% | +1,8 (−3,6 a +7,1) | 19,0% | 62,5% | +2,9 (−1,4 a +6,8) |
| Média: ajuste fino + ridge ridge 224 | 0,2491 | 58,7 / 58,2 / 58,3% | 57,9 / 56,4 / 57,5% | +7,9 (+2,1 a +13,6) | 18,2% | 63,9% | +4,3 (−1,1 a +9,3) |
| SigLIP 2 384×512 + ridge, 2 orientações + GeM + grade 2×2 | 0,2478 | 60,2 / 59,2 / 59,2% | 58,9 / 55,4 / 56,4% | +8,9 (+2,1 a +15,4) | 19,0% | 64,3% | +4,6 (−1,1 a +10,4) |
| SigLIP 2 384×512 + ridge, 2 orientações | 0,2471 | 59,7 / 58,3 / 59,1% | 61,1 / 60,4 / 61,4% | +11,1 (+4,6 a +17,5) | 18,3% | 67,1% | +7,5 (+2,1 a +13,2) |
| Média: ridge 224 + mlp | 0,2468 | 59,9 / 58,9 / 58,7% | 60,0 / 58,9 / 59,6% | +10,0 (+3,9 a +16,1) | 18,0% | 61,4% | +1,8 (−2,9 a +6,4) |
| Média: ajuste fino + ridge 224 com laterais | 0,2428 | 60,0 / 57,4 / 58,6% | 55,7 / 57,1 / 58,2% | +5,7 (+0,0 a +11,4) | 18,0% | 62,5% | +2,9 (−1,8 a +7,5) |
| Média: ajuste fino + mlp | 0,2427 | 61,3 / 60,8 / 61,1% | 61,1 / 60,4 / 61,8% | +11,1 (+4,6 a +17,1) | 17,6% | 60,7% | +1,1 (−3,9 a +6,4) |
| Média: ridge 224 + 384×512 | 0,2401 | 60,1 / 59,1 / 60,0% | 60,7 / 58,9 / 60,7% | +10,7 (+4,6 a +16,8) | 17,1% | 65,7% | +6,1 (+1,1 a +11,1) |
| Média: ajuste fino + mlp + ridge 224 com laterais **[1 backbone]** | 0,2395 | 61,5 / 59,9 / 61,2% | 61,1 / 58,6 / 60,4% | +11,1 (+5,4 a +16,8) | 17,5% | 62,1% | +2,5 (−2,5 a +7,1) |
| Média: ajuste fino + ridge 224 + 384×512 | 0,2378 | 60,9 / 60,1 / 60,4% | 62,5 / 59,6 / 60,7% | +12,5 (+6,8 a +18,6) | 17,0% | 64,6% | +5,0 (+0,4 a +9,7) |
| Média: ajuste fino + 384×512 **[só imagem]** | 0,2351 | 61,9 / 61,3 / 62,2% | 61,8 / 58,6 / 61,1% | +11,8 (+5,7 a +17,9) | 16,7% | 65,4% | +5,7 (+0,4 a +11,1) |
| Média: ajuste fino + ridge 224 com laterais + 384×512 | 0,2338 | 61,5 / 60,8 / 60,8% | 61,8 / 60,4 / 60,0% | +11,8 (+6,1 a +17,5) | 16,9% | 65,4% | +5,7 (+1,1 a +10,4) |
| Média: ajuste fino + mlp + ridge 224 com laterais + 384×512 **[massa]** | 0,2329 | 61,8 / 61,5 / 61,7% | 62,1 / 60,7 / 62,1% | +12,1 (+6,4 a +17,9) | 16,9% | 63,9% | +4,3 (−0,4 a +8,9) |


Como ler: a coluna de CV decide, a de teste só relata. Várias configurações que a CV não escolheu têm número de teste maior que as escolhidas (o MLP sozinho dá 63,6% só de imagem; o ridge a 384×512 dá 67,1% no híbrido simples). Isso é o esperado quando as diferenças reais são menores que o ruído de 280 pratos, e é o motivo de não escolher pelo teste.

### 8.4 Pipeline só de imagem (totalmente no aparelho)

- **Escolhido pela CV (critério 62,2%):** média de duas previsões, o SigLIP 2 B/16 a 224 px com ajuste fino parcial e o SigLIP 2 B/16 a 384×512 com ridge, pela rota "geo". **Teste: 61,1%** (IC 55,4 a 66,8), MAE 6,9 g, viés −2,7 g, erro de massa de 16,7%. No `n5k_v1` (50 pratos): 62%.
- **Com um único backbone (CV 61,5%):** média de três cabeças sobre o mesmo SigLIP 2 B/16 a 224 px (ajuste fino parcial, MLP multitarefa e ridge treinado também com quadros laterais), rota direta. **Teste: 61,1%** (IC 55,4 a 66,8), MAE 7,3 g, viés −3,9 g, erro de massa de 17,5%; `n5k_v1`: 58%. É o modelo exportado em 8.9.
- **Contra as referências:** +11,1 pontos sobre o 2d só de imagem (IC +4,6 a +17,2), +6,1 sobre o LLM (IC −1,1 a +13,2) e +1,4 sobre o híbrido 2d (IC −5,4 a +8,2). Em palavras: melhor que o modelo treinado anterior, **no mesmo nível** do LLM e do híbrido anterior.
- **Por faixa de proteína real** (escolhido / um backbone): 0–5 g 91 / 91%; 5–15 g 73 / 80%; 15–25 g 48 / 55%; 25–40 g 43 / 34%; > 40 g 50 / 45%. O acerto alto vem dos pratos com pouca proteína, em que a tolerância de ±5 g resolve. **Nos pratos com mais de 15 g, que são os que importam para a usuária, o acerto fica entre 34% e 55%.**
- **Oráculos:** com a massa real e a densidade do modelo, 67,5%; com a massa do modelo e a densidade real, 81,4%. A densidade aprendida só da imagem ainda perde para a do LLM (71,4% com a massa real), mas a distância caiu: no ridge com 8 orientações esse oráculo era 59%.

### 8.5 Híbrido com o LLM

Massa: o conjunto de 4 membros escolhido pela CV (RMSE 0,2329; no teste, MAPE de 16,9% contra 22,1% do 2d e 30% do LLM). Densidade e proteína direta do modelo: as do pipeline só de imagem escolhido. Do LLM entram as gramas e a densidade por componente (`estimativas_v2_comp_casados.csv`). Pesos ajustados só no `calib`; a forma foi escolhida pelo critério no `calib` com validação deixa-fora.

| Forma | Calib (deixa-fora) | Teste | MAE / viés | Δ vs híbrido 2d (IC 95%) | Δ vs LLM (IC 95%) | n5k v1 |
|---|---|---|---|---|---|---|
| Massa do modelo × densidade do LLM (a forma da seção 2) | 63,3% | 63,9% | 6,6 g / −2,6 g | +4,3 (−0,4 a +8,9) | +8,9 (+2,5 a +15,4) | 54% |
| √(massa do modelo · massa do LLM) × densidade do LLM | 62,5% | 61,8% | 7,2 g / −3,4 g | +2,1 (−3,2 a +7,5) | +6,8 (+2,9 a +11,1) | 60% |
| Massa modeloᵃ · LLM¹⁻ᵃ × densidade do LLM (1 peso) | 61,7% | 63,2% | 6,6 g / −2,8 g | +3,6 (−1,1 a +8,2) | +8,2 (+2,1 a +14,3) | 54% |
| **Massa misturada × densidade misturada (2 pesos) [escolhida]** | 65,0% | **62,9%** | 6,3 g / −3,3 g | +3,2 (−2,1 a +8,6) | +7,9 (+1,8 a +14,3) | 60% |
| Mistura log-linear de massas e densidades (5 pesos) | 65,0% | 65,4% | 5,9 g / −1,2 g | +5,7 (0,0 a +11,4) | +10,4 (+3,6 a +17,1) | 66% |
| Mistura log-linear + proteína direta do modelo e do LLM (7 pesos) | 64,2% | 65,4% | 5,9 g / −1,0 g | +5,7 (0,0 a +11,8) | +10,4 (+3,6 a +17,1) | 70% |
| Por grupo: gramas do LLM misturadas com a massa por grupo do modelo × densidade do LLM | 55,0% | 56,1% | 8,5 g / −4,6 g | −3,6 (−9,6 a +2,1) | +1,1 (−2,9 a +4,6) | 54% |
| Só proteína animal: gramas do LLM misturadas com as do modelo | 55,0% | 53,9% | 8,4 g / −4,4 g | −5,7 (−12,1 a +0,4) | −1,1 (−5,0 a +2,9) | 56% |
| Média geométrica da proteína do modelo e do LLM | 63,3% | 61,1% | 6,9 g / −3,8 g | +1,4 (−4,6 a +7,5) | +6,1 (+0,4 a +11,4) | 64% |

1. **A forma escolhida dá 62,9%** (IC 57,1 a 68,6). Ela empatou no `calib` com a mistura de 5 pesos (65,0% cada, ou 78 de 120 pratos); fiquei com a de menos pesos. A outra dá 65,4% no teste. Com 120 pratos, o `calib` não separa formas que diferem em 2 ou 3 pontos.
2. **Contra o híbrido 2d (59,6%) o ganho é de +2 a +6 pontos conforme a forma, e o intervalo inclui zero em quase todas** (nas duas misturas log-lineares o limite inferior é exatamente zero). Contra o LLM sozinho (55,0%), todas as formas simples ganham com folga.
3. **O modelo de massa quase não muda o híbrido dentro do ruído.** Com outro conjunto quase empatado na CV (ajuste fino + 384×512, RMSE 0,2351 contra 0,2329), a mesma regra escolhe a mesma forma e o teste dá 66,1%. Três pontos de diferença entre dois modelos de massa praticamente iguais é ruído de amostra, não efeito.
4. **Usar a massa por grupo do modelo piorou** (54–56%). No `calib`, trocar as gramas do LLM pelas gramas reais de cada grupo levaria a 87,5%, e só as da proteína animal, a 75%. O ganho potencial existe, mas a massa por grupo que o modelo prevê é ruidosa demais para substituir a do LLM. Seria preciso um modelo que segmente os componentes, o que o Nutrition5k não tem rótulo para treinar.
5. **Por faixa de proteína real** (híbrido escolhido / LLM sozinho): 0–5 g 89 / 84%; 5–15 g 79 / 71%; 15–25 g 62 / 57%; 25–40 g 38 / 36%; > 40 g 46 / 27%. O híbrido melhora mais onde o LLM é pior (pratos grandes), mas **entre 25 e 40 g ninguém passa de 40%**.
6. **Oráculos no teste:** massa real × densidade do LLM, 71,4%; massa do modelo × densidade real, 81,4%; massa do LLM × densidade real, 62,9%. O erro de massa do modelo já é o menor dos dois problemas.

### 8.6 Confiança: quanto dá para sugerir sozinho

Uma regressão logística estima a probabilidade de a sugestão cair dentro da tolerância. No modelo só de imagem ela usa a proteína prevista, a discordância entre as rotas (direta, massa × densidade, soma por grupo) e a dispersão entre as cabeças do conjunto, e é ajustada nas previsões fora-do-fold do **treino**. No híbrido usa a discordância entre modelo e LLM em massa e em densidade e a confiança declarada pelo LLM, e é ajustada no `calib`. O limiar para "≥ 70% de acerto entre os aceitos" foi escolhido no `calib`; o teste só relata.

| Fração mais confiante do teste | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|
| Só imagem: acerto | 96% | 91% | 82% | 76% | 70% | 68% | 64% | 62% | 62% | 61% |
| Híbrido: acerto | 96% | 93% | 86% | 81% | 79% | 73% | 69% | 66% | 65% | 63% |

| | Limiar do calib: cobertura no teste | Acerto nos aceitos | Acerto nos recusados | Maior cobertura com ≥ 70% no teste (descritivo) |
|---|---|---|---|---|
| Só imagem | 52% | 69,4% | 51,9% | 50% |
| Híbrido | 80% | 65,5% | 52,6% | 65% |

1. **Cerca de metade dos pratos pode ser sugerida sozinha com ~70% de acerto** no modelo só de imagem (52% dos pratos, 69,4%). No híbrido, de 60% a 65% dos pratos chegam a 70–73%, mas o limiar tirado dos 120 pratos do `calib` aceitou demais (80% dos pratos, 65,5% de acerto). Para fixar esse limiar com segurança é preciso um conjunto de calibração maior.
2. **A probabilidade está só razoavelmente calibrada.** Em quatro faixas de mesma contagem, previsto → observado: só imagem 38→56, 56→49, 71→53, 89→87; híbrido 46→54, 63→40, 74→67, 87→90. Os extremos funcionam e o meio não. Serve para separar "sugerir" de "pedir confirmação", não para mostrar um percentual à usuária.
3. **O que a confiança separa é, em grande parte, prato com pouca proteína de prato com muita.** No modelo só de imagem, os aceitos têm em média 11 g de proteína real e os recusados, 36 g; 69% dos aceitos têm até 15 g. Entre os aceitos com mais de 15 g (46 pratos), o acerto é de **30%**. No híbrido: 17 g contra 46 g, e 47% de acerto nos aceitos com mais de 15 g (112 pratos). Ou seja: a sugestão automática "confiável" cobre bem lanches, frutas e saladas, e **não** cobre o prato com carne, que é onde a usuária mais precisa do número. Para esses pratos o seletor de porção continua necessário.

### 8.7 Robustez de escala: o que acontece no celular

No Nutrition5k a câmera fica a altura fixa, então o tamanho em pixels equivale ao tamanho real. No celular a distância varia. Medi isso no `teste` com zoom artificial (recorte central para "mais perto", mais mesa em volta para "mais longe") e com o recorte pelo prato, que imita o círculo-guia do app. Para o recorte, a caixa do prato veio do mapa de profundidade (`zen_prato.py`); a profundidade só faz o papel da usuária que enquadra o prato, e o modelo recebe apenas o recorte RGB. Tudo com SigLIP 2 B/16, ridge e uma orientação, para isolar o efeito da escala.

| Treino | Entrada no teste | Só imagem (direto) | Erro de massa (MAPE) | Viés mediano da massa | Massa × densidade do LLM |
|---|---|---|---|---|---|
| (a) quadro inteiro, escala fixa | zoom 0,70 (30% mais longe) | 46,8% | 28,7% | −26% | 46,4% |
| | zoom 0,85 | 48,9% | 22,1% | −12% | 55,0% |
| | **sem zoom** | 50,0% | 21,1% | −2% | 60,0% |
| | zoom 1,15 | 55,7% | 29,3% | +16% | 60,7% |
| | zoom 1,30 (30% mais perto) | 53,9% | 41,2% | +31% | 53,6% |
| (b) + 2 cópias por prato com escala aleatória (0,7 a 1,3) | zoom 0,70 | 48,2% | 22,4% | −3% | 57,9% |
| | zoom 0,85 | 52,5% | 20,8% | −1% | 61,4% |
| | **sem zoom** | 50,0% | 20,2% | −3% | 58,9% |
| | zoom 1,15 | 54,6% | 20,4% | −3% | 61,8% |
| | zoom 1,30 | 56,8% | 22,4% | +4% | 59,6% |
| (c) recorte pelo prato (prato ocupa 90% do lado) | enquadramento exato | 54,3% | 21,7% | −3% | 62,5% |
| | erro de enquadramento de ±10% | 55,4% | 23,0% | −4% | 60,4% |
| | erro de enquadramento de ±30% | 52,1% | 25,9% | −2% | 57,9% |
| (d) recorte pelo prato + erro de ±10% no treino | enquadramento exato | 55,4% | 21,4% | −3% | 61,1% |
| | erro de ±10% | 54,6% | 22,3% | −4% | 59,6% |
| | erro de ±30% | 55,0% | 24,9% | −1% | 57,9% |
| (e) 8 orientações + escala aleatória | zoom 0,70 / 0,85 / 1 / 1,15 / 1,30 | 51,4 / 50,4 / 52,1 / 56,1 / 52,5% | 21,6 / 20,8 / 19,6 / 19,9 / 24,0% | −3 a +6% | 58,6 / 59,3 / 61,4 / 61,8 / 59,3% |

1. **O modelo treinado só na escala do Nutrition5k não pode ir para o celular como está.** Com a foto 30% mais longe, ele subestima a massa em 26%; 30% mais perto, superestima em 31%. O híbrido cai de 60% para 46–54%. O erro de massa cresce quase na proporção do zoom, não do quadrado dele: o modelo já se apoia em parte no prato, não só na área em pixels.
2. **Treinar com escala aleatória resolve isso dentro do Nutrition5k, sem custo.** O erro de massa fica em 20–22% em toda a faixa de ±30% (viés entre −3% e +4%), e na escala original não piora (20,2% contra 21,1%). Combinado com as 8 orientações (linha e), a CV fica igual à da configuração sem escala (RMSE de log-massa 0,2545 contra 0,2551).
3. **O recorte pelo prato custa pouco:** com o prato ocupando 90% do quadro, o erro de massa é 21,7% (21,1% no quadro inteiro), e o critério não cai. Um erro de enquadramento de ±10% custa ~1 ponto de MAPE; de ±30%, ~4 pontos de MAPE e ~4 pontos no híbrido. Treinar com esse erro (linha d) recupera uma parte.
4. **Ressalva que o Nutrition5k não deixa medir:** o conjunto tem poucos tipos de prato e tigela, sempre os mesmos. O modelo com escala aleatória e o do recorte pelo prato aprendem o tamanho real **desses** pratos e usam o prato como régua. Na casa da usuária o prato pode ter 20 ou 28 cm, o que muda a área em quase 2×, e nenhuma foto de cima sem referência resolve essa ambiguidade. O resultado acima mostra que a **distância** deixa de ser problema quando o prato enche o guia. Ele não mostra que o **tamanho do prato** deixa de ser. Isso só se mede com fotos de celular em pratos variados, ou pedindo o diâmetro do prato uma vez no app.
5. **Quanto a precisão depende do guia:** com o prato dentro de ±10% do círculo, quase nada (≤ 2 pontos). Com ±30%, ~4 pontos no híbrido. Sem guia nenhum e sem aumento de escala, até 14 pontos.

### 8.8 Itens brasileiros (secundários sob a nova premissa)

Olhei as 144 fotos: **UFPR (103) e Liv Up (34) não são fotos de cima** (prato a ~45° e marmita em foto de estúdio inclinada). **Só as 7 da Seara são vistas de cima**, e são a arte recortada da bandeja sobre fundo branco, não uma foto de celular. Não existe, neste material, nenhum prato brasileiro fotografado de cima em condição real.

| Fonte | n | Só imagem (seção 8) | 2d só imagem | Erro de massa: seção 8 / 2d | LLM | Híbrido escolhido | √(modelo·LLM) × dens. LLM | 2d massa × dens. LLM |
|---|---|---|---|---|---|---|---|---|
| UFPR (a 45°) | 103 | 39,8% | 41,7% | 41% (viés −53 g) / 46% | 46,6% | 50,5% | 56,3% | 42,7% |
| Liv Up (estúdio, inclinada) | 34 | 2,9% | 26,5% | 61% (viés +182 g) / 21% | 23,5% | 5,9% | 8,8% | 35,3% |
| Seara (de cima, arte da embalagem) | 7 | 0% | 0% | 64% (viés +192 g) / 20% | 14,3% | 14,3% | 14,3% | 28,6% |
| Todos | 144 | 29,2% | 36,1% | | 39,6% | 38,2% | 43,1% | 40,3% |

- O modelo desta seção foi treinado para a foto de cima do Nutrition5k e **volta a errar a escala fora dela**: na marmita da Liv Up e na bandeja da Seara, que enchem o quadro, ele vê 180 a 190 g a mais. O 2d, treinado também com quadros laterais, não tinha esse viés. É o mesmo efeito de escala de 8.7, e é o preço de especializar o modelo na foto zenital a altura fixa.
- Mesmo nas 7 imagens de cima da Seara o resultado é ruim (0% só imagem). São só 7 e são arte de embalagem, então isso não mede o app; mostra apenas que "de cima" não basta quando a distância e o recipiente são outros.
- Sob a premissa nova, esta tabela não avalia o produto. O que falta é um conjunto brasileiro **fotografado de cima com celular**, com peso por componente.

### 8.9 Modelo final, tamanho e Core ML

`zen_final.py` treina e exporta o modelo de um backbone escolhido pela CV: SigLIP 2 B/16 a 224 px, com os 9 primeiros blocos compartilhados e dois topos de 3 blocos (o congelado, que alimenta o MLP e o ridge, e o ajustado). Ele confere que o modelo montado reproduz as previsões dos experimentos (diferença máxima de 0,001 em log nos 16 pratos conferidos).

| Item | Valor |
|---|---|
| Parâmetros | 122,1 M |
| Core ML `mlprogram`, FP16, iOS 17+ | **convertido**, 230,8 MB |
| Core ML com pesos em 8 bits | **convertido**, 115,9 MB (efeito na precisão **não medido**) |
| Latência em CPU x86, por orientação | 392 ms (1 thread), 132 ms (4 threads) |
| Orientações por foto | 4 no topo ajustado, 8 nos congelados, como foi avaliado |

- A conversão do ViT-B funcionou por `torch.jit.trace` com imagem de tamanho fixo e `coremltools` 9 (na seção 5 a do ViT-S tinha falhado; não investiguei a diferença). O `.mlpackage` não roda no Linux, então a latência no iPhone **não foi medida**. Com 8 orientações o custo é 8 passagens; em CPU x86 isso dá ~1 s com 4 threads.
- Saída: 22 valores em log por cabeça (massa, proteína, kcal, densidade, gordura, carboidrato, massa e proteína por grupo). O app faz a média das orientações e das cabeças e desfaz o log.
- O pipeline de dois backbones (224 px ajustado + 384×512) tem o mesmo resultado no teste (61,1%) e custaria um segundo ViT-B com 3,9 vezes mais tokens. Não vale o custo no aparelho. No servidor, como modelo de massa do híbrido, vale.
- Este modelo **não** tem aumento de escala nem recorte pelo prato (8.7). Antes de ir para o celular ele precisa ser treinado de novo com um dos dois.

### 8.10 Distância da meta e o que seria preciso

1. **Onde estamos:** só imagem 61,1% (IC 55 a 67); híbrido escolhido 62,9% (IC 57 a 69), com formas e modelos de massa equivalentes entre 62% e 66%. A meta é 70%. **Nenhuma configuração avaliada chegou a 70% no teste**; a maior foi 67,1%, não escolhida. O limite superior do intervalo das melhores encosta em 71–72%, então 70% não está excluído pela estatística, mas também não foi demonstrado.
2. **O que mudou em relação às seções 2 e 7:** o erro de massa caiu de 22% para 17% (orientações, resolução e conjunto) e o modelo só de imagem alcançou o LLM. O teto de "massa perfeita × densidade do LLM" continua em 71%. Mesmo zerando o erro de massa, o híbrido atual mal passaria da meta.
3. **O que seria preciso, em ordem de ganho esperado:**
   - **Densidade melhor.** É onde está a maior parte do erro que sobra: identificar a preparação (frango grelhado ou empanado, carne magra ou gorda) e a fração de proteína de pratos mistos. Caminhos: testar um LLM de visão mais forte ou com a foto em resolução maior só para a identificação; juntar a densidade do LLM com a do modelo por alimento, e não por prato.
   - **Massa por componente.** O oráculo do `calib` diz que acertar as gramas de cada grupo levaria a ~87%. Isso pede segmentação dos componentes, com rótulo de máscara ou de peso por componente em foto de cima, que hoje não existe.
   - **Mais do que já funcionou, com GPU:** ajuste fino na resolução maior, 8 orientações na resolução maior, um backbone do tipo DINOv2 (TIPS) no conjunto. Na CV, orientações, resolução e conjunto renderam de 2 a 5 pontos cada. Minha estimativa, não medida, é de mais 2 a 4 pontos somando esses passos. Não fecha a distância sozinho.
   - **Dados brasileiros de cima, com celular:** sem eles não há como saber se os 61–66% do Nutrition5k valem no app. Pelo que 8.7 e 8.8 mostram, a escala e o recipiente são o risco principal.
4. **Recomendação que se mantém:** o número final vem do seletor de porção (71–76% no v2). O modelo desta seção serve para (a) dar o valor inicial do seletor com erro de massa menor, (b) sugerir sozinho a metade dos pratos em que a confiança é alta, sabendo que essa metade é, na maior parte, de pratos com pouca proteína, e (c) ser a base de um pipeline sem nuvem, já no nível do LLM no Nutrition5k.

## Reproduzir

```
TD=<scratchpad>/treino   # dados, features e pesos ficam fora do repositório
python3 eval/treino/prep_dados.py $TD                     # treino/avaliação + checagem de sobreposição
python3 eval/treino/extrai_features.py $TD vits16_in21k siglip_b16 siglip2_b16 effb0
python3 eval/treino/features_profundidade.py $TD
OMP_NUM_THREADS=1 python3 eval/treino/experimentos.py      # A, B, C, curva de aprendizado, transferência BR
OMP_NUM_THREADS=1 python3 eval/treino/finetune.py 8        # D
python3 eval/treino/on_device.py                           # F
python3 eval/treino/extrai_features_lateral.py $TD siglip2_b16 --quadros 4 --shards 4   # seção 7 (~30 min)
OMP_NUM_THREADS=1 python3 eval/treino/experimentos_lateral.py siglip2_b16              # seção 7 (~17 min)
```
Os pesos devem estar em `$TD/pesos/`: `webli_en_b16_224_63724782.npz` e `siglip2_b16_224.npz` de `storage.googleapis.com/big_vision/…`, `S_16-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0.npz` de `gs://vit_models/augreg/` e `efficientnet_b0_ra-3dd342df.pth` do release `v0.1-weights` do timm. Nesta máquina compartilhada, um processo com uma thread foi o mais rápido; com 4 threads, o PyTorch ficou de 4 a 20 vezes mais lento.

Seção 8 (uma única foto de cima). As 631 fotos zenitais que faltavam vêm de `gs://nutrition5k_dataset/…/realsense_overhead/<dish_id>/{rgb,depth_raw}.png` para `$TD/img/`; os pesos extras, `siglip2_b16_512.npz`, de `storage.googleapis.com/big_vision/siglip2/`.
```
python3 eval/treino/zen_dados.py $TD                       # treino zenital estrito (1.684 pratos), alvos multitarefa, checagem de sobreposição
OMP_NUM_THREADS=1 python3 eval/treino/zen_base2d.py        # linha de base 2d com previsões por prato (bootstrap pareado)
python3 eval/treino/zen_profundidade.py $TD && python3 eval/treino/zen_prato.py $TD      # alvos auxiliares de profundidade; caixa do prato
python3 eval/treino/zen_extrai.py $TD siglip2_b16 --orient 8 --tokens 9                  # ~30 min; grava 2,7 GB de tokens
python3 eval/treino/zen_extrai.py $TD siglip2_b16_512 --orient 2                         # ~40 min
for z in 0.7 0.85 1.15 1.3; do python3 eval/treino/zen_extrai.py $TD siglip2_b16 --orient 1 --variante zoom:$z --conj teste --saida siglip2_b16_zoom$z; done
for s in 1 2; do python3 eval/treino/zen_extrai.py $TD siglip2_b16 --orient 1 --variante escala:$s --conj treino --saida siglip2_b16_escala$s; done
python3 eval/treino/zen_extrai.py $TD siglip2_b16 --orient 1 --variante prato:0.9 --conj treino,aval --saida siglip2_b16_prato0.9
python3 eval/treino/zen_extrai.py $TD siglip2_b16 --orient 1 --variante prato:0.9:0.1:1 --conj teste --saida siglip2_b16_prato0.9_ruido0.1
python3 eval/treino/zen_extrai.py $TD siglip2_b16 --orient 1 --variante prato:0.9:0.3:1 --conj teste --saida siglip2_b16_prato0.9_ruido0.3
python3 eval/treino/zen_extrai.py $TD siglip2_b16 --orient 1 --variante prato:0.9:0.1:2 --conj treino --saida siglip2_b16_prato0.9_ruido0.1_treino
cd eval/treino; export OMP_NUM_THREADS=4
python3 zen_experimentos.py ridge siglip2_b16 siglip2_b16_512    # ridge por orientação/blocos
python3 zen_experimentos.py lateral && python3 zen_experimentos.py mlp siglip2_b16
python3 zen_experimentos.py escala && python3 zen_experimentos.py escala_d4
python3 zen_finetune.py holdout                                  # fold 0, validação a cada época (~16 min)
for k in 1 2 3 4; do python3 zen_finetune.py fold --fold $k; done  # ~14 min cada
python3 zen_finetune.py final && python3 zen_finetune.py junta   # ~17 min
python3 zen_diag_effb0.py original 8                             # diagnóstico do ajuste fino antigo (~12 min)
python3 zen_avalia.py                                            # seleção por CV, teste, híbridos, bootstrap, confiança, Brasil, escala
python3 zen_final.py                                             # ponto de entrada único: treina o modelo final e exporta Core ML
```
`zen_final.py` chama sozinho `zen_dados.py`, `zen_extrai.py` e `zen_finetune.py final` se as saídas ainda não existirem. Tempo total de CPU nesta máquina (4 núcleos): cerca de 3 h.
