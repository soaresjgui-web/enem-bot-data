# Modelo treinado: proteína só pela foto, sem informar porção (09/11/2026)

**Pergunta:** um modelo **treinado** (e não um LLM em nuvem) consegue sugerir a proteína só pela foto e atingir a meta de **≥ 70% dos pratos dentro de ±20% ou ±5 g**?
**Resposta curta: não, com os dados que existem hoje.** Com higiene de dados rigorosa, o melhor modelo só de imagem fica em **~48–52%** no `teste` (280 pratos), abaixo do LLM (55%). Profundidade (RGB-D) não ajudou. O melhor resultado foi o **híbrido** "LLM identifica e dá a densidade, modelo treinado dá o peso", com **~60% no teste (59,6%) e 60–62% no n5k v1**: melhora o LLM em 5 pontos, mas ainda fica 10 pontos abaixo da meta. A comida brasileira em foto que não é zenital **quebra** o modelo de peso (Liv Up 3%). **Atualização (seção 7):** treinar também com os quadros laterais do Nutrition5k deixa o modelo robusto ao ângulo (foto lateral 44% → 49%) e leva o Brasil de 22% para 36% (Liv Up 3% → 26%), mas o híbrido continua em ~59–60%, longe de 70%.

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
