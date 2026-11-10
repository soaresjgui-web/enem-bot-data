# Modelo treinado: proteína só pela foto, sem informar porção (09/11/2026)

**Pergunta:** um modelo **treinado** (e não um LLM em nuvem) consegue sugerir a proteína só pela foto e atingir a meta de **≥ 70% dos pratos dentro de ±20% ou ±5 g**?
**Resposta curta: não, com os dados que existem hoje.** Com higiene de dados rigorosa, o melhor modelo só de imagem fica em **~48–52%** no `teste` (280 pratos), abaixo do LLM (55%). Profundidade (RGB-D) não ajudou. O melhor resultado foi o **híbrido** "LLM identifica e dá a densidade, modelo treinado dá o peso", com **~60% no teste (59,6%) e 60–62% no n5k v1**: melhora o LLM em 5 pontos, mas ainda fica 10 pontos abaixo da meta. A comida brasileira em foto que não é zenital **quebra** o modelo de peso (Liv Up 3%). **Atualização (seção 7):** treinar também com os quadros laterais do Nutrition5k deixa o modelo robusto ao ângulo (foto lateral 44% → 49%) e leva o Brasil de 22% para 36% (Liv Up 3% → 26%), mas o híbrido continua em ~59–60%, longe de 70%. **Atualização (seção 8, premissa de uma única foto de cima):** com 8 orientações da foto, resolução maior, ajuste fino parcial e média de modelos, o modelo **só de imagem sobe de 50% para 61%** no teste (IC 55 a 67; +11 pontos sobre o anterior, IC +5 a +17) e alcança o LLM (55%). O erro de massa cai de 22% para 17%. O **híbrido escolhido fica em 63%** (IC 57 a 69; formas equivalentes dão de 62% a 66%), sem diferença segura para os 59,6% de antes. **A meta de 70% continua não atingida**, e o erro que sobra está mais na densidade (que alimento é) do que no peso. Cerca de metade dos pratos pode ser sugerida sozinha com ~70% de acerto, mas essa metade é, na maior parte, de pratos com pouca proteína. Fora da distância fixa do Nutrition5k o modelo erra a escala, a menos que seja treinado com escala aleatória ou com o prato enquadrado num guia. **Atualização (seção 9, densidade de proteína):** prever a fração da massa do prato que é cada fonte de proteína, somar a recuperação de pratos parecidos e um terceiro backbone melhora a densidade do modelo de 67,5% para 76,1% quando a massa é a real (IC do ganho: +2,9 a +14,3). No resultado final o ganho é menor: **só imagem 61,1% → 64,6%** (IC 59 a 70; ganho não distinguível) e **híbrido 62,9% → 67,1%** (IC 61 a 72,5; ganho de +4,3, IC +0,4 a +8,6). Melhoram os pratos de 5 a 25 g; **os de mais de 25 g praticamente não melhoram** (38–46%), porque ali o erro de massa já pesa tanto quanto o de densidade. No LLM, o erro estava na parte da massa que é proteína e na fonte, não na tabela; a instrução v3 pede identificação e partes da massa em vez de gramas. **Com as estimativas v3 (uma rodada cega), o híbrido chega a 70,4% no teste** (IC 65,0 a 75,7; +7,5 sobre o híbrido da seção 8, IC +3,2 a +11,8; +3,2 sobre o híbrido com v2, IC 0,0 a +6,4). **É a primeira estimativa pontual na meta de 70%, e não é a meta demonstrada:** o intervalo começa em 65%, é uma rodada só em 280 pratos do Nutrition5k, o `n5k_v1` (50 pratos) dá 62%, o pipeline depende de um LLM em nuvem, e os pratos com mais de 25 g continuam em 41% a 54%, agora limitados pelo erro de massa.

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

## 9. Densidade de proteína e pratos ricos em proteína (10/11/2026)

**Pergunta:** a seção 8 terminou com o gargalo na **densidade** (gramas de proteína por grama de comida, isto é, saber que alimento é) e com os pratos acima de 15 g de proteína em 34–55%. Dá para melhorar a densidade, sobretudo nesses pratos, com a mesma premissa de uma foto de cima?
**Resposta curta: a densidade melhorou de forma mensurável, no modelo treinado e no LLM. Com a instrução nova do LLM (v3), o híbrido chega a 70,4% no teste: a estimativa pontual alcança a meta de 70% pela primeira vez, mas o intervalo vai de 65 a 76 e é uma única rodada cega. Sem LLM, o ganho é pequeno. Os pratos com mais de 25 g continuam fracos (41% a 54%).** Números do `teste` (280 pratos), com tudo escolhido por validação cruzada no treino e pesos do híbrido só no `calib`.

| Medida no `teste` | Seção 8 (escolhido) | Seção 9 (escolhido) | Diferença (IC 95%, bootstrap pareado) |
|---|---|---|---|
| **Densidade isolada:** massa real × densidade do modelo | 67,5% | **76,1%** | **+8,6 (+2,9 a +14,3)** |
| Só imagem, sem LLM | 61,1% | **64,6%** (IC 59,3 a 70,0) | +3,6 (−1,8 a +8,9): não distinguível |
| Híbrido: massa do modelo × densidade (LLM v2 e modelo) | 62,9% | **67,1%** (IC 61,4 a 72,5) | **+4,3 (+0,4 a +8,6)** |
| **Híbrido com o LLM v3** (uma rodada cega; 9.7) | 62,9% | **70,4%** (IC 65,0 a 75,7) | **+7,5 (+3,2 a +11,8)**; contra o híbrido com v2: +3,2 (0,0 a +6,4) |
| Densidade do LLM isolada: massa real × densidade (v2 → v3) | 71,4% | **80,7%** | **+9,3 (+4,6 a +13,9)** |
| Pratos com mais de 15 g: só imagem / híbrido v2 / híbrido v3 | 47,0% / 48,8% / — | 47,6% / 53,0% / 55,4% | |
| Pratos com mais de 25 g: híbrido v2 / híbrido v3 | 42,0% / — | 44,6% / 47,3% | |
| LLM sozinho (referência) | 55,0% (39,9% acima de 15 g) | | |
| **Meta** | **70%** | | |

- **A densidade do modelo treinado passou de 67,5% para 76,1%** quando medida com a massa real, e ficou acima da do LLM v2 na estimativa pontual (71,4%; diferença de +4,6, IC −1,4 a +10,4, não segura). O ganho veio de três coisas: prever **quanto da massa do prato é cada fonte de proteína** em vez da densidade direta, **recuperar pratos parecidos** do treino, e um terceiro backbone (TIPS) para essa recuperação.
- **O ganho não chegou inteiro ao resultado final.** Só imagem: 61,1% → 64,6%, dentro do ruído. Híbrido: 62,9% → 67,1%, com o intervalo encostado em zero. É a primeira vez que o híbrido escolhido se separa do anterior, mas por pouco, e formas não escolhidas vão de 65% a 70%.
- **Os pratos com mais de 25 g não melhoraram de forma que se distinga do ruído** (só imagem: 43% → 38% e 50% → 46%; híbrido: 38% → 45% e 46% → 45%; são 56 pratos por faixa). O que melhorou foram os pratos de 5 a 25 g. Nos pratos grandes, o erro que sobra já é **tanto de massa quanto de densidade**, e os dois puxam para baixo (9.5).
- **Lado do LLM: a instrução v3 leva o híbrido a 70,4%** (IC 65,0 a 75,7), +7,5 sobre o híbrido da seção 8 (IC +3,2 a +11,8). A análise mostrava que o LLM errava a parte da massa e a fonte, não a tabela; a v3 (`eval/prompt_v3.md`) pede identificação e partes da massa em vez de gramas. A densidade do LLM subiu de 71,4% para 80,7% com a massa real. **Ressalvas:** uma rodada cega só; contra o híbrido com v2 o ganho é de +3,2 com limite inferior zero; no `n5k_v1` (50 pratos) dá 62%; no `calib`, a fonte certa e a parte das fontes não mudaram, então o mecanismo do ganho não está isolado; e o pipeline precisa de um LLM em nuvem (9.7 e 9.8).
- **Acima de 25 g o limite agora é a massa.** Com a massa real, a densidade v3 daria 77% e 68% nas duas faixas de cima; com a massa do modelo, 45% e 50%.

### 9.1 Dados, higiene e regras de escolha

- **Nada mudou nos dados:** o mesmo treino zenital estrito da seção 8 (1.684 pratos em 511 cadeias, interseção com os 450 pratos de avaliação = 0), os mesmos 5 folds por cadeia, o mesmo critério de CV ponderado para as cinco faixas de proteína terem o mesmo peso, e a mesma entrada na inferência: **uma foto RGB de cima**.
- **Sinal novo de treino:** a lista de ingredientes de cada prato de **treino** (nome e gramas, do `dish_metadata` do Nutrition5k). Dela saem a fração de massa por ingrediente e por **fonte de proteína** (10 grupos, `dens_comum.py`). Os ingredientes dos pratos do `teste` e do `n5k_v1` nunca são lidos: a função que os carrega recusa esses ids. Os do `calib` são lidos só na análise de erro (9.2).
- **Modelo de massa:** o da seção 8, sem tocar (conjunto de 4 membros, RMSE de log-massa 0,2329). Esta seção só troca a densidade.
- **Regras, fixadas antes de olhar o teste** (estão no cabeçalho de `dens_avalia.py`, como R2 a R5; a R1 é o modelo de massa, que não muda):
  1. **densidade:** maior critério do produto na CV do treino, ponderado por faixa, calculado com a **massa real × densidade prevista**, para isolar a densidade. Concorrem todas as cabeças de densidade da seção 8, todas as desta seção, as médias declaradas e uma seleção gulosa de conjunto avaliada com ajuste cruzado;
  2. **desencolhimento** (esticar a densidade em torno da média): só é adotado se melhorar a CV com ajuste cruzado em mais de 0,5 ponto;
  3. **só imagem:** maior critério ponderado na CV entre todas as configurações e rotas da seção 8, a massa da seção 8 × cada densidade candidata, e a média geométrica disso com a rota direta da seção 8;
  4. **híbrido:** pesos só no `calib`; forma escolhida pelo critério no `calib` com validação deixa-fora em 12 blocos; em empate, a forma com menos pesos. As formas com a densidade do LLM v3 foram declaradas no script **antes** de as estimativas existirem.
- **Teste:** uma execução de `dens_avalia.py`, com todas as configurações. Enquanto as configurações eram montadas, o script rodou com `--so-cv`, que para antes de ler o gabarito do teste. Todas as configurações aparecem em 9.4 e 9.6, inclusive as que perderam.
- **Referências para os intervalos:** os dois pipelines escolhidos na seção 8, refeitos prato a prato pelo mesmo script (que confere que dão 61,1% e 62,9%). Bootstrap pareado, 4.000 sorteios.

### 9.2 Análise de erro: onde a densidade falha

Tudo aqui vem das previsões **fora-do-fold do treino** (1.684 pratos, CV por cadeia) e do `calib` (120). O `teste` não foi lido (`dens_analise.py`; o código recusa os ingredientes de qualquer prato do `teste`). "Fonte principal" é o grupo que dá pelo menos metade da proteína do prato, pelos ingredientes do Nutrition5k.

**a) Em prato com muita proteína, o erro é mais de densidade do que de massa; acima de 40 g é dos dois.** Modelo da seção 8 no treino, fora-do-fold:

| Faixa de proteína real | n | Acerto (massa × densidade previstas) | Com a massa real (só a densidade erra) | Com a densidade real (só a massa erra) | Viés mediano da massa | Viés mediano da densidade |
|---|---|---|---|---|---|---|
| 0–5 g | 643 | 97,0% | 97,8% | 100% | 0% | +7% |
| 5–15 g | 360 | 81,4% | 83,3% | 95,8% | 0% | −10% |
| 15–25 g | 220 | 53,6% | 56,4% | 83,2% | −2% | −5% |
| 25–40 g | 282 | 42,6% | 45,4% | 77,3% | −4% | −6% |
| > 40 g | 179 | 32,4% | 45,8% | 73,2% | −10% | −13% |

Corrigir só a densidade levaria os pratos de 15 a 40 g a 77–83%; corrigir só a massa quase não muda nada (45–56%). Acima de 40 g os dois erros pesam: a massa é subestimada em 10% e a densidade em 13%, e os dois vieses se somam.

**b) Regressão à média.** A densidade prevista varia menos que a real: a inclinação de log(prevista) contra log(real) é 0,84 em todos os pratos e **0,68 nos pratos com mais de 15 g** (correlação de 0,64). No quinto de pratos mais densos (12,6 a 37 g/100 g) a previsão fica 20% abaixo da real; no quinto menos denso, 22% acima. A massa tem o mesmo efeito, mais fraco (inclinação 0,89). No `calib`, a inclinação do modelo é 0,80 e a do LLM é 0,96: **o LLM quase não encolhe**, porque dá nome ao alimento e lê a tabela.

**c) Por fonte de proteína (pratos com mais de 15 g, treino fora-do-fold).**

| Fonte principal | n | Acerto | Com a massa real | Com a densidade real | Viés da densidade |
|---|---|---|---|---|---|
| Frango e peru | 197 | 34,5% | 43,7% | 69,5% | **−18%** |
| Bovina | 81 | 39,5% | 54,3% | 79,0% | −8% |
| Suína e embutidos | 93 | 54,8% | 46,2% | 82,8% | −9% |
| Peixe e frutos do mar | 118 | 40,7% | 54,2% | 82,2% | −3% |
| Ovo | 77 | 50,6% | 61,0% | 81,8% | −6% |
| Prato pronto (pizza etc.) | 21 | 66,7% | 57,1% | 76,2% | +12% |
| Sem fonte dominante | 78 | 46,2% | 41,0% | 85,9% | +8% |

O frango é a fonte mais frequente e a pior. Há duas causas. A primeira é a regressão à média: na tabela do Nutrition5k, `chicken` tem 31 g/100 g, o valor mais alto entre as fontes comuns. A segunda é da própria tabela: `chicken` (31), `chicken breast` (31), `chicken thighs` (24) e `grilled chicken` (23) têm densidades que diferem em até 37%, e a foto não diz qual nome o restaurante registrou. O mesmo vale para `fish` (18) e `salmon` (24), e para `pork` (26), `bacon` (37) e `sausage` (19). Parte do erro em frango é **ruído de rótulo** que nenhum modelo de imagem tira.

**d) O erro cresce quando a proteína é minoria no prato, e quando é quase tudo.** Nos pratos com mais de 15 g em que as fontes de proteína são menos de 25% da massa (63 pratos), o acerto com a massa real é de 32% e a densidade sai 17% **acima**; quando são 75% ou mais (135 pratos), 43%, com a densidade 11% **abaixo**. É a regressão à média vista pelo lado da composição: o modelo puxa todo prato para "metade proteína, metade acompanhamento". O número de ingredientes não explica o erro (1–2 ingredientes: 45%; 11 ou mais: 49%), então "ingrediente escondido em prato complexo" não é a causa principal.

**e) O que bastaria saber.** Se o modelo soubesse só a **fração da massa de cada uma de 10 fontes** (frango e peru, bovina, suína e embutidos, peixe, ovo, leguminosa e soja, laticínio, prato pronto, castanhas, resto) e usasse a densidade média de cada fonte no treino, a massa real × essa densidade daria **92,6%** na CV ponderada (87,7% a 95,0% nas faixas acima de 15 g), e 76,6% com a massa do modelo. Ou seja: **não é preciso reconhecer o ingrediente exato; é preciso acertar quanto do prato é cada fonte.** Foi isso que orientou o passo 2.

**f) LLM v2 no `calib`: erra a parte da massa e a fonte, não a tabela.** Todas as linhas usam a **massa real**, para isolar a densidade.

| Densidade usada (calib, massa real) | Acerto | Pratos com > 15 g |
|---|---|---|
| Do LLM, como está | 76,7% | 64% |
| Do LLM, trocando todas as densidades pela tabela do Nutrition5k | 76,7% | 64% |
| Frações do LLM × densidade **real** de cada fonte no prato | 75,8% | 68% |
| Frações **reais** × densidade do LLM para cada fonte | **90,0%** | **89%** |
| Frações reais × densidade média da fonte no treino | 86,7% | 88% |
| Frações do LLM × densidade média da fonte no treino | 67,5% | 58% |
| Do modelo só de imagem da seção 8 | 69,2% | 58% |

- **A tabela não é o problema.** No `calib`, 93% da massa estimada pelo LLM já usa nomes genéricos que casam com a tabela do Nutrition5k; só 7% usa a TACO. Trocar a tabela não muda o resultado. A hipótese "grelhado da TACO contra a tabela do Nutrition5k" não se confirma: o LLM já usava os nomes em inglês nos pratos do Nutrition5k. O que existe é o problema do item (c): em 19 componentes de frango, o LLM escreveu `chicken breast` (31) 6 vezes, `grilled chicken` (23) 5 vezes, `chicken` (31) 5 vezes e `roast chicken` (24) 2 vezes, e o gabarito quase sempre é `chicken` (31). Usar a média do treino para a família do frango (28,2 g/100 g) leva o LLM de 76,7% para 78,3% no `calib`: dois pratos.
- **Dar ao LLM a densidade real de cada fonte não melhora nada (75,8%). Dar a ele as frações reais leva a 90%.** O erro do LLM está em **quanto da massa é a fonte de proteína** e em **qual é a fonte**.
- **O LLM acerta a fonte principal em 77% dos 86 pratos do `calib` que têm fonte dominante.** Quando acerta, 74% ficam dentro da tolerância (com a massa real); quando erra, 55%. As confusões: frango visto como porco, bovina, peixe ou laticínio (5 de 17); porco visto como bovina ou frango (4 de 24); ovo não visto (3 de 19). O modelo de imagem do passo 2 acerta a fonte principal em 78–79% na CV do treino: **na identificação da fonte, modelo treinado e LLM estão no mesmo nível.**
- Em média, a fração de massa das fontes de proteína que o LLM dá não tem viés (63% contra 65% reais nos pratos com mais de 15 g; correlação de 0,87). O erro é de dispersão, prato a prato.

**Resposta à pergunta "densidade, massa ou os dois?":** nos pratos de 15 a 40 g, densidade (com a massa real o acerto fica em 45–56%, com a densidade real em 77–83%). Acima de 40 g, os dois. E "densidade" quer dizer, na prática, **fração da massa que é fonte de proteína**, mais a identidade da fonte; o valor de tabela por 100 g importa pouco.

### 9.3 Lado da imagem: o que foi testado e o que cada coisa deu

Todas as cabeças usam features congeladas: as da seção 8 (SigLIP 2 B/16 a 224 px com 8 orientações; a 384×512 com 2 orientações) e, no item 9, as do TIPS B/14. "Critério de densidade" = critério do produto na CV, ponderado por faixa, com a massa real × a densidade prevista; entre parênteses, só os pratos com mais de 15 g. Ponto de partida: a densidade do pipeline escolhido na seção 8 dá **65,7% (49,2%)**; a melhor cabeça isolada da seção 8 (MLP multitarefa a 224 px) dá 68,2% (53,4%).

| # | Ideia | O que foi feito | Critério de densidade na CV | Veredito |
|---|---|---|---|---|
| 1 | Peso por faixa de proteína no treino | Ridge de log-densidade com peso por amostra para as 5 faixas pesarem igual (e meio caminho) | 384×512: 64,1 → 64,8 → 65,2% (46,7 → 48,9%); 224 px: 61,0 → 61,7% | Ajuda pouco sozinho (+1) |
| 2 | Recuperação (k vizinhos) | Média da log-densidade dos 80 pratos de treino mais parecidos no embedding do SigLIP 2 (cosseno, peso exponencial) | 67,5% (53,9%) a 384×512; 68,0% (54,9%) a 224 px | **Ajuda:** sem treinar nada, empata com o melhor MLP da seção 8, e erra de outro jeito |
| 3 | Fração de massa por **fonte de proteína** (10 grupos) × densidade média da fonte | Ridge linear nas frações | 62,3% (**55,3%**) | Pior no total, **melhor acima de 15 g** (ridge de log-densidade: 46,7%) |
| 3b | | MLP com softmax e entropia cruzada | 64,2% (48,3%) | A perda errada: não ajuda |
| 3c | | MLP com softmax e **erro quadrático nas frações** | **70,9% (59,9%)**; três sementes: 70,6 a 70,9% | **Ajuda muito** |
| 3d | | 3c + perda direta em log da densidade composta + peso por faixa | **72,2% (61,4%)**; outras sementes: 71,8% e 72,2% | **A melhor cabeça isolada** |
| 3e | | 3c com erro absoluto; com erro quadrático ponderado pela densidade da classe | 66,9% (53,8%); 72,0% (61,4%) | Absoluto piora; ponderado empata com 3d |
| 4 | Classificador de **ingredientes** (165 classes) → Σ fração × densidade do ingrediente | MLP softmax, entropia cruzada; + perda direta; com erro quadrático | 59,8% (41,0%); 62,7% (45,0%); 69,8% (58,3%) | Pior que as 10 fontes: classes demais para 1.684 pratos |
| 4b | Classes "finas" (32: os ingredientes que mais dão proteína, um a um, + o resto por fonte) | Como 3d | 71,3% (58,9%) | Não ganha das 10 fontes. Separar `chicken` de `chicken thighs` pela foto não funcionou |
| 4c | Contribuição de proteína por ingrediente | Ridge linear | 33–35% | Muito pior |
| 5 | Mais entrada | 224 + 384×512 concatenados; blocos espaciais (GeM e grade 2×2) | 72,2% (60,3%); 69,0% (55,7%) | Não ajuda. A 224 px a fonte principal sai certa em 74% dos pratos; a 384×512, em 78–80% |
| 6 | Cabeça maior | 1.024 unidades, dropout 0,5 | 68,8% (58,4%) | Piora |
| 7 | Médias (em log) de cabeças diferentes | 22 médias declaradas: frações (3d) com vizinhos, com classes finas, com o MLP da seção 8, com outras sementes, com o TIPS | de 71,4% a **75,4% (63,7%)** | **Ajuda** (+3 sobre a melhor isolada) |
| 7b | Seleção gulosa de conjunto | Escolhida em 4 folds, aplicada no 5º | 74,3% (62,9%) | Não ganha das médias simples: parte do que a escolha entre médias rende é ruído |
| 8 | Desencolhimento | Esticar em log, em torno da média, a densidade escolhida e a proteína final (fator b), com ajuste cruzado | 75,4 → 75,5% (b = 1,05); proteína final 66,3 → 65,4% (b = 1,10) | Não adotado (regra: mais de 0,5 ponto) |
| 9 | Terceiro backbone: TIPS B/14 (tipo DINOv2), 336×448, 1 orientação | Vizinhos no embedding do TIPS; cabeça 3d sobre TIPS; 3d sobre TIPS + SigLIP 2 concatenados | vizinhos **72,4% (60,7%)**; 3d 70,3% (56,9%); concatenados 72,2% (59,7%) | **Ajuda na recuperação:** os vizinhos no TIPS, sem treino, empatam com a melhor cabeça treinada. A cabeça treinada sobre TIPS não ganha da do SigLIP 2 |

O que os números mostram:

1. **Prever "quanto do prato é cada fonte" funciona melhor do que prever a densidade direto**, desde que a perda seja o erro quadrático das frações. A mesma cabeça com entropia cruzada perde 7 pontos. A densidade sai de uma conta transparente (fração × densidade média da fonte), e isso também dá ao app o que mostrar: "frango ≈ 45% do prato".
2. **O ganho está nos pratos com mais de 15 g** (49% → 61–62% com a massa real). Nos pratos com pouca proteína a rota por frações é pior que a regressão (a tolerância de ±5 g esconde isso), e por isso a média com as cabeças de log-densidade ganha das frações sozinhas.
3. **Resolução conta para identificar.** A 384×512 a fonte principal sai certa em 78–80% dos pratos; a 224 px, em 74%.
4. **Ingrediente exato não ajuda.** O Nutrition5k tem 199 ingredientes no treino e poucos pratos por ingrediente; as 10 fontes bastam (o oráculo de 9.2 dá 92,6%) e são aprendidas melhor.
5. **O TIPS, que a seção 8 não chegou a extrair, rende na recuperação** (72,4% contra 67,5% do SigLIP 2 na mesma resolução). A extração levou 38 min de CPU com uma orientação. Ele entrou só na densidade; o modelo de massa continua o da seção 8.

### 9.4 Todas as densidades avaliadas

**Escolha pela CV (regra 1):** a média, em log, de cinco estimativas de densidade: três cabeças de fração por fonte (item 3d de 9.3) sobre SigLIP 2 a 384×512, sobre TIPS e sobre os dois concatenados; os vizinhos no embedding do TIPS; e o MLP multitarefa a 224 px da seção 8. Critério de densidade na CV: **75,4% (63,7% acima de 15 g)**, contra 65,7% (49,2%) do escolhido na seção 8.

- **Desencolhimento não adotado.** Esticar a log-densidade escolhida (b = 1,05) dá 75,4% → 75,5% com ajuste cruzado; esticar a proteína final do pipeline só de imagem (b = 1,10) dá 66,3% → 65,4%. Nenhum passa da regra de 0,5 ponto. A regressão à média continua lá (inclinação de 0,60 acima de 15 g, contra 0,68 antes; a correlação subiu de 0,64 para 0,72): a média de cinco estimativas encolhe mais, mas acerta mais a ordem. Esticar não conserta, porque o erro é de informação, não de escala.
- **Por fonte, na CV (pratos com mais de 15 g, massa real × densidade):**

| Fonte principal | n | Seção 8 | Seção 9 | Viés da densidade (antes → depois) |
|---|---|---|---|---|
| Frango e peru | 197 | 43,7% | 54,8% | −18% → −15% |
| Bovina | 81 | 54,3% | 60,5% | −8% → −5% |
| Suína e embutidos | 93 | 46,2% | 64,5% | −9% → −10% |
| Peixe e frutos do mar | 118 | 54,2% | 68,6% | −3% → −1% |
| Ovo | 77 | 61,0% | 88,3% | −6% → +1% |
| Prato pronto (pizza etc.) | 21 | 57,1% | 42,9% | +12% → +24% |

O frango melhora menos e continua 15% abaixo: é a fonte em que a tabela do Nutrition5k tem mais nomes para a mesma aparência (9.2 c). Os pratos prontos pioraram (são 21). Por faixa de proteína, na CV: 15–25 g 56,4% → 65,9%; 25–40 g 45,4% → 63,8%; > 40 g 45,8% → 61,5%.

**Resumo das principais** (a tabela completa, com as 151 densidades, vem logo abaixo). "Massa s8" é o modelo de massa da seção 8; a coluna "geo" é a média geométrica com a rota direta da seção 8.

| Densidade | CV: critério de densidade (todos / > 15 g) | Teste: massa real × dens. (todos / > 15 g) | Teste: massa s8 × dens. | Teste: geo |
|---|---|---|---|---|
| Seção 8, escolhida (ajuste fino + 384×512) | 65,7 / 49,2% | 67,5 / 55,4% | 60,0% | 61,1% |
| Seção 8, MLP multitarefa 224 px | 68,2 / 53,4% | 68,9 / 56,0% | 63,9% | 65,0% |
| Ridge de log-densidade 384×512, peso por faixa | 65,2 / 48,9% | 66,8 / 55,4% | 62,1% | 63,9% |
| Vizinhos, SigLIP 2 384×512 | 67,5 / 53,9% | 67,9 / 56,0% | 58,9% | 65,0% |
| Vizinhos, TIPS | 72,4 / 60,7% | 72,9 / 63,7% | 59,6% | 62,5% |
| Fração por fonte, ridge linear, 384×512 | 62,3 / 55,3% | 62,1 / 57,1% | 55,0% | 64,6% |
| Fração por fonte, MLP com entropia cruzada, 384×512 | 64,2 / 48,3% | 70,4 / 60,1% | 63,2% | 65,7% |
| Fração por fonte, MLP com erro quadrático (3c), 384×512 | 70,9 / 59,9% | 73,2 / 63,7% | 61,1% | 65,4% |
| 3c + perda de densidade + peso por faixa (3d), 384×512 | 72,2 / 61,4% | 73,6 / 63,7% | 62,5% | 65,4% |
| 3d sobre TIPS | 70,3 / 56,9% | 71,4 / 63,7% | 59,6% | 67,1% |
| Ingredientes (165 classes), erro quadrático | 69,8 / 58,3% | 72,1 / 66,1% | 62,5% | 67,1% |
| Classes finas (32), como 3d | 71,3 / 58,9% | 73,6 / 63,1% | 63,2% | 66,4% |
| Seleção gulosa de conjunto | 74,3 / 62,9% | 76,8 / 67,9% | 66,1% | 63,2% |
| **Média escolhida pela CV [densidade]** | **75,4 / 63,7%** | **76,1 / 66,7%** | **64,6%** | 64,6% |

Como ler: a coluna "massa real × dens." mede a densidade sozinha, e nela as cabeças de fração por fonte com erro quadrático ficam entre 69,6% e 75,0% no teste, e as médias declaradas, entre 71,8% e 77,5%, contra 67,5% da seção 8. Esse ganho é consistente. A coluna "massa s8 × dens." é o pipeline completo, e nela as mesmas cabeças vão de 59% a 64% e as médias, de 62% a 67%: com 280 pratos, duas densidades quase iguais na CV diferem em vários pontos no produto final. Por isso a regra 3 não escolhe entre densidades pelo produto.

<details>
<summary>Tabela completa: as 151 densidades, em ordem do critério de CV (CV e teste)</summary>

Legenda dos nomes: `s8:` = cabeça de densidade de uma configuração da seção 8; `soft-mlp fonte` / `ingrediente` / `fina` = MLP com softmax sobre as 10 fontes / 165 ingredientes / 32 classes finas; `-l2`, `-l1`, `-l2d` = perda nas frações (erro quadrático, absoluto, quadrático ponderado pela densidade da classe; sem sufixo = entropia cruzada); `+perda-dens` = perda direta em log da densidade composta; `peso-faixa` = peso por faixa de proteína; `[frações]` = densidade pela rota Σ fração × densidade da classe; `[log-dens]` = a cabeça de log-densidade do mesmo modelo; `knn` = vizinhos; `ambos` = 224 px + 384×512 concatenados; `+esp` = com blocos espaciais; `tips_b14+siglip2_b16_512` = os dois concatenados. A diferença é de "massa s8 × dens." contra o só imagem da seção 8 (61,1%).

| Densidade | CV: RMSE log-dens. (todos / > 15 g) | CV: critério de densidade (todos / > 15 g) | Teste: massa real × dens. (todos / > 15 g) | Teste: massa s8 × dens. | Δ vs só imagem s8 (IC 95%) | Teste: geo com a direta s8 |
|---|---|---|---|---|---|---|
| ridge contribuição por ingrediente siglip2_b16_512 | 0,875 / 0,422 | 33,3 / 28,9% | 35,4 / 32,7% | 35,7% | −25,4 (−33,2 a −17,5) | 52,5% |
| ridge contribuição por ingrediente tips_b14 | 0,866 / 0,429 | 34,4 / 31,0% | 35,7 / 31,0% | 38,2% | −22,9 (−30,7 a −14,6) | 55,0% |
| ridge contribuição por ingrediente siglip2_b16 | 0,864 / 0,408 | 35,0 / 31,8% | 36,8 / 32,7% | 33,6% | −27,5 (−35,4 a −20,0) | 53,6% |
| soft-linear fonte siglip2_b16_512 [frações] | 0,874 / 0,472 | 37,1 / 27,4% | 46,8 / 41,7% | 40,7% | −20,4 (−27,9 a −13,2) | 56,4% |
| soft-linear ingrediente siglip2_b16_512 [frações] | 0,451 / 0,544 | 56,6 / 36,2% | 61,8 / 48,2% | 53,9% | −7,1 (−13,6 a −1,1) | 60,0% |
| s8: ridge siglip2_b16 o4 | 0,389 / 0,379 | 58,1 / 38,8% | 57,5 / 43,5% | 53,2% | −7,9 (−13,2 a −2,9) | 58,9% |
| s8: ridge siglip2_b16 o1 | 0,402 / 0,381 | 59,7 / 41,5% | 55,0 / 39,3% | 51,4% | −9,6 (−15,0 a −3,9) | 56,4% |
| s8: ridge siglip2_b16 o2 | 0,396 / 0,377 | 59,8 / 41,6% | 55,7 / 40,5% | 52,5% | −8,6 (−13,6 a −3,6) | 58,6% |
| soft-mlp ingrediente siglip2_b16_512 [frações] | 0,383 / 0,447 | 59,8 / 41,0% | 66,1 / 52,4% | 60,0% | −1,1 (−7,1 a +5,0) | 62,9% |
| s8: ridge siglip2_b16 o8 + escala aleatória no treino | 0,374 / 0,361 | 60,7 / 42,9% | 58,2 / 45,2% | 57,1% | −3,9 (−8,9 a +1,4) | 59,6% |
| s8: ridge siglip2_b16 o8 | 0,377 / 0,365 | 61,0 / 43,4% | 59,3 / 44,0% | 55,7% | −5,4 (−10,4 a −0,4) | 58,9% |
| ridge logdens siglip2_b16 | 0,377 / 0,365 | 61,0 / 43,4% | 59,3 / 44,0% | 55,7% | −5,4 (−10,4 a −0,4) | 58,9% |
| s8: ridge siglip2_b16 o1-tta8 | 0,394 / 0,375 | 61,2 / 43,7% | 57,5 / 42,3% | 55,7% | −5,4 (−10,4 a +0,0) | 60,4% |
| soft-linear fonte siglip2_b16_512 [log-dens] | 0,379 / 0,366 | 61,3 / 42,5% | 62,5 / 48,8% | 56,8% | −4,3 (−8,9 a +0,0) | 60,4% |
| soft-linear ingrediente siglip2_b16_512 [log-dens] | 0,379 / 0,366 | 61,3 / 42,5% | 62,5 / 48,8% | 56,8% | −4,3 (−8,9 a +0,0) | 60,4% |
| ridge logdens peso-faixa0.5 siglip2_b16 | 0,377 / 0,354 | 61,4 / 44,2% | 62,9 / 50,6% | 56,4% | −4,6 (−9,6 a +0,7) | 59,3% |
| ridge logdens peso-faixa1.0 siglip2_b16 | 0,381 / 0,341 | 61,7 / 44,4% | 64,3 / 52,4% | 56,4% | −4,6 (−9,6 a +0,7) | 62,1% |
| s8: ridge siglip2_b16 o8 + quadros laterais no treino | 0,377 / 0,362 | 61,8 / 44,2% | 58,9 / 44,6% | 54,3% | −6,8 (−11,8 a −1,4) | 60,4% |
| s8: ridge siglip2_b16 o8+gem | 0,376 / 0,362 | 62,0 / 44,7% | 63,2 / 53,0% | 57,9% | −3,2 (−8,2 a +2,1) | 60,0% |
| soft-mlp fonte siglip2_b16 [frações] | 0,435 / 0,433 | 62,0 / 44,8% | 67,5 / 57,7% | 58,9% | −2,1 (−8,2 a +4,3) | 65,4% |
| ridge fração por fonte siglip2_b16_512 | 0,609 / 0,275 | 62,3 / 55,3% | 62,1 / 57,1% | 55,0% | −6,1 (−11,8 a −0,4) | 64,6% |
| s8: ridge siglip2_b16_512 o1 | 0,377 / 0,346 | 62,4 / 44,6% | 60,7 / 45,2% | 59,3% | −1,8 (−6,1 a +2,5) | 63,2% |
| ridge fração por fonte siglip2_b16 | 0,611 / 0,277 | 62,6 / 56,3% | 64,6 / 58,9% | 55,0% | −6,1 (−12,1 a +0,4) | 64,6% |
| soft-mlp ingrediente +perda-dens siglip2_b16_512 [frações] | 0,352 / 0,393 | 62,7 / 45,0% | 70,7 / 58,9% | 61,1% | +0,0 (−5,7 a +5,7) | 63,2% |
| s8: média: ajuste fino + ridge siglip2 o8 | 0,357 / 0,355 | 62,9 / 45,5% | 65,4 / 52,4% | 58,2% | −2,9 (−7,5 a +1,8) | 59,6% |
| s8: ridge siglip2_b16_512 o2+gem+grade | 0,375 / 0,355 | 62,9 / 45,5% | 61,1 / 46,4% | 56,8% | −4,3 (−8,9 a +0,4) | 61,1% |
| soft-mlp ingrediente só-frações +perda-dens siglip2_b16_512 [frações] | 0,352 / 0,390 | 63,1 / 46,0% | 71,4 / 59,5% | 64,3% | +3,2 (−2,9 a +9,3) | 64,6% |
| soft-mlp ingrediente +perda-dens peso-faixa siglip2_b16 [frações] | 0,379 / 0,426 | 63,1 / 46,1% | 73,9 / 64,3% | 64,6% | +3,6 (−2,9 a +9,6) | 65,0% |
| s8: ridge siglip2_b16 o8+gem+grade | 0,376 / 0,362 | 63,2 / 46,2% | 60,7 / 48,8% | 58,2% | −2,9 (−7,5 a +2,1) | 60,7% |
| ridge fração por fonte tips_b14 | 0,594 / 0,273 | 63,2 / 55,4% | 64,6 / 56,5% | 56,4% | −4,6 (−11,1 a +1,8) | 63,2% |
| soft-mlp ingrediente +perda-dens peso-faixa siglip2_b16_512 [frações] | 0,352 / 0,384 | 63,2 / 46,3% | 72,1 / 60,7% | 64,6% | +3,6 (−2,5 a +9,3) | 64,6% |
| soft-mlp ingrediente +perda-dens peso-faixa0.5 siglip2_b16_512 [frações] | 0,351 / 0,388 | 63,3 / 46,4% | 71,4 / 59,5% | 63,6% | +2,5 (−3,2 a +8,2) | 64,6% |
| s8: média: siglip2 o8 + siglip2-512 | 0,353 / 0,335 | 63,3 / 45,8% | 63,9 / 50,6% | 58,2% | −2,9 (−6,4 a +0,7) | 61,4% |
| s8: ft cv | 0,354 / 0,360 | 63,8 / 46,6% | 63,6 / 48,8% | 56,4% | −4,6 (−9,3 a +0,0) | 58,6% |
| s8: ridge siglip2_b16_512 o2 | 0,365 / 0,342 | 64,1 / 46,7% | 63,6 / 50,0% | 59,6% | −1,4 (−5,7 a +2,9) | 64,6% |
| ridge logdens siglip2_b16_512 | 0,365 / 0,342 | 64,1 / 46,7% | 63,6 / 50,0% | 59,6% | −1,4 (−5,7 a +2,9) | 64,6% |
| soft-mlp fonte siglip2_b16_512 [frações] | 0,409 / 0,383 | 64,2 / 48,3% | 70,4 / 60,1% | 63,2% | +2,1 (−3,6 a +7,9) | 65,7% |
| soft-mlp ingrediente +perda-dens oculto1024 drop0.5 siglip2_b16_512 [frações] | 0,337 / 0,349 | 64,5 / 47,9% | 73,2 / 63,7% | 65,7% | +4,6 (−1,1 a +10,4) | 65,0% |
| s8: média: ajuste fino + ridge o8 com laterais | 0,353 / 0,348 | 64,7 / 48,3% | 64,3 / 51,8% | 57,9% | −3,2 (−7,5 a +1,1) | 60,7% |
| ridge logdens peso-faixa0.5 siglip2_b16_512 | 0,365 / 0,330 | 64,8 / 47,8% | 65,4 / 53,0% | 60,4% | −0,7 (−5,0 a +3,6) | 63,9% |
| soft-mlp fonte peso-faixa siglip2_b16_512 [frações] | 0,405 / 0,370 | 64,8 / 48,6% | 72,1 / 61,3% | 65,0% | +3,9 (−2,1 a +9,6) | 65,0% |
| s8: média: ajuste fino + ridge o8 + siglip2-512 (sem TIPS) | 0,344 / 0,335 | 65,0 / 48,2% | 66,4 / 54,2% | 58,9% | −2,1 (−6,1 a +1,8) | 61,8% |
| ridge logdens peso-faixa1.0 siglip2_b16_512 | 0,370 / 0,320 | 65,2 / 48,9% | 66,8 / 55,4% | 62,1% | +1,1 (−3,2 a +5,0) | 63,9% |
| s8: média: ajuste fino + ridge o8 com laterais + siglip2-512 (sem TIPS) | 0,343 / 0,331 | 65,4 / 49,2% | 65,7 / 53,0% | 61,1% | +0,0 (−3,6 a +3,6) | 61,4% |
| s8: média: siglip2 o8 + mlp | 0,344 / 0,332 | 65,7 / 49,3% | 65,0 / 52,4% | 60,0% | −1,1 (−6,1 a +3,6) | 63,2% |
| s8: média: ajuste fino + siglip2-512 | 0,339 / 0,330 | 65,7 / 49,2% | 67,5 / 55,4% | 60,0% | −1,1 (−3,9 a +1,8) | 61,1% |
| soft-mlp fonte +perda-dens siglip2_b16 [frações] | 0,402 / 0,385 | 65,8 / 49,4% | 68,9 / 56,0% | 62,1% | +1,1 (−5,4 a +7,5) | 66,4% |
| ridge logdens tips_b14 | 0,356 / 0,352 | 65,9 / 50,2% | 65,0 / 51,2% | 63,2% | +2,1 (−3,2 a +7,1) | 64,6% |
| s8: mlp+aux-profundidade siglip2_b16 o8 | 0,345 / 0,339 | 66,8 / 51,1% | 67,5 / 54,2% | 61,8% | +0,7 (−5,0 a +6,1) | 63,6% |
| ridge logdens peso-faixa0.5 tips_b14 | 0,357 / 0,343 | 66,9 / 51,7% | 65,4 / 53,6% | 62,9% | +1,8 (−3,2 a +6,8) | 64,3% |
| soft-mlp fonte-l1 siglip2_b16_512 [frações] | 0,398 / 0,328 | 66,9 / 53,8% | 74,3 / 64,9% | 62,5% | +1,4 (−4,3 a +7,1) | 64,6% |
| knn fração por fonte siglip2_b16 | 0,451 / 0,304 | 67,0 / 56,8% | 67,5 / 60,7% | 57,1% | −3,9 (−10,0 a +2,1) | 63,2% |
| s8: média: ajuste fino + mlp + ridge o8 com laterais | 0,338 / 0,331 | 67,0 / 51,3% | 65,7 / 53,6% | 61,1% | +0,0 (−4,3 a +4,3) | 62,1% |
| soft-mlp fonte-l2 oculto1024 drop0.5 siglip2_b16_512 [log-dens] | 0,351 / 0,318 | 67,1 / 51,2% | 72,1 / 61,3% | 64,6% | +3,6 (−2,1 a +8,6) | 62,9% |
| soft-mlp fonte-l2 siglip2_b16_512+esp [log-dens] | 0,352 / 0,312 | 67,1 / 51,5% | 70,7 / 58,9% | 65,0% | +3,9 (−1,4 a +9,3) | 63,6% |
| s8: média: ajuste fino + mlp + ridge o8 com laterais + siglip2-512 (sem TIPS) | 0,334 / 0,322 | 67,1 / 51,2% | 67,1 / 55,4% | 60,7% | −0,4 (−3,9 a +3,2) | 62,5% |
| soft-mlp ingrediente-l2 siglip2_b16_512 [log-dens] | 0,342 / 0,315 | 67,1 / 51,2% | 70,4 / 58,9% | 62,5% | +1,4 (−3,9 a +6,4) | 62,9% |
| ridge logdens peso-faixa1.0 tips_b14 | 0,362 / 0,335 | 67,1 / 51,9% | 66,4 / 54,8% | 62,1% | +1,1 (−3,9 a +6,1) | 63,2% |
| soft-mlp fonte-l2 siglip2_b16_512 [log-dens] | 0,339 / 0,303 | 67,3 / 51,7% | 70,4 / 57,7% | 63,6% | +2,5 (−2,9 a +7,9) | 63,6% |
| knn fração por fonte siglip2_b16_512 | 0,433 / 0,293 | 67,4 / 56,4% | 69,3 / 60,7% | 57,1% | −3,9 (−10,0 a +2,1) | 62,9% |
| soft-mlp ingrediente +perda-dens siglip2_b16_512 [log-dens] | 0,339 / 0,312 | 67,5 / 51,6% | 70,0 / 57,7% | 64,6% | +3,6 (−1,1 a +8,2) | 63,9% |
| knn densidade siglip2_b16_512 | 0,343 / 0,338 | 67,5 / 53,9% | 67,9 / 56,0% | 58,9% | −2,1 (−8,2 a +3,9) | 65,0% |
| s8: média: ajuste fino + mlp | 0,332 / 0,330 | 67,5 / 51,8% | 68,2 / 56,0% | 62,1% | +1,1 (−3,6 a +5,7) | 61,8% |
| soft-mlp fonte siglip2_b16 [log-dens] | 0,338 / 0,322 | 67,6 / 52,0% | 69,6 / 56,5% | 62,1% | +1,1 (−4,6 a +6,4) | 64,3% |
| soft-mlp fonte-l2 tips_b14 [log-dens] | 0,346 / 0,330 | 67,7 / 51,9% | 68,2 / 54,8% | 61,4% | +0,4 (−5,4 a +6,1) | 65,4% |
| soft-mlp ingrediente +perda-dens oculto1024 drop0.5 siglip2_b16_512 [log-dens] | 0,343 / 0,308 | 67,7 / 52,1% | 67,9 / 54,8% | 62,1% | +1,1 (−3,9 a +5,7) | 63,9% |
| soft-mlp fonte +perda-dens siglip2_b16 [log-dens] | 0,340 / 0,322 | 67,9 / 52,3% | 68,2 / 54,2% | 60,0% | −1,1 (−6,8 a +4,3) | 63,2% |
| soft-mlp fonte-l2d siglip2_b16_512 [log-dens] | 0,339 / 0,306 | 67,9 / 52,3% | 70,4 / 58,9% | 62,5% | +1,4 (−3,9 a +6,4) | 62,5% |
| soft-mlp ingrediente siglip2_b16_512 [log-dens] | 0,338 / 0,316 | 68,0 / 52,4% | 70,7 / 58,9% | 65,4% | +4,3 (−0,4 a +8,9) | 64,6% |
| knn densidade siglip2_b16 | 0,363 / 0,357 | 68,0 / 54,9% | 72,1 / 60,7% | 61,8% | +0,7 (−5,4 a +6,8) | 62,5% |
| soft-mlp fonte-l2 semente2 siglip2_b16_512 [log-dens] | 0,341 / 0,309 | 68,1 / 52,7% | 71,8 / 60,1% | 61,1% | +0,0 (−5,0 a +5,0) | 62,5% |
| s8: mlp siglip2_b16 o8 | 0,336 / 0,328 | 68,2 / 53,4% | 68,9 / 56,0% | 63,9% | +2,9 (−2,5 a +7,9) | 65,0% |
| s8: mlp wd0.1 siglip2_b16 o8 | 0,336 / 0,328 | 68,3 / 53,5% | 69,3 / 56,5% | 64,3% | +3,2 (−2,1 a +8,2) | 64,6% |
| soft-mlp fonte-l2 tips_b14 [frações] | 0,386 / 0,305 | 68,3 / 53,8% | 70,4 / 61,9% | 59,3% | −1,8 (−7,9 a +4,6) | 66,8% |
| soft-mlp fonte +perda-dens siglip2_b16_512 [log-dens] | 0,332 / 0,305 | 68,4 / 52,8% | 72,9 / 62,5% | 65,0% | +3,9 (−1,4 a +9,3) | 63,6% |
| soft-mlp fonte +perda-dens peso-faixa siglip2_b16_512 [frações] | 0,380 / 0,326 | 68,5 / 54,0% | 71,8 / 61,9% | 64,3% | +3,2 (−2,9 a +8,9) | 65,0% |
| soft-mlp fonte-l2 ambos [log-dens] | 0,339 / 0,313 | 68,5 / 53,9% | 69,3 / 58,3% | 62,5% | +1,4 (−3,6 a +6,1) | 64,3% |
| s8: mlp oculto1024 drop0.5 siglip2_b16 o8 | 0,333 / 0,327 | 68,7 / 53,7% | 70,0 / 57,1% | 63,2% | +2,1 (−3,2 a +7,5) | 63,2% |
| soft-mlp fonte-l2 semente1 siglip2_b16_512 [log-dens] | 0,339 / 0,304 | 68,8 / 54,2% | 73,9 / 62,5% | 63,9% | +2,9 (−2,1 a +7,9) | 64,6% |
| soft-mlp fonte-l2 oculto1024 drop0.5 siglip2_b16_512 [frações] | 0,409 / 0,282 | 68,8 / 58,4% | 69,6 / 63,1% | 60,4% | −0,7 (−6,4 a +5,0) | 65,4% |
| soft-mlp fonte siglip2_b16_512 [log-dens] | 0,329 / 0,303 | 68,8 / 53,2% | 73,6 / 62,5% | 65,4% | +4,3 (−0,4 a +8,9) | 64,3% |
| soft-mlp ingrediente +perda-dens peso-faixa0.5 siglip2_b16_512 [log-dens] | 0,341 / 0,298 | 68,8 / 54,1% | 71,1 / 58,9% | 66,4% | +5,4 (+0,4 a +10,0) | 62,5% |
| soft-mlp fonte-l2 +perda-dens peso-faixa tips_b14 [log-dens] | 0,357 / 0,310 | 68,8 / 54,8% | 71,1 / 61,3% | 61,1% | +0,0 (−5,7 a +5,7) | 65,4% |
| soft-mlp fonte-l1 siglip2_b16_512 [log-dens] | 0,343 / 0,299 | 68,9 / 54,6% | 70,4 / 59,5% | 65,4% | +4,3 (−1,1 a +9,3) | 64,6% |
| soft-mlp fonte +perda-dens siglip2_b16_512 [frações] | 0,380 / 0,332 | 69,0 / 54,9% | 71,8 / 61,9% | 64,3% | +3,2 (−2,5 a +8,9) | 65,4% |
| soft-mlp fonte-l2 siglip2_b16_512+esp [frações] | 0,391 / 0,305 | 69,0 / 55,7% | 72,5 / 66,7% | 62,5% | +1,4 (−4,6 a +7,5) | 66,4% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente1 tips_b14 [log-dens] | 0,357 / 0,305 | 69,1 / 55,1% | 70,7 / 58,9% | 62,1% | +1,1 (−5,0 a +6,8) | 66,1% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente2 siglip2_b16_512 [log-dens] | 0,353 / 0,295 | 69,1 / 54,7% | 72,5 / 63,7% | 61,8% | +0,7 (−4,3 a +5,4) | 63,2% |
| soft-mlp fonte-l2 +perda-dens peso-faixa tips_b14+siglip2_b16_512 [log-dens] | 0,349 / 0,298 | 69,2 / 55,1% | 72,5 / 61,9% | 64,3% | +3,2 (−2,5 a +8,9) | 65,4% |
| soft-mlp fina-l2 siglip2_b16_512 [log-dens] | 0,341 / 0,309 | 69,2 / 54,7% | 72,1 / 60,1% | 63,2% | +2,1 (−2,9 a +7,1) | 63,9% |
| soft-mlp ingrediente +perda-dens peso-faixa siglip2_b16 [log-dens] | 0,360 / 0,306 | 69,2 / 55,9% | 70,0 / 58,3% | 63,2% | +2,1 (−2,9 a +7,1) | 62,9% |
| soft-mlp fonte-l2 tips_b14+siglip2_b16_512 [log-dens] | 0,337 / 0,313 | 69,4 / 54,3% | 68,6 / 54,8% | 61,4% | +0,4 (−5,0 a +5,7) | 63,9% |
| soft-mlp fonte +perda-dens peso-faixa siglip2_b16_512 [log-dens] | 0,342 / 0,289 | 69,5 / 55,2% | 72,1 / 61,3% | 66,4% | +5,4 (+0,4 a +10,4) | 63,6% |
| knn fração por fonte tips_b14+siglip2_b16_512 | 0,413 / 0,271 | 69,5 / 59,2% | 71,1 / 63,7% | 58,6% | −2,5 (−8,6 a +3,6) | 63,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa siglip2_b16 [log-dens] | 0,361 / 0,297 | 69,5 / 56,2% | 67,9 / 54,8% | 61,8% | +0,7 (−4,6 a +6,1) | 63,6% |
| soft-mlp fina-l2 +perda-dens peso-faixa siglip2_b16_512 [log-dens] | 0,351 / 0,283 | 69,6 / 55,8% | 70,4 / 58,9% | 64,3% | +3,2 (−2,1 a +8,2) | 63,9% |
| soft-mlp fina-l2 siglip2_b16_512 [frações] | 0,381 / 0,304 | 69,7 / 57,5% | 72,1 / 63,1% | 62,9% | +1,8 (−3,9 a +7,5) | 65,0% |
| soft-mlp ingrediente-l2 siglip2_b16_512 [frações] | 0,339 / 0,284 | 69,8 / 58,3% | 72,1 / 66,1% | 62,5% | +1,4 (−4,3 a +7,5) | 67,1% |
| soft-mlp ingrediente +perda-dens peso-faixa siglip2_b16_512 [log-dens] | 0,347 / 0,287 | 69,8 / 55,9% | 70,0 / 60,1% | 63,6% | +2,5 (−2,5 a +7,1) | 63,2% |
| soft-mlp fonte peso-faixa siglip2_b16_512 [log-dens] | 0,336 / 0,285 | 70,0 / 55,7% | 72,9 / 63,1% | 68,9% | +7,9 (+2,9 a +12,9) | 65,4% |
| soft-mlp fonte-l2 ambos [frações] | 0,391 / 0,294 | 70,1 / 58,1% | 74,3 / 66,1% | 63,6% | +2,5 (−3,2 a +8,6) | 64,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa siglip2_b16 [frações] | 0,394 / 0,312 | 70,2 / 57,5% | 70,7 / 61,9% | 61,1% | +0,0 (−6,1 a +6,1) | 65,4% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente1 siglip2_b16_512 [log-dens] | 0,352 / 0,287 | 70,2 / 56,8% | 72,5 / 61,3% | 64,6% | +3,6 (−1,4 a +8,6) | 64,6% |
| soft-mlp fonte-l2 tips_b14+siglip2_b16_512 [frações] | 0,377 / 0,292 | 70,3 / 57,4% | 71,4 / 61,3% | 61,1% | +0,0 (−6,4 a +6,4) | 66,1% |
| soft-mlp fonte-l2 +perda-dens peso-faixa tips_b14 [frações] | 0,379 / 0,298 | 70,3 / 56,9% | 71,4 / 63,7% | 59,6% | −1,4 (−7,5 a +4,6) | 67,1% |
| soft-mlp fonte-l2 peso-faixa siglip2_b16_512 [log-dens] | 0,350 / 0,283 | 70,4 / 56,5% | 72,1 / 63,1% | 65,7% | +4,6 (−0,4 a +9,6) | 63,9% |
| soft-mlp fonte-l2 +perda-dens peso-faixa siglip2_b16_512 [log-dens] | 0,350 / 0,284 | 70,4 / 56,8% | 70,7 / 61,3% | 66,1% | +5,0 (+0,0 a +10,0) | 63,9% |
| soft-mlp fonte-l2 +perda-dens2 peso-faixa siglip2_b16_512 [log-dens] | 0,350 / 0,283 | 70,5 / 57,1% | 70,7 / 61,3% | 65,4% | +4,3 (−0,7 a +9,6) | 64,6% |
| soft-mlp fonte-l2 só-frações siglip2_b16_512 [frações] | 0,389 / 0,288 | 70,5 / 58,7% | 73,9 / 64,3% | 61,4% | +0,4 (−5,4 a +6,1) | 65,0% |
| soft-mlp fonte-l2 semente1 siglip2_b16_512 [frações] | 0,392 / 0,284 | 70,6 / 59,5% | 73,9 / 64,3% | 62,9% | +1,8 (−3,9 a +7,5) | 66,1% |
| soft-mlp fonte-l2 +perda-dens peso-faixa 100 épocas siglip2_b16_512 [log-dens] | 0,346 / 0,282 | 70,8 / 57,2% | 70,7 / 60,1% | 63,9% | +2,9 (−2,1 a +7,9) | 63,6% |
| soft-mlp fonte-l2 semente2 siglip2_b16_512 [frações] | 0,391 / 0,283 | 70,9 / 59,8% | 74,6 / 65,5% | 62,1% | +1,1 (−5,0 a +7,1) | 65,4% |
| soft-mlp fonte-l2 siglip2_b16_512 [frações] | 0,389 / 0,281 | 70,9 / 59,9% | 73,2 / 63,7% | 61,1% | +0,0 (−5,7 a +5,7) | 65,4% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente1 tips_b14 [frações] | 0,382 / 0,298 | 71,0 / 58,0% | 71,1 / 63,1% | 60,7% | −0,4 (−6,4 a +5,7) | 66,1% |
| soft-mlp fonte-l2 +perda-dens peso-faixa ambos [log-dens] | 0,346 / 0,293 | 71,0 / 57,8% | 71,4 / 61,9% | 63,9% | +2,9 (−2,1 a +7,9) | 63,9% |
| knn fração por fonte tips_b14 | 0,405 / 0,270 | 71,1 / 61,2% | 71,8 / 63,1% | 59,6% | −1,4 (−7,9 a +5,0) | 63,9% |
| soft-mlp fina-l2 +perda-dens peso-faixa siglip2_b16_512 [frações] | 0,366 / 0,294 | 71,3 / 58,9% | 73,6 / 63,1% | 63,2% | +2,1 (−3,6 a +7,9) | 66,4% |
| média: frações-l2 512, 3 sementes | 0,389 / 0,279 | 71,4 / 60,8% | 74,6 / 64,9% | 63,6% | +2,5 (−3,2 a +7,9) | 65,4% |
| soft-mlp fonte-l2 +perda-dens peso-faixa 100 épocas siglip2_b16_512 [frações] | 0,378 / 0,285 | 71,4 / 59,8% | 75,0 / 64,9% | 61,8% | +0,7 (−5,4 a +6,4) | 63,6% |
| soft-mlp fonte-l2 peso-faixa siglip2_b16_512 [frações] | 0,396 / 0,282 | 71,7 / 61,0% | 72,9 / 64,3% | 62,9% | +1,8 (−3,9 a +7,5) | 65,7% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente1 siglip2_b16_512 [frações] | 0,378 / 0,276 | 71,8 / 60,4% | 73,9 / 63,1% | 62,1% | +1,1 (−4,3 a +6,4) | 66,1% |
| soft-mlp fonte-l2 +perda-dens2 peso-faixa siglip2_b16_512 [frações] | 0,376 / 0,278 | 72,0 / 60,9% | 74,3 / 63,7% | 63,2% | +2,1 (−3,6 a +7,9) | 65,4% |
| soft-mlp fonte-l2d siglip2_b16_512 [frações] | 0,390 / 0,277 | 72,0 / 61,4% | 73,2 / 64,9% | 60,4% | −0,7 (−6,4 a +5,0) | 65,4% |
| média: frações-l2 512 + ajuste fino (s8) | 0,320 / 0,285 | 72,1 / 58,9% | 71,8 / 61,3% | 64,6% | +3,6 (−1,1 a +8,2) | 62,5% |
| soft-mlp fonte-l2 +perda-dens peso-faixa ambos [frações] | 0,379 / 0,283 | 72,2 / 60,3% | 74,6 / 66,7% | 62,1% | +1,1 (−4,6 a +6,8) | 65,7% |
| soft-mlp fonte-l2 +perda-dens peso-faixa siglip2_b16_512 [frações] | 0,380 / 0,278 | 72,2 / 61,4% | 73,6 / 63,7% | 62,5% | +1,4 (−4,3 a +7,1) | 65,4% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente2 siglip2_b16_512 [frações] | 0,379 / 0,277 | 72,2 / 61,3% | 74,6 / 66,1% | 60,7% | −0,4 (−6,1 a +5,4) | 65,4% |
| soft-mlp fonte-l2 +perda-dens peso-faixa tips_b14+siglip2_b16_512 [frações] | 0,372 / 0,284 | 72,2 / 59,7% | 73,6 / 64,9% | 61,4% | +0,4 (−5,4 a +6,1) | 65,4% |
| knn densidade tips_b14+siglip2_b16_512 | 0,322 / 0,311 | 72,3 / 60,7% | 72,5 / 61,3% | 61,4% | +0,4 (−5,4 a +6,4) | 64,3% |
| knn densidade tips_b14 | 0,319 / 0,309 | 72,4 / 60,7% | 72,9 / 63,7% | 59,6% | −1,4 (−7,5 a +4,6) | 62,5% |
| média: frações-l2 512, 3 sementes (+perda-dens peso-faixa) | 0,377 / 0,274 | 72,6 / 61,9% | 74,3 / 64,9% | 63,9% | +2,9 (−2,9 a +8,6) | 66,1% |
| média: frações-l2 512 + knn 512 | 0,325 / 0,287 | 72,6 / 61,0% | 74,3 / 64,3% | 62,9% | +1,8 (−3,6 a +7,1) | 65,0% |
| média: frações-l2 512 + classes finas-l2 512 | 0,366 / 0,277 | 72,8 / 61,5% | 77,1 / 69,0% | 63,6% | +2,5 (−3,2 a +8,2) | 64,6% |
| média: frações-l2 512 + s8 escolhido (ajuste fino + 512) | 0,323 / 0,277 | 72,9 / 60,4% | 72,1 / 61,3% | 63,6% | +2,5 (−1,8 a +6,8) | 63,2% |
| média: frações-l2 512 + 224 + ambos | 0,376 / 0,279 | 72,9 / 61,7% | 76,8 / 68,5% | 63,6% | +2,5 (−3,2 a +8,2) | 64,3% |
| média: frações-l2 512 + frações-l2 224 | 0,377 / 0,282 | 73,0 / 61,9% | 74,3 / 66,1% | 65,4% | +4,3 (−1,4 a +10,0) | 64,6% |
| média: frações-l2 512 + sua cabeça log-dens | 0,338 / 0,266 | 73,0 / 61,0% | 75,0 / 64,3% | 66,8% | +5,7 (+0,7 a +10,7) | 63,9% |
| média: frações-l2 512 + ambos | 0,376 / 0,276 | 73,1 / 62,0% | 75,0 / 66,7% | 62,5% | +1,4 (−3,9 a +6,8) | 65,0% |
| média: frações-l2 512 + knn 512 + mlp 224 (s8) | 0,312 / 0,285 | 73,3 / 61,1% | 73,2 / 61,9% | 62,9% | +1,8 (−3,6 a +7,1) | 65,7% |
| média: frações-l2 512, 3 sementes + knn 512 + mlp 224 (s8) | 0,326 / 0,271 | 73,5 / 62,1% | 74,6 / 63,7% | 62,5% | +1,4 (−3,9 a +6,8) | 65,7% |
| média: frações-l2 512 + frações-l2 TIPS + knn 512 | 0,329 / 0,276 | 73,8 / 62,0% | 73,9 / 63,7% | 63,2% | +2,1 (−3,2 a +7,5) | 65,7% |
| média: frações-l2 512 + classes finas-l2 512 + knn 512 + mlp 224 (s8) | 0,316 / 0,279 | 73,9 / 61,8% | 74,3 / 63,7% | 63,2% | +2,1 (−2,9 a +7,1) | 64,6% |
| média: frações-l2 (512, 224, ambos) + knn 512 + ridge fração por fonte 512 | 0,373 / 0,266 | 74,0 / 63,1% | 74,6 / 66,1% | 65,0% | +3,9 (−1,4 a +9,3) | 64,6% |
| média: frações-l2 512 + classes finas-l2 512 + knn TIPS + mlp 224 (s8) | 0,308 / 0,270 | 74,2 / 62,0% | 73,9 / 63,1% | 65,4% | +4,3 (−1,1 a +9,6) | 64,3% |
| seleção gulosa de conjunto | 0,317 / 0,266 | 74,3 / 62,9% | 76,8 / 67,9% | 66,1% | +5,0 (+0,0 a +9,6) | 63,2% |
| média: frações-l2 512 + knn TIPS + mlp 224 (s8) | 0,301 / 0,273 | 74,4 / 62,7% | 74,6 / 64,3% | 64,3% | +3,2 (−2,1 a +8,9) | 65,0% |
| média: frações-l2 512 + frações-l2 TIPS | 0,369 / 0,272 | 74,5 / 63,4% | 74,6 / 67,9% | 62,5% | +1,4 (−4,3 a +7,1) | 65,0% |
| média: frações-l2 512 + frações-l2 TIPS + knn 512 + knn TIPS + mlp 224 (s8) | 0,304 / 0,275 | 74,7 / 63,2% | 75,4 / 65,5% | 62,9% | +1,8 (−3,6 a +7,1) | 65,4% |
| média: frações-l2 512 + knn TIPS | 0,310 / 0,271 | 74,7 / 63,5% | 76,8 / 68,5% | 64,3% | +3,2 (−2,1 a +8,6) | 63,9% |
| média: frações-l2 512 + frações-l2 TIPS + knn TIPS | 0,322 / 0,269 | 74,8 / 63,3% | 77,5 / 70,8% | 63,2% | +2,1 (−3,6 a +7,9) | 64,3% |
| média: frações-l2 (512, TIPS, TIPS+512) + knn TIPS + mlp 224 (s8) **[densidade]** | 0,316 / 0,268 | 75,4 / 63,7% | 76,1 / 66,7% | 64,6% | +3,6 (−1,8 a +8,9) | 64,6% |

</details>

### 9.5 Pipeline só de imagem

- **Escolhido pela regra 3 (CV 66,3%, contra 62,2% do escolhido na seção 8):** massa da seção 8 × densidade escolhida. **Teste: 64,6%** (IC 59,3 a 70,0), MAE 6,3 g, viés −3,4 g. No `n5k_v1` (50 pratos): 58% (seção 8: 62%). No `calib`: 60,0% → 66,7%.
- **Contra as referências:** +3,6 sobre o só imagem da seção 8 (IC −1,8 a +8,9, não distinguível), +9,6 sobre o LLM (IC +2,5 a +16,4), +1,8 sobre o híbrido da seção 8 (IC −3,2 a +6,8).
- **Por faixa de proteína real no teste** (56 pratos por faixa; um prato = 1,8 ponto):

| Faixa | LLM v2 | Só imagem s8 | **Só imagem s9** | Híbrido s8 | **Híbrido s9** | Massa real × dens. s8 | Massa real × dens. s9 | Massa real × dens. LLM v2 | Massa s8 × dens. real |
|---|---|---|---|---|---|---|---|---|---|
| 0–5 g | 84% | 91% | 93% | 89% | 89% | 93% | 93% | 89% | 100% |
| 5–15 g | 71% | 73% | **88%** | 79% | **88%** | 79% | 88% | 88% | 93% |
| 15–25 g | 57% | 48% | **59%** | 62% | **70%** | 64% | 73% | 68% | 77% |
| 25–40 g | 36% | 43% | 38% | 38% | 45% | 55% | 62% | 55% | 62% |
| > 40 g | 27% | 50% | 46% | 46% | 45% | 46% | 64% | 57% | 75% |
| **Todos** | 55,0% | 61,1% | **64,6%** | 62,9% | **67,1%** | 67,5% | 76,1% | 71,4% | 81,4% |
| **> 15 g** | 39,9% | 47,0% | 47,6% | 48,8% | 53,0% | 55,4% | 66,7% | 60,1% | 71,4% |

1. **A densidade melhorou em todas as faixas acima de 5 g** quando a massa é a real (colunas 7 e 8): 79 → 88, 64 → 73, 55 → 62 e 46 → 64. Acima de 40 g o ganho é de 10 pratos em 56.
2. **No pipeline completo o ganho aparece de 5 a 25 g e some acima de 25 g.** Entre 25 e 40 g, o modelo de massa sozinho já limita o acerto a 62% mesmo com a densidade real, e a densidade nova sozinha também dá 62%. Os dois erros têm o mesmo sinal: no teste, a massa sai 12% abaixo na faixa de 25–40 g e 8% abaixo acima de 40 g, e a densidade, 9% e 7% abaixo. Multiplicados, viram −20% e −15%, no limite da tolerância ou fora dela. É regressão à média nos dois fatores.
3. **A CV prometia mais nas faixas altas do que o teste entregou** (CV: 25–40 g 43% → 53%; > 40 g 35% → 39%; teste: 43% → 38% e 50% → 46%). Com 56 pratos por faixa, 5 pontos são 3 pratos, e a seção 8 já tinha a faixa > 40 g bem acima da própria CV (50% contra 35%).
4. **Esta configuração não é para o aparelho:** usa três passagens de ViT-B (SigLIP 2 a 224 e a 384×512, TIPS a 336×448) e uma busca nos 1.684 embeddings do treino. Com a densidade de um backbone só (a cabeça 3d a 384×512, com a mesma massa da seção 8), o teste dá 62,5% pela rota massa × densidade e 65,4% na média geométrica com a rota direta; pela CV ela vale 72,2% de critério de densidade, 3 pontos abaixo da escolhida. O modelo exportado em 8.9 não muda.

### 9.6 Híbrido com o LLM v2

Massa: a da seção 8. Densidade do modelo: a escolhida em 9.4. Do LLM v2 entram a densidade e a massa por prato, e as frações e densidades por fonte (`estimativas_v2_comp_casados.csv`). Pesos só no `calib`; forma escolhida pelo `calib` deixa-fora. As 7 primeiras linhas são as formas da seção 8, refeitas; as demais usam a densidade nova.

| Forma | Pesos | Calib (deixa-fora) | Teste | MAE / viés | Δ vs híbrido s8 (IC 95%) | Δ vs LLM (IC 95%) | > 15 g | n5k v1 |
|---|---|---|---|---|---|---|---|---|
| Dens. s8 · massa modelo × dens LLM | 0 | 63,3% | 63,9% | 6,6 g / −2,6 g | +1,1 (−3,6 a +5,7) | +8,9 (+2,5 a +15,4) | 49,4% | 54% |
| Dens. s8 · massa √(modelo·LLM) × dens LLM | 0 | 62,5% | 61,8% | 7,2 g / −3,4 g | −1,1 (−6,8 a +4,3) | +6,8 (+2,9 a +11,1) | 47,6% | 60% |
| Dens. s8 · massa modelo^a·LLM^(1−a) × dens LLM | 1 | 61,7% | 63,2% | 6,6 g / −2,8 g | +0,4 (−4,3 a +5,0) | +8,2 (+2,1 a +14,3) | 47,0% | 54% |
| Dens. s8 · massa^a × dens (LLM^b·modelo^(1−b)) [escolhida na seção 8] | 2 | 65,0% | 62,9% | 6,3 g / −3,3 g | +0,0 (+0,0 a +0,0) | +7,9 (+1,8 a +14,3) | 48,8% | 60% |
| Dens. s8 · blend log-linear (5 pesos) | 5 | 65,0% | 65,4% | 5,9 g / −1,2 g | +2,5 (−2,1 a +7,1) | +10,4 (+3,6 a +17,1) | 56,0% | 66% |
| Dens. s8 · blend log-linear + proteína direta e do LLM (7 pesos) | 7 | 64,2% | 65,4% | 5,9 g / −1,0 g | +2,5 (−2,1 a +7,1) | +10,4 (+3,6 a +17,1) | 56,0% | 70% |
| Dens. s8 · média geométrica: proteína direta do modelo e do LLM | 0 | 63,3% | 61,1% | 6,9 g / −3,8 g | −1,8 (−6,1 a +2,1) | +6,1 (+0,4 a +11,4) | 44,6% | 64% |
| Dens. s9 · massa modelo × dens LLM | 0 | 63,3% | 63,9% | 6,6 g / −2,6 g | +1,1 (−3,6 a +5,7) | +8,9 (+2,5 a +15,4) | 49,4% | 54% |
| Dens. s9 · massa √(modelo·LLM) × dens LLM | 0 | 62,5% | 61,8% | 7,2 g / −3,4 g | −1,1 (−6,8 a +4,3) | +6,8 (+2,9 a +11,1) | 47,6% | 60% |
| Dens. s9 · massa modelo^a·LLM^(1−a) × dens LLM | 1 | 61,7% | 63,2% | 6,6 g / −2,8 g | +0,4 (−4,3 a +5,0) | +8,2 (+2,1 a +14,3) | 47,0% | 54% |
| Dens. s9 · massa^a × dens (LLM^b·modelo^(1−b)) | 2 | 68,3% | 67,1% | 5,9 g / −3,3 g | +4,3 (+0,4 a +8,6) | +12,1 (+5,7 a +18,6) | 53,0% | 66% |
| Dens. s9 · blend log-linear (5 pesos) | 5 | 67,5% | 66,4% | 5,6 g / −1,8 g | +3,6 (−1,1 a +8,2) | +11,4 (+5,0 a +18,2) | 56,0% | 72% |
| Dens. s9 · blend log-linear + proteína direta e do LLM (7 pesos) | 7 | 68,3% | 69,6% | 5,6 g / −1,5 g | +6,8 (+2,1 a +11,4) | +14,6 (+8,2 a +21,1) | 59,5% | 72% |
| Dens. s9 · média geométrica: proteína direta do modelo e do LLM | 0 | 60,8% | 60,0% | 6,9 g / −4,0 g | −2,9 (−7,9 a +2,1) | +5,0 (+0,0 a +10,0) | 44,0% | 62% |
| Dens. s9 · massa modelo × dens (LLM^b · modelo^(1−b)) **[escolhida]** | 1 | 70,0% | 67,1% | 5,9 g / −3,3 g | +4,3 (+0,4 a +8,6) | +12,1 (+5,7 a +18,6) | 53,0% | 66% |
| Dens. s9 · massa modelo × Σ fonte (fração LLM^w ⊕ modelo) × dens por fonte do LLM | 1 | 62,5% | 65,4% | 6,0 g / −2,7 g | +2,5 (−2,1 a +7,1) | +10,4 (+3,9 a +16,8) | 50,6% | 62% |
| Dens. s9 · massa^a × dens (LLM com frango pela média do treino ^b · modelo^(1−b)) | 2 | 68,3% | 66,8% | 5,9 g / −3,2 g | +3,9 (+0,0 a +7,9) | +11,8 (+5,0 a +18,6) | 52,4% | 64% |
| Dens. s9 · massa modelo × dens (LLM com frango pela média do treino ^b · modelo^(1−b)) | 1 | 69,2% | 66,8% | 5,9 g / −3,2 g | +3,9 (+0,0 a +7,9) | +11,8 (+5,0 a +18,6) | 52,4% | 64% |
| Dens. s9 · massa modelo × dens do modelo (sem LLM) | 0 | 66,7% | 64,6% | 6,3 g / −3,4 g | +1,8 (−3,2 a +6,8) | +9,6 (+2,5 a +16,4) | 47,6% | 58% |
| Dens. s9 · massa modelo × dens (LLM^b · frações do modelo com a fonte do LLM como prior^(1−b)) | 2 | 67,5% | 67,9% | 6,0 g / −3,1 g | +5,0 (+0,7 a +9,3) | +12,9 (+6,4 a +19,3) | 52,4% | 64% |

1. **Forma escolhida: massa do modelo × densidade^(0,4 do LLM · 0,6 do modelo). Teste: 67,1%** (IC 61,4 a 72,5), MAE 5,9 g, viés −3,3 g; +4,3 sobre o híbrido da seção 8 (IC +0,4 a +8,6), +12,1 sobre o LLM (IC +5,7 a +18,6), +6,1 sobre o só imagem da seção 8 (IC +1,1 a +11,4). No `calib` deixa-fora: 70,0% (seção 8: 65,0%). No `n5k_v1`: 66% (seção 8: 60%).
2. **O peso do LLM na densidade caiu para 0,4.** Com a densidade nova, o modelo treinado contribui mais do que o LLM. Misturar a massa do LLM não ajuda (a forma de dois pesos escolhe peso 1 para a massa do modelo e dá o mesmo resultado).
3. **As formas que misturam a densidade do LLM com a densidade nova ficam entre 65% e 70% no teste**, contra 63% a 65% das mesmas formas com a densidade da seção 8. A mistura log-linear de 7 pesos dá 69,6% (IC 64,3 a 75,0); não foi a escolhida (68,3% no `calib`, com 7 pesos ajustados em 120 pratos). O valor honesto é "67%, com formas equivalentes entre 65% e 70%".
4. **Combinar por fonte não funcionou com o LLM v2.** Misturar as frações de massa por fonte do LLM com as do modelo e usar a densidade por fonte do LLM dá 62,5% no `calib` (65,4% no teste). Usar a fonte principal do LLM como prior sobre as frações do modelo não muda as frações (o `calib` escolhe fator 1). A correção da família do frango pela média do treino dá 69,2% no `calib` e 66,8% no teste: efeito nulo.
5. **Acima de 15 g: 48,8% → 53,0%.** O ganho está na faixa de 15–25 g (62% → 70%) e na de 25–40 g (38% → 45%); acima de 40 g não muda (46% → 45%).

### 9.7 Lado do LLM: instrução v3 (identificação e partes da massa)

A análise de 9.2 (f) diz o que pedir ao LLM: **não gramas, e sim o que é cada componente e que parte da massa ele ocupa**. A massa total vem do modelo treinado, que erra 17% contra 30% do LLM.

- **`eval/prompt_v3.md`:** instrução cega nova. Para cada componente visível: o alimento de referência mais próximo, de uma **lista fechada** (os nomes da tabela do Nutrition5k, agrupados por tipo, e os nomes da TACO para comida brasileira), uma **segunda hipótese** quando houver dúvida (ex.: frango ou porco), a certeza, o preparo e a **parte da massa total em %** (as partes somam 100). Por prato, a declaração da **principal fonte de proteína** (10 grupos, os mesmos de 9.2). A instrução ensina a passar de área para massa (área × altura × compactação: folhas soltas pesam um décimo de carne com a mesma área), dá pistas visuais para as confusões medidas no `calib` (frango × porco × peixe, clara de ovo × tofu × queijo) e pede a conferência com unidades conhecidas (um ovo, uma tira de bacon). Nada nela veio do `teste`: as lições são do `calib` e do treino, e as listas são tabelas de referência, não rótulos de pratos.
- **`eval/protein_from_shares.py`:** transforma a saída em densidade do prato, Σ parte × proteína por grama do alimento de referência. Com certeza `media` ou `baixa`, mistura 25% ou 40% da densidade da segunda hipótese. Com mais de uma rodada de estimativa, usa a média geométrica das rodadas. Com `--massa`, multiplica pela massa e devolve a proteína. Conferido com as estimativas v2 convertidas de gramas para partes: reproduz a proteína da v2 nos 594 pratos (diferença máxima de 0,1 g).
- **Formas de híbrido com a v3, declaradas em `dens_avalia.py` antes de as estimativas existirem** (entram sozinhas quando `docs/dados/precisao/estimativas_v3_partes.csv` existe, com pesos no `calib` e a mesma regra de escolha): massa do modelo × densidade v3; massa do modelo × mistura da densidade v3 com a do modelo; a mesma com a massa misturada com a do LLM v2; mistura das três densidades (v2, v3, modelo); frações por fonte do v3 misturadas com as do modelo; densidade v3 com a família do frango pela média do treino; e a fonte principal declarada no v3 como prior sobre as frações do modelo.

**Resultados do LLM v3 (10/11/2026).** Uma rodada cega, 15 estimadores, as 450 fotos do Nutrition5k (400 `d*` = `calib` + `teste`, e as 50 do `n5k_v1`); os itens brasileiros não foram estimados. Todas as partes somam 100 e todos os nomes casaram com as tabelas. Nos 400 pratos `d*` são 1.258 componentes, 3,15 por prato (na v2, 2,67). Metade dos componentes veio com certeza `alta` (627); 484 com `media` e 147 com `baixa`; 625 trazem segunda hipótese. Um estimador (lote 03) gerou recortes ampliados de 4 das suas próprias imagens para ver detalhes; nada além das imagens foi lido. As formas de híbrido e a regra de escolha são as declaradas antes de as estimativas existirem; `dens_avalia.py` rodou sem mudança.

**a) Densidade do LLM, isolada (massa real × densidade).**

| | `calib` (120) | `teste` (280) | `teste`, > 15 g |
|---|---|---|---|
| LLM v2 (gramas por componente) | 76,7% | 71,4% | 60,1% |
| **LLM v3 (identificação + partes)** | **80,8%** | **80,7%** | **72,0%** |
| Modelo treinado (9.4) | 74,2% | 76,1% | 66,7% |
| v3 contra v2, no teste (IC 95% pareado) | | **+9,3 (+4,6 a +13,9)** | |
| v3 contra o modelo treinado, no teste | | +4,6 (−0,7 a +10,0) | |

- **A densidade do v3 é melhor que a do v2**, e o intervalo sustenta isso no teste. Contra a densidade do modelo treinado a diferença não é segura.
- No `calib`, por faixa (v2 → v3): 0–5 g 100 → 92%; 5–15 g 92 → 83%; 15–25 g 71 → 79%; 25–40 g 71 → 71%; > 40 g 50 → 79%. O ganho no `calib` está nos pratos com muita proteína e há perda nos de pouca. No teste o v3 ganha em todas as faixas (tabela em c).

**b) O que a instrução queria melhorar não mudou de forma mensurável no `calib`.** (Só o `calib` tem ingredientes que podem ser lidos; 86 pratos com fonte dominante.)

| No `calib` | LLM v2 | LLM v3 | Modelo treinado (cabeça de frações) |
|---|---|---|---|
| Fonte principal certa | 76,7% (derivada dos componentes) | **77,9%** (declarada); 75,6% (derivada dos componentes) | 79,1% |
| Fração da massa que é fonte de proteína, pratos > 15 g: correlação com a real | 0,87 | 0,88 | |
| Idem: erro absoluto mediano | 6,9 pontos | 6,7 pontos | |
| Idem: média (real: 65,2%) | 62,7% | 63,0% | |

- **A identificação da fonte ficou igual** (77,9% contra 76,7%: um prato). As confusões são as mesmas da v2: frango certo em 10 de 17 (visto como peixe 3 vezes, e como ovo, bovina, laticínio e porco uma vez cada), porco em 17 de 24, bovina em 3 de 6 (as outras 3 como porco). Ovo (19 de 19), prato pronto (7 de 7) e castanhas (9 de 9) saem sempre certos. Quando o v3 acerta a fonte, 82% dos pratos ficam dentro da tolerância com a massa real; quando erra, 63%.
- **A parte da massa das fontes de proteína também ficou igual** (correlação 0,88 contra 0,87).
- **Então o ganho de densidade do v3 não veio pelo caminho que a instrução mirava**, ou veio por uma soma de efeitos pequenos que estas duas medidas não captam. Candidatos, **não isolados aqui**: a lista fechada de nomes (o v2 escrevia nomes livres), a segunda hipótese misturada na densidade (metade dos componentes tem uma), e mais componentes por prato (3,15 contra 2,67), o que inclui queijo, castanhas e molhos em pequena quantidade. Qual deles pesa não foi medido.
- **LLM e modelo erram a fonte em pratos diferentes:** concordam em 71% dos pratos, e pelo menos um dos dois acerta em 92%. A forma que tentava usar isso (a fonte declarada como prior sobre as frações do modelo) não funcionou: o `calib` escolhe fator 1, que não muda nada.

**c) Híbridos com a densidade v3.** Mesma massa (seção 8), mesma regra (pesos e forma no `calib`). As diferenças são contra o híbrido escolhido na seção 8 (62,9%) e contra o híbrido escolhido em 9.6, sem v3 (67,1%).

| Forma | Pesos ajustados | Calib (deixa-fora) | Teste | MAE / viés | Δ vs híbrido s8 (IC 95%) | Δ vs híbrido de 9.6 (IC 95%) | > 15 g | n5k v1 |
|---|---|---|---|---|---|---|---|---|
| Massa do modelo × densidade v3 | nenhum | 66,7% | 70,0% (IC 64,6 a 75,4) | 6,0 g / −2,3 g | +7,1 (+2,1 a +12,1) | +2,9 (−1,4 a +7,1) | 55,4% | 60% |
| Massa do modelo × densidade (v3^b · modelo^(1−b)) | b = 0,5 | 70,8% | 70,4% (IC 65,0 a 75,4) | 5,8 g / −3,0 g | +7,5 (+3,2 a +11,8) | +3,2 (+0,7 a +6,1) | 56,0% | 64% |
| Massa^a (modelo, LLM v2) × densidade (v3^b · modelo^(1−b)) | a = 1; b = 0,5 | 70,0% | 70,4% (IC 65,0 a 75,4) | 5,8 g / −3,0 g | +7,5 (+3,2 a +11,8) | +3,2 (+0,7 a +6,1) | 56,0% | 64% |
| Massa do modelo × densidade (v2 · v3 · modelo) | 0,3 / 0,3 / 0,4 | 69,2% | 69,6% (IC 64,3 a 75,0) | 5,8 g / −3,1 g | +6,8 (+2,8 a +10,7) | +2,5 (0,0 a +5,0) | 56,0% | 66% |
| Massa do modelo × Σ fonte (frações v3 ⊕ modelo) × densidade por fonte do v3 | 0,6 para o v3 | 67,5% | 70,4% (IC 65,0 a 75,7) | 5,8 g / −2,5 g | +7,5 (+2,5 a +12,5) | +3,2 (0,0 a +6,4) | 56,0% | 62% |
| Massa do modelo × densidade (v3 com frango pela média do treino ^b · modelo^(1−b)) | b = 0,5 | 70,0% | 70,0% (IC 64,6 a 75,0) | 5,8 g / −2,9 g | +7,1 (+2,5 a +11,4) | +2,9 (0,0 a +5,7) | 56,0% | 66% |
| **Massa do modelo × densidade (v3^b · frações do modelo com a fonte declarada no v3 como prior^(1−b)) [escolhida]** | fator do prior = 1; b = 0,6 | **72,5%** | **70,4%** (IC 65,0 a 75,7) | 5,8 g / −2,8 g | **+7,5 (+3,2 a +11,8)** | +3,2 (0,0 a +6,4) | 55,4% | 62% |
| Massa do modelo × frações do modelo com a fonte declarada no v3 como prior (sem a densidade do LLM) | fator = 1 | 64,2% | 61,4% (IC 55,7 a 67,1) | 6,4 g / −3,0 g | −1,4 (−6,8 a +3,9) | −5,7 (−10,0 a −1,4) | 45,2% | 64% |

Por faixa de proteína real no teste (56 pratos por faixa):

| Faixa | LLM v2 sozinho | Híbrido s8 | Híbrido de 9.6 (sem v3) | **Híbrido com v3 (escolhido)** | Massa real × dens. v2 | Massa real × dens. v3 | Massa real × dens. do modelo | Massa do modelo × dens. v3 | Massa do modelo × dens. real |
|---|---|---|---|---|---|---|---|---|---|
| 0–5 g | 84% | 89% | 89% | **93%** | 89% | 95% | 93% | 95% | 100% |
| 5–15 g | 71% | 79% | 88% | **93%** | 88% | 93% | 88% | 89% | 93% |
| 15–25 g | 57% | 62% | 70% | **71%** | 68% | 71% | 73% | 71% | 77% |
| 25–40 g | 36% | 38% | 45% | **41%** | 55% | 77% | 62% | 45% | 62% |
| > 40 g | 27% | 46% | 45% | **54%** | 57% | 68% | 64% | 50% | 75% |
| **Todos** | 55,0% | 62,9% | 67,1% | **70,4%** | 71,4% | 80,7% | 76,1% | 70,0% | 81,4% |
| **> 15 g** | 39,9% | 48,8% | 53,0% | **55,4%** | 60,1% | 72,0% | 66,7% | 55,4% | 71,4% |

1. **Forma escolhida pelo `calib`: 70,4% no teste** (IC 65,0 a 75,7), MAE 5,8 g, viés −2,8 g. É +7,5 sobre o híbrido da seção 8 (IC +3,2 a +11,8), +15,4 sobre o LLM v2 sozinho (IC +8,9 a +22,1), +5,7 sobre o só imagem de 9.5 (IC +1,4 a +10,0) e +3,2 sobre o híbrido de 9.6 (IC 0,0 a +6,4: o limite inferior é zero).
2. **O nome da forma escolhida diz mais do que ela faz.** O `calib` ajustou o fator do prior em 1, ou seja, a fonte declarada pelo v3 **não entra**. O que sobra é: densidade = (densidade v3)^0,6 × (frações por fonte de uma cabeça do modelo × densidade média da fonte)^0,4. Ela difere da segunda linha só por usar uma cabeça de frações em vez da média de 9.4, e ganhou dela no `calib` por 2 pratos em 120.
3. **O resultado não depende da forma:** as sete formas que usam a densidade v3 dão de 69,6% a 70,4% no teste, e a mais simples, sem nenhum peso ajustado (massa do modelo × densidade v3), dá 70,0%. No `calib` elas vão de 66,7% a 72,5%.
4. **No `n5k_v1` (50 pratos) o ganho não aparece:** 62% na forma escolhida, contra 66% do híbrido de 9.6 e 60% do da seção 8; as formas com v3 vão de 60% a 66%. Com 50 pratos o intervalo é de ±14 pontos, então isso não contradiz o teste, mas também não o confirma.
5. **Acima de 15 g: 55,4%** (IC 47,6 a 62,5), contra 53,0% do híbrido de 9.6 (+2,4, IC −2,4 a +7,1) e 48,8% do da seção 8. **Acima de 25 g: 47,3%** (9.6: 44,6%). Por faixa, 25–40 g fica em 41% e > 40 g em 54%. O ganho do v3 no resultado final está nos pratos de 0 a 15 g (89 → 93% e 88 → 93%) e, dentro do ruído, acima de 40 g.
6. **Acima de 25 g, o que limita agora é a massa, não a densidade.** Com a massa real, a densidade v3 daria 77% (25–40 g) e 68% (> 40 g). Com a massa do modelo, 45% e 50%. O modelo de massa sozinho, com a densidade real, dá 62% e 75%, e subestima a massa em 12% e 8% nessas faixas.
7. **Uma rodada só.** O item 4 do plano (média de duas rodadas independentes) não pôde ser medido.

### 9.8 Conclusão e o que falta

1. **Onde estamos, no `teste` (280 pratos do Nutrition5k).**
   - Só imagem: 64,6% (IC 59,3 a 70,0).
   - Híbrido com o LLM v2: 67,1% (IC 61,4 a 72,5).
   - **Híbrido com o LLM v3: 70,4% (IC 65,0 a 75,7).** É a primeira vez neste estudo que a estimativa pontual de um pipeline escolhido sem olhar o teste chega à meta de 70%.
2. **O que os dados sustentam.**
   - O híbrido com v3 é melhor que o híbrido da seção 8 (+7,5, IC +3,2 a +11,8), que o só imagem (+5,7, IC +1,4 a +10,0) e que o LLM sozinho (+15,4, IC +8,9 a +22,1).
   - O resultado não depende da forma de combinar: sete formas dão de 69,6% a 70,4%, e a que não ajusta nenhum peso dá 70,0%.
   - A densidade do LLM v3, isolada, é melhor que a do v2 (+9,3 com a massa real, IC +4,6 a +13,9), e a do modelo treinado é melhor que a da seção 8 (+8,6, IC +2,9 a +14,3).
3. **O que os dados não sustentam.**
   - **Que a meta de 70% foi atingida.** O intervalo vai de 65,0% a 75,7%: o valor verdadeiro pode estar 5 pontos abaixo da meta. Contra o híbrido sem v3 (67,1%) o ganho é de +3,2 com limite inferior zero.
   - **Que o resultado se repete.** Foi uma única rodada cega. No `n5k_v1` (50 pratos) a mesma forma dá 62%, abaixo dos 66% do híbrido sem v3; com 50 pratos isso não refuta nem confirma.
   - **Que os pratos ricos em proteína estão resolvidos.** Acima de 15 g: 55,4% (IC 47,6 a 62,5). Acima de 25 g: 47,3%. Nas faixas de 25–40 g e > 40 g: 41% e 54%. Os 70% do total vêm de 93% nos pratos com até 15 g e 71% nos de 15 a 25 g.
   - **Que a instrução v3 funciona pelo motivo para o qual foi escrita.** No `calib`, a fonte principal certa (78% contra 77%) e a parte da massa das fontes (correlação 0,88 contra 0,87) não mudaram. A densidade melhorou, mas o mecanismo não foi isolado (9.7 b).
   - **Qualquer coisa fora do Nutrition5k.** São 280 pratos de um refeitório, com câmera a altura fixa, e a lista de nomes da instrução é a tabela do próprio Nutrition5k. Nenhum prato brasileiro foi estimado com a v3. O problema de escala de 8.7 e 8.8 continua.
4. **O que este pipeline exige.** Um LLM de visão em nuvem para a densidade, mais três passagens de ViT-B e uma busca nos embeddings do treino para a massa e a densidade do modelo. Não roda no aparelho. O pipeline sem nuvem é o só imagem: 64,6%, ou 61,1% com o modelo de um backbone exportado em 8.9.
5. **Pratos ricos em proteína, que eram o alvo: melhora parcial.** De 15 a 25 g, sim (híbrido: 62% na seção 8 → 70% em 9.6 → 71% com v3). Acima de 25 g, nada que se distinga do ruído: 38% → 45% → 41% e 46% → 45% → 54%.
6. **Por que os pratos grandes continuam fracos: agora é a massa.** Com a massa real, a densidade v3 daria 77% e 68% nas duas faixas acima de 25 g; com a massa do modelo, dá 45% e 50%. O modelo de massa subestima esses pratos em 8% a 12%, e mesmo com a densidade real ficaria em 62% e 75%. Esticar as previsões não resolve (testado com ajuste cruzado em 9.4).
7. **O que falta, em ordem de ganho esperado.**
   - **Corrigir a massa dos pratos grandes.** É o que limita acima de 25 g, e esta seção não tocou no modelo de massa. O peso por faixa, que ajudou a densidade, não foi testado na massa.
   - **Uma segunda rodada cega da v3.** Serve para duas coisas: ver se os 70% se repetem, e medir a média de duas rodadas, que `protein_from_shares.py` já calcula.
   - **Entender de onde vem o ganho da v3** (lista fechada, segunda hipótese ou mais componentes), antes de depender dela.
   - **Ajuste fino em GPU com a perda de frações por fonte na resolução maior.** A cabeça que funcionou aqui foi treinada sobre features congeladas. Não cabe em CPU.
   - **Dados brasileiros de cima, com celular:** continua valendo o que 8.10 diz.
8. **Para o produto.** A recomendação de 8.10 não muda: o seletor de porção continua necessário, sobretudo nos pratos com mais de 25 g, em que nenhum pipeline passa de 54%. O que muda é o ponto de partida do seletor: com a v3 e o modelo de massa, 93% dos pratos com até 15 g e 71% dos de 15 a 25 g já saem dentro da tolerância no Nutrition5k. A saída "fração da massa por fonte de proteína" é uma pergunta que a usuária consegue responder ("frango é mais ou menos metade do prato?").
9. **Registro de método.** (a) Antes de qualquer leitura do teste, a regra 3 foi restringida à densidade escolhida (a primeira versão deixava concorrer a massa × cada uma das densidades, e a escolha trocava de vencedor a cada configuração nova), o TIPS foi extraído e a grade dos vizinhos foi ampliada; `dens_avalia.py --so-cv` e um ensaio no `n5k_v1` foram usados nessa fase. (b) O script completo rodou quatro vezes no teste: a primeira é a avaliação; a segunda e a terceira só acrescentaram relatos (a decomposição por faixa de 9.5 e o intervalo da densidade isolada); a quarta regravou o resumo depois de um ensaio do caminho do LLM v3 feito só no `calib`, com as estimativas v2 convertidas para o formato v3. As escolhas e as tabelas são as mesmas nas quatro (a saída da quarta é idêntica à da terceira). (c) Os números de teste das 150 densidades não escolhidas são relato, não seleção. (d) LLM v3: as estimativas chegaram depois de a instrução, o script de conversão, as formas de híbrido e a regra de escolha estarem fixados. `dens_avalia.py` rodou uma vez, sem mudança, e escolheu a forma pelo `calib`. Depois acrescentei ao script um bloco só de relato ("4d": densidade v3 com a massa real, intervalos contra o híbrido de 9.6, tabela por faixa e, no `calib`, a fonte principal e a parte das fontes) e rodei mais uma vez; a saída até o bloco novo é idêntica à da execução anterior. No total, o teste foi lido em seis execuções, todas com as mesmas escolhas. (e) Um estimador da v3 (lote 03) ampliou recortes de 4 das suas próprias imagens; nenhum outro arquivo foi lido.

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

Seção 9 (densidade de proteína). Usa as features e o banco de previsões da seção 8; o TIPS (`tips_b14.npz`, de `gs://tips_data`) é extraído aqui.
```
cd eval/treino; export OMP_NUM_THREADS=2
python3 dens_analise.py                                           # análise de erro: só treino fora-do-fold e calib
python3 zen_extrai.py $TD tips_b14 --orient 1 --conj treino,aval  # ~40 min
python3 dens_experimentos.py ridge siglip2_b16_512 siglip2_b16 tips_b14
python3 dens_experimentos.py knn tips_b14 siglip2_b16_512 siglip2_b16 "tips_b14|siglip2_b16_512"
python3 dens_experimentos.py soft siglip2_b16_512                 # todas as cabeças softmax (~1 min cada, 2 threads)
F="soft-mlp fonte-l2 +perda-dens peso-faixa"
DENS_CFG="$F;soft-mlp fonte-l2" python3 dens_experimentos.py soft ambos "tips_b14|siglip2_b16_512"
DENS_CFG="$F;soft-mlp fonte-l2;$F semente1" python3 dens_experimentos.py soft tips_b14
DENS_CFG="$F;soft-mlp fonte;soft-mlp fonte +perda-dens;soft-mlp ingrediente +perda-dens peso-faixa" python3 dens_experimentos.py soft siglip2_b16
DENS_CFG="soft-mlp fonte-l2" python3 dens_experimentos.py soft siglip2_b16_512+esp
python3 dens_avalia.py --so-cv   # escolhas por CV e calib, sem ler o teste
python3 dens_avalia.py           # avaliação no teste; com estimativas_v3_partes.csv, inclui as formas com o LLM v3
python3 ../protein_from_shares.py ../../docs/dados/precisao/estimativas_v3_partes.csv ../../docs/dados/precisao/estimativas_v3_pratos.csv
```
As previsões ficam em `$TD/resultados/dens/banco/` e os resumos em `$TD/resultados/dens/{analise,avalia}.json`. Tempo de CPU: cerca de 1 h 15 min, dos quais 40 min são a extração do TIPS.
