# Modelo treinado: proteína só pela foto, sem informar porção (09/11/2026)

**Pergunta:** um modelo **treinado** (e não um LLM em nuvem) consegue sugerir a proteína só pela foto e atingir a meta de **≥ 70% dos pratos dentro de ±20% ou ±5 g**?
**Resposta curta: não, com os dados que existem hoje.** Com higiene de dados rigorosa, o melhor modelo só de imagem fica em **~48–52%** no `teste` (280 pratos), abaixo do LLM (55%). Profundidade (RGB-D) não ajudou. O melhor resultado foi o **híbrido** "LLM identifica e dá a densidade, modelo treinado dá o peso", com **~60% no teste (59,6%) e 60–62% no n5k v1**: melhora o LLM em 5 pontos, mas ainda fica 10 pontos abaixo da meta. A comida brasileira em foto que não é zenital **quebra** o modelo de peso (Liv Up 3%). **Atualização (seção 7):** treinar também com os quadros laterais do Nutrition5k deixa o modelo robusto ao ângulo (foto lateral 44% → 49%) e leva o Brasil de 22% para 36% (Liv Up 3% → 26%), mas o híbrido continua em ~59–60%, longe de 70%. **Atualização (seção 8, premissa de uma única foto de cima):** com 8 orientações da foto, resolução maior, ajuste fino parcial e média de modelos, o modelo **só de imagem sobe de 50% para 61%** no teste (IC 55 a 67; +11 pontos sobre o anterior, IC +5 a +17) e alcança o LLM (55%). O erro de massa cai de 22% para 17%. O **híbrido escolhido fica em 63%** (IC 57 a 69; formas equivalentes dão de 62% a 66%), sem diferença segura para os 59,6% de antes. **A meta de 70% continua não atingida**, e o erro que sobra está mais na densidade (que alimento é) do que no peso. Cerca de metade dos pratos pode ser sugerida sozinha com ~70% de acerto, mas essa metade é, na maior parte, de pratos com pouca proteína. Fora da distância fixa do Nutrition5k o modelo erra a escala, a menos que seja treinado com escala aleatória ou com o prato enquadrado num guia. **Atualização (seção 9, densidade de proteína):** prever a fração da massa do prato que é cada fonte de proteína, somar a recuperação de pratos parecidos e um terceiro backbone melhora a densidade do modelo de 67,5% para 76,1% quando a massa é a real (IC do ganho: +2,9 a +14,3). No resultado final o ganho é menor: **só imagem 61,1% → 64,6%** (IC 59 a 70; ganho não distinguível) e **híbrido 62,9% → 67,1%** (IC 61 a 72,5; ganho de +4,3, IC +0,4 a +8,6). Melhoram os pratos de 5 a 25 g; **os de mais de 25 g praticamente não melhoram** (38–46%), porque ali o erro de massa já pesa tanto quanto o de densidade. No LLM, o erro estava na parte da massa que é proteína e na fonte, não na tabela; a instrução v3 pede identificação e partes da massa em vez de gramas. **Com as estimativas v3, o híbrido fica em cerca de 70% no teste: 71,4% com a média de duas rodadas cegas independentes** (IC 66,4 a 76,4; +8,6 sobre o híbrido da seção 8, IC +3,9 a +13,2; +4,3 sobre o híbrido com v2, IC +1,4 a +7,5), 70,4% na primeira rodada sozinha e 68,6% na segunda, feita como réplica. **É a primeira estimativa pontual na meta de 70%, e não é a meta demonstrada:** o intervalo começa em 66%, a réplica sozinha ficou abaixo de 70%, são 280 pratos do Nutrition5k, o `n5k_v1` (50 pratos) dá 62%, o pipeline depende de um LLM em nuvem, e os pratos com mais de 25 g continuam em 45% e 55%, agora limitados pelo erro de massa. As duas rodadas erram nos mesmos pratos, então repetir a estimativa não melhora o resultado. **Atualização (seção 10, treino em GPU):** 15 configurações treinadas no Kaggle (2 × T4, 5,7 h) dão um modelo de massa melhor: a média de cinco backbones congelados de alta resolução baixa o erro de massa de 16,9% para 15,3% e reduz o viés nos pratos com mais de 25 g de −12% e −8% para −6% e −2%. Com ele, o **só imagem sobe de 64,6% para 69,6%** (IC 64 a 75; ganho de +5,0, IC +0,4 a +9,6), e os pratos com mais de 25 g saem de 42% para 54%. O **híbrido com o LLM v3 fica em 72,9%** (IC 67,5 a 78,2), +1,4 sobre os 71,4% de antes, sem diferença segura. Todas as formas com a densidade v3 dão de 70% a 75% no teste nas três leituras do LLM, mas **a meta de 70% continua não demonstrada**: o limite inferior do intervalo é 67,5%, e a forma de híbrido que o `calib` prefere entre todas dá 68,2%. Um modelo único que caiba no aparelho dá 62,9%, sem ganho seguro sobre os 61,1% da seção 8; o ganho está em juntar backbones grandes, no servidor. **Atualização (seção 11, conjunto de teste novo):** um segundo conjunto, o `teste2` (829 pratos do Nutrition5k nunca vistos por nenhum modelo, em 190 cadeias), foi lido **uma vez**, com o pipeline congelado e a regra escrita antes. **Híbrido com o LLM v3: 70,8%** (ponderado para as cinco faixas de proteína terem o mesmo peso; IC por cadeia de 66,3 a 75,2). **Só imagem: 69,7%** (IC 65,2 a 74,2). Os dois conjuntos juntos (1.109 pratos): 71,5% (IC 67,9 a 75,2). **Veredito pela regra: 70% não demonstrado e não refutado.** O nível de cerca de 70% se repetiu; o limite inferior do intervalo, não. No `teste2` o LLM não acrescentou nada distinguível ao só imagem (+1,1, IC −2,1 a +4,3), os pratos com mais de 25 g ficaram em 52%, e o modelo único de aparelho, em 64,5%. O `teste2` não é independente do `teste` (722 dos 829 pratos são o mesmo prato do `teste` em outro momento da montagem), e tudo continua sendo Nutrition5k. Servir o pipeline só de imagem custa cerca de 1,7 a 2,0 s por foto numa GPU T4, com 3,6 GB de pesos. **Seção 12 (desenho e pré-registro, sem resultados):** com a decisão de usar só imagem e reduzir as nove redes a um modelo (no máximo dois), o treino do Nutrition5k cresce de 1.684 para 2.281 pratos sem tocar no teste2, a próxima execução em GPU está especificada, e o teste2 pode ser lido mais duas vezes no máximo, uma por candidato congelado. A expectativa registrada é que um modelo só fique de 1 a 3 pontos abaixo das nove redes e que a meta não seja demonstrada só com o Nutrition5k. **Atualização (seção 12.6, terceira execução em GPU):** a regra pré-registrada escolheu como modelo único o **controle sem destilação** (SigLIP 2 B/16 a 224 px, a receita de antes treinada com o treino_v2, uma orientação): os alunos destilados com ajuste fino ficaram 4 a 5 pontos abaixo dele na validação cruzada, e nenhum par de modelos passou no critério. Na única leitura do teste2 ele dá **66,0%** (IC por cadeia 61,6 a 70,1): **70% não demonstrado**, 3,7 pontos abaixo das nove redes (IC −9,0 a +1,4) e 1,5 acima do mesmo modelo treinado com o treino antigo (IC −2,2 a +5,5); os pratos com mais de 25 g ficam em 48%. Ele custa 8 ms por foto numa T4 e 186 MB, contra 1,7 a 2,0 s e 3,6 GB das nove redes. Os 35% a mais de treino deram +2 a +3 pontos de validação cruzada à mesma receita. A destilação falhou por um erro de receita (os pesos das perdas novas abafaram a perda de massa) e, onde a comparação é limpa, custou 1,5 ponto. A base externa Flavoria não foi adotada (−1 ponto no Nutrition5k). Fora da regra de 12.3, um único backbone grande congelado (So400m) com duas cabeças empata com o conjunto de cinco redes na validação cruzada (71 a 72%). **Atualização (seção 12.7, emenda aprovada pelo fundador em 12/11/2026):** esse modelo foi lido no teste2, uma vez, com a definição fixada antes: **69,7%** (IC por cadeia 65,0 a 74,3; com a correção para as leituras repetidas, 64,1 a 75,4). **70% continua não demonstrado.** Ele repete o nível das nove redes (diferença pareada 0,0; IC −4,7 a +4,3) com um backbone só, a 61 ms por foto numa T4 e 859 MB, e fica 3,7 pontos acima do modelo de aparelho (IC −1,8 a +9,5; na validação cruzada, +3,2 com IC +0,4 a +6,0). É um modelo de servidor: não cabe no aparelho como está. Os pratos com mais de 25 g ficam em 54%. Foi a terceira leitura do teste2, decidida depois de ver a segunda: ele está gasto como teste de registro, e a prova da meta precisa de um conjunto novo. **Seção 13 (desenho e pré-registro da quarta execução, sem resultados):** com a decisão de usar qualquer base de treino, com a procedência registrada, entram cinco bases externas pesadas (OmniFood8K, a única com proteína; Flavoria, Food Portion Benchmark, Mensa e LeFood, só com massa). As bases de massa entram só na cabeça de massa, porque juntar o Flavoria nas duas cabeças custou 2 pontos no Nutrition5k. A regra de adoção protege o Nutrition5k (queda máxima de 1 ponto) e mede o ganho fora dele; partes de cada base ficam seguradas como teste de registro, o OmniFood8K por prato. O teste2 terá no máximo mais uma leitura. Nenhum desses conjuntos mede prato brasileiro em foto de celular: a meta do produto só pode ser julgada num conjunto brasileiro pesado, que ainda não existe. **Seção 12.8 (calorias, carboidratos e gordura, só validação cruzada):** com o mesmo tipo de critério da proteína, o modelo de servidor acerta 70,5% das kcal (piso de 50 kcal), 78,6% dos carboidratos (piso de 10 g) e 68,4% da gordura (piso de 5 g), e o modelo de aparelho, 65,0%, 75,0% e 65,6%. Nas refeições grandes a estimativa fica cerca de 20% abaixo do real (carboidrato acima de 50 g: 46% de acerto). Dá para mostrar kcal e carboidrato como estimativa, não como número para dose de insulina. **Atualização (seção 13.11, quarta execução):** com cinco bases externas pesadas, a regra pré-registrada só admitiu a massa do OmniFood8K na cabeça de massa. O modelo de servidor escolhido dá **69,2% no teste2** (IC por cadeia de 65,0 a 73,7; na quarta e última leitura, com a correção para quatro leituras, 64,1 a 74,7): **70% não demonstrado**. Fica no mesmo nível do D1 e das nove redes, com 57% nos pratos acima de 25 g: o maior número do estudo nessa faixa, sem diferença segura. Nas cenas de pratos chineses nunca vistos do OmniFood8K, os três modelos lidos ficam entre 59% e 66%. Cada base externa melhora muito a massa do próprio domínio (Flavoria de 52% para 67% a ±20%), mas não transfere para os outros e custa de 2 a 3 pontos no Nutrition5k. A escala da foto é o limite que mais aparece: quem fotografa de mais perto muda a massa em 3 a 5 vezes. A próxima alavanca é enquadrar pelo prato (o círculo-guia), e o que falta para julgar os 70% é um conjunto brasileiro pesado de celular. O teste2 está encerrado. kcal e carboidrato no teste2: 71% e 84%.

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
**Resposta curta: a densidade melhorou de forma mensurável, no modelo treinado e no LLM. Com a instrução nova do LLM (v3), o híbrido fica em cerca de 70% no teste: 71,4% com a média de duas rodadas cegas (IC 66 a 76), 70,4% e 68,6% em cada rodada. A estimativa pontual alcança a meta de 70% pela primeira vez, mas o intervalo começa em 66% e a réplica sozinha ficou abaixo de 70%. Sem LLM, o ganho é pequeno. Os pratos com mais de 25 g continuam fracos (45% e 55%).** Números do `teste` (280 pratos), com tudo escolhido por validação cruzada no treino e pesos do híbrido só no `calib`.

| Medida no `teste` | Seção 8 (escolhido) | Seção 9 (escolhido) | Diferença (IC 95%, bootstrap pareado) |
|---|---|---|---|
| **Densidade isolada:** massa real × densidade do modelo | 67,5% | **76,1%** | **+8,6 (+2,9 a +14,3)** |
| Só imagem, sem LLM | 61,1% | **64,6%** (IC 59,3 a 70,0) | +3,6 (−1,8 a +8,9): não distinguível |
| Híbrido: massa do modelo × densidade (LLM v2 e modelo) | 62,9% | **67,1%** (IC 61,4 a 72,5) | **+4,3 (+0,4 a +8,6)** |
| **Híbrido com o LLM v3, média de duas rodadas cegas** (9.7 d) | 62,9% | **71,4%** (IC 66,4 a 76,4) | **+8,6 (+3,9 a +13,2)**; contra o híbrido com v2: +4,3 (+1,4 a +7,5) |
| Idem, rodada 1 sozinha / rodada 2 sozinha (réplica) | | 70,4% (IC 65,0 a 75,7) / 68,6% (IC 63,2 a 73,9) | +7,5 (+3,2 a +11,8) / +5,7 (+1,1 a +10,4); rodada 2 contra rodada 1: −1,8 (−5,4 a +1,4) |
| Densidade do LLM isolada: massa real × densidade (v2 → v3) | 71,4% | **80,7%** (rodada 2: 80,4%; média: 80,7%) | **+9,3 (+4,6 a +13,9)** |
| Pratos com mais de 15 g: só imagem / híbrido v2 / híbrido v3 (média das rodadas) | 47,0% / 48,8% / — | 47,6% / 53,0% / 57,7% | |
| Pratos com mais de 25 g: híbrido v2 / híbrido v3 (média das rodadas) | 42,0% / — | 44,6% / 50,0% | |
| LLM sozinho (referência) | 55,0% (39,9% acima de 15 g) | | |
| **Meta** | **70%** | | |

- **A densidade do modelo treinado passou de 67,5% para 76,1%** quando medida com a massa real, e ficou acima da do LLM v2 na estimativa pontual (71,4%; diferença de +4,6, IC −1,4 a +10,4, não segura). O ganho veio de três coisas: prever **quanto da massa do prato é cada fonte de proteína** em vez da densidade direta, **recuperar pratos parecidos** do treino, e um terceiro backbone (TIPS) para essa recuperação.
- **O ganho não chegou inteiro ao resultado final.** Só imagem: 61,1% → 64,6%, dentro do ruído. Híbrido: 62,9% → 67,1%, com o intervalo encostado em zero. É a primeira vez que o híbrido escolhido se separa do anterior, mas por pouco, e formas não escolhidas vão de 65% a 70%.
- **Os pratos com mais de 25 g não melhoraram de forma que se distinga do ruído** (só imagem: 43% → 38% e 50% → 46%; híbrido: 38% → 45% e 46% → 45%; são 56 pratos por faixa). O que melhorou foram os pratos de 5 a 25 g. Nos pratos grandes, o erro que sobra já é **tanto de massa quanto de densidade**, e os dois puxam para baixo (9.5).
- **Lado do LLM: com a instrução v3 o híbrido fica em cerca de 70%.** Média de duas rodadas cegas independentes: 71,4% (IC 66,4 a 76,4), +8,6 sobre o híbrido da seção 8 (IC +3,9 a +13,2) e +4,3 sobre o híbrido com v2 (IC +1,4 a +7,5). Rodada 1 sozinha: 70,4%; rodada 2 sozinha, a réplica: 68,6%. A análise mostrava que o LLM errava a parte da massa e a fonte, não a tabela; a v3 (`eval/prompt_v3.md`) pede identificação e partes da massa em vez de gramas. A densidade do LLM subiu de 71,4% para 80,7% com a massa real, e isso se repetiu na segunda rodada (80,4%). **Ressalvas:** a meta não está demonstrada (o intervalo começa em 66% e a réplica sozinha ficou em 68,6%); no `n5k_v1` (50 pratos) dá 62% nas três leituras; no `calib`, a fonte certa e a parte das fontes não mudaram, então o mecanismo do ganho não está isolado; as duas rodadas erram nos mesmos pratos (correlação de 0,98), então repetir não melhora; e o pipeline precisa de um LLM em nuvem (9.7 e 9.8).
- **Acima de 25 g o limite agora é a massa.** Com a massa real, a densidade v3 (média das rodadas) daria 77% e 68% nas duas faixas de cima; com a massa do modelo, 46% e 54%.

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

**Resultados do LLM v3 (10/11/2026), rodada 1.** (A segunda rodada e a média das duas estão em (d), no fim desta subseção; os itens (a) a (c) são da rodada 1 e ficam como foram escritos.) Uma rodada cega, 15 estimadores, as 450 fotos do Nutrition5k (400 `d*` = `calib` + `teste`, e as 50 do `n5k_v1`); os itens brasileiros não foram estimados. Todas as partes somam 100 e todos os nomes casaram com as tabelas. Nos 400 pratos `d*` são 1.258 componentes, 3,15 por prato (na v2, 2,67). Metade dos componentes veio com certeza `alta` (627); 484 com `media` e 147 com `baixa`; 625 trazem segunda hipótese. Um estimador (lote 03) gerou recortes ampliados de 4 das suas próprias imagens para ver detalhes; nada além das imagens foi lido. As formas de híbrido e a regra de escolha são as declaradas antes de as estimativas existirem; `dens_avalia.py` rodou sem mudança.

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
7. **Uma rodada só, até aqui.** A segunda rodada e a média das duas estão em (d), logo abaixo.

**d) Segunda rodada cega: regra de leitura, escrita antes de rodar qualquer coisa no teste.** Uma segunda rodada independente foi feita com a mesma instrução e as mesmas 450 fotos: 15 estimadores novos, com as saídas da rodada 1 retiradas da pasta antes. `estimativas_v3_partes.csv` passou a ter as duas rodadas (coluna `rodada`; 1.431 e 1.425 linhas; todas as partes somam 100). A leitura fica fixada assim:

- **Número principal:** o mesmo procedimento já declarado, aplicado à **média das duas rodadas** (`protein_from_shares.py` já faz a média geométrica da densidade por prato). Era o item 4 do plano original.
- **Réplica:** a **rodada 2 sozinha**, pelo mesmo procedimento. É ela que diz se os 70,4% da rodada 1 se repetem.
- **Rodada 1 sozinha:** fica como já foi relatada (70,4%).
- **Mesma regra de escolha para cada entrada:** pesos e forma no `calib`, com validação deixa-fora; as mesmas formas; nenhuma forma nova; nenhuma mudança na instrução. Cada entrada (rodada 2, média) escolhe a sua forma no `calib`.
- **Como julgar:** "confirmado" se a rodada 2 sozinha e a média ficarem em 70% ou mais no teste; "enfraquecido" se a rodada 2 ficar claramente abaixo (fora do que duas rodadas do mesmo procedimento costumam diferir) ou se a média cair abaixo de 70%; em qualquer caso o intervalo de confiança continua valendo, e 70% só estaria demonstrado com o limite inferior acima de 70%.
- Mudança de código necessária, só de mecânica: um filtro de rodada em `llm_v3` (variável `DENS_V3_RODADA`), a gravação das estimativas por prato de cada execução e um script que compara as rodadas (`dens_rodadas.py`). No conversor, o desempate da fonte principal entre rodadas passa a ser determinístico (vale a da primeira rodada).

**Resultado das duas rodadas.** Cada entrada escolheu a sua forma e os seus pesos no `calib`. A rodada 1 foi refeita pelo código novo e saiu idêntica ao que já estava relatado. Rodada 2: 1.256 componentes nos 400 pratos `d*` (3,14 por prato), 656 com certeza `alta`, 590 com segunda hipótese.

| Entrada do LLM v3 | Forma escolhida no `calib` | Calib (deixa-fora) | Teste | Δ vs híbrido s8 (62,9%) | Δ vs híbrido de 9.6 (67,1%) | Δ vs rodada 1 | > 15 g | > 25 g | n5k v1 |
|---|---|---|---|---|---|---|---|---|---|
| Rodada 1 sozinha (já relatada) | v3^0,6 · frações do modelo^0,4 | 72,5% | 70,4% (IC 65,0 a 75,7) | +7,5 (+3,2 a +11,8) | +3,2 (0,0 a +6,4) | | 55,4% | 47,3% | 62% |
| **Rodada 2 sozinha (réplica)** | v3^0,5 · modelo^0,5 | 73,3% | **68,6%** (IC 63,2 a 73,9) | +5,7 (+1,1 a +10,4) | +1,4 (−1,8 a +4,6) | −1,8 (−5,4 a +1,4) | 54,2% | 45,5% | 62% |
| **Média das duas rodadas (número principal)** | v3^0,5 · modelo^0,5 | 72,5% | **71,4%** (IC 66,4 a 76,4) | **+8,6 (+3,9 a +13,2)** | **+4,3 (+1,4 a +7,5)** | +1,1 (−1,4 a +3,6) | 57,7% (IC 50,0 a 64,9) | 50,0% | 62% |

| No teste | Rodada 1 | Rodada 2 | Média das duas |
|---|---|---|---|
| Massa real × densidade v3 (todos / > 15 g) | 80,7 / 72,0% | 80,4 / 72,0% | 80,7 / 72,6% |
| Idem, no `calib` | 80,8% | 80,0% | 80,8% |
| Massa do modelo × densidade v3, sem nenhum peso | 70,0% | 69,3% | 70,7% |
| As sete formas com a densidade v3 | 69,6 a 70,4% | 67,9 a 70,4% | 69,3 a 71,4% |
| Híbrido escolhido, 0–5 g | 93% | 91% | 93% |
| 5–15 g | 93% | 89% | 91% |
| 15–25 g | 71% | 71% | 73% |
| 25–40 g | 41% | 41% | 45% |
| > 40 g | 54% | 50% | 55% |
| Massa real × densidade v3: 0–5 / 5–15 / 15–25 / 25–40 / > 40 g | 95 / 93 / 71 / 77 / 68% | 93 / 93 / 73 / 75 / 68% | 93 / 93 / 73 / 77 / 68% |
| `calib`: fonte principal declarada certa | 77,9% | 80,2% | 77,9% |

**Concordância entre as rodadas 1 e 2** (`dens_rodadas.py`):

- **Densidade por prato:** correlação de 0,977 no teste (0,971 acima de 15 g), 0,984 no `calib` e 0,988 no `n5k_v1`. A diferença relativa mediana entre as duas rodadas é de 3%; em 92% dos pratos do teste uma fica a menos de 20% da outra.
- **Fonte principal declarada:** a mesma em 94% dos pratos do teste (93% acima de 15 g), 93% no `calib` e 98% no `n5k_v1`. As trocas são as confusões de sempre (porco × frango × bovina, leguminosa × frango, prato pronto × frango).
- **Pratos que entram e saem da tolerância no teste:** no híbrido escolhido de cada rodada, 182 estão dentro nas duas, 15 só na rodada 1, 10 só na rodada 2 e 73 em nenhuma (25 pratos trocam, 8,9%). Com a massa real × densidade v3, 15 trocam (8 e 7). Com a massa do modelo × densidade v3 sem pesos, 20 trocam (11 e 9).

O que as duas rodadas mostram:

1. **A réplica ficou em 68,6%, abaixo de 70%, e dentro do que duas rodadas diferem.** A diferença para a rodada 1 é de −1,8 ponto (IC −5,4 a +1,4): 5 pratos líquidos em 280, com 25 trocando de lado. Parte dela vem da forma que o `calib` escolheu em cada rodada: na rodada 2, a forma que a rodada 1 tinha escolhido dá 70,4% no teste, e a forma sem pesos dá 69,3% contra 70,0%.
2. **O número principal, a média das duas rodadas, é 71,4%** (IC 66,4 a 76,4), +8,6 sobre o híbrido da seção 8 e +4,3 sobre o híbrido sem v3, agora com o intervalo acima de zero (+1,4 a +7,5).
3. **Fazer a média de duas rodadas não melhorou a densidade** (80,7% com a massa real, igual à rodada 1). As duas rodadas concordam demais para a média reduzir erro: correlação de 0,98, mesma fonte em 94% dos pratos. Os erros do LLM são **sistemáticos** (os mesmos pratos, a mesma confusão), não ruído de uma leitura para outra. A diferença de +1,1 ponto entre a média e a rodada 1 no híbrido está dentro do ruído (IC −1,4 a +3,6). Uma terceira rodada não deve mudar o quadro.
4. **O que se repetiu sem ambiguidade:** a densidade v3 com a massa real (80,7% e 80,4% no teste; 80,8% e 80,0% no `calib`), o nível do híbrido (entre 68,6% e 71,4% nas três leituras), o `n5k_v1` em 62% nas três, e os pratos acima de 25 g fracos (41–45% e 50–55%).
5. **Pela regra escrita acima:** não é "confirmado", porque a rodada 2 sozinha não chegou a 70%; não é "enfraquecido", porque ela não ficou claramente abaixo da rodada 1 e a média ficou acima de 70%. **A leitura fica inalterada e mais firme: o híbrido com a v3 vale cerca de 70% no teste (68,6% a 71,4% conforme a rodada), com intervalo de 66% a 76% no número principal.** A meta de 70% continua não demonstrada: o limite inferior está abaixo dela nas três leituras.

### 9.8 Conclusão e o que falta

1. **Onde estamos, no `teste` (280 pratos do Nutrition5k).**
   - Só imagem: 64,6% (IC 59,3 a 70,0).
   - Híbrido com o LLM v2: 67,1% (IC 61,4 a 72,5).
   - **Híbrido com o LLM v3, média de duas rodadas cegas: 71,4% (IC 66,4 a 76,4).** Rodada 1 sozinha: 70,4% (IC 65,0 a 75,7). Rodada 2 sozinha, a réplica: 68,6% (IC 63,2 a 73,9). É a primeira vez neste estudo que a estimativa pontual de um pipeline escolhido sem olhar o teste chega à meta de 70%; a réplica sozinha ficou 1,4 ponto abaixo dela.
2. **O que os dados sustentam.**
   - O híbrido com v3 (média das duas rodadas) é melhor que o híbrido da seção 8 (+8,6, IC +3,9 a +13,2), que o híbrido com v2 (+4,3, IC +1,4 a +7,5), que o só imagem (+6,8, IC +2,9 a +10,7) e que o LLM sozinho (+16,4, IC +9,6 a +23,2). Com uma rodada só, o ganho sobre o híbrido com v2 não se separa de zero (+3,2 e +1,4).
   - **O nível se repete: cerca de 70%.** Duas rodadas independentes dão 70,4% e 68,6%, e a média, 71,4%. As formas de combinar dão de 67,9% a 71,4% entre rodadas e formas, e a que não ajusta nenhum peso, 70,0%, 69,3% e 70,7%.
   - **Os erros do LLM são sistemáticos, não ruído de leitura:** as duas rodadas têm correlação de 0,98 na densidade por prato e a mesma fonte principal em 94% dos pratos. A média das duas não melhora a densidade (80,7% com a massa real, nas duas leituras).
   - A densidade do LLM v3, isolada, é melhor que a do v2 (+9,3 com a massa real, IC +4,6 a +13,9), e a do modelo treinado é melhor que a da seção 8 (+8,6, IC +2,9 a +14,3).
3. **O que os dados não sustentam.**
   - **Que a meta de 70% foi atingida.** No número principal o intervalo vai de 66,4% a 76,4%, e a réplica sozinha deu 68,6%. O que os dados dizem é "cerca de 70%, entre 66% e 76%", não "70% ou mais". Pela regra escrita antes de rodar (9.7 d), a segunda rodada **não confirma nem enfraquece**: a leitura fica a mesma, com menos incerteza sobre o efeito da rodada.
   - **Que o ganho vale no `n5k_v1`.** Nos 50 pratos, as três leituras dão 62%, abaixo dos 66% do híbrido sem v3; com 50 pratos isso não refuta nem confirma.
   - **Que os pratos ricos em proteína estão resolvidos.** Na média das duas rodadas, acima de 15 g: 57,7% (IC 50,0 a 64,9). Acima de 25 g: 50,0%. Nas faixas de 25–40 g e > 40 g: 45% e 55% (41% e 54% na rodada 1; 41% e 50% na rodada 2). Os 71% do total vêm de 93% e 91% nos pratos com até 15 g e 73% nos de 15 a 25 g.
   - **Que a instrução v3 funciona pelo motivo para o qual foi escrita.** No `calib`, a fonte principal certa (78% na rodada 1 e 80% na rodada 2, contra 77% da v2) e a parte da massa das fontes (correlação 0,88 e 0,87, contra 0,87) não mudaram. A densidade melhorou, mas o mecanismo não foi isolado (9.7 b).
   - **Qualquer coisa fora do Nutrition5k.** São 280 pratos de um refeitório, com câmera a altura fixa, e a lista de nomes da instrução é a tabela do próprio Nutrition5k. Nenhum prato brasileiro foi estimado com a v3. O problema de escala de 8.7 e 8.8 continua.
4. **O que este pipeline exige.** Um LLM de visão em nuvem para a densidade, mais três passagens de ViT-B e uma busca nos embeddings do treino para a massa e a densidade do modelo. Não roda no aparelho. O pipeline sem nuvem é o só imagem: 64,6%, ou 61,1% com o modelo de um backbone exportado em 8.9.
5. **Pratos ricos em proteína, que eram o alvo: melhora parcial.** De 15 a 25 g, sim (híbrido: 62% na seção 8 → 70% em 9.6 → 73% com v3, média das rodadas). Acima de 25 g, nada que se distinga do ruído: 38% → 45% → 45% e 46% → 45% → 55%.
6. **Por que os pratos grandes continuam fracos: agora é a massa.** Com a massa real, a densidade v3 (média das rodadas) daria 77% e 68% nas duas faixas acima de 25 g; com a massa do modelo, dá 46% e 54%. O modelo de massa subestima esses pratos em 8% a 12%, e mesmo com a densidade real ficaria em 62% e 75%. Esticar as previsões não resolve (testado com ajuste cruzado em 9.4).
7. **O que falta, em ordem de ganho esperado.**
   - **Corrigir a massa dos pratos grandes.** É o que limita acima de 25 g, e esta seção não tocou no modelo de massa. O peso por faixa, que ajudou a densidade, não foi testado na massa.
   - **Mais rodadas do LLM não são o caminho.** A segunda rodada foi feita (9.7 d): o nível se repete e a média não melhora a densidade, porque os erros são os mesmos nas duas. O que resta no lado do LLM é mudar o que ele vê ou o que se pede (resolução, recortes, outra instrução), não repetir.
   - **Entender de onde vem o ganho da v3** (lista fechada, segunda hipótese ou mais componentes), antes de depender dela.
   - **Ajuste fino em GPU com a perda de frações por fonte na resolução maior.** A cabeça que funcionou aqui foi treinada sobre features congeladas. Não cabe em CPU.
   - **Dados brasileiros de cima, com celular:** continua valendo o que 8.10 diz.
8. **Para o produto.** A recomendação de 8.10 não muda: o seletor de porção continua necessário, sobretudo nos pratos com mais de 25 g, em que nenhum pipeline passa de 55%. O que muda é o ponto de partida do seletor: com a v3 e o modelo de massa, mais de 90% dos pratos com até 15 g e 73% dos de 15 a 25 g já saem dentro da tolerância no Nutrition5k. A saída "fração da massa por fonte de proteína" é uma pergunta que a usuária consegue responder ("frango é mais ou menos metade do prato?").
9. **Registro de método.** (a) Antes de qualquer leitura do teste, a regra 3 foi restringida à densidade escolhida (a primeira versão deixava concorrer a massa × cada uma das densidades, e a escolha trocava de vencedor a cada configuração nova), o TIPS foi extraído e a grade dos vizinhos foi ampliada; `dens_avalia.py --so-cv` e um ensaio no `n5k_v1` foram usados nessa fase. (b) O script completo rodou quatro vezes no teste: a primeira é a avaliação; a segunda e a terceira só acrescentaram relatos (a decomposição por faixa de 9.5 e o intervalo da densidade isolada); a quarta regravou o resumo depois de um ensaio do caminho do LLM v3 feito só no `calib`, com as estimativas v2 convertidas para o formato v3. As escolhas e as tabelas são as mesmas nas quatro (a saída da quarta é idêntica à da terceira). (c) Os números de teste das 150 densidades não escolhidas são relato, não seleção. (d) LLM v3: as estimativas chegaram depois de a instrução, o script de conversão, as formas de híbrido e a regra de escolha estarem fixados. `dens_avalia.py` rodou uma vez, sem mudança, e escolheu a forma pelo `calib`. Depois acrescentei ao script um bloco só de relato ("4d": densidade v3 com a massa real, intervalos contra o híbrido de 9.6, tabela por faixa e, no `calib`, a fonte principal e a parte das fontes) e rodei mais uma vez; a saída até o bloco novo é idêntica à da execução anterior. No total, o teste foi lido em seis execuções, todas com as mesmas escolhas. (e) Um estimador da v3 (lote 03, rodada 1) ampliou recortes de 4 das suas próprias imagens; nenhum outro arquivo foi lido. (f) Segunda rodada da v3: a regra de leitura (número principal = média das duas rodadas; rodada 2 sozinha = réplica; como julgar) foi escrita em 9.7 (d) antes de qualquer execução no teste. As mudanças de código foram só de mecânica (filtro de rodada, gravação das estimativas por prato, `dens_rodadas.py`, desempate determinístico da fonte entre rodadas). `dens_avalia.py` rodou mais três vezes no teste: rodada 1 (saída idêntica à já relatada), rodada 2 e média; antes, só no `calib` (`--so-cv`), para a rodada 2 e para a média. `dens_rodadas.py` lê as estimativas gravadas e o gabarito para comparar as rodadas. No total, nove execuções de `dens_avalia.py` leram o teste, e nenhuma escolha dependeu dele. `estimativas_v3_pratos.csv` e `estimativas_v3_comp_casados.csv` foram regerados com as duas rodadas (densidade por prato = média geométrica).

## 10. Treino em GPU (Kaggle) (11/11/2026)

**Pergunta:** os experimentos que a seção 8 deixou sem rodar por falta de CPU (ajuste fino completo, 8 orientações na resolução maior, backbones maiores e do tipo DINOv2, destilação de um professor RGB-D) melhoram o modelo de massa e o resultado final? A seção 9 terminou com o erro de massa como o limite dos pratos com mais de 25 g.
**Resposta curta: melhoram a massa e o pipeline só de imagem; o híbrido com o LLM v3 sobe pouco; a meta de 70% continua não demonstrada.** Números do `teste` (280 pratos), com a regra de 10.2.

| Medida no `teste` | Seção 9 (escolhido) | Seção 10 (escolhido) | Diferença (IC 95%, bootstrap pareado) |
|---|---|---|---|
| Massa: MAPE / massa × densidade real | 16,9% / 81,4% | **15,3% / 87,5%** | +6,1 (+2,5 a +9,6) |
| Viés mediano da massa, 25–40 g / > 40 g | −12% / −8% | **−6% / −2%** | |
| Só imagem, sem LLM (conjunto de servidor) | 64,6% | **69,6%** (IC 64,3 a 75,0) | **+5,0 (+0,4 a +9,6)** |
| **Híbrido com o LLM v3, média das rodadas (número principal)** | 71,4% | **72,9%** (IC 67,5 a 78,2) | +1,4 (−2,1 a +5,4): não distinguível |
| Idem, rodada 1 / rodada 2 sozinhas | 70,4% / 68,6% | 72,9% (IC 67,5 a 78,2) / 74,3% (IC 68,9 a 79,3) | +2,5 (−1,1 a +6,1) / +5,7 (+1,8 a +9,6) |
| Forma de híbrido que o `calib` escolhe entre todas (usa só o LLM v2) | 67,1% (híbrido com v2) | 68,2% (IC 62,9 a 73,6) | +1,1 (−3,9 a +5,7) |
| Pratos com mais de 25 g: só imagem / híbrido v3 | 42,0% / 50,0% | **53,6% / 56,2%** | |
| Modelo único do tamanho de um ViT-B (para o aparelho) | 61,1% (seção 8.9) | 62,9% (IC 57,5 a 68,6) | +1,8 (−3,9 a +7,5): não distinguível |
| **Meta** | **70%** | limite inferior do IC: 67,5% | **não demonstrada** |


### 10.1 O que rodou

`eval/treino/kaggle/` (pacote descrito em `LEIA-ME.md`): 15 configurações declaradas em `treino_gpu.py`, todas completas em 2 × Tesla T4, 5,7 h de relógio. Cada uma entrega previsões fora-do-fold no treino (os mesmos 1.684 pratos e os mesmos 5 folds por cadeia) e previsões nos 450 pratos de avaliação, com as mesmas 22 saídas em log do banco da seção 8. O Kaggle recebeu só ids e alvos do treino; os 450 ids de avaliação foram sem gabarito. Os itens brasileiros não foram ao Kaggle.

**Conferência de paridade (só CV):** `gpu ridge siglip2_b16_224 o8` dá RMSE de log-massa 0,2550 e critério 56,6 / 54,1 / 55,3% na CV; a mesma configuração na CPU dava 0,2551 e 56,6 / 54,2 / 54,9%. O ambiente da GPU reproduz o da CPU.

### 10.2 Regra de leitura, escrita antes de ler qualquer número de teste dos modelos novos

Até este ponto só foram lidos números de **CV no treino** dos modelos de GPU (o `manifesto.json` do Kaggle não contém gabarito de avaliação). A leitura do teste fica fixada assim:

1. **Massa:** a regra 1 da seção 8, sem mudança: menor RMSE de log-massa na CV, entre **todas** as configurações completas e conjuntos declarados, antigos e novos. Os conjuntos novos são só os 13 de `conjuntos_gpu.json`, escritos antes de qualquer resultado de GPU.
2. **Densidade:** a regra 1 da seção 9 (maior critério de CV com a massa real × densidade prevista). As configurações de GPU têm cabeça de densidade (as mesmas 22 saídas) e entram como candidatas, com os conjuntos de GPU. **Não há cabeça de frações por fonte nem recuperação sobre os backbones novos:** o Kaggle entregou previsões, não features. As 22 médias de 9.3 ficam como estão; a seleção gulosa é o mesmo procedimento sobre a lista maior. Desencolhimento: mesma regra (mais de 0,5 ponto com ajuste cruzado).
3. **Só imagem:** a regra 3 da seção 9: maior critério de CV entre (a) todas as configurações e rotas do banco, antigas e novas, (b) massa escolhida × densidade escolhida e (c) a média geométrica de (b) com a rota direta. A "rota direta" de (c) passa a ser a da configuração que a regra 2 da seção 8 escolhe no banco ampliado (na seção 9 era a da configuração escolhida na seção 8).
4. **Híbrido:** as mesmas formas de 9.6 e 9.7, com a massa e a densidade escolhidas acima; pesos e forma no `calib` com validação deixa-fora; empate para a forma com menos pesos. **Número principal: o híbrido com a média das duas rodadas do LLM v3.** Também saem as rodadas 1 e 2 sozinhas e o melhor híbrido só com o LLM v2.
5. **Comparações:** bootstrap pareado contra os pipelines escolhidos até a seção 9: só imagem 64,6%, híbrido com v2 67,1%, híbrido com v3 (média das rodadas) 71,4%. As previsões por prato desses três foram gravadas **antes** de os modelos de GPU entrarem no banco, e o script confere que elas dão esses números.
6. **"70% demonstrado"** quer dizer: limite inferior do IC de 95% do número principal em 70% ou mais. Abaixo disso, a meta continua não demonstrada, qualquer que seja a estimativa pontual.
7. **Modelo único para o aparelho:** entre as configurações de um backbone só, a de maior critério de CV (rotas direta, massa × densidade ou média geométrica). Relato separado para os backbones do tamanho do ViT-B (o que 8.9 mostrou caber) e para qualquer tamanho. O teste só relata.
8. **Nada novo:** nenhuma forma de híbrido nova, nenhum conjunto além dos já declarados, nenhuma mudança na instrução do LLM. Toda configuração avaliada aparece nas tabelas.
9. As execuções a mais no teste são contadas em 10.7.

**Esclarecimento acrescentado depois da escolha no `calib` e antes de qualquer leitura do teste pelo pipeline novo.** Com os modelos de GPU, a escolha no `calib` (sem teste) saiu assim: massa = média dos cinco backbones congelados de alta resolução; densidade = seleção gulosa de conjunto; só imagem = massa × essa densidade; e, entre **todas** as formas de híbrido, o `calib` prefere uma forma que usa só o LLM v2 (mistura log-linear de 5 pesos, 73,3% deixa-fora) às formas com o v3 (a melhor, 70,8%). A regra 4 dizia duas coisas que até a seção 9 coincidiam: "forma escolhida no `calib`" e "número principal = híbrido com a média das duas rodadas do v3". Como agora divergem, fica fixado, antes de ler o teste:

- **Número principal (letra da regra 4):** a melhor forma **com o v3** (média das rodadas) pelo `calib`; empate para a de menos pesos.
- **Ao lado, sempre:** a forma que o `calib` escolhe entre todas (com v2 ou v3), que é a regra 4 da seção 9 aplicada sem restrição.
- **"70% demonstrado" só vale se as duas tiverem o limite inferior do IC em 70% ou mais.** Se só uma tiver, não está demonstrado.


### 10.3 Ingestão e modelo de massa

`ingere_kaggle.py` conferiu listas, folds e ordem dos pratos, copiou as 15 configurações para o banco, somou os 13 conjuntos declarados e rodou `zen_avalia.py` sem mudança. A CV recalculada aqui é igual à registrada no Kaggle nas 15 configurações. O ensaio (`--ensaio`) e a execução para valer deram a mesma saída.

**Todas as configurações e conjuntos de GPU** (ordem do pior para o melhor RMSE de log-massa na CV; a coluna de CV decide, a de teste só relata):

| Configuração | CV: RMSE log-massa | CV: critério (direto / m×d / geo) | Teste (direto / m×d / geo) | Teste: MAPE massa | Teste: massa × dens. LLM v2 |
|---|---|---|---|---|---|
| gpu destila-rgbd siglip2_b16_224 r1 | 0,2708 | 61,0 / 62,0 / 61,7% | 61,1 / 60,0 / 60,0% | 18,8% | 60,7% |
| gpu ft siglip2_b16_224 r3 | 0,2664 | 61,2 / 61,5 / 61,8% | 58,2 / 58,2 / 56,8% | 18,6% | 58,9% |
| média gpu: ft 224 r1 + destila-rgbd 224 r1 | 0,2637 | 61,6 / 61,9 / 62,1% | 61,4 / 57,9 / 60,0% | 17,9% | 61,1% |
| gpu ft siglip2_b16_224 r1 | 0,2608 | 60,9 / 60,6 / 60,8% | 60,7 / 58,2 / 60,0% | 17,2% | 63,9% |
| gpu ridge siglip2_b16_224 o8 | 0,2550 | 56,6 / 54,1 / 55,3% | 55,0 / 58,6 / 56,4% | 18,6% | 62,9% |
| gpu ft dinov2_b14_336x448 r1 | 0,2539 | 61,0 / 61,9 / 62,0% | 64,3 / 65,7 / 65,4% | 18,5% | 61,4% |
| média gpu: ft 224 r1 + r2 + r3 | 0,2539 | 62,3 / 63,1 / 63,1% | 60,7 / 60,4 / 61,4% | 17,2% | 62,5% |
| gpu ft siglip2_b16_384x512 r1 | 0,2498 | 64,7 / 63,6 / 64,6% | 66,4 / 63,6 / 64,3% | 18,5% | 61,8% |
| média gpu: ft 224 r1 + ft 384x512 r1 | 0,2490 | 64,6 / 63,3 / 63,9% | 64,6 / 63,6 / 62,9% | 17,2% | 62,1% |
| gpu ft siglip2_b16_224 r2 **[único ViT-B]** | 0,2474 | 65,0 / 64,7 / 65,7% | 63,2 / 61,1 / 62,9% | 16,9% | 63,9% |
| gpu ft siglip2_b16_224 r1 sem-escala | 0,2467 | 61,4 / 62,5 / 62,2% | 63,2 / 60,0 / 63,2% | 17,8% | 66,8% |
| gpu ft siglip2_b16_384x512 r2 | 0,2464 | 65,2 / 64,2 / 64,8% | 62,9 / 60,4 / 61,4% | 17,3% | 59,6% |
| média gpu: ft 384x512 r1 + r2 | 0,2463 | 65,3 / 64,6 / 64,6% | 63,2 / 61,1 / 61,8% | 17,6% | 59,6% |
| gpu ft dinov2_l14_224x308 r1 | 0,2452 | 62,8 / 62,6 / 63,0% | 63,2 / 63,9 / 63,6% | 18,5% | 58,9% |
| gpu ridge siglip2_b16_384x512 o8 | 0,2426 | 61,5 / 59,8 / 61,1% | 60,4 / 59,6 / 60,4% | 17,4% | 66,4% |
| média gpu: ft 224 r1 + ft 384x512 r1 + ft dinov2_b14 r1 | 0,2418 | 63,9 / 64,3 / 63,8% | 65,0 / 63,6 / 64,3% | 16,8% | 62,5% |
| média gpu: ft 384x512 r1 + ft dinov2_b14 r1 | 0,2409 | 64,7 / 65,3 / 65,2% | 65,4 / 63,9 / 66,1% | 17,2% | 62,5% |
| média gpu: ft 224 r1 + ft 384x512 r1 + ft dinov2_b14 r1 + ft dinov2_l14 r1 | 0,2367 | 64,3 / 66,0 / 64,5% | 64,6 / 63,9 / 63,6% | 16,7% | 61,1% |
| gpu ridge dinov2_b14_336x448 o8 | 0,2349 | 61,2 / 60,1 / 61,2% | 58,9 / 58,9 / 59,6% | 18,1% | 65,0% |
| média gpu+cpu: só imagem da seção 8 + ft 224 r1 + ft 384x512 r1 | 0,2336 | 64,4 / 64,1 / 64,3% | 66,1 / 63,2 / 63,9% | 16,2% | 65,0% |
| média gpu+cpu: conjunto de massa da seção 8 + ft 384x512 r1 | 0,2305 | 63,9 / 63,1 / 63,0% | 63,9 / 63,6 / 64,6% | 16,6% | 63,6% |
| gpu ridge dinov2_l14_336x448 o8 | 0,2283 | 62,9 / 61,9 / 62,6% | 60,7 / 61,8 / 62,5% | 18,0% | 64,6% |
| média gpu+cpu: conjunto de massa da seção 8 + ft 384x512 r1 + ft dinov2_b14 r1 | 0,2271 | 63,9 / 64,4 / 64,2% | 66,1 / 65,4 / 65,7% | 16,3% | 63,2% |
| gpu ridge siglip2_l16_384x512 o8 | 0,2265 | 65,1 / 64,4 / 65,2% | 61,8 / 62,1 / 61,4% | 17,3% | 65,4% |
| média gpu: ridge 384x512 o8 + ridge dinov2_b14 o8 | 0,2220 | 63,1 / 62,7 / 63,2% | 61,8 / 61,8 / 61,4% | 16,5% | 67,9% |
| gpu ridge siglip2_so400m16_384x512 o8 **[único]** | 0,2190 | 65,4 / 64,9 / 65,9% | 61,1 / 62,9 / 62,1% | 15,9% | 67,9% |
| média gpu: ft 384x512 r1 + ft dinov2_b14 r1 + ridge so400m o8 + ridge dinov2_l14 o8 **[rota direta]** | 0,2120 | 66,9 / 67,0 / 67,6% | 67,9 / 67,9 / 68,2% | 15,4% | 65,7% |
| média gpu: ridge congelados de alta resolução (5 backbones) **[massa]** | 0,2051 | 66,7 / 66,7 / 67,1% | 65,7 / 68,6 / 66,1% | 15,3% | 67,9% |

- **Massa escolhida pela regra 1: a média dos cinco backbones congelados de alta resolução** (SigLIP 2 B/16, L/16 e So400m/16 a 384×512; DINOv2 B/14 e L/14 a 336×448; ridge, 8 orientações). RMSE de log-massa na CV: **0,2051**, contra 0,2329 do conjunto da seção 8.
- **O ajuste fino completo não foi o que mais ajudou a massa.** Os backbones maiores congelados com ridge (0,219 a 0,228) ganham de todos os ajustes finos (0,245 a 0,266). O ajuste fino ajuda o critério de proteína no mesmo backbone B/16 (a 224 px, 65% contra 57% do ridge; a 384×512, 65% contra 62%), não a massa.
- **Destilação do professor RGB-D: não ajudou** (0,2708; pior que o ajuste fino simples com a mesma receita, 0,2608). É a terceira vez que a profundidade do Nutrition5k não rende (seções 2-B e 8).
- **Escala aleatória no treino custa na CV** (0,2608 com, 0,2467 sem, na receita r1), como se espera num conjunto em que a câmera está sempre à mesma altura. O teste desta seção também é à altura fixa; a robustez de 8.7 não foi medida de novo.

**Massa no teste, por faixa de proteína** (MAPE e, entre parênteses, viés mediano):

| | 0–5 g | 5–15 g | 15–25 g | 25–40 g | > 40 g | Todos (MAPE) | Massa × densidade real |
|---|---|---|---|---|---|---|---|
| Seção 8, CV no treino | 19,7% (0%) | 16,7% (0%) | 14,3% (−2%) | 13,0% (−4%) | 14,7% (−10%) | 16,7% | |
| Seção 10, CV no treino | 17,3% (+1%) | 14,4% (+1%) | 12,5% (0%) | 10,6% (−1%) | 13,0% (−5%) | 14,5% | |
| Seção 8, teste | 19,1% (+3%) | 17,3% (+5%) | 15,0% (−7%) | 18,5% (**−12%**) | 14,4% (**−8%**) | 16,9% | 81,4% |
| **Seção 10, teste** | 18,7% (+5%) | 15,9% (+7%) | 13,3% (−6%) | 15,3% (**−6%**) | 13,5% (**−2%**) | **15,3%** | **87,5%** |

**O viés acima de 25 g caiu pela metade ou mais:** de −12% para −6% entre 25 e 40 g, e de −8% para −2% acima de 40 g. Com a densidade real, a massa nova daria 87,5% no teste, contra 81,4% da antiga (+6,1, IC +2,5 a +9,6); por faixa, 100 / 96 / 89 / 71 / 80%, contra 100 / 93 / 77 / 62 / 75%. A faixa de 25 a 40 g continua a mais fraca (71%).

### 10.4 Densidade

**Escolhida pela regra 2: a seleção gulosa de conjunto**, com critério de densidade de 76,2% (65,0% acima de 15 g) na CV com ajuste cruzado, contra 75,4% (63,7%) da média escolhida na seção 9. Os membros escolhidos com todos os folds são cinco: a cabeça de densidade da média de quatro ajustes finos de GPU (SigLIP 2 B/16 a 224 e a 384×512, DINOv2 B/14 e L/14), a do ridge sobre o So400m, e três cabeças de frações da seção 9 (ingredientes com erro quadrático, fontes com erro ponderado, fontes sobre 224 + 384×512). O TIPS deixou de entrar. Desencolhimento não adotado (76,2% → 74,9%).

- **No teste, a densidade não mudou:** 75,4% com a massa real (65,5% acima de 15 g), contra 76,1% (66,7%) da escolhida na seção 9. As cabeças de densidade dos modelos de GPU chegam a 73,8% na CV e 70,7% no teste (a melhor, uma média de quatro ajustes finos); as cabeças de frações da seção 9, em média, davam 75,4% e 76,1%.
- **O ganho desta seção vem da massa, não da densidade.**

<details>
<summary>Tabela completa: as 179 densidades (151 da seção 9 e 28 cabeças de densidade de GPU), em ordem do critério de CV</summary>

Mesma legenda de 9.4; `s8:` marca a cabeça de densidade de uma configuração do banco de massa e proteína (seções 8 e 10). "Massa escolhida" é a de 10.3; a diferença é contra o só imagem da seção 9 (64,6%).

| Densidade | CV: critério de densidade (todos / > 15 g) | Teste: massa real × dens. (todos / > 15 g) | Teste: massa escolhida × dens. | Δ vs só imagem da seção 9 (IC 95%) | Teste: geo com a rota direta |
|---|---|---|---|---|---|
| ridge contribuição por ingrediente siglip2_b16_512 | 33,3 / 28,9% | 35,4 / 32,7% | 32,1% | −32,5 (−40,7 a −24,6) | 56,8% |
| ridge contribuição por ingrediente tips_b14 | 34,4 / 31,0% | 35,7 / 31,0% | 35,4% | −29,3 (−37,5 a −21,1) | 58,6% |
| ridge contribuição por ingrediente siglip2_b16 | 35,0 / 31,8% | 36,8 / 32,7% | 30,7% | −33,9 (−41,8 a −26,4) | 58,2% |
| soft-linear fonte siglip2_b16_512 [frações] | 37,1 / 27,4% | 46,8 / 41,7% | 40,7% | −23,9 (−31,4 a −16,8) | 61,4% |
| soft-linear ingrediente siglip2_b16_512 [frações] | 56,6 / 36,2% | 61,8 / 48,2% | 56,1% | −8,6 (−14,3 a −2,9) | 65,4% |
| s8: ridge siglip2_b16 o4 | 58,1 / 38,8% | 57,5 / 43,5% | 55,0% | −9,6 (−15,7 a −3,6) | 63,6% |
| s8: ridge siglip2_b16 o1 | 59,7 / 41,5% | 55,0 / 39,3% | 51,4% | −13,2 (−19,6 a −7,1) | 60,7% |
| s8: ridge siglip2_b16 o2 | 59,8 / 41,6% | 55,7 / 40,5% | 52,9% | −11,8 (−17,9 a −5,7) | 62,5% |
| soft-mlp ingrediente siglip2_b16_512 [frações] | 59,8 / 41,0% | 66,1 / 52,4% | 62,5% | −2,1 (−7,5 a +3,2) | 66,4% |
| s8: ridge siglip2_b16 o8 + escala aleatória no treino | 60,7 / 42,9% | 58,2 / 45,2% | 58,2% | −6,4 (−12,5 a +0,0) | 64,6% |
| s8: gpu ridge siglip2_b16_224 o8 | 60,8 / 42,9% | 58,6 / 44,0% | 59,6% | −5,0 (−10,7 a +1,1) | 65,7% |
| s8: ridge siglip2_b16 o8 | 61,0 / 43,4% | 59,3 / 44,0% | 59,6% | −5,0 (−11,1 a +1,1) | 65,4% |
| ridge logdens siglip2_b16 | 61,0 / 43,4% | 59,3 / 44,0% | 59,6% | −5,0 (−11,1 a +1,1) | 65,4% |
| s8: ridge siglip2_b16 o1-tta8 | 61,2 / 43,7% | 57,5 / 42,3% | 52,9% | −11,8 (−17,9 a −6,1) | 61,8% |
| soft-linear fonte siglip2_b16_512 [log-dens] | 61,3 / 42,5% | 62,5 / 48,8% | 58,6% | −6,1 (−12,1 a −0,4) | 65,4% |
| soft-linear ingrediente siglip2_b16_512 [log-dens] | 61,3 / 42,5% | 62,5 / 48,8% | 58,6% | −6,1 (−12,1 a −0,4) | 65,4% |
| ridge logdens peso-faixa0.5 siglip2_b16 | 61,4 / 44,2% | 62,9 / 50,6% | 61,1% | −3,6 (−9,3 a +2,5) | 65,0% |
| ridge logdens peso-faixa1.0 siglip2_b16 | 61,7 / 44,4% | 64,3 / 52,4% | 61,1% | −3,6 (−9,3 a +2,5) | 66,1% |
| s8: ridge siglip2_b16 o8 + quadros laterais no treino | 61,8 / 44,2% | 58,9 / 44,6% | 55,7% | −8,9 (−14,7 a −3,2) | 62,9% |
| s8: ridge siglip2_b16 o8+gem | 62,0 / 44,7% | 63,2 / 53,0% | 59,3% | −5,4 (−11,4 a +0,7) | 66,1% |
| soft-mlp fonte siglip2_b16 [frações] | 62,0 / 44,8% | 67,5 / 57,7% | 61,4% | −3,2 (−8,2 a +1,8) | 69,3% |
| ridge fração por fonte siglip2_b16_512 | 62,3 / 55,3% | 62,1 / 57,1% | 57,5% | −7,1 (−13,6 a −0,7) | 67,5% |
| s8: ridge siglip2_b16_512 o1 | 62,4 / 44,6% | 60,7 / 45,2% | 58,9% | −5,7 (−11,4 a +0,0) | 63,2% |
| ridge fração por fonte siglip2_b16 | 62,6 / 56,3% | 64,6 / 58,9% | 57,5% | −7,1 (−13,2 a −1,1) | 65,7% |
| soft-mlp ingrediente +perda-dens siglip2_b16_512 [frações] | 62,7 / 45,0% | 70,7 / 58,9% | 65,0% | +0,4 (−5,0 a +5,7) | 68,9% |
| s8: média: ajuste fino + ridge siglip2 o8 | 62,9 / 45,5% | 65,4 / 52,4% | 60,4% | −4,3 (−10,0 a +1,4) | 66,4% |
| s8: ridge siglip2_b16_512 o2+gem+grade | 62,9 / 45,5% | 61,1 / 46,4% | 57,1% | −7,5 (−13,2 a −1,8) | 65,0% |
| soft-mlp ingrediente só-frações +perda-dens siglip2_b16_512 [frações] | 63,1 / 46,0% | 71,4 / 59,5% | 66,8% | +2,1 (−3,6 a +7,9) | 69,6% |
| soft-mlp ingrediente +perda-dens peso-faixa siglip2_b16 [frações] | 63,1 / 46,1% | 73,9 / 64,3% | 66,4% | +1,8 (−3,6 a +6,8) | 70,4% |
| s8: ridge siglip2_b16 o8+gem+grade | 63,2 / 46,2% | 60,7 / 48,8% | 58,6% | −6,1 (−12,1 a +0,0) | 65,0% |
| ridge fração por fonte tips_b14 | 63,2 / 55,4% | 64,6 / 56,5% | 57,1% | −7,5 (−13,6 a −1,4) | 67,1% |
| soft-mlp ingrediente +perda-dens peso-faixa siglip2_b16_512 [frações] | 63,2 / 46,3% | 72,1 / 60,7% | 64,6% | +0,0 (−5,7 a +5,4) | 69,6% |
| soft-mlp ingrediente +perda-dens peso-faixa0.5 siglip2_b16_512 [frações] | 63,3 / 46,4% | 71,4 / 59,5% | 64,6% | +0,0 (−5,4 a +5,7) | 68,2% |
| s8: média: siglip2 o8 + siglip2-512 | 63,3 / 45,8% | 63,9 / 50,6% | 58,9% | −5,7 (−11,4 a +0,0) | 66,8% |
| s8: ft cv | 63,8 / 46,6% | 63,6 / 48,8% | 61,8% | −2,9 (−8,9 a +3,2) | 66,4% |
| s8: ridge siglip2_b16_512 o2 | 64,1 / 46,7% | 63,6 / 50,0% | 61,1% | −3,6 (−9,3 a +2,1) | 65,7% |
| ridge logdens siglip2_b16_512 | 64,1 / 46,7% | 63,6 / 50,0% | 61,1% | −3,6 (−9,3 a +2,1) | 65,7% |
| soft-mlp fonte siglip2_b16_512 [frações] | 64,2 / 48,3% | 70,4 / 60,1% | 62,5% | −2,1 (−7,1 a +2,9) | 67,5% |
| soft-mlp ingrediente +perda-dens oculto1024 drop0.5 siglip2_b16_512 [frações] | 64,5 / 47,9% | 73,2 / 63,7% | 65,7% | +1,1 (−4,6 a +6,4) | 70,0% |
| s8: média: ajuste fino + ridge o8 com laterais | 64,7 / 48,3% | 64,3 / 51,8% | 62,1% | −2,5 (−7,9 a +3,2) | 64,6% |
| s8: gpu ridge dinov2_b14_336x448 o8 | 64,8 / 47,1% | 65,7 / 54,2% | 62,9% | −1,8 (−7,9 a +3,9) | 66,4% |
| ridge logdens peso-faixa0.5 siglip2_b16_512 | 64,8 / 47,8% | 65,4 / 53,0% | 61,4% | −3,2 (−8,9 a +2,5) | 66,1% |
| soft-mlp fonte peso-faixa siglip2_b16_512 [frações] | 64,8 / 48,6% | 72,1 / 61,3% | 62,9% | −1,8 (−6,8 a +3,2) | 68,2% |
| s8: média: ajuste fino + ridge o8 + siglip2-512 (sem TIPS) | 65,0 / 48,2% | 66,4 / 54,2% | 60,7% | −3,9 (−9,6 a +1,8) | 67,1% |
| s8: gpu ridge siglip2_b16_384x512 o8 | 65,1 / 48,2% | 62,5 / 48,8% | 57,1% | −7,5 (−13,6 a −1,8) | 64,3% |
| ridge logdens peso-faixa1.0 siglip2_b16_512 | 65,2 / 48,9% | 66,8 / 55,4% | 61,4% | −3,2 (−8,9 a +2,5) | 66,4% |
| s8: média: ajuste fino + ridge o8 com laterais + siglip2-512 (sem TIPS) | 65,4 / 49,2% | 65,7 / 53,0% | 62,5% | −2,1 (−7,5 a +3,2) | 65,7% |
| s8: média: siglip2 o8 + mlp | 65,7 / 49,3% | 65,0 / 52,4% | 64,6% | +0,0 (−5,4 a +5,4) | 66,4% |
| s8: média: ajuste fino + siglip2-512 | 65,7 / 49,2% | 67,5 / 55,4% | 62,5% | −2,1 (−8,2 a +3,9) | 67,1% |
| soft-mlp fonte +perda-dens siglip2_b16 [frações] | 65,8 / 49,4% | 68,9 / 56,0% | 65,4% | +0,7 (−3,9 a +5,4) | 68,2% |
| ridge logdens tips_b14 | 65,9 / 50,2% | 65,0 / 51,2% | 62,1% | −2,5 (−7,9 a +2,9) | 65,7% |
| s8: gpu ridge dinov2_l14_336x448 o8 | 66,4 / 50,1% | 68,6 / 54,2% | 63,6% | −1,1 (−6,8 a +4,6) | 66,1% |
| s8: mlp+aux-profundidade siglip2_b16 o8 | 66,8 / 51,1% | 67,5 / 54,2% | 64,6% | +0,0 (−5,4 a +5,4) | 66,4% |
| ridge logdens peso-faixa0.5 tips_b14 | 66,9 / 51,7% | 65,4 / 53,6% | 61,8% | −2,9 (−8,6 a +2,9) | 66,4% |
| soft-mlp fonte-l1 siglip2_b16_512 [frações] | 66,9 / 53,8% | 74,3 / 64,9% | 63,6% | −1,1 (−5,7 a +3,6) | 69,6% |
| knn fração por fonte siglip2_b16 | 67,0 / 56,8% | 67,5 / 60,7% | 59,6% | −5,0 (−10,4 a +0,4) | 68,9% |
| s8: média: ajuste fino + mlp + ridge o8 com laterais | 67,0 / 51,3% | 65,7 / 53,6% | 62,9% | −1,8 (−6,8 a +3,6) | 65,4% |
| soft-mlp fonte-l2 oculto1024 drop0.5 siglip2_b16_512 [log-dens] | 67,1 / 51,2% | 72,1 / 61,3% | 63,6% | −1,1 (−6,4 a +3,9) | 69,6% |
| soft-mlp fonte-l2 siglip2_b16_512+esp [log-dens] | 67,1 / 51,5% | 70,7 / 58,9% | 66,8% | +2,1 (−2,9 a +7,5) | 69,6% |
| s8: média: ajuste fino + mlp + ridge o8 com laterais + siglip2-512 (sem TIPS) | 67,1 / 51,2% | 67,1 / 55,4% | 63,6% | −1,1 (−6,4 a +4,3) | 66,1% |
| soft-mlp ingrediente-l2 siglip2_b16_512 [log-dens] | 67,1 / 51,2% | 70,4 / 58,9% | 64,6% | +0,0 (−5,4 a +5,4) | 67,1% |
| ridge logdens peso-faixa1.0 tips_b14 | 67,1 / 51,9% | 66,4 / 54,8% | 62,9% | −1,8 (−7,9 a +3,9) | 66,4% |
| soft-mlp fonte-l2 siglip2_b16_512 [log-dens] | 67,3 / 51,7% | 70,4 / 57,7% | 64,6% | +0,0 (−5,0 a +5,0) | 68,6% |
| knn fração por fonte siglip2_b16_512 | 67,4 / 56,4% | 69,3 / 60,7% | 58,9% | −5,7 (−11,1 a −0,4) | 66,4% |
| s8: gpu ft siglip2_b16_224 r3 | 67,4 / 52,2% | 67,1 / 52,4% | 67,9% | +3,2 (−2,5 a +8,9) | 69,3% |
| soft-mlp ingrediente +perda-dens siglip2_b16_512 [log-dens] | 67,5 / 51,6% | 70,0 / 57,7% | 63,9% | −0,7 (−5,7 a +4,3) | 67,1% |
| knn densidade siglip2_b16_512 | 67,5 / 53,9% | 67,9 / 56,0% | 60,7% | −3,9 (−8,9 a +1,1) | 67,9% |
| s8: média: ajuste fino + mlp | 67,5 / 51,8% | 68,2 / 56,0% | 63,9% | −0,7 (−6,1 a +4,6) | 67,5% |
| soft-mlp fonte siglip2_b16 [log-dens] | 67,6 / 52,0% | 69,6 / 56,5% | 65,0% | +0,4 (−4,6 a +5,4) | 67,5% |
| soft-mlp fonte-l2 tips_b14 [log-dens] | 67,7 / 51,9% | 68,2 / 54,8% | 64,6% | +0,0 (−5,7 a +5,4) | 66,4% |
| soft-mlp ingrediente +perda-dens oculto1024 drop0.5 siglip2_b16_512 [log-dens] | 67,7 / 52,1% | 67,9 / 54,8% | 60,0% | −4,6 (−9,6 a +0,4) | 67,5% |
| soft-mlp fonte +perda-dens siglip2_b16 [log-dens] | 67,9 / 52,3% | 68,2 / 54,2% | 64,6% | +0,0 (−5,0 a +5,0) | 67,9% |
| soft-mlp fonte-l2d siglip2_b16_512 [log-dens] | 67,9 / 52,3% | 70,4 / 58,9% | 64,3% | −0,4 (−5,4 a +4,6) | 68,9% |
| s8: média gpu: ridge 384x512 o8 + ridge dinov2_b14 o8 | 68,0 / 51,7% | 68,9 / 58,3% | 62,1% | −2,5 (−7,9 a +2,9) | 67,9% |
| soft-mlp ingrediente siglip2_b16_512 [log-dens] | 68,0 / 52,4% | 70,7 / 58,9% | 65,4% | +0,7 (−4,6 a +5,7) | 67,1% |
| knn densidade siglip2_b16 | 68,0 / 54,9% | 72,1 / 60,7% | 63,9% | −0,7 (−5,7 a +4,3) | 67,9% |
| soft-mlp fonte-l2 semente2 siglip2_b16_512 [log-dens] | 68,1 / 52,7% | 71,8 / 60,1% | 64,6% | +0,0 (−5,4 a +5,4) | 68,9% |
| s8: mlp siglip2_b16 o8 | 68,2 / 53,4% | 68,9 / 56,0% | 62,9% | −1,8 (−6,8 a +3,2) | 68,6% |
| s8: mlp wd0.1 siglip2_b16 o8 | 68,3 / 53,5% | 69,3 / 56,5% | 62,9% | −1,8 (−6,8 a +3,2) | 68,6% |
| soft-mlp fonte-l2 tips_b14 [frações] | 68,3 / 53,8% | 70,4 / 61,9% | 64,3% | −0,4 (−5,4 a +4,3) | 68,6% |
| soft-mlp fonte +perda-dens siglip2_b16_512 [log-dens] | 68,4 / 52,8% | 72,9 / 62,5% | 63,6% | −1,1 (−6,1 a +3,9) | 67,9% |
| soft-mlp fonte +perda-dens peso-faixa siglip2_b16_512 [frações] | 68,5 / 54,0% | 71,8 / 61,9% | 64,3% | −0,4 (−5,0 a +3,9) | 69,6% |
| soft-mlp fonte-l2 ambos [log-dens] | 68,5 / 53,9% | 69,3 / 58,3% | 66,1% | +1,4 (−3,6 a +6,8) | 69,3% |
| s8: mlp oculto1024 drop0.5 siglip2_b16 o8 | 68,7 / 53,7% | 70,0 / 57,1% | 64,3% | −0,4 (−5,4 a +5,0) | 66,8% |
| s8: gpu ft dinov2_l14_224x308 r1 | 68,8 / 53,4% | 70,7 / 58,9% | 67,1% | +2,5 (−3,2 a +8,2) | 67,9% |
| s8: gpu ridge siglip2_l16_384x512 o8 | 68,8 / 53,2% | 67,1 / 53,6% | 62,1% | −2,5 (−8,9 a +3,6) | 66,1% |
| soft-mlp fonte-l2 semente1 siglip2_b16_512 [log-dens] | 68,8 / 54,2% | 73,9 / 62,5% | 64,3% | −0,4 (−5,7 a +4,6) | 70,4% |
| soft-mlp fonte-l2 oculto1024 drop0.5 siglip2_b16_512 [frações] | 68,8 / 58,4% | 69,6 / 63,1% | 61,1% | −3,6 (−9,6 a +2,5) | 69,6% |
| soft-mlp fonte siglip2_b16_512 [log-dens] | 68,8 / 53,2% | 73,6 / 62,5% | 65,4% | +0,7 (−3,9 a +5,4) | 68,6% |
| soft-mlp ingrediente +perda-dens peso-faixa0.5 siglip2_b16_512 [log-dens] | 68,8 / 54,1% | 71,1 / 58,9% | 66,4% | +1,8 (−2,9 a +6,8) | 66,4% |
| s8: gpu ft siglip2_b16_224 r1 sem-escala | 68,8 / 53,9% | 66,1 / 51,8% | 64,6% | +0,0 (−6,1 a +5,7) | 68,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa tips_b14 [log-dens] | 68,8 / 54,8% | 71,1 / 61,3% | 65,4% | +0,7 (−5,0 a +6,1) | 67,9% |
| soft-mlp fonte-l1 siglip2_b16_512 [log-dens] | 68,9 / 54,6% | 70,4 / 59,5% | 67,1% | +2,5 (−2,5 a +7,5) | 67,9% |
| s8: gpu ft dinov2_b14_336x448 r1 | 69,0 / 54,3% | 67,1 / 54,2% | 65,4% | +0,7 (−5,4 a +6,8) | 68,6% |
| soft-mlp fonte +perda-dens siglip2_b16_512 [frações] | 69,0 / 54,9% | 71,8 / 61,9% | 64,6% | +0,0 (−4,6 a +4,6) | 68,9% |
| soft-mlp fonte-l2 siglip2_b16_512+esp [frações] | 69,0 / 55,7% | 72,5 / 66,7% | 62,9% | −1,8 (−6,8 a +2,9) | 68,9% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente1 tips_b14 [log-dens] | 69,1 / 55,1% | 70,7 / 58,9% | 65,0% | +0,4 (−5,0 a +5,7) | 66,4% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente2 siglip2_b16_512 [log-dens] | 69,1 / 54,7% | 72,5 / 63,7% | 65,0% | +0,4 (−5,0 a +5,4) | 66,8% |
| soft-mlp fonte-l2 +perda-dens peso-faixa tips_b14+siglip2_b16_512 [log-dens] | 69,2 / 55,1% | 72,5 / 61,9% | 66,4% | +1,8 (−3,6 a +7,1) | 68,2% |
| s8: média gpu+cpu: conjunto de massa da seção 8 + ft 384x512 r1 | 69,2 / 53,9% | 70,4 / 60,7% | 63,9% | −0,7 (−6,1 a +4,6) | 67,5% |
| soft-mlp fina-l2 siglip2_b16_512 [log-dens] | 69,2 / 54,7% | 72,1 / 60,1% | 63,9% | −0,7 (−6,1 a +4,6) | 67,9% |
| soft-mlp ingrediente +perda-dens peso-faixa siglip2_b16 [log-dens] | 69,2 / 55,9% | 70,0 / 58,3% | 65,0% | +0,4 (−4,6 a +5,7) | 68,9% |
| s8: gpu ft siglip2_b16_224 r1 | 69,3 / 54,6% | 68,9 / 56,5% | 64,6% | +0,0 (−5,4 a +5,7) | 68,2% |
| soft-mlp fonte-l2 tips_b14+siglip2_b16_512 [log-dens] | 69,4 / 54,3% | 68,6 / 54,8% | 64,3% | −0,4 (−5,7 a +5,0) | 66,8% |
| soft-mlp fonte +perda-dens peso-faixa siglip2_b16_512 [log-dens] | 69,5 / 55,2% | 72,1 / 61,3% | 63,2% | −1,4 (−6,4 a +3,2) | 68,2% |
| knn fração por fonte tips_b14+siglip2_b16_512 | 69,5 / 59,2% | 71,1 / 63,7% | 62,9% | −1,8 (−6,8 a +3,6) | 68,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa siglip2_b16 [log-dens] | 69,5 / 56,2% | 67,9 / 54,8% | 61,8% | −2,9 (−7,5 a +2,1) | 68,9% |
| soft-mlp fina-l2 +perda-dens peso-faixa siglip2_b16_512 [log-dens] | 69,6 / 55,8% | 70,4 / 58,9% | 62,5% | −2,1 (−7,1 a +2,9) | 67,5% |
| s8: média gpu+cpu: só imagem da seção 8 + ft 224 r1 + ft 384x512 r1 | 69,7 / 54,5% | 71,8 / 61,9% | 67,1% | +2,5 (−2,9 a +7,9) | 68,2% |
| soft-mlp fina-l2 siglip2_b16_512 [frações] | 69,7 / 57,5% | 72,1 / 63,1% | 63,9% | −0,7 (−5,7 a +4,3) | 68,9% |
| soft-mlp ingrediente-l2 siglip2_b16_512 [frações] | 69,8 / 58,3% | 72,1 / 66,1% | 65,4% | +0,7 (−5,0 a +6,4) | 70,0% |
| soft-mlp ingrediente +perda-dens peso-faixa siglip2_b16_512 [log-dens] | 69,8 / 55,9% | 70,0 / 60,1% | 66,1% | +1,4 (−3,6 a +6,4) | 67,5% |
| soft-mlp fonte peso-faixa siglip2_b16_512 [log-dens] | 70,0 / 55,7% | 72,9 / 63,1% | 66,1% | +1,4 (−3,6 a +6,8) | 68,2% |
| soft-mlp fonte-l2 ambos [frações] | 70,1 / 58,1% | 74,3 / 66,1% | 64,3% | −0,4 (−5,0 a +4,3) | 68,2% |
| soft-mlp fonte-l2 +perda-dens peso-faixa siglip2_b16 [frações] | 70,2 / 57,5% | 70,7 / 61,9% | 66,4% | +1,8 (−2,5 a +6,1) | 66,8% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente1 siglip2_b16_512 [log-dens] | 70,2 / 56,8% | 72,5 / 61,3% | 66,1% | +1,4 (−3,9 a +6,8) | 69,3% |
| soft-mlp fonte-l2 tips_b14+siglip2_b16_512 [frações] | 70,3 / 57,4% | 71,4 / 61,3% | 63,6% | −1,1 (−6,1 a +3,9) | 68,2% |
| soft-mlp fonte-l2 +perda-dens peso-faixa tips_b14 [frações] | 70,3 / 56,9% | 71,4 / 63,7% | 66,1% | +1,4 (−3,6 a +6,4) | 67,5% |
| s8: gpu destila-rgbd siglip2_b16_224 r1 | 70,4 / 56,7% | 67,5 / 53,0% | 63,2% | −1,4 (−7,1 a +4,3) | 66,8% |
| soft-mlp fonte-l2 peso-faixa siglip2_b16_512 [log-dens] | 70,4 / 56,5% | 72,1 / 63,1% | 66,1% | +1,4 (−3,6 a +6,4) | 69,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa siglip2_b16_512 [log-dens] | 70,4 / 56,8% | 70,7 / 61,3% | 65,4% | +0,7 (−4,6 a +5,7) | 69,6% |
| soft-mlp fonte-l2 +perda-dens2 peso-faixa siglip2_b16_512 [log-dens] | 70,5 / 57,1% | 70,7 / 61,3% | 63,6% | −1,1 (−6,4 a +3,9) | 69,6% |
| soft-mlp fonte-l2 só-frações siglip2_b16_512 [frações] | 70,5 / 58,7% | 73,9 / 64,3% | 63,2% | −1,4 (−6,1 a +3,2) | 68,9% |
| soft-mlp fonte-l2 semente1 siglip2_b16_512 [frações] | 70,6 / 59,5% | 73,9 / 64,3% | 64,6% | +0,0 (−4,6 a +5,0) | 68,9% |
| soft-mlp fonte-l2 +perda-dens peso-faixa 100 épocas siglip2_b16_512 [log-dens] | 70,8 / 57,2% | 70,7 / 60,1% | 64,6% | +0,0 (−5,0 a +5,4) | 68,2% |
| s8: média gpu: ft 224 r1 + destila-rgbd 224 r1 | 70,8 / 57,2% | 70,0 / 57,7% | 62,1% | −2,5 (−8,2 a +3,2) | 68,9% |
| soft-mlp fonte-l2 semente2 siglip2_b16_512 [frações] | 70,9 / 59,8% | 74,6 / 65,5% | 63,9% | −0,7 (−5,4 a +3,9) | 68,9% |
| soft-mlp fonte-l2 siglip2_b16_512 [frações] | 70,9 / 59,9% | 73,2 / 63,7% | 63,9% | −0,7 (−5,4 a +4,3) | 67,9% |
| s8: média gpu+cpu: conjunto de massa da seção 8 + ft 384x512 r1 + ft dinov2_b14 r1 | 70,9 / 56,5% | 70,0 / 60,1% | 64,6% | +0,0 (−5,0 a +5,0) | 68,9% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente1 tips_b14 [frações] | 71,0 / 58,0% | 71,1 / 63,1% | 65,4% | +0,7 (−3,9 a +5,4) | 68,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa ambos [log-dens] | 71,0 / 57,8% | 71,4 / 61,9% | 67,9% | +3,2 (−1,8 a +8,2) | 69,3% |
| s8: gpu ridge siglip2_so400m16_384x512 o8 | 71,0 / 56,2% | 68,9 / 54,8% | 65,4% | +0,7 (−4,6 a +6,1) | 69,6% |
| knn fração por fonte tips_b14 | 71,1 / 61,2% | 71,8 / 63,1% | 65,0% | +0,4 (−4,6 a +5,4) | 68,9% |
| s8: gpu ft siglip2_b16_384x512 r1 | 71,3 / 57,2% | 69,6 / 60,1% | 66,1% | +1,4 (−4,3 a +7,1) | 67,5% |
| soft-mlp fina-l2 +perda-dens peso-faixa siglip2_b16_512 [frações] | 71,3 / 58,9% | 73,6 / 63,1% | 65,7% | +1,1 (−3,9 a +5,7) | 69,6% |
| média: frações-l2 512, 3 sementes | 71,4 / 60,8% | 74,6 / 64,9% | 64,3% | −0,4 (−5,0 a +4,3) | 68,9% |
| soft-mlp fonte-l2 +perda-dens peso-faixa 100 épocas siglip2_b16_512 [frações] | 71,4 / 59,8% | 75,0 / 64,9% | 65,0% | +0,4 (−4,3 a +5,0) | 67,5% |
| soft-mlp fonte-l2 peso-faixa siglip2_b16_512 [frações] | 71,7 / 61,0% | 72,9 / 64,3% | 63,2% | −1,4 (−6,4 a +3,6) | 68,6% |
| s8: média gpu: ft 224 r1 + ft 384x512 r1 | 71,8 / 58,1% | 70,0 / 59,5% | 65,4% | +0,7 (−5,0 a +6,8) | 68,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente1 siglip2_b16_512 [frações] | 71,8 / 60,4% | 73,9 / 63,1% | 63,6% | −1,1 (−5,7 a +3,6) | 69,3% |
| s8: média gpu: ft 224 r1 + r2 + r3 | 71,9 / 58,2% | 71,4 / 59,5% | 64,6% | +0,0 (−5,4 a +5,7) | 70,0% |
| s8: média gpu: ridge congelados de alta resolução (5 backbones) | 71,9 / 57,2% | 71,8 / 60,1% | 68,6% | +3,9 (−1,4 a +9,3) | 68,6% |
| soft-mlp fonte-l2 +perda-dens2 peso-faixa siglip2_b16_512 [frações] | 72,0 / 60,9% | 74,3 / 63,7% | 64,3% | −0,4 (−5,4 a +4,6) | 68,2% |
| soft-mlp fonte-l2d siglip2_b16_512 [frações] | 72,0 / 61,4% | 73,2 / 64,9% | 63,6% | −1,1 (−6,1 a +3,9) | 68,2% |
| média: frações-l2 512 + ajuste fino (s8) | 72,1 / 58,9% | 71,8 / 61,3% | 64,3% | −0,4 (−5,4 a +4,3) | 68,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa ambos [frações] | 72,2 / 60,3% | 74,6 / 66,7% | 66,8% | +2,1 (−2,5 a +6,4) | 68,6% |
| soft-mlp fonte-l2 +perda-dens peso-faixa siglip2_b16_512 [frações] | 72,2 / 61,4% | 73,6 / 63,7% | 64,6% | +0,0 (−4,6 a +4,7) | 68,2% |
| soft-mlp fonte-l2 +perda-dens peso-faixa semente2 siglip2_b16_512 [frações] | 72,2 / 61,3% | 74,6 / 66,1% | 63,9% | −0,7 (−5,4 a +3,9) | 69,3% |
| soft-mlp fonte-l2 +perda-dens peso-faixa tips_b14+siglip2_b16_512 [frações] | 72,2 / 59,7% | 73,6 / 64,9% | 64,6% | +0,0 (−4,6 a +4,3) | 68,9% |
| knn densidade tips_b14+siglip2_b16_512 | 72,3 / 60,7% | 72,5 / 61,3% | 66,4% | +1,8 (−2,9 a +6,4) | 68,6% |
| knn densidade tips_b14 | 72,4 / 60,7% | 72,9 / 63,7% | 67,5% | +2,9 (−1,8 a +7,5) | 71,1% |
| s8: gpu ft siglip2_b16_384x512 r2 | 72,6 / 58,9% | 73,9 / 64,3% | 67,9% | +3,2 (−2,1 a +8,6) | 70,0% |
| média: frações-l2 512, 3 sementes (+perda-dens peso-faixa) | 72,6 / 61,9% | 74,3 / 64,9% | 64,6% | +0,0 (−4,6 a +4,6) | 68,9% |
| média: frações-l2 512 + knn 512 | 72,6 / 61,0% | 74,3 / 64,3% | 65,4% | +0,7 (−3,2 a +5,0) | 69,3% |
| média: frações-l2 512 + classes finas-l2 512 | 72,8 / 61,5% | 77,1 / 69,0% | 65,4% | +0,7 (−3,6 a +5,4) | 68,9% |
| média: frações-l2 512 + s8 escolhido (ajuste fino + 512) | 72,9 / 60,4% | 72,1 / 61,3% | 66,4% | +1,8 (−2,5 a +6,4) | 67,5% |
| média: frações-l2 512 + 224 + ambos | 72,9 / 61,7% | 76,8 / 68,5% | 66,8% | +2,1 (−2,1 a +6,4) | 68,9% |
| s8: média gpu: ft 224 r1 + ft 384x512 r1 + ft dinov2_b14 r1 | 72,9 / 59,5% | 71,4 / 60,1% | 66,4% | +1,8 (−3,9 a +7,5) | 69,3% |
| s8: média gpu: ft 384x512 r1 + ft dinov2_b14 r1 | 72,9 / 59,6% | 69,6 / 58,3% | 67,1% | +2,5 (−2,9 a +8,2) | 69,3% |
| s8: gpu ft siglip2_b16_224 r2 | 73,0 / 59,5% | 70,0 / 58,3% | 66,1% | +1,4 (−3,6 a +6,4) | 67,9% |
| média: frações-l2 512 + frações-l2 224 | 73,0 / 61,9% | 74,3 / 66,1% | 66,8% | +2,1 (−1,8 a +6,4) | 69,3% |
| média: frações-l2 512 + sua cabeça log-dens | 73,0 / 61,0% | 75,0 / 64,3% | 66,4% | +1,8 (−2,9 a +6,4) | 68,6% |
| média: frações-l2 512 + ambos | 73,1 / 62,0% | 75,0 / 66,7% | 66,4% | +1,8 (−2,5 a +6,1) | 68,9% |
| s8: média gpu: ft 384x512 r1 + ft dinov2_b14 r1 + ridge so400m o8 + ridge dinov2_l14 o8 | 73,1 / 59,5% | 73,2 / 60,7% | 70,4% | +5,7 (+0,7 a +11,1) | 69,3% |
| média: frações-l2 512 + knn 512 + mlp 224 (s8) | 73,3 / 61,1% | 73,2 / 61,9% | 66,4% | +1,8 (−2,1 a +6,1) | 67,9% |
| s8: média gpu: ft 384x512 r1 + r2 | 73,4 / 60,0% | 72,9 / 63,7% | 67,1% | +2,5 (−2,9 a +8,2) | 67,9% |
| média: frações-l2 512, 3 sementes + knn 512 + mlp 224 (s8) | 73,5 / 62,1% | 74,6 / 63,7% | 66,8% | +2,1 (−1,8 a +6,4) | 69,6% |
| média: frações-l2 512 + frações-l2 TIPS + knn 512 | 73,8 / 62,0% | 73,9 / 63,7% | 64,6% | +0,0 (−4,3 a +4,3) | 69,3% |
| s8: média gpu: ft 224 r1 + ft 384x512 r1 + ft dinov2_b14 r1 + ft dinov2_l14 r1 | 73,8 / 61,1% | 70,7 / 59,5% | 67,1% | +2,5 (−3,2 a +7,9) | 68,2% |
| média: frações-l2 512 + classes finas-l2 512 + knn 512 + mlp 224 (s8) | 73,9 / 61,8% | 74,3 / 63,7% | 67,5% | +2,9 (−1,1 a +7,1) | 69,6% |
| média: frações-l2 (512, 224, ambos) + knn 512 + ridge fração por fonte 512 | 74,0 / 63,1% | 74,6 / 66,1% | 65,7% | +1,1 (−2,9 a +5,4) | 69,3% |
| média: frações-l2 512 + classes finas-l2 512 + knn TIPS + mlp 224 (s8) | 74,2 / 62,0% | 73,9 / 63,1% | 68,2% | +3,6 (−0,4 a +7,5) | 69,3% |
| média: frações-l2 512 + knn TIPS + mlp 224 (s8) | 74,4 / 62,7% | 74,6 / 64,3% | 69,3% | +4,6 (+1,1 a +8,6) | 69,6% |
| média: frações-l2 512 + frações-l2 TIPS | 74,5 / 63,4% | 74,6 / 67,9% | 64,6% | +0,0 (−4,3 a +4,3) | 68,6% |
| média: frações-l2 512 + frações-l2 TIPS + knn 512 + knn TIPS + mlp 224 (s8) | 74,7 / 63,2% | 75,4 / 65,5% | 67,9% | +3,2 (−0,4 a +7,1) | 68,6% |
| média: frações-l2 512 + knn TIPS | 74,7 / 63,5% | 76,8 / 68,5% | 68,6% | +3,9 (+0,0 a +7,9) | 70,7% |
| média: frações-l2 512 + frações-l2 TIPS + knn TIPS | 74,8 / 63,3% | 77,5 / 70,8% | 67,9% | +3,2 (−0,7 a +7,5) | 69,6% |
| média: frações-l2 (512, TIPS, TIPS+512) + knn TIPS + mlp 224 (s8) | 75,4 / 63,7% | 76,1 / 66,7% | 68,6% | +3,9 (+0,0 a +8,2) | 69,3% |
| seleção gulosa de conjunto **[densidade]** | 76,2 / 65,0% | 75,4 / 65,5% | 69,6% | +5,0 (+0,4 a +9,6) | 71,4% |

</details>

### 10.5 Só imagem e modelo único

**Escolhido pela regra 3 (CV 70,5%, contra 66,3% na seção 9 e 62,2% na seção 8): massa de 10.3 × densidade de 10.4. Teste: 69,6%** (IC 64,3 a 75,0), MAE 5,6 g, viés −2,0 g. No `calib`: 71,7% (seção 9: 66,7%). No `n5k_v1`: 64% (seção 9: 58%).

- **Contra os pipelines anteriores:** +5,0 sobre o só imagem da seção 9 (IC +0,4 a +9,6), +8,6 sobre o da seção 8 (IC +3,6 a +13,6), +14,6 sobre o LLM v2 sozinho (IC +7,5 a +21,8), +2,5 sobre o híbrido com v2 da seção 9 (IC −2,5 a +7,5).
- **Os pratos com mais de 25 g melhoraram pela primeira vez:** 42,0% → 53,6% (25–40 g: 38% → 48%; > 40 g: 46% → 59%). Acima de 15 g: 47,6% → 57,7%.
- A regra da seção 8 aplicada ao banco ampliado (`zen_avalia.py`, sem as cabeças de frações) escolhe a média de quatro modelos de GPU pela rota geométrica: CV 67,6%, teste 68,2% (IC 62,5 a 73,6).

**Modelo único (um backbone), para o aparelho** (regra 7; a tabela completa, com os 32 modelos únicos, está abaixo):

| Modelo único | CV: critério | Teste | Teste: MAPE massa | Δ vs modelo exportado em 8.9 (61,1%) | n5k v1 |
|---|---|---|---|---|---|
| **Tamanho ViT-B, escolhido pela CV:** SigLIP 2 B/16 a 224 px, ajuste fino completo `r2` (lr 6e-5, 25 épocas, orientações e escala aleatórias), rota geo | 65,7% | **62,9%** (IC 57,5 a 68,6) | 16,9% | +1,8 (−3,9 a +7,5) | 64% |
| Qualquer tamanho, escolhido pela CV: SigLIP 2 So400m/16 a 384×512 congelado + ridge, rota geo | 65,9% | 62,1% (IC 56,4 a 67,9) | 15,9% | +1,1 (−5,0 a +7,1) | 62% |
| Não escolhido, maior no teste: SigLIP 2 B/16 a 384×512, ajuste fino `r1`, rota direta | 64,7% | 66,4% (IC 60,7 a 71,8) | 18,5% | +5,4 (0,0 a +11,1) | 64% |
| Referência: modelo exportado em 8.9 (três cabeças sobre o SigLIP 2 B/16 a 224 px) | 61,5% | 61,1% (IC 55,4 a 66,8) | 17,5% | | 58% |

- **O melhor modelo único para o aparelho, pela CV, é o `gpu ft siglip2_b16_224 r2`.** No teste ele dá 62,9%, que não se distingue dos 61,1% do modelo de 8.9. O que ele tem a mais é ser um ViT-B inteiro com uma cabeça só (o de 8.9 tem dois topos de 3 blocos), ter erro de massa um pouco menor (16,9% contra 17,5%) e ter sido treinado com escala aleatória, que 8.7 apontou como condição para ir ao celular. Os pesos estão em `pesos__gpu_ft_siglip2_b16_224_r2.pt` (fora do repositório). A conversão para Core ML e a latência no iPhone **não foram feitas**, e a robustez de escala desse modelo **não foi medida**.
- **Um único modelo continua 7 pontos abaixo do conjunto** (62,9% contra 69,6%). O ganho desta seção está em juntar backbones grandes, o que não cabe no aparelho.

<details>
<summary>Tabela completa: os 32 modelos de um backbone (CPU e GPU), em ordem do critério de CV</summary>

| Modelo único | Tamanho | Rota (CV) | CV: critério | Teste | MAE / viés | Teste: MAPE massa | Δ vs modelo exportado em 8.9 (61,1%) | > 15 g | n5k v1 |
|---|---|---|---|---|---|---|---|---|---|
| ridge siglip2_b16 o1 | ViT-B | direta | 54,6% | 50,0% (IC 44,3 a 55,7) | 8,7 g / −4,1 g | 21,1% | −11,1 (−16,4 a −5,7) | 30,4% | 44% |
| ridge siglip2_b16 o8 + escala aleatória no treino | ViT-B | direta | 55,9% | 56,4% (IC 50,4 a 62,1) | 7,8 g / −2,4 g | 18,5% | −4,6 (−9,3 a +0,4) | 39,3% | 52% |
| ridge siglip2_b16 o1-tta8 | ViT-B | direta | 56,0% | 55,0% (IC 49,3 a 60,7) | 8,2 g / −3,9 g | 20,0% | −6,1 (−11,1 a −0,7) | 38,7% | 54% |
| ridge siglip2_b16 o4 | ViT-B | direta | 56,6% | 51,8% (IC 46,1 a 57,5) | 8,3 g / −3,6 g | 20,1% | −9,3 (−14,3 a −4,3) | 34,5% | 48% |
| gpu ridge siglip2_b16_224 o8 | ViT-B | direta | 56,6% | 55,0% (IC 49,3 a 60,7) | 7,7 g / −2,9 g | 18,6% | −6,1 (−10,7 a −1,4) | 38,1% | 56% |
| ridge siglip2_b16 o8 | ViT-B | direta | 56,6% | 55,4% (IC 49,6 a 61,1) | 7,7 g / −2,8 g | 18,6% | −5,7 (−10,4 a −1,1) | 38,7% | 56% |
| ridge siglip2_b16 o8+gem | ViT-B | direta | 56,9% | 56,4% (IC 50,7 a 62,1) | 7,8 g / −3,1 g | 19,1% | −4,6 (−9,3 a +0,0) | 40,5% | 50% |
| ridge siglip2_b16 o2 | ViT-B | direta | 57,0% | 51,4% (IC 45,7 a 57,1) | 8,4 g / −3,5 g | 20,4% | −9,6 (−14,6 a −4,6) | 34,5% | 52% |
| ridge siglip2_b16 o8 + quadros laterais no treino | ViT-B | direta | 57,3% | 51,8% (IC 45,7 a 57,9) | 8,3 g / −3,6 g | 19,0% | −9,3 (−13,6 a −5,0) | 32,7% | 48% |
| ridge siglip2_b16 o8+gem+grade | ViT-B | direta | 57,3% | 57,1% (IC 51,1 a 62,9) | 7,7 g / −2,6 g | 18,9% | −3,9 (−8,2 a +0,4) | 39,9% | 52% |
| ridge siglip2_b16_512 o1 | ViT-B | direta | 59,4% | 58,2% (IC 52,5 a 63,9) | 7,4 g / −3,8 g | 19,3% | −2,9 (−7,9 a +2,1) | 42,9% | 54% |
| ft cv | ViT-B | direta | 59,4% | 57,5% (IC 51,8 a 63,2) | 7,6 g / −2,0 g | 18,9% | −3,6 (−7,9 a +0,7) | 41,7% | 60% |
| mlp+aux-profundidade siglip2_b16 o8 | ViT-B | direta | 59,5% | 60,0% (IC 54,3 a 65,7) | 7,7 g / −5,8 g | 20,0% | −1,1 (−6,1 a +3,6) | 39,3% | 50% |
| ridge siglip2_b16_512 o2 | ViT-B | direta | 59,7% | 61,1% (IC 55,4 a 66,8) | 7,0 g / −3,4 g | 18,3% | +0,0 (−5,4 a +5,4) | 47,6% | 60% |
| ridge siglip2_b16_512 o2+gem+grade | ViT-B | direta | 60,2% | 58,9% (IC 53,2 a 64,6) | 7,3 g / −3,5 g | 19,0% | −2,1 (−7,5 a +3,2) | 44,0% | 56% |
| mlp oculto1024 drop0.5 siglip2_b16 o8 | ViT-B | geo | 60,5% | 59,6% (IC 53,9 a 65,0) | 7,6 g / −5,6 g | 19,9% | −1,4 (−6,1 a +3,2) | 40,5% | 50% |
| mlp siglip2_b16 o8 | ViT-B | geo | 60,6% | 62,1% (IC 56,4 a 67,9) | 7,3 g / −5,2 g | 19,4% | +1,1 (−3,6 a +5,7) | 44,0% | 54% |
| mlp wd0.1 siglip2_b16 o8 | ViT-B | geo | 60,6% | 62,5% (IC 56,8 a 68,2) | 7,3 g / −5,2 g | 19,4% | +1,4 (−3,2 a +6,1) | 44,6% | 52% |
| gpu ft siglip2_b16_224 r1 | ViT-B | direta | 60,9% | 60,7% (IC 55,0 a 66,4) | 6,8 g / −2,7 g | 17,2% | −0,4 (−5,4 a +4,7) | 47,0% | 56% |
| gpu ridge dinov2_b14_336x448 o8 | ViT-B | direta | 61,2% | 58,9% (IC 53,2 a 64,3) | 7,3 g / −3,0 g | 18,1% | −2,1 (−8,2 a +3,9) | 44,6% | 54% |
| gpu ridge siglip2_b16_384x512 o8 | ViT-B | direta | 61,5% | 60,4% (IC 54,6 a 66,1) | 7,0 g / −3,1 g | 17,4% | −0,7 (−6,1 a +4,6) | 45,2% | 62% |
| gpu ft siglip2_b16_224 r3 | ViT-B | geo | 61,8% | 56,8% (IC 50,7 a 62,9) | 7,4 g / −2,2 g | 18,6% | −4,3 (−10,0 a +1,4) | 39,9% | 54% |
| gpu ft dinov2_b14_336x448 r1 | ViT-B | geo | 62,0% | 65,4% (IC 59,6 a 70,7) | 7,0 g / −2,4 g | 18,5% | +4,3 (−1,4 a +10,0) | 50,0% | 56% |
| gpu destila-rgbd siglip2_b16_224 r1 | ViT-B | m×d | 62,0% | 60,0% (IC 53,9 a 65,7) | 7,3 g / −2,3 g | 18,8% | −1,1 (−7,1 a +4,6) | 43,5% | 56% |
| gpu ft siglip2_b16_224 r1 sem-escala | ViT-B | m×d | 62,5% | 60,0% (IC 53,9 a 65,4) | 7,5 g / −0,5 g | 17,8% | −1,1 (−7,1 a +5,0) | 43,5% | 56% |
| gpu ridge dinov2_l14_336x448 o8 | maior | direta | 62,9% | 60,7% (IC 55,0 a 66,4) | 7,1 g / −2,9 g | 18,0% | −0,4 (−6,4 a +5,4) | 44,0% | 66% |
| gpu ft dinov2_l14_224x308 r1 | maior | geo | 63,0% | 63,6% (IC 57,9 a 69,3) | 6,9 g / −3,2 g | 18,5% | +2,5 (−3,6 a +8,6) | 48,8% | 64% |
| gpu ft siglip2_b16_384x512 r1 | ViT-B | direta | 64,7% | 66,4% (IC 60,7 a 71,8) | 6,2 g / −1,8 g | 18,5% | +5,4 (+0,0 a +11,1) | 54,2% | 64% |
| gpu ridge siglip2_l16_384x512 o8 | maior | geo | 65,2% | 61,4% (IC 55,7 a 67,1) | 7,2 g / −2,3 g | 17,3% | +0,4 (−6,1 a +6,4) | 45,2% | 66% |
| gpu ft siglip2_b16_384x512 r2 | ViT-B | direta | 65,2% | 62,9% (IC 57,1 a 68,2) | 6,3 g / −3,2 g | 17,3% | +1,8 (−3,9 a +7,2) | 47,6% | 58% |
| gpu ft siglip2_b16_224 r2 **[único ViT-B]** | ViT-B | geo | 65,7% | 62,9% (IC 57,5 a 68,6) | 6,8 g / −2,7 g | 16,9% | +1,8 (−3,9 a +7,5) | 46,4% | 64% |
| gpu ridge siglip2_so400m16_384x512 o8 **[único]** | maior | geo | 65,9% | 62,1% (IC 56,4 a 67,9) | 6,8 g / −2,2 g | 15,9% | +1,1 (−5,0 a +7,1) | 44,0% | 62% |

</details>

### 10.6 Híbridos

Massa de 10.3, densidade de 10.4, as mesmas formas de 9.6 e 9.7. Leitura principal: LLM v3 com a média das duas rodadas. As diferenças são contra os híbridos escolhidos na seção 9. As 7 primeiras linhas são as formas da seção 8 com a massa e a densidade da seção 8 (referência).

| Forma | Calib (deixa-fora) | Teste | MAE / viés | Δ vs híbrido v3 da seção 9 (71,4%) | Δ vs híbrido v2 da seção 9 (67,1%) | > 15 g | > 25 g | n5k v1 |
|---|---|---|---|---|---|---|---|---|
| Massa e dens. da seção 8 · massa modelo × dens LLM | 63,3% | 63,9% (IC 58,6 a 69,3) | 6,6 g / −2,6 g | −7,5 (−12,5 a −2,9) | −3,2 (−7,9 a +1,4) | 49,4% | 38,4% | 54% |
| Massa e dens. da seção 8 · massa √(modelo·LLM) × dens LLM | 62,5% | 61,8% (IC 56,1 a 67,2) | 7,2 g / −3,4 g | −9,6 (−15,7 a −3,6) | −5,4 (−11,4 a +0,7) | 47,6% | 34,8% | 60% |
| Massa e dens. da seção 8 · massa modelo^a·LLM^(1−a) × dens LLM | 61,7% | 63,2% (IC 57,5 a 68,6) | 6,6 g / −2,8 g | −8,2 (−13,6 a −2,9) | −3,9 (−8,9 a +1,1) | 47,0% | 34,8% | 54% |
| Massa e dens. da seção 8 · massa^a × dens (LLM^b·modelo^(1−b)) | 65,0% | 62,9% (IC 57,1 a 68,6) | 6,3 g / −3,3 g | −8,6 (−13,2 a −3,9) | −4,3 (−8,6 a −0,4) | 48,8% | 42,0% | 60% |
| Massa e dens. da seção 8 · blend log-linear (5 pesos) | 65,0% | 65,4% (IC 59,6 a 71,1) | 5,9 g / −1,2 g | −6,1 (−11,1 a −1,1) | −1,8 (−6,8 a +3,2) | 56,0% | 50,0% | 66% |
| Massa e dens. da seção 8 · blend log-linear + proteína direta e do LLM (7 pesos) | 64,2% | 65,4% (IC 59,6 a 71,1) | 5,9 g / −1,0 g | −6,1 (−11,1 a −0,7) | −1,8 (−6,8 a +3,6) | 56,0% | 50,9% | 70% |
| Massa e dens. da seção 8 · média geométrica: proteína direta do modelo e do LLM | 63,3% | 61,1% (IC 55,4 a 66,8) | 6,9 g / −3,8 g | −10,4 (−15,7 a −5,0) | −6,1 (−11,4 a −1,1) | 44,6% | 36,6% | 64% |
| v2 · massa modelo × dens LLM | 63,3% | 67,9% (IC 62,5 a 73,2) | 6,3 g / −1,8 g | −3,6 (−8,9 a +1,8) | +0,7 (−4,6 a +6,1) | 54,8% | 45,5% | 66% |
| v2 · massa √(modelo·LLM) × dens LLM | 62,5% | 63,6% (IC 57,9 a 69,3) | 7,0 g / −3,1 g | −7,9 (−14,3 a −1,4) | −3,6 (−9,6 a +2,5) | 50,0% | 36,6% | 58% |
| v2 · massa modelo^a·LLM^(1−a) × dens LLM | 64,2% | 67,9% (IC 62,1 a 73,2) | 6,4 g / −2,2 g | −3,6 (−8,9 a +1,8) | +0,7 (−4,7 a +6,1) | 55,4% | 46,4% | 68% |
| v2 · massa^a × dens (LLM^b·modelo^(1−b)) | 70,8% | 69,3% (IC 63,9 a 74,6) | 5,5 g / −2,1 g | −2,1 (−6,4 a +2,1) | +2,1 (−2,1 a +6,4) | 57,7% | 51,8% | 70% |
| v2 · blend log-linear (5 pesos) **[escolhida pelo calib entre todas]** | 73,3% | 68,2% (IC 62,9 a 73,6) | 5,4 g / −1,4 g | −3,2 (−8,2 a +1,8) | +1,1 (−3,9 a +5,7) | 59,5% | 55,4% | 68% |
| v2 · blend log-linear + proteína direta e do LLM (7 pesos) | 71,7% | 70,0% (IC 64,6 a 75,4) | 5,3 g / −1,2 g | −1,4 (−6,4 a +3,6) | +2,9 (−2,1 a +7,9) | 60,7% | 58,0% | 70% |
| v2 · média geométrica: proteína direta do modelo e do LLM | 66,7% | 64,3% (IC 58,6 a 69,6) | 6,5 g / −3,4 g | −7,1 (−12,9 a −1,8) | −2,9 (−8,2 a +2,5) | 51,2% | 42,9% | 70% |
| v2 · massa modelo × dens (LLM^b · modelo^(1−b)) | 71,7% | 69,3% (IC 63,9 a 74,6) | 5,5 g / −2,1 g | −2,1 (−6,4 a +2,1) | +2,1 (−2,1 a +6,4) | 57,7% | 51,8% | 70% |
| v2 · massa modelo × Σ fonte (fração LLM^w ⊕ modelo) × dens por fonte do LLM | 65,8% | 69,6% (IC 64,3 a 75,0) | 5,7 g / −1,9 g | −1,8 (−6,8 a +2,9) | +2,5 (−2,1 a +7,1) | 57,1% | 50,9% | 70% |
| v2 · massa^a × dens (LLM com frango pela média do treino ^b · modelo^(1−b)) | 70,8% | 69,3% (IC 63,9 a 74,6) | 5,5 g / −2,1 g | −2,1 (−6,4 a +2,1) | +2,1 (−2,1 a +6,1) | 57,7% | 51,8% | 68% |
| v2 · massa modelo × dens (LLM com frango pela média do treino ^b · modelo^(1−b)) | 71,7% | 69,3% (IC 63,9 a 74,6) | 5,5 g / −2,1 g | −2,1 (−6,4 a +2,1) | +2,1 (−2,1 a +6,1) | 57,7% | 51,8% | 68% |
| v2 · massa modelo × dens do modelo (sem LLM) | 71,7% | 69,6% (IC 64,3 a 75,0) | 5,6 g / −2,0 g | −1,8 (−6,4 a +2,5) | +2,5 (−2,5 a +7,5) | 57,7% | 53,6% | 64% |
| v2 · massa modelo × dens (LLM^b · frações do modelo com a fonte do LLM como prior^(1−b)) | 67,5% | 68,6% (IC 62,9 a 73,9) | 5,6 g / −2,3 g | −2,9 (−7,5 a +1,8) | +1,4 (−3,2 a +5,7) | 56,0% | 48,2% | 66% |
| v3 · massa modelo × dens LLM v3 | 69,2% | 73,9% (IC 68,6 a 78,9) | 5,5 g / −1,5 g | +2,5 (−2,1 a +7,1) | +6,8 (+1,4 a +11,8) | 63,1% | 58,0% | 62% |
| v3 · massa modelo × dens (v3^b · modelo^(1−b)) | 70,0% | 73,6% (IC 68,2 a 78,6) | 5,3 g / −1,9 g | +2,1 (−1,4 a +5,7) | +6,4 (+2,1 a +10,7) | 62,5% | 56,2% | 68% |
| v3 · massa^a (modelo, LLM v2) × dens (v3^b · modelo^(1−b)) | 69,2% | 73,6% (IC 68,2 a 78,6) | 5,3 g / −1,9 g | +2,1 (−1,4 a +5,7) | +6,4 (+2,1 a +10,7) | 62,5% | 56,2% | 68% |
| v3 · massa modelo × dens (v2^b1 · v3^b2 · modelo^(1−b1−b2)) | 70,0% | 71,1% (IC 65,7 a 76,1) | 5,4 g / −2,1 g | −0,4 (−4,6 a +3,9) | +3,9 (−0,4 a +7,9) | 60,1% | 54,5% | 70% |
| v3 · massa modelo × Σ fonte (fração v3^w ⊕ modelo) × dens por fonte do v3 | 69,2% | 74,3% (IC 69,3 a 79,3) | 5,3 g / −1,6 g | +2,9 (−1,1 a +6,8) | +7,1 (+2,9 a +11,4) | 63,7% | 58,9% | 60% |
| v3 · massa modelo × dens (v3 com frango pela média do treino ^b · modelo^(1−b)) **[melhor com v3: número principal]** | 70,8% | 72,9% (IC 67,5 a 78,2) | 5,4 g / −1,8 g | +1,4 (−2,1 a +5,4) | +5,7 (+1,1 a +10,4) | 61,3% | 56,2% | 70% |
| v3 · massa modelo × dens (v3^b · frações do modelo com a fonte declarada no v3 como prior^(1−b)) | 70,8% | 75,0% (IC 70,0 a 80,0) | 5,3 g / −1,9 g | +3,6 (−0,0 a +7,1) | +7,9 (+3,6 a +12,1) | 63,7% | 57,1% | 66% |
| v3 · massa modelo × frações do modelo com a fonte declarada no v3 como prior (sem densidade do LLM) | 65,8% | 64,6% (IC 58,9 a 70,4) | 6,0 g / −2,3 g | −6,8 (−11,8 a −1,8) | −2,5 (−7,5 a +2,5) | 52,4% | 47,3% | 64% |

**As três leituras do LLM v3** (a forma é escolhida no `calib` em cada uma):

| Leitura | Melhor forma com v3 no `calib` | Calib (deixa-fora) | Teste | Δ vs híbrido v3 da seção 9, mesma leitura | Δ vs híbrido v2 da seção 9 (67,1%) | > 15 g | > 25 g | n5k v1 | Massa × densidade v3, sem nenhum peso |
|---|---|---|---|---|---|---|---|---|---|
| **Média das duas rodadas (número principal)** | (v3 com frango pela média do treino)^0,4 · modelo^0,6 | 70,8% | **72,9%** (IC 67,5 a 78,2) | +1,4 (−2,1 a +5,4), contra 71,4% | +5,7 (+1,1 a +10,4) | 61,3% | 56,2% | 70% | 73,9% (IC 68,6 a 78,9) |
| Rodada 1 sozinha | v3^0,5 · modelo^0,5 | 70,8% | 72,9% (IC 67,5 a 78,2) | +2,5 (−1,1 a +6,1), contra 70,4% | +5,7 (+1,8 a +10,0) | 61,3% | 55,4% | 70% | 73,2% (IC 67,9 a 78,6) |
| Rodada 2 sozinha | v3^0,6 · frações do modelo^0,4 | 71,7% | 74,3% (IC 68,9 a 79,3) | +5,7 (+1,8 a +9,6), contra 68,6% | +7,1 (+2,5 a +11,8) | 63,1% | 56,2% | 62% | 73,2% (IC 67,9 a 78,2) |
| **Forma que o `calib` escolhe entre todas** (igual nas três leituras) | só LLM v2: mistura log-linear de 5 pesos | 73,3% | **68,2%** (IC 62,9 a 73,6) | −3,2 (−8,2 a +1,8), contra 71,4% | +1,1 (−3,9 a +5,7) | 59,5% | 55,4% | 68% | |

Por faixa de proteína real no teste (56 pratos por faixa):

| Faixa | Só imagem s9 | **Só imagem s10** | Híbrido v3 s9 (média) | **Híbrido v3 s10 (média)** | Forma do `calib` entre todas (v2) | Massa s10 × dens. v3, sem pesos | Massa s8 × dens. real | Massa s10 × dens. real | Massa real × dens. do modelo s10 | Massa real × dens. v3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–5 g | 93% | 91% | 93% | 91% | 88% | 93% | 100% | 100% | 93% | 93% |
| 5–15 g | 88% | 84% | 91% | 89% | 75% | 88% | 93% | 96% | 88% | 93% |
| 15–25 g | 59% | 66% | 73% | 71% | 68% | 73% | 77% | 89% | 71% | 73% |
| 25–40 g | 38% | **48%** | 45% | **54%** | 57% | 52% | 62% | 71% | 66% | 77% |
| > 40 g | 46% | **59%** | 55% | **59%** | 54% | 64% | 75% | 80% | 59% | 68% |
| **Todos** | 64,6% | **69,6%** | 71,4% | **72,9%** | 68,2% | 73,9% | 81,4% | 87,5% | 75,4% | 80,7% |
| **> 15 g** | 47,6% | 57,7% | 57,7% | 61,3% | 59,5% | 63,1% | 71,4% | 80,4% | 65,5% | 72,6% |
| **> 25 g** | 42,0% | 53,6% | 50,0% | 56,2% | 55,4% | 58,0% | 68,8% | 75,9% | 62,5% | 72,3% |

1. **Número principal: 72,9%** (IC 67,5 a 78,2), MAE 5,4 g, viés −1,8 g. Contra o híbrido com v3 da seção 9 (71,4%) o ganho é de +1,4 e **não se distingue de zero** (IC −2,1 a +5,4). Contra o híbrido com v2 da seção 9: +5,7 (IC +1,1 a +10,4). Contra o só imagem desta seção: +3,2 (IC +0,4 a +6,4).
2. **A forma que o `calib` prefere entre todas dá 68,2% no teste** (IC 62,9 a 73,6), abaixo do só imagem (69,6%). Ela usa só o LLM v2 e ajusta 5 pesos em 120 pratos. No `calib` ela parecia 2,5 pontos melhor que a melhor forma com v3; no teste ficou 4,6 pontos pior (IC da diferença: +0,4 a +9,3 a favor da forma com v3). **Os 120 pratos do `calib` não separam formas de híbrido com segurança:** as formas com a densidade v3 valem 69% a 71% no `calib` e 71% a 75% no teste; as formas com v2, 62% a 73% no `calib` e 64% a 70% no teste.
3. **Todas as formas que usam a densidade v3 ficam em 70% ou mais no teste, nas três leituras:** 71,1% a 75,0% com a média das rodadas, 71,4% a 74,3% com a rodada 1, 70,4% a 74,3% com a rodada 2. A forma sem nenhum peso ajustado (massa × densidade v3) dá 73,9%, 73,2% e 73,2%. Uma forma não escolhida chega a 75,0% (IC 70,0 a 80,0).
4. **O LLM ajuda menos do que ajudava.** Com a massa nova, o só imagem está a 3 pontos do híbrido com v3 (69,6% contra 72,9%); na seção 9 a distância era de 7 (64,6% contra 71,4%). No `calib`, o só imagem (71,7%) fica acima das formas com v3 (até 70,8%) e abaixo só da mistura de 5 pesos com v2 (73,3%).
5. **Acima de 25 g:** 56,2% no híbrido com v3 (seção 9: 50,0%) e 53,6% no só imagem (42,0%). A faixa de 25 a 40 g saiu de 38–45% para 48–54% e continua a mais fraca.
6. **No `n5k_v1`:** 70% no número principal (seção 9: 62%), 62% a 70% nas formas com v3. Com 50 pratos, ±14 pontos.

### 10.7 Conclusão

1. **Onde estamos, no `teste`.**
   - **Só imagem (conjunto de servidor): 69,6%** (IC 64,3 a 75,0). Seção 9: 64,6%. Ganho de +5,0 (IC +0,4 a +9,6).
   - **Híbrido com o LLM v3, média das rodadas: 72,9%** (IC 67,5 a 78,2). Seção 9: 71,4%. Ganho de +1,4 (IC −2,1 a +5,4), não distinguível.
   - Forma que o `calib` escolhe entre todas (só LLM v2): 68,2% (IC 62,9 a 73,6).
   - Modelo único do tamanho de um ViT-B: 62,9% (IC 57,5 a 68,6). Seção 8: 61,1%.
2. **A meta de 70% está demonstrada? Não.** Pela regra de 10.2, seria preciso o limite inferior do IC em 70% ou mais no número principal **e** na forma que o `calib` escolhe entre todas. Os limites são 67,5% e 62,9%. O que os dados sustentam é mais fraco e mais específico:
   - a estimativa pontual do híbrido com v3 está acima de 70% em **todas** as formas que usam a densidade v3 e nas três leituras (70,4% a 75,0%), inclusive na forma sem nenhum peso ajustado (73,2% a 73,9%, com limite inferior de 67,9% a 68,6%);
   - nenhuma forma escolhida tem o limite inferior em 70%; uma não escolhida tem (75,0%, IC 70,0 a 80,0), e escolher por ela seria escolher pelo teste;
   - com 280 pratos, um IC de 95% tem cerca de ±5,4 pontos: para o limite inferior chegar a 70% a estimativa teria de estar perto de 75%, ou o teste teria de ser maior.
3. **O que os modelos de GPU compraram, com segurança:** um modelo de massa melhor (com a densidade real, 81,4% → 87,5%, IC do ganho +2,5 a +9,6; viés acima de 25 g de −12% e −8% para −6% e −2%) e, por causa dele, um pipeline só de imagem melhor (+5,0, IC +0,4 a +9,6), com os pratos acima de 25 g saindo de 42% para 54%. **O que não compraram:** densidade melhor (75,4% contra 76,1% com a massa real), ganho distinguível no híbrido com v3, um modelo único claramente melhor para o aparelho.
4. **Limites que continuam.**
   - **Pratos de 25 a 40 g:** 48% (só imagem) e 54% (híbrido). Ali a massa nova ainda só dá 71% com a densidade real, e a densidade do modelo, 66% com a massa real.
   - **O `calib` é pequeno para escolher a forma do híbrido** (item 2 de 10.6). Com mais pratos de calibração, ou fixando de antemão a forma sem pesos, essa fonte de variação some.
   - **O teste já foi lido muitas vezes** (item 6). Nenhuma escolha o usou, mas 280 pratos lidos em 12 execuções de `dens_avalia.py` e em várias de `zen_avalia.py` deixam de ser um teste intocado: as próximas mudanças pedem um conjunto de teste novo.
   - **Tudo continua sendo Nutrition5k:** um refeitório, câmera a altura fixa, sem prato brasileiro. Os modelos de GPU não viram os itens brasileiros, então 8.8 não foi refeita. A robustez de escala (8.7) não foi medida de novo.
5. **Para o aparelho e para o servidor.**
   - **No aparelho:** o `gpu ft siglip2_b16_224 r2`, um SigLIP 2 B/16 a 224 px com ajuste fino completo e uma cabeça de 22 saídas, usado pela rota geométrica com média de 8 orientações. Teste: 62,9%. Falta converter para Core ML, medir a latência e medir a escala.
   - **No servidor, só imagem (69,6%):** dez redes. Massa: cinco backbones congelados em alta resolução (SigLIP 2 B/16, L/16 e So400m/16 a 384×512; DINOv2 B/14 e L/14 a 336×448), 8 orientações cada, com ridge. Densidade: quatro modelos com ajuste fino (SigLIP 2 B/16 a 224 e a 384×512, DINOv2 B/14 e L/14), mais três cabeças de frações sobre o SigLIP 2 B/16 congelado (224 px e 384×512) e a cabeça de densidade do So400m. Hoje só existem as previsões: nem os pesos dos ajustes finos (fora o `r2`) nem as cabeças ridge foram guardados, e servir isso pede rodar o treino de novo com `--guarda-pesos todos`.
   - **No servidor, híbrido (72,9%):** o mesmo, mais uma chamada ao LLM de visão em nuvem com a instrução v3. Uma rodada basta (9.7 d).
6. **Registro de método.** (a) A regra de leitura (10.2) foi escrita antes de qualquer número de teste dos modelos novos; até ali só a CV do Kaggle tinha sido lida. (b) As previsões por prato dos três pipelines de referência da seção 9 foram gravadas antes de os modelos de GPU entrarem no banco, sem ler gabarito; o script confere que elas dão 64,6%, 67,1% e 71,4% (e 70,4% e 68,6% nas rodadas). (c) `dens_avalia.py` passou a escolher a massa pela CV no banco inteiro e a somar os 13 conjuntos de GPU; `--sem-gpu` refaz o estado da seção 9, e foi conferido no `calib` que as escolhas saem iguais às de antes. (d) Depois de ver a escolha no `calib` e antes de ler o teste, registrei em 10.2 qual das duas leituras da regra 4 é o número principal e que a meta só conta como demonstrada com as duas. (e) Execuções que leram o teste nesta seção: duas de `zen_avalia.py` (ensaio e ingestão, saída idêntica) e três de `dens_avalia.py` (média das rodadas, rodada 1, rodada 2). Com as nove da seção 9, `dens_avalia.py` leu o teste 12 vezes. Só no `calib` ou no `n5k_v1`: `--sem-gpu --so-cv`, `--so-cv` e `--ensaio`. (f) Nenhuma forma nova, nenhum conjunto novo, nenhuma mudança na instrução.

## 11. Conjunto de teste novo (teste2) (11/11/2026)

**Por quê:** o `teste` (280 pratos) já foi lido em mais de uma dezena de execuções, e com 280 pratos o intervalo de 95% tem cerca de ±5 pontos. Para confirmar ou refutar a meta de 70% é preciso um conjunto maior, nunca pontuado, e um pipeline congelado. **Esta seção foi escrita antes de existir qualquer previsão para o `teste2`.** Os resultados entram em 11.5.

### 11.1 Quem entra no teste2

Regra fixa (`eval/treino/teste2_monta.py`), sem amostragem por rótulo: entram **todos** os pratos do Nutrition5k que passam em tudo abaixo.

| Passo | Pratos |
|---|---|
| Com foto zenital | 3.490 |
| Passam nos filtros do v2 (massa ≥ 20 g; sem ingrediente vazio, `deprecated` ou `plate only`; soma das proteínas e das gramas coerente com o total) | 3.353 |
| Fora dos 450 ids de avaliação (`calib`, `teste`, `n5k_v1`) | 2.918 |
| Nunca usados em treino (fora dos 1.684 do treino zenital estrito e dos 2.999 pratos cujos quadros laterais treinaram o ridge da seção 8) | 1.234 |
| Desses, a menos de 120 s de algum prato de treino (mesma cadeia) | 0 |
| Desses, na cadeia de algum prato do `calib` (saem: o `calib` ajustou os pesos e a forma do híbrido) | 405 |
| **teste2** | **829 pratos, em 190 cadeias** |

- **Conferência de cada componente dos pipelines da seção 10:** os cinco backbones congelados com ridge (massa), os quatro ajustes finos e as três cabeças de frações (densidade), a recuperação por vizinhos e todas as outras cabeças de CPU e de GPU foram treinados **só** nos 1.684 pratos do treino zenital estrito (no Kaggle, as mesmas listas, conferidas por hash na ingestão). O único treino com outros pratos é o ridge com quadros laterais da seção 8, que não entra nos pipelines da seção 10; mesmo assim, nenhum dos 2.999 pratos dele é candidato nem está na cadeia de um candidato. Os backbones vêm de pré-treino em dados da web, não do Nutrition5k.
- **De onde vêm:** split oficial de treino 635, de teste 169, sem split 25; café 1: 804, café 2: 25. Por faixa de proteína (só contagem): 0–5 g 316; 5–15 g 169; 15–25 g 132; 25–40 g 110; > 40 g 102. Por número de ingredientes: 1: 191; 2 a 4: 235; 5 ou mais: 403.
- **Limite que não dá para contornar com o Nutrition5k, dito antes dos resultados: o teste2 não é independente do `teste`.** Todo prato que nunca entrou no treino ficou de fora justamente por estar na cadeia de um prato de avaliação. Dos 829, **722 são "irmãos" de pratos do `teste` antigo** (o mesmo prato, fotografado segundos antes ou depois, com um ingrediente a mais ou a menos) e 107 estão em cadeias que só têm pratos do `n5k_v1`. O teste2 é novo no sentido de que nenhum modelo viu essas fotos e nenhuma previsão para elas foi pontuada; não é uma amostra nova de pratos. Três consequências, já fixadas:
  1. os intervalos são calculados **por cadeia** (bootstrap que sorteia cadeias inteiras), não por prato: a unidade independente é a cadeia, e são 190;
  2. o resultado conjunto `teste` + `teste2` também é por cadeia, juntando irmãos na mesma unidade;
  3. os 107 pratos (24 cadeias) sem irmão no `teste` são relatados à parte. É a única parte independente do `teste`, e é pequena.
- **A composição é outra.** O `teste` foi sorteado com 56 pratos por faixa de proteína. O teste2 é o que sobrou: 38% dos pratos têm até 5 g, que são os fáceis (acerto acima de 90% pela tolerância de ±5 g). Um acerto simples no teste2 sairia inflado e não seria comparável aos números das seções 8 a 10. Por isso a medida principal é **ponderada para as cinco faixas terem o mesmo peso** (11.3).
- **Lotes cegos:** ids neutros `e0001` a `e0829` em ordem aleatória (semente fixa), 28 lotes de 29 ou 30 fotos embaralhadas em `<scratchpad>/cego4/loteNN/`, com o mesmo texto do estimador v3 (`INSTRUCOES.md`). O gabarito está em `gabarito_pratos_teste2.csv` (mesmo esquema do v2) e `n5k_teste2_amostra.csv` (com a cadeia e o tipo de cadeia de cada prato); a lista sem rótulo para o Kaggle, em `eval/treino/kaggle/listas/aval2_ids.csv`.

### 11.2 Pipeline congelado

Nada é escolhido nem ajustado de novo. São os dois pipelines escolhidos na seção 10, com as constantes de lá.

- **Massa:** exp da média dos logs de massa de cinco backbones congelados com ridge e 8 orientações: `gpu ridge siglip2_b16_384x512 o8`, `gpu ridge dinov2_b14_336x448 o8`, `gpu ridge dinov2_l14_336x448 o8`, `gpu ridge siglip2_l16_384x512 o8`, `gpu ridge siglip2_so400m16_384x512 o8`.
- **Densidade do modelo:** média em log (com o mesmo piso de 0,01) de cinco estimativas:
  1. a cabeça de densidade da média dos quatro ajustes finos `gpu ft siglip2_b16_224 r1`, `gpu ft siglip2_b16_384x512 r1`, `gpu ft dinov2_b14_336x448 r1`, `gpu ft dinov2_l14_224x308 r1`;
  2. a cabeça de densidade de `gpu ridge siglip2_so400m16_384x512 o8`;
  3. `soft-mlp ingrediente-l2 siglip2_b16_512` (rota por frações);
  4. `soft-mlp fonte-l2d siglip2_b16_512` (rota por frações);
  5. `soft-mlp fonte-l2 +perda-dens peso-faixa ambos` (rota por frações).
- **Só imagem (congelado):** massa × densidade do modelo. No `teste`: 69,6%.
- **Híbrido (congelado, número principal):** massa × [(densidade v3 com os componentes de frango e peru na densidade média do treino, 28,25 g/100 g)^0,4 × (densidade do modelo)^0,6]. É a forma e o peso que o `calib` escolheu na seção 10 com a média das duas rodadas (0,4 para o LLM). No `teste`: 72,9% com a média das rodadas.
- **LLM v3:** a mesma instrução (`eval/prompt_v3.md`), convertida por `protein_from_shares.py` com os mesmos pesos da segunda hipótese. **O teste2 recebe UMA rodada cega.** A seção 9.7 (d) mostrou que duas rodadas concordam em 0,977 na densidade por prato e que a média não melhora a densidade; na seção 10, cada rodada sozinha deu 72,9% e 74,3% no `teste` (com a forma escolhida para cada rodada).
- **O que muda por força:** as previsões para o teste2 só existem depois de rodar de novo, no Kaggle, as nove configurações acima com os pesos guardados, e de treinar de novo as três cabeças de CPU (os pesos não tinham sido guardados). As receitas, as listas de treino e as sementes são as mesmas; o ridge é determinístico, o ajuste fino em GPU não é idêntico bit a bit. Antes de pontuar, a paridade é conferida **sem gabarito**: correlação e diferença mediana entre as previsões novas e as da seção 10 nos 450 pratos antigos, e a CV no treino. O peso 0,4, as definições dos conjuntos e a densidade do frango **não** são reajustados com os modelos novos.
- **Fora do congelado, só como relato:** massa × densidade v3 sem nenhum peso; o modelo único `gpu ft siglip2_b16_224 r2` pela rota geométrica, se as previsões dele forem geradas. As formas com o LLM v2 (inclusive a que o `calib` preferia entre todas na seção 10) **não** são avaliadas no teste2: não há rodada v2 para esses pratos.

### 11.3 Regra de decisão, escrita antes de pontuar

1. **Medida principal:** critério do produto (±20% ou ±5 g) do **híbrido congelado** no teste2, **ponderado para as cinco faixas de proteína real terem o mesmo peso** (as faixas de todas as tabelas: até 5, 5–15, 15–25, 25–40, mais de 40 g).
2. **Intervalo:** bootstrap de 95% por percentil, 4.000 sorteios de **cadeias** com reposição, com os pesos por faixa recalculados em cada sorteio.
3. **Decisão:**
   - **70% demonstrado** só se o limite inferior do intervalo da medida principal no teste2 for 70% ou mais;
   - **70% refutado** se o limite superior ficar abaixo de 70%;
   - caso contrário, **não demonstrado** (compatível com 70%, sem prova).
4. **Também saem, sem mudar a decisão:** o só imagem congelado, pela mesma medida; o acerto simples (sem ponderar), marcado como não comparável; por faixa; acima de 15 g e de 25 g; os 107 pratos sem irmão no `teste`; e **`teste` + `teste2` juntos** (acerto por faixa com os pratos dos dois conjuntos, cinco faixas com o mesmo peso, intervalo por cadeia sobre a união), com o número de pratos e de cadeias. No conjunto, o `teste` entra com o mesmo pipeline congelado e com a rodada 1 do LLM v3, para que os dois lados usem uma rodada só.
5. **O teste2 é lido uma vez.** O script de pontuação roda uma única vez no teste2. Antes dele, só conferências sem gabarito (paridade nos 450 pratos antigos, pratos sem previsão, partes que somam 100, nomes sem casamento). Se alguma conferência falhar, conserta-se a mecânica e registra-se aqui o que mudou; a regra e o pipeline não mudam.

### 11.4 Preparação do lado da CPU (sem gabarito do teste2)

Feito antes de chegarem a rodada cega v3 e a nova execução no Kaggle. Nenhum rótulo do teste2 foi lido e nada foi pontuado nele.

- **Features do teste2:** SigLIP 2 B/16 a 224 px e a 384×512, 2 orientações, para as 829 fotos (`zen_extrai.py --conj teste2`; 3,5 min e 30 min de CPU). Conferência da extração: 50 pratos antigos (`n5k_v1`) extraídos de novo dão as mesmas features do banco (diferença máxima de 0,0002, que é o arredondamento de 16 bits).
- **As três cabeças de frações** (`teste2_cabecas.py`): treinadas de novo em todo o treino com o mesmo código, a mesma configuração e a mesma semente. Nos 450 pratos antigos, as previsões novas são **idênticas** às do banco (diferença máxima de 0,00000 g/g nas três cabeças e nos três conjuntos). São, portanto, os mesmos modelos da seção 10, agora com previsão para o teste2.
- **Script de pontuação** (`teste2_avalia.py`): implementa 11.2 e 11.3 e tem três modos. `--confere` só faz conferências sem gabarito (pode rodar à vontade). `--ensaio` roda a pontuação inteira com o `teste` **antigo** no lugar do teste2, sem ler o teste2. Sem opção, pontua o teste2 uma vez, grava uma marca e recusa rodar de novo.
- **Ensaio no `teste` antigo, com as previsões da primeira execução no Kaggle:** o script refaz os números da seção 10: só imagem 69,6%; erro de massa de 15,3%; massa × densidade v3 da rodada 1, sem pesos, 73,2%; modelo único `r2` 62,9%. O híbrido congelado (forma e peso de 11.2) com a rodada 1 dá 72,1% no `teste` (IC por cadeia: 66,8 a 77,3; o `teste` tem 230 cadeias para 280 pratos). Esse é o lado do `teste` na leitura conjunta de 11.3. Foi mais uma leitura do `teste` antigo, só de conferência.
- **Comando da leitura única**, quando as duas entradas existirem (`<scratchpad>/kaggle_out2/` e `docs/dados/precisao/estimativas_v3_teste2_partes.csv`):

```
cd eval/treino
python3 teste2_avalia.py <scratchpad>/kaggle_out2 --confere   # sem gabarito; tem de terminar com "PRONTO para a leitura única do teste2."
python3 teste2_avalia.py <scratchpad>/kaggle_out2             # a leitura única
```

### 11.5 Resultados

**Veredito pela regra de 11.3: 70% NÃO DEMONSTRADO, e também não refutado.** No teste2, o híbrido congelado dá **70,8%** na medida principal (ponderado por faixa), com intervalo de 95% por cadeia de **66,3% a 75,2%**. O limite inferior está abaixo de 70% e o superior, acima. O resultado é compatível com 70%, sem prova.

**Entradas e conferências antes da leitura (sem gabarito).**

- **Nova execução no Kaggle** (2 × T4, 11 configurações, pesos guardados): os cinco ridge repetem a seção 10 exatamente nos 450 pratos antigos; os ajustes finos repetem com correlação de 0,9998 ou mais no log da massa (diferença mediana de 0,0002 a 0,0081). O pipeline congelado refeito dá a mesma massa e, na proteína só de imagem, diferença relativa mediana de 0,1%, com 100% dos pratos a menos de 5% do valor da seção 10. CV da massa no treino: 0,2051, igual.
- **Rodada cega v3 do teste2:** 28 estimadores, 829 pratos, 2.966 componentes (3,58 por prato), todas as partes somam 100, todos os nomes casaram com as tabelas. Certeza `alta` em 1.575 componentes, `media` em 1.083, `baixa` em 308; 1.355 com segunda hipótese.
- `teste2_avalia.py --confere` terminou com "PRONTO para a leitura única do teste2". **A pontuação rodou uma vez** (11/11/2026, 19:13); a marca de leitura está gravada e o script recusa rodar de novo.

**Resultado, por conjunto.** "Ponderado" = cinco faixas de proteína com o mesmo peso (a medida da regra); intervalo por bootstrap de cadeias.

| Conjunto | Pratos / cadeias | **Híbrido congelado (ponderado)** | Só imagem congelado (ponderado) | Híbrido − só imagem |
|---|---|---|---|---|
| **teste2 (medida principal)** | 829 / 190 | **70,8%** (IC 66,3 a 75,2) | 69,7% (IC 65,2 a 74,2) | +1,1 (−2,1 a +4,3) |
| teste2, só os 107 sem irmão no `teste` | 107 / 24 | 69,3% (IC 56,0 a 88,5) | 66,6% (IC 56,0 a 80,4) | |
| teste2, os 722 irmãos de pratos do `teste` | 722 / 166 | 71,0% (IC 66,4 a 75,5) | 70,3% (IC 65,4 a 75,3) | |
| `teste` antigo, mesmo pipeline congelado (modelos refeitos, LLM v3 rodada 1): releitura | 280 / 230 | 72,5% (IC 67,1 a 77,6) | 69,3% (IC 63,8 a 74,6) | |
| **`teste` + `teste2` juntos** | 1.109 / 254 | **71,5%** (IC 67,9 a 75,2) | 69,8% (IC 65,8 a 73,5) | |

- **Acerto simples no teste2, sem ponderar: 78,8% no híbrido e 77,2% no só imagem. Não é comparável com a meta nem com as seções anteriores:** 38% dos pratos do teste2 têm até 5 g de proteína, e neles o acerto é de 97–98%.
- **Os 107 pratos sem irmão no `teste`** dão 69,3%, com intervalo de 56% a 88%: são 24 cadeias, e não dizem nada sozinhos.
- **Juntando os dois conjuntos** (1.109 pratos, 254 cadeias; o teste2 só acrescenta 24 cadeias às 230 do `teste`, o resto são irmãos): 71,5%, intervalo de 67,9% a 75,2%. O limite inferior continua abaixo de 70%.

**Por faixa de proteína real no teste2.**

| Faixa | Pratos | **Híbrido congelado** | Só imagem congelado | Massa × dens. v3, sem pesos | Modelo único `r2` (aparelho) | Massa real × dens. do modelo | Massa real × dens. v3 | Massa do modelo × dens. real | Massa: MAPE (viés mediano) |
|---|---|---|---|---|---|---|---|---|---|
| 0–5 g | 316 | 98% | 97% | 97% | 99% | 98% | 98% | 99% | 23,4% (+4%) |
| 5–15 g | 169 | 87% | 82% | 91% | 81% | 88% | 91% | 97% | 15,3% (+3%) |
| 15–25 g | 132 | 65% | 64% | 67% | 53% | 69% | 64% | 89% | 13,9% (+4%) |
| 25–40 g | 110 | 52% | 51% | 52% | 47% | 74% | 59% | 81% | 13,2% (+5%) |
| > 40 g | 102 | 52% | 55% | 41% | 42% | 65% | 44% | 77% | 13,9% (+3%) |
| **Ponderado (IC por cadeia)** | 829 | **70,8%** (66,3 a 75,2) | 69,7% (65,2 a 74,2) | 69,6% (65,0 a 74,3) | 64,5% (60,1 a 68,8) | 78,8% (73,7 a 83,4) | 71,2% (66,4 a 76,0) | 88,8% (85,5 a 92,1) | 17,7% |
| Simples (não comparável) | 829 | 78,8% | 77,2% | 78,2% | 74,1% | 84,2% | 79,4% | 92,2% | |
| > 15 g (três faixas, mesmo peso) | 344 | 56,3% | 56,7% | 53,5% | 47,5% | 69,1% | 55,9% | 82,6% | |
| > 25 g (duas faixas, mesmo peso) | 212 | 51,9% | 52,9% | 46,5% | 44,7% | 69,2% | 51,6% | 79,2% | |

O que a leitura mostra:

1. **O nível se repete: cerca de 70%.** O híbrido deu 72,9% no `teste` (seção 10, média das rodadas), 72,5% no `teste` com os modelos refeitos e uma rodada, e 70,8% no teste2. O só imagem deu 69,6%, 69,3% e 69,7%. O só imagem é o número mais estável deste estudo.
2. **A meta não está demonstrada.** O limite inferior é 66,3% no teste2 e 67,9% nos dois conjuntos juntos. Com 190 cadeias o intervalo tem ±4,5 pontos; com 254, ±3,7. Para o limite inferior chegar a 70%, a estimativa teria de estar perto de 74%.
3. **A meta não está refutada.** O limite superior é 75,2%.
4. **No teste2, o LLM v3 não acrescentou nada que se distinga de zero:** híbrido menos só imagem = +1,1 ponto (IC −2,1 a +4,3). No `teste` a diferença era de +3,2 (IC +0,4 a +6,4, na seção 10). E a densidade do v3, isolada, **não repetiu** a vantagem que tinha no `teste`: com a massa real, ela dá 71,2% no teste2, abaixo da densidade do modelo treinado (78,8%); no `teste` era o contrário (80,7% contra 75,4%). A queda está nos pratos com mais de 25 g (59% e 44% com a massa real). É outra rodada, com outros 28 estimadores e outros pratos; o que os dados dizem é que a vantagem do LLM medida no `teste` não se confirmou.
5. **Pratos com mais de 25 g: 52%**, no híbrido e no só imagem (25–40 g: 52% e 51%; > 40 g: 52% e 55%). É o mesmo patamar do `teste`, em que o mesmo pipeline dá 52% e 57% no híbrido nessas duas faixas. Continuam sendo o ponto fraco.
6. **A massa no teste2:** erro de 17,7% (15,3% no `teste`), com viés **positivo** de 3% a 5% em todas as faixas; no `teste` o viés acima de 15 g era negativo (−6% a −2%). Com a densidade real, a massa do modelo daria 88,8% (87,5% no `teste`). O erro de massa não é o que limita o teste2: a densidade do modelo com a massa real dá 78,8%, e os dois erros juntos, 69,7%.
7. **Modelo único para o aparelho (`gpu ft siglip2_b16_224 r2`):** 64,5% ponderado (IC 60,1 a 68,8) no teste2; 62,9% no `teste`. Fica 5 pontos abaixo do conjunto de servidor.

**O que os dados sustentam, depois do teste2:** um pipeline só de imagem (de servidor) em 70% no Nutrition5k, com intervalo de 65% a 74%; um híbrido com o LLM v3 em 71%, com intervalo de 66% a 75%, sem ganho seguro sobre o só imagem; um modelo único de aparelho em 63–65%. **O que não sustentam:** que a meta de 70% foi atingida; que o LLM em nuvem compensa; qualquer coisa nos pratos com mais de 25 g (52%); qualquer coisa fora do Nutrition5k.

**Registro de método.** (a) Tudo em 11.1 a 11.3 foi escrito antes de existir previsão para o teste2, e 11.4 antes de chegarem as duas entradas. (b) O teste2 foi lido uma vez. Antes dela rodaram, sem gabarito do teste2: `--confere` (duas vezes: antes e depois de as entradas chegarem), a conversão das partes v3 para contar nomes sem casamento (nenhum) e o ensaio no `teste` antigo. (c) O `teste` antigo foi relido duas vezes nesta seção (no ensaio e, com os modelos refeitos, dentro da leitura única), só como relato. (d) Nenhuma escolha, peso ou forma foi mudado depois de 11.2. (e) O teste2 não é independente do `teste` (11.1): os intervalos são por cadeia por causa disso, e a parte independente (107 pratos) é pequena demais para decidir.

### 11.6 Custo de servir (medido na segunda execução no Kaggle)

Medidas de `servico.json` e `tamanhos.json` (pasta de saída do Kaggle, fora do repositório). GPU: uma Tesla T4, meia precisão. CPU: Xeon de 2,0 GHz, 4 threads, precisão simples. O tempo inclui redimensionar, normalizar, a rede e a cabeça, a partir da imagem 480×640 já na memória; não inclui decodificar o PNG nem copiar para a GPU. Um modelo por vez, uma imagem por vez. "8 em lote" = as 8 orientações da mesma imagem num lote.

| Modelo | Parâmetros | Pesos em memória (fp16) | T4: 1 orientação | T4: 8 orientações (em sequência / em lote) | CPU: 1 orientação | CPU: 8 orientações | Conferência (dif. máx. em log) |
|---|---|---|---|---|---|---|---|
| `gpu ridge siglip2_b16_384x512 o8` | 93 M | 187 MB | 13 ms | 108 / 88 ms | 0,73 s | 6,6 s | 0,0008 |
| `gpu ridge dinov2_b14_336x448 o8` | 86 M | 172 MB | 14 ms | 112 / 89 ms | 0,82 s | 6,7 s | 0,0002 |
| `gpu ridge dinov2_l14_336x448 o8` | 304 M | 608 MB | 44 ms | 359 / 311 ms | 2,5 s | 20,5 s | 0,0002 |
| `gpu ridge siglip2_l16_384x512 o8` | 317 M | 633 MB | 44 ms | 376 / 331 ms | 2,7 s | 21,8 s | 0,0004 |
| `gpu ridge siglip2_so400m16_384x512 o8` | 429 M | 857 MB | 66 ms | 552 / 517 ms | 3,4 s | 27,6 s | 0,0003 |
| `gpu ft siglip2_b16_224 r1` | 93 M | 186 MB | 8 ms | 62 / 23 ms | 0,20 s | 1,6 s | 0,0025 |
| `gpu ft siglip2_b16_384x512 r1` | 93 M | 187 MB | 13 ms | 111 / 93 ms | 0,86 s | 6,9 s | 0,0030 |
| `gpu ft dinov2_b14_336x448 r1` | 86 M | 172 MB | 14 ms | 115 / 94 ms | 0,85 s | 6,8 s | 0,0027 |
| `gpu ft dinov2_l14_224x308 r1` | 303 M | 607 MB | 24 ms | 195 / 166 ms | 1,2 s | 9,5 s | 0,0020 |
| **Massa: os 5 backbones congelados** | **1.228 M** | **2,46 GB** | **181 ms** | **1.507 / 1.335 ms** | **10,1 s** | **83 s** | |
| **Só imagem: os 9 modelos** | **1.804 M** | **3,61 GB** | **241 ms** | **1.990 / 1.711 ms** | **13,2 s** | **108 s** | |
| `gpu ft siglip2_b16_224 r2` (modelo único, aparelho) | 93 M | 186 MB | 8 ms | 64 / 24 ms | 0,21 s | 1,6 s | 0,0034 |
| `gpu ridge siglip2_b16_224 o8` (backbone a 224 px da cabeça `ambos`; fora dos totais) | 93 M | 186 MB | 7 ms | 58 / 23 ms | não medido | não medido | 0,0008 |

- **O pipeline só de imagem, como foi avaliado (8 orientações), custa cerca de 1,7 a 2,0 s por foto numa T4** e 3,6 GB de pesos em memória (pico de 1,9 GB de memória de GPU no maior modelo, um por vez). Em CPU são quase 2 minutos por foto: não serve sem GPU. Com uma orientação só, 0,24 s na T4; a precisão com uma orientação **não** foi medida para este conjunto (na seção 8, treinar e prever com 8 orientações baixou o erro de massa de 21,1% para 18,6%).
- **A massa sozinha** (5 backbones) custa 1,3 a 1,5 s por foto na T4. Mais de 80% disso são os três backbones grandes (L/14, L/16, So400m).
- **Arquivos:** os pesos públicos dos cinco backbones somam 11,1 GB para baixar (os arquivos originais trazem também a torre de texto e estão em precisão simples); as cabeças ridge guardadas somam 1,8 MB; os quatro ajustes finos guardados, 1,15 GB. A saída inteira da execução tem 1,27 GB.
- **Conferência:** cada modelo remontado a partir dos arquivos de servir reproduz as previsões gravadas com diferença máxima, em log, de 0,0002 a 0,0034 (8 pratos por modelo). Os arquivos guardados servem para servir.
- **O modelo único `r2`:** 93 M de parâmetros, 186 MB em meia precisão, 8 ms por orientação na T4 e 0,2 s numa CPU x86 de 4 threads (1,6 s com 8 orientações).
- **Não medido:** nada no iPhone (nem conversão para Core ML, nem latência, nem memória); as três cabeças de frações de CPU (MLPs pequenas sobre features já calculadas, custo presumido desprezível, não cronometrado); o backbone a 224 px em CPU; a chamada ao LLM de visão (latência e preço por foto); decodificação da imagem e transferência; vazão com várias fotos ao mesmo tempo; custo em dinheiro por foto.


## 12. Um modelo só e mais dados de treino (11/11/2026)

**Direção nova (registrada em `docs/produto.md`, seção 7):** 70% é o **mínimo**, não o teto; a v1 é **só de imagem** (sem LLM em nuvem: no teste2 ele não mostrou ganho seguro); o conjunto de nove redes tem de virar **um modelo, no máximo dois**, a partir de uma foto de cima; e é preciso **mais dados de treino**. De 12.1 a 12.5 é desenho e pré-registro: **nada ali leu o teste2**, e nenhum modelo novo tinha sido treinado. Os resultados da terceira execução e a leitura do teste2 estão em 12.6; a emenda que permitiu mais uma leitura, em 12.7.

### 12.1 Protocolo de avaliação daqui para a frente

1. **Teste de registro: o teste2** (829 pratos, 190 cadeias), com a medida de 11.3: critério do produto ponderado para as cinco faixas de proteína terem o mesmo peso, intervalo de 95% por bootstrap de cadeias. Ele foi lido **uma vez** (11.5). Referências nessa medida:

| Pipeline (seção 11) | teste2, ponderado (IC por cadeia) | > 25 g |
|---|---|---|
| Só imagem, nove redes | 69,7% (65,2 a 74,2) | 52,9% |
| Modelo único `gpu ft siglip2_b16_224 r2` | 64,5% (60,1 a 68,8) | 44,7% |
| (Híbrido com LLM v3, fora da v1) | 70,8% (66,3 a 75,2) | 51,9% |

2. **Regra de sucesso:** meta demonstrada só se o **limite inferior do intervalo de 95% for 70% ou mais** nessa medida. Com 190 cadeias o intervalo tem cerca de ±4,5 pontos: **a estimativa precisa estar perto de 74,5%**. É 5 pontos acima do melhor pipeline de hoje e 10 acima do melhor modelo único.
3. **Como se escolhe sem tocar no teste2:** validação cruzada agrupada por cadeia (5 folds) no treino, com o critério ponderado por faixa, como nas seções 8 a 10. Toda escolha (modelo, rota, número de orientações, uso de dados externos) sai da CV. O `calib` deixa de existir como conjunto separado (12.2).
4. **Leituras do teste2 permitidas a partir de agora: no máximo duas, uma por candidato final congelado.**
   - **C1, o modelo único** que a regra de 12.3 escolher pela CV. É o candidato principal.
   - **C2, um pipeline de dois modelos,** só se ajudar com clareza na CV: critério de CV pelo menos **2,0 pontos** acima do C1 **e** limite inferior de 95% (por cadeia, nas previsões fora-do-fold) da diferença pareada acima de zero.
   - As duas leituras, se houver duas, saem **na mesma execução**. Com dois candidatos lidos, o intervalo de 97,5% (correção para duas leituras) é relatado ao lado do de 95%; "passou com folga" só se o limite inferior de 97,5% também for 70% ou mais.
   - O professor (12.3) e os candidatos que perderem na CV **não** são pontuados no teste2.
5. **O que não pode entrar em nenhuma escolha:**
   - as fotos do teste2, nem sem rótulo (nada de destilação ou pseudo-rótulo nelas);
   - os resultados por prato do teste2 da seção 11 (o arquivo existe no scratchpad; não é aberto para desenhar nada);
   - os pratos da avaliação antiga que ficam fora do treino (204 do `teste` e 41 do `n5k_v1` em cadeia do teste2): são irmãos de pratos do teste2, então escolher por eles seria escolher pelo teste2. Não são pontuados antes da leitura final.
6. **O `teste` antigo (280) está aposentado como número de manchete.** 65 dos seus pratos entram no treino (12.2); os outros são irmãos do teste2. Os números das seções 2 a 10 ficam como história.
7. **Antes de cada leitura:** conferências sem gabarito, como em 11.4 (previsões completas e na ordem de `aval2_ids.csv`; modelo remontado dos arquivos de servir reproduz as previsões gravadas).

### 12.2 Mais treino do Nutrition5k: treino_v2

Com o LLM fora, o `calib` não ajusta mais nada e pode treinar. A regra (`eval/treino/treino_v2_monta.py`): entra todo prato com foto zenital que passa no filtro do treino, **não é do teste2 e não está na cadeia (< 120 s) de nenhum prato do teste2**.

| Origem | Pratos |
|---|---|
| Treino antigo (seções 8 a 11) | 1.684 |
| Pratos que estavam fora só por estar em cadeia do `calib` | 405 |
| `calib` (2 dos 120 não passam no filtro do treino) | 118 |
| `teste` antigo, só os de cadeias que não tocam o teste2 | 65 |
| `n5k_v1`, idem | 9 |
| **treino_v2** | **2.281 pratos, 646 cadeias** (antes: 1.684 e 511) |

| Faixa de proteína | 0–5 g | 5–15 g | 15–25 g | 25–40 g | > 40 g | > 25 g | > 15 g |
|---|---|---|---|---|---|---|---|
| Treino antigo | 643 | 360 | 220 | 282 | 179 | 461 | 681 |
| **treino_v2** | 846 | 450 | 310 | 391 | 284 | **675** | **985** |
| Aumento | +32% | +25% | +41% | +39% | +59% | **+46%** | +45% |

- **Conferência com o grafo de cadeias** (o script confere com `assert`): 0 pratos do teste2 no treino_v2; 0 cadeias em comum; cada cadeia num fold só; os alvos dos 1.684 pratos antigos são idênticos aos de `listas/treino.csv`.
- **O que continua fora de tudo:** os 829 do teste2; 204 pratos do `teste` antigo e 41 do `n5k_v1` que estão em cadeia do teste2 (confirmado: 166 das 230 cadeias do `teste` antigo tocam o teste2); 11 do `teste` e 2 do `calib` que não passam no filtro do treino.
- **`eval/treino/kaggle/listas/treino_v2.csv`:** mesmo esquema de `treino.csv` (cadeia, fold por cadeia, alvos, massa e proteína por grupo), com 10 colunas a mais no fim, `ff_<fonte>`: a fração da massa do prato em cada fonte de proteína (alvo das cabeças de frações de 12.3). Os folds são novos (GroupKFold de 5 sobre as 646 cadeias; 456 ou 457 pratos por fold, 125 a 154 com mais de 25 g). `n5k_treino_v2_origem.csv` guarda a origem de cada prato.
- **Quanto esperar disso:** o ganho de dados é de 35% no total e de 46% acima de 25 g, do mesmo refeitório. A curva da seção 3 já estava quase plana, então conto com 1 a 2 pontos, concentrados nos pratos grandes. Não resolve sozinho.

### 12.3 Especificação da terceira execução em GPU

Orçamento: cerca de 3,5 h de relógio em 2 × T4 (a cota semanal está no fim). Tempos de referência da segunda execução: os cinco backbones congelados, 75 min de GPU; ajuste fino do SigLIP 2 B/16 a 224 px (`r2`), 45 min; a 384×512 (`r1`), 112 min. Com 35% a mais de treino, os ajustes finos passam para cerca de 61 e 151 min. **Só entram no treino os pratos de `treino_v2.csv`. As fotos do teste2 só servem para gerar previsões no fim.**

**O que as seções 8 a 11 ensinaram e o desenho usa:**

- Massa: backbones grandes **congelados** com ridge ganham de todo ajuste fino (0,219 a 0,228 contra 0,245 a 0,266 de RMSE em log); a média de cinco chega a 0,205. O ajuste fino ajuda a proteína, não a massa.
- Densidade: prever a **fração da massa por fonte de proteína** com erro quadrático ganha de prever a densidade direto (+6 pontos com a massa real); classes mais finas e ingredientes não ajudam.
- Resolução ajuda a identificar (fonte certa em 78–80% a 384×512 contra 74% a 224 px). Profundidade e destilação RGB-D não ajudam (três tentativas). Esticar previsões não ajuda. Peso por faixa ajuda um pouco a densidade e nunca foi testado na massa.
- Um modelo único fica 5 pontos abaixo do conjunto (64,5% contra 69,7% no teste2). O conjunto ganha por juntar backbones, não por uma receita.

**Configurações, em ordem de prioridade** (se o orçamento acabar, as últimas ficam de fora e a regra vale para as que terminarem):

| # | Nome | O que é | Custo de GPU (estimativa) |
|---|---|---|---|
| 1–5 | `gpu3 ridge <backbone> o8`, para `siglip2_so400m16_384x512`, `siglip2_l16_384x512`, `dinov2_l14_336x448`, `siglip2_b16_384x512`, `dinov2_b14_336x448` | Como na segunda execução, treinados no treino_v2. As features (8 orientações) ficam em cache para as cabeças abaixo | 80 min (dividir entre as duas GPUs) |
| 6–8 | `gpu3 fracoes so400m`, `gpu3 fracoes l16`, `gpu3 fracoes b16_384x512` | Cabeça de frações por fonte sobre as features congeladas de cada backbone (nunca testada nos backbones grandes) | 2 min |
| 9 | `gpu3 professor` | **Professor:** massa = média em log das 5 massas ridge; densidade = média em log (piso 0,01) das 3 cabeças de frações e da cabeça de densidade do ridge So400m; proteína = massa × densidade. Só modelos sobre features congeladas, para permitir alvos aninhados | 3 min |
| 10 | `gpu3 unico so400m mlp destilado` (**U1**) | **Um backbone congelado** (So400m/16 a 384×512) + cabeça MLP com rótulos e destilação. Modelo único de servidor | 2 min |
| 11 | `gpu3 ft siglip2_b16_224 r2 destilado` (**U4**) | Ajuste fino completo, receita `r2`, com cabeça de frações, peso por faixa e destilação. Modelo único do tamanho do aparelho | 61 min |
| 12 | `gpu3 ft siglip2_b16_384x512 r1 destilado` (**U3**) | Idem a 384×512, receita `r1` (12 épocas) | 151 min |
| 13 | `gpu3 ft siglip2_b16_224 r2` (**U0**) | A receita `r2` sem mudança, só com o treino_v2. Mede quanto vem dos dados e quanto do resto | 61 min |

Ordem sugerida: GPU A extrai So400m e B/16; GPU B extrai L/16 e os dois DINOv2 (~40 min); cabeças e professor (~5 min); depois GPU A roda U3 e GPU B roda U4 e U0. Total: cerca de 3,3 h de relógio, mais as medidas de serviço. **Com a base externa de 12.5**, as configurações 1 a 10 rodam também na versão `+flavoria`, a decisão de adoção sai antes dos ajustes finos, e o total passa para cerca de 3,8 h.

**Detalhes que precisam ser iguais ao que está escrito:**

- **Cabeça de frações (6–8 e dentro de U1, U3, U4):** como `soft-mlp fonte-l2 +perda-dens peso-faixa` de `dens_experimentos.py`. Nas features congeladas: MLP de uma camada oculta (512, GELU, dropout 0,3) sobre features padronizadas, cada orientação como amostra; saídas = as 22 em log (padronizadas) + 10 logits; AdamW (lr 1e-3, decaimento 1e-2), 60 épocas, lote 256, OneCycle. Perdas: Huber nas 22 saídas; **10 × erro quadrático entre o softmax e as frações `ff_*`**; Huber em log da densidade composta Σ fração × densidade média da fonte. Peso por amostra = as cinco faixas de proteína com o mesmo peso. A densidade média de cada fonte sai do treino do fold (mínimos quadrados não negativos de proteína ≈ Σ `ff` × massa × densidade da fonte; no treino antigo: frango e peru 28,2; bovina 25,7; suína 26,2; peixe 21,2; ovo 10,1; leguminosa 8,4; laticínio 12,9; prato pronto 10,7; castanhas 21,4; resto 1,9 g/100 g). Nos ajustes finos (U3, U4), a mesma cabeça de 22 + 10 saídas no lugar da de 22, com as mesmas três perdas.
- **Destilação (U1, U3, U4):** soma à perda de rótulos um Huber entre a saída do aluno e a do professor em três quantidades (log da massa, log da densidade com piso 0,01, log1p da proteína), com peso 0,5. **Alvos aninhados:** ao treinar o aluno sem o fold k, o alvo do professor para um prato do fold j vem de um professor treinado sem os folds j e k. Para o modelo final (treino inteiro), usam-se as previsões fora-do-fold comuns do professor. Isso é barato porque o professor só tem ridge e MLP sobre features em cache. Sem o aninhamento, a CV do aluno enxergaria os rótulos do fold de validação através do professor.
- **Peso por faixa (U1, U3, U4):** cinco faixas com o mesmo peso em todas as perdas, inclusive a de massa. É a medida voltada aos pratos com mais de 25 g, junto com a cabeça de frações e os 46% a mais de pratos grandes. U0 fica sem peso, sem frações e sem destilação, de propósito.
- **Aumento de dados nos ajustes finos:** orientações e escala aleatória (0,7 a 1,3), como nas receitas `r1` e `r2`. Nada de fotos do teste2, nem sem rótulo.
- **Orientações na previsão:** cada configuração grava as previsões com **1** (a original), **4** (as que preservam o quadro) e **8** orientações, fora-do-fold e no teste2, para a regra abaixo decidir o custo.

**O que cada configuração grava:** `pred__<nome>.npz` com `oof` [2.281 × 22] e `teste2` [829 × 22] na ordem de `listas/treino_v2.csv` e `listas/aval2_ids.csv`, mais `oof_ff` e `teste2_ff` [× 10] quando há cabeça de frações, e as variantes `_o1` e `_o4`; os ids, os folds e o hash das listas; registro por época; pesos em meia precisão e os arquivos de servir (como na segunda execução), com a conferência do modelo remontado; medidas de serviço (T4 e CPU, 1, 4 e 8 orientações). O professor grava também os alvos aninhados. **Não** gerar previsões para os 450 pratos da avaliação antiga.

**Regra que escolhe O modelo (C1), aplicada aqui, na CPU, só com as previsões fora-do-fold:**

1. Para cada candidato único (U0, U1, U3, U4): a **rota** de proteína de maior critério de CV ponderado, com 8 orientações, entre: direta; massa × densidade da regressão; massa × densidade das frações; massa × média em log das duas densidades; média geométrica da direta com a anterior.
2. **Orientações:** o menor número entre 1, 4 e 8 cujo critério de CV fique a até 1,0 ponto do de 8.
3. **C1 = o candidato de maior critério de CV.** Se outro ficar a menos de 1,0 ponto do melhor, vale o **mais barato de servir** entre eles (tempo por foto na T4 com as orientações escolhidas).
4. **C2:** entre os três pares declarados (U1 + U3, U1 + U4, U3 + U4; média em log das estimativas de proteína de cada um na sua rota), o de maior critério de CV; só é lido no teste2 se passar no critério de 12.1 (2,0 pontos e diferença pareada acima de zero).
5. O professor não concorre: são cinco redes. Ele entra na tabela de CV como referência do que um conjunto dá no treino_v2.

**O que eu espero, com a incerteza que tenho:**

| | CV no treino_v2 (critério ponderado) | teste2 (ponderado) |
|---|---|---|
| Referência de hoje, CV no treino antigo: `r2` 65,7%; nove redes 70,5% | | 64,5%; 69,7% |
| U0 (`r2` com mais dados) | 66 a 68% | 65 a 67% |
| U4 (224 px, destilado) | 67 a 69,5% | 66 a 68,5% |
| U3 (384×512, destilado) | 67 a 70% | 66 a 69% |
| U1 (So400m congelado + MLP) | 67 a 70% | 66 a 69% |
| Melhor par | +1 a +1,5 sobre o melhor único | 67 a 70% |
| Professor (cinco redes congeladas) | 70 a 73% | não é lido |

- **Não espero que um modelo só chegue ao nível das nove redes.** O mais provável é ficar de 1 a 3 pontos abaixo de 69,7% no teste2. Dou cerca de uma chance em quatro de empatar ou passar.
- **Não espero que nada treinado só com o Nutrition5k demonstre a meta no teste2.** Para o limite inferior chegar a 70%, a estimativa teria de estar perto de 74,5%. Dou menos de 5% de chance a isso nesta execução, com um ou com dois modelos.
- **Onde posso estar errado para melhor:** as cabeças de frações nunca foram treinadas sobre os backbones grandes, e o peso por faixa nunca foi usado na massa. Se os dois renderem como renderam na densidade da seção 9, o U1 pode surpreender.
- **O que seria preciso para um modelo só chegar lá:** (a) mais pratos pesados com mais de 25 g, de outra origem, que é o que 12.4 prepara; (b) um backbone grande com ajuste fino a 384×512 (So400m: cerca de 8 h de GPU, fora deste orçamento), se o U1 mostrar que o backbone é o que importa; (c) fotos sem rótulo, que não sejam do teste2, para a destilação. E, para a **demonstração**, o intervalo: mesmo um modelo de 73% de verdade falharia na regra metade das vezes com 190 cadeias. Um conjunto de teste maior e independente (a coleta brasileira) é parte da resposta, não só um modelo melhor.

### 12.4 Gancho para dados externos

A pesquisa de bases externas está em curso (`docs/dados/precisao/fontes_treino.md`). Uma base nova entra assim, sem mudar o protocolo de 12.1:

1. **Lista própria**, no mesmo esquema de `treino_v2.csv`, com três colunas a mais: `dominio` (`n5k_zenital` para o que existe hoje; um nome por base externa), `grupo` (a unidade que não pode ser separada entre folds: prato, sessão ou usuária) e `vista` (`zenital` ou outra). **Alvo ausente fica vazio**, e a perda daquele alvo é mascarada para a amostra (uma base só com proteína e calorias, sem massa, treina proteína e densidade, não massa).
2. **Dados externos só treinam.** Entram em todos os folds de treino (fold −1) e **nunca** nos folds de validação que escolhem o modelo: o critério de CV continua calculado só nos pratos do treino_v2. A exceção é uma base zenital, com escala comparável e massa e proteína pesadas: ela pode ter CV própria, relatada à parte, e só passa a contar na escolha se isso for declarado antes de rodar.
3. **Peso por domínio:** a soma dos pesos das amostras externas fica em no máximo metade da do Nutrition5k, para a base maior não mandar no modelo.
4. **Escala:** se a distância da câmera for outra, a base entra com a escala aleatória ligada e com um viés de log-massa por domínio na cabeça (aprendido no treino; na previsão vale o do `n5k_zenital`). Sem isso, a massa aprende a distância da base, que é o problema de 8.7.
5. **Higiene:** qualquer base derivada do Nutrition5k passa pela regra de cadeia contra o teste2 antes de entrar; bases com fotos repetidas do mesmo prato são agrupadas por `grupo`.
6. **Adoção pela CV:** cada base é uma ablação pré-declarada (com e sem), e só fica se o critério de CV no treino_v2 subir pelo menos 1,0 ponto. As leituras do teste2 continuam sendo no máximo duas.

### 12.5 Primeira base externa: FlavoriaFoodWeight1700 (pré-registro, antes de qualquer resultado da terceira execução)

**O que é** (`docs/dados/precisao/fontes_treino.md`): cerca de 2.000 fotos, de 850 a 1.000 pratos depois da limpeza, de uma linha de almoço da Universidade de Turku (Zenodo 5850856, CC BY 4.0). Cada porção foi **pesada por componente** pela balança da linha (246 nomes); **não há proteína nem calorias medidas**. Fotos de cima, 1024×1024, com o prato recortado para encher o quadro, dois quadros por bandeja. Pratos bem mais pesados que os do Nutrition5k (mediana de ~470 g contra 170 g). A lista vem em `eval/treino/kaggle/listas/externo_flavoria.csv`, no esquema de 12.4. Se a lista não estiver pronta ou não passar nas conferências, a execução segue como em 12.3, sem ela.

**Como entra no treino.**

| Alvo | No Flavoria | Peso na perda |
|---|---|---|
| Massa total (log) | Medida | 1 |
| Massa por grupo (`m_*`) e frações por fonte (`ff_*`) | Derivadas dos gramas pesados, por um mapa de nomes | 0,5 (o mapa de nomes tem erro, e a linha serve pratos prontos que o Nutrition5k separa em ingredientes) |
| Proteína, densidade, kcal, gordura, carboidrato, proteína por grupo, densidade composta das frações | Só existe a estimativa **fraca** (nome → tabela) | **0: não é usada.** Fica para uma ablação futura, se a massa mostrar ganho |
| Destilação do professor | | 0: as fotos do Flavoria só entram com rótulos |

- **Só treina:** fold −1, nunca nos folds de validação. O critério de CV continua calculado só nos pratos do treino_v2. Os dois quadros da mesma bandeja têm o mesmo `grupo`.
- **Teto de peso:** a soma dos pesos das fotos de treino do Flavoria é **metade** da soma dos pesos do Nutrition5k; dentro do Flavoria, peso igual por foto (não há faixa de proteína para ponderar).
- **Enquadramento:** as fotos do Nutrition5k **não** são recortadas pelo prato nesta execução. Motivos: mudaria a entrada e, com ela, a referência do teste2; a caixa do prato não está no pacote do Kaggle; e a seção 8.7 mediu um custo de 0,6 ponto de MAPE para o recorte. Fica como o passo seguinte, quando o círculo-guia do app estiver definido (aí os dois domínios ficam "prato enchendo o quadro"). Nesta execução:
  - a foto quadrada do Flavoria é **encaixada no centro de um quadro 4:3** (laterais preenchidas com a cor mediana da borda da própria foto) e daí em diante passa pelo mesmo pré-processamento das fotos do Nutrition5k, em todos os backbones;
  - **viés de log-massa por domínio:** de cada alvo de massa do Flavoria (total e por grupo, em log) subtrai-se a diferença entre a média do domínio e a média do Nutrition5k no treino do fold; na previsão de fotos do Nutrition5k não se soma nada, e na de fotos do Flavoria soma-se a diferença de volta. Vale igual para ridge, MLP e ajuste fino. O viés absorve junto a diferença de enquadramento e o fato de os pratos serem mesmo mais pesados; o que o Flavoria ensina é a variação **dentro** do domínio e as frações em pratos grandes;
  - nos ajustes finos, a escala aleatória (0,7 a 1,3) vale para os dois domínios.

**Parte segurada do Flavoria (teste externo de massa).** 20% dos `grupo`s, fixos: os dos últimos dias de coleta, se a lista tiver data; senão, os grupos com `sha1(grupo) mod 5 == 0`. Marcados como `teste_ext`; nunca treinam e **nunca entram em nenhuma escolha**.

**Ablação que decide a adoção, só nas configurações baratas.** As configurações 1 a 10 de 12.3 (cinco ridge, três cabeças de frações, professor, U1) rodam **duas vezes**: só com o treino_v2 (`gpu3 …`) e com o treino_v2 mais o treino do Flavoria (`gpu3 … +flavoria`). Custo a mais: features das ~2.000 fotos nos cinco backbones com 8 orientações, cerca de **50 min de GPU**, mais ~10 min de cabeças: perto de **+30 min de relógio** nas duas GPUs (total da execução: ~3,8 h). Ordem: features do Nutrition5k e do Flavoria; cabeças e professor nas duas versões; decisão pela regra abaixo, gravada no registro; só então os ajustes finos.

**Regra de adoção, aplicada às previsões fora-do-fold do professor no treino_v2 (com Flavoria menos sem Flavoria). Para o Flavoria, ela vale no lugar do item 6 de 12.4, que só tinha a primeira condição:**

1. **Adota** se o critério de CV ponderado por faixa subir **1,0 ponto ou mais**;
2. **ou** se ele não cair mais que 0,3 ponto **e** o critério nas duas faixas acima de 25 g (média das duas) subir **2,0 pontos ou mais**;
3. senão, **não adota**.

- Os ajustes finos caros rodam **uma vez**, com o resultado da regra: U4 e U3 treinam com o Flavoria se ele for adotado, e sem ele se não for. **U0 treina sempre só com o treino_v2**, para continuar medindo o efeito dos dados do Nutrition5k. O U1 e o professor que concorrem em 12.3 são os da versão adotada.
- Se o orçamento apertar, a ordem de corte é: U0, depois U3. As duplicatas com Flavoria não são cortadas antes dos ajustes finos, porque a decisão depende delas.

**Relato secundário, fora do domínio, com rótulo claro.** A CV no Nutrition5k não enxerga o que mais se espera do Flavoria (pratos grandes, prato enchendo o quadro, isto é, o uso no celular). Por isso sai também, no `teste_ext` do Flavoria, o **erro de massa** (MAPE e viés mediano, no total e nos pratos com mais de 600 g) do professor e do U1 **com e sem** o Flavoria no treino, e dos ajustes finos como foram treinados. Para os modelos sem Flavoria, dois números: a previsão crua e a previsão corrigida por uma constante (a razão mediana medida no treino do Flavoria), para separar "erra a escala" de "erra a ordem". **Esse relato não escolhe nada, não é o critério do produto (não há proteína medida) e não conta como leitura do teste2.**

**O que eu espero:** no critério de CV do Nutrition5k, efeito perto de zero (entre −0,5 e +1 ponto), porque o domínio e o enquadramento são outros e a proteína do Flavoria não entra. Acho mais provável que a regra **não** adote do que adote. No teste externo, espero uma queda grande do erro de massa com o Flavoria no treino (terceiros relatam 45% → 25%); isso mede transferência para pratos grandes, não o produto.

**Adendo ao pré-registro (11/11/2026, depois de a lista `externo_flavoria.csv` ficar pronta e antes de qualquer resultado da terceira execução):**

- **Tamanho real:** a lista tem 1.264 pratos em 1.967 fotos (uma linha por foto), e não os 850–1.000 pratos estimados acima, porque as bandejas que só têm a foto `enter` foram mantidas. O custo de extração não muda (cerca de 2.000 fotos).
- **Fotos ocluídas:** as fotos com `oclusao = 1` (119, quase todas com a cabeça de quem carrega a bandeja por cima do prato) **não entram no treino nem na métrica do `teste_ext`**. O filtro é automático e imperfeito (27 de 32 corretas numa amostra; mãos pequenas na borda escapam).
- **Métrica por prato no `teste_ext`:** usa só a foto com `quadro_preferido == 1` de cada prato, para que um prato com dois quadros não conte duas vezes. No treino vale o peso uniforme por foto já definido.
- **Divisão:** vale a coluna `split_ext` (dias 10, 11, 12, 13, 16, 19 e 20 de novembro de 2020 em `teste_ext`: 230 pratos, 18,2%), e não as pastas `train`/`val`/`test` do arquivo original, que ignoram os dias.
- **Nomes dos componentes** são menos confiáveis que os pesos (vêm do cardápio atribuído à balança); isso já está coberto pelo peso 0,5 dos alvos derivados dos nomes.

### 12.6 Resultado da terceira execução em GPU (12/11/2026)

A terceira execução rodou no Kaggle (2 × T4, 3,6 h de relógio, orçamento de 4,5 h). Terminaram 23 configurações: as 13 de 12.3 e as 10 duplicatas `+flavoria` de 12.5. Os dois ajustes finos `+flavoria` foram cancelados pela regra de 12.5 (o Flavoria não foi adotado; item (f)). Nada foi cortado por orçamento. A ingestão conferiu ids, folds e hash das listas em todas (`ingere_kaggle.py --so-confere`: 23 aceitas, 0 recusadas). A escolha e a leitura do teste2 são de `eval/treino/um_modelo_avalia.py`.

**a) O que a regra de 12.3 diz sobre o U0, escrito antes de abrir qualquer número do teste2.**

A especificação não previu o caso em que os alunos destilados perdem para o controle sem destilação. Foi o que aconteceu na CV (item (b)), e a resposta muda o candidato. Por isso a leitura do texto fica registrada aqui, antes da leitura do teste2:

- **O U0 é elegível como C1 pelo texto.** O item 1 da regra diz "para cada candidato único (U0, U1, U3, U4)", e o item 3 diz "C1 = o candidato de maior critério de CV", com o desempate pelo mais barato de servir. O U0 está na lista por nome. A tabela de configurações o descreve como a receita que "mede quanto vem dos dados e quanto do resto", e 12.5 diz que ele "treina sempre só com o treino_v2, para continuar medindo o efeito dos dados". Essas frases dizem para que ele serve como comparação. Nenhuma o tira da regra de escolha, e o único excluído por escrito é o professor (item 5). Sigo o texto: o U0 concorre.
- **O U0 não entra em nenhum par.** O item 4 declara três pares (U1 + U3, U1 + U4, U3 + U4). Um par com o U0 não foi declarado, então não pode ser o C2, mesmo que fosse o melhor par na CV. Antes da leitura não calculei nenhum par com o U0 (os que aparecem no item (e) são exploratórios e foram calculados depois).
- **Rotas do U0.** Ele não tem cabeça de frações. Das cinco rotas do item 1 existem três: direta; massa × densidade da regressão; e a média geométrica da direta com massa × densidade da regressão (a "anterior" do texto, a média em log das duas densidades, se reduz à única densidade que ele tem). É o que a execução gravou.
- **Pontos em que o texto admite mais de uma leitura, e o que vale** (calculei as duas para mostrar se a escolha depende disso):
  - "Critério de CV" do item 3: vale o da rota escolhida **com o número de orientações escolhido** no item 2, que é o modelo que seria servido e a que o tempo na T4 se refere. Ao lado sai o critério com 8 orientações.
  - "Tempo por foto na T4": uma foto por vez, as orientações em sequência (`lote1_oK` de `servico.json`).
  - Pares: média em log (log1p) das estimativas de proteína de cada modelo na sua rota e com as suas orientações; a comparação com o C1 é o bootstrap de cadeias (4.000 sorteios das 646 cadeias do treino_v2) da diferença do critério ponderado, nas previsões fora-do-fold.
  - O U1 e o professor que valem são os treinados sem o Flavoria, porque a regra de 12.5 não o adotou.

**b) Aplicação da regra, só com as previsões fora-do-fold do treino_v2 (2.281 pratos, 646 cadeias).** Critério ponderado por faixa. Os números reproduzem os do registro da execução.

| Candidato | Rota escolhida (8 orientações) | CV com 8 / 4 / 1 orientações | Orientações (item 2) | Critério de CV | RMSE de log-massa | > 25 g | T4 por foto |
|---|---|---|---|---|---|---|---|
| **U0** `gpu3 ft siglip2_b16_224 r2` (sem destilação) | massa × densidade | 68,1 / 68,1 / 68,0 | **1** | **68,0%** | 0,234 | 48,5% | **8 ms** |
| U1 `gpu3 unico so400m mlp destilado` | geométrica (direta com massa × média das duas densidades) | 68,7 / 68,3 / 67,3 | 4 | 68,3% | 0,238 | 44,8% | 248 ms |
| U4 `gpu3 ft siglip2_b16_224 r2 destilado` | massa × média das duas densidades | 64,5 / 64,7 / 64,2 | 1 | 64,2% | 0,286 | 42,4% | 9 ms |
| U3 `gpu3 ft siglip2_b16_384x512 r1 destilado` | massa × densidade das frações | 63,0 / 63,2 / 62,6 | 1 | 62,6% | 0,303 | 41,1% | 15 ms |
| Professor (cinco redes; não concorre) | massa × densidade | 71,7 / 71,6 / 70,9 | | 71,7% | 0,197 | 53,0% | 1.564 ms |

Todas as rotas de cada candidato, com 8 orientações: U0 direta 67,9, massa × densidade 68,1, geométrica 67,9. U1 direta 68,0, massa × densidade da regressão 67,9, das frações 66,0, média das duas 68,4, geométrica 68,7. U4 61,2 / 61,3 / 63,6 / 64,5 / 62,8. U3 60,5 / 59,8 / 63,0 / 62,2 / 62,6.

- **Item 3.** O maior critério é o do U1 (68,3%; com 8 orientações, 68,7%). O U0 fica a 0,25 ponto (com 8 orientações, a 0,56): menos de 1,0 nas duas leituras. Entre os dois, o mais barato de servir é o U0 (8 ms contra 248 ms na T4). **C1 = U0, `gpu3 ft siglip2_b16_224 r2`, rota massa × densidade, 1 orientação.** A diferença pareada U1 − U0 na CV é +0,2 ponto (IC por cadeia −2,3 a +2,8): os dois são indistinguíveis.
- **Item 4.** Pares declarados: U1 + U4 67,9%; U1 + U3 67,6%; U3 + U4 65,4%. O melhor (U1 + U4) fica **0,2 ponto abaixo** do C1 (IC −2,4 a +2,1). O critério de 12.1 pede 2,0 pontos acima e limite inferior acima de zero. **Não há C2.**
- **Leituras do teste2 nesta seção: uma**, a do C1. U1, U3, U4, os pares e o professor não são pontuados no teste2.
- **O que isso quer dizer, sem enfeite:** o candidato escolhido é o controle. É a mesma rede e a mesma receita do modelo único da seção 11 (64,5% no teste2), treinada com 35% mais pratos. Nenhuma das novidades de 12.3 (destilação, cabeça de frações, peso por faixa na massa, 384×512) entrou no modelo que a regra escolheu. A expectativa registrada em 12.3 para o U0 no teste2 é de 65 a 67%.

**c) Leitura do teste2: uma, só do C1 (12/11/2026, 01:52).**

Antes, sem gabarito (`--confere`): 829 previsões na ordem de `aval2_ids.csv`, todas finitas; o modelo remontado dos arquivos de servir reproduz as previsões gravadas (diferença máxima em log de 0,005, em 8 pratos); a proteína prevista no teste2 tem a mesma distribuição das previsões fora-do-fold do treino (percentis 10/50/90: 0,8 / 10,1 / 39,9 g contra 0,8 / 10,5 / 39,5 g). A marca de leitura está gravada e o script se recusa a ler de novo.

**Veredito pela regra de 12.1: 70% NÃO DEMONSTRADO.** O C1 dá **66,0%** no teste2 (ponderado por faixa), com intervalo de 95% por cadeia de **61,6% a 70,1%**. O limite inferior teria de ser 70%; está 8 pontos abaixo. O limite superior é 70,1%: o intervalo fica quase todo abaixo da meta.

| Faixa de proteína real | Pratos | **C1: `gpu3 ft siglip2_b16_224 r2`, 1 orientação** | `r2` antigo (11.5; treino antigo, 8 orientações) | Nove redes (11.5) | Massa do C1: MAPE (viés mediano) | Massa das nove redes (11.5) |
|---|---|---|---|---|---|---|
| 0–5 g | 316 | 98,7% | 99% | 97% | 21,6% (−4%) | 23,4% (+4%) |
| 5–15 g | 169 | 82,2% | 81% | 82% | 15,8% (−3%) | 15,3% (+3%) |
| 15–25 g | 132 | 53,0% | 53% | 64% | 13,8% (−5%) | 13,9% (+4%) |
| 25–40 g | 110 | 52,7% | 47% | 51% | 14,2% (−6%) | 13,2% (+5%) |
| > 40 g | 102 | 43,1% | 42% | 55% | 15,1% (−11%) | 13,9% (+3%) |
| **Ponderado (IC por cadeia)** | 829 | **66,0%** (61,6 a 70,1) | 64,5% (60,1 a 68,8) | 69,7% (65,2 a 74,2) | 17,4% (−5%) | 17,7% (+4%) |
| Simples (não comparável com a meta) | 829 | 75,2% | 74,1% | 77,2% | | |
| > 15 g (três faixas) | 344 | 49,6% | 47,5% | 56,7% | | |
| > 25 g (duas faixas) | 212 | 47,9% | 44,7% | 52,9% | | |

- **Contra as nove redes (69,7%): −3,7 pontos** (diferença pareada, IC por cadeia −9,0 a +1,4). A perda está nos pratos de 15 a 25 g (53% contra 64%) e nos de mais de 40 g (43% contra 55%).
- **Contra o `r2` antigo (64,5%): +1,5 ponto** (IC −2,2 a +5,5); acima de 25 g, +3,2 (IC −4,2 a +10,5). A direção é a esperada com mais dados; a diferença não se distingue de zero. O modelo novo usa 1 orientação e o antigo usava 8.
- **Massa:** erro de 17,4%, igual ao das cinco redes de massa da seção 11 (17,7%). O viés é que muda de lado: o C1 subestima em todas as faixas (−3% a −6%), e mais nos pratos com mais de 40 g (−11%). A proteína acompanha: viés mediano de −7%, −8%, −12% e −14% nas quatro faixas acima de 5 g.
- **Os 107 pratos sem irmão no `teste` antigo:** 57,7% (24 cadeias; IC 48 a 77). Os 722 irmãos: 66,7% (IC 62,0 a 71,1). Como em 11.5, o grupo pequeno não decide nada sozinho.
- **Contra o que eu tinha escrito em 12.3:** U0 na CV, 66 a 68% (deu 68,0%); U0 no teste2, 65 a 67% (deu 66,0%); "um modelo só de 1 a 3 pontos abaixo das nove redes" (deu 3,7); professor na CV, 70 a 73% (deu 71,7%); U1, 67 a 70% (68,3%). **Errei nos destilados:** U4 e U3 deram 64,2% e 62,6% onde eu esperava 67 a 70%, e o melhor par ficou abaixo do melhor único, não 1 a 1,5 ponto acima.

**d) Por que a destilação piorou os ajustes finos.** Só com as previsões fora-do-fold e os registros de treino; nada do teste2.

| | U0 (receita `r2`, sem nada) | U4 (`r2` + frações + peso por faixa + destilação) | U3 (idem, 384×512, `r1`) |
|---|---|---|---|
| RMSE de log-massa na validação, **época 0** (cabeça = ridge sobre o backbone congelado; média dos 5 folds, 1 orientação) | 0,254 | 0,254 | 0,242 |
| Idem, época 5 | 0,277 | 0,345 | 0,327 |
| Idem, **última época** | **0,233** | **0,285** | **0,303** |
| Perda de treino: 1ª época / última | 0,12 / 0,021 | 6,4 / 0,71 a 0,93 | 6,4 / 1,35 a 1,52 |
| RMSE de log-massa por faixa de proteína (8 orientações) | 0,23 / 0,22 / 0,20 / 0,20 / 0,29 | 0,29 / 0,27 / 0,25 / 0,26 / 0,35 | 0,33 / 0,30 / 0,24 / 0,25 / 0,36 |
| Critério com a **densidade real** × massa prevista | 85,9% | 78,6% | 77,2% |
| Critério com a **massa real** × densidade prevista (regressão / frações / média das duas) | 75,9% | 73,6 / 73,4 / 77,1% | 69,1 / 73,5 / 74,2% |
| Distância da log-massa até a do professor | 0,142 | 0,199 | 0,228 |

1. **O que quebrou foi a massa, e em todo lugar.** Os dois alunos terminam com a massa **pior do que a do ponto de partida** (a cabeça ridge sobre o backbone ainda congelado), nos 5 folds, e pior que o U0 em todas as faixas de proteína. O U0 parte do mesmo ponto e melhora. A densidade quase não mudou: +1,2 ponto no U4, −1,7 no U3, com a massa real.
2. **Não achei erro nos alvos nem na conta da destilação.** Os alvos aninhados do professor estão alinhados e sãos (erro de 0,19 a 0,21 em log-massa contra o rótulo; correlação de 0,998 com os alvos fora-do-fold comuns; nenhum prato do fold k tem alvo no aluno treinado sem k). A destilação é feita no mesmo espaço padronizado das saídas, com a mesma média e o mesmo desvio. Na época 0 o U4 é idêntico ao U0.
3. **A causa mais provável é o equilíbrio das perdas, e o erro é da especificação de 12.3, que eu escrevi.** A perda dos rótulos é a média de Huber sobre 22 saídas: a massa pesa 1/22. As três perdas novas entraram por cima, em escala cheia: 10 × a soma do erro quadrático das 10 frações, o Huber da densidade composta e 0,5 × a destilação. O resultado está no registro: a perda de treino do U4 é **cerca de 40 vezes** a do U0 na última época e 53 vezes na primeira (a do U3, mais de 60 vezes). O termo dos rótulos, que no U0 é a perda inteira, fica na ordem de 2% do total. O treino corta o gradiente em norma 1,0, então quase todo passo vai para as frações. Os logits das frações partem de zero e aprendem com o passo pequeno do ajuste fino (6e-5 ou 3e-5; na cabeça sobre features congeladas o passo é 1e-3), o que mantém esse termo grande até o fim. Nas cabeças sobre features congeladas a mesma soma não faz esse estrago, porque não há backbone para desviar.
4. **Não é o aluno copiando os erros do professor.** Se fosse, o aluno ficaria mais perto do professor que o U0. Ficou mais longe (0,20 e 0,23 contra 0,14).
5. **O peso por faixa não parece ser a causa principal:** o erro de massa sobe por igual em todas as faixas, inclusive nas que ganharam peso. Mas não há ablação: U4 e U3 mudaram três coisas de uma vez (frações, peso por faixa, destilação), e os dados não separam uma da outra.
6. **Há também um efeito real da destilação, pequeno e negativo, medido onde a comparação é limpa.** O U1 é a cabeça `gpu3 fracoes so400m` com a destilação somada, no mesmo backbone congelado. Na CV: 68,7% contra 70,2% na rota geométrica, **−1,6 ponto** (IC por cadeia −3,3 a +0,0); na rota direta, −1,8 (IC −3,6 a −0,05). Com o Flavoria no treino a diferença se repete (67,2% contra 68,9%; os dois pares não são independentes). A massa piora de 0,232 para 0,237, e os pratos com mais de 40 g caem de 41% para 37%. A explicação que os números sustentam: o alvo do professor é uma cópia do rótulo com ruído e encolhida (nos alvos fora-do-fold comuns, desvio-padrão de 95% do real na massa, 83% na densidade e 90% na proteína), e os alvos aninhados vêm de professores treinados com só 60% dos pratos; e, pelo mesmo 1/22, ele pesa 3,7 vezes mais que o rótulo verdadeiro na massa e na proteína. Sem fotos novas sem rótulo, o professor não mostra ao aluno nada que os rótulos já não mostrem.

**Resumo do diagnóstico:** nos ajustes finos, problema de implementação da receita (pesos das perdas, definidos em 12.3 e executados como escritos), não prova contra a cabeça de frações nem contra o peso por faixa. Na destilação em si, um efeito real de cerca de −1,5 ponto do jeito que foi especificada.

**e) O que os dados a mais renderam, e o que custa ter um modelo só.**

- **Efeito dos dados, mesma receita, mesmos pratos.** Nos 1.684 pratos do treino antigo, previsões fora-do-fold do `r2` da segunda execução (treinado com o treino antigo) contra as do U0 (treinado com o treino_v2): RMSE de log-massa de 0,247 para 0,237; critério de CV de 65,0% para 68,1% na rota massa × densidade, **+3,1 pontos** (IC por cadeia +0,9 a +5,6); +3,0 na direta (IC +0,7 a +5,3) e +2,2 na geométrica (IC +0,0 a +4,4). Acima de 25 g, de 43–46% para 48–49%. É mais que o 1 a 2 pontos que eu esperava em 12.2. Ressalvas: os folds são outros, e a variação entre duas rodadas do mesmo ajuste fino não foi medida.
- **Nos backbones congelados o ganho é menor:** ridge So400m de 0,2190 para 0,2106 nos mesmos pratos; média das cinco massas de 0,2051 para 0,2013. Quem mais ganha com dados é o ajuste fino.
- **No teste2** o mesmo par dá +1,5 ponto (IC −2,2 a +5,5), item (c).
- **Um modelo só contra o conjunto.** Na CV: C1 68,0% contra o professor 71,7%, **−3,7 pontos** (IC −6,3 a −0,9); massa 0,234 contra 0,197. No teste2: 66,0% contra 69,7% das nove redes, −3,7 (IC −9,0 a +1,4). O preço de servir cai de 1,7–2,0 s para 8 ms (item (g)).
- **Achado exploratório, só de CV, feito depois da leitura e fora da regra.** O melhor modelo de um backbone só nesta execução não era candidato. Com o **So400m congelado e duas cabeças que já foram treinadas** (a ridge para a massa e a cabeça de frações sem destilação para a densidade, na média em log das duas densidades dela), a CV dá **71,3% com 1 orientação** (61 ms na T4) e 72,0% com 8 (532 ms): empata com o professor de cinco redes (−0,4, IC −2,5 a +1,6, com 1 orientação) e fica +3,2 pontos acima do C1 (IC +0,4 a +6,1). A média dele com o U0 dá 72,6% (+4,6 sobre o C1, IC +2,7 a +6,5; 70 ms na T4). O U1 declarado em 12.3 perdeu isso por dois motivos: usa a massa da própria cabeça MLP (0,232 a 0,237) em vez da ridge sobre as mesmas features (0,209), e leva a destilação. **Isso não foi lido no teste2 e não pode ser, pela regra em vigor:** não era candidato, e foi achado olhando umas 17 variantes na CV, o que por si só infla o número em algo como meio ponto a um ponto. O que ele diz com segurança é de onde vem a vantagem do professor: massa por ridge num backbone grande congelado mais densidade pela cabeça de frações, e não o número de redes.

**f) Flavoria: não adotado.**

- **No domínio (CV do treino_v2, professor com e sem a base):** 71,7% → 70,8%, **−0,95 ponto** (IC por cadeia −2,3 a +0,5); acima de 25 g, 53,0% → 50,3% (−2,7; IC −5,5 a +0,1); massa de 0,1968 para 0,1996. A regra de 12.5 pedia +1,0, ou no máximo −0,3 com +2,0 acima de 25 g. Os cinco ridge pioram um pouco na massa (o So400m, de 0,2090 para 0,2108), duas das três cabeças de frações caem de 1 a 1,5 ponto, e o U1 cai de 68,7% para 67,2%. A faixa que mais perde é a de mais de 40 g (53,5% → 47,9% no professor), justo a que se queria ajudar. Eu esperava entre −0,5 e +1; ficou um pouco abaixo disso. Por causa da regra, os dois ajustes finos com Flavoria não rodaram: o resultado vale para modelos sobre features congeladas.
- **Fora do domínio (`teste_ext` do Flavoria: 227 pratos, 46 com mais de 600 g; só massa, não é o critério do produto e não é leitura do teste2).** Treinaram 1.493 fotos de 1.001 pratos.

| Modelo | Sem o Flavoria no treino: MAPE (viés mediano) | Idem, corrigido por uma constante | Com o Flavoria no treino | Acima de 600 g: sem / constante / com |
|---|---|---|---|---|
| Professor (cinco redes) | 21,8% (−15%) | 19,6% (−4%) | **18,7%** (−4%) | 29,7% / 20,4% / **18,4%** (viés −29% / −20% / −17%) |
| U1 | 22,9% (−14%) | 21,2% | 21,5% (−6%) | 33,5% / 25,5% / 27,3% |
| U0 = C1 | 27,8% (+4%) | 26,2% | não treinado | 19,5% (viés −14%) / 22,5% |
| U3 | 26,7% (+5%) | 24,8% | não treinado | 21,3% / 24,6% |
| U4 | 33,9% (+2%) | 33,2% | não treinado | 24,0% / 24,7% |

  - Os modelos que nunca viram o Flavoria já erram só 22% nesses pratos (terceiros relatam 45%; são outros modelos). Treinar com a base leva o professor a 18,7%: 3 pontos de ganho, dos quais 2 vêm de uma constante de escala. Nos pratos com mais de 600 g o ganho é maior (29,7% → 18,4%), e mesmo aí a constante sozinha já chega a 20,4%. O viés nesses pratos continua em −17%.
  - **Leitura:** do jeito que entrou (só massa, teto de peso de metade, outro enquadramento, viés por domínio), o Flavoria ensinou quase só uma constante de escala, e custou cerca de 1 ponto no Nutrition5k. Eu esperava uma queda grande do erro fora do domínio; foi pequena.
  - O C1 erra 27,8% no Flavoria, contra 21,8% do professor: o modelo único com ajuste fino é mais frágil fora do Nutrition5k que os backbones congelados.

**g) Custo de servir do C1** (`servico.json` da terceira execução; mesmas condições de 11.6).

| | C1: `gpu3 ft siglip2_b16_224 r2`, 1 orientação | Nove redes (11.6), 8 orientações |
|---|---|---|
| Parâmetros | 92,9 M | 1.804 M |
| Pesos em memória (fp16) | 186 MB (um arquivo de 186 MB) | 3,61 GB |
| T4, uma foto | **8,3 ms** (3,1 ms por foto em lotes de 8) | 1.711 a 1.990 ms |
| Pico de memória de GPU | 381 MB (436 MB em lotes de 8) | 1,9 GB (maior modelo, um por vez) |
| CPU x86 (Xeon 2,0 GHz, 4 threads) | 0,24 s | 108 s |
| Conferência do modelo remontado (dif. máx. em log) | 0,005 (0,0006 na log-massa) | |

A regra escolheu 1 orientação: na CV, 68,0% contra 68,1% com 8. O tempo por foto é mais de 200 vezes menor que o das nove redes, e a memória, 19 vezes menor. **Não medido:** nada no iPhone (conversão para Core ML, latência, memória), decodificação da imagem, vazão com várias fotos.

**h) O que foi aprendido para o produto, e o que eu rodaria agora.**

1. **Um modelo só, do jeito que 12.3 declarou, dá 66% no teste2, 4 pontos abaixo das nove redes** (não distinguível com 190 cadeias, e igual à diferença de CV, que é segura). A meta não foi demonstrada e este modelo está abaixo dela. Os pratos com mais de 25 g ficam em 48%.
2. **Mais dados do mesmo tipo funcionam.** 35% a mais de pratos (46% a mais acima de 25 g) deram +2 a +3 pontos de CV à mesma receita e +1,5 (incerto) no teste2. É a única alavanca desta execução com efeito positivo medido. O Nutrition5k acabou: o treino_v2 já usa tudo que não toca o teste2.
3. **O Flavoria, como entrou, não ajuda no Nutrition5k** (−1 ponto) e fora dele ensina quase só uma constante. Não vale repetir igual. Se voltar, é com o prato enquadrado do mesmo jeito nos dois domínios (o guia do app), que 12.5 deixou para depois.
4. **A destilação sem fotos novas não ajuda** (−1,5 ponto onde a comparação é limpa). Só volta a fazer sentido com fotos sem rótulo que não sejam do teste2.
5. **A cabeça de frações e o peso por faixa em ajuste fino continuam sem teste válido:** os pesos das perdas estragaram a massa antes que desse para ver qualquer coisa.

O que eu rodaria, em ordem:

1. **Sem GPU, se o dono autorizar por escrito antes (autorizado e feito: 12.7):** pré-registrar como candidato o So400m congelado com as duas cabeças (e, como segundo candidato, a média dele com o U0) e ler no teste2. As previsões já estão gravadas. Isso é uma **emenda** a 12.1: as duas leituras de lá eram do C1 e do C2 desta execução, e a do C2 não foi usada porque nenhum par passou. Expectativa honesta: entre 68,5% e 71% no teste2 (a CV de 71,3% a 72,6%, menos os 1 a 2 pontos que a CV perdeu para o teste2 nos dois casos medidos, menos o otimismo de ter escolhido depois de olhar). **Não demonstraria a meta,** que pede cerca de 74,5%.
2. **Na próxima execução em GPU, uma mudança por vez sobre o U0** (61 min cada a 224 px): (a) cabeça de frações com as perdas novas normalizadas para o termo dos rótulos manter pelo menos metade do gradiente, logits iniciados da cabeça congelada e passo próprio para a cabeça; (b) o mesmo com peso por faixa; sem destilação. Critério de aceitação no próprio registro: a massa na última época não pode ser pior que na época 0.
3. **Ajuste fino do So400m a 384×512** (parcial, últimos blocos; cerca de 3 h; completo, cerca de 8 h), partindo da cabeça ridge. É o backbone que dá a melhor massa congelada (0,209) e a melhor densidade (79,0% com a massa real, contra 75,9% do U0), e o ajuste fino foi o que mais ganhou com dados.
4. **Dados e teste:** mais pratos pesados com mais de 25 g, de cima, e um conjunto de teste maior e independente. Com 190 cadeias, um modelo de 72% de verdade falharia na regra em cerca de 85% das vezes.

**Registro de método.** (a) O item (a) e a aplicação da regra (item (b)) foram escritos neste documento antes de qualquer número do teste2 dos modelos novos. (b) O teste2 foi lido **uma vez** nesta seção, para um candidato (C1); é a segunda leitura do teste2 no estudo (a primeira é a de 11.5). A leitura que 12.1 reservava ao C2 não foi usada. (c) Antes da leitura rodaram, sem gabarito do teste2: `--so-cv` e `--confere`. (d) Depois da leitura: `--relato` abriu os resultados por prato da seção 11 só para calcular as diferenças pareadas e o grupo dos 107 pratos (a primeira versão do script marcava esse grupo pela coluna errada e não o imprimiu; a correção não muda nenhum número já lido); e rodaram as análises só de CV dos itens (d), (e) e (f). (e) U1, U3, U4, os pares, o professor, as cabeças de frações e as combinações exploratórias **não** foram pontuados no teste2; as previsões deles para o teste2 existem nos arquivos e não foram abertas. (f) O `teste` antigo não foi lido. (g) A conferência de paridade que a execução não conseguiu fazer ("sem referência da execução 2") foi feita aqui: a réplica dos cinco ridge no treino antigo dá os mesmos RMSE da segunda execução (0,2190, 0,2265, 0,2283, 0,2426, 0,2349).

### 12.7 Emenda a 12.1: leitura do So400m congelado com duas cabeças (12/11/2026)

**Escrito antes de abrir qualquer número do teste2 deste candidato.**

**a) O que muda em 12.1, e quem aprovou.** 12.1 permitia duas leituras do teste2 depois da seção 11: a do C1 e a do C2 da terceira execução. A do C1 foi feita (12.6); a do C2 não, porque nenhum par passou. O item (h) de 12.6 propôs ler um candidato que não estava na regra, com a condição de o dono autorizar por escrito antes. **O fundador aprovou em 12/11/2026**; a aprovação me foi repassada pelo coordenador do trabalho, e é essa mensagem que eu registro aqui. A emenda vale só para o que está escrito abaixo. O resto de 12.1 continua igual: a medida, a regra de sucesso e o que não pode entrar em nenhuma escolha.

**b) Como o candidato foi achado, sem enfeite.** Ele **não** estava em 12.3. Apareceu depois da leitura do C1, numa análise só de validação cruzada (12.6, item (e)), entre umas 17 variantes olhadas. Duas consequências:

- a CV dele está otimista, por ter sido o melhor de vários: algo como 0,5 a 1 ponto;
- a decisão de fazer mais uma leitura veio **depois** de ver que o C1 deu 66,0%. A escolha do candidato não usou o teste2, mas a de ler de novo, sim. Isso é uma forma de parada opcional e pesa na multiplicidade (item (f)).

**c) Definição congelada.**

- **D1, o candidato lido.** Um backbone, o SigLIP 2 So400m/16 a 384×512, **congelado** (checkpoint público), com duas cabeças treinadas na terceira execução com o treino_v2 inteiro:
  - massa = `exp` da log-massa da cabeça ridge (`gpu3 ridge siglip2_so400m16_384x512 o8`);
  - densidade = média em log (piso 0,01) das duas densidades da cabeça de frações **sem destilação** (`gpu3 fracoes so400m`): a da regressão e a composta pelas frações por fonte;
  - proteína = massa × densidade. **Rota única, fixada aqui:** é a que 12.6 descreveu. A variante "média geométrica com a proteína direta da cabeça" dá 0,1 ponto a mais na CV com 8 orientações e fica de fora.
  - **Orientações: 1** (a foto original). Sai da mesma regra do item 2 de 12.3, o menor número entre 1, 4 e 8 a até 1,0 ponto do de 8: a CV dá 72,0% / 71,4% / 71,3% com 8 / 4 / 1.
  - CV no treino_v2: **71,3%** (IC por cadeia 68,8 a 73,8); por faixa 98 / 86 / 69 / 51 / 53; acima de 25 g, 51,9%; RMSE de log-massa 0,217. Contra o C1: +3,2 pontos (IC +0,4 a +6,0). Contra o professor de cinco redes (71,7%): −0,4 (IC −2,4 a +1,7).
- **D2, a média do D1 com o U0** (média em log1p da proteína do D1 e da do C1 de 12.6, cada um com 1 orientação): CV 72,6% (IC 70,1 a 75,2). **Critério para valer uma leitura, o mesmo que 12.1 já tinha para um segundo modelo:** CV pelo menos 2,0 pontos acima do modelo único (aqui, o D1) **e** limite inferior de 95% da diferença pareada acima de zero. Uso o critério antigo de propósito: é o único escrito antes de eu conhecer estes números, e qualquer limiar inventado agora seria escolhido sabendo o resultado. A diferença D2 − D1 na CV é **+1,3 ponto (IC −0,8 a +3,4)**: não passa em nenhuma das duas condições. **O D2 não é lido.**

**d) Leituras desta emenda: uma, a do D1.** Na mesma execução saem, só como relato pareado, as diferenças para os pipelines já lidos (C1 de 12.6; nove redes e híbrido de 11.5), com os resultados por prato já gravados. Nenhum outro modelo é pontuado.

**e) Regra, sem mudança.** Meta demonstrada só se o **limite inferior do intervalo de 95%** (bootstrap de cadeias, critério ponderado por faixa) for **70% ou mais**. O veredito sai dessa conta e de mais nada.

**f) Multiplicidade.** Esta é a **terceira leitura do teste2 no estudo** (11.5, 12.6 e esta), e a segunda depois de 12.1. Por isso o intervalo de 97,5% sai ao lado do de 95% (e o de 98,3%, que é a correção para três leituras); "passou com folga" só se o limite inferior do intervalo mais largo também ficar em 70% ou mais. **O teste2 está perto de gasto como teste de registro:** três pipelines já foram pontuados nele, o último escolhido depois de se ver o resultado do segundo. Os resultados por prato de três leituras existem em disco. Daqui em diante ele serve para comparar modelos de forma pareada; para **demonstrar** a meta, o próximo número precisa de um conjunto novo.

**g) O que eu espero.** Entre **68,5% e 71%** no teste2, como escrito em 12.6; o palpite central é 69,5%. Conta: CV de 71,3%, menos os 1 a 2 pontos que a CV perdeu para o teste2 nos dois casos medidos (C1: 68,0% → 66,0%; nove redes: 70,5% → 69,7%), menos o otimismo da escolha. Com intervalo de ±4,5 pontos, o limite inferior deve ficar perto de 65%. **Dou menos de 5% de chance de a meta ser demonstrada.** O que a leitura pode responder é outra coisa: se um backbone só chega ao nível das nove redes (69,7%) e quanto fica acima do C1 (66,0%).

**h) Antes da leitura, sem gabarito:** previsões das duas cabeças completas, finitas e na ordem de `aval2_ids.csv`; conferência de remontagem da cabeça ridge gravada na execução. **Ressalva conhecida de antemão:** a cabeça de frações não teve medida de serviço na execução, então não há conferência de remontagem dela (o arquivo da cabeça existe; a do U1, que tem a mesma arquitetura sobre o mesmo backbone, conferiu com diferença de 0,0001).

**i) Resultado da leitura (12/11/2026, 16:10; uma leitura, só do D1).**

Conferências sem gabarito, antes: 829 previsões das duas cabeças, finitas, na ordem de `aval2_ids.csv`; cabeça ridge remontada com diferença máxima em log de 0,0003; proteína prevista com percentis 10/50/90 de 1,3 / 11,1 / 42,6 g no teste2 contra 1,2 / 10,8 / 41,2 g fora-do-fold. A marca de leitura está gravada e o script se recusa a ler de novo.

**Veredito pela regra: 70% NÃO DEMONSTRADO.** O D1 dá **69,7%** no teste2 (ponderado por faixa), com intervalo de 95% por cadeia de **65,0% a 74,3%** (97,5%: 64,5 a 75,0; 98,3%: 64,1 a 75,4). O limite inferior teria de ser 70%. O resultado é compatível com 70%, sem prova, igual ao das nove redes na seção 11.

| Faixa de proteína real | Pratos | **D1: So400m congelado, duas cabeças, 1 orientação** | C1 (12.6) | Nove redes (11.5) | D1 na CV do treino_v2 | Professor na CV (não lido) | Massa do D1: MAPE (viés mediano) |
|---|---|---|---|---|---|---|---|
| 0–5 g | 316 | 98,1% | 98,7% | 97% | 98% | 98% | 24,9% (+4%) |
| 5–15 g | 169 | 78,7% | 82,2% | 82% | 86% | 88% | 17,7% (+4%) |
| 15–25 g | 132 | 64,4% | 53,0% | 64% | 69% | 66% | 16,7% (+3%) |
| 25–40 g | 110 | 55,5% | 52,7% | 51% | 51% | 52% | 12,9% (+3%) |
| > 40 g | 102 | 52,0% | 43,1% | 55% | 53% | 54% | 13,8% (−3%) |
| **Ponderado (IC por cadeia)** | 829 | **69,7%** (65,0 a 74,3) | 66,0% (61,6 a 70,1) | 69,7% (65,2 a 74,2) | 71,3% (68,8 a 73,8) | 71,7% | 19,2% (+3%) |
| Simples (não comparável com a meta) | 829 | 77,4% | 75,2% | 77,2% | | | |
| > 15 g (três faixas) | 344 | 57,3% | 49,6% | 56,7% | | | |
| > 25 g (duas faixas) | 212 | 53,7% | 47,9% | 52,9% | 51,9% | 53,0% | |

- **Contra as nove redes (69,7%): 0,0 ponto** (diferença pareada, IC por cadeia −4,7 a +4,3); acima de 25 g, +0,8 (IC −8,5 a +9,3). **Um backbone só repete o nível do conjunto de nove redes no teste2.**
- **Contra o C1 (66,0%): +3,7 pontos** (IC −1,8 a +9,5); acima de 25 g, +5,8 (IC −4,8 a +17,1). No teste2 sozinho a diferença não se distingue de zero; na CV ela é do mesmo tamanho e é segura (+3,2; IC +0,4 a +6,0). O ganho está nos pratos de 15 a 25 g (64% contra 53%) e nos de mais de 40 g (52% contra 43%). Nos de 5 a 15 g o D1 perde 3,5 pontos.
- **Contra o híbrido com LLM v3 da seção 11 (70,8%): −1,1** (IC −5,9 a +3,6).
- **Da CV para o teste2: 71,3% → 69,7%**, queda de 1,6 ponto, dentro do que eu tinha escrito (68,5 a 71%; palpite de 69,5%).
- **Massa:** erro de 19,2% (viés mediano +3%), **pior** que o das cinco redes da seção 11 (17,7%) e que o do C1 (17,4%); RMSE em log de 0,243, contra 0,217 na CV. É uma cabeça ridge só, com uma orientação. O que segura o resultado é a densidade: a proteína sai quase sem viés (mediana de +5%, +3%, 0% e −7% nas quatro faixas acima de 5 g), enquanto o C1 subestimava de 7% a 14%.
- **Os 107 pratos sem irmão no `teste` antigo:** 76,1% (24 cadeias; IC 67 a 86). Os 722 irmãos: 69,1% (IC 64,0 a 74,2). O grupo pequeno não decide nada sozinho; no C1 ele tinha dado 57,7%.
- **Pratos com mais de 25 g: 53,7%.** Continuam no mesmo patamar de todos os pipelines deste estudo (52 a 54%).

**j) Custo de servir do D1** (`servico.json` e os arquivos de serviço da terceira execução; mesmas condições de 11.6).

| | D1: So400m congelado + duas cabeças, 1 orientação | C1 (12.6) | Nove redes (11.6), 8 orientações |
|---|---|---|---|
| Parâmetros | 429,7 M (428,5 M do backbone; 1,2 M das duas cabeças) | 92,9 M | 1.804 M |
| Pesos em memória (fp16) | 859 MB | 186 MB | 3,61 GB |
| T4, uma foto | **61 ms** (56 ms por foto em lotes de 8) | 8,3 ms | 1.711 a 1.990 ms |
| Pico de memória de GPU | 1,9 GB | 381 MB | 1,9 GB (maior modelo, um por vez) |
| CPU x86 (Xeon 2,0 GHz, 4 threads) | 3,8 s | 0,24 s | 108 s |
| teste2, ponderado | 69,7% | 66,0% | 69,7% |

- Os tempos são os medidos para o mesmo backbone com cada cabeça em separado (61,7 ms com a ridge; 61,4 ms com uma MLP igual à de frações). O backbone roda uma vez e as duas cabeças somam 1,2 M de parâmetros, então o tempo do D1 é esse. **A combinação das duas cabeças não foi cronometrada como um pipeline.**
- **É um modelo de servidor:** cerca de 30 vezes mais rápido que as nove redes na T4 e 4 vezes menor, mas 7 vezes mais lento e 4,6 vezes maior que o C1. Com 859 MB e 3,8 s por foto numa CPU de servidor, não vai para o aparelho como está.
- **O que falta para servir de verdade:**
  - **Nada precisa ser treinado de novo.** O backbone é o checkpoint público (`siglip2_so400m16_512.npz`, 4,55 GB para baixar, com a torre de texto e em precisão simples). As duas cabeças finais, treinadas com o treino_v2 inteiro, estão gravadas na saída da execução: `cabeca__gpu3_ridge_siglip2_so400m16_384x512_o8.npz` (0,4 MB) e `cabeca__gpu3_fracoes_so400m.npz` (4,8 MB, com as densidades por fonte). **Ficam fora do repositório;** precisam ser guardadas num lugar que não se perca com o scratchpad.
  - **Falta conferir a remontagem da cabeça de frações.** A execução só conferiu a ridge (0,0003) e a cabeça do U1, de mesma arquitetura (0,0001).
  - **Falta o código de serviço que junta as duas cabeças** sobre as mesmas features (massa da ridge × média em log das duas densidades da cabeça de frações) e a medida de latência desse conjunto.
  - Não medido: vazão com várias fotos, decodificação da imagem, custo em dinheiro por foto, nada em aparelho.

**k) O que a leitura muda.**

1. **O conjunto de nove redes pode ser trocado por um backbone só, sem perda medida:** 69,7% contra 69,7% no teste2, a 61 ms em vez de 1,7 a 2,0 s. A vantagem do conjunto vinha de massa por ridge num backbone grande congelado e densidade por frações, não do número de redes.
2. **A meta continua não demonstrada.** Três pipelines diferentes lidos no teste2 dão 69,7%, 66,0% e 69,7%; nenhum tem limite inferior acima de 66%.
3. **Há dois modelos únicos na mesa, com papéis diferentes:** o de aparelho (C1: 66%, 186 MB, 8 ms na T4, 0,24 s em CPU) e o de servidor (D1: 70%, 859 MB, 61 ms na T4). Os 3,7 pontos entre eles são o preço de rodar no aparelho hoje.
4. **O próximo passo de modelo fica mais claro:** o ajuste fino parcial do So400m (item 3 de 12.6 (h)) parte agora de um modelo que já vale 70%, e a ablação da cabeça de frações no ajuste fino pequeno (item 2) é o que pode fechar a distância do modelo de aparelho.
5. **O teste2 está gasto para demonstração.** Foram três leituras, a última decidida depois de ver a segunda. Ele ainda serve para comparações pareadas entre modelos; a prova da meta precisa de um conjunto novo e maior.

**Registro de método da emenda.** (a) Os itens (a) a (h) foram escritos neste documento antes da leitura; o `--so-cv` que fixou as orientações e o critério do D2 rodou antes, só com previsões fora-do-fold. (b) O teste2 foi lido **uma vez** nesta emenda, para um candidato (D1); é a terceira leitura do estudo. O D2 não foi lido. (c) Na mesma execução foram abertos os resultados por prato já gravados das leituras anteriores (C1; nove redes e híbrido da seção 11), só para as diferenças pareadas. (d) A aprovação do fundador é a que o coordenador me repassou em 12/11/2026; eu não a vi diretamente. (e) Nenhum outro modelo da terceira execução foi pontuado no teste2. (f) O `teste` antigo não foi lido.

### 12.8 Calorias, carboidratos e gordura (validação cruzada) (12/11/2026)

**Pergunta da fundadora:** dá para contar carboidratos e calorias também, e não só proteína? As cabeças já preveem 22 saídas, entre elas kcal, gordura e carboidrato, mas nada disso tinha sido medido. **Esta subseção usa só as previsões fora-do-fold da terceira execução no treino_v2** (2.281 pratos, 646 cadeias). **Não lê o teste2**, que fica reservado para a leitura única da seção 13. Script: `eval/treino/macros_cv.py`.

**Critério, fixado antes de calcular qualquer acerto (só olhei a distribuição dos rótulos).** É o mesmo formato do de proteína: acerto se o erro for de até 20% do valor real **ou** de até um piso absoluto; média das cinco faixas do próprio nutriente com o mesmo peso; intervalo por bootstrap de cadeias.

| Nutriente | Faixas (pratos no treino_v2) | Piso | Por que esse piso |
|---|---|---|---|
| kcal | 0–100 (654), 100–250 (642), 250–400 (487), 400–600 (323), > 600 (175) | ± 50 kcal | 20% de 250 kcal, uma refeição pequena; 50 kcal é um pedaço de pão |
| Carboidrato | 0–5 (446), 5–15 (679), 15–30 (658), 30–50 (359), > 50 g (139) | ± 10 g | Uma "troca" de carboidrato na contagem europeia (10 g; a americana usa 15 g). Abaixo disso a contagem não muda |
| Gordura | 0–2 (564), 2–10 (619), 10–20 (555), 20–35 (373), > 35 g (170) | ± 5 g | O mesmo piso em gramas da proteína (cerca de 45 kcal) |

Ao lado saem dois controles: o mesmo critério com um piso estrito (25 kcal, 5 g, 3 g) e só com os 20%, sem piso.

**Rotas.**

- **D1** tem cinco rotas para cada nutriente: a direta da cabeça ridge; a direta da cabeça de frações; a massa da ridge vezes a densidade implícita na cabeça de frações (nutriente ÷ massa da própria cabeça); a média em log das duas diretas; e a média geométrica da direta da ridge com a anterior. A rota sai da CV com **1 orientação**, a do D1:
  - kcal e carboidrato: **média em log das duas diretas**;
  - gordura: **geométrica**.
  - Escolher entre cinco na mesma CV deixa o número uns 0,5 ponto otimista.
  - A cabeça de frações não tem densidade por fonte para esses nutrientes: a densidade composta só existe para proteína.
- **U0** (o C1 de 12.6, aparelho) tem só a direta: a densidade implícita na própria massa dá o mesmo número.

**Resultado (critério ponderado, IC por cadeia).**

| | D1 (servidor, 61 ms na T4) | U0 = C1 (aparelho, 8 ms) | D1 − U0 | D1 por faixa (da menor para a maior) | D1: viés mediano por faixa | Piso estrito / só 20% (D1) |
|---|---|---|---|---|---|---|
| **kcal** | **70,5%** (67,9 a 73,1) | 65,0% (62,2 a 67,8) | +5,5 (+2,2 a +8,7) | 94 / 71 / 63 / 66 / **58** | +10 / +1 / 0 / −8 / **−17%** | 63,9 / 57,0 |
| **Carboidrato** | **78,6%** (76,0 a 81,0) | 75,0% (72,2 a 77,7) | +3,6 (+0,8 a +6,4) | 99 / 96 / 89 / **63** / **46** | +37 / +5 / −6 / −14 / **−20%** | 67,3 / 44,7 |
| **Gordura** | **68,4%** (66,0 a 70,7) | 65,6% (63,2 a 68,1) | +2,8 (+0,3 a +5,3) | 97 / 88 / 70 / **46** / **40** | +68 / +6 / −8 / −15 / **−20%** | 60,1 / 35,0 |
| Proteína, para comparar (rotas de 12.7) | 71,3% | 68,0% | +3,2 | 98 / 86 / 69 / 51 / 53 | | |
| Massa, ±20% sem piso | 70,3% | 69,9% | | MAPE 16,1% e 16,4%; viés −0,4% e −5,0% | | |

- **Erro médio absoluto (D1):** 54 kcal, 5,3 g de carboidrato e 3,9 g de gordura por prato. O MAPE (27%, 36% e 96%) não serve para carboidrato e gordura, porque muitos pratos têm quase zero: um prato com 0,5 g de gordura estimado em 2 g conta 300%.
- **Com as mesmas rotas genéricas,** a proteína do D1 daria 70,1%, abaixo dos 71,3% da rota com a densidade composta pelas frações. A cabeça de frações foi feita para a proteína; para os outros nutrientes ela entra só como uma segunda regressão.
- **O padrão de erro é o mesmo da proteína: regressão à média.** Superestima os pratos leves e subestima os grandes. Na faixa mais alta de cada nutriente, a estimativa fica cerca de 20% abaixo: −17% nas refeições acima de 600 kcal, −20% acima de 50 g de carboidrato e acima de 35 g de gordura. O erro de kcal anda junto com o de massa (correlação de 0,56 em log); o de carboidrato, menos (0,40).

**Conclusão.**

- **kcal, no servidor: no nível da proteína.** O D1 acerta 70,5% na validação cruzada do Nutrition5k com o mesmo tipo de critério, e o modelo de aparelho, 65%.
- **Carboidrato: número maior, mas que não deve ser lido como "melhor que a proteína".** 79% no servidor e 75% no aparelho. Sai mais alto porque o piso de 10 g cobre quase todos os pratos com até 30 g de carboidrato. Nas refeições que importam para quem conta carboidrato (30 g ou mais), o acerto é de 63% e 46%, com a estimativa 14% a 20% abaixo. Sem piso, só com os 20%, fica em 45%.
- **Gordura** fica perto do kcal (68%), fraca acima de 20 g.

**Ressalvas** (as mesmas da proteína, e mais algumas):
1. É validação cruzada, não teste. Na proteína, o D1 perdeu 1,6 ponto da CV para o teste2.
2. É um refeitório americano. Comida brasileira tem muito arroz, feijão, farinha e fritura, e nada disso foi medido.
3. Os pisos são escolha minha.
4. A subestimação dos pratos grandes é sistemática.

**O app pode mostrar kcal e carboidrato hoje como estimativa,** com faixa ou arredondamento ("cerca de 450 kcal", "40 a 55 g de carboidrato"), no mesmo nível de confiança da proteína. **Não pode** apresentar o carboidrato como preciso para decidir dose de insulina, e o texto do app precisa dizer isso.

## 13. Mais bases de treino (quarta execução) (12/11/2026)

**Desenho e pré-registro. Nada aqui tem resultado da quarta execução, e nada aqui leu o teste2.** Decisões da fundadora de 12/11/2026 (`docs/produto.md`, seção 7): 70% é o mínimo; e o treino deixa de estar restrito a bases de licença comercial ("enriquecer o modelo, qualquer que seja a base"), com a procedência registrada por base (colunas `licenca` e `uso_declarado` em cada lista), para que um modelo só com dados liberados possa ser refeito. Base atrás de controle de acesso não é contornada. As bases e os problemas de rótulo de cada uma estão em `fontes_treino.md` (2.1, 10 e 11).

### 13.1 O que a terceira execução deixou, e o que isso muda no desenho

- O ponto de partida é o **D1** (12.7): So400m congelado, cabeça ridge para a massa e cabeça de frações para a densidade, uma orientação. 71,3% na CV do treino_v2 e 69,7% no teste2.
- **Um número novo, calculado agora só com as previsões fora-do-fold da terceira execução:** as duas cabeças do D1 treinadas **com o Flavoria**, do jeito de 12.5, dão 69,2% na CV do Nutrition5k com uma orientação, contra 71,3% sem ele: **−2,1 pontos (IC por cadeia −3,7 a −0,4)**; acima de 25 g, 51,9% → 48,0% (−4,0; IC −7,4 a −0,5). Com 8 orientações, −0,95 (IC −2,6 a +0,7). Juntar uma base externa nas mesmas cabeças **custa** no Nutrition5k, e o custo está mais na cabeça de frações (70,2% → 68,9% sozinha) que na massa (0,2090 → 0,2108).
- Por isso o desenho separa as duas cabeças. **Bases que só têm massa medida entram só na cabeça de massa.** A cabeça de densidade só recebe a base que tem proteína (OmniFood8K). Como a proteína do D1 é massa × densidade, qualquer cabeça de massa combina com qualquer cabeça de densidade depois, na CPU, sem treinar de novo.
- A destilação saiu. O ajuste fino com cabeça de frações volta com a receita corrigida (13.7), uma mudança por vez.

### 13.2 Objetivo, dito com honestidade

O produto é foto de celular de refeição de verdade, no Brasil. **Não existe nenhum conjunto pesado desse domínio.** O que dá para medir são substitutos:

| Componente | Onde | O que mede | Rótulo |
|---|---|---|---|
| **N** | Nutrition5k, CV do treino_v2 por cadeia | Critério do produto (proteína a ±20% ou ±5 g), ponderado nas cinco faixas | Pesado, ingrediente a ingrediente |
| **O** | OmniFood8K, CV por prato (13.3) | O mesmo critério, em quatro faixas (0–5, 5–15, 15–25, > 25 g) com o mesmo peso, em cenas cujo prato principal o modelo nunca viu | Pesado × tabela, recalculado; fração comestível suposta em osso e casca |
| **M** | Flavoria, FPB (fotos de cima) e Mensa, CV por dia, por grupo de porção e por participante | % dos pratos com a **massa** a ±20%, média dos três domínios | Massa pesada |

- O OmniFood8K tem só 32 cenas acima de 40 g; por isso as duas faixas de cima viram uma.
- O LeFood (um item pequeno por prato, no máximo 343 g) é relatado à parte e não entra em **M**.
- Em **O** e **M** a massa prevista leva uma **constante por domínio** (a mediana de log real − log previsto, estimada nos grupos de treino daquele domínio nos outros folds), igual para todas as configurações, treinem ou não com o domínio. Sem ela a medida seria só a diferença de enquadramento entre as câmeras. A previsão crua sai ao lado. **A densidade não leva correção nenhuma.**
- **O objetivo desta execução: subir O e M sem derrubar N.** N é onde a meta de 70% é medida hoje; O e M dizem se o modelo aguenta outra cozinha, outro prato e outra câmera.

### 13.3 Divisões das bases externas, fixadas agora

`eval/treino/externo_r4_divide.py` grava uma tabela por base (`divisao_r4_<dominio>.csv`: papel, fold, folds proibidos, quadro que conta na métrica, unidade do bootstrap). Só usa rótulos das bases; nenhum resultado de modelo.

| Base | Treino (com CV de 5 folds) | Unidade dos folds | **Parte de registro** (nunca treina, nunca entra em escolha) | Fotos usadas |
|---|---|---|---|---|
| **OmniFood8K** | 3.862 cenas de 130 pratos principais (121 acima de 25 g; 572 de 15 a 25 g) | Prato principal | **1.928 cenas de 34 pratos principais** (151 acima de 25 g, 39 delas sem a ressalva de osso ou casca; 553 de 15 a 25 g). Mais 2.227 cenas "mistas", relatadas à parte | Treino: 3 dos 6 quadros (`camera_0`, `2`, `4`). Métrica: `camera_0` |
| **Flavoria** | 1.001 pratos, 1.493 fotos sem oclusão, 25 dias | Dia | 227 pratos dos 7 últimos dias (a mesma de 12.5) | Métrica: o quadro preferido |
| **FPB, só de cima** | 747 bandejas, 2.106 fotos, 107 grupos de porção | Grupo de porção | 121 bandejas, 312 fotos, 18 grupos | Métrica: média das fotos da bandeja |
| **Mensa** | 166 bandejas de 5 participantes | Participante | 34 bandejas de 1 participante | |
| **LeFood** | 433 porções, 661 fotos | Porção | 91 porções, 150 fotos | Métrica: a foto "antes" |
| **SNAPMe** | não treina | | 1.163 fotos, só avaliação com rótulo fraco, **se a fundadora autorizar** (13.5) | |

**Como o OmniFood8K é segurado por prato.** O teste dos autores repete os 165 pratos do treino, então mede porção e combinação, não prato novo. Uma cena junta de 1 a 7 pratos (o arroz está em 2.524), e segurar "cenas só com pratos novos" jogaria fora mais da metade da base. A regra:

1. A densidade de proteína de cada prato sai de mínimos quadrados nas próprias cenas (proteína da cena ≈ Σ gramas de cada prato × densidade; o ajuste é exato, erro máximo de 0,3 g). O **prato principal** de uma cena é o que mais contribui com proteína (na mediana, 53% da proteína da cena).
2. **Pratos de registro:** os 35 com `sha1(nome) mod 5 == 3`. Entre os cinco restos possíveis, é o que deixa as cenas com mais de 25 g mais bem repartidas entre registro e treino (151 e 121). Escolhi pelo rótulo, antes de qualquer modelo.
3. **Registro** = cenas cujo prato principal é de registro. **Treino** = cenas sem **nenhum** prato de registro. As outras 2.227 têm um prato de registro como acompanhamento: não treinam, e saem num relato secundário.
4. **CV dentro do treino:** cada prato principal num fold; quando o fold k valida, saem do treino todas as cenas que contêm qualquer prato do fold k (coluna `folds_proibidos`). Sobram de 1.588 a 2.321 cenas de OmniFood8K por fold.

**O que essa parte de registro mede:** cenas cujo prato principal de proteína nunca foi visto, no mesmo laboratório, com o mesmo prato branco e as mesmas câmeras, e com acompanhamentos que podem ter sido vistos. É "prato chinês novo no mesmo cenário". Não mede outra cozinha nem foto de celular.

**Avisos sobre as partes de registro.** A do Flavoria já foi olhada uma vez (12.6, massa de cinco modelos), e aquele resultado é parte do motivo desta seção: ela não é mais virgem. As do OmniFood8K, FPB, Mensa e LeFood nunca foram pontuadas. As cenas acima de 25 g do OmniFood8K vêm de poucos pratos (no registro: amendoim frito, costela, pele de porco, tofu), quase todos com a ressalva de osso ou casca: o intervalo, por bootstrap de prato principal, vai ser largo.

### 13.4 Como cada base supervisiona

| Base | Cabeça de massa (ridge) | Cabeça de densidade (frações) | Peso |
|---|---|---|---|
| Nutrition5k (treino_v2) | Todos os alvos | Todos os alvos, frações e densidade composta, peso por faixa | 1 |
| OmniFood8K | Massa | Proteína, densidade, kcal, gordura e carboidrato (recalculados); massa e proteína por grupo; frações por fonte | Massa 1. Nutrição: 1, e **0,5 nas cenas com `osso_ou_casca`**; vazia nas 97 cenas sem nutrição. Grupos e frações: 0,5 |
| Flavoria, FPB de cima, Mensa, LeFood | **Só a massa total** | **Nada** nas configurações principais. Só na H2: frações por fonte, peso 0,5 | Massa 1 |
| Colunas `*_fraca` (proteína de tabela por nome, autorrelato) | Não usadas | Não usadas | 0 |
| SNAPMe | Não treina | Não treina | |

- **Peso por domínio.** Dentro de um domínio, peso igual por grupo (cena, prato, bandeja), repartido entre as suas fotos. A soma dos pesos de um domínio é `min(0,5; grupos de treino ÷ 2.281)` vezes a do Nutrition5k: OmniFood8K 0,50; Flavoria 0,44; FPB 0,33; LeFood 0,19; Mensa 0,07. Se a soma dos domínios presentes passar de 1,0, todos são multiplicados pelo mesmo fator para ela dar 1,0. O Nutrition5k nunca fica com menos da metade do peso.
- **Viés por domínio,** como em 12.5, nos alvos extensivos (massa, proteína, kcal, gordura, carboidrato e os alvos por grupo, em log): do alvo externo subtrai-se a diferença entre a média do domínio e a do Nutrition5k no treino do fold. **Nenhum viés** na densidade nem nas frações. A densidade composta pelas frações continua com as densidades por fonte do Nutrition5k e só entra na perda para fotos do Nutrition5k.
- **As previsões das fotos externas são gravadas cruas,** como se fossem do Nutrition5k (sem somar o viés de volta). A constante por domínio de 13.2 é aplicada depois, na CPU, igual para todas as configurações.
- **Entrada.** Toda foto que não é 4:3 é encaixada no centro de um quadro 4:3 com a cor mediana da borda, nunca recortada (como o Flavoria em 12.5); foto em retrato é girada para paisagem antes. O OmniFood8K já é 640×480.
- **Orientações nas features congeladas:** Nutrition5k, 8 (como antes); bases externas de uma foto, 2 (original e 180°); OmniFood8K, uma por quadro (os três quadros já variam altura, posição e giro).
- **Quadros do OmniFood8K.** As seis câmeras são todas de cima ou quase, então cada quadro é, sozinho, "uma foto de cima": usar três deles no treino não fere a premissa. Na previsão entra sempre uma foto só, e na métrica conta um quadro por cena. **As vistas lateral e oblíqua do FPB não entram** (10.048 fotos): não são de cima.
- **Recorte pelo prato: ainda não.** Normalizar o Nutrition5k pelo prato (o círculo-guia do app) é o que tiraria a necessidade da constante por domínio. Fica fora desta execução por três motivos: não há detector de prato conferido para o OmniFood8K (prato branco em mesa branca), o FPB e o Mensa (bandejas); o recorte muda a entrada do Nutrition5k e, com ela, a única referência comparável (69,7%); e o círculo-guia ainda não tem medida em pixels. **Consequência que precisa ficar escrita:** a massa numa foto de celular continua sem calibração de escala. As medidas O e M valem "com a escala do domínio conhecida".

### 13.5 Teste de registro e regra de sucesso

**Onde a meta de 70% vai ser julgada, e o que cada lugar diz.**

1. **Nutrition5k: uma última leitura do teste2,** só do modelo de servidor **S** que a regra de 13.9 escolher. Mesma medida e mesma regra: demonstrado só se o **limite inferior do intervalo de 95%** por cadeia, no critério ponderado, for 70% ou mais. Será a **quarta leitura** do teste2. Sai junto o intervalo de 98,75% (correção para quatro leituras), e eu só escrevo "demonstrado no Nutrition5k" se o limite inferior desse intervalo também ficar em 70% ou mais. Se S for a réplica do D1 (nada adotado), **não há leitura nova**: o número é o de 12.7. Depois dessa leitura o teste2 não é lido de novo para veredito.
2. **Fora do Nutrition5k, com proteína:** a parte de registro do OmniFood8K (1.928 cenas, 34 pratos principais). Critério ponderado nas quatro faixas, intervalo de 95% por bootstrap de prato principal; ao lado, o mesmo só nas cenas sem `osso_ou_casca`. A regra é a mesma (limite inferior de 70% ou mais), e o veredito sai com o nome certo: "70% em pratos chineses novos de laboratório", não "70% no produto".
3. **Fora do Nutrition5k, só massa:** partes de registro do Flavoria, do FPB de cima, do Mensa e do LeFood. MAPE, viés mediano e % a ±20%, crus e com a constante. Não há critério de proteína: não entram no veredito de 70%.
4. **SNAPMe: só avaliação, e só com autorização da fundadora.** É o único conjunto com foto de celular em casa. Os autores pedem para não treinar com ele, e é isso que eu recomendo: **não treinar; usar como benchmark, que é o uso que os autores preveem.** O rótulo é autorrelato com porção padrão: serve para ver se a fonte principal de proteína é reconhecida e se a proteína prevista acompanha a declarada (correlação de postos), não para medir acerto em gramas. Se a fundadora não se pronunciar até a execução, o SNAPMe fica de fora por inteiro.

**Quantos modelos são pontuados nas partes de registro externas, na mesma execução da CPU, uma vez:** no máximo quatro, com papel declarado em 13.9: a referência **A0** (réplica do D1, sem base externa), o **S**, o **X** (o de maior nota multi-domínio, se for outro) e o **Ad** (modelo de aparelho, se houver um novo). No teste2, só o S.

**Relato secundário, que não decide nada: kcal, carboidrato e gordura** (escrito em 12/11/2026, antes de qualquer resultado da quarta execução).

- **Quem tem rótulo:**
  - **Nutrition5k:** sim, na mesma natureza da proteína (ingredientes pesados × tabela).
  - **OmniFood8K:** sim, **recalculado**. A lista traz `kcal`, `gordura` e `carbo` em 7.862 cenas: gramas crus × tabela dos autores, com a mesma fração comestível suposta em osso e casca, e sem o valor publicado pelos autores para conferir.
  - **Flavoria, FPB, Mensa e LeFood:** não têm (nem fraco).
  - **SNAPMe:** só autorrelato (`kcal_fraca`, `gordura_fraca_g`, `carbo_fraca_g`).
- **O que sai, para A0, S, X e Ad:**
  1. na CV do treino_v2, o critério de 12.8 (mesmas faixas e pisos);
  2. na parte de registro do OmniFood8K, o mesmo critério, as faixas vazias ficando de fora, com bootstrap por prato principal e, ao lado, só as cenas sem `osso_ou_casca`. As previsões levam a mesma constante de massa por domínio de 13.2, multiplicando o nutriente inteiro; os números crus saem ao lado;
  3. se o S for lido no teste2, o critério de 12.8 sai **dentro da mesma leitura única**, sem leitura a mais;
  4. no SNAPMe, se autorizado, só a correlação de postos com o autorrelato.
- **Rotas:** as de 12.8, fixas. Para kcal e carboidrato, média em log das duas diretas; para gordura, geométrica. Para os modelos de ajuste fino, a direta.
- **Esses números não entram na R13, na escolha do S nem na do Ad.** Nenhuma base passa a ser adotada ou recusada por eles.

**O que nenhum desses números diz.** Um resultado no Nutrition5k é sobre um refeitório americano fotografado por uma câmera fixa. Um resultado no OmniFood8K é sobre 165 pratos chineses num laboratório. **Nenhum dos dois mede prato brasileiro em foto de celular,** e os dois juntos também não. A meta de 70% **do produto** só pode ser julgada num conjunto brasileiro pesado, de celular, que ainda não existe (`fontes_treino.md`, 5.2). Até lá, o que esta execução pode entregar é: o nível no Nutrition5k, o nível numa segunda cozinha com rótulo de tabela, e o erro de massa em quatro cenários de refeitório.

### 13.6 Configurações da quarta execução, em ordem de prioridade

Orçamento: cerca de 4,5 h de relógio em 2 × T4 (540 minutos de GPU); a cota é incerta, então a ordem importa. Só o So400m é extraído: os outros quatro backbones serviam ao professor, que saiu. Ritmo medido na terceira execução: cerca de 15 fotos por segundo no So400m a 384×512.

| Prioridade | Nome | O que é | GPU (estimativa) |
|---|---|---|---|
| 1 | features `so400m` | Nutrition5k (treino_v2 e teste2, 8 orientações: 24,9 mil passadas) e bases externas (27,5 mil passadas: OmniFood8K 15,7 mil; FPB 4,8 mil; Flavoria 3,7 mil; LeFood 1,6 mil; Mensa 0,4 mil; SNAPMe 1,2 mil, se autorizado) | 60 min |
| 1 | `gpu4 ridge so400m` (**R0**) | Cabeça ridge só com o treino_v2. Réplica da do D1 | 1 min |
| 1 | `gpu4 ridge so400m +ok` (**R1**) | + massa de Flavoria, LeFood e Mensa (as três `comercial_ok`) | 2 min |
| 1 | `gpu4 ridge so400m +ok+fpb` (**R2**) | + massa do FPB de cima | 2 min |
| 1 | `gpu4 ridge so400m +omni` (**R3**) | treino_v2 + massa do OmniFood8K | 2 min |
| 1 | `gpu4 ridge so400m +tudo` (**R4**) | treino_v2 + massa das cinco bases | 3 min |
| 1 | `gpu4 fracoes so400m` (**H0**) | Cabeça de frações só com o treino_v2. Réplica da do D1 | 5 min |
| 1 | `gpu4 fracoes so400m +omni` (**H1**) | + OmniFood8K com os alvos de 13.4 | 8 min |
| 2 | `gpu4 ft siglip2_b16_224 r2 fracoes` (**F1**) | Modelo de aparelho: receita `r2` + cabeça de frações com as perdas normalizadas (13.7). Só treino_v2 | 64 min |
| 3 | `gpu4 ftparcial so400m` (**P1**) | Ajuste fino **parcial** do So400m a 384×512: só os 4 últimos blocos, a norma final, o agrupador e a cabeça (13.7). Só treino_v2 | 150 min |
| 4 | `gpu4 ft siglip2_b16_224 r2 +tudo` (**F3**) | Modelo de aparelho: receita `r2` **sem mudança** (a do U0) + as cinco bases externas, com os alvos de 13.4 numa cabeça só | 96 min |
| 5 | `gpu4 fracoes so400m +omni+ff` (**H2**) | H1 + frações por fonte de Flavoria, FPB, Mensa e LeFood (peso 0,5). Testa se foi o rótulo de frações que custou no Flavoria | 10 min |
| 6 | `gpu4 ft siglip2_b16_224 r2 fracoes faixa` (**F2**) | F1 + peso por faixa em todas as perdas | 64 min |

- **Ordem nas GPUs.** GPU A: features, cabeças (R0 a R4, H0, H1, H2), depois P1: cerca de 245 min. GPU B: F1, F3, F2: cerca de 225 min. A GPU B começa o F1 no minuto zero, porque ele só precisa do treino_v2. Total: cerca de 470 dos 540 minutos, mais as medidas de serviço.
- **Se uma base não baixar ou não passar na conferência,** a execução segue sem ela; as configurações que dependem dela saem marcadas, e as regras valem para o que terminar. O OmniFood8K (2,7 GB no Google Drive, que corta por cota) deve ser guardado antes como dataset privado do Kaggle.
- **Nada de decisão dentro do kernel.** Nenhuma configuração depende do resultado de outra; toda escolha é feita depois, na CPU. O kernel só imprime métricas de CV das partes de treino.
- **O ajuste fino parcial do So400m cabe?** Cabe apertado. Conta: os 23 blocos congelados custam uma passada de inferência (52 ms por foto); os 4 treináveis, ida e volta (27 ms): cerca de 13 fotos por segundo, 2,4 min por época num fold, 6 épocas, 5 folds mais o final, e 30 min de previsões. O ajuste fino completo (cerca de 8 h) não cabe. Se o orçamento apertar, o kernel cancela o P1 pela projeção de tempo, como na terceira execução.

### 13.7 Receitas

**Cabeças sobre features congeladas (R e H):** as de 12.3, sem mudança (ridge com alfa por CV interna; MLP de uma camada oculta de 512, 60 épocas, Huber nas 22 saídas, 10 × erro quadrático das frações, Huber da densidade composta, peso por faixa), com as amostras externas mascaradas e pesadas como em 13.4. Sem destilação. **Conferência de paridade, antes de qualquer outra coisa:** R0 e H0 têm de repetir as cabeças do D1 (correlação de 0,999 ou mais com as previsões fora-do-fold gravadas na terceira execução; CV do D1 em 71,3% ± 0,3).

**Ajuste fino com cabeça de frações, corrigido (F1, F2, P1).** O erro de 12.3 foi somar as perdas novas em escala cheia. A correção:

1. Perda = `L_rot` + `α` × (`L_frac` + `L_dens`). `L_rot` é a perda do U0, sem mudança (média de Huber nas 22 saídas, com os pesos por alvo da receita). `L_frac` é a soma do erro quadrático das 10 frações; `L_dens`, o Huber em log da densidade composta.
2. **`α` é recalculado a cada passo** para o gradiente auxiliar nunca passar do gradiente dos rótulos nas features: `α = min(1, ‖∂L_rot/∂f‖ ÷ ‖∂(L_frac + L_dens)/∂f‖)`, com `f` = as features de entrada da cabeça, as normas tiradas no lote, `α` sem gradiente e suavizado por média móvel de 20 passos. Assim **o termo dos rótulos fica com pelo menos metade do gradiente que chega ao backbone**, por construção.
3. Os logits das frações partem de zero, têm passo 10 vezes o da cabeça, e **na primeira época o backbone fica parado** (só a cabeça treina).
4. Sem destilação. Sem peso por faixa no F1 e no P1; o F2 é o F1 com peso por faixa e mais nada.
5. O registro grava, por época: `L_rot`, `L_frac`, `L_dens`, o `α` médio e a validação a cada 5 épocas e na última.
6. **Critério de aceitação, no próprio registro:** a média dos folds do RMSE de log-massa na **última época não pode ser maior que na época 0**. Quem falhar não concorre a nada.

**P1, o resto da receita:** backbone em meia precisão; blocos congelados em modo de avaliação e sem gradiente; 6 épocas; lote 16 com acúmulo de 2; passo 2e-5 no topo com decaimento de 0,75 por bloco nos 4 treináveis; aquecimento de 10%, cosseno; decaimento de pesos 0,05; `drop_path` 0,1 só nos blocos treináveis; média móvel dos pesos 0,99; corte de gradiente em 1,0; orientações e escala aleatória (0,7 a 1,3); cabeça de 22 + 10 saídas com as 22 iniciadas da ridge sobre o backbone congelado. Folds de `treino_v2.csv`.

**F3:** a receita do U0 sem tocar em nada (22 saídas, uma perda), mais as fotos externas: por época, todas as do treino_v2 e um sorteio de 1.140 fotos externas (metade do treino_v2), repartidas entre os domínios na proporção dos pesos de 13.4; perda de cada alvo mascarada onde o rótulo falta; viés por domínio; escala aleatória nos dois lados. Mede uma coisa só: o efeito das bases no modelo de aparelho.

### 13.8 O que cada configuração grava

- **`pred__<nome>.npz`:** `oof`, `oof_o1`, `oof_o4` do treino_v2 (e `oof_ff`, `oof_dens_ff` quando há frações); para **cada** domínio externo e **todas** as configurações, treinem ou não com ele: `extcv__<dominio>` (previsão crua de cada foto de treino pelo modelo do fold em que ela valida, `fold_ext`) e `extreg__<dominio>` (previsão crua do modelo final para as linhas de `registro`, `registro_misto` e `avaliacao_fraca`), com os ids; `teste2`, `teste2_o1`, `teste2_o4`; os vieses por domínio de cada fold; ids, folds e hash de todas as listas e tabelas de divisão.
- **O kernel não pode abrir os rótulos das partes de registro.** Ao carregar cada lista, ele apaga as colunas de alvo de toda linha que não seja `treino`. Não há "relato externo" automático desta vez: as previsões de registro e as do teste2 ficam gravadas e **fechadas** até a leitura de 13.9.
- **Para servir:** as cabeças finais (treinadas com tudo o que é `treino`), os pesos do P1 e dos ajustes finos em meia precisão, os arquivos de servir e a conferência de remontagem de **todas** as cabeças, inclusive as de frações (faltou na terceira execução); medidas de serviço (T4 e CPU, 1 orientação) do So400m com as duas cabeças juntas, do P1 e dos ajustes finos.
- **Procedência, no manifesto de cada modelo:** `treinado_com` = lista de `{dominio, lista, sha256 da lista, grupos, fotos, alvos usados, licenca, uso_declarado}`, e `so_comercial_ok` (verdadeiro se todas as bases forem `comercial_ok`; o Nutrition5k é CC BY 4.0). R0, R1, H0, F1, F2 e P1 são `so_comercial_ok`; tudo que leva OmniFood8K ou FPB, não.

### 13.9 Regra de escolha, aplicada na CPU, só com validação cruzada

**Modelos de servidor congelados.** São as 15 combinações de uma cabeça de massa (R0 a R4) com uma de densidade (H0 a H2): proteína = massa da ridge × média em log das duas densidades da cabeça de frações, **uma orientação**, como o D1. **A0 = R0 × H0.** Para cada uma saem N, O, M e a nota **S = (N + O + M) ÷ 3**.

**Regra R13 (adoção de dados externos):**

1. **Guarda do Nutrition5k.** Uma combinação é admissível se N não cair mais de **1,0 ponto** em relação ao A0 **e** o critério acima de 25 g não cair mais de **2,5 pontos**. O A0 é sempre admissível.
2. **A\* = a combinação admissível de maior S.** Empate (menos de 0,5 ponto de S): a que usa menos bases não comerciais; depois, a que usa menos bases.
3. **X = a combinação de maior S sem olhar a guarda.** Se for diferente da A\*, ela é o "modelo enriquecido": é pontuada nas partes de registro externas, para a fundadora ver a troca com números de teste, e não é lida no teste2.
4. Sai sempre, ao lado, **a melhor combinação admissível só com dados liberados** (R0 ou R1 com H0), que é o modelo que pode ser refeito sem as bases não comerciais.

**O que a R13 mostra e o que não mostra.**

- Mostra se um modelo só consegue servir três domínios sem perder no primeiro. Em O e M, uma combinação que treina com o domínio leva vantagem sobre uma que não treina: **S sobe quase por construção quando a base entra.** Por isso a guarda é o que decide de verdade, e S só ordena quem passou nela.
- Com os números da terceira execução, o Flavoria do jeito de 12.5 (−2,1 em N e −4,0 acima de 25 g, no D1) **seria barrado**. A aposta desta seção é que, com as bases de massa só na cabeça de massa, o custo some.
- **Não mostra transferência para um terceiro domínio.** A única evidência disso que os dados permitem sai numa tabela à parte, que não decide nada: (a) o efeito do OmniFood8K na massa de Flavoria, FPB e Mensa quando nenhum dos três treina (R3 contra R0); (b) o efeito das bases de massa na proteína do OmniFood8K quando ele não treina (R1 e R2 contra R0, com H0); (c) o efeito do OmniFood8K em N (H1 contra H0). Se (a), (b) e (c) derem zero, a leitura honesta é "o modelo aprendeu mais domínios", não "o modelo generaliza melhor".
- Intervalos: N por bootstrap de cadeia; O por prato principal; M por dia, grupo de porção e participante. O Mensa tem cinco participantes: o intervalo dele quase não informa.

**Modelo de servidor para a leitura (S).** S = P1 se ele passar na aceitação de 13.7, o critério dele em N (mesma rota, uma orientação) ficar **1,5 ponto ou mais** acima do da A\* **e** o limite inferior de 95% da diferença pareada por cadeia ficar acima de zero. Senão, S = A\*. O P1 treina só com o treino_v2, então a comparação com o A0 mede só o ajuste fino parcial.

**Modelo de aparelho (Ad).** Candidatos: F1, F2 e F3 que passarem na aceitação, cada um na rota e nas orientações escolhidas pelos itens 1 e 2 de 12.3. Há um Ad novo só se o melhor deles ficar **1,0 ponto ou mais** acima do U0 da terceira execução em N (68,0%; mesmos folds, mesma lista). Senão o modelo de aparelho continua sendo o C1 de 12.6. O Ad é pontuado nas partes de registro externas e **não** no teste2.

**Antes de qualquer leitura:** conferências sem gabarito, como em 12.1 (previsões completas e na ordem; remontagem de todas as cabeças; paridade do A0 com o D1 nas previsões do teste2 gravadas na terceira execução, sem abrir rótulo). A leitura é uma execução só, com marca que impede a segunda.

### 13.10 O que eu espero

| | Expectativa | Chance |
|---|---|---|
| A0 (réplica do D1) na CV | 71,3% ± 0,3 | |
| Bases de massa na cabeça de massa (R1, R2): efeito em N | −0,5 a +0,3 ponto | Passar na guarda: 85% |
| Idem: efeito em M | +5 a +15 pontos de pratos a ±20% | |
| OmniFood8K na cabeça de densidade (H1): efeito em N | −1,5 a +0,5 ponto | **Piorar N além da guarda: 40%.** Ficar dentro de ±1: 45%. Subir 1 ponto ou mais: 15% |
| Idem: O | de 40–50% (sem a base) para 55–65% | |
| A\* diferente do A0 | | 60%, quase sempre pelas bases de massa |
| P1 sobre o A0 em N | +0,5 a +2,5 pontos | Passar na regra (1,5 ponto e diferença segura): 25% |
| F1 em N | 67,5 a 69,5% (U0: 68,0%) | Virar Ad: 30% |
| F3 em N | 67 a 69% | |
| S no teste2, se houver leitura | 68,5 a 71,5% | **Demonstrar 70%: menos de 5%** |
| Parte de registro do OmniFood8K, melhor modelo com a base | 50 a 62% | Demonstrar 70% ali: menos de 5% |

- **O ruído do OmniFood8K pode custar mais do que rende no Nutrition5k, e eu acho isso quase tão provável quanto o contrário.** A proteína é recalculada por tabela; em 863 das 3.765 cenas de treino com nutrição ela depende de uma fração comestível suposta; os pratos são refogados chineses, em que a divisão por fonte segue outra lógica que a do Nutrition5k; e o Flavoria já mostrou que rótulo de frações de outra base derruba a cabeça de densidade. Se o H1 piorar N, a R13 fica com o H0 e o OmniFood8K entra só pela massa, ou não entra.
- **Onde o OmniFood8K pode ajudar de verdade:** na faixa de 15 a 25 g (572 cenas de treino), que no teste2 é onde o modelo de aparelho mais perde, e em ensinar proteína fora do refeitório do Nutrition5k.
- **Não espero que esta execução demonstre a meta em lugar nenhum.** O que ela pode entregar: um modelo de servidor que não perde no Nutrition5k e erra menos a massa fora dele; a primeira medida de proteína numa segunda cozinha; a resposta sobre a cabeça de frações no ajuste fino; e saber se o ajuste fino parcial do backbone grande vale as 8 horas do completo.
- **O que continua faltando para o produto, com ou sem esta execução:** o recorte pelo círculo-guia (escala), e um conjunto brasileiro pesado, de celular, para treinar e, principalmente, para julgar os 70%.

### 13.11 Resultado da quarta execução (13/11/2026)

A quarta execução rodou no Kaggle (2 × T4, 3,7 h de relógio, nada cortado pelo orçamento). As seis bases chegaram: Mensa (GitHub), LeFood (Mendeley), FPB (Hugging Face), OmniFood8K (Google Drive, 2,54 GB, 15.741 fotos de cima), Flavoria (Zenodo) e SNAPMe (Figshare, só avaliação). A ingestão aceitou as 12 configurações (`ingere_kaggle.py --so-confere`). As tabelas de divisão usadas pelo kernel são as de 13.3: os sha256 batem com os gravados em cada previsão. **Paridade com o D1:** as cabeças R0 e H0 repetem as da terceira execução, com correlação de 1,00000 e CV de 71,29%. A fundadora delegou as decisões de treino (`docs/produto.md`): o que a regra de 13.9 permite é feito sem esperar aprovação. Script: `eval/treino/r4_avalia.py`.

**a) Regra R13, só com validação cruzada, escrita aqui antes de abrir qualquer parte de registro ou o teste2.** As 15 combinações de uma cabeça de massa com uma de densidade, 1 orientação. N é o critério ponderado no treino_v2. O é o critério em quatro faixas no treino do OmniFood8K, com folds por prato principal e só o quadro `camera_0`. M é a % de pratos com a massa a ±20% em Flavoria, FPB de cima e Mensa. O e M levam a constante de massa por domínio.

| Combinação | Bases | N | > 25 g | O | M | S = (N+O+M)/3 | Guarda |
|---|---|---|---|---|---|---|---|
| **R0×H0 = A0** (réplica do D1) | nenhuma | **71,3** | 51,9 | 54,2 | 43,2 | 56,2 | sim |
| **R3×H0 = A\*** | OmniFood8K (só massa) | **70,5** | 51,8 | 55,8 | 47,1 | **57,8** | **sim** |
| R4×H1 = **X** | as cinco | 69,2 | 48,9 | 56,0 | 57,4 | **60,9** | não (N −2,1) |
| R4×H2 | as cinco | 68,2 | 48,6 | 55,7 | 57,4 | 60,4 | não |
| R4×H0 | as cinco | 68,6 | 49,3 | 53,8 | 57,4 | 59,9 | não |
| R2×H1 | as cinco | 68,1 | 46,4 | 52,6 | 56,8 | 59,2 | não |
| R2×H2 | as cinco | 67,7 | 47,0 | 52,0 | 56,8 | 58,9 | não |
| R2×H0 | Flavoria, LeFood, Mensa, FPB | 68,0 | 48,4 | 50,1 | 56,8 | 58,3 | não |
| R3×H1 | OmniFood8K | 69,3 | 48,6 | 57,7 | 47,1 | 58,0 | não |
| R1×H1 | Flavoria, LeFood, Mensa, OmniFood8K | 69,0 | 48,0 | 49,6 | 53,7 | 57,4 | não |
| R3×H2 | as cinco | 68,0 | 48,5 | 56,6 | 47,1 | 57,3 | não |
| R1×H0 | Flavoria, LeFood, Mensa | 69,2 | 49,3 | 48,8 | 53,7 | 57,3 | não (N −2,1) |
| R1×H2 | as cinco | 67,8 | 47,3 | 49,0 | 53,7 | 56,8 | não |
| R0×H1 | OmniFood8K | 69,6 | 48,1 | 56,4 | 43,2 | 56,4 | não |
| R0×H2 | as cinco | 69,2 | 50,0 | 55,9 | 43,2 | 56,1 | não |

- **Só duas combinações passam na guarda: o A0 e o R3×H0.** A guarda pede N a no máximo 1,0 ponto do A0 e o critério acima de 25 g a no máximo 2,5 pontos. Toda cabeça de massa com Flavoria, LeFood e Mensa (R1, R2, R4) derruba N em 2 a 3 pontos, e toda cabeça de densidade com OmniFood8K (H1, H2) em 1,7 a 2,0.
- **A\* = R3×H0:** a cabeça de massa treinada também com a massa do OmniFood8K e a cabeça de densidade do D1, sem base externa. Contra o A0: N −0,8 (IC por cadeia −2,2 a +0,7); acima de 25 g −0,1 (IC −3,0 a +2,8); O +1,6 (IC por prato −3,1 a +7,9); M +3,9 pontos. Nenhuma dessas diferenças se distingue de zero. **O A\* é admissível e é outro modelo que não o A0, mas não é melhor que ele de forma demonstrável.**
- **X = R4×H1** (todas as bases, nas duas cabeças): a maior nota S (60,9), com M de 57,4% (o A0 tem 43,2%), mas N −2,1 (IC −3,8 a −0,4) e acima de 25 g −3,0. É o "modelo enriquecido" de 13.9: vai para as partes de registro externas, não para o teste2.
- **Melhor admissível só com dados liberados: o próprio A0.** O R1×H0 (Flavoria, LeFood e Mensa, todas CC BY) cai 2,1 em N. Hoje, o modelo que pode ser refeito sem bases não comerciais é o D1.
- **Transferência (não decide nada, 13.9).**
  - Massa do OmniFood8K na ridge, medida em domínios que não treinaram com ela: Flavoria +2,7, FPB +1,9, Mensa +7,2 pontos de pratos a ±20%. Nenhum intervalo exclui zero.
  - Massas de Flavoria, LeFood e Mensa na ridge, medidas na proteína do OmniFood8K: **−5,4** (IC −9,3 a −0,8); com o FPB, −4,2 (IC −7,4 a −0,3).
  - OmniFood8K na cabeça de densidade, medido no Nutrition5k: **−1,7** (IC −3,0 a −0,4).
  - **Leitura:** o que cada base ensina vale para o domínio dela. Fora dele, ou não se mede ganho, ou se mede perda. Os dados não mostram generalização.
- **S = A\* = R3×H0.** O P1 não passou na aceitação (abaixo). **Como o S não é o A0, a regra de 13.5 abre a última leitura do teste2 para ele.**
- **Modelo de aparelho.**
  - F1 e F2 não passaram na aceitação.
  - O F3 (receita do U0 com as cinco bases) passou, mas a regra de 12.3 dá a ele 67,7% (geométrica, 1 orientação), **0,4 abaixo do U0** (68,0%). A regra pedia 1,0 acima.
  - **Não há modelo de aparelho novo:** continua o C1 de 12.6. O F3 não é pontuado nas partes de registro.
- **Relato secundário de 12.8, na CV:** kcal / carboidrato / gordura de 70,5 / 78,6 / 68,4% no A0, 70,5 / 78,6 / 68,7% no A\* e 69,7 / 78,7 / 69,5% no X. Nada muda.

**b) Por que o ajuste fino com a cabeça de frações ainda estragou a massa.**

| | Época 0 → última (RMSE de log-massa, média dos folds) | `α` médio | L_rot ÷ (L_frac + L_dens), última época | L_rot, última época |
|---|---|---|---|---|
| F3 (só a perda de rótulos, + bases) | 0,2545 → **0,2313** | | | 0,017 |
| F1 (frações, perdas normalizadas) | 0,2545 → 0,2665 | 0,94 a 0,98 | 0,19 | 0,029 |
| F2 (F1 + peso por faixa) | 0,2545 → 0,2622 | 0,94 a 0,97 | 0,21 | 0,030 |
| P1 (So400m, 4 últimos blocos, frações) | 0,2167 → 0,2457 | 0,92 a 1,00 | 0,14 | 0,022 |

- **O balanço funcionou como foi escrito.** O `α` ficou entre 0,92 e 1,00 o tempo todo. O gradiente auxiliar, medido na entrada da cabeça, ficou do tamanho do gradiente dos rótulos, nunca maior: os rótulos receberam metade, como 13.7 pedia.
- **Metade não basta, e a falha é da minha especificação.** Eu equilibrei contra o termo dos rótulos inteiro, que é a média de 22 saídas; a massa é 1/22 dele. Com metade do gradiente indo para frações e densidade, a massa recebe uns 2%. As features se moveram para servir às frações, e a perda de rótulos terminou 70% acima da do F3 (0,029 contra 0,017).
- **O problema não é treinar o backbone, é treiná-lo com a perda de frações.** O F3, mesma receita sem frações e com cinco bases a mais, melhora a massa como o U0. No P1 a queda é a maior (0,217 → 0,246), porque o ponto de partida já era a ridge boa do So400m.
- **O que eu faria:** a cabeça de frações **sem gradiente para o backbone** (treinada sobre as features, com `stop-gradient`), ou o ajuste fino só com a perda de rótulos e a cabeça de frações treinada depois sobre as features congeladas do modelo ajustado. Nas features congeladas, a cabeça de frações sempre funcionou.

**c) Antes da última leitura do teste2 (escrito antes de rodar).** O S é o R3×H0, e não o A0, então 13.5 manda ler. É a **quarta leitura** do teste2: sai o intervalo de 98,75% ao lado do de 95%, e "demonstrado no Nutrition5k" só se os dois limites inferiores ficarem em 70% ou mais.
- **Expectação:** o S difere do D1 só na cabeça de massa, e na CV fica 0,8 ponto abaixo dele. Espero de 68 a 70,5% no teste2, com o palpite em 69%. A chance de demonstrar a meta é **menor que 3%**.
- **Conferências sem gabarito, antes:** previsões completas e na ordem; o A0 refeito tem de reproduzir as previsões do D1 gravadas em 12.7; remontagem das duas cabeças.
- **Também sai,** dentro da mesma leitura e sem leitura a mais, o critério de 12.8 para kcal, carboidrato e gordura.

**d) Partes de registro externas: uma leitura, de três modelos** (A0, S e X, como 13.5 permite; não há modelo de aparelho novo). Rodou uma vez (`r4_avalia.py --registro`, com marca). A massa leva a constante do domínio, estimada nas partes de treino; os números crus saem ao lado.

*OmniFood8K, cenas cujo prato principal de proteína nunca foi visto* (1.916 cenas com rótulo, 34 pratos principais; critério em quatro faixas; IC por bootstrap de prato principal):

| | A0 (D1) | **S** (R3×H0) | X (R4×H1) |
|---|---|---|---|
| **Proteína, critério de quatro faixas** | **62,5%** (56,2 a 70,2) | **59,3%** (54,3 a 67,9) | **65,5%** (56,2 a 80,3) |
| Por faixa: 0–5 / 5–15 / 15–25 / > 25 g | 94 / 76 / 51 / 28 | 98 / 79 / 51 / **9** | 99 / 87 / 43 / 32 |
| Sem a ressalva de osso ou casca (1.453 cenas) | 53,1% (45,5 a 77,8) | 54,8% (48,6 a 81,9) | 52,5% (48,6 a 58,5) |
| Sem a constante de massa | 56,1% | 57,1% | 64,5% |
| Massa: a ±20% / MAPE / viés | 48,6% / 24,7% / −3% | 72,7% / 14,9% / −5% | 70,0% / 15,5% / −5% |
| kcal (12.8; viés) | 32,3% (−38%) | 30,3% (−44%) | 25,8% (−47%) |
| Carboidrato (12.8) | 64,8% (−20%) | 61,2% (−27%) | 58,4% (−33%) |
| Gordura (12.8) | 35,4% (−56%) | 35,5% (−58%) | 46,8% (−47%) |
| Cenas "mistas" (prato de registro como acompanhamento; 2.181) | 62,0% | 61,6% | 67,8% |

- **Nenhum modelo chega a 70% nas cenas novas do OmniFood8K;** o melhor limite inferior é 56%.
- A massa do OmniFood8K na cabeça de massa (S) acerta muito mais a massa dessas cenas (73% a ±20%, contra 49%). Mesmo assim, a proteína não melhora (−3,2, dentro do ruído), e os pratos com mais de 25 g caem de 28% para 9%: a densidade, que é a mesma nos dois, fica baixa nesses pratos, e a massa do A0 errava para cima e compensava. São 151 cenas de poucos pratos (amendoim, costela, pele de porco, tofu).
- **kcal e gordura são muito subestimadas nessa cozinha:** viés de −38% a −58%. Refogado chinês leva óleo que a foto não mostra. Para o produto, kcal e gordura de comida frita ou com óleo vão sair baixas.

*Bases só de massa (% a ±20%, MAPE e viés mediano, com a constante; cru entre parênteses; IC por cluster):*

| Parte de registro | A0 (D1) | **S** | X | Pratos acima de 600 g, MAPE (A0 / S / X) |
|---|---|---|---|---|
| Flavoria, 227 pratos de 7 dias | 52,0% (41,5 a 66,3); 25,6%; −10% (cru 23%) | 57,3% (47,2 a 72,5); 22,8%; −7% | **66,5%** (60,0 a 73,6); 19,3%; −4% | 28,6% / 26,4% / **17,6%** (46 pratos) |
| FPB de cima, 121 bandejas, 18 grupos | 31,4% (18,6 a 42,5); 34,6%; −20% | 35,5% (25,0 a 49,0); 33,4%; −13% | 42,1% (35,6 a 48,4); 25,0%; −15% | 28,3% / 27,8% / 25,1% (29) |
| LeFood, 91 porções | 26,4%; 43,9% | 30,8%; 36,0% | **68,1%**; 16,8% | |
| Mensa, 34 fotos de um participante | **inválido** (abaixo) | | | |

- **Cada base melhora a massa do próprio domínio,** e só nele. O X, que treinou com as quatro, acerta 67% no Flavoria (A0: 52%) e 68% no LeFood (A0: 26%). Nos pratos de mais de 600 g do Flavoria, o erro cai de 29% para 18%, como em 12.6.
- **O Mensa mostrou um erro meu e um problema real.**
  - O participante segurado fotografou **itens soltos de perto** (33 das 34 fotos têm um item só); os outros cinco fotografaram a **bandeja inteira** (mediana de 600 a 740 g). Eu fixei a parte segurada por participante sem olhar o que cada um fotografou.
  - A constante aprendida nas bandejas (×3,3) não serve para o close: o erro passa de 500%, e mesmo cru só 12% a 32% das fotos ficam a ±20%.
  - **É o problema de escala de 8.7 em forma pura:** a mesma comida, fotografada por pessoas diferentes, muda de escala em 3 a 5 vezes, e nenhuma constante por domínio resolve.
*SNAPMe* (1.163 fotos de celular em casa, 95 pessoas; **autorrelato, só correlação de postos**, IC por pessoa):

| | Proteína | kcal | Gordura | Massa | Carboidrato |
|---|---|---|---|---|---|
| A0 | 0,64 (0,59 a 0,68) | 0,60 | 0,63 | 0,42 | 0,38 |
| S | 0,59 (0,54 a 0,64) | 0,60 | 0,61 | 0,37 | 0,38 |
| X | 0,62 (0,58 a 0,67) | 0,58 | 0,65 | 0,40 | 0,37 |

  - A proteína prevista acompanha a declarada com correlação de 0,6 em foto de celular de verdade. A massa (0,4) acompanha menos que a proteína, o que é o esperado: a porção declarada é a padrão do recordatório, e a foto não tem escala.
  - Nada disso é acerto em gramas.

**e) Última leitura do teste2: o S** (13/11/2026; a quarta leitura do estudo; marca gravada). As conferências passaram: o A0 refeito reproduz o D1 de 12.7 com correlação de 1,000000, e as cabeças foram remontadas com diferença de 0,0003 e 0,0001 em log.

**Veredito pela regra: 70% NÃO DEMONSTRADO.** O S dá **69,2%** (IC por cadeia de 95%: **65,0 a 73,7**; 98,75%: 64,1 a 74,7). Simples, 76,7%.

| Faixa | Pratos | **S** (R3×H0) | D1 (12.7) | C1 (12.6) | Nove redes (11.5) | Massa do S: MAPE (viés) |
|---|---|---|---|---|---|---|
| 0–5 g | 316 | 98,1% | 98,1% | 98,7% | 97% | 25,8% (+5%) |
| 5–15 g | 169 | 76,3% | 78,7% | 82,2% | 82% | 17,0% (+2%) |
| 15–25 g | 132 | 57,6% | 64,4% | 53,0% | 64% | 16,1% (+5%) |
| 25–40 g | 110 | 56,4% | 55,5% | 52,7% | 51% | 12,7% (0%) |
| > 40 g | 102 | 57,8% | 52,0% | 43,1% | 55% | 14,1% (−1%) |
| **Ponderado** | 829 | **69,2%** (65,0 a 73,7) | 69,7% | 66,0% | 69,7% | 19,3% (+3%) |
| > 25 g | 212 | **57,1%** | 53,7% | 47,9% | 52,9% | |

- **Contra o D1: −0,5** (IC −3,2 a +2,3); acima de 25 g, **+3,4** (IC −2,2 a +9,1).
- **Contra as nove redes: −0,4** (IC −4,3 a +3,2). **Contra o C1: +3,3** (IC −1,9 a +8,6).
- O S troca um pouco de acerto nos pratos de 5 a 25 g por um pouco nos de mais de 25 g: **57%**, o maior número nessa faixa em todo o estudo. A diferença não se distingue de zero.
- Os 107 pratos sem irmão no `teste` antigo dão 76,7% (IC 67,7 a 84,8).
- **kcal, carboidrato e gordura no teste2** (relato secundário de 12.8, na mesma leitura): **71,1%** (IC 65,9 a 76,5), **84,3%** (79,3 a 89,0) e **65,5%** (61,4 a 69,6). Os vieses medianos ficam entre +2% e +3%. Por faixa:
  - kcal: 94 / 71 / 61 / 69 / 61;
  - carboidrato: 99 / 96 / 90 / 69 / 68;
  - gordura: 98 / 86 / 67 / 49 / 28.
  - A CV de 12.8 se confirmou para kcal e carboidrato. A gordura acima de 35 g é fraca (28%).
- **O teste2 está encerrado como teste de registro.** Quatro leituras; nenhum pipeline passou de 70% no limite inferior, e os quatro modelos lidos ficam entre 66% e 70%.

**f) Custo de servir e procedência.**

| Modelo | Papel | Treinado com (licença) | Só dados liberados? | Custo |
|---|---|---|---|---|
| **S** = R3×H0 | Servidor, escolhido pela R13 | Nutrition5k (CC BY 4.0) + **massa** do OmniFood8K (**CC BY-NC 4.0**) | **Não** | So400m congelado + 2 cabeças: 429,7 M de parâmetros, 860 MB em fp16, **66 ms** por foto na T4 (56 ms em lotes de 8), 3,5 s em CPU de 4 threads |
| A0 = D1 = R0×H0 | Referência; **o melhor só com dados liberados** | Nutrition5k (CC BY 4.0) | **Sim** | O mesmo |
| X = R4×H1 | Enriquecido, não lido no teste2 | Nutrition5k + Flavoria, LeFood, Mensa (CC BY 4.0) + FPB e OmniFood8K (**CC BY-NC 4.0**) | **Não** | O mesmo |
| C1 | Aparelho (sem mudança) | Nutrition5k (CC BY 4.0) | Sim | 93 M, 186 MB, 8 ms na T4, 0,24 s em CPU |

- **Para servir o S, nada precisa ser treinado de novo:** o backbone é público, e as cabeças `cabeca__gpu4_ridge_so400m__omni.npz` e `cabeca__gpu4_fracoes_so400m.npz` estão na saída da execução, fora do repositório, com a remontagem conferida.
- **Atribuição obrigatória:** Nutrition5k, e OmniFood8K para o S; Flavoria, LeFood, Mensa e FPB também para o X.

**g) O que isto significa para o produto.**

1. **A meta de 70% não está demonstrada em lugar nenhum.**
   - No Nutrition5k, o S dá 69,2% (65,0 a 73,7), o mesmo nível do D1 e das nove redes.
   - Nas cenas novas do OmniFood8K, de 59% a 66%, todas com limite inferior abaixo de 57%.
   - Em comida brasileira de celular, nada foi medido.
2. **O que os dados externos mudaram.**
   - **No Nutrition5k, quase nada, e só para pior:** só a massa do OmniFood8K na cabeça de massa passou na guarda (−0,8 na CV, −0,5 no teste2), e toda outra entrada custou de 1,7 a 3 pontos.
   - **Fora dele, muito, e só no próprio domínio:** a massa a ±20% sobe de 52% para 67% no Flavoria e de 26% para 68% no LeFood quando a base entra no treino, e não sobe onde ela não entra.
   - A proteína nas cenas novas do OmniFood8K melhora pouco (X +3, dentro do ruído) e piora acima de 25 g no S.
   - **Não houve generalização mensurável.** Houve "mais domínios aprendidos", com custo no Nutrition5k.
3. **A escala da foto é o limite que mais aparece, e é a próxima alavanca.**
   - Sem a constante por domínio, a massa externa cai para 1% a 35% a ±20%.
   - No Mensa, um fotógrafo que chegou mais perto muda a escala em 3 a 5 vezes.
   - O app vai ter exatamente esse problema. O que pode resolver não é mais dado da mesma foto: é **enquadrar a foto pelo prato** (o círculo-guia), treinar e prever com o prato normalizado, e medir isso. Isso pede um detector de prato conferido nas bases e o tamanho do guia em pixels.
4. **Segunda alavanca: um conjunto brasileiro pesado, de celular, com o guia.** Serve para treinar e, principalmente, para julgar os 70% do produto. O teste2 acabou.
5. **No modelo de aparelho, nada mudou** (C1, 66% no teste2).
   - A receita com frações ainda estraga a massa: falta a versão sem gradiente para o backbone, item (b).
   - As bases externas no ajuste fino (F3) não ajudaram no Nutrition5k.
6. **kcal e carboidrato:** 71% e 84% no teste2, com o servidor, no critério de 12.8. Dá para mostrar como estimativa com faixa. Gordura e kcal de comida com óleo ficam baixas (OmniFood8K: −40% a −58%).
7. **Licença:** o S, escolhido pela regra, usa uma base não comercial e não ganha de forma demonstrável do D1 em lugar nenhum. Se a licença pesar, o D1 entrega o mesmo número hoje. A decisão de treino delegada segue a regra, então o modelo de servidor de registro passa a ser o S.

**Registro de método da seção 13.**
- (a) As regras de 13.1 a 13.10 foram escritas antes de qualquer resultado da quarta execução.
- (b) A aplicação da R13 e o diagnóstico (itens a e b) foram escritos antes de qualquer leitura. A nota do item (c) foi escrita depois da leitura das partes externas e antes da do teste2.
- (c) As partes de registro externas foram lidas uma vez, para três modelos; o teste2, uma vez, só para o S (quarta leitura do estudo).
- (d) Antes das leituras rodaram, sem gabarito delas: `--so-cv` e as conferências embutidas nos modos de leitura.
- (e) Nenhuma escolha foi refeita depois das leituras.
- (f) A parte segurada do Mensa ficou inválida por um erro de desenho meu, achado na leitura; registro aqui e não troco a parte.
- (g) As divisões que o kernel usou são as de 13.3 (sha256 conferidos).



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
DENS_V3_RODADA=1 python3 dens_avalia.py && DENS_V3_RODADA=2 python3 dens_avalia.py   # cada rodada cega do LLM v3 sozinha (o padrão é a média das rodadas)
python3 dens_rodadas.py          # compara as rodadas (9.7 d)
python3 dens_avalia.py --sem-gpu  # refaz o estado da seção 9 (sem os modelos de GPU da seção 10)
python3 ../protein_from_shares.py ../../docs/dados/precisao/estimativas_v3_partes.csv ../../docs/dados/precisao/estimativas_v3_pratos.csv
```
As previsões ficam em `$TD/resultados/dens/banco/` e os resumos em `$TD/resultados/dens/{analise,avalia}.json`. Tempo de CPU: cerca de 1 h 15 min, dos quais 40 min são a extração do TIPS.

Seção 10 (treino em GPU). O pacote e as instruções para rodar no Kaggle ou no Colab estão em `eval/treino/kaggle/LEIA-ME.md`; a saída (previsões, registros e pesos) fica fora do repositório.
```
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida> --ensaio <pasta_qualquer>   # confere e avalia numa cópia do banco
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida>                              # copia para o banco e roda zen_avalia.py
cd eval/treino
python3 dens_avalia.py --so-cv                                  # escolhas por CV e calib com os modelos de GPU, sem ler o teste
python3 dens_avalia.py                                          # massa escolhida pela CV no banco inteiro; LLM v3 = média das rodadas
DENS_V3_RODADA=1 python3 dens_avalia.py && DENS_V3_RODADA=2 python3 dens_avalia.py
```
`dens_avalia.py` lê `resultados/dens/ref_pre_gpu.npz` (previsões por prato dos pipelines da seção 9, gravadas antes da ingestão) para as comparações pareadas.

Seção 11 (teste2). As fotos já estão em `$TD/img/`; os lotes cegos ficam fora do repositório.
```
cd eval/treino
python3 teste2_monta.py $TD <scratchpad>/cego4            # quem entra, ids neutros, gabarito, lista para o Kaggle, lotes cegos
python3 zen_extrai.py $TD siglip2_b16 --orient 2 --conj teste2 --saida siglip2_b16_teste2          # ~4 min
python3 zen_extrai.py $TD siglip2_b16_512 --orient 2 --conj teste2 --saida siglip2_b16_512_teste2  # ~30 min
OMP_NUM_THREADS=2 python3 teste2_cabecas.py               # as três cabeças de frações, com paridade nos 450 antigos
python3 teste2_avalia.py <pasta_kaggle_1> --ensaio        # ensaio no teste antigo: refaz os números da seção 10
python3 teste2_avalia.py <pasta_kaggle_2> --confere       # conferências sem gabarito
python3 teste2_avalia.py <pasta_kaggle_2>                 # leitura única do teste2
```
A segunda execução no Kaggle (`eval/treino/kaggle/`, com `listas/aval2_ids.csv`) grava também `servico.json` e `tamanhos.json`, de onde saem os números de 11.6. A rodada cega v3 do teste2 está em `docs/dados/precisao/estimativas_v3_teste2_partes.csv`; o resultado da leitura única, em `$TD/resultados/dens/teste2/avalia_teste2.json`.

Seção 12 (um modelo só e mais dados).
```
cd eval/treino
python3 treino_v2_monta.py $TD       # treino_v2 (2.281 pratos), folds por cadeia, conferência de cadeias contra o teste2
python3 um_modelo_avalia.py <pasta_kaggle_3> --so-cv      # regra de 12.3 só com as previsões fora-do-fold; grava selecao.json
python3 um_modelo_avalia.py <pasta_kaggle_3> --confere    # conferências sem gabarito do(s) candidato(s) escolhido(s)
python3 um_modelo_avalia.py <pasta_kaggle_3>              # leitura única do teste2 (C1 e, se a regra deixar, C2); recusa rodar de novo
python3 um_modelo_avalia.py <pasta_kaggle_3> --relato     # depois da leitura: diferenças pareadas para os pipelines da seção 11
python3 um_modelo_emenda.py <pasta_kaggle_3> --so-cv      # 12.7: definição congelada do D1 (e critério do D2), só com fora-do-fold
python3 um_modelo_emenda.py <pasta_kaggle_3> --confere    # 12.7: conferências sem gabarito
python3 um_modelo_emenda.py <pasta_kaggle_3>              # 12.7: leitura única do D1 no teste2; recusa rodar de novo
```
Seção 12.8: `cd eval/treino; python3 macros_cv.py <pasta_kaggle_3>` (só previsões fora-do-fold do treino_v2; grava `$TD/resultados/dens/um_modelo/macros_cv.json`).

Seção 13.11: `cd eval/treino; python3 r4_avalia.py <pasta_kaggle_4> <pasta_divisoes> --so-cv` (R13 só com validação cruzada), depois `--registro` (leitura única das partes de registro externas) e `--teste2` (última leitura do teste2, só do S); resultados em `$TD/resultados/dens/r4/`.

Seção 13 (quarta execução): `python3 eval/treino/externo_r4_divide.py <pasta_de_saida>` grava as divisões das bases externas (`divisao_r4_<dominio>.csv`: papel, folds, parte de registro), a partir das listas `externo_*.csv`; as do OmniFood8K, do FPB e do SNAPMe derivam de bases não comerciais e ficam fora do repositório.

A terceira execução em GPU (12.3 e 12.5) usa `eval/treino/kaggle/listas/treino_v2.csv`, `aval2_ids.csv` e `externo_flavoria.csv`; a pasta de saída (previsões, registros, `servico.json`, `decisao_flavoria.json`, `relato_externo.json`) fica fora do repositório. Os resultados de 12.6 ficam em `$TD/resultados/dens/um_modelo/` (`selecao.json`, `confere.json`, `lido.json`, `avalia_teste2.json`, `relato_teste2.json`), e os de 12.7, na mesma pasta, com o prefixo `emenda_`. O diagnóstico da destilação, o efeito dos dados e as combinações exploratórias de 12.6 (d) e (e) foram calculados à parte, só com as previsões fora-do-fold dessa pasta e as da segunda execução.
