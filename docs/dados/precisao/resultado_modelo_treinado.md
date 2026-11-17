# Modelo treinado: proteína só pela foto, sem informar porção (09/11/2026)

**Pergunta:** um modelo **treinado** (e não um LLM em nuvem) consegue sugerir a proteína só pela foto e atingir a meta de **≥ 70% dos pratos dentro de ±20% ou ±5 g**?
**Resposta curta: não, com os dados que existem hoje.** Com higiene de dados rigorosa, o melhor modelo só de imagem fica em **~48–52%** no `teste` (280 pratos), abaixo do LLM (55%). Profundidade (RGB-D) não ajudou. O melhor resultado foi o **híbrido** "LLM identifica e dá a densidade, modelo treinado dá o peso", com **~60% no teste (59,6%) e 60–62% no n5k v1**: melhora o LLM em 5 pontos, mas ainda fica 10 pontos abaixo da meta. A comida brasileira em foto que não é zenital **quebra** o modelo de peso (Liv Up 3%). **Atualização (seção 7):** treinar também com os quadros laterais do Nutrition5k deixa o modelo robusto ao ângulo (foto lateral 44% → 49%) e leva o Brasil de 22% para 36% (Liv Up 3% → 26%), mas o híbrido continua em ~59–60%, longe de 70%. **Atualização (seção 8, premissa de uma única foto de cima):** com 8 orientações da foto, resolução maior, ajuste fino parcial e média de modelos, o modelo **só de imagem sobe de 50% para 61%** no teste (IC 55 a 67; +11 pontos sobre o anterior, IC +5 a +17) e alcança o LLM (55%). O erro de massa cai de 22% para 17%. O **híbrido escolhido fica em 63%** (IC 57 a 69; formas equivalentes dão de 62% a 66%), sem diferença segura para os 59,6% de antes. **A meta de 70% continua não atingida**, e o erro que sobra está mais na densidade (que alimento é) do que no peso. Cerca de metade dos pratos pode ser sugerida sozinha com ~70% de acerto, mas essa metade é, na maior parte, de pratos com pouca proteína. Fora da distância fixa do Nutrition5k o modelo erra a escala, a menos que seja treinado com escala aleatória ou com o prato enquadrado num guia. **Atualização (seção 9, densidade de proteína):** prever a fração da massa do prato que é cada fonte de proteína, somar a recuperação de pratos parecidos e um terceiro backbone melhora a densidade do modelo de 67,5% para 76,1% quando a massa é a real (IC do ganho: +2,9 a +14,3). No resultado final o ganho é menor: **só imagem 61,1% → 64,6%** (IC 59 a 70; ganho não distinguível) e **híbrido 62,9% → 67,1%** (IC 61 a 72,5; ganho de +4,3, IC +0,4 a +8,6). Melhoram os pratos de 5 a 25 g; **os de mais de 25 g praticamente não melhoram** (38–46%), porque ali o erro de massa já pesa tanto quanto o de densidade. No LLM, o erro estava na parte da massa que é proteína e na fonte, não na tabela; a instrução v3 pede identificação e partes da massa em vez de gramas. **Com as estimativas v3, o híbrido fica em cerca de 70% no teste: 71,4% com a média de duas rodadas cegas independentes** (IC 66,4 a 76,4; +8,6 sobre o híbrido da seção 8, IC +3,9 a +13,2; +4,3 sobre o híbrido com v2, IC +1,4 a +7,5), 70,4% na primeira rodada sozinha e 68,6% na segunda, feita como réplica. **É a primeira estimativa pontual na meta de 70%, e não é a meta demonstrada:** o intervalo começa em 66%, a réplica sozinha ficou abaixo de 70%, são 280 pratos do Nutrition5k, o `n5k_v1` (50 pratos) dá 62%, o pipeline depende de um LLM em nuvem, e os pratos com mais de 25 g continuam em 45% e 55%, agora limitados pelo erro de massa. As duas rodadas erram nos mesmos pratos, então repetir a estimativa não melhora o resultado. **Atualização (seção 10, treino em GPU):** 15 configurações treinadas no Kaggle (2 × T4, 5,7 h) dão um modelo de massa melhor: a média de cinco backbones congelados de alta resolução baixa o erro de massa de 16,9% para 15,3% e reduz o viés nos pratos com mais de 25 g de −12% e −8% para −6% e −2%. Com ele, o **só imagem sobe de 64,6% para 69,6%** (IC 64 a 75; ganho de +5,0, IC +0,4 a +9,6), e os pratos com mais de 25 g saem de 42% para 54%. O **híbrido com o LLM v3 fica em 72,9%** (IC 67,5 a 78,2), +1,4 sobre os 71,4% de antes, sem diferença segura. Todas as formas com a densidade v3 dão de 70% a 75% no teste nas três leituras do LLM, mas **a meta de 70% continua não demonstrada**: o limite inferior do intervalo é 67,5%, e a forma de híbrido que o `calib` prefere entre todas dá 68,2%. Um modelo único que caiba no aparelho dá 62,9%, sem ganho seguro sobre os 61,1% da seção 8; o ganho está em juntar backbones grandes, no servidor. **Atualização (seção 11, conjunto de teste novo):** um segundo conjunto, o `teste2` (829 pratos do Nutrition5k nunca vistos por nenhum modelo, em 190 cadeias), foi lido **uma vez**, com o pipeline congelado e a regra escrita antes. **Híbrido com o LLM v3: 70,8%** (ponderado para as cinco faixas de proteína terem o mesmo peso; IC por cadeia de 66,3 a 75,2). **Só imagem: 69,7%** (IC 65,2 a 74,2). Os dois conjuntos juntos (1.109 pratos): 71,5% (IC 67,9 a 75,2). **Veredito pela regra: 70% não demonstrado e não refutado.** O nível de cerca de 70% se repetiu; o limite inferior do intervalo, não. No `teste2` o LLM não acrescentou nada distinguível ao só imagem (+1,1, IC −2,1 a +4,3), os pratos com mais de 25 g ficaram em 52%, e o modelo único de aparelho, em 64,5%. O `teste2` não é independente do `teste` (722 dos 829 pratos são o mesmo prato do `teste` em outro momento da montagem), e tudo continua sendo Nutrition5k. Servir o pipeline só de imagem custa cerca de 1,7 a 2,0 s por foto numa GPU T4, com 3,6 GB de pesos. **Seção 12 (desenho e pré-registro, sem resultados):** com a decisão de usar só imagem e reduzir as nove redes a um modelo (no máximo dois), o treino do Nutrition5k cresce de 1.684 para 2.281 pratos sem tocar no teste2, a próxima execução em GPU está especificada, e o teste2 pode ser lido mais duas vezes no máximo, uma por candidato congelado. A expectativa registrada é que um modelo só fique de 1 a 3 pontos abaixo das nove redes e que a meta não seja demonstrada só com o Nutrition5k. **Atualização (seção 12.6, terceira execução em GPU):** a regra pré-registrada escolheu como modelo único o **controle sem destilação** (SigLIP 2 B/16 a 224 px, a receita de antes treinada com o treino_v2, uma orientação): os alunos destilados com ajuste fino ficaram 4 a 5 pontos abaixo dele na validação cruzada, e nenhum par de modelos passou no critério. Na única leitura do teste2 ele dá **66,0%** (IC por cadeia 61,6 a 70,1): **70% não demonstrado**, 3,7 pontos abaixo das nove redes (IC −9,0 a +1,4) e 1,5 acima do mesmo modelo treinado com o treino antigo (IC −2,2 a +5,5); os pratos com mais de 25 g ficam em 48%. Ele custa 8 ms por foto numa T4 e 186 MB, contra 1,7 a 2,0 s e 3,6 GB das nove redes. Os 35% a mais de treino deram +2 a +3 pontos de validação cruzada à mesma receita. A destilação falhou por um erro de receita (os pesos das perdas novas abafaram a perda de massa) e, onde a comparação é limpa, custou 1,5 ponto. A base externa Flavoria não foi adotada (−1 ponto no Nutrition5k). Fora da regra de 12.3, um único backbone grande congelado (So400m) com duas cabeças empata com o conjunto de cinco redes na validação cruzada (71 a 72%). **Atualização (seção 12.7, emenda aprovada pelo fundador em 12/11/2026):** esse modelo foi lido no teste2, uma vez, com a definição fixada antes: **69,7%** (IC por cadeia 65,0 a 74,3; com a correção para as leituras repetidas, 64,1 a 75,4). **70% continua não demonstrado.** Ele repete o nível das nove redes (diferença pareada 0,0; IC −4,7 a +4,3) com um backbone só, a 61 ms por foto numa T4 e 859 MB, e fica 3,7 pontos acima do modelo de aparelho (IC −1,8 a +9,5; na validação cruzada, +3,2 com IC +0,4 a +6,0). É um modelo de servidor: não cabe no aparelho como está. Os pratos com mais de 25 g ficam em 54%. Foi a terceira leitura do teste2, decidida depois de ver a segunda: ele está gasto como teste de registro, e a prova da meta precisa de um conjunto novo. **Seção 13 (desenho e pré-registro da quarta execução, sem resultados):** com a decisão de usar qualquer base de treino, com a procedência registrada, entram cinco bases externas pesadas (OmniFood8K, a única com proteína; Flavoria, Food Portion Benchmark, Mensa e LeFood, só com massa). As bases de massa entram só na cabeça de massa, porque juntar o Flavoria nas duas cabeças custou 2 pontos no Nutrition5k. A regra de adoção protege o Nutrition5k (queda máxima de 1 ponto) e mede o ganho fora dele; partes de cada base ficam seguradas como teste de registro, o OmniFood8K por prato. O teste2 terá no máximo mais uma leitura. Nenhum desses conjuntos mede prato brasileiro em foto de celular: a meta do produto só pode ser julgada num conjunto brasileiro pesado, que ainda não existe. **Seção 12.8 (calorias, carboidratos e gordura, só validação cruzada):** com o mesmo tipo de critério da proteína, o modelo de servidor acerta 70,5% das kcal (piso de 50 kcal), 78,6% dos carboidratos (piso de 10 g) e 68,4% da gordura (piso de 5 g), e o modelo de aparelho, 65,0%, 75,0% e 65,6%. Nas refeições grandes a estimativa fica cerca de 20% abaixo do real (carboidrato acima de 50 g: 46% de acerto). Dá para mostrar kcal e carboidrato como estimativa, não como número para dose de insulina. **Atualização (seção 13.11, quarta execução):** com cinco bases externas pesadas, a regra pré-registrada só admitiu a massa do OmniFood8K na cabeça de massa. O modelo de servidor escolhido dá **69,2% no teste2** (IC por cadeia de 65,0 a 73,7; na quarta e última leitura, com a correção para quatro leituras, 64,1 a 74,7): **70% não demonstrado**. Fica no mesmo nível do D1 e das nove redes, com 57% nos pratos acima de 25 g: o maior número do estudo nessa faixa, sem diferença segura. Nas cenas de pratos chineses nunca vistos do OmniFood8K, os três modelos lidos ficam entre 59% e 66%. Cada base externa melhora muito a massa do próprio domínio (Flavoria de 52% para 67% a ±20%), mas não transfere para os outros e custa de 2 a 3 pontos no Nutrition5k. A escala da foto é o limite que mais aparece: quem fotografa de mais perto muda a massa em 3 a 5 vezes. A próxima alavanca é enquadrar pelo prato (o círculo-guia), e o que falta para julgar os 70% é um conjunto brasileiro pesado de celular. O teste2 está encerrado. kcal e carboidrato no teste2: 71% e 84%. **Seção 14 (desenho da quinta execução, sem resultados):** enquadrar as fotos de treino pelo prato, como o círculo-guia do app, em todos os domínios e sem constante por domínio; consertar a receita do modelo de aparelho (cabeça de frações sem gradiente para o backbone); e ajustar em parte o backbone grande só com a perda de rótulos. Não há mais teste de registro do Nutrition5k: as escolhas saem da validação cruzada e de uma segunda leitura das partes seguradas externas, e os 70% do produto só vão ser julgados no conjunto brasileiro. **Atualização (seção 14.10, quinta execução):** enquadrar as fotos de treino pelo prato e treinar a massa com as cinco bases externas resolveu a escala fora do Nutrition5k. Sem constante nenhuma, a massa a ±20% foi de 2–35% para 48–66% em Flavoria, FPB, Mensa e LeFood, e as bases deixaram de custar no Nutrition5k. Mas o recorte tira cerca de 3 pontos do Nutrition5k (71% → 68%), porque a câmera fixa dele dava a escala de graça. A regra pré-registrada protege o Nutrition5k com a foto inteira e manteve o servidor de 13.11 (R3×H0). O modelo com recorte e as cinco bases (68% no Nutrition5k, massa usável fora dele) é o candidato mais parecido com o app, a ser julgado no conjunto brasileiro. A receita corrigida do aparelho conserta a massa, mas não passa de 68%. O ajuste fino do backbone grande falhou na aceitação por 0,0009. **Seção 15 (desenho da sexta execução, sem resultados):** a partir de agora o Nutrition5k é julgado no recorte, contra o modelo do app (68,3%). Entram: mais orientações e rotas combinadas na inferência; o ajuste fino do backbone grande no recorte, com parada pela validação interna; um modelo de aparelho treinado no recorte; e a média entre servidor e aparelho, cada um pelas regras de antes. Não há teste de registro do Nutrition5k, e os 70% continuam a ser julgados no conjunto brasileiro. **Atualização (seção 15.8, sexta execução):** no recorte, o modelo do app passa de 68,3% para **70,7%** na validação cruzada do Nutrition5k (+2,4; IC +0,3 a +4,6), com 52% acima de 25 g. O ganho vem da inferência (4 orientações e uma rota que junta a proteína direta), não de treino novo: o ajuste fino do backbone grande não teve efeito, e a média com o modelo de aparelho deixou de ajudar. Nas partes seguradas nunca usadas para escolher, a massa sem constante acerta 45% a 69% dos pratos a ±20% (antes, de 1% a 37%), e a proteína nos pratos chineses novos fica em 65%. Custa 262 ms por foto numa T4. O modelo de aparelho passa a ser um treinado no recorte (67%, 10 ms). 70% não está demonstrado em lugar nenhum; a próxima alavanca é o conjunto brasileiro com o guia e o diâmetro do prato. **Seção 16 (critério de parada):** a meta só conta como atingida em dados públicos com um teste de registro novo, de bases com proteína derivada de pesagem: pelo menos 600 pratos e 100 unidades independentes, seguradas por hash antes de qualquer previsão. O limite inferior do intervalo tem de passar de 70%, e após k leituras vale a correção de Bonferroni, com no máximo 3 leituras. Isso exige acerto verdadeiro de 73% a 75%, contra cerca de 65% na medida mais próxima de hoje. Se em 3 ciclos nada passar, fica escrito que dados públicos não bastam e que é preciso a coleta brasileira. **Seção 16.5:** essa saída já foi atingida no primeiro ciclo. Nenhuma base pública forma o teste de registro (o ACETADA tem a proteína medida, mas a foto é oblíqua e da bandeja inteira), então **com dados públicos não se consegue demonstrar os 70%**; a demonstração exige o conjunto brasileiro. **Seção 17 (desenho da sétima execução):** só melhora, sem demonstração, mirando os pratos acima de 25 g e a densidade, com o ACETADA, as receitas e o FoodBD em cabeças sobre features congeladas. **Atualização (seção 17.8, sétima execução):** nenhuma das três bases melhorou o modelo. O ACETADA não mexe na faixa acima de 25 g (+0,2 no Nutrition5k, −0,8 no OmniFood8K), as receitas pioram um pouco, e o recorte do FoodBD foi reprovado na conferência a olho. Pelo objetivo pré-registrado, que pesa a faixa alta, o app passa a usar uma cabeça de densidade maior e a rota que mistura as três estimativas: 71,4% e 55,6% acima de 25 g na validação cruzada do recorte. Mas na parte segurada do OmniFood8K fica igual (64,6% contra 64,9%), e lá a faixa acima de 25 g cai. Sem o guia (FoodBD, foto inteira) a proteína sai quase o dobro da real; o guia ou o tamanho do prato é condição para estes números valerem no app. **Seção 16.6 (emenda, antes de qualquer previsão):** o DFood-TJ (CC0; refeições hospitalares chinesas, cada alimento pesado, um prato fotografado de cima) passa a ser o teste de registro em dados públicos e não entra em treino. São 711 refeições de 89 pacientes, com o visor da balança mascarado e conferido a olho; em três faixas com o mesmo peso (até 15, 15–25 e acima de 25 g) tem 75 / 417 / 219 refeições. O candidato é o pipeline do app (S7), em uma leitura. Passar exige limite inferior de 70%, isto é, acerto verdadeiro perto de 75%. A expectativa é de 55% a 66%. **Atualização (seção 16.7, a leitura do DFood-TJ):** **meta não atingida.** O pipeline do app acerta **26,9%** (IC por paciente de 22,1 a 31,9), mas o número mede sobretudo um defeito da entrada. A máscara do visor marcou a balança inteira, que fica embaixo do prato, e cobriu de cinza mais de um quarto da janela em 37% das fotos; nelas o acerto é de 3%. A conferência a olho pegou o defeito antes da leitura, mas a regra registrada não tinha saída para ele. Nas 349 refeições com até 10% de cinza, o acerto fica em 38% a 40%, com a proteína e a massa bem abaixo das reais. É 15 a 28 pontos abaixo do que eu esperava, e o sinal aponta para a escala: um prato maior que os do treino continua sendo lido como um prato comum, mesmo com o guia. A demonstração dos 70% com dados públicos continua fora de alcance; a alavanca é o tamanho do prato, medido no conjunto brasileiro. **Seção 16.8 (emenda, antes do kernel novo):** uma segunda leitura (2 de 3, Bonferroni de 97,5%) com a máscara só do visor, que poupa o prato, e o mesmo candidato. Espero de 32% a 45%: o veredito não deve mudar. O objetivo é diagnóstico: separar o erro de escala do erro de densidade. **Atualização (seção 16.8, leitura 2 de 3):** com a máscara só do visor, a comida voltou à vista, mas o visor colado ao prato oval ficou legível em 41 das 247 fotos de 2024–2025 (16,6%), e o período saiu pela regra dos dígitos. Em 457 refeições de 2020–2023 (59 pacientes), o pipeline do app acerta **44,1%** (IC de 97,5% de 37,0 a 51,3): **meta não atingida**, no topo da faixa esperada (32% a 45%). Os pratos que a máscara antiga cobria passaram de 3% para 49%. O diagnóstico depois da leitura mostra dois erros que se compensam em parte: a massa sai cerca de 42% abaixo da pesada (refeições de ~500 g) e a densidade sai cerca de 50% acima. Não é só escala, é também a composição de uma cozinha nova. Com dados públicos, os 70% não se demonstram. **Seção 18 (desenho, sem resultados):** um pipeline geométrico. Ele segmenta e identifica cada alimento (OWLv2 com máscaras do SAM 2.1), tira a escala do diâmetro do prato (no DFood-TJ, pelos ovos com casca; no app, informado uma vez pela usuária), calcula a massa como área × k por grupo (calibrado só na validação cruzada do Nutrition5k) ou área × altura × densidade de tabela (FAO/INFOODS), e a proteína como massa × tabela. A leitura 3 do DFood-TJ só acontece se quatro condições sem rótulo passarem. Não muda o veredito: o que ela mede é se a escala pelo prato conserta a massa (−42% hoje). A densidade (+57%) a geometria não conserta sozinha. **Atualização (seção 18.11, nona execução):** a segmentação (OWLv2 com o SAM 2.1, guiado por caixas) reprovou na conferência a olho nas três bases: 13 de 40 pratos bons no DFood-TJ, 10 no Nutrition5k e 23 no FPB, contra os 36 exigidos. O SAM devolve o prato inteiro como arroz, não acha alimentos, e no Nutrition5k conta o prato vazio como comida. Pela regra pré-registrada, a geometria não entra: não houve calibração, escolha nem leitura, e o DFood-TJ guarda a leitura 3. A escala pelos ovos funcionou sem rótulo (prato de 23,6 cm, dispersão de 5%). O gargalo é a segmentação, não a escala. **Seção 18.12 (emenda, antes da décima execução):** um segmentador treinado em comida (FoodSeg103 ou UECFoodPix, com critérios de aceitação e o mapa de classes escritos antes), o branco do prato medido foto a foto no anel da borda, todos os recipientes do FPB e amostras novas para a conferência (semente 20261118). Espero de 15% a 25% de chance de o DFood-TJ e o Nutrition5k passarem juntos. Se um dos dois reprovar, a rota geométrica fica parada até a coleta brasileira.

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



## 14. Quinta execução: escala da foto e receitas corrigidas (13/11/2026)

**Desenho e pré-registro. Nada aqui tem resultado da quinta execução.** A fundadora pediu para continuar melhorando o modelo e delegou as decisões de treino. O que a quarta execução deixou (13.11):
- o modelo de servidor está em 69 a 70% no Nutrition5k e não sai disso;
- a massa fora do Nutrition5k, sem constante por domínio, acerta só de 1% a 35% dos pratos a ±20%;
- cada base externa só ajuda o próprio domínio;
- as duas receitas de ajuste fino com frações estragaram a massa, por um erro de especificação meu.

### 14.1 O que ainda pode julgar uma melhora (dito antes de tudo)

- **Não existe mais teste de registro do Nutrition5k.** O teste2 foi lido quatro vezes e está encerrado: nesta execução o kernel **nem gera previsões para ele**.
- No Nutrition5k, sobra só a validação cruzada agrupada por cadeia do treino_v2 (**N**). Ela escolhe, mas não demonstra nada, e cada escolha entre muitas candidatas a deixa um pouco otimista.
- Fora dele: a CV das bases externas (**O** no OmniFood8K, **M** em Flavoria, FPB e Mensa, como em 13.2) e uma **segunda leitura** das partes de registro do OmniFood8K, Flavoria, FPB e LeFood, para no máximo dois candidatos novos congelados (14.7). É segunda leitura, com intervalo de 97,5%; essas partes não são mais virgens.
- **A meta de 70% do produto só vai ser julgada no conjunto brasileiro pesado, de celular, com o guia** (`docs/coleta-brasil.md`), quando ele existir. Nenhum número desta seção é esse veredito.

### 14.2 As alavancas, na ordem que os dados sustentam

A ordem proposta pelo coordenador era 1) escala da foto, 2) capacidade do servidor, 3) modelo de aparelho, 4) orientações e pares, 5) massa por domínio. **Mudo uma coisa:** o conserto do modelo de aparelho vem antes do servidor.
- O conserto do aparelho é um erro de receita conhecido, custa 64 minutos e tem efeito quase certo sobre a massa.
- O ajuste fino parcial do So400m custa 150 minutos. Nenhum ajuste fino ganhou de um backbone grande congelado na massa até hoje (seção 10). O P1 já piorava na época 5, antes de a perda de frações dominar.

| Ordem | Alavanca | Evidência a favor | Evidência contra |
|---|---|---|---|
| 1 | **Enquadrar pelo prato** em todos os domínios | Fora do Nutrition5k o erro dominante é de escala: sem constante, a massa acerta 1–35% a ±20%, e com a constante, 31–67% (13.11). No Mensa, outro enquadramento mudou a escala em 3 a 5 vezes. Em 8.7, o recorte pelo prato não custou nada no critério do Nutrition5k | O tamanho do prato continua desconhecido (8.7, item 4): o recorte troca "distância da câmera" por "tamanho do prato" |
| 2 | **Aparelho: cabeça de frações sem gradiente para o backbone** (F4) | A receita do U0 sem frações funciona (F3 passou na aceitação); a cabeça de frações funciona nas features congeladas (D1) | F3 e U0 ficam em 67,7–68,0%: o teto da rede pequena pode ser esse |
| 3 | **Servidor: ajuste fino parcial do So400m só com a perda de rótulos, massa com peso próprio** (P2) | O P1 falhou pela perda de frações; a massa congelada do So400m é a melhor que temos (0,209) | Ajuste fino nunca ganhou de backbone grande congelado na massa (seção 10) |
| 4 | Recorte também no aparelho (F4c) e bases externas no aparelho (F5) | O app vai ter o guia; o F3 passou na aceitação | O F3 não ganhou do U0 em N |
| — | 8 orientações no servidor | +0,7 na CV do D1 | **Fica de fora:** a regra de 12.3 (item 2) pede 1,0 ponto, e custa 8 vezes o tempo (532 contra 61 ms) |
| — | Média do servidor com o U0 | +1,3 na CV (12.7) | **Fica a regra de 12.1:** só entra com 2,0 pontos e diferença segura. É reavaliada com os modelos novos |
| — | Cabeça de massa condicionada ao domínio | Cada base ajuda o próprio domínio | **Fica de fora:** o app teria de escolher um domínio, e "foto de celular brasileira com guia" não está entre os que existem. O recorte pelo prato é a forma de tornar os domínios parecidos sem escolher |
| — | Tamanho do prato em pixels como entrada (1b) | | **Fica de fora:** com o guia, o prato ocupa sempre a mesma fração do quadro, então essa entrada seria constante no app. Ela só informa a distância da câmera nas bases de treino, que é justamente o que o recorte quer apagar |

### 14.3 Enquadramento pelo prato

**Definição, igual à do app.** O guia do app é um círculo com diâmetro de **80% do lado menor do quadro** (o que `docs/coleta-brasil.md` já pede: "prato ocupando cerca de 80% da largura"). No treino, cada foto vira uma janela 4:3 (a entrada do modelo, 512×384):
- centrada na caixa da refeição;
- com altura igual ao maior lado da caixa dividido por 0,80;
- preenchida com a cor mediana da borda onde a janela sai da foto.

**Na inferência do app não há detector:** a janela é o próprio guia. O detector só serve para fazer as fotos de treino parecerem fotos tiradas com o guia.

**Detector (etapa 0, um kernel curto antes da execução principal).**
1. **OWLv2** (`google/owlv2-base-patch16-ensemble`, Apache 2.0, pesos públicos), em meia precisão, imagem a 960 px.
   - Consultas: "a plate of food", "a bowl of food", "a food tray", "a lunch box", "a food container", "a plate", "a bowl".
   - Caixas com nota de 0,15 ou mais e área de 2% ou mais da foto, depois de NMS a 0,5.
   - **A caixa da refeição é a união dessas caixas.** Numa bandeja com vários pratos, é a refeição inteira, que é o que a pessoa poria dentro do guia.
2. **Sem caixa:** reserva clássica com OpenCV. Maior elipse ajustada às bordas (Canny e `fitEllipse`), com área de 10% a 95% da foto e razão de eixos de 0,5 ou mais.
3. **Sem elipse:** a foto inteira, com a marca `sem_prato`. A taxa de `sem_prato` sai por domínio.
4. No Nutrition5k, ao lado sai a concordância com a caixa da profundidade de 8.7 (`zen_bbox.csv`): IoU mediana e % com IoU de 0,7 ou mais.

Custo: cerca de 25,4 mil fotos (Nutrition5k 3.110, OmniFood8K 15.741, Flavoria 1.967, FPB 2.418, Mensa 200, LeFood 811, SNAPMe 1.163), cerca de **40 minutos de GPU**, ou 20 de relógio nas duas T4. Grava, por domínio, `caixas_r5_<dominio>.csv`: id, caixa, nota, método (`owlv2`, `elipse`, `sem_prato`) e a versão do detector. As tabelas das bases não comerciais ficam fora do repositório.

**Conferência a olho, antes da execução principal, sem rótulo nenhum.**
- A etapa 0 grava folhas de contato com 40 fotos sorteadas por domínio (semente 0), com a caixa desenhada e o recorte ao lado. Só na saída do Kaggle e no scratchpad, nunca no repositório.
- Eu classifico cada uma em **certa** (a refeição inteira dentro, pouco além dela), **cortada** (parte da comida fora) ou **errada** (outra coisa).
- **Um domínio passa com 36 ou mais certas em 40 e no máximo 1 errada.**
- Domínio que não passa entra nas configurações com recorte usando a foto inteira, marcado, e fica fora da comparação recorte × foto inteira daquele domínio.
- **Se o Nutrition5k não passar, a alavanca 1 é cancelada.** O tempo dela vai para a próxima da fila.
- A execução principal só começa depois desta conferência.

**Recorte nas features congeladas.**
- Nutrition5k com 8 orientações. A orientação 0 é o recorte exato; nas outras 7 entra um erro de enquadramento sorteado (escala de ±10% e deslocamento de ±5% do lado), que imita a pessoa que não acerta o guia (8.7, linha d).
- Bases externas com 2 orientações; OmniFood8K com os três quadros, como em 13.4.
- A previsão de 1 orientação usa só o recorte exato.

**Sem viés de massa por domínio nas configurações com recorte.** Esta é a hipótese da alavanca: com o prato enchendo o quadro, os domínios ficam comparáveis sem constante. As configurações sobre a foto inteira continuam como na quarta execução.

**Emenda a 14.3 (13/11/2026, antes de a etapa 0 rodar e antes de qualquer resultado da quinta execução):** a regra pré-registrada (união das caixas das sete consultas) foi testada no ensaio em CPU com o detector real em 22 fotos, sem rótulos de massa ou proteína. A consulta "a food tray" dispara (escore 0,26–0,31) na plataforma da balança do Nutrition5k e na bandeja marrom do FPB, e a união vira quase a foto inteira: a caixa difere da caixa só de prato e tigela (IoU < 0,7) em 9 de 10 fotos do Nutrition5k e em 12 de 12 do FPB. Isso derrota o propósito de 14.3 (enquadrar o prato como o círculo-guia do app). **A regra passa a ser a união das caixas só das consultas de prato e tigela** ("a plate of food", "a bowl of food", "a plate", "a bowl"; índices 0, 1, 5, 6), mantidos o limiar, a área mínima, o NMS e os recursos à elipse e a `sem_prato`. O kernel grava as caixas brutas de todas as consultas, então a regra original continua reconstruível e será relatada ao lado (taxa de `sem_prato` e IoU com a caixa da profundidade nas duas regras). A conferência a olho de 14.3 (36 de 40 certas, no máximo 1 errada) é feita nas folhas da regra emendada; se uma base falhar, ela usa a foto inteira, como já previsto. Consequência conhecida: em bandejas com vários pratos (Mensa, FPB), a caixa é a união dos pratos, não um prato só.

### 14.4 Receitas corrigidas

**Aparelho, F4: a receita do U0 sem mudança, mais uma cabeça de frações que não manda gradiente para o backbone.**
- A cabeça é uma MLP como a do H0 (oculta de 512, GELU, dropout de 0,3; 22 saídas, mais 10 logits) sobre `stop_gradient(features)`.
- Tem o otimizador dela (AdamW, passo de 1e-3, decaimento de 1e-2), as três perdas de 12.3 e o peso por faixa, que só agem nela.
- A rede e a cabeça linear de 22 saídas treinam **exatamente** como no U0. A massa do F4 tem de sair igual à do U0, dentro do ruído; se não sair, é erro de implementação.
- Rotas de proteína: as cinco de 12.3, com a densidade das frações vindo da cabeça nova.

**Aparelho, F4c:** o F4 sobre o recorte pelo prato. No aumento de dados, o erro de enquadramento de ±10% entra no lugar da escala aleatória de 0,7 a 1,3.

**Aparelho, F5:** o F4 com as bases externas do F3: só massa, viés por domínio, foto inteira. A cabeça de frações treina só com o Nutrition5k.

**Servidor, P2: So400m a 384×512, 4 últimos blocos, só com a perda de rótulos e a massa com peso próprio,** mais a cabeça de frações sem gradiente do F4.
- **Pesos por alvo** (o erro de 13.7 foi a massa pesar 1/22): massa 6, proteína direta 3, densidade 3, kcal 0,5, gordura 0,3, carboidrato 0,3, cada massa e proteína por grupo 0,15. A perda é a média ponderada; a massa fica com 39% do peso.
- O resto da receita é a do P1 (13.7): 6 épocas, passo de 2e-5 no topo com decaimento de 0,75, média móvel de 0,99, escala aleatória, foto inteira, só o treino_v2.

**Critério de aceitação (o de 13.7, sem mudança).** A média dos folds do RMSE de log-massa na última época não pode passar da época 0. Referências: 0,2545 para a rede B/16; 0,2167 para o P2. Quem falha não concorre.

### 14.5 Configurações, em ordem de prioridade

| Prioridade | Nome | O que é | Bases | GPU (min) |
|---|---|---|---|---|
| 0 | etapa 0 | Detector e folhas de contato (14.3); kernel separado, antes da execução principal | todas as fotos | 40 |
| 1 | `gpu5 feat so400m prato` | Features do So400m sobre o recorte: Nutrition5k com 8 orientações e bases externas | | 60 |
| 1 | `gpu5 ridge so400m prato` (**R0c**), `+ok` (**R1c**), `+ok+fpb` (**R2c**), `+omni` (**R3c**), `+tudo` (**R4c**) | Cabeças de massa sobre o recorte, sem viés por domínio | como R0–R4 | 5 |
| 1 | `gpu5 fracoes so400m prato` (**H0c**), `+omni` (**H1c**) | Cabeças de frações sobre o recorte | como H0, H1 | 8 |
| 2 | `gpu5 ft siglip2_b16_224 r2 fracoes-sg` (**F4**) | 14.4 | treino_v2 | 64 |
| 3 | `gpu5 ftparcial so400m rotulos` (**P2**) | 14.4 | treino_v2 | 150 |
| 4 | `gpu5 ft siglip2_b16_224 r2 fracoes-sg prato` (**F4c**) | 14.4 | treino_v2 | 64 |
| 5 | `gpu5 ft siglip2_b16_224 r2 fracoes-sg +tudo` (**F5**) | 14.4 | as cinco (massa) | 96 |

- **As combinações sobre a foto inteira (R0–R4 × H0–H2) não são refeitas.** As previsões da quarta execução valem: a paridade com o D1 foi conferida duas vezes, com correlação de 1,00000. A referência do Nutrition5k "nos dois jeitos" é o A0 (foto inteira, da quarta execução) contra o **A0c = R0c×H0c** (recorte, desta).
- **Divisão das GPUs:**
  - GPU A: features e cabeças do recorte (cerca de 75 min), depois o P2 (150). Total: cerca de 225 min.
  - GPU B: F4, F4c e F5. Total: cerca de 224 min.
  - Com a etapa 0, a execução toda fica em cerca de 490 minutos de GPU e 4,1 h de relógio.
- **Ordem de corte se a cota faltar:** F5, depois F4c, depois P2. As features e cabeças do recorte e o F4 não são cortados.
- **Nada de decisão dentro do kernel.** Nenhuma configuração depende do resultado de outra, e **nenhuma previsão do teste2 é gerada**.

### 14.6 O que o kernel grava

- **Tudo o que a quarta execução gravava**, menos o teste2: `oof`, `oof_o1`, `oof_o4` (e `oof_ff*`, `oof_dens_ff*`); `extcv__*` e `extreg__*` crus, para todos os domínios e todas as configurações; ids, folds, hashes; vieses por domínio quando houver.
- **Para o recorte:** a tabela de caixas usada (com o sha256), o método por foto, os parâmetros da janela (0,80; 4:3; erro de enquadramento) e a versão do detector.
- **Para servir os vencedores:**
  - as cabeças finais (ridge e frações) de todas as configurações;
  - os pesos em meia precisão do F4, F4c, F5 e P2, com a cabeça de frações sem gradiente separada;
  - os arquivos de servir, com a conferência de remontagem de **todas** as cabeças;
  - medidas de serviço (T4 e CPU, 1 orientação) de cada configuração que possa ser escolhida.
  - O arquivo de servir das configurações com recorte descreve a janela do guia, que é o que o app aplica. **O detector não faz parte do modelo servido.**
- **Procedência no manifesto de cada modelo:** `treinado_com` (bases, listas, sha256, alvos, licença, `uso_declarado`) e `so_comercial_ok`. A licença do detector (Apache 2.0) vai à parte, porque ele só prepara os dados de treino.

### 14.7 Regra de escolha (R14), na CPU, só com validação cruzada

**Candidatos a servidor:**
- as 15 combinações da quarta execução (foto inteira);
- as 10 do recorte (R0c a R4c × H0c, H1c);
- o P2 com as próprias cabeças;
- a massa do P2 × a densidade do H0.

**Candidatos a aparelho:** F4, F4c e F5, os que passarem na aceitação. Sempre com 1 orientação, e com rota e orientações pelos itens 1 e 2 de 12.3.

**Medidas.** N e o critério acima de 25 g, como em 13.9. **O e M agora contam crus, sem a constante por domínio,** porque o app não tem constante nenhuma: a pessoa não diz em que "domínio" está. Os números com constante saem ao lado.

**Servidor.**
1. **Guarda:** N a no máximo 1,0 ponto do A0 (71,29%), e o critério acima de 25 g a no máximo 2,5 do dele (51,94%).
2. **Nota:** S14 = (N + O cru + M cru) / 3.
3. **S5 = a admissível de maior S14.** Empate (menos de 0,5): primeiro a que usa o recorte, porque é a entrada do app; depois, menos bases não comerciais; depois, menos bases.
4. Ao lado sai sempre a melhor admissível só com dados liberados.

**Aparelho.** Ad5 = a aceita de maior N, **se** ela ficar 1,0 ponto ou mais acima do U0 (68,04%). Senão, o aparelho continua o C1.

**Par.** S5 com o modelo de aparelho vigente (média em log da proteína). Só se a CV subir 2,0 pontos ou mais sobre o S5 **e** o limite inferior de 95% da diferença pareada ficar acima de zero.

**Segunda leitura das partes de registro** (OmniFood8K com proteína e 12.8; Flavoria, FPB e LeFood só com massa; SNAPMe com correlação de postos):
- só para o S5, se ele não for o A0 nem o S de 13.11, e para o Ad5, se houver;
- no máximo dois modelos, numa execução só, com marca;
- números crus e com constante; intervalos de 95% e de 97,5%;
- a parte do Mensa sai só como "teste de escala" (fotos de perto de um participante) e não julga nada;
- comparação pareada com o A0 e o S de 13.11, que já foram lidos;
- **antes da leitura,** o item correspondente de 14.10 é escrito.

**O que esta regra não mostra.** Escolher entre 28 candidatos de servidor na mesma CV deixa o vencedor otimista, talvez em 1 ponto. A segunda leitura das partes de registro é a única conferência fora da escolha, e é fora do Nutrition5k. Nenhuma das duas coisas mede comida brasileira.

### 14.8 O que eu espero

| Alavanca | Expectativa | Chance |
|---|---|---|
| **Recorte, A0c contra A0** | N de −1,0 a +0,5. M cru de 13% (A0: Flavoria 35, FPB 4, Mensa 2) para 30–50%. O cru de 53% para 50–58% | Passar na guarda: 65%. Fazer M cru subir 15 pontos ou mais: 60% |
| Bases de massa com o recorte (R1c, R4c) | Com escala comparável, o custo em N deve cair (era de 2 a 3 pontos sem recorte) | Alguma combinação com base externa passar na guarda: 40% |
| F4 | N de 68 a 69,5% (rota com a densidade das frações) | Passar na aceitação: 90%. Virar o Ad5: 25% |
| P2 | Massa de 0,217 para 0,205–0,22 com 1 orientação; N de 70 a 72% | Passar na aceitação: 50%. Ser o S5: 20% |
| F4c, F5 | N de 67,5 a 69%; F5 com M melhor | Virar o Ad5: 10% cada |
| Par S5 + aparelho | +0,5 a +1,5 | Passar na regra: 15% |
| **Resultado de conjunto** | O S5 é um modelo com recorte em 55% dos casos | O melhor N passar de 72%: 20% |

- **A aposta principal não é subir o Nutrition5k, é tornar a massa usável fora dele sem constante.** Se o recorte não fizer M cru subir, o problema da escala fica para o conjunto brasileiro (prato e distância conhecidos) e para pedir na coleta o diâmetro do prato, que `coleta-brasil.md` já anota.
- **Risco honesto do recorte:** o detector pode errar diferente em cada domínio, e "a refeição inteira" numa bandeja não é "um prato" no guia. A conferência a olho pega o grosso; o resto aparece como diferença entre domínios em M.

### 14.9 Conferência a olho do detector (etapa 0), antes da execução principal

A etapa 0 rodou no Kaggle (29,5 min nas duas T4; OWLv2 `google/owlv2-base-patch16-ensemble@cfd3195ba4ea`). Conferi a olho as folhas da **regra emendada** (união das caixas de prato e tigela; emenda de 14.3): 40 fotos por base, com a caixa e a janela recortada lado a lado. Nada de rótulo de massa ou proteína foi aberto.

**Como julguei.** Pelo que o modelo recebe: a janela.
- **Certa:** a janela contém a refeição inteira (todo item de comida que entra na massa do rótulo; bebida não conta), e o prato ou recipiente é o objeto principal.
- **Cortada:** um item da refeição fica parcial ou totalmente fora da janela.
- **Errada:** a janela está centrada em outra coisa.

A caixa às vezes fica dentro do fundo do prato ou deslocada (Nutrition5k 31 e 32, Flavoria 27 e 30). A margem da janela (o lado maior da caixa ocupa 80% da altura) recupera a refeição, e eu contei como certa.

| Base | Certa | Cortada | Errada | Veredito | Recurso, regra emendada: elipse / `sem_prato` (todas as fotos) | `sem_prato` na regra original de sete consultas | Caixa da regra emendada difere da original (IoU < 0,7) |
|---|---|---|---|---|---|---|---|
| Nutrition5k | 40 | 0 | 0 | **passa** | 2 / 3 (0,1%) | 0% | 8,8% |
| Flavoria | 40 | 0 | 0 | **passa** | 0 / 0 | 0% | 0,1% |
| LeFood | 40 | 0 | 0 | **passa** | 0 / 0 | 0% | 0% |
| FPB (de cima) | 40 | 0 | 0 | **passa** | 28 / 0 | 0% | 80,0% |
| OmniFood8K | 40 | 0 | 0 | **passa** | 13 / 0 | 0% | 2,8% |
| Mensa | 36 | 4 | 0 | **passa, no limite** | 9 / 4 (2,0%) | 0% | 24,1% |
| SNAPMe | 37 | 1 | **2** | **reprovada** | 123 / 51 (4,4%) | 4,0% | 22,6% |

- **Mensa.** As quatro cortadas:
  - duas bandejas em que um prato ou um pão na borda fica fora da janela;
  - dois potes de vidro em que a elipse caiu no meio ou num canto do pote.
  - É a consequência anunciada na emenda: em bandeja, a caixa é a união dos pratos, e pão, fruta e iogurte fora de prato só entram pela margem da janela.
- **SNAPMe.** As duas erradas:
  - uma elipse no canto da mesa, com o copo da bebida fora;
  - uma caixa num pilão branco ao fundo, com a caneca no centro.
  - **O SNAPMe fica com a foto inteira nas configurações com recorte.** Ele só serve de avaliação fraca, então isso não muda nenhum treino.
- **A amostra de 40 sobrerrepresenta os casos de recurso** (o Mensa teve 8 de 40 fotos por elipse ou `sem_prato`, contra 6,5% no total): a conferência foi mais dura que um sorteio simples.
- **Nutrition5k, concordância com a caixa da profundidade de 8.7** (1.874 pratos com a caixa gravada):

| Regra | IoU mediana | Pratos com IoU de 0,7 ou mais |
|---|---|---|
| Original, sete consultas | 0,952 | 86,1% |
| **Emendada, só prato e tigela** | **0,956** | **94,2%** |

  Com a caixa da profundidade refeita no kernel (2.272 pratos), a regra original dá 0,953 e 86,7%. A emenda aproxima a caixa do prato que a profundidade vê; a consulta de bandeja era o que a afastava.
- **Decisão gravada em `eval/treino/kaggle/listas/conferencia_r5.json`:** `"conferida": true`, regra `so_prato` (consultas 0, 1, 5, 6), `reprovados: ["snapme"]`.
- As cópias das folhas com o veredito por foto ficam no scratchpad (`conferencia_r5/`), fora do repositório: as fotos são de terceiros.

### 14.10 Resultado da quinta execução (13/11/2026)

A execução principal rodou no Kaggle dentro do orçamento, e nada foi cortado. A ingestão (`ingere_kaggle.py --so-confere --caixas`) casou todas as tabelas de recorte com as da etapa 0, menos a do SNAPMe. **Isso é o esperado, e confirmo:** o registro do kernel diz "snapme reprovada na conferência a olho: foto inteira | tabela None" e "recortes de snapme: 1163 fotos (1163 com a foto inteira)". O SNAPMe foi reprovado em 14.9, entrou com a foto inteira e não usou tabela de caixas nenhuma. Nas outras seis bases, os recortes são os da regra emendada (só prato e tigela), com os métodos de 14.9. Script: `eval/treino/r5_avalia.py`.

**a) Aceitação (13.7).**

| Configuração | RMSE de log-massa, época 0 → última | Aceita? |
|---|---|---|
| F4 (U0 + cabeça de frações sem gradiente) | 0,2545 → 0,2315 | **sim** |
| F4c (o mesmo, no recorte) | 0,2638 → 0,2329 | **sim** |
| F5 (F4 + bases externas) | 0,2545 → 0,2303 | **sim** |
| P2 (So400m, 4 blocos, só rótulos) | 0,2167 → 0,2176 | **não**, por 0,0009 |

**A regra é a regra: o P2 não concorre.** Para registro, ele também não ganharia: a massa dele na CV com 8 orientações (0,2099) fica igual à da ridge congelada (0,2090), e o N dele (66,4% com as próprias cabeças; 69,0% com a densidade do H0) fica abaixo do A0.

**b) Regra R14, só com validação cruzada.** São 27 candidatos (15 + 10 + 2; em 14.7 escrevi "28" por erro de conta). O e M crus, sem constante por domínio; com constante ao lado.

| Candidato | Bases | N | > 25 g | O cru | M cru | M com constante | S14 | Guarda |
|---|---|---|---|---|---|---|---|---|
| **R0×H0 = A0** (D1, foto inteira) | nenhuma | **71,3** | 51,9 | 53,4 | **13,3** | 43,2 | 46,0 | sim |
| **R3×H0 = S5** (o S de 13.11) | OmniFood8K (massa) | 70,5 | 51,8 | 55,7 | 18,0 | 47,1 | **48,1** | **sim** |
| R0c×H0c = A0c (recorte) | nenhuma | 68,0 | 48,1 | 58,1 | 20,2 | 45,5 | 48,8 | **não** (N −3,3) |
| R3c×H0c | OmniFood8K | 68,3 | 48,5 | 58,2 | 19,9 | 49,8 | 48,8 | não |
| **R4c×H0c** (recorte + as cinco bases na massa) | as cinco | 68,3 | 48,8 | 55,7 | **59,2** | 58,5 | **61,1** (a maior) | não (N −3,0) |
| R4c×H1c | as cinco | 67,1 | 45,9 | 55,7 | 59,2 | 58,5 | 60,7 | não |
| R1c×H0c | Flavoria, LeFood, Mensa | 68,4 | 48,8 | 54,0 | 51,7 | 54,9 | 58,0 | não |
| R4×H1 (o X de 13.11) | as cinco | 69,2 | 48,9 | 56,2 | 7,3 | 57,4 | 44,2 | não |
| P2×H0 / P2 | nenhuma | 69,0 / 66,4 | 47,7 / 50,6 | 56,4 / 58,9 | 19,9 | 47,1 | 48,4 | não (aceitação) |

(As outras 17 combinações estão em `r5_selecao.json`; nenhuma passa na guarda.)

- **S5 = R3×H0, o mesmo S de 13.11.** Só A0 e R3×H0 passam na guarda; nenhuma configuração com recorte passa.
- **A melhor só com dados liberados continua o A0.**
- **Aparelho: não há Ad5.** O aparelho continua o C1 de 12.6.
  - F4: rota massa × média das duas densidades, 1 orientação, **67,3%** (−0,7 contra o U0; IC −2,5 a +1,2).
  - F5: geométrica, **67,5%** (−0,6; IC −2,2 a +1,2).
  - F4c: **65,9%** (−2,1; IC −4,5 a +0,3).
- **Par S5 + C1:** 72,1% na CV, +1,55 sobre o S5 (IC −0,7 a +3,8). Não passa na regra de 2,0 pontos.
- **Segunda leitura das partes de registro: não acontece.** 14.7 só a permite para um S5 novo (que não seja o A0 nem o S de 13.11) e para um Ad5. Não há nenhum dos dois. As partes de registro externas não foram abertas nesta seção.
- **Relato secundário de 12.8, na CV** (kcal / carboidrato / gordura): A0 70,5 / 78,6 / 68,4%; S5 70,5 / 78,6 / 68,7%. Sem mudança.

**c) O enquadramento pelo prato consertou a massa fora do Nutrition5k? Sozinho, não; junto com as bases externas, sim, e muito.** Massa a ±20%, **crua** (sem constante nenhuma, como no app), na CV de cada base:

| Base | A0 (foto inteira) | A0c (recorte, só Nutrition5k) | **R4c (recorte + as cinco bases)** | R4c − A0 (IC) | Constante que o R4c ainda precisaria | MAPE cru: A0 → R4c |
|---|---|---|---|---|---|---|
| Flavoria | 34,7% | 45,1% | **66,2%** | +31,6 (+24,7 a +38,4) | ×1,03 | 31% → 23% |
| FPB de cima | 3,5% | 13,7% | **48,2%** | +44,7 (+39,8 a +52,3) | ×1,07 | 60% → 27% |
| Mensa | 1,8% | 1,8% | **63,3%** | +61,4 (+51,7 a +68,2) | ×1,02 | 68% → 21% |
| LeFood | 18,5% | 6,2% | **61,9%** | +43,4 (+37,6 a +48,7) | ×0,99 | 99% → 21% |

- **Com o recorte, as bases externas deixaram de brigar com o Nutrition5k.**
  - Com a foto inteira, as massas de Flavoria, LeFood e Mensa na ridge custavam de 2 a 3 pontos de N (13.11).
  - Com o recorte, o R4c×H0c fica **+0,35 sobre o A0c** (IC −1,5 a +2,2). O R1c, +0,45; o R3c, +0,26.
  - A constante por domínio, que com a foto inteira ia de ×0,3 a ×3,6, cai para **×0,99 a ×1,07**.
  - É a hipótese de 14.3, confirmada na CV: **com o prato enchendo o quadro, um modelo só serve os cinco domínios sem saber em qual está.**
- **Só o recorte não basta.** O A0c, que só viu o Nutrition5k, ainda erra a escala dos outros domínios (constante de ×0,5 a ×2,5). O tamanho dos pratos muda entre os domínios, e o modelo só aprende isso vendo os pratos.
- **O custo: o recorte tira cerca de 3 pontos do Nutrition5k** (A0c contra A0: N −3,3, IC −5,5 a −0,9; acima de 25 g −3,8). Contrariou 8.7, onde o recorte não custava nada.
  - A massa isolada não piora: com a densidade real, R0 86% e R0c 86%.
  - A densidade isolada não piora: com a massa real, H0 78% e H0c 79%.
  - A perda aparece quando as duas se juntam. Na foto inteira, os erros de massa e de densidade, vindos da mesma escala absoluta da câmera fixa, se compensavam em parte; no recorte, não. **(Correção em 15.2: essa explicação não se sustenta; a correlação entre os dois erros é quase zero nos dois casos, e a causa da perda não foi identificada.)**
- **O que isso quer dizer para o produto, e não está na regra.**
  - O N da foto inteira (71%) usa uma vantagem que o app não tem: a câmera do Nutrition5k fica sempre à mesma altura, então o tamanho em pixels é o tamanho real.
  - O app recebe a foto enquadrada pelo guia, que é o recorte. **No Nutrition5k, o número parecido com o do app é o do recorte: cerca de 68%.**
  - A R14, escrita antes, protege o N da foto inteira. **Ela escolheu o modelo certo para o Nutrition5k e não o mais parecido com o app.** Não mudo a escolha depois de ver os números: o S5 de registro é o R3×H0.
  - O próximo pré-registro deve medir o Nutrition5k pelo recorte (14.10 e). O R4c×H0c é o candidato natural ao app, e a decisão só pode sair de uma regra escrita antes, julgada no conjunto brasileiro.

**d) A receita corrigida do aparelho funcionou?** Na massa, sim. A cabeça de frações sem gradiente deixou a massa como a do U0 (0,2315 e 0,2303, contra 0,233), e as três passaram na aceitação, o que o F1 e o F2 não conseguiram. Na proteína, não. A densidade das frações sobre a rede pequena não acrescenta: 67,3% contra 68,0% do U0, e a rota só pelas frações dá 64%. **O teto da rede B/16 a 224 px parece ser 68% no Nutrition5k.** Com o recorte, 66%.

**e) Os melhores modelos hoje, e a próxima alavanca.**

| Papel | Modelo | Nutrition5k, CV (foto inteira) | Nutrition5k, CV (recorte) | Massa crua fora do Nutrition5k (média de Flavoria, FPB, Mensa) | Procedência |
|---|---|---|---|---|---|
| **Servidor de registro (R14)** | R3×H0 | 70,5% (teste2: 69,2%) | — | 18% | Nutrition5k + OmniFood8K (CC BY-NC) |
| Servidor só com dados liberados | A0 = D1 | 71,3% (teste2: 69,7%) | 68,0% (A0c) | 13% | Nutrition5k |
| Candidato ao app (não escolhido pela regra) | R4c×H0c | — | 68,3% | **59%** | Nutrition5k + as cinco bases (FPB e OmniFood8K CC BY-NC) |
| **Aparelho** | C1 (U0) | 68,0% (teste2: 66,0%) | — | não medida | Nutrition5k |

- O R4c×H0c tem o mesmo custo de servir que o D1: So400m congelado mais duas cabeças, 430 M de parâmetros, cerca de 64 ms por foto na T4 e 3,9 s em CPU. A conferência de remontagem passou, e as cabeças estão na saída da execução, fora do repositório.
- **Próximas alavancas, em ordem:**
  1. **O conjunto brasileiro com o guia** (`docs/coleta-brasil.md`) é o único lugar que julga os 70% do app e o único que pode escolher entre "melhor no Nutrition5k" (R3×H0) e "melhor fora dele" (R4c×H0c). Nada no Nutrition5k decide isso.
  2. **Devolver a escala perdida no recorte:** o diâmetro do prato. A coleta já anota o diâmetro. O app pode perguntar uma vez o tamanho do prato de casa. No treino, um token de tamanho de prato, que nas bases sem essa informação fica "desconhecido". É a única informação que separa "prato pequeno cheio" de "prato grande cheio".
  3. **O ajuste fino do So400m, agora no recorte e com as cinco bases.** Só a perda de rótulos funciona (P2 ficou a 0,0009 da aceitação); a próxima receita deve partir do R4c. Não é prioridade antes dos itens 1 e 2.
  4. **No aparelho,** a rede B/16 a 224 px parou em 68%. Subir pede outra rede (maior ou a 384×512) ou destilar do servidor com fotos sem rótulo que não sejam do teste2. Não vale gastar GPU nisso antes da coleta.

**Registro de método.**
- (a) A R14 foi aplicada como escrita: o P2 ficou fora por 0,0009 e nenhuma configuração com recorte passou na guarda.
- (b) A observação "o recorte é o mais parecido com o app" foi feita depois de ver os números. Por isso **não muda a escolha** e entra só como proposta para o próximo pré-registro.
- (c) Nenhuma parte de registro externa e nenhum teste do Nutrition5k foi lido nesta seção. Os números são todos de validação cruzada.
- (d) As misturas R0c×H0 e R0×H0c servem só para o diagnóstico do item (c); não são candidatas, porque o app recebe uma foto só.


## 15. Sexta execução: protocolo no recorte (13/11/2026)

**Desenho e pré-registro. Nada aqui tem resultado da sexta execução.** A decisão já registrada em `docs/produto.md`, seção 7, é esta:
- **servidor do app:** o R4c×H0c (recorte pelo prato, massa treinada com o Nutrition5k e as cinco bases externas);
- **números publicados do Nutrition5k:** continuam os do R3×H0, até o conjunto brasileiro decidir;
- **alternativa só com dados liberados:** o D1;
- **aparelho:** o C1.

A partir daqui, **o Nutrition5k é julgado no recorte**.

### 15.1 Protocolo no recorte

- **N_c:** critério ponderado nas cinco faixas, na validação cruzada do treino_v2 (folds por cadeia), com a foto **recortada pelo prato** como em 14.3, com a regra emendada.
- **Referência REF = R4c×H0c**, rota massa × média das duas densidades, **1 orientação**, como está no app: N_c **68,34%**, > 25 g **48,82%**, O cru **55,65%**, M cru **59,23%**, nota S15 = (N_c + O cru + M cru) / 3 = **61,07**.
- **O e M continuam crus, sem constante por domínio,** com os números com constante ao lado. Nas bases externas a previsão é a gravada (média de duas orientações, ou dos quadros do OmniFood8K), porque o kernel não grava variantes de orientação para elas.
- **Não existe teste de registro do Nutrition5k.** O teste2 está fechado e o kernel não gera previsões para ele.
- **A meta de 70% do produto só é julgada no conjunto brasileiro com o guia.**
- **Leituras já feitas das partes de registro externas:**

| Parte | Leituras até agora | Próxima |
|---|---|---|
| OmniFood8K | 1 (13.11) | 2ª |
| FPB de cima | 1 (13.11) | 2ª |
| LeFood | 1 (13.11) | 2ª |
| Mensa (inválida para massa, 13.11) | 1 | 2ª, só como teste de escala |
| Flavoria (últimos 7 dias) | **2** (relato externo de 12.6 e 13.11) | 3ª |

### 15.2 Correção de 14.10 e o que a CV já mostra (escrito antes de qualquer resultado da sexta execução)

**Correção.** Em 14.10 (c) expliquei a perda de cerca de 3 pontos do recorte por "erros de massa e de densidade que se compensavam na foto inteira e deixam de se compensar no recorte". **Isso não se sustenta.**
- A correlação entre o erro de log-massa e o de log-densidade é quase zero nos dois casos: +0,03 na foto inteira e +0,04 no recorte; por faixa, de −0,12 a +0,20 nos dois.
- O erro de massa por faixa também é quase igual.
- **Não sei explicar a perda.** Ela fica registrada como medida (−3,3 pontos; IC −5,5 a −0,9), sem causa.

**Números da CV já visíveis antes deste desenho.** Saem das previsões fora-do-fold da quinta execução, olhadas para desenhar esta seção. Por isso as escolhas entre essas alternativas vão ficar otimistas: quem confere é a segunda leitura externa e, de verdade, o conjunto brasileiro.

| Na REF (R4c×H0c) | N_c, 1 / 4 / 8 orientações | > 25 g (1 orientação) |
|---|---|---|
| r1, massa × média das duas densidades (rota atual) | 68,3 / 69,9 / 70,0 | 48,8 |
| Proteína direta da ridge | 65,6 / 66,7 / 67,9 | 45,1 |
| Proteína direta da cabeça de frações | 67,2 / 68,1 / 67,5 | 43,4 |
| r2, geométrica (r1 com a direta da ridge) | 68,4 / 70,6 / 70,9 | 48,7 |
| r3, geométrica das três (r1 e as duas diretas) | 69,7 / 70,7 / 70,9 | 49,2 |
| r4, mistura log-linear das três, com pesos de cada fold ajustados nos outros quatro | 69,3 (+1,0 sobre r1; IC −0,5 a +2,5) | 50,8 |
| Par REF + F4c (aparelho no recorte), média em log | 70,9 (+2,5; IC +0,5 a +4,7) | |
| Par REF + C1 (aparelho na foto inteira) | 71,0 (+2,7; IC +0,6 a +4,8) | |

- **No recorte, as orientações valem mais que na foto inteira:** +1,6 de 1 para 4, contra +0,7 no D1. Pela regra de 12.3 (item 2), a REF seria servida com 4 orientações.
- **A alavanca 3 do coordenador (OmniFood8K na cabeça de densidade, no recorte) já foi testada na quinta execução:** R4c×H1c dá N_c **67,1** (−1,2 contra a REF) e O cru **55,7** (+0,1). Piora no Nutrition5k e não ajuda no OmniFood8K. **Sai do desenho; não gasta GPU.**
- **A alavanca 5 (tamanho do prato como entrada) não tem dado:** nenhuma das seis bases anota o diâmetro do prato.
  - O FPB tem um xadrez de calibração em quase toda foto, e o SNAPMe tem um marcador de tamanho conhecido. Os dois dariam escala absoluta, mas medir o prato por eles é um projeto à parte (detectar o marcador, medir em pixels) e cobriria só duas bases.
  - **Fica para o conjunto brasileiro,** que já anota o diâmetro (`coleta-brasil.md`).

### 15.3 Alavancas, na ordem que eu defendo

| Ordem | Alavanca | Custo | Por quê nesta ordem |
|---|---|---|---|
| 1 | **Inferência e rotas** sobre os modelos que já existem: orientações (1/4/8), rotas r1 a r4, par com o aparelho | 0 GPU (CPU) | O que a CV já mostra (+1 a +2,7) é maior que qualquer ganho esperado de treino. Mas é a parte mais otimista |
| 2 | **P3:** ajuste fino do So400m no recorte, só a perda de rótulos, com as seis bases na massa, saindo da ridge do R4c, com aceitação robusta; depois, a cabeça de densidade nas features congeladas dele | ~210 min | É o único treino que pode subir a massa e a densidade ao mesmo tempo no domínio do app. O P2 falhou por 0,0009 |
| 3 | **F5c:** aparelho no recorte, receita F4c (com a cabeça sem gradiente) mais as cinco bases na massa | 96 min | O aparelho no app vê o recorte, e o C1 nunca foi medido nele. O par com o aparelho é a maior alavanca vista (+2,5) |
| 4 | **H0c-m:** cabeça de densidade que recebe também a log-massa prevista pela ridge (fora-do-fold) | 60 min de features + 5 | A sugestão de densidade condicionada à massa do coordenador. É a mais barata de testar, e a que menos espero |

**Ordem de corte se a cota faltar:** H0c-m, depois F5c. O P3 não é cortado.

### 15.4 Receitas

**P3 (`gpu6 ftparcial so400m prato rotulos +tudo`).**
- **Entrada:** recorte de 14.3 em todos os domínios, com o erro de enquadramento de ±10% como aumento; sem viés de massa por domínio.
- **Alvos:**
  - Nutrition5k: os 22, com os pesos do P2 (massa 6; proteína e densidade 3; o resto pequeno).
  - Bases externas: só a massa total, com os tetos de peso de 13.4.
- **Início:** a cabeça linear de 22 saídas começa na solução da ridge sobre as features congeladas do recorte, calculada no treino de cada fold (como o R4c).
- **Mudanças para a aceitação ser robusta,** todas fixadas agora:
  1. **Duas épocas de aquecimento com o backbone parado** (só a cabeça treina, passo de 1e-3). Depois, **5 épocas** com os 4 últimos blocos liberados, passo de **1e-5** no topo (metade do P2), decaimento de 0,75 por bloco e cosseno.
  2. **Média móvel dos pesos de 0,999** (era 0,99).
  3. **Parada pela validação interna:**
     - dentro de cada fold, 10% das cadeias de treino (`sha1(cadeia) mod 10 == 0`) ficam de fora como validação interna, e o RMSE de log-massa (1 orientação, pesos da média móvel) é medido na época 0 (a ridge) e no fim de cada época;
     - **vale o checkpoint de menor RMSE interno, e a época 0 é candidata**;
     - o modelo final, com tudo, treina pela mediana do número de épocas escolhido nos folds.
  - **A aceitação continua a de 13.7,** sobre o fold de fora: a média dos folds do RMSE de log-massa do checkpoint escolhido não pode passar da época 0.
  - **Consequência honesta:** se a parada escolher a época 0, o P3 passa na aceitação e é igual à REF. Isso não é ganho, e sai relatado como "ajuste fino sem efeito".
- **Cabeça de densidade do P3 (HP3).** Em cada fold, depois do ajuste fino:
  - extrair as features do backbone ajustado: treino do fold com 2 orientações (exata e uma com erro de enquadramento), validação com 1, 4 e 8;
  - ajustar uma cabeça igual à H0c (MLP de 512, frações, densidade composta, peso por faixa, 60 épocas);
  - as fotos externas de cada fold passam uma vez e dão a massa da cabeça principal e as features para a HP3.

**F5c (`gpu6 ft siglip2_b16_224 r2 fracoes-sg prato +tudo`).** É a receita do F4c (U0, cabeça de frações sem gradiente, recorte com erro de enquadramento), mais as massas das cinco bases como no F3, sem viés por domínio. Aceitação de 13.7.

**H0c-m (`gpu6 fracoes so400m prato +massa`).**
- A H0c com uma entrada a mais: a log-massa prevista pela ridge R4c, padronizada, concatenada às features.
- No treino de cada fold entra a previsão **aninhada** da ridge (feita sem o fold de validação nem o fold do próprio prato, como os alvos aninhados de 12.3), para não vazar o rótulo da massa.
- Pede as features do So400m no recorte de novo: a quinta execução não as guardou.
- **Conferência de paridade:** a R4c e a H0c refeitas repetem as da quinta execução (correlação de 0,999 ou mais).

### 15.5 Candidatos e regra R15 (na CPU, só com validação cruzada)

**Servidor.**
- **Cabeça de massa:** R4c (quinta execução) ou P3 (se aceito).
- **Cabeça de densidade:** H0c, H0c-m (se rodou) ou HP3 (só com o P3).
- **Rota:** r1 a r4, com as direções de 15.2. Na r2, r3 e r4, a "direta da ridge" é a proteína direta da cabeça de massa e a "direta das frações" a da cabeça de densidade.
- **Orientações:** de cada candidato, pela regra de 12.3 (item 2): o menor número entre 1, 4 e 8 a até 1,0 ponto do critério com 8.
- São de 4 a 20 candidatos (até 5 pares de cabeças × 4 rotas), conforme o que terminar.

**Regra R15.**
1. **Guarda:** N_c a no máximo 1,0 ponto da REF (67,34% ou mais), e o critério acima de 25 g no recorte a no máximo 2,5 pontos dela (46,32% ou mais).
2. **S6 = o admissível de maior S15 = (N_c + O cru + M cru) / 3.**
   - Empate (menos de 0,5): menos orientações (latência), depois menos bases não comerciais, depois a rota mais simples (r1 < r2 < r3 < r4), depois menos bases.
   - A REF é sempre admissível e concorre.
3. **Melhor só com dados liberados,** sempre ao lado: o mesmo cálculo restrito a R1c (Flavoria, LeFood e Mensa, todas CC BY) com H0c. Hoje ele dá N_c 68,4 e M cru 51,7. A REF usa FPB e OmniFood8K (CC BY-NC).

**Aparelho no recorte.**
- Referência: **F4c** (65,92%; geométrica, 1 orientação), porque o C1 foi treinado na foto inteira e nunca foi medido no recorte.
- **Ad6 = o F5c, se for aceito e ficar 1,0 ponto ou mais acima do F4c em N_c.** Senão, o F4c.
- **Recomendação para o produto, que não é regra:** trocar o C1 pelo modelo de aparelho treinado no recorte. O app só vai mostrar fotos enquadradas pelo guia, e o C1 nunca foi medido nelas.

**Par S6 + aparelho** (média em log da proteína; Ad6 ou F4c): entra se subir a CV **2,0 pontos ou mais** sobre o S6 **e** o limite inferior de 95% da diferença pareada ficar acima de zero.
- No app, o par quer dizer que a estimativa final espera o servidor; a do aparelho sai antes, como prévia.
- Latência: a soma dos dois.

**Segunda leitura das partes de registro externas.** No máximo **dois** modelos congelados, numa execução só, com marca:
1. **a REF,** que nunca foi lida e é o modelo do app;
2. o **candidato final do app:** o par, se ele passar; senão, o S6, se não for a REF. Se nenhum dos dois existir, só a REF é lida.

Intervalos de 95% e de 97,5%. Os números sem constante são os principais. O teste de escala do Mensa sai à parte. O relato de 12.8 (kcal, carboidrato, gordura) no OmniFood8K sai como antes. A comparação pareada é com o A0 e o S de 13.11, que já foram lidos.

### 15.6 Configurações e GPU

| Prioridade | Configuração | GPU | Minutos |
|---|---|---|---|
| 1 | P3 + HP3 (5 folds + final, parada interna, extração de features e cabeça por fold) | A | ~210 |
| 2 | F5c | B | ~96 |
| 3 | Features do So400m no recorte + R4c e H0c refeitas (paridade) + H0c-m | B | ~65 |
| | Medidas de serviço: REF com 1 e 4 orientações; P3+HP3 com 1 e 4; F5c; o par | A e B | ~15 |

**Total:** cerca de 385 minutos de GPU e 3,7 h de relógio, com folga dentro de 4,5–5 h.

**O kernel grava:**
- tudo o que a quinta execução gravava (sem teste2);
- os checkpoints escolhidos pela parada interna e a curva interna por época;
- os pesos treináveis do P3 (os 4 blocos, a norma e a cabeça) em meia precisão, mais as cabeças HP3, H0c-m e F5c;
- os arquivos de servir com a janela do guia e a remontagem de todas as cabeças;
- a procedência por modelo (`treinado_com`, `so_comercial_ok`).

### 15.7 O que eu espero

| Alavanca | Expectativa | Chance |
|---|---|---|
| Orientações: a regra escolhe 4 para o S6 | +1,5 sobre a mesma configuração com 1 | 85% |
| Rotas r2 a r4 | +0,5 a +1,3 sobre r1, já com 4 orientações (o que se viu: +0,6 a +0,8) | Uma rota diferente da r1 ganhar: 60% |
| P3 | Aceito: 85% (a parada pela época 0 garante quase tudo). Ganho em N_c de 0 a +1,5 e em M cru de 0 a +5 | Entrar no S6: 30% |
| H0c-m | 0 a +0,5 | Entrar no S6: 15% |
| F5c | N_c de 65,5 a 67,5; M cru do aparelho de cerca de 23% para 50–60% | Virar Ad6: 30% |
| Par S6 + aparelho | +1,5 a +2,5 (visto: +2,5 com a REF a 1 orientação; com 4 orientações deve ser menor) | Passar na regra: 45% |
| **Candidato final do app, na CV** | **N_c de 70 a 72%** | N_c acima de 72%: 15% |

- Fora do Nutrition5k, espero M cru parado (de 58 a 62%) e O cru de 55 a 59%.
- **Nada disso demonstra os 70%:** é validação cruzada, com escolhas feitas depois de ver uma parte dos números. Quem decide é o conjunto brasileiro.

### 15.8 Resultado da sexta execução (14/11/2026)

A execução rodou no Kaggle dentro do orçamento, e nada foi cortado. A ingestão aceitou as 6 configurações, e a paridade com a R4c e a H0c da quinta execução é exata: correlação de 1,00000, CV de 68,34%, e as previsões são idênticas, inclusive nas bases externas. Script: `eval/treino/r6_avalia.py`.

**a) Regra R15, só com validação cruzada, escrita antes de abrir as partes de registro.**

*Aceitação (13.7):*
- F4c: 0,2638 → 0,2329, **passou**.
- F5c: 0,2638 → 0,2295, **passou**.
- P3: 0,2265 → 0,2266, **não passou**, por 0,0001.
  - A parada pela validação interna escolheu a época 0 nos folds 0 a 2 e a época 7 nos folds 3 e 4; o modelo final ficou com a época 0.
  - Nos dois folds em que treinou, a massa do fold de fora piorou um pouco.
  - **Ajuste fino sem efeito,** como 15.4 previa para esse caso. O P3 não concorre. Para registro: N_c de 68,4% com a cabeça HP3 e com a H0c, igual à REF; RMSE de log-massa de 0,2274, contra 0,2257 da R4c.

*Candidatos a servidor:*
- massa R4c × densidade H0c ou H0c-m, quatro rotas, orientações pela regra de 12.3: 8 candidatos, mais a REF fixa com 1 orientação;
- ao lado, o mesmo com a R1c (só dados liberados).

| Candidato | N_c (1 / 4 / 8 orientações) | Orientações escolhidas | N_c | > 25 g | O cru | M cru | S15 | Guarda |
|---|---|---|---|---|---|---|---|---|
| **REF** R4c×H0c r1 (app hoje) | 68,3 / 69,9 / 70,0 | 1 (fixa) | 68,34 | 48,82 | 55,65 | 59,23 | 61,07 | sim |
| R4c×H0c r1 | 68,3 / 69,9 / 70,0 | 4 | 69,93 | 51,89 | 55,65 | 59,23 | 61,60 | sim |
| R4c×H0c r2 | 68,4 / 70,6 / 70,9 | 4 | 70,57 | 51,46 | 59,37 | 59,23 | 63,06 | sim |
| R4c×H0c r3 | 69,7 / 70,7 / 70,9 | 4 | 70,69 | 51,66 | 59,06 | 59,23 | 62,99 | sim |
| R4c×H0c r4 | 69,3 / 71,4 / 72,1 | 4 | 71,40 | 54,51 | 57,88 | 59,23 | 62,84 | sim |
| R4c×H0c-m r1 | 68,1 / 70,2 / 70,1 | 4 | 70,15 | 51,29 | 55,26 | 59,23 | 61,55 | sim |
| **R4c×H0c-m r2 = S6** | 68,7 / 70,7 / 70,4 | 4 | **70,73** | **52,11** | **60,21** | 59,23 | **63,39** | sim |
| R4c×H0c-m r3 | 69,3 / 71,1 / 71,2 | 4 | 71,14 | 52,06 | 60,24 | 59,23 | 63,54 (a maior) | sim |
| R4c×H0c-m r4 | 70,4 / 71,1 / 71,8 | 4 | 71,07 | 53,66 | 58,81 | 59,23 | 63,04 | sim |

- **Todos passam na guarda.** A regra das orientações escolheu 4 em todos. A M é a mesma em todos, porque a cabeça de massa é a mesma.
- **S6 = R4c×H0c-m, rota r2, 4 orientações.**
  - A maior nota é a da R4c×H0c-m r3 (63,54). A menos de 0,5 dela ficam também a H0c-m r2 (63,39) e a H0c r2 (63,06).
  - O desempate de 15.5 segue a ordem: orientações (iguais), bases não comerciais (iguais), **rota mais simples (r2 antes de r3)**, número de bases (iguais), e por fim a nota.
- **S6 contra a REF:** N_c **+2,39** (IC por cadeia +0,28 a +4,55); acima de 25 g, +3,30 (IC −0,66 a +7,48); O cru +4,6. **É a primeira melhora com intervalo acima de zero desde a quarta execução.**
  - Vem quase toda da inferência (4 orientações e a rota r2), que 15.2 já mostrava antes da execução.
  - A H0c-m acrescenta pouco: +0,2 em N_c e +0,8 em O cru sobre a H0c na mesma rota.
- **Melhor só com dados liberados** (R1c×H0c, 15.5): rota r3 com 4 orientações. N_c **71,26** (+2,93 sobre a REF; IC +0,87 a +4,99), O cru 58,35 e **M cru 51,68**: perde 7,5 pontos de massa fora do Nutrition5k contra o S6.
- **Aparelho no recorte.**
  - F5c: rota massa × média das densidades, 1 orientação, N_c **67,37**, **+1,45 sobre o F4c** (IC −0,37 a +3,19). Passa no limiar de 1,0 ponto: **Ad6 = F5c.**
  - A massa crua do aparelho fora do Nutrition5k vai de 23,4% (F4c) para **50,9%** (Flavoria 65,7, FPB 39,4, Mensa 47,6).
- **Par S6 + F5c:** N_c 71,63, **+0,90 sobre o S6** (IC −0,96 a +2,83). **Não passa** na regra de 2,0 pontos. O par só ajudava quando a REF usava 1 orientação e a rota r1: o ganho que ele dava foi absorvido pelas 4 orientações e pela rota r2.
- **Candidato final do app = S6.** A segunda leitura (15.5) é da REF e do S6.
- **12.8 na CV do recorte** (kcal / carboidrato / gordura, 1 orientação): REF 67,2 / 78,6 / 68,3; S6 67,5 / 77,6 / 68,6.

**b) Segunda leitura das partes de registro externas: REF e S6** (14/11/2026; uma execução, com marca). Números crus, sem constante por domínio. Nas bases externas, a previsão do S6 usa a rota r2 sobre as previsões gravadas: as 4 orientações valem só no Nutrition5k, porque o kernel não grava variantes de orientação para as fotos externas. Depois desta leitura, as partes ficam com **2 leituras** cada (OmniFood8K, FPB, LeFood, Mensa) e **3** no Flavoria.

| OmniFood8K, cenas com prato principal nunca visto (1.916; 34 pratos) | REF | **S6** | S6 − REF |
|---|---|---|---|
| Proteína, quatro faixas, cru (IC 95%; 97,5%) | 63,1% (55,8 a 70,8; 54,9 a 71,8) | **64,9%** (56,7 a 75,7; 56,0 a 77,0) | +1,8 (95%: −2,0 a +6,4; 97,5%: −2,5 a +7,5) |
| Por faixa: 0–5 / 5–15 / 15–25 / > 25 g | 98 / 75 / 52 / 28 | 98 / 74 / 43 / **44** | |
| Sem a ressalva de osso ou casca | 56,5% | 53,4% | |
| Massa crua a ±20% / MAPE / viés | 68,7% / 16,3% / −3% | igual (mesma cabeça) | |
| kcal / carboidrato / gordura (12.8, cru) | 25% / 65% / 30% | 24% / 68% / 30% | |
| Cenas mistas (2.181) | 59,6% | 60,6% | |

- Na quarta execução (13.11), essas mesmas cenas deram de 56% a 65% crus (A0 56%, S 57%, X 65%). Os dois modelos com recorte ficam no topo dessa faixa, e o intervalo de nenhum chega a 70%.
- **kcal e gordura continuam muito subestimados nessa cozinha:** viés de −52% e −75%. O óleo do refogado chinês não aparece na foto.

| Bases só de massa, massa crua a ±20% (a cabeça é a R4c nos dois modelos) | 13.11, foto inteira, cru (A0 / S / X) | **R4c, recorte, cru** (IC 95%; 97,5%) | Com constante | MAPE cru | Viés cru |
|---|---|---|---|---|---|
| Flavoria (227 pratos, 7 dias) | 23% / 37% / 1% | **68,7%** (62,0 a 75,3; 60,8 a 77,1) | 72,2% (×1,03) | 18,7%; acima de 600 g, 19,6% | −6% |
| FPB de cima (121 bandejas) | 3% / 5% / 12% | **44,6%** (33,9 a 57,9; 32,4 a 60,0) | 50,4% (×1,06) | 24,6% | −14% |
| LeFood (91 porções) | 17% / 18% / 0% | **65,9%** (56,0 a 75,8; 54,9 a 76,9) | 67,0% (×0,99) | 18,3% | +1% |
| Mensa, um participante que fotografou itens soltos de perto (só teste de escala) | 24% / 12% / 32% | 29,4% | 29,4% | 77% | +48% |

- **A massa usável fora do Nutrition5k se confirmou nas partes seguradas:** de 1% a 37% crus para 45% a 69% crus, com a constante que ainda faltaria entre ×0,99 e ×1,06. É o mesmo nível da CV (48% a 66%).
- **O Mensa continua errado em +48%.** Um item solto fotografado de perto, enquadrado como se fosse um prato inteiro, ainda engana a escala. É o caso de uso que o guia do app tem de evitar (ou o tamanho do prato tem de resolver).
- **SNAPMe** (autorrelato, foto inteira, correlação de postos): proteína **0,71** na REF e **0,73** no S6 (era de 0,59 a 0,64 em 13.11); massa 0,49; kcal 0,62; gordura 0,65; carboidrato 0,40.

**c) O pipeline final do app (servidor), definido com precisão.**

| Parte | Definição |
|---|---|
| Entrada | A foto enquadrada pelo círculo-guia, com diâmetro de 80% do lado menor. A janela 4:3 do guia é redimensionada para 512×384 (bicúbico); valores em [0, 1], média e desvio de 0,5 |
| Backbone | SigLIP 2 So400m/16 a 384×512, **congelado** (`siglip2_so400m16_512.npz`, checkpoint público; timm `vit_so400m_patch16_siglip_512.v2_webli`), em meia precisão. Features = [saída agregada ‖ média dos tokens], 2.304 dimensões |
| Orientações | **4:** original, 180°, espelho horizontal e espelho vertical (as que preservam o quadro). As saídas das cabeças são calculadas por orientação, e a média é feita em log |
| Cabeça de massa | Ridge **R4c** (`cabeca__gpu5_ridge_so400m_prato__tudo.npz`; a refeita na sexta execução é idêntica). Dá a log-massa *m* e a proteína direta *dR* |
| Cabeça de densidade | **H0c-m** (`cabeca__gpu6_fracoes_so400m_prato__massa.npz`). MLP de 512 sobre [features ‖ log-massa da R4c]. Dá a densidade da regressão e a densidade composta pelas frações (10 fontes, densidades por fonte no arquivo) |
| Rota | **r2:** proteína = √(*m* × *d*_méd × *dR*), com *d*_méd = exp(média dos logs de (densidade da regressão + 0,01) e (densidade das frações + 0,01)) − 0,01 |
| Custo (T4, meia precisão; CPU de 4 threads) | 430 M de parâmetros, 860 MB. **262 ms por foto com as 4 orientações num lote** (284 ms em sequência; 70 ms com 1 orientação). Em CPU, 14,6 s (3,6 s com 1). Medido para a REF; a H0c-m só acrescenta uma entrada à MLP. A remontagem das duas cabeças confere (diferença máxima de 0,0007 e 0,0003 em log) |
| Procedência | Cabeça de massa: Nutrition5k, Flavoria, LeFood e Mensa (CC BY 4.0), mais FPB e OmniFood8K (**CC BY-NC 4.0**). A cabeça de densidade treinou só com o Nutrition5k, mas recebe a massa da R4c. **O pipeline não é só de dados liberados.** Atribuição obrigatória às seis bases |
| Validação cruzada | N_c **70,7%** (> 25 g 52,1%); O cru 60,2%; M cru 59,2%. Parte segurada do OmniFood8K: 64,9% |

- **Só com dados liberados:** R1c×H0c, rota r3, 4 orientações. N_c 71,3%, O cru 58,4%, **M cru 51,7%**, mesmo custo.
- **Aparelho:** **F5c** (Ad6). SigLIP 2 B/16 a 224 px com ajuste fino no recorte, cabeça de frações sem gradiente, rota massa × média das densidades, 1 orientação.
  - Medidas: N_c 67,4%; massa crua fora do Nutrition5k 50,9%.
  - Custo: 93 M de parâmetros, 186 MB, 10 ms na T4, 0,21 s em CPU.
  - Treinou com as cinco bases externas: **também não é só de dados liberados**, diferente do C1.
  - O par servidor + aparelho não passou (+0,9), então o app usa o servidor para a estimativa final e o aparelho só para a prévia.

**d) Conclusão.**

1. **No protocolo do app, o modelo melhorou com segurança.** No recorte, N_c foi de 68,3% para 70,7% (+2,4; IC +0,3 a +4,6), e acima de 25 g de 48,8% para 52,1%. No OmniFood8K segurado, +1,8 (dentro do ruído).
   - A melhora vem quase toda da **inferência**: 4 orientações em vez de 1 e a rota que junta a proteína direta. Isso já estava visível antes desta execução (15.2), então o número da CV tem algum otimismo de escolha.
   - **O treino novo não rendeu:** o ajuste fino do backbone grande não teve efeito pela segunda vez, a densidade condicionada à massa deu +0,2, e o par com o aparelho deixou de ajudar.
2. **A massa usável fora do Nutrition5k se confirmou numa parte nunca usada para escolher:** 45% a 69% dos pratos a ±20% sem constante (antes, de 1% a 37%). O caso que ainda quebra é o item solto fotografado de perto (Mensa, +48%).
3. **70% não está demonstrado em lugar nenhum.**
   - O número do Nutrition5k é de validação cruzada e passa um pouco de 70% (70,7%, IC por cadeia de cerca de ±2,5), sem teste de registro.
   - No OmniFood8K, 65%.
   - kcal e gordura erram muito em comida com óleo (−52% e −75% no OmniFood8K).
4. **A alavanca seguinte não é mais uma execução em GPU sobre as mesmas bases.**
   - Três execuções seguidas mostraram que o ajuste fino do backbone grande não melhora a massa, e o que subiu foi a inferência.
   - **Agora o que falta é o conjunto brasileiro com o guia** (`docs/coleta-brasil.md`), pelos três motivos de sempre: é o único que julga os 70%; traz o diâmetro do prato, que resolve o caso do close; e traz comida brasileira para treinar.
   - **Enquanto ele não existe,** duas coisas baratas, sem GPU nova:
     - (a) medir no app real o erro de enquadramento que as pessoas cometem com o guia;
     - (b) avisar no app quando o prato não enche o guia (item solto de perto).

**Registro de método.**
- (a) A R15 foi aplicada como escrita. O P3 ficou fora por 0,0001. O desempate de 15.5 escolheu a r2 contra a r3, que tinha 0,15 a mais de nota.
- (b) A parte (a) desta seção foi escrita antes da leitura das partes externas.
- (c) A leitura das partes externas rodou uma vez, para dois modelos (REF e S6). Os intervalos de 97,5% saem ao lado.
- (d) Nenhum teste do Nutrition5k foi lido.
- (e) A causa da perda de cerca de 3 pontos do recorte no Nutrition5k continua desconhecida (15.2).



## 16. Critério de parada: o que conta como meta atingida (14/11/2026)

**Escrito antes de qualquer base nova ou execução nova.** A fundadora pediu para continuar explorando bases públicas até a meta ser atingida. Isso só tem sentido com uma definição que dados públicos possam cumprir (o conjunto brasileiro ainda não existe) e com uma saída honesta. São dois níveis, e não se confundem:
- **Meta atingida em dados públicos:** a regra desta seção.
- **Meta do produto:** 70% em comida brasileira, de celular, com o guia. Só o conjunto brasileiro julga. Cumprir a primeira não autoriza afirmar a segunda.

### 16.1 Teste de registro

- **O teste2 está aposentado** (quatro leituras), e o Nutrition5k não tem mais pratos fora do treino. Não há teste de registro do Nutrition5k.
- **A parte segurada do OmniFood8K não serve como teste de registro.**
  - Foi lida duas vezes (13.11 e 15.8), e as leituras de 13.11 influenciaram o desenho das execuções 5 e 6.
  - Tem só 34 pratos principais: o intervalo por prato tem ±7,5 pontos, e passar exigiria acerto verdadeiro perto de 78%.
  - A proteína é de tabela, com fração comestível suposta.
  - **Fica como relato secundário,** sem decidir nada.
- **O teste de registro é novo:** uma parte segurada das bases com rótulo que a varredura de dados ainda achar. Requisitos, todos verificados antes da primeira previsão:
  1. **Proteína derivada de pesagem.** Vale o prato pesado ou os ingredientes pesados × tabela; não vale porção nominal nem autorrelato. **Foto de cima, ou o enquadramento do guia.**
  2. **A parte é fixada por hash da unidade de independência** (prato ou receita; sessão; pessoa; dia), nunca por foto. A tabela de divisão é gravada com sha256 **antes** de qualquer modelo ver as fotos. A parte nunca treina e nunca escolhe nada; o kernel apaga os rótulos dela.
  3. **Tamanho mínimo:** 600 pratos e 100 unidades de independência, com pelo menos 60 pratos em cada uma das cinco faixas de proteína. Se uma base sozinha não chega a isso, o teste é composto de várias bases, como em (5).
  4. **Métrica:** o critério do produto (±20% ou ±5 g), nas cinco faixas (0–5, 5–15, 15–25, 25–40, > 40 g) com o mesmo peso. Faixa com menos de 60 pratos se junta à vizinha de baixo, com o número de faixas declarado na divisão. O intervalo é por bootstrap da unidade de independência. Entrada: o recorte do guia, crua, sem constante por domínio.
  5. **Teste composto:** a média das bases com o mesmo peso por base (cada uma ponderada por faixa). O bootstrap sorteia as unidades dentro de cada base.
- **Regra de sucesso:** **limite inferior do intervalo (16.2) de 70% ou mais no teste composto.**
  - Sai também cada base em separado. Se uma base com 150 pratos ou mais ficar com o ponto abaixo de 60%, isso é escrito ao lado do sucesso, e o sucesso não é anunciado sem essa ressalva.
  - **O que o teste representa:** só as cozinhas, os pratos e as câmeras das bases que o compõem.
  - **O que não representa:** comida brasileira; foto de celular em casa (a não ser que alguma base seja isso); pratos de tamanho desconhecido.

### 16.2 Disciplina de leitura

- **No máximo uma leitura do teste de registro por execução,** de **um** candidato do app, congelado antes pela regra de escolha em vigor (a R15 e as que vierem, sempre com validação cruzada e as partes de treino).
- **Depois da leitura número k, o intervalo é o de Bonferroni:** nível 1 − 0,05/k (k = 2: 97,5%; k = 3: 98,3%). O sucesso de 16.1 vale com esse intervalo.
- **Teto: 3 leituras.** Depois da terceira, o conjunto se aposenta e vira mais uma base de treino; uma nova meta exige um teste novo, com os mesmos requisitos.
- **Uma leitura só se gasta quando vale:** a estimativa pontual do candidato na validação cruzada das mesmas bases (partes de treino, mesma métrica) tem de estar no acerto necessário de 16.3 para o tamanho do teste. Se não estiver, não se lê.

### 16.3 O que seria preciso

**Acerto verdadeiro necessário para o limite inferior passar de 70%.** Cinco faixas com o mesmo peso e o perfil por faixa do modelo atual. "Efeito de agrupamento" é a inflação da variância quando os pratos de uma mesma unidade se parecem: 1 é sem agrupamento, 2 é um agrupamento moderado.

| Pratos no teste | 1ª leitura, sem agrupamento | 3ª leitura, sem agrupamento | 1ª leitura, agrupamento 2 | 3ª leitura, agrupamento 2 |
|---|---|---|---|---|
| 400 | 74,0% | 74,9% | 75,6% | 76,7% |
| 600 | 73,3% | 74,0% | 74,6% | 75,6% |
| 800 | 72,9% | 73,5% | 74,0% | 74,9% |
| 1.200 | 72,4% | 72,9% | 73,3% | 74,0% |
| 1.600 | 72,1% | 72,5% | 72,9% | 73,5% |

**Onde o modelo do app está.**

| Medida | Valor | Natureza |
|---|---|---|
| Nutrition5k no recorte | 70,7% | Validação cruzada, com escolha otimista |
| Cenas de pratos chineses nunca vistos (OmniFood8K) | **64,9%** (IC 56,7 a 75,7) | A medida mais próxima de um teste de pratos novos |
| Pratos com mais de 25 g, OmniFood8K | 44% | |
| Pratos com mais de 25 g, Nutrition5k | 52% | |

- Com 600 a 1.000 pratos de teste, a distância é de **8 a 10 pontos** no acerto verdadeiro sobre a medida mais próxima, e de **2 a 4** sobre a CV do Nutrition5k.
- **O que poderia fechar essa distância,** em ordem de efeito esperado:
  1. **Pratos ricos em proteína, com componentes pesados.** A faixa acima de 25 g está em 44–52%, e cada faixa vale 1/5 do critério. Subir as duas faixas de cima para 65% já soma uns 4 pontos.
  2. **Rótulo de identidade e densidade da fonte de proteína em muitos pratos.** Com a massa real, a densidade do modelo dá 78–79% no Nutrition5k, ainda um teto baixo.
  3. **Escala:** o diâmetro do prato ou um marcador. Resolve o caso do close (Mensa, +48%) e o resto do erro de massa.

**Minha estimativa honesta:** a chance de bases públicas fecharem 8 a 10 pontos nos próximos três ciclos é **menor que 15%**. Nenhuma base pública conhecida tem muitos pratos pesados, ricos em proteína, com tamanho de prato.

### 16.4 Orçamento e saída

- **Um ciclo** = uma varredura de dados, mais no máximo uma execução em GPU (cerca de 4,5 a 5 h em 2 × T4), mais a escolha na CPU e, se 16.2 permitir, uma leitura.
- **Saída: 3 ciclos depois desta seção.** Se até lá nenhum candidato passar na regra de 16.1, ou se não aparecer base que permita montar o teste de 16.1, o relatório diz por escrito: **"com dados públicos não se chega a 70% demonstrado; é preciso a coleta brasileira"**, com os números de cada ciclo.
- **Saída antecipada:** se dois ciclos seguidos não subirem a estimativa pontual da validação cruzada do teste composto em pelo menos 1 ponto, a saída vem no segundo.
- **Durante os ciclos, nada para a coleta brasileira.** Ela é o único caminho para a meta do produto, qualquer que seja o resultado aqui.

### 16.5 Ciclo 1: a condição de saída foi atingida (15/11/2026)

**A saída de 16.4 por falta de base ("não aparecer base que permita montar o teste de 16.1") foi atingida no fim do primeiro ciclo.** A varredura (`fontes_treino.md`, seção 12, incluindo 12.8) não achou nenhuma base pública que forme o teste de registro de 16.1, nem sozinha nem em composição.
- **ACETADA** (802 refeições, 152 participantes, pesagem antes e depois): atende o rótulo, o tamanho e as faixas. **Falha na foto:** é oblíqua, da bandeja inteira com vários pratos, e o rótulo é da bandeja.
- **FoodBD:** a proteína é estimada por nutricionista, não pesada.
- **Receitas:** o rótulo é proteína por grama dos ingredientes crus, e a foto não é a porção.
- **Flavoria, FPB e LeFood:** a proteína delas é só de tabela.
- **OmniFood8K:** excluído por 16.1.

**Conclusão, pela regra escrita antes:**
- **com dados públicos não se consegue demonstrar os 70%**;
- **a demonstração exige o conjunto brasileiro, enquadrado pelo guia** (`docs/coleta-brasil.md`);
- mudar 16.1 para aceitar um teste "de bandeja, oblíquo" mediria outra coisa que não o app, e não é decisão minha.

**O que muda daqui em diante.** As próximas execuções servem para **melhorar** o modelo, julgadas pela validação cruzada no recorte (seção 15). **Nenhuma delas pode ser apresentada como meta atingida.** As leituras de partes seguradas continuam limitadas como em 15.5 e 16.2.

### 16.6 Emenda: o DFood-TJ como teste de registro em dados públicos (16/11/2026)

**Escrito antes de qualquer modelo prever estas fotos.**

**Decisão do coordenador** (a fundadora delegou as decisões de treino): o DFood-TJ (`fontes_treino.md`, 13.5.1; CC0 1.0) passa a ser **o teste de registro em dados públicos** e **não é usado em treino nenhum**. Ele atende o que o ACETADA não atendia:
- foto de cima de **um prato só**, com todos os alimentos;
- cada alimento **pesado** (balança de 0,1 g);
- proteína = massa pesada × tabela dos autores, recalculada; o fígado de porco foi corrigido.

O que esta seção faz é fixar a população, a métrica e a disciplina de leitura. **Os desvios em relação a 16.1 foram decididos depois de ver as contagens, mas antes de qualquer previsão,** e estão marcados como desvios.

**a) População (tabela `teste_r8_dfoodtj.csv`, gerada por `eval/treino/dfoodtj_teste_divide.py`).**

| Passo | Fotos (= refeições) | Pacientes |
|---|---|---|
| Fotos `zenital` | 920 | 90 |
| Com proteína (sem `nome_difere_do_id`, cujos alvos estão vazios) | 908 | 90 |
| Sem os pares de vista incerta (`olhada_em_par_incerta`, `olhada_sozinha_incerta`) | 733 | 89 |
| Sem `pesos_redondos` (pesos estimados) | 721 | 89 |
| Sem `rotulo_difere_no_par` (rótulo duvidoso entre as duas fotos) | 711 | 89 |
| **População final** | **711** | **89** |
| … 2020–2023 (prato redondo, Tianjin) | 464 | 59 |
| … 2024–2025 (prato oval, inclui Xangai) | 247 | 30 |

- **Ovo pesado com casca (163 refeições):** a proteína do ovo cozido (`food_id` 6, 13,1 g/100 g) é multiplicada pela **fração comestível de 0,88**. É a casca de 12% da tabela do USDA para ovo de galinha. Corrigir, em vez de excluir, mantém os cafés da manhã, que são a maior parte da faixa baixa.
- **`food_id_repetido`, `kcal_incoerente` e `proteina_autores_difere` ficam:** não mexem na proteína depois da correção da lista.
- **O visor da balança: o rótulo escrito na foto.** Olhei duas folhas de contato das zenitais de 2024–2025 (50 fotos). Na maioria, o visor aceso, com o peso legível, está ao lado do prato; no resto, está parcialmente escondido por ele. Nas fotos de 2020–2023 a balança é uma plataforma azul embaixo do prato, com o visor quase sempre apagado ou fora do quadro. A regra, sem olhar rótulo nenhum:
  1. **Em todas as fotos do teste, o kernel mascara o visor.** O OWLv2 procura "a digital display", "an LCD screen" e "a kitchen scale display" (nota de 0,15 ou mais), e cada caixa encontrada é pintada com a cor mediana da borda da foto, antes do recorte.
  2. **Conferência a olho, antes da leitura:** em folhas com o recorte já mascarado de **todas** as 711 fotos, cada foto em que ainda se lê algum dígito do visor é excluída, uma a uma. A lista das excluídas fica gravada antes da leitura.
  3. **Se mais de 10% das fotos de 2024–2025 ainda mostrarem dígitos,** o mascaramento falhou: o período 2024–2025 sai inteiro, e o teste fica com os 464 pratos de 2020–2023, de 59 pacientes.
- **Faixas (desvio de 16.1).** As cinco faixas têm 1 / 74 / 417 / 200 / 19 refeições. Juntando a faixa de 0–5 g à de 5–15 e a acima de 40 à de 25–40, como 16.1 (4) permite, ficam **três faixas com o mesmo peso (1/3 cada): até 15 g (75), 15–25 g (417), acima de 25 g (219).** O teste **não mede separado os pratos acima de 40 g**: são só 19.
- **Unidade: o paciente (desvio de 16.1).** São 89 pacientes, abaixo dos 100 de 16.1 (3). Aceito, pelas contas:
  - a meia-largura do IC de 95% por bootstrap de paciente, simulada com a estrutura real (refeições por paciente e por faixa) e um acerto de 75 / 65 / 50% por faixa, é de **4,4 pontos** sem efeito de paciente, **5,0** com um efeito moderado (desvio de 0,5 em logit) e **6,5** com um forte (1,0);
  - a largura vem mais da faixa de até 15 g, que tem só 75 refeições e pesa 1/3, do que do número de pacientes;
  - com os 59 pacientes de 2020–2023, as larguras seriam de 4,8, 5,9 e 7,4.
- **Tabelas, com sha256 gravado:**
  - `teste_r8_dfoodtj.csv`, sem rótulo, a única que vai para o kernel: sha256 `534a4627…dce5ef5d`;
  - `gabarito_r8_dfoodtj.csv`, com a proteína corrigida, só para a leitura na CPU: sha256 `fdf2e172…1fc07fde6`.
  - As duas ficam fora do repositório, no scratchpad; o script que as refaz está no repositório.

**b) Entrada, métrica e regra.**
- **Entrada:** a foto zenital com o visor mascarado, recortada na janela do guia pela regra só de prato e tigela da etapa 0 (14.3 e emenda; detector e versão iguais). Se não houver prato, vale a elipse; se não houver elipse, a foto inteira, marcada.
- **Conferência do recorte, como em 14.9:** 40 fotos por período, com 36 certas e no máximo 1 errada. Se um período falhar, ele entra com a foto inteira, sinalizado.
- **Métrica:** o critério do produto (±20% ou ±5 g), crua, **sem constante de domínio**, nas três faixas com o mesmo peso. IC de 95% por bootstrap de paciente (4.000 sorteios).
- **Sucesso:** **limite inferior de 95% de 70% ou mais**, no intervalo de Bonferroni da leitura número k (16.2).

**c) Candidato e plano de leitura.**
- **Um candidato só para a regra: o pipeline do app em vigor, o S7** (17.8 g), congelado: R4c × H-cap, rota r4 com os pesos de 17.8, 4 orientações.
- **Na mesma leitura, como linhas descritivas que nunca podem reclamar a meta:**
  - o S6 (R4c × H0c-m, r2, 4 orientações);
  - o modelo de aparelho F5c;
  - a subpopulação 2020–2023 e cada período em separado;
  - os pratos que não tiveram ovo corrigido;
  - o erro de massa cru;
  - kcal, carboidrato e gordura (12.8).
- Se algum desses parecer melhor que o S7, ele só pode reclamar a meta numa **leitura nova**, como candidato congelado, com Bonferroni k = 2.
- **Teto de 3 leituras** (16.2). A primeira é esta.

**d) O que o teste representa e o que não representa.**
- **Representa:** refeições de hospital de pacientes diabéticos na China (Tianjin e pelo menos parte em Xangai, 2020–2025), de uma ou duas cozinhas, com um modelo de prato por período, tudo num prato branco grande, fotografado de celular quase de cima.
- **Não representa:**
  - comida brasileira;
  - pratos de casa ou de restaurante;
  - pratos de tamanhos variados;
  - porções acima de 40 g de proteína;
  - fotos tiradas pelo usuário com o guia: o enquadramento aqui é o do detector.
- Um sucesso aqui seria "70% demonstrado em refeições hospitalares chinesas de cima, num prato padrão". A meta do produto continua dependendo do conjunto brasileiro.

**e) O que eu espero.**
- **Medida mais próxima:** as cenas de pratos chineses nunca vistos do OmniFood8K, com 64,6–64,9% em quatro faixas.
- **A favor:** um prato só, de cima, e muita refeição na faixa de 15 a 25 g, que no Nutrition5k acerta 64–69%.
- **Contra:**
  - prato e escala diferentes dos do treino: a massa crua fora do Nutrition5k acerta 45–69% a ±20%;
  - pratos mistos (refogados com carne) e pão de vapor, que o treino quase não tem;
  - tabela por prato pronto.
- **Minha estimativa:** de **55% a 66%** nas três faixas, com o ponto em torno de 60%.
- **Chance de o limite inferior chegar a 70%: menos de 3%.** Para isso o acerto verdadeiro teria de estar perto de 75%.

**f) O que o kernel faz** (especificação para o agente do Kaggle; nada é pontuado nele).
1. **Entrada:** `teste_r8_dfoodtj.csv` (sem rótulo; o kernel confere o sha256). Nenhuma coluna de alvo entra no pacote.
2. **Download por Range**, sem baixar o zip inteiro:
   - lê o diretório central de `https://china.scidb.cn/download?fileId=cb2751a22a516c057913e15b7d3cc5c4` (4.247.942.139 bytes; `DFoodTJ-F-Nutr.zip`, md5 `b67cc4116e2c734296fbd6cd99cd50bc`) e baixa, em até 14 conexões paralelas, só os 711 PNG da coluna `arquivo` (cerca de 0,7 GB; uns 15 a 25 minutos, mais lento numa conexão só);
   - confere cada PNG (assinatura e `IEND`) e grava o sha256 de cada um.
3. **Detecção:**
   - (a) a máscara do visor, como em (a), com caixas e nota gravadas;
   - (b) o prato, pela regra só de prato e tigela de 14.3: OWLv2 `google/owlv2-base-patch16-ensemble@cfd3195ba4ea`, 960 px, nota de 0,15, área de 2%, NMS de 0,5, elipse e `sem_prato` como reserva;
   - (c) a janela do guia: 4:3, com o lado maior da caixa a 80% da altura, preenchida com a cor mediana.
4. **Folhas de contato, só na saída do Kaggle:**
   - todas as 711 janelas já mascaradas, em 30 por folha, em tamanho suficiente para ler um visor;
   - 40 por período com a caixa desenhada, para a conferência de 14.9;
   - um CSV com posição na folha → `dish_id` e a coluna `veredito` vazia.
5. **Features e previsões:**
   - So400m (o mesmo checkpoint) sobre as janelas, com 1, 4 e 8 orientações;
   - saídas das cabeças `cabeca__gpu5_ridge_so400m_prato__tudo.npz` (R4c), `cabeca__gpu7_fracoes_so400m_prato__massa_cap.npz` (H-cap, 3 sementes) e `cabeca__gpu7_fracoes_so400m_prato__massa.npz` (H0c-m, 3 sementes), com a entrada de massa da R4c como no serviço;
   - o F5c (`pesos__gpu6_ft_siglip2_b16_224_r2_fracoes_sg_prato__tudo.pt` e `cabecasg__gpu6_ft_siglip2_b16_224_r2_fracoes_sg_prato__tudo.npz`), na mesma janela, com 1 orientação;
   - grava as 22 saídas, as frações e a densidade das frações por orientação, mais a foto inteira como variante de relato;
   - **as rotas (r2, r4) e as métricas são aplicadas depois, na CPU.**
6. **Registro:** versões, hashes, tempos e a contagem de fotos com recorte, elipse, `sem_prato` e máscara de visor.
7. **Custo:** uns 30 minutos de GPU, mais o download.

**g) Ordem dos passos.**
1. Kernel.
2. Conferência a olho das folhas: visor e recorte. O resultado é escrito aqui antes da leitura.
3. Leitura única na CPU, com marca, pelo gabarito.
4. Resultado em 16.7.

**h) O kernel (oitava execução, 17/11/2026).** Escrito antes de olhar qualquer previsão.
- **Fotos:** 711 de 711, baixadas do espelho verificado e conferidas (CRC e sha256), sem nenhuma excluída. O sha256 da tabela de teste confere no manifesto.
- **Um arquivo fora do padrão:** o `dftj_000548` é um JPEG gravado com extensão `.png`. Decodifica normalmente e foi **aceito por decisão do coordenador**. A decisão não depende do rótulo: é sobre o formato do arquivo, tomada antes de qualquer previsão.
- **Detecção pela regra só de prato:** OWLv2 em 539 fotos, elipse em 50 e **`sem_prato` (foto inteira) em 122**, ou 17%. É mais que em qualquer domínio anterior.
- **Máscara de visor:** aplicada em 440 fotos.
- **Remontagem conferida:** `confere` verdadeiro, com diferença máxima de 0,0178.

**i) Passo 1: o visor nas 711 janelas mascaradas (17/11/2026).** Olhei as 24 folhas (`folha_r8_janelas_01` a `24`), uma a uma.
- **Nenhuma janela mostra dígito legível de balança.** A lista de exclusões está vazia: 0 de 464 em 2020–2023 e 0 de 247 em 2024–2025.
  - A coluna `veredito` foi preenchida com `sem_digitos` nas 711 linhas, numa cópia fora do repositório: `folha_r8_janelas_veredito.csv`, sha256 `492ba89b…2a414b03`.
  - As quatro fotos de 2024–2025 sem máscara (`dftj_003101`, `003192`, `003237`, `003260`) mostram a borda da balança sob o prato, mas não dígitos legíveis no tamanho da folha.
- **Texto que não é da balança, e fica:** a marca d'água de câmera "SHOT ON MI 8", um cartão azul, uma etiqueta rosa escrita à mão (`dftj_000729`) e folhas impressas (`dftj_000360`, `000754`). Nenhum deles é o peso.
- **Pela regra de (a), nenhuma foto sai e o período 2024–2025 não sai** (0% com dígito, contra o limite de 10%).
- **Mas a máscara falhou de outro jeito, que a regra de (a) não previa: ela esconde a comida.**
  - A consulta "a kitchen scale display" disparou 228 vezes em 2020–2023 e 367 em 2024–2025. Muitas vezes ela marca a balança inteira, que fica embaixo do prato, e não só o visor. A caixa é pintada de cinza por cima do prato.
  - **Medida na janela que o modelo vê** (fração da área da janela, dentro da foto, coberta por caixa de visor), pelo CSV de caixas:

| Período | Fotos | Janela sem máscara | ≤10% | 10–25% | 25–50% | >50% |
|---|---|---|---|---|---|---|
| 2020–2023 | 464 | 280 | 301 | 10 | 28 | 125 |
| 2024–2025 | 247 | 4 | 48 | 92 | 51 | 56 |

  (A coluna "≤10%" inclui as janelas sem máscara.)
  - **As 172 fotos de elipse ou `sem_prato` têm todas uma caixa de visor:** 50 de 50 e 122 de 122. A janela mascarada tem mediana de 52% (elipse, 2020–2023), 30% (elipse, 2024–2025), 60% (`sem_prato`, 2020–2023) e 54% (`sem_prato`, 2024–2025). **Foi a máscara que tirou o prato do detector, não o modelo de prato de um período.** Nas folhas, essas fotos aparecem como um retângulo cinza sobre o prato, às vezes com só uma lasca da borda à vista.
  - Em 2024–2025, o caso típico com OWLv2 é uma faixa cinza na lateral do prato oval, onde fica o visor. Ela cobre de 10% a 25% da janela e corta parte do arroz ou de outro alimento.
  - Há também casos em que a máscara acerta: cobre só um visor pequeno na borda (por exemplo, `dftj_000789`, `000803`, `000851`, `000854`, `000877`, e as últimas fotos de 2024–2025, `dftj_003152` em diante).
  - **No total, 260 de 711 janelas (37%) têm mais de 25% da área coberta de cinza:** 153 em 2020–2023 (33%) e 107 em 2024–2025 (43%). Nelas a comida, que é o que o modelo mede, está em boa parte escondida.

**j) Passo 2: conferência do recorte, 40 por período (17/11/2026).** Escrito antes da leitura.
- **O que eu julguei:** a janela que o modelo recebe, já mascarada, porque a máscara faz parte da entrada registrada em (b). As classes são as de 14.3:
  - **certa:** a refeição inteira à vista;
  - **cortada:** parte da comida fora da janela ou sob o cinza;
  - **errada:** a comida quase toda escondida, ou a janela em outra coisa.
- **2020–2023: 30 certas, 0 cortadas, 10 erradas. Reprovado** (a exigência é de 36 certas e no máximo 1 errada).
  - As 10 erradas são todas efeito da máscara: as 8 da amostra com elipse ou `sem_prato` (`dftj_000071`, `000093`, `000164`, `000176`, `000254`, `000594`, `000673`, `000879`) e duas com OWLv2 de nota baixa, que acharam um pedaço de prato fora do cinza (`dftj_000573`, nota 0,20, e `000764`, nota 0,32).
  - As 30 certas são fotos sem máscara ou com a máscara fora do prato. Nelas a caixa do prato está certa.
- **2024–2025: 8 certas, 16 cortadas, 16 erradas. Reprovado.**
  - **Erradas:** um retângulo cinza sobre o centro do prato (`dftj_001124`, `001158`, `001192`, `001322`, `001407`, `001452`, `001515`, `001532`, `001537`, `001549`, `001583`, `001891`, `002086`, `002199`, `002418`, `002636`).
  - **Cortadas:** a faixa cinza do visor entra no prato e cobre parte do arroz ou de outro alimento (`dftj_001424`, `001435`, `001441`, `001543`, `001628`, `001674`, `002210`, `002821`, `002843`, `002921`, `002927`, `002989`, `003073`, `003152`, `003209`, `003266`).
  - **Certas:** `dftj_001991`, `002216`, `002553`, `002714`, `002883`, `003001`, `003198` e `003237`.
  - Como em 2020–2023, a caixa vermelha do prato quase sempre pega o prato. O defeito é a máscara, não o detector de prato.
- **Os 122 `sem_prato` e os 50 de elipse:** todos têm caixa de visor, e a janela tem mediana de 30% a 60% de cinza. **A causa é a máscara**, que cobre o prato e tira do OWLv2 o que ele procuraria. Não é o modelo de prato de um período: em 2020–2023, o mesmo detector acerta o prato redondo em todas as fotos sem máscara da amostra.
- **O que a regra registrada manda, aplicada ao pé da letra.** Em (b): "se um período falhar, ele entra com a foto inteira, sinalizado". **Os dois períodos falharam, então os dois entram com a foto inteira, sinalizados.**
  - **Isso não conserta o defeito.** A foto inteira do kernel é recortada da mesma imagem já mascarada (`teste_r8.py`, `inteira[i] = E.recorta(m, …)`, com `m` mascarada). A comida escondida na janela continua escondida na foto inteira. Pela coluna de cobertura, o cinza ocupa uma mediana de 59% da foto inteira nas 197 fotos mascaradas de 2020–2023 e de 12% nas 243 de 2024–2025.
  - A reserva de (b) foi pensada para falha de enquadramento, não para uma entrada sem comida. **Não invento regra nova:** a leitura sai com a foto inteira, como manda (b), com esta marca. Ela **conta como a leitura 1 de 3**.
  - **Consequência para a interpretação, declarada antes de ver o número:**
    - um fracasso não pode ser atribuído ao modelo: ele mistura o modelo com uma entrada em que 37% das janelas têm mais de um quarto de cinza;
    - um sucesso continuaria válido como sucesso, porque o defeito só pode baixar o acerto. Mas a chance de sucesso já era abaixo de 3% (e) e agora é menor.
  - Um teste que meça o modelo neste conjunto exige refazer a máscara (só o visor, não a balança inteira) e repetir os passos 1 e 2. Fica como proposta para o coordenador em 16.7, não como regra aplicada aqui.

**k) Passo 3: o que a leitura única calcula (fixado antes de rodar; `eval/treino/r8_avalia.py`, com marca).**
- **Regra:** o S7 na **foto inteira mascarada**, 4 orientações (as 4 primeiras do grupo diedral, como no treino), rota r4 com os pesos de 17.8. Critério de três faixas, IC de 95% por paciente (4.000 sorteios). Sucesso se o limite inferior for de 70% ou mais.
- **Linhas descritivas de (c), todas na foto inteira:**
  - S6 (R4c × H0c-m, r2, 4 orientações; a H0c-m da sétima execução, média de 3 sementes);
  - F5c (1 orientação, rota massa × densidade média);
  - S7 por período;
  - S7 com a proteína da lista, sem a correção do ovo;
  - massa crua do S7 (R4c) a ±20%, sem piso, com viés mediano;
  - kcal, carboidrato e gordura do S7, com as rotas e os critérios de 12.8 (cinco faixas e pisos de 12.8). O gabarito traz os macros da lista, sem tirar a casca do ovo.
- **Linhas descritivas a mais, declaradas agora, que também não podem reclamar a meta:**
  - o S7 na **janela do guia mascarada**, a entrada que valeria se a conferência tivesse passado;
  - o S7, nas duas entradas, separado pela cobertura de cinza na janela (até 10%: 349 fotos; mais de 25%: 260). Os grupos foram fixados pela tabela de caixas, sem rótulo.
- O acerto por faixa e o viés mediano saem para a regra e para as linhas descritivas.

### 16.7 Resultado do teste de registro DFood-TJ (17/11/2026)

**Leitura 1 de 3**, feita uma vez em 17/11/2026 às 01:45. A marca `r8_leitura_lida.json` impede a repetição. Script: `eval/treino/r8_avalia.py`.
- Antes da leitura, o mesmo script rodou com rótulos sorteados (`--ensaio`), sem abrir o gabarito, só para não perder a leitura por erro de código.
- O sha256 do gabarito conferiu com o de 16.6.
- **Pesos da rota r4:** usei os de 17.8 (g), ajustados com 4 orientações (0,4903 / 0,2609 / 0,3018 / −0,1558). O `r7_selecao.json` guarda no campo `W` os pesos de 1 orientação (0,5035 / 0,2071 / 0,3446 / −0,1586). O código de 17.8 usa esses nas bases externas, o que está certo, porque lá as previsões são de 1 orientação (conferido em 17.8 f).

**a) Veredito: meta não atingida.**
- **S7 na foto inteira mascarada: 26,9%** (IC de 95% por paciente: 22,1 a 31,9).
- O limite inferior, 22,1, fica 48 pontos abaixo de 70.
- O número também fica uns 30 pontos abaixo da faixa que eu esperava (55% a 66%; 16.6 e).

| Linha (711 refeições, 89 pacientes, salvo indicação) | Acerto | IC 95% | Por faixa (até 15 / 15–25 / acima de 25 g) | Viés mediano por faixa |
|---|---|---|---|---|
| **S7, foto inteira (regra)** | **26,9%** | 22,1 a 31,9 | 35 / 28 / 18 | −17% / −40% / −45% |
| S6, foto inteira | 24,4% | 19,9 a 29,5 | 36 / 23 / 15 | −29% / −50% / −55% |
| F5c, foto inteira, 1 orientação | 25,5% | 21,1 a 30,2 | 33 / 25 / 18 | −29% / −44% / −51% |
| S7, 2020–2023 (464; 59 pacientes) | 24,8% | 19,7 a 30,2 | 35 / 28 / 11 | −21% / −39% / −49% |
| S7, 2024–2025 (247; 30 pacientes) | 27,7% | 16,0 a 36,7 | 25 / 27 / 31 | +45% / −48% / −33% |
| S7, proteína da lista (sem a correção do ovo) | 26,0% | 21,0 a 31,2 | 34 / 26 / 18 | −18% / −42% / −45% |
| S7, janela do guia mascarada | 28,9% | 23,9 a 34,2 | 33 / 30 / 24 | −12% / −36% / −40% |
| S7, foto inteira, janela com até 10% de cinza (349; 72 pacientes) | **38,3%** | 31,3 a 46,1 | 53 / 40 / 21 | +22% / −24% / −39% |
| S7, janela do guia, janela com até 10% de cinza | **39,8%** | 32,3 a 47,7 | 51 / 43 / 25 | +26% / −21% / −32% |
| S7, foto inteira, janela com mais de 25% de cinza (260; 69 pacientes) | 2,8% | 0,2 a 6,8 | 7 / 1 / 0 | −92% / −94% / −95% |
| S7, janela do guia, janela com mais de 25% de cinza | 3,7% | 0,7 a 8,0 | 7 / 1 / 3 | −91% / −93% / −94% |

- **Massa (R4c) a ±20%:**
  - foto inteira: 11,7%, com viés mediano de −54%; por período, 5,6% (2020–2023) e 23,1% (2024–2025);
  - janela do guia: 15,8% e −49%; por período, 6,9% e 32,4%.
- **kcal, carboidrato e gordura** (S7, foto inteira, critério de 12.8): 21,9%, 29,3% e 30,5%, com viés mediano de −61%, −79% e −56%.
  - **Os intervalos desses três são suspeitos:** kcal de 5,7 a 29,9 e carboidrato de 0,0 a 33,5, com o valor pontual quase no limite de cima.
  - A causa provável são faixas de 12.8 com pouquíssimas refeições nesta população: o critério tira do cálculo a faixa que fica vazia num sorteio.
  - Não investiguei (ver d). Dos macros, só vale que estão muito abaixo do que a validação cruzada dava.

**b) Interpretação.**
1. **Primeiro a máscara, depois o modelo.**
   - Nas 260 janelas com mais de um quarto de cinza, o acerto é de 3%, e a estimativa sai cerca de 93% abaixo do real. O modelo vê um retângulo cinza e responde "quase nada".
   - Essas fotos são 37% do teste e puxam o número da regra para baixo. **Os 26,9% medem o pipeline com esta máscara, não o modelo neste conjunto.** Isso foi declarado antes da leitura (16.6 j).
2. **Sem a máscara no caminho, o modelo ainda fica muito abaixo do esperado.**
   - Nas 349 refeições cuja janela tem até 10% de cinza, o S7 acerta **38% a 40%**. Isso é cerca de 25 pontos abaixo dos 64,6% da parte segurada do OmniFood8K e de 15 a 28 pontos abaixo da faixa que eu esperava.
   - O erro é de subestimação: −21% a −24% na faixa de 15 a 25 g e −32% a −39% acima de 25 g.
   - A massa da R4c sai, no conjunto, 49% a 54% abaixo da pesada. As linhas declaradas não separam massa e densidade dentro do grupo com pouco cinza. **A hipótese mais provável, não testada:**
     - a janela do guia põe o prato a 80% da altura, e o modelo aprendeu o tamanho de prato das bases de treino;
     - o prato do hospital é grande e as porções são de hospital, e a janela não carrega o tamanho real.
   - É o aviso de 17.8: **o guia ou o tamanho do prato é condição para estes números valerem.** Aqui o guia existe, mas um prato maior que os do treino continua sendo lido como um prato de tamanho comum.
3. **Não é um defeito só do S7.** O S6 (24,4%) e o F5c (25,5%) caem do mesmo jeito.
4. **A janela do guia fica 2 pontos à frente da foto inteira** (28,9% contra 26,9%; 39,8% contra 38,3% com pouco cinza). É o sentido esperado, porque o treino é no recorte.
5. **A correção do ovo não muda o quadro** (26,0% sem ela).
6. **Os períodos não se separam** (24,8% e 27,7%). O intervalo de 2024–2025 é largo, com 30 pacientes.
7. **O grupo com pouco cinza não é uma amostra ao acaso.** São quase todas fotos de 2020–2023 (301 das 349), a maioria sem caixa de visor nenhuma na janela (280). Ele serve de diagnóstico, não de estimativa do acerto no conjunto.

**c) O que o teste representa e o que não representa.**
- **Não representa** o acerto do pipeline do app em refeições hospitalares chinesas de cima, que é o que 16.6 (d) queria medir. A entrada registrada escondeu a comida em mais de um terço das fotos.
- **Representa:**
  - que a meta **não foi demonstrada em dados públicos**, pela regra escrita antes;
  - que, mesmo onde a máscara não atrapalha, um prato grande e padronizado, diferente dos do treino, leva a proteína para 38% a 40%, com subestimação forte nas porções maiores.
- O segundo ponto é a observação mais útil do teste: **sem a escala real (o diâmetro do prato ou outra referência), o recorte pelo guia não transfere para pratos de outro tamanho.**

**d) Disciplina e o que ficou de fora.**
- **Leituras:** 1 de 3 usada. Uma segunda leitura do S7 teria Bonferroni de 97,5%.
  - Com 38% a 40% nas fotos limpas, nenhuma leitura realista chega a 70%.
  - **A condição de saída de 16.5 se confirma, agora com um teste de registro de verdade:** com os dados públicos e este pipeline, os 70% não se demonstram.
- **Minha expectativa errou por cerca de 30 pontos.** Errei também no passo de desenho:
  - a conferência do visor em 16.6 olhou duas folhas de fotos originais, sem máscara;
  - não testei a máscara numa amostra antes do kernel;
  - a consulta "a kitchen scale display" marca a balança inteira, que fica embaixo do prato.
  - A conferência a olho dos passos 1 e 2 pegou o defeito antes da leitura, como devia. **Mas a regra registrada não tinha saída para "a máscara esconde a comida"**: a reserva da foto inteira herda a mesma máscara.
- **Rótulo visto fora da leitura:** ao conferir o cabeçalho do gabarito, imprimi a primeira linha (`dftj_000001`, com a proteína e a massa). É um rótulo só, sem previsão ao lado, e não mudou nenhuma decisão.
- **Análises que não fiz:** depois da leitura, tentei rodar diagnósticos além das linhas declaradas:
  - massa real × densidade prevista no grupo com pouco cinza;
  - o número de refeições por faixa de cada macro.
  - A tentativa foi bloqueada como uma passada a mais pelos rótulos. Concordo: seria análise depois de ver o número, fora do declarado. Fica para o coordenador decidir se vale, com a marca de análise post hoc. **Decidido (17/11/2026):** rodam só depois da leitura 2, marcados como post hoc (16.8 h).

**e) Proposta para o coordenador (não aplicada).**
- Para medir o modelo neste conjunto, e não a máscara, seria preciso um kernel novo:
  - máscara só do visor: as consultas "a digital display" e "an LCD screen", com a área da caixa limitada (por exemplo, até 5% da foto);
  - depois, repetir os passos 1 e 2.
- Seria a leitura 2 de 3. Pelo grupo com pouco cinza, a expectativa honesta é de 35% a 45%: **o veredito não mudaria**.
- O valor dela seria só de diagnóstico: separar quanto do erro é escala (massa) e quanto é densidade num domínio novo.
- A alavanca que 16.7 aponta continua a de 17.8: **o tamanho do prato como entrada**, a ser medido no conjunto brasileiro.
- **Aceita pelo coordenador em 17/11/2026:** vira a emenda 16.8. A nota sobre os pesos da r4 foi conferida em 17.8 (f): não havia erro.

### 16.8 Emenda: segunda leitura do DFood-TJ com a máscara só do visor (17/11/2026)

**Escrita antes do kernel novo, sem nenhuma previsão nova.** O coordenador aceitou a proposta de 16.7 (e); as decisões de treino e avaliação são dele. Esta é a **leitura 2 de 3** do teste de registro DFood-TJ.
- **O que esta leitura quer medir:** o pipeline do app neste conjunto, sem o defeito da máscara de 16.7.
- **O que ela não muda:** a população, a métrica, a regra e o candidato.

**a) O que muda no kernel** (especificação para o agente do Kaggle; eu não mexo em `eval/treino/kaggle/`).
1. **O prato vem primeiro, na foto sem máscara.**
   - É a mesma regra só de prato e tigela de 16.6 (f) 3(b): OWLv2 `google/owlv2-base-patch16-ensemble@cfd3195ba4ea`, 960 px, nota de 0,15, área de 2%, NMS de 0,5, com elipse e `sem_prato` como reserva.
   - A janela do guia também é a mesma: 4:3, com o lado maior da caixa a 80% da altura.
2. **A máscara cobre só o visor.**
   - Só as consultas "a digital display" e "an LCD screen", com nota de 0,15 ou mais. A consulta "a kitchen scale display" sai.
   - Uma caixa só é aceita se ocupar **até 5% da área da foto** e se no máximo **10% da área dela** cair dentro da caixa do prato (ou da caixa da elipse).
   - A caixa aceita é pintada com a cor mediana da borda, como antes, **só fora da caixa do prato**: nenhum pixel dentro da caixa do prato é pintado.
   - Leio "fora da janela do prato" como "fora da caixa do prato", porque o visor que importa fica dentro da janela 4:3, ao lado do prato.
   - Num `sem_prato` não há caixa a proteger, e as caixas aceitas são pintadas inteiras.
3. **A foto inteira** é tirada da mesma imagem com a máscara nova, como variante de relato e como reserva.
4. **Saídas no mesmo formato da oitava execução:**
   - `pred9__dfoodtj.npz`, com as mesmas chaves de `pred8__dfoodtj.npz`: R4c, H-cap e H0c-m com 8 orientações, e F5c com 1, nas variantes `recorte` e `inteira`;
   - `caixas_r9_dfoodtj.csv`, com as colunas da oitava mais `fracao_cinza_janela` (a fração da janela pintada de fato) e `caixas_visor_descartadas` (cada caixa recusada, com o motivo: área ou sobreposição);
   - `folha_r9_janelas_NN.jpg` com todas as 711 janelas já mascaradas, e `folha_r9_janelas.csv`;
   - `folha_r9_conferencia_<período>_{1,2}.jpg` com 40 por período, a caixa do prato e as caixas de visor (aceitas e recusadas, em cores diferentes), e os CSVs;
   - `manifesto_r9.json`.
   - **Os nomes podem continuar os da oitava** (`pred8__…`, `caixas_r8_…`): o kernel em preparo, `teste_r8.py` com a etapa 8b, mantém esses nomes. Nesse caso, as saídas ficam numa pasta nova.
   - O script aceita os dois nomes. Ele recusa a pasta da oitava e recusa uma tabela de caixas sem a coluna `fracao_cinza_janela`.
5. **Registro no log:** as contagens de método (OWLv2 / elipse / `sem_prato`), as caixas de visor aceitas e recusadas por motivo, e a distribuição de `fracao_cinza_janela` por período.
   - O mesmo download conferido (sha256 das 711 fotos, iguais às da oitava).
   - `dftj_000548` continua aceito.

**b) População.** A mesma de 16.6 (a): 711 refeições, 89 pacientes, com as mesmas regras de queda.
- **A conferência dos dígitos é refeita nas folhas novas, foto a foto.** Cada foto em que ainda se lê um dígito do visor sai.
- **Se mais de 10% das fotos de 2024–2025 mostrarem dígitos, o período 2024–2025 sai inteiro**, e o teste fica com 464 refeições de 59 pacientes.
- **Esse risco agora é real.** Na oitava, o visor de 2024–2025 fica colado ao prato oval, e as caixas pequenas de visor encostam na caixa do prato (folhas de conferência da oitava).
  - Uma caixa com mais de 10% dentro do prato é recusada e não pinta nada.
  - Uma caixa aceita não pinta a parte que cai dentro do prato.
  - Nos dois casos o dígito pode ficar à vista. **Acho mais provável que sim do que que não** que o período 2024–2025 caia por essa regra. Se cair, a leitura segue com 2020–2023, como a regra manda.

**c) Conferência a olho, gravada antes da leitura.**
- Os dois passos de 16.6 (g), na mesma ordem: (1) dígitos em todas as janelas; (2) recorte, 40 por período, pela régua de 14.3 (36 certas e no máximo 1 errada).
- **Se um período falhar no recorte, ele entra com a foto inteira, sinalizado.** A regra é a de 16.6 (b), e agora a foto inteira tem a máscara nova.
- O resultado vai para `conferencia_r9.json`, no scratchpad, com:
  - `digitos_legiveis`: a lista de `dish_id`;
  - `recorte_reprovado`: os períodos reprovados.
- O sha256 desse arquivo é escrito aqui antes da leitura. O script lê esse arquivo e aplica sozinho a queda do período e a entrada por período.

**d) Métrica, regra e candidato: os de 16.6, com a correção de Bonferroni da segunda leitura.**
- **Candidato:** o S7 congelado. R4c × H-cap, média das 4 primeiras orientações, rota r4 com os pesos de 4 orientações de 17.8 (g): 0,4903 / 0,2609 / 0,3018 / −0,1558. São os mesmos de 16.7, conferidos em 17.8 (f).
- **Entrada:** a janela do guia com a máscara nova; num período reprovado no recorte, a foto inteira com a máscara nova.
- **Métrica:** o critério do produto, cru, sem constante de domínio. Três faixas de proteína corrigida com o mesmo peso (até 15 / 15–25 / acima de 25 g).
- **Intervalo:** bootstrap por paciente, com 4.000 sorteios.
- **Sucesso: o limite inferior do intervalo de 97,5%** (percentil 1,25; Bonferroni para k = 2) **de 70% ou mais.**
- **Script:** `eval/treino/r8_avalia.py --leitura2`, com marca própria (`r8_leitura2_lida.json`). Ele recusa rodar sem a marca da leitura 1, ou se a da leitura 2 já existir.
  - Antes da leitura, o modo `--ensaio2` roda os mesmos cálculos com rótulos sorteados, sem abrir o gabarito.
  - O código da leitura 2 foi escrito e ensaiado agora, antes do kernel, com as previsões da oitava no lugar das da nona.

**e) Linhas descritivas** (nenhuma reclama a meta). Todas saem com os intervalos de 95% e de 97,5%:
- na entrada da regra: o S6 (R4c × H0c-m, r2, 4 orientações), o F5c (1 orientação), o S7 por período e o S7 com a proteína da lista, sem a correção do ovo;
- o S7 na janela do guia e na foto inteira, as duas com a máscara nova;
- **a separação pelo cinza:**
  - pela cobertura da máscara nova na janela (até 10% / mais de 25%);
  - **pelos mesmos pratos da oitava execução** (os 349 com até 10% de cinza e os 260 com mais de 25% na máscara antiga). A comparação com a leitura 1 fica feita nos mesmos pratos;
- a massa crua da R4c a ±20%, com o viés mediano, por período, nas duas variantes;
- kcal, carboidrato e gordura pelo critério de 12.8, na variante da regra.

**f) O que eu espero.**
- **Ponto de partida:** na leitura 1, as 349 refeições com até 10% de cinza deram 38% a 40%.
  - Com a máscara só do visor, as 260 que estavam cobertas recuperam a comida. Se elas se comportarem como as outras, o total fica perto disso.
  - A subestimação da massa (−49% a −54% na leitura 1) não depende da máscara e continua.
- **Minha estimativa:** **de 32% a 45%**, com o ponto em torno de 38%. O limite inferior de 97,5% deve ficar em torno de 30%.
  - Se 2024–2025 cair pela regra dos dígitos, o número fica perto do que 2020–2023 der sozinho, com o intervalo mais largo.
  - Nos 260 pratos que estavam cobertos, espero de 30% a 40%. Se ficar bem abaixo disso, o defeito não era só a máscara.
- **Chance de sucesso: praticamente zero (bem abaixo de 1%).** **O veredito de 16.7 não deve mudar.**
- O valor desta leitura é de diagnóstico: medir o pipeline sem o defeito e separar escala e densidade em (h).

**g) Ordem dos passos.**
1. Kernel.
2. Conferência dos dígitos.
3. Conferência do recorte.
4. `conferencia_r9.json` e o sha256 dele, escritos aqui.
5. `--ensaio2` com as previsões da nona.
6. Leitura única com `--leitura2`.
7. Diagnósticos post hoc de (h).
8. Resultado, mais abaixo nesta seção.

**h) Diagnósticos post hoc, pedidos pelo coordenador. Só depois da leitura 2, marcados como post hoc.** Explicam o resultado; não escolhem nada e não reclamam a meta.
- O modo `--pos-hoc` recusa rodar antes da marca da leitura 2 e calcula, para a oitava e para a nona, na janela do guia, em todas as refeições e no grupo com até 10% de cinza, por período:
  - **massa real × densidade prevista** (a densidade média do S7) pelo critério de três faixas;
  - **massa prevista × densidade real**;
  - a massa a ±20%, a densidade a ±20% e os vieses medianos de cada uma, com a massa real e a prevista medianas.
- Ele também conta as refeições por faixa de 12.8 para kcal, carboidrato e gordura, para explicar os intervalos estranhos dos macros em 16.7 (a).
- **Leitura dos diagnósticos:**
  - se a massa real × densidade prevista chegar perto de 70%, o problema é escala;
  - se a massa prevista × densidade real ficar perto de 70%, o problema é a densidade (comida chinesa de hospital).

**i) O que esta leitura representa e o que não representa.** O mesmo que 16.6 (d), sem o defeito da máscara: refeições hospitalares chinesas de cima, num prato padrão grande, com o enquadramento do detector. Não mede comida brasileira, prato de casa, nem foto tirada pelo usuário com o guia.

**j) O kernel (execução 8b, 17/11/2026).** Escrito antes de qualquer leitura.
- O kernel é o `teste_r8.py` com a etapa 8b, e mantém os nomes de arquivo da oitava. Saídas em `kaggle_out8b/`.
- **Fotos:** 711 de 711, do espelho verificado, sem nenhuma excluída.
- **Prato:** o OWLv2 achou o prato nas 711 (nenhuma elipse, nenhum `sem_prato`). Sem a máscara antiga, o detector não falha mais.
- **Caixas de visor:**
  - 211 pintadas, 11 descartadas por área e 132 descartadas por sobreposição com o prato;
  - visor pintado em 179 fotos: 2 em 2020–2023 e 177 em 2024–2025.
- **Cinza na janela:** praticamente 0 em 2020–2023. Em 2024–2025, média de 1,3%, mediana de 1,6% e p90 de 2,2%. A remontagem confere.

**k) Passo 1: os dígitos, nas folhas novas (17/11/2026).**
- **Como olhei:**
  - as 24 folhas de janelas;
  - um zoom de cada região de visor, com as caixas aceitas e descartadas (290 fotos; `zoom_visor.py`, no scratchpad);
  - uma varredura de LCD aceso (azul ou verde, saturado) em todas as janelas de 2020–2023, para os visores que o detector não pegou: 6 a mais, nenhum com dígito.
- **Critério:** a foto sai quando dá para **nomear** pelo menos um dígito do visor. Fragmentos de segmento sem dígito nomeável ficam e estão listados no JSON. Os números das teclas da balança vermelha (0, 1, 2, 3, 6…) não são leitura de peso e não contam.
- **2020–2023: 7 fotos com dígito legível.** São visores azuis meio escondidos pelo prato: `dftj_000137`, `000165`, `000179`, `000315`, `000656`, `000665` e `000690`. Elas saem.
  - Ficam 9 fragmentos não nomeáveis: `000307`, `000308`, `000360`, `000511`, `000536`, `000643`, `000654`, `000720` e `000754`.
- **2024–2025: 41 de 247 fotos (16,6%) com o peso legível.** Todas são visores descartados por sobreposição com o prato.
  - A leitura aparece inteira, como "316.0", "588.5", "554.0" ou "612.5".
  - Há mais 12 fragmentos duvidosos, no fim do período, com o visor meio escondido pela borda do prato.
- **Pela regra de 16.6 (a) (3), mais de 10%: o período 2024–2025 sai inteiro.** Era o risco previsto em 16.8 (b).
  - A causa é a própria regra da máscara nova. No prato oval, o visor encosta no prato, e a caixa com mais de 10% dentro da caixa do prato não é pintada.
- **População da leitura 2: 457 refeições de 59 pacientes, todas de 2020–2023** (464 menos 7).

**l) Passo 2: o recorte, 40 por período (17/11/2026).**
- **2020–2023: 39 certas, 0 cortadas, 0 erradas. Aprovado.**
  - Uma janela folgada: `dftj_000699`, em que a caixa pegou o prato e um pedaço da mesa, e o prato ocupa um terço da largura. Não é cortada nem errada. Conto como "não certa", e o período passa do mesmo jeito.
- **2024–2025: 40 certas.** Não importa para a leitura, porque o período já saiu pelos dígitos.
- **Entrada da regra: a janela do guia em 2020–2023.**
- **Arquivo da conferência:** `conferencia_r9.json` (`digitos_legiveis` com os 48 `dish_id`; `recorte_reprovado` vazio), **sha256 `59916f6921cb0a32fc7b2381588c420f62c9cba352e47dbb41be394a58bdc881`**.
- **A separação pelo cinza fica quase vazia nesta leitura:** só 2 das 457 janelas têm cinza, com no máximo 0,9%. O que vale é a separação pelos mesmos pratos da oitava.

**m) Leitura 2 de 3 (17/11/2026), com `--leitura2`.**
- **Antes da leitura:**
  - o `--ensaio2`, com rótulos sorteados, quebrou num grupo vazio (nenhuma janela com mais de 25% de cinza). Corrigi o código para registrar o grupo vazio, sem mudar nenhum cálculo, e o ensaio passou;
  - o ensaio da leitura 1 repete os números de antes;
  - o sha256 da conferência confere (`59916f69…`).
- A marca é `r8_leitura2_lida.json`, e o resultado está em `r8_leitura2.json`, no scratchpad.

**Veredito: meta não atingida.** O S7 na janela do guia, com 457 refeições de 59 pacientes de 2020–2023, acerta **44,1%**.
- Intervalo de 95%: 38,0 a 50,1.
- **Intervalo de 97,5% (Bonferroni, k = 2): 37,0 a 51,3.**
- O limite inferior, 37,0, fica 33 pontos abaixo de 70.
- **O resultado está no topo da faixa que eu esperava (32% a 45%).**

| Linha (457 refeições, 59 pacientes, 2020–2023) | Acerto | IC 95% | IC 97,5% | Por faixa (até 15 / 15–25 / acima de 25 g) | Viés mediano por faixa |
|---|---|---|---|---|---|
| **S7, janela do guia (regra)** | **44,1%** | 38,0 a 50,1 | **37,0 a 51,3** | 60 / 47 / 25 | +21% / −21% / −34% |
| S6, janela | 36,1% | 30,0 a 42,2 | 28,9 a 42,9 | 60 / 33 / 15 | +4% / −30% / −43% |
| F5c, janela, 1 orientação | 38,3% | 33,8 a 43,1 | 33,1 a 43,8 | 35 / 44 / 35 | +55% / −2% / −28% |
| S7, proteína da lista (sem a correção do ovo) | 42,9% | 36,5 a 49,2 | | 60 / 45 / 24 | |
| S7, foto inteira | 40,7% | 35,1 a 45,9 | | 66 / 40 / 15 | +13% / −27% / −40% |
| S7, janela; pratos com até 10% de cinza na oitava (301; 54 pacientes) | 41,4% | 33,3 a 49,5 | | 55 / 48 / 22 | |
| S7, janela; pratos com mais de 25% de cinza na oitava (146; 42 pacientes) | **49,0%** | 40,9 a 56,0 | | 71 / 47 / 29 | |

- **Massa (R4c) a ±20%:** 12,0% na janela, com viés mediano de −42%; 6,8% na foto inteira, com −47%.
- **kcal, carboidrato e gordura** (S7, janela, critério de 12.8): 22,5%, 30,9% e 39,5%, com viés de −47%, −64% e −34%.
  - Os intervalos continuam estranhos (carboidrato de 0,0 a 37,0). O diagnóstico de (n) explica por quê.
- **Os pratos que estavam cobertos na oitava agora acertam 49% (eram 3%).** A máscara era o defeito de 16.7.
  - Os que já estavam limpos acertam 41%, contra 38% a 40% na leitura 1, que incluía 48 pratos de 2024–2025.
  - As duas leituras concordam onde a entrada era a mesma.

**n) Diagnósticos post hoc** (`--pos-hoc`, rodado depois da leitura 2; explicam, não escolhem nada). Na janela do guia, nas 464 refeições de 2020–2023 da nona execução, sem exclusão por dígito:

| | 8ª (máscara antiga) | 9ª (máscara só do visor) |
|---|---|---|
| Massa real, mediana | 503 g | 503 g |
| Massa prevista, mediana | 228 g | 296 g |
| Massa a ±20% (viés mediano) | 6,9% (−51%) | 11,9% (−42%) |
| Densidade de proteína a ±20% (viés mediano) | 15,5% (+23%) | 19,6% (**+57%**) |
| Massa **real** × densidade prevista (três faixas) | 25,8% | 21,0% |
| Massa prevista × densidade **real** (três faixas) | 11,9% | 22,7% |

- **Os dois componentes erram muito, e em sentidos opostos.**
  - A massa sai pouco mais da metade da pesada.
  - A densidade (proteína por grama) sai cerca de 50% acima da real.
  - Os dois erros se compensam em parte, e é por isso que o S7 inteiro (44%) acerta mais do que qualquer um dos dois componentes com o outro trocado pelo valor real (21% a 23%).
  - **Não é só escala.** Com a massa real, a proteína sairia cerca de 50% acima da verdadeira.
- **Uma leitura possível, não testada:**
  - a massa pesada inclui caldo, molho e o arroz denso de uma refeição hospitalar grande (mediana de 503 g, contra cerca de 170 g no Nutrition5k);
  - o modelo vê "um prato cheio" na janela e devolve uma massa e uma composição típicas do treino.
  - Em 2024–2025, só como diagnóstico, a massa prevista × densidade real chega a 55%, e lá o erro maior é o da densidade.
- **Os macros:** as faixas de 12.8 quase não têm refeições nesta população.
  - Carboidrato: 0 / 0 / 5 / 50 / 656 refeições por faixa.
  - kcal: 0 / 7 / 151 / 388 / 165.
  - Num sorteio sem as 5 refeições da faixa de 15 a 30 g, a média muda de base, e por isso os intervalos saem estranhos.
  - **O critério de cinco faixas de 12.8 não serve para esta população.** Os números de macros daqui não devem ser citados.

**o) O que fica.**
1. **Meta não demonstrada, em duas leituras.** A leitura 1 media sobretudo a máscara. A leitura 2 mede o pipeline sem esse defeito: **44%**, com o limite inferior de 97,5% em 37%.
   - Resta 1 leitura (16.2), e nenhuma realista chega a 70%.
   - **A condição de saída de 16.5 está confirmada num teste de registro de verdade:** com os dados públicos e este pipeline, não se demonstram os 70%.
2. **O que a leitura 2 representa:** refeições hospitalares chinesas de 2020–2023, num prato redondo grande, de cima, com o enquadramento do detector.
   - Não representa 2024–2025: saiu pela regra dos dígitos.
   - Não representa comida brasileira.
3. **A regra dos dígitos derrubou 2024–2025.** A máscara nova poupa o prato e, com isso, deixa à vista o visor colado ao prato oval. Era o risco que 16.8 (b) previa. Não vou propor uma terceira máscara: o teto de leituras e o resultado de 2020–2023 tornam isso inútil para a meta.
4. **Para o produto:** em pratos grandes, fora do treino, a massa sai pela metade e a densidade sai alta. **O guia sozinho não carrega a escala nem a composição de uma cozinha nova.** Isso reforça o que 16.7 e 17.8 apontavam (o tamanho do prato como entrada) e acrescenta a densidade como problema de domínio. As duas coisas precisam ser medidas no conjunto brasileiro.

## 17. Sétima execução: a faixa acima de 25 g e a densidade (15/11/2026)

**Desenho e pré-registro, antes de qualquer resultado.** Pela seção 16.5, isto é **melhora, não demonstração**. O alvo é o ponto fraco:
- pratos com mais de 25 g de proteína: 52% no recorte do Nutrition5k e 44% no OmniFood8K segurado;
- o erro de densidade: a densidade do modelo, com a massa real, dá 78–79%.

### 17.1 Protocolo e referência

- **Protocolo:** o da seção 15 (Nutrition5k no recorte, N_c; > 25 g no recorte; O e M crus).
- **Referência: o S6** (R4c×H0c-m, rota r2, 4 orientações): N_c **70,73**, > 25 g **52,11**, O cru **60,21**, M cru **59,23**.
- **Objetivo novo, fixado agora, para pesar a faixa fraca:** **S17 = (N_c + > 25 g no recorte + O cru + M cru) / 4.** O S6 tem 60,57.
- Só validação cruzada. Nenhum teste do Nutrition5k existe, e nenhuma parte segurada é lida, a não ser como em 17.6.

### 17.2 Como cada base nova entra

| Base | Foto e entrada | O que supervisiona | Peso | Parte segurada (nunca escolhe nada) |
|---|---|---|---|---|
| **ACETADA** (639 refeições de treino, 120 participantes; 400 acima de 25 g) | **Foto inteira, oblíqua, sem recorte.** O recorte não isola a bandeja rotulada. Entra encaixada em 4:3, como o Flavoria | Na cabeça de densidade: **log-densidade da bandeja** (proteína ÷ massa servidas, sem viés: é a parte que pode transferir para o prato), frações por fonte (peso 0,5) e proteína direta (com viés por domínio). Na cabeça de massa, só na variante R4c+A: massa da bandeja, **com viés por domínio**, porque a escala oblíqua não é a do guia | Soma de pesos de min(0,5; 639/2.281) = **0,28** do Nutrition5k | **163 refeições de 32 participantes** (`split_ext`): relato de **proteína medida** fora do domínio, em quatro faixas (0–15, 15–25, 25–40, > 40 g; 32 / 40 / 33 / 58 refeições), com bootstrap por participante. A entrada é a foto oblíqua inteira, que não é a do app |
| **Receitas DataHive** (17.494 de treino) | Foto do site, encaixada em 4:3, 1 orientação | **Só a log-densidade, marcada como fraca** (proteína por grama dos ingredientes crus). Nenhum outro alvo. A entrada de massa da H0c-m é a previsão da R4c na própria foto, sem rótulo | Soma de pesos de **0,15** do Nutrition5k, por ser rótulo fraco | 4.365 receitas: relato do erro de densidade (rótulo fraco) |
| **FoodBD** (1.444 de treino, 58 grupos) | De cima, recorte pelo prato com o detector da etapa 0 (regra só de prato e tigela) | **Só a proteína direta, fraca** (estimativa de nutricionista; sem massa, então sem densidade), com viés por domínio | **0,15** | **384 pratos de 17 grupos:** relato com rótulo fraco, no critério do produto, em três faixas (0–5, 5–15, acima de 15 g; quase não há acima de 25 g) |

- **Por que não há vazamento.**
  - Nenhuma base nova está no Nutrition5k. Elas entram em todos os folds de treino (fold −1) e nunca nos folds de validação.
  - A entrada de massa da H0c-m continua sendo a previsão **aninhada** da R4c em cada fold.
  - As partes seguradas das três bases são fixadas pelos `split_ext` das listas (sha256 gravado), e o kernel apaga os rótulos delas.
- **Conferência do recorte no FoodBD (14.3).** Folha de contato de 40 fotos, conferida a olho depois da execução e **antes da escolha**: passa com 36 certas e no máximo 1 errada. Se não passar, as configurações com FoodBD saem e o relato dele usa a foto inteira.
- **Features novas:** só as das fotos novas. Também refaz as do recorte do Nutrition5k e das cinco bases da seção 13, porque as execuções 5 e 6 não as guardaram. **Desta vez o kernel guarda todas as features**, em meia precisão, como dataset privado do Kaggle (cerca de 0,3 GB).

### 17.3 Configurações, em ordem de prioridade

Todas são cabeças sobre features congeladas do So400m. Cada cabeça de densidade treina com **3 sementes** e entra pela média das três: com diferenças de 0,5 ponto em jogo, a variância entre sementes também sai relatada.

| Prioridade | Configuração | GPU (min) |
|---|---|---|
| 1 | Features do recorte do Nutrition5k (8 orientações) e das cinco bases da seção 13, guardadas; paridade da R4c e da H0c-m com a sexta execução (correlação de 0,999 ou mais) | 65 |
| 2 | Features do ACETADA (802 fotos, 2 orientações) | 3 |
| 2 | **H-A:** H0c-m + ACETADA. **R4c+A:** R4c + massa do ACETADA | 10 |
| 3 | Download e features das receitas (21.859 fotos, 1 orientação) | 30 |
| 3 | **H-R:** H0c-m + receitas. **H-AR:** H0c-m + ACETADA + receitas | 15 |
| 4 | Detector e recorte do FoodBD (1.828 fotos), folha de contato, features (2 orientações) | 10 |
| 4 | **H-B:** H0c-m + FoodBD. **H-ARB:** as três | 15 |
| 5 | **H-cap:** H0c-m com duas camadas ocultas de 1.024. Testa se a capacidade da cabeça, e não o dado, é o limite da densidade | 10 |
| 5 | Medidas de serviço do candidato que vier a ser escolhido (1 e 4 orientações) | 10 |

- **Total:** cerca de 170 minutos de GPU e 1,5 a 2 h de relógio nas duas T4. A folga de 4 h fica para o download das receitas, que é o passo incerto.
- **Corte, se faltar cota:** prioridade 5, depois 4, depois 3. A 1 e a 2 não são cortadas.
- **Nada de ajuste fino.** Três execuções mostraram que ele não melhora a massa nem a densidade.

### 17.4 Candidatos e regra R17

- **Candidatos:** massa ∈ {R4c, R4c+A} × densidade ∈ {H0c-m, H-A, H-R, H-AR, H-B, H-ARB, H-cap} × rotas r1–r4 (15.2) × orientações pela regra de 12.3 (item 2). São até 56 candidatos: **o vencedor fica otimista**, e o tamanho desse otimismo não se mede sem teste.
- **Guarda:** N_c ≥ 69,73 e > 25 g no recorte ≥ 49,61 (1,0 e 2,5 abaixo do S6).
- **S7 = o admissível de maior S17.** Empate (menos de 0,5): menos orientações; menos bases não comerciais (ACETADA e receitas são CC BY-NC; FoodBD é CC BY); rota mais simples; menos bases. **O S6 concorre e fica, a menos que alguém tenha S17 de 0,5 ou mais acima dele.**
- **Melhor só com dados liberados,** ao lado: R1c × {H0c, H-B} (FoodBD é CC BY).
- **Aparelho:** sem mudança (F5c). O **par S7 + F5c** segue a regra de 2,0 pontos de 15.5.
- **Relatos que não escolhem nada, para o S6 e o S7:** ACETADA segurado (proteína medida, foto oblíqua); FoodBD segurado (rótulo de nutricionista); densidade nas receitas seguradas; acerto acima de 25 g por base nas partes de treino do OmniFood8K, ACETADA e Nutrition5k.

### 17.5 A pergunta principal: o volume acima de 25 g do ACETADA mexe na faixa fraca?

Medidas fixadas agora, todas na CV:
1. A diferença pareada do critério acima de 25 g no Nutrition5k (recorte), H-A × H0c-m com a R4c e rota r2, com bootstrap por cadeia.
2. O mesmo nas cenas acima de 25 g do OmniFood8K (CV por prato principal, só a faixa > 25 g), com bootstrap por prato principal.
3. O acerto da densidade com a **massa real** (o teto da densidade, 78–79% hoje), H-A × H0c-m. Mostra se o ganho, se houver, é da densidade.

### 17.6 Leituras

- **Nenhuma parte segurada da seção 13 é lida,** a menos que o S7 seja diferente do S6. Nesse caso há uma leitura, só do S7.
  - OmniFood8K, FPB e LeFood: é a terceira leitura, com intervalo de 98,3% (Bonferroni de 16.2), e o teto de 3 é atingido.
  - Mensa: só como teste de escala.
  - **Flavoria, com 3 leituras, está aposentado:** não é lido.
- As partes seguradas do ACETADA, do FoodBD e das receitas são lidas **uma vez**, para o S6 e o S7, como relato. Nunca escolhem nada.

### 17.7 O que eu espero

| Alavanca | Expectativa | Chance |
|---|---|---|
| ACETADA na densidade (H-A) | > 25 g no Nutrition5k: de −1 a +2. No OmniFood8K acima de 25 g: de 0 a +5. O rótulo é da bandeja inteira (dilui a densidade de cada prato), a foto é oblíqua e o cardápio é de laboratório | Melhorar a faixa com IC acima de zero: **20%**. Piorar N_c mais de 1 ponto: 25% |
| ACETADA na massa (R4c+A) | M cru de −3 a +1: a escala oblíqua da bandeja não é a do guia | Ser escolhida: 10% |
| Receitas (H-R) | N_c de −0,5 a +1; densidade crua contra cozida, foto de site | Ajudar: 25%. Atrapalhar (como as frações do Flavoria em 13.1): **40%** |
| FoodBD (H-B) | N_c de 0 ± 0,5. O que vale mais é o relato no prato caseiro de celular | Ser escolhida: 10% |
| Capacidade (H-cap) | 0 a +0,5 | 15% |
| **Conjunto** | Melhor caso: N_c +1 e > 25 g +2 a +3 | S7 diferente do S6: **35%** |
| Relato ACETADA segurado, S6 | Foto oblíqua de bandeja, que o modelo nunca viu: 40% a 55%. Com H-A: 50% a 60% | |

- **O que não espero:** que alguma configuração leve a faixa acima de 25 g do Nutrition5k para perto de 60%.
- O limite da densidade parece ser a identidade e a quantidade da fonte de proteína na foto. Nenhuma das três bases ensina isso no enquadramento do app. Só o conjunto brasileiro, com componentes pesados, ensina.

### 17.8 Resultado da sétima execução (15/11/2026)

A execução rodou no Kaggle sem nada cortado. A ingestão aceitou as 16 configurações, e a paridade com a sexta execução está confirmada: R4c e H0c-m com correlação de 1,00000, critério de CV de 68,12% em r1 com 1 orientação. As features ficaram guardadas (331 MB em meia precisão). Script: `eval/treino/r7_avalia.py`.

**Nota de implementação, decidida pelo coordenador (a fundadora delegou as decisões de treino).** O agente do Kaggle pôs as linhas de treino das bases novas em folds por grupo (participante do ACETADA, nome da receita, bloco do FoodBD), e não no fold −1 de 17.2. Por isso existem previsões fora-do-fold nelas. **Os folds de validação do Nutrition5k nunca contêm essas linhas,** então N_c e > 25 g continuam limpos.

**a) Conferência a olho do recorte do FoodBD (14.3): reprovada.**
- Resultado: 35 certas, 5 cortadas, 0 erradas; a regra pede 36 certas.
- As cinco cortadas são fotos de perto, com a comida enchendo o quadro, ou com mais de um prato. Nesses casos a caixa do prato deixa comida fora da janela.
- **Valem as variantes com a foto inteira do FoodBD** (H-Bi, H-ARBi, H0c+Bi). As de recorte (H-B, H-ARB, H0c+B) **não concorrem**, mas aparecem na tabela.
- A folha com o veredito por foto fica no scratchpad (`conferencia_r7/`), fora do repositório.

**b) Regra R17, só com validação cruzada, escrita antes de abrir as partes seguradas.**
- São 84 combinações: massa {R4c, R4c+A} × 9 cabeças de densidade × rotas r1–r4, mais R1c × {H0c, H0c+Bi} para a referência só com dados liberados.
- As orientações seguem a regra de 12.3; quase todas ficam em 4.
- **Guarda:** N_c ≥ 69,73 e > 25 g ≥ 49,61.
- **S6 de referência**, como registrado em 15.8: S17 = **60,57**.

| Candidato | N_c | > 25 g | O cru | O cru > 25 g | M cru | **S17** | Densidade com massa real |
|---|---|---|---|---|---|---|---|
| **S6** (R4c×H0c-m r2, 4 orientações; 15.8) | 70,73 | 52,11 | 60,21 | — | 59,23 | 60,57 | — |
| S6 refeito (H0c-m da 7ª, média de 3 sementes) | 70,63 | 51,81 | 60,17 | 32,2 | 59,23 | 60,46 | 78,06 |
| **R4c×H-cap r4, 4 orientações = S7** | **71,36** | **55,63** | 59,63 | 21,5 | 59,23 | **61,46** | 78,82 |
| R4c×H-Bi r4 (FoodBD, foto inteira) | 72,05 | 55,04 | 58,21 | 11,6 | 59,23 | 61,13 | 79,00 |
| R4c×H-R r4 (receitas) | 71,80 | 54,43 | 58,83 | 15,7 | 59,23 | 61,07 | 78,17 |
| R4c×H0c-m r4 | 71,78 | 54,70 | 58,53 | 15,7 | 59,23 | 61,06 | 78,06 |
| R4c×H-A r2 (ACETADA) | 70,78 | 51,98 | 60,13 | 31,4 | 59,23 | 60,53 | 78,34 |
| R4c×H-A r4 | 71,01 | 52,88 | 58,74 | 19,0 | 59,23 | 60,46 | 78,34 |
| R4c×H-AR r3 (ACETADA e receitas) | 71,13 | 52,56 | 60,50 | 25,6 | 59,23 | 60,85 | 78,56 |
| R4c+A×H-R r4 (massa com ACETADA) | 72,34 | 54,96 | 58,92 | 16,5 | **57,20** | 60,86 | 78,17 |
| Só dados liberados: R1c×H0c r4 | 72,07 | 54,88 | 57,39 | 14,9 | **51,68** | 59,01 | 77,94 |

(A tabela completa está em `r7_selecao.json`.)

- **S7 = R4c×H-cap, rota r4, 4 orientações.**
  - A H-cap é a mesma cabeça de densidade da H0c-m, com mais capacidade (duas camadas ocultas de 1.024), treinada **só com o Nutrition5k**.
  - O S17 dela (61,46) passa a exigência de 17.4 (S6 + 0,5 = 61,07). É o único candidato que passa, sem empate.
- **De onde vem o ganho, dito sem enfeite: quase todo da mudança de objetivo, não de dado novo.**
  - O S17 dá peso à faixa acima de 25 g, e a rota r4 (a mistura log-linear das três estimativas) acerta mais nessa faixa: 54,7% contra 51,8% da r2 com a mesma H0c-m. Esse ganho já existia na sexta execução (a r4 perdeu lá no desempate do S15).
  - A H-cap acrescenta uns 0,4 ponto, de forma estável: nas três sementes, com r2 e 4 orientações, dá 71,04 / 70,69 / 70,87, contra 70,73 / 70,22 / 70,46 da H0c-m.
  - **Custo:** no OmniFood8K, a r4 acerta menos as cenas acima de 25 g (21,5% contra 32,2% da r2). O ganho na faixa alta do Nutrition5k não se repete fora dele.
- **Dispersão entre sementes** (N_c com R4c, r2, 4 orientações): amplitude de 0,2 a 0,8 ponto. Diferenças de até cerca de 0,5 ponto entre cabeças estão dentro do ruído de treino.
- **Melhor só com dados liberados:** R1c×H0c, r4, 4 orientações. N_c 72,07 e > 25 g 54,88, mas **M cru 51,68**: perde 7,5 pontos de massa fora do Nutrition5k.
- **A massa com o ACETADA (R4c+A) piora M cru** (59,2 → 57,2), como 17.7 esperava.

**c) A pergunta de 17.5: o volume acima de 25 g do ACETADA mexe na faixa fraca? Não.** H-A contra H0c-m, ambas com R4c, r2 e 4 orientações:

| Medida | H0c-m | H-A | Diferença (IC) |
|---|---|---|---|
| 1. > 25 g no Nutrition5k (recorte) | 51,81 | 51,98 | +0,18 (−0,87 a +1,25) |
| 2. Cenas acima de 25 g do OmniFood8K (121 cenas, 27 pratos principais) | 32,2 | 31,4 | −0,83 (−3,77 a +6,03) |
| 3. Densidade com a massa real, todos os pratos / acima de 25 g | 78,06 / 64,83 | 78,34 / 65,50 | +0,28 (−0,59 a +1,20) / +0,67 (−1,18 a +2,64) |

- As 400 refeições acima de 25 g com proteína medida não mexem na faixa. O rótulo é da bandeja inteira e a foto é oblíqua, e isso não ensina a densidade de um prato visto de cima.
- **Receitas (H-R contra H0c-m):** N_c −0,51 (IC −1,15 a +0,15); > 25 g −0,30. Não ajudam.
- **FoodBD (H-Bi, foto inteira):** +0,3 em N_c com a rota r4, dentro do ruído das sementes.

**d) Leitura das partes seguradas da seção 13 (17.6), decidida antes de ler.**
- O S7 usa a **mesma cabeça de massa (R4c)** do S6, e as previsões dela são idênticas. Os números de massa do FPB, do LeFood e do Mensa em 15.8 valem igualmente para o S7. **Ler essas partes de novo não daria número novo e só gastaria leitura:** elas não são lidas e ficam com 2 leituras cada.
- **O Flavoria está aposentado** (3 leituras).
- **Só a parte segurada do OmniFood8K é lida, para o S7.** É a terceira leitura dela, com o intervalo de 98,3%, e com isso ela chega ao teto e se aposenta.

**e) Partes seguradas das bases novas (lidas uma vez, para o S6 e o S7; relato, sem escolher nada).** No S6 a cabeça de densidade é a H0c-m refeita na sétima execução: a da sexta não tem previsões nessas bases.

| Parte segurada | Medida | S6 | S7 |
|---|---|---|---|
| **ACETADA**, 163 refeições de 32 participantes. Proteína medida; foto oblíqua da bandeja, que não é a do app | Critério em quatro faixas, cru | 14,4% (8,1 a 20,9) | 19,7% (14,8 a 25,4) |
| | Com a constante do domínio, estimada nas refeições de treino | 33,7% (×2,97) | **46,4%** (×2,25; IC 35,9 a 55,9) |
| | Massa real × densidade prevista | 52,1% (46,0 a 58,4) | 49,8% (41,1 a 58,0) |
| **Receitas**, 4.365 (rótulo fraco: proteína por grama dos ingredientes crus) | Densidade a ±20% / correlação de postos / viés mediano | 25,9% / 0,61 / +36% | 27,9% / 0,62 / +29% |
| **FoodBD**, 384 pratos caseiros de celular (rótulo de nutricionista; foto inteira, porque o recorte foi reprovado) | Critério em três faixas, cru / com constante | 28,5% / 71,1% (×0,52) | 25,7% / 77,3% (×0,48) |
| | Correlação de postos | 0,54 | 0,57 |

- **ACETADA:**
  - A massa da bandeja oblíqua sai 60–70% abaixo do real (constante de ×2,2 a ×3,0): o modelo do recorte não sabe ler escala numa foto de bandeja inteira.
  - **Mesmo com a massa real, a densidade acerta só 50–52%,** contra 78% no Nutrition5k. A bandeja, com bebidas, sachês e pratos que a foto oblíqua esconde, é outro problema.
- **FoodBD:**
  - Cru, o modelo **superestima em cerca de 90%** a proteína dos pratos caseiros de Bangladesh. Pela foto inteira, de perto, sem guia, a escala parece maior; e o rótulo de nutricionista segue porções padrão.
  - Com uma constante de ×0,5, chega a 71–77%. Isso diz que a **ordem** está razoável (correlação de 0,55), não a escala.
  - **É o caso do app sem o guia,** e reforça que o guia (ou o tamanho do prato) é indispensável.

**f) Terceira e última leitura da parte segurada do OmniFood8K, só do S7** (1.916 cenas, 34 pratos principais; números crus).
- **S7: 64,6%** (IC 95%: 57,1 a 73,3; Bonferroni de 98,3%: 55,6 a 75,4).
- S6 (lido em 15.8): 64,9%.
- **S7 − S6: −0,3** (IC 95%: −3,1 a +1,7).
- Por faixa: S7 99 / 72 / 54 / **33**; S6 98 / 74 / 43 / **44**. **Fora do Nutrition5k, a faixa acima de 25 g cai com a rota r4.**
- Com isso a parte do OmniFood8K chega ao teto de 3 leituras e se aposenta, como o Flavoria. FPB e LeFood ficam com 2 leituras; o Mensa só serve de teste de escala.

**g) O pipeline do app passa a ser o S7.** Muda só a cabeça de densidade e a rota; o resto é o de 15.8 (c).

| Parte | Definição |
|---|---|
| Entrada, backbone e orientações | Como em 15.8: janela do guia; So400m/16 a 384×512 congelado; **4 orientações** |
| Cabeça de massa | **R4c**, a mesma de antes (`cabeca__gpu5_ridge_so400m_prato__tudo.npz`, igual à refeita na sétima) |
| Cabeça de densidade | **H-cap** (`cabeca__gpu7_fracoes_so400m_prato__massa_cap.npz`). Três MLPs (sementes), cada uma com duas camadas ocultas de 1.024, sobre [features ‖ log-massa da R4c]; as saídas são a média das três. Remontagem com diferença máxima de 0,00007 em log |
| Rota | **r4:** log(1 + proteína) = 0,4903 · log(1 + *r1*) + 0,2609 · log(1 + *dR*) + 0,3018 · log(1 + *dH*) − 0,1558. Os pesos são os ajustados no Nutrition5k inteiro com 4 orientações; *r1* = massa × média das duas densidades, *dR* = proteína direta da R4c, *dH* = proteína direta da H-cap. Nos folds, o peso de *r1* variou de 0,47 a 0,54 |
| Custo | ~236 ms por foto na T4 com 4 orientações num lote (258 ms em sequência; 64 ms com 1). CPU: 14,6 s. São 432 M de parâmetros e 864 MB, medidos com a H0c-m de 3 sementes; a H-cap acrescenta cerca de 7 M de parâmetros à cabeça (três MLPs de 3,4 M contra três de 1,2 M) |
| Procedência | A mesma do S6: a cabeça de massa usa FPB e OmniFood8K (CC BY-NC); a H-cap treinou só com o Nutrition5k, mas recebe a massa da R4c. **Não é só de dados liberados.** Alternativa liberada: R1c×H0c, r4, 4 orientações, com N_c 72,1 mas M cru de 51,7% |
| Validação cruzada | N_c **71,4%**, > 25 g no recorte **55,6%**, O cru 59,6%, M cru 59,2% |

O aparelho continua o F5c. O par não foi recalculado: a regra de 15.5 exige 2,0 pontos, e com o S6 ele dava +0,9.

**h) Conclusão.**

1. **As três bases novas não melhoraram o modelo.**
   - O ACETADA não mexe na faixa acima de 25 g: +0,2 no Nutrition5k e −0,8 no OmniFood8K, ICs em torno de zero.
   - As receitas pioram um pouco (N_c −0,5).
   - O FoodBD só serve com a foto inteira e acrescenta +0,3, dentro do ruído das sementes.
   - **A troca do S6 pelo S7 vem do objetivo novo (peso na faixa alta), da rota r4, que já existia, e de uma cabeça de densidade maior (+0,4).** Não vem de dado novo.
2. **A faixa acima de 25 g sobe no Nutrition5k (52% → 56%) e cai no OmniFood8K (44% → 33%).** O ganho nos pratos ricos em proteína não se transfere: a mistura r4 aprendeu a calibração do Nutrition5k.
3. **A meta continua sem demonstração possível com dados públicos (16.5).** As partes seguradas do OmniFood8K e do Flavoria estão aposentadas, e as do FPB e do LeFood medem só massa.
4. **O que os relatos externos mostram para o produto.** Sem o guia (FoodBD, foto inteira) a proteína sai quase o dobro da real. Em bandeja oblíqua (ACETADA) sai menos de um terço. **O guia, ou o tamanho do prato, é a condição para os números deste estudo valerem no app.**
5. **Próxima alavanca:** a coleta brasileira com o guia e o diâmetro do prato (`docs/coleta-brasil.md`). Em dados públicos, a cabeça de densidade sobre o So400m congelado parece ter chegado ao teto: 78–79% com a massa real em todas as variantes desta execução.

**Registro de método.**
- (a) A conferência do FoodBD e a R17 (itens a a d) foram escritas antes de qualquer leitura.
- (b) As partes seguradas das três bases novas foram lidas uma vez, para S6 e S7.
- (c) A parte do OmniFood8K foi lida pela terceira e última vez, só para o S7.
- (d) Nenhum teste do Nutrition5k existe ou foi lido.
- (e) Os pesos da rota r4 nas bases externas são os de 1 orientação, porque as previsões externas não têm variantes de orientação.
- (f) **Nota de conferência (17/11/2026), sem correção de número.** Em 16.7 levantei a suspeita de que a R17 e a leitura do OmniFood8K tivessem aplicado os pesos de 1 orientação a previsões de 4 orientações. **Conferi, e não aplicaram.**
  - No Nutrition5k, `r7_avalia.py` ajusta os pesos da r4 separadamente para cada número de orientações (`pesos_mistura(cs[o], …)`). Refeitos agora pela validação cruzada do treino:
    - 1 orientação: 0,5035 / 0,2071 / 0,3446 / −0,1586;
    - **4 orientações: 0,4903 / 0,2609 / 0,3018 / −0,1558**, os de (g);
    - 8 orientações: 0,4912 / 0,2732 / 0,2893 / −0,1576.
  - O N_c e a faixa acima de 25 g da R17 usam os pesos certos para cada orientação.
  - **Nas bases externas, as previsões são de uma orientação só.** O kernel usa só a vista original nas features externas (`F0[:, :1]`), e o OmniFood8K tem uma vista só (`feat7__omnifood__recorte.npy`, forma [15741, 1, 2304]). O campo `W` de `r7_selecao.json` guarda os pesos de 1 orientação, e é com eles que o O cru da R17 e a leitura de (f) foram feitos. **Está certo e não há o que recalcular.**
  - **A ressalva que fica é de representação, não de conta.** O O cru, os 64,6% do OmniFood8K segurado e a massa crua das bases externas medem o S7 com **1 orientação**. O app serve com 4.
    - No Nutrition5k, passar de 1 para 4 orientações vale +1,6 ponto para o S7 (N_c de 69,78 para 71,36; com 8 orientações, 70,87).
    - Fora dele, a diferença não foi medida.
  - O `r8_avalia.py` (16.7 e 16.8) usa os pesos de 4 orientações, com previsões de 4 orientações.

## 18. Pipeline geométrico (17/11/2026): desenho pré-registrado, sem resultados

**Pergunta da fundadora:** em vez de aprender massa e proteína de ponta a ponta com fotos pesadas, dá para **medir**? A ideia é segmentar e identificar cada alimento, tirar a escala do prato (ou da profundidade), estimar volume e massa por classe e aplicar a tabela. Isso precisa de pouco dado pesado. Esta seção fixa o desenho, os dados, as regras e o que eu espero, **antes de qualquer número**.

### 18.1 O que o DFood-TJ já ensinou (16.8 n)

Nas refeições de 2020–2023, o S7 erra a **massa em −42%** (cerca de 500 g pesados contra 296 g previstos) e a **densidade de proteína em cerca de +57%**.
- **A geometria pode consertar a massa,** desde que a escala esteja certa (o diâmetro do prato) e que a altura e a densidade aparente de cada classe transfiram.
- **A geometria não conserta, sozinha, a densidade.** Ela depende de identificar o alimento e de a tabela da classe valer naquela cozinha. Num refogado misto de hospital chinês, "carne" ou "misto" com a tabela do Nutrition5k pode errar tanto quanto a cabeça aprendida.
- O ganho possível é parcial. Separar os itens dá a cada um a sua densidade: o arroz com a do amido, e não uma média do prato. Isso só ajuda se a segmentação e a identificação acertarem.

### 18.2 Dados: o que calibra e o que testa

| Base | Para quê | Leituras | Escala absoluta |
|---|---|---|---|
| **Nutrition5k, treino (2.281 pratos)** | Calibrar e escolher, **só por validação cruzada** (5 folds por cadeia, os mesmos de sempre). Tem a massa e a proteína por grupo (`m_*` e `p_*`), a profundidade de cima e a câmera fixa | Livre (validação cruzada) | Distância da câmera à mesa pela profundidade (mediana da mesa) e campo de visão nominal do sensor. Conferência: o diâmetro dos pratos que resulta disso tem de cair em tamanhos plausíveis (18 a 30 cm), sem rótulo nenhum |
| **FPB, parte segurada** | Massa absoluta fora do domínio | 2 usadas; **resta 1** | O xadrez de calibração presente em quase toda foto. **Só vale se o tamanho do quadrado estiver documentado** (cartão ou artigo do FPB); senão, o FPB entra só como relato de ordem (correlação de postos), sem massa absoluta |
| **LeFood, parte segurada** | Massa fora do domínio | 2 usadas; resta 1 | Não há referência métrica conhecida. **Não entra** neste teste: sem escala, a geometria não tem o que medir |
| **DFood-TJ (teste de registro)** | O teste que importa: cozinha nova, prato padrão, cada item pesado | 2 usadas; **resta a 3 de 3** | Um modelo de prato por período. O diâmetro físico vem dos **ovos cozidos com casca**: eixo menor de cerca de 4,3 cm, valor de tabela para ovo de galinha, sem rótulo. Ver 18.4 |
| OmniFood8K e Flavoria | — | Aposentados (3 leituras) | — |

- **O que nunca entra:**
  - os polígonos e as fotos `item_balanca` do DFood-TJ (16.6: a base não entra em treino nenhum);
  - os nomes de alimento por foto do DFood-TJ (são rótulo);
  - qualquer rótulo de teste para escolher prior.
- A `density.xlsx` do DFood-TJ (a tabela dos autores) também não entra: ela define o gabarito.

### 18.3 Segmentação e identificação

- **Fonte:**
  - OWLv2 (o mesmo checkpoint de 14.3, 960 px) com um vocabulário fixo, mapeado para os 8 grupos de massa do treino (`proteina_animal`, `leguminosa`, `laticinio`, `amido`, `vegetal`, `gordura_molho`, `misto`, `outros`) e, para a proteína, para as 10 fontes das cabeças de frações;
  - em cada caixa, uma máscara do **SAM 2.1** (`facebook/sam2.1-hiera-large`, do Hugging Face, com a caixa como prompt). Se o SAM não baixar no Kaggle, a reserva é a caixa recortada pela elipse do prato, marcada.
- **Vocabulário, fixado agora:**
  - "rice", "noodles", "bread", "steamed bun", "potato", "corn";
  - "meat", "pork", "beef", "chicken", "fish", "shrimp", "egg", "tofu", "beans", "cheese";
  - "leafy vegetables", "cucumber", "carrot", "cabbage", "broccoli", "mushroom";
  - "sauce", "soup", "stew", "mixed stir-fry".
  - Cada termo tem um grupo e, quando é fonte de proteína, uma fonte. A tabela fica em `eval/geom/vocabulario.csv`, escrita antes do kernel.
- **Regras de máscara:**
  - só dentro do prato;
  - um pixel com duas máscaras fica com a de maior nota;
  - comida que nenhuma caixa cobriu: a área do prato menos as máscaras e menos o branco do prato (limiar de cor fixado no treino) vira `misto`.
- **Conferência a olho, antes de qualquer número:** 40 pratos por base, com as máscaras coloridas por grupo.
  - Régua: **36 pratos ou mais com a comida coberta sem falha grande** (menos de 20% da comida fora de máscara, a olho), e **no máximo 2 com a fonte de proteína principal no grupo errado**.
  - Uma base reprovada não entra com a geometria (não há reserva que meça área sem máscara).
- **No Brasil, depois:** os mesmos grupos, com nomes da TACO no vocabulário.

### 18.4 Escala absoluta, base a base, e o que o app pediria

- **Nutrition5k:**
  - cm por pixel = 2·d·tan(campo/2)/largura, com d a distância mediana câmera–mesa na profundidade de cada prato (fora das máscaras);
  - o campo de visão é o nominal da câmera de cor do sensor usado pelos autores;
  - conferência sem rótulo: a distribuição do diâmetro dos pratos em cm.
- **DFood-TJ:**
  - em cada período, mede-se o eixo menor de cada ovo inteiro, detectado como "a boiled egg in shell", com a elipse ajustada na máscara, junto com o diâmetro do prato em pixels na mesma foto;
  - o diâmetro físico do prato do período = mediana de (prato_px / ovo_px) × 4,3 cm;
  - **a conta usa só a imagem e um valor de tabela.** O peso do ovo no gabarito não é aberto.
  - Depois, cada foto tem a escala pelo prato detectado: o diâmetro físico do período sobre o diâmetro do prato em pixels, com a elipse do prato.
- **FPB:** os cantos do xadrez com `cv2.findChessboardCorners`, mais o tamanho do quadrado documentado (ou nada; ver 18.2).
- **No app:**
  - a usuária informa o **diâmetro do prato uma vez**: escolhe entre 20, 23, 25 e 27 cm, ou mede com a régua da tela;
  - o círculo-guia é casado com a borda do prato;
  - num iPhone com LiDAR, a profundidade com os intrínsecos (`AVDepthData`) dá a escala e, depois, o volume. **Isso precisa ser medido no piloto brasileiro, não suposto:** em 2-B a profundidade do Nutrition5k não serviu para volume.

### 18.5 Volume, massa, proteína e agregação

- **Área por item:** pixels da máscara × (cm/pixel)², em cm².
- **Massa por item:**
  - variante calibrada: área × k_grupo, em g/cm² (altura × densidade aparente);
  - variante de prior: área × h_grupo × d_grupo.
- **Proteína por item:** massa × ρ_fonte (ou ρ_grupo, quando não há fonte de proteína). **A refeição é a soma dos itens.**
- **De onde vêm os parâmetros (nunca de rótulo de teste):**
  - **k_grupo (variante calibrada):** mínimos quadrados não negativos, prato a prato, de massa_real ≈ Σ_g k_g · A_g, **no treino de cada fold do Nutrition5k**. Os `m_*` por grupo entram como restrição auxiliar (a massa de cada grupo contra a área do grupo). Para o teste externo, o ajuste é feito nos 2.281 pratos.
  - **h_grupo (variante de prior):** a mediana da altura da comida acima da superfície do prato, dentro das máscaras do grupo, pela profundidade do Nutrition5k (treino). Não usa grama nenhuma.
  - **d_grupo:** a densidade aparente (g/mL) de alimentos representativos de cada grupo na **FAO/INFOODS Density Database v2.0**; onde ela não tiver, o peso por xícara do USDA FNDDS ÷ 236,6 mL. Os valores ficam em `eval/geom/priors.csv`, com a fonte linha a linha, antes do kernel.
  - **ρ_fonte e ρ_grupo:** proteína ÷ massa por grupo e por fonte, no treino do Nutrition5k (Σ `p_*` / Σ `m_*`). No Brasil, depois, a TACO.
- **Variante de profundidade** (só no Nutrition5k, descritiva): o volume integrado na profundidade dentro de cada máscara × d_grupo. Pela experiência de 2-B, espero pouco dela.

### 18.6 Candidatos, métrica e regra de escolha

| | Massa | Densidade de proteína |
|---|---|---|
| **G1** | Geométrica calibrada (k_grupo) | ρ por item (tabela por grupo e fonte) |
| **G2** | Geométrica de prior (h × d, sem gramas) | ρ por item |
| **G3** | Geométrica calibrada (k_grupo) | A aprendida do S7 (média das duas densidades da H-cap) |
| **G4** | Mistura log-linear, como a r4, de log(1 + G1), log(1 + G3) e das três estimativas do S7, com pesos por fold | — |
| G-prof | Volume pela profundidade × d (só Nutrition5k) | ρ por item |

- **Métrica:** o critério do produto (±20% ou ±5 g), cru, nas faixas de sempre de cada base: 5 faixas no Nutrition5k e 3 no DFood-TJ. A massa a ±20% e o viés mediano saem ao lado.
- **IC:** bootstrap por cadeia no Nutrition5k e por paciente no DFood-TJ.
- **Escolha no Nutrition5k (validação cruzada), pela ordem:**
  1. **Para o app:** um candidato só troca o S7 se N_c ≥ S7 + 1,0 (71,36 + 1,0), com o limite inferior do IC pareado por cadeia acima de 0, e a faixa acima de 25 g ≥ S7 − 2,5. **Espero que nenhum passe:** no Nutrition5k a câmera fixa já dá a escala ao modelo aprendido, e a geometria não tem o que acrescentar.
  2. **Para o DFood-TJ:** o candidato é o melhor de {G1, G3} por N_c. Os dois têm massa **só geométrica**, porque a pergunta é se a geometria conserta a massa. G2 e G4 saem como linhas descritivas.
- **FPB (última leitura, se a escala do xadrez estiver documentada):** a massa a ±20% e o viés de G1, G2 e da R4c. É relato, e lê uma vez.

### 18.7 O DFood-TJ merece a leitura 3? Sim, com condições pré-registradas

- **A leitura 3 não pode mudar o veredito de 16.8.** Com Bonferroni k = 3 (intervalo de 98,3%), passar exigiria acerto verdadeiro perto de 78%, e o S7 está em 44%.
- **O valor dela é outro:** é o único teste público que mede, numa cozinha nova e com cada item pesado, se a **escala pelo diâmetro do prato** conserta a massa. É a aposta do produto brasileiro.
- Guardar a leitura tem pouco valor. Qualquer pipeline futuro enfrenta a mesma densidade de cozinha nova.
- **Ela só acontece se as quatro condições abaixo passarem, todas sem rótulo:**
  1. a conferência das máscaras do DFood-TJ (18.3) passa;
  2. a escala pelos ovos é estável: pelo menos 30 ovos medidos em 2020–2023, dispersão robusta (MAD/mediana × 1,48) do diâmetro implícito do prato de **8% ou menos**, e diâmetro entre 22 e 32 cm;
  3. no Nutrition5k, a massa geométrica do candidato a ±20% fica **no máximo 15 pontos abaixo** da massa da R4c. Isso mostra que a geometria não está quebrada onde a escala é conhecida;
  4. a regra de escolha de 18.6 (2) foi aplicada e o candidato foi congelado e gravado com hash.
- **Se alguma falhar, a leitura 3 não acontece,** e isso fica escrito.
- **Desenho da leitura 3:**
  - **a mesma população da leitura 2:** as 457 refeições de 2020–2023 de `conferencia_r9.json` (sha256 `59916f69…`). Os ovos que dão a escala são de fotos de 2020–2023, que incluem a população; o rótulo não entra em lugar nenhum;
  - a mesma regra de proteína (limite inferior do intervalo de 98,3% de 70% ou mais), com o S7 da leitura 2 lado a lado, **pareado nas mesmas refeições**;
  - **a medida que interessa é a massa:** a massa a ±20% e o viés mediano da massa geométrica, contra os −42% do S7;
  - linhas descritivas: G2, G4, por faixa e por grupo principal, e a massa por item contra o peso do item **só depois da leitura, como post hoc**.

### 18.8 O que eu espero

- **Nutrition5k (validação cruzada):**
  - massa geométrica a ±20% de 35% a 55%, contra cerca de 70% da R4c;
  - G1 e G3 de 50% a 62% no critério;
  - G4 de 71% a 72%, dentro do ruído do S7.
  - **Nenhum troca o app.**
- **FPB:** massa geométrica a ±20% de 30% a 55%, contra 45% a 69% da massa aprendida (15.8). Os pratos do FPB têm altura e recipiente variados, e a área sozinha não carrega isso.
- **DFood-TJ, se a leitura acontecer:**
  - **a massa sai do viés de −42% para algo entre −15% e +15%**, se a escala pelos ovos estiver certa; a massa a ±20% fica de 25% a 45%, porque a altura do arroz amontoado e dos refogados varia muito;
  - **a densidade continua o problema:** com G3, a densidade aprendida mantém os +57%, e a proteína sairia **cerca de 50% acima**, com o critério de 30% a 45%. Com G1 (tabela por grupo), a densidade depende de a identificação separar arroz, verdura e carne, e espero de 35% a 50%;
  - **o critério de proteína não passa de 70%** em nenhum caso.
- **O que seria uma boa notícia:** a massa do DFood-TJ com viés pequeno e de 40% ou mais a ±20%. Isso diria que, com o diâmetro do prato, a massa vira um problema de altura e de densidade aparente, e não de escala, e o conjunto brasileiro com o diâmetro do prato ficaria mais promissor. **Uma má notícia** seria a massa geométrica com viés forte mesmo com a escala certa: a área × prior não transferiria entre cozinhas.

### 18.9 Custo e implementação

- **Kaggle, 2 × T4:**
  - OWLv2 a 960 px, cerca de 0,3 s por foto, e SAM 2.1 com 5 a 10 caixas, de 0,2 a 0,4 s por foto;
  - cerca de 3.500 fotos: 2.281 do Nutrition5k, 711 do DFood-TJ (o kernel gera as máscaras das 711, mas o script usa só as 457) e a parte segurada do FPB;
  - **uns 40 a 60 minutos de GPU**, mais o download.
  - O kernel novo, `eval/treino/kaggle/geom_r9.py`, é escrito pelo agente do Kaggle. Eu não mexo em `eval/treino/kaggle/`.
  - **Saídas:**
    - uma tabela por item (foto, termo, grupo, fonte, nota, área em pixels, máscara em RLE comprimido), de 20 a 50 MB;
    - as folhas de conferência;
    - nenhuma máscara como imagem;
    - o mapa de profundidade do Nutrition5k já está no scratchpad.
- **CPU, aqui:**
  - escala do Nutrition5k pela profundidade, alguns minutos;
  - k_grupo por mínimos quadrados não negativos e h_grupo por fold, segundos;
  - ovos e escala do DFood-TJ, segundos;
  - bootstrap, alguns minutos.
  - Scripts `eval/geom/` (`escala.py`, `calibra.py`, `avalia_geom.py`, com marca para a leitura 3) e as tabelas `vocabulario.csv` e `priors.csv`, escritas antes do kernel.
  - O disco está apertado (cerca de 1,9 GB livres), então nada de máscara densa em disco.
- **Ordem:**
  1. vocabulário e priors escritos aqui;
  2. kernel;
  3. conferências a olho;
  4. escala;
  5. validação cruzada no Nutrition5k e escolha;
  6. FPB, se a escala do xadrez estiver documentada;
  7. as condições de 18.7;
  8. a leitura 3 ou o registro de que ela não aconteceu;
  9. o resultado em 18.11.

### 18.10 O lado da CPU, pronto antes do kernel (17/11/2026), e o formato que o kernel deve gravar

**a) O que já está em `eval/geom/`.** Nenhum rótulo de teste foi aberto.

| Arquivo | O que faz | Estado |
|---|---|---|
| `vocabulario.csv` | 27 termos do OWLv2 → grupo de massa (8) e fonte de proteína (10), mais `resto` (comida que nenhuma caixa cobriu) | Pronto. Inclui "a boiled egg in shell", que serve à escala do DFood-TJ |
| `priors.csv` | Densidade aparente de 24 alimentos, com a fonte linha a linha; altura por grupo; eixo menor do ovo (4,3 cm); campo de visão do Nutrition5k (54,8°) | Densidades prontas. **As alturas ficam vazias até o kernel** (ver b) |
| `escala.py` | `n5k`: profundidade + campo de visão. `dfoodtj`: diâmetro do prato por período pelos ovos, mais a condição 18.7 (2). `fpb`: xadrez | Pronto. O `n5k` já rodou (ver c) |
| `calibra.py` | Áreas por grupo; k por grupo (mínimos quadrados não negativos, por fold); ρ por grupo; alturas e volumes pela profundidade local; G1, G2, G3, G-prof e a mistura G4 | Pronto |
| `avalia_geom.py` | `--cv` (escolha de 18.6, congela o candidato com sha256); `--condicoes` (18.7); `--fpb` (última leitura, só ordem, com marca); `--leitura3` (marca própria; exige as quatro condições, as marcas das leituras 1 e 2 e o sha do gabarito); `--ensaio` (tabelas sintéticas e rótulos sorteados) | Pronto. **O ensaio passou de ponta a ponta** sem gravar seleção nem marca |

**b) Decisões e limites que surgiram ao construir.**
- **Densidades: não consegui a tabela da FAO/INFOODS.** O site da FAO e o do USDA estão bloqueados nesta sessão.
  - Usei os pesos por xícara do USDA FoodData Central, compilados pelo aqua-calc e publicados em `huggingface.co/datasets/arunapb/food-density-nutrition` (CC BY 4.0), linha a linha com o `class_id`. As colunas de nutrientes dessa compilação têm erros visíveis (por exemplo, 18,8 g de proteína por 100 g de vagem), e por isso **só a densidade é usada**. A proteína vem do treino do Nutrition5k.
  - Várias linhas são aproximações fracas, marcadas no `vocabulario.csv`: pão de vapor, queijo (a linha é manteiga), ensopado e refogado misto.
  - A G1 não depende dessas densidades: o k sai do treino. A G2 depende.
- **Altura por grupo:** sai da profundidade do Nutrition5k (que está no scratchpad) dentro das máscaras de cada grupo. As máscaras só existem depois do kernel.
  - Depois do kernel, `calibra.py --alturas` calcula as alturas, e elas são copiadas para `priors.csv` **antes de qualquer validação cruzada**. Não usa grama nenhuma.
  - O kernel não precisa mexer na profundidade.
- **Escala do FPB: o tamanho do quadrado do xadrez não está documentado** no cartão do Hugging Face nem no README dos autores. O artigo (IEEE Access, 2025) não está no arXiv nem na Europe PMC, e o IEEE está bloqueado. **Pela regra de 18.2, o FPB entra só como relato de ordem:** a correlação de postos entre a massa geométrica relativa (área em pixels × k) e a massa pesada, na última leitura.
- **Ovo:** os 4,3 cm são um valor de referência para o ovo de galinha médio a grande. Não consegui conferi-lo numa tabela online. Um erro de 5% nele vira 10% na área e na massa, para o período inteiro.
- **Ajuste do k:** na primeira versão, um grupo identificado errado (com área, mas sem massa no rótulo) puxava o k para zero. As linhas por grupo agora usam a massa do prato como escala e entram com peso 0,5. A mudança foi feita no ensaio sintético, antes de qualquer número real.

**c) Escala do Nutrition5k (sem rótulo; `escala.py n5k`, 2,5 minutos de CPU).**
- Nos 2.281 pratos do treino, o plano do prato fica a **35,0 cm** da câmera (p10 34,4; p90 35,4), e o pixel vale 0,0567 cm.
- **O diâmetro do prato sai com mediana de 25,6 cm** (p10 20,0; p90 26,8): **passa** na conferência de 18 a 30 cm. Com o campo da câmera de cor em 16:9 (69,4°), daria uns 34 cm, que não é plausível.
- 2 pratos ficaram sem caixa e usam a distância mediana, marcados.

**d) O que o kernel deve gravar** (para o agente do Kaggle; nada aqui é pontuado). São duas tabelas CSV, mais as folhas de conferência. Bases: o treino do Nutrition5k (2.281 fotos de cima), as 711 do DFood-TJ (as mesmas da leitura 2, mas sem a máscara de visor) e a parte de registro do FPB (312 fotos).

`fotos_geom.csv`: uma linha por foto.

| Coluna | Conteúdo |
|---|---|
| `base` | `n5k`, `dfoodtj` ou `fpb` |
| `id` | `dish_id` do Nutrition5k, `dftj_…` ou o `dish_id` da lista do FPB |
| `arquivo`, `sha256` | Arquivo de origem e o hash dos bytes |
| `largura`, `altura`, `girada` | Tamanho da foto depois da orientação do EXIF (e do giro de retrato), como no kernel da oitava |
| `periodo`, `paciente` | Só do DFood-TJ, copiados de `teste_r8_dfoodtj.csv` (sem rótulo) |
| `prato_metodo` | `owlv2`, `elipse` ou `sem_prato`, pela regra só de prato e tigela de 14.3 |
| `prato_caixa` | `x0 y0 x1 y1`, em pixels da foto orientada |
| `prato_elipse` | `cx cy eixo_maior eixo_menor angulo_graus`, com eixos **inteiros** (não semieixos), em pixels da foto orientada, ajustada por `cv2.fitEllipse` no contorno da máscara do SAM do prato |
| `prato_rle`, `mascara_largura`, `mascara_altura` | Máscara do prato em RLE, na resolução de trabalho |
| `xadrez_detectado`, `xadrez_passo_px` | Só do FPB: 1 se `cv2.findChessboardCorners` achar o padrão; passo mediano entre cantos vizinhos, em pixels da foto orientada |
| `n_itens`, `versoes` | Quantos itens; versões do OWLv2, do SAM e das bibliotecas |

`itens_geom.csv.gz`: uma linha por item.

| Coluna | Conteúdo |
|---|---|
| `base`, `id` | Como em `fotos_geom.csv` |
| `item` | Inteiro, a partir de 0, na foto |
| `termo`, `consulta` | `termo` do `vocabulario.csv` e o texto exato da consulta. O item `resto` não tem consulta |
| `nota` | A nota do OWLv2 (0 no `resto`) |
| `caixa` | `x0 y0 x1 y1` da caixa do OWLv2, em pixels da foto orientada |
| `rle` | **RLE do COCO sem compressão:** contagens separadas por espaço, em ordem de coluna (Fortran), começando pela contagem de zeros, na resolução de trabalho `mascara_largura × mascara_altura` (lado maior de até 1.024 px; o Nutrition5k fica em 640 × 480, alinhado à profundidade) |
| `mascara_largura`, `mascara_altura` | Tamanho da máscara |
| `area_px` | Pixels da máscara **em unidades da foto orientada** (contagem × (largura / mascara_largura)²) |
| `elipse` | `cx cy eixo_maior eixo_menor angulo_graus` da máscara, eixos inteiros, em pixels da foto orientada (`cv2.fitEllipse`) |
| `origem_mascara` | `sam2` ou `caixa_elipse` (reserva) |
| `dentro_prato` | Fração da máscara dentro da máscara do prato, antes do recorte |

**Regras do kernel para as máscaras:**
- consultas, com nota de 0,15 ou mais e NMS de 0,5 por termo;
- SAM 2.1 (`facebook/sam2.1-hiera-large`), com a caixa como prompt;
- recorte pela máscara do prato;
- um pixel em duas máscaras fica com a de maior nota;
- **`resto`** = máscara do prato − itens − branco do prato. O branco do prato é S < 0,15 e V > 0,75 em HSV, limiar fixo; os componentes de `resto` com menos de 0,5% da área do prato são descartados;
  - **Emenda (17/11/2026, antes de o kernel rodar e sem nenhum rótulo):** no ensaio em CPU com 9 fotos (3 por base), o prato vazio tem S ≈ 0,05 mas V de só 0,64–0,74 sob a luz das cenas, e com V > 0,75 o `resto` cobria 75–87% do prato no Nutrition5k e 37–46% no DFood-TJ, contando o prato vazio como comida. **O limiar passa a ser S < 0,15 e V > 0,55.** O arroz branco não é afetado, porque o `resto` só existe fora das máscaras dos itens. O kernel grava a mediana de S e V do prato livre e a fração de `resto` por foto (`livre_s_mediano`, `livre_v_mediano`, `resto_fracao_prato`), para que a escolha possa ser conferida a olho nas folhas antes de qualquer leitura.
- **nada de máscara densa em disco.**
- **Ovos:** a consulta "a boiled egg in shell" também roda nas fotos do DFood-TJ.

**Folhas de conferência:** 40 fotos por base, a foto com as máscaras coloridas por grupo e o nome do termo, para a régua de 18.3.

**Tamanho esperado:** de 20 a 50 MB, com 4 a 8 itens por foto.

**e) Ordem a partir daqui (como em 18.9).**
1. Kernel.
2. Conferência das máscaras: o resultado vai para `conferencia_mascaras.json`, com `{"dfoodtj": {"certas": n, "proteina_errada": n}}` e o mesmo para o Nutrition5k e o FPB.
3. `calibra.py --alturas` e as alturas copiadas para `priors.csv`, com a data.
4. `avalia_geom.py --cv`, que congela o candidato.
5. `--fpb`.
6. `--condicoes`.
7. `--leitura3`, se as condições passarem.
8. Resultado em 18.11.

### 18.11 Nona execução: a segmentação reprovou nas três bases, e a sequência parou no passo 2 (17/11/2026)

**a) O kernel** (`kaggle_out9/`, 59 minutos, sem falhas).

| Base | Fotos | Itens | Prato pelo OWLv2 | `resto` / prato (mediana; p90) |
|---|---|---|---|---|
| Nutrition5k (treino) | 2.281 | 18.318 (mediana de 4 por foto; p90 de 21) | 2.276 (2 elipses, 3 sem prato) | 0,119; 0,787 |
| DFood-TJ | 711 | 5.143 | 711 | 0,005; 0,111 |
| FPB (registro) | 312 | 1.796 | — | 0,225; 0,941 |

- O limiar de branco foi o emendado (V > 0,55).
- O xadrez do FPB não foi achado em nenhuma foto, embora apareça na maioria. O padrão de cantos do kernel deve estar errado. Isso não muda nada: sem o tamanho do quadrado, o FPB seria só de ordem de qualquer jeito (18.10 b).

**b) Passo 2: a conferência das máscaras, antes de qualquer número.**
- `conferencia_mascaras.json`, sha256 `71e2f8c1ed0ce84453a445ba98dffc3eee2d1949d694e1ed78dd55ddcc455308`.
- Régua de 18.3: 36 ou mais de 40 com a comida coberta sem falha grande, e no máximo 2 com a proteína principal no grupo errado.
- Conto como falha grande também **um prato vazio grande contado como comida**. Uma máscara que cobre o prato inteiro mede o prato, não a comida.

| Base | Certas (de 40) | Proteína no grupo errado | Falhas principais | Resultado |
|---|---|---|---|---|
| **DFood-TJ** | **13** | 2 | Prato inteiro como arroz ou massa: 12. Comida sem máscara: 15 | **Reprovado** |
| **Nutrition5k** | **10** | 0 | Comida sem máscara: 16. Prato vazio contado como `resto`: 14 | **Reprovado** |
| **FPB** | **23** | 0 | Bandeja com vários recipientes e só um segmentado: 10. Bandeja inteira como prato: 1. Comida parcial: 6 | **Reprovado** |

- **No DFood-TJ:**
  - a máscara do SAM, com a caixa do arroz (ou da massa) como prompt, muitas vezes devolve o prato inteiro;
  - noutras fotos o OWLv2 não acha quase nada, e a comida fica sem máscara;
  - os dois erros de proteína são pepino-do-mar marcado como pão (`dftj_000010`) e pastéis de carne como "noodles" (`dftj_000017`).
- **A pergunta do coordenador sobre o `resto` do Nutrition5k** (p90 de 0,79): **é prato vazio contado como comida**, não comida sem caixa. Sob a luz da bancada, o prato do Nutrition5k fica cinza e escapa do limiar de branco; nas folhas, o `resto` azul cobre o prato inteiro em 14 das 40 fotos.
- **No FPB:** as bandejas têm vários recipientes, e o kernel segmenta só o prato que o detector escolheu.

**c) O que a regra manda.** Pela 18.3, **"uma base reprovada não entra com a geometria".** As três reprovaram, e por isso:
- **não rodei `calibra.py --alturas` nem `avalia_geom.py --cv`.** As alturas e o k sairiam de máscaras reprovadas, e a escolha seria feita com elas;
- **não rodei `--fpb`.** O FPB continua com 1 leitura;
- **rodei `--condicoes`, que não usa rótulo nenhum**, para registrar a escala pelos ovos:

| Condição de 18.7 | Resultado |
|---|---|
| (1) Máscaras do DFood-TJ | **Falha** (13 de 40) |
| (2) Escala pelos ovos | **Passa.** 2020–2023: prato de **23,6 cm**, de 50 ovos, dispersão de 5,4%. 2024–2025: 22,8 cm, de 41 ovos, 2,0% |
| (3) Massa geométrica no Nutrition5k | Não avaliada (sem seleção) |
| (4) Candidato congelado | Não há |

- **A leitura 3 do DFood-TJ não acontece.** O DFood-TJ continua com 1 leitura. Nenhum rótulo foi aberto nesta execução.

**d) O que fica.**
1. **O gargalo não foi a escala, foi a segmentação.**
   - A escala pelos ovos funcionou, sem rótulo: 23,6 cm para o prato redondo, com dispersão baixa. É um tamanho comum de prato.
   - **O OWLv2 com o SAM, guiado por caixas, não segmenta pratos de cozinha nem de hospital** com a qualidade que a régua pede: ora devolve o prato inteiro, ora não acha o alimento.
2. A pergunta de 18.1 (a geometria conserta a massa de −42%?) **continua sem resposta.** Não há máscara confiável para medir área.
3. **Proposta para o coordenador (não aplicada).** Uma décima execução, só de segmentação, com a mesma régua antes de qualquer número:
   - um segmentador treinado em comida (por exemplo, no FoodSeg103, Apache 2.0, com a licença conferida antes) no lugar do OWLv2 com caixas;
   - ou, no SAM, prompts de ponto e uma regra de sanidade: máscara de mais de 60% do prato só quando houver um item só;
   - o branco do prato amostrado na borda de cada foto, em vez de um limiar fixo;
   - no FPB, todos os recipientes da bandeja.
   - A leitura 3 do DFood-TJ continua guardada para ela, com as mesmas condições de 18.7.
4. **Para o produto:** num app, a usuária pode corrigir a segmentação (tocar no alimento, ajustar o contorno), e isso não existe nestes testes. Mesmo assim, uma segmentação que falha em dois terços a três quartos dos pratos (DFood-TJ e Nutrition5k) tornaria o pipeline geométrico inviável sem essa ajuda.

### 18.12 Emenda: décima execução, só de segmentação (17/11/2026)

**Escrita antes de a execução rodar.** O coordenador aceitou a proposta de 18.11 (d) 3. Muda só a segmentação; tudo depois da conferência a olho fica como em 18.4 a 18.7.

**a) O segmentador.**
- **Aceito se cumprir tudo isto**, conferido pelo agente do Kaggle e escrito aqui antes de qualquer saída nas nossas fotos:
  1. **Pesos públicos**, baixáveis no Kaggle (Hugging Face ou release do GitHub), com o hash dos pesos gravado.
  2. **Treinado em comida:** FoodSeg103 (103 classes mais fundo) ou UECFoodPix Complete (102 mais fundo), declarado pelos autores.
  3. **Licença** gravada (código e pesos). Licença só de pesquisa serve para esta avaliação e fica marcada como impedimento para o produto.
  4. **Nada dos nossos testes no treino.** O treino declarado não pode incluir o DFood-TJ (nem os polígonos dele), o FPB ou o Nutrition5k.
  5. **Qualidade publicada:** mIoU de 40 ou mais no FoodSeg103, ou de 55 ou mais no UECFoodPix, no conjunto de teste da própria base.
  6. **Custo:** até 2 s por foto numa T4, ou seja, até 2 h para as cerca de 3.500 fotos.
  7. **Teste de fumaça:** só nas imagens de exemplo do próprio modelo, nunca nas nossas.
- **Escolha entre candidatos aceitos, pela ordem:**
  1. FoodSeg103 antes de UECFoodPix: as classes do FoodSeg103 são ingredientes (arroz, carne de porco, tofu, repolho) e casam com os grupos; as do UECFoodPix são pratos japoneses inteiros;
  2. o maior mIoU publicado;
  3. a licença mais aberta.
  - **Nenhuma saída nas nossas fotos entra nessa escolha.**
- **Mapa das classes para os 8 grupos e as 10 fontes:**
  - **FoodSeg103: já está escrito** em `eval/geom/vocabulario.csv` (termos `fs103:<classe>`, 103 linhas): 55 vegetal, 13 amido, 11 proteína animal, 7 outros, 4 misto, 3 leguminosa, 3 laticínio e 1 gordura e molho. A densidade de cada classe está em `priors.csv`, com a mesma fonte.
  - **As 6 bebidas (café, suco, chá, vinho, leite, milk-shake) ficam fora da massa de comida.** As listas também não contam bebida na massa (`bebida_g`).
  - Se o escolhido for o UECFoodPix, eu escrevo o mapa das 102 classes de prato **antes de abrir qualquer saída**, a partir da lista de classes que o agente informar.
- **Saída:** o mapa de classes por pixel, só dentro do prato. Cada componente conexo de uma classe vira um item da tabela de 18.10 (d), com `termo` = `fs103:<classe>`. Componentes com menos de 0,3% da área do prato são descartados.
- **O SAM 2.1 fica desligado nesta execução.** A pergunta é se um segmentador de comida passa na régua sozinho; refinar com o SAM seria uma segunda variante, e isso pediria uma segunda conferência.
- **Regra de sanidade:** um componente com mais de 60% da área do prato só vale se for **a única classe de comida** no prato. Se não for, o componente é descartado, e a foto é marcada com `sanidade_falhou`. A conferência a olho julga as máscaras já depois da regra.

**b) O branco do prato, foto a foto.**
- **Anel:** os pixels entre 0,88 e 0,98 dos semieixos da elipse do prato, menos os pixels marcados como comida.
- **Estatística:** a mediana e o MAD de cada canal, em Lab.
  - Um pixel dentro do prato é **prato**, e não comida, se a distância dele à mediana do anel for no máximo 3 × 1,4826 × MAD (o MAD tomado como a norma dos três canais), com tolerância mínima de ΔE = 8.
- **Com menos de 500 pixels no anel,** vale o limiar fixo de 18.10 (S < 0,15 e V > 0,55), e a foto é marcada.
- **`resto`** = prato − comida segmentada − pixels de prato. Componentes de `resto` com menos de 0,5% do prato são descartados.

**c) FPB: todos os recipientes e o xadrez.**
- **Recipientes:** o OWLv2 procura "a plate", "a bowl", "a food container" e "a cup" dentro da bandeja (nota de 0,15 ou mais, NMS de 0,5).
  - Cada recipiente é segmentado como um prato, com o anel do próprio recipiente.
  - A refeição é a soma dos recipientes, sem as bebidas.
  - A bandeja nunca é tomada como prato. Uma caixa com mais de 60% da foto não é recipiente.
- **Xadrez:**
  - `cv2.findChessboardCornersSB` na foto em tons de cinza, ampliada 2 vezes dentro da caixa do OWLv2 para "a checkerboard";
  - padrões de cantos internos procurados, em ordem: 3×3, 4×4, 5×5, 6×6, 7×7, 4×5, 5×4, 5×6, 6×5, 6×7, 7×6, 7×8 e 8×7;
  - **o kernel grava qual padrão achou** em cada foto, mais a contagem por padrão.
  - O passo em pixels serve só de registro. **Sem o tamanho do quadrado documentado, o FPB continua só de ordem** (18.2).
  - **Emenda do xadrez (18/11/2026, antes do kernel):** o tabuleiro do FPB tem 4 × 5 quadrados, isto é, **3 × 4 cantos internos**, padrão ausente da lista acima. Passa a valer: primeiro `cv2.findChessboardCorners` comum (limiar adaptativo + normalização) na foto inteira em tons de cinza, com 3×4 e 4×3 no início da lista e depois os demais padrões; o `findChessboardCornersSB` só como reserva, aceito apenas se a dispersão entre cantos vizinhos for ≤ 5%. **Registro:** a contagem de cantos foi feita pelo agente do kernel em 10 fotos do registro do FPB, olhadas antes desta emenda e antes de qualquer saída de segmentação; o FPB continua só de ordem, então isso não afeta nenhuma decisão.

**Segmentador escolhido (18/11/2026), pela regra de (a), sem nenhuma saída nas nossas fotos:** SETR-MLA ViT-B/16, o modelo oficial do FoodSeg103-Benchmark-v1 (mIoU publicado 45,1 no teste do FoodSeg103; único candidato que passa os sete critérios). Pesos `iter_80000.pth` do espelho `PhilippMuellerR/FoodSAMWeights` no Hugging Face, revisão `454015fde30211cdd6a897183a9e51b1748218b0`, sha256 `f72bf120eb00dca469f292cc2bde956883f3b64d22768f203c17aea8eaf5001a`; metadados do arquivo conferem com o original (mmseg 0.11, 80.000 iterações, 104 classes). Código e anotações Apache-2.0; imagens do Recipe1M, de uso de pesquisa: **bloqueia o produto, serve para esta avaliação**. Carregado por uma reimplementação em PyTorch puro (timm ViT-B/16 a 768 + cabeça MLA), conferida em 50 imagens de validação do FoodSeg103 (aAcc 81,9 contra 83,53 publicado no conjunto inteiro). Os outros candidatos falharam: rótulos fora da ordem oficial e sem licença (SegFormer-B2 e Mask2Former de `mericberktas`), mIoU abaixo de 40 (DeepLabV3+ R50), treino com o Nutrition5k (`yolo26l-seg-foodseg103-nutrition5k`), pesos inacessíveis (UECFoodPix DeepLabV3+), ou lentidão e SAM (FoodSAM completo).

**d) A conferência a olho: a mesma régua, em amostras novas.**
- A régua de 18.3, com a leitura de 18.11 (b): **36 ou mais de 40** com a comida coberta sem falha grande, e **no máximo 2** com a proteína principal no grupo errado. **Prato vazio contado como comida é falha grande.**
- **Amostras novas de 40 fotos por base, com a semente `20261118`** (`numpy.random.default_rng(20261118).choice(n, 40, replace=False)` sobre a lista ordenada por id):
  - Nutrition5k: entre as 2.281 do treino;
  - DFood-TJ: **entre as 457 refeições da população da leitura 2** (`conferencia_r9.json`), as que a leitura 3 usaria;
  - FPB: entre as 312 fotos de registro.
- **Repetir as amostras da nona execução enviesaria a conferência.** Eu já vi aquelas 120 fotos, e as mudanças (a) a (c) foram desenhadas olhando as falhas delas: o arroz que vira o prato, o prato cinza, a bandeja com vários recipientes. Conferir nelas mediria o conserto nos casos que o motivaram, o que é otimista.
  - As 40 da nona saem também, **como relato, sem decidir nada.**
- O resultado vai para `conferencia_mascaras_r10.json`, com o sha256 escrito aqui, antes de qualquer número.

**e) Tudo depois da conferência fica como em 18.4 a 18.7.**
- Escala: Nutrition5k pela profundidade; DFood-TJ pelos ovos, recalculada com as saídas novas. Se o segmentador não tiver a classe de ovo separada, valem os ovos da nona, que passaram a olho.
- Alturas pela profundidade, depois a validação cruzada e a escolha (G1 a G4 e G-prof), depois as condições de 18.7.
- **A leitura 3 do DFood-TJ e a última leitura do FPB continuam guardadas, com as mesmas condições.**
- **O Nutrition5k e o DFood-TJ precisam passar os dois.**
  - Sem o Nutrition5k, não há de onde calibrar k nem alturas, e nenhum candidato existe.
  - Sem o DFood-TJ, não há leitura 3.
  - Se só o FPB reprovar, cai só o relato do FPB.

**f) O que eu espero e a regra de parada.**
- **Expectativa honesta:**
  - O FoodSeg103 é de fotos de receitas da internet (Recipe1M), tiradas de ângulos variados. O mIoU publicado fica em torno de 40 a 50 dentro da própria base.
  - Pratos de hospital chineses de cima, e as bancadas do Nutrition5k, ficam fora desse domínio.
  - O branco por foto deve resolver o prato vazio contado como comida no Nutrition5k. A cobertura da comida depende do segmentador.
  - **Chance de passar:** de 25% a 35% no DFood-TJ, de 35% a 45% no Nutrition5k e de 40% a 50% no FPB. **Chance de o DFood-TJ e o Nutrition5k passarem juntos: de 15% a 25%.**
- **Regra de parada:** **se a décima execução reprovar na conferência do DFood-TJ, ou na do Nutrition5k, a rota geométrica fica parada até a coleta brasileira.** Lá a fundadora pode fotografar pratos mais simples, com o diâmetro do prato anotado, e a segmentação pode ser conferida e corrigida na fonte.
  - Não haverá uma décima primeira tentativa de segmentação em dados públicos.
  - A leitura 3 do DFood-TJ fica sem uso, registrada assim.
  - O que se aprendeu fica: a escala pelo prato dá, sem rótulo, um diâmetro plausível e estável (os ovos, em 18.11; não conferido contra o real), e o gargalo é a segmentação.


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

Seção 16.6: `python3 eval/treino/dfoodtj_teste_divide.py <pasta>` grava a população do teste de registro DFood-TJ (tabela sem rótulo para o kernel e gabarito para a CPU, com os sha256).

Seção 16.7: `python3 -I eval/treino/r8_avalia.py <pasta_kaggle_8> <gabarito_r8_dfoodtj.csv>` faz a leitura única do DFood-TJ (marca em `resultados/dens/r8/`; não roda duas vezes). `--ensaio` no lugar do gabarito roda os mesmos cálculos com rótulos sorteados. Seção 16.8: `--leitura2 <pasta_kaggle_9> <gabarito> <conferencia_r9.json> <pasta_kaggle_8>` (marca própria), `--ensaio2` (sem gabarito) e `--pos-hoc` (só depois da leitura 2).

Seção 18: `python3 -I eval/geom/escala.py n5k <scratchpad>/treino <DENS>/geom/escala_n5k.csv` (escala do Nutrition5k, sem rótulo); `python3 -I eval/geom/avalia_geom.py --ensaio <scratchpad>/treino <kaggle_out7> <kaggle_out8b> <conferencia_r9.json>` (ensaio com tabelas sintéticas); os modos `--cv`, `--condicoes`, `--fpb` e `--leitura3` esperam as tabelas do kernel (18.10 d).

Seção 17.8: `cd eval/treino; python3 r7_avalia.py <pasta_kaggle_7> <pasta_divisoes> --so-cv` (R17, sementes e 17.5), depois `--relatos` (partes seguradas de ACETADA, receitas e FoodBD) e `--omnifood` (terceira e última leitura do OmniFood8K, só do S7); resultados em `$TD/resultados/dens/r7/`.

Seção 15.8: `cd eval/treino; python3 r6_avalia.py <pasta_kaggle_5> <pasta_kaggle_6> <pasta_divisoes> --so-cv` (R15 no recorte), depois `--registro` (segunda leitura das partes externas: REF e candidato final); resultados em `$TD/resultados/dens/r6/`.

Seção 14.10: `cd eval/treino; python3 r5_avalia.py <pasta_kaggle_4> <pasta_kaggle_5> <pasta_divisoes> --so-cv` (R14 só com validação cruzada; a segunda leitura, `--registro`, só roda se a regra a abrir); resultados em `$TD/resultados/dens/r5/`. A etapa 0 (detector) e a conferência a olho estão em 14.9; a decisão vai em `eval/treino/kaggle/listas/conferencia_r5.json`.

Seção 13.11: `cd eval/treino; python3 r4_avalia.py <pasta_kaggle_4> <pasta_divisoes> --so-cv` (R13 só com validação cruzada), depois `--registro` (leitura única das partes de registro externas) e `--teste2` (última leitura do teste2, só do S); resultados em `$TD/resultados/dens/r4/`.

Seção 13 (quarta execução): `python3 eval/treino/externo_r4_divide.py <pasta_de_saida>` grava as divisões das bases externas (`divisao_r4_<dominio>.csv`: papel, folds, parte de registro), a partir das listas `externo_*.csv`; as do OmniFood8K, do FPB e do SNAPMe derivam de bases não comerciais e ficam fora do repositório.

A terceira execução em GPU (12.3 e 12.5) usa `eval/treino/kaggle/listas/treino_v2.csv`, `aval2_ids.csv` e `externo_flavoria.csv`; a pasta de saída (previsões, registros, `servico.json`, `decisao_flavoria.json`, `relato_externo.json`) fica fora do repositório. Os resultados de 12.6 ficam em `$TD/resultados/dens/um_modelo/` (`selecao.json`, `confere.json`, `lido.json`, `avalia_teste2.json`, `relato_teste2.json`), e os de 12.7, na mesma pasta, com o prefixo `emenda_`. O diagnóstico da destilação, o efeito dos dados e as combinações exploratórias de 12.6 (d) e (e) foram calculados à parte, só com as previsões fora-do-fold dessa pasta e as da segunda execução.
