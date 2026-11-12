# Onde conseguir mais dados de treino para "uma foto de cima → massa e proteína" (11/11/2026)

**Pergunta da fundadora:** onde buscar mais fotos para melhorar o treino, já que 70% é o piso e o modelo está em ~70% (69,7% só com imagem no teste novo, IC 65–74), com 52% nos pratos acima de 25 g de proteína.

**Resposta curta.** Não existe um segundo Nutrition5k com licença comercial. O que existe, por ordem de valor:

1. **Um dataset novo, comercial, de cima e pesado por componente que ainda não usamos: FlavoriaFoodWeight1700** (Finlândia, CC BY 4.0, **1.264 pratos e 1.967 fotos** depois da nossa limpeza, de uma linha de almoço com balança automática). Só tem **peso**, sem proteína, e são pratos grandes (mediana de 459 g, contra 170 g no nosso treino). **Atualização do mesmo dia: licença e arquivos conferidos, lista de treino pronta** (seção 2.1).
2. **Pratos do próprio Nutrition5k que hoje estão presos nos conjuntos de avaliação:** 1.669 pratos de cima fora do treino, uns 500 deles com mais de 25 g (o treino tem 461). Usá-los custa o teste.
3. **Três datasets com peso medido, todos com licença não comercial, dois deles bem maiores que o nosso treino:** OmniFood8K (8.036 pratos chineses), Food Portion Benchmark (14.083 imagens) e MetaFood3D (637 objetos). Só entram no produto se os autores licenciarem. **Isso depende de e-mail da fundadora.**
4. **Fotos sem rótulo** para destilar as nove redes num modelo só: há fontes com licença comercial (MyFoodRepo, Open Images), mas pequenas ou ruidosas depois de filtrar por "prato visto de cima". O SNAPMe saiu desta lista na segunda varredura: a licença é CC BY-SA 4.0 e os autores pedem para não treinar com ele (seção 10).
5. **Dado brasileiro continua tendo de ser coletado.** Marmita de e-commerce rende dezenas de itens, com foto de estúdio em ângulo: serve de teste, não de treino.

**Segunda varredura (12/11/2026), seção 10:** com Zenodo, Figshare, Dataverse, arXiv e Mendeley liberados, não apareceu nenhum conjunto novo de pratos compostos pesados com licença comercial. Saíram duas listas a mais (LeFood-Set, com peso medido de itens isolados; SNAPMe, com rótulo fraco e uso pendente de decisão) e um pedido de acesso que vale fazer (ClinicPlate).

Este documento não repete `datasets_internacionais.md` (09/11) nem `fontes.md` (08/11). O que mudou desde então: acesso à API do Kaggle e a possibilidade de anexar datasets ou baixar da internet dentro do notebook de treino.

**Nenhum experimento de treino foi rodado aqui.** Toda coluna "ganho esperado" é julgamento, com a evidência que existe ao lado. A especificação de leitura das fontes prontas está em `eval/dados_ext/fontes_treino.json`; o leitor do NutritionVerse-Real, em `eval/dados_ext/manifestos_kaggle.py`.

## 0. Como foi feito e o que limita a confiança

- **Kaggle:** cerca de 100 buscas na API (`datasets list -s`), em inglês e português: nutrition5k, nutritionverse, metafood3d, ecustfd, food weight/portion/mass/volume/depth/RGB-D, calorie estimation, plate waste, leftover, canteen/tray, school lunch, hospital meal, FoodSeg103, UECFood, Food-101, Food2K, Open Images, Recipe1M, MyFoodRepo, CNFOOD, brazilian food, comida brasileira, flavoria e outras. Para cada candidato li o metadado (licença declarada), a listagem de arquivos e, nos principais, uma amostra dos rótulos.
- **A licença que aparece no Kaggle é a que o uploader escolheu.** Em cópias de terceiros ela não vale nada: o MetaFood3D aparece como MIT e Apache-2.0 e o original é CC BY-NC. Só confio no rótulo do Kaggle quando a conta é dos autores (NutritionVerse-Real) ou quando coincide com o original (Nutrition5k).
- **Hosts que respondem daqui:** `www.kaggle.com`, `huggingface.co` (cartões e arquivos pequenos; os arquivos grandes ficam numa CDN bloqueada), `raw.githubusercontent.com`, `storage.googleapis.com`, `open-images-dataset.s3.amazonaws.com`, `www.livup.com.br`, `www.swift.com.br`, `www.seara.com.br`, `www.sadia.com.br`, `www.paodeacucar.com`, `mercado.carrefour.com.br`, `www.tbca.net.br`, `edisciplinas.usp.br`, `nutritotal.com.br`, `agdatacommons.nal.usda.gov` e, desde a tarde de 11/11, `zenodo.org`.
- **Bloqueados (403 do proxy):** `arxiv.org`, `doi.org`, `www.mdpi.com`, `data.mendeley.com`, `lorenz.ecn.purdue.edu`, `drive.google.com`, `*.github.io`, `figshare.com`, `aihub.or.kr`, `universe.roboflow.com`, `mrfit.com.br`, `99app.com`, `ifood.com.br`, todos os sites de receita testados e todos os repositórios de universidades brasileiras e portuguesas testados. Para esses, o que está escrito vem de **resumo de busca na web**, e está marcado assim.
- Níveis de verificação usados nas tabelas: **A** = conferido aqui no arquivo ou no texto da própria fonte; **B** = lido em resumo de busca ou em material de terceiros; **C** = de memória, sem conferência nesta sessão.

## 1. Tabela ranqueada

Ordem pelo valor esperado para o modelo do produto. "Comercial" = a licença permite treinar o modelo que vai para o app.

| # | Fonte | O que tem | Licença e uso permitido | Onde está | Como entra no treino | Ganho esperado e por quê | Verificação |
|---|---|---|---|---|---|---|---|
| 1 | **FlavoriaFoodWeight1700** (U. Turku, 2022) | 2.013 fotos 1024×1024 de pratos **de cima**, já recortadas e mascaradas no prato, 1 a 4 quadros por bandeja; cada porção pesada pela balança da linha, **por componente** (246 nomes). 3.343 sessões de pesagem, 1.292 com foto; depois da limpeza, **1.264 pratos / 1.967 fotos**. Massa mediana de 459 g, 202 pratos acima de 600 g. **Sem proteína nem kcal** | **CC BY 4.0: comercial, com atribuição** | Zenodo 5850856 (2,58 GB). **Não está no Kaggle**; o notebook baixa do Zenodo | Lista pronta em `eval/treino/kaggle/listas/externo_flavoria.csv`, no esquema de 12.4/12.5: massa, massa por grupo e frações por fonte; proteína fica vazia. 230 pratos dos últimos dias são o teste externo de massa | **Massa em pratos grandes e fora do Nutrition5k: provável. No teste do Nutrition5k: pequeno (0 a 2 pontos, chute).** Evidência de terceiros: um modelo treinado só com Nutrition5k errava a massa desses pratos em 45% (207 g a menos, em média); com 555 pratos do Flavoria no treino, 25%. No nosso teste2 o erro de massa não é o que mais limita | **A**: licença no registro do Zenodo, listagem completa do tar, os 11 CSVs e cerca de 110 fotos olhadas |
| 2 | **Nutrition5k: pratos de cima fora do treino** | 1.669 pratos que passam nos filtros e não estão nos 1.684 do treino: 435 de avaliação (`calib`, `teste`, `n5k_v1`), 829 do teste2, 405 em cadeias do `calib`. **Uns 500 com mais de 25 g** (525 contando só o filtro de massa ≥ 20 g; o treino tem 461) | CC BY 4.0: comercial | Já baixados (bucket do Google) | Treinar o modelo de **produção** com tudo, depois de fechar a avaliação | **É a maior reserva de pratos ricos em proteína, e dobra a faixa fraca. Ganho incerto:** a curva de aprendizado com backbone congelado estabiliza em ~400 pratos, mas ela não foi medida por faixa nem com ajuste fino. **Custo: acaba o teste do Nutrition5k**; a medida passaria a ser validação cruzada por cadeia e conjuntos externos | **A** (contagens feitas aqui) |
| 3 | **OmniFood8K** (CVPR 2026, U. Yunnan / CAS) | **8.036 cenas reais** de pratos chineses, várias vistas por cena, ingredientes pesados, nutrientes | **CC BY-NC 4.0: não comercial.** Só entra com licença dos autores | Google Drive (link no repositório dos autores); bloqueado daqui. Não está no Kaggle nem no HF | Se licenciado: treino direto de massa e proteína, 5 vezes o nosso treino | **O maior ganho possível entre os rotulados, se houver licença.** Cozinha diferente da brasileira, mas com arroz, carnes e pratos mistos em porções de refeição | **A** para a licença (README dos autores, lido no GitHub); **B** para o conteúdo (não vi nenhuma foto nem rótulo) |
| 4 | **Fotos sem rótulo para destilação** (MyFoodRepo, Open Images, SNAPMe, quadros do Nutrition5k) | Ver seção 4. Depois de filtrar por "prato de cima": algo como 20 a 35 mil fotos com licença comercial (estimativa) | CC BY 4.0, CC BY 2.0, domínio público | Kaggle (MyFoodRepo), S3 e GCS (Open Images), USDA (SNAPMe) | Pseudo-rótulo das nove redes → um modelo só | **Aproxima o modelo único (63–65%) do conjunto (70%); não sobe o teto.** Sem evidência nossa: é o efeito usual de destilação, não medido aqui | **A** para Open Images (contado e amostrado); **B** para o resto |
| 5 | **Food Portion Benchmark** (ISSAI, 2025) | 14.083 imagens, 138 classes da Ásia Central, peso por item em balança; vistas de cima e de 4 lados; sem proteína | **CC BY-NC 4.0: não comercial** | HF `issai/Food_Portion_Benchmark`. Imagens bloqueadas daqui; do Kaggle devem baixar. Não está no Kaggle | Se licenciado: massa. Sem licença: teste externo de massa | Médio para massa, se licenciado | **A** para a licença (cartão do HF); rótulos conferidos em 09/11 |
| 6 | **FLIC** (UNIMIB, 2026) | Refeitório italiano, câmera a 65 cm **de cima**; 401 pares bandeja cheia/sobra (700 fotos cheias), cada prato pesado, máscaras | Artigo em CC BY; **licença do dataset e link de download não localizados** | Não localizado (artigo na MDPI, bloqueada) | Massa por prato da bandeja | Pequeno a médio: +400 a 700 pratos pesados de cima, outro refeitório | **B** |
| 7 | **ACETADA** (Curtin / Purdue, 2025) | 806 fotos "antes da refeição" (e as "depois") de 152 adultos em estudo de alimentação controlada; comida pesada a 0,1 g e conferida por nutricionistas; foto de celular | **CC BY-NC 4.0: não comercial** (corrigido em 12/11: a página do dataset e um registro de terceiros dizem NC; na primeira passada estava como CC BY 4.0) | `skynet.ecn.purdue.edu/~coburn6/ACETADA/` (bloqueado daqui) | Só com licença dos autores | Médio se licenciado: é o domínio certo (celular, refeição real, peso medido) | **B** |
| 8 | **MetaFood3D** (Purdue, 2024) | 637 objetos (743 na v2), 108 classes; cada um com peso, kcal, **proteína** e vídeo RGB-D real de celular; mediana de 93 g e 4 g de proteína (alimentos isolados) | **CC BY-NC 4.0, com formulário de pedido** (página oficial) | Purdue (bloqueado). No Kaggle há 4 cópias de terceiros com licenças inventadas (MIT, Apache-2.0) | Se licenciado: massa de alimento isolado | Baixo: alimento único, pouca proteína | **A** para a planilha de rótulos (637 linhas, lida aqui); **B** para a licença |
| 9 | **Nutrition5k: pratos só com vídeo lateral** | 1.504 pratos sem foto de cima (1.432 passam nos filtros); só **119 com mais de 25 g** | CC BY 4.0 | Bucket; já temos 8 quadros por prato | Sinal auxiliar de treino | **Baixo.** Já testado na seção 8 (item 10): ganho pequeno na CV, nenhum no teste. Os quadros são oblíquos e cortam o prato | **A** |
| 10 | **LeFood-Set** (Indonésia, 2025) | 524 porções de comida de hospital, um item por prato, foto de cima antes e depois de comer, peso em balança nas duas | **CC BY 4.0: comercial** (conferido no registro do Mendeley em 12/11) | Mendeley Data `cchsk79jkt`, 52 MB | **Lista pronta** (`externo_lefood.csv`, 811 fotos; seção 10) | Baixo para a faixa fraca (nenhum prato passa de 16 g de proteína); útil para identidade e peso de frango, peixe, ovo, tofu e tempeh | **A** |
| 11 | **ECUSTFD** | 2.968 imagens, 19 alimentos isolados, peso e volume | Sem licença declarada | Já baixado | Massa | Baixo (já avaliado em `datasets_internacionais.md`) | **A** |
| 12 | **NutritionVerse-Real** | 889 fotos de celular, 225 pratos com foto; **peso nominal por tipo de item, não medido por prato** (ver 2.2) | **CC BY-NC-SA 4.0: não comercial** | Kaggle `nutritionverse/nutritionverse-real` (1,1 GB); baixa daqui | Só avaliação, com filtro | Nenhum no treino; teste externo fraco | **A** |

A coleta brasileira própria (`docs/coleta-brasil.md`) fica fora desta tabela porque não é uma fonte pronta, mas continua sendo o item de maior valor: é o único jeito de ter prato, recipiente e porção daqui.

## 2. Rotulados com peso medido: detalhes

### 2.1 FlavoriaFoodWeight1700 (verificado e montado em 11/11/2026)

> Sarapisto, T., Koivunen, L., Mäkilä, T., Klami, A., Ojansivu, P. (2022). *FlavoriaFoodWeight1700: Automated Lunch Line Meal Pictures with Automatic Measurement of Weight and Contents* (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5850856. Licença **CC BY 4.0**: a atribuição acima é obrigatória em qualquer uso, inclusive no modelo do produto.

**Estado: pronto para o treino.** A lista está em `eval/treino/kaggle/listas/externo_flavoria.csv`; quem a monta é `eval/dados_ext/flavoria_monta.py`, a partir do `dataset_csv.tar` (1,3 MB) e de duas tabelas nossas: `flavoria_componentes.csv` (os 246 nomes) e `flavoria_arquivos.csv` (a listagem do tar de imagens, com as medidas por foto). As imagens não estão no repositório nem nesta máquina: o notebook baixa `https://zenodo.org/api/records/5850856/files/dataset_images_and_csv.tar/content` (2,58 GB).

**O que foi conferido.**

- **Licença:** `cc-by-4.0`, acesso aberto, publicado em 10/06/2022, no registro do Zenodo (API do registro, depois que `zenodo.org` foi liberado).
- **Arquivos:** `dataset_images_and_csv.tar` (2.576.977.920 bytes, md5 `56270b20f0261777bd5e81f46cfd9097`) e `dataset_csv.tar` (1.277.440 bytes, md5 `74410f922287ceffbd8092d7dd4e5530`). Listei o tar de imagens inteiro em fluxo, sem gravar: **2.013 PNG, todos 1024×1024**, em `images/train` (1.716), `images/val` (104) e `images/test` (193), mais os mesmos 11 CSVs. **Não há README.**
- **As fotos** (vi cerca de 110, em mosaicos): um prato raso branco visto **de cima**, já recortado e mascarado (fundo preto fora do prato), enchendo o quadro. É a refeição de verdade: salada, acompanhamento e prato quente no mesmo prato. A luz é fraca e uniforme. Cada bandeja tem um quadro `enter` (1.130) e, quando a pessoa se identifica, um `identification` (883), tirados com 2 a 13 s de diferença.
- **Os CSVs:** 12.284 pesagens, 3.343 sessões, 687 bandejas, 246 nomes de componente, 30 balanças (duas linhas "koti", comida caseira, e uma "vege", vegetariana; cubas de salada e cubas quentes), 32 dias de almoço entre 05/10 e 20/11/2020. Colunas: `session_id, weighing_event_time, [weighting_type,] scale_identifier, weight_of_a_component, component_name, tray_id, user_identification_time`.
- **`registered` e `non_registered`** não são explicados em lugar nenhum. O que os dados mostram: só as sessões dos arquivos `registered` têm foto `identification`; as dos `non_registered` têm no máximo a foto `enter`. Leio como "a pessoa se identificou no leitor" contra "passou sem se identificar". Duas sessões aparecem nos dois arquivos.
- **Horários:** um dos arquivos está em UTC e os outros em hora local (UTC+3 até 24/10, UTC+2 depois). Com isso, a época no nome da foto `identification` coincide com `user_identification_time` **ao segundo em 877 de 883 fotos**: o casamento foto ↔ pesagem é firme.
- **Só 1.292 das 3.343 sessões têm foto.** As outras 2.051 não entram.

**Limpeza** (regras e contagens que o script imprime):

| Regra | O que faz | Quantos |
|---|---|---|
| E1 | Linhas de pesagem repetidas, e as duas sessões exportadas em dois arquivos (vale a `registered`) | 13 linhas |
| E2 | Pesagens abaixo de 5 g: descartadas. O histograma tem um pico em 3 g (toque no talher) | 660 pesagens |
| P1 | Bandeja com sopa ou sobremesa (18 nomes): sai, porque são servidas em tigela ou prato à parte, fora da foto | 18 pratos com foto |
| P2 | Bandeja com pesagem implausível (> 300 g numa cuba de salada ou > 650 g numa cuba quente; são os vales dos dois histogramas): sai inteira | 9 pratos com foto |
| P3 | Massa total fora de 30 a 1.200 g | 1 prato com foto |
| F1 | Foto sem sessão da mesma bandeja a até 90 s | 0 fotos |
| | **Ficam** | **1.264 pratos, 1.967 fotos** (861 `identification`, 1.106 `enter`) |

- As falhas de balança existem (no CSV inteiro, 440 pesagens passam de 600 g; o padrão típico é "620 g de alface" seguido de "58 g" na mesma cuba, 7 s depois), mas **quase não caem nas sessões com foto**: os autores parecem ter publicado fotos só de sessões já filtradas. Por isso a limpeza tira 28 pratos, e não centenas.
- Pesagens repetidas na mesma cuba são somadas (68 pratos, aviso `repeticao`).
- Um projeto de terceiros (`shalev396/ml-lab`) chega a 856 pratos porque só usa as fotos `identification`. Nós ficamos também com as bandejas que só têm a foto `enter`.

**O que saiu.**

| | Pratos | Fotos | Massa mediana | Acima de 600 g | Proteína fraca > 25 g |
|---|---|---|---|---|---|
| Tudo | 1.264 | 1.967 | 459 g (p10 310, p90 650, máx. 1.090) | 202 | 653 |
| `treino_ext` | 1.034 | 1.591 | 457 g | 154 | 532 |
| `teste_ext` (10 a 20/11/2020, 7 dias) | 230 (18,2%) | 376 | 485 g | 48 | 120 |

- **Parte segurada (12.5):** os últimos dias inteiros de coleta. Com os 7 últimos dias dá 18,2% dos grupos; com 8 (incluindo 09/11) daria 23,7%. Fiquei com o corte mais perto de 20%. Nenhum dia está dos dois lados.
- **Composição pela massa:** vegetal 29%, prato pronto (misto) 28%, amido 25%, proteína animal 13%, leguminosa 4%, molho 1,5%. Fontes de proteína presentes (≥ 20 g no prato): prato pronto em 632 pratos, leguminosa e soja em 193, peixe em 184, frango em 141, suína e embutidos em 135, bovina em 38. Quase não há ovo, laticínio nem castanhas.
- **Proteína fraca** (gramas pesados × densidade de tabela pelo nome): mediana de 25,5 g por prato; 653 pratos acima de 25 g; em 450 deles, mais de 25 g vêm de componentes que são fonte de proteína; em cerca de 430, de um único componente. **68% dessa proteína vem de densidades de confiança C** (preparações com receita suposta). Ela está na lista em colunas separadas (`proteina_fraca_g`, `proteina_fraca_conf`, `proteina_fraca_fontes_g`); a coluna `proteina` fica **vazia**.

**O mapa de nomes** (`eval/dados_ext/flavoria_componentes.csv`): para cada um dos 246 nomes, a tradução, o grupo (os 8 de `grupos_alimentos.py`), a fonte de proteína (as 10 de `dens_comum.py`), a referência de tabela (`taco:` ou `n5k:`, com mistura explícita quando é preparação) e a densidade calculada dela, a confiança (A 43, B 88, C 115) e a razão nos casos ambíguos, quase todos nomes finlandeses. Convenções: peça de proteína, com ou sem molho por cima (filé empanado, almôndega, salsicha), vai para a fonte dela; proteína cozida dentro de molho, massa ou legumes (ensopado, curry, lasanha, paella, wok, gratinado) vai para `prato_misto`, como no Nutrition5k; mistura de salada com leguminosa vai para leguminosa quando ela é metade ou mais.

**Problemas conhecidos do rótulo.**

1. **O nome do componente vem do cardápio programado na cuba, não do que foi servido.** Na sessão exportada duas vezes, 2 dos 6 nomes diferem entre os arquivos, com a mesma balança e o mesmo peso. Em 8 fotos comparadas com a lista, os pratos quentes bateram em todas; em 2, um item de salada não batia ("Watermelon" era salada de tomate, pepino e azeitona; "Kurkku-mandariini" era tzatziki). **O peso é confiável; o nome das saladas, nem sempre.** Como quase tudo no balcão de saladas cai em `vegetal`, isso pesa pouco nos grupos e nas frações.
2. **Pratos prontos entram inteiros.** 28% da massa está em `prato_misto`, onde o Nutrition5k separaria frango, arroz e molho. É a razão do peso 0,5 nesses alvos em 12.5.
3. **Itens não pesados.** Não dá para saber pelo CSV. Nas fotos conferidas, o molho servido sobre o prato na linha vegetariana (tártaro, creme azedo com gergelim) **estava** pesado, e o pão pita também (entra como "pita recheada de falafel"). Pão da mesa de pães e molho de salada do autosserviço podem não estar.
4. **Oclusão.** Nas fotos `enter` aparece com frequência a **cabeça** de quem carrega a bandeja, cobrindo até metade do prato. A medida barata que montei (maior arco da borda branca encoberto, `arco_borda_oculta`, calculada por `eval/dados_ext/flavoria_quadros.py` lendo o tar em fluxo) marca 119 fotos com `oclusao = 1` (110 `enter`, 9 `identification`). Num sorteio de 32 marcadas, 27 eram oclusão de verdade; ela não pega mão pequena na borda e erra quando a comida passa da borda. Em 36 pratos todos os quadros estão marcados. `quadro_preferido` aponta o quadro mais limpo de cada prato (856 `identification`, 408 `enter`).
5. **Nem tudo é prato.** Pelo menos uma foto mostra uma embalagem preta para viagem.
6. **Enquadramento e escala.** O prato enche o quadro e o fundo é preto: não é o enquadramento do Nutrition5k. 12.5 trata disso com o viés de massa por domínio.
7. **O split oficial do tar** (`train`/`val`/`test`, guardado em `split_oficial`) não respeita dias. O nosso `split_ext` é por dia.

**Evidência de que ajuda (de terceiros, não nossa).** O projeto calorie-lens treinou com Nutrition5k + 555 pratos do Flavoria e relata, em 301 pratos de dias não vistos, erro de massa de 45,4% → 25,5%, e de 410 g → 86 g nos pratos acima de 600 g. Não reproduzi.

**Pista adicional.** O grupo mantém um catálogo (`flavoriadatacatalog.tt.utu.fi`) e um sistema mais novo, Flavoria Flex, com foto antes e depois e pesagem. É um contato possível para mais dados do mesmo tipo.

### 2.2 NutritionVerse-Real: licença verificada e um problema nos rótulos

- **Licença:** CC BY-NC-SA 4.0, no metadado do Kaggle, em conta dos autores. Não comercial. Em `datasets_internacionais.md` isso estava como "não verificada".
- **Rótulos:** 251 pratos no CSV; 889 fotos de celular (3024×4032) que cobrem 225 pratos, em geral 4 fotos por prato, de cima e inclinadas, sem o ângulo rotulado. Proteína mediana de 50 g: 165 dos 225 pratos têm mais de 25 g.
- **Problema, achado ao conferir os números:** cada um dos 45 tipos de item tem **um único peso no CSV inteiro**. Toda "chicken-wing" pesa 43 g, toda "french-fry" 5 g, todo "meatloaf" 701 g, toda "lasagna" 307 g. O total do prato é a soma desses pesos nominais. A foto do prato 100 mostra duas fatias finas de bolo de carne num prato raso, e o rótulo diz 2 × 701 g = 1.402 g e 364 g de proteína. Ou seja: **o peso não foi medido prato a prato**, ao contrário do que a descrição dá a entender.
- **Consequência:** para itens contados (asinha, sushi, ovo, almôndega) o rótulo é plausível; para itens cortados, não. Marquei três tipos como suspeitos (meatloaf, near-whole-chicken, lasagna), o que tira 38 pratos; sobram 187, sendo 129 com mais de 25 g. Abri só duas fotos, então pode haver outros tipos com o mesmo problema. **Vale como teste externo fraco de pratos ricos em proteína, não como gabarito.**

### 2.3 Os três grandes com licença não comercial

- **OmniFood8K.** O README dos autores diz: "publicly available for non-commercial research purposes... licensed under CC BY-NC 4.0". O e-mail de contato está no próprio README (`github.com/yudongjian/OmniFood8K-food`). O dataset está num arquivo do Google Drive; os pesos do modelo, em `huggingface.co/dongjianjjjj/OmniFood8k`. Os autores também publicam um conjunto sintético (NutritionSynth-115K), que não serve (critério: só foto real).
- **Food Portion Benchmark.** Já descrito em `datasets_internacionais.md`. Novo: não existe cópia no Kaggle; o caminho é `hf download` dentro do notebook.
- **MetaFood3D.** Segundo a página oficial (lida por resumo de busca), CC BY-NC 4.0, uso de pesquisa, com formulário de pedido. As cópias no Kaggle: `darrinbright/metafood3d` (7 GB, "MIT"), `darrinbright/metafood3d-dataset` (19 GB, "Apache-2.0"; inclui imagens **renderizadas** no Blender), `jauzanneau/metafood3d` (3 GB, sem licença; 9.121 quadros reais de 108 objetos, um por classe) e `brightdarrin/metafood3d-xlsx` (a planilha de rótulos). **Não usar essas cópias no produto:** a licença delas foi posta por quem subiu.
- **Avaliação com dataset NC.** Usar um conjunto não comercial só para medir, dentro de uma empresa, é zona cinzenta: a leitura conservadora é pedir autorização também para isso. A decisão é da fundadora.

### 2.4 O que o Kaggle tem e não serve

| Dataset no Kaggle | Por que não |
|---|---|
| `mirzaashifmugdho/foodbd-...` (FoodBD, 3.523 fotos de celular de refeições de Bangladesh; CC BY-SA 4.0) | Nutrição de 1.837 imagens **estimada por especialistas**, não pesada |
| `crispen5gar/recipe-images-with-nutritional-information` (1.258 receitas) | Nutrição por porção da receita; a foto não mostra uma porção medida. Fotos de editora |
| `dwiiyy/nutricheck-image-dataset`, `sanadalali/food-101-nutritional-information`, `mariamabdelfattah76/...` | Classe + tabela de calorias por classe; sem peso |
| `darsh22blc1378/foodd-ieee-datasets`, `rusqi29/food-detection-dataset-for-calorie-measurement` (FooDD) | Alimento isolado com o polegar como escala; sem peso por imagem |
| `warcoder/fruit-recognition-and-calories-estimation`, `mypapit/mangomassnet552-dataset` | Frutas |
| `thezaza102/tray-food-segmentation`, `luongnhatminh/trayfood`, `tommyngx/foodtraydl`, `smakkkk/tray-of-food` | Bandejas com segmentação, sem peso |
| `visang2022/school-lunch-waste-data-for-education`, `rahuljangir78/food-waste-dataset-from-a-university-canteen` | Peso servido e sobra em planilha, **sem imagem** |
| `hadeelahmadalshlafi/restaurant-food-waste-and-leftovers-images` | Fotos da web, sem peso |
| `catarinaantelo/portuguese-meals` (6.771 imagens, 23 classes), `sc0v1n0/26-tipical-brazilian-foods` (CC BY-NC-SA) | Só classe; imagens da web |
| `tianbaochenchen/finalversion`, `tianbaochenchen/dip-2025` | Cópias do ECUSTFD, que já temos |

## 3. Nutrition5k: o que ainda não foi usado

Contas feitas com os metadados do bucket e as nossas listas (`zen_treino.csv`, `n5k_teste2_amostra.csv`).

| Reserva | Pratos | Com mais de 25 g | Situação |
|---|---|---|---|
| Fotos de cima, total | 3.490 | 986 (com massa ≥ 20 g) | Uma foto por prato: `rgb.png`, `depth_raw.png`, `depth_color.png`. **Não há outros quadros nem outras câmeras de cima** |
| Treino atual (estrito) | 1.684 | 461 | Em uso |
| Avaliação antiga (`calib`, `teste`, `n5k_v1`) | 450 (435 passam nos filtros) | ~313 nesta linha, na de baixo e nos reprovados com massa ≥ 20 g, somados | Fora do treino por desenho |
| Cadeias do `calib` | 405 | | Fora do treino; ficaram de fora do teste2 porque o `calib` ajustou o híbrido |
| teste2 | 829 | 212 | Lido uma vez em 11/11 |
| Reprovados nos filtros do v2 | 137 | incluídos nos ~313 acima | Massa < 20 g, ingrediente `deprecated`/`plate only`/sem nome, soma incoerente |
| Só com vídeo lateral | 1.504 (1.432 nos filtros) | 119 | 8 quadros por prato já baixados (seção 7) |
| Café 2 | 238 no total | | **Já entra:** 83 no treino e 25 no teste2; os totais zerados são reconstruídos pelos ingredientes |

- **A reserva que importa são os 1.669 pratos de cima fora do treino.** São do mesmo refeitório e da mesma câmera, então não trazem variedade; trazem uns 500 pratos ricos em proteína. Decisão de método, não de dados: treinar o modelo de produção com eles significa não ter mais teste dentro do Nutrition5k. Antes de decidir, dá para medir de graça, só com o treino atual: **curva de aprendizado por faixa** (treinar com 50% e 100% dos pratos e olhar o acerto de CV só na faixa > 25 g). Se a faixa alta ainda sobe de 50% para 100%, vale dobrar; se já está parada, não vale gastar o teste.
- **Recuperar parte dos 137 reprovados.** O projeto calorie-lens documenta correções conferíveis: 10 pesos com erro de escala de ×1000, o item "plate only" (peso do prato vazio) e duas linhas `deprecated` que são duplicatas de maçã e melancia. Aplicar isso devolveria algumas dezenas de pratos. Ganho desprezível; só vale junto com o item acima.
- **Cópias no Kaggle.** `gillesokhin/nutrition5k-dataset` (23 GB, CC BY 4.0) tem exatamente as 3.490 pastas de cima do bucket (conferi: 10.470 arquivos) mais quadros laterais já extraídos, 1 a cada 5 (~115 por prato). **Não acrescenta nenhum prato nem foto de cima.** Serve só para não baixar do bucket a cada execução. Os CSVs dessa cópia têm 4.768 pratos: **faltam os 238 do café 2**. `salman719/nutrition5k` e `saeikhabdullah/nutrition5k-dataset-2` (195 GB) são o bucket inteiro. `siddhantrout/nutrition5k-dataset` é um pickle de 2,5 GB (não abrir: pickle executa código). As demais são recortes.

## 4. Fotos sem rótulo, ou com rótulo fraco, para destilação

Objetivo: usar as nove redes do servidor como professor e treinar um modelo só com as previsões delas em muitas fotos novas. Isso **não ensina nada que o professor não saiba**; serve para o modelo único chegar perto do conjunto e para ele ver fotos de celular, pratos e fundos que o Nutrition5k não tem.

| Fonte | Tamanho | Licença | No Kaggle? | Observação |
|---|---|---|---|---|
| **MyFoodRepo-273** (Food Recognition Benchmark, EPFL) | 24.119 fotos de refeição tiradas com celular (versão do artigo) | CC BY 4.0 segundo o artigo (B); conferir na página do AIcrowd | Sim: `awsaf49/food-recognition-2022-dataset` (2,6 GB), `sainikhileshreddy/food-recognition-2022` (5,3 GB) | **A melhor fonte pronta:** foto de usuário de app de dieta, o nosso domínio. Sem peso |
| **Open Images, classes de comida** | **88.516** fotos com rótulo humano positivo (84.813 "Food") no treino | Imagens CC BY 2.0 (A: 4.922 de 4.922 na validação); anotações CC BY 4.0 | Não como recorte de comida; baixa direto do S3 dentro do notebook (e daqui) | **Muito ruído (A):** em 24 fotos sorteadas, 1 ou 2 eram um prato visto de cima; o resto, pessoas, feiras, embalagens, bebidas. Depois de um filtro zero-shot devem sobrar 5 a 10 mil (estimativa). Tem rostos: filtrar |
| **SNAPMe** (USDA) | 3.311 fotos de celular; 1.163 fotos "antes" de refeição não embalada | **CC BY-SA 4.0** no registro (corrigido em 12/11; estava como domínio público), e os autores pedem para **não treinar** com ele | Não; baixa de `ndownloader.figshare.com` | Reavaliado na seção 10 como rótulo fraco: lista pronta (`externo_snapme.csv`), uso pendente de decisão |
| **Quadros laterais do Nutrition5k** | 36.970 já extraídos; ~115 por prato na cópia do Kaggle | CC BY 4.0 | Sim | Mesmo refeitório: pouca variedade |
| **Fotos do Flavoria fora da lista** | 46 de 2.013 (pratos tirados pela limpeza) | CC BY 4.0 | Não | Quase todas as fotos do Flavoria já entram com rótulo |
| Food-101 | 101.000 | Imagens do Foodspotting; o acordo original só cobre uso científico (B). **Não é limpa para produto** | Sim (`dansbecker/food-101`, `kmader/food41`) | No Kaggle aparece como "other" e "copyright-authors" |
| UEC FOOD-256, Food2K, VireoFood, ChineseFoodNet, ISIA | 30 mil a 1 milhão | Pesquisa não comercial (C; ChineseFoodNet: "free for academic research" na própria descrição do Kaggle, A) | UECFOOD256 e ChineseFoodNet sim; **Food2K não** | Não usar no produto |
| Recipe1M+ | 13 milhões de imagens | Acordo de pesquisa (C) | Só derivados de texto (`shlokpatel0812/culinaai-datasets`); imagens não | Não usar |
| FoodSeg103 | 7.118 | Imagens vêm do Recipe1M | Sim, várias cópias | Herda a restrição |
| CNFOOD-241 | 191 mil | "other" no Kaggle; fotos da web com marca d'água | `zachaluza/cnfood-241` | Não usar |

- **A fonte que vai importar de verdade não está nesta tabela:** as fotos do próprio app, tiradas de cima com o círculo-guia, com consentimento. Mil fotos do beta valem mais que as 88 mil do Open Images.
- **Rótulo fraco de receita** (proteína e kcal por porção, com foto empratada): no Kaggle só achei coleções pequenas e em inglês, com foto de editora. A razão proteína/kcal é independente da escala e poderia servir de alvo auxiliar de densidade, mas a foto não mostra uma porção medida e o direito sobre a foto não é limpo. Prioridade baixa.

## 5. Criar dado brasileiro rotulado gastando pouco

Além do protocolo de `docs/coleta-brasil.md`.

### 5.1 Marmitas e pratos prontos: o que dá para coletar de hosts liberados

| Marca | Host | O que há | Limite |
|---|---|---|---|
| Liv Up | `www.livup.com.br` (liberado) | 85 pratos completos com peso e tabela; já usamos 34 | Foto de estúdio **em ângulo** |
| Swift Prato Feito / Pro&Fit | `www.swift.com.br` (liberado; API do catálogo responde) | **8 pratos** de 300 g (estrogonofe, costela com mandioca, carne moída com arroz e feijão, tilápia com purê, sobrecoxa...) | Fotos e tabela nutricional (uma imagem) ficam em `swiftbr.vteximg.com.br`, **bloqueado** |
| Seara | `www.seara.com.br` (liberado) | Já usamos 7 | Foto de embalagem ou sugestão |
| Sadia | `www.sadia.com.br` (responde) | Não explorado | Provável foto de embalagem |
| Carrefour Mercado | `mercado.carrefour.com.br` | A página responde; a API do catálogo devolveu 503 | Foto de embalagem |
| Pão de Açúcar | `www.paodeacucar.com` | Já descartado em `coleta.md`: só foto da caixa | |
| Mr. Fit, 99Food, iFood, Perdigão | `mrfit.com.br`, `99app.com`, `ifood.com.br` bloqueados; `www.perdigao.com.br` devolve 403 do próprio site | Não coletável daqui | |
| Vapza, Beleaf, Pratí (ex-Lucco Fit), Bálsamo, Urban Farmcy | Hosts bloqueados | Só nomes, vindos de busca (B) | Não verificado se publicam foto de cima e tabela |

**Conclusão honesta:** e-commerce rende algumas dezenas de itens a mais, quase todos com foto de estúdio em ângulo. É material de **teste**; como treino, não muda nada.

### 5.2 O que eu faria no lugar: comprar e fotografar

- **Comprar as marmitas e fotografar de cima.** O rótulo dá o peso e a proteína; a balança confirma o peso depois de aquecer. Cada marmita, servida num prato de casa, vira uma refeição rotulada em 3 a 4 minutos, com proteína de rótulo (melhor que tabela). Só a Liv Up tem 85 pratos completos, e as linhas "performance" têm os pratos de 30 a 45 g que faltam. Quando os componentes vêm separados na embalagem, pesar cada um ao servir. Ressalva: a tabela do rótulo tem tolerância legal (±20%, de memória; conferir), então é um gabarito mais frouxo que a pesagem por ingrediente.
- **Série de porções da fonte de proteína.** É o que os álbuns de nutrição fazem, e ataca direto a faixa fraca: frango, carne, peixe ou ovo já prontos, fotografados no mesmo prato com 50, 100, 150, 200 e 250 g, com e sem acompanhamento. Cinco fotos rotuladas em dois minutos, todas acima de 15 g de proteína. As fotos da mesma série são a mesma refeição para efeito de separar treino e teste.
- **Restaurante por quilo e curso de Nutrição** continuam sendo as fontes de volume (já estão no protocolo).
- **Depois do lançamento:** toda correção de porção feita pela usuária é um rótulo fraco. Guardar foto, sugestão e correção, com consentimento, é a fonte que cresce sozinha.

### 5.3 Álbuns e manuais fotográficos de porções (além do UFPR 2017)

Todos em hosts bloqueados daqui. Existência e conteúdo vêm de resumo de busca (B). **Não sei o ângulo das fotos de nenhum deles**; o UFPR 2017 é a 45° e é provável que os outros também sejam, porque é o padrão desses álbuns.

| Material | Onde | Conteúdo segundo a busca |
|---|---|---|
| Manual fotográfico de quantificação alimentar infantil/escolar (UFPR) | `gupea.ufpr.br/wp-content/uploads/2024/10/Manual_Infatil_Quantificação_Escolar.pdf` | 73 alimentos e 44 medidas caseiras, com gramas |
| Álbum fotográfico de quantificação alimentar para crianças (Editora UFPE) | `editora.ufpe.br/books/catalog/book/513` | PDF aberto; conteúdo não visto |
| Álbuns fotográficos digitais de porções por grupo de alimento (UFLA, TCCs) | `sip.prg.ufla.br` | Porções 1, 2 e 3 com gramas (pães, queijos, leites, pastosos) |
| Manual Fotográfico de Quantificação Alimentar do IAN-AF (Portugal) | `ian-af.up.pt` (cópia no Scribd) | 176 séries de 159 alimentos ou refeições, porções crescentes; em português, com pratos compostos |
| UnB (Lopez, 2007), UFRN, UNICESUMAR | Ver `fontes.md`, seção 3 | |

Uso realista: mais algumas centenas de fotos de **alimento isolado** com gramas, em ângulo, sem direito de redistribuir. Bom para teste de porção; fraco para treino de prato de cima. O IAN-AF é o que mais vale abrir, por ter refeições.

## 6. O que eu faria primeiro

1. **Flavoria na próxima execução do Kaggle.** Feito o que dava para fazer daqui: licença conferida, lista montada (`externo_flavoria.csv`), regra de uso pré-registrada em 12.5 de `resultado_modelo_treinado.md`. Falta rodar. O primeiro número a olhar é o erro de massa do modelo atual nos 230 pratos do `teste_ext`: se for de dezenas de por cento, isso por si só mostra quanto o modelo sofre fora do Nutrition5k.
2. **Curva de aprendizado por faixa de proteína, na CV** (barato, sem tocar em teste): decide se vale gastar o teste do Nutrition5k para dobrar os pratos acima de 25 g.
3. **Cinco e-mails** (seção 8): OmniFood8K primeiro.
4. **Série de porções de proteína + marmitas compradas, de cima** (um fim de semana, ~100 refeições): é o primeiro dado brasileiro de cima que o projeto teria, e cai na faixa fraca.
5. **Destilação**, só quando a escolha das redes vencedoras estiver feita: MyFoodRepo + SNAPMe + Open Images filtrado + fotos do beta.

O que eu **não** faria: gastar tempo com os quadros laterais do Nutrition5k (já testado, sem ganho), com MetaFood3D e ECUSTFD (alimento isolado), nem treinar com qualquer cópia do Kaggle cuja licença original seja NC.

## 7. O que NÃO existe (até onde esta busca alcança)

- **Nenhum dataset público brasileiro, latino-americano ou em português com prato fotografado e peso medido.** Busca no Kaggle ("brazilian food", "comida brasileira"), no HF e na web: só classificação (MyFood, Brazilian food images, 26 Tipical Brazilian Foods, Portuguese Meals) e álbuns de nutrição em PDF.
- **Nenhum dataset com proteína medida em pratos compostos, licença comercial e tamanho relevante além do Nutrition5k.** O Flavoria tem peso, não proteína.
- **Nenhum dataset no Kaggle, além das cópias do Nutrition5k, com peso medido e licença comercial válida.** O NutritionVerse-Real é NC-SA; as cópias do MetaFood3D estão com licença errada.
- **Nenhuma foto de cima a mais no Nutrition5k:** uma por prato.
- **Nenhum conjunto grande de fotos de comida de cima, sem rótulo, com licença comercial.** As fontes limpas são pequenas (MyFoodRepo, SNAPMe) ou ruidosas (Open Images).
- **Food2K e as imagens do Recipe1M+ não estão no Kaggle**, e as licenças originais são de pesquisa.
- **Marmitaria brasileira com foto de cima e tabela nutricional por prato, em volume para treino:** não encontrei.

## 8. O que depende da fundadora

| Ação | Para quem | Para quê |
|---|---|---|
| Pedir licença comercial (ou parceria) | OmniFood8K: autores na U. Yunnan / ICT-CAS (e-mail no README do repositório) | 8.036 pratos pesados; o maior ganho possível |
| Pedir licença comercial | ISSAI, Universidade Nazarbayev (Food Portion Benchmark) | 14 mil imagens com peso |
| Pedir licença comercial e acesso | Purdue (MetaFood3D: formulário na página) | 637 a 743 objetos com proteína |
| Perguntar onde está o dataset e qual a licença | FLIC: Marco Buzzelli, UNIMIB | 401 bandejas pesadas, de cima |
| Pedir licença comercial | ACETADA: Curtin University e Purdue (autores do arXiv 2507.07048); já publicado, CC BY-NC 4.0 | 806 fotos de celular com comida pesada |
| Confirmar permissão de uso em produto | ECUSTFD: Liang & Li, ECUST | Já baixado, sem licença declarada |
| Decidir | Uso de datasets NC só para **avaliar** dentro da empresa | NutritionVerse-Real, FPB, OmniFood8K como testes externos |
| Decidir | Gastar o teste do Nutrition5k para treinar o modelo de produção | Depois do item 2 da seção 6 |
| Liberar hosts no ambiente | `us.aws.cdn.hf.co`, `drive.google.com`, `swiftbr.vteximg.com.br`, `ian-af.up.pt`, `gupea.ufpr.br`, `www.mdpi.com`, `data.mendeley.com` | Conferir daqui o que hoje está como B |

## 9. O que não consegui verificar

- No Flavoria: o que `registered` e `non_registered` querem dizer (não há README; a leitura da seção 2.1 é inferência), e quais itens do autosserviço não são pesados.
- Se o notebook do Kaggle alcança de fato o Zenodo, a CDN do HF e o Google Drive. É a premissa da tarefa; não testei. Daqui, o Zenodo entregou os 2,6 GB em cerca de 20 min.
- O conteúdo do OmniFood8K (ângulo das fotos, formato dos rótulos, se o peso é por prato). Depois do que apareceu no NutritionVerse-Real, **não assumir que "pesado" quer dizer pesado prato a prato** antes de abrir os rótulos.
- Licença e download do FLIC (a MDPI continua bloqueada; ver seção 10). O LeFood-Set foi aberto e o ACETADA teve a licença esclarecida na segunda varredura.
- As licenças de MyFoodRepo, Food-101, UEC FOOD-256, Food2K e Recipe1M+ nas páginas oficiais (B ou C).
- A listagem completa dos quadros laterais na cópia do Nutrition5k no Kaggle (vi 31 pratos).
- AI Hub (Coreia): continua sem verificação; exige cadastro coreano.
- Se alguma marmitaria brasileira publica foto de cima: quase todos os hosts estão bloqueados.
- Todos os números de ganho.

## 10. Segunda varredura (12/11/2026): repositórios, pistas pendentes e SNAPMe

**Por quê:** a terceira execução em GPU deixou o modelo único em 66,0% no teste2, o Nutrition5k acabou (tudo o que serve está no treino ou no teste) e o Flavoria não foi adotado pela regra de CV, embora tenha cortado o erro de massa fora do domínio. A instrução é buscar mais pratos, de preferência ricos em proteína, com peso medido e licença comercial.

**Resposta curta: não achei nenhum conjunto novo de pratos compostos, de cima, pesados, com licença comercial.** Saíram duas listas novas, as duas com ressalva séria, e três pedidos que valem o tempo da fundadora.

| | Fotos / pratos | Rótulo | Licença | Pratos com mais de 25 g de proteína (fraca) | Lista |
|---|---|---|---|---|---|
| **SNAPMe** (USDA, 2022) | 1.163 fotos "antes", 95 pessoas | **Autorrelato** (ASA24): nada pesado | **CC BY-SA 4.0**, e os autores pedem para não treinar com ele | **277** (203 contando só os itens que são fonte de proteína) | `externo_snapme.csv`: todos os alvos vazios, autorrelato em colunas `*_fraca` |
| **LeFood-Set** (Indonésia, 2025) | 811 fotos, 524 porções | Peso **medido** em balança, um item por prato | CC BY 4.0 | **0** (máximo de 15,6 g); 512 fotos são de uma fonte de proteína | `externo_lefood.csv`: massa, grupo e fonte preenchidos |
| **GT Mensa 2025** (Itália, 2026; achado depois, ver 10.9) | 200 fotos de bandeja inteira | Peso **medido** por item em balança de cozinha (870 itens) | CC BY 4.0 | 39 bandejas com mais de 25 g vindos de fontes de proteína; 17 com um único item acima de 25 g | `externo_mensa.csv`: massa, grupo e fonte preenchidos |
| Flavoria (para comparar) | 1.967 fotos, 1.264 pratos | Peso medido por componente | CC BY 4.0 | 450 | `externo_flavoria.csv` |

### 10.1 O que foi buscado

Hosts novos desde a primeira passada: `zenodo.org`, `doi.org`, `arxiv.org`, `data.mendeley.com`, `api.figshare.com`, `ndownloader.figshare.com`, `dataverse.harvard.edu`. As mesmas 54 consultas foram a cada repositório (40 em inglês, 14 em português e espanhol: *food image weight*, *food portion estimation images*, *meal images weight*, *plate waste images*, *canteen tray images*, *food volume estimation dataset*, *calorie estimation images dataset*, *dietary assessment images dataset*, *nutrient estimation food images*, *food scale images*, *hospital meal images*, *school lunch images*, *imagens alimentos peso*, *álbum fotográfico porções alimentos*, *estimación porciones alimentos imágenes* e variantes; a lista inteira está em `<scratchpad>/varredura2/scripts/consultas.py`).

| Onde | Como | Registros vistos | Passaram no filtro de título e descrição | Abertos (licença, arquivos, amostra) |
|---|---|---|---|---|
| Zenodo | API de registros, só datasets: 54 consultas (50 primeiros de cada) + 12 consultas booleanas (até 200 de cada) + 4 dirigidas | 1.443 + 289 + 24 | cerca de 190 títulos lidos um a um | 6 |
| Figshare | API de busca, só datasets: 54 consultas | 420 | 39 | 5 |
| Harvard Dataverse | API de busca: 54 consultas | 1.249 | 16 | 0 (nenhum era de imagem de comida com peso) |
| arXiv | Busca do site: 25 consultas | 111 artigos | 39 de 2024 a 2026 que falam de dataset e de peso ou porção | 5 artigos lidos |
| Mendeley Data | **Sem busca:** a API pública ignora o termo de busca. Usei busca na web e a API de registro para o que já era conhecido | 4 registros achados pela web | | 1 (LeFood-Set, inteiro) |
| OSF | **Não buscado:** `api.osf.io` está bloqueado | | | |
| Kaggle e Hugging Face | Só os nomes novos desta passada (11 e 13 consultas) | | | 0 novos |

O filtro automático pega pouco e erra muito nesses repositórios (a busca do Zenodo devolve ecologia e geologia para "food weight"), então o que decide é a leitura dos títulos. É possível que exista um dataset pequeno com título pouco óbvio que eu não vi.

### 10.2 SNAPMe: reavaliado como rótulo fraco

> Chin, E. L.; Larke, J. A.; Bouzid, Y. Y.; Nguyen, T.; Vainberg, Y.; Smilowitz, J. T.; Lemay, D. G. (2022). *SNAPMe: A Benchmark Dataset of Food Photos with Food Records for Evaluation of Computer Vision Algorithms in the Context of Dietary Assessment.* Ag Data Commons. https://doi.org/10.15482/USDA.ADC/1528346

- **Baixa daqui:** `https://ndownloader.figshare.com/files/44532971` (2,03 GB, `tar.gz`). Li o arquivo inteiro em fluxo, guardando só os rótulos e miniaturas de 256 px.
- **Licença: CC BY-SA 4.0** no registro (API do Figshare). **Não é domínio público**, como estava em `fontes.md` e na primeira passada. Dois pontos para a fundadora, não para mim:
  1. o registro tem um campo *Use limitations* que diz, textualmente: **"This data set should not be used to train models as it is a small curated data set. Models trained with this small set of images would not be generalizable."** Não é cláusula da licença; é o pedido dos autores, e o SNAPMe é usado como benchmark por outros grupos;
  2. o compartilha-igual: se um modelo treinado com as fotos conta como obra derivada é questão em aberto.
- **O que tem:** 3.368 imagens (2.893 já reduzidas a 800×800). O arquivo `master_SNAPME_linkfile.csv` liga 4.761 itens do recordatório ASA24 à foto "antes" em que aparecem, com gramas declarados, nutrientes do FNDDS e os grupos do FPED. São 95 pessoas e até 3 dias cada.
- **Depois das regras** (S1 item sem foto: 69 itens; S2 foto de embalagem: 199; S3 só bebida: 271; S4 massa declarada fora de 20 a 2.000 g: 25): **1.163 fotos**. Proteína declarada por foto: mediana 11,9 g; 162 fotos de 25 a 40 g e 115 acima de 40 g.
- **As fotos** (vi 96 sorteadas): é o domínio do produto. Celular, em casa, prato ou tigela de verdade, mesa e luz variadas, com um marcador quadriculado de tamanho conhecido na cena. **Cerca de 80 das 96 foram tiradas de cima ou quase**; o resto, em ângulo. O ângulo não é rotulado (`vista = celular_nao_rotulada`).
- **O rótulo:** em 14 fotos comparadas com a lista de itens, **a identidade bateu em todas** (salmão, coxa de frango, ovo, queijo). **A quantidade é porção padrão do ASA24**: "Pasta… 375 g" aparece igual em tigelas de tamanhos diferentes, e na ponta alta há valores que não cabem no prato (76, 78 e 90 g de proteína em refeições comuns). A quantidade declarada é a **comida**, não a servida: quando sobrou, a foto mostra mais do que o rótulo.
- **Como ficou na lista:** todos os alvos do esquema **vazios**; o autorrelato vai em `massa_fraca_g`, `proteina_fraca_g`, `kcal_fraca`, `gordura_fraca_g`, `carbo_fraca_g`, `m_fraca_<grupo>` e `ff_fraca_<fonte>`. Grupo e fonte saem das mesmas regras do Nutrition5k aplicadas à descrição do FNDDS, com três ajustes documentados em `snapme_monta.py`. `grupo` = participante; 116 fotos de 12 participantes ficam como `teste_ext` pela regra de reserva de 12.5 (não há data de coleta utilizável).
- **Para que serviria:** identidade da fonte de proteína em foto de celular e, com peso baixo, proteína e frações fracas. Não serve para massa. E só depois da decisão sobre a licença e o pedido dos autores.

### 10.3 LeFood-Set

> Sari, Yuita Arum; Wani, Yudi Arimba; Nakazawa, Atsushi (2025). *LeFood-Set: Leftovers Food Dataset.* Mendeley Data, V1. https://doi.org/10.17632/cchsk79jkt.1. Licença CC BY 4.0.

- Baixei tudo (1.049 arquivos, 52 MB) pela API pública do Mendeley. São fotos 749×548 **de cima**, um prato raso branco sobre toalha quadriculada, **um item por prato**, fotografado antes e depois de comer, com o peso em balança digital nas duas.
- O `data_original.xlsx` tem 678 linhas, mas **só 524 têm foto** (as preparações de legumes e as sopas não foram publicadas). Sobram 34 comidas: arroz e arroz papa (154 porções), frango em cinco preparos, tofu, tempeh, ovo, peixe, almôndega, croquete.
- **Lista:** 811 fotos (524 "antes", 287 "depois" com sobra de 5 g ou mais; 237 pratos vazios saíram). Massa mediana de 42 g, máxima de 343 g. 512 fotos são de uma fonte de proteína (soja 189, frango 145, ovo 66, peixe 52, bovina 30, embutido 6, prato pronto 24).
- **Limite:** são porções de hospital, pequenas. **Nenhuma foto passa de 16 g de proteína.** Não ataca a faixa acima de 25 g; ensina identidade e peso de itens de proteína isolados, e as fotos "depois" dão porções parciais do mesmo item. Cozinha indonésia.
- O mapa das 34 comidas está em `eval/dados_ext/lefood_componentes.csv`, com a tradução e a razão.

### 10.4 Pistas da primeira passada

| Pista | O que achei agora |
|---|---|
| **ACETADA** | **CC BY-NC 4.0**, não CC BY 4.0. Está publicado em `skynet.ecn.purdue.edu/~coburn6/ACETADA/` (bloqueado daqui); um registro do Zenodo (23947978) e um artigo de 2026 (arXiv 2604.06352) já o usam. 806 pares antes/depois, comida de laboratório pesada a 0,1 g, foto de celular. Só com licença dos autores |
| **FLIC** | **Continua sem abrir, agora por outro motivo** (10.9): a MDPI devolve "Access Denied" a acesso automatizado, e o dataset está num Google Drive, bloqueado daqui |
| **OmniFood8K** | Li o artigo (arXiv 2604.12356): não há subconjunto com licença comercial. O que é CC BY é o artigo; o dataset segue CC BY-NC 4.0 |
| **Grupo do Flavoria** | Nenhum outro registro no Zenodo (busquei por *Flavoria*, *Sarapisto* e *lunch line*). O catálogo do grupo e o repositório da universidade estão bloqueados |
| **Artigos de 2024 a 2026 que liberaram dados** | Dos 39 do arXiv, os que trazem peso medido são os já conhecidos (OmniFood8K, ACETADA, MetaFood3D, SimpleFood45, NutritionVerse-Real), todos não comerciais. Um artigo de outubro de 2026 (arXiv 2610.29025) testou a transferência de modelos de massa entre conjuntos e escreve que "poucos datasets públicos juntam foto e peso"; a lista dele é a nossa mais a biblioteca de porções do ASA24 |

### 10.5 O que não passou, e por quê

| Dataset | Onde | Licença | Por que não |
|---|---|---|---|
| **ClinicPlate** (Roma, 2026) | Zenodo 21626434 | CC BY 4.0, mas **acesso restrito** | **É o que mais vale pedir.** 9.868 fotos de porções individuais de 7 comidas de hospital (peixe cozido, presunto cozido, omelete, arroz, cenoura, erva-doce, abobrinha), 10 níveis de porção cada, **pesadas a 1 g**, com macronutrientes, em prato branco e bandeja. Não vi nenhum arquivo |
| FoodWeight1.4M, amostra | Zenodo 20312279 | CC BY 4.0 | Baixei as 2.500 fotos: são produtos crus em saco plástico sobre a balança do supermercado (mediana de 448 g). Não é prato. O conjunto de 1,4 milhão ainda não foi publicado |
| FoodSight (Pádua, 2025) | Zenodo 14887385 | CC BY 4.0 | 15.600 fotos de bandeja de refeitório, de cima, 1280×720, com segmentação e **sem peso**. Em 10 fotos sorteadas, 8 eram bandejas depois da refeição. Serviria só como foto sem rótulo |
| FOODITY, desperdício | Zenodo 17023233 | **Sem licença** | 486 fotos de restos de um alimento só, com peso e volume |
| NutriConv / PANCAKE | Zenodo 15800511 | CC BY 4.0 | 210 imagens de 182×136 px de um alimento só |
| O'Hara et al., fotos de refeições (Dublin, 2025) | Figshare 28271003 | CC BY 4.0 | A planilha tem o peso de cada alimento e 16 nutrientes de 114 fotos (39 refeições em 3 tamanhos de porção), **mas as fotos não estão no registro** |
| Benchmark de cantina com balança de cozinha (*Nutrients*, 2026) | GitHub `Metabolic-Intelligence-Lab/vlm-food-benchmark` | CC BY 4.0 | **Passou**, depois de localizado: virou a lista `externo_mensa.csv` (10.9) |
| Biblioteca de porções do ASA24 (NCI) | `epi.grants.cancer.gov` (bloqueado) | Governo dos EUA; não conferida | 6.329 miniaturas de estúdio de um alimento só, com peso de porção padrão. Não é prato nem peso medido por foto |
| Central Asian Food Dataset; PUST Cafeteria; DishSeg24k; FoodBD | Figshare, Mendeley, arXiv | Variadas | Só classe ou segmentação; sem peso |
| DFood-TJ (hospital, China) | Não localizado | Não conferida | Só por resumo de busca |
| Atlas fotográficos de porções (Singapura 2026, Emirados 2018) | Figshare, material suplementar em PDF | CC BY 4.0 | Não abertos; são álbuns em PDF, um alimento por série |

### 10.6 Ordem pelo valor para a faixa acima de 25 g

1. **SNAPMe:** 277 fotos de celular com mais de 25 g declarados, o domínio do produto. Rótulo fraco e uso pendente de decisão.
2. **Flavoria:** 450 pratos com mais de 25 g de proteína fraca vinda de fontes de proteína, com massa medida. Já está no pacote; a regra de CV não o adotou.
3. **ClinicPlate, se o acesso for concedido:** três das sete comidas são fonte de proteína, em 10 níveis de porção pesados. Não sei quantas fotos passam de 25 g.
4. **ACETADA, se licenciado:** 806 refeições de laboratório pesadas, em foto de celular.
5. **OmniFood8K, se licenciado:** 8.036 pratos.
6. **GT Mensa 2025:** 39 bandejas com mais de 25 g vindos de fontes de proteína (17 com um único item acima de 25 g), com peso medido por item. São só 200 fotos, e de bandeja inteira.
7. **LeFood-Set:** zero pratos acima de 25 g; entra pelo peso e pela identidade de itens de proteína.

Nada disso muda a conta da primeira passada: **prato composto, rico em proteína, de cima, pesado e com licença comercial só existe no Nutrition5k e no Flavoria.** O resto é pedido de licença ou coleta própria.

### 10.7 O que depende da fundadora, em ordem

1. **Pedir acesso ao ClinicPlate** no próprio Zenodo (registro 21626434, botão de pedido de acesso). A licença já é CC BY 4.0; falta só a liberação dos arquivos. É o pedido mais barato.
2. **Decidir sobre o SNAPMe:** a licença permite uso comercial com atribuição e compartilha-igual, mas os autores pedem para não treinar. O caminho limpo é escrever para a pessoa de contato do registro (USDA) explicando o uso como rótulo fraco e pedindo a posição dela.
3. **Pedir licença comercial do ACETADA e do OmniFood8K** (os dois CC BY-NC 4.0).
4. **FLIC:** abrir o artigo no navegador (a MDPI recusa acesso automatizado daqui), copiar o link do Google Drive e a licença, e, se a licença não estiver escrita, perguntar aos autores (10.9). Também ajudariam no ambiente `skynet.ecn.purdue.edu`, `api.osf.io` e `epi.grants.cancer.gov`.

### 10.8 O que não consegui abrir

- A MDPI (o host foi liberado, mas o site devolve "Access Denied" a acesso automatizado; ver 10.9), o Google Drive do FLIC, a página do ACETADA, o catálogo do Flavoria, a biblioteca do ASA24 e toda a busca do OSF.
- A busca do Mendeley Data: só li registros cujo endereço eu já tinha.
- Os arquivos do ClinicPlate (restritos) e as fotos de O'Hara et al. (não publicadas).
- No SNAPMe, as imagens em tamanho original: trabalhei com miniaturas de 256 px, e o ângulo foi contado à mão em 96 fotos.

### 10.9 Depois da liberação da MDPI (12/11/2026, tarde): FLIC e o benchmark de cantina

**MDPI: liberada no ambiente, mas não abre.** `https://www.mdpi.com/2076-3417/16/11/5465` e `https://www.mdpi.com/2072-6643/18/21/3582` chegam ao servidor e voltam com **HTTP 403 "Access Denied"** (página do Akamai, com número de referência): é a proteção do próprio site contra acesso automatizado, não o proxy do ambiente. Não tentei contornar. O `doi.org` redireciona os metadados para `api.crossref.org`, que está bloqueado.

**FLIC: continua sem verificação.**

- Pelo resumo de uma busca na web, a página do artigo aponta um **Google Drive** para o dataset (acesso em 26/05/2026), coletado em 22 dias de junho e julho de 2024: 401 pares de bandeja cheia e sobra, com máscaras e massa medida. **Não vi o link, a licença nem nenhum arquivo.**
- `drive.google.com` está bloqueado daqui; num notebook do Kaggle deve baixar, mas falta o identificador da pasta.
- Busquei de novo no Zenodo (por autor e por tema) e no Figshare: nada. Os hosts da UNIMIB (`ivl.disco.unimib.it`, `boa.unimib.it`) também estão bloqueados.
- **O que falta é de dois minutos num navegador:** abrir o artigo, copiar o link do Drive e ler a declaração de disponibilidade. Se a licença não estiver escrita, vale o pedido 4 de `pedidos_licenca.md`.

**Benchmark de cantina: localizado, conferido e montado.**

> Marchetti, T.; Capezzone, S.; Esposito, R.; De Spirito, M.; Maulucci, G. *Benchmarking Open-Source Vision-Language Models for Food Image Recognition and Portion Size Estimation.* Nutrients 2026, 18(21), 3582. https://doi.org/10.3390/nu18213582. Dados em `github.com/Metabolic-Intelligence-Lab/vlm-food-benchmark`, licença **CC BY 4.0** (arquivo `LICENSE-DATA`; o código é MIT).

- **Baixa daqui** por `raw.githubusercontent.com`: `labels_flat.csv`, `GT_Mensa_2025/image_participants.csv` e as 200 fotos (20 MB). Baixei tudo para o scratchpad.
- **O que é:** 200 fotos de celular (lado maior de até 748 px, sem EXIF) tiradas por seis pessoas num refeitório universitário italiano, quase todas de cima. **A foto é da bandeja inteira**: primeiro prato, segundo, acompanhamento, pão, iogurte, fruta e água, cada um no seu prato. 38 fotos são de um item só (pote de vidro, embalagem).
- **Rótulo:** 870 itens, cada um com nome em inglês e **peso em balança de cozinha na hora da coleta, com a tara do prato descontada**. Conferi 8 bandejas contra a lista: os itens batem com o que se vê.
- **Lista** (`externo_mensa.csv`, uma linha por foto): massa sem as bebidas (água, café e suco vão em `bebida_g`), massa por grupo e frações por fonte; proteína só fraca, em colunas à parte. Massa mediana de 602 g; 101 bandejas acima de 600 g. `grupo` = participante; as 34 fotos de um participante ficam como `teste_ext` pela regra de reserva de 12.5.
- **Proteína fraca:** mediana de 28,5 g por bandeja, mas a maior parte vem de massa, pão e iogurte. **Só 39 bandejas têm mais de 25 g vindos de fontes de proteína, e 17 têm um único item acima de 25 g** (85 têm um item acima de 15 g). É cardápio de refeitório italiano: muito primeiro prato de massa, segundo prato pequeno (omelete, peixe, hambúrguer de legumes).
- **O mapa de nomes** (`eval/dados_ext/mensa_componentes.csv`, 314 nomes) sai de 87 regras ordenadas escritas em `mensa_monta.py`; cada linha guarda a regra que casou. Li a tabela inteira e corrigi dois casos. Confiança: A 35, B 171, C 108.
- **Ressalvas:** são bandejas com vários pratos, não um prato (`vista = zenital_bandeja`); iogurte e biscoito aparecem em embalagem fechada (somados em `embalado_g`); fruta inteira é pesada com casca; só seis fotógrafos; imagens pequenas.

## Reproduzir

```
S=<scratchpad>; K="PYTHONPATH=$S/kaggle_pkg KAGGLE_API_TOKEN=... python3 -m kaggle"
$K datasets list -s "<consulta>" --csv
$K datasets metadata <ref> -p <pasta>           # licença declarada
$K datasets files <ref> --csv --page-size 200   # listagem (paginar com --page-token)
$K datasets download nutritionverse/nutritionverse-real -f nutritionverse_dish_metadata3.csv -p <pasta>
python3 -I eval/dados_ext/manifestos_kaggle.py nvreal <raiz do dataset> <saida.csv>
# Flavoria: listagem do tar em fluxo (sem gravar) e lista de treino
curl -sL https://zenodo.org/api/records/5850856/files/dataset_images_and_csv.tar/content | tar -tvf - > listagem.txt
curl -sL -o dataset_csv.tar https://zenodo.org/api/records/5850856/files/dataset_csv.tar/content
python3 -I eval/dados_ext/flavoria_monta.py monta dataset_csv.tar     # grava eval/treino/kaggle/listas/externo_flavoria.csv
# Segunda varredura: LeFood-Set e SNAPMe
python3 -I eval/dados_ext/lefood_monta.py monta data_original.xlsx     # o xlsx vem da URL de lefood_arquivos.csv
curl -sL https://ndownloader.figshare.com/files/44532971 | tar -xzO --occurrence=1 snapme_db_09Dec2022/snapme_cs_db/master_SNAPME_linkfile.csv > master.csv
python3 -I eval/dados_ext/snapme_monta.py monta master.csv
# GT Mensa 2025 (benchmark de cantina)
R=https://raw.githubusercontent.com/Metabolic-Intelligence-Lab/vlm-food-benchmark/main
curl -sO $R/labels_flat.csv; curl -sO $R/GT_Mensa_2025/image_participants.csv
python3 -I eval/dados_ext/mensa_monta.py monta labels_flat.csv image_participants.csv
# Open Images: contar fotos de comida no treino (lê ~400 MB em fluxo, não grava)
curl -s https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv | <filtro por LabelName e Confidence=1>
```

Scripts de apoio, amostras e listagens desta sessão ficaram em `<scratchpad>/fontes_treino/` (fora do repositório).
