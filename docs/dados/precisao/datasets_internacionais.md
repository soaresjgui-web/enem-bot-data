# Datasets internacionais para TREINO do estimador de massa/proteína (2026-11-09)

Objetivo: ampliar os dados de treino do modelo que estima massa (g) e proteína (g) a partir de uma foto. Este levantamento não repete o trabalho do outro agente, que treina com as imagens overhead RGB/depth do Nutrition5k em `scratchpad/treino`.

**Critérios obrigatórios** (os mesmos do v2):
1. fotos reais de câmera, sem imagem sintética, renderizada ou gerada por IA;
2. rótulo **medido**: comida pesada em balança, ou ingredientes pesados em laboratório ou receita. Rótulo estimado por modelo ou autorrelatado não serve;
3. licença que permita pesquisa. Licença não comercial (NC) serve só para P&D interno e avaliação, nunca para o modelo do produto, que é comercial.

As imagens e os manifestos ficam **só no scratchpad** (`scratchpad/dados_ext/<nome>/`), nunca no repositório. Os scripts estão em `eval/dados_ext/`.

## Resumo

| Dataset | Foto real? | Rótulo medido? | Imagens / pratos | Campos do rótulo | Licença | Uso comercial? | Status | Hosts necessários |
|---|---|---|---|---|---|---|---|---|
| **Nutrition5k side_angles** (Google, CVPR 2021) | Sim. Vídeos de 4 câmeras laterais (Raspberry Pi) com o prato girando | Sim. Pesagem incremental por ingrediente, com nutrientes da USDA | **NN_IMG quadros / NN_DISH pratos** (8 por prato) | massa total, kcal, proteína, gordura, carboidrato, gramas por ingrediente | CC BY 4.0 | **Sim**, com atribuição | **Baixado** (quadros extraídos) | `storage.googleapis.com` (liberado) |
| **ECUSTFD** (Liang & Li, 2017) | Sim. Fotos de celular, vista superior e lateral, com moeda de 25 mm como escala | Sim. Peso em balança e volume por deslocamento de água, por objeto | 2.978 imagens (2.968 no manifesto) / 271 porções / 19 tipos | massa (g), volume (ml), caixas VOC; **sem proteína** | **Não declarada.** O README diz "free public dataset" e pede citação | **Incerto**: pedir permissão aos autores antes de usar no produto | **Baixado** (`github.com`) | `github.com` (liberado) |
| Food Portion Benchmark (ISSAI, 2025) | Sim. RealSense D455 e celulares, em laboratório | Sim. Peso em gramas por item (balança) | 14.083 imagens, 138 classes (Ásia Central) | caixa YOLO + peso por item; sem proteína | **CC BY-NC 4.0** | **Não** (só P&D interno) | **Bloqueado**: os rótulos (.txt) baixam, as imagens não | `us.aws.cdn.hf.co` (CDN Xet do HF; também `cas-server.xethub.hf.co` e `cdn-lfs-us-1.hf.co`) |
| NutritionVerse-Real (U. Waterloo, 2024) | Sim, segundo o artigo (celular, vários ângulos) | Sim, segundo o artigo (ingredientes pesados) | 889 imagens / 251 pratos | massa e nutrientes por ingrediente | não verificada (página do Kaggle) | a verificar | **Bloqueado** | `www.kaggle.com` |
| FLIC (UNIMIB/Parma, *Appl. Sci.* 2026) | Sim. Bandejas de refeitório universitário italiano, vista zenital a 65 cm, 640x480 | Sim. Cada prato pesado antes e depois da refeição | 401 pares cheio/sobra (~1.160 itens) | gramas por prato (primeiro, segundo, acompanhamento, pão), máscaras | CC BY 4.0 (segundo terceiros; não verificado) | provável, com atribuição | **Bloqueado**: o host do download não foi localizado | `doi.org`, `www.mdpi.com` (e provavelmente `zenodo.org` ou o servidor da UNIMIB) |
| MetaFood3D / SimpleFood45 (Purdue) | Em parte: objetos reais escaneados, com vídeo real. MetaFood3D também traz vistas **renderizadas** | Sim (peso por objeto) | 637 objetos / 45 alimentos | peso, energia, malha 3D | a verificar | a verificar | **Bloqueado** | `lorenz.ecn.purdue.edu` |
| MADiMa 2023 (U. Bern) | Sim nas 10 "Real Food". As 10 "Plastic Food" são comida de plástico | **Não para massa**: o gabarito é volume, por nuvem de pontos | 20 refeições | volume por item, profundidade (RealSense, LiDAR) | Apache-2.0 | sim | **Rejeitado** para massa; também tem acesso restrito (*gated*) no HF | `huggingface.co` com login + aceite |
| Diabetes60 (Christ et al., ICCV-W 2017) | Sim (Kinect V2, RGB-D) | A verificar: carboidrato por prato caseiro; não está claro se pesado | 60 pratos | carboidrato (BE) | não declarada | a verificar | **Bloqueado** | `drive.google.com`, `drive.usercontent.google.com` |
| UNIMIB2016 | Sim (bandejas de refeitório) | **Não**: só classe e segmentação, sem peso | 1.027 bandejas | classes, polígonos | — | — | **Rejeitado** (sem massa) | — |
| NutritionVerse-Synth / NutritionVerse-3D | **Não**: cenas renderizadas (Omniverse) a partir de modelos 3D | — | — | — | — | — | **Rejeitado** (sintético) | — |
| ChinaMartFood-109 | — | — | — | — | — | — | **Não encontrado** em GitHub nem HF; não verificado | — |
| AI Hub (Coreia; imagens de comida com peso) | Não verificado | Não verificado | — | — | Exige cadastro coreano | — | **Não verificado** | `aihub.or.kr` |
| Comida brasileira / latino-americana | MyFood (UFPR), "Brazilian food images" (Mendeley): fotos reais | **Não**: sem peso | — | — | — | — | **Rejeitado** (sem massa). Nenhum dataset latino com peso foi encontrado | — |
| SNAPMe, MM-Food-100K, CalorieBench, espelhos do Nutrition5k no HF | ver `coleta.md` | autorrelato ou estimado por LLM | — | — | — | — | **Rejeitados** (já registrados em `coleta.md`) | — |

**Total utilizável para treino comercial:** NN_IMG quadros do Nutrition5k side (massa **e** proteína). Disso, NN_TRAIN_IMG imagens/NN_TRAIN_DISH pratos estão fora do holdout de avaliação. Há ainda 2.968 imagens do ECUSTFD, só com massa e com a licença a confirmar.

## Nutrition5k side_angles: detalhes

- Bucket: `gs://nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles/dish_*/camera_{A,B,C,D}.h264`. São 4.792 pratos e 18.486 vídeos, **178,5 GiB** no total. Cada vídeo é H.264 cru, 1920x1080, ~141 quadros, com keyframe a cada 60 quadros, e mostra o prato girando numa base.
- Não baixamos os vídeos inteiros. `eval/dados_ext/n5k_side_frames.py` usa **HTTP Range** (`curl -r`) no endpoint público e, por câmera, extrai:
  - `ini`: o último quadro decodificável dos primeiros 600 KB (≈ quadro 8);
  - `fim`: o keyframe (IDR) mais próximo do fim (≈ quadro 120, com o prato já girado). O script acha o SPS (NAL tipo 7, byte `0x27`, e não `0x67`) nos últimos 30% do arquivo.

  São 8 imagens por prato (4 câmeras × 2 momentos), em 960x540 JPEG. Foram ~6,7 MB baixados por vídeo, contra ~10 MB do vídeo inteiro, e NN_GB GB de disco.
- Rótulos: `dish_metadata_cafe{1,2}.csv`. No café 2 os totais vêm **zerados** no CSV. Reconstruí proteína e kcal pela soma dos ingredientes e completei os nomes vazios com o `ingredients_metadata.csv`, como no v2. Essas linhas levam a nota `totais_reconstruidos_pelos_ingredientes`. Pratos com ingrediente `deprecated`, `plate only` ou sem nome levam a nota `ingrediente_deprecated_ou_sem_nome`, para filtrar se for o caso.
- Split: o oficial `rgb_{train,test}_ids.txt`, que cobre 4.606 dos 4.792 pratos com vídeo. O resto fica com split vazio.
- **eval_holdout=1** para os 450 `dish_id` que aparecem em `gabarito_pratos*.csv`, `gabarito*.csv`, `n5k_v2_amostra.csv` e `coleta_n5k*.py`. São os nossos pratos de avaliação v1 e v2: **nunca treinar com eles**. Total no manifesto: NN_HOLD_IMG imagens / NN_HOLD_DISH pratos.
- **Atenção ao vazamento:** os quadros laterais são dos mesmos pratos das imagens overhead que o outro agente usa. Qualquer split treino/val precisa ser **por `dish_id`**, igual nas duas fontes.
- QA visual: um mosaico de 24 quadros sorteados mostrou o prato visível e nítido em todos. Em alguns o prato aparece cortado na borda.

## ECUSTFD: detalhes

- Clonado de `github.com/Liang-yc/ECUSTFD-resized-` (versão redimensionada, < 1000 px, com anotações). `density.xls` traz `id, tipo, volume, peso(g)` por objeto. A unidade aparece como "mm^3", mas os valores são de ml. Nas imagens `mix*` há 2 alimentos, e o manifesto soma as massas e lista os dois.
- 19 tipos: maçã, banana, pão, bun, doughnut, ovo, fried dough twist, uva, limão, lichia, manga, mooncake, laranja, pera, pêssego, ameixa, kiwi, sachima e tomate. São quase todos alimentos únicos e inteiros, então serve para escala e porção, mas pouco para prato composto. **Não traz proteína.** O espelho `ai5labsOfficial/ecustfd` do HF calcula kcal com tabela da USDA e declara CC BY 4.0, mas essa licença foi dada por terceiros; vale a original, que não declara licença.
- O split oficial (`ImageSets/Main`) é **por imagem**: a mesma porção cai em treino e teste. Para treino, refazer o split por `dish_id`.
- `mix002T(2)` e `mix005S(4)` não têm a moeda (nota `sem_moeda`).

## Food Portion Benchmark (ISSAI): por que ficou bloqueado

- Os rótulos YOLO, com a 6ª coluna em gramas, são arquivos git comuns e baixam por `huggingface.co`. Numa amostra de 300 rótulos, ~95% das imagens têm peso em todos os itens, **inclusive no split de teste**: o card diz que o teste é oculto, mas os pesos estão publicados.
- As imagens estão no armazenamento Xet do HF. `resolve/main/...` redireciona para `us.aws.cdn.hf.co`, que o proxy recusa (o mesmo vale para `cas-server.xethub.hf.co` e `cdn-lfs-us-1.hf.co`).
- Os nomes `*.rf.<hash>.jpg` indicam exportação do Roboflow. No treino, ~400 imagens aparecem em 2 a 5 cópias, provavelmente aumentação clássica (flip/rotação, não IA). Ao baixar, deduplicar pelo nome antes do `.rf.`.
- Licença CC BY-NC 4.0: **não usar no modelo do produto**. Serve como avaliação extra de porção por peso.
- `manifestos.py` já tem a função `fpb`, pronta para quando as imagens puderem ser baixadas (`hf download issai/Food_Portion_Benchmark --repo-type dataset --include "FPB_Dataset/*"`).

## Log de rejeição (resumo)

| Dataset | Motivo |
|---|---|
| MADiMa 2023 | O gabarito é volume (nuvem de pontos), não massa. Metade é comida de plástico. Acesso *gated* |
| UNIMIB2016 | Sem peso nem nutrientes |
| NutritionVerse-Synth / -3D | Imagens renderizadas (sintéticas) |
| MyFood (UFPR), Brazilian food images (Mendeley) | Sem peso |
| SNAPMe | Porção autorrelatada (ASA24) |
| MM-Food-100K, CalorieBench-80K, CalorieData etc. | Rótulo estimado por LLM ou de proveniência incerta |
| Espelhos do Nutrition5k no HF | Duplicam a fonte original, e alguns trazem nomes gerados pelo Gemini |
| Repositórios de projetos (Food-Volume-Estimation, FoodMassEstimation etc.) | Código sem dataset próprio publicado, ou com dados de poucas refeições e sem licença |

## Hosts a liberar (em ordem de valor)

1. `us.aws.cdn.hf.co` (e `cas-server.xethub.hf.co`, `cdn-lfs-us-1.hf.co`): imagens do FPB, ~14 mil imagens com peso (só P&D, NC). Também libera os arquivos LFS de qualquer dataset do HF.
2. `www.kaggle.com`: NutritionVerse-Real. Os downloads do Kaggle redirecionam para `storage.googleapis.com`, que já está liberado.
3. `doi.org`, `www.mdpi.com`, `zenodo.org`: localizar e baixar o FLIC (CC BY 4.0, refeitório com pesagem antes e depois).
4. `lorenz.ecn.purdue.edu`: MetaFood3D e SimpleFood45.
5. `drive.google.com`, `drive.usercontent.google.com`: Diabetes60.

## Reproduzir

```
S=<scratchpad>/dados_ext
gsutil ls -l 'gs://nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles/**' > $S/n5k_side/sizes.txt
python3 -I eval/dados_ext/n5k_side_frames.py $S/n5k_side/sizes.txt $S/n5k_side/frames --workers 32   # ~80 min, ~120 GB trafegados
gsutil cp gs://nutrition5k_dataset/nutrition5k_dataset/metadata/*.csv gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/rgb_*_ids.txt $S/n5k_side/meta/
git clone --depth 1 https://github.com/Liang-yc/ECUSTFD-resized-.git $S/ecustfd/raw
python3 -I eval/dados_ext/manifestos.py $S docs/dados/precisao n5k_side ecustfd
```
