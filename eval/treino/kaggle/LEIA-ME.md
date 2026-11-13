# Treino em GPU (Kaggle ou Colab) — seções 8, 10 a 14, uma única foto de cima

Roda, sem interação, os treinos em GPU do modelo de proteína pela foto, descritos em `docs/dados/precisao/resultado_modelo_treinado.md`. A inferência usa sempre **uma** foto RGB zenital. O critério é o do produto: proteína dentro de ±20% ou ±5 g.

**Estado (12/11/2026):**

- **Execução 1 (seção 10)** rodou no Kaggle: 2× T4, 15 configurações em ~5,7 h; a paridade com a CPU conferiu (RMSE 0,2550 contra 0,2551).
- **Execução 2 (seção 11)** rodou no Kaggle: 11 configurações em 4,5 h de relógio (torch 2.11.0+cu130, timm 1.0.30), com previsões no teste2, arquivos de servir e `servico.json`; saída de 1,27 GB.
- **Execução 3 (seções 12.3 a 12.5)** rodou no Kaggle: 23 configurações em 3,6 h; resultados em 12.6 e 12.7.
- **Execução 4 (seção 13)** rodou no Kaggle: as seis bases baixaram, a paridade com o D1 saiu exata e nada foi cortado; resultados em 13.11.
- **Execução 5 (seção 14), etapa 0 (14.3): pronta para o push e ainda não rodou em GPU.** Kernel separado, `nutrelle-deteccao-r5` (`deteccao_r5/`), descrito logo abaixo. A execução principal da quinta execução só roda depois da conferência a olho das folhas de contato.

Para baixar a saída é preciso acesso a `www.kaggleusercontent.com`.

## Quinta execução, etapa 0: a caixa da refeição em cada foto (14.3)

`deteccao_r5.py`, num kernel próprio (`deteccao_r5/nutrelle_deteccao_r5.ipynb`, slug `nutrelle-deteccao-r5`). Não treina nada e não lê rótulo nenhum.

- **Fotos:** os 2.281 pratos de treino_v2 (o teste2 está encerrado e não entra) e todas as fotos das bases externas que a quarta execução usava, inclusive as das partes de registro: Flavoria 1.848, LeFood 811, Mensa 200, FPB de cima 2.418, OmniFood8K 15.741 (três quadros por cena de treino e o `camera_0` das de registro), SNAPMe 1.163. Total: 24.462. As bases baixam como na quarta execução (mesmas funções, prazo de 40 min), em paralelo com a detecção.
- **Detector:** OWLv2 base (`google/owlv2-base-patch16-ensemble`, revisão `cfd3195ba4ea`, sha256 dos pesos conferido; Apache 2.0) em meia precisão, um processo por GPU, imagem a 960 px, as sete consultas de 14.3; caixas com nota ≥ 0,15 e área ≥ 2%, NMS a 0,5, **união** das que sobram. O pré-processamento é feito na GPU, igual ao do `Owlv2Processor` (em 12 fotos reais de seis bases a caixa saiu com IoU de 0,93 a 1,00 contra a do processador). Sem caixa: a maior elipse do OpenCV (área de 10% a 95%, razão de eixos ≥ 0,5). Sem elipse: `sem_prato`, foto inteira.
- **Janela do guia:** 4:3, centrada na caixa, altura = maior lado ÷ 0,80, cor mediana da borda onde sai da foto; `sem_prato` = o menor 4:3 que contém a foto. A foto é tomada como o modelo a vê: orientação do EXIF aplicada e retrato girado para paisagem.
- **Nutrition5k:** IoU com a caixa da profundidade de 8.7 (`zen_prato.py`): a gravada naquela época (`listas/caixas_profundidade_n5k.csv`, 1.874 dos 2.281 pratos) e a refeita no kernel, com a mesma conta, para todos (`depth_raw.png` do mesmo bucket). Em 9 pratos do teste a refeita repetiu a gravada (IoU 1,00).
- **Saída:** `caixas_r5_<base>.csv` (id da foto, arquivo, sha256 da foto, tamanho, `girada`, método, caixa, nota, nº de caixas, janela, versão do detector; e, para diagnóstico, a caixa só das consultas de prato e tigela, a elipse e até 40 caixas brutas com a consulta), `folha_r5_<base>_<1|2>.jpg` (40 fotos sorteadas com semente 0: a foto com a caixa em vermelho e a janela em amarelo, e o recorte ao lado) com `folha_r5_<base>.csv` (posição → foto, coluna `veredito` vazia para a conferência), as mesmas 40 fotos em `folha_r5_<base>_so_prato_<k>.jpg`, `resumo_r5.json` (por base: métodos, taxa de `sem_prato`, IoU com a profundidade, diagnóstico), `manifesto_r5.json` (detector, licença, parâmetros) e o registro.
- **Refazer a regra na CPU, sem GPU nem fotos:** `python3 deteccao_r5.py --refaz-regra <saída da etapa 0> --saida <pasta> [--consultas 0,1,5,6]` regrava as tabelas com outra seleção de consultas, a partir das caixas brutas e da elipse gravadas. Com as sete consultas, repete a tabela original.
- **Teste:** `python3 deteccao_r5.py --autoteste`; `python3 deteccao_r5.py --smoke 2 --cpu --workers 2 --listas <listas com as tabelas de divisão> --externo-imagens <pasta com uma subpasta por base> --saida <pasta> --cache <cache>` (com o OWLv2 de verdade; `--sem-detector` testa só a reserva).
- **Push:** `NUTRELLE_SCRATCH=<scratchpad> python3 monta_pacote.py push <pasta fora do repositório> --kernel r5a --usuario <você>` e `python3 monta_pacote.py confere <pasta> --kernel r5a`. O notebook do repositório (`monta_pacote.py notebook --kernel r5a`) leva só as bases liberadas. Tempo estimado: ~20 min de detecção nas duas T4 depois que as bases chegam (~40 min de GPU, como 14.3), ~1 h de relógio com os downloads; orçamento de 1,5 h.

**Atenção, visto no teste em CPU antes de rodar:** com a regra de 14.3 (união de todas as consultas), a consulta "a food tray" pega a **plataforma da balança** no Nutrition5k (nota 0,26 a 0,31, 80% da foto) e a **bandeja marrom** no FPB; a união vira quase a foto inteira e o recorte não enquadra o prato. Em 9 de 10 fotos do Nutrition5k e em 12 de 12 do FPB, a caixa da regra difere da caixa só de prato e tigela (IoU < 0,7). As folhas `_so_prato` mostram a alternativa nas mesmas 40 fotos, e `--refaz-regra` regrava a tabela na CPU se a conferência a olho preferir ela. A regra não foi mudada aqui: é pré-registrada.

## O que a execução 4 faz

Seção 13 de `docs/dados/precisao/resultado_modelo_treinado.md`: o D1 da execução 3 (So400m congelado, cabeça ridge para a massa, cabeça de frações para a densidade) com mais bases de treino, cada uma só na cabeça que ela supervisiona, mais três ajustes finos pequenos e um ajuste fino parcial do So400m. **Nada é decidido dentro do kernel:** ele só grava previsões e métricas de CV das partes de treino; a regra de 13.9 roda depois, na CPU.

| Prioridade | Apelido | Configuração | O que é | GPU | Min (estimativa) |
|---|---|---|---|---|---|
| 1 | | `gpu4 feat so400m n5k` e `gpu4 feat so400m <base>` | features do So400m: Nutrition5k com 8 orientações; bases de uma foto com 2 (original e 180°); OmniFood8K e SNAPMe com 1 | A | 29 + 39 |
| 1 | R0 | `gpu4 ridge so400m` | cabeça ridge, só treino_v2: réplica da do D1 | A | 1 |
| 1 | H0 | `gpu4 fracoes so400m` | cabeça de frações, só treino_v2: réplica da do D1 | A | 5 |
| 1 | R1 a R4 | `gpu4 ridge so400m +ok`, `+ok+fpb`, `+omni`, `+tudo` | R0 com a **massa total** de Flavoria, LeFood e Mensa; + FPB; só OmniFood8K; as cinco | A | 2, 2, 2, 3 |
| 1 | H1 | `gpu4 fracoes so400m +omni` | H0 + OmniFood8K com os alvos de 13.4 | A | 8 |
| 2 | F1 | `gpu4 ft siglip2_b16_224 r2 fracoes` | receita r2 + cabeça de frações com a perda corrigida (α, 13.7) | B | 64 |
| 3 | P1 | `gpu4 ftparcial so400m` | ajuste fino parcial: 4 últimos blocos, norma, agrupador e cabeças; 6 épocas | A | 150 |
| 4 | F3 | `gpu4 ft siglip2_b16_224 r2 +tudo` | receita do U0 sem mudança + as cinco bases | B | 96 |
| 5 | H2 | `gpu4 fracoes so400m +omni+ff` | H1 + frações por fonte das bases de massa (peso 0,5) | A | 10 |
| 6 | F2 | `gpu4 ft siglip2_b16_224 r2 fracoes faixa` | F1 + peso por faixa | B | 64 |

- **Bases externas** (uma lista e uma tabela de divisão de `eval/treino/externo_r4_divide.py` por base). Todas baixam numa linha à parte desde o minuto zero, com novas tentativas, retomada e conferência, e prazo de 40 min; a que não chegar fica de fora, as configurações que dependem dela são canceladas e o manifesto diz por quê (`dominios.<base>.situacao`):

  | Base | Licença, uso | De onde | Conferência |
  |---|---|---|---|
  | Flavoria | CC BY 4.0, comercial | Zenodo 5850856, um `.tar` de 2,58 GB | tamanho e md5 |
  | LeFood-Set | CC BY 4.0, comercial | Mendeley Data, foto a foto (coluna `url`) | imagem íntegra |
  | GT Mensa | CC BY 4.0, comercial | GitHub (raw), foto a foto | imagem íntegra |
  | FPB, só de cima | CC BY-NC 4.0 | Hugging Face, foto a foto (`resolve/main/<arquivo>`) | tamanho e sha256 de cada foto |
  | OmniFood8K | CC BY-NC 4.0 | espelho anexado ao kernel (árvore ou o zip); senão Google Drive (um zip de 2,73 GB), com a confirmação de arquivo grande | tamanho e CRC de cada foto |
  | SNAPMe | CC BY-SA 4.0, autores pedem para não treinar | Figshare 44532971, `.tar.gz` de 2,03 GB | tamanho e integridade do gzip |

  O **SNAPMe nunca treina**: só tem as fotos previstas por todas as configurações, cruas, para a avaliação fraca feita depois na CPU.
- **Partes de registro fechadas (13.8).** Ao carregar cada base, o kernel apaga as colunas de alvo de toda linha que não é `treino` (`registro`, `registro_misto`, `avaliacao_fraca`); o notebook já vai com essas colunas vazias. O kernel não pontua nenhuma foto de registro nem o teste2: grava as previsões cruas e mais nada. O `--autoteste` confere isso.
- **Como cada base supervisiona (13.4):** bases de massa só na cabeça de massa (R1 a R4) e no F3; o OmniFood8K na cabeça de densidade (H1, H2) com massa 1, nutrição 1 (0,5 com osso ou casca, 0 sem nutrição), grupos e frações 0,5; peso igual por grupo dentro da base, soma `min(0,5; grupos ÷ 2.281)` × a do Nutrition5k, todos reescalados se a soma passar de 1; viés por domínio retirado dos alvos extensivos (não da densidade nem das frações); fotos de outra base sempre encaixadas num quadro 4:3 com a cor mediana da borda, retrato girado para paisagem.
- **Validação cruzada conjunta:** cada foto de treino de uma base externa valida no fold dela (`fold_ext`) e não treina nos `folds_proibidos` (no OmniFood8K, os folds de todos os pratos da cena).
- **Conferência de paridade (13.7):** assim que R0 e H0 terminam, o kernel compara as previsões fora-do-fold delas com as do D1 da execução 3 (`listas/ref_d1_r3.csv`): correlação de 0,999 ou mais na massa (1 e 8 orientações) e nas duas densidades, e o critério de CV do D1 em 71,3 ± 0,3. Se falhar, `manifesto.json` marca tudo como não confiável (`confianca`, `confiavel` em cada configuração) e a ingestão não grava nada no banco.
- **Ajustes finos com frações (F1, F2, P1):** perda = `L_rot` + α × (`L_frac` + `L_dens`), com α = min(1, ‖∂L_rot/∂f‖ ÷ ‖∂(L_frac + L_dens)/∂f‖) recalculado a cada passo e suavizado em 20 passos; logits das frações a partir de zero, com passo 10×; backbone parado na primeira época. O registro grava por época `L_rot`, `L_frac`, `L_dens` e `alfa`, e o `log__` traz a aceitação de 13.7 (RMSE de log-massa da última época não maior que o da época 0, média dos folds).
- **O que cada configuração grava:** `pred__<config>.npz` com `oof`, `teste2` (1, 4 e 8 orientações), frações quando há, e para cada base `extcv__<base>` (fotos de treino, pelo modelo do fold em que validam) e `extreg__<base>` (registro, registro misto e avaliação fraca, pelo modelo final), cruas, com os ids, o fold e o papel; `vies__<base>` por fold; `procedencia` (13.8: `treinado_com` e `so_comercial_ok`); hashes de todas as listas e tabelas. Para servir: cabeças finais com conferência de remontagem (inclusive as de frações), pesos dos ajustes finos (o P1 só com os tensores treinados) e `servico.json` (T4 e CPU, 1 orientação: o So400m com as duas cabeças juntas, o P1 e os três ajustes finos).

**Tempo e tamanho (estimativa, 2 × T4, com a fila simulada pelas mesmas funções do escalonador):** GPU A ~241 min (features 68, cabeças 31, P1 150 incluindo ~30 min de previsões nas 27,5 mil fotos externas), GPU B ~224 min (F1, F3, F2; ela também pega trabalho da A quando fica sem o seu), mais ~12 min de serviço: **~4,2 h de relógio**, dentro de `ORCAMENTO_H = 4.5`. Com 5% a mais de tempo que o estimado, o P1 não cabe e é cancelado pela projeção de tempo, como 13.6 prevê; sem o OmniFood8K, R3, R4, H1, H2 e F3 caem e a execução termina em ~3,3 h. Saída: ~0,9 GB (três `pesos__` de ~186 MB, o do P1 com ~150 MB, as partes das cabeças de frações com os pesos de cada fold). Cache em `/kaggle/temp`: ~31 GB (as fotos decodificadas do OmniFood8K são 14,5 GB).

### Dois notebooks: o do repositório e o do push

- `nutrelle_treino_gpu.ipynb` (no repositório, `python3 monta_pacote.py notebook`): só as bases **liberadas** (Flavoria, LeFood, Mensa). Roda sozinho: OmniFood8K, FPB e SNAPMe ficam de fora e o manifesto diz por quê. É a quarta execução "só com dados que podem ser refeitos comercialmente".
- Notebook completo (`python3 monta_pacote.py push <pasta fora do repositório> --usuario <seu usuário> [--espelho-omnifood <usuário>/<dataset>]`): as seis bases. As listas e tabelas de divisão das bases fechadas são ignoradas pelo git e lidas de `listas/`, de `--divisoes <pasta>` ou da variável `NUTRELLE_SCRATCH` (e das subpastas `listas_nc/` e `listas_r4/` dela). **Não commitar essa pasta.**
- `python3 monta_pacote.py confere [<pasta do push>]` falha se o notebook do repositório estiver desatualizado ou embutir qualquer arquivo ignorado pelo git, id (`omn_`, `fpb_`, `snp_`) ou licença de uma base fechada; com a pasta, confere também o notebook completo. Os dois ficam abaixo de 1 MB (listas enxutas, comprimidas com lzma).
- **Espelho do OmniFood8K (recomendado):** o Google Drive corta por cota. Crie um dataset **privado** do Kaggle com o arquivo `0-OminiFood8k.zip` (2.731.585.789 bytes) do Drive, como está (o Kaggle pode extraí-lo; o kernel aceita a árvore extraída `0-OminiFood8k/8036/<cena>/camera_<k>.jpg` em qualquer nível até 4 pastas abaixo de `/kaggle/input`, ou o próprio zip), e passe `--espelho-omnifood <usuário>/<slug>` ao `push`, que o põe em `dataset_sources`. Sem o espelho, o kernel tenta o Drive.

**O que olhar nas primeiras linhas do registro** (`kaggle kernels logs <usuário>/nutrelle-treino-gpu-r4`):

1. `nutrelle-treino-gpu/4 | rodada 4: 25 configurações | treino 2281 pratos, avaliação teste2 829 | listas 460c298cb8e57fb0, teste2 4aad94d89dc2440e | orçamento 4.5 h`.
2. Uma linha por base: `base externa omnifood (13.3): {'fotos': 15741, 'treino': 11586, 'grupos_treino': 3862, 'registro': 1928, 'registro_misto': 2227} | CC BY-NC 4.0, nao_comercial | lista fd6caa4415e4e652 (versão enxuta no pacote) …`; nas outras: flavoria 1848/1493/1001/355, lefood 811/661/433/150, mensa 200/166/166/34, fpb 2418/2106/747/312, snapme 1163 de avaliação fraca. `ATENÇÃO: a base externa … fica de FORA` diz o que não entrou.
3. `[cuda:1] começa gpu4 ft siglip2_b16_224 r2 fracoes | fold0` logo no começo (a GPU B não espera as features).
4. Nos primeiros 40 min: `base externa <base>: … baixados e conferidos` e `cache das fotos de <base>: … fotos`; o OmniFood8K diz se veio do espelho ou do Drive.
5. Perto de 35 min: `== PARIDADE com o D1 da execução 3 (13.7): OK | correlação fora-do-fold: massa 1.0000 …, critério de CV do D1 71.29%`. Se sair `FALHOU`, nada desta execução deve ser lido.
6. No F1, época 1 com `so_cabeca: True` e `alfa` perto de 1; nas seguintes, `alfa` abaixo de 1 se o termo das frações dominar; `L_rot` não pode subir.
7. Nas linhas `== <config>: CV no treino`, a massa nas fotos de treino das bases externas (com a constante do domínio) e, nos ajustes finos, `aceitação (13.7): … PASSOU/NÃO PASSOU`.

## O que a execução 3 fez

Treina com `listas/treino_v2.csv` (2.281 pratos, 646 cadeias, folds novos, frações da massa por fonte de proteína) e prevê **só o teste2** (829 pratos, sem gabarito). As fotos do teste2 passam apenas pelos modelos finais, para gerar previsões: não entram em treino, destilação, nem em nada sem rótulo. Os 450 pratos da avaliação antiga não são previstos (parte deles agora é treino).

As 13 configurações de 12.3, na ordem de prioridade (`python3 treino_gpu.py --lista`):

| Nº | Configuração | O que é | Min de GPU (estimativa) |
|---|---|---|---|
| 1–5 | `gpu3 ridge <backbone> o8` | SigLIP 2 So400m, L/16, B/16 a 384×512 e DINOv2 L/14, B/14 a 336×448, congelados, 8 orientações, ridge com alfa por alvo | 31, 20, 20, 8, 8 |
| 6–8 | `gpu3 fracoes so400m`, `l16`, `b16_384x512` | Cabeça de frações por fonte (MLP de 512, a receita `soft-mlp fonte-l2 +perda-dens peso-faixa` da CPU) sobre as features de três deles | 3 cada |
| 9 | `gpu3 professor` | Massa = média em log das cinco massas ridge; densidade = média em log das três densidades por frações e da cabeça de densidade do ridge So400m; proteína = massa × densidade. Só combina previsões | 1 |
| 10 | `gpu3 unico so400m mlp destilado` (U1) | Um backbone congelado + a cabeça de frações, com destilação do professor | 2 |
| 11 | `gpu3 ft siglip2_b16_224 r2 destilado` (U4) | Ajuste fino a 224 px: cabeça de 22 + 10 saídas, perdas de frações e de densidade composta, peso por faixa de proteína, destilação | 64 |
| 12 | `gpu3 ft siglip2_b16_384x512 r1 destilado` (U3) | O mesmo a 384×512 | 151 |
| 13 | `gpu3 ft siglip2_b16_224 r2` (U0) | A receita r2 da execução 2 sem mudança, só com o treino novo | 64 |

- **Destilação com alvos aninhados.** O aluno treinado sem o fold k recebe, para um prato do fold j, a previsão de um professor treinado sem os folds j e k (10 pares de folds; a conta extra é feita dentro das etapas ridge e frações). Assim a CV do aluno nunca enxerga rótulos do fold k. O `--autoteste` troca todos os rótulos de um fold e exige que os alvos e as previsões desse aluno não mudem (com e sem a base externa); os alvos ficam em `pred__gpu3_professor.npz` (`alvos_oof`, `alvos_aninhados`).
- **Toda previsão sai com 1, 4 e 8 orientações** (chaves com sufixo `_o1`, `_o4` e sem sufixo), fora-do-fold e no teste2; onde há cabeça de frações saem também `*_ff` (frações), `*_dens_ff` (densidade pelas frações) e `dens_fonte` [6 × 10] (folds 0 a 4 e modelo final).
- **Nada é escolhido no Kaggle** e nenhum gabarito de avaliação é lido. A regra de 12.3 (rota, orientações, C1, C2) é aplicada na CPU, a partir dos arquivos fora-do-fold. O registro mostra o critério de CV de cada rota só para acompanhar.
- **Paridade com a execução 2:** cada `gpu3 ridge` treina também uma réplica do ridge da execução 2 (lista e folds antigos) sobre as features desta execução e grava as previsões dela no teste2 (`r2_teste2`). A ingestão compara com a execução 2, sem gabarito.
- **Escalonador:** dependências entre configurações (frações depende do ridge do mesmo backbone, professor depende de 1 a 8, alunos dependem do professor), ordem estrita nos alunos e simulação do orçamento: a configuração que não cabe inteira no tempo que resta não começa. Ordem de corte: U0, depois U3.

### Base externa: FlavoriaFoodWeight1700 (seção 12.5 e adendo)

O notebook liga a base (`EXTERNO = 'flavoria'` na primeira célula; `''` volta à execução de 12.3 pura) e embute a lista `listas/externo_flavoria.csv`, como as outras. O que o kernel faz com ela:

1. **Download em paralelo desde o início:** `dataset_images_and_csv.tar` (2,58 GB, Zenodo 5850856) numa linha à parte, com novas tentativas, retomada por `Range`, tamanho e md5 conferidos, e prazo de 45 min. Enquanto isso as GPUs extraem as features do Nutrition5k. **Se a lista não passar nas conferências ou o download falhar, a execução segue como em 12.3** e o manifesto diz por quê (`externo.situacao`, estado de `externo flavoria`).
2. **O que entra:** uma linha por foto. Ficam fora as fotos com `oclusao = 1` e, no `teste_ext`, os quadros que não são o preferido do prato. Na lista atual: 1.493 fotos de treino (1.001 pratos) e 227 pratos de `teste_ext` (coluna `split_ext`; nunca treinam, nunca entram em escolha). As pastas `train/val/test` do arquivo original não são usadas. As colunas `proteina_fraca_*` não são lidas.
3. **Como treina:** só massa total (peso 1), massa por grupo e frações por fonte (peso 0,5); proteína, densidade, kcal, gordura, carboidrato, densidade composta e destilação não se aplicam às fotos do Flavoria. Fold −1: entram em todo treino, nunca na validação. A soma dos pesos das fotos do Flavoria é metade da do Nutrition5k, com peso igual por foto. A foto quadrada vai para o centro de um quadro 4:3 com a cor mediana da borda nas laterais. De cada alvo de massa do Flavoria sai o viés de domínio (média do Flavoria menos média do Nutrition5k no treino do fold); ele volta só nas previsões para fotos do Flavoria.
4. **Ablação e decisão:** as configurações 1 a 10 rodam também na versão `gpu3 … +flavoria`. Quando os dois professores ficam prontos, `decisao flavoria` aplica a regra de 12.5 às previsões fora-do-fold deles (adota se o critério de CV sobe 1,0 ponto ou mais; ou se não cai mais que 0,3 e a média das duas faixas acima de 25 g sobe 2,0 ou mais), grava `decisao_flavoria.json` e a linha `== DECISÃO` no registro. **U4 e U3 rodam uma vez, na versão indicada** (`… destilado +flavoria` se adotada); U0 roda sempre só com o treino_v2. As duplicatas não são cortadas antes dos ajustes finos.
5. **Relato secundário** em `relato_externo.json` e no fim do registro: MAPE e viés mediano da massa no `teste_ext`, no total e acima de 600 g, do professor e do U1 com e sem a base e dos ajustes finos como foram treinados; para modelo sem a base, a previsão crua e a corrigida por uma constante (razão mediana no treino do Flavoria). **Não escolhe nada, não é o critério do produto e não é leitura do teste2.**

Tempo e tamanho estimados (2 × T4, tempos da execução 2 × 1,05 nos congelados e × 1,35 nos ajustes finos):

| | Sem a base (`EXTERNO = ''`) | Com o Flavoria |
|---|---|---|
| GPU até o professor e o U1 | ~1,7 h (50 min de relógio) | ~2,9 h (~1 h 25 de relógio, com a decisão) |
| Ajustes finos | 279 min de GPU | não adotada: 279 min; adotada: U4 80 + U3 189 (+ U0 64 se couber) |
| Relógio total, com a fase de serviço | ~3,3 a 3,5 h | ~3,9 h com `ORCAMENTO_H = 4.0`; nesse orçamento o U0 fica no limite (não adotada) ou de fora (adotada) |
| Saída | ~0,65 GB (três `pesos__*.pt` de ~186 MB) | ~0,7 GB |

Pela simulação da fila com essas estimativas e `ORCAMENTO_H = 4.0`: não adotada, os três ajustes finos cabem por 3 min (5% de lentidão corta o U0); adotada, cabem U4 e U3 e o U0 fica de fora, mas **5% de lentidão já corta o U3 com o Flavoria** (aí rodam U4 e U0 e a execução acaba em ~3,1 h). Com `ORCAMENTO_H = 4.5` cabe tudo nos dois desfechos. No ajuste fino com o Flavoria, cada época usa todas as fotos do Nutrition5k e um sorteio de fotos do Flavoria igual a 25% do número de pratos do treino (constante `FRACAO_EXT_EPOCA`), cada uma com peso 2, o que mantém o teto de metade do peso e limita o custo a +25%.

**O que olhar nas primeiras linhas do registro** (`kaggle kernels logs USUARIO/nutrelle-treino-gpu`):

1. Primeira linha: `rodada 3: 27 configurações | treino 2281 pratos, avaliação teste2 829 | listas 460c298cb8e57fb0, teste2 4aad94d89dc2440e | orçamento 4.0 h`. Sem a base são 13 configurações.
2. `torch …, timm … | dispositivos: ['Tesla T4, 15360 MiB', 'Tesla T4, 15360 MiB'] -> 2 processo(s)`: as mesmas versões da execução 2 (torch 2.11.0+cu130 e timm 1.0.30, na imagem fixada).
3. `base externa flavoria (12.5): lista 30d974fa0521e29e, {'n_lista': 1967, 'fora_por_oclusao': 119, … 'treino': 1493, … 'teste_ext': 227 …}, parte segurada por coluna split_ext`. Se aparecer `ATENÇÃO: a base externa flavoria fica de FORA`, a execução virou a de 12.3.
4. Em até ~25 min: `base externa flavoria: 2.40 GB baixados e conferidos (md5)`, `… fotos extraídas`, `cache das fotos externas: 1720 fotos`.
5. Em cada `gpu3 ridge`: a linha `== … CV no treino (n=2281) …` com `réplica da execução 2: RMSE …`. Os valores da execução 2 são 0,2190 (So400m), 0,2265 (L/16), 0,2283 (DINOv2 L/14), 0,2426 (B/16 a 384×512) e 0,2349 (DINOv2 B/14), em 1.684 pratos; diferença acima de ±0,001 indica que o ambiente mudou.
6. Perto de 1 h 25: `== DECISÃO sobre a base externa flavoria: …` com os dois critérios e as diferenças.
7. Nos ajustes finos destilados, a perda da primeira época fica perto de 8 (domina o termo das frações, que parte de frações iguais) e cai nas seguintes; `val_rmse_logm` sai a cada 5 épocas.
8. Linhas `… não cabe no orçamento …: fica de fora` mostram o que foi cortado.

## O que a execução 2 faz

1. **Roda de novo 11 das 15 configurações**, com as mesmas receitas, sementes, folds e ordem de lotes: as nove do pipeline congelado da seção 11.2 (cinco backbones congelados com ridge e quatro ajustes finos), mais `gpu ft siglip2_b16_224 r2` (candidato a rodar no aparelho) e `gpu ridge siglip2_b16_224 o8` (paridade).
2. **Prevê também o teste2**: os 829 pratos de `listas/aval2_ids.csv`, sem gabarito. Cada `pred__<config>.npz` ganha `teste2` [829 × 22] e `ids_teste2`, na ordem da lista. O treino não muda: os pratos novos só entram na previsão, e as features do treino e dos 450 são extraídas nos mesmos lotes da execução 1.
3. **Grava o que é preciso para servir** cada modelo: `pesos__<config>.pt` para os cinco ajustes finos (rede inteira, meia precisão), `cabeca__<config>.npz` para os seis ridges (o backbone é o checkpoint público; ficam a URL, o tamanho e o sha256) e `servir__<config>.json` (pré-processamento, orientações, fórmula da cabeça, leitura das 22 saídas).
4. **Mede o custo de servir** em `servico.json`, numa fase à parte depois de todo o treino, um modelo por vez: parâmetros, bytes de pesos e latência por imagem na resolução de avaliação, com 1 e com 8 orientações, numa T4 em fp16 (lotes de 1 e de 8) e na CPU (float32, lote de 1, número de threads registrado). Cada modelo é remontado **só** com os arquivos de servir e comparado com as previsões gravadas (`conferencia`). Os totais saem para `massa_5_backbones` e `imagem_9_modelos`. Nenhuma métrica de acerto no teste2 é calculada no Kaggle.
5. `tamanhos.json` lista os bytes de cada arquivo da saída.

O ajuste fino em GPU não repete bit a bit; o ridge sobre backbone congelado deve repetir quase exatamente. A paridade com a execução 1 é conferida na ingestão, sem gabarito.

## Arquivos

| Arquivo | Para quê |
|---|---|
| `treino_gpu.py` | O script. Configurações em `CONFIGS` (execuções 1 e 2) e `CONFIGS3` (execução 3); quais rodam em cada execução, em `RODADAS`; base externa em `EXTERNOS` e `com_externo`; pipelines dos totais de serviço, em `PIPELINES` e `PIPELINES3`. |
| `nutrelle_treino_gpu.ipynb` | Notebook gerado que embute o script e as listas. É o único arquivo enviado ao Kaggle. **Não edite à mão.** |
| `kernel-metadata.json` | Metadados do `kaggle kernels push` (GPU T4 ×2, internet ligada, privado, ambiente fixado no original). |
| `listas/treino.csv`, `listas/aval_ids.csv` | Os 1.684 pratos do treino estrito com cadeia, fold e alvos; os 450 ids de avaliação **sem gabarito**. |
| `listas/aval2_ids.csv` | Os 829 ids do teste2, **sem gabarito** (gravado por `eval/treino/teste2_monta.py`). |
| `listas/treino_v2.csv` | Os 2.281 pratos do treino da seção 12, com cadeia, fold, alvos e frações por fonte (gravado por `eval/treino/treino_v2_monta.py`). |
| `listas/externo_<base>.csv` | Rótulos de uma base externa e onde está cada foto (gravados por `eval/dados_ext/<base>_monta.py`). Nenhuma imagem. Os do OmniFood8K, FPB e SNAPMe são ignorados pelo git (licença não comercial ou compartilha-igual). |
| `listas/divisao_r4_<base>.csv` | Divisão da quarta execução (papel, fold, folds proibidos, quadro da métrica), de `eval/treino/externo_r4_divide.py`. As das bases fechadas também são ignoradas pelo git. |
| `listas/ref_d1_r3.csv` | Previsões fora-do-fold das duas cabeças do D1 na execução 3, para a conferência de paridade (13.7). |
| `monta_pacote.py` | Regrava as listas e o notebook; `confere` verifica se o notebook está em dia, se o teste2 é o mesmo da execução 2 (conteúdo fixado) e não cruza com nenhum treino, e se a lista externa não traz prato do Nutrition5k nem alvo fora do pré-registro. |
| `conjuntos_gpu.json` | Conjuntos (médias) declarados antes de qualquer resultado de GPU. |
| `ingere_kaggle.py` | Confere a saída e a leva para os bancos de previsões (ver "Trazer os resultados"). |

As listas só têm ids e metadados públicos do Nutrition5k (CC BY 4.0) e rótulos derivados do FlavoriaFoodWeight1700 (CC BY 4.0). Depois de mudar `treino_gpu.py` ou as listas: `python3 monta_pacote.py notebook && python3 monta_pacote.py confere`.

## Rodar no Kaggle pela API

```
pip install kaggle
export KAGGLE_API_TOKEN=...                    # ou ~/.kaggle/access_token (ou o kaggle.json antigo)
cd eval/treino/kaggle
export NUTRELLE_SCRATCH=<pasta com listas_nc/ e listas_r4/>      # as listas e tabelas das bases fechadas, fora do repositório
python3 monta_pacote.py push <pasta_push> --usuario <seu usuário> --espelho-omnifood <seu usuário>/<dataset do OmniFood8K>
python3 monta_pacote.py confere <pasta_push>
kaggle kernels push -p <pasta_push>            # execução 4: kernel novo, nutrelle-treino-gpu-r4 (o do repositório fica com USUARIO_KAGGLE)
kaggle kernels status USUARIO/nutrelle-treino-gpu-r4      # ou: kaggle kernels logs USUARIO/nutrelle-treino-gpu-r4 --follow
kaggle kernels output USUARIO/nutrelle-treino-gpu-r4 -p <pasta_saida>
```

A conta precisa estar verificada por telefone para ter GPU e internet. `"docker_image_pinning_type": "original"` pede o mesmo ambiente da primeira versão do notebook (execução 1: torch 2.11.0+cu130); as versões de torch e timm usadas ficam nas primeiras linhas do registro e em `manifesto.json`. A saída da execução 2 tem 1,27 GB e a da execução 3, cerca de 0,7 GB, quase tudo em `pesos__*.pt`; para baixar sem os pesos: `--file-pattern '^(?!pesos__)'`. O usuário real do Kaggle nunca entra no repositório: `kernel-metadata.json` fica com `USUARIO_KAGGLE`.

## Rodar no Kaggle pelo site

*Create → New Notebook → File → Import Notebook* e envie `nutrelle_treino_gpu.ipynb`. Em *Settings*: *Accelerator = GPU T4 x2*, *Internet = On*. Depois *Save Version → Save & Run All (Commit)*. Rodar de forma interativa também funciona, mas a sessão cai se a aba fechar. Ao terminar, baixe tudo na aba *Output*.

## Rodar no Colab

Abra o notebook (*Arquivo → Fazer upload*), escolha *Ambiente de execução → Alterar tipo → GPU* e rode tudo. A saída vai para `/content/nutrelle_saida` e se perde quando a sessão cai. Para retomar, monte o Drive e, na primeira célula, aponte `SAIDA` para uma pasta dele (por exemplo `/content/drive/MyDrive/nutrelle_saida`): ao rodar de novo, o script pula o que já está lá.

## O que roda e quanto demora

Cada configuração gera previsões fora-do-fold no treino (os mesmos 5 folds por cadeia da CPU) e previsões nos pratos de avaliação por um modelo treinado em todo o treino, sempre com média de 8 orientações. "ridge" = backbone congelado + ridge; "ft" = ajuste fino completo com cabeça multitarefa (22 saídas em log) iniciada na solução do ridge, orientações e escala aleatórias (0,7 a 1,3), decaimento de lr por camada, média móvel dos pesos e precisão mista. Não há parada antecipada.

Minutos de GPU **medidos na execução 1** (2 × T4, 2.134 pratos), na ordem de prioridade. "Ex. 2" marca o que a execução 2 roda.

| Configuração | Minutos de GPU (execução 1) | Ex. 2 |
|---|---|---|
| `gpu ridge siglip2_b16_224 o8` — paridade com a CPU | 2,0 | sim |
| `gpu ridge siglip2_b16_384x512 o8` | 4,6 | sim |
| `gpu ridge dinov2_b14_336x448 o8` | 3,7 | sim |
| `gpu ft siglip2_b16_224 r1` (lr 3e-5, decaimento 0,75, 15 épocas) | 26,0 | sim |
| `gpu ft siglip2_b16_384x512 r1` (12 épocas) | 100,9 | sim |
| `gpu ft dinov2_b14_336x448 r1` (12 épocas) | 99,4 | sim |
| `gpu ft siglip2_b16_224 r2` (lr 6e-5, decaimento 0,65, 25 épocas) | 41,1 | sim |
| `gpu ft siglip2_b16_224 r3` (lr 2e-5, decaimento 0,9, 8 épocas) | 14,9 | |
| `gpu ridge dinov2_l14_336x448 o8` | 11,2 | sim |
| `gpu ridge siglip2_l16_384x512 o8` | 12,7 | sim |
| `gpu ridge siglip2_so400m16_384x512 o8` | 18,4 | sim |
| `gpu ft siglip2_b16_384x512 r2` (16 épocas) | 132,5 | |
| `gpu ft siglip2_b16_224 r1 sem-escala` (ablação) | 25,4 | |
| `gpu ft dinov2_l14_224x308 r1` (10 épocas) | 121,7 | sim |
| `gpu destila-rgbd siglip2_b16_224 r1` | 58,8 | |

- **Execução 1:** 673 min de GPU nas 15 configurações, 341 min de relógio (o script abre um processo por GPU).
- **Execução 2 (medido):** 11 configurações, 271 min de relógio, com o teste2 e a fase de serviço. Os ridges levaram 28,0 (So400m), 17,4 (L/16), 17,5 (DINOv2 L/14), 6,1 (B/16 a 384×512) e 5,9 min (DINOv2 B/14); os ajustes finos `siglip2_b16_224 r2` e `siglip2_b16_384x512 r1`, 45,3 e 111,9 min. As estimativas da execução 3 partem desses tempos. O que não cabe no orçamento fica marcado como `sem_orcamento` e não gera arquivo pela metade.

`python3 treino_gpu.py --lista` mostra as configurações da rodada (com `--externo flavoria`, as 27 da execução 3 com a base); `--rodada 2` e `--rodada 1` voltam às execuções anteriores; `--so "texto"` roda só parte; `--sem-servico` pula a fase de serviço; `--guarda-pesos melhor` guarda só os pesos do melhor modelo (o padrão é `todos`); `--externo-imagens <pasta>` usa fotos do Flavoria já em disco, sem baixar.

## Retomar

Cada parte pronta (`fold0`…`fold4`, `final`; `tudo` no ridge) vira um `parte__*.npz` na saída. Para continuar em outra sessão do Kaggle: suba a pasta baixada como dataset privado, acrescente-o em `dataset_sources` de `kernel-metadata.json` e faça o push de novo. O script procura `manifesto.json` em `/kaggle/input`, copia o que está pronto e segue de onde parou. Fora do Kaggle: `--retoma <pasta>`. Só são aceitas partes feitas com as mesmas listas de treino e avaliação.

Na execução 3, uma parte treinada com a base externa só é reaproveitada se a lista externa for a mesma (o hash fica gravado na parte), e a decisão de adoção só é reaproveitada se foi tomada pela regra, com a mesma lista.

Uma parte `final` ou `tudo` só conta como pronta se tiver as previsões do teste2 e os arquivos de servir. Por isso, **se a saída da execução 1 for anexada, os folds dela são reaproveitados** (o fora-do-fold fica igual ao da execução 1) e só as partes `final`/`tudo` são refeitas. Para a execução 2 valer como repetição completa, não anexe nada.

## Saída (tudo na raiz da pasta, sem subpastas)

- `pred__<config>.npz`: `oof` [1.684 × 22], `calib` [120 × 22], `teste` [280 × 22], `n5k_v1` [50 × 22] e `teste2` [829 × 22], em log, na ordem das listas, mais os ids para conferência. `pred__<config>.csv` traz o mesmo em texto.
- `pesos__<config>.pt`: rede inteira do ajuste fino (média móvel dos pesos, meia precisão) + `tm`, `ts`; carrega com `torch.load(..., weights_only=True)`. De 173 a 608 MB cada. **Não commitar.**
- `cabeca__<config>.npz`: `mu`, `sd`, `W`, `mt`, `alfa` do ridge. `servir__<config>.json`: como reproduzir a previsão.
- `servico__<config>.json` e `servico.json`: custo de servir por modelo e totais por pipeline. `tamanhos.json`: bytes por arquivo.
- `log__<config>.json`: receita, perda e validação por época, fotos/s, memória de GPU, tempo por parte.
- `manifesto.json`, `eventos.jsonl`, `treino_gpu.log`: estado de cada parte, eventos e registro completo.

Na execução 3, `pred__<config>.npz` traz `oof` [2.281 × 22] e `teste2` [829 × 22] nas três variantes de orientações, as frações e a densidade por frações quando existem, `dens_fonte`, e, conforme a configuração, `oof_aninhado`, `aninhado_dens_ff`, `alvos_oof`, `alvos_aninhados` e a réplica `r2_teste2`. `cabeca__<config>.npz` das cabeças MLP traz `mu`, `sd`, `W1`, `b1`, `W2`, `b2`, `tm`, `ts`, `dens_fonte`. Com a base externa saem também `decisao_flavoria.json`, `relato_externo.json`, e nas previsões `ext` (fotos do Flavoria; `ext_sem` = o modelo sem a base nas mesmas fotos), `ext_ids`, `ext_treina`, `ext_delta` e `ext_lista`. A fase de serviço mede primeiro os candidatos a modelo único (do U1, só a versão adotada) e depois os cinco backbones congelados.

Em `servico.json`, `loteB_oK` é um lote de B imagens com K orientações em sequência (como as previsões foram geradas) e `lote1_o8_em_lote` são as 8 orientações de uma imagem num lote só (ou dois de 4, se a entrada não é quadrada). O tempo inclui redimensionar e normalizar a partir de uint8 480×640 já no dispositivo; não inclui decodificar o PNG nem copiar para a GPU. O total de um pipeline é a soma dos modelos, um depois do outro. As três cabeças de frações da densidade (seção 11.2) foram treinadas na CPU e não são medidas: ver `nota_pipelines` no arquivo.

## Trazer os resultados

```
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida> --so-confere               # só confere (e, com teste2, imprime a paridade)
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida> --ensaio <pasta_qualquer>  # ensaio: grava tudo numa pasta à parte
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida>                             # para valer
```

**Saída da execução 3.** Nada lê gabarito de avaliação e nada é escolhido. O script confere listas, ordem, folds, as três variantes de orientações, as frações, o padrão de NaN dos alvos aninhados, a ausência de previsões para os 450 pratos antigos e a paridade das features com a execução 2 (`--referencia <pasta da saída da execução 2>`; padrão: `<TD>/resultados/zen_banco_r2`). Com a base externa: a lista do manifesto é a desta pasta, nenhuma amostra externa tem id do Nutrition5k, cada `+flavoria` foi treinada com essa lista, a decisão gravada se repete a partir das previsões dos dois professores; imprime quais versões concorrem em 12.3 (as adotadas) e o relato do `teste_ext`. Sem `--so-confere`, grava tudo em `<TD>/resultados/zen_banco_r3/` e o resumo em `ingestao_kaggle_r3.json` (com `decisao_externo` e `concorrem_em_12_3`).

**Saída com teste2 (execução 2).** Nada lê gabarito de avaliação. O script confere listas, folds e ordem dos pratos (inclusive do teste2), imprime a paridade contra a execução 1 e grava cada configuração em `<TD>/resultados/zen_banco_r2/<nome>.npz` (`oof`, `calib`, `teste`, `n5k_v1`, `teste2`, `ids_teste2`, `pratos_teste2`). **O banco da seção 10 (`zen_banco`) não é alterado e `zen_avalia.py` não roda.** A paridade, por configuração: correlação e diferença absoluta mediana, em log, da massa, da proteína direta e da densidade nos 450 pratos antigos e no fora-do-fold do treino, a CV no treino das duas execuções, e as mesmas medidas para a massa média dos cinco backbones e a densidade média dos quatro ajustes finos. A referência é o `zen_banco` (`--referencia <pasta>` aponta outra: banco ou saída do Kaggle). `--substitui` troca também as entradas do `zen_banco` (as antigas vão para `zen_banco_antes_r2/`) e roda `zen_avalia.py`, o que lê o `teste` antigo outra vez; não é o padrão.

**Saída sem teste2 (execução 1).** Como antes: copia cada configuração completa para `<TD>/resultados/zen_banco/`, soma os conjuntos de `conjuntos_gpu.json` e roda `zen_avalia.py` sem mudar nenhuma regra. O `zen_avalia.json` anterior fica em `zen_avalia.antes_kaggle.json`.

**Primeira coisa a olhar:** `gpu ridge siglip2_b16_224 o8` tem de dar na CV quase o mesmo que `ridge siglip2_b16 o8` da CPU (RMSE de log-massa 0,2551; critério 56,6 / 54,2 / 54,9%; a execução 1 deu 0,2550). Na execução 2, os seis ridges têm de repetir a execução 1 com correlação de 1,000; se não repetirem, o ambiente mudou e nada mais deve ser lido.

## Máquina de estados

Cada parte (configuração × `fold0`…`fold4`/`final`, ou `tudo` no ridge, e `servico` na fase de serviço) tem um estado explícito, mudado só por `transicao(estado, evento)` em `treino_gpu.py` (função pura, testada por `python3 treino_gpu.py --autoteste`) e registrado em `eventos.jsonl` e `manifesto.json`.

| De | Evento | Para |
|---|---|---|
| pendente | `retomou` (arquivo já existe e está completo) | feita |
| pendente | `iniciou` | rodando |
| pendente | `sem_orcamento` | sem_orcamento |
| pendente | `reduziu_lote` (outra parte estourou a memória) | pendente, com lote menor |
| pendente | `cancelou` (outra parte falhou ou ficou sem orçamento) | cancelada |
| rodando | `concluiu` | feita |
| rodando | `estourou_memoria` | pendente com lote ÷ 2 (até ÷ 4); depois, falhou |
| rodando | `falhou` | pendente (1ª vez); falhou (2ª vez) |
| rodando | `sem_orcamento` (a projeção de tempo não cabe) | sem_orcamento |

A configuração é `completa` quando todas as partes de treino estão `feita`; só então saem `pred__` e `log__`. Uma falha derruba só a configuração dela. A parte `servico` roda depois e não muda o estado da configuração: se falhar, as previsões e os arquivos de servir continuam valendo.

Na execução 3 a fila também é decidida por funções puras, testadas no `--autoteste`:

| Função | Entrada → saída |
|---|---|
| `proxima_acao(configs, estado, cabe)` | → `roda` (primeira parte pendente com as dependências completas) \| `espera` (depende de algo que ainda roda) \| `cancela` (uma dependência não ficou pronta nem vai ficar) \| `sem_orcamento` (decidido só quando chega a vez da configuração) \| `fim` |
| `simula_orcamento(configs, estado, livres, seg_parte, limite)` | → quais configurações cabem inteiras, em ordem de prioridade |
| `seg_parte(config, configs, estado)` | → segundos por parte: mediana do que já foi medido, ou a estimativa corrigida pela razão medido/estimado do mesmo tipo |
| `regra_adocao(dados, oof_sem, oof_com)` | → decisão sobre a base externa, com os dois critérios e as diferenças |

Duas pseudo-configurações usam a mesma máquina de estados, sem ocupar GPU: `externo flavoria` (download e preparo das fotos; `falhou` cancela por dependência todas as versões `+flavoria`) e `decisao flavoria` (fica `feita` quando a decisão é gravada; o evento cancela, dos ajustes finos destilados, a versão que a decisão exclui). Se a versão com a base não fica pronta, a decisão é "não adotada", com o motivo gravado.

## Teste rápido em CPU

```
python3 treino_gpu.py --autoteste             # inclui o fechamento dos rótulos de registro, pesos por base, procedência, paridade, fila da execução 4
python3 treino_gpu.py --smoke --listas <pasta com todas as listas e tabelas> --externo-imagens <pasta com uma subpasta de fotos por base> --saida <pasta> --cache <pasta_cache>
python3 treino_gpu.py --rodada 3 --smoke --saida <pasta> --cache <pasta_cache>     # execução 3: ~3 min, 40 pratos + 4 do teste2, ViT-Ti
python3 treino_gpu.py --rodada 2 --smoke --saida <pasta2> --cache <pasta_cache2>   # execução 2
python3 ingere_kaggle.py <pasta> --so-confere
```

Na execução 4, com `--externo-imagens`, o teste usa de cada base só as fotos que estão em `<pasta>/<base>/` (com o caminho da lista ou só o nome do arquivo): algumas de treino em cada fold e algumas de registro; nada é baixado. Leva ~4 min em CPU e passa por todas as configurações, a paridade (que falha, de propósito: o backbone é outro) e a fase de serviço. Sem a pasta de uma base, o teste baixa as fotos dela de verdade (as bases de foto a foto, só as escolhidas; as de arquivo único, o arquivo inteiro).

## Dados e licenças

As fotos vêm de `https://storage.googleapis.com/nutrition5k_dataset/` na hora de rodar (Nutrition5k, Google Research, CC BY 4.0). As do Flavoria vêm de `https://zenodo.org/records/5850856` (Sarapisto e outros, 2022, FlavoriaFoodWeight1700, CC BY 4.0; atribuição obrigatória em qualquer publicação que use os modelos treinados com ela). Os pesos do SigLIP 2 vêm de `storage.googleapis.com/big_vision/siglip2/` (os mesmos arquivos usados na CPU; Apache 2.0) e os do DINOv2, do Hugging Face via `timm` (Apache 2.0). Nenhuma imagem, feature ou peso entra no repositório.
