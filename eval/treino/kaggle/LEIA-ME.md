# Treino em GPU (Kaggle ou Colab) — seções 8, 10, 11 e 12, uma única foto de cima

Roda, sem interação, os treinos em GPU do modelo de proteína pela foto, descritos em `docs/dados/precisao/resultado_modelo_treinado.md`. A inferência usa sempre **uma** foto RGB zenital. O critério é o do produto: proteína dentro de ±20% ou ±5 g.

**Estado (11/11/2026):**

- **Execução 1 (seção 10)** rodou no Kaggle: 2× T4, 15 configurações em ~5,7 h; a paridade com a CPU conferiu (RMSE 0,2550 contra 0,2551).
- **Execução 2 (seção 11)** rodou no Kaggle: 11 configurações em 4,5 h de relógio (torch 2.11.0+cu130, timm 1.0.30), com previsões no teste2, arquivos de servir e `servico.json`; saída de 1,27 GB.
- **Execução 3 (seções 12.3 a 12.5) está pronta para o push e ainda não rodou em GPU.** É o padrão do script (`--rodada 3`) e do notebook. Foi testada só em CPU, com um backbone minúsculo, 40 pratos e 19 fotos do Flavoria; os tempos abaixo são estimativas feitas com os tempos medidos na execução 2.

Para baixar a saída é preciso acesso a `www.kaggleusercontent.com`.

## O que a execução 3 faz

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
| `listas/externo_flavoria.csv` | Rótulos do Flavoria, uma linha por foto, e o caminho de cada foto dentro do `.tar` do Zenodo (gravado por `eval/dados_ext/flavoria_monta.py`). Nenhuma imagem. |
| `monta_pacote.py` | Regrava as listas e o notebook; `confere` verifica se o notebook está em dia, se o teste2 é o mesmo da execução 2 (conteúdo fixado) e não cruza com nenhum treino, e se a lista externa não traz prato do Nutrition5k nem alvo fora do pré-registro. |
| `conjuntos_gpu.json` | Conjuntos (médias) declarados antes de qualquer resultado de GPU. |
| `ingere_kaggle.py` | Confere a saída e a leva para os bancos de previsões (ver "Trazer os resultados"). |

As listas só têm ids e metadados públicos do Nutrition5k (CC BY 4.0) e rótulos derivados do FlavoriaFoodWeight1700 (CC BY 4.0). Depois de mudar `treino_gpu.py` ou as listas: `python3 monta_pacote.py notebook && python3 monta_pacote.py confere`.

## Rodar no Kaggle pela API

```
pip install kaggle
export KAGGLE_API_TOKEN=...                    # ou ~/.kaggle/access_token (ou o kaggle.json antigo)
cd eval/treino/kaggle
# troque USUARIO_KAGGLE pelo seu usuário em kernel-metadata.json ("id"), numa cópia fora do repositório
python3 monta_pacote.py confere
kaggle kernels push -p .                       # cria uma versão nova do mesmo notebook e já roda (Save & Run All)
kaggle kernels status USUARIO/nutrelle-treino-gpu      # ou: kaggle kernels logs USUARIO/nutrelle-treino-gpu --follow
kaggle kernels output USUARIO/nutrelle-treino-gpu -p <pasta_saida>
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
python3 treino_gpu.py --autoteste
python3 treino_gpu.py --smoke --saida <pasta> --cache <pasta_cache>     # execução 3: ~3 min, 40 pratos + 4 do teste2, ViT-Ti, com a fase de serviço
python3 treino_gpu.py --smoke --externo flavoria --externo-imagens <pasta com algumas fotos do Flavoria> --saida <pasta> --cache <pasta_cache>
python3 treino_gpu.py --rodada 2 --smoke --saida <pasta2> --cache <pasta_cache2>   # execução 2
python3 ingere_kaggle.py <pasta> --so-confere
```

Com `--externo-imagens`, o teste usa só as linhas da lista cujas fotos estão na pasta (até 16 de treino e 8 pratos de `teste_ext`) e não baixa nada.

## Dados e licenças

As fotos vêm de `https://storage.googleapis.com/nutrition5k_dataset/` na hora de rodar (Nutrition5k, Google Research, CC BY 4.0). As do Flavoria vêm de `https://zenodo.org/records/5850856` (Sarapisto e outros, 2022, FlavoriaFoodWeight1700, CC BY 4.0; atribuição obrigatória em qualquer publicação que use os modelos treinados com ela). Os pesos do SigLIP 2 vêm de `storage.googleapis.com/big_vision/siglip2/` (os mesmos arquivos usados na CPU; Apache 2.0) e os do DINOv2, do Hugging Face via `timm` (Apache 2.0). Nenhuma imagem, feature ou peso entra no repositório.
