# Treino em GPU (Kaggle ou Colab) — seção 8, uma única foto de cima

Roda, sem interação, os experimentos que a seção 8 de `docs/dados/precisao/resultado_modelo_treinado.md` deixou "sem rodar por falta de tempo de CPU": ajuste fino completo do SigLIP 2 B/16 a 224 px e a 384×512, 8 orientações na resolução maior, backbones DINOv2 e SigLIP 2 maiores e, por último, destilação de um professor RGB-D. A inferência continua usando **uma** foto RGB zenital. O critério é o do produto: proteína dentro de ±20% ou ±5 g.

**Estado (11/11/2026): rodou uma vez no Kaggle (2× T4, 15 configurações completas em ~5,7 h; a paridade com a CPU conferiu: RMSE 0,2550 contra 0,2551).** Resultados na seção 10 de `docs/dados/precisao/resultado_modelo_treinado.md`. Os tempos por configuração medidos estão em `treino_gpu.log` da saída; os da tabela abaixo eram estimativas feitas antes da execução. Para baixar a saída é preciso acesso a `www.kaggleusercontent.com`.

## Arquivos

| Arquivo | Para quê |
|---|---|
| `treino_gpu.py` | O script. Baixa as fotos, treina, grava previsões, registros e pesos. Configurações declaradas em `CONFIGS`. |
| `nutrelle_treino_gpu.ipynb` | Notebook gerado que embute o script e as listas. É o único arquivo enviado ao Kaggle. **Não edite à mão.** |
| `kernel-metadata.json` | Metadados do `kaggle kernels push` (GPU T4 ×2, internet ligada, privado). |
| `listas/treino.csv`, `listas/aval_ids.csv` | Os 1.684 pratos do treino estrito com cadeia, fold e alvos; os 450 ids de avaliação **sem gabarito**. Só ids e metadados públicos do Nutrition5k (CC BY 4.0). |
| `monta_pacote.py` | Regrava as listas e o notebook; `confere` verifica se o notebook está em dia. |
| `conjuntos_gpu.json` | Conjuntos (médias) declarados antes de qualquer resultado de GPU. |
| `ingere_kaggle.py` | Leva a saída para o banco de previsões e roda `zen_avalia.py`. |

Depois de mudar `treino_gpu.py` ou as listas: `python3 monta_pacote.py notebook && python3 monta_pacote.py confere`.

## Rodar no Kaggle pela API

```
pip install kaggle
export KAGGLE_API_TOKEN=...                    # ou ~/.kaggle/access_token (ou o kaggle.json antigo)
cd eval/treino/kaggle
# troque USUARIO_KAGGLE pelo seu usuário em kernel-metadata.json ("id")
python3 monta_pacote.py confere
kaggle kernels push -p .                       # cria a versão e já roda (Save & Run All)
kaggle kernels status USUARIO/nutrelle-treino-gpu      # ou: kaggle kernels logs USUARIO/nutrelle-treino-gpu --follow
kaggle kernels output USUARIO/nutrelle-treino-gpu -p <pasta_saida>
```

A conta precisa estar verificada por telefone para ter GPU e internet. Para não baixar os pesos (centenas de MB): `--file-pattern '^(?!pesos__)'`.

## Rodar no Kaggle pelo site

*Create → New Notebook → File → Import Notebook* e envie `nutrelle_treino_gpu.ipynb`. Em *Settings*: *Accelerator = GPU T4 x2*, *Internet = On*. Depois *Save Version → Save & Run All (Commit)*. Rodar de forma interativa também funciona, mas a sessão cai se a aba fechar. Ao terminar, baixe tudo na aba *Output*.

## Rodar no Colab

Abra o notebook (*Arquivo → Fazer upload*), escolha *Ambiente de execução → Alterar tipo → GPU* e rode tudo. A saída vai para `/content/nutrelle_saida` e se perde quando a sessão cai. Para retomar, monte o Drive e, na primeira célula, aponte `SAIDA` para uma pasta dele (por exemplo `/content/drive/MyDrive/nutrelle_saida`): ao rodar de novo, o script pula o que já está lá.

## O que roda, em ordem de prioridade

Cada configuração gera previsões fora-do-fold no treino (os mesmos 5 folds por cadeia da CPU) e previsões nos 450 pratos de avaliação por um modelo treinado em todo o treino, sempre com média de 8 orientações. "ridge" = backbone congelado + ridge; "ft" = ajuste fino completo com cabeça multitarefa (22 saídas em log) iniciada na solução do ridge, orientações e escala aleatórias (0,7 a 1,3), decaimento de lr por camada, média móvel dos pesos e precisão mista. Não há parada antecipada.

| # | Configuração | Minutos numa T4 (estimativa) |
|---|---|---|
| 1 | `gpu ridge siglip2_b16_224 o8` — paridade com a CPU | 3 |
| 2 | `gpu ridge siglip2_b16_384x512 o8` — 8 orientações na resolução maior | 7 |
| 3 | `gpu ridge dinov2_b14_336x448 o8` | 7 |
| 4 | `gpu ft siglip2_b16_224 r1` (lr 3e-5, decaimento 0,75, 15 épocas) | 35 |
| 5 | `gpu ft siglip2_b16_384x512 r1` (12 épocas) | 110 |
| 6 | `gpu ft dinov2_b14_336x448 r1` (12 épocas) | 110 |
| 7 | `gpu ft siglip2_b16_224 r2` (lr 6e-5, decaimento 0,65, 25 épocas) | 50 |
| 8 | `gpu ft siglip2_b16_224 r3` (lr 2e-5, decaimento 0,9, 8 épocas: a receita da CPU no backbone inteiro) | 22 |
| 9–11 | `gpu ridge` com DINOv2 L/14, SigLIP 2 L/16 e SigLIP 2 So400m/16, todos em alta resolução | 16 + 16 + 25 |
| 12 | `gpu ft siglip2_b16_384x512 r2` (16 épocas) | 140 |
| 13 | `gpu ft siglip2_b16_224 r1 sem-escala` (ablação) | 35 |
| 14 | `gpu ft dinov2_l14_224x308 r1` (10 épocas) | 125 |
| 15 | `gpu destila-rgbd siglip2_b16_224 r1` (professor RGB-D → aluno RGB) | 75 |

Soma: cerca de 13 h numa T4, ou perto de 6,5 h de relógio com as duas T4 do Kaggle (o script abre um processo por GPU). As estimativas supõem ~90 fotos/s no treino do ViT-B a 224 px e ~18 fotos/s com 768 tokens; os tempos medidos ficam em `manifesto.json` e nos `log__*.json`. O orçamento padrão é 11 h de relógio (`ORCAMENTO_H` na primeira célula): o que não couber fica marcado como `sem_orcamento` e não gera arquivo pela metade.

`python3 treino_gpu.py --lista` mostra as configurações; `--so "texto"` roda só parte.

## Retomar

Cada parte pronta (`fold0`…`fold4`, `final`) vira um `parte__*.npz` na saída. Para continuar em outra sessão do Kaggle: suba a pasta baixada como dataset privado, acrescente-o em `dataset_sources` de `kernel-metadata.json` e faça o push de novo. O script procura `manifesto.json` em `/kaggle/input`, copia as partes prontas e segue de onde parou. Fora do Kaggle: `--retoma <pasta>`. Só são aceitas partes feitas com as mesmas listas.

## Saída (tudo na raiz da pasta, sem subpastas)

- `pred__<config>.npz`: `oof` [1.684 × 22], `calib` [120 × 22], `teste` [280 × 22], `n5k_v1` [50 × 22], em log, na ordem das listas — o esquema do banco `zen_banco` — mais os ids para conferência. `pred__<config>.csv` traz o mesmo conteúdo em texto.
- `log__<config>.json`: receita, perda e validação por época, fotos/s, memória de GPU, tempo por parte.
- `pesos__<melhor>.pt`: pesos (meia precisão) do melhor modelo único pela CV no treino. `--guarda-pesos todos` guarda todos. **Não commitar.**
- `manifesto.json`, `eventos.jsonl`, `treino_gpu.log`: estado de cada parte, eventos e registro completo.

## Trazer os resultados

```
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida> --ensaio <pasta_qualquer>   # ensaio: não altera o banco
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida>                              # para valer
```

O script confere listas, folds e ordem dos pratos, copia cada configuração completa para `<TD>/resultados/zen_banco/`, soma os conjuntos de `conjuntos_gpu.json` e roda `zen_avalia.py` sem mudar nenhuma regra (escolha por CV no treino, híbrido no `calib`, `teste` só para relatar). O `zen_avalia.json` anterior fica em `zen_avalia.antes_kaggle.json`. O Kaggle não vê os itens brasileiros: se a CV escolher uma configuração de GPU, a seção 5 de `zen_avalia.py` (Brasil) fica marcada como não avaliada.

**Primeira coisa a olhar:** `gpu ridge siglip2_b16_224 o8` tem de dar na CV quase o mesmo que `ridge siglip2_b16 o8` da CPU (RMSE de log-massa 0,2551; critério 56,6 / 54,2 / 54,9%). Se não der, há algo errado no ambiente (pesos, redimensionamento) e nada mais deve ser lido.

## Máquina de estados

Cada parte (configuração × `fold0`…`fold4`/`final`, ou `tudo` no ridge) tem um estado explícito, mudado só por `transicao(estado, evento)` em `treino_gpu.py` (função pura, testada por `python3 treino_gpu.py --autoteste`) e registrado em `eventos.jsonl` e `manifesto.json`.

| De | Evento | Para |
|---|---|---|
| pendente | `retomou` (arquivo já existe) | feita |
| pendente | `iniciou` | rodando |
| pendente | `sem_orcamento` | sem_orcamento |
| pendente | `reduziu_lote` (outra parte estourou a memória) | pendente, com lote menor |
| pendente | `cancelou` (outra parte falhou ou ficou sem orçamento) | cancelada |
| rodando | `concluiu` | feita |
| rodando | `estourou_memoria` | pendente com lote ÷ 2 (até ÷ 4); depois, falhou |
| rodando | `falhou` | pendente (1ª vez); falhou (2ª vez) |
| rodando | `sem_orcamento` (a projeção de tempo não cabe) | sem_orcamento |

A configuração é `completa` quando todas as partes estão `feita`; só então saem `pred__` e `log__`. Uma falha derruba só a configuração dela.

## Teste rápido em CPU

```
python3 treino_gpu.py --autoteste
python3 treino_gpu.py --smoke --saida <pasta> --cache <pasta_cache>     # ~3 min, 52 pratos, ViT-Ti
python3 ingere_kaggle.py <pasta> --ensaio <pasta_ensaio>
```

## Dados e licenças

As fotos vêm de `https://storage.googleapis.com/nutrition5k_dataset/` na hora de rodar (Nutrition5k, Google Research, CC BY 4.0). Os pesos do SigLIP 2 vêm de `storage.googleapis.com/big_vision/siglip2/` (os mesmos arquivos usados na CPU; Apache 2.0) e os do DINOv2, do Hugging Face via `timm` (Apache 2.0). Nenhuma imagem, feature ou peso entra no repositório.
