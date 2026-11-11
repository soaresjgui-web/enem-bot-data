# Treino em GPU (Kaggle ou Colab) — seções 8, 10 e 11, uma única foto de cima

Roda, sem interação, os treinos em GPU do modelo de proteína pela foto: ajuste fino completo do SigLIP 2 B/16 a 224 px e a 384×512, backbones DINOv2 e SigLIP 2 congelados em alta resolução com 8 orientações e o restante da seção 10 de `docs/dados/precisao/resultado_modelo_treinado.md`. A inferência usa sempre **uma** foto RGB zenital. O critério é o do produto: proteína dentro de ±20% ou ±5 g.

**Estado (11/11/2026): rodou uma vez no Kaggle (2× T4, 15 configurações completas em ~5,7 h; a paridade com a CPU conferiu: RMSE 0,2550 contra 0,2551).** Resultados na seção 10 de `docs/dados/precisao/resultado_modelo_treinado.md`. Para baixar a saída é preciso acesso a `www.kaggleusercontent.com`.

**A execução 2 (seção 11) está pronta para o push e ainda não rodou em GPU.** O padrão do script passou a ser ela. Foi testada só em CPU, com um backbone minúsculo e 56 pratos; os tempos dela abaixo são estimativas feitas com os tempos medidos na execução 1.

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
| `treino_gpu.py` | O script. Configurações em `CONFIGS`; quais rodam em cada execução, em `RODADAS`; pipelines dos totais de serviço, em `PIPELINES`. |
| `nutrelle_treino_gpu.ipynb` | Notebook gerado que embute o script e as listas. É o único arquivo enviado ao Kaggle. **Não edite à mão.** |
| `kernel-metadata.json` | Metadados do `kaggle kernels push` (GPU T4 ×2, internet ligada, privado, ambiente fixado no original). |
| `listas/treino.csv`, `listas/aval_ids.csv` | Os 1.684 pratos do treino estrito com cadeia, fold e alvos; os 450 ids de avaliação **sem gabarito**. |
| `listas/aval2_ids.csv` | Os 829 ids do teste2, **sem gabarito** (gravado por `eval/treino/teste2_monta.py`). |
| `monta_pacote.py` | Regrava as listas e o notebook; `confere` verifica se o notebook está em dia e se o teste2 não cruza com treino nem com os 450. |
| `conjuntos_gpu.json` | Conjuntos (médias) declarados antes de qualquer resultado de GPU. |
| `ingere_kaggle.py` | Confere a saída e a leva para os bancos de previsões (ver "Trazer os resultados"). |

As listas só têm ids e metadados públicos do Nutrition5k (CC BY 4.0). Depois de mudar `treino_gpu.py` ou as listas: `python3 monta_pacote.py notebook && python3 monta_pacote.py confere`.

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

A conta precisa estar verificada por telefone para ter GPU e internet. `"docker_image_pinning_type": "original"` pede o mesmo ambiente da primeira versão do notebook (execução 1: torch 2.11.0+cu130); as versões de torch e timm usadas ficam nas primeiras linhas do registro e em `manifesto.json`. A saída da execução 2 tem cerca de 1,4 GB, quase tudo em `pesos__*.pt`; para baixar sem os pesos: `--file-pattern '^(?!pesos__)'`.

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
- **Execução 2 (estimativa):** as 11 configurações somaram 442 min de GPU na execução 1. O teste2 acrescenta perto de 30 min (a extração de features cresce 39% e o modelo final prevê 829 pratos a mais) e a fase de serviço, de 15 a 30 min numa GPU só. Total: perto de 8 h de GPU, **4,5 a 5 h de relógio**, dentro do orçamento padrão de 11 h (`ORCAMENTO_H` na primeira célula). O que não couber fica marcado como `sem_orcamento` e não gera arquivo pela metade.

`python3 treino_gpu.py --lista` mostra as configurações da rodada; `--rodada 1` volta às 15 da seção 10; `--so "texto"` roda só parte; `--sem-servico` pula a fase de serviço; `--guarda-pesos melhor` guarda só os pesos do melhor modelo (o padrão agora é `todos`).

## Retomar

Cada parte pronta (`fold0`…`fold4`, `final`; `tudo` no ridge) vira um `parte__*.npz` na saída. Para continuar em outra sessão do Kaggle: suba a pasta baixada como dataset privado, acrescente-o em `dataset_sources` de `kernel-metadata.json` e faça o push de novo. O script procura `manifesto.json` em `/kaggle/input`, copia o que está pronto e segue de onde parou. Fora do Kaggle: `--retoma <pasta>`. Só são aceitas partes feitas com as mesmas listas de treino e avaliação.

Uma parte `final` ou `tudo` só conta como pronta se tiver as previsões do teste2 e os arquivos de servir. Por isso, **se a saída da execução 1 for anexada, os folds dela são reaproveitados** (o fora-do-fold fica igual ao da execução 1) e só as partes `final`/`tudo` são refeitas. Para a execução 2 valer como repetição completa, não anexe nada.

## Saída (tudo na raiz da pasta, sem subpastas)

- `pred__<config>.npz`: `oof` [1.684 × 22], `calib` [120 × 22], `teste` [280 × 22], `n5k_v1` [50 × 22] e `teste2` [829 × 22], em log, na ordem das listas, mais os ids para conferência. `pred__<config>.csv` traz o mesmo em texto.
- `pesos__<config>.pt`: rede inteira do ajuste fino (média móvel dos pesos, meia precisão) + `tm`, `ts`; carrega com `torch.load(..., weights_only=True)`. De 173 a 608 MB cada. **Não commitar.**
- `cabeca__<config>.npz`: `mu`, `sd`, `W`, `mt`, `alfa` do ridge. `servir__<config>.json`: como reproduzir a previsão.
- `servico__<config>.json` e `servico.json`: custo de servir por modelo e totais por pipeline. `tamanhos.json`: bytes por arquivo.
- `log__<config>.json`: receita, perda e validação por época, fotos/s, memória de GPU, tempo por parte.
- `manifesto.json`, `eventos.jsonl`, `treino_gpu.log`: estado de cada parte, eventos e registro completo.

Em `servico.json`, `loteB_oK` é um lote de B imagens com K orientações em sequência (como as previsões foram geradas) e `lote1_o8_em_lote` são as 8 orientações de uma imagem num lote só (ou dois de 4, se a entrada não é quadrada). O tempo inclui redimensionar e normalizar a partir de uint8 480×640 já no dispositivo; não inclui decodificar o PNG nem copiar para a GPU. O total de um pipeline é a soma dos modelos, um depois do outro. As três cabeças de frações da densidade (seção 11.2) foram treinadas na CPU e não são medidas: ver `nota_pipelines` no arquivo.

## Trazer os resultados

```
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida> --so-confere               # só confere (e, com teste2, imprime a paridade)
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida> --ensaio <pasta_qualquer>  # ensaio: grava tudo numa pasta à parte
python3 eval/treino/kaggle/ingere_kaggle.py <pasta_saida>                             # para valer
```

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

## Teste rápido em CPU

```
python3 treino_gpu.py --autoteste
python3 treino_gpu.py --smoke --saida <pasta> --cache <pasta_cache>     # ~3 min, 56 pratos (4 do teste2), ViT-Ti, com a fase de serviço
python3 ingere_kaggle.py <pasta> --ensaio <pasta_ensaio>
```

## Dados e licenças

As fotos vêm de `https://storage.googleapis.com/nutrition5k_dataset/` na hora de rodar (Nutrition5k, Google Research, CC BY 4.0). Os pesos do SigLIP 2 vêm de `storage.googleapis.com/big_vision/siglip2/` (os mesmos arquivos usados na CPU; Apache 2.0) e os do DINOv2, do Hugging Face via `timm` (Apache 2.0). Nenhuma imagem, feature ou peso entra no repositório.
