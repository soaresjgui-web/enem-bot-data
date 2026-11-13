# Coleta do gabarito: proteína por foto (2026-11-08)

Arquivos: `gabarito_pratos.csv` (1 linha por prato, com fonte, URL, confiança e notas), `gabarito.csv` (formato do `eval/score_precisao.py`: uma linha `prato inteiro` por prato, `proteina_100g = proteina_g / peso × 100`, com 2 casas) e os scripts `coleta_ufpr.py`, `coleta_n5k.py`, `coleta.py` (Seara + montagem final) e `coleta_livup.py` (lote 5, Liv Up).
**As imagens não estão no repositório**, porque são de terceiros e o repositório é público. Ficam no scratchpad da sessão (`precisao/imagens/<prato_id>.png`). O arquivo `precisao/lista_imagens.txt` traz só `prato_id<TAB>caminho`, sem rótulos, e é o que o estimador cego recebe. Os ids `b001…`/`n001…` foram sorteados (seed 20261108) e não revelam conteúdo. O lote 5 (Liv Up, ids `c001…`, seed 20261109) fica em `cego/lote5/<id>.jpg` no scratchpad, pasta só com as imagens.

## Resultado

| Fonte | Itens | Categorias | Proteína (g) | Confiança |
|---|---|---|---|---|
| Manual Fotográfico UFPR 2017 + TACO 4ª ed. | 103 | almoco_pf 52, jantar 22, lanche 13, restaurante 10, cafe 6 | 0,9–100,4 | A 62 · B 29 · C 12 |
| Seara Marmita 300 g (site oficial) | 7 | congelado | 13,8–20,1 | B |
| Liv Up (site oficial), lote 5 | 34 | marmita | 10,8–44,4 | A 30 · B 4 |
| **Brasileiros** | **144** | | 0,9–100,4 | |
| Nutrition5k (CC BY 4.0) | 50 | n5k | 0,2–78 | A |

### UFPR (Crispim et al., 2017; PDF em nutritotal.com.br)
- O álbum traz séries de 4–6 fotos do mesmo alimento num prato padrão, todas tiradas em 45°. A gramatura de cada foto está na tabela das pp. 143–144. Cada foto foi recortada pela caixa da imagem no PDF (`pdfplumber`), sem o rótulo `Pxxx – n`.
- Proteína = gramas × proteína/100 g da TACO 4ª ed. (CSV de `raulfdm/taco-api`). O item TACO usado e o valor aparecem em `observacoes`.
- Foram usados 25 alimentos: arroz, feijão sem e com caldo, feijoada, baião de dois, lentilha, bife, filé de frango, filé mignon, estrogonofe, vaca atolada, porco em fatias, carne seca, torresmo, presunto, salame, filé de peixe, camarão pequeno e médio, macarronada, batata, batata frita, batata-doce, mandioca e brócolis. Nos alimentos de mais proteína entram várias porções da mesma série, porque o teste é justamente estimar o tamanho da porção.
- Confiança: **A** quando a TACO tem a preparação. **B** quando o corte ou a espécie não é informado (bife → contra-filé grelhado, peixe → merluza assada, filé mignon, porco → pernil assado, frango → peito grelhado) ou quando o feijão com caldo pode estar diluído. **C** em três casos: feijoada (o manual dá a porção em ml; assumi 1 g/ml), torresmo (usei "toucinho frito") e estrogonofe (a foto do álbum IARC parece um ragu de frango com legumes).
- Excluídos: P302 omelete/ovos mexidos, P147 queijos (vários tipos numa tábua), P047 almôndegas (uma grande e várias pequenas) e P086 fígado (dois tipos). Nessas fotos aparecem alternativas lado a lado e não fica claro a que se refere a gramatura. Também ficaram de fora alimentos sem item na TACO (lasanha, pizza, risoto, moqueca, purê, tutu, cuscuz marroquino), itens com peso que inclui osso ou casca (posta de peixe, camarão com casca, frango em partes) e a sopa (porção em volume).
- Ressalva: são alimentos isolados (um por prato), não pratos compostos.

### Seara: 7 marmitas de 300 g
- Proteína = proteína/100 g da tabela no site oficial × 300 g (a embalagem inteira é uma porção). Os valores batem com o selo da embalagem (17, 18, 20, 13, 17, 18 g).
- Só existe foto de embalagem. Na arte das "Marmitas" aparece a bandeja aberta, vista de cima, com a porção inteira. Recortei a bandeja e pintei de branco todo o fundo fora dela (`mask_tray`), então nome, selo de proteína e o padrão "Marmita" ficaram fora. Conferi cada recorte visualmente; restam só alguns pontos laranja sem texto legível na feijoada. Confiança B, porque é uma ilustração da porção e não uma foto do produto aquecido.
- Excluídos: linha **Protein** (marca "PROTEIN" em destaque, selo "30 g proteína" sobre a comida e prato cortado pela borda), **Panelinhas** (comida só parcialmente visível), lasanhas e massas (porção diferente da embalagem ou só garfada/fatia ilustrativa), empanados, pizzas e frios.

### Liv Up: 34 marmitas (lote 5, ids c001–c034)
- Lista de produtos tirada da home e das categorias de `/marmitas` em www.livup.com.br (90 páginas). Foram descartados 2 acompanhamentos (edamame, arroz jasmim) e 2 "proteínas" de 110 g sem foto. Dos 85 pratos completos, escolhi 34 distintos: um por receita, evitando variantes 370/430/580 g de foto quase igual. Entraram PF brasileiros, frango, carne, peixe e frutos do mar, massas e alguns vegetarianos para cobrir a faixa baixa.
- Gabarito = proteína/100 g da tabela nutricional da página × peso da embalagem (300–370 g; a embalagem é uma porção). Todas as tabelas usadas são "Porção de 100g". A URL da página e a da foto estão em `gabarito_pratos.csv`.
- Foto: foto de estúdio da marmita aberta (`livup.imgix.net/.../photos-original/...`, w=1200), recortada automaticamente na caixa para tirar o prato de salada e o copo que decoram a cena e não fazem parte da porção. Conferi as 34 imagens finais visualmente. Não há texto nutricional, peso nem alegação de proteína; aparece só o logo Liv Up e o slogan "Comer bem nunca foi tão fácil", impressos na caixa. O nome do arquivo da foto traz o mesmo peso da página em todos os casos.
- Confiança: A 30. B 4: o nome do arquivo da foto descreve uma versão anterior da receita (tilápia/saint peter com couscous, bobó de camarão, frango teriyaki, moqueca). Em 3 itens (2 escondidinhos e a lasanha) só a superfície é visível, o que está anotado em `observacoes`.

### Nutrition5k
- Amostra estratificada (seed fixa) de 10 pratos por faixa de proteína: 0–5, 5–15, 15–25, 25–40 e >40 g. Os candidatos são o split `rgb_test` com imagem overhead e massa ≥ 20 g. A imagem é o `rgb.png` zenital e o gabarito vem de `total_protein`/`total_mass` do CSV. Ingredientes em `descricao`.

## Fontes planejadas que não renderam (bloqueios)

- **Mr. Fit / marmitarias no 99Food (`99app.com`)**: Vercel Security Checkpoint (HTTP 429 com desafio JS). `www.mrfit.com.br` redireciona para `mrfit.com.br` (sem www), host não liberado. Nada coletado.
- **Liv Up via Pão de Açúcar**: só há foto da caixa, com a comida parcial atrás de textos. Foi excluído e substituído pelo site oficial depois que `livup.imgix.net` foi liberado (lote 5).
- **Pão de Açúcar**: o JSON do produto não traz tabela nutricional. Imagens (`static.paodeacucar.com`) são só de embalagem. A busca usa `api.vendas.gpa.digital` e linximpulse (não liberados).
- **Carrefour**: as APIs VTEX (`/api/catalog_system`, `/api/io/...intelligent-search`) devolveram 503. A busca é renderizada no cliente. As imagens vêm de `carrefourbrfood.vtexassets.com` (também servidas por `carrefourbr.vtexassets.com`).
- **Perdigão / Sadia**: HTTP 403 nos sites oficiais.
- **Swift**: a página tem pratos feitos de 300 g, mas as imagens estão em **`swiftbr.vteximg.com.br`**, host não liberado.
- **Open Food Facts**: a busca (`/api/v2/search` e facetas) deu 503 durante toda a coleta. A API de produto funciona, mas os produtos brasileiros consultados só têm foto de embalagem.
- **UFPR acervodigital**: erro de certificado TLS. O PDF da USP (e-disciplinas) exige login. Usei a cópia do nutritotal.

Para ganhar pratos compostos brasileiros com foto real, os passos seriam: liberar `swiftbr.vteximg.com.br` (Liv Up já incluído no lote 5), e abrir o 99Food com navegador.

---

# v2 — dados ampliados (2026-11-08)

Objetivo: ter mais pratos com **gabarito por componente**, para medir e corrigir os dois erros do v1. O primeiro é o peso de porções grandes de alimento único, subestimado. O segundo é a proporção de carne em prato composto, superestimada. Nada foi estimado nesta etapa: aqui só estão os dados, as ferramentas e os lotes cegos.
**Regra de uso de fontes** (exigida pela usuária): só **fotos reais de câmera** (nada sintético, gerado ou aumentado por IA, nem de proveniência incerta) e só rótulos **medidos** (pesagem ou laboratório), nunca estimados por modelo. Dataset que falha em um dos dois critérios fica de fora, com o motivo anotado abaixo.

## Resultado

| Fonte | Itens | Split | Gabarito | Confiança |
|---|---|---|---|---|
| Nutrition5k v2, café 1 | 328 | calib 100 · teste 228 | pesagem incremental por ingrediente + USDA (CSV do prato) | A |
| Nutrition5k v2, café 2 | 72 | calib 20 · teste 52 | massa total pesada; gramas por ingrediente fracionárias (rateio por receita); proteína = soma dos ingredientes (os totais do prato vêm zerados no CSV do café 2) | B |
| Itens do v1 (UFPR 103, Nutrition5k 50, Liv Up 34, Seara 7) | 194 | `v1` (fora do ajuste) | como no v1 | como no v1 |
| SNAPMe (USDA) | 150 selecionadas, **0 com imagem** | — | autorrelato ASA24 | C |
| **Total nos lotes cegos v2** | **594** | | | |

- `gabarito_componentes.csv` (`prato_id,componente,gramas,proteina_g,grupo,origem,confianca`): 2.870 componentes de 553 pratos. São 2.395 do Nutrition5k v2, 372 do Nutrition5k v1 e 103 do UFPR (um componente por foto). Liv Up e Seara não têm gabarito por componente.
- `gabarito_v2.csv`: os 594 pratos no formato do `eval/score_precisao.py`, com uma linha por ingrediente no Nutrition5k (soma = proteína do prato, diferença ≤ 0,005 g).
- `gabarito_pratos_v2.csv`: uma linha por prato com `origem`, `split`, `categoria`, peso, proteína, confiança e `lote_v2`. **É aqui que fica o mapeamento id → lote. As pastas dos lotes não têm nenhum rótulo.**
- `n5k_v2_amostra.csv`: os 400 pratos novos (`d0001…d0400`) com `dish_id`, split, café, estratos, massa, proteína e ingredientes em gramas.
- `snapme_selecao.csv`: 150 fotos do SNAPMe com o gabarito ASA24 (por foto e por alimento), prontas para quando as imagens puderem ser baixadas.

### Nutrition5k v2 (`coleta_n5k_v2.py`, seed 20261110)
- Candidatos: pratos com pasta `realsense_overhead` (3.490). Foram excluídos os 50 usados no v1, os pratos com massa < 20 g, os que têm ingrediente `deprecated`, `plate only` ou sem nome, e os que têm soma das proteínas (ou das gramas) dos ingredientes diferente do total do prato (±5% + 0,5 g / +2 g). Sobraram 3.353, dos quais 226 do café 2.
- A amostra é estratificada por proteína (0–5, 5–15, 15–25, 25–40, > 40 g; 80 por faixa) × nº de ingredientes (1, 2–4, 5+; ~27 por célula, com as sobras completadas dentro da faixa). Em cada célula, ~20% vêm do café 2 quando há pratos disponíveis. Só existem 7 pratos > 40 g com 1 ingrediente, e o café 2 tem apenas 37 pratos ≥ 25 g.
- O split é determinístico dentro de cada faixa de proteína: 24 calib + 56 teste, o que dá **120 calib / 280 teste**. As distribuições ficaram parecidas: proteína mediana de 20,2 g no calib e 19,4 g no teste; massa mediana de 200 g e 202 g.
- **Achado:** o `dish_metadata_cafe2.csv` traz `total_calories/fat/carb/protein = 0` em todos os 238 pratos (só a massa total vem preenchida) e 77 nomes de ingrediente vazios. Reconstruí os totais pela soma dos ingredientes e os nomes pelo `ingr_id` do `ingredients_metadata.csv`. No café 1, conferi que gramas × proteína/g do `ingredients_metadata` reproduz a proteína por ingrediente (27.224 linhas, mediana do desvio 0).
- Imagens: os 400 `rgb.png` (640×480, zenitais) foram baixados para o scratchpad. **Não estão no repositório.**

### Profundidade overhead (análogo ao LiDAR/TrueDepth): viável como dado, não como solução pronta
- `depth_raw.png` é 16 bits com 1 unidade = 0,1 mm, valores até ~4.000 (0,4 m) e ~14% de pixels sem leitura (mediana em 65 pratos). O dataset **não distribui intrínsecos da câmera**. O artigo (Thames et al., CVPR 2021) dá a distância câmera–base (35,9 cm) e a área por pixel nessa distância (5,957 × 10⁻³ cm²), o que basta para volume por soma de alturas.
- Teste rápido em 65 pratos do calib: plano de apoio pela mediana da borda da imagem, altura > 8 mm, volume = Σ altura × área do pixel. A correlação de log(volume) com log(massa pesada) foi de só **r = 0,36**. Para servir de verdade seria preciso ajustar o plano por RANSAC, segmentar o prato/bandeja, separar a comida do prato e usar a densidade de cada alimento. No celular a situação é melhor, porque o iPhone entrega intrínsecos calibrados em `AVDepthData`. Conclusão: dá para usar como trilha de pesquisa; não entra no pipeline v2.

### SNAPMe (USDA, Larke et al. 2023, *Nutrients* 15:4972): gabarito utilizável com ressalva, imagens bloqueadas
- O repositório `github.com/JulesLarke-USDA/SNAPMe` é **CC BY-SA 4.0**. O `input/master_SNAPME_linkfile.csv` liga cada foto (`filename`) aos alimentos do recordatório **ASA24** (código FNDDS, `FoodAmt` em g e `PROT`, entre outros): 4.761 linhas, 1.679 fotos, 95 participantes. As fotos são reais, de celular, tiradas pelas participantes.
- **O gabarito é autorrelatado, não pesado.** A participante escolhe a porção nas imagens e medidas do ASA24, e a proteína vem da FNDDS 2018. Por isso a confiança é **C**: serve para medir identificação e robustez em fotos do mundo real, **não** para calibrar peso de porção.
- `coleta_snapme.py` seleciona 150 fotos sem alimento embalado e com no máximo 6 alimentos, 30 por faixa de proteína e no máximo 3 por participante por faixa.
- **Imagens não obtidas.** O pacote de fotos está no Ag Data Commons, e `agdatacommons.nal.usda.gov` responde **HTTP 202 com desafio JavaScript do AWS WAF** (`x-amzn-waf-action: challenge`) a qualquer requisição sem navegador. Tentei de novo depois de alguns minutos, com o mesmo resultado. Para descobrir os ids dos arquivos (o download em si sai por `ndownloader.figshare.com`, que está liberado) seria preciso `api.figshare.com`, que está **bloqueado no proxy**. Caminhos possíveis: liberar `api.figshare.com` (se o Ag Data Commons expuser o artigo pela API pública do figshare) ou baixar o zip uma vez num navegador e copiar para o scratchpad.

### Hugging Face (busca na API: nutrition, nutrition5k, food weight, calorie, portion, snapme…)
| Dataset | Fotos reais? | Rótulo medido? | Decisão |
|---|---|---|---|
| **Codatta/MM-Food-100K** | Crowdsourcing (campanha Codatta × Binance); o próprio card traz `online_download_prob`, ou seja, parte das imagens pode ter sido baixada da web | **Não**: porção e nutrientes vêm de pré-anotação humana mais **GPT-4o/Qwen-max**, revisados por usuários; nada pesado | **Rejeitado** (rótulo estimado por modelo; proveniência das fotos incerta; OpenRAIL-M **não comercial**) |
| issai/Food_Portion_Benchmark (FPB) | Sim: RealSense D455 e celulares, em laboratório | **Peso pesado** por item (g); sem proteína | Utilizável só para **peso**, mas é cozinha da Ásia Central, a licença é **CC BY-NC 4.0** e os rótulos de teste estão ocultos. Não baixado; candidato a teste extra de peso por porção |
| zy12123/CalorieBench-80K | Card vazio | Desconhecido | **Rejeitado** (proveniência incerta) |
| pinkieseb/nutrition_dataset (266 mil imagens com massa e macros) | Sem card (parece derivado do Nutrition5k) | Desconhecido | **Rejeitado**; usar a fonte original |
| TeeA/nutrition5k-food-name-gemini, AustinNaung/…-processed | Derivados do Nutrition5k | Nomes gerados pelo Gemini | **Rejeitados** |
| ai5labsOfficial/nutrition5k, panikos/…, thi3rrr/…, shashs/… | Espelhos do Nutrition5k | Herdam o original | Sem necessidade: usamos o bucket original do Google |
| YashArya16/CalorieData, aryachakraborty/Food_Calorie_Dataset, sayanroy058/… | — | Texto de porção sem método, ou respostas de modelo | **Rejeitados** |
| arunapb/food-density-nutrition | Sem imagens | Densidade e nutrientes por classe do FoodSeg103 (tabela) | Fora do escopo |

### Referências brasileiras (`ref/`, gerado por `ref/monta_ref.py`)
- `taco.csv`: TACO 4ª ed. (NEPA/UNICAMP, 2011), com 597 itens e as colunas `id,nome,grupo,proteina_100g,kcal_100g,lipidios_100g,carboidrato_100g,fibra_100g,umidade_100g`. Vem do CSV de `raulfdm/taco-api` (código MIT). A TACO é de uso livre com citação. **Conferência: 10/10 valores batem** com a 4ª ed.: arroz tipo 1 cozido 2,5; feijão carioca cozido 4,8; peito de frango sem pele grelhado 32,0; ovo cozido 13,3; pão francês 8,0; patinho grelhado 35,9; feijão preto cozido 4,5; queijo minas frescal 17,4; batata inglesa cozida 1,2; banana prata 1,3. Ressalva: 15 itens têm proteína vazia na fonte (óleos, sal, leite fluido integral e desnatado UHT, iogurte de abacaxi, aguardente, coco verde). O pipeline os ignora; para leite, use `USDA: milk`.
- `medidas_ibge.csv`: IBGE, POF 2008-2009, *Tabela de Medidas Referidas para os Alimentos Consumidos no Brasil* (`tabelamedidas_bd.zip` do ftp.ibge.gov.br), com 11.801 linhas de alimento × preparação × medida caseira → gramas. As 722 linhas `PORCAO` são as porções típicas usadas como prior em `calibra.py`.
- `composicao_ibge.csv`: IBGE, POF 2008-2009, *Tabelas de Composição Nutricional dos Alimentos Consumidos no Brasil* (`tabelacompleta.zip`), com 1.971 combinações de alimento × preparação, proteína e kcal por 100 g, montada a partir de TACO e USDA. É complemento para preparações que a TACO não tem (ex.: macarrão cozido 5,8 g). Dados públicos do IBGE: reprodução permitida com citação da fonte.
- `n5k_ingredients.csv`: cópia do `ingredients_metadata.csv` do Nutrition5k (555 ingredientes, proteína por g da USDA; CC BY 4.0), usada para nomes genéricos.
- **TBCA (USP/FoRC)**: `www.tbca.net.br` responde, mas só tem consulta item a item na web. A base completa exige assinar o "Termo de Confidencialidade TBCA". Não raspei o site. A composição do IBGE cobre as preparações que faltavam.

## Pipeline v2 (`eval/`)
1. `prompt_v2.md`: instrução ao estimador cego. A saída é **por componente** (`prato_id,componente,alimento_ref,gramas_estimadas,confianca,pista_escala`), com nome TACO ou genérico USDA, e **sem cálculo de proteína**. Inclui âncoras de escala (diâmetro de prato, talheres, marmitas, medidas caseiras do IBGE), o aviso de que porções grandes de carne costumam ser subestimadas e de que prato composto costuma ter menos carne do que parece, e listas de nomes conferidos contra as tabelas.
2. `protein_from_components.py`: casa `alimento_ref` com a TACO e com o Nutrition5k/USDA. A ordem é: sinônimos, nome exato, sinônimo contido no nome e, por último, pontuação difflib + tokens + palavra-núcleo, com penalidade para itens crus. Grava a proteína por prato (formato do `score_precisao.py`) e por componente, e lista os nomes sem casamento em `<saida>.nao_casados.txt`.
3. `calibra.py`: ajusta **só no calib** (multiplicador por grupo × tipo de prato, log-linear por grupo, ou encolhimento para a porção típica do IBGE), escolhe o modelo pelo erro no calib e reporta antes/depois no **teste** e nos itens do v1, todos fora do ajuste.
4. `simula_seletor_componentes.py`: seletor de 6 tamanhos **no componente com mais proteína** (e, numa variante, nos dois primeiros), comparado com a foto sozinha e com o seletor do prato inteiro do v1.
5. `grupos_alimentos.py`: grupos grossos (proteína animal, leguminosa, laticínio, amido, vegetal, gordura/molho, misto, outros), definidos por palavra-chave em pt/en e pelo grupo da TACO.

Teste de fumaça: rodei as quatro ferramentas de ponta a ponta com um estimador **sintético** (gerado do gabarito com ruído; fica só no scratchpad e não é resultado). Todas rodaram, e só 1 nome ficou sem casamento (`deprecated`).

Reproduzir a calibração quando chegarem as estimativas cegas:
```
python3 eval/protein_from_components.py est_v2.csv pratos_v2.csv --componentes comp_v2.csv --sistema nuvem_llm_v2
python3 eval/score_precisao.py docs/dados/precisao/gabarito_v2.csv pratos_v2.csv
python3 eval/calibra.py comp_v2.csv --saida pratos_v2_calibrado.csv --parametros calib.json
python3 eval/simula_seletor_componentes.py comp_v2.csv
```

## Lotes cegos v2
- `scratchpad/cego2/lote01 … lote15`: 594 imagens, 40 por pasta (a lote15 tem 34). Os itens vêm todos embaralhados juntos (seed 20261112). Os 400 novos têm ids `d0001…d0400`, sorteados sem relação com o conteúdo; os 194 do v1 mantêm os ids antigos (`b…`, `n…`, `c…`). As pastas têm **só imagens**. O mapeamento id → lote e o gabarito ficam somente em `gabarito_pratos_v2.csv`.
- Montagem: `python3 monta_gabarito_v2.py <dir_n5k> <img_n5k_v2> <scratchpad>/cego <scratchpad>/cego2`.

## Hosts
- Usados: `storage.googleapis.com` (gsutil), `raw.githubusercontent.com`, `github.com` (git clone), `huggingface.co`, `ftp.ibge.gov.br`, `www.tbca.net.br`.
- Bloqueios: `agdatacommons.nal.usda.gov` (desafio AWS WAF, não é bloqueio do proxy) e `api.figshare.com` (bloqueado no proxy). A API REST do GitHub (`gh api`) recusou o repositório do SNAPMe por política da sessão; o `git clone` público funcionou.
