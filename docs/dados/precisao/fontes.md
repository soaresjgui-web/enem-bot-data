# Fontes para o conjunto de teste de precisão do Nutrelle (proteína a partir de foto)

Data do levantamento: 2026-11-08. Escopo: **descoberta de fontes** (nenhuma imagem foi baixada, nenhum dataset montado).

Objetivo: ≥100 pratos **brasileiros** com (a) foto do prato/porção real e (b) ground truth confiável de peso total e proteína (g) daquela porção — idealmente com pesos por componente — mais um subconjunto opcional de ~50 pratos do Nutrition5k para calibração.

## 0. Como o levantamento foi feito e o que limita a confiança

- A rede do sandbox é allowlist. Durante este levantamento só resolviam `github.com`, `raw.githubusercontent.com` e `storage.googleapis.com` (via `gsutil`), além dos registries de pacotes. **Todos os sites brasileiros (Liv Up, Mr. Fit, Pão de Açúcar, 99Food, FatSecret, USP, UFPR, Zenodo, Mendeley, Kaggle, Hugging Face, arXiv…) foram bloqueados (`403` do proxy)**.
- Portanto, para as fontes brasileiras, as conclusões vêm de **snippets de busca web** (títulos, trechos indexados, URLs). Onde digo "a confirmar", é porque não abri a página. O Nutrition5k, ao contrário, foi inspecionado diretamente no bucket GCS (contagens, colunas e estatísticas abaixo são medidas, não estimadas).
- Nível de confiança por ficha: **A** = verificado diretamente; **B** = inferido de vários snippets consistentes; **C** = inferido de 1 snippet ou de conhecimento prévio.

## 1. Marmitas / pratos prontos (foto do prato + tabela nutricional por porção)

Esta é a categoria mais promissora: produtos embalados têm tabela nutricional obrigatória (ANVISA) **por porção e por 100 g**, peso líquido declarado e foto do produto. Caveat universal: a foto de e-commerce é frequentemente "sugestão de apresentação" — é preciso **marcar manualmente** se a foto mostra a porção vendida (marmita aberta) ou só a embalagem.

### 1.1 Mr. Fit (via listagens do 99Food em `99app.com`) — confiança B

- URLs: `https://99app.com/99food/<cidade>/<loja>/<id>/` — ex.: `https://99app.com/99food/rio-de-janeiro/mr-fit-rocha-miranda-marmitas-congeladas/5764609000302447448/`, `https://99app.com/99food/campinas/mr-fit-alimentaao-saudavel-hortolandia/5764608387862758636/`, `.../mr-fit-alimentaao-saudavel-ipanema/5764609520215787268/`, `.../mr-fit-sao-gonalo/5764611244959729715/`, `.../mr-fit-alimentaao-saudavel-nova-campinas/5764613920669829860/`, `.../mr-fit-alimentaao-saudavel-campo-grande/5764609591242131191/`.
- O que há: páginas **renderizadas no servidor** (indexadas pelo buscador com texto completo), uma por unidade franqueada, com cada marmita listada com foto + descrição no padrão "**400g, 62g de Proteínas, 392 kcal**". Exemplos capturados: Sobrecoxa Fit 400 g / 62 g prot / 392 kcal; Parmegiana Light 400 g / 66 g / 416 kcal; Mr. Alcatra 400 g / 66 g / 396 kcal; Feijoada Light 400 g / 59 g / 372 kcal; Panqueca Lowcarb Frango 300 g / 48 g / 278 kcal; Mr. Maromba 70 g prot; Tilápia (Mr. Metabolismo) 56 g / 352 kcal; Chicken Lowcarb 300 g / 243 kcal.
- GT: **por porção = embalagem inteira** (300–400 g). Sem pesos por componente.
- Nº estimado de pratos distintos: 25–40 (o cardápio Mr. Fit repete entre unidades).
- Brasilidade: alta (feijoada light, strogonoff, parmegiana, panqueca, PF fit).
- Risco de qualidade: **divergências entre unidades** (Chicken Lowcarb aparece com 62 g e com 48 g de proteína; Strogonoff Light com 60 g e 66 g). Regra sugerida: aceitar só valores que coincidam em ≥2 listagens ou com o site oficial `mrfit.com.br` (não verificado; provavelmente lista produtos com tabela).
- Licença: conteúdo comercial da marca; uso interno para avaliação é razoável, **não republicar imagens** — guardar URL + rótulos.
- Hosts: `99app.com` (+ CDN de imagens a identificar no HTML após liberar; `www.mrfit.com.br` para o site oficial).
- Dificuldade: baixa (HTML estático indexável).

### 1.2 Liv Up — via Pão de Açúcar (`paodeacucar.com`) e site próprio — confiança B

- URLs de produto no Pão de Açúcar: `https://www.paodeacucar.com/produto/<id>/<slug>` — ex.: `/produto/1640875/liv-up-carne-moida-pure-de-mandioquinha-e-legumes-assados-300g`, `/produto/1606919/sobrecoxa-com-ervilha-pure-de-batata-e-cenoura-liv-up-300g`, `/produto/1624591/estrogonofe-de-frango-com-batata-liv-up-300g`, `/produto/1645800/marmita-liv-up-baixa-caloria-frango-em-tiras-300g`, `/produto/1606922/carne-moida-arroz-integral-e-feijao-carioca-liv-up-300g`, `/produto/1640874/liv-up-salmao-em-lascas-com-molho-branco-arroz-e-legumes-300g`.
- O que há: foto do produto + **tabela nutricional por porção** (os snippets de lasanhas no mesmo site mostram "porção de 300 g … Proteínas 17 g"). Marmitas Liv Up são de 300 g (linha Dia a Dia / Baixa Caloria) e 430–580 g (linha Performance). Notícia de 2025: ~20 opções de pratos prontos Liv Up nas lojas GPA.
- Complemento sem foto: `tabelatacoonline.com.br/tabela-nutricional/marcas/liv-up-marmita-...` lista **30+ marmitas Liv Up** por porção (Dia a Dia, Performance, Performance Pro, Baixa Caloria, Vegetariana, Especiais) — útil para cruzar valores; `fatsecret.com.br/calorias-nutrição/liv-up/...` lista por 100 g (crowd-sourced).
- Site próprio `www.livup.com.br`: bloqueado no sandbox; não indexado pelo buscador (provavelmente SPA/Next.js com catálogo via API) — **a confirmar**.
- GT: por porção (300 g); sem componentes. Brasilidade: alta (arroz, feijão, carne moída, estrogonofe, frango).
- Nº estimado: 20–35 pratos com foto + tabela.
- Licença: comercial; guardar só URL + rótulos.
- Hosts: `www.paodeacucar.com` (site React; imagens provavelmente em `static.paodeacucar.com` — a confirmar; pode haver API em `api.gpa.digital` — a confirmar), `www.livup.com.br`, `www.tabelatacoonline.com.br`.
- Dificuldade: média (Pão de Açúcar é JS-renderizado, mas as páginas são indexadas; provavelmente JSON embutido ou API pública).

### 1.3 Refeições congeladas de indústria: Seara (Protein, Gourmet, lasanhas), Perdigão Meu Menu, Sadia, Swift Good4You — confiança B/C

- Seara Protein: "primeira linha de refeições proteicas congeladas"; após expansão **13 produtos**, 15–36 g de proteína por porção, embalagens de 150 g e 180 g; ex. "frango com legumes e arroz com brócolis — 30 g proteína / 329 kcal". Lasanhas Seara 600 g (porção 300 g; ex. 4 Queijos: 17 g prot / 418 kcal), Lasanha Airfryer 300 g.
- Perdigão Meu Menu: **8–9 refeições completas de 300 g**, lançamentos com 42 g de proteína por porção (ex.: feijoada e arroz com couve; carne moída com purê e arroz com brócolis; fettuccine com peru e brócolis; nhoque à bolonhesa).
- Sadia: lasanhas 350/600 g, almôndegas, pratos congelados; Swift: linha Good4You (porções individuais, só e-commerce da marca; ex. "Prato feito Strogonoff de Frango com Arroz", "Costela e Mandioca com Arroz").
- Onde pegar foto + tabela: páginas de produto em `paodeacucar.com` (confirmado por snippets que exibem tabela por porção), `mercado.carrefour.com.br` (não apareceu indexado — a confirmar), sites oficiais `seara.com.br`, `perdigao.com.br`, `sadia.com.br`, `swift.com.br` (não verificados). Open Food Facts tem alguns códigos (ex. `world.openfoodfacts.org/product/7893000674073` lasanha), mas com **foto de embalagem**, não do prato.
- Caveat importante: a foto é quase sempre da embalagem/sugestão; para lasanhas a porção (300 g) é **metade** do produto — a foto não corresponde à porção. Usar preferencialmente **refeições individuais** (Seara Protein, Meu Menu, Good4You, Liv Up), cuja porção = embalagem inteira.
- Nº estimado utilizável (refeições individuais): 20–30. Brasilidade: alta.
- Hosts: `www.paodeacucar.com`, `mercado.carrefour.com.br` (+ `carrefourbr.vtexassets.com` provável CDN — a confirmar), `www.seara.com.br`, `www.perdigao.com.br`, `www.sadia.com.br`, `www.swift.com.br`.

### 1.4 Outras marmitarias no 99Food (`99app.com`) — confiança C

- Páginas: Tribo Fit (RJ; "marmitas de 400 g com 150 g de frango" — útil como peso de componente), Easy Fit (200/400 g), Athletic Food Tijuca (~400 g), Gelato Fit (300 g), Viva Leve (Salvador), ElumaaFit (Campinas; ex. "Frangolino 384 kcal, 64 g proteína, 400 g"), Nutri Ponto (Goiânia), Happy Sabor, Mamma Fit, Mais Fit Mania (SP).
- Só uma parte informa proteína; valores são declarados pela loja (sem laudo). Útil para +10–15 pratos de PF brasileiro com foto, com confiança menor. Sebrae tem e-book "como fazer marmitas fit" (não é fonte de GT).

### 1.5 iFood — confiança C

- Páginas `ifood.com.br/delivery/<cidade>/<loja>/<uuid>` são SPA; catálogo vem de API (`marketplace.ifood.com.br`), imagens em `static.ifood-static.com.br` (de memória — a confirmar). Há scrapers comerciais (Apify, Bright Data) e um repo `ricardotachinardi/ifood-scraper`. Poucas lojas publicam macros. **Baixa prioridade**; só se faltarem pratos.

### 1.6 Agregadores nutricionais (sem foto confiável; apenas para cruzar valores)

- `fatsecret.com.br` (crowd-sourced, declara não garantir precisão; muitos itens por 100 g), `tabelatacoonline.com.br/tabela-nutricional/marcas/...` (por "1 porção", frequentemente sem gramas no snippet), `vitat.com.br` (dados com erros, ex. 0 kcal), `arise-app.com` (estimativas). Não usar como GT primário.

### 1.7 Open Food Facts (BR) — confiança B (conhecimento da plataforma)

- Dados ODbL, fotos CC BY-SA (única fonte aqui cuja **republicação de imagens é permitida**, com atribuição e share-alike). API pública (`world.openfoodfacts.org/api/v2/product/<ean>.json`), imagens em `images.openfoodfacts.org`. Porém as fotos são de **embalagem**, não do prato empratado — serve para GT de rótulo (porção, proteína), não para a foto a ser avaliada. Útil para validar rótulos de Seara/Sadia/Perdigão/Liv Up.
- Hosts: `world.openfoodfacts.org`, `br.openfoodfacts.org`, `images.openfoodfacts.org`.

## 2. Redes de restaurantes — resultado majoritariamente negativo

Pesquisei Madero, Outback, Spoleto, Giraffas, China in Box, Habib's, Subway, McDonald's, Burger King, Bob's, Coco Bambu, Divino Fogão, Montana Grill, Vivenda do Camarão. **Nenhuma tabela nutricional oficial por prato apareceu indexada**; os resultados são FatSecret/Tabela TACO (não oficiais).

- Outback: há reclamação no Reclame Aqui ("não disponibilizam tabela nutricional"); o PDF oficial de café da manhã (`cdn.outback.com.br/wp-data/wp-content/uploads/2026/02/Outback_cafedamanha.pdf`) remete a uma "tabela completa no site" (não localizada). Existe a **"Outback – Tabela de Gramaturas"** oficial (`https://s3.amazonaws.com/institucional-statics-files/site/uploads/Outback-Tabela-de-Gramaturas.pdf`) com pesos aproximados por item (ex. Arroz Pilaf ≈ 200 g) — sem nutrientes. Combinável com fotos do cardápio + TACO/TBCA para proteína estimada (confiança baixa).
- McDonald's BR: blogs indicam que a informação nutricional está no site/app oficial (`mcdonalds.com.br`), mas a página não apareceu; Burger King e Subway BR idem. Fast food tem baixa brasilidade e pouco valor para "prato de comida"; **baixa prioridade**.
- Conclusão: não contar com redes para o núcleo do conjunto; no máximo 5–10 itens de Outback (gramaturas oficiais) como teste adicional.

## 3. Fontes acadêmicas brasileiras (atlas fotográficos de porções com gramas)

Não existe (até onde a busca alcança) dataset brasileiro de **pratos completos** com peso e nutrientes por prato. O que existe são **álbuns fotográficos de porções** (foto de alimento único em porções pesadas, em gramas) — muito úteis: a proteína da porção vem de TACO/TBCA aplicada ao peso conhecido, e a foto é de porção real pesada.

### 3.1 Manual Fotográfico de Quantificação Alimentar (UFPR, Crispim et al., 2017; álbum GloboDiet-Brasil/IARC) — confiança B

- PDF: `https://edisciplinas.usp.br/pluginfile.php/5296370/mod_resource/content/1/Manual%20Fotogra%CC%81fico%20de%20Quantificac%CC%A7a%CC%83o%20Alimentar.pdf` (cópia em disciplina da USP, 147 p., 1ª ed.); também em `https://nutritotal.com.br/pro/material/manual-fotografico-de-quantificacao-alimentar/` (download, possivelmente com cadastro). Referência: Curitiba, UFPR, 2017.
- Conteúdo: séries de fotos de porções (padrão GloboDiet: 4–6 tamanhos por alimento, cada um com **peso em gramas**), fotos a 45°, alimentos escolhidos a partir do Guia Alimentar e da POF 2008-09; planejamento previa ~91 fotos (séries) no álbum brasileiro. Fotos por fotógrafo profissional, logo após o preparo. Validações: Nichelle et al. 2019 (*Nutrients*), Ferreira et al. 2021 (*Measurement: Food*) — fotos válidas em geral, pior para almôndega; tese UFPR 2019 (`acervodigital.ufpr.br/handle/1884/64688`).
- Uso: ~40–90 fotos de alimentos brasileiros (arroz, feijão, carnes, farofa, purê…) com gramas → proteína via TACO/TBCA. Não são "pratos", mas são exatamente o cenário de porção única pesada; serve também para testar o modelo componente a componente.
- Licença: direitos da UFPR/autores; sem licença aberta declarada. **Uso interno OK; não republicar imagens** (guardar nº da página/série + gramas).
- Hosts: `edisciplinas.usp.br`, `nutritotal.com.br`, `acervodigital.ufpr.br`.
- Versão infantil (ENANI/UFRJ): 101 alimentos e 81 medidas caseiras, porções infantis — `enani.nutricao.ufrj.br` (menos útil).

### 3.2 Dissertação UnB (Lopez, 2007) "Estimativas de porções alimentares: elaboração e teste de um procedimento com registro fotográfico" — confiança C

- PDF aberto: `https://repositorio.unb.br/bitstream/10482/3354/1/2007_RosanaPosseSLopez.pdf`. Base do livro "Álbum fotográfico de porções alimentares" (Lopez & Botelho, Metha, 2008; impresso, 239 p., não disponível online). Verificar se os anexos trazem as fotos com gramas (porções P/M/G). Host: `repositorio.unb.br`.

### 3.3 UFRN — "Registro fotográfico de porções de preparações culinárias habitualmente consumidas pela população" (Santos & Ribeiro) — confiança C

- `https://acervo.ufrn.br/Record/ri-123456789-34785/Description` (TCC; fichas técnicas + fotos das porções das preparações do Guia Alimentar; porções pesadas). Também "Manual Técnico: Padronização e registro fotográfico de medidas caseiras de Frutas e Hortaliças" (`acervo.ufrn.br/Record/ri-123456789-40009`). Hosts: `acervo.ufrn.br`, `repositorio.ufrn.br`. Texto completo não confirmado.

### 3.4 Outros registros (apenas impressos / regionais)

- Zabotto, Vianna & Gil (1996) "Registro fotográfico para inquéritos dietéticos: utensílios e porções" (Metha/UNICAMP) — impresso. Monteiro et al. (2007) "Consumo alimentar: visualizando porções" (Guanabara Koogan) — impresso. Posters UNICESUMAR (registro fotográfico regional, porções nos percentis 25/50/75, pesadas em balança de 1 g) — PDFs em `unicesumar.edu.br` sem o álbum completo.

### 3.5 Datasets de imagens de comida brasileira **sem** peso/nutriente (só para teste qualitativo)

- MyFood (UFPR/SIBGRAPI 2020; `zenodo.org`; 1.250 imagens, 9 classes: arroz, feijão, ovo cozido/frito, massa, salada, carne assada, maçã, peito de frango; máscaras de segmentação). Sem GT de peso.
- "Brazilian food images" (Mendeley, `data.mendeley.com/datasets/7n36jtcpv3/2`; 20 classes, máscaras). Sem GT de peso.
- Nenhum dataset no Kaggle/Hugging Face com comida brasileira + nutrientes foi encontrado.

## 4. Conjuntos internacionais com pesagem (calibração)

### 4.1 Nutrition5k (Google) — confiança A (inspecionado via `gsutil`)

- Bucket: `gs://nutrition5k_dataset/nutrition5k_dataset/` (host HTTPS: `storage.googleapis.com`; `gsutil ls`/`cp` já funcionam no sandbox). Repo: `https://github.com/google-research-datasets/Nutrition5k`. Licença: **CC BY 4.0** (imagens e metadados — republicar é permitido com atribuição).
- Estrutura: `metadata/dish_metadata_cafe1.csv` (2,2 MB) e `dish_metadata_cafe2.csv` (0,1 MB) — **sem cabeçalho**; colunas: `dish_id, total_calories, total_mass(g), total_fat, total_carb, total_protein(g)`, depois, por ingrediente, `ingr_id, ingr_name, grams, calories, fat, carb, protein` (repetido). `metadata/ingredients_metadata.csv` (nutrientes por g, USDA). `dish_ids/splits/{rgb,depth}_{train,test}_ids.txt`. `imagery/realsense_overhead/<dish_id>/{rgb.png, depth_raw.png, depth_color.png}` (~410 KB + 190 KB + 290 KB por prato). `imagery/side_angles/` (vídeos, 4 câmeras; tar completo 181 GB — não precisamos).
- Contagens medidas: **5.006 pratos** com metadados; **3.490 pratos com imagem overhead RGB**; split `rgb_test_ids.txt` = 709 ids. Mediana de 3 ingredientes/prato.
- Distribuição medida: proteína mediana **7,1 g**, média 14,5 g, p10 0,4 g, p90 39,4 g, máx 147,5 g; **1.690 pratos com ≥15 g** e 1.069 com ≥25 g; massa mediana 145 g (p10 41 g, p90 381 g). Ou seja, muitos pratos são porções pequenas/baixa proteína — amostrar estratificado por proteína.
- Proposta de subconjunto (~50 pratos): ids em `rgb_test_ids.txt` ∩ pastas de `realsense_overhead`, estratificar em 5 faixas de proteína (0–5, 5–15, 15–25, 25–40, >40 g), 10 por faixa; baixar só `rgb.png` (~20 MB no total; com depth ~45 MB).
- Caveats: visão zenital a distância fixa de bandeja de refeitório (EUA) — distribuição diferente de foto de celular; proteína vem de pesagem incremental + USDA (ótimo GT por componente).

### 4.2 NutritionVerse-Real (Waterloo, 2024) — confiança B

- 889 imagens, 251 pratos, 45 alimentos; **pesos por ingrediente medidos**, nutrientes via rótulo/Canada Nutrient File. Download: `https://www.kaggle.com/datasets/nutritionverse/nutritionverse-real` (requer login Kaggle; host `www.kaggle.com`). Licença não explicitada — verificar na página. Fotos a vários ângulos (mais parecidas com celular que o Nutrition5k).

### 4.3 ECUSTFD (2017) — confiança B

- 2.978 imagens, 19 alimentos **simples** (frutas, pão etc.), peso em balança + volume; `github.com/Liang-yc/ECUSTFD-resized-` (host `github.com`; licença não declarada). Sem proteína direta (derivar por tabela). Pouco relevante para pratos compostos.

### 4.4 MM-Food-100K (Codatta, 2025) — confiança B

- 100k imagens, nutrição (kcal, P/G/C) e porção **anotadas por contribuidores, não pesadas**; licença OpenRAIL; `huggingface.co/datasets/Codatta/MM-Food-100K`. Baixa confiança como GT; não usar para calibração.

### 4.5 SNAPMe (USDA, 2023) — confiança B

- 3.311 fotos de refeições reais ligadas a 275 registros ASA24 de 95 participantes — **autorrelato**, não pesagem; domínio público (Ag Data Commons, `agdatacommons.nal.usda.gov`). Útil só como teste "do mundo real" com GT ruidoso.

## 5. Shortlist ranqueada e mix proposto

| # | Fonte | Papel | Pratos-alvo | Confiança do GT | Foto = porção real? |
|---|-------|-------|-------------|-----------------|---------------------|
| 1 | Mr. Fit via 99Food (+ `mrfit.com.br`) | núcleo PF brasileiro, alta proteína | 30 | média (cruzar ≥2 listagens) | sim (marmita aberta, a checar) |
| 2 | Liv Up via Pão de Açúcar (+ `livup.com.br`, Tabela TACO para cruzar) | núcleo PF brasileiro | 25 | alta (rótulo) | geralmente sim |
| 3 | Seara Protein + Perdigão Meu Menu + Swift Good4You (PdA / Carrefour / sites) | refeições individuais industrializadas | 20–25 | alta (rótulo) | parcial (checar) |
| 4 | Manual Fotográfico UFPR 2017 (porções pesadas) | alimentos brasileiros isolados, porções em g | 20–40 fotos | alta (pesado) + TACO | sim |
| 5 | Outras marmitarias 99Food (Tribo Fit, ElumaaFit, Easy Fit…) | reforço de variedade | 10–15 | baixa–média | sim |
| 6 | Outback "Tabela de Gramaturas" + fotos do cardápio | teste extra de restaurante | 5–10 | baixa (proteína estimada) | não (foto institucional) |
| 7 | Nutrition5k (CC BY 4.0) | calibração internacional | 50 | muito alta (pesado por componente) | sim (zenital) |
| 8 | NutritionVerse-Real | calibração extra, ângulos variados | opcional 20 | alta | sim |

Total brasileiro esperado: **105–145 itens** (1+2+3+4+5), dos quais ~75–95 pratos compostos e 20–40 porções de alimento único. Lasanhas 600 g, Open Food Facts e agregadores ficam como **validação de rótulo**, não como itens do conjunto.

Esquema de rótulo sugerido por item: `fonte, url_pagina, url_imagem, nome, peso_porcao_g, proteina_g, kcal, base (porção/100 g), componentes[{nome, g}] (quando houver), foto_mostra_porcao (sim/não/parcial), confianca (A/B/C), data_captura`. Como o repositório é público, **guardar só URLs + rótulos** para fontes comerciais/acadêmicas; imagens só do Nutrition5k (CC BY 4.0) e, se usado, Open Food Facts (CC BY-SA).

## 6. Hosts a liberar na rede do sandbox

Já acessíveis: `github.com`, `raw.githubusercontent.com`, `storage.googleapis.com`.

Prioridade alta (núcleo brasileiro):
- `99app.com` (+ CDN de imagens do 99Food, a identificar no HTML)
- `www.paodeacucar.com`, `static.paodeacucar.com` (CDN provável), `api.gpa.digital` (API provável)
- `www.livup.com.br`
- `www.mrfit.com.br`
- `mercado.carrefour.com.br`, `carrefourbr.vtexassets.com` (CDN provável)
- `www.seara.com.br`, `www.perdigao.com.br`, `www.sadia.com.br`, `www.swift.com.br`
- `edisciplinas.usp.br`, `nutritotal.com.br`, `acervodigital.ufpr.br`

Prioridade média (cruzamento e acadêmicos):
- `www.tabelatacoonline.com.br`, `www.fatsecret.com.br`
- `repositorio.unb.br`, `acervo.ufrn.br`, `repositorio.ufrn.br`, `enani.nutricao.ufrj.br`
- `world.openfoodfacts.org`, `br.openfoodfacts.org`, `images.openfoodfacts.org`
- `s3.amazonaws.com` (PDF de gramaturas do Outback), `cdn.outback.com.br`

Opcional (calibração extra):
- `www.kaggle.com` (NutritionVerse-Real), `huggingface.co`, `zenodo.org`, `data.mendeley.com`, `agdatacommons.nal.usda.gov`
- `www.ifood.com.br`, `marketplace.ifood.com.br`, `static.ifood-static.com.br` (baixa prioridade)

Após liberar, o primeiro passo deve ser abrir 1 página de cada host e registrar o hostname real das imagens (`<img src>`/JSON) para completar a lista.

## 7. Riscos e cuidados

1. **Foto ≠ porção**: e-commerce usa fotos de embalagem/sugestão. Exigir revisão manual (campo `foto_mostra_porcao`).
2. **Peso**: rótulo traz peso líquido; marmitas ultracongeladas perdem água no aquecimento — considerar tolerância (±10%) ao comparar com estimativa do app.
3. **Inconsistência de declarações** (Mr. Fit entre unidades; FatSecret crowd-sourced). Priorizar rótulo de fabricante (Liv Up, Seara, Perdigão) e exigir concordância dupla para lojas.
4. **Direitos autorais**: imagens comerciais e do manual UFPR não podem ser republicadas no repositório público; apenas Nutrition5k (CC BY 4.0) e Open Food Facts (CC BY-SA) permitem.
5. **Termos de uso / scraping**: 99Food e iFood têm termos restritivos; volume pequeno e uso interno de avaliação; evitar automação agressiva (poucas centenas de requisições).
6. **Lacuna**: não há dataset brasileiro com pratos compostos pesados; se o teste exigir pesos por componente em pratos brasileiros, a alternativa é **coleta própria** (balança + foto de ~50 PFs), que pode ser calibrada com o Nutrition5k.

## 8. Próximos passos sugeridos

1. Liberar os hosts de prioridade alta; confirmar renderização (HTML vs JSON) e CDNs de imagem de 99Food, Pão de Açúcar e Liv Up.
2. Baixar `rgb_test_ids.txt` + CSVs do Nutrition5k (já possível) e amostrar os 50 pratos estratificados; baixar só `rgb.png`.
3. Montar planilha de rótulos com o esquema da seção 5; começar por Mr. Fit (99Food) e Liv Up (Pão de Açúcar).
4. Extrair do Manual UFPR as séries de alimentos-base (arroz, feijão, carnes, farofa, purê, macarrão) com gramas e calcular proteína por TACO/TBCA.
