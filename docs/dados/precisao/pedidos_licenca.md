# Pedidos de licença e de acesso a bases de pratos pesados

> Rascunhos para a fundadora enviar. As bases abaixo têm comida pesada, mas a licença não permite uso comercial ou o acesso não é público (ver [`fontes_treino.md`](fontes_treino.md)). Os e-mails vão em inglês, que é a língua de trabalho dos autores. Os endereços dos autores não ficam neste repositório, que é público: estão na página ou no artigo de cada base.

## Ordem sugerida

| # | Base | O que tem | O que pedir | Por que vale |
|---|---|---|---|---|
| 1 | OmniFood8K | 8.036 cenas reais de pratos chineses, ingredientes pesados | Licença comercial para treino (hoje CC BY-NC 4.0) | Maior volume de pratos pesados que existe |
| 2 | Food Portion Benchmark (ISSAI) | 14.083 imagens com peso medido | Licença comercial para treino (hoje CC BY-NC 4.0) | Volume; sem proteína, serve para o peso |
| 3 | MetaFood3D (Purdue) | 637 alimentos com peso e nutrientes | Licença comercial (hoje CC BY-NC 4.0, com formulário) | Tem proteína medida |
| 4 | FLIC (UNIMIB) | 401 bandejas de refeitório pesadas, foto de cima | Onde baixar e qual a licença | Mesmo formato do nosso uso |
| 5 | ACETADA (Curtin) | 806 fotos de celular de refeições pesadas | Se a base será publicada e sob qual licença | Fotos de celular, o domínio real |

| 6 | ClinicPlate (Zenodo, registro 21626434) | 9.868 fotos de porções hospitalares pesadas a 1 g, com macronutrientes; três dos sete alimentos são fontes de proteína, em dez tamanhos de porção | **Pedir acesso** no próprio Zenodo (a licença já é CC BY 4.0; só o acesso é restrito) | É o pedido mais simples e o de retorno mais provável: basta uma conta no Zenodo e uma frase dizendo o uso |
| 7 | SNAPMe (USDA) | 1.163 fotos de celular de refeições reais, com nutrientes declarados pelos participantes | Perguntar se aceitam o uso como rótulo fraco de treino | Os autores pedem que a base não seja usada para treinar modelos; a licença é CC BY-SA 4.0 |

**Correções depois da segunda busca (12/11/2026):** a ACETADA é CC BY-NC 4.0 e já está publicada (num servidor da Purdue), então o pedido 5 passa a ser de licença comercial, não de disponibilidade. A SNAPMe não é de domínio público, como constava antes.

Antes de enviar: confirmar no site de cada base que a licença e o contato continuam os mesmos.

## O que dizer em todos

- Quem somos: uma empresa brasileira pequena fazendo um app de nutrição para mulheres (Nutrelle), que estima a proteína de uma refeição a partir de uma foto.
- O que queremos fazer com os dados: **treinar e avaliar** um modelo; não redistribuir as imagens nem os rótulos.
- O que oferecemos: citação do artigo no app e na documentação, relato dos resultados (inclusive os negativos), e disposição para pagar uma licença ou assinar um acordo de uso.
- O que perguntar se a resposta for não: se ao menos o uso para **avaliação interna** é permitido.

## 1. OmniFood8K

**Subject:** Commercial training licence for OmniFood8K

Dear Dr. [name],

I am the founder of Nutrelle, a small Brazilian company building a nutrition app for women. The app estimates the protein of a meal from a single overhead photo, and its accuracy matters to us because people will use it to plan what they eat.

Our model is trained on Nutrition5k, which is the only weighed dataset we have found that allows commercial use. It reaches about 70% of dishes within 20% of the measured protein, and it is weakest on protein-rich dishes. OmniFood8K is the largest set of real dishes with weighed ingredients that we know of, and it would address exactly that weakness.

The dataset is released under CC BY-NC 4.0. Would you consider granting us a licence to use it for training and evaluating a commercial model? We would not redistribute the images or labels. We would cite your paper in the app and in our documentation, share our results with you, and we are open to a licence fee or a data-use agreement on your terms.

If commercial training is not possible, could you tell us whether internal evaluation only would be acceptable?

Thank you for making this dataset and for considering the request.

Best regards,
[name], founder, Nutrelle
[contact]

## 2. Food Portion Benchmark (ISSAI)

**Subject:** Commercial training licence for the Food Portion Benchmark

Dear ISSAI team,

I am the founder of Nutrelle, a small Brazilian company building a nutrition app for women that estimates the protein of a meal from a single overhead photo.

The hardest part of that estimate is portion weight. Our model is trained on Nutrition5k, the only weighed dataset we have found that allows commercial use, and its weight error is about 17%. Your Food Portion Benchmark, with more than 14,000 weighed images, would let us train and measure this directly.

The dataset is released under CC BY-NC 4.0. Would you consider granting us a licence to use it for training and evaluating a commercial model? We would not redistribute the images or labels. We would cite your work in the app and in our documentation, share our results with you, and we are open to a licence fee or a data-use agreement.

If commercial training is not possible, could you tell us whether internal evaluation only would be acceptable?

Thank you for considering the request.

Best regards,
[name], founder, Nutrelle
[contact]

## 3. MetaFood3D (Purdue)

**Subject:** Commercial licence request for MetaFood3D

Dear Prof. [name],

I am the founder of Nutrelle, a small Brazilian company building a nutrition app for women that estimates the protein of a meal from a single overhead photo.

MetaFood3D is one of the few datasets with measured weight and nutrient values per food item. We would like to use it to train and evaluate our portion and protein models. The dataset is offered under CC BY-NC 4.0 through a request form, so I am writing to ask whether a commercial licence is possible, and on what terms.

We would not redistribute the data. We would cite your paper in the app and in our documentation and share our results with you, and we are open to a licence fee or a data-use agreement.

If commercial use is not possible, could you tell us whether internal evaluation only would be acceptable?

Thank you for considering the request.

Best regards,
[name], founder, Nutrelle
[contact]

## 4. FLIC (UNIMIB)

**Subject:** Access to the FLIC canteen tray dataset

Dear Prof. [name],

I am the founder of Nutrelle, a small Brazilian company building a nutrition app that estimates the protein of a meal from a single overhead photo.

Your paper describes FLIC, a set of canteen trays photographed from above and weighed before and after the meal. This is very close to how our app is used. I could not find where the dataset can be downloaded or under which licence it is released.

Could you tell us where it is available, and whether its licence allows training and evaluating a commercial model? If it does not, we would be glad to discuss a licence or a data-use agreement, and we would cite your work.

Thank you for your time.

Best regards,
[name], founder, Nutrelle
[contact]

## 5. ACETADA (Curtin)

**Subject:** Availability of the ACETADA meal image dataset

Dear Prof. [name],

I am the founder of Nutrelle, a small Brazilian company building a nutrition app that estimates the protein of a meal from a single photo taken with a phone.

Your work describes ACETADA, a set of phone photos of meals with weighed food records. Phone photos with measured weights are what our model lacks most. I could not find a public release of the dataset.

Could you tell us whether it is or will be available, and under which licence? If it is not public, would you consider sharing it for training and evaluating a commercial model under a data-use agreement? We would not redistribute it, and we would cite your work.

Thank you for your time.

Best regards,
[name], founder, Nutrelle
[contact]

## 6. ClinicPlate (pedido de acesso no Zenodo)

Não é e-mail: é o botão "Request access" no registro 21626434 do Zenodo, com uma conta sua. Texto sugerido para o campo de justificativa:

> I am the founder of Nutrelle, a small Brazilian company building a nutrition app that estimates the protein of a meal from a single overhead photo. We would use ClinicPlate to train and evaluate portion-weight and protein estimation, under its CC BY 4.0 licence, citing the dataset in the app and in our documentation. We will not redistribute the images or labels.

## 7. SNAPMe (USDA)

**Subject:** Use of SNAPMe as weak-label training data

Dear Dr. [name],

I am the founder of Nutrelle, a small Brazilian company building a nutrition app that estimates the protein of a meal from a single photo taken with a phone.

SNAPMe is the closest public dataset to how the app is used: real meals photographed at home. The record asks that it not be used to train models because it is a small curated set, and we want to respect that. I am writing to ask whether you would object to its use as weak-label training data alongside larger weighed datasets, with SNAPMe cited and its share-alike terms followed, or whether you would prefer that we use it for evaluation only.

Thank you for your time.

Best regards,
[name], founder, Nutrelle
[contact]
