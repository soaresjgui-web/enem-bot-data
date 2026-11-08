# Coleta do gabarito: proteína por foto (2026-11-08)

Arquivos: `gabarito_pratos.csv` (1 linha por prato, com fonte, URL, confiança e notas), `gabarito.csv` (formato do `eval/score_precisao.py`: uma linha `prato inteiro` por prato, `proteina_100g = proteina_g / peso × 100`, com 2 casas) e os scripts `coleta_ufpr.py`, `coleta_n5k.py` e `coleta.py` (Seara + montagem final).
**As imagens não estão no repositório**, porque são de terceiros e o repositório é público. Ficam no scratchpad da sessão (`precisao/imagens/<prato_id>.png`). O arquivo `precisao/lista_imagens.txt` traz só `prato_id<TAB>caminho`, sem rótulos, e é o que o estimador cego recebe. Os ids `b001…`/`n001…` foram sorteados (seed 20261108) e não revelam conteúdo.

## Resultado

| Fonte | Itens | Categorias | Proteína (g) | Confiança |
|---|---|---|---|---|
| Manual Fotográfico UFPR 2017 + TACO 4ª ed. | 103 | almoco_pf 52, jantar 22, lanche 13, restaurante 10, cafe 6 | 0,9–100,4 | A 62 · B 29 · C 12 |
| Seara Marmita 300 g (site oficial) | 7 | congelado | 13,8–20,1 | B |
| **Brasileiros** | **110** | | 0,9–100,4 | |
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

### Nutrition5k
- Amostra estratificada (seed fixa) de 10 pratos por faixa de proteína: 0–5, 5–15, 15–25, 25–40 e >40 g. Os candidatos são o split `rgb_test` com imagem overhead e massa ≥ 20 g. A imagem é o `rgb.png` zenital e o gabarito vem de `total_protein`/`total_mass` do CSV. Ingredientes em `descricao`.

## Fontes planejadas que não renderam (bloqueios)

- **Mr. Fit / marmitarias no 99Food (`99app.com`)**: Vercel Security Checkpoint (HTTP 429 com desafio JS). `www.mrfit.com.br` redireciona para `mrfit.com.br` (sem www), host não liberado. Nada coletado.
- **Liv Up (`www.livup.com.br`)**: a página tem a tabela nutricional por 100 g e o peso (gabarito ótimo), mas as fotos da marmita aberta estão em **`livup.imgix.net`**, host não liberado. No Pão de Açúcar só há foto da caixa (comida parcial atrás de textos), por isso foi excluído.
- **Pão de Açúcar**: o JSON do produto não traz tabela nutricional. Imagens (`static.paodeacucar.com`) são só de embalagem. A busca usa `api.vendas.gpa.digital` e linximpulse (não liberados).
- **Carrefour**: as APIs VTEX (`/api/catalog_system`, `/api/io/...intelligent-search`) devolveram 503. A busca é renderizada no cliente. As imagens vêm de `carrefourbrfood.vtexassets.com` (também servidas por `carrefourbr.vtexassets.com`).
- **Perdigão / Sadia**: HTTP 403 nos sites oficiais.
- **Swift**: a página tem pratos feitos de 300 g, mas as imagens estão em **`swiftbr.vteximg.com.br`**, host não liberado.
- **Open Food Facts**: a busca (`/api/v2/search` e facetas) deu 503 durante toda a coleta. A API de produto funciona, mas os produtos brasileiros consultados só têm foto de embalagem.
- **UFPR acervodigital**: erro de certificado TLS. O PDF da USP (e-disciplinas) exige login. Usei a cópia do nutritotal.

Para ganhar pratos compostos brasileiros com foto real, os passos seriam: liberar `livup.imgix.net` (cerca de 30 marmitas Liv Up com tabela oficial) e `swiftbr.vteximg.com.br`, e abrir o 99Food com navegador.
