# Instrução para o estimador cego, v2 (por componente)

> Este é o texto que o estimador recebe junto com a pasta de imagens (`cego2/loteNN/`). Ele não recebe gabarito, tabelas nem o nome da fonte das fotos. O estimador **não calcula proteína**: a proteína sai de `eval/protein_from_components.py`, que usa TACO/USDA.

---

Você vai olhar fotos de refeições, uma de cada vez. Para cada foto, liste **cada alimento visível como um componente separado** e estime quantos **gramas** de cada um há no prato (comida pronta, como está servida). **Não calcule proteína nem calorias.** Isso é feito depois, a partir da sua lista.

O nome do arquivo sem a extensão é o `prato_id` (ex.: `d0137.png` → `d0137`, `b012.png` → `b012`). Não tente descobrir de onde vem a foto. O id não diz nada sobre o conteúdo.

## Saída

Um CSV com uma linha **por componente**, com estas colunas:

```
prato_id,componente,alimento_ref,gramas_estimadas,confianca,pista_escala
d0137,arroz branco,"Arroz, tipo 1, cozido",140,media,"prato raso ~26 cm; arroz ocupa ~1/4"
d0137,feijão,"Feijão, carioca, cozido",90,media,"concha rasa ao lado do arroz"
d0137,bife,"Carne, bovina, contra-filé, sem gordura, grelhado",120,baixa,"bife ~12×8 cm, ~1,2 cm de espessura vs. garfo"
d0137,alface,"Alface, crespa, crua",20,alta,"3 folhas"
```

- `componente`: como você descreveria o item, em poucas palavras.
- `alimento_ref`: o nome **mais próximo** da tabela de referência:
  - comida brasileira → nome **exato** da TACO (lista abaixo; se o item não estiver nela, escreva no mesmo estilo, ex. `"Carne, bovina, cupim, assado"`);
  - comida não brasileira ou genérica → nome genérico em inglês no estilo USDA (lista abaixo), opcionalmente com o prefixo `USDA:` (ex.: `USDA: grilled chicken`).
- `gramas_estimadas`: número inteiro de gramas do componente pronto, sem o prato. Peso comestível: sem osso nem casca quando dá para ver que eles não serão comidos. Some o caldo do feijão ao feijão e o molho a quem ele cobre, a menos que o molho seja um componente à parte (ex.: `molho de tomate` 40 g).
- `confianca`: `alta`, `media` ou `baixa`, só para o peso deste componente.
- `pista_escala`: a referência de tamanho usada (prato, talher, recipiente, mão, embalagem, comparação com outro item). Se não houver nenhuma, escreva `nenhuma`.

Se a foto tiver um único alimento, a saída é uma linha só. Se um item estiver escondido (por baixo de outro, dentro de recipiente opaco), estime só o que dá para inferir e diga isso em `pista_escala`.

## Como estimar (lições da rodada v1)

Na rodada v1, quem estimou acertou **o que** era o alimento (erro de densidade de cerca de 7%), mas errou **quanto** havia. Siga estes passos:

1. **Ache a escala primeiro.** Antes de olhar a comida, identifique o recipiente e meça tudo em relação a ele:
   - prato raso de jantar: 24 a 27 cm de diâmetro (a área útil, sem a borda, tem 18 a 21 cm); prato de sobremesa: 19 a 21 cm; prato fundo: cerca de 22 cm; bandeja/prato de refeitório: confira a borda e os talheres;
   - garfo de mesa: cerca de 19 cm; faca: 21 a 23 cm; colher de sopa: 17 a 19 cm (a concha da colher tem cerca de 4 × 6 cm);
   - marmita descartável de alumínio de 750 ml: cerca de 19 × 14 cm; de 500 ml: cerca de 16 × 12 cm; pote redondo de 500 ml: cerca de 12 cm de diâmetro;
   - sem nenhuma referência, use o tamanho típico do próprio alimento (ovo inteiro ≈ 50 g; pão francês ≈ 50 g; fatia de tomate ≈ 15 g; folha de alface ≈ 10 g) e marque `confianca` como `baixa`.
2. **Estime área × altura de cada componente**, nunca o prato inteiro. Descreva em `pista_escala` a área (cm ou fração do prato) e a altura que você supôs. Medidas caseiras de referência (IBGE, POF 2008-09): colher de servir de arroz ≈ 45 g; escumadeira de arroz ≈ 85 g; concha de feijão ≈ 80 a 140 g; colher de servir de carne moída ≈ 60 g; bife ≈ 80 a 100 g; filé de frango ≈ 100 g; filé de peixe ≈ 120 g; gomo de linguiça ≈ 60 g; escumadeira de macarrão ≈ 110 g.
3. **Porções grandes de carne, peixe e frango costumam ser subestimadas.** Na v1, as peças grandes de proteína animal (acima de ~200 g) foram estimadas, em média, com 30 a 50% menos peso do que tinham. Quando a peça de proteína ocupa mais de 1/4 do prato ou é alta (mais de 2 cm), confira a espessura e não deixe a estimativa "voltar para a média". Peças de 200 a 350 g são comuns em prato de restaurante.
4. **Prato composto costuma ter menos carne do que parece.** Em marmitas e pratos montados, a v1 superestimou a proporção de carne (+16%): pedaços de carne por cima do arroz ou do molho escondem que, por baixo, a maior parte é arroz, massa, purê ou legumes. Estime a carne pelo que está **visível** (conte os pedaços e o tamanho de cada um) e coloque o resto do volume nos acompanhamentos.
5. **Separe tudo que der para separar:** arroz, feijão, carne, salada e farofa são componentes diferentes. Pratos que não dá para separar (lasanha, feijoada, estrogonofe, risoto, sopa, sanduíche) entram como um componente com o nome do prato pronto (ex.: `Feijoada`, `Estrogonofe de frango`, `USDA: lasagna`).
6. Óleo, molho, temperos e ervas: liste somente se forem visíveis em quantidade relevante (≥ 5 g). Não estime sal.
7. Para cada componente, pense **duas vezes** no tamanho: primeiro uma estimativa, depois "ela é compatível com a escala do passo 1?". Escreva só a final.

## Nomes TACO frequentes (use exatamente assim)

Cereais e massas: `Arroz, tipo 1, cozido` · `Arroz, integral, cozido` · `Macarrão, trigo, cozido (USDA 20121)` · `Pão, trigo, francês` · `Pão, trigo, forma, integral` · `Mandioca, farofa, temperada` · `Cuscuz, de milho, cozido com sal` · `Tapioca, com manteiga` · `Milho, verde, enlatado, drenado`
Leguminosas: `Feijão, carioca, cozido` · `Feijão, preto, cozido` · `Lentilha, cozida` · `Grão-de-bico, cru` (prefira `USDA: chickpeas` para grão cozido) · `Feijoada`
Carnes: `Carne, bovina, contra-filé, sem gordura, grelhado` · `Carne, bovina, miolo de alcatra, sem gordura, grelhado` · `Carne, bovina, patinho, sem gordura, grelhado` · `Carne, bovina, filé mingnon, sem gordura, grelhado` · `Carne, bovina, acém, moído, cozido` · `Carne, bovina, costela, assada` · `Carne, bovina, charque, cozido` · `Porco, lombo, assado` · `Porco, pernil, assado` · `Lingüiça, porco, grelhada` · `Presunto, sem capa de gordura`
Aves e ovos: `Frango, peito, sem pele, grelhado` · `Frango, peito, sem pele, cozido` · `Frango, coxa, com pele, assada` · `Frango, sobrecoxa, sem pele, assada` · `Ovo, de galinha, inteiro, cozido/10minutos` · `Ovo, de galinha, inteiro, frito`
Peixes: `Merluza, filé, assado` · `Pescada, filé, frito` · `Salmão, sem pele, fresco, grelhado` · `Atum, conserva em óleo` · `Sardinha, assada` · `Camarão, Rio Grande, grande, cozido`
Tubérculos: `Batata, inglesa, cozida` · `Batata, inglesa, frita` · `Batata, doce, cozida` · `Mandioca, cozida` · `Mandioca, frita` · `Polenta, pré-cozida`
Leite e queijos: `Queijo, minas, frescal` · `Queijo, mozarela` · `Queijo, prato` · `Iogurte, natural` · `USDA: milk` (a TACO não tem proteína do leite fluido)
Hortaliças e frutas: `Alface, crespa, crua` · `Tomate, com semente, cru` · `Cenoura, cozida` · `Brócolis, cozido` · `Abobrinha, italiana, refogada` · `Couve, manteiga, refogada` · `Repolho, branco, cru` · `Banana, prata, crua` · `Mamão, Papaia, cru`
Pratos prontos: `Estrogonofe de frango` · `Baião de dois, arroz e feijão-de-corda` · `Cuscuz, paulista` (genérico) · `Salpicão, de frango` · `Feijoada`

## Nomes genéricos (USDA) para o que não é brasileiro

`grilled chicken` · `chicken breast` · `chicken thighs` · `roast chicken` · `steak` · `beef` · `ground beef` · `pork` · `pork chops` · `bacon` · `sausage` · `ham` · `turkey` · `salmon` · `fish` · `cod` · `tuna` · `shrimp` · `scrambled eggs` · `fried eggs` · `hard boiled eggs` · `egg whites` · `omelets` · `tofu` · `white rice` · `brown rice` · `pasta` · `noodles` · `bread` · `toast` · `tortilla` · `quinoa` · `couscous` · `oatmeal` · `potatoes` · `roasted potatoes` · `mashed potatoes` · `french fries` · `sweet potato` · `black beans` · `chickpeas` · `lentils` · `hummus` · `cheese` · `cottage cheese` · `greek yogurt` · `mixed greens` · `lettuce` · `spinach (cooked)` · `broccoli` · `cauliflower` · `carrot` · `green beans` · `corn` · `tomatoes` · `cucumbers` · `bell peppers` · `squash` · `zucchini` · `mushroom` · `avocado` · `olive oil` · `salad dressing` · `pizza` · `lasagna` · `burritos` · `sandwiches`
