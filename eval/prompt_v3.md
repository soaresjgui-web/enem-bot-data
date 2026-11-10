# Instrução para o estimador cego, v3 (identificação e partes da massa)

> Este é o texto que o estimador recebe junto com a pasta de imagens (`cego2/loteNN/`). Ele não recebe gabarito, nem a origem das fotos, nem a divisão calib/teste. O estimador **não estima gramas nem proteína**: ele identifica os alimentos e diz que **parte da massa** do prato cada um representa. A proteína sai de `eval/protein_from_shares.py`, que junta essas partes com as tabelas (Nutrition5k/USDA e TACO) e com uma massa total vinda de outro lugar (o modelo treinado).
>
> Por que mudou em relação à v2: na análise do `calib` (seção 9 de `docs/dados/precisao/resultado_modelo_treinado.md`), o erro de densidade do LLM não veio da tabela nem do valor por 100 g de cada alimento. Veio de **qual é a fonte de proteína** (77% de acerto no `calib`) e de **quanto da massa do prato ela ocupa**. Esta instrução pede só essas duas coisas.
>
> Para quem roda (não mostrar ao estimador): ver "Para o orquestrador" no fim do arquivo.

---

Você vai olhar fotos de refeições tiradas de cima, uma de cada vez. Para cada foto:

1. liste **cada alimento visível como um componente separado**;
2. dê a cada componente o nome **mais próximo** de uma lista fechada de alimentos de referência (abaixo);
3. diga que **parte da massa total da comida** (em %) cada componente representa;
4. declare qual é a **principal fonte de proteína** do prato.

**Não estime gramas, proteína nem calorias.** O nome do arquivo sem a extensão é o `prato_id` (ex.: `d0137.png` → `d0137`; os ids `x0001` e `x0002` do exemplo abaixo são inventados). Não tente descobrir de onde vem a foto; o id não diz nada sobre o conteúdo.

## Saída

Um CSV (UTF-8, separado por vírgula, campos com vírgula entre aspas duplas) com uma linha **por componente** e exatamente estas colunas, nesta ordem:

```
prato_id,componente,alimento_ref,alimento_alt,certeza,preparo,parte_pct,fonte_principal
x0001,filé de salmão,salmon,fish,media,assado,45,peixe_marisco
x0001,arroz integral,brown rice,,alta,cozido,30,peixe_marisco
x0001,brócolis,broccoli,,alta,cozido,22,peixe_marisco
x0001,fio de azeite,olive oil,,baixa,nao_se_aplica,3,peixe_marisco
x0002,omelete,omelets,scrambled eggs,media,frito,48,ovo
x0002,queijo derretido por cima,cheese,,media,nao_se_aplica,6,ovo
x0002,torrada,toast,,alta,assado,16,ovo
x0002,morangos,strawberries,,alta,cru,30,ovo
```

- `componente`: como você descreveria o item, em poucas palavras (texto livre, em português).
- `alimento_ref`: o alimento de referência **mais provável**, escrito **exatamente** como nas listas abaixo.
  - comida genérica ou internacional → nome em inglês da lista "Alimentos de referência";
  - comida brasileira → nome da TACO (lista no fim; se o item não estiver nela, escreva no mesmo estilo, ex. `"Carne, bovina, cupim, assado"`).
- `alimento_alt`: a **segunda hipótese** para o mesmo componente, da mesma lista, quando você não tiver certeza do que é (ex.: pode ser frango ou porco; pode ser clara de ovo ou tofu). Deixe **vazio** quando `certeza` for `alta`.
- `certeza`: `alta`, `media` ou `baixa` — sua certeza sobre **o que é** o componente (não sobre a quantidade).
- `preparo`: um destes: `cru`, `cozido`, `grelhado`, `assado`, `frito`, `empanado`, `refogado`, `ensopado` (com molho/caldo), `mexido`, `processado` (embutidos, frios, enlatados), `nao_se_aplica`.
- `parte_pct`: número inteiro de 1 a 100: a parte da **massa total da comida do prato** que é deste componente. **As partes de um prato somam 100.** Prato com um único alimento: uma linha com `100`.
- `fonte_principal`: o grupo que mais contribui com **proteína** (não com massa) no prato, **repetido em todas as linhas do prato**. Um destes:
  `frango_peru` · `bovina` · `suina_embutido` (porco, bacon, linguiça, presunto) · `peixe_marisco` · `ovo` · `leguminosa_soja` (feijão, lentilha, grão-de-bico, tofu) · `laticinio` (queijo, iogurte, leite) · `prato_misto` (pizza, lasanha, sanduíche, prato pronto que não dá para separar) · `nozes_sementes` · `nenhuma` (prato sem fonte relevante de proteína: frutas, saladas, legumes, grãos, pão).

## Como identificar

1. **Olhe primeiro para a proteína.** Antes de listar os acompanhamentos, pergunte: há carne, ave, peixe, ovo, queijo, tofu, leguminosa, castanha ou um prato pronto (pizza, lasanha) nesta foto? Onde? Quanto? É isso que mais pesa no resultado.
2. **Diferencie pelas pistas visuais:**
   - **frango** (peito): branco a bege, fibras longas, cubos ou tiras secas; **coxa/sobrecoxa** (`chicken thighs`): mais escura, irregular, brilhante. **Porco** (`pork`): rosado-acinzentado, fibra mais fina, borda de gordura. **Peru** moído ou fatiado: claro, textura fina;
   - **bovina**: marrom-escura por fora, pode ser rosada por dentro; `steak` é peça ou fatia; `beef` é picada, desfiada ou em cubos; `ground beef` é moída;
   - **peixe branco** (`fish`, `cod`, `tilapia`): lascas largas e brancas que se separam em camadas; **salmão** (`salmon`): laranja-rosado, lascas com linhas brancas; **atum** (`tuna`): rosa-escuro a marrom, fibra compacta; **camarão** (`shrimp`): curvo, rosado;
   - **ovo**: `scrambled eggs` é amarelo e em grumos; `egg whites` é branco, em grumos ou placa (não confundir com tofu, queijo feta, couve-flor ou arroz); `hard boiled eggs`, `fried eggs`, `omelets` pelo formato;
   - **embutidos**: `bacon` (tiras onduladas vermelho-escuras com gordura), `turkey bacon` (tiras lisas, rosadas, sem gordura visível), `sausage` (gomos ou rodelas), `chicken apple sausage` (gomo claro), `ham`;
   - **tofu**: cubos brancos ou dourados de borda reta e superfície lisa; **queijos**: `feta cheese` (branco esfarelado), `parmesan cheese` (ralado), `cheese` (fatia ou cubo amarelo);
   - **grãos**: `white rice`, `brown rice`, `quinoa` (grãos miúdos com "anelzinho"), `couscous`, `wheat berry` (grão marrom maior), `millet` (amarelo miúdo), `bulgur`, `lentils`, `chickpeas`.
3. **Prato que não dá para separar** (pizza, lasanha, sanduíche, burrito, sopa, ensopado) entra como **um** componente com o nome do prato pronto (`pizza`, `lasagna`, `sandwiches`, `Feijoada`…) e `fonte_principal` = `prato_misto`.
4. **Itens pequenos que carregam proteína ou gordura contam:** queijo ralado ou esfarelado, castanhas, sementes, bacon em pedaços, molho cremoso. Liste-os mesmo que sejam 1 a 5% da massa. Ervas, sal e enfeites abaixo de 1% ficam de fora.
5. Se não der para decidir entre dois alimentos, ponha o mais provável em `alimento_ref`, o outro em `alimento_alt` e marque `certeza` como `media` ou `baixa`. **Não escolha o mais genérico para fugir da dúvida** (ex.: `chicken` × `pork` são ambos específicos; `seafood` ou `beef` genérico só se realmente não der para ver).

## Como estimar a parte da massa (o passo em que mais se erra)

**Área na foto não é massa.** Uma cama de folhas pode cobrir o prato inteiro e pesar 40 a 50 g, enquanto a carne em cubos por cima dela cobre um terço da área e pesa três vezes mais. Faça assim, componente por componente:

1. **Área:** que fração da área de comida o componente ocupa (conte também o que está **por baixo** de outro item: a base de arroz ou de folhas continua embaixo da carne).
2. **Altura:** rasa (≤ 1 cm), média (1 a 2,5 cm), alta (> 2,5 cm, monte ou peça grossa).
3. **Compactação** (quanto pesa o mesmo volume):
   - **folhas cruas soltas** (alface, rúcula, espinafre cru, mix de folhas, couve crua): muito leves. Peso relativo **0,1**;
   - **vegetais cozidos ou picados, frutas em cubos, cogumelos, legumes assados**: peso relativo **0,6**;
   - **grãos cozidos, massas, purês, batatas, leguminosas**: peso relativo **0,8**;
   - **carnes, aves, peixes, ovos, tofu, queijos, embutidos, pizza**: compactos. Peso relativo **1,0**;
   - **castanhas e sementes**: **0,6**; **pão e torradas**: **0,3**; **bacon crocante**: **0,4**;
   - **molhos, azeite, manteiga**: quase não têm área; dê de 1 a 5% conforme o brilho e a poça visível.
4. **Peso relativo = área × altura × compactação.** Some os pesos relativos de todos os componentes e converta cada um em porcentagem. Arredonde para inteiros que somem 100.
5. **Confira com unidades conhecidas** quando houver: um ovo ≈ 50 g; uma tira de bacon ≈ 10 g; um gomo de linguiça ≈ 60 g; um tomate-cereja ≈ 15 g; uma amêndoa ≈ 1,2 g; uma azeitona ≈ 4 g; um morango ≈ 12 g; uma fatia de pizza ≈ 100 g; um peito de frango inteiro ≈ 150 g. Se as partes que você deu não batem com a contagem das unidades, corrija as partes.
6. **Faça duas vezes.** Primeiro escreva as partes; depois pergunte: "se o prato inteiro pesasse 300 g, cada componente teria um peso plausível?". Escreva só o resultado final.

Erros a evitar:
- dar às folhas a mesma parte que a área delas sugere (o erro mais comum);
- esquecer a base escondida (arroz, grão ou purê sob a carne) ou, ao contrário, supor uma base que não se vê;
- dividir em partes iguais por falta de critério ("33/33/33");
- juntar duas proteínas diferentes num componente só (ex.: ovo e bacon são dois componentes).

## Alimentos de referência (nomes genéricos; use exatamente assim)

**Aves:** `chicken` · `chicken breast` · `chicken thighs` · `chicken drumsticks` · `chicken wings` · `grilled chicken` · `roast chicken` · `fried chicken` · `chicken nuggets` · `ground chicken` · `chicken apple sausage` · `turkey` · `turkey breast` · `roast turkey` · `ground turkey` · `turkey bacon` · `duck`
**Bovina e cordeiro:** `beef` · `steak` · `ground beef` · `roast beef` · `brisket` · `corned beef` · `meatballs` · `hamburgers` · `veal` · `lamb` · `lamb chops` · `bison` · `jerky`
**Suína e embutidos:** `pork` · `pork chops` · `roast pork` · `ground pork` · `ribs` · `bacon` · `ham` · `sausage` · `hot dogs` · `pepperoni` · `salami` · `pastrami` · `bologna`
**Peixes e frutos do mar:** `fish` · `salmon` · `smoked salmon` · `tuna` · `cod` · `tilapia` · `halibut` · `trout` · `catfish` · `snapper` · `swordfish` · `mahi mahi` · `mackerel` · `sardines` · `anchovies` · `shrimp` · `crab` · `lobster` · `scallops` · `clams` · `mussels` · `oysters` · `squid` · `calamari` · `octopus` · `seafood`
**Ovos:** `scrambled eggs` · `egg whites` · `eggs` · `fried eggs` · `hard boiled eggs` · `poached eggs` · `omelets` · `egg yolks`
**Leguminosas e soja:** `tofu` · `tempeh` · `edamame` · `black beans` · `white beans` · `kidney beans` · `pinto beans` · `lima beans` · `baked beans` · `refried beans` · `chickpeas` · `lentils` · `green peas` · `hummus` · `falafel`
**Laticínios:** `cheese` · `cheddar cheese` · `mozzarella cheese` · `parmesan cheese` · `feta cheese` · `goat cheese` · `blue cheese` · `swiss cheese` · `provolone cheese` · `cream cheese` · `cottage cheese` · `greek yogurt` · `yogurt` · `milk`
**Pratos prontos (um componente só):** `pizza` · `cheese pizza` · `pepperoni pizza` · `lasagna` · `macaroni and cheese` · `sandwiches` · `burritos` · `tacos` · `quesadillas` · `enchiladas` · `wraps` · `dumplings` · `fried rice` · `chow mein` · `lo mein` · `sushi` · `ravioli` · `tortellini` · `quiche` · `meatloaf` · `stews` · `curries` · `chili` · `chicken salad` · `tuna salad` · `egg salad` · `chicken soup` · `tomato soup` · `miso soup` · `risotto` · `paella` · `nachos`
**Castanhas e sementes:** `almonds` · `walnuts` · `pecans` · `cashews` · `peanuts` · `pistachios` · `mixed nuts` · `pine nuts` · `pumpkin seeds` · `sunflower seeds` · `chia seeds` · `peanut butter` · `almond butter` · `tahini`
**Grãos, massas e pães:** `white rice` · `brown rice` · `wild rice` · `country rice` · `quinoa` · `couscous` · `bulgur` · `millet` · `wheat berry` · `barley` · `oatmeal` · `granola` · `cereal` · `pasta` · `spaghetti` · `macaroni` · `noodles` · `rice noodles` · `orzo` · `gnocchi` · `pasta salad` · `bread` · `white bread` · `whole wheat bread` · `multigrain bread` · `sourdough bread` · `toast` · `bagels` · `english muffins` · `croissants` · `buns` · `rolls` · `biscuits` · `muffins` · `pita bread` · `naan` · `flatbread` · `tortilla` · `tortilla chips` · `crackers` · `croutons` · `pancakes` · `waffles` · `french toast` · `cornbread` · `polenta` · `grits`
**Batatas e raízes:** `potatoes` · `roasted potatoes` · `baked potatoes` · `mashed potatoes` · `red potatoes` · `french fries` · `hash browns` · `potato salad` · `potato chips` · `sweet potato` · `yam` · `corn` · `corn on the cob` · `parsnips` · `turnips` · `beets`
**Verduras e legumes:** `mixed greens` · `lettuce` · `arugula` · `spinach (raw)` · `spinach (cooked)` · `kale` · `chard` · `collards` · `mustard greens` · `bok choy` · `cabbage` · `brussels sprouts` · `broccoli` · `cauliflower` · `green beans` · `snow peas` · `asparagus` · `zucchini` · `squash` · `eggplant` · `bell peppers` · `mushroom` · `tomatoes` · `cherry tomatoes` · `sun dried tomatoes` · `cucumbers` · `carrot` · `baby carrots` · `celery` · `onions` · `green onions` · `shallots` · `leeks` · `garlic` · `radishes` · `jalapenos` · `artichokes` · `fennel` · `okra` · `olives` · `pickles` · `avocado` · `mixed vegetables` · `caesar salad` · `garden salad` · `greek salad` · `coleslaw` · `sauerkraut` · `kimchi` · `seaweed`
**Frutas:** `apple` · `banana` · `orange` · `mandarin oranges` · `grapefruits` · `grapes` · `berries` · `strawberries` · `blueberries` · `raspberries` · `blackberries` · `cherries` · `pineapple` · `cantaloupe` · `honeydew melons` · `watermelon` · `mangos` · `papayas` · `kiwi` · `peaches` · `pears` · `plums` · `apricots` · `figs` · `dates` · `raisins` · `cranberries` · `pomegranate` · `lemon` · `lime` · `fruit salad`
**Gorduras, molhos e temperos:** `olive oil` · `vegetable oil` · `butter` · `mayonnaise` · `salad dressing` · `caesar dressing` · `ranch dressing` · `italian dressing` · `vinaigrette` · `balsamic vinegar` · `pesto` · `salsa` · `guacamole` · `sour cream` · `gravy` · `pasta sauce` · `barbecue sauce` · `teriyaki sauce` · `soy sauce` · `ketchup` · `mustard` · `honey` · `maple syrup` · `jams`
**Doces e outros:** `brownies` · `cookies` · `cakes` · `chocolate` · `dark chocolate` · `donuts` · `pastries` · `pies` · `puddings` · `ice creams` · `granola bars` · `nutrition bars` · `trail mix` · `popcorn`

## Nomes TACO para comida brasileira (use exatamente assim)

Cereais e massas: `Arroz, tipo 1, cozido` · `Arroz, integral, cozido` · `Macarrão, trigo, cozido (USDA 20121)` · `Pão, trigo, francês` · `Pão, trigo, forma, integral` · `Mandioca, farofa, temperada` · `Cuscuz, de milho, cozido com sal` · `Tapioca, com manteiga` · `Milho, verde, enlatado, drenado`
Leguminosas: `Feijão, carioca, cozido` · `Feijão, preto, cozido` · `Lentilha, cozida` · `Feijoada`
Carnes: `Carne, bovina, contra-filé, sem gordura, grelhado` · `Carne, bovina, miolo de alcatra, sem gordura, grelhado` · `Carne, bovina, patinho, sem gordura, grelhado` · `Carne, bovina, filé mingnon, sem gordura, grelhado` · `Carne, bovina, acém, moído, cozido` · `Carne, bovina, costela, assada` · `Carne, bovina, charque, cozido` · `Porco, lombo, assado` · `Porco, pernil, assado` · `Lingüiça, porco, grelhada` · `Presunto, sem capa de gordura`
Aves e ovos: `Frango, peito, sem pele, grelhado` · `Frango, peito, sem pele, cozido` · `Frango, coxa, com pele, assada` · `Frango, sobrecoxa, sem pele, assada` · `Ovo, de galinha, inteiro, cozido/10minutos` · `Ovo, de galinha, inteiro, frito`
Peixes: `Merluza, filé, assado` · `Pescada, filé, frito` · `Salmão, sem pele, fresco, grelhado` · `Atum, conserva em óleo` · `Sardinha, assada` · `Camarão, Rio Grande, grande, cozido`
Tubérculos: `Batata, inglesa, cozida` · `Batata, inglesa, frita` · `Batata, doce, cozida` · `Mandioca, cozida` · `Mandioca, frita` · `Polenta, pré-cozida`
Leite e queijos: `Queijo, minas, frescal` · `Queijo, mozarela` · `Queijo, prato` · `Iogurte, natural`
Hortaliças e frutas: `Alface, crespa, crua` · `Tomate, com semente, cru` · `Cenoura, cozida` · `Brócolis, cozido` · `Abobrinha, italiana, refogada` · `Couve, manteiga, refogada` · `Repolho, branco, cru` · `Banana, prata, crua` · `Mamão, Papaia, cru`
Pratos prontos: `Estrogonofe de frango` · `Baião de dois, arroz e feijão-de-corda` · `Cuscuz, paulista` · `Salpicão, de frango` · `Feijoada`

---

## Para o orquestrador (não faz parte do texto do estimador)

**O que estimar.** As imagens são as mesmas da v2, em `<scratchpad>/cego2/loteNN/` (ids neutros).

| Prioridade | Arquivos | Quantos | Para quê |
|---|---|---|---|
| **Obrigatório** | todos os `d*.png` dos lotes `lote01` a `lote15` | 400 (120 `calib` + 280 `teste`, misturados; o estimador não sabe qual é qual) | ajustar os pesos no `calib` e medir no `teste` |
| Recomendado | os `n*.png` dos mesmos lotes | 50 (`n5k_v1`) | checagem secundária |
| Dispensável | `b*.png`, `c*.jpg` (itens brasileiros) | 144 | quase nenhum é foto de cima e o modelo de massa não vale para eles (seção 8.8); só serviriam para testar o caminho TACO da instrução |

**Rodadas.** Uma rodada basta. Se o custo permitir, faça **duas rodadas independentes** (estimadores diferentes, sem ver a saída um do outro) pelo menos nos `d*.png`: `protein_from_shares.py` faz a média das rodadas, e a média de duas estimativas independentes de partes costuma reduzir o ruído.

**Formato de entrega.** Um CSV por lote e rodada, com o cabeçalho exato da seção "Saída": `cego3/out/loteNN.csv` (rodada 1) e `cego3/out/loteNN_r2.csv` (rodada 2). Depois de juntar (com uma coluna extra `rodada` = 1 ou 2), o arquivo vai para `docs/dados/precisao/estimativas_v3_partes.csv`:

```
prato_id,componente,alimento_ref,alimento_alt,certeza,preparo,parte_pct,fonte_principal,rodada
```

**Como vira proteína.**

```
python3 eval/protein_from_shares.py docs/dados/precisao/estimativas_v3_partes.csv docs/dados/precisao/estimativas_v3_pratos.csv \
    --componentes docs/dados/precisao/estimativas_v3_comp_casados.csv [--massa massas.csv]
python3 eval/treino/dens_avalia.py        # reajusta o híbrido no calib com a densidade v3 e avalia no teste
```

`protein_from_shares.py` casa `alimento_ref`/`alimento_alt` com `ref/n5k_ingredients.csv` (nomes em inglês) e `ref/taco.csv`, calcula a densidade do prato como Σ parte × proteína por grama (com `certeza` = `media`/`baixa`, mistura 25%/40% da densidade de `alimento_alt`) e, se receber `--massa` (CSV `prato_id,massa_g`), multiplica pela massa. `dens_avalia.py` usa a massa do modelo treinado e tem as configurações com a densidade v3 **declaradas antes** de os dados existirem.

**Higiene.** Nada nesta instrução veio do `teste`: as lições vêm da análise do `calib` e do treino, e as listas de nomes são as tabelas de referência (`ref/n5k_ingredients.csv`, `ref/taco.csv`), não rótulos de pratos.
