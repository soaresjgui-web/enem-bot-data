#!/usr/bin/env python3
"""Monta a lista de treino externo do Food Portion Benchmark (mesmo esquema de externo_mensa.csv; seção 12.4).

Dataset: Sanatbyek, A.; Rakhimzhanova, T.; Nurmanova, B.; Omarova, Z.; Rakhmankulova, A.; Orazbayev, R.; Varol, H. A.;
Chan, M. Y. (ISSAI, Nazarbayev University). "A multitask deep learning model for food scene recognition and portion
estimation — the Food Portion Benchmark (FPB) dataset". IEEE Access, 2025.
https://huggingface.co/datasets/issai/Food_Portion_Benchmark — licença CC BY-NC 4.0 (uso NÃO comercial).
Esta lista existe por decisão da fundadora (2026-11-12) de não restringir o treino a bases de licença comercial; as
colunas `licenca` e `uso_declarado` levam a procedência em cada linha.

O que é: 138 classes de comida da Ásia Central, servidas em três tamanhos (small / average / big) num prato branco
sobre uma bandeja marrom de refeitório, cada item pesado em balança. Rótulo no formato YOLO com uma 6ª coluna, o
peso em gramas: `classe x y w h peso`. Quando a classe tem mais de uma caixa na foto, o peso vem DIVIDIDO entre as
caixas (39 + 39 = 78 g), então o peso do item é a SOMA. Peso faltando vem como -1 (ou -1 dividido: -0.5, -0.333).

Vista (conferida em cerca de 500 imagens abertas, dos 12 dias de gravação e das duas campanhas):
  zenital   câmera RealSense fixa acima da bandeja. 1ª campanha (nomes com data, `<item>_<tamanho>_<dia>_<NN>_Color`):
            quadros 01-04 de cada sessão; a bandeja enche o quadro e às vezes passa da borda (comida cortada).
            2ª campanha (nomes sem data, `..._depth|RGB_ref|no_ref_Color`, 914 arquivos): todos de cima, bandeja
            inteira no quadro (70 de 70 abertos). Os quadros `_ref` têm talheres, régua e guardanapo na bandeja
            (`tem_referencia`, pelo nome, que tem erros de digitação); quase todos têm um xadrez de calibração.
  lateral   a mesma RealSense, quadros 05-12: bandeja pequena no meio da sala, vista baixa (não serve para a
            foto de cima);
  obliqua   celulares (3024x4032 ou 3456x4608), de 30 a 70 graus, quatro ou oito giros da bandeja.
No dia 09Dec a ordem dos quadros não foi essa em três sessões; nesse dia a vista sai do tamanho da caixa dentro da
sessão (regra V2) e não do número do quadro. Os 260 quadros desse dia foram abertos: 89 zenitais e 171 laterais, todos
certos; nos outros dias, 73 quadros abertos, todos conforme o número.

Uso:
  git clone com GIT_LFS_SKIP_SMUDGE=1 de https://huggingface.co/datasets/issai/Food_Portion_Benchmark  (173 MB:
  rótulos em texto e as imagens como ponteiros LFS; as imagens somam 11,4 GB e NÃO são baixadas aqui)
  python3 -I fpb_monta.py mapa  <clone>/FPB_Dataset/RGB       # regrava fpb_classes.csv
  python3 -I fpb_monta.py monta <clone>/FPB_Dataset/RGB [--saida <csv>]
Cada imagem baixa de  https://huggingface.co/datasets/issai/Food_Portion_Benchmark/resolve/main/<arquivo>
(ou huggingface_hub.snapshot_download(repo_type="dataset", allow_patterns=...)). As zenitais somam ~0,2 GB.

UMA LINHA POR FOTO. `massa` = soma dos itens pesados, SEM as bebidas (que vão em `bebida_g`), como em
externo_mensa.csv. Só a massa e as frações por grupo/fonte são alvo; proteína é FRACA (classe -> densidade de
tabela), em colunas separadas, nunca em `proteina`. Regras, com contagem impressa:
  B1  rótulo sem imagem: sai;
  B2  a mesma imagem publicada duas vezes (mesmo sha256 do arquivo): fica uma linha; se as duas anotações dão pesos
      diferentes, a imagem sai (não dá para saber qual vale);
  B3  nenhum item de comida com peso (todos -1) ou foto só de bebida: sai;
  B4  parte dos itens de comida sem peso: a linha fica com `massa` e frações VAZIAS, a soma do que foi pesado em
      `massa_pesada_g` e o aviso `peso_parcial` (o salmão pesado com a guarnição sem peso, por exemplo);
  V1  vista pelo nome do arquivo (acima);  V2  dia 09Dec: zenital se a maior caixa do quadro passa da média geométrica
      entre a maior e a menor da sessão, quando a sessão tem as duas vistas (razão >= 1,8); senão vale V1 com o
      aviso `vista_incerta`.
`prato` = a mesma bandeja servida (mesmos itens com os mesmos pesos); uma foto em que um item da bandeja não foi
anotado (escondido atrás de outro) leva o aviso `item_sem_anotacao` e fica no prato maior. `grupo` junta os pratos
que repetem a mesma porção de comida (mesma classe e mesmo peso, fora bebida, condimento e embalado), para a parte
segurada não ver a mesma porção do outro lado: regra de reserva de 12.5, sha1(grupo) mod 5 == 0 -> teste_ext.
A divisão dos autores (train/val/test) é por imagem, com o mesmo prato dos três lados; fica em `split_autores`.
"""
import ast, csv, hashlib, os, re, sys
from collections import Counter, defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from flavoria_monta import FONTES, GRUPOS, confere_esquema, densidade, tabelas  # noqa: E402

MAPA = os.path.join(AQUI, "fpb_classes.csv")
SAIDA = os.path.join(AQUI, "..", "treino", "kaggle", "listas", "externo_fpb.csv")
LICENCA, USO = "CC BY-NC 4.0", "nao_comercial"
PREFIXO_ARQ = "FPB_Dataset/RGB/"
DIA_IRREGULAR = "09Dec"
RAZAO_VISTAS = 1.8
PROTEINA_FRACA_ALTA_G = 25.0

# classe: (grupo, fonte, referência de densidade, confiança, tipo) — tipo: "" | bebida | embalado | condimento | sopa
# Confiança: A = alimento simples; B = receita típica suposta; C = receita incerta (prato regional, recheio que não
# se vê). Carne de cavalo curada (kazy, zhal-zhaya) e frios vão em suina_embutido (embutidos); pratos em que a carne
# está misturada ao amido ou ao caldo vão em prato_misto. Classes conferidas em recortes de 48 imagens.
SOPA = "0.6*DILUENTE"                  # caldo: ~1 g/100 g, contado como zero ("broth" do N5k é concentrado: 16,5)
PAO = "taco:Pão, trigo, francês"        # 8,0 g/100 g ("bread" do N5k dá 13,5)
C = {
    "achichuk": ("vegetal", "sem_proteina", "0.7*T_TOMATE|0.3*n5k:onions", "B", ""),
    "almond": ("outros", "nozes_sementes", "n5k:almonds", "A", ""),
    "apple_juice": ("outros", "sem_proteina", "n5k:apple juice", "A", "bebida"),
    "apple_strudel": ("outros", "sem_proteina", "n5k:strudels", "B", ""),
    "balqaymaq": ("laticinio", "laticinio", "0.7*n5k:cream|0.3*n5k:sugar", "C", ""),
    "bauyrsak": ("amido", "sem_proteina", "0.85*" + PAO + "|0.15*n5k:oil", "B", ""),
    "beef": ("proteina_animal", "bovina", "n5k:roast beef", "B", ""),
    "beef_cutlet": ("proteina_animal", "bovina", "0.75*n5k:ground beef|0.25*" + PAO, "B", ""),
    "beefstroganov": ("misto", "prato_misto", "0.5*n5k:beef|0.5*n5k:gravy", "B", ""),
    "beet_salad": ("vegetal", "sem_proteina", "0.9*T_BETERRABA|0.1*n5k:mayonnaise", "B", ""),
    "belise_tea": ("outros", "sem_proteina", "n5k:tea", "B", "bebida"),
    "beshbarmak": ("misto", "prato_misto", "0.3*n5k:beef|0.45*n5k:noodles|0.15*n5k:potatoes|0.1*n5k:onions", "B", ""),
    "black_tea": ("outros", "sem_proteina", "n5k:tea", "A", "bebida"),
    "bliny": ("amido", "sem_proteina", "n5k:crepes", "B", ""),
    "borek": ("misto", "prato_misto", "n5k:empanadas", "C", ""),
    "borsh": ("misto", "prato_misto", "0.1*n5k:beef|0.3*n5k:mixed vegetables|" + SOPA, "C", "sopa"),
    "bread": ("amido", "sem_proteina", PAO, "A", ""),
    "broccoli": ("vegetal", "sem_proteina", "T_BROCOLIS", "A", ""),
    "buckwheat": ("amido", "sem_proteina", "0.28*n5k:bulgur|0.72*DILUENTE", "B", ""),
    "bukteme": ("misto", "prato_misto", "n5k:empanadas", "C", ""),
    "bun": ("amido", "sem_proteina", "n5k:buns", "A", ""),
    "burger": ("misto", "prato_misto", "n5k:hamburgers", "B", ""),
    "butter": ("gordura_molho", "sem_proteina", "n5k:butter", "A", "condimento"),
    "caesar_salad": ("misto", "prato_misto", "0.2*n5k:grilled chicken|0.8*n5k:caesar salad", "C", ""),
    "capuccino": ("outros", "sem_proteina", "n5k:cappuccino", "A", "bebida"),
    "cereal": ("amido", "sem_proteina", "n5k:cereal", "B", ""),
    "chak_chak": ("outros", "sem_proteina", "0.6*n5k:donuts|0.4*n5k:honey", "C", ""),
    "cheburek": ("misto", "prato_misto", "n5k:empanadas", "B", ""),
    "cheese_sticks": ("laticinio", "laticinio", "0.6*n5k:mozzarella cheese|0.4*n5k:bread crumbs", "B", ""),
    "cheesecake": ("outros", "sem_proteina", "n5k:cheesecake", "B", ""),
    "chelpek": ("amido", "sem_proteina", "0.85*n5k:flatbread|0.15*n5k:oil", "B", ""),
    "chicken": ("proteina_animal", "frango_peru", "0.85*n5k:chicken thighs|0.15*DILUENTE", "C", ""),
    "chicken_in_plum": ("misto", "prato_misto", "0.55*n5k:chicken thighs|0.15*n5k:mushroom|0.3*n5k:cream", "C", ""),
    "chocolate": ("outros", "sem_proteina", "n5k:chocolate", "A", ""),
    "ciabatta": ("amido", "sem_proteina", PAO, "A", ""),
    "coke": ("outros", "sem_proteina", "DILUENTE", "A", "bebida"),
    "compote": ("outros", "sem_proteina", "n5k:fruit punch", "B", "bebida"),
    "cottage_cheese": ("laticinio", "laticinio", "n5k:cottage cheese", "B", ""),
    "croissant": ("amido", "sem_proteina", "n5k:croissants", "A", ""),
    "cupcake": ("outros", "sem_proteina", "n5k:cupcakes", "A", ""),
    "dapanji": ("misto", "prato_misto", "0.35*n5k:chicken thighs|0.3*n5k:potatoes|0.25*n5k:noodles|0.1*DILUENTE", "B", ""),
    "doner": ("misto", "prato_misto", "n5k:gyros", "B", ""),
    "dried_apricot": ("vegetal", "sem_proteina", "n5k:raisins", "B", ""),
    "dumpling_with_soup": ("misto", "prato_misto", "0.5*n5k:dumplings|0.5*DILUENTE", "B", "sopa"),
    "egg": ("proteina_animal", "ovo", "T_OVO_FRITO", "B", ""),
    "fanta": ("outros", "sem_proteina", "DILUENTE", "A", "bebida"),
    "fettucini": ("misto", "prato_misto", "0.8*n5k:pasta|0.1*n5k:cheese|0.1*T_TOMATE", "C", ""),
    "fish_soup": ("misto", "prato_misto", "0.15*n5k:fish|0.25*n5k:mixed vegetables|" + SOPA, "C", "sopa"),
    "french_fries": ("amido", "sem_proteina", "n5k:french fries", "A", ""),
    "fresh_salad": ("vegetal", "sem_proteina", "0.5*T_TOMATE|0.5*T_PEPINO", "B", ""),
    "fried_aubergine": ("vegetal", "sem_proteina", "0.9*n5k:eggplant|0.1*n5k:oil", "B", ""),
    "fried_dumplings": ("misto", "prato_misto", "n5k:dumplings", "B", ""),
    "fried_lagman": ("misto", "prato_misto", "0.6*n5k:noodles|0.2*n5k:beef|0.2*n5k:mixed vegetables", "B", ""),
    "funchosa": ("amido", "sem_proteina", "0.7*n5k:rice noodles|0.3*T_CENOURA", "B", ""),
    "golubcy": ("misto", "prato_misto", "0.35*n5k:ground beef|0.25*T_ARROZ|0.4*n5k:cabbage", "B", ""),
    "greek_salad": ("vegetal", "laticinio", "n5k:greek salad", "B", ""),
    "green_tea": ("outros", "sem_proteina", "n5k:tea", "A", "bebida"),
    "grilled_vegetables": ("vegetal", "sem_proteina", "n5k:mixed vegetables", "B", ""),
    "honey": ("outros", "sem_proteina", "n5k:honey", "A", "condimento"),
    "hvorost": ("outros", "sem_proteina", "n5k:pastries", "B", ""),
    "icecream": ("outros", "sem_proteina", "n5k:ice creams", "A", ""),
    "irimshik": ("laticinio", "laticinio", "n5k:cottage cheese", "C", ""),
    "kazy": ("proteina_animal", "suina_embutido", "n5k:sausage", "C", ""),
    "kefir": ("laticinio", "laticinio", "n5k:buttermilk", "B", "bebida"),
    "kespe": ("misto", "prato_misto", "0.25*n5k:noodles|0.1*n5k:beef|0.65*DILUENTE", "C", "sopa"),
    "ketchup": ("gordura_molho", "sem_proteina", "n5k:ketchup", "A", "condimento"),
    "khachapuri": ("misto", "laticinio", "0.6*" + PAO + "|0.3*n5k:cheese|0.1*n5k:eggs", "B", ""),
    "kirieshki_cheese": ("amido", "sem_proteina", "n5k:croutons", "A", "embalado"),
    "kirieshki_shashlyk": ("amido", "sem_proteina", "n5k:croutons", "A", "embalado"),
    "koktal": ("misto", "prato_misto", "0.5*n5k:fish|0.4*n5k:mixed vegetables|0.1*n5k:cheese", "C", ""),
    "korean_carrot": ("vegetal", "sem_proteina", "0.92*T_CENOURA|0.08*n5k:oil", "B", ""),
    "kurt": ("laticinio", "laticinio", "n5k:swiss cheese", "C", ""),
    "kuyrdak": ("misto", "prato_misto", "0.6*n5k:beef|0.3*n5k:potatoes|0.1*n5k:onions", "C", ""),
    "kymyz": ("laticinio", "laticinio", "0.65*n5k:milk|0.35*DILUENTE", "B", "bebida"),
    "lemonade": ("outros", "sem_proteina", "n5k:lemonade", "A", "bebida"),
    "lentiil_soup": ("leguminosa", "leguminosa_soja", "0.3*n5k:lentils|0.7*DILUENTE", "B", "sopa"),
    "lime": ("vegetal", "sem_proteina", "n5k:lime", "A", ""),
    "lulya_kebab": ("misto", "prato_misto", "0.45*n5k:lamb|0.25*n5k:flatbread|0.3*n5k:coleslaw", "C", ""),
    "malibu_salad": ("misto", "prato_misto", "0.3*n5k:ham|0.2*n5k:cheese|0.3*T_PEPINO|0.2*n5k:mayonnaise", "C", ""),
    "manpar": ("misto", "prato_misto", "0.2*n5k:pasta|0.1*n5k:beef|0.2*n5k:mixed vegetables|0.5*DILUENTE", "C", "sopa"),
    "mashed_potato": ("amido", "sem_proteina", "n5k:mashed potatoes", "A", ""),
    "mayonnaise": ("gordura_molho", "sem_proteina", "n5k:mayonnaise", "A", "condimento"),
    "meat_assorty": ("proteina_animal", "suina_embutido", "0.8*n5k:roast beef|0.2*n5k:garden salad", "C", ""),
    "meatball": ("proteina_animal", "bovina", "n5k:meatballs", "B", ""),
    "multifruit": ("outros", "sem_proteina", "n5k:fruit punch", "B", "bebida"),
    "napoleon": ("outros", "sem_proteina", "n5k:cakes", "B", ""),
    "naryn": ("misto", "prato_misto", "0.35*n5k:noodles|0.2*n5k:beef|0.45*DILUENTE", "C", "sopa"),
    "nauryz_koje": ("misto", "prato_misto", "0.5*n5k:buttermilk|0.2*n5k:barley|0.05*n5k:beef|0.25*DILUENTE", "C", "sopa"),
    "nuggets": ("proteina_animal", "frango_peru", "n5k:chicken nuggets", "A", ""),
    "okroshka": ("misto", "prato_misto", "0.5*n5k:buttermilk|0.2*n5k:potatoes|0.1*n5k:eggs|0.1*T_PEPINO|0.1*n5k:bologna", "C", "sopa"),
    "olivie_salad": ("misto", "prato_misto", "0.8*n5k:potato salad|0.1*n5k:eggs|0.1*n5k:bologna", "B", ""),
    "onion": ("vegetal", "sem_proteina", "n5k:onions", "A", ""),
    "orama_nan": ("misto", "prato_misto", "n5k:dumplings", "B", ""),
    "pahlava": ("outros", "sem_proteina", "n5k:pastries", "B", ""),
    "pie": ("misto", "prato_misto", "0.7*" + PAO + "|0.3*n5k:cheese", "C", ""),
    "pirojki": ("misto", "prato_misto", "0.75*n5k:buns|0.25*n5k:mashed potatoes", "C", ""),
    "pistachio": ("outros", "nozes_sementes", "0.53*n5k:pistachios|0.47*DILUENTE", "B", ""),
    "pizza": ("misto", "prato_misto", "n5k:pizza", "A", ""),
    "plov": ("misto", "prato_misto", "0.6*T_ARROZ|0.2*n5k:beef|0.15*T_CENOURA_COZ|0.05*n5k:oil", "B", ""),
    "pod_shuboi": ("misto", "prato_misto", "0.2*n5k:herring|0.3*T_BETERRABA|0.25*n5k:potatoes|0.1*n5k:eggs|0.15*n5k:mayonnaise", "B", ""),
    "pomergranate_juice": ("outros", "sem_proteina", "n5k:grape juice", "B", "bebida"),
    "pop_corn": ("amido", "sem_proteina", "n5k:popcorn", "A", ""),
    "potato": ("amido", "sem_proteina", "n5k:roasted potatoes", "A", ""),
    "quesadilla": ("misto", "prato_misto", "n5k:quesadillas", "B", ""),
    "raisins": ("vegetal", "sem_proteina", "n5k:raisins", "A", ""),
    "ramen": ("misto", "prato_misto", "0.3*n5k:noodles|0.05*n5k:eggs|0.05*n5k:pork|" + SOPA, "C", "sopa"),
    "raspberry_jam": ("outros", "sem_proteina", "n5k:jams", "A", "condimento"),
    "rice": ("amido", "sem_proteina", "T_ARROZ", "A", ""),
    "rice_porridge": ("amido", "laticinio", "0.3*T_ARROZ|0.7*n5k:milk", "B", ""),
    "rollton": ("amido", "sem_proteina", "0.6*n5k:noodles|0.4*DILUENTE", "C", ""),
    "salad_mushrooms": ("vegetal", "sem_proteina", "0.9*n5k:mushroom|0.1*n5k:oil", "B", ""),
    "salmon": ("proteina_animal", "peixe_marisco", "n5k:salmon", "A", ""),
    "samsa": ("misto", "prato_misto", "n5k:empanadas", "B", ""),
    "sandwich": ("misto", "prato_misto", "n5k:sandwiches", "B", ""),
    "sausage": ("proteina_animal", "suina_embutido", "n5k:hot dogs", "B", ""),
    "sausage_in_dough": ("misto", "prato_misto", "0.5*n5k:hot dogs|0.5*n5k:buns", "B", ""),
    "shashlyk_beef": ("proteina_animal", "bovina", "n5k:steak", "B", ""),
    "shashlyk_chicken": ("proteina_animal", "frango_peru", "n5k:grilled chicken", "B", ""),
    "shorpa": ("misto", "prato_misto", "0.15*n5k:beef|0.25*n5k:potatoes|" + SOPA, "C", "sopa"),
    "shubat": ("laticinio", "laticinio", "n5k:milk", "B", "bebida"),
    "sirne": ("misto", "prato_misto", "0.5*n5k:lamb|0.35*n5k:potatoes|0.15*DILUENTE", "C", ""),
    "sorpa": ("outros", "sem_proteina", "0.05*n5k:beef|0.95*DILUENTE", "C", "sopa"),
    "spinach": ("vegetal", "sem_proteina", "n5k:spinach (cooked)", "A", ""),
    "sprite": ("outros", "sem_proteina", "DILUENTE", "A", "bebida"),
    "syrniki": ("laticinio", "laticinio", "0.7*n5k:cottage cheese|0.15*n5k:flour|0.1*n5k:eggs|0.05*n5k:sugar", "B", ""),
    "taba_nan": ("amido", "sem_proteina", PAO, "A", ""),
    "tea_with_milk": ("outros", "sem_proteina", "0.2*n5k:milk|0.8*n5k:tea", "B", "bebida"),
    "tiramisu": ("outros", "sem_proteina", "n5k:cakes", "B", ""),
    "toast": ("amido", "sem_proteina", PAO, "B", ""),
    "tom_yam": ("misto", "prato_misto", "0.1*n5k:shrimp|0.1*n5k:mushroom|0.2*n5k:coconut milk|" + SOPA, "C", "sopa"),
    "tongue_salad": ("misto", "prato_misto", "0.3*n5k:beef|0.1*n5k:cheese|0.4*n5k:garden salad|0.2*n5k:mayonnaise", "C", ""),
    "tuc": ("amido", "sem_proteina", "n5k:crackers", "A", "embalado"),
    "udon": ("misto", "prato_misto", "0.65*n5k:noodles|0.15*n5k:grilled chicken|0.2*n5k:mixed vegetables", "C", ""),
    "vinegret": ("vegetal", "sem_proteina", "0.4*T_BETERRABA|0.3*T_BATATA|0.2*T_CENOURA_COZ|0.1*n5k:pickles", "B", ""),
    "walnut": ("outros", "nozes_sementes", "n5k:walnuts", "A", ""),
    "water": ("outros", "sem_proteina", "DILUENTE", "A", "bebida"),
    "zhal_zhaya": ("proteina_animal", "suina_embutido", "n5k:pastrami", "C", ""),
    "zhent": ("outros", "sem_proteina", "0.4*n5k:millet|0.25*n5k:cottage cheese|0.2*n5k:butter|0.15*n5k:sugar", "C", ""),
}
TIPOS = {"", "bebida", "embalado", "condimento", "sopa"}
_RX_RS = re.compile(r"^(.*?)_(\d+[A-Za-z]{3})_(\d+)(_ref)?_Color_png$")
_RX_PORCAO = re.compile(r"(?:^|_)(small|average|big)(?:_|$)")


def ler_nomes(raiz):
    """Lista de classes de data.yaml, lida como literal (o arquivo é dado de terceiros; nada dele é executado)."""
    with open(os.path.join(raiz, "data.yaml"), encoding="utf-8") as f:
        for l in f:
            if l.startswith("names:"):
                nomes = ast.literal_eval(l.split(":", 1)[1].strip())
                assert isinstance(nomes, list) and all(isinstance(n, str) for n in nomes) and len(nomes) == 138
                return nomes
    raise ValueError("data.yaml sem a linha names:")


def ler_ponteiro(caminho):
    """sha256 e tamanho do arquivo de imagem, do ponteiro LFS (ou do próprio arquivo, se o clone trouxe as imagens)."""
    with open(caminho, "rb") as f:
        ini = f.read(200)
        if ini.startswith(b"version https://git-lfs"):
            m = re.search(rb"oid sha256:([0-9a-f]{64})\s+size (\d+)", ini)
            return m.group(1).decode(), int(m.group(2))
        h = hashlib.sha256(ini)
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest(), os.path.getsize(caminho)


def ler_fotos(raiz, nomes):
    """[(split, nome sem extensão, [(classe, peso, altura da caixa)], sha256, bytes)] e os rótulos sem imagem (B1)."""
    fotos, sem_img = [], []
    for sp in ("train", "val", "test"):
        for arq in sorted(os.listdir(os.path.join(raiz, sp, "labels"))):
            b = arq[:-4]
            img = os.path.join(raiz, sp, "images", b + ".jpg")
            if not os.path.exists(img):
                sem_img.append(f"{sp}/{b}"); continue
            itens = []
            with open(os.path.join(raiz, sp, "labels", arq), encoding="utf-8") as f:
                for l in f:
                    p = l.split()
                    if p:
                        assert len(p) == 6, (arq, l)
                        itens.append((nomes[int(p[0])], float(p[5]), float(p[4])))
            fotos.append((sp, b, itens) + ler_ponteiro(img))
    return fotos, sem_img


def pesos_por_classe(itens):
    """{classe: gramas} somando as caixas; None para a classe com alguma caixa sem peso."""
    tot = {}
    for c, g, _ in itens:
        tot[c] = None if (g <= 0 or tot.get(c, 0) is None) else tot.get(c, 0) + g
    return tot


def cmd_mapa(raiz):
    tab = tabelas(); nomes = ler_nomes(raiz)
    assert set(nomes) == set(C), set(nomes) ^ set(C)
    fotos, _ = ler_fotos(raiz, nomes)
    n_fotos = Counter(); pesos = defaultdict(list); sem = Counter()
    for _, _, itens, _, _ in fotos:
        for c, g in pesos_por_classe(itens).items():
            n_fotos[c] += 1
            if g is None:
                sem[c] += 1
            else:
                pesos[c].append(g)
    with open(MAPA, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["classe", "fotos", "fotos_sem_peso", "mediana_g", "grupo", "fonte", "ref", "proteina_100g", "confianca", "tipo"])
        for c in nomes:
            g, fo, ref, conf, tipo = C[c]
            assert g in GRUPOS and fo in FONTES and conf in "ABC" and tipo in TIPOS, c
            v = sorted(pesos[c])
            w.writerow([c, n_fotos[c], sem[c], f"{v[len(v) // 2]:.0f}" if v else "", g, fo, ref, f"{densidade(ref, tab):.2f}", conf, tipo])
    print(MAPA, len(nomes), "classes |", dict(Counter(C[c][3] for c in nomes)), "|", dict(Counter(C[c][4] or "comida" for c in nomes)))


def vista_de(nome_base, alturas_sessao):
    """(dispositivo, vista, tem_referencia, avisos). `alturas_sessao`: maior caixa de cada quadro da sessão (dia 09Dec)."""
    if "Color_png" not in nome_base:
        return "celular", "obliqua", int(bool(re.search(r"(?<!no)_ref_", nome_base))), []
    m = _RX_RS.match(nome_base)
    if not m:                                # 2ª campanha (nomes sem data: depth/RGB x ref/no_ref): tudo de cima
        n = nome_base.lower()
        return "realsense", "zenital", int("ref" in n and not re.search(r"n[op][_-]?ref", n)), []
    por_numero = "zenital" if int(m.group(3)) <= 4 else "lateral"
    if m.group(2) != DIA_IRREGULAR:
        return "realsense", por_numero, int(bool(m.group(4))), []
    alt, todas = alturas_sessao
    if max(todas) / max(min(todas), 1e-6) >= RAZAO_VISTAS:                     # V2
        return "realsense", ("zenital" if alt > (max(todas) * min(todas)) ** 0.5 else "lateral"), int(bool(m.group(4))), []
    return "realsense", por_numero, int(bool(m.group(4))), ["vista_incerta"]


def cmd_monta(raiz, saida):
    tab = tabelas(); nomes = ler_nomes(raiz)
    with open(MAPA, encoding="utf-8", newline="") as f:
        mapa = {r["classe"]: r for r in csv.DictReader(f)}
    for c in nomes:                                           # o mapa gravado tem de ser o que a tabela dá hoje
        r = mapa[c]
        assert (r["grupo"], r["fonte"], r["ref"], r["confianca"], r["tipo"]) == C[c], c
        assert abs(float(r["proteina_100g"]) - densidade(r["ref"], tab)) < 0.006, c
    dens = {c: float(mapa[c]["proteina_100g"]) for c in nomes}
    tipo = {c: C[c][4] for c in nomes}
    fotos, sem_img = ler_fotos(raiz, nomes)
    cont = Counter(B1_rotulo_sem_imagem=len(sem_img), imagens_lidas=len(fotos))

    # B2: a mesma imagem (sha256) duas vezes
    por_sha = defaultdict(list)
    for f in fotos:
        por_sha[f[3]].append(f)
    unicas = []
    for sha, v in por_sha.items():
        pesos = {tuple(sorted((c, round(g, 1) if g is not None else None) for c, g in pesos_por_classe(f[2]).items())) for f in v}
        if len(v) > 1:
            cont["B2_copias_retiradas"] += len(v) - 1
        if len(pesos) > 1:
            cont["B2_copias_com_pesos_diferentes"] += len(v); cont["B2_copias_retiradas"] -= len(v) - 1; continue
        unicas.append(min(v, key=lambda f: (f[0] != "train", f[1])))

    # sessões da RealSense do dia irregular (V2)
    alturas = defaultdict(list)
    for sp, b, itens, _, _ in unicas:
        m = _RX_RS.match(b.split(".rf.")[0])
        if m and m.group(2) == DIA_IRREGULAR:
            alturas[m.group(1)].append(max(a for _, _, a in itens))

    # pratos: assinatura = itens com peso; a assinatura contida em outra do mesmo prefixo fica no prato maior
    def prefixo(base):
        m = _RX_RS.match(base)
        if m:
            return m.group(1) + "@" + m.group(2)
        return re.split(r"_(?:RGB|rgb|RGb)_", base)[0] if re.search(r"_(?:RGB|rgb|RGb)_", base) else ""
    regs = []
    for sp, b, itens, sha, tam in unicas:
        base = b.split(".rf.")[0]
        pc = pesos_por_classe(itens)
        comida = {c: g for c, g in pc.items() if tipo[c] != "bebida"}
        if not comida or all(g is None for g in comida.values()):
            cont["B3_so_bebida" if not comida else "B3_comida_sem_peso"] += 1; continue
        ass = frozenset((c, round(g, 1)) for c, g in pc.items() if g is not None)
        regs.append(dict(sp=sp, b=b, base=base, pc=pc, comida=comida, ass=ass, sha=sha, tam=tam, pre=prefixo(base),
                         alt=max(a for _, _, a in itens)))
    por_pre = defaultdict(set)
    for r in regs:
        por_pre[r["pre"]].add(r["ass"])
    maior = {}
    for pre, conj in por_pre.items():
        for a in conj:
            sup = [b for b in conj if a < b]
            tops = [b for b in sup if not any(b < c for c in conj)]
            maior[(pre, a)] = tops[0] if len(tops) == 1 else a      # contida em duas bandejas diferentes: fica sozinha
    pai = {}

    def acha(x):
        while pai.setdefault(x, x) != x:
            pai[x] = pai[pai[x]]; x = pai[x]
        return x
    for r in regs:
        r["prato_ass"] = maior[(r["pre"], r["ass"])]
        r["prato"] = "fpb_p" + hashlib.sha1(repr(sorted(r["prato_ass"])).encode()).hexdigest()[:10]
        for c, g in r["prato_ass"]:
            if tipo[c] in ("", "sopa"):
                pai[acha(("prato", r["prato"]))] = acha(("porcao", c, g))
    menor = {}                                               # nome do grupo = menor prato do componente (estável)
    for r in regs:
        raiz_g = acha(("prato", r["prato"]))
        menor[raiz_g] = min(menor.get(raiz_g, r["prato"]), r["prato"])
    for r in regs:
        r["grupo"] = "fpb_g" + menor[acha(("prato", r["prato"]))][5:]

    base_cab = confere_esquema()
    cab = base_cab + ["dominio", "grupo", "vista", "split_ext", "arquivo", "sha256", "bytes", "dispositivo", "porcao",
                      "tem_referencia", "split_autores", "prato", "componentes", "n_componentes", "bebida_g", "embalado_g",
                      "massa_pesada_g", "proteina_fraca_g", "proteina_fraca_conf", "proteina_fraca_fontes_g",
                      "maior_fonte_fraca_g", "avisos", "licenca", "uso_declarado"]
    linhas = []; ids = Counter()
    for r in sorted(regs, key=lambda r: (r["base"], r["b"])):
        disp, vista, ref, avisos = vista_de(r["base"], (r["alt"], alturas.get(r["pre"].split("@")[0], [r["alt"]])))
        mg = dict.fromkeys(GRUPOS, 0.0); ff = dict.fromkeys(FONTES, 0.0)
        pesada = beb = emb = pf = pff = maior_f = 0.0; por_conf = Counter(); comp = []
        for c, g in sorted(r["pc"].items()):
            if g is None:
                comp.append(f"{c}:?"); continue
            if tipo[c] == "bebida":
                beb += g; comp.append(f"{c}:{g:.0f}"); continue
            pesada += g; mg[C[c][0]] += g; ff[C[c][1]] += g
            emb += g if tipo[c] == "embalado" else 0
            p = g * dens[c] / 100
            pf += p; por_conf[C[c][3]] += p
            if C[c][1] != "sem_proteina":
                pff += p; maior_f = max(maior_f, p)
            comp.append(f"{c}:{g:.0f}")
        parcial = any(g is None for g in r["comida"].values())
        cont["B4_peso_parcial"] += parcial
        avisos += [a for a, s in (("peso_parcial", parcial), ("item_sem_anotacao", r["ass"] != r["prato_ass"]),
                                  ("tem_sopa", any(tipo[c] == "sopa" for c in r["comida"])),
                                  ("so_condimento", all(tipo[c] == "condimento" for c in r["comida"])),
                                  ("tem_embalado", emb > 0)) if s]
        split = "teste_ext" if int(hashlib.sha1(r["grupo"].encode()).hexdigest(), 16) % 5 == 0 else "treino_ext"
        ids[r["base"]] += 1
        dish = "fpb_" + r["base"] + ("" if ids[r["base"]] == 1 else f"~{ids[r['base']]}")
        m_por = _RX_PORCAO.search(r["base"])
        alvos = ([f"{pesada:.1f}", "", "", "", ""] + [f"{mg[g]:.1f}" for g in GRUPOS] + [""] * len(GRUPOS)
                 + [f"{ff[k] / pesada:.4f}" for k in FONTES]) if not parcial else [""] * (5 + 2 * len(GRUPOS) + len(FONTES))
        linhas.append([dish, -1, -1, ""] + alvos
                      + ["fpb", r["grupo"], vista, split, f"{PREFIXO_ARQ}{r['sp']}/images/{r['b']}.jpg", r["sha"], r["tam"], disp,
                         m_por.group(1) if m_por else "", ref, r["sp"], r["prato"], ";".join(comp), len(r["comida"]),
                         f"{beb:.0f}", f"{emb:.0f}", f"{pesada:.1f}", f"{pf:.1f}", max(por_conf, key=por_conf.get),
                         f"{pff:.1f}", f"{maior_f:.1f}", "|".join(avisos), LICENCA, USO])
    assert len({l[0] for l in linhas}) == len(linhas) and len({l[cab.index("sha256")] for l in linhas}) == len(linhas)
    lado = {}
    for l in linhas:
        assert lado.setdefault(l[cab.index("grupo")], l[cab.index("split_ext")]) == l[cab.index("split_ext")]
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(cab); w.writerows(linhas)
    resumo(linhas, cab, cont, saida)


def resumo(linhas, cab, cont, saida):
    i = {k: cab.index(k) for k in cab}
    print("regras:", dict(cont))
    print(f"fotos (linhas): {len(linhas)} | pratos: {len({l[i['prato']] for l in linhas})} | grupos: {len({l[i['grupo']] for l in linhas})}"
          f" | maior grupo: {Counter(l[i['grupo']] for l in linhas).most_common(1)[0][1]} fotos")
    for nome, sel in (("TODAS", linhas), ("ZENITAL", [l for l in linhas if l[i["vista"]] == "zenital"]),
                      ("OBLIQUA (celular)", [l for l in linhas if l[i["vista"]] == "obliqua"]),
                      ("LATERAL", [l for l in linhas if l[i["vista"]] == "lateral"])):
        com = [l for l in sel if l[i["massa"]] != ""]
        if not com:
            print(f"  {nome}: {len(sel)} fotos"); continue
        m = sorted(float(l[i["massa"]]) for l in com); q = lambda p: m[int(p * (len(m) - 1))]
        alto = [l for l in com if float(l[i["proteina_fraca_g"]]) > PROTEINA_FRACA_ALTA_G]
        alto_f = [l for l in com if float(l[i["proteina_fraca_fontes_g"]]) > PROTEINA_FRACA_ALTA_G]
        um = [l for l in com if float(l[i["maior_fonte_fraca_g"]]) > PROTEINA_FRACA_ALTA_G]
        print(f"  {nome}: {len(sel)} fotos, {len({l[i['prato']] for l in sel})} pratos | com massa: {len(com)} fotos,"
              f" {len({l[i['prato']] for l in com})} pratos | split {dict(Counter(l[i['split_ext']] for l in com))}"
              f" | {sum(int(l[i['bytes']]) for l in sel) / 1e9:.2f} GB")
        print(f"     massa sem bebida (g): p10 {q(.1):.0f} mediana {q(.5):.0f} p90 {q(.9):.0f} máx {m[-1]:.0f}"
              f" | proteína FRACA > 25 g: {len(alto)} fotos / {len({l[i['prato']] for l in alto})} pratos"
              f" | só das fontes > 25 g: {len(alto_f)} / {len({l[i['prato']] for l in alto_f})}"
              f" | um item-fonte > 25 g: {len(um)} / {len({l[i['prato']] for l in um})}")
    com = [l for l in linhas if l[i["massa"]] != ""]
    tot = sum(float(l[i["massa"]]) for l in com)
    print("  avisos:", dict(Counter(a for l in linhas for a in l[i["avisos"]].split("|") if a)))
    print("  massa por grupo %:", {g: round(100 * sum(float(l[i[f'm_{g}']]) for l in com) / tot, 1) for g in GRUPOS})
    print("  pratos com a fonte (>= 20 g de comida da fonte):",
          {k: len({l[i['prato']] for l in com if float(l[i[f'ff_{k}']]) * float(l[i['massa']]) >= 20}) for k in FONTES})
    print("  confiança da proteína fraca (fotos):", dict(Counter(l[i["proteina_fraca_conf"]] for l in linhas)))
    print("gravado:", os.path.relpath(saida), f"({len(cab)} colunas)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "mapa":
        cmd_mapa(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "monta":
        cmd_monta(sys.argv[2], sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else SAIDA)
    else:
        sys.exit(__doc__)
