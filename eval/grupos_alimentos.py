"""Grupos grossos de alimentos usados no pipeline v2 (calibração e seletor por componente).

grupo_de(nome, grupo_taco=None) -> um de GRUPOS. Funciona com nomes da TACO (pt), da tabela de
medidas do IBGE (pt, maiúsculas sem acento) e do Nutrition5k/USDA (en). Regras por palavra-chave,
na ordem: misto > proteína animal > leguminosa > laticínio > amido > gordura/molho > vegetal.
"""
import re
import unicodedata

GRUPOS = ["proteina_animal", "leguminosa", "laticinio", "amido", "vegetal", "gordura_molho", "misto", "outros"]

# palavras inteiras (sem acento, minúsculas); "*" no fim = prefixo
_KW = {
    "misto": """pizza* lasanha* lasagna burrito* sandwich* sanduiche* hamburguer* burger* cheeseburger* taco tacos
        enchilada* quesadilla* calzone* empanada* empada* pastel coxinha* esfiha* quiche* pot_pies gumbo jambalaya
        paella chili stews curries goulash feijoada estrogonofe strogonoff escondidinho* risoto risotto
        macaroni_and_cheese fried_rice chow_mein lo_mein sushi wraps gyros fajitas chimichangas nachos
        baiao* acaraje vatapa moqueca* yakisoba salpicao cuscuz_paulista virado* tutu""",
    "proteina_animal": """carne* bovin* frango* galinha peru chester pato porco* suin* lombo pernil costela* bisteca
        bife* file* fil* alcatra picanha patinho acem musculo cupim contra-file contrafile maminha fraldinha
        lagarto hamburguer linguica* salsicha* presunto* mortadela salame bacon toucinho torresmo charque
        carne-seca figado* coracao* moela* almondega* kibe quibe peixe* pescad* atum sardinha* salmao tilapia
        merluza bacalhau pescada corvina cacao camarao* lula polvo marisco* mexilhao ostra* caranguejo siri
        lagosta ovo ovos clara gema omelete* frango* steak beef chicken* turkey* duck pork* ham sausage*
        bacon fish salmon tuna tilapia cod halibut trout shrimp crab lobster scallops clams mussels oysters
        squid calamari octopus eel herring sardines anchovies mackerel catfish snapper swordfish flounder
        haddock carp mahi artic_char veal bison lamb* brisket meatballs meatloaf pastrami bologna salami
        pepperoni jerky ribs hot_dogs corned roast egg eggs omelets scrambled poached fried_eggs
        hard_boiled egg_whites egg_yolks crawfish seafood tempura protein_powder""",
    "leguminosa": """feijao* lentilha* grao-de-bico grao_de_bico ervilha* soja tofu tempeh edamame
        beans bean(seed) lentils chickpeas hummus falafel peas green_peas black_beans pinto_beans kidney_beans
        refried_beans baked_beans lima_beans white_beans soy_milk soy_nuts soy_yogurt soy_sausage""",
    "laticinio": """queijo* requeijao leite* iogurte* coalhada ricota mussarela muçarela parmesao provolone
        cheese* cottage milk yogurt* greek_yogurt buttermilk cream_cheese kefir""",
    "amido": """arroz* macarrao* massa* espaguete talharim nhoque pao* paes torrada* biscoito* bolacha* bolo*
        farofa farinha* cuscuz tapioca beiju polenta angu fuba milho* aveia granola cereal* batata* mandioca
        aipim macaxeira inhame cara mandioquinha baroa* rice* pasta* spaghetti noodles macaroni orzo gnocchi
        ravioli tortellini bread* toast bagels buns rolls croissants muffins biscuits tortilla* pita naan
        flatbread focaccia crackers pretzels chips quinoa couscous bulgur barley millet oats oatmeal granola
        cereal polenta grits hominy corn* potato* potatoes hash_browns french_fries sweet_potato yam
        tabouli stuffing pancakes waffles crepes french_toast pie_crust pizza_dough wheat_berry wild_rice
        brown_rice white_rice country_rice""",
    "gordura_molho": """oleo azeite manteiga margarina maionese molho* creme_de_leite nata catchup ketchup mostarda
        oil olive_oil vegetable_oil butter margarine mayonnaise dressing* vinaigrette sauce* gravy pesto
        salsa guacamole sour_cream cream half_and_half coffee_creamer syrup honey sugar jams spreads
        tahini peanut_butter almond_butter soy_sauce teriyaki_sauce barbecue_sauce
        salt vinegar pepper mustard broth sal vinagre caldo*""",
    "vegetal": """alface tomate* cenoura* brocolis couve* repolho abobrinha abobora chuchu pepino pimentao
        cebola* alho beterraba* espinafre agriao rucula vagem quiabo jilo berinjela salada* legume* hortali*
        banana* maca* laranja* mamao manga uva* abacaxi melancia melao morango* pera goiaba acai abacate
        kiwi limao* fruta* lettuce tomatoes cherry_tomatoes carrot* broccoli cauliflower cabbage kale spinach*
        cucumbers celery bell_peppers onions garlic mushroom* asparagus green_beans zucchini squash eggplant
        beets radishes arugula chard collards greens mixed_greens salads garden_salad caesar_salad
        apple banana orange grapes berries strawberries blueberries raspberries blackberries pineapple
        melon* cantaloupe watermelon peach* pears plums mangos papayas avocado lemon lime figs dates
        raisins cherries apricots pomegranate okra artichokes leeks fennel parsley shallots thyme cilantro
        basil rosemary chive ginger bok_choy brussels_sprouts jalapenos olives cranberries tomatillo coleslaw
        salsinha cebolinha coentro manjericao alecrim azeitona* gengibre pimenta""",
}


def normaliza(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9()\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _compila():
    regras = []
    for grupo, txt in _KW.items():
        # termos separados por espaço; "_" junta termos de várias palavras ("grao_de_bico")
        termos = sorted({normaliza(t.replace("_", " ")) + ("*" if t.endswith("*") else "") for t in txt.split()},
                        key=len, reverse=True)
        pats = []
        for t in termos:
            if not t:
                continue
            if t.endswith("*"):
                pats.append(re.escape(t[:-1]) + r"\w*")
            else:
                pats.append(re.escape(t))
        regras.append((grupo, re.compile(r"(?<![a-z])(?:" + "|".join(pats) + r")(?![a-z])")))
    return regras


_REGRAS = None
_TACO_GRUPO = {
    "Carnes e derivados": "proteina_animal", "Pescados e frutos do mar": "proteina_animal",
    "Ovos e derivados": "proteina_animal", "Leguminosas e derivados": "leguminosa",
    "Leite e derivados": "laticinio", "Cereais e derivados": "amido",
    "Verduras, hortaliças e derivados": "vegetal", "Frutas e derivados": "vegetal",
    "Gorduras e óleos": "gordura_molho", "Alimentos preparados": "misto",
}


def grupo_de(nome, grupo_taco=None):
    """Grupo grosso de um alimento. grupo_taco (coluna 'grupo' do taco.csv) tem precedência,
    exceto 'Alimentos preparados', que é refinado pelas palavras-chave."""
    global _REGRAS
    if grupo_taco and grupo_taco in _TACO_GRUPO and _TACO_GRUPO[grupo_taco] != "misto":
        # TACO classifica batata/mandioca em "Verduras, hortaliças"; corrige para amido
        g = _TACO_GRUPO[grupo_taco]
        n = normaliza(nome)
        if g == "vegetal" and re.search(r"(?<![a-z])(batata|mandioca|aipim|inhame|cara|mandioquinha|milho)", n):
            return "amido"
        return g
    if _REGRAS is None:
        _REGRAS = _compila()
    n = normaliza(nome)
    for grupo, rx in _REGRAS:
        if rx.search(n):
            return grupo
    return "misto" if grupo_taco == "Alimentos preparados" else "outros"
