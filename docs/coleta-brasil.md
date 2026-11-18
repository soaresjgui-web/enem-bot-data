# Coleta de pratos brasileiros pesados, com foto de cima

> Complementa [`prototipo-precisao.md`](prototipo-precisao.md). Aqui está o **como fazer** da coleta que falta: nenhuma base pública tem comida brasileira fotografada de cima com o peso medido (ver [`dados/precisao/datasets_internacionais.md`](dados/precisao/datasets_internacionais.md)), e todos os resultados de [`dados/precisao/resultado_modelo_treinado.md`](dados/precisao/resultado_modelo_treinado.md) vêm de um refeitório americano.

## 1. Para que serve

1. **Medir** o modelo atual em comida brasileira sob a premissa do produto (uma única foto de cima). Hoje esse número não existe.
2. **Treinar**: adaptar o modelo de peso e a identificação aos pratos, recipientes e porções daqui.

Os dois usos pedem os mesmos dados; a separação entre treino e teste é feita depois, por **refeição** (nunca por foto).

## 2. Quanto coletar

| Fase | Pratos | Para quê |
|---|---|---|
| Piloto | 50 | Validar o protocolo, o tempo por prato e a primeira medida do modelo |
| Teste | 300 | Medir com margem de erro de cerca de ±5 pontos |
| Treino | 400–800 | Adaptar o modelo; a curva de aprendizado no Nutrition5k estabiliza a partir de ~400 pratos |

**Mistura sugerida** (a mesma de `prototipo-precisao.md`): almoço tipo PF, marmita, café da manhã, lanches, jantar. Pelo menos **40% dos pratos com mais de 25 g de proteína**: é a faixa em que o modelo mais erra (54–59% de acerto).

## 3. Material

- Balança de cozinha com resolução de 1 g (conferir com um peso conhecido no início de cada sessão).
- iPhone (de preferência com LiDAR: o mapa de profundidade é salvo como sinal de treino; a inferência continua sendo só a foto).
- Os pratos e recipientes do dia a dia: prato raso, prato fundo, marmita de alumínio, pote, tigela. **Variar** é importante; anotar o diâmetro do prato.
- Fundo e luz variados (mesa de madeira, toalha, bancada; luz de janela e luz artificial).

## 4. Passo a passo por prato

1. **Tarar** a balança com o prato vazio. Anotar o tipo e o diâmetro do recipiente.
2. **Servir um componente de cada vez**, anotando o peso de cada um logo depois de servir (peso acumulado ou com nova tara): arroz, feijão, carne, salada, molho. Pratos que não dá para separar (feijoada, estrogonofe, lasanha) entram como um componente com o nome do prato pronto.
3. Para cada componente, anotar o **alimento da TACO** mais próximo e o **preparo** (grelhado, cozido, frito, assado, cru). Se houver rótulo (marmita pronta, iogurte, pão de forma), anotar a proteína por 100 g do rótulo: ela vale mais que a tabela.
4. **Foto principal, de cima:** celular paralelo à mesa, prato inteiro dentro do quadro, ocupando cerca de 80% da largura (é o que o círculo-guia do app vai pedir). Sem mão, sem sombra forte do celular.
5. **Fotos extras do mesmo prato** (valem para treino de robustez, não contam como pratos novos): uma mais de perto, uma mais de longe, uma levemente inclinada (10–15°), uma com outra luz.
6. Não fotografar rostos, documentos, telas nem embalagens com dados pessoais. As fotos **não entram no repositório**, que é público: ficam num armazenamento privado, e o repositório guarda só as planilhas.

Tempo esperado: 4 a 6 minutos por prato depois de pegar o ritmo.

## 5. O que registrar

Duas planilhas CSV (modelos em `eval/coleta_br/modelo_pratos.csv` e `eval/coleta_br/modelo_componentes.csv`):

- **pratos:** `prato_id`, `refeicao_id` (o mesmo para todas as fotos da mesma refeição servida), `data`, `categoria`, `recipiente`, `diametro_cm`, `foto_principal`, `fotos_extras`, `aparelho`, `tem_profundidade`, `quem_coletou`, `observacoes`.
- **componentes:** `prato_id`, `componente`, `alimento_taco`, `preparo`, `gramas`, `proteina_100g_rotulo` (vazio quando não há rótulo).

`python3 eval/coleta_br/valida.py pratos.csv componentes.csv` confere as planilhas (ids, somas, nomes da TACO, faixas plausíveis), calcula a proteína de cada prato e mostra quantos pratos há por faixa de proteína. Com `--gabarito saida.csv`, grava o gabarito no mesmo formato de `docs/dados/precisao/gabarito_pratos_v2.csv`.

## 6. Quem pode coletar

- **Você e pessoas próximas**, nas refeições de casa (piloto).
- **Restaurante por quilo parceiro:** a balança já faz parte do fluxo; pesar por componente exige combinar com o restaurante.
- **Curso de Nutrição** (laboratório de técnica dietética): tem balanças, fichas técnicas e interesse acadêmico; é a fonte com melhor qualidade de rótulo.
- **Usuárias do beta com balança**, com consentimento explícito para uso das fotos em treino (dado sensível de saúde: LGPD).

## 7. Critérios de qualidade

- Soma dos componentes confere com o peso total do prato (±3 g).
- Pelo menos 5 recipientes diferentes e 3 ambientes de luz em cada 50 pratos.
- Nenhuma refeição com fotos em mais de um conjunto (treino e teste).
- Proteína pela tabela conferida com o rótulo sempre que houver rótulo.
