# Nutrelle

**Fotografe o prato e veja a proteína.** App iOS de proteína para mulheres que estão mudando o corpo — com modo caneta (GLP-1) e IA rodando no próprio celular.

## Documentos

- [`docs/produto.md`](docs/produto.md) — definição de produto: posicionamento, funcionalidades da v1, preços por país, ordem de lançamento, stack.
- [`docs/dados/precisao/resultado_modelo_treinado.md`](docs/dados/precisao/resultado_modelo_treinado.md) — resultados de precisão (LLM, modelo treinado, híbrido); resumo em `docs/produto.md` §7.
- [`docs/coleta-brasil.md`](docs/coleta-brasil.md) — como coletar pratos brasileiros pesados com foto de cima; planilhas e validador em `eval/coleta_br/`.
- [`docs/prototipo-precisao.md`](docs/prototipo-precisao.md) — protocolo do teste de precisão da foto com ~100 pratos brasileiros.
- [`docs/pesquisa-mercado.md`](docs/pesquisa-mercado.md) — setores de apps para mulheres (Google Trends 12 m / 3 m + RevenueCat 2026).
- [`docs/concorrencia-precificacao.md`](docs/concorrencia-precificacao.md) — 60 concorrentes, preços por país (App Store/Play), lacunas.
- `docs/dados/` — dados brutos (séries do Trends, fichas e avaliações das lojas) e scripts de coleta.

## Ferramentas

- `eval/score_precisao.py` — pontua as estimativas de proteína contra o gabarito pesado (`python3 eval/score_precisao.py gabarito.csv estimativas.csv`).
