#!/usr/bin/env python3
"""Treino em GPU (Kaggle/Colab) do modelo de proteína pela foto: seções 8, 10, 11 e 12 de resultado_modelo_treinado.md.

Premissa do produto: a inferência usa UMA foto RGB tirada de cima. Critério: proteína dentro de ±20% ou ±5 g.

Uso (Kaggle/Colab, pelo notebook nutrelle_treino_gpu.ipynb, ou direto):
  python3 treino_gpu.py                       execução 3 (padrão): as 13 configurações da seção 12.3, até o orçamento
  python3 treino_gpu.py --externo flavoria    execução 3 com a base externa da seção 12.5 (é como o notebook chama)
  python3 treino_gpu.py --rodada 2            as 11 configurações da execução 2 (seção 11); --rodada 1: as 15 da seção 10
  python3 treino_gpu.py --lista               mostra as configurações da rodada e sai
  python3 treino_gpu.py --so "ft siglip2"     só as configurações cujo nome contém o texto
  python3 treino_gpu.py --smoke               teste rápido em CPU (backbone minúsculo, poucas dezenas de pratos)
  python3 treino_gpu.py --checa-modelos       monta cada backbone sem pesos e passa uma imagem (sem rede)
  python3 treino_gpu.py --autoteste           testes da máquina de estados, do escalonador, da regra de adoção da base
                                              externa e dos alvos aninhados

Execução 3 (seção 12.3; um modelo só, sem LLM):
  - treino = listas/treino_v2.csv (2.281 pratos, folds novos por cadeia, frações da massa por fonte de proteína);
    previsões só para o teste2 (listas/aval2_ids.csv). As fotos do teste2 só passam pelos modelos finais, para gerar
    previsões: não entram em treino, destilação, nem em nada sem rótulo. Os 450 pratos da avaliação antiga não são previstos.
  - 1–5: cinco backbones congelados + ridge; 6–8: cabeça de frações por fonte sobre três deles; 9: professor (média em
    log das massas ridge × média em log das densidades); 10: U1, um backbone congelado + MLP com destilação;
    11–12: U4 e U3, ajuste fino com cabeça de frações, peso por faixa e destilação; 13: U0, a receita r2 sem mudança.
  - destilação com alvos ANINHADOS: o aluno treinado sem o fold k recebe, para um prato do fold j, a previsão de um
    professor treinado sem os folds j e k. Assim a CV do aluno nunca enxerga rótulos do fold k (conferido no autoteste).
  - toda previsão sai com 1, 4 e 8 orientações (sufixos _o1, _o4 e nenhum), fora-do-fold e no teste2.
  - escalonador: dependências entre configurações, ordem estrita de prioridade nos alunos e simulação do orçamento:
    a configuração que não cabe inteira no tempo que resta não começa (fica 'sem_orcamento').

Base externa (seção 12.5 e seu adendo; --externo flavoria): FlavoriaFoodWeight1700, Zenodo 5850856, CC BY 4.0.
  - rótulos em listas/externo_flavoria.csv (uma linha por foto); as fotos vêm do .tar do Zenodo (2,58 GB), baixado numa
    linha à parte desde o início, com retomada, md5 conferido e prazo de 45 min. Se a lista não passar nas conferências
    ou o download falhar, a execução segue como em 12.3, e o manifesto diz por quê.
  - entram só a massa total (peso 1), a massa por grupo e as frações por fonte (peso 0,5), só no treino (fold −1), com a
    soma dos pesos igual à metade da do Nutrition5k e o viés de log-massa por domínio retirado; a foto quadrada vai para
    o centro de um quadro 4:3. Fotos com oclusão ficam fora; 20% dos pratos (split_ext = teste_ext) nunca treinam.
  - as configurações 1 a 10 rodam também na versão '+flavoria'; 'decisao flavoria' aplica a regra de adoção às previsões
    fora-do-fold dos dois professores (regra_adocao, função pura) e grava decisao_flavoria.json; U4 e U3 rodam uma vez,
    na versão indicada; U0 é sempre só com o treino_v2. relato_externo.json traz o erro de massa no teste_ext (por
    prato, só o quadro preferido), com e sem a base: não escolhe nada.

O que o script faz, em qualquer rodada:
  1. Baixa do bucket público do Nutrition5k as fotos zenitais dos pratos das listas, em paralelo, com novas tentativas
     e retomada.
  2. Para cada configuração (ordem = prioridade): previsões fora-do-fold no treino (5 folds por cadeia, coluna `fold`
     da lista) + previsões nos pratos de avaliação por um modelo treinado em todo o treino.
  3. Grava o que é preciso para servir: pesos__<config>.pt (ajuste fino, meia precisão), cabeca__<config>.npz (ridge ou
     MLP sobre backbone congelado; o backbone público não é regravado: ficam a URL, o tamanho e o sha256) e
     servir__<config>.json (pré-processamento, orientações, saídas).
  4. Fase de serviço, depois de todo o treino: remonta cada modelo só com esses arquivos, confere que ele reproduz as
     previsões gravadas e mede latência (uma T4 em fp16; CPU com o número de threads registrado). Resultado em servico.json.
  5. Tudo na pasta de saída, sem subpastas: pred__<config>.npz/.csv, log__<config>.json, parte__*.npz (retomada),
     manifesto.json, eventos.jsonl, tamanhos.json e treino_gpu.log.

Higiene: este script NÃO lê nenhum gabarito de avaliação (calib/teste/n5k_v1/teste2): só grava previsões; os pratos de
avaliação nunca entram no treino (o script confere as interseções); sem parada antecipada (épocas fixas por receita).

Retomada: cada parte (fold0..fold4, final; tudo nas etapas sobre features congeladas) vira um arquivo parte__*.npz ao
terminar. Ao iniciar, o script copia partes de --retoma (ou de /kaggle/input/**, se houver um manifesto.json com as
mesmas listas) e pula o que já existe e está completo.
"""
import argparse, csv, glob, hashlib, json, math, os, shutil, subprocess, sys, tarfile, tempfile, threading, time, traceback, zlib
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from types import SimpleNamespace
import numpy as np

FORMATO = "nutrelle-treino-gpu/3"          # /2 = /1 + teste2, arquivos de servir e servico.json; /3 = /2 + rodada 3 (seção 12)
FORMATOS_ACEITOS = ("nutrelle-treino-gpu/1", "nutrelle-treino-gpu/2", FORMATO)
AQUI = os.path.dirname(os.path.abspath(__file__))
BUCKET = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead"
SPLITS = ["calib", "teste", "n5k_v1"]
EPS_D = 0.01
FAIXAS = np.array([5, 15, 25, 40])   # faixas de proteína do sorteio do v2
I_M, I_P, I_K, I_D, I_F, I_C = 0, 1, 2, 3, 4, 5
SAI_OK, SAI_SEM_ORCAMENTO, SAI_MEMORIA = 0, 3, 4

# ------------------------------------------------------------------ backbones e configurações (pré-declaradas)
_BV, _AUG, _FB = ("https://storage.googleapis.com/big_vision/siglip2/", "https://storage.googleapis.com/vit_models/augreg/",
                  "https://dl.fbaipublicfiles.com/dinov2/")
BACKBONES = {
    # chave: timm = arquitetura.etiqueta dos pesos no Hugging Face; hw = (H, W) da entrada; url = o mesmo checkpoint no
    # servidor de quem publicou; fontes = ordem em que as origens são tentadas. SigLIP 2: primeiro o .npz do big_vision,
    # que é o arquivo usado na CPU (zen_extrai.py); DINOv2: primeiro o Hugging Face.
    "siglip2_b16_224": dict(timm="vit_base_patch16_siglip_224.v2_webli", hw=(224, 224), url=_BV + "siglip2_b16_224.npz", fontes=("url", "hf")),  # foto inteira achatada, como na CPU
    "siglip2_b16_384x512": dict(timm="vit_base_patch16_siglip_512.v2_webli", hw=(384, 512), url=_BV + "siglip2_b16_512.npz", fontes=("url", "hf")),  # proporção original, pesos de 512 px
    "siglip2_l16_384x512": dict(timm="vit_large_patch16_siglip_512.v2_webli", hw=(384, 512), url=_BV + "siglip2_l16_512.npz", fontes=("url", "hf")),
    "siglip2_so400m16_384x512": dict(timm="vit_so400m_patch16_siglip_512.v2_webli", hw=(384, 512), url=_BV + "siglip2_so400m16_512.npz", fontes=("url", "hf")),
    "dinov2_b14_336x448": dict(timm="vit_base_patch14_reg4_dinov2.lvd142m", hw=(336, 448), url=_FB + "dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth", fontes=("hf", "url")),
    "dinov2_l14_336x448": dict(timm="vit_large_patch14_reg4_dinov2.lvd142m", hw=(336, 448), url=_FB + "dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth", fontes=("hf", "url")),
    "dinov2_l14_224x308": dict(timm="vit_large_patch14_reg4_dinov2.lvd142m", hw=(224, 308), url=_FB + "dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth", fontes=("hf", "url")),
    # só para --smoke
    "vit_tiny_224": dict(timm="vit_tiny_patch16_224.augreg_in21k", hw=(224, 224), url=_AUG + "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", fontes=("url", "hf")),
    "vit_tiny_96x128": dict(timm="vit_tiny_patch16_224.augreg_in21k", hw=(96, 128), url=_AUG + "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz", fontes=("url", "hf")),
}
PASTA_PESOS = os.path.join(tempfile.gettempdir(), "nutrelle_cache", "pesos")   # redefinida a partir de --cache
# Bases externas de treino (gancho de 12.4). A lista de rótulos fica em listas/; as fotos são baixadas na hora de rodar.
EXTERNOS = {
    "flavoria": dict(lista="externo_flavoria.csv", arquivo="dataset_images_and_csv.tar", bytes=2576977920, md5="56270b20f0261777bd5e81f46cfd9097",
                     url="https://zenodo.org/api/records/5850856/files/dataset_images_and_csv.tar/content",
                     origem="FlavoriaFoodWeight1700 (Universidade de Turku), Zenodo 5850856, CC BY 4.0",
                     # 12.5: massa total medida (peso 1); massa por grupo e frações por fonte derivadas de um mapa de nomes
                     # (peso 0,5); proteína, densidade, kcal, gordura, carboidrato e proteína por grupo não são usados (peso 0)
                     pesos=dict(massa=1.0, grupos=0.5, fracoes=0.5)),
}
TETO_EXTERNO = 0.5            # soma dos pesos das fotos externas de treino = metade da soma dos pesos do Nutrition5k
FRACAO_EXT_EPOCA = 0.25       # ajuste fino: fotos externas sorteadas por época, em fração do nº de pratos do Nutrition5k (custo: +25%)
# receitas de ajuste fino (lr do topo, decaimento de lr por camada, épocas); fixadas antes de qualquer resultado
RECEITAS = {
    "r1": dict(lr=3e-5, lld=0.75, epocas=15),   # receita principal
    "r2": dict(lr=6e-5, lld=0.65, epocas=25),   # passo maior no topo, base quase parada, mais épocas
    "r3": dict(lr=2e-5, lld=0.90, epocas=8),    # a receita da CPU (lr 2e-5, 8 épocas) estendida ao backbone inteiro
}
PARTES_FT = ["fold0", "fold1", "fold2", "fold3", "fold4", "final"]


def _ridge(bb, est_min):
    return dict(nome=f"gpu ridge {bb} o8", tipo="ridge", bb=bb, bs=32, est_min=est_min)


def _ft(bb, receita, est_min, tipo="ft", sufixo="", prefixo="gpu", **kw):
    c = dict(nome=f"{prefixo} {tipo} {bb} {receita}{sufixo}", tipo=tipo, bb=bb, receita=receita, bs=32, acum=1, escala=(0.7, 1.3),
             ckpt_grad=False, wd=0.05, drop_path=0.1, ema=0.99, aquec=0.1, est_min=est_min, **RECEITAS[receita])
    c.update(kw)
    return c


_ALTA = dict(bs=16, acum=2, ckpt_grad=True)     # 768 tokens: lote menor + checkpoint de gradiente (16 GB)
# Ordem = prioridade. est_min = minutos de GPU MEDIDOS na execução 1 (10/11/2026, 2 × Tesla T4) para a configuração
# inteira (2.134 pratos; com o teste2 a extração das features cresce ~39%). Só informa; não entra no treino.
CONFIGS = [
    _ridge("siglip2_b16_224", 2),                                   # paridade com a CPU: "ridge siglip2_b16 o8" (RMSE 0,2551)
    _ridge("siglip2_b16_384x512", 5),                               # 8 orientações na resolução maior (não rodado na CPU)
    _ridge("dinov2_b14_336x448", 4),                                # backbone alternativo, congelado
    _ft("siglip2_b16_224", "r1", 26),                               # ajuste fino completo a 224
    _ft("siglip2_b16_384x512", "r1", 101, epocas=12, **_ALTA),      # ajuste fino completo na resolução maior
    _ft("dinov2_b14_336x448", "r1", 99, epocas=12, **_ALTA),        # backbone alternativo, ajuste fino completo
    _ft("siglip2_b16_224", "r2", 41),
    _ft("siglip2_b16_224", "r3", 15),
    _ridge("dinov2_l14_336x448", 11),
    _ridge("siglip2_l16_384x512", 13),
    _ridge("siglip2_so400m16_384x512", 18),
    _ft("siglip2_b16_384x512", "r2", 133, epocas=16, **_ALTA),
    _ft("siglip2_b16_224", "r1", 25, sufixo=" sem-escala", escala=None),   # ablação: só orientações, sem escala aleatória
    _ft("dinov2_l14_224x308", "r1", 122, epocas=10, lld=0.85, bs=16, acum=2, ckpt_grad=True),
    _ft("siglip2_b16_224", "r1", 59, tipo="destila-rgbd"),          # professor RGB-D -> aluno RGB (opcional, por último)
]
_MASSA5 = ["gpu ridge siglip2_b16_384x512 o8", "gpu ridge dinov2_b14_336x448 o8", "gpu ridge dinov2_l14_336x448 o8",
           "gpu ridge siglip2_l16_384x512 o8", "gpu ridge siglip2_so400m16_384x512 o8"]
_FT4 = ["gpu ft siglip2_b16_224 r1", "gpu ft siglip2_b16_384x512 r1", "gpu ft dinov2_b14_336x448 r1", "gpu ft dinov2_l14_224x308 r1"]
# Execução 2 (seção 11): só o que o pipeline congelado usa + o candidato a rodar no aparelho + a configuração de paridade.
RODADAS = {"1": [c["nome"] for c in CONFIGS],
           "2": ["gpu ridge siglip2_b16_224 o8"] + _MASSA5 + _FT4 + ["gpu ft siglip2_b16_224 r2"]}
# Pipelines congelados da seção 11.2, para os totais de servico.json (o custo de cada modelo aparece uma vez).
PIPELINES = {"massa_5_backbones": list(_MASSA5), "imagem_9_modelos": _MASSA5 + _FT4}
NOTA_PIPELINES = ("imagem_9_modelos: a densidade do pipeline congelado usa ainda três cabeças de frações treinadas na CPU "
                  "(soft-mlp) sobre features congeladas do SigLIP 2 B/16 a 384×512 (2 orientações) e, numa delas ('ambos'), "
                  "também a 224 px. O custo do backbone a 384×512 já está contado em 'gpu ridge siglip2_b16_384x512 o8' se as "
                  "features forem compartilhadas; o do backbone a 224 px é o de 'gpu ridge siglip2_b16_224 o8' (fora dos totais). "
                  "As cabeças em si são MLPs pequenas: custo desprezível, não medido aqui.")
# ---- Execução 3 (seção 12.3): um modelo só, treino_v2, previsões só no teste2. Tudo aqui foi escrito antes de rodar.
# est_min = minutos de GPU estimados com os tempos medidos na execução 2 (× 1,05 nos congelados, × 1,35 nos ajustes finos).
MLP3 = dict(oculto=512, drop=0.3, lr=1e-3, wd=1e-2, epocas=60, lote=256)     # cabeça de frações, como soft-mlp fonte-l2 +perda-dens peso-faixa
PESO_DESTILA = 0.5


def _ridge3(bb, est_min, prefixo="gpu3", paridade=True):
    return dict(nome=f"{prefixo} ridge {bb} o8", tipo="ridge", bb=bb, bs=32, est_min=est_min, v3=True, livre=True,
                paridade_r2=f"gpu ridge {bb} o8" if paridade else None)


def _fracoes3(apelido, bb, est_min, prefixo="gpu3", **kw):
    return dict(nome=f"{prefixo} fracoes {apelido}", tipo="fracoes", bb=bb, est_min=est_min, v3=True, livre=True, servico=False,
                depende=[f"{prefixo} ridge {bb} o8"], **dict(MLP3, **kw))


def _alunos3(prefixo, professor):
    return dict(prefixo=prefixo, v3=True, fracoes=True, peso_faixa=True, professor=professor, peso_destila=PESO_DESTILA, depende=[professor], val_cada=5)


_R3 = ["siglip2_so400m16_384x512", "siglip2_l16_384x512", "dinov2_l14_336x448", "siglip2_b16_384x512", "dinov2_b14_336x448"]
_F3 = [("so400m", "siglip2_so400m16_384x512"), ("l16", "siglip2_l16_384x512"), ("b16_384x512", "siglip2_b16_384x512")]
CONFIGS3 = (
    [_ridge3(bb, m) for bb, m in zip(_R3, (31, 20, 20, 8, 8))]                       # 1–5: congelados + ridge no treino_v2
    + [_fracoes3(ap, bb, 3) for ap, bb in _F3]                                        # 6–8: cabeça de frações por fonte
    + [dict(nome="gpu3 professor", tipo="professor", bb=None, est_min=1, v3=True, servico=False,
            massa=[f"gpu3 ridge {bb} o8" for bb in _R3], fracoes=[f"gpu3 fracoes {ap}" for ap, _ in _F3],
            dens_ridge="gpu3 ridge siglip2_so400m16_384x512 o8",
            depende=[f"gpu3 ridge {bb} o8" for bb in _R3] + [f"gpu3 fracoes {ap}" for ap, _ in _F3])]   # 9
    + [dict(nome="gpu3 unico so400m mlp destilado", tipo="unico", bb="siglip2_so400m16_384x512", est_min=2, v3=True,
            professor="gpu3 professor", peso_destila=PESO_DESTILA,
            depende=["gpu3 professor", "gpu3 ridge siglip2_so400m16_384x512 o8"], **MLP3)]                # 10 (U1)
    + [_ft("siglip2_b16_224", "r2", 64, sufixo=" destilado", **_alunos3("gpu3", "gpu3 professor")),       # 11 (U4)
       _ft("siglip2_b16_384x512", "r1", 151, sufixo=" destilado", epocas=12, **_ALTA, **_alunos3("gpu3", "gpu3 professor")),   # 12 (U3)
       _ft("siglip2_b16_224", "r2", 64, prefixo="gpu3", v3=True, val_cada=5)])                            # 13 (U0): r2 sem mudança
_U1, _U4, _U3, _U0 = (c["nome"] for c in CONFIGS3[-4:])
RODADAS["3"] = [c["nome"] for c in CONFIGS3]
PIPELINES3 = {"professor_5_backbones": [f"gpu3 ridge {bb} o8" for bb in _R3], "U1": [_U1], "U4": [_U4], "U3": [_U3], "U0": [_U0],
              "par_U1_U3": [_U1, _U3], "par_U1_U4": [_U1, _U4], "par_U3_U4": [_U3, _U4]}
ORDEM_SERVICO3 = [_U1, _U4, _U3, _U0]      # na fase de serviço, os candidatos a modelo único vêm antes dos congelados

CONFIGS_SMOKE3 = (
    [_ridge3("vit_tiny_224", 1, "smoke3"), _ridge3("vit_tiny_96x128", 1, "smoke3")]
    + [_fracoes3("tiny224", "vit_tiny_224", 1, "smoke3", epocas=3), _fracoes3("tiny96", "vit_tiny_96x128", 1, "smoke3", epocas=3)]
    + [dict(nome="smoke3 professor", tipo="professor", bb=None, est_min=1, v3=True, servico=False,
            massa=["smoke3 ridge vit_tiny_224 o8", "smoke3 ridge vit_tiny_96x128 o8"], fracoes=["smoke3 fracoes tiny224", "smoke3 fracoes tiny96"],
            dens_ridge="smoke3 ridge vit_tiny_224 o8",
            depende=["smoke3 ridge vit_tiny_224 o8", "smoke3 ridge vit_tiny_96x128 o8", "smoke3 fracoes tiny224", "smoke3 fracoes tiny96"])]
    + [dict(nome="smoke3 unico tiny224 mlp destilado", tipo="unico", bb="vit_tiny_224", est_min=1, v3=True, professor="smoke3 professor",
            peso_destila=PESO_DESTILA, depende=["smoke3 professor", "smoke3 ridge vit_tiny_224 o8"], **dict(MLP3, epocas=3))]
    + [_ft("vit_tiny_224", "r2", 1, sufixo=" destilado", epocas=1, bs=8, **_alunos3("smoke3", "smoke3 professor")),
       _ft("vit_tiny_96x128", "r1", 1, sufixo=" destilado", epocas=1, bs=4, acum=2, ckpt_grad=True, **_alunos3("smoke3", "smoke3 professor")),
       _ft("vit_tiny_224", "r2", 1, prefixo="smoke3", v3=True, val_cada=5, epocas=1, bs=8)])

CONFIGS_SMOKE = [
    dict(nome="smoke ridge vit_tiny_224 o8", tipo="ridge", bb="vit_tiny_224", bs=8, est_min=1),
    _ft("vit_tiny_224", "r1", 1, epocas=1, bs=8),
    _ft("vit_tiny_96x128", "r1", 1, epocas=1, bs=4, acum=2, ckpt_grad=True),
    _ft("vit_tiny_224", "r1", 1, tipo="destila-rgbd", epocas=1, bs=8),
]
for _c in CONFIGS_SMOKE:
    _c["nome"] = _c["nome"].replace("gpu ", "smoke ")


def slug(nome):
    return "".join(ch if ch.isalnum() else "_" for ch in nome)


def partes_de(cfg):
    return ["tudo"] if cfg["tipo"] in ("ridge", "fracoes", "professor", "unico", "externo", "decisao") else list(PARTES_FT)


def todas_as_configs():
    return CONFIGS + CONFIGS3 + CONFIGS_SMOKE + CONFIGS_SMOKE3


def configs_da_rodada(rodada, smoke=False):
    if rodada == "3":
        return list(CONFIGS_SMOKE3 if smoke else CONFIGS3)
    if smoke:
        return list(CONFIGS_SMOKE)
    return [c for c in CONFIGS if c["nome"] in RODADAS[rodada]]     # a ordem de CONFIGS (prioridade) é mantida


EST_EXT = {"ridge": {"siglip2_so400m16_384x512": 17, "siglip2_l16_384x512": 11, "dinov2_l14_336x448": 11, "siglip2_b16_384x512": 5, "dinov2_b14_336x448": 5},
           "fracoes": 5, "professor": 1, "unico": 3, "ft": 1 + FRACAO_EXT_EPOCA}     # min de GPU das versões com a base externa (ft: fator sobre a versão sem)


def com_externo(cfgs, nome):
    """Seção 12.5: como a base externa entra na rodada 3.
      - 'externo <nome>': baixar e preparar as fotos (sem GPU, numa linha à parte);
      - cada configuração barata (ridge, frações, professor, U1) ganha uma gêmea '+<nome>', treinada com o treino_v2
        mais o treino da base externa;
      - 'decisao <nome>': a regra de adoção, aplicada às previsões fora-do-fold dos dois professores;
      - cada ajuste fino destilado existe em duas versões e só roda a que a decisão indicar; o U0 (sem professor) é
        sempre só com o treino_v2.
    A ordem mantém as duplicatas antes dos ajustes finos, porque a decisão depende delas."""
    suf = f" +{nome}"
    ren = lambda n: n + suf
    prep = dict(nome=f"externo {nome}", tipo="externo", bb=None, est_min=0, v3=True, livre=True, servico=False)

    def gemea(c):
        g = dict(c, nome=ren(c["nome"]), externo=True, plano=c["nome"], paridade_r2=None)
        g["depende"] = [prep["nome"]] + [ren(d) for d in c.get("depende", [])] + ([c["nome"]] if c["tipo"] in ("ridge", "unico") else [])
        for k in ("massa", "fracoes"):
            if isinstance(c.get(k), list):
                g[k] = [ren(x) for x in c[k]]
        for k in ("dens_ridge", "professor"):
            if c.get(k):
                g[k] = ren(c[k])
        e = EST_EXT[c["tipo"]]
        g["est_min"] = max(1, round(c["est_min"] * e)) if c["tipo"] == "ft" else (e.get(c["bb"], 1) if isinstance(e, dict) else e)
        if c["tipo"] not in ("ft", "unico"):
            g["servico"] = False        # mesmo backbone e mesmo tipo de cabeça da versão sem a base: a latência é a mesma
        return g

    saida = [prep]
    for tipo in ("ridge", "fracoes", "professor", "unico"):
        saida += [c for c in cfgs if c["tipo"] == tipo] + [gemea(c) for c in cfgs if c["tipo"] == tipo]
    profs = [c["nome"] for c in cfgs if c["tipo"] == "professor"]
    dec = dict(nome=f"decisao {nome}", tipo="decisao", bb=None, est_min=0, v3=True, servico=False, sem=profs[0], com=ren(profs[0]),
               depende=[profs[0], ren(profs[0])]) if profs else None
    if dec:
        saida.append(dec)
    for c in cfgs:
        if c["tipo"] != "ft":
            continue
        if dec and c.get("professor"):
            saida.append(dict(c, depende=list(c.get("depende", [])) + [dec["nome"]], se_adotado=False))
            saida.append(dict(gemea(c), se_adotado=True, depende=gemea(c)["depende"] + [dec["nome"]]))
        else:
            saida.append(c)
    return saida


def regra_adocao(D, oof_sem, oof_com):
    """Regra de adoção da base externa (12.5), nas previsões fora-do-fold do professor no treino_v2 (com menos sem):
    adota se o critério de CV ponderado por faixa sobe 1,0 ponto ou mais; ou se não cai mais que 0,3 ponto e o critério
    nas duas faixas acima de 25 g (média das duas) sobe 2,0 pontos ou mais. Função pura; a proteína do professor é
    massa × densidade (coluna de proteína das suas previsões)."""
    v = ~(np.isnan(oof_sem[:, I_P]) | np.isnan(oof_com[:, I_P]))
    real, w, fx = D.proteina[v], D.w_cv[v], np.digitize(D.proteina[v], FAIXAS)
    ac = lambda oof: acerto(real, np.expm1(oof[v, I_P]).clip(0))
    crit = lambda oof: float(100 * (ac(oof) * w).sum() / w.sum())
    alto = lambda oof: float(np.mean([100 * ac(oof)[fx == f].mean() for f in (3, 4) if (fx == f).any()] or [0.0]))
    r = dict(n=int(v.sum()), criterio_sem=crit(oof_sem), criterio_com=crit(oof_com), acima_25g_sem=alto(oof_sem), acima_25g_com=alto(oof_com))
    r["dif_criterio"], r["dif_acima_25g"] = r["criterio_com"] - r["criterio_sem"], r["acima_25g_com"] - r["acima_25g_sem"]
    t = 1e-9        # folga de arredondamento: "1,0 ponto ou mais" inclui o 1,0 exato
    r["adotado"] = bool(r["dif_criterio"] >= 1.0 - t or (r["dif_criterio"] >= -0.3 - t and r["dif_acima_25g"] >= 2.0 - t))
    r["regra"] = "adota se dif_criterio >= 1,0; ou se dif_criterio >= -0,3 e dif_acima_25g >= 2,0 (12.5)"
    return r


def relato_externo(D, preds):
    """Relato secundário de 12.5 (não escolhe nada, não é o critério do produto, não é leitura do teste2): erro de massa
    na parte segurada da base externa (teste_ext), por foto. preds = {nome: arrays da pred__*.npz}. Para modelo treinado
    sem a base: previsão crua e previsão corrigida por uma constante (razão mediana real/prevista no treino externo)."""
    real, te = D.ext["massa"], ~D.ext["treina"]

    def med(lm, sel):
        sel = sel & np.isfinite(lm)
        p = np.exp(lm[sel]); r = real[sel]
        return dict(n=int(sel.sum()), mape=round(float(100 * np.mean(np.abs(p - r) / r)), 2), vies_mediano_pct=round(float(100 * (np.median(p / r) - 1)), 2)) if sel.any() else dict(n=0)

    def bloco(lm):
        return dict(total=med(lm, te), acima_600g=med(lm, te & (real > 600)))

    out, outra_lista = {}, []
    for nome, z in preds.items():
        if "ext" not in z:
            continue
        if "ext_ids" in z and [str(x) for x in z["ext_ids"]] != list(D.ext["ids"]):      # previsões de uma retomada com outra lista externa
            outra_lista.append(nome); continue
        r = {}
        com_base = " +" in nome
        sem = z["ext_sem"][:, I_M] if "ext_sem" in z else (None if com_base else z["ext"][:, I_M])
        if sem is not None:
            tr_ok = ~te & np.isfinite(sem)
            k = float(np.median(real[tr_ok] / np.exp(sem[tr_ok]))) if tr_ok.any() else 1.0
            r["sem_a_base_cru"], r["sem_a_base_corrigido"], r["constante"] = bloco(sem), bloco(sem + np.log(k)), round(k, 4)
        if com_base:
            r["com_a_base"] = bloco(z["ext"][:, I_M])
        out[nome] = r
    return dict(nota="relato fora do domínio (12.5): erro de massa por foto no teste_ext da base externa; não escolhe nada e não é leitura do teste2",
                metrica="por prato: só o quadro preferido e sem oclusão de cada prato do teste_ext (adendo de 12.5); MAPE e viés mediano da massa total, em %",
                n_teste_ext=int(te.sum()), n_teste_ext_acima_600g=int((te & (real > 600)).sum()), n_treino_ext=int((~te).sum()),
                regra_teste_ext=D.ext["regra_teste_ext"], modelos=out, fora_por_serem_de_outra_lista_externa=outra_lista)


# ------------------------------------------------------------------ máquina de estados de cada parte (config × fold)
class Estado(str, Enum):
    PENDENTE = "pendente"            # na fila (tentativa e divisor de lote guardados para a próxima execução)
    RODANDO = "rodando"              # um processo de treino está com ela
    FEITA = "feita"                  # parte__*.npz gravado (nesta sessão ou retomado)
    FALHOU = "falhou"                # erro repetido: a configuração inteira é abandonada
    SEM_ORCAMENTO = "sem_orcamento"  # não cabia no tempo que restava
    CANCELADA = "cancelada"          # outra parte da mesma configuração falhou ou ficou sem orçamento


TERMINAIS = {Estado.FEITA, Estado.FALHOU, Estado.SEM_ORCAMENTO, Estado.CANCELADA}
MAX_TENTATIVAS, MAX_DIV_LOTE = 2, 4


def parte_inicial():
    return dict(estado=Estado.PENDENTE.value, tentativa=0, div_lote=1, seg=None, erro=None, dispositivo=None, retomada=False)


def transicao(p, ev):
    """Função pura (estado da parte, evento) -> novo estado. Eventos: retomou | iniciou | concluiu | estourou_memoria |
    reduziu_lote | falhou | sem_orcamento | cancelou. Transição não prevista levanta ValueError (estado impossível não é gravado)."""
    e, t = Estado(p["estado"]), ev["tipo"]
    n = dict(p)
    if e is Estado.PENDENTE and t == "retomou":
        n.update(estado=Estado.FEITA.value, retomada=True, seg=ev.get("seg"))
    elif e is Estado.PENDENTE and t == "iniciou":
        n.update(estado=Estado.RODANDO.value, tentativa=p["tentativa"] + 1, dispositivo=ev["dispositivo"])
    elif e is Estado.PENDENTE and t == "sem_orcamento":
        n.update(estado=Estado.SEM_ORCAMENTO.value)
    elif e is Estado.PENDENTE and t == "reduziu_lote":       # outra parte da configuração estourou a memória
        n.update(div_lote=max(p["div_lote"], ev["div_lote"]))
    elif e is Estado.PENDENTE and t == "cancelou":
        n.update(estado=Estado.CANCELADA.value, erro=ev.get("motivo"))
    elif e is Estado.RODANDO and t == "concluiu":
        n.update(estado=Estado.FEITA.value, seg=ev["seg"], erro=None)
    elif e is Estado.RODANDO and t == "sem_orcamento":
        n.update(estado=Estado.SEM_ORCAMENTO.value)
    elif e is Estado.RODANDO and t == "estourou_memoria":
        if p["div_lote"] < MAX_DIV_LOTE:      # tenta de novo com metade do lote (e o dobro de acumulação)
            n.update(estado=Estado.PENDENTE.value, div_lote=p["div_lote"] * 2, tentativa=0, erro="memória")
        else:
            n.update(estado=Estado.FALHOU.value, erro="memória insuficiente mesmo com lote ÷ %d" % MAX_DIV_LOTE)
    elif e is Estado.RODANDO and t == "falhou":
        n.update(estado=(Estado.PENDENTE if p["tentativa"] < MAX_TENTATIVAS else Estado.FALHOU).value, erro=ev.get("erro"))
    else:
        raise ValueError(f"transição inválida: {e.value} + {t}")
    return n


def estado_config(partes):
    """Estado da configuração, derivado das partes: completa | falhou | sem_orcamento | cancelada | em_andamento | pendente."""
    es = {Estado(p["estado"]) for p in partes.values()}
    if es == {Estado.FEITA}:
        return "completa"
    if Estado.FALHOU in es:
        return "falhou"
    if Estado.SEM_ORCAMENTO in es:
        return "sem_orcamento"
    if es == {Estado.PENDENTE}:
        return "pendente"
    if Estado.CANCELADA in es and not es & {Estado.PENDENTE, Estado.RODANDO}:
        return "cancelada"
    return "em_andamento"


def simula_orcamento(cfgs, estado, livres, seg_parte, limite):
    """Quais configurações cabem no tempo que resta? Função pura. livres = instante em que cada processo fica livre;
    seg_parte(cfg) = segundos estimados por parte. Em ordem de prioridade, as partes pendentes de cada configuração vão
    para o processo que fica livre primeiro; a configuração cabe se todas terminam até `limite`. Quem não cabe não ocupa
    tempo, e as de prioridade menor ainda podem caber. Devolve {nome: bool}."""
    livres, cabe = list(livres), {}
    for c in cfgs:
        n = sum(p["estado"] == Estado.PENDENTE.value for p in estado[slug(c["nome"])].values())
        if not n or c["tipo"] in ("externo", "decisao"):      # o preparo da base externa e a decisão não ocupam GPU
            continue
        f, fim = list(livres), 0.0
        for _ in range(n):
            g = min(range(len(f)), key=f.__getitem__)
            f[g] += seg_parte(c); fim = max(fim, f[g])
        cabe[c["nome"]] = fim <= limite
        if cabe[c["nome"]]:
            livres = f
    return cabe


def seg_parte(c, cfgs, estado):
    """Segundos estimados por parte de uma configuração. Função pura. Com partes já medidas nesta sessão: a mediana
    delas. Sem nenhuma: a estimativa declarada (est_min), corrigida pela razão medido/declarado das partes já medidas
    em outras configurações do mesmo tipo (se os ajustes finos estão saindo 20% mais lentos, o próximo também sairá)."""
    medidas = lambda x: [q["seg"] for q in estado[slug(x["nome"])].values() if q["estado"] == Estado.FEITA.value and q["seg"] and not q.get("retomada")]
    decl = lambda x: x["est_min"] * 60.0 / len(partes_de(x))
    m = medidas(c)
    if m:
        return float(np.median(m))
    razoes = [v / decl(x) for x in cfgs if x["tipo"] == c["tipo"] for v in medidas(x)]
    return decl(c) * (float(np.median(razoes)) if razoes else 1.0)


def proxima_acao(cfgs, estado, cabe):
    """O que um processo livre faz agora? Função pura (estado das partes, o que cabe) -> uma ação:
      ("cancela", cfg, motivo)      uma dependência não ficou pronta nem vai ficar;
      ("sem_orcamento", cfg, parte) a configuração não cabe no tempo que resta;
      ("roda", cfg, parte)          primeira parte pendente, na ordem de prioridade, com as dependências completas;
      ("espera", None, None)        há partes pendentes, mas dependem de algo que ainda está rodando;
      ("fim", None, None)           nada pendente.
    Uma configuração só passa à frente de outra bloqueada se for marcada `livre` (as etapas sobre features congeladas);
    os alunos seguem a ordem estrita, para o orçamento cortar sempre a de menor prioridade."""
    est = lambda c: estado[slug(c["nome"])]
    viva = lambda c: any(p["estado"] in (Estado.PENDENTE.value, Estado.RODANDO.value) for p in est(c).values())
    por_nome = {c["nome"]: c for c in cfgs}
    bloqueio = espera = False
    for c in cfgs:
        pend = [p for p in partes_de(c) if est(c)[p]["estado"] == Estado.PENDENTE.value]
        if not pend or c["tipo"] in ("externo", "decisao"):       # preparo da base externa e decisão: fora das GPUs
            continue
        deps = [por_nome[d] for d in c.get("depende", []) if d in por_nome]
        mortas = [d["nome"] for d in deps if estado_config(est(d)) != "completa" and not viva(d)]
        if mortas:
            return "cancela", c, f"dependência não ficou pronta: {mortas[0]}"
        if any(estado_config(est(d)) != "completa" for d in deps):
            bloqueio = espera = True
            continue
        if bloqueio and not c.get("livre"):
            espera = True
            continue
        if not cabe.get(c["nome"], True):       # só se decide o corte quando chega a vez da configuração
            return "sem_orcamento", c, pend[0]
        return "roda", c, pend[0]
    return ("espera", None, None) if espera else ("fim", None, None)


def autoteste():
    p = parte_inicial()
    assert Estado(transicao(p, dict(tipo="retomou"))["estado"]) is Estado.FEITA
    r = transicao(p, dict(tipo="iniciou", dispositivo="cuda:0"))
    assert r["estado"] == "rodando" and r["tentativa"] == 1 and p["tentativa"] == 0           # pura: não altera a entrada
    assert transicao(r, dict(tipo="concluiu", seg=10.0)) == dict(r, estado="feita", seg=10.0)
    m = transicao(r, dict(tipo="estourou_memoria"))
    assert m["estado"] == "pendente" and m["div_lote"] == 2 and m["tentativa"] == 0
    m4 = dict(r, div_lote=MAX_DIV_LOTE)
    assert transicao(m4, dict(tipo="estourou_memoria"))["estado"] == "falhou"
    f1 = transicao(r, dict(tipo="falhou", erro="x"))
    assert f1["estado"] == "pendente" and f1["erro"] == "x"                                   # 1ª falha: tenta de novo
    f2 = transicao(transicao(f1, dict(tipo="iniciou", dispositivo="cpu")), dict(tipo="falhou", erro="y"))
    assert f2["estado"] == "falhou"                                                           # 2ª falha: desiste
    assert transicao(r, dict(tipo="sem_orcamento"))["estado"] == "sem_orcamento"
    assert transicao(p, dict(tipo="cancelou", motivo="z"))["estado"] == "cancelada"
    assert transicao(p, dict(tipo="reduziu_lote", div_lote=2)) == dict(p, div_lote=2)
    for terminal in TERMINAIS:                                                                # terminais não saem do lugar
        for t in ("retomou", "iniciou", "concluiu", "estourou_memoria", "reduziu_lote", "falhou", "sem_orcamento", "cancelou"):
            try:
                transicao(dict(p, estado=terminal.value), dict(tipo=t, dispositivo="x", seg=1, div_lote=2))
                raise AssertionError((terminal, t))
            except ValueError:
                pass
    for t in ("concluiu", "estourou_memoria", "falhou"):                                      # só quem está rodando termina
        try:
            transicao(p, dict(tipo=t, seg=1)); raise AssertionError(t)
        except ValueError:
            pass
    f = lambda *es: estado_config({i: dict(estado=e) for i, e in enumerate(es)})
    assert f("feita", "feita") == "completa" and f("feita", "falhou", "cancelada") == "falhou"
    assert f("feita", "sem_orcamento") == "sem_orcamento" and f("pendente", "pendente") == "pendente"
    assert f("feita", "pendente") == "em_andamento" and f("rodando", "pendente") == "em_andamento"
    assert f("cancelada", "cancelada") == "cancelada" and f("feita", "cancelada") == "cancelada" and f("cancelada", "pendente") == "em_andamento"
    assert json.loads(json.dumps(m)) == m                                                     # serializável
    assert len({slug(c["nome"]) for c in todas_as_configs()}) == len(todas_as_configs())
    assert all(c["bb"] in BACKBONES for c in todas_as_configs() if c["bb"])
    for lista in (CONFIGS3, CONFIGS_SMOKE3):      # dependências declaradas existem e vêm antes na fila
        for i, c in enumerate(lista):
            assert all(d in [x["nome"] for x in lista[:i]] for d in c.get("depende", [])), c["nome"]
    assert [c["nome"] for c in CONFIGS3] == [
        "gpu3 ridge siglip2_so400m16_384x512 o8", "gpu3 ridge siglip2_l16_384x512 o8", "gpu3 ridge dinov2_l14_336x448 o8",
        "gpu3 ridge siglip2_b16_384x512 o8", "gpu3 ridge dinov2_b14_336x448 o8", "gpu3 fracoes so400m", "gpu3 fracoes l16",
        "gpu3 fracoes b16_384x512", "gpu3 professor", "gpu3 unico so400m mlp destilado", "gpu3 ft siglip2_b16_224 r2 destilado",
        "gpu3 ft siglip2_b16_384x512 r1 destilado", "gpu3 ft siglip2_b16_224 r2"]
    u4, u3, u0 = CONFIGS3[-3:]
    assert (u4["lr"], u4["lld"], u4["epocas"], u3["lr"], u3["lld"], u3["epocas"], u3["bs"], u3["acum"]) == (6e-5, 0.65, 25, 3e-5, 0.75, 12, 16, 2)
    assert not u0.get("fracoes") and not u0.get("peso_faixa") and not u0.get("professor") and (u0["lr"], u0["epocas"]) == (6e-5, 25)
    nomes = {c["nome"] for c in CONFIGS + CONFIGS3}
    assert all(set(v) <= nomes and len(set(v)) == len(v) for v in list(RODADAS.values()) + list(PIPELINES.values()) + list(PIPELINES3.values()))
    assert len(RODADAS["2"]) == 11 and all(set(v) <= set(RODADAS["2"]) for v in PIPELINES.values())
    assert [c["nome"] for c in configs_da_rodada("2")][0] == "gpu ridge siglip2_b16_224 o8"      # a paridade roda primeiro
    # ---- escalonador (funções puras)
    cs = [dict(nome="r", tipo="ridge", livre=True), dict(nome="f", tipo="fracoes", livre=True, depende=["r"]),
          dict(nome="p", tipo="professor", depende=["r", "f"]), dict(nome="a", tipo="ft", depende=["p"]), dict(nome="z", tipo="ft")]
    novo = lambda: {slug(c["nome"]): {q: parte_inicial() for q in partes_de(c)} for c in cs}
    marca = lambda e, nome, est, partes=None: [e[nome][q].update(estado=est) for q in (partes or e[nome])]
    e = novo()
    assert proxima_acao(cs, e, {})[:2] == ("roda", cs[0]) and proxima_acao(cs, e, {})[2] == "tudo"
    marca(e, "r", "rodando")
    assert proxima_acao(cs, e, {})[0] == "espera"                    # 'z' não tem dependência, mas não fura a fila dos alunos
    marca(e, "r", "feita")
    assert proxima_acao(cs, e, {})[1]["nome"] == "f"
    marca(e, "f", "feita"); marca(e, "p", "feita")
    assert proxima_acao(cs, e, {})[1:] == (cs[3], "fold0")
    marca(e, "a", "rodando")
    assert proxima_acao(cs, e, {})[1:] == (cs[4], "fold0")           # todas as partes de 'a' já estão com alguém: 'z' pode ir
    e = novo(); marca(e, "r", "falhou")
    assert proxima_acao(cs, e, {})[:2] == ("cancela", cs[1])         # dependência morta: cancela quem depende dela
    marca(e, "f", "cancelada"); marca(e, "p", "cancelada"); marca(e, "a", "cancelada")
    assert proxima_acao(cs, e, {})[:2] == ("roda", cs[4])            # sem professor, o U0 ainda roda
    assert proxima_acao(cs, e, {"z": False})[0] == "sem_orcamento"
    e2 = novo(); marca(e2, "r", "rodando")
    assert proxima_acao(cs, e2, {"z": False})[0] == "espera"         # o corte de orçamento só se decide quando chega a vez
    cz = [dict(nome="x", tipo="ft", est_min=60), dict(nome="y", tipo="ft", est_min=120), dict(nome="w", tipo="ridge", est_min=5)]
    ez = {q["nome"]: {r: parte_inicial() for r in partes_de(q)} for q in cz}
    assert seg_parte(cz[1], cz, ez) == 1200.0 and seg_parte(cz[2], cz, ez) == 300.0
    ez["x"]["fold0"].update(estado="feita", seg=900.0)                  # x saiu 1,5 × mais lento que o declarado
    assert seg_parte(cz[0], cz, ez) == 900.0 and seg_parte(cz[1], cz, ez) == 1800.0 and seg_parte(cz[2], cz, ez) == 300.0
    marca(e, "z", "feita")
    assert proxima_acao(cs, e, {})[0] == "fim"
    e = novo(); marca(e, "r", "feita"); marca(e, "f", "feita"); marca(e, "p", "feita")
    seg = lambda c: 100.0
    assert simula_orcamento(cs, e, [0.0, 0.0], seg, 600) == {"a": True, "z": True}       # 12 partes de 100 s em 2 processos: 600 s
    assert simula_orcamento(cs, e, [0.0, 0.0], seg, 599) == {"a": True, "z": False}      # corta a de menor prioridade
    assert simula_orcamento(cs, e, [0.0, 0.0], lambda c: 250.0 if c["nome"] == "a" else 100.0, 700) == {"a": False, "z": True}   # a cara sai; a barata ainda cabe
    assert simula_orcamento(cs, e, [0.0, 400.0], seg, 600) == {"a": True, "z": False}    # um processo ocupado até 400 s
    teste_externo()
    try:
        import torch  # noqa: F401
    except ImportError:
        return print("autoteste: ok (sem torch: alvos aninhados não testados)")
    teste_aninhado()
    print("autoteste: ok")


def teste_externo():
    """Seção 12.5, sem torch: a fila com a base externa, a regra de adoção, o corte das versões excluídas e o relato."""
    cs = com_externo(CONFIGS3, "flavoria")
    nomes = [c["nome"] for c in cs]
    assert len(set(nomes)) == len(nomes) and len({slug(n) for n in nomes}) == len(nomes)
    for i, c in enumerate(cs):          # dependências existem e vêm antes
        assert all(d in nomes[:i] for d in c.get("depende", [])), c["nome"]
    gem = [c for c in cs if c.get("externo")]
    assert sorted(c["plano"] for c in gem if c["tipo"] != "ft") == sorted(c["nome"] for c in CONFIGS3[:10])     # as configurações 1 a 10, duas vezes
    assert [c["plano"] for c in gem if c["tipo"] == "ft"] == [CONFIGS3[10]["nome"], CONFIGS3[11]["nome"]]       # U4 e U3; o U0 não tem versão com a base
    i_dec = nomes.index("decisao flavoria")
    assert all(c["tipo"] != "ft" for c in cs[:i_dec]) and all(c["tipo"] == "ft" for c in cs[i_dec + 1:])       # a decisão sai antes de todo ajuste fino
    assert [c["nome"] for c in cs[i_dec + 1:]] == [_U4, _U4 + " +flavoria", _U3, _U3 + " +flavoria", _U0]
    assert all(("se_adotado" in c) == (c["nome"] != _U0) for c in cs[i_dec + 1:]) and "depende" not in cs[-1]
    pf = next(c for c in cs if c["nome"] == "gpu3 professor +flavoria")
    assert all(x.endswith(" +flavoria") for x in pf["massa"] + pf["fracoes"] + [pf["dens_ridge"]])
    assert next(c for c in cs if c["nome"] == _U3 + " +flavoria")["professor"] == pf["nome"]
    for c in gem:                       # mesma receita da versão sem a base (só mudam nome, dependências, estimativa e os dados)
        o = next(x for x in CONFIGS3 if x["nome"] == c["plano"])
        fora = {"nome", "externo", "plano", "paridade_r2", "depende", "massa", "fracoes", "dens_ridge", "professor", "est_min", "servico", "se_adotado"}
        assert {k: v for k, v in c.items() if k not in fora} == {k: v for k, v in o.items() if k not in fora}, c["nome"]
    # regra de adoção: função pura sobre as previsões fora-do-fold dos dois professores
    rng = np.random.default_rng(1)
    prot = np.r_[rng.uniform(1, 5, 200), rng.uniform(5, 15, 200), rng.uniform(15, 25, 200), rng.uniform(25, 40, 200), rng.uniform(40, 80, 200)]
    fx = np.digitize(prot, FAIXAS)
    D = SimpleNamespace(proteina=prot, w_cv=(1 / np.bincount(fx, minlength=5))[fx])

    def oof(certos):        # certos = nº de pratos certos em cada faixa (de 200); os errados ficam longe (2 × + 10 g)
        o = np.zeros((len(prot), 6))
        for f in range(5):
            i = np.where(fx == f)[0]
            o[i, I_P] = np.log1p(np.where(np.arange(len(i)) < certos[f], prot[i], 2 * prot[i] + 10))
        return o
    base = oof([100] * 5)
    r = regra_adocao(D, base, oof([110] * 5))
    assert r["adotado"] and abs(r["dif_criterio"] - 5.0) < 1e-9 and abs(r["criterio_sem"] - 50.0) < 1e-9
    assert regra_adocao(D, base, oof([102] * 5))["adotado"]                             # +1,0 ponto: adota
    assert not regra_adocao(D, base, oof([101, 101, 101, 102, 102]))["adotado"]        # +0,7 e faixas altas +1,0: não
    r = regra_adocao(D, base, oof([96, 96, 100, 104, 104]))
    assert r["adotado"] and abs(r["dif_criterio"]) < 1e-9 and abs(r["dif_acima_25g"] - 2.0) < 1e-9     # empate no critério e +2,0 nas faixas altas: adota
    assert not regra_adocao(D, base, oof([84, 90, 100, 110, 110]))["adotado"]          # faixas altas +5, mas o critério cai 0,6: não
    assert regra_adocao(D, base, oof([95, 95, 100, 104, 104]))["adotado"]              # critério cai 0,2 (dentro de 0,3) e faixas altas +2,0: adota
    assert not regra_adocao(D, base, oof([100, 100, 100, 104, 102]))["adotado"]        # +0,6 e faixas altas +1,5: não
    assert not regra_adocao(D, base, base)["adotado"]
    nanb = base.copy(); nanb[::7] = np.nan                                             # pratos fora do subconjunto (--smoke) não contam
    assert regra_adocao(D, nanb, nanb)["n"] == int((~np.isnan(nanb[:, 0])).sum())
    # a fila: os ajustes finos esperam a decisão; depois dela, só a versão indicada roda, e o U0 roda sempre
    for adotado in (False, True):
        e = {slug(c["nome"]): {q: parte_inicial() for q in partes_de(c)} for c in cs}
        for c in cs[:i_dec]:
            e[slug(c["nome"])]["tudo"]["estado"] = "feita"
        assert proxima_acao(cs, e, {})[0] == "espera"                                   # sem decisão, nenhum ajuste fino começa (nem o U0)
        e[slug("decisao flavoria")]["tudo"]["estado"] = "feita"
        for c in cs:
            if "se_adotado" in c and c["se_adotado"] != adotado:
                for q in e[slug(c["nome"])].values():
                    q["estado"] = "cancelada"
        ordem = []
        while True:
            acao, c, q = proxima_acao(cs, e, {})
            if acao != "roda":
                break
            e[slug(c["nome"])][q]["estado"] = "feita"; ordem.append(c["nome"])
        suf = " +flavoria" if adotado else ""
        assert acao == "fim" and list(dict.fromkeys(ordem)) == [_U4 + suf, _U3 + suf, _U0], ordem
        cabe = simula_orcamento(cs, {k: {q: parte_inicial() for q in v} if "ft" in k and all(x["estado"] == "feita" for x in v.values()) else v for k, v in e.items()},
                                [0.0, 0.0], lambda c: 60.0 * c["est_min"] / 6, 1e9)
        assert set(cabe) == {_U4 + suf, _U3 + suf, _U0}                                 # a versão excluída não ocupa orçamento
    # sem a base (download falhou): as gêmeas caem por dependência
    e = {slug(c["nome"]): {q: parte_inicial() for q in partes_de(c)} for c in cs}
    e[slug("externo flavoria")]["tudo"]["estado"] = "falhou"
    canceladas = []
    while True:
        acao, c, q = proxima_acao(cs, e, {})
        if acao == "cancela":
            for x in e[slug(c["nome"])].values():
                x["estado"] = "cancelada"
            canceladas.append(c["nome"])
        elif acao == "roda":
            e[slug(c["nome"])][q]["estado"] = "feita"
        else:
            break
    assert acao == "espera" and set(canceladas) == {c["nome"] for c in gem}        # todas as versões com a base; falta só a decisão (do orquestrador)
    assert all(e[slug(c["nome"])]["tudo"]["estado"] == "feita" for c in CONFIGS3[:10])
    # relato no teste externo: cru, corrigido por uma constante e com a base
    massa = np.array([400.0, 500, 700, 300, 800, 650]); tre = np.array([True, True, True, False, False, False])
    Dx = SimpleNamespace(ext=dict(massa=massa, treina=tre, regra_teste_ext="x"))
    P = lambda fator: np.c_[np.log(massa * fator), np.zeros((6, 5))]
    com = P(1.1); com[1] = np.nan
    Dx.ext["ids"] = list("abcdef")
    rel = relato_externo(Dx, {"a +b": dict(ext=com, ext_sem=P(0.5)), "c": dict(ext=P(2.0), ext_ids=np.array(list("abcdef"))), "d": dict(oof=P(1)),
                              "e": dict(ext=P(1.0)[:5], ext_ids=np.array(list("abcde")))})
    r = rel["modelos"]
    assert rel["fora_por_serem_de_outra_lista_externa"] == ["e"] and rel["n_teste_ext_acima_600g"] == 2
    assert set(r) == {"a +b", "c"} and r["a +b"]["constante"] == 2.0 and r["c"]["constante"] == 0.5
    assert r["a +b"]["sem_a_base_cru"]["total"] == dict(n=3, mape=50.0, vies_mediano_pct=-50.0) and r["a +b"]["sem_a_base_corrigido"]["total"]["mape"] == 0.0
    assert r["a +b"]["com_a_base"]["total"] == dict(n=3, mape=10.0, vies_mediano_pct=10.0) and r["a +b"]["com_a_base"]["acima_600g"]["n"] == 2
    assert set(r["c"]) == {"sem_a_base_cru", "sem_a_base_corrigido", "constante"} and r["c"]["sem_a_base_cru"]["acima_600g"]["mape"] == 100.0


def teste_aninhado():
    """Os alvos de treino do aluno que fica sem o fold k NUNCA dependem dos rótulos do fold k. Dados sintéticos, mesmas
    funções da produção (ridge_aninhado, fracoes_aninhado, combina_professor, alvos_do_aluno, cabeca_folds): troca-se
    todo rótulo do fold k e exige-se que (a) os alvos aninhados de k, nos pratos fora de k, não mudem; (b) as previsões
    do aluno para o fold k não mudem; e, para o teste ter força, (c) os alvos dos outros alunos mudem."""
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    n, o, Dm, K = 60, 2, 12, 4
    fold = np.arange(n) % 5
    X = rng.normal(size=(n, o, Dm)).astype(np.float32)
    cfg = dict(MLP3, oculto=8, epocas=6, lote=16)
    dev = torch.device("cpu")

    def rotulos(semente):
        r = np.random.default_rng(semente)
        massa = np.exp(4.5 + 0.5 * X[:, 0, 0] + 0.2 * r.normal(size=n))
        FF = np.abs(X[:, 0, 1:1 + K]) + 0.1 * r.random((n, K)); FF /= FF.sum(1, keepdims=True)
        prot = massa * (FF @ np.array([0.28, 0.2, 0.1, 0.02])) * np.exp(0.1 * r.normal(size=n))
        T = np.c_[np.log(massa), np.log1p(prot), np.log1p(4 * prot), np.log(prot / massa + EPS_D), r.normal(size=n), r.normal(size=n)]
        return massa, prot, FF, T

    # amostras externas de treino (12.5): só massa e frações têm rótulo; entram em todo treino, com o viés de domínio retirado
    ne = 20
    Xe = (rng.normal(size=(ne, o, Dm)) + 0.5).astype(np.float32)
    Te = np.full((ne, 6), np.nan); Te[:, I_M] = 5.6 + 0.5 * Xe[:, 0, 0] + 0.1 * rng.normal(size=ne)
    FFe = np.abs(Xe[:, 0, 1:1 + K]) + 0.05; FFe /= FFe.sum(1, keepdims=True)
    pa = np.zeros(6, np.float32); pa[I_M] = 1.0
    ed = dict(X=Xe, T=Te, FF=FFe, peso_alvo=pa, peso_ff=0.5, cols_massa=[I_M])
    todos_f = frozenset(range(5))

    def alvos(massa, prot, FF, T, ext=None):
        an_r = ridge_aninhado(X, T, fold)                                             # [5, n, alvos]
        P = ridge_cv_or(X, T, fold)[0].mean(1)
        if ext is not None:     # como no ridge '+base': só a massa é refeita com as amostras externas
            R = RidgeSubExt(X, T[:, [I_M]], fold, Xe, Te[:, [I_M]], lambda n_i: TETO_EXTERNO * n_i / ne)
            an_r[:, :, [I_M]] = ridge_aninhado_de(R)
            alfa = R.alfa_cv(todos_f)
            for k in range(5):
                P[fold == k, I_M] = R.prev_x(todos_f - {k}, X[fold == k], alfa).numpy()[:, 0]
        an_f = fracoes_aninhado(X, T, FF, massa, prot, fold, cfg, dev, ext=ext)       # [5, n]
        an = np.stack([np.stack(combina_professor([an_r[k][:, I_M]], [lg_dens(an_f[k]), an_r[k][:, I_D]]), 1) for k in range(5)])
        _, F, dens = cabeca_folds(X, T, FF, massa, prot, fold, cfg, dev, ext=ext)
        oof = np.stack(combina_professor([P[:, I_M]], [lg_dens(np.array([F[i].mean(0) @ dens[fold[i]] for i in range(n)])), P[:, I_D]]), 1)
        return oof, an

    base = rotulos(1); outro = rotulos(2)
    for ext in (None, ed):      # sem e com a base externa no treino do professor e do aluno
        oof0, an0 = alvos(*base, ext=ext)
        if ext is None:
            sem_ext = an0
        else:                   # a base externa mexe de fato nos alvos (o caminho está ligado)…
            assert np.nanmax(np.abs(an0 - sem_ext)) > 1e-3
        for k in range(5):
            assert np.isnan(an0[k][fold == k]).all() and not np.isnan(an0[k][fold != k]).any()
        for k in ((0, 3) if ext is None else (2,)):
            troca = [np.where((fold == k).reshape((-1,) + (1,) * (b.ndim - 1)), o2, b) for b, o2 in zip(base, outro)]    # só os rótulos do fold k mudam
            oof1, an1 = alvos(*troca, ext=ext)
            fora = fold != k
            assert np.abs(alvos_do_aluno(oof1, an1, k)[fora] - alvos_do_aluno(oof0, an0, k)[fora]).max() < 1e-5, "alvo do aluno depende do fold de validação"
            assert max(np.abs(an1[j][fold != j] - an0[j][fold != j])[~np.isnan(an0[j][fold != j])].max() for j in range(5) if j != k) > 1e-3   # os outros alunos mudam
            assert np.abs(oof1 - oof0).max() > 1e-3                                   # e os alvos comuns (modelo final) também
            # o aluno sem o fold k (cabeça destilada, U1): mesmas previsões para o fold k com ou sem os rótulos verdadeiros de k
            p0 = cabeca_folds(X, base[3], base[2], base[0], base[1], fold, cfg, dev, an0, ext=ext)[0][fold == k]
            p1 = cabeca_folds(X, troca[3], troca[2], troca[0], troca[1], fold, cfg, dev, an1, ext=ext)[0][fold == k]
            assert np.abs(p1 - p0).max() < 1e-4, "a previsão fora-do-fold do aluno depende dos rótulos do fold de validação"
    # …e com peso zero ela some: o ridge com amostras externas de peso 0 é o ridge comum (mesmas previsões aninhadas)
    R0 = RidgeSubExt(X, base[3][:, [I_M]], fold, Xe, Te[:, [I_M]], 0.0)
    assert np.nanmax(np.abs(ridge_aninhado_de(R0)[:, :, 0] - ridge_aninhado(X, base[3], fold)[:, :, I_M])) < 1e-5
    # NNLS da densidade por fonte e peso por faixa
    d = dens_por_fonte(base[2], base[0], base[0] * (base[2] @ np.array([0.28, 0.2, 0.1, 0.02])))
    assert np.abs(d - np.array([0.28, 0.2, 0.1, 0.02])).max() < 1e-6
    w = peso_faixa(np.array([1.0, 2, 3, 10, 30, 50, 60]))
    assert abs(w.mean() - 1) < 1e-9 and abs(w[:3].sum() - w[3]) < 1e-9 and abs(w[5:].sum() - w[4]) < 1e-9


# ------------------------------------------------------------------ dados (listas do repositório; sem gabarito de avaliação)
class Dados:
    """Treino zenital estrito (ids, folds por cadeia, alvos em log) + ids de avaliação (sem rótulo):
    aval = calib/teste/n5k_v1 (aval_ids.csv); conj = aval + os conjuntos de aval2_ids.csv (teste2), se o arquivo existir."""

    def __init__(self, pasta, smoke=0, rodada="2", externo=None, externo_imagens=None, externo_pesos=None):
        self.rodada, v3 = rodada, rodada == "3"       # rodada 3: treino_v2 (com frações por fonte) e só o teste2 para prever
        arq_tr, arq_av = os.path.join(pasta, "treino_v2.csv" if v3 else "treino.csv"), os.path.join(pasta, "aval_ids.csv")
        with open(arq_tr) as f:
            rows = list(csv.DictReader(f))
        self.grupos = [c[2:] for c in rows[0] if c.startswith("m_")]
        G = len(self.grupos)
        self.N = 6 + 2 * G
        self.I_MG, self.I_PG = slice(6, 6 + G), slice(6 + G, 6 + 2 * G)
        self.ids = [r["dish_id"] for r in rows]
        self.fold = np.array([int(r["fold"]) for r in rows])
        col = lambda c: np.array([float(r[c]) for r in rows])
        m, p = col("massa"), col("proteina")
        mg = np.array([[float(r[f"m_{g}"]) for g in self.grupos] for r in rows])
        pg = np.array([[float(r[f"p_{g}"]) for g in self.grupos] for r in rows])
        self.massa, self.proteina = m, p
        self.fontes = [c[3:] for c in rows[0] if c.startswith("ff_")]          # fração da massa por fonte de proteína (treino_v2)
        self.FF = np.array([[float(r[f"ff_{g}"]) for g in self.fontes] for r in rows]) if self.fontes else None
        # mesmos alvos de zen_comum.alvos: log m | log1p p | log1p kcal | log(p/m + eps) | log1p gordura | log1p carbo | grupos
        self.T = np.c_[np.log(m), np.log1p(p), np.log1p(col("kcal")), np.log(p / m + EPS_D), np.log1p(col("gordura")),
                       np.log1p(col("carbo")), np.log1p(mg), np.log1p(pg)]
        fx = np.digitize(p, FAIXAS)
        self.w_cv = (1 / np.bincount(fx, minlength=5))[fx]; self.w_cv /= self.w_cv.sum()   # 5 faixas com o mesmo peso
        if v3:      # os 450 da avaliação antiga não são previstos (parte deles agora é treino)
            self.aval, ev = {}, []
            assert len(set(self.ids)) == len(self.ids), "ids repetidos nas listas"
            self.hash = hashlib.sha256(open(arq_tr, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16]     # conteúdo: CRLF ou LF dá o mesmo
            arq_v1 = os.path.join(pasta, "treino.csv")        # lista e folds antigos: só para a réplica de paridade com a execução 2
            self.antigo = {r["dish_id"]: int(r["fold"]) for r in csv.DictReader(open(arq_v1))} if os.path.exists(arq_v1) else {}
        else:
            with open(arq_av) as f:
                av = list(csv.DictReader(f))
            self.aval = {s: [r["dish_id"] for r in av if r["split"] == s] for s in SPLITS}
            ev = [d for s in SPLITS for d in self.aval[s]]
            assert len(set(self.ids)) == len(self.ids) and len(set(ev)) == len(ev), "ids repetidos nas listas"
            assert not set(self.ids) & set(ev), "prato de avaliação na lista de treino"
            self.hash = hashlib.sha256(open(arq_tr, "rb").read() + open(arq_av, "rb").read()).hexdigest()[:16]
        # conjuntos novos, só para previsão (teste2): não mudam o hash das listas de treino/avaliação da execução 1
        self.conj, self.hash2 = dict(self.aval), None
        arq_av2 = os.path.join(pasta, "aval2_ids.csv")
        if os.path.exists(arq_av2):
            with open(arq_av2) as f:
                av2 = list(csv.DictReader(f))
            for s in dict.fromkeys(r["split"] for r in av2):
                assert s not in self.conj and s != "oof", f"conjunto repetido: {s}"
                self.conj[s] = [r["dish_id"] for r in av2 if r["split"] == s]
            ev2 = [r["dish_id"] for r in av2]
            assert len(set(ev2)) == len(ev2), "ids repetidos em aval2_ids.csv"
            assert not set(ev2) & set(self.ids), "prato do teste2 na lista de treino"
            assert not set(ev2) & set(ev), "prato do teste2 já está na avaliação antiga"
            # hash do CONTEÚDO (o arquivo do repositório tem fim de linha CRLF e o notebook o regrava com LF)
            self.hash2 = hashlib.sha256("\n".join(f"{r['prato_id']},{r['split']},{r['dish_id']}" for r in av2).encode()).hexdigest()[:16]
        self.extras = [s for s in self.conj if s not in SPLITS]
        # --smoke: poucos pratos por fold e por conjunto; o resto fica NaN nas saídas
        self.usa_tr = np.ones(len(self.ids), bool)
        self.usa_av = {s: np.ones(len(v), bool) for s, v in self.conj.items()}
        if smoke:
            self.usa_tr[:] = False
            for k in range(5):
                self.usa_tr[np.where(self.fold == k)[0][:smoke]] = True
            for s in self.conj:
                self.usa_av[s][:] = False; self.usa_av[s][:4] = True
        usados = lambda nomes: [d for s in nomes for d, u in zip(self.conj[s], self.usa_av[s]) if u]
        base = [d for d, u in zip(self.ids, self.usa_tr) if u] + usados([s for s in SPLITS if s in self.conj])
        self.n_base = len(base)                     # treino (+ 450 nas rodadas 1 e 2): mesma ordem e mesmos lotes de extração
        self.todos = base + usados(self.extras)
        # blocos de extração de features: treino (+ 450) | pratos só de previsão (teste2), cada um em lotes próprios
        self.segmentos = [(i, f) for i, f in ((0, self.n_base), (self.n_base, len(self.todos))) if f > i]
        self.chave = hashlib.sha256("\n".join(self.todos).encode()).hexdigest()[:10]
        # gancho de 12.4 (desligado por padrão): amostras externas, só de treino, com cache de fotos e de features à parte
        self.ext = le_externo(externo, externo_imagens, self, smoke, externo_pesos) if externo else None


def le_externo(arq, raiz, D, smoke=0, pesos=None):
    """Gancho de 12.4/12.5: lista externa no esquema de treino_v2.csv mais `dominio`, `grupo`, `vista` e a foto de cada
    amostra (coluna `arquivo` ou `imagem`: nome do arquivo, com ou sem o caminho dentro do .tar). Uma linha por FOTO; os
    quadros do mesmo prato têm o mesmo `grupo`. Alvo ausente = campo vazio: vira NaN e a perda daquele alvo
    é mascarada; alvo com peso 0 em `pesos` também é mascarado. As amostras só treinam (fold −1): nunca entram em
    validação. Parte segurada (`teste_ext`, 20% dos grupos; nunca treina nem entra em escolha): a coluna `split` da
    lista (`split_ext`), se existir; senão os grupos dos últimos dias (coluna `data`), até somar 20%; senão sha1(grupo)
    mod 5 == 0. Adendo de 12.5: fotos com `oclusao` = 1 ficam fora de tudo; na parte segurada fica só o quadro com
    `quadro_preferido` = 1 de cada prato (métrica por prato). Colunas de proteína fraca (`proteina_fraca_*`) não são lidas.
    raiz = pasta das fotos (None: as fotos vêm do arquivo da base, baixado em prepara_externo)."""
    import re
    with open(arq, newline="") as f:
        rows = list(csv.DictReader(f))
    c_id = next((c for c in ("id", "amostra_id", "prato_id", "dish_id") if c in rows[0]), None)
    c_img = next((c for c in ("arquivo", "imagem", "arquivo_imagem", "foto", "filename") if c in rows[0]), None)
    falta = [c for c in ("dominio", "grupo", "vista") if c not in rows[0]] + ([] if c_id else ["id"]) + ([] if c_img else ["arquivo"])
    assert not falta, f"lista externa sem as colunas {falta}"
    c_split = next((c for c in ("split_ext", "split", "conjunto") if c in rows[0]), None)
    seg_de = lambda r: bool(c_split) and "teste" in r[c_split].lower()
    if smoke:       # teste rápido: poucas amostras (treino e parte segurada), só as que têm foto na pasta dada
        rows = [r for r in rows if raiz is None or acha_foto(raiz, r[c_img])]
        usa = lambda r: r.get("oclusao") != "1" and (not seg_de(r) or r.get("quadro_preferido", "1") == "1")
        rows = ([r for r in rows if not seg_de(r) and usa(r)][:smoke * 2] + [r for r in rows if seg_de(r) and usa(r)][:smoke]
                + [r for r in rows if not usa(r)][:2])
    assert rows, "lista externa vazia"
    conteudo = "\n".join(",".join(str(r.get(c, "")) for c in rows[0]) for r in rows)      # o hash é o da lista inteira, antes dos filtros
    # adendo de 12.5: foto com oclusao = 1 não entra no treino nem na métrica; na parte segurada, a métrica é por prato
    # (só o quadro com quadro_preferido = 1). As linhas que não servem a nenhum dos dois saem aqui (nem features têm).
    n_lista = len(rows)
    n_ocl = sum(r.get("oclusao") == "1" for r in rows)
    rows = [r for r in rows if r.get("oclusao") != "1"]
    n_npref = sum(seg_de(r) and r.get("quadro_preferido", "1") != "1" for r in rows) if c_split else 0
    rows = [r for r in rows if not (seg_de(r) and r.get("quadro_preferido", "1") != "1")] if c_split else rows
    assert rows, "lista externa vazia depois dos filtros"
    num = lambda r, c: float(r[c]) if r.get(c) not in (None, "") else np.nan
    col = lambda c: np.array([num(r, c) for r in rows])
    m, p = col("massa"), col("proteina")
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.c_[np.log(m), np.log1p(p), np.log1p(col("kcal")), np.log(p / m + EPS_D), np.log1p(col("gordura")), np.log1p(col("carbo")),
                  np.log1p(np.array([[num(r, f"m_{g}") for g in D.grupos] for r in rows])),
                  np.log1p(np.array([[num(r, f"p_{g}") for g in D.grupos] for r in rows]))]
    T[~np.isfinite(T)] = np.nan
    FF = np.array([[num(r, f"ff_{g}") for g in D.fontes] for r in rows])
    FF[np.isnan(FF).any(1)] = np.nan
    # peso de cada alvo na perda (12.5); alvo com peso 0 não é usado mesmo que a lista traga um valor
    pa = np.ones(D.N, np.float32)
    if pesos is not None:
        pa[:] = 0.0; pa[I_M] = pesos.get("massa", 0.0); pa[D.I_MG] = pesos.get("grupos", 0.0)
    T[:, pa == 0] = np.nan
    peso_ff = float(1.0 if pesos is None else pesos.get("fracoes", 0.0))
    if peso_ff == 0:
        FF[:] = np.nan
    # higiene: nenhum prato das listas do Nutrition5k; e nenhum prato do Nutrition5k a menos de 120 s de um prato do
    # teste2 (aproximação da regra de cadeia: uma base derivada do Nutrition5k tem de passar antes pela regra completa)
    t2 = {d for s in D.extras for d in D.conj[s]}
    ts2 = np.sort([int(d.split("_")[1]) for d in t2 if re.fullmatch(r"dish_\d+", d)])
    conhecidos = t2 | set(D.ids) | {d for v in D.conj.values() for d in v}
    for r in rows:
        assert r[c_id] not in conhecidos, f"{r[c_id]}: amostra externa repetida numa lista do Nutrition5k"
        if re.fullmatch(r"dish_\d+", r[c_id]) and len(ts2):
            t = int(r[c_id].split("_")[1]); i = np.searchsorted(ts2, t)
            perto = min(abs(int(ts2[j]) - t) for j in (max(i - 1, 0), min(i, len(ts2) - 1)))
            assert perto >= 120, f"{r[c_id]}: a {perto} s de um prato do teste2"
    doms = sorted({r["dominio"] for r in rows})
    ids = [f"ext:{r['dominio']}:{r[c_id]}" for r in rows]
    assert len(set(ids)) == len(ids), "ids repetidos na lista externa"
    # parte segurada: 20% dos grupos, fixos
    grupo = [r["grupo"] for r in rows]
    c_data = next((c for c in ("data", "dia", "date") if c in rows[0]), None)
    if c_split:
        fora, regra = {g for g, r in zip(grupo, rows) if "teste" in r[c_split].lower()}, f"coluna {c_split}"
        assert not fora & {g for g, r in zip(grupo, rows) if "teste" not in r[c_split].lower()}, "grupo da lista externa dos dois lados (treino e teste_ext)"
    elif c_data:
        dia = {}
        for g, r in zip(grupo, rows):
            dia[g] = max(dia.get(g, ""), r[c_data][:10])
        fora, alvo = set(), 0.2 * len(dia)
        for d in sorted(set(dia.values()), reverse=True):       # dias inteiros, do último para trás, até 20% dos grupos
            if len(fora) >= alvo:
                break
            fora |= {g for g, x in dia.items() if x == d}
        regra = f"grupos dos últimos dias (coluna {c_data})"
    else:
        fora, regra = {g for g in set(grupo) if int(hashlib.sha1(g.encode()).hexdigest(), 16) % 5 == 0}, "sha1(grupo) mod 5 == 0"
    treina = np.array([g not in fora for g in grupo])
    assert treina.any(), "lista externa sem amostras de treino"
    assert np.isfinite(m).all() and (m > 0).all(), "lista externa com massa total ausente ou não positiva"
    assert not (set(np.array(grupo)[treina]) & set(np.array(grupo)[~treina])), "grupo da lista externa dos dois lados (treino e teste_ext)"
    nomes_img = [os.path.basename(r[c_img]) for r in rows]
    assert len(set(nomes_img)) == len(nomes_img), "lista externa com fotos de mesmo nome"
    if not c_split and "quadro_preferido" in rows[0]:      # divisão feita aqui: a métrica continua por prato
        treina_ou_pref = np.array([t or r["quadro_preferido"] == "1" for t, r in zip(treina, rows)])
        assert treina_ou_pref.all(), "lista externa sem coluna de divisão e com mais de um quadro por prato na parte segurada"
    return dict(ids=ids, imagens=[r[c_img] for r in rows], raiz=raiz, T=T, FF=FF, massa=m, dominios=doms, dom=np.array([doms.index(r["dominio"]) for r in rows]),
                grupo=grupo, vista=[r["vista"] for r in rows], treina=treina, regra_teste_ext=regra, peso_alvo=pa, peso_ff=peso_ff,
                hash=hashlib.sha256(conteudo.encode()).hexdigest()[:16],
                resumo=dict(n_lista=n_lista, fora_por_oclusao=n_ocl, fora_por_nao_ser_o_quadro_preferido_do_teste_ext=n_npref,
                            n=len(rows), grupos=len(set(grupo)), treino=int(treina.sum()), grupos_treino=len(set(np.array(grupo)[treina])),
                            teste_ext=int((~treina).sum()), grupos_teste_ext=len(set(np.array(grupo)[~treina])),
                            com_massa=int(np.isfinite(T[:, I_M]).sum()), com_fracoes=int(np.isfinite(FF).all(1).sum()), massa_mediana_g=float(np.nanmedian(m))))


def acha_foto(raiz, nome):
    """Caminho de uma foto externa numa pasta: com o caminho da lista (ex.: images/train/x.png) ou só com o nome do arquivo."""
    for c in (os.path.join(raiz, nome), os.path.join(raiz, os.path.basename(nome))):
        if os.path.exists(c):
            return c
    return None


def vies_dominio(Te, T, cols):
    """Viés de domínio (12.5): diferença entre a média do alvo nas amostras externas de treino e a média no Nutrition5k
    do treino do fold, só nos alvos de massa (cols). Subtrai-se dos alvos externos no treino; na previsão de fotos do
    Nutrition5k não se soma nada, e na de fotos externas soma-se de volta."""
    d = np.zeros(T.shape[1])
    d[cols] = np.nanmean(Te[:, cols], 0) - T[:, cols].mean(0)
    return np.nan_to_num(d)


def proteinas(T):
    massa = np.exp(T[:, I_M]); dens = np.clip(np.exp(T[:, I_D]) - EPS_D, 0, None)
    direto = np.expm1(T[:, I_P]).clip(0)
    return dict(direto=direto, mxd=massa * dens, geo=np.sqrt(direto * massa * dens), massa=massa)


def acerto(real, est):
    err = np.abs(est - real)
    return (err <= 5) | (err <= 0.20 * np.maximum(real, 1e-9))


def resumo_cv(D, oof):
    """Métricas de CV no TREINO (as mesmas de zen_avalia): RMSE de log-massa e critério ponderado por faixa."""
    v = ~np.isnan(oof[:, 0])
    if not v.any():
        return dict(n=0)
    P = proteinas(oof[v]); w = D.w_cv[v]
    cr = lambda est: float(100 * (acerto(D.proteina[v], est) * w).sum() / w.sum())
    return dict(n=int(v.sum()), rmse_logm=float(np.sqrt(np.mean((oof[v, I_M] - D.T[v, I_M]) ** 2))),
                mape_massa=float(100 * np.mean(np.abs(P["massa"] - D.massa[v]) / D.massa[v])),
                crit_direto=cr(P["direto"]), crit_mxd=cr(P["mxd"]), crit_geo=cr(P["geo"]))


# ------------------------------------------------------------------ ridge (cópia de zen_comum: mesmo cálculo da CPU)
class Ridge:
    ALFAS = np.logspace(-1, 5, 19)

    def _decomp(self, X, T):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        mt = T.mean(0)
        s2, V = np.linalg.eigh((Z.T @ Z).astype(np.float64))
        return mu, sd, mt, s2.clip(0), V, V.T @ (Z.T @ (T - mt))

    def caminho(self, X, T, Xv):
        mu, sd, mt, s2, V, VtZtT = self._decomp(X, T)
        ZvV = ((Xv - mu) / sd) @ V
        return np.stack([ZvV @ (VtZtT / (s2 + a)[:, None]) + mt for a in self.ALFAS])

    def fit(self, X, T, alfa):
        mu, sd, mt, s2, V, VtZtT = self._decomp(X, T)
        self.mu, self.sd, self.mt = mu, sd, mt
        self.W = np.stack([V @ (VtZtT[:, j] / (s2 + a)) for j, a in enumerate(alfa)], 1)
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.W + self.mt


def ridge_cv(X, T, fold):
    """X[n, orientações, D]: cada orientação é uma amostra de treino; previsão = média nas orientações.
    Devolve (previsões fora-do-fold, alfa por alvo, MSE de CV por alvo)."""
    n, o, Dm = X.shape
    r = Ridge()
    P = np.zeros((len(r.ALFAS), n, T.shape[1]))
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        cam = r.caminho(X[tr].reshape(-1, Dm), np.repeat(T[tr], o, 0), X[va].reshape(-1, Dm))
        P[:, va] = cam.reshape(len(r.ALFAS), va.sum(), o, -1).mean(2)
    mse = ((P - T) ** 2).mean(1)
    best = mse.argmin(0)
    return np.stack([P[best[j], :, j] for j in range(T.shape[1])], 1), r.ALFAS[best], mse.min(0)


def ridge_final(X, T, alfa):
    n, o, Dm = X.shape
    m = Ridge().fit(X.reshape(-1, Dm), np.repeat(T, o, 0), alfa)
    return (lambda Xn: m.predict(Xn.reshape(-1, Dm)).reshape(Xn.shape[0], Xn.shape[1], -1).mean(1)), m


# ------------------------------------------------------------------ rodada 3: ridge por orientação e aninhado, frações, professor
ORIENT = {"_o1": 1, "_o4": 4, "": 8}      # sufixo das variantes -> nº de orientações na previsão (as 4 primeiras preservam o quadro)


def media_or(P, k):
    """Média nas k primeiras orientações de P [n, 8, ...]."""
    return P[:, :k].mean(1)


def ridge_cv_or(X, T, fold):
    """Como ridge_cv (mesma escolha de alfa, pela previsão média das orientações), mas devolve a previsão fora-do-fold
    de cada orientação: [n, o, alvos]. A média nas 8 é a de ridge_cv."""
    n, o, Dm = X.shape
    r = Ridge()
    P = np.zeros((len(r.ALFAS), n, o, T.shape[1]))
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        cam = r.caminho(X[tr].reshape(-1, Dm), np.repeat(T[tr], o, 0), X[va].reshape(-1, Dm))
        P[:, va] = cam.reshape(len(r.ALFAS), va.sum(), o, -1)
    mse = ((P.mean(2) - T) ** 2).mean(1)
    best = mse.argmin(0)
    return np.stack([P[best[j], :, :, j] for j in range(T.shape[1])], 2), r.ALFAS[best], mse.min(0)


class RidgeSub:
    """A conta de Ridge para qualquer subconjunto de folds de treino, com a decomposição em cache (torch, float64 na
    decomposição; GPU se houver). Serve aos alvos aninhados do professor: 20 subconjuntos por backbone."""

    def __init__(self, X, T, fold, dev=None):
        import torch
        self.t, self.dev = torch, dev or torch.device("cpu")
        self.X = torch.from_numpy(np.ascontiguousarray(X, np.float32)).to(self.dev)
        self.T = torch.from_numpy(np.ascontiguousarray(T, np.float64)).to(self.dev)
        self.fold, self.cache = np.asarray(fold), {}
        self.alfas = torch.from_numpy(Ridge.ALFAS).to(self.dev)

    def _m(self, m):
        return self.t.from_numpy(np.where(m)[0]).to(self.dev)

    def dec(self, folds):
        if folds not in self.cache:
            t, i = self.t, self._m(np.isin(self.fold, list(folds)))
            o, Dm = self.X.shape[1:]
            Xt = self.X[i].reshape(-1, Dm); Tt = self.T[i].repeat_interleave(o, 0)
            mu, sd = Xt.mean(0), Xt.std(0, unbiased=False) + 1e-6
            Z = (Xt - mu) / sd
            mt = Tt.mean(0)
            s2, V = t.linalg.eigh((Z.T @ Z).double())
            self.cache[folds] = (mu, sd, mt, s2.clamp(min=0), V, V.T @ (Z.double().T @ (Tt - mt)))
        return self.cache[folds]

    def prev(self, folds, m, alfa=None):
        """Previsão (média nas orientações) nos pratos da máscara m, por um ridge treinado nos folds dados:
        para todos os alfas [A, n, alvos] ou, com um alfa por alvo, [n, alvos]."""
        return self.prev_x(folds, self.X[self._m(m)], alfa)

    def prev_x(self, folds, Xv, alfa=None, por_or=False):
        """Como prev, para features quaisquer Xv [n, o, D] (tensor ou numpy). por_or: uma previsão por orientação [n, o, alvos]."""
        mu, sd, mt, s2, V, VtZtT = self.dec(folds)[:6]
        if not self.t.is_tensor(Xv):
            Xv = self.t.from_numpy(np.ascontiguousarray(Xv, np.float32)).to(self.dev)
        nv, o, Dm = Xv.shape
        Zv = ((Xv.reshape(-1, Dm) - mu) / sd).double()
        if alfa is None:
            ZvV = Zv @ V
            return self.t.stack([ZvV @ (VtZtT / (s2 + a)[:, None]) + mt for a in self.alfas]).reshape(len(self.alfas), nv, o, -1).mean(2)
        P = (Zv @ (V @ (VtZtT / (s2[:, None] + alfa[None, :]))) + mt).reshape(nv, o, -1)
        return P if por_or else P.mean(1)

    def afim(self, folds, alfa):
        """O ridge como mapa afim no espaço das features cruas: s = f @ A + b (para gravar a cabeça de servir)."""
        mu, sd, mt, s2, V, VtZtT = self.dec(folds)[:6]
        A = (V @ (VtZtT / (s2[:, None] + alfa[None, :]))) / sd.double()[:, None]
        return A.cpu().numpy(), (mt - mu.double() @ A).cpu().numpy()

    def alfa_cv(self, folds):
        """Alfa por alvo pela CV nos folds dados (erro quadrático das previsões fora-do-fold, só nos pratos do Nutrition5k)."""
        erro = 0
        for f in sorted(folds):
            erro = erro + ((self.prev(folds - {f}, self.fold == f) - self.T[self._m(self.fold == f)]) ** 2).sum(1)
        return self.alfas[erro.argmin(0)]


class RidgeSubExt(RidgeSub):
    """RidgeSub com amostras externas de treino (12.4/12.5): entram em todo subconjunto de treino (fold −1), com peso
    `peso` por amostra (as do Nutrition5k têm peso 1) e com os alvos deslocados pelo viés de domínio do subconjunto.
    A padronização e o ridge são ponderados (a conta de ridge_alvos de dens_experimentos.py). T e Te trazem só os alvos
    de massa deste grupo."""

    def __init__(self, X, T, fold, Xe, Te, peso, dev=None):
        """peso: função (nº de pratos do Nutrition5k no treino) -> peso de cada amostra externa."""
        super().__init__(X, T, fold, dev)
        self.Xe = self.t.from_numpy(np.ascontiguousarray(Xe, np.float32)).to(self.dev)
        self.Te = self.t.from_numpy(np.ascontiguousarray(Te, np.float64)).to(self.dev)
        self.peso = peso if callable(peso) else (lambda n: float(peso))

    def dec(self, folds):
        if folds not in self.cache:
            t, i = self.t, self._m(np.isin(self.fold, list(folds)))
            o, Dm = self.X.shape[1:]
            delta = self.Te.mean(0) - self.T[i].mean(0)                       # viés de domínio no treino deste subconjunto
            Xt = t.cat([self.X[i].reshape(-1, Dm), self.Xe.reshape(-1, Dm)])
            Tt = t.cat([self.T[i].repeat_interleave(o, 0), (self.Te - delta).repeat_interleave(o, 0)])
            w = t.cat([t.ones(len(i) * o, device=self.dev), t.full((len(self.Xe) * o,), float(self.peso(len(i))), device=self.dev)])
            mu = (Xt * w[:, None]).sum(0) / w.sum()
            sd = (((Xt - mu) ** 2 * w[:, None]).sum(0) / w.sum()).sqrt() + 1e-6
            mt = (Tt * w.double()[:, None]).sum(0) / w.double().sum()
            Z = (Xt - mu) / sd * w.sqrt()[:, None]
            s2, V = t.linalg.eigh((Z.T @ Z).double())
            self.cache[folds] = (mu, sd, mt, s2.clamp(min=0), V, V.T @ (Z.double().T @ ((Tt - mt) * w.double().sqrt()[:, None])), delta)
        return self.cache[folds]

    def delta(self, folds):
        return self.dec(folds)[6].cpu().numpy()


def pares_aninhados(fold):
    """Os treinos aninhados: para cada par de folds {j, k}, o conjunto S dos outros folds. O modelo treinado em S dá o
    alvo dos pratos do fold j para o aluno que fica sem o fold k, e vice-versa. Nenhum rótulo de j ou k entra em S."""
    fs = sorted(set(int(f) for f in fold))
    return [(j, k, frozenset(fs) - {j, k}) for a, j in enumerate(fs) for k in fs[a + 1:]]


def ridge_aninhado(X, T, fold, dev=None):
    """[5, n, alvos]: na posição [k, i], a previsão para o prato i (fold j != k) de um ridge treinado SEM os folds j e k
    (alfa por alvo escolhido por CV interna só nos folds de treino). Linhas do próprio fold k: NaN."""
    return ridge_aninhado_de(RidgeSub(X, T, fold, dev))


def ridge_aninhado_de(R):
    """ridge_aninhado para um RidgeSub já montado (com ou sem amostras externas de treino)."""
    fold = R.fold
    out = np.full((int(fold.max()) + 1, len(fold), R.T.shape[1]), np.nan, np.float32)
    for j, k, S in pares_aninhados(fold):
        alfa = R.alfa_cv(S)
        out[k, fold == j] = R.prev(S, fold == j, alfa).cpu().numpy()
        out[j, fold == k] = R.prev(S, fold == k, alfa).cpu().numpy()
    return out


def peso_faixa(prot):
    """Peso por amostra: as cinco faixas de proteína com o mesmo peso, contadas só nos pratos dados (os de treino);
    média 1."""
    fx = np.digitize(prot, FAIXAS)
    w = (1 / np.maximum(np.bincount(fx, minlength=5), 1))[fx]
    return w / w.mean()


def dens_por_fonte(FF, massa, prot, it=2000):
    """Densidade média de proteína de cada fonte (g/g) nos pratos dados: mínimos quadrados não negativos de
    proteína ≈ Σ fração × massa × densidade da fonte (descida por coordenada, exata para este problema convexo)."""
    A = FF * massa[:, None]
    G, c = A.T @ A, A.T @ prot
    d = np.zeros(A.shape[1])
    for _ in range(it):
        ant = d.copy()
        for j in range(len(d)):
            if G[j, j] > 1e-12:
                d[j] = max(0.0, (c[j] - G[j] @ d + G[j, j] * d[j]) / G[j, j])
        if np.abs(d - ant).max() < 1e-12:
            break
    return d


def lg_dens(d):
    """Log da densidade com piso: a mesma conta da coluna de densidade das 22 saídas."""
    return np.log(np.maximum(d, 0) + EPS_D)


def treina_cabeca(X, T, FF, w, dc, cfg, dev, alvos_prof=None, semente=0, ext=None):
    """Cabeça de frações sobre features congeladas, como treina_soft('fonte-l2 +perda-dens peso-faixa') de
    dens_experimentos.py: MLP de uma camada oculta sobre features padronizadas, cada orientação como amostra; saídas =
    22 em log (padronizadas) + logits das fontes. Perdas, todas com o peso por amostra w: Huber nas 22; 10 × erro
    quadrático entre o softmax e as frações; Huber em log da densidade composta Σ fração × dc. Com alvos_prof [n, 3]
    (log-massa, log-densidade com piso, log1p-proteína do professor), soma PESO_DESTILA × Huber nas saídas 0, 3 e 1.
    ext (12.5): amostras externas só de treino: dict(X [m, o, D], T [m, 22] com NaN nos alvos sem rótulo e já sem o viés
    de domínio, FF [m, K] ou NaN, peso_alvo [22], peso_ff, c = peso por amostra). Entram com a perda de cada alvo
    mascarada e pesada; densidade composta e destilação não se aplicam a elas. Padronização: só do Nutrition5k.
    Devolve (pesos para servir, função X [n, o, D] -> (22 saídas em log [n, o, 22], frações [n, o, K]))."""
    import torch
    torch.manual_seed(semente)
    n, o, Dm = X.shape
    K, nt = FF.shape[1], T.shape[1]
    mu, sd = X.reshape(-1, Dm).mean(0), X.reshape(-1, Dm).std(0) + 1e-6
    tm, ts = T.mean(0), T.std(0) + 1e-6
    q = [I_M, I_D, I_P]
    rep = lambda A: np.repeat(A, o, 0).astype(np.float32)
    Xn, Yn_, Fn, Wn = ((X - mu) / sd).reshape(-1, Dm).astype(np.float32), rep((T - tm) / ts), rep(FF), rep(w / w.mean())
    Pnn = None if alvos_prof is None else rep((alvos_prof - tm[q]) / ts[q])
    if ext is not None:      # linhas externas depois das do Nutrition5k; máscaras = presença do alvo × peso do alvo
        me = len(ext["X"])
        Me = rep(np.isfinite(ext["T"]) * ext["peso_alvo"] * (ts > 1e-3))     # alvo sem variação no Nutrition5k (só no --smoke): fora
        Mfe = rep(np.isfinite(ext["FF"]).all(1) * ext["peso_ff"])
        Xn = np.r_[Xn, ((ext["X"] - mu) / sd).reshape(-1, Dm).astype(np.float32)]
        Yn_ = np.r_[Yn_, rep(np.nan_to_num((ext["T"] - tm) / ts))]
        Fn = np.r_[Fn, rep(np.nan_to_num(ext["FF"]))]
        Wn = np.r_[Wn, np.full(me * o, ext["c"], np.float32)]
        Pnn = None if Pnn is None else np.r_[Pnn, np.zeros((me * o, 3), np.float32)]
        Mt = torch.from_numpy(np.r_[np.ones((n * o, nt), np.float32), Me]).to(dev)           # por alvo
        Mf = torch.from_numpy(np.r_[np.ones(n * o, np.float32), Mfe]).to(dev)                # frações
        Mn = torch.from_numpy(np.r_[np.ones(n * o, np.float32), np.zeros(me * o, np.float32)]).to(dev)   # só Nutrition5k: densidade composta e destilação
    Xt, Y, Fr, W = (torch.from_numpy(v).to(dev) for v in (Xn, Yn_, Fn, Wn))
    dct = torch.from_numpy(dc.astype(np.float32)).to(dev)
    Pn = None if Pnn is None else torch.from_numpy(Pnn).to(dev)
    net = torch.nn.Sequential(torch.nn.Dropout(cfg["drop"]), torch.nn.Linear(Dm, cfg["oculto"]), torch.nn.GELU(), torch.nn.Dropout(cfg["drop"]),
                              torch.nn.Linear(cfg["oculto"], nt + K)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epocas"] * (len(Xt) // cfg["lote"] + 1), pct_start=0.1)
    g = torch.Generator().manual_seed(semente)
    sl1 = torch.nn.functional.smooth_l1_loss
    for _ in range(cfg["epocas"]):
        net.train()
        for j in torch.randperm(len(Xt), generator=g).split(cfg["lote"]):
            j = j.to(dev)
            out, wj = net(Xt[j]), W[j]
            fr = torch.softmax(out[:, nt:], 1)
            alvo_d = Y[j][:, I_D] * float(ts[I_D]) + float(tm[I_D])
            p_d = sl1((torch.log(fr @ dct + EPS_D) - alvo_d) / float(ts[I_D]), torch.zeros_like(alvo_d), reduction="none")
            if ext is None:
                perda = (sl1(out[:, :nt], Y[j], reduction="none").mean(1) * wj).mean()
                perda = perda + 10 * (((fr - Fr[j]) ** 2).sum(1) * wj).mean()
                perda = perda + (p_d * wj).mean()
                if Pn is not None:
                    perda = perda + cfg.get("peso_destila", PESO_DESTILA) * (sl1(out[:, q], Pn[j], reduction="none").mean(1) * wj).mean()
            else:
                perda = ((sl1(out[:, :nt], Y[j], reduction="none") * Mt[j]).mean(1) * wj).mean()
                perda = perda + 10 * (((fr - Fr[j]) ** 2).sum(1) * Mf[j] * wj).mean()
                perda = perda + (p_d * Mn[j] * wj).mean()
                if Pn is not None:
                    perda = perda + cfg.get("peso_destila", PESO_DESTILA) * (sl1(out[:, q], Pn[j], reduction="none").mean(1) * Mn[j] * wj).mean()
            opt.zero_grad(); perda.backward(); opt.step(); sched.step()
    net.eval()
    tm_t, ts_t = torch.from_numpy(tm.astype(np.float32)).to(dev), torch.from_numpy(ts.astype(np.float32)).to(dev)

    def prev(Xv):
        with torch.inference_mode():
            z = net(torch.from_numpy(((Xv - mu) / sd).reshape(-1, Dm).astype(np.float32)).to(dev))
            return ((z[:, :nt] * ts_t + tm_t).reshape(Xv.shape[0], Xv.shape[1], nt).cpu().numpy(),
                    torch.softmax(z[:, nt:], 1).reshape(Xv.shape[0], Xv.shape[1], K).cpu().numpy())
    est = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    pesos = dict(mu=mu.astype(np.float32), sd=sd.astype(np.float32), tm=tm.astype(np.float32), ts=ts.astype(np.float32),
                 W1=est["1.weight"], b1=est["1.bias"], W2=est["4.weight"], b2=est["4.bias"], dens_fonte=dc.astype(np.float32))
    return pesos, prev


def _uma_cabeca(X, T, FF, massa, prot, m, cfg, dev, alvos=None, ext=None):
    """Treina a cabeça de frações SÓ com os pratos da máscara m: rótulos, peso por faixa, densidade por fonte e alvos
    do professor (já recortados em m) saem todos de m. ext (12.5): amostras externas de treino dict(X, T, FF, peso_alvo,
    peso_ff, cols_massa); entram com a soma de pesos igual a TETO_EXTERNO × a do Nutrition5k e com os alvos de massa
    sem o viés de domínio medido em m. Devolve (densidade por fonte, pesos para servir, função de previsão)."""
    assert alvos is None or not np.isnan(alvos).any(), "alvo do professor faltando para um prato de treino"
    dc = dens_por_fonte(FF[m], massa[m], prot[m])
    e, delta = None, None
    if ext is not None:
        delta = vies_dominio(ext["T"], T[m], ext["cols_massa"])
        e = dict(ext, T=ext["T"] - delta, c=TETO_EXTERNO * m.sum() / len(ext["X"]))
    pesos, prev = treina_cabeca(X[m], T[m], FF[m], peso_faixa(prot[m]), dc, cfg, dev, alvos, ext=e)
    if delta is not None:
        pesos["delta_ext"] = delta.astype(np.float32)
    return dc, pesos, prev


def cabeca_folds(X, T, FF, massa, prot, fold, cfg, dev, alvos_aninhados=None, tempo=lambda: None, ext=None):
    """Previsões fora-do-fold da cabeça de frações: para cada fold k, treina sem k e prevê k. Com alvos_aninhados
    [5, n, 3] (U1), o aluno sem o fold k é destilado com alvos_aninhados[k], que vêm de professores que também não
    viram o fold k. Devolve (22 saídas [n, o, 22], frações [n, o, K], densidade por fonte de cada fold [5, K])."""
    n, o = X.shape[:2]
    P, F = np.zeros((n, o, T.shape[1]), np.float32), np.zeros((n, o, FF.shape[1]), np.float32)
    dens = np.zeros((int(fold.max()) + 1, FF.shape[1]), np.float32)
    for k in sorted(set(fold.tolist())):
        tempo()
        m = fold != k
        dens[k], _, prev = _uma_cabeca(X, T, FF, massa, prot, m, cfg, dev, None if alvos_aninhados is None else alvos_aninhados[k][m], ext)
        P[~m], F[~m] = prev(X[~m])
    return P, F, dens


def fracoes_aninhado(X, T, FF, massa, prot, fold, cfg, dev, tempo=lambda: None, ext=None):
    """[5, n]: na posição [k, i], a densidade pelas frações do prato i (fold j != k) prevista por uma cabeça treinada SEM
    os folds j e k (8 orientações). Linhas do próprio fold k: NaN."""
    an = np.full((int(fold.max()) + 1, len(fold)), np.nan, np.float32)
    for j, k, S in pares_aninhados(fold):
        tempo()
        dc, _, prev = _uma_cabeca(X, T, FF, massa, prot, np.isin(fold, list(S)), cfg, dev, ext=ext)
        an[k, fold == j] = prev(X[fold == j])[1].mean(1) @ dc
        an[j, fold == k] = prev(X[fold == k])[1].mean(1) @ dc
    return an


def combina_professor(lm, ld):
    """Professor: massa = média em log das massas; densidade = média em log (com piso) das densidades; proteína =
    massa × densidade. Entradas: listas de log-massa e de log(densidade + piso). Devolve (log-massa, log-densidade com
    piso, log1p-proteína)."""
    lm, ld = np.mean(lm, 0), np.mean(ld, 0)
    return lm, ld, np.log1p(np.exp(lm) * np.clip(np.exp(ld) - EPS_D, 0, None))


def alvos_do_aluno(alvos_oof, alvos_aninhados, k):
    """Alvos do professor para o aluno treinado sem o fold k (k = None: modelo final, com as previsões fora-do-fold
    comuns). Para k dado, o alvo de um prato do fold j vem de um professor treinado sem os folds j e k."""
    return alvos_oof if k is None else alvos_aninhados[k]


def rotas_cv(D, pred, sufixo=""):
    """Critério de CV ponderado (cinco faixas com o mesmo peso) das rotas de proteína da regra de 12.3, numa variante de
    orientações. Só resume no registro; a escolha é feita na CPU pelo agente de avaliação."""
    oof = pred["oof" + sufixo]
    v = ~np.isnan(oof[:, 0])
    w, real = D.w_cv[v], D.proteina[v]
    cr = lambda est: round(float(100 * (acerto(real, est) * w).sum() / w.sum()), 2)
    massa, direta = np.exp(oof[v, I_M]), np.expm1(oof[v, I_P]).clip(0)
    d_reg = np.clip(np.exp(oof[v, I_D]) - EPS_D, 0, None)
    r = dict(direta=cr(direta), massa_x_dens=cr(massa * d_reg))
    if "oof_dens_ff" + sufixo in pred:
        d_ff = pred["oof_dens_ff" + sufixo][v]
        d_med = np.clip(np.exp((lg_dens(d_reg) + lg_dens(d_ff)) / 2) - EPS_D, 0, None)
        r.update(massa_x_dens_fracoes=cr(massa * d_ff), massa_x_dens_media=cr(massa * d_med), geometrica=cr(np.sqrt(direta * massa * d_med)))
    else:
        r.update(geometrica=cr(np.sqrt(direta * massa * d_reg)))
    return r


# ------------------------------------------------------------------ download e cache das imagens
def _png_ok(dados):
    return len(dados) > 64 and dados[:8] == b"\x89PNG\r\n\x1a\n" and dados[-8:-4] == b"IEND"


def baixa_um(dish, arquivo, raiz, tentativas=6):
    """Baixa <BUCKET>/<dish>/<arquivo> para raiz/img/<dish>/<arquivo>. Retomada: arquivo PNG íntegro é mantido."""
    import urllib.error, urllib.request
    dest = os.path.join(raiz, "img", dish, arquivo)
    if os.path.exists(dest):
        with open(dest, "rb") as f:
            if _png_ok(f.read()):
                return "tinha"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    erro = None
    for t in range(tentativas):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{dish}/{arquivo}", timeout=60) as r:
                dados = r.read()
                tam = r.headers.get("Content-Length")
            if (tam and int(tam) != len(dados)) or not _png_ok(dados):
                raise IOError(f"arquivo truncado ({len(dados)} bytes)")
            tmp = f"{dest}.parte{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(dados)
            os.replace(tmp, dest)
            return "baixou"
        except urllib.error.HTTPError as e:
            erro = f"HTTP {e.code}"
            if e.code in (403, 404):     # não existe no bucket: tentar de novo não adianta
                break
        except Exception as e:           # rede: espera e tenta de novo
            erro = f"{type(e).__name__}: {e}"
        time.sleep(min(30, 2 ** t) * (0.5 + np.random.rand()))
    return f"erro: {erro}"


class Trava:
    """Trava entre processos por arquivo: dois processos não baixam os mesmos pesos nem extraem as mesmas features."""

    def __init__(self, arq):
        self.arq = arq

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.arq, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode()); os.close(fd)
                return self
            except FileExistsError:
                try:
                    os.kill(int(open(self.arq).read() or 0), 0)     # o dono ainda existe?
                except (ValueError, ProcessLookupError, FileNotFoundError):
                    try:
                        os.remove(self.arq)                          # trava órfã
                    except FileNotFoundError:
                        pass
                    continue
                except PermissionError:
                    pass
                time.sleep(2)

    def __exit__(self, *exc):
        try:
            os.remove(self.arq)
        except FileNotFoundError:
            pass


def baixa_arquivo(url, dest, tentativas=8, ate=None):
    """Download de arquivo grande (pesos, base externa) com novas tentativas e retomada por Range. Só existe em `dest` se
    completo. ate: instante-limite (desiste com TimeoutError; o parcial fica para uma retomada)."""
    import urllib.error, urllib.request
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with Trava(dest + ".trava"):
        parte, erro = dest + ".parte", None
        for t in range(tentativas):
            if os.path.exists(dest):
                return dest
            try:
                ja = os.path.getsize(parte) if os.path.exists(parte) else 0
                req = urllib.request.Request(url, headers={"Range": f"bytes={ja}-"} if ja else {})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if ja and r.status != 206:      # servidor ignorou o Range: recomeça
                        ja = 0
                    total = ja + int(r.headers.get("Content-Length") or 0)
                    with open(parte, "ab" if ja else "wb") as f:
                        while True:
                            bloco = r.read(1 << 22)
                            if not bloco:
                                break
                            f.write(bloco)
                            if ate and time.time() > ate:
                                raise TimeoutError(f"download de {os.path.basename(dest)} passou do prazo")
                if total and os.path.getsize(parte) != total:
                    raise IOError(f"incompleto: {os.path.getsize(parte)} de {total} bytes")
                os.replace(parte, dest)
                print(f"download: {os.path.basename(dest)} ({os.path.getsize(dest) / 1e6:.0f} MB) de {url}", flush=True)
                return dest
            except TimeoutError:
                raise
            except urllib.error.HTTPError as e:
                erro = f"HTTP {e.code}"
                if e.code == 416 and os.path.exists(parte):    # Range além do fim: o parcial não serve
                    os.remove(parte)
                elif e.code in (401, 403, 404):
                    break
            except Exception as e:
                erro = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** t) * (0.5 + np.random.rand()))
    raise IOError(f"não foi possível baixar {url}: {erro}")


def baixa(dishes, arquivo, raiz, threads=16):
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(lambda d: baixa_um(d, arquivo, raiz), dishes))
    falhas = [(d, r) for d, r in zip(dishes, res) if r.startswith("erro")]
    diz(f"download {arquivo}: {res.count('baixou')} baixados, {res.count('tinha')} já no disco, {len(falhas)} falhas ({time.time() - t0:.0f}s)")
    if falhas:
        raise SystemExit(f"Não foi possível baixar {len(falhas)} arquivos (ex.: {falhas[:3]}). Sem todos os pratos a higiene "
                         "do treino/avaliação não é a mesma da CPU; rode de novo (o download é retomado).")


def md5_arquivo(arq):
    h = hashlib.md5()
    with open(arq, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 24), b""):
            h.update(bloco)
    return h.hexdigest()


def prepara_externo(a, D):
    """Fotos da base externa prontas no cache. Base do registro EXTERNOS: baixa o arquivo (com retomada), confere o
    tamanho e o md5 e extrai só as fotos que a lista cita (pelo nome do arquivo; o caminho dentro do .tar é ignorado).
    Com --externo-imagens, as fotos já estão numa pasta e nada é baixado."""
    C = Cache(a.cache, D)
    if os.path.exists(os.path.join(C.pasta_ext(), "pronto.json")):
        return
    raiz, info = D.ext["raiz"], EXTERNOS[a.externo]
    if raiz is None:
        base = os.path.join(a.cache, "ext", a.externo)
        raiz, tar = os.path.join(base, "img"), os.path.join(base, info["arquivo"])
        os.makedirs(raiz, exist_ok=True)
        quer = {os.path.basename(f) for f in D.ext["imagens"]}
        if not all(os.path.exists(os.path.join(raiz, f)) for f in quer):
            t0 = time.time()
            baixa_arquivo(info["url"], tar, ate=getattr(a, "externo_ate", None))
            if os.path.getsize(tar) != info["bytes"] or md5_arquivo(tar) != info["md5"]:
                os.remove(tar)
                raise IOError(f"{info['arquivo']}: tamanho ou md5 diferente do registrado; arquivo apagado para baixar de novo")
            diz(f"base externa {a.externo}: {os.path.getsize(tar) / 2 ** 30:.2f} GB baixados e conferidos (md5) em {time.time() - t0:.0f}s")
            n = 0
            with tarfile.open(tar) as t:
                for m in t:
                    nome = os.path.basename(m.name)
                    if m.isfile() and nome in quer and not os.path.exists(os.path.join(raiz, nome)):
                        with t.extractfile(m) as fo, open(os.path.join(raiz, nome + ".parte"), "wb") as fd:
                            shutil.copyfileobj(fo, fd, 1 << 22)
                        os.replace(os.path.join(raiz, nome + ".parte"), os.path.join(raiz, nome)); n += 1
            diz(f"base externa {a.externo}: {n} fotos extraídas ({time.time() - t0:.0f}s desde o início do download)")
    C.prepara_ext(raiz)


class Cache:
    """Imagens decodificadas em disco (memmap), na ordem de Dados.todos: rgb uint8 [n, 480, 640, 3], cor da mesa
    [n, 3] (mediana da moldura, para o aumento de escala) e, se pedido, altura acima da mesa em cm [n, 240, 320]."""
    H, W = 480, 640

    def __init__(self, raiz, D):
        self.raiz, self.D = raiz, D
        self.pasta = os.path.join(raiz, f"cache_{D.chave}")
        self.pos = {d: i for i, d in enumerate(D.todos)}
        self.rgb = self.cor = self.alt = None

    def _arq(self, nome):
        return os.path.join(self.pasta, nome)

    def prepara(self, profundidade=False):
        os.makedirs(self.pasta, exist_ok=True)
        n = len(self.D.todos)
        if not os.path.exists(self._arq("rgb.npy")):
            baixa(self.D.todos, "rgb.png", self.raiz)
            from PIL import Image
            t0 = time.time()
            mm = np.lib.format.open_memmap(self._arq("rgb.tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, self.H, self.W, 3))
            cor = np.zeros((n, 3), np.float32)

            def um(i):
                im = Image.open(os.path.join(self.raiz, "img", self.D.todos[i], "rgb.png")).convert("RGB")
                fora = im.size != (self.W, self.H)
                if fora:
                    im = im.resize((self.W, self.H), Image.BICUBIC)
                a = np.asarray(im); mm[i] = a
                k = max(2, int(0.03 * min(a.shape[:2])))   # moldura externa de 3%, como zen_extrai._borda
                cor[i] = np.median(np.concatenate([a[:k].reshape(-1, 3), a[-k:].reshape(-1, 3), a[:, :k].reshape(-1, 3), a[:, -k:].reshape(-1, 3)]), 0) / 255
                return fora
            with ThreadPoolExecutor(os.cpu_count() or 2) as ex:
                fora = sum(ex.map(um, range(n)))
            mm.flush(); del mm
            np.save(self._arq("cor.npy"), cor)
            os.replace(self._arq("rgb.tmp.npy"), self._arq("rgb.npy"))
            diz(f"cache RGB: {n} fotos decodificadas ({fora} fora de 640×480, redimensionadas) em {time.time() - t0:.0f}s")
        if profundidade and not os.path.exists(self._arq("alt.npy")):
            baixa(self.D.todos, "depth_raw.png", self.raiz)
            from PIL import Image
            t0 = time.time()
            alt = np.zeros((n, self.H // 2, self.W // 2), np.float16)

            def um(i):
                d = np.asarray(Image.open(os.path.join(self.raiz, "img", self.D.todos[i], "depth_raw.png"))).astype(np.float32)
                if d.shape != (self.H, self.W):
                    return 1
                val = d > 0
                if val.sum() < 1000:
                    return 1
                if np.median(d[val]) < 1000:      # primeiros pratos vêm em mm; o resto em 0,1 mm
                    d = d * 10
                k = int(0.08 * self.H)
                borda = np.concatenate([d[:k].ravel(), d[-k:].ravel(), d[:, :k].ravel(), d[:, -k:].ravel()])
                mesa = np.median(borda[borda > 0]) if (borda > 0).any() else np.median(d[val])
                a = np.where(val, (mesa - d) / 100.0, 0.0).clip(-2, 15)     # cm acima da mesa; sem leitura = 0
                alt[i] = a.reshape(self.H // 2, 2, self.W // 2, 2).mean((1, 3))
                return 0
            with ThreadPoolExecutor(os.cpu_count() or 2) as ex:
                ruins = sum(ex.map(um, range(n)))
            np.save(self._arq("alt.tmp.npy"), alt); os.replace(self._arq("alt.tmp.npy"), self._arq("alt.npy"))
            diz(f"cache de profundidade: {n} mapas ({ruins} sem leitura útil, ficam em zero) em {time.time() - t0:.0f}s")
        return self

    def pasta_ext(self):
        return os.path.join(self.raiz, f"ext_{self.D.ext['hash']}")

    def prepara_ext(self, raiz_fotos):
        """Cache das fotos externas (à parte do Nutrition5k): cada foto cabe inteira em 640×480, com a cor da moldura em
        volta (a escala de cada base é outra; no treino elas entram com escala aleatória e viés de massa por domínio)."""
        e, pasta = self.D.ext, self.pasta_ext()
        if os.path.exists(os.path.join(pasta, "pronto.json")):
            return self
        os.makedirs(pasta, exist_ok=True)
        from PIL import Image
        t0, n = time.time(), len(e["ids"])
        falta = [f for f in e["imagens"] if not acha_foto(raiz_fotos, f)]
        if falta:
            raise IOError(f"{len(falta)} fotos da lista externa não estão em {raiz_fotos} (ex.: {falta[:3]})")
        mm = np.lib.format.open_memmap(os.path.join(pasta, "rgb.tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, self.H, self.W, 3))
        cor = np.zeros((n, 3), np.float32)

        def um(i):
            im = Image.open(acha_foto(raiz_fotos, e["imagens"][i])).convert("RGB")
            z = min(self.W / im.size[0], self.H / im.size[1])
            pq = np.asarray(im.resize((max(1, round(im.size[0] * z)), max(1, round(im.size[1] * z))), Image.BICUBIC))
            kb = max(2, int(0.03 * min(pq.shape[:2])))
            cor[i] = np.median(np.concatenate([pq[:kb].reshape(-1, 3), pq[-kb:].reshape(-1, 3), pq[:, :kb].reshape(-1, 3), pq[:, -kb:].reshape(-1, 3)]), 0) / 255
            tela = np.empty((self.H, self.W, 3), np.uint8); tela[:] = (cor[i] * 255).astype(np.uint8)
            y0, x0 = (self.H - pq.shape[0]) // 2, (self.W - pq.shape[1]) // 2
            tela[y0:y0 + pq.shape[0], x0:x0 + pq.shape[1]] = pq
            mm[i] = tela
        with ThreadPoolExecutor(os.cpu_count() or 2) as ex:
            list(ex.map(um, range(n)))
        mm.flush(); del mm
        np.save(os.path.join(pasta, "cor.npy"), cor)
        os.replace(os.path.join(pasta, "rgb.tmp.npy"), os.path.join(pasta, "rgb.npy"))
        grava_json(os.path.join(pasta, "pronto.json"), dict(n=n, lista=e["hash"], seg=round(time.time() - t0, 1)))
        diz(f"cache das fotos externas: {n} fotos em {time.time() - t0:.0f}s")
        return self

    def abre(self, profundidade=False, ext=False):
        self.rgb = np.load(self._arq("rgb.npy"), mmap_mode="r")
        self.cor = np.load(self._arq("cor.npy"))
        if profundidade:
            self.alt = np.load(self._arq("alt.npy"), mmap_mode="r")
        assert len(self.rgb) == len(self.D.todos)
        self.ext_rgb = self.ext_cor = None
        if ext and self.D.ext:        # linhas das fotos externas: depois das do Nutrition5k
            self.ext_rgb = np.load(os.path.join(self.pasta_ext(), "rgb.npy"), mmap_mode="r")
            self.ext_cor = np.load(os.path.join(self.pasta_ext(), "cor.npy"))
            self.pos.update({d: len(self.rgb) + i for i, d in enumerate(self.D.ext["ids"])})
        return self


# ------------------------------------------------------------------ modelo e pré-processamento (torch só aqui para baixo)
def orienta(x, k):
    """k-ésimo elemento do grupo diedral aplicado a [..., H, W]; mesma ordem de zen_extrai.orienta
    (original, 180°, espelho H, espelho V, 90°, 270°, transposta, antitransposta)."""
    if k == 0: return x
    if k == 1: return x.flip(-2, -1)
    if k == 2: return x.flip(-1)
    if k == 3: return x.flip(-2)
    t = x.transpose(-2, -1)
    return [t.flip(-2), t.flip(-1), t, t.flip(-2, -1)][k - 4]


def orienta_lote(x, ks):
    """Orientação por amostra. Entrada não quadrada: todas as amostras do lote no mesmo formato (ks < 4 ou ks >= 4)."""
    ks = np.asarray(ks)
    if (ks == ks[0]).all():
        return orienta(x, int(ks[0]))
    assert x.shape[-1] == x.shape[-2] or (ks < 4).all() or (ks >= 4).all()
    forma = x.shape if ks[0] < 4 else (*x.shape[:-2], x.shape[-1], x.shape[-2])
    import torch
    out = x.new_empty(forma)
    for k in np.unique(ks):
        m = torch.from_numpy(np.where(ks == k)[0]).to(x.device)
        out[m] = orienta(x[m], int(k))
    return out


def redimensiona(x, hw, zoom=None, cor=None):
    """x [B, C, H, W] em [0, 1] -> [B, C, h, w]. zoom por amostra, com a semântica de zen_extrai.zoom: z > 1 aproxima
    (recorte central de 1/z); z < 1 afasta (mais 'mesa' em volta, na cor mediana da moldura da foto)."""
    import torch
    F = torch.nn.functional
    if zoom is None:
        return F.interpolate(x, size=hw, mode="bicubic", antialias=True, align_corners=False)
    B, Cn, H, W = x.shape
    out = []
    for i in range(B):
        z, xi = float(zoom[i]), x[i:i + 1]
        if z > 1:
            ch, cw = H / z, W / z
            xi = xi[:, :, round((H - ch) / 2):round((H + ch) / 2), round((W - cw) / 2):round((W + cw) / 2)]
        elif z < 1:
            ph, pw = int(round((H / z - H) / 2)), int(round((W / z - W) / 2))
            tela = cor[i].view(1, Cn, 1, 1).expand(1, Cn, H + 2 * ph, W + 2 * pw).clone()
            tela[:, :, ph:ph + H, pw:pw + W] = xi
            xi = tela
        out.append(F.interpolate(xi, size=hw, mode="bicubic", antialias=True, align_corners=False))
    return torch.cat(out)


def monta_lote(C, linhas, dev, hw, canais=3, zoom=None):
    """Lê as fotos do cache e devolve [B, canais, h, w]: RGB em [0, 1] (+ altura em cm / 5 no 4º canal)."""
    import torch
    linhas = np.asarray(linhas)
    n5k = len(C.rgb)
    if getattr(C, "ext_rgb", None) is not None and (linhas >= n5k).any():     # lote com fotos externas (cache à parte)
        rgb = np.stack([C.rgb[i] if i < n5k else C.ext_rgb[i - n5k] for i in linhas])
        cor = np.stack([C.cor[i] if i < n5k else C.ext_cor[i - n5k] for i in linhas])
    else:
        rgb, cor = np.ascontiguousarray(C.rgb[linhas]), C.cor[linhas]
    x = torch.from_numpy(rgb).to(dev).permute(0, 3, 1, 2).float().div_(255)
    cor = torch.from_numpy(cor).to(dev)
    if canais == 4:
        a = torch.from_numpy(np.stack([C.alt[i] for i in linhas]).astype(np.float32)).to(dev)[:, None] / 5
        x = torch.cat([x, torch.nn.functional.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)], 1)
        cor = torch.cat([cor, torch.zeros(len(linhas), 1, device=dev)], 1)
    x = redimensiona(x, hw, zoom, cor)
    x[:, :3].clamp_(0, 1)
    return x


def cria_rede(bb, n_saidas, pretrained=True, drop_path=0.0, ckpt_grad=False, canais=3):
    """Backbone do timm + cabeça linear sobre [agregado ‖ média dos tokens] (os blocos 'agregado' e 'média' da CPU)."""
    import timm, torch

    class Rede(torch.nn.Module):
        def __init__(self):
            super().__init__()
            b = BACKBONES[bb]
            arq_timm = b["timm"].split(".")[0]
            kw = dict(num_classes=0, img_size=b["hw"], dynamic_img_size=b["hw"][0] != b["hw"][1], drop_path_rate=drop_path)
            erros = []
            for fonte in (b["fontes"] if pretrained else ()):
                try:
                    if fonte == "url":   # checkpoint original num arquivo local; o timm converte (.npz) ou filtra (.pth)
                        arq = baixa_arquivo(b["url"], os.path.join(PASTA_PESOS, os.path.basename(b["url"])))
                        self.m = timm.create_model(arq_timm, pretrained=True, pretrained_cfg_overlay=dict(file=arq, custom_load=arq.endswith(".npz")), **kw)
                    else:
                        self.m = timm.create_model(b["timm"], pretrained=True, **kw)
                    _anota_origem(bb, fonte)
                    break
                except Exception as e:
                    erros.append(f"{fonte}: {type(e).__name__}: {str(e)[:300]}")
                    print(f"pesos de {bb} pela origem '{fonte}' falharam ({erros[-1]})", flush=True)
            else:
                if pretrained:
                    raise RuntimeError(f"sem pesos para {bb}: {erros}")
                self.m = timm.create_model(arq_timm, pretrained=False, **kw)
            if ckpt_grad:
                self.m.set_grad_checkpointing(True)
            pc = self.m.pretrained_cfg
            media, desvio = list(pc.get("mean", (0.5,) * 3)), list(pc.get("std", (0.5,) * 3))
            if canais == 4:   # 4º canal (altura) entra com peso zero: o professor parte exatamente do modelo RGB
                p = self.m.patch_embed.proj
                novo = torch.nn.Conv2d(4, p.out_channels, p.kernel_size, p.stride, p.padding, bias=p.bias is not None)
                with torch.no_grad():
                    novo.weight.zero_(); novo.weight[:, :3] = p.weight
                    if p.bias is not None:
                        novo.bias.copy_(p.bias)
                self.m.patch_embed.proj = novo
                media, desvio = media + [0.0], desvio + [1.0]
            self.register_buffer("media", torch.tensor(media).view(1, -1, 1, 1))
            self.register_buffer("desvio", torch.tensor(desvio).view(1, -1, 1, 1))
            self.dim = 2 * self.m.num_features
            self.head = torch.nn.Linear(self.dim, n_saidas)

        def feats(self, x):
            t = self.m.forward_features((x - self.media) / self.desvio)
            return torch.cat([self.m.forward_head(t, pre_logits=True), t[:, self.m.num_prefix_tokens:].mean(1)], 1)

        def forward(self, x):
            f = self.feats(x)
            return self.head(f), f

    return Rede()


def _anota_origem(bb, fonte):
    """Registra de qual origem os pesos públicos do backbone foram de fato carregados (para servir__*.json)."""
    try:
        os.makedirs(PASTA_PESOS, exist_ok=True)
        with open(os.path.join(PASTA_PESOS, f"origem__{bb}.txt"), "w") as f:
            f.write(fonte)
    except OSError:
        pass


def sha256_arquivo(arq):
    h = hashlib.sha256()
    with open(arq, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 24), b""):
            h.update(bloco)
    return h.hexdigest()


def descreve_backbone(bb):
    """Checkpoint público do backbone: origem usada, URL ou repositório, arquivo, tamanho e sha256. É o que permite
    servir sem regravar o backbone. Chamar antes de apagar o arquivo do cache."""
    b = BACKBONES[bb]
    try:
        fonte = open(os.path.join(PASTA_PESOS, f"origem__{bb}.txt")).read().strip()
    except OSError:
        fonte = b["fontes"][0]
    o = dict(chave=bb, timm=b["timm"], fonte=fonte, url=b.get("url"), hf_id=None, arquivo=None, bytes=None, sha256=None)
    arq = None
    try:
        import timm
        o["timm_versao"] = timm.__version__
        o["hf_id"] = timm.models.get_pretrained_cfg(b["timm"]).hf_hub_id
    except Exception:
        pass
    if fonte == "url":
        arq = os.path.join(PASTA_PESOS, os.path.basename(b["url"]))
    elif o["hf_id"]:
        try:
            from huggingface_hub import try_to_load_from_cache
            for nome in ("model.safetensors", "open_clip_model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin"):
                cam = try_to_load_from_cache(o["hf_id"], nome)
                if isinstance(cam, str) and os.path.exists(cam):
                    arq = cam
                    break
        except Exception:
            pass
    if arq and os.path.exists(arq):
        o.update(arquivo=os.path.basename(arq), bytes=os.path.getsize(arq), sha256=sha256_arquivo(arq))
    return o


def grupos_lr(rede, lr, lld, wd):
    """Decaimento de lr por camada: a cabeça e o topo usam lr; cada bloco abaixo multiplica por lld."""
    L = len(rede.m.blocks)
    grupos = {}
    for n, p in rede.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("m.blocks."):
            cam = int(n.split(".")[2]) + 1
        elif n.startswith(("m.patch_embed", "m.pos_embed", "m.cls_token", "m.reg_token")):
            cam = 0
        else:                      # cabeça, norma final, agregação por atenção
            cam = L + 1
        sem_wd = p.ndim <= 1 or n.endswith(("pos_embed", "cls_token", "reg_token", "latent"))
        grupos.setdefault((cam, sem_wd), []).append(p)
    return [dict(params=ps, lr=lr * lld ** (L + 1 - cam), weight_decay=0.0 if sem_wd else wd) for (cam, sem_wd), ps in sorted(grupos.items())]


class SemOrcamento(Exception):
    pass


def autocast(dev):
    import torch
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


def feats_congeladas(bb, D, C, dev, bs, prazo, so_treino=False):
    """Features do backbone congelado, 8 orientações: [n, 8, 2·dim] float16 (cache em disco). É a entrada do ridge, das
    cabeças de frações e do ponto de partida da cabeça no ajuste fino. so_treino: só os pratos de treino (as fotos de
    previsão não são tocadas quando o backbone só serve para iniciar a cabeça de um ajuste fino)."""
    arq = os.path.join(C.pasta, f"feat__{bb}.npy")
    if so_treino and not os.path.exists(arq):
        arq = os.path.join(C.pasta, f"feat__{bb}__treino.npy")
    with Trava(arq + ".trava"):
        if not os.path.exists(arq):
            _extrai_feats(arq, bb, D, C, dev, bs, prazo, D.segmentos[:1] if arq.endswith("__treino.npy") else D.segmentos)
    return np.load(arq)


def feats_ext(bb, D, C, dev, bs, prazo):
    """Features congeladas das fotos externas (todas: treino e teste_ext), 8 orientações, num cache à parte."""
    arq = os.path.join(C.pasta_ext(), f"feat__{bb}.npy")
    with Trava(arq + ".trava"):
        if not os.path.exists(arq):
            n = len(C.rgb)
            _extrai_feats(arq, bb, D, C, dev, bs, prazo, [(n, n + len(D.ext["ids"]))], base=n)
    return np.load(arq)


def aplica_mlp(z, X):
    """Aplica uma cabeça MLP gravada (cabeca__*.npz) a features X [n, o, D]: 22 saídas em log, média das orientações."""
    import torch
    with torch.inference_mode():
        h = torch.nn.functional.gelu(torch.from_numpy(((X - z["mu"]) / z["sd"]).reshape(-1, X.shape[-1]).astype(np.float32)) @ torch.from_numpy(z["W1"]).T + torch.from_numpy(z["b1"]))
        o = (h @ torch.from_numpy(z["W2"]).T + torch.from_numpy(z["b2"])).numpy()
    nt = len(z["tm"])
    return (o[:, :nt] * z["ts"] + z["tm"]).reshape(X.shape[0], X.shape[1], nt).mean(1)


def _extrai_feats(arq, bb, D, C, dev, bs, prazo, segmentos, base=0):
    import torch
    t0 = time.time()
    rede = cria_rede(bb, D.N).to(dev).eval()
    hw = BACKBONES[bb]["hw"]
    X = np.zeros((segmentos[-1][1] - base, 8, rede.dim), np.float16)
    with torch.inference_mode():
        for ini, fim in segmentos:   # treino (+ 450) nos mesmos lotes de sempre; externos e teste2 em lotes à parte, depois
            for i in range(ini, fim, bs):
                if time.time() > prazo:
                    raise SemOrcamento()
                x = monta_lote(C, list(range(i, min(i + bs, fim))), dev, hw)
                for k in range(8):
                    with autocast(dev):
                        f = rede.feats(orienta(x, k))
                    X[i - base:i - base + len(x), k] = f.float().cpu().numpy()
    tmp = f"{arq}.{os.getpid()}.npy"
    np.save(tmp, X); os.replace(tmp, arq)
    print(f"features congeladas {bb}: {X.shape} em {time.time() - t0:.0f}s ({len(X) * 8 / (time.time() - t0):.0f} img/s)", flush=True)


def pesos_perda(D, n_saidas):
    w = np.ones(n_saidas, np.float32)   # mesmos pesos por alvo do ajuste fino da CPU (zen_finetune.py)
    w[I_K] = 0.5; w[I_F] = 0.3; w[I_C] = 0.3; w[D.I_MG] = 0.15; w[D.I_PG] = 0.15
    return w


def treina(cfg, D, C, tr, va, dev, prazo, div_lote=1, canais=3, professor=None, fator_orc=1.0, semente=0, rotulo="", alvos_prof=None):
    """Ajuste fino completo. tr/va: índices em D.ids (va só para registrar a curva de validação; não decide nada).
    Rodada 3 (cfg): fracoes = cabeça de 22 + 10 saídas com as perdas de frações e de densidade composta; peso_faixa =
    cinco faixas de proteína com o mesmo peso em todas as perdas (contadas só em tr); alvos_prof [len(D.ids), 3] =
    alvos do professor (log-massa, log-densidade com piso, log1p-proteína) para a destilação; externo = soma ao treino
    as amostras da lista externa (gancho de 12.4). Sem nada disso, a conta é a das execuções 1 e 2.
    Devolve (rede com os pesos da média móvel, função de previsão, histórico por época, (média, desvio) dos alvos)."""
    import copy, torch
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    hw = BACKBONES[cfg["bb"]]["hw"]
    quad = hw[0] == hw[1]
    bs = max(1, min(cfg["bs"] // div_lote, len(tr))); acum = cfg["acum"] * div_lote
    T, N = D.T, D.N
    K = len(D.fontes) if cfg.get("fracoes") else 0
    ext = D.ext if cfg.get("externo") else None
    novo = bool(K or cfg.get("peso_faixa") or alvos_prof is not None or ext)     # perdas da rodada 3
    tm, ts = T[tr].mean(0), T[tr].std(0) + 1e-6
    # cabeça iniciada na solução do ridge sobre as features congeladas dos mesmos pratos (alfa por CV interna):
    # o ajuste fino parte do melhor modelo congelado, como na CPU
    Xc = feats_congeladas(cfg["bb"], D, C, dev, max(1, cfg["bs"] // div_lote), prazo, so_treino=bool(cfg.get("v3")))
    Xr = Xc[[C.pos[D.ids[i]] for i in tr]].astype(np.float32)
    _, alfa, _ = ridge_cv(Xr, T[tr], D.fold[tr])
    _, m = ridge_final(Xr, T[tr], alfa)
    del Xc, Xr
    rede = cria_rede(cfg["bb"], N + K, drop_path=cfg["drop_path"], ckpt_grad=cfg["ckpt_grad"], canais=canais).to(dev)
    with torch.no_grad():
        if K:       # logits das fontes partem de zero (frações iguais); as 22 saídas, do ridge
            rede.head.weight.zero_(); rede.head.bias.zero_()
        rede.head.weight[:N].copy_(torch.from_numpy((((m.W / m.sd[:, None]) / ts).T).astype(np.float32)))
        rede.head.bias[:N].copy_(torch.from_numpy(((m.mt - (m.mu / m.sd) @ m.W - tm) / ts).astype(np.float32)))
    ema = copy.deepcopy(rede).eval()
    ema.m.set_grad_checkpointing(False)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(grupos_lr(rede, cfg["lr"], cfg["lld"], cfg["wd"]))
    delta_ext = None
    if ext:         # só as fotos externas de treino; alvos de massa sem o viés de domínio (12.5)
        ie = np.where(ext["treina"])[0]
        delta_ext = vies_dominio(ext["T"][ie], T[tr], [I_M] + list(range(D.I_MG.start, D.I_MG.stop)))
    # por época: todas as fotos do Nutrition5k e um sorteio (sem reposição) de FRACAO_EXT_EPOCA × n fotos externas, cada
    # uma com o peso que faz a soma dos pesos externos ser TETO_EXTERNO × a do Nutrition5k (peso igual por foto externa)
    n_ext_ep = max(1, min(len(ie), int(FRACAO_EXT_EPOCA * len(tr)))) if ext else 0
    n_ep = len(tr) + n_ext_ep
    passos_ep = max(1, n_ep // (bs * acum)); total = cfg["epocas"] * passos_ep
    n_aq = max(1, int(cfg["aquec"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / n_aq if s < n_aq else 0.5 * (1 + math.cos(math.pi * (s - n_aq) / max(1, total - n_aq))))
    try:
        scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    w = torch.from_numpy(pesos_perda(D, N)).to(dev)
    tm_t, ts_t = torch.from_numpy(tm.astype(np.float32)).to(dev), torch.from_numpy(ts.astype(np.float32)).to(dev)
    huber = torch.nn.functional.smooth_l1_loss
    linha = np.array([C.pos.get(d, -1) for d in D.ids] + ([C.pos[ext["ids"][i]] for i in ie] if ext else []))   # -1: fora do subconjunto do --smoke
    # alvos por amostra; as externas vêm depois das do Nutrition5k, com NaN onde o alvo falta (perda mascarada)
    T_tudo = np.r_[T, ext["T"][ie] - delta_ext] if ext else T
    Yn = torch.from_numpy(np.nan_to_num((T_tudo - tm) / ts).astype(np.float32)).to(dev)
    if novo:
        n5k = len(D.ids)
        q = [I_M, I_D, I_P]
        tem_np = np.isfinite(T_tudo).astype(np.float32)                                               # máscara por alvo…
        if ext:     # …× peso do alvo nas externas (alvo sem variação no Nutrition5k deste treino, só no --smoke: fora)
            tem_np[n5k:] *= ext["peso_alvo"] * (ts > 1e-3)
        tem = torch.from_numpy(tem_np).to(dev)
        w_am = np.ones(len(T_tudo), np.float32)
        if cfg.get("peso_faixa"):
            w_am[tr] = peso_faixa(D.proteina[tr])
        if ext:
            w_am[n5k:] = TETO_EXTERNO * float(w_am[tr].sum()) / n_ext_ep
        Wam = torch.from_numpy(w_am).to(dev)
        so_n5k = torch.from_numpy((np.arange(len(T_tudo)) < n5k).astype(np.float32)).to(dev)          # densidade composta: só Nutrition5k
        if K:
            dc = dens_por_fonte(D.FF[tr], D.massa[tr], D.proteina[tr])
            dct = torch.from_numpy(dc.astype(np.float32)).to(dev)
            FF_tudo = np.r_[D.FF, ext["FF"][ie]] if ext else D.FF
            FFt = torch.from_numpy(np.nan_to_num(FF_tudo).astype(np.float32)).to(dev)
            tem_ff_np = np.isfinite(FF_tudo).all(1).astype(np.float32)
            if ext:
                tem_ff_np[n5k:] *= ext["peso_ff"]
            tem_ff = torch.from_numpy(tem_ff_np).to(dev)
        if alvos_prof is not None:
            assert not np.isnan(alvos_prof[tr]).any(), "alvo do professor faltando para um prato de treino"
            ap = np.full((len(T_tudo), 3), np.nan); ap[:n5k] = alvos_prof
            Pn = torch.from_numpy(np.nan_to_num((ap - tm[q]) / ts[q]).astype(np.float32)).to(dev)
            tem_p = torch.from_numpy(np.isfinite(ap).all(1).astype(np.float32)).to(dev)

    def perda_nova(out, jt):
        """Perdas da rodada 3, todas com o peso por amostra: Huber nas 22 saídas (com os pesos por alvo da receita);
        10 × erro quadrático das frações; Huber em log da densidade composta; destilação nas saídas 0, 3 e 1."""
        wj = Wam[jt]
        o22 = out[:, :N]
        p = ((huber(o22, Yn[jt], reduction="none") * w * tem[jt]).mean(1) * wj).mean()
        if K:
            fr = torch.softmax(out[:, N:], 1)
            p = p + 10 * (((fr - FFt[jt]) ** 2).sum(1) * tem_ff[jt] * wj).mean()
            alvo_d = Yn[jt][:, I_D] * ts_t[I_D] + tm_t[I_D]
            p = p + (huber((torch.log(fr @ dct + EPS_D) - alvo_d) / ts_t[I_D], torch.zeros_like(alvo_d), reduction="none") * so_n5k[jt] * wj).mean()
        if alvos_prof is not None:
            p = p + cfg.get("peso_destila", PESO_DESTILA) * (huber(o22[:, q], Pn[jt], reduction="none").mean(1) * tem_p[jt] * wj).mean()
        return p

    def preve(ids, n_or=8, variantes=False):
        """Média das previsões (em log) nas n_or orientações, com os pesos da média móvel. variantes=True: devolve
        ({sufixo: 22 saídas}, {sufixo: frações} ou None) para 1, 4 e 8 orientações (ORIENT), numa passada só."""
        out = np.zeros((len(ids), 8 if variantes else 1, N), np.float32)
        fr = np.zeros((len(ids), 8, K), np.float32) if variantes and K else None
        with torch.inference_mode():
            for i in range(0, len(ids), bs):
                x = monta_lote(C, [C.pos[d] for d in ids[i:i + bs]], dev, hw, canais)
                soma = 0
                for k in range(8 if variantes else n_or):
                    with autocast(dev):
                        o = ema(orienta(x, k))[0].float()
                    if variantes:
                        out[i:i + len(x), k] = (o[:, :N] * ts_t + tm_t).cpu().numpy()
                        if K:
                            fr[i:i + len(x), k] = torch.softmax(o[:, N:], 1).cpu().numpy()
                    else:
                        soma = soma + o[:, :N]
                if not variantes:
                    out[i:i + len(x), 0] = (soma / n_or * ts_t + tm_t).cpu().numpy()
        if not variantes:
            return out[:, 0]
        return {suf: media_or(out, k) for suf, k in ORIENT.items()}, ({suf: media_or(fr, k) for suf, k in ORIENT.items()} if K else None)

    preve.dens_fonte = dc if K else None
    preve.delta_ext = delta_ext

    def valida():
        pv = preve([D.ids[i] for i in va], 1); P = proteinas(pv); wv = D.w_cv[va]
        cr = lambda est: float(100 * (acerto(D.proteina[va], est) * wv).sum() / wv.sum())
        return dict(val_rmse_logm=float(np.sqrt(np.mean((pv[:, I_M] - T[va, I_M]) ** 2))), val_crit_direto=cr(P["direto"]), val_crit_mxd=cr(P["mxd"]))

    val_cada = cfg.get("val_cada", 1)       # a validação só registra a curva; nas rodada 3, a cada 5 épocas e na última
    hist, passo, t0 = [], 0, time.time()
    if va is not None and len(va):
        hist.append(dict(epoca=0, **valida()))
        print(f"{rotulo} época 0 (cabeça = ridge): {hist[-1]}", flush=True)
    t_treino = time.time()
    for ep in range(cfg["epocas"]):
        rede.train()
        amostras = tr if not ext else np.r_[tr, len(D.ids) + rng.choice(len(ie), n_ext_ep, replace=False)]
        perm = rng.permutation(len(amostras)); perdas = []
        for b in range(passos_ep):
            for a in range(acum):
                j = amostras[perm[(b * acum + a) * bs:(b * acum + a + 1) * bs]]
                if quad:
                    ks = rng.integers(0, 8, len(j))
                else:   # lote inteiro em paisagem (0–3) ou em retrato (4–7); a orientação dentro do formato é por amostra
                    ks = rng.integers(0, 4, len(j)) + 4 * int(rng.integers(0, 2))
                zoom = rng.uniform(*cfg["escala"], len(j)) if cfg["escala"] else None
                x = orienta_lote(monta_lote(C, linha[j], dev, hw, 4 if professor is not None else canais, zoom), ks)
                x4, x = (x, x[:, :3]) if professor is not None else (None, x)
                jt = torch.from_numpy(j).to(dev)
                with autocast(dev):
                    out, f = rede(x)
                    if novo:
                        perda = perda_nova(out.float(), jt)
                    else:
                        perda = (huber(out.float(), Yn[jt], reduction="none") * w).mean()
                    if professor is not None:   # destilação RGB-D (execução 1): saídas e features do professor na mesma vista
                        with torch.no_grad():
                            op, fp = professor(x4)
                        perda = perda + (huber(out.float(), op.float(), reduction="none") * w).mean() \
                            + 0.5 * (1 - torch.nn.functional.cosine_similarity(f.float(), fp.float(), dim=1)).mean()
                if not torch.isfinite(perda):     # precisão mista pode estourar: melhor falhar do que gravar lixo
                    raise RuntimeError(f"perda não finita no passo {passo} (época {ep + 1})")
                scaler.scale(perda / acum).backward()
                perdas.append(float(perda.detach()))
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            passo += 1
            d = min(cfg["ema"], (1 + passo) / (10 + passo))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), rede.parameters()):
                    pe.lerp_(p, 1 - d)
            if passo == min(total, max(3, total // 20)) or passo % 50 == 0:   # projeção de tempo: cabe no orçamento?
                por_passo = (time.time() - t_treino) / passo
                fim = time.time() + (total - passo) * por_passo * 1.1 + (total * por_passo * 1.1 + 120) * (fator_orc - 1) + 90
                if fim > prazo:
                    print(f"{rotulo} sem orçamento: faltariam {(fim - time.time()) / 60:.0f} min e restam {(prazo - time.time()) / 60:.0f} min", flush=True)
                    raise SemOrcamento()
        h = dict(epoca=ep + 1, perda=float(np.mean(perdas)), lr=float(sched.get_last_lr()[-1]), seg=round(time.time() - t0, 1),
                 img_s=round((ep + 1) * passos_ep * acum * bs / (time.time() - t_treino), 1))
        if va is not None and len(va) and ((ep + 1) % val_cada == 0 or ep + 1 == cfg["epocas"]):
            h.update(valida())
        hist.append(h)
        print(f"{rotulo} {h}", flush=True)
    return ema, preve, hist, (tm, ts)


# ------------------------------------------------------------------ uma parte (processo filho)
def arq_parte(saida, cfg, parte):
    return os.path.join(saida, f"parte__{slug(cfg['nome'])}__{parte}.npz")


def arq_de(saida, cfg, tipo):
    """Arquivos por configuração: pesos (.pt), cabeca (.npz), servir (.json), servico (.json)."""
    return os.path.join(saida, f"{tipo}__{slug(cfg['nome'])}." + dict(pesos="pt", cabeca="npz", servir="json", servico="json")[tipo])


def nomes_saidas(D):
    return ["log_massa", "log1p_proteina", "log1p_kcal", "log_densidade", "log1p_gordura", "log1p_carbo"] \
        + [f"log1p_m_{g}" for g in D.grupos] + [f"log1p_p_{g}" for g in D.grupos]


def parte_pronta(a, cfg, parte, D):
    """A parte está em disco COM tudo o que esta execução precisa? Folds: só o arquivo. final/tudo: previsões de todos
    os conjuntos das listas (inclusive teste2) e os arquivos de servir. Parte de execução antiga sem isso é refeita."""
    if cfg["tipo"] == "externo":       # pronto = cache das fotos externas montado (não gera previsões)
        return os.path.exists(os.path.join(Cache(a.cache, D).pasta_ext(), "pronto.json"))
    if cfg["tipo"] == "decisao":
        return os.path.exists(os.path.join(a.saida, f"{slug(cfg['nome'])}.json"))
    arq = arq_parte(a.saida, cfg, parte)
    if not os.path.exists(arq):
        return False
    if cfg.get("externo"):      # treinada com a base externa: só vale se foi com a mesma lista
        with np.load(arq, allow_pickle=False) as z:
            if D.ext is None or "ext_lista" not in z.files or str(z["ext_lista"]) != D.ext["hash"]:
                return False
    if parte in ("tudo", "final"):
        with np.load(arq, allow_pickle=False) as z:
            if any(sp not in z.files for sp in D.conj):
                return False
        if cfg["tipo"] == "professor":      # composição de outros modelos: não tem arquivos de servir próprios
            return True
        if not os.path.exists(arq_de(a.saida, cfg, "servir")):
            return False
        if parte == "tudo" and not os.path.exists(arq_de(a.saida, cfg, "cabeca")):
            return False
        if parte == "final" and a.guarda_pesos == "todos" and not os.path.exists(arq_de(a.saida, cfg, "pesos")):
            return False
    return True


def grava_servir(a, cfg, D, dim, cabeca):
    """servir__<config>.json: o que é preciso para reproduzir a previsão fora deste script (pré-processamento, backbone
    público com URL e sha256, orientações, cabeça, leitura das 22 saídas)."""
    b = BACKBONES[cfg["bb"]]
    try:        # nada aqui pode derrubar uma parte já treinada: o que falhar fica registrado no próprio arquivo
        import timm
        pc = timm.models.get_pretrained_cfg(b["timm"])
        media, desvio = list(pc.mean), list(pc.std)
        bk = descreve_backbone(cfg["bb"])
    except Exception as e:
        media = desvio = None
        bk = dict(chave=cfg["bb"], timm=b["timm"], url=b.get("url"), hf_id=None, fonte=None, arquivo=None, bytes=None, sha256=None, erro=f"{type(e).__name__}: {e}")
    spec = dict(
        formato=FORMATO, nome=cfg["nome"], tipo=cfg["tipo"], listas_hash=D.hash, n_treino=int(D.usa_tr.sum()),
        backbone=dict(bk, basta_o_checkpoint_publico=cfg["tipo"] == "ridge"),
        entrada=dict(foto="RGB zenital inteira, sem recorte (640×480 no Nutrition5k)", hw=list(b["hw"]),
                     redimensiona="bicúbico com antialias, da foto inteira direto para (H, W) = hw "
                                  "(torch.nn.functional.interpolate, align_corners=False); com hw quadrado a proporção é achatada",
                     escala="uint8 / 255, limitado a [0, 1] depois de redimensionar", media=media, desvio=desvio),
        orientacoes=dict(n=8, ordem=["original", "180°", "espelho horizontal", "espelho vertical", "90°", "270°", "transposta", "antitransposta"],
                         regra="média das 22 saídas (em log) nas 8 orientações; com hw não quadrado, as 4 últimas entram em retrato (W × H)"),
        features="[saída agregada do backbone (timm forward_head com pre_logits=True) ‖ média dos tokens de patch depois da norma final]",
        dim_features=int(dim), precisao="backbone em fp16 (torch.autocast) na GPU, como nas previsões gravadas",
        cabeca=cabeca,
        saidas=dict(nomes=nomes_saidas(D),
                    leitura="massa_g = exp(s[0]); proteína direta_g = expm1(s[1]); kcal = expm1(s[2]); densidade (g de proteína por g) = "
                            "exp(s[3]) − 0.01; gordura_g = expm1(s[4]); carboidrato_g = expm1(s[5]); massa e proteína por grupo: expm1"),
        cfg=cfg)
    grava_json(arq_de(a.saida, cfg, "servir"), spec)


def roda_parte(a, cfg, parte, cfgs=()):
    global PASTA_PESOS
    import torch
    t0 = time.time()
    PASTA_PESOS = os.path.join(a.cache, "pesos")
    os.environ.setdefault("HF_HOME", os.path.join(a.cache, "hf"))     # pesos do Hugging Face fora da pasta de saída
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    dev = torch.device(a.dispositivo)
    D = Dados(a.listas, a.smoke, a.rodada, a.externo_lista, a.externo_imagens, a.externo_pesos)
    destila = cfg["tipo"] == "destila-rgbd"
    C = Cache(a.cache, D)
    C.abre(profundidade=destila, ext=D.ext is not None and os.path.exists(os.path.join(C.pasta_ext(), "pronto.json")))
    assert not cfg.get("externo") or C.ext_rgb is not None, "configuração com dados externos sem o cache das fotos externas"
    prazo = a.prazo or float("inf")
    rot = f"[{cfg['nome']} | {parte}]"
    if cfg.get("v3"):
        return roda_parte3(a, cfg, parte, cfgs, D, C, dev, prazo, rot, t0)
    tr_tudo = np.where(D.usa_tr)[0]
    nan = lambda n: np.full((n, D.N), np.nan, np.float32)
    saida = dict(cfg=json.dumps(cfg), parte=parte, dispositivo=str(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))
    if cfg["tipo"] == "ridge":
        X = feats_congeladas(cfg["bb"], D, C, dev, cfg["bs"], prazo).astype(np.float32)
        Xtr = X[[C.pos[D.ids[i]] for i in tr_tudo]]
        oof = nan(len(D.ids))
        oof[tr_tudo], alfa, mse = ridge_cv(Xtr, D.T[tr_tudo], D.fold[tr_tudo])
        pred, m = ridge_final(Xtr, D.T[tr_tudo], alfa)
        saida.update(oof=oof, alfa=alfa, cv_mse=mse)
        for s in D.conj:
            p = nan(len(D.conj[s])); u = np.where(D.usa_av[s])[0]
            p[u] = pred(X[[C.pos[D.conj[s][i]] for i in u]])
            saida[s] = p
        hist = []
        # para servir: a cabeça do ridge (o backbone é o checkpoint público, descrito em servir__*.json)
        arq_c = arq_de(a.saida, cfg, "cabeca")
        np.savez(arq_c + ".tmp.npz", mu=m.mu, sd=m.sd, W=m.W, mt=m.mt, alfa=alfa, nome=cfg["nome"], listas_hash=D.hash)
        os.replace(arq_c + ".tmp.npz", arq_c)
        grava_servir(a, cfg, D, X.shape[-1], dict(
            tipo="ridge sobre o backbone congelado", arquivo=os.path.basename(arq_c), bytes=os.path.getsize(arq_c),
            formula="s = ((f − mu) / sd) @ W + mt, com as features f em float32 (cabeça fora do autocast); arrays mu, sd, W, mt no arquivo"))
        url = BACKBONES[cfg["bb"]].get("url")       # checkpoint grande que nenhuma outra configuração usa: libera o disco se ele estiver curto
        arq = os.path.join(PASTA_PESOS, os.path.basename(url or "-"))
        if url and os.path.exists(arq) and sum(BACKBONES[c["bb"]].get("url") == url for c in cfgs) == 1 \
                and shutil.disk_usage(PASTA_PESOS).free < 15 * 2 ** 30:
            os.remove(arq)
    else:
        k = None if parte == "final" else int(parte[4:])
        tr = tr_tudo if k is None else tr_tudo[D.fold[tr_tudo] != k]
        va = None if k is None else tr_tudo[D.fold[tr_tudo] == k]
        sem = zlib.crc32(f"{cfg['nome']}|{parte}".encode()) % 2 ** 31
        prof, hist_prof = None, None
        if destila:   # professor RGB-D treinado nos mesmos pratos (nunca vê o fold de fora); só o aluno RGB faz previsões
            prof, _, hist_prof, _ = treina(cfg, D, C, tr, va, dev, prazo, a.div_lote, canais=4, fator_orc=2.0, semente=sem + 1, rotulo=rot + " professor")
        rede, preve, hist, (tm, ts) = treina(cfg, D, C, tr, va, dev, prazo, a.div_lote, professor=prof, semente=sem, rotulo=rot)
        if hist_prof is not None:
            saida["hist_professor"] = json.dumps(hist_prof)
        if k is not None:
            oof = nan(len(D.ids)); oof[va] = preve([D.ids[i] for i in va])
            saida["oof"] = oof
        else:
            for s in D.conj:
                p = nan(len(D.conj[s])); u = np.where(D.usa_av[s])[0]
                p[u] = preve([D.conj[s][i] for i in u])
                saida[s] = p
            if a.guarda_pesos != "nenhum":   # pesos da média móvel em meia precisão (ficam fora do git); só tensores e tipos simples
                arq = arq_de(a.saida, cfg, "pesos")
                torch.save(dict(formato=FORMATO, nome=cfg["nome"], cfg=cfg, timm=BACKBONES[cfg["bb"]], tm=[float(v) for v in tm], ts=[float(v) for v in ts],
                                grupos=list(D.grupos), listas_hash=D.hash,
                                estado={n: (v.half() if v.is_floating_point() else v).cpu() for n, v in rede.state_dict().items()}), arq + ".tmp")
                os.replace(arq + ".tmp", arq)
                grava_servir(a, cfg, D, rede.dim, dict(
                    tipo="linear, treinada junto com o backbone", arquivo=os.path.basename(arq), bytes=os.path.getsize(arq),
                    formula="s = head(f) * ts + tm; o arquivo traz 'estado' (rede inteira com a média móvel dos pesos, meia precisão), 'tm' e 'ts'",
                    carregar="torch.load(arquivo, weights_only=True); cria_rede(bb, 22, pretrained=False).load_state_dict(estado em float32)"))
    for k in ["oof"] + list(D.conj):      # só previsões finitas viram arquivo (NaN é reservado a pratos fora da parte)
        if k in saida and not np.isfinite(saida[k][~np.isnan(saida[k][:, 0])]).all():
            raise RuntimeError(f"previsões não finitas em '{k}'")
    saida.update(hist=json.dumps(hist), seg=time.time() - t0)
    if dev.type == "cuda":
        saida["mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    dest = arq_parte(a.saida, cfg, parte)
    np.savez(dest + ".tmp.npz", **saida); os.replace(dest + ".tmp.npz", dest)
    print(f"{rot} pronto em {(time.time() - t0) / 60:.1f} min", flush=True)


def roda_parte3(a, cfg, parte, cfgs, D, C, dev, prazo, rot, t0):
    """Uma parte da rodada 3 (seção 12.3). Treina só com os pratos de treino_v2; as fotos do teste2 só passam pelo modelo
    final, para gerar previsões. Toda previsão sai com 1, 4 e 8 orientações (sufixos _o1, _o4 e nenhum)."""
    import torch
    tr = np.where(D.usa_tr)[0]
    n, N, K = len(D.ids), D.N, len(D.fontes)
    fold_tr = D.fold[tr]
    lin_tr = [C.pos[D.ids[i]] for i in tr]
    usa = {s: np.where(D.usa_av[s])[0] for s in D.conj}
    lin = {s: [C.pos[D.conj[s][i]] for i in usa[s]] for s in D.conj}
    nan = lambda *forma: np.full(forma, np.nan, np.float32)
    por_nome = {c["nome"]: c for c in cfgs}
    parte_de = lambda nome: np.load(arq_parte(a.saida, por_nome[nome], "tudo"), allow_pickle=False)
    saida = dict(cfg=json.dumps(cfg), parte=parte, dispositivo=str(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))
    hist = []

    def guarda(chave, idx, tam, P, F=None, dc=None):
        """Grava as variantes de orientações de P [m, 8, 22] (e das frações F [m, 8, K]) nas linhas idx de um array [tam, ...]."""
        for suf, k in ORIENT.items():
            saida.setdefault(chave + suf, nan(tam, N))[idx] = media_or(P, k)
            if F is not None:
                saida.setdefault(chave + "_ff" + suf, nan(tam, K))[idx] = media_or(F, k)
                saida.setdefault(chave + "_dens_ff" + suf, nan(tam))[idx] = media_or(F, k) @ dc

    def tempo():
        if time.time() > prazo:
            raise SemOrcamento()

    ext = D.ext if cfg.get("externo") else None
    if ext is not None:
        tre, cols_m = ext["treina"], [I_M] + list(range(D.I_MG.start, D.I_MG.stop))
        ids_ext = dict(ext_ids=np.array(ext["ids"]), ext_treina=tre)
    if cfg["tipo"] == "ridge" and ext is not None:
        # ridge com o treino externo (12.5): só os alvos de massa (total e por grupo) são refeitos, com as fotos externas
        # de treino em todo fold (peso total = TETO_EXTERNO × o do Nutrition5k) e sem o viés de domínio; os outros alvos
        # (sem rótulo na base externa) são os do ridge só com o treino_v2, copiados da configuração de origem.
        X = feats_congeladas(cfg["bb"], D, C, dev, cfg["bs"], prazo).astype(np.float32)
        Xe = feats_ext(cfg["bb"], D, C, dev, cfg["bs"], prazo).astype(np.float32)
        Xtr, Dm = X[lin_tr], X.shape[-1]
        pl = parte_de(cfg["plano"])
        for c in pl.files:
            if c.startswith(tuple(["oof"] + list(D.conj))):
                saida[c] = pl[c].copy()
        zc = np.load(arq_de(a.saida, por_nome[cfg["plano"]], "cabeca"), allow_pickle=False)
        A = zc["W"] / zc["sd"][:, None]; b = zc["mt"] - (zc["mu"] / zc["sd"]) @ zc["W"]          # a cabeça de origem como mapa afim
        ext_sem = (Xe.reshape(-1, Dm).astype(np.float64) @ A + b).reshape(len(Xe), 8, -1).mean(1)
        ext_com, delta, todos_f = ext_sem.copy(), np.zeros(N), frozenset(int(k) for k in set(fold_tr.tolist()))
        for cols, lam in (([I_M], float(ext["peso_alvo"][I_M])), (cols_m[1:], float(ext["peso_alvo"][D.I_MG.start]))):
            ok = tre & np.isfinite(ext["T"][:, cols]).all(1)
            if lam <= 0 or not ok.any():
                continue
            tempo()
            R = RidgeSubExt(Xtr, D.T[tr][:, cols], fold_tr, Xe[ok], ext["T"][ok][:, cols], lambda n_i, lam=lam, ne=int(ok.sum()): TETO_EXTERNO * n_i / ne * lam, dev)
            alfa = R.alfa_cv(todos_f)
            for k in sorted(todos_f):
                Pk = R.prev_x(todos_f - {k}, Xtr[fold_tr == k], alfa, por_or=True).cpu().numpy()
                for suf, kk in ORIENT.items():
                    saida["oof" + suf][np.ix_(tr[fold_tr == k], cols)] = media_or(Pk, kk)
            for sc in D.conj:
                Pv = R.prev_x(todos_f, X[lin[sc]], alfa, por_or=True).cpu().numpy()
                for suf, kk in ORIENT.items():
                    saida[sc + suf][np.ix_(usa[sc], cols)] = media_or(Pv, kk)
            an = ridge_aninhado_de(R)
            for k in range(an.shape[0]):
                saida["oof_aninhado"][k][np.ix_(tr, cols)] = an[k]
            A[:, cols], b[cols] = R.afim(todos_f, alfa)
            delta[cols] = R.delta(todos_f)
            ext_com[:, cols] = R.prev_x(todos_f, Xe, alfa).cpu().numpy() + delta[cols]      # nas fotos externas, o viés volta
        saida.update(ext=ext_com.astype(np.float32), ext_sem=ext_sem.astype(np.float32), ext_delta=delta.astype(np.float32), **ids_ext)
        for sc in D.conj:       # a cabeça gravada (mapa afim) tem de repetir as previsões gravadas
            if len(lin[sc]):
                ref = (X[lin[sc]].reshape(-1, Dm).astype(np.float64) @ A + b).reshape(len(lin[sc]), 8, -1).mean(1)
                dif = float(np.abs(ref - saida[sc][usa[sc]]).max())
                assert dif < 2e-3, f"a cabeça gravada não repete as previsões de {sc} (dif {dif:.2g})"
        arq_c = arq_de(a.saida, cfg, "cabeca")
        np.savez(arq_c + ".tmp.npz", mu=np.zeros(Dm, np.float32), sd=np.ones(Dm, np.float32), W=A, mt=b, delta_ext=delta, nome=cfg["nome"], listas_hash=D.hash)
        os.replace(arq_c + ".tmp.npz", arq_c)
        grava_servir(a, cfg, D, Dm, dict(
            tipo="ridge sobre o backbone congelado, com o treino externo nos alvos de massa", arquivo=os.path.basename(arq_c), bytes=os.path.getsize(arq_c),
            formula="s = f @ W + mt, com as features f em float32 (cabeça fora do autocast; mu = 0 e sd = 1 no arquivo). Em fotos do domínio externo, "
                    "soma-se delta_ext às saídas de massa; em fotos do Nutrition5k e do app, nada"))
    elif cfg["tipo"] == "ridge":
        X = feats_congeladas(cfg["bb"], D, C, dev, cfg["bs"], prazo).astype(np.float32)
        Xtr = X[lin_tr]
        P, alfa, mse = ridge_cv_or(Xtr, D.T[tr], fold_tr)
        guarda("oof", tr, n, P)
        _, m = ridge_final(Xtr, D.T[tr], alfa)
        por_or = lambda Xv: m.predict(Xv.reshape(-1, Xv.shape[-1])).reshape(Xv.shape[0], Xv.shape[1], -1)
        for s in D.conj:
            guarda(s, usa[s], len(D.conj[s]), por_or(X[lin[s]]))
        saida.update(alfa=alfa, cv_mse=mse)
        tempo()
        # alvos aninhados do professor: previsões de ridges treinados sem dois folds (ver ridge_aninhado)
        an = nan(5, n, N); an[:, tr] = ridge_aninhado(Xtr, D.T[tr], fold_tr, dev)
        saida["oof_aninhado"] = an
        # paridade com a execução 2: o mesmo ridge, na lista e nos folds antigos, sobre as features desta execução.
        # As previsões dele para o teste2 têm de repetir as da execução 2 (confere-se na ingestão, sem gabarito).
        ia = np.array([i for i in tr if D.ids[i] in D.antigo], int)
        if cfg.get("paridade_r2") and len(ia) and len({D.antigo[D.ids[i]] for i in ia}) > 1:
            Xa, fa = X[[C.pos[D.ids[i]] for i in ia]], np.array([D.antigo[D.ids[i]] for i in ia])
            oof_a, alfa_a, _ = ridge_cv(Xa, D.T[ia], fa)
            pred_a, _ = ridge_final(Xa, D.T[ia], alfa_a)
            for s in D.conj:
                r2 = nan(len(D.conj[s]), N); r2[usa[s]] = pred_a(X[lin[s]])
                saida["r2_" + s] = r2
            saida.update(r2_nome=cfg["paridade_r2"], r2_n=len(ia), r2_cv_rmse_logm=float(np.sqrt(np.mean((oof_a[:, I_M] - D.T[ia, I_M]) ** 2))))
        arq_c = arq_de(a.saida, cfg, "cabeca")
        np.savez(arq_c + ".tmp.npz", mu=m.mu, sd=m.sd, W=m.W, mt=m.mt, alfa=alfa, nome=cfg["nome"], listas_hash=D.hash)
        os.replace(arq_c + ".tmp.npz", arq_c)
        grava_servir(a, cfg, D, X.shape[-1], dict(
            tipo="ridge sobre o backbone congelado", arquivo=os.path.basename(arq_c), bytes=os.path.getsize(arq_c),
            formula="s = ((f − mu) / sd) @ W + mt, com as features f em float32 (cabeça fora do autocast); arrays mu, sd, W, mt no arquivo"))
    elif cfg["tipo"] in ("fracoes", "unico"):
        # cabeça de frações sobre features congeladas; no 'unico' (U1), também destilação com alvos aninhados do professor
        X = feats_congeladas(cfg["bb"], D, C, dev, 32, prazo).astype(np.float32)
        Xtr, Ttr, FFtr, mtr, ptr = X[lin_tr], D.T[tr], D.FF[tr], D.massa[tr], D.proteina[tr]
        prof = parte_de(cfg["professor"]) if cfg.get("professor") else None
        dens_fonte = np.zeros((6, K), np.float32)
        ed = None
        if ext is not None:       # fotos externas de treino (12.5): alvos mascarados, peso por alvo, viés de domínio retirado
            Xe = feats_ext(cfg["bb"], D, C, dev, 32, prazo).astype(np.float32)
            ed = dict(X=Xe[tre], T=ext["T"][tre], FF=ext["FF"][tre], peso_alvo=ext["peso_alvo"], peso_ff=ext["peso_ff"], cols_massa=cols_m)
        P, F, dens_fonte[:5] = cabeca_folds(Xtr, Ttr, FFtr, mtr, ptr, fold_tr, cfg, dev, None if prof is None else prof["alvos_aninhados"][:, tr], tempo, ed)
        for k in range(5):
            guarda("oof", tr[fold_tr == k], n, P[fold_tr == k], F[fold_tr == k], dens_fonte[k])
        tempo()
        dc, pesos, prev = _uma_cabeca(Xtr, Ttr, FFtr, mtr, ptr, np.ones(len(tr), bool), cfg, dev,
                                      None if prof is None else alvos_do_aluno(prof["alvos_oof"], None, None)[tr], ed)
        dens_fonte[5] = dc
        for s in D.conj:
            guarda(s, usa[s], len(D.conj[s]), *prev(X[lin[s]]), dc)
        saida.update(dens_fonte=dens_fonte, fontes=np.array(D.fontes))
        if cfg["tipo"] == "fracoes":     # densidade por frações aninhada, para os alvos do professor
            an = nan(5, n); an[:, tr] = fracoes_aninhado(Xtr, Ttr, FFtr, mtr, ptr, fold_tr, cfg, dev, tempo, ed)
            saida["aninhado_dens_ff"] = an
        if ext is not None and cfg["tipo"] == "unico":      # relato no teste externo: o U1 com e sem o treino externo, nas fotos externas
            saida.update(ext=(prev(Xe)[0].mean(1) + pesos["delta_ext"]).astype(np.float32), ext_delta=pesos["delta_ext"], **ids_ext,
                         ext_sem=aplica_mlp(np.load(arq_de(a.saida, por_nome[cfg["plano"]], "cabeca"), allow_pickle=False), Xe).astype(np.float32))
        arq_c = arq_de(a.saida, cfg, "cabeca")
        np.savez(arq_c + ".tmp.npz", nome=cfg["nome"], listas_hash=D.hash, fontes=np.array(D.fontes), **pesos)
        os.replace(arq_c + ".tmp.npz", arq_c)
        grava_servir(a, cfg, D, X.shape[-1], dict(
            tipo="MLP de uma camada oculta sobre o backbone congelado (22 saídas em log + frações por fonte)", arquivo=os.path.basename(arq_c),
            bytes=os.path.getsize(arq_c), fontes=list(D.fontes), dens_fonte_g_por_g=[float(v) for v in dc],
            formula="z = W2 @ gelu(W1 @ ((f − mu) / sd) + b1) + b2, com f em float32, por orientação; s = z[:22] * ts + tm; "
                    "frações = softmax(z[22:]); média das orientações; densidade pelas frações = frações @ dens_fonte"))
    elif cfg["tipo"] == "professor":
        # massa = média em log das massas ridge; densidade = média em log (piso 0,01) das densidades das cabeças de
        # frações e da cabeça de densidade de um ridge; proteína = massa × densidade. Só combina previsões já gravadas.
        R, Fz, Dz = [parte_de(x) for x in cfg["massa"]], [parte_de(x) for x in cfg["fracoes"]], parte_de(cfg["dens_ridge"])
        combina = combina_professor
        for chave in ["oof"] + list(D.conj):
            for suf in ORIENT:
                P = np.mean([r[chave + suf] for r in R], 0)       # colunas que o professor não define: média dos ridges
                P[:, I_M], P[:, I_D], P[:, I_P] = combina([r[chave + suf][:, I_M] for r in R],
                                                          [lg_dens(f[chave + "_dens_ff" + suf]) for f in Fz] + [Dz[chave + suf][:, I_D]])
                saida[chave + suf] = P.astype(np.float32)
        saida["alvos_oof"] = np.stack([saida["oof"][:, c] for c in (I_M, I_D, I_P)], 1)
        saida["alvos_aninhados"] = np.stack([np.stack(combina([r["oof_aninhado"][k][:, I_M] for r in R],
                                                               [lg_dens(f["aninhado_dens_ff"][k]) for f in Fz] + [Dz["oof_aninhado"][k][:, I_D]]), 1)
                                             for k in range(5)]).astype(np.float32)
        saida["alvos_nomes"] = np.array(["log_massa", "log_densidade_piso_0.01", "log1p_proteina"])
        if all("ext" in r for r in R):      # massa do professor nas fotos externas, com e sem o treino externo (só a coluna de massa interessa)
            saida.update(ext=np.mean([r["ext"] for r in R], 0), ext_sem=np.mean([r["ext_sem"] for r in R], 0), ext_ids=R[0]["ext_ids"], ext_treina=R[0]["ext_treina"])
        for k in range(5):      # o aluno que fica sem o fold k só usa alvos de pratos dos outros folds
            assert np.isnan(saida["alvos_aninhados"][k][tr[fold_tr == k]]).all() and not np.isnan(saida["alvos_aninhados"][k][tr[fold_tr != k]]).any()
    else:       # ajuste fino (U0, U3, U4)
        k = None if parte == "final" else int(parte[4:])
        trk = tr if k is None else tr[fold_tr != k]
        va = None if k is None else tr[fold_tr == k]
        sem = zlib.crc32(f"{cfg['nome']}|{parte}".encode()) % 2 ** 31
        alvos = None
        if cfg.get("professor"):    # alvos aninhados: para o aluno sem o fold k, o professor também nunca viu o fold k
            zp = parte_de(cfg["professor"])
            alvos = alvos_do_aluno(zp["alvos_oof"], zp["alvos_aninhados"], k)
        rede, preve, hist, (tm, ts) = treina(cfg, D, C, trk, va, dev, prazo, a.div_lote, semente=sem, rotulo=rot, alvos_prof=alvos)
        dc = preve.dens_fonte
        if k is None and C.ext_rgb is not None:      # relato no teste externo: o ajuste fino como foi treinado, nas fotos externas
            de = preve.delta_ext if preve.delta_ext is not None else np.zeros(N)
            i_tr = np.where(D.ext["treina"])[0]         # toda a parte segurada + ~400 fotos do treino externo (para a constante do relato)
            sel = np.sort(np.r_[np.where(~D.ext["treina"])[0], i_tr[::max(1, len(i_tr) // 400)]])
            pe = nan(len(D.ext["ids"]), N); pe[sel] = preve([D.ext["ids"][i] for i in sel]) + de
            saida.update(ext=pe, ext_delta=de.astype(np.float32), ext_ids=np.array(D.ext["ids"]), ext_treina=D.ext["treina"])
        if dc is not None:
            saida.update(dens_fonte=dc.astype(np.float32), fontes=np.array(D.fontes))
        if k is not None:
            P, F = preve([D.ids[i] for i in va], variantes=True)
            for suf in ORIENT:
                saida.setdefault("oof" + suf, nan(n, N))[va] = P[suf]
                if F is not None:
                    saida.setdefault("oof_ff" + suf, nan(n, K))[va] = F[suf]
                    saida.setdefault("oof_dens_ff" + suf, nan(n))[va] = F[suf] @ dc
        else:
            for s in D.conj:
                P, F = preve([D.conj[s][i] for i in usa[s]], variantes=True)
                for suf in ORIENT:
                    saida.setdefault(s + suf, nan(len(D.conj[s]), N))[usa[s]] = P[suf]
                    if F is not None:
                        saida.setdefault(s + "_ff" + suf, nan(len(D.conj[s]), K))[usa[s]] = F[suf]
                        saida.setdefault(s + "_dens_ff" + suf, nan(len(D.conj[s])))[usa[s]] = F[suf] @ dc
            if a.guarda_pesos != "nenhum":
                arq = arq_de(a.saida, cfg, "pesos")
                torch.save(dict(formato=FORMATO, nome=cfg["nome"], cfg={c: v for c, v in cfg.items() if c != "depende"}, timm=BACKBONES[cfg["bb"]],
                                tm=[float(v) for v in tm], ts=[float(v) for v in ts], grupos=list(D.grupos), listas_hash=D.hash, n_saidas=N + (K if dc is not None else 0),
                                fontes=list(D.fontes) if dc is not None else [], dens_fonte=[float(v) for v in dc] if dc is not None else [],
                                estado={nm: (v.half() if v.is_floating_point() else v).cpu() for nm, v in rede.state_dict().items()}), arq + ".tmp")
                os.replace(arq + ".tmp", arq)
                grava_servir(a, cfg, D, rede.dim, dict(
                    tipo="linear, treinada junto com o backbone" + (" (22 saídas em log + 10 logits de fração por fonte)" if dc is not None else ""),
                    arquivo=os.path.basename(arq), bytes=os.path.getsize(arq), fontes=list(D.fontes) if dc is not None else None,
                    dens_fonte_g_por_g=[float(v) for v in dc] if dc is not None else None,
                    formula="z = head(f); s = z[:22] * ts + tm" + ("; frações = softmax(z[22:]) por orientação; densidade pelas frações = frações @ dens_fonte" if dc is not None else "")
                            + "; o arquivo traz 'estado' (rede inteira com a média móvel dos pesos, meia precisão), 'tm', 'ts', 'n_saidas' e 'dens_fonte'",
                    carregar="torch.load(arquivo, weights_only=True); cria_rede(bb, n_saidas, pretrained=False).load_state_dict(estado em float32)"))
    # NaN só é aceito em pratos fora da parte (outros folds, subconjunto do --smoke) e nos alvos aninhados do próprio fold
    for c, v in saida.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not np.isfinite(v[~np.isnan(v)]).all():
            raise RuntimeError(f"valores não finitos em '{c}'")
    saida.update(hist=json.dumps(hist), seg=time.time() - t0)
    if cfg.get("externo"):
        saida["ext_lista"] = D.ext["hash"]
    if dev.type == "cuda":
        saida["mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    dest = arq_parte(a.saida, cfg, parte)
    np.savez(dest + ".tmp.npz", **saida); os.replace(dest + ".tmp.npz", dest)
    print(f"{rot} pronto em {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ fase de serviço (processo filho, um modelo por vez)
def mede_servico(a, cfg):
    """Remonta o modelo SÓ com os arquivos de servir, confere que ele reproduz as previsões gravadas e mede o custo:
    parâmetros, tamanho dos pesos e latência por imagem (GPU em fp16, lotes de 1 e de 8; 1, 4 e 8 orientações; CPU em
    float32 com o número de threads registrado). O tempo inclui redimensionar e normalizar, a partir de uint8 480×640
    já no dispositivo; não inclui decodificar o PNG nem a cópia para a GPU. Grava servico__<config>.json.
    Três tipos de cabeça: ridge (linear, float32), MLP de frações sobre o backbone congelado (float32) e a camada
    linear do ajuste fino (22 saídas, ou 22 + 10 logits de fração)."""
    global PASTA_PESOS
    import torch
    F = torch.nn.functional
    t0 = time.time()
    PASTA_PESOS = os.path.join(a.cache, "pesos")
    os.environ.setdefault("HF_HOME", os.path.join(a.cache, "hf"))
    n_thr = max(1, int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_threads(n_thr)
    dev = torch.device(a.dispositivo)
    D = Dados(a.listas, a.smoke, a.rodada, a.externo_lista, a.externo_imagens, a.externo_pesos)
    C = Cache(a.cache, D).abre()
    prazo = a.prazo or float("inf")
    hw, N = BACKBONES[cfg["bb"]]["hw"], D.N
    quad = hw[0] == hw[1]
    congelado = cfg["tipo"] in ("ridge", "fracoes", "unico")       # backbone público + cabeça gravada, em float32 fora do autocast
    spec = json.load(open(arq_de(a.saida, cfg, "servir")))
    K = 0
    if congelado:
        rede = cria_rede(cfg["bb"], N)
        z = np.load(arq_de(a.saida, cfg, "cabeca"), allow_pickle=False)
        if cfg["tipo"] == "ridge":
            with torch.no_grad():
                rede.head.weight.copy_(torch.from_numpy((z["W"] / z["sd"][:, None]).T.astype(np.float32)))
                rede.head.bias.copy_(torch.from_numpy((z["mt"] - (z["mu"] / z["sd"]) @ z["W"]).astype(np.float32)))
            tm, ts = np.zeros(N, np.float32), np.ones(N, np.float32)
        else:
            class MLP(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("mu", torch.from_numpy(z["mu"])); self.register_buffer("sd", torch.from_numpy(z["sd"]))
                    self.l1 = torch.nn.Linear(z["W1"].shape[1], z["W1"].shape[0]); self.l2 = torch.nn.Linear(z["W2"].shape[1], z["W2"].shape[0])
                    with torch.no_grad():
                        self.l1.weight.copy_(torch.from_numpy(z["W1"])); self.l1.bias.copy_(torch.from_numpy(z["b1"]))
                        self.l2.weight.copy_(torch.from_numpy(z["W2"])); self.l2.bias.copy_(torch.from_numpy(z["b2"]))

                def forward(self, f):
                    return self.l2(F.gelu(self.l1((f - self.mu) / self.sd)))
            rede.head = MLP()
            tm, ts, K = z["tm"], z["ts"], z["W2"].shape[0] - N
        arquivos = [dict(arquivo=spec["backbone"]["arquivo"], bytes=spec["backbone"]["bytes"], publico=True, url=spec["backbone"]["url"],
                         hf_id=spec["backbone"]["hf_id"], fonte=spec["backbone"]["fonte"], sha256=spec["backbone"]["sha256"]),
                    dict(arquivo=os.path.basename(arq_de(a.saida, cfg, "cabeca")), bytes=os.path.getsize(arq_de(a.saida, cfg, "cabeca")), publico=False)]
    else:          # ajuste fino: só o arquivo de pesos gravado (sem baixar nada)
        ck = torch.load(arq_de(a.saida, cfg, "pesos"), map_location="cpu", weights_only=True)
        K = ck.get("n_saidas", N) - N
        rede = cria_rede(cfg["bb"], N + K, pretrained=False)
        rede.load_state_dict({n: (v.float() if v.is_floating_point() else v) for n, v in ck["estado"].items()})
        tm, ts = np.array(ck["tm"], np.float32), np.array(ck["ts"], np.float32)
        arquivos = [dict(arquivo=os.path.basename(arq_de(a.saida, cfg, "pesos")), bytes=os.path.getsize(arq_de(a.saida, cfg, "pesos")), publico=False)]
    rede.eval()
    n_par, n_cab = sum(p.numel() for p in rede.parameters()), sum(p.numel() for p in rede.head.parameters())

    def infere(d, xu8, n_or, agrupado=False):
        """uint8 [B, 480, 640, 3] no dispositivo -> (22 saídas em log por imagem, frações por fonte ou None), média das
        n_or primeiras orientações. agrupado: as orientações num lote só (com entrada não quadrada, as 8 em dois de 4)."""
        x = F.interpolate(xu8.permute(0, 3, 1, 2).float().div(255), size=hw, mode="bicubic", antialias=True, align_corners=False).clamp_(0, 1)
        grupos = [[k] for k in range(n_or)] if not agrupado else ([list(range(n_or))] if quad or n_or <= 4 else [[0, 1, 2, 3], [4, 5, 6, 7]])
        st, sf = 0, 0
        for g in grupos:
            xb = torch.cat([orienta(x, k) for k in g])
            with autocast(d):
                f = rede.feats(xb)
                o = None if congelado else rede.head(f)
            if congelado:      # cabeça em float32, fora do autocast (como as previsões gravadas)
                o = rede.head(f.float())
            o = o.float().reshape(len(g), len(x), -1)
            st = st + o[..., :N].sum(0)
            if K:
                sf = sf + torch.softmax(o[..., N:], -1).sum(0)
        return st / n_or * torch.from_numpy(ts).to(d) + torch.from_numpy(tm).to(d), (sf / n_or if K else None)

    def cronometra(d, xu8, n_or, agrupado, reps, aquece):
        sinc = torch.cuda.synchronize if d.type == "cuda" else (lambda: None)
        with torch.inference_mode():
            for _ in range(aquece):
                infere(d, xu8, n_or, agrupado)
            if d.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            tempos = []
            for _ in range(reps):
                sinc(); t = time.perf_counter(); infere(d, xu8, n_or, agrupado); sinc()
                tempos.append(time.perf_counter() - t)
        r = dict(lote=len(xu8), orientacoes=n_or, orientacoes_em_lote=bool(agrupado), repeticoes=reps,
                 ms_por_imagem=round(1000 * float(np.median(tempos)) / len(xu8), 2), ms_por_imagem_min=round(1000 * min(tempos) / len(xu8), 2),
                 ms_por_imagem_max=round(1000 * max(tempos) / len(xu8), 2))
        if d.type == "cuda":
            r["mem_pico_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        return r

    u8 = lambda ids, d: torch.from_numpy(np.ascontiguousarray(C.rgb[[C.pos[i] for i in ids]])).to(d)
    rede.to(dev)
    # 1. o modelo remontado reproduz as previsões gravadas? (até 8 pratos de um conjunto de previsão, 8 orientações; sem gabarito)
    parte = np.load(arq_parte(a.saida, cfg, "tudo" if congelado else "final"), allow_pickle=False)
    conj = "calib" if "calib" in D.conj else next(iter(D.conj))
    u = np.where(D.usa_av[conj])[0][:8]
    with torch.inference_mode():
        p, fr = infere(dev, u8([D.conj[conj][i] for i in u], dev), 8)
    dif = np.abs(p.cpu().numpy() - parte[conj][u])
    conf = dict(conjunto=conj, pratos=int(len(u)), max_dif_log=float(dif.max()), max_dif_log_massa=float(dif[:, I_M].max()), max_dif_log1p_proteina=float(dif[:, I_P].max()),
                dispositivo=dev.type, nota="diferença máxima, em log, entre o modelo remontado com os arquivos de servir e as previsões gravadas")
    if K:
        conf["max_dif_fracao"] = float(np.abs(fr.cpu().numpy() - parte[conj + "_ff"][u]).max())
    res = dict(formato=FORMATO, nome=cfg["nome"], tipo=cfg["tipo"], backbone=cfg["bb"], timm=BACKBONES[cfg["bb"]]["timm"], hw=list(hw),
               parametros=dict(total=int(n_par), backbone=int(n_par - n_cab), cabeca=int(n_cab)), mb_fp16_em_memoria=round(n_par * 2 / 1e6, 1),
               arquivos=arquivos, bytes_pesos=int(sum(x["bytes"] or 0 for x in arquivos)), bytes_gravados_na_saida=int(sum(x["bytes"] or 0 for x in arquivos if not x["publico"])),
               conferencia=conf,
               inclui="redimensionar (bicúbico) + normalizar + rede + cabeça, a partir de uint8 480×640 já no dispositivo; não inclui decodificar o PNG nem a cópia para a GPU",
               gpu=None, cpu=None)
    print(f"[{cfg['nome']} | servico] {n_par / 1e6:.1f} M parâmetros; remontado × gravado ({conj}): diferença máxima {dif.max():.4f} em log"
          + (f", {conf['max_dif_fracao']:.4f} nas frações" if K else ""), flush=True)
    ids8 = D.todos[:8]
    # 2. GPU (uma só, fp16)
    if dev.type == "cuda":
        x8 = u8(ids8, dev)
        med = {}
        for nome, (lote, n_or, agr, reps, aq) in dict(lote1_o1=(1, 1, False, 30, 5), lote1_o4=(1, 4, False, 10, 2), lote1_o8=(1, 8, False, 10, 2),
                                                       lote1_o4_em_lote=(1, 4, True, 10, 2), lote1_o8_em_lote=(1, 8, True, 10, 2),
                                                       lote8_o1=(8, 1, False, 10, 2), lote8_o8=(8, 8, False, 5, 1)).items():
            if time.time() + 60 > prazo:
                raise SemOrcamento()
            med[nome] = cronometra(dev, x8[:lote], n_or, agr, reps, aq)
        res["gpu"] = dict(nome=torch.cuda.get_device_name(0), precisao="fp16 (torch.autocast)", medidas=med)
        print(f"[{cfg['nome']} | servico] GPU ms por imagem: " + ", ".join(f"{k} {v['ms_por_imagem']}" for k, v in med.items()), flush=True)
    # 3. CPU (float32), poucas imagens
    rede.float().cpu()
    cpu = torch.device("cpu")
    x8 = u8(ids8, cpu)
    try:
        modelo_cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        modelo_cpu = None
    med = dict(lote1_o1=cronometra(cpu, x8[:1], 1, False, 2 if a.smoke else 3, 1))
    rapido = med["lote1_o1"]["ms_por_imagem"] < 500
    for nome, n_or in (("lote1_o4", 4), ("lote1_o8", 8)):
        if time.time() + 1.3 * n_or * med["lote1_o1"]["ms_por_imagem"] / 1000 + 30 < prazo:
            med[nome] = cronometra(cpu, x8[1:2], n_or, False, 3 if rapido and not a.smoke else 1, 0)
    res["cpu"] = dict(threads=n_thr, nucleos=os.cpu_count(), modelo=modelo_cpu, precisao="float32", medidas=med)
    print(f"[{cfg['nome']} | servico] CPU ({n_thr} threads) ms por imagem: " + ", ".join(f"{k} {v['ms_por_imagem']}" for k, v in med.items()), flush=True)
    res["seg"] = round(time.time() - t0, 1)
    grava_json(arq_de(a.saida, cfg, "servico"), res)


def principal_parte(a, cfgs):
    nome, parte = a.job.rsplit(":", 1)
    cfg = next(c for c in cfgs if slug(c["nome"]) == nome)
    simula = os.environ.get("NUTRELLE_SIMULA", "")     # só para testar o orquestrador: "memoria" ou "erro"
    if simula == "memoria" and a.div_lote == 1 and parte in ("fold0", "tudo"):
        sys.exit(SAI_MEMORIA)
    if simula == "erro" and parte == "fold1":
        sys.exit(1)
    try:
        if parte == "servico":
            mede_servico(a, cfg)
        else:
            roda_parte(a, cfg, parte, cfgs)
    except SemOrcamento:
        sys.exit(SAI_SEM_ORCAMENTO)
    except Exception as e:
        traceback.print_exc()
        sys.exit(SAI_MEMORIA if "out of memory" in str(e).lower() and parte != "servico" else 1)


# ------------------------------------------------------------------ orquestrador
_trava_log = threading.Lock()
_arq_log = None


def diz(*partes):
    s = time.strftime("%H:%M:%S ") + " ".join(str(p) for p in partes)
    with _trava_log:
        print(s, flush=True)
        if _arq_log:
            _arq_log.write(s + "\n"); _arq_log.flush()


def grava_json(arq, obj):
    with open(arq + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
    os.replace(arq + ".tmp", arq)


def garante_timm(cfgs):
    """Kaggle/Colab trazem timm; as etiquetas '.v2_webli' (SigLIP 2) pedem timm >= 1.0.15. Atualiza se faltar."""
    def faltam():
        import timm
        tem = set(timm.list_pretrained())
        return sorted({BACKBONES[c["bb"]]["timm"] for c in cfgs if c["bb"]} - tem), timm.__version__
    try:
        f, v = faltam()
    except ImportError:
        f, v = ["timm"], None
    if f:
        diz(f"timm {v} não conhece {f}: atualizando com pip")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm"], check=False)
        r = subprocess.run([sys.executable, "-c", "import timm,sys; t=set(timm.list_pretrained()); sys.exit(any(x not in t for x in sys.argv[1:]))"] + f)
        if r.returncode:
            raise SystemExit(f"Mesmo depois de atualizar, o timm não tem os pesos {f}.")


def retoma(a):
    """Copia partes prontas de execuções anteriores (mesmas listas) para a pasta de saída. Partes de fold da execução 1
    são aceitas como estão; partes final/tudo sem teste2 ou sem os arquivos de servir são refeitas (ver parte_pronta)."""
    fontes = list(a.retoma)
    if os.path.isdir("/kaggle/input"):
        fontes += [os.path.dirname(m) for m in glob.glob("/kaggle/input/**/manifesto.json", recursive=True)]
    n = 0
    for pasta in fontes:
        try:
            man = json.load(open(os.path.join(pasta, "manifesto.json")))
        except Exception:
            continue
        if man.get("formato") not in FORMATOS_ACEITOS or man.get("listas_hash") != a._hash or bool(man.get("smoke")) != bool(a.smoke):
            diz(f"retomada: {pasta} ignorada (outro formato, outras listas ou smoke)")
            continue
        for padrao in ("parte__*.npz", "pesos__*.pt", "cabeca__*.npz", "servir__*.json", "servico__*.json", "decisao_*.json"):
            for arq in glob.glob(os.path.join(pasta, padrao)):
                dest = os.path.join(a.saida, os.path.basename(arq))
                mesma_ext = bool(a.externo) and (man.get("externo") or {}).get("lista") == a._hash_ext
                if not mesma_ext and any(slug(f" +{x}") in os.path.basename(arq) for x in EXTERNOS):
                    continue        # treinado com a base externa de outra lista (ou esta execução não usa a base)
                if padrao.startswith("decisao") and (json.load(open(arq)).get("motivo") or man.get("externo") is None or not a.externo
                                                     or man["externo"].get("lista") != a._hash_ext):
                    continue        # só vale de novo a decisão tomada pela regra, com a mesma lista externa
                if not os.path.exists(dest):
                    shutil.copy(arq, dest); n += 1
    if n:
        diz(f"retomada: {n} arquivos copiados de {fontes}")


def junta(a, cfg, D):
    """Reúne as partes de uma configuração completa: pred__*.npz (esquema do banco + teste2), pred__*.csv e log__*.json."""
    s = slug(cfg["nome"])
    partes = {p: dict(np.load(arq_parte(a.saida, cfg, p), allow_pickle=False)) for p in partes_de(cfg)}
    if cfg["tipo"] == "ridge":
        oof, av = partes["tudo"]["oof"], partes["tudo"]
    else:
        oof = np.full((len(D.ids), D.N), np.nan, np.float32)
        for k in range(5):
            m = (D.fold == k) & D.usa_tr
            oof[m] = partes[f"fold{k}"]["oof"][m]
        av = partes["final"]
    assert not np.isnan(oof[D.usa_tr]).any(), "previsão fora-do-fold faltando"
    for sp in D.conj:
        assert av[sp].shape == (len(D.conj[sp]), D.N) and not np.isnan(av[sp][D.usa_av[sp]]).any(), f"previsão de {sp} faltando"
    pred = dict(oof=oof, **{sp: av[sp] for sp in D.conj})
    np.savez(os.path.join(a.saida, f"pred__{s}.tmp.npz"), nome=cfg["nome"], listas_hash=D.hash, listas2_hash=D.hash2 or "", fold=D.fold,
             ids_treino=np.array(D.ids), **{f"ids_{sp}": np.array(D.conj[sp]) for sp in D.conj}, **pred)
    os.replace(os.path.join(a.saida, f"pred__{s}.tmp.npz"), os.path.join(a.saida, f"pred__{s}.npz"))
    with open(os.path.join(a.saida, f"pred__{s}.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["conjunto", "dish_id", "fold"] + nomes_saidas(D))
        for conj, ids, M, folds in [("treino_oof", D.ids, oof, D.fold)] + [(sp, D.conj[sp], av[sp], None) for sp in D.conj]:
            for i, d in enumerate(ids):
                if not np.isnan(M[i, 0]):
                    wr.writerow([conj, d, "" if folds is None else int(folds[i])] + [f"{v:.5f}" for v in M[i]])
    cv = resumo_cv(D, oof)
    extra = {"hist_professor": {p: json.loads(str(v["hist_professor"])) for p, v in partes.items() if "hist_professor" in v}}
    grava_json(os.path.join(a.saida, f"log__{s}.json"),
               dict(nome=cfg["nome"], cfg=cfg, cv_treino=cv, seg_total=float(sum(float(v["seg"]) for v in partes.values())),
                    partes={p: dict(seg=float(v["seg"]), dispositivo=str(v["dispositivo"]), mem_gb=float(v["mem_gb"]) if "mem_gb" in v else None,
                                    hist=json.loads(str(v["hist"]))) for p, v in partes.items()}, **(extra if extra["hist_professor"] else {})))
    return cv


def junta3(a, cfg, D):
    """Rodada 3: reúne as partes numa pred__*.npz com tudo o que a regra de 12.3 precisa: oof e teste2 com 1, 4 e 8
    orientações (sufixos _o1, _o4 e nenhum), frações (_ff) e densidade pelas frações (_dens_ff) quando há cabeça de
    frações, dens_fonte [6 × 10] (folds 0–4 e modelo final), e o que mais a parte gravou (alvos aninhados, réplica de
    paridade). Também pred__*.csv (8 orientações) e log__*.json."""
    s = slug(cfg["nome"])
    partes = {p: dict(np.load(arq_parte(a.saida, cfg, p), allow_pickle=False)) for p in partes_de(cfg)}
    fora = ("cfg", "parte", "dispositivo", "hist", "seg", "mem_gb")
    marca = {"ext_lista": D.ext["hash"]} if cfg.get("externo") else {}
    if "tudo" in partes:
        pred = {c: v for c, v in partes["tudo"].items() if c not in fora}
    else:
        pred, fin = {}, partes["final"]
        for c in [c for c in partes["fold0"] if c.startswith("oof")]:
            arr = np.full_like(partes["fold0"][c], np.nan)
            for k in range(5):
                m = (D.fold == k) & D.usa_tr
                arr[m] = partes[f"fold{k}"][c][m]
            pred[c] = arr
        pred.update({c: v for c, v in fin.items() if any(c.startswith(sp) for sp in list(D.conj) + ["ext"])})
        if "dens_fonte" in fin:
            pred.update(dens_fonte=np.stack([partes[f"fold{k}"]["dens_fonte"] for k in range(5)] + [fin["dens_fonte"]]), fontes=fin["fontes"])
    principais = [c for c in pred if (c.startswith("oof") and "aninhado" not in c) or any(c.startswith(sp) for sp in D.conj)]
    for c in principais:      # nenhuma previsão faltando nos pratos usados
        usados = D.usa_tr if c.startswith("oof") else D.usa_av[next(sp for sp in D.conj if c.startswith(sp))]
        assert not np.isnan(pred[c][usados]).any(), f"previsão faltando em '{c}'"
    for suf in ORIENT:
        assert "oof" + suf in pred and all(sp + suf in pred for sp in D.conj), f"variante '{suf}' faltando"
    pred.update(marca)
    np.savez(os.path.join(a.saida, f"pred__{s}.tmp.npz"), nome=cfg["nome"], listas_hash=D.hash, listas2_hash=D.hash2 or "", fold=D.fold,
             ids_treino=np.array(D.ids), **{f"ids_{sp}": np.array(D.conj[sp]) for sp in D.conj}, **pred)
    os.replace(os.path.join(a.saida, f"pred__{s}.tmp.npz"), os.path.join(a.saida, f"pred__{s}.npz"))
    with open(os.path.join(a.saida, f"pred__{s}.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["conjunto", "dish_id", "fold"] + nomes_saidas(D))
        for conj, ids, M, folds in [("treino_oof", D.ids, pred["oof"], D.fold)] + [(sp, D.conj[sp], pred[sp], None) for sp in D.conj]:
            for i, d in enumerate(ids):
                if not np.isnan(M[i, 0]):
                    wr.writerow([conj, d, "" if folds is None else int(folds[i])] + [f"{v:.5f}" for v in M[i]])
    cv = resumo_cv(D, pred["oof"])
    cv["rotas"] = {f"o{k}": rotas_cv(D, pred, suf) for suf, k in ORIENT.items()}
    if "r2_cv_rmse_logm" in pred:
        cv["replica_execucao2_rmse_logm"] = float(pred["r2_cv_rmse_logm"])
    grava_json(os.path.join(a.saida, f"log__{s}.json"),
               dict(nome=cfg["nome"], cfg=cfg, cv_treino=cv, seg_total=float(sum(float(v["seg"]) for v in partes.values())),
                    partes={p: dict(seg=float(v["seg"]), dispositivo=str(v["dispositivo"]), mem_gb=float(v["mem_gb"]) if "mem_gb" in v else None,
                                    hist=json.loads(str(v["hist"]))) for p, v in partes.items()}))
    return cv


def monta_servico(a, cfgs, pipelines):
    """servico.json: uma entrada por modelo medido + totais de cada pipeline (soma das latências por imagem, um modelo
    depois do outro numa só GPU; soma de parâmetros e de bytes de pesos; maior pico de memória de GPU)."""
    modelos = {}
    for c in cfgs:
        if os.path.exists(arq_de(a.saida, c, "servico")):
            modelos[c["nome"]] = json.load(open(arq_de(a.saida, c, "servico")))

    def total(nomes):
        tem, falta = [n for n in nomes if n in modelos], [n for n in nomes if n not in modelos]
        soma = lambda disp: {k: round(sum(modelos[n][disp]["medidas"][k]["ms_por_imagem"] for n in tem), 1)
                             for k in (modelos[tem[0]][disp] or {}).get("medidas", {})
                             if all(modelos[n][disp] and k in modelos[n][disp]["medidas"] for n in tem)} if tem else {}
        picos = [v["mem_pico_mb"] for n in tem if modelos[n]["gpu"] for v in modelos[n]["gpu"]["medidas"].values()]
        return dict(modelos=list(nomes), completo=not falta, faltam=falta,
                    parametros=int(sum(modelos[n]["parametros"]["total"] for n in tem)),
                    bytes_pesos=int(sum(modelos[n]["bytes_pesos"] for n in tem)),
                    bytes_gravados_na_saida=int(sum(modelos[n]["bytes_gravados_na_saida"] for n in tem)),
                    mb_fp16_em_memoria=round(sum(modelos[n]["mb_fp16_em_memoria"] for n in tem), 1),
                    gpu_ms_por_imagem=soma("gpu"), gpu_mem_pico_mb_maior_modelo=max(picos) if picos else None, cpu_ms_por_imagem=soma("cpu"))

    um = next(iter(modelos.values()), {})
    obj = dict(formato=FORMATO, gerado=time.strftime("%Y-%m-%d %H:%M:%S"),
               condicoes=dict(gpu=(um.get("gpu") or {}).get("nome"), gpu_precisao="fp16 (torch.autocast)", cpu=um.get("cpu"),
                              inclui=um.get("inclui"), medido="um modelo por vez, com a máquina sem outro treino em andamento",
                              chaves="loteB_oK = lote de B imagens, K orientações em sequência; lote1_o8_em_lote = as 8 orientações de uma imagem "
                                     "num lote só (entrada quadrada) ou em dois de 4 (paisagem e retrato); ms_por_imagem = mediana do tempo do lote ÷ B"),
               pipelines={k: total(v) for k, v in pipelines.items()},
               nota_pipelines=NOTA_PIPELINES if pipelines is PIPELINES else "total de um pipeline = soma dos modelos, um depois do outro; "
               "professor_5_backbones: as três cabeças de frações do professor reusam as features de três desses backbones (MLPs pequenas, não medidas)",
               modelos=modelos)
    for v in obj["condicoes"]["cpu"] and [obj["condicoes"]["cpu"]] or []:
        v.pop("medidas", None)
    grava_json(os.path.join(a.saida, "servico.json"), obj)
    return obj


def grava_tamanhos(a):
    """tamanhos.json: bytes de cada arquivo da pasta de saída, por tipo e no total (limite do Kaggle: ~20 GB)."""
    arqs = {n: os.path.getsize(os.path.join(a.saida, n)) for n in sorted(os.listdir(a.saida))
            if os.path.isfile(os.path.join(a.saida, n)) and n != "tamanhos.json" and ".tmp" not in n}
    tipo = lambda n: n.split("__")[0] if "__" in n else "outros"
    por_tipo = {}
    for n, b in arqs.items():
        por_tipo[tipo(n)] = por_tipo.get(tipo(n), 0) + b
    obj = dict(total_bytes=int(sum(arqs.values())), total_gb=round(sum(arqs.values()) / 2 ** 30, 3), n_arquivos=len(arqs), limite_kaggle_gb=20,
               maior_arquivo=max(arqs.items(), key=lambda x: x[1]) if arqs else None,
               por_tipo_mb={k: round(v / 2 ** 20, 1) for k, v in sorted(por_tipo.items())}, arquivos=arqs)
    grava_json(os.path.join(a.saida, "tamanhos.json"), obj)
    return obj


def orquestra(a, cfgs):
    global _arq_log
    t_ini = time.time()
    os.makedirs(a.saida, exist_ok=True)
    try:
        os.makedirs(a.cache, exist_ok=True)
    except OSError:      # /kaggle/temp pode não existir ou não aceitar escrita
        a.cache = os.path.join(tempfile.gettempdir(), "nutrelle_cache"); os.makedirs(a.cache, exist_ok=True)
    for t in glob.glob(os.path.join(a.cache, "**", "*.trava"), recursive=True):   # nenhum processo filho vivo aqui
        os.remove(t)
    _arq_log = open(os.path.join(a.saida, "treino_gpu.log"), "a")
    prazo = t_ini + a.orcamento_h * 3600 - 300          # 5 min de reserva para fechar os arquivos
    D = Dados(a.listas, a.smoke, a.rodada, a.externo_lista, a.externo_imagens, a.externo_pesos)
    a._hash, a._hash_ext = D.hash, (D.ext or {}).get("hash")
    livre = lambda pasta: round(shutil.disk_usage(pasta).free / 2 ** 30, 1)
    diz(f"{FORMATO} | rodada {a.rodada}{' (smoke)' if a.smoke else ''}: {len(cfgs)} configurações | treino {int(D.usa_tr.sum())} pratos, avaliação "
        + ", ".join(f"{s} {int(D.usa_av[s].sum())}" for s in D.conj) + f" | listas {D.hash}, teste2 {D.hash2} | orçamento {a.orcamento_h} h"
        f" | saída {a.saida} ({livre(a.saida)} GB livres) | cache {a.cache} ({livre(a.cache)} GB livres) | pesos: {a.guarda_pesos}")
    garante_timm(cfgs)
    import torch
    n_gpu = torch.cuda.device_count()
    disp = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu and not a.cpu else ["cpu"] * a.workers
    try:      # nomes pelo nvidia-smi: perguntar ao torch abriria um contexto CUDA (memória de GPU) neste processo
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception:
        gpus = []
    gpus = gpus if len(gpus) == n_gpu else [f"cuda:{i}" for i in range(n_gpu)]
    try:
        import timm
        v_timm = timm.__version__
    except Exception:
        v_timm = None
    diz(f"torch {torch.__version__}, timm {v_timm} | dispositivos: {gpus or 'CPU'} -> {len(disp)} processo(s) em paralelo")
    if not n_gpu and not (a.smoke or a.cpu):
        raise SystemExit("Sem GPU. No Kaggle: Settings > Accelerator > GPU T4 x2. (Para testar em CPU: --smoke.)")
    retoma(a)
    estado = {slug(c["nome"]): {p: parte_inicial() for p in partes_de(c)} for c in cfgs}
    est_serv = {slug(c["nome"]): {"servico": parte_inicial()} for c in cfgs}      # fase de serviço: mesma máquina de estados
    trava = threading.Lock()
    if D.ext:
        diz(f"base externa {a.externo} (12.5): lista {D.ext['hash']}, {D.ext['resumo']}, parte segurada por {D.ext['regra_teste_ext']}, "
            f"pesos por alvo {EXTERNOS[a.externo]['pesos']}; só treina nas configurações '+{a.externo}'")
    info = dict(formato=FORMATO, rodada=a.rodada, smoke=bool(a.smoke), listas_hash=D.hash, listas2_hash=D.hash2,
                externo=dict(nome=a.externo, lista=D.ext["hash"], resumo=D.ext["resumo"], regra_teste_ext=D.ext["regra_teste_ext"],
                             pesos=EXTERNOS[a.externo]["pesos"], teto=TETO_EXTERNO, fracao_por_epoca_no_ajuste_fino=FRACAO_EXT_EPOCA,
                             origem=EXTERNOS[a.externo]["origem"], situacao="aguardando as fotos") if D.ext
                else (dict(situacao=a.externo_fora) if getattr(a, "externo_fora", None) else None),
                n_treino=int(D.usa_tr.sum()), conjuntos={s: int(D.usa_av[s].sum()) for s in D.conj}, inicio=time.strftime("%Y-%m-%d %H:%M:%S"),
                orcamento_h=a.orcamento_h, gpus=gpus, torch=torch.__version__, timm=v_timm, python=sys.version.split()[0], guarda_pesos=a.guarda_pesos,
                disco_livre_gb_inicio=dict(saida=livre(a.saida), cache=livre(a.cache)), argv=sys.argv[1:], configs={}, melhor_modelo=None)

    def evento(cfg, parte, **ev):     # único lugar em que o estado muda (sempre com a trava)
        s = slug(cfg["nome"])
        mapa = est_serv if parte == "servico" else estado
        mapa[s][parte] = transicao(mapa[s][parte], ev)
        with open(os.path.join(a.saida, "eventos.jsonl"), "a") as f:
            f.write(json.dumps(dict(t=round(time.time(), 1), config=cfg["nome"], parte=parte, **ev, estado=mapa[s][parte]["estado"]), ensure_ascii=False) + "\n")

    def manifesto():
        for c in cfgs:
            s = slug(c["nome"])
            info["configs"].setdefault(c["nome"], dict(slug=s, tipo=c["tipo"], bb=c["bb"], cv_treino=None, min=None))
            info["configs"][c["nome"]].update(estado=estado_config(estado[s]), partes=estado[s], servico=est_serv[s]["servico"])
        info["duracao_min"] = round((time.time() - t_ini) / 60, 1)
        grava_json(os.path.join(a.saida, "manifesto.json"), info)

    def fecha_config(c):
        """Configuração completa: junta as partes, calcula a CV no treino e, com --guarda-pesos melhor, mantém só os pesos do melhor modelo."""
        s = slug(c["nome"])
        if c["tipo"] in ("externo", "decisao"):
            return
        cv = junta3(a, c, D) if c.get("v3") else junta(a, c, D)
        ic = info["configs"][c["nome"]]
        ic["cv_treino"] = cv; ic["min"] = round(sum(p["seg"] or 0 for p in estado[s].values()) / 60, 1)
        diz(f"== {c['nome']}: CV no treino (n={cv['n']}) RMSE log-massa {cv['rmse_logm']:.4f} | MAPE massa {cv['mape_massa']:.1f}% | "
            f"critério ponderado direto {cv['crit_direto']:.1f}% m×d {cv['crit_mxd']:.1f}% geo {cv['crit_geo']:.1f}% | {ic['min']} min de GPU")
        if "rotas" in cv:
            diz(f"   rotas (critério de CV ponderado), 8 / 4 / 1 orientações: " + " | ".join(
                f"{r} {cv['rotas']['o8'][r]:.1f} / {cv['rotas']['o4'][r]:.1f} / {cv['rotas']['o1'][r]:.1f}" for r in cv["rotas"]["o8"])
                + (f" | réplica da execução 2: RMSE {cv['replica_execucao2_rmse_logm']:.4f}" if "replica_execucao2_rmse_logm" in cv else ""))
        nota = lambda n: max(info["configs"][n]["cv_treino"][r] for r in ("crit_direto", "crit_mxd", "crit_geo"))
        com_pesos = [x["nome"] for x in cfgs if info["configs"][x["nome"]]["cv_treino"] and os.path.exists(arq_de(a.saida, x, "pesos"))]
        if com_pesos:
            melhor = max(com_pesos, key=nota)
            info["melhor_modelo"] = dict(nome=melhor, arquivo=f"pesos__{slug(melhor)}.pt", criterio_cv=nota(melhor),
                                         regra="maior critério ponderado na CV do treino entre as rotas direto, m×d e geo")
            if a.guarda_pesos == "melhor":
                for n in com_pesos:
                    if n != melhor:
                        os.remove(os.path.join(a.saida, f"pesos__{slug(n)}.pt"))

    def lanca(c, p, dev, div_lote, threads):
        """Roda uma parte num processo filho e devolve (código de saída, últimas linhas, segundos)."""
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(threads))
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--job", f"{slug(c['nome'])}:{p}", "--prazo", str(prazo), "--div-lote", str(div_lote),
               "--saida", a.saida, "--cache", a.cache, "--listas", a.listas, "--guarda-pesos", a.guarda_pesos, "--rodada", a.rodada]
        if a.smoke:
            cmd += ["--smoke", str(a.smoke)]
        if a.externo:
            cmd += ["--externo", a.externo, "--externo-lista", a.externo_lista] + (["--externo-imagens", a.externo_imagens] if a.externo_imagens else [])
        if dev.startswith("cuda"):
            env["CUDA_VISIBLE_DEVICES"] = dev.split(":")[1]; cmd += ["--dispositivo", "cuda:0"]
        else:
            cmd += ["--dispositivo", "cpu"]
        t0 = time.time()
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        cauda = []
        for linha in pr.stdout:
            diz(f"[{dev}] {linha.rstrip()}"); cauda = (cauda + [linha.rstrip()])[-8:]
        return pr.wait(), cauda, time.time() - t0

    with trava:
        for c in cfgs:                      # partes já em disco e completas para esta execução (retomada) -> FEITA
            for p in partes_de(c):
                if parte_pronta(a, c, p, D):
                    evento(c, p, tipo="retomou", seg=None if c["tipo"] in ("externo", "decisao") else float(np.load(arq_parte(a.saida, c, p))["seg"]))
                elif os.path.exists(arq_parte(a.saida, c, p)):
                    diz(f"retomada: {c['nome']} | {p} existe, mas sem teste2, sem os arquivos de servir ou de outra lista externa: será refeita")
        manifesto()
        for c in cfgs:
            if estado_config(estado[slug(c["nome"])]) == "completa":
                fecha_config(c)
        manifesto()
        nomes, todas = {c["nome"] for c in cfgs}, {c["nome"]: c for c in getattr(a, "_todas", None) or todas_as_configs()}
        for c in cfgs:      # dependência fora desta execução (--so): tem de estar pronta em disco
            for d in c.get("depende", []):
                if d not in nomes and not (d in todas and all(parte_pronta(a, todas[d], q, D) for q in partes_de(todas[d]))):
                    for q, e in estado[slug(c["nome"])].items():
                        if e["estado"] == Estado.PENDENTE.value:
                            evento(c, q, tipo="cancelou", motivo=f"dependência fora desta execução e sem arquivo: {d}")
        manifesto()
    pendentes = [c for c in cfgs if estado_config(estado[slug(c["nome"])]) != "completa"]
    a.externo_ate = t_ini + 45 * 60              # a base externa tem 45 min para chegar; depois disso a execução segue sem ela

    def linha_externa(c):
        """Baixa e prepara a base externa numa linha à parte (rede e CPU), desde o começo: as GPUs seguem com o
        Nutrition5k enquanto isso. Se falhar, só as configurações com a base são canceladas, e o manifesto diz por quê."""
        while estado[slug(c["nome"])]["tudo"]["estado"] == Estado.PENDENTE.value:
            with trava:
                evento(c, "tudo", tipo="iniciou", dispositivo="cpu"); manifesto()
            t0 = time.time()
            try:
                prepara_externo(a, D)
                with trava:
                    evento(c, "tudo", tipo="concluiu", seg=time.time() - t0)
                    info["externo"]["situacao"] = "fotos prontas"; manifesto()
            except BaseException as ex:      # qualquer falha aqui só tira a base externa da execução
                diz(f"base externa {a.externo}: {type(ex).__name__}: {ex}")
                with trava:
                    evento(c, "tudo", tipo="falhou", erro=f"{type(ex).__name__}: {ex}")
                    if estado[slug(c["nome"])]["tudo"]["estado"] == Estado.FALHOU.value:
                        info["externo"]["situacao"] = f"FORA desta execução: as fotos não ficaram prontas ({type(ex).__name__}: {ex})"
                        diz(f"ATENÇÃO: a base externa {a.externo} fica de FORA desta execução; segue como em 12.3")
                    manifesto()

    if D.ext and any(c["tipo"] == "externo" and estado_config(estado[slug(c["nome"])]) == "completa" for c in cfgs):
        info["externo"]["situacao"] = "fotos prontas"
    ts_ext = [threading.Thread(target=linha_externa, args=(c,), daemon=True) for c in cfgs if c["tipo"] == "externo"] if pendentes else []
    [t.start() for t in ts_ext]
    if pendentes or not a.sem_servico:
        Cache(a.cache, D).prepara(profundidade=any(c["tipo"] == "destila-rgbd" for c in pendentes))

    def cancela(c, motivo):
        for p, e in estado[slug(c["nome"])].items():
            if e["estado"] == Estado.PENDENTE.value:
                evento(c, p, tipo="cancelou", motivo=motivo)

    def tenta_decisao():
        """Regra de adoção da base externa (12.5): roda aqui, sem GPU, assim que os dois professores estão prontos (ou
        quando a versão com a base não vai mais ficar pronta). Cancela, dos ajustes finos, a versão que a decisão exclui."""
        est = lambda nome: estado[slug(nome)]
        viva = lambda nome: any(q["estado"] in (Estado.PENDENTE.value, Estado.RODANDO.value) for q in est(nome).values())
        for c in cfgs:
            if c["tipo"] != "decisao":
                continue
            arq = os.path.join(a.saida, f"{slug(c['nome'])}.json")
            if est(c["nome"])["tudo"]["estado"] == Estado.PENDENTE.value:
                es, ec = estado_config(est(c["sem"])), estado_config(est(c["com"]))
                if es == "completa" and ec == "completa":
                    try:
                        z1, z2 = (np.load(os.path.join(a.saida, f"pred__{slug(x)}.npz"), allow_pickle=False) for x in (c["sem"], c["com"]))
                        r = regra_adocao(D, z1["oof"], z2["oof"])
                    except Exception as ex:
                        r = dict(adotado=False, motivo=f"não foi possível comparar os dois professores ({type(ex).__name__}: {ex})")
                elif es == "completa" and not viva(c["com"]):
                    r = dict(adotado=False, motivo="a versão com a base externa não ficou pronta (download, lista ou orçamento)")
                elif not viva(c["sem"]) and es != "completa":
                    cancela(c, "o professor sem a base externa não ficou pronto"); continue
                else:
                    continue
                grava_json(arq, dict(r, base=a.externo, sem=c["sem"], com=c["com"], quando=time.strftime("%Y-%m-%d %H:%M:%S")))
                evento(c, "tudo", tipo="iniciou", dispositivo="cpu"); evento(c, "tudo", tipo="concluiu", seg=0.0)
                diz(f"== DECISÃO sobre a base externa {a.externo}: {'ADOTADA' if r['adotado'] else 'NÃO adotada'} | " + (r.get("motivo") or
                    f"critério de CV do professor {r['criterio_sem']:.2f} -> {r['criterio_com']:.2f} ({r['dif_criterio']:+.2f}); acima de 25 g "
                    f"{r['acima_25g_sem']:.2f} -> {r['acima_25g_com']:.2f} ({r['dif_acima_25g']:+.2f})"))
            if est(c["nome"])["tudo"]["estado"] == Estado.FEITA.value:
                adotado = json.load(open(arq))["adotado"]
                info["decisao_externo"] = json.load(open(arq))
                for q in cfgs:
                    if "se_adotado" in q and q["se_adotado"] != adotado:
                        cancela(q, f"decisão sobre a base externa: {'adotada' if adotado else 'não adotada'}")

    with trava:
        tenta_decisao(); manifesto()

    rodando = {}        # processo -> (configuração, início) da parte em execução
    reserva = 0 if a.sem_servico else 330        # tempo guardado para a fase de serviço dos candidatos a modelo único

    seg = lambda c: seg_parte(c, cfgs, estado)

    def proxima(i, dev):
        """Decide o que o processo i faz (ver proxima_acao e simula_orcamento); aplica cancelamentos e cortes de orçamento."""
        while True:
            agora = time.time()
            tenta_decisao()
            livres = [agora if j not in rodando else max(agora, rodando[j][1] + seg(rodando[j][0])) for j in range(len(disp))]
            acao, c, x = proxima_acao(cfgs, estado, simula_orcamento(cfgs, estado, livres, seg, prazo - reserva))
            if acao == "cancela":
                diz(f"{c['nome']} cancelada: {x}"); cancela(c, x)
            elif acao == "sem_orcamento":
                diz(f"{c['nome']} não cabe no orçamento ({seg(c) / 60:.1f} min por parte, restam {(prazo - reserva - agora) / 60:.0f} min): fica de fora")
                evento(c, x, tipo="sem_orcamento"); cancela(c, "sem orçamento para a configuração inteira")
            elif acao == "roda":
                evento(c, x, tipo="iniciou", dispositivo=dev); rodando[i] = (c, agora)
                return c, x
            elif acao == "espera" and agora > prazo - reserva and not rodando:      # esperando algo que não chegou a tempo (ex.: base externa)
                for q in cfgs:
                    if any(e["estado"] == Estado.PENDENTE.value for e in estado[slug(q["nome"])].values()) and q["tipo"] not in ("externo", "decisao"):
                        diz(f"{q['nome']}: a dependência não ficou pronta dentro do orçamento"); cancela(q, "dependência não ficou pronta dentro do orçamento")
                return "fim"
            else:
                return acao

    def trabalhador(i, dev):
        while True:
            with trava:
                rodando.pop(i, None)
                job = proxima(i, dev); manifesto()
            if job == "fim":
                return
            if job == "espera":      # depende de algo que outro processo está rodando
                time.sleep(1 if a.smoke else 5)
                continue
            c, p = job
            e = estado[slug(c["nome"])][p]
            diz(f"[{dev}] começa {c['nome']} | {p} (tentativa {e['tentativa']}, lote ÷ {e['div_lote']}); restam {(prazo - time.time()) / 60:.0f} min")
            rc, cauda, seg = lanca(c, p, dev, e["div_lote"], max(1, (os.cpu_count() or 2) // len(disp)) if not a.smoke else 1)
            with trava:
                if rc == SAI_OK and parte_pronta(a, c, p, D):
                    evento(c, p, tipo="concluiu", seg=seg)
                    if estado_config(estado[slug(c["nome"])]) == "completa":
                        try:
                            fecha_config(c)
                        except Exception as ex:
                            diz(f"ERRO ao juntar {c['nome']}: {ex}")
                elif rc == SAI_SEM_ORCAMENTO:
                    evento(c, p, tipo="sem_orcamento"); cancela(c, "sem orçamento para a configuração inteira")
                elif rc == SAI_MEMORIA:
                    evento(c, p, tipo="estourou_memoria")
                    for q, eq in estado[slug(c["nome"])].items():   # as outras partes da configuração usam o mesmo lote reduzido
                        if eq["estado"] == Estado.PENDENTE.value and q != p:
                            evento(c, q, tipo="reduziu_lote", div_lote=estado[slug(c["nome"])][p]["div_lote"])
                else:
                    evento(c, p, tipo="falhou", erro=f"código {rc}: " + " | ".join(cauda[-3:]))
                if estado[slug(c["nome"])][p]["estado"] == Estado.FALHOU.value:
                    diz(f"[{dev}] {c['nome']} | {p} FALHOU: {estado[slug(c['nome'])][p]['erro']}")
                    cancela(c, f"a parte {p} falhou")
                manifesto()

    if pendentes:
        ts = [threading.Thread(target=trabalhador, args=(i, d)) for i, d in enumerate(disp)]
        [t.start() for t in ts]; [t.join() for t in ts + ts_ext]

    # ---- fase de serviço: depois de todo o treino, um modelo por vez numa só GPU, com a CPU livre (latências limpas)
    def serve(c):
        s = slug(c["nome"])
        if os.path.exists(arq_de(a.saida, c, "servico")) and estado_config(estado[s]) == "completa":
            return evento(c, "servico", tipo="retomou")
        if c.get("servico") is False:
            return evento(c, "servico", tipo="cancelou", motivo="sem modelo próprio a medir (mesmo backbone de um ridge, ou composição)")
        if c["tipo"] == "unico" and a.externo and bool(c.get("externo")) != bool((info.get("decisao_externo") or {}).get("adotado")):
            return evento(c, "servico", tipo="cancelou", motivo="mede-se só a versão que a decisão sobre a base externa indicou")
        if estado_config(estado[s]) != "completa" or not os.path.exists(arq_de(a.saida, c, "servir")) \
                or not os.path.exists(arq_de(a.saida, c, "pesos" if c["tipo"] in ("ft", "destila-rgbd") else "cabeca")):
            return evento(c, "servico", tipo="cancelou", motivo="configuração incompleta ou sem arquivos de servir")
        while est_serv[s]["servico"]["estado"] == Estado.PENDENTE.value:
            if time.time() + (60 if c["tipo"] in ("ft", "destila-rgbd") else 200) > prazo:     # backbone congelado grande: carregar + medir leva ~2,5 min
                return evento(c, "servico", tipo="sem_orcamento")
            evento(c, "servico", tipo="iniciou", dispositivo=disp[0])
            diz(f"[{disp[0]}] serviço: {c['nome']}; restam {(prazo - time.time()) / 60:.0f} min")
            rc, cauda, seg = lanca(c, "servico", disp[0], 1, (os.cpu_count() or 2) if not a.smoke else 1)
            if rc == SAI_OK and os.path.exists(arq_de(a.saida, c, "servico")):
                evento(c, "servico", tipo="concluiu", seg=seg)
            elif rc == SAI_SEM_ORCAMENTO:
                evento(c, "servico", tipo="sem_orcamento")
            else:
                evento(c, "servico", tipo="falhou", erro=f"código {rc}: " + " | ".join(cauda[-3:]))

    with trava:
        tenta_decisao()
        if not a.sem_servico:
            # na rodada 3, primeiro os candidatos a modelo único (a regra de 12.3 usa o tempo deles); depois os congelados
            for c in sorted(cfgs, key=lambda c: 0 if c.get("v3") and c["tipo"] in ("unico", "ft") else 1):
                serve(c); manifesto()
            adot = bool((info.get("decisao_externo") or {}).get("adotado"))      # nos totais entra a versão adotada de cada candidato
            troca = lambda nome: next((c["nome"] for c in cfgs if adot and c.get("plano") == nome and c["tipo"] in ("unico", "ft")), nome)
            sv = monta_servico(a, cfgs, {"smoke_todos": [c["nome"] for c in cfgs if est_serv[slug(c["nome"])]["servico"]["estado"] == Estado.FEITA.value]} if a.smoke
                               else ({k: [troca(x) for x in v] for k, v in PIPELINES3.items()} if a.rodada == "3" else PIPELINES))
            for k, v in sv["pipelines"].items():
                diz(f"serviço, {k}: {'completo' if v['completo'] else 'FALTAM ' + str(v['faltam'])} | {v['parametros'] / 1e6:.0f} M parâmetros | "
                    f"pesos {v['bytes_pesos'] / 2 ** 30:.2f} GB | GPU ms/imagem {v['gpu_ms_por_imagem']} | CPU ms/imagem {v['cpu_ms_por_imagem']}")
        if D.ext is not None:       # relato secundário de 12.5: erro de massa no teste_ext, com e sem a base no treino
            preds = {}
            for c in cfgs:
                arq = os.path.join(a.saida, f"pred__{slug(c['nome'])}.npz")
                if os.path.exists(arq) and (c["tipo"] in ("professor", "unico", "ft")):
                    preds[c["nome"]] = np.load(arq, allow_pickle=False)
            try:        # relato secundário: nada aqui pode derrubar o fechamento da execução
                rel = relato_externo(D, preds)
                grava_json(os.path.join(a.saida, "relato_externo.json"), rel)
                for nome, r in rel["modelos"].items():
                    diz(f"relato externo (teste_ext, {rel['n_teste_ext']} pratos; não escolhe nada) {nome}: " + " | ".join(
                        f"{k}: MAPE de massa {v['total'].get('mape')}%, viés mediano {v['total'].get('vies_mediano_pct')}%" for k, v in r.items() if isinstance(v, dict)))
            except Exception as ex:
                diz(f"relato externo: não foi possível montar ({type(ex).__name__}: {ex})")
        info["disco_livre_gb_fim"] = dict(saida=livre(a.saida), cache=livre(a.cache))
        manifesto()
        tam = grava_tamanhos(a)
        info["tamanho_saida"] = {k: tam[k] for k in ("total_gb", "n_arquivos", "por_tipo_mb", "maior_arquivo")}
        manifesto()
    diz("\n" + "\n".join(f"  {ic['estado']:14s} {n:44s} " + (f"RMSE {ic['cv_treino']['rmse_logm']:.4f} crit {ic['cv_treino']['crit_direto']:.1f}/{ic['cv_treino']['crit_mxd']:.1f}/{ic['cv_treino']['crit_geo']:.1f}% {ic['min']} min"
                                                                  if ic["cv_treino"] else "") + f" | serviço: {ic['servico']['estado']}" for n, ic in info["configs"].items()))
    diz(f"melhor modelo único pela CV: {info['melhor_modelo']}")
    diz(f"saída: {tam['total_gb']} GB em {tam['n_arquivos']} arquivos {tam['por_tipo_mb']} (MB por tipo)")
    diz(f"fim: {info['duracao_min']} min. Baixe a pasta de saída e rode eval/treino/kaggle/ingere_kaggle.py <pasta>.")


def checa_modelos():
    """Monta cada backbone SEM pesos e passa uma imagem em paisagem e em retrato (confere nomes, formatos e orientações)."""
    import torch
    for bb, b in BACKBONES.items():
        t0 = time.time()
        rede = cria_rede(bb, 22, pretrained=False).eval()
        x = torch.rand(1, 3, *b["hw"])
        with torch.inference_mode():
            formas = [tuple(rede(orienta(x, k))[0].shape) for k in (0, 5)]
        print(f"{bb:28s} {b['timm']:44s} {sum(p.numel() for p in rede.parameters()) / 1e6:6.1f} M parâmetros | dim {rede.dim} | "
              f"blocos {len(rede.m.blocks)} | grupos de lr {len(grupos_lr(rede, 1e-4, 0.75, 0.05))} | saídas {formas} | {time.time() - t0:.1f}s", flush=True)


def main():
    kaggle = os.path.isdir("/kaggle/working")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--saida", default="/kaggle/working" if kaggle else os.path.abspath("nutrelle_saida"))
    ap.add_argument("--cache", default="/kaggle/temp/nutrelle_cache" if kaggle else os.path.join(tempfile.gettempdir(), "nutrelle_cache"))
    ap.add_argument("--listas", default=os.path.join(AQUI, "listas"))
    ap.add_argument("--orcamento-h", type=float, default=11.0, help="horas de relógio desde o início; o que não couber fica 'sem_orcamento'")
    ap.add_argument("--so", action="append", default=[], help="só configurações cujo nome contém o texto (pode repetir)")
    ap.add_argument("--retoma", action="append", default=[], help="pasta com a saída de uma execução anterior (pode repetir)")
    ap.add_argument("--rodada", choices=sorted(RODADAS), default="3",
                    help="quais configurações: 1 = as 15 da seção 10; 2 = as 11 da seção 11; 3 = as 13 da seção 12.3 (treino_v2, só teste2)")
    ap.add_argument("--externo", choices=sorted(EXTERNOS), help="base externa de treino (12.4 e 12.5; desligada por padrão): entra como ablação com decisão de adoção")
    ap.add_argument("--externo-lista", help="lista da base externa (padrão: a de listas/ indicada em EXTERNOS)")
    ap.add_argument("--externo-imagens", help="pasta com as fotos da base externa já em disco (com ela nada é baixado)")
    ap.add_argument("--guarda-pesos", choices=["melhor", "todos", "nenhum"], default="todos",
                    help="pesos finais dos ajustes finos: todos (padrão; é o que serve), só o melhor pela CV, ou nenhum")
    ap.add_argument("--sem-servico", action="store_true", help="não mede latência nem grava servico.json")
    ap.add_argument("--smoke", type=int, nargs="?", const=8, default=0, help="teste rápido: N pratos por fold (padrão 8), backbone minúsculo, CPU")
    ap.add_argument("--workers", type=int, default=1, help="processos em paralelo quando não há GPU (--smoke/--cpu)")
    ap.add_argument("--cpu", action="store_true", help="ignora a GPU")
    ap.add_argument("--lista", action="store_true"); ap.add_argument("--checa-modelos", action="store_true"); ap.add_argument("--autoteste", action="store_true")
    ap.add_argument("--job"); ap.add_argument("--dispositivo", default="cpu"); ap.add_argument("--prazo", type=float); ap.add_argument("--div-lote", type=int, default=1)
    a = ap.parse_args()
    a.saida, a.cache, a.listas = (os.path.abspath(p) for p in (a.saida, a.cache, a.listas))
    if a.autoteste:
        return autoteste()
    if a.checa_modelos:
        return checa_modelos()
    a.externo_pesos = None
    if a.externo:
        assert a.rodada == "3", "--externo só existe na rodada 3"
        a.externo_lista = os.path.abspath(a.externo_lista or os.path.join(a.listas, EXTERNOS[a.externo]["lista"]))
        a.externo_imagens = os.path.abspath(a.externo_imagens) if a.externo_imagens else None
        a.externo_pesos = EXTERNOS[a.externo]["pesos"]
        try:        # 12.5: se a lista não estiver pronta ou não passar nas conferências, a execução segue sem ela
            Dados(a.listas, a.smoke, a.rodada, a.externo_lista, a.externo_imagens, a.externo_pesos)
        except (OSError, AssertionError, KeyError, ValueError) as e:
            assert not a.job, f"lista externa inválida num processo filho: {e}"
            a.externo_fora = f"FORA desta execução: a lista de {a.externo} não está pronta ou não passou nas conferências ({type(e).__name__}: {e})"
            print(f"ATENÇÃO: a base externa {a.externo} fica de FORA desta execução ({type(e).__name__}: {e}); segue como em 12.3", flush=True)
            a.externo = None
    if not a.externo:
        a.externo_lista = a.externo_imagens = a.externo_pesos = None
    gemeas = lambda cs: com_externo(cs, a.externo) if a.externo else cs
    if a.job:       # processo filho: procura a configuração entre todas as declaradas para a rodada
        return principal_parte(a, gemeas(configs_da_rodada("3", a.smoke)) if a.rodada == "3" else (CONFIGS_SMOKE if a.smoke else CONFIGS))
    cfgs = a._todas = gemeas(configs_da_rodada(a.rodada, a.smoke))
    if a.so:
        cfgs = [c for c in cfgs if any(t in c["nome"] for t in a.so)]
    if a.lista:
        for c in cfgs:
            b = BACKBONES.get(c["bb"], dict(timm="(combina previsões)", hw=""))
            print(f"{c['nome']:46s} {c['tipo']:13s} {b['timm']:44s} {b['hw']} {c['est_min']:4d} min de GPU "
                  + (f"| lr {c['lr']:g} lld {c['lld']} épocas {c['epocas']} lote {c['bs']}×{c['acum']} escala {c['escala']}" if "lld" in c else "")
                  + "".join(f" | {k}" for k in ("fracoes", "peso_faixa", "professor", "externo") if c.get(k)))
        return print(f"rodada {a.rodada}: {len(cfgs)} configurações, {sum(c['est_min'] for c in cfgs) / 60:.1f} h de GPU "
                     + ("estimadas com os tempos da execução 2" if a.rodada == "3" else "medidas na execução 1")
                     + " (2 × T4: perto de metade disso em relógio), mais a fase de serviço")
    orquestra(a, cfgs)


if __name__ == "__main__":
    main()
