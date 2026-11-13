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

FORMATO = "nutrelle-treino-gpu/6"          # /2 = /1 + teste2, arquivos de servir e servico.json; /3 = /2 + rodada 3 (seção 12); /4 = /3 + rodada 4 (seção 13);
                                            # /5 = /4 + rodada 5 (seção 14: recorte pelo prato, cabeça sem gradiente, conjunto 'remontagem');
                                            # /6 = /5 + rodada 6 (seção 15: P3 com a HP3 e a escolha da parada interna, H0c-m)
FORMATOS_ACEITOS = ("nutrelle-treino-gpu/1", "nutrelle-treino-gpu/2", "nutrelle-treino-gpu/3", "nutrelle-treino-gpu/4", "nutrelle-treino-gpu/5", FORMATO)
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
    if cfg.get("v4"):       # rodada 4: cada configuração declara as suas partes (treino + 'ext', as previsões nas fotos externas)
        return list(cfg["partes"])
    return ["tudo"] if cfg["tipo"] in ("ridge", "fracoes", "professor", "unico", "externo", "decisao") else list(PARTES_FT)


def todas_as_configs():
    return (CONFIGS + CONFIGS3 + CONFIGS_SMOKE + CONFIGS_SMOKE3 + CONFIGS4 + [c for c in CONFIGS_SMOKE4 if c["tipo"] != "externo"]
            + [c for c in CONFIGS5 if c["tipo"] != "externo"] + [c for c in CONFIGS_SMOKE5 if c["tipo"] != "externo"]
            + [c for c in CONFIGS6 if c["tipo"] != "externo"] + [c for c in CONFIGS_SMOKE6 if c["tipo"] != "externo"])


def configs_da_rodada(rodada, smoke=False):
    if rodada == "6":
        return list(CONFIGS_SMOKE6 if smoke else CONFIGS6)
    if rodada == "5":
        return list(CONFIGS_SMOKE5 if smoke else CONFIGS5)
    if rodada == "4":
        return list(CONFIGS_SMOKE4 if smoke else CONFIGS4)
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
    tempo, e as de prioridade menor ainda podem caber. Devolve {nome: bool, ou "qualquer_gpu"}.
    Rodada 5 (`gpu_flexivel`): a configuração que não cabe na GPU preferida, mas cabe se as partes forem para o processo
    que fica livre primeiro, vale "qualquer_gpu": a outra GPU pega as partes dela antes do próprio trabalho de prioridade
    menor. É o que faz a ordem de corte de 14.5 (F5, depois F4c, depois P2) valer entre as duas GPUs."""
    livres, cabe = list(livres), {}
    for c in cfgs:
        n = sum(p["estado"] == Estado.PENDENTE.value for p in estado[slug(c["nome"])].values())
        if not n or c["tipo"] in ("externo", "decisao"):      # o preparo da base externa e a decisão não ocupam GPU
            continue
        fixa = c.get("gpu") if c.get("gpu") is not None and c["gpu"] < len(livres) else None     # rodada 4: GPU preferida da configuração (13.6)

        def projeta(fixa):
            f, fim = list(livres), 0.0
            for _ in range(n):
                g = fixa if fixa is not None else min(range(len(f)), key=f.__getitem__)
                f[g] += seg_parte(c); fim = max(fim, f[g])
            return f, fim
        f, fim = projeta(fixa)
        cabe[c["nome"]] = fim <= limite
        if not cabe[c["nome"]] and fixa is not None and c.get("gpu_flexivel"):
            f, fim = projeta(None)
            cabe[c["nome"]] = "qualquer_gpu" if fim <= limite else False
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


def proxima_acao(cfgs, estado, cabe, gpu=None):
    """O que um processo livre faz agora? Função pura (estado das partes, o que cabe, nº da GPU do processo) -> uma ação:
      ("cancela", cfg, motivo)      uma dependência não ficou pronta nem vai ficar;
      ("sem_orcamento", cfg, parte) a configuração não cabe no tempo que resta;
      ("roda", cfg, parte)          primeira parte pendente, na ordem de prioridade, com as dependências completas;
      ("espera", None, None)        há partes pendentes, mas dependem de algo que ainda está rodando;
      ("fim", None, None)           nada pendente.
    Uma configuração só passa à frente de outra bloqueada se for marcada `livre` (as etapas sobre features congeladas);
    os alunos seguem a ordem estrita, para o orçamento cortar sempre a de menor prioridade.
    Rodada 4: `gpu` na configuração é a GPU preferida: o processo pega primeiro o que é da sua GPU (ou de nenhuma) e só
    leva trabalho da outra quando não tem nada seu para rodar. `tardia` = dict(parte, terminal): essa parte só roda
    depois de todas as outras partes da configuração e quando as configurações de `terminal` já terminaram, bem ou mal."""
    est = lambda c: estado[slug(c["nome"])]
    viva = lambda c: any(p["estado"] in (Estado.PENDENTE.value, Estado.RODANDO.value) for p in est(c).values())
    por_nome = {c["nome"]: c for c in cfgs}
    bloqueio = espera = False
    outra = None
    for c in cfgs:
        pend = [p for p in partes_de(c) if est(c)[p]["estado"] == Estado.PENDENTE.value]
        if not pend or c["tipo"] in ("externo", "decisao"):       # preparo da base externa e decisão: fora das GPUs
            continue
        deps = [por_nome[d] for d in c.get("depende", []) if d in por_nome]
        # dependência pronta = todas as partes dela feitas, menos a parte tardia (as previsões nas fotos externas da
        # rodada 4 não são entrada de ninguém): assim R1 a R4 não esperam o R0 terminar as previsões externas
        pronta = lambda d: all(est(d)[p]["estado"] == Estado.FEITA.value for p in partes_de(d) if p != d.get("tardia", {}).get("parte"))
        mortas = [d["nome"] for d in deps if not pronta(d) and not viva(d)]
        if mortas:
            return "cancela", c, f"dependência não ficou pronta: {mortas[0]}"
        if any(not pronta(d) for d in deps):
            bloqueio = espera = True
            continue
        if bloqueio and not c.get("livre"):
            espera = True
            continue
        tard = c.get("tardia")
        if tard and pend[0] == tard["parte"] and (any(est(c)[p]["estado"] != Estado.FEITA.value for p in partes_de(c) if p != tard["parte"])
                                                  or any(viva(por_nome[d]) for d in tard["terminal"] if d in por_nome)):
            espera = True
            continue
        if not cabe.get(c["nome"], True):       # só se decide o corte quando chega a vez da configuração
            return "sem_orcamento", c, pend[0]
        if gpu is None or c.get("gpu") in (None, gpu) or cabe.get(c["nome"]) == "qualquer_gpu":
            return "roda", c, pend[0]
        outra = outra or ("roda", c, pend[0])   # trabalho da outra GPU: só se não houver nada desta
    return outra or (("espera", None, None) if espera else ("fim", None, None))


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
    nomes = {c["nome"] for c in CONFIGS + CONFIGS3 + CONFIGS4 + CONFIGS5 + CONFIGS6}
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
    teste_rodada4()
    teste_rodada5()
    teste_rodada6()
    try:
        import torch  # noqa: F401
    except ImportError:
        return print("autoteste: ok (sem torch: alvos aninhados e cabeças com bases externas não testados)")
    teste_aninhado()
    teste_rodada4_torch()
    teste_rodada5_torch()
    teste_rodada6_torch()
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

    def __init__(self, pasta, smoke=0, rodada="2", externo=None, externo_imagens=None, externo_pesos=None, dominios=None):
        self.rodada, v3 = rodada, rodada in ("3", "4", "5", "6")   # rodadas 3 a 6: treino_v2 (com frações por fonte); 3 e 4 preveem o teste2
        arq_tr, arq_av = os.path.join(pasta, "treino_v2.csv" if v3 else "treino.csv"), os.path.join(pasta, "aval_ids.csv")
        with open(arq_tr) as f:
            rows = list(csv.DictReader(f))
        self.grupos = [c[2:] for c in rows[0] if c.startswith("m_")]
        G = len(self.grupos)
        self.N = 6 + 2 * G
        self.I_MG, self.I_PG = slice(6, 6 + G), slice(6 + G, 6 + 2 * G)
        self.ids = [r["dish_id"] for r in rows]
        self.fold = np.array([int(r["fold"]) for r in rows])
        self.cadeia = [r.get("cadeia") or r["dish_id"] for r in rows]       # rodada 6: a validação interna do P3 sai por cadeia (15.4)
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
        if rodada in ("5", "6"):    # 14.1/15.1: o teste2 está encerrado e não é previsto. O único "conjunto" é o de remontagem: os 8 primeiros
            # pratos de treino, previstos pelo modelo final só para conferir que os arquivos de servir repetem as previsões (nunca pontuado)
            assert not self.hash2, "a quinta e a sexta execuções não levam o teste2 (14.1, 15.1): tire aval2_ids.csv do pacote"
            prim = [d for d, u in zip(self.ids, self.usa_tr) if u][:8]
            self.conj, self.usa_av, self.extras = {"remontagem": prim}, {"remontagem": np.ones(len(prim), bool)}, []
        # gancho de 12.4 (desligado por padrão): amostras externas, só de treino, com cache de fotos e de features à parte
        self.ext = le_externo(externo, externo_imagens, self, smoke, externo_pesos) if externo else None
        # rodada 4 (seção 13): um domínio externo por lista + tabela de divisão presentes na pasta; os outros ficam de fora
        self.dom, self.dom_fora = {}, {}
        if rodada in ("4", "5", "6"):
            for chave in DOMINIOS4:
                if dominios is not None and chave not in dominios:
                    self.dom_fora[chave] = "fora desta execução por --dominios"
                    continue
                try:
                    self.dom[chave] = le_dominio4(pasta, chave, self, smoke, os.path.join(externo_imagens, chave) if externo_imagens else None)
                except FileNotFoundError as e:
                    self.dom_fora[chave] = f"lista ou tabela de divisão não está neste pacote ({os.path.basename(str(e.filename))})"
                except (AssertionError, KeyError, ValueError) as e:
                    self.dom_fora[chave] = f"a lista não passou nas conferências ({type(e).__name__}: {e})"


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
        if ext["X"].ndim == 2:   # rodada 4: cada linha já é uma amostra (foto × orientação), com peso por amostra e por alvo
            rep_e, oe = (lambda A: np.asarray(A, np.float32)), 1
        else:
            rep_e, oe = rep, o
        Me = rep_e(np.isfinite(ext["T"]) * ext["peso_alvo"] * (ts > 1e-3))     # alvo sem variação no Nutrition5k (só no --smoke): fora
        Mfe = rep_e(np.isfinite(ext["FF"]).all(1) * ext["peso_ff"])
        Xn = np.r_[Xn, ((ext["X"] - mu) / sd).reshape(-1, Dm).astype(np.float32)]
        Yn_ = np.r_[Yn_, rep_e(np.nan_to_num((ext["T"] - tm) / ts))]
        Fn = np.r_[Fn, rep_e(np.nan_to_num(ext["FF"]))]
        Wn = np.r_[Wn, np.broadcast_to(np.asarray(ext["c"], np.float32), (me,)).repeat(oe)]
        Pnn = None if Pnn is None else np.r_[Pnn, np.zeros((me * oe, 3), np.float32)]
        Mt = torch.from_numpy(np.r_[np.ones((n * o, nt), np.float32), Me]).to(dev)           # por alvo
        Mf = torch.from_numpy(np.r_[np.ones(n * o, np.float32), Mfe]).to(dev)                # frações
        Mn = torch.from_numpy(np.r_[np.ones(n * o, np.float32), np.zeros(me * oe, np.float32)]).to(dev)   # só Nutrition5k: densidade composta e destilação
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


def baixa_arquivo(url, dest, tentativas=8, ate=None, silencio=False):
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
            if ate and time.time() > ate:
                raise TimeoutError(f"download de {os.path.basename(dest)} passou do prazo")
            try:
                ja = os.path.getsize(parte) if os.path.exists(parte) else 0
                req = urllib.request.Request(url, headers=dict({"User-Agent": "python-urllib"}, **({"Range": f"bytes={ja}-"} if ja else {})))   # o Mendeley recusa o agente padrão
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
                if not silencio:
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


def monta_lote(C, linhas, dev, hw, canais=3, zoom=None, enquadramento=None):
    """Lê as fotos do cache e devolve [B, canais, h, w]: RGB em [0, 1] (+ altura em cm / 5 no 4º canal). Rodada 5, sobre o
    recorte (C.prato): as fotos do Nutrition5k são recortadas na janela do guia no dispositivo, com o erro de enquadramento
    [B, 3] (escala, dx, dy) se dado; as das bases externas já vêm recortadas do cache."""
    import torch
    linhas = np.asarray(linhas)
    n5k = len(C.rgb)
    if getattr(C, "blocos", None) and (linhas >= n5k).any():                  # rodada 4: um cache por domínio externo
        def de(i):
            if i < n5k:
                return C.rgb, C.cor, i
            ini, r, c = next(b for b in reversed(C.blocos) if i >= b[0])
            return r, c, i - ini
        tr = [de(int(i)) for i in linhas]
        rgb, cor = np.stack([r[j] for r, _, j in tr]), np.stack([c[j] for _, c, j in tr])
    elif getattr(C, "ext_rgb", None) is not None and (linhas >= n5k).any():   # lote com fotos externas (cache à parte)
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
    if getattr(C, "prato", False):
        js = C.janelas_pos
        out = []
        for b, i in enumerate(linhas):
            jan = js.get(int(i))
            if jan is None and enquadramento is not None and i >= n5k:     # rodada 6: foto externa, já recortada no cache; o erro de
                jan = (x.shape[-1] / 2, x.shape[-2] / 2, float(x.shape[-2]), float(x.shape[-1]))     # enquadramento vale sobre o recorte inteiro
            if jan is not None and enquadramento is not None:
                jan = janela_desvio(jan, *enquadramento[b])
            out.append(recorte_gpu(x[b], jan, cor[b], hw) if jan is not None else redimensiona(x[b:b + 1], hw)[0])
        x = torch.stack(out)
    else:
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
    if getattr(C, "prato", False):          # rodada 5: sobre o recorte do guia (14.3), 8 vistas com erro de enquadramento
        arq = os.path.join(C.pasta, f"feat__{bb}__prato_{C.prato_hash['n5k']}.npy")
        with Trava(arq + ".trava"):
            if not os.path.exists(arq):
                feats_prato_n5k(arq, bb, D, C, dev, bs, prazo)
        return np.load(arq)
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


def _extrai_feats(arq, bb, D, C, dev, bs, prazo, segmentos, base=0, n_or=8):
    import torch
    t0 = time.time()
    rede = cria_rede(bb, D.N).to(dev).eval()
    hw = BACKBONES[bb]["hw"]
    X = np.zeros((segmentos[-1][1] - base, n_or, rede.dim), np.float16)
    with torch.inference_mode():
        for ini, fim in segmentos:   # treino (+ 450) nos mesmos lotes de sempre; externos e teste2 em lotes à parte, depois
            for i in range(ini, fim, bs):
                if time.time() > prazo:
                    raise SemOrcamento()
                x = monta_lote(C, list(range(i, min(i + bs, fim))), dev, hw)
                for k in range(n_or):
                    with autocast(dev):
                        f = rede.feats(orienta(x, k))
                    X[i - base:i - base + len(x), k] = f.float().cpu().numpy()
    tmp = f"{arq}.{os.getpid()}.npy"
    np.save(tmp, X); os.replace(tmp, arq)
    print(f"features congeladas {bb}: {X.shape} em {time.time() - t0:.0f}s ({len(X) * n_or / (time.time() - t0):.0f} img/s)", flush=True)


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
    X0_sg = Xr if cfg.get("fracoes_sg") else None      # rodada 5: padronização da cabeça de frações sem gradiente
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
    # rodada 5 (14.4): cabeça de frações sem gradiente para o backbone, com geradores próprios (não muda nada da rede)
    sg = CabecaSG(X0_sg, D, tr, dev, semente, total * acum) if X0_sg is not None else None
    del X0_sg
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
        if variantes and sg is not None:
            fr_sg, o_sg = np.zeros((len(ids), 8, len(D.fontes)), np.float32), np.zeros((len(ids), 8, N), np.float32)
        with torch.inference_mode():
            for i in range(0, len(ids), bs):
                x = monta_lote(C, [C.pos[d] for d in ids[i:i + bs]], dev, hw, canais)
                soma = 0
                for k in range(8 if variantes else n_or):
                    with autocast(dev):
                        o, f_ = ema(orienta(x, k))
                        o = o.float()
                    if variantes and sg is not None:
                        p_sg, q_sg = sg.preve(f_)
                        o_sg[i:i + len(x), k], fr_sg[i:i + len(x), k] = p_sg.cpu().numpy(), q_sg.cpu().numpy()
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
        if sg is not None:      # frações da cabeça sem gradiente; as 22 saídas dela ficam em preve.ultima_sg
            preve.ultima_sg = {suf: media_or(o_sg, k) for suf, k in ORIENT.items()}
            return {suf: media_or(out, k) for suf, k in ORIENT.items()}, {suf: media_or(fr_sg, k) for suf, k in ORIENT.items()}
        return {suf: media_or(out, k) for suf, k in ORIENT.items()}, ({suf: media_or(fr, k) for suf, k in ORIENT.items()} if K else None)

    preve.dens_fonte = dc if K else (sg.dc if sg is not None else None)
    preve.sg = sg
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
                enq = desvios5(rng, len(j), cfg["enquadramento"]) if cfg.get("enquadramento") else None      # F4c: erro de enquadramento (14.4)
                x = orienta_lote(monta_lote(C, linha[j], dev, hw, 4 if professor is not None else canais, zoom, enquadramento=enq), ks)
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
                if sg is not None:
                    sg.passo(f, jt)
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
        if sg is not None:
            h["perda_cabeca_sg"] = sg.media_perda(-passos_ep * acum)
        if va is not None and len(va) and ((ep + 1) % val_cada == 0 or ep + 1 == cfg["epocas"]):
            h.update(valida())
        hist.append(h)
        print(f"{rotulo} {h}", flush=True)
    return ema, preve, hist, (tm, ts)


# ------------------------------------------------------------------ uma parte (processo filho)
def dados_de(a):
    """As listas desta execução, como os argumentos pedem (a mesma chamada no orquestrador e em cada processo filho)."""
    return Dados(a.listas, a.smoke, a.rodada, a.externo_lista, a.externo_imagens, a.externo_pesos, dominios=getattr(a, "dominios", None))


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
    if cfg.get("v4"):
        return parte_pronta4(a, cfg, parte, D)
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
        **(dict(recorte_do_guia=dict(PRATO5, como="no app, a janela é o próprio círculo-guia (diâmetro de 80% do lado menor do quadro); a foto é recortada "
                                                   "nessa janela 4:3 e redimensionada para hw, sem detector. No treino, a janela veio da caixa da refeição da etapa 0 (OWLv2), "
                                                   "que não faz parte do modelo servido",
                                     treino_n5k="8 vistas: recorte exato e 7 recortes com erro de enquadramento sorteado, cada um numa orientação do grupo diedral")) if cfg.get("prato") else {}),
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
    D = dados_de(a)
    destila = cfg["tipo"] == "destila-rgbd"
    C = Cache(a.cache, D)
    if cfg.get("v5"):           # rodada 5: as janelas do guia vêm da etapa 0; as configurações sobre o recorte abrem os recortes
        P = carrega_caixas5(a.listas, [a.caixas], D, a.smoke)
        C.P5 = P
        ab = abre_prato5(C, D, P) if cfg.get("prato") else abre_dominios4(C, D)
        return roda_parte4(a, cfg, parte, cfgs, D, C, ab, dev, a.prazo or float("inf"), f"[{cfg['nome']} | {parte}]", t0)
    if cfg.get("v4"):
        return roda_parte4(a, cfg, parte, cfgs, D, C, abre_dominios4(C, D), dev, a.prazo or float("inf"), f"[{cfg['nome']} | {parte}]", t0)
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


# <<<<< RODADA 4: início
# ===================================================================== RODADA 4 (seção 13): mais bases de treino
# Tudo aqui foi escrito antes de rodar. Nada lê rótulo de avaliação: nem do teste2, nem das partes de registro externas.
DOMINIOS4 = {       # chave = nome usado nas listas (externo_<chave>.csv, divisao_r4_<chave>.csv), nas pastas e nas configurações
    "flavoria": dict(dominio="flavoria", orient=2, origem="tar", arquivo=EXTERNOS["flavoria"]["arquivo"], bytes=EXTERNOS["flavoria"]["bytes"],
                     md5=EXTERNOS["flavoria"]["md5"], url=EXTERNOS["flavoria"]["url"], de="Zenodo 5850856"),
    "lefood": dict(dominio="lefood", orient=2, origem="url", de="Mendeley Data cchsk79jkt, foto a foto"),
    "mensa": dict(dominio="mensa", orient=2, origem="url", de="GitHub Metabolic-Intelligence-Lab/vlm-food-benchmark, foto a foto"),
    "fpb": dict(dominio="fpb", orient=2, origem="url", base="https://huggingface.co/datasets/issai/Food_Portion_Benchmark/resolve/main/",
                de="Hugging Face issai/Food_Portion_Benchmark, foto a foto"),
    "omnifood": dict(dominio="omnifood8k", orient=1, origem="zip", arquivo="0-OminiFood8k.zip", bytes=2731585789, drive_id="1tUIa1jXMLxRGLyex_jVkmqmHneIMU0wG",
                     de="Google Drive (um zip) ou espelho num dataset do Kaggle"),
    "snapme": dict(dominio="snapme", orient=1, origem="targz", arquivo="snapme_db_09Dec2022.tar.gz", bytes=2034227035,
                   url="https://ndownloader.figshare.com/files/44532971", de="Figshare 44532971 (Ag Data Commons)"),
}
DOMINIOS_TREINO4 = ["flavoria", "lefood", "mensa", "fpb", "omnifood"]      # o SNAPMe nunca treina (13.5): só previsões gravadas
PAPEIS_REGISTRO4 = ("registro", "registro_misto", "avaliacao_fraca")       # nunca treinam; o kernel só grava as previsões
PRAZO_DOWNLOAD4 = 40 * 60        # segundos, desde o início, para cada base externa ficar pronta
TETO_DOMINIO4 = 0.5              # soma dos pesos de um domínio = min(0,5; grupos de treino ÷ pratos do treino_v2) × a do Nutrition5k
# colunas que o kernel pode ler de uma linha que NÃO é de treino (ids, onde está a foto, procedência e a divisão): nenhuma de rótulo
COLUNAS_ABERTAS4 = ("dish_id", "dominio", "grupo", "vista", "arquivo", "arquivo_base", "quadros", "url", "sha256", "bytes", "licenca", "uso_declarado",
                    "papel", "fold_ext", "folds_proibidos", "quadro_metrica", "quadro_metrica_nome", "quadros_treino", "grupo_metrica")
COLUNAS_ALVO4 = ("massa", "proteina", "kcal", "gordura", "carbo")           # + m_*, p_*, ff_* (o esquema de treino_v2.csv)
REF_R5, REF_R5_CV = "ref_r5_prato.csv", 68.34      # rodada 6: R4c e H0c fora-do-fold da quinta execução; critério da REF (15.1)
REF_D1 = "ref_d1_r3.csv"         # previsões fora-do-fold das duas cabeças do D1 na execução 3 (conferência de paridade de 13.7)
D1_CV, D1_TOL, D1_CORR = 71.3, 0.3, 0.999

_SO4, _FT4B = "siglip2_so400m16_384x512", "siglip2_b16_224"
P1_RECEITA = dict(lr=2e-5, lld=0.75, epocas=6, bs=16, acum=2, escala=(0.7, 1.3), ckpt_grad=False, wd=0.05, drop_path=0.1, ema=0.99, aquec=0.1, blocos_treinaveis=4)


def configs4(smoke=False):
    """As configurações da quarta execução (13.6), em ordem de prioridade, mais as etapas sem modelo: uma por base externa
    (download e preparo das fotos, numa linha à parte) e uma por bloco de features do backbone congelado. Nenhuma
    configuração decide nada sobre outra; `depende` só diz o que precisa existir em disco antes."""
    pre, so, ftb = ("smoke4", "vit_tiny_96x128", "vit_tiny_224") if smoke else ("gpu4", _SO4, _FT4B)
    ap = "tiny96" if smoke else "so400m"
    mlp = dict(MLP3, epocas=3) if smoke else MLP3
    ext = lambda d: f"externo {d}"
    feat = lambda d: f"{pre} feat {ap} {d}"
    todos_feat = [feat(d) for d in DOMINIOS4]
    cs = [dict(nome=ext(d), tipo="externo", chave=d, bb=None, est_min=0, v4=True, livre=True, servico=False, partes=["tudo"]) for d in DOMINIOS4]
    cs.append(dict(nome=feat("n5k"), tipo="feat", chave="n5k", bb=so, bs=32, est_min=1 if smoke else 29, v4=True, livre=True, servico=False, gpu=0, partes=["tudo"]))

    def cabeca(nome, tipo, apelido, est, doms=(), ff_de=(), **kw):
        return dict(dict(nome=f"{pre} {tipo} {ap}{nome}", tipo=tipo + "4", apelido=apelido, bb=so, bs=32, est_min=est, v4=True, livre=True, gpu=0, servico=False,
                         treina_com=list(doms), ff_de=list(ff_de), partes=["tudo", "ext"], depende=[feat("n5k")] + [feat(d) for d in list(doms) + list(ff_de)],
                         tardia=dict(parte="ext", terminal=todos_feat)), **kw)
    r0, h0 = cabeca("", "ridge", "R0", 1, servico=None), cabeca("", "fracoes", "H0", 5, **mlp)
    cs += [r0, h0]
    est_feat = dict(flavoria=5, lefood=3, mensa=2, fpb=7, omnifood=19, snapme=3)       # passadas ÷ 15 fotos/s + carga do modelo
    cs += [dict(nome=feat(d), tipo="feat", chave=d, bb=so, bs=32, est_min=1 if smoke else est_feat[d], v4=True, livre=True, servico=False, gpu=0,
                partes=["tudo"], depende=[ext(d)]) for d in DOMINIOS4]
    ok = ["flavoria", "lefood", "mensa"]
    for nome, apel, est, doms in ((" +ok", "R1", 2, ok), (" +ok+fpb", "R2", 2, ok + ["fpb"]), (" +omni", "R3", 2, ["omnifood"]), (" +tudo", "R4", 3, DOMINIOS_TREINO4)):
        c = cabeca(nome, "ridge", apel, est, doms); c["depende"].append(r0["nome"]); c["plano"] = r0["nome"]
        cs.append(c)
    cs.append(cabeca(" +omni", "fracoes", "H1", 8, ["omnifood"], **mlp))
    todos_ext = [ext(d) for d in DOMINIOS4]
    ft = lambda **kw: dict(v4=True, livre=True, gpu=1, val_cada=5, partes=PARTES_FT + ["ext"], tardia=dict(parte="ext", terminal=todos_ext), **kw)
    rs = dict(epocas=2, bs=8) if smoke else {}
    f1 = _ft(ftb, "r2", 1 if smoke else 64, prefixo=pre, sufixo=" fracoes", apelido="F1", fracoes4=True, **rs, **ft())
    p1 = dict(nome=f"{pre} ftparcial {ap}", tipo="ftparcial", apelido="P1", bb=so, receita="p1", est_min=1 if smoke else 150, fracoes4=True, depende=[feat("n5k")],
              **dict(P1_RECEITA, **(dict(epocas=2, bs=4, blocos_treinaveis=2) if smoke else {})), **dict(ft(), gpu=0))
    f3 = _ft(ftb, "r2", 1 if smoke else 96, prefixo=pre, sufixo=" +tudo", apelido="F3", externo4=list(DOMINIOS_TREINO4), depende=[ext(d) for d in DOMINIOS_TREINO4], **rs, **ft())
    h2 = cabeca(" +omni+ff", "fracoes", "H2", 10, ["omnifood"], ["flavoria", "fpb", "mensa", "lefood"], **mlp)
    f2 = _ft(ftb, "r2", 1 if smoke else 64, prefixo=pre, sufixo=" fracoes faixa", apelido="F2", fracoes4=True, peso_faixa=True, **rs, **ft())
    return cs + [f1, p1, f3, h2, f2]


CONFIGS4, CONFIGS_SMOKE4 = configs4(), configs4(True)
RODADAS["4"] = [c["nome"] for c in CONFIGS4]


def fracoes_dominio4(grupos_treino, n_n5k, presentes):
    """Peso de cada domínio externo presente, em fração da soma dos pesos do Nutrition5k (13.4): min(0,5; grupos de treino ÷
    pratos do treino_v2); se a soma dos presentes passa de 1,0, todos são multiplicados pelo mesmo fator para ela dar 1,0
    (o Nutrition5k nunca fica com menos da metade do peso). Função pura."""
    f = {d: min(TETO_DOMINIO4, grupos_treino[d] / n_n5k) for d in presentes}
    s = sum(f.values())
    return {d: (v / s if s > 1.0 else v) for d, v in f.items()}


def fecha_rotulos4(lista, divisao):
    """Junta cada linha de uma lista externa com a sua linha da tabela de divisão e FECHA os rótulos (13.8): de toda linha
    que não é de treino só sobram as colunas de COLUNAS_ABERTAS4 (ids, onde está a foto, procedência e divisão); as colunas
    de alvo (massa, proteína, kcal, gordura, carboidrato, m_*, p_*, ff_*) e qualquer outra são apagadas antes de alguém
    ler. Das linhas de treino sobram essas e as de alvo, mais `peso_nutricao`. Colunas de rótulo fraco (`*fraca*`), de
    componentes e as demais nunca passam, em linha nenhuma. Linhas `fora` saem. Função pura."""
    assert [r["dish_id"] for r in lista] == [r["dish_id"] for r in divisao], "lista e tabela de divisão com linhas diferentes"
    out = []
    for r, v in zip(lista, divisao):
        if v["papel"] == "fora":
            continue
        assert v["papel"] == "treino" or v["papel"] in PAPEIS_REGISTRO4, f"papel desconhecido: {v['papel']}"
        j = dict(r, **v)
        alvo = lambda c: v["papel"] == "treino" and (c in COLUNAS_ALVO4 or c.startswith(("m_", "p_", "ff_")) or c == "peso_nutricao") and "fraca" not in c
        out.append({c: x for c, x in j.items() if c in COLUNAS_ABERTAS4 or alvo(c)})
    return out


def le_dominio4(pasta, chave, D, smoke=0, raiz=None):
    """Um domínio externo da rodada 4: externo_<chave>.csv + divisao_r4_<chave>.csv (13.3), com os rótulos das partes de
    registro fechados (fecha_rotulos4). Devolve as FOTOS (uma linha da lista pode dar mais de uma: os três quadros de
    treino de uma cena do OmniFood8K), com papel, fold em que valida, folds proibidos, grupo (a unidade que pesa igual:
    cena, prato, bandeja) e alvos em log só nas fotos de treino. raiz: pasta com as fotos já em disco (teste); no --smoke
    ficam só algumas fotos que existem nela."""
    info = DOMINIOS4[chave]
    arq_l, arq_d = os.path.join(pasta, f"externo_{chave}.csv"), os.path.join(pasta, f"divisao_r4_{chave}.csv")
    with open(arq_l, newline="") as f:
        L = list(csv.DictReader(f))
    with open(arq_d, newline="") as f:
        V = list(csv.DictReader(f))
    assert L and all(r["dominio"] == info["dominio"] for r in L), "domínio da lista"
    lic, uso = {r["licenca"] for r in L}, {r["uso_declarado"] for r in L}
    assert len(lic) == 1 and len(uso) == 1 and "" not in lic | uso, "licença ou uso declarado ausente ou misturado"
    n_lista = len(L)
    grupos_treino = len({v["grupo_metrica"] for v in V if v["papel"] == "treino"})
    R = fecha_rotulos4(L, V)
    del L, V
    conhecidos = set(D.ids) | {d for v in D.conj.values() for d in v} | set(getattr(D, "antigo", {}))
    assert not [r["dish_id"] for r in R if r["dish_id"] in conhecidos or r["dish_id"].startswith("dish_")], "prato do Nutrition5k numa lista externa"
    assert chave in DOMINIOS_TREINO4 or not any(r["papel"] == "treino" for r in R), f"{chave} não treina"
    fotos = []          # (linha, arquivo, id da foto)
    for i, r in enumerate(R):
        if "arquivo_base" in r:     # uma linha por cena, vários quadros
            qs = r["quadros_treino"].split(";") if r["papel"] == "treino" else [r["quadro_metrica_nome"]]
            assert all(q and q in r["quadros"].split(";") for q in qs), f"{r['dish_id']}: quadro fora da lista"
            fotos += [(i, r["arquivo_base"] + q, f"ext:{chave}:{r['dish_id']}#{os.path.splitext(q)[0]}") for q in qs]
        else:
            fotos.append((i, r["arquivo"], f"ext:{chave}:{r['dish_id']}"))
    if smoke:           # teste rápido: poucas fotos, só as que estão na pasta dada, com treino em mais de um fold e algum registro
        tem = [f for f in fotos if raiz is None or acha_foto(raiz, f[1])]
        tr, por_fold = [], {}
        for f in tem:
            if R[f[0]]["papel"] == "treino" and por_fold.setdefault(R[f[0]]["fold_ext"], 0) < max(2, smoke // 2):
                por_fold[R[f[0]]["fold_ext"]] += 1; tr.append(f)
        fotos = tr + [f for f in tem if R[f[0]]["papel"] != "treino"][:max(2, smoke // 2)]
    assert fotos, "nenhuma foto do domínio"
    lin = [f[0] for f in fotos]
    papel = np.array([R[i]["papel"] for i in lin])
    treina = papel == "treino"
    num = lambda r, c: float(r[c]) if r.get(c) not in (None, "") else np.nan
    T = np.full((len(fotos), D.N), np.nan); FF = np.full((len(fotos), len(D.fontes)), np.nan); pn = np.zeros(len(fotos))
    for j, i in enumerate(lin):
        r = R[i]
        if r["papel"] != "treino":      # parte de registro: nenhum rótulo é lido (as colunas nem existem mais)
            continue
        m, p = num(r, "massa"), num(r, "proteina")
        assert np.isfinite(m) and m > 0, f"{r['dish_id']}: linha de treino sem massa"
        with np.errstate(invalid="ignore", divide="ignore"):
            T[j] = np.r_[np.log(m), np.log1p(p), np.log1p(num(r, "kcal")), np.log(p / m + EPS_D), np.log1p(num(r, "gordura")), np.log1p(num(r, "carbo")),
                         np.log1p([num(r, f"m_{g}") for g in D.grupos]), np.log1p([num(r, f"p_{g}") for g in D.grupos])]
        ff = np.array([num(r, f"ff_{g}") for g in D.fontes])
        FF[j] = ff if np.isfinite(ff).all() else np.nan
        pn[j] = num(r, "peso_nutricao") if r.get("peso_nutricao") not in (None, "") else 0.0
    T[~np.isfinite(T)] = np.nan
    proib = np.zeros((len(fotos), 5), bool)
    for j, i in enumerate(lin):
        if R[i]["papel"] == "treino":
            ks = [int(k) for k in R[i]["folds_proibidos"].split(";")]
            assert int(R[i]["fold_ext"]) in ks, f"{R[i]['dish_id']}: o fold em que valida tem de ser proibido no treino"
            proib[j, ks] = True
    sha = lambda arq: hashlib.sha256(open(arq, "rb").read()).hexdigest()
    try:        # sha256 das listas inteiras, quando o pacote embute uma versão enxuta delas (monta_pacote.py)
        proc = json.load(open(os.path.join(pasta, "procedencia_r4.json")))[chave]
    except (OSError, KeyError, ValueError):
        proc = {}
    ids = [f[2] for f in fotos]
    assert len(set(ids)) == len(ids), "foto repetida"
    cont = lambda p: int((papel == p).sum())
    return dict(chave=chave, dominio=info["dominio"], licenca=lic.pop(), uso_declarado=uso.pop(), n_lista=n_lista,
                sha256_lista=proc.get("sha256_lista") or sha(arq_l), sha256_divisao=proc.get("sha256_divisao") or sha(arq_d),
                lista_enxuta=bool(proc), ids=ids, arquivos=[f[1] for f in fotos], url=[R[i].get("url") or None for i in lin],
                sha256=[R[i].get("sha256") or None for i in lin], bytes=[int(R[i]["bytes"]) if R[i].get("bytes") else None for i in lin],
                papel=papel, treina=treina, reg=~treina, fold_ext=np.array([int(R[i]["fold_ext"]) for i in lin]), proib=proib,
                grupo=np.array([R[i]["grupo_metrica"] for i in lin]), metrica=np.array([R[i]["quadro_metrica"] == "1" for i in lin]),
                T=T, FF=FF, peso_nut=pn, grupos_treino_lista=grupos_treino, raiz=raiz,
                hash=hashlib.sha256("\n".join(f"{a}|{b}" for a, b in zip(ids, [f[1] for f in fotos])).encode()).hexdigest()[:16],
                resumo=dict(fotos=len(fotos), treino=cont("treino"), grupos_treino=len(set(np.array([R[i]["grupo_metrica"] for i in lin])[treina])),
                            **{p: cont(p) for p in PAPEIS_REGISTRO4 if cont(p)}))


def pesos_dominio4(d, k, total):
    """Peso de cada foto de treino de um domínio no modelo que deixa o fold k de fora (k = None: modelo final): peso igual
    por grupo, repartido entre as fotos do grupo que podem entrar; a soma é `total`. Fotos proibidas no fold k (13.3) e
    as que não são de treino ficam com peso 0. Função pura."""
    pode = d["treina"] & (~d["proib"][:, k] if k is not None else True)
    w = np.zeros(len(pode))
    if pode.any():
        _, inv, n = np.unique(d["grupo"][pode], return_inverse=True, return_counts=True)
        w[pode] = total / len(n) / n[inv]
    return w


def pesos_dominio6(d, ks, total):
    """pesos_dominio4 para o modelo que deixa de fora os folds ks (lista; vazia = modelo final): fora as fotos proibidas
    em qualquer um deles. Com um fold só, é pesos_dominio4. Função pura."""
    ks = list(ks)
    if len(ks) <= 1:
        return pesos_dominio4(d, ks[0] if ks else None, total)
    pode = d["treina"] & ~d["proib"][:, ks].any(1)
    w = np.zeros(len(pode))
    if pode.any():
        _, inv, n = np.unique(d["grupo"][pode], return_inverse=True, return_counts=True)
        w[pode] = total / len(n) / n[inv]
    return w


def alvos_dominio4(d, D, cfg):
    """Peso de cada alvo em cada foto de treino de um domínio, para a cabeça de frações e para o ajuste fino (13.4):
    (peso [n, 22], peso das frações [n]). Bases só de massa: massa total 1 (e, só onde `ff_de` pede, frações 0,5).
    OmniFood8K: massa 1; proteína, densidade, kcal, gordura e carboidrato com o peso de nutrição da cena (1; 0,5 com osso
    ou casca; 0 sem nutrição); massa e proteína por grupo e frações 0,5. Função pura."""
    P = np.zeros((len(d["ids"]), D.N), np.float32); pf = np.zeros(len(d["ids"]), np.float32)
    ch = d["chave"]
    if ch in cfg.get("ff_de", ()):                  # H2: só as frações por fonte
        pf[:] = 0.5
        return P, pf
    if ch not in (cfg.get("treina_com") or cfg.get("externo4") or ()):
        return P, pf
    P[:, I_M] = 1.0
    if ch == "omnifood":
        P[:, [I_P, I_K, I_D, I_F, I_C]] = d["peso_nut"][:, None]
        P[:, D.I_MG] = 0.5; P[:, D.I_PG] = 0.5
        pf[:] = 0.5
    if cfg["tipo"] == "ridge4" or cfg.get("so_massa_ext"):     # cabeça de massa (e P3, F5c da rodada 6): só a massa total, em qualquer base
        P[:, 1:] = 0.0; pf[:] = 0.0
    return P, pf


def colunas_extensivas4(D):
    """Alvos que levam o viés por domínio (13.4): massa, proteína, kcal, gordura, carboidrato e os alvos por grupo, em log.
    A densidade e as frações não levam."""
    return [I_M, I_P, I_K, I_F, I_C] + list(range(D.I_MG.start, D.I_PG.stop))


def procedencia4(cfg, D):
    """`treinado_com` e `so_comercial_ok` de uma configuração (13.8): o Nutrition5k e cada base externa de que ela usa
    rótulo, com o hash da lista, grupos, fotos, alvos, licença e uso declarado."""
    tc = [dict(dominio="nutrition5k", lista="treino_v2.csv", sha256=D.hash, grupos=int(D.usa_tr.sum()), fotos=int(D.usa_tr.sum()),
               alvos="todos (22 saídas" + (", frações por fonte e densidade composta)" if cfg["tipo"] == "fracoes4" or cfg.get("fracoes4") else ")"),
               licenca="CC BY 4.0", uso_declarado="comercial_ok")]
    herda = [ch for ch in (cfg.get("herda_massa_de") or []) if ch not in (cfg.get("treina_com") or cfg.get("externo4") or [])]
    for ch in list(cfg.get("treina_com") or cfg.get("externo4") or []) + list(cfg.get("ff_de") or []) + herda:
        d = D.dom.get(ch)
        if d is None:
            tc.append(dict(dominio=ch, ausente=D.dom_fora.get(ch, "sem lista")))
            continue
        if ch in herda:         # H0c-m (rodada 6): o rótulo de massa da base entra pela log-massa da ridge R4c, que é uma entrada da cabeça
            tc.append(dict(dominio=d["dominio"], lista=f"externo_{ch}.csv", sha256=d["sha256_lista"], divisao=f"divisao_r4_{ch}.csv", sha256_divisao=d["sha256_divisao"],
                           grupos=d["resumo"]["grupos_treino"], fotos=d["resumo"]["treino"], alvos="massa total, pela entrada de log-massa da ridge R4c",
                           licenca=d["licenca"], uso_declarado=d["uso_declarado"]))
            continue
        P, pf = alvos_dominio4(d, D, cfg)
        usados = [n for n, c in (("massa", I_M), ("proteína", I_P), ("kcal", I_K), ("densidade", I_D), ("gordura", I_F), ("carboidrato", I_C)) if P[:, c].any()]
        usados += ["massa por grupo"] * bool(P[:, D.I_MG].any()) + ["proteína por grupo"] * bool(P[:, D.I_PG].any()) + ["frações por fonte"] * bool(pf.any())
        tc.append(dict(dominio=d["dominio"], lista=f"externo_{ch}.csv", sha256=d["sha256_lista"], divisao=f"divisao_r4_{ch}.csv", sha256_divisao=d["sha256_divisao"],
                       grupos=d["resumo"]["grupos_treino"], fotos=d["resumo"]["treino"], alvos=", ".join(usados), licenca=d["licenca"], uso_declarado=d["uso_declarado"]))
    return dict(treinado_com=tc, so_comercial_ok=all(x.get("uso_declarado") == "comercial_ok" for x in tc))


def paridade_d1(D, r0, h0, ref, esperado=None, tolerancia=None, contra="D1 da execução 3"):
    """Conferência de paridade de 13.7: as cabeças R0 e H0 repetem as do D1 da execução 3? Correlação das previsões
    fora-do-fold (log-massa da ridge com 1 e 8 orientações; as duas densidades da cabeça de frações, 1 orientação) com as
    gravadas na execução 3 (ref = {dish_id: [4 valores]}), e o critério de CV do D1 (massa da ridge × média em log das
    duas densidades, 1 orientação), que tem de dar 71,3 ± 0,3. Só usa rótulos do treino. Função pura."""
    v = ~np.isnan(r0["oof_o1"][:, I_M]) & ~np.isnan(h0["oof_o1"][:, I_D]) & np.array([d in ref for d in D.ids])
    if v.sum() < 3:
        return dict(passou=False, motivo="sem pratos em comum com a referência da execução 3", n=int(v.sum()))
    R = np.array([ref[d] for d, u in zip(D.ids, v) if u])
    nosso = np.c_[r0["oof_o1"][v, I_M], r0["oof"][v, I_M], h0["oof_o1"][v, I_D], lg_dens(h0["oof_dens_ff_o1"][v])]
    corr = [float(np.corrcoef(nosso[:, j], R[:, j])[0, 1]) for j in range(4)]
    ld = 0.5 * (nosso[:, 2] + nosso[:, 3])
    p = np.exp(nosso[:, 0]) * np.clip(np.exp(ld) - EPS_D, 0, None)
    w = D.w_cv[v]
    crit = float(100 * (acerto(D.proteina[v], p) * w).sum() / w.sum())
    esp, tol = D1_CV if esperado is None else esperado, D1_TOL if tolerancia is None else tolerancia      # rodada 6: a REF da quinta execução (68,34%)
    r = dict(n=int(v.sum()), corr=dict(zip(["log_massa_o1", "log_massa_o8", "log_dens_regressao_o1", "log_dens_fracoes_o1"], corr)),
             dif_abs_mediana=[float(np.median(np.abs(nosso[:, j] - R[:, j]))) for j in range(4)], criterio_cv_d1=crit, esperado=esp, tolerancia=tol, corr_minima=D1_CORR, contra=contra)
    r["passou"] = bool(min(corr) >= D1_CORR and abs(crit - esp) <= tol + 1e-9 and v.sum() == len(D.ids))
    if not r["passou"]:
        r["motivo"] = ("correlação abaixo de %.3f" % D1_CORR if min(corr) < D1_CORR else "critério de CV fora de %.2f ± %.1f" % (esp, tol)
                       if abs(crit - esp) > tol + 1e-9 else "nem todos os pratos do treino_v2 têm previsão (execução parcial)")
    return r


# ------------------------------------------------------------------ rodada 4: fotos das bases externas (download, conferência, cache)
def url_drive4(file_id, ate=None):
    """URL que de fato entrega o arquivo do Google Drive. O Drive pode responder com uma página HTML no lugar do
    arquivo: aviso de arquivo grande (traz um formulário com `confirm` e `uuid`; refaz-se o pedido com eles) ou recusa
    por cota (não há o que fazer: IOError). Só lê o primeiro byte."""
    import re, urllib.error, urllib.parse, urllib.request
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    for _ in range(3):
        if ate and time.time() > ate:
            raise TimeoutError("Google Drive: prazo esgotado")
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.headers.get_content_type() != "text/html":
                    return url
                html = r.read(1 << 20).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise IOError(f"Google Drive respondeu HTTP {e.code} (cota ou acesso)")
        campos = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', html))
        acao = re.search(r'<form[^>]+action="([^"]+)"', html)
        if "confirm" not in campos or not acao:
            motivo = "cota de download excedida" if re.search(r"quota|too many users|muitos usu", html, re.I) else "página sem o formulário de confirmação"
            raise IOError(f"Google Drive não entregou o arquivo: {motivo}")
        url = acao.group(1).replace("&amp;", "&") + "?" + urllib.parse.urlencode({k: v for k, v in campos.items() if k in ("id", "export", "confirm", "uuid", "at", "authuser")})
    raise IOError("Google Drive não entregou o arquivo depois da confirmação")


def acha_no_espelho4(raizes, rel, prof=4):
    """Procura, em pastas anexadas (datasets do Kaggle), uma árvore já extraída que contenha o arquivo `rel` (ou um sufixo
    do caminho dele: o Kaggle pode tirar ou pôr pastas ao extrair um zip). Devolve (pasta, nº de componentes cortados do
    início de rel) ou None."""
    partes = rel.split("/")
    for raiz in raizes:
        if not raiz or not os.path.isdir(raiz):
            continue
        nivel = [raiz]
        for _ in range(prof + 1):
            for d in nivel:
                for corte in range(len(partes)):
                    if os.path.isfile(os.path.join(d, *partes[corte:])):
                        return d, corte
            prox = []
            for d in nivel:
                try:
                    prox += [e.path for e in os.scandir(d) if e.is_dir()]
                except OSError:
                    pass
            nivel = prox[:2000]
    return None


def acha_arquivo4(raizes, nome, bytes_=None, prof=4):
    """Procura um arquivo pelo nome (e pelo tamanho, se dado) em pastas anexadas. Devolve o caminho ou None."""
    for raiz in raizes:
        if not raiz or not os.path.isdir(raiz):
            continue
        for base, dirs, arqs in os.walk(raiz):
            if base[len(raiz):].count(os.sep) >= prof:
                dirs[:] = []
            for a in arqs:
                c = os.path.join(base, a)
                if (a == nome or (bytes_ and a.lower().endswith(os.path.splitext(nome)[1]))) and (not bytes_ or os.path.getsize(c) == bytes_):
                    return c
    return None


def _foto_ok(arq):
    try:
        from PIL import Image
        with Image.open(arq) as im:
            im.verify()
        return True
    except Exception:
        return False


def baixa_fotos4(d, raiz, ate, threads=16):
    """Fotos de um domínio servidas uma a uma (coluna `url`, ou base + caminho): baixa em paralelo, com novas tentativas e
    retomada, e confere cada uma (tamanho e sha256 quando a lista traz; senão, que é uma imagem íntegra). Uma foto que
    falha derruba o domínio: treinar com a lista furada mudaria o conjunto pré-registrado sem aviso."""
    import urllib.parse
    info = DOMINIOS4[d["chave"]]

    def um(j):
        dest = os.path.join(raiz, d["arquivos"][j])
        url = d["url"][j] or info["base"] + urllib.parse.quote(d["arquivos"][j])
        for tentativa in range(3):
            try:
                baixa_arquivo(url, dest, tentativas=6, ate=ate, silencio=True)
                if d["bytes"][j] and os.path.getsize(dest) != d["bytes"][j]:
                    raise IOError("tamanho diferente do da lista")
                if d["sha256"][j] and sha256_arquivo(dest) != d["sha256"][j]:
                    raise IOError("sha256 diferente do da lista")
                if not d["sha256"][j] and not _foto_ok(dest):
                    raise IOError("imagem corrompida")
                return None
            except TimeoutError as e:
                return f"{type(e).__name__}: {e}"
            except Exception as e:
                erro = f"{type(e).__name__}: {e}"
                if os.path.exists(dest):
                    os.remove(dest)
        return erro
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(um, range(len(d["ids"]))))
    falhas = [(d["arquivos"][j], r) for j, r in enumerate(res) if r]
    if falhas:
        raise IOError(f"{len(falhas)} de {len(res)} fotos não baixaram ou não conferem (ex.: {falhas[0]})")


def baixa_dominio4(a, d):
    """Deixa em disco as fotos que a lista de um domínio cita e devolve a pasta. Tudo com novas tentativas, retomada,
    conferência e prazo (a.externo_ate). O que falhar levanta erro: a execução segue sem o domínio.
      tar / targz   um arquivo só (Zenodo, Figshare): baixa, confere tamanho (e md5, se registrado) e extrai só as fotos citadas;
      zip           OmniFood8K: primeiro um espelho anexado ao kernel (árvore extraída ou o próprio zip, em /kaggle/input
                    ou em --espelho); senão o Google Drive, com a confirmação de arquivo grande;
      url           uma foto por pedido (Mendeley, GitHub, Hugging Face)."""
    import zipfile
    ch, info, ate = d["chave"], DOMINIOS4[d["chave"]], getattr(a, "externo_ate", None)
    base = os.path.join(a.cache, "ext4", ch)
    raiz = os.path.join(base, "img")
    os.makedirs(raiz, exist_ok=True)
    falta = lambda: [f for f in d["arquivos"] if not os.path.exists(os.path.join(raiz, f))]
    if not falta():
        return raiz
    t0 = time.time()
    espelhos = [getattr(a, "espelho", None), "/kaggle/input"]
    if info["origem"] == "url":
        baixa_fotos4(d, raiz, ate)
        diz(f"base externa {ch}: {len(d['ids'])} fotos baixadas e conferidas em {time.time() - t0:.0f}s ({info['de']})")
        return raiz
    quer = set(d["arquivos"])

    def grava(nome, fo):
        dest = os.path.join(raiz, nome)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".parte", "wb") as fd:
            shutil.copyfileobj(fo, fd, 1 << 22)
        os.replace(dest + ".parte", dest)
    if info["origem"] == "zip":
        achado = acha_no_espelho4(espelhos, d["arquivos"][0])
        if achado:                              # árvore já extraída num dataset anexado: usa no lugar, sem copiar
            pasta, corte = achado
            de = lambda f: os.path.join(pasta, *f.split("/")[corte:])
            faltam = [f for f in d["arquivos"] if not os.path.isfile(de(f))]
            if faltam:
                raise IOError(f"espelho em {pasta} sem {len(faltam)} fotos da lista (ex.: {faltam[0]})")
            d["raiz_espelho"] = (pasta, corte)
            diz(f"base externa {ch}: {len(quer)} fotos achadas no espelho {pasta} (sem download)")
            return pasta
        arq = acha_arquivo4(espelhos, info["arquivo"], info["bytes"])
        if arq:
            diz(f"base externa {ch}: usando o zip do espelho {arq}")
        else:
            arq = os.path.join(base, info["arquivo"])
            baixa_arquivo(url_drive4(info["drive_id"], ate), arq, ate=ate)
            if os.path.getsize(arq) != info["bytes"] or not zipfile.is_zipfile(arq):
                os.remove(arq)
                raise IOError(f"{info['arquivo']}: tamanho diferente do registrado ou não é um zip (o Google Drive pode ter entregue uma página); arquivo apagado")
            diz(f"base externa {ch}: {os.path.getsize(arq) / 2 ** 30:.2f} GB baixados do Google Drive em {time.time() - t0:.0f}s")
        with zipfile.ZipFile(arq) as z:         # a leitura confere o CRC de cada foto
            nomes = set(z.namelist())
            sem = [f for f in quer if f not in nomes]
            if sem:
                raise IOError(f"{info['arquivo']} sem {len(sem)} fotos da lista (ex.: {sem[0]})")
            for f in sorted(quer):
                if not os.path.exists(os.path.join(raiz, f)):
                    with z.open(f) as fo:
                        grava(f, fo)
    else:                                        # tar ou tar.gz
        arq = acha_arquivo4(espelhos, info["arquivo"], info["bytes"]) or os.path.join(base, info["arquivo"])
        if not os.path.exists(arq):
            baixa_arquivo(info["url"], arq, ate=ate)
        if os.path.getsize(arq) != info["bytes"] or (info.get("md5") and md5_arquivo(arq) != info["md5"]):
            if arq.startswith(base):
                os.remove(arq)
            raise IOError(f"{info['arquivo']}: tamanho ou md5 diferente do registrado; arquivo apagado para baixar de novo")
        diz(f"base externa {ch}: {os.path.getsize(arq) / 2 ** 30:.2f} GB baixados e conferidos em {time.time() - t0:.0f}s ({info['de']})")
        por_nome = {os.path.basename(f): f for f in quer}       # o caminho dentro do .tar pode vir com prefixo ('./')
        with tarfile.open(arq, "r:gz" if info["origem"] == "targz" else "r:") as t:
            for m in t:
                f = m.name[2:] if m.name.startswith("./") else m.name
                f = f if f in quer else por_nome.get(os.path.basename(f))
                if m.isfile() and f and not os.path.exists(os.path.join(raiz, f)):
                    with t.extractfile(m) as fo:
                        grava(f, fo)
                if ate and time.time() > ate + 600:
                    raise TimeoutError(f"extração de {info['arquivo']} passou do prazo")
    if falta():
        raise IOError(f"{len(falta())} fotos da lista não estão em {info['arquivo']} (ex.: {falta()[0]})")
    diz(f"base externa {ch}: {len(quer)} fotos extraídas ({time.time() - t0:.0f}s desde o início)")
    return raiz


def pasta_dom4(C, d):
    if getattr(C, "prato", False):          # rodada 5, configurações sobre o recorte: o cache dos recortes da base
        return pasta_prato5(C, d)
    return os.path.join(C.raiz, f"ext4_{d['chave']}_{d['hash']}")


def prepara_dominio4(a, D, ch):
    """Fotos de um domínio externo prontas no cache (uint8 640×480, como as do Nutrition5k). Entrada (13.4): foto em
    retrato é girada para paisagem; a que não é 4:3 é encaixada no centro de um quadro 4:3 com a cor mediana da sua
    borda, nunca recortada. Com --externo-imagens/<chave>/ as fotos já estão em disco e nada é baixado."""
    d = D.dom[ch]
    C = Cache(a.cache, D)
    pasta = pasta_dom4(C, d)
    if os.path.exists(os.path.join(pasta, "pronto.json")):
        return
    raiz = d["raiz"] if d["raiz"] and os.path.isdir(d["raiz"]) else baixa_dominio4(a, d)
    corte = d.get("raiz_espelho", (None, 0))[1]
    de = lambda f: acha_foto(raiz, "/".join(f.split("/")[corte:]))
    falta = [f for f in d["arquivos"] if not de(f)]
    if falta:
        raise IOError(f"{len(falta)} fotos da lista de {ch} não estão em {raiz} (ex.: {falta[:2]})")
    os.makedirs(pasta, exist_ok=True)
    from PIL import Image, ImageOps
    t0, n = time.time(), len(d["ids"])
    mm = np.lib.format.open_memmap(os.path.join(pasta, "rgb.tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, C.H, C.W, 3))
    cor = np.zeros((n, 3), np.float32)
    giradas = [0]

    def um(i):
        im = ImageOps.exif_transpose(Image.open(de(d["arquivos"][i]))).convert("RGB")     # foto de celular: a orientação gravada no EXIF vale
        if im.size[1] > im.size[0]:             # retrato -> paisagem
            im = im.transpose(Image.ROTATE_90); giradas[0] += 1
        z = min(C.W / im.size[0], C.H / im.size[1])
        pq = np.asarray(im if im.size == (C.W, C.H) else im.resize((max(1, round(im.size[0] * z)), max(1, round(im.size[1] * z))), Image.BICUBIC))
        kb = max(2, int(0.03 * min(pq.shape[:2])))
        cor[i] = np.median(np.concatenate([pq[:kb].reshape(-1, 3), pq[-kb:].reshape(-1, 3), pq[:, :kb].reshape(-1, 3), pq[:, -kb:].reshape(-1, 3)]), 0) / 255
        tela = np.empty((C.H, C.W, 3), np.uint8); tela[:] = (cor[i] * 255).astype(np.uint8)
        y0, x0 = (C.H - pq.shape[0]) // 2, (C.W - pq.shape[1]) // 2
        tela[y0:y0 + pq.shape[0], x0:x0 + pq.shape[1]] = pq
        mm[i] = tela
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as ex:
        list(ex.map(um, range(n)))
    mm.flush(); del mm
    np.save(os.path.join(pasta, "cor.npy"), cor)
    os.replace(os.path.join(pasta, "rgb.tmp.npy"), os.path.join(pasta, "rgb.npy"))
    grava_json(os.path.join(pasta, "pronto.json"), dict(n=n, hash=d["hash"], giradas=giradas[0], seg=round(time.time() - t0, 1)))
    diz(f"cache das fotos de {ch}: {n} fotos em {time.time() - t0:.0f}s ({giradas[0]} em retrato, giradas)")


def abre_dominios4(C, D, so=None):
    """Abre o cache do Nutrition5k e o de cada domínio externo que já está pronto; as fotos externas ganham posições
    depois das do Nutrition5k (C.pos). Devolve as chaves abertas."""
    C.abre()
    C.blocos, abertos, ini = [], [], len(C.rgb)
    for ch, d in D.dom.items():
        pasta = pasta_dom4(C, d)
        if (so is None or ch in so) and os.path.exists(os.path.join(pasta, "pronto.json")):
            rgb = np.load(os.path.join(pasta, "rgb.npy"), mmap_mode="r")
            assert len(rgb) == len(d["ids"])
            C.blocos.append((ini, rgb, np.load(os.path.join(pasta, "cor.npy"))))
            C.pos.update({x: ini + i for i, x in enumerate(d["ids"])})
            d["pos0"] = ini
            ini += len(rgb); abertos.append(ch)
    return abertos


def feats_dom4(bb, D, C, ch, dev, bs, prazo):
    """Features congeladas das fotos de um domínio externo (13.4): 2 orientações (original e 180°) nas bases de uma foto
    por prato; 1 por quadro no OmniFood8K e no SNAPMe. [fotos, orientações, 2·dim] float16, no cache do domínio."""
    d = D.dom[ch]
    arq = os.path.join(pasta_dom4(C, d), f"feat__{bb}.npy")
    with Trava(arq + ".trava"):
        if not os.path.exists(arq):
            _extrai_feats(arq, bb, D, C, dev, bs, prazo, [(d["pos0"], d["pos0"] + len(d["ids"]))], base=d["pos0"], n_or=DOMINIOS4[ch]["orient"])
    return np.load(arq)


# ------------------------------------------------------------------ rodada 4: cabeças com amostras externas
class RidgeMulti(RidgeSub):
    """RidgeSub com amostras de treino de vários domínios externos (13.4). Cada domínio traz features [fotos, orientações,
    D], o alvo [fotos, alvos] e, por foto, os folds em que ela NÃO pode treinar e o seu grupo. No modelo treinado com os
    folds `folds` do Nutrition5k entram as fotos externas que não são proibidas em nenhum fold deixado de fora; dentro
    do domínio o peso é igual por grupo (repartido entre as fotos e as orientações) e a soma é `frac` × a do
    Nutrition5k; do alvo sai o viés do domínio (média ponderada do domínio − média do Nutrition5k nesse treino).
    Padronização e ridge ponderados, como em RidgeSubExt. As previsões saem cruas (o viés não volta)."""

    def __init__(self, X, T, fold, doms, dev=None, sem_vies=False, todos=None):
        """doms = [dict(chave, X [m, o, D], t [m, alvos], d = o domínio (pesos_dominio4), frac)]. todos: os folds do
        Nutrition5k (rodada 6: o conjunto de linhas pode não ter algum fold, e as fotos externas proibidas nos folds
        deixados de fora continuam fora)."""
        super().__init__(X, T, fold, dev)
        self.sem_vies = sem_vies
        self.doms = doms
        self.todos = frozenset(todos) if todos is not None else frozenset(int(k) for k in set(self.fold.tolist()))
        for e in doms:
            e["Xt"] = self.t.from_numpy(np.ascontiguousarray(e["X"], np.float32)).to(self.dev)

    def dec(self, folds):
        if folds not in self.cache:
            t, i = self.t, self._m(np.isin(self.fold, list(folds)))
            o, Dm = self.X.shape[1:]
            fora = sorted(self.todos - set(folds))      # rodada 6 (H0c-m aninhada): até dois folds de fora
            Xs, Ts, ws, deltas = [self.X[i].reshape(-1, Dm)], [self.T[i].repeat_interleave(o, 0)], [t.ones(len(i) * o, device=self.dev, dtype=t.float64)], {}
            for e in self.doms:
                w = pesos_dominio6(e["d"], fora, e["frac"] * len(i) * o)
                j = np.where(w > 0)[0]
                if not len(j):
                    continue
                oe = e["Xt"].shape[1]
                wj = t.from_numpy(w[j] / oe).to(self.dev)
                te = t.from_numpy(np.ascontiguousarray(e["t"][j], np.float64)).to(self.dev)
                delta = (te * wj[:, None]).sum(0) / wj.sum() - self.T[i].mean(0)
                if self.sem_vies:       # rodada 5, recorte: sem viés por domínio (14.3)
                    delta = delta * 0
                deltas[e["chave"]] = delta.cpu().numpy()
                Xs.append(e["Xt"][t.from_numpy(j).to(self.dev)].reshape(-1, Dm)); Ts.append((te - delta).repeat_interleave(oe, 0)); ws.append(wj.repeat_interleave(oe))
            Xt, Tt, w = t.cat(Xs), t.cat(Ts), t.cat(ws)
            wf = w.float()
            mu = (Xt * wf[:, None]).sum(0) / wf.sum()
            sd = (((Xt - mu) ** 2 * wf[:, None]).sum(0) / wf.sum()).sqrt() + 1e-6
            mt = (Tt * w[:, None]).sum(0) / w.sum()
            Z = (Xt - mu) / sd * wf.sqrt()[:, None]
            s2, V = t.linalg.eigh((Z.T @ Z).double())
            self.cache[folds] = (mu, sd, mt, s2.clamp(min=0), V, V.T @ (Z.double().T @ ((Tt - mt) * w.sqrt()[:, None])), deltas)
        return self.cache[folds]

    def vies(self, folds):
        return self.dec(folds)[6]


def ext_cabeca4(D, cfg, feats, fracs, k, T_n5k):
    """Amostras externas para treinar a cabeça de frações no modelo que deixa o fold k de fora (k = None: final), no
    formato de treina_cabeca com linhas prontas: X [M, D] (uma linha por foto × orientação), T [M, 22] com NaN onde não
    há rótulo e já sem o viés do domínio nos alvos extensivos, FF, peso por alvo [M, 22], peso das frações [M] e peso por
    amostra c [M] (a soma de cada domínio é `frac` × a do Nutrition5k, que é nº de pratos × 8 orientações). Devolve
    (dict ou None, {domínio: viés [22]})."""
    Xs, Ts, Fs, Ps, Pfs, cs, vies = [], [], [], [], [], [], {}
    ext_cols = colunas_extensivas4(D)
    for ch in list(cfg.get("treina_com", [])) + list(cfg.get("ff_de", [])):
        d = D.dom[ch]
        w = pesos_dominio4(d, k, fracs[ch] * len(T_n5k) * 8)
        j = np.where(w > 0)[0]
        if not len(j):
            continue
        P, pf = alvos_dominio4(d, D, cfg)
        T = d["T"][j].copy()
        T[:, P[j].max(0) == 0] = np.nan             # alvo que a configuração não usa neste domínio: fora
        delta = np.zeros(D.N)
        with np.errstate(invalid="ignore"):
            m = np.nansum(T[:, ext_cols] * w[j, None], 0) / np.maximum((np.isfinite(T[:, ext_cols]) * w[j, None]).sum(0), 1e-12)
        tem = np.isfinite(T[:, ext_cols]).any(0)
        if not cfg.get("sem_vies"):     # rodada 5, recorte: sem viés por domínio (14.3)
            delta[np.array(ext_cols)[tem]] = (m - T_n5k[:, ext_cols].mean(0))[tem]
        vies[ch] = delta
        oe = feats[ch].shape[1]
        rep = lambda A: np.repeat(A, oe, 0)
        Xs.append(feats[ch][j].reshape(-1, feats[ch].shape[-1]).astype(np.float32)); Ts.append(rep(T - delta)); Fs.append(rep(d["FF"][j] if pf[j].any() else np.full_like(d["FF"][j], np.nan)))
        Ps.append(rep(P[j])); Pfs.append(rep(pf[j])); cs.append(rep(w[j] / oe))
    if not Xs:
        return None, vies
    return dict(X=np.concatenate(Xs), T=np.concatenate(Ts), FF=np.concatenate(Fs), peso_alvo=np.concatenate(Ps), peso_ff=np.concatenate(Pfs),
                c=np.concatenate(cs).astype(np.float32)), vies


def alfa_aux4(g_rot, g_aux):
    """α de 13.7: min(1, ‖∂L_rot/∂f‖ ÷ ‖∂(L_frac + L_dens)/∂f‖), com as normas tiradas no lote. Com ele o gradiente auxiliar
    que chega às features nunca passa do gradiente dos rótulos."""
    return float(min(1.0, float(g_rot.norm()) / (float(g_aux.norm()) + 1e-12)))


def feats_parcial4(rede, x, n, ckpt=False):
    """Features [agregado ‖ média dos tokens] com os blocos abaixo dos n últimos sem gradiente (P1). É a mesma conta de
    Rede.feats; a igualdade é conferida no começo do treino."""
    import torch
    m = rede.m
    with torch.no_grad():
        t = m.patch_embed((x - rede.media) / rede.desvio)
        t = m.norm_pre(m.patch_drop(m._pos_embed(t)))
        for b in m.blocks[:-n]:
            t = b(t)
    for b in m.blocks[-n:]:
        t = torch.utils.checkpoint.checkpoint(b, t, use_reentrant=False) if ckpt and t.requires_grad else b(t)
    t = m.norm(t)
    return torch.cat([m.forward_head(t, pre_logits=True), t[:, m.num_prefix_tokens:].mean(1)], 1)


def cria_rede4(cfg, D, pretrained=True, treino=False):
    """Rede de um ajuste fino da rodada 4: a de cria_rede com 22 saídas e, se a configuração tem cabeça de frações, uma
    segunda camada linear `head_ff` (10 logits, a partir de zero). No ajuste fino parcial, só os últimos blocos, a norma
    final, o agrupador e as cabeças ficam com gradiente."""
    import torch
    rede = cria_rede(cfg["bb"], D.N, pretrained=pretrained, drop_path=cfg["drop_path"] if treino else 0.0, ckpt_grad=bool(treino and cfg["ckpt_grad"] and not cfg.get("blocos_treinaveis")))
    if cfg.get("fracoes4"):
        rede.head_ff = torch.nn.Linear(rede.dim, len(D.fontes))
        with torch.no_grad():
            rede.head_ff.weight.zero_(); rede.head_ff.bias.zero_()
    n = cfg.get("blocos_treinaveis")
    if n:
        L = len(rede.m.blocks)
        for nome, p in rede.named_parameters():
            topo = nome.startswith(("head", "m.norm.", "m.fc_norm", "m.attn_pool")) or (nome.startswith("m.blocks.") and int(nome.split(".")[2]) >= L - n)
            p.requires_grad_(bool(topo))
    return rede


def treina4(cfg, D, C, tr, va, dev, prazo, div_lote=1, semente=0, rotulo="", k=None, ini=None, fotos_epoca=None):
    """Ajuste fino da rodada 4 (13.7). tr/va: índices em D.ids; k: fold deixado de fora (None: modelo final).
      F3 (externo4): a receita do U0 sem mudança (22 saídas, uma perda) mais, por época, um sorteio de fotos externas
        igual à metade dos pratos do treino, repartido entre os domínios na proporção dos pesos de 13.4, com a perda de
        cada alvo mascarada e pesada e o viés por domínio retirado.
      F1, F2, P1 (fracoes4): perda = L_rot + α × (L_frac + L_dens), com α recalculado a cada passo (alfa_aux4), sem
        gradiente e suavizado por média móvel de 20 passos; logits das frações a partir de zero, com passo 10 × o da
        cabeça; backbone parado na primeira época. F2: peso por faixa em todas as perdas. P1 (blocos_treinaveis): só os
        últimos blocos, a norma final, o agrupador e as cabeças treinam; os outros blocos ficam em modo de avaliação.
      P3 (rodada 6, 15.4): `ini` = alfas da R0c e da R4c refeitas: a cabeça começa na ridge R4c refeita só com as linhas
        de treino deste modelo (sem a validação interna); `aquec_cabeca` épocas só de cabeça (otimizador próprio, passo
        `lr_cabeca`) antes das `epocas` com os blocos liberados; `parada_interna`: 10% das cadeias de treino do fold
        (sha1(cadeia) mod 10 == 0) ficam de fora e o RMSE de log-massa delas (1 orientação, média móvel) na época 0 e no
        fim de cada época escolhe o checkpoint (a época 0 concorre). No modelo final (k = None), `fotos_epoca` = pasta onde
        vai a média móvel dos tensores treinados no fim de cada época (a escolha da época é feita depois, pelos folds).
    A cabeça fica em float32, fora do autocast (o backbone, em meia precisão). Sem destilação.
    Devolve (rede com a média móvel dos pesos, função de previsão, histórico por época, (média, desvio), extras)."""
    import copy, torch
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    hw = BACKBONES[cfg["bb"]]["hw"]
    quad = hw[0] == hw[1]
    bs = max(1, min(cfg["bs"] // div_lote, len(tr))); acum = cfg["acum"] * div_lote
    T, N = D.T, D.N
    K = len(D.fontes) if cfg.get("fracoes4") else 0
    n_par = cfg.get("blocos_treinaveis")
    interna = np.zeros(len(tr), bool)
    if cfg.get("parada_interna") and k is not None:     # rodada 6: validação interna por cadeia, dentro das linhas de treino do fold
        interna = np.array([particao_interna6(D.cadeia[i]) for i in tr], bool)
    tr_int, tr = tr[interna], tr[~interna]
    tm, ts = T[tr].mean(0), T[tr].std(0) + 1e-6
    info_ini = {}
    if ini is not None:         # P3: a ridge R4c refeita com as linhas deste modelo (15.4)
        A_ini, b_ini, info_ini = ini_r4c6(cfg, D, C, tr, k, dev, prazo, ini, tr_int)
        X0_sg = None
    else:
        Xc = feats_congeladas(cfg["bb"], D, C, dev, max(1, cfg["bs"] // div_lote), prazo, so_treino=True)
        Xr = Xc[[C.pos[D.ids[i]] for i in tr]].astype(np.float32)
        _, alfa, _ = ridge_cv(Xr, T[tr], D.fold[tr])
        _, m = ridge_final(Xr, T[tr], alfa)
        A_ini, b_ini = m.W / m.sd[:, None], m.mt - (m.mu / m.sd) @ m.W
        X0_sg = Xr if cfg.get("fracoes_sg") else None      # rodada 5: padronização da cabeça de frações sem gradiente
        del Xc, Xr
    rede = cria_rede4(cfg, D, treino=True).to(dev)
    with torch.no_grad():       # as 22 saídas partem da ridge sobre o backbone congelado; os logits das frações, de zero
        rede.head.weight.copy_(torch.from_numpy(((A_ini / ts).T).astype(np.float32)))
        rede.head.bias.copy_(torch.from_numpy(((b_ini - tm) / ts).astype(np.float32)))
    feats = (lambda r, x: feats_parcial4(r, x, n_par, cfg["ckpt_grad"])) if n_par else (lambda r, x: r.feats(x))
    if n_par:                   # a conta dividida tem de ser a mesma de Rede.feats
        rede.eval()
        with torch.inference_mode(), autocast(dev):
            x0 = monta_lote(C, [C.pos[D.ids[i]] for i in tr[:2]], dev, hw)
            dif = float((feats_parcial4(rede, x0, n_par).float() - rede.feats(x0).float()).abs().max())
        assert dif < 5e-2, f"feats_parcial4 difere de Rede.feats ({dif:.3g}): a versão do timm mudou a ordem das operações"
    ema = copy.deepcopy(rede).eval()
    ema.m.set_grad_checkpointing(False)
    for p in ema.parameters():
        p.requires_grad_(False)
    treinaveis = [(pe, p) for pe, p in zip(ema.parameters(), rede.parameters()) if p.requires_grad]
    # grupos de lr: decaimento por camada; logits das frações com passo 10 ×; backbone (tudo menos as cabeças) parado na 1ª época
    grupos, e_backbone = [], []
    ff_ids = {id(p) for p in rede.head_ff.parameters()} if K else set()
    cab_ids = ff_ids | {id(p) for p in rede.head.parameters()}
    for g in grupos_lr(rede, cfg["lr"], cfg["lld"], cfg["wd"]):
        for tipo, ps in (("ff", [p for p in g["params"] if id(p) in ff_ids]), ("cabeca", [p for p in g["params"] if id(p) in cab_ids - ff_ids]),
                         ("backbone", [p for p in g["params"] if id(p) not in cab_ids])):
            if ps:
                grupos.append(dict(params=ps, lr=g["lr"] * (10 if tipo == "ff" else 1), weight_decay=g["weight_decay"])); e_backbone.append(tipo == "backbone")
    opt = torch.optim.AdamW(grupos)
    # fotos externas (F3)
    doms = [ch for ch in cfg.get("externo4", []) if ch in D.dom]
    ext, vies = {}, {}
    if doms:
        assert doms == list(cfg["externo4"]), "domínio de treino ausente"
        fr = fracoes_dominio4({ch: D.dom[ch]["grupos_treino_lista"] for ch in doms}, len(D.ids), doms)
        n_ext_ep = max(len(doms), int(round(0.5 * len(tr))))
        cotas = {ch: max(1, int(round(n_ext_ep * fr[ch] / sum(fr.values())))) for ch in doms}
        ext_cols = colunas_extensivas4(D)
        base, Te, Pe, lin_e = len(D.ids), [], [], []
        for ch in doms:
            d = D.dom[ch]
            w = pesos_dominio4(d, k, 1.0)
            j = np.where(w > 0)[0]
            P, _ = alvos_dominio4(d, D, cfg)
            Td = d["T"][j].copy(); Td[:, P[j].max(0) == 0] = np.nan
            delta = np.zeros(N)
            with np.errstate(invalid="ignore"):
                md = np.nansum(Td[:, ext_cols] * w[j, None], 0) / np.maximum((np.isfinite(Td[:, ext_cols]) * w[j, None]).sum(0), 1e-12)
            tem = np.isfinite(Td[:, ext_cols]).any(0)
            if not cfg.get("sem_vies"):     # rodada 6 (P3, F5c), sobre o recorte: sem viés por domínio (15.4)
                delta[np.array(ext_cols)[tem]] = (md - T[tr][:, ext_cols].mean(0))[tem]
            vies[ch] = delta
            ext[ch] = dict(ini=base + len(lin_e), n=len(j), p=w[j] / w[j].sum())
            Te.append(Td - delta); Pe.append(P[j]); lin_e += [C.pos[d["ids"][i]] for i in j]
        T_tudo = np.r_[T, np.concatenate(Te)]
        mascara = np.r_[np.ones_like(T), np.isfinite(np.concatenate(Te)) * np.concatenate(Pe) * (ts > 1e-3)].astype(np.float32)
        linha = np.array([C.pos.get(d, -1) for d in D.ids] + lin_e)
    else:
        n_ext_ep, T_tudo, mascara = 0, T, None
        linha = np.array([C.pos.get(d, -1) for d in D.ids])
    n_ep = len(tr) + (sum(cotas.values()) if doms else 0)
    n_cab = int(cfg.get("aquec_cabeca") or 0)          # P3: épocas só de cabeça antes das de ajuste fino (15.4)
    passos_ep = max(1, n_ep // (bs * acum)); total = cfg["epocas"] * passos_ep; total_tudo = (n_cab + cfg["epocas"]) * passos_ep
    n_aq = max(1, int(cfg["aquec"] * total))
    curva = lambda s: (s + 1) / n_aq if s < n_aq else 0.5 * (1 + math.cos(math.pi * (s - n_aq) / max(1, total - n_aq)))
    parado = bool(K)            # backbone parado na primeira época (só nas configurações com cabeça de frações)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [(lambda s: 0.0 if s < passos_ep else curva(s)) if (b and parado) else curva for b in e_backbone])
    opt_c = torch.optim.AdamW(list(rede.head.parameters()), lr=cfg["lr_cabeca"], weight_decay=0.0) if n_cab else None
    try:
        scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    pa = cfg.get("pesos_alvo")         # P2 (14.4): pesos próprios por alvo, média ponderada
    w = torch.from_numpy(pesos_alvo5(D, pa) if pa else pesos_perda(D, N)).to(dev)
    sg = CabecaSG(X0_sg, D, tr, dev, semente, total * acum) if X0_sg is not None else None
    del X0_sg
    tm_t, ts_t = torch.from_numpy(tm.astype(np.float32)).to(dev), torch.from_numpy(ts.astype(np.float32)).to(dev)
    huber = torch.nn.functional.smooth_l1_loss
    Yn = torch.from_numpy(np.nan_to_num((T_tudo - tm) / ts).astype(np.float32)).to(dev)
    Mk = None if mascara is None else torch.from_numpy(mascara).to(dev)
    w_am = np.ones(len(T_tudo), np.float32)
    if cfg.get("peso_faixa"):
        w_am[tr] = peso_faixa(D.proteina[tr])
    Wam = torch.from_numpy(w_am).to(dev) if cfg.get("peso_faixa") else None
    dc = None
    if K:
        dc = dens_por_fonte(D.FF[tr], D.massa[tr], D.proteina[tr])
        dct = torch.from_numpy(dc.astype(np.float32)).to(dev)
        FFt = torch.from_numpy(D.FF.astype(np.float32)).to(dev)

    def cabecas(r, f):
        """22 saídas padronizadas e logits das frações, em float32."""
        f = f.float()
        return r.head(f), (r.head_ff(f) if K else None)

    def preve(ids, n_or=8, variantes=False):
        """Previsões em log com os pesos da média móvel. variantes: ({sufixo: 22 saídas}, {sufixo: frações} ou None) para
        1, 4 e 8 orientações; senão (22 saídas, frações ou None) com a média das n_or primeiras orientações."""
        nk = 8 if variantes else n_or
        out = np.zeros((len(ids), nk, N), np.float32)
        Kp = K or (len(D.fontes) if sg is not None else 0)
        fr = np.zeros((len(ids), nk, Kp), np.float32) if Kp else None
        o_sg = np.zeros((len(ids), nk, N), np.float32) if sg is not None else None
        with torch.inference_mode():
            for i in range(0, len(ids), bs):
                x = monta_lote(C, [C.pos[d] for d in ids[i:i + bs]], dev, hw)
                for q in range(nk):
                    with autocast(dev):
                        f = ema.feats(orienta(x, q))
                    o, lg = cabecas(ema, f)
                    out[i:i + len(x), q] = (o * ts_t + tm_t).cpu().numpy()
                    if K:
                        fr[i:i + len(x), q] = torch.softmax(lg, 1).cpu().numpy()
                    elif sg is not None:        # frações da cabeça sem gradiente (rodada 5)
                        p_sg, q_sg = sg.preve(f)
                        o_sg[i:i + len(x), q], fr[i:i + len(x), q] = p_sg.cpu().numpy(), q_sg.cpu().numpy()
        if not variantes:
            return out.mean(1), (fr.mean(1) if Kp else None)
        if sg is not None:
            preve.ultima_sg = {suf: media_or(o_sg, q) for suf, q in ORIENT.items()}
        return {suf: media_or(out, q) for suf, q in ORIENT.items()}, ({suf: media_or(fr, q) for suf, q in ORIENT.items()} if Kp else None)

    def valida():
        pv = preve([D.ids[i] for i in va], 1)[0]; P = proteinas(pv); wv = D.w_cv[va]
        cr = lambda est: float(100 * (acerto(D.proteina[va], est) * wv).sum() / wv.sum())
        return dict(val_rmse_logm=float(np.sqrt(np.mean((pv[:, I_M] - T[va, I_M]) ** 2))), val_crit_direto=cr(P["direto"]), val_crit_mxd=cr(P["mxd"]))

    val_cada = cfg.get("val_cada", 5)
    hist, passo, t0 = [], 0, time.time()
    nomes_tr = [nm for nm, p in rede.named_parameters() if p.requires_grad]

    def int_rmse():
        """RMSE de log-massa na validação interna (1 orientação, média móvel dos pesos)."""
        pv = preve([D.ids[i] for i in tr_int], 1)[0]
        return float(np.sqrt(np.mean((pv[:, I_M] - T[tr_int, I_M]) ** 2)))

    def foto(ep):
        """Os tensores treinados da média móvel, na CPU (o checkpoint da época ep)."""
        sd_ = ema.state_dict()
        return {nm: sd_[nm].detach().float().cpu().clone() for nm in nomes_tr}

    def grava_foto(ep):
        torch.save(dict(estado={nm: v.half() for nm, v in foto(ep).items()}, tm=tm.tolist(), ts=ts.tolist(), epoca=ep), os.path.join(fotos_epoca, f"ep{ep}.pt.tmp"))
        os.replace(os.path.join(fotos_epoca, f"ep{ep}.pt.tmp"), os.path.join(fotos_epoca, f"ep{ep}.pt"))
    melhor = None
    if va is not None and len(va):
        hist.append(dict(epoca=0, **valida()))
        if len(tr_int):
            hist[-1]["int_rmse_logm"] = int_rmse()
            melhor = (hist[-1]["int_rmse_logm"], 0, foto(0))
        print(f"{rotulo} época 0 (cabeça = ridge): {hist[-1]}", flush=True)
    if fotos_epoca:
        os.makedirs(fotos_epoca, exist_ok=True)
        grava_foto(0)
    t_ref, p_ref, alfas = time.time(), 0, []
    for ep in range(n_cab + cfg["epocas"]):
        rede.train()
        if n_par:
            for b in rede.m.blocks[:-n_par]:
                b.eval()
        so_cabeca = (parado and ep == 0) or ep < n_cab
        if doms:        # sorteio das fotos externas da época: por domínio, com probabilidade igual por grupo
            amostras = np.r_[tr, np.concatenate([e["ini"] + rng.choice(e["n"], cotas[ch], replace=cotas[ch] > e["n"], p=e["p"]) for ch, e in ext.items()])]
        else:
            amostras = tr
        perm = rng.permutation(len(amostras)); soma = dict(perda=[], L_rot=[], L_frac=[], L_dens=[], alfa=[])
        for b in range(passos_ep):
            for a_ in range(acum):
                j = amostras[perm[(b * acum + a_) * bs:(b * acum + a_ + 1) * bs]]
                ks = rng.integers(0, 8, len(j)) if quad else rng.integers(0, 4, len(j)) + 4 * int(rng.integers(0, 2))
                zoom = rng.uniform(*cfg["escala"], len(j)) if cfg["escala"] else None
                enq = desvios5(rng, len(j), cfg["enquadramento"]) if cfg.get("enquadramento") else None      # rodada 6: erro de enquadramento (15.4)
                x = orienta_lote(monta_lote(C, linha[j], dev, hw, 3, zoom, enquadramento=enq), ks)
                jt = torch.from_numpy(j).to(dev)
                if so_cabeca:
                    with torch.no_grad(), autocast(dev):
                        f = feats(rede, x)
                    f = f.float().requires_grad_(True)
                else:
                    with autocast(dev):
                        f = feats(rede, x)
                    f = f.float()
                o22, lg = cabecas(rede, f)
                h = huber(o22, Yn[jt], reduction="none") * w
                if Mk is not None:
                    h = h * Mk[jt]
                if pa:
                    L_rot = ((h.sum(1) / w.sum()) * (Wam[jt] if Wam is not None else 1.0)).mean()
                else:
                    L_rot = (h.mean(1) * Wam[jt]).mean() if Wam is not None else h.mean()
                if K:
                    wj = Wam[jt] if Wam is not None else 1.0
                    fr = torch.softmax(lg, 1)
                    L_frac = (((fr - FFt[jt]) ** 2).sum(1) * wj).mean()
                    alvo_d = Yn[jt][:, I_D] * ts_t[I_D] + tm_t[I_D]
                    L_dens = (huber((torch.log(fr @ dct + EPS_D) - alvo_d) / ts_t[I_D], torch.zeros_like(alvo_d), reduction="none") * wj).mean()
                    g_rot, g_aux = torch.autograd.grad(L_rot, f, retain_graph=True)[0], torch.autograd.grad(L_frac + L_dens, f, retain_graph=True)[0]
                    alfas.append(alfa_aux4(g_rot, g_aux))
                    al = float(np.mean(alfas[-20:]))
                    perda = L_rot + al * (L_frac + L_dens)
                    soma["L_frac"].append(float(L_frac.detach())); soma["L_dens"].append(float(L_dens.detach())); soma["alfa"].append(al)
                else:
                    perda = L_rot
                if not torch.isfinite(perda):
                    raise RuntimeError(f"perda não finita no passo {passo} (época {ep + 1})")
                scaler.scale(perda / acum).backward()
                soma["perda"].append(float(perda.detach())); soma["L_rot"].append(float(L_rot.detach()))
                if sg is not None:
                    sg.passo(f, jt)
            o_ = opt_c if (n_cab and ep < n_cab) else opt      # P3: nas épocas só de cabeça, o otimizador da cabeça (passo fixo)
            scaler.unscale_(o_)
            torch.nn.utils.clip_grad_norm_([p for _, p in treinaveis], 1.0)
            scaler.step(o_); scaler.update(); opt.zero_grad(set_to_none=True)
            if o_ is opt:
                sched.step()
            else:
                opt_c.zero_grad(set_to_none=True)
            passo += 1
            dk = min(cfg["ema"], (1 + passo) / (10 + passo))
            with torch.no_grad():
                for pe, p in treinaveis:
                    pe.lerp_(p, 1 - dk)
            if not so_cabeca and (passo - p_ref == min(total - p_ref, max(3, total // 20)) or passo % 50 == 0):   # projeção de tempo: cabe no orçamento?
                por_passo = (time.time() - t_ref) / max(1, passo - p_ref)
                fim = time.time() + (total_tudo - passo) * por_passo * 1.1 + 90
                if fim > prazo:
                    print(f"{rotulo} sem orçamento: faltariam {(fim - time.time()) / 60:.0f} min e restam {(prazo - time.time()) / 60:.0f} min", flush=True)
                    raise SemOrcamento()
        if so_cabeca:           # a época só de cabeça é mais rápida: a projeção de tempo conta a partir daqui
            t_ref, p_ref = time.time(), passo
        hh = dict(epoca=ep + 1, perda=float(np.mean(soma["perda"])), L_rot=float(np.mean(soma["L_rot"])), lr=float(sched.get_last_lr()[-1]), seg=round(time.time() - t0, 1),
                  img_s=round((ep + 1) * passos_ep * acum * bs / (time.time() - t0), 1))
        if K:
            hh.update(L_frac=float(np.mean(soma["L_frac"])), L_dens=float(np.mean(soma["L_dens"])), alfa=float(np.mean(soma["alfa"])), so_cabeca=bool(so_cabeca))
        if sg is not None:
            hh["perda_cabeca_sg"] = sg.media_perda(-passos_ep * acum)
        if n_cab:
            hh["so_cabeca"] = bool(so_cabeca)
        if va is not None and len(va) and not cfg.get("parada_interna") and ((ep + 1) % val_cada == 0 or ep + 1 == cfg["epocas"]):
            hh.update(valida())
        if len(tr_int):         # P3: a validação interna escolhe o checkpoint; o fold de fora só é olhado na época 0 e no escolhido
            hh["int_rmse_logm"] = int_rmse()
            if hh["int_rmse_logm"] < melhor[0]:
                melhor = (hh["int_rmse_logm"], ep + 1, foto(ep + 1))
        if fotos_epoca:
            grava_foto(ep + 1)
        hist.append(hh)
        print(f"{rotulo} {hh}", flush=True)
    escolha = {}
    if melhor is not None:
        with torch.no_grad():
            sd_ = ema.state_dict()
            for nm, v in melhor[2].items():
                sd_[nm].copy_(v.to(sd_[nm].device, sd_[nm].dtype))
        escolha = dict(epoca_escolhida=int(melhor[1]), int_rmse_logm=float(melhor[0]), n_interna=int(len(tr_int)), n_treino=int(len(tr)),
                       curva_interna=[h_.get("int_rmse_logm") for h_ in hist])
        hist.append(dict(epoca="escolhida", **escolha, **(valida() if va is not None and len(va) else {})))
        print(f"{rotulo} checkpoint escolhido pela validação interna: época {melhor[1]} ({hist[-1]})", flush=True)
    preve.sg = sg
    return ema, preve, hist, (tm, ts), dict(dens_fonte=dc if K else (sg.dc if sg is not None else None), vies=vies, n_ext_epoca=int(sum(cotas.values())) if doms else 0,
                                            escolha=escolha, ini=info_ini, tr_modelo=tr, tr_interna=tr_int)


# ------------------------------------------------------------------ rodada 4: uma parte (processo filho) e a junção
def massa_aninhada6(cfg, D, C, X, Xtr, tr, fold_tr, todos_f, dev, prazo, p4, lins):
    """Entrada de massa da H0c-m (15.4): a log-massa da ridge R4c (refeita nesta execução; mesmas bases, pesos, alfa e sem
    viés), por vista. No modelo que deixa o fold q de fora, o prato do fold j ≠ q recebe a previsão da ridge treinada
    sem q e sem j (aninhada: o rótulo de massa do próprio prato nunca entra), e o prato do fold q a da ridge sem q (a
    R4c fora-do-fold). No modelo final, cada prato recebe a R4c fora-do-fold, e os conjuntos de previsão a ridge com
    tudo. Devolve ({q: [n_treino, 8]}, {conjunto: [n, 8]}, informações); confere que a coluna do fold de fora repete a
    R4c refeita (oof gravado)."""
    import torch
    c4 = json.loads(str(p4["cfg"]))
    usados = [ch for ch in (c4.get("treina_com") or []) if ch in D.dom]
    fr = json.loads(str(p4["pesos_dominio"]))
    doms = [dict(chave=ch, X=feats_dom4(c4["bb"], D, C, ch, dev, 32, prazo), t=D.dom[ch]["T"][:, [I_M]], d=D.dom[ch], frac=fr[ch]) for ch in usados]
    R = RidgeMulti(Xtr, D.T[tr][:, [I_M]], fold_tr, doms, dev, sem_vies=bool(c4.get("sem_vies")))
    alfa = torch.from_numpy(np.asarray(p4["alfa_massa"], np.float64)).to(dev)
    prev = lambda fs, Xv: R.prev_x(frozenset(fs), Xv, alfa, por_or=True).cpu().numpy()[..., 0].astype(np.float32)
    M = colunas_aninhadas6(prev, Xtr, fold_tr, todos_f)
    Mconj = {s: prev(todos_f, X[l]) if len(l) else np.zeros((0, X.shape[1]), np.float32) for s, l in lins.items()}
    ref = np.asarray(p4["oof"])[tr, I_M]
    meu = np.zeros(len(tr))
    for q in sorted(todos_f):
        meu[fold_tr == q] = M[q][fold_tr == q].mean(1)
    dif = float(np.abs(meu - ref).max())
    assert dif < 5e-3, f"a massa da ridge fora-do-fold não repete a R4c refeita (dif {dif:.3g})"
    aninh = float(np.mean([np.abs(M[q][fold_tr != q] - M[5][fold_tr != q]).mean() for q in sorted(todos_f)]))
    return M, Mconj, dict(bases=usados, dif_r4c6_oof=dif, mudanca_media_aninhada_log=aninh, ridges=1 + len(todos_f) + len(todos_f) * (len(todos_f) - 1) // 2)


def pasta_fotos6(a, cfg):
    """Onde o modelo final do P3 deixa a média móvel de cada época (no cache, fora da saída)."""
    return os.path.join(a.cache, f"fotos6__{slug(cfg['nome'])}")


def feats_rede6(rede, C, ids, dev, hw, bs, prazo, vistas):
    """Features [agregado ‖ média dos tokens] de uma rede (ajustada) para pratos do Nutrition5k, sobre o recorte:
    vistas = [(orientação, com erro de enquadramento?)]; o erro é sorteado por prato e vista (semente fixa).
    Devolve [n, vistas, D] float32."""
    import torch
    out = np.zeros((len(ids), len(vistas), rede.dim), np.float32)
    rede.eval()
    with torch.inference_mode():
        for i in range(0, len(ids), bs):
            if time.time() > prazo:
                raise SemOrcamento()
            lote = ids[i:i + bs]
            linhas = [C.pos[d] for d in lote]
            for v, (k, jit) in enumerate(vistas):
                enq = np.stack([desvios5(np.random.default_rng(zlib.crc32(f"{d}|hp3|{v}".encode())), 1)[0] for d in lote]) if jit else None
                x = orienta(monta_lote(C, linhas, dev, hw, enquadramento=enq), k)
                with autocast(dev):
                    f = rede.feats(x)
                out[i:i + len(lote), v] = f.float().cpu().numpy()
    return out


def cabeca_linear6(rede, F, tm, ts):
    """A cabeça de 22 saídas de um ajuste fino sobre features já extraídas [n, o, D]: 22 saídas em log [n, o, 22]."""
    import torch
    with torch.inference_mode():
        W, b = rede.head.weight.detach().float().cpu(), rede.head.bias.detach().float().cpu()
        o = torch.from_numpy(F.reshape(-1, F.shape[-1])) @ W.T + b
    return (o.numpy() * ts + tm).reshape(F.shape[0], F.shape[1], -1).astype(np.float32)


def hp3_6(rede, C, D, trq, va_ids, dev, hw, bs, prazo, tm, ts, sem):
    """A cabeça de densidade do P3 (HP3, 15.4) num modelo: features do backbone ajustado nos pratos de treino (2 vistas:
    o recorte exato e um com erro de enquadramento, orientação original), cabeça igual à H0c (MLP3: oculta de 512,
    frações, densidade composta, peso por faixa, 60 épocas); nas fotos de previsão, 8 orientações. Devolve (cabeça
    principal [n, 8, 22], HP3 [n, 8, 22], frações da HP3 [n, 8, K], pesos da HP3, densidade por fonte)."""
    Ftr = feats_rede6(rede, C, [D.ids[i] for i in trq], dev, hw, bs, prazo, [(0, False), (0, True)])
    Fv = feats_rede6(rede, C, va_ids, dev, hw, bs, prazo, [(k, False) for k in range(8)])
    dc = dens_por_fonte(D.FF[trq], D.massa[trq], D.proteina[trq])
    pesos, prev = treina_cabeca(Ftr, D.T[trq], D.FF[trq], peso_faixa(D.proteina[trq]), dc, dict(MLP3), dev, semente=sem)
    Ph, Fh = prev(Fv)
    return cabeca_linear6(rede, Fv, tm, ts), Ph, Fh, pesos, dc


def monta_final_p3(a, cfg, D, C, dev, prazo, saida, parte_de, todos_f, usa, nan):
    """P3, modelo final (15.4): a época é a mediana das escolhidas pela validação interna nos folds; a média móvel dessa
    época (gravada pela parte 'final', que treina todas as épocas) vira o arquivo de pesos; a HP3 final treina sobre as
    features dele; os conjuntos de previsão (remontagem) recebem as duas cabeças, com 1, 4 e 8 orientações."""
    import torch
    escolhas = [int(json.loads(str(parte_de(cfg["nome"], f"fold{q}")["escolha"]))["epoca_escolhida"]) for q in sorted(todos_f)]
    e = int(np.median(escolhas)) if len(escolhas) % 2 else int(sorted(escolhas)[len(escolhas) // 2 - 1])
    fp = torch.load(os.path.join(pasta_fotos6(a, cfg), f"ep{e}.pt"), map_location="cpu", weights_only=True)
    rede = cria_rede4(cfg, D, pretrained=True).to(dev).eval()
    with torch.no_grad():
        sd_ = rede.state_dict()
        for nm, v in fp["estado"].items():
            sd_[nm].copy_(v.to(sd_[nm].device, sd_[nm].dtype))
    tm, ts = np.array(fp["tm"], np.float64), np.array(fp["ts"], np.float64)
    tr = np.where(D.usa_tr)[0]
    hw = BACKBONES[cfg["bb"]]["hw"]
    N, K = D.N, len(D.fontes)
    sem = zlib.crc32(f"{cfg['nome']}|final".encode()) % 2 ** 31
    ph = dch = None
    for s in D.conj:
        ids = [D.conj[s][i] for i in usa[s]]
        Pm, Ph, Fh, ph, dch = hp3_6(rede, C, D, tr, ids, dev, hw, max(8, cfg["bs"]), prazo, tm, ts, sem)
        tam = len(D.conj[s])
        for suf, kk in ORIENT.items():
            saida.setdefault(s + suf, nan(tam, N))[usa[s]] = media_or(Pm, kk)
            saida.setdefault(s + "_hp3" + suf, nan(tam, N))[usa[s]] = media_or(Ph, kk)
            saida.setdefault(s + "_ff" + suf, nan(tam, K))[usa[s]] = media_or(Fh, kk)
            saida.setdefault(s + "_dens_ff" + suf, nan(tam))[usa[s]] = media_or(Fh, kk) @ dch
    arq = arq_modelo4(a.saida, cfg, "final")
    grava_modelo4(arq, cfg, D, rede, tm, ts, dch, hp3=ph)
    arq_h = os.path.join(a.saida, f"cabecahp3__{slug(cfg['nome'])}.npz")
    np.savez(arq_h + ".tmp.npz", nome=cfg["nome"], listas_hash=D.hash, fontes=np.array(D.fontes), **ph)
    os.replace(arq_h + ".tmp.npz", arq_h)
    saida.update(dens_fonte=dch.astype(np.float32), fontes=np.array(D.fontes), final_p3=json.dumps(dict(epoca=e, escolhas_dos_folds=escolhas)))
    grava_servir(a, cfg, D, rede.dim, dict(
        tipo="ajuste fino parcial: só os últimos blocos, a norma final, o agrupador e a cabeça mudaram; o resto é o checkpoint público (22 saídas em log); "
             "mais a cabeça de densidade HP3 (MLP como a H0c) sobre as features do backbone ajustado",
        arquivo=os.path.basename(arq), bytes=os.path.getsize(arq), fontes=list(D.fontes), dens_fonte_g_por_g=[float(v) for v in dch],
        formula="s = head(f) * ts + tm, com f em float32; o arquivo traz 'estado' (só os tensores treinados, média móvel, meia precisão), 'tm', 'ts' e 'cabeca_hp3'",
        carregar="torch.load(arquivo, weights_only=True); cria_rede4(cfg, D, pretrained=True); carrega_modelo4(rede, ck)",
        parada_interna=dict(epocas_escolhidas_nos_folds=escolhas, epoca_do_modelo_final=e,
                            regra="10% das cadeias de treino de cada fold (sha1(cadeia) mod 10 == 0) como validação interna; vale o checkpoint de menor RMSE "
                                  "de log-massa (1 orientação, média móvel), com a época 0 (a ridge) concorrendo; o final usa a mediana das épocas escolhidas"),
        cabeca_hp3=dict(arquivo=os.path.basename(arq_h), tambem_em="'cabeca_hp3' do arquivo de pesos",
                        formula="sobre as mesmas features f (float32): z = W2 @ gelu(W1 @ ((f − mu) / sd) + b1) + b2; s_hp3 = z[:22] * ts + tm; "
                                "frações = softmax(z[22:]); densidade pelas frações = frações @ dens_fonte")))
    print(f"[{cfg['nome']} | ext] modelo final: época {e} (escolhidas nos folds: {escolhas})", flush=True)


def colunas_aninhadas6(prev, Xtr, fold_tr, todos_f):
    """A coluna de massa aninhada da H0c-m (15.4), dada prev(folds de treino, X) -> previsão por vista. Modelo q (fold q
    de fora): prato do fold j ≠ q <- ridge sem q e sem j; prato do fold q <- ridge sem q. Modelo final (5): cada prato
    <- ridge sem o fold dele. Função pura (o rótulo de um prato nunca entra na coluna dele em nenhum modelo)."""
    M = {q: np.full((len(fold_tr), Xtr.shape[1]), np.nan, np.float32) for q in list(todos_f) + [5]}
    for q in sorted(todos_f):
        M[q][fold_tr == q] = prev(todos_f - {q}, Xtr[fold_tr == q])
        for j in sorted(todos_f - {q}):
            M[q][fold_tr == j] = prev(todos_f - {q, j}, Xtr[fold_tr == j])
        M[5][fold_tr == q] = M[q][fold_tr == q]
    assert all(not np.isnan(v).any() for v in M.values()), "coluna de massa aninhada incompleta"
    return M


def particao_interna6(cadeia):
    """A cadeia fica na validação interna do P3 (15.4)? sha1(cadeia) mod 10 == 0. Função pura."""
    return int(hashlib.sha1(str(cadeia).encode()).hexdigest(), 16) % 10 == 0


def ini_r4c6(cfg, D, C, tr, k, dev, prazo, ini, tr_int=()):
    """Ponto de partida da cabeça do P3 (15.4): a ridge da R4c refeita só com as linhas de treino deste modelo (o fold k
    e a validação interna de fora): as 22 saídas da R0c (só Nutrition5k, alfas da R0c refeita) com a massa trocada pela
    da R4c (Nutrition5k + as cinco bases, sem viés por domínio, alfa da R4c refeita), sobre as features congeladas do
    recorte. Devolve (A [D, 22], b [22], informações). Confere também que, com a validação interna de volta, a conta
    repete o mapa da R4c refeita para o mesmo fold (ini['afim_A'], ini['afim_b'])."""
    import torch
    X = feats_congeladas(cfg["bb"], D, C, dev, 32, prazo).astype(np.float32)
    usados = [ch for ch in ini["doms"] if ch in D.dom]
    fr = fracoes_dominio4({ch: D.dom[ch]["grupos_treino_lista"] for ch in usados}, len(D.ids), usados)
    doms = [dict(chave=ch, X=feats_dom4(cfg["bb"], D, C, ch, dev, 32, prazo), t=D.dom[ch]["T"][:, [I_M]], d=D.dom[ch], frac=fr[ch]) for ch in usados]
    todos = frozenset(int(f) for f in set(D.fold[D.usa_tr].tolist()))
    fs = todos - {k} if k is not None else todos
    a0 = torch.from_numpy(np.asarray(ini["alfa0"], np.float64)).to(dev)
    am = torch.from_numpy(np.asarray(ini["alfa_m"], np.float64)).to(dev)

    def conta(linhas):
        Xl = X[[C.pos[D.ids[i]] for i in linhas]]
        f_ = D.fold[linhas]
        A, b = RidgeSub(Xl, D.T[linhas], f_, dev).afim(frozenset(int(x) for x in set(f_.tolist())), a0)
        Am, bm = RidgeMulti(Xl, D.T[linhas][:, [I_M]], f_, [dict(e) for e in doms], dev, sem_vies=True, todos=todos).afim(fs, am)
        A[:, I_M], b[I_M] = Am[:, 0], bm[0]
        return A, b
    A, b = conta(tr)
    info = dict(n_nutrition5k=int(len(tr)), n_interna_fora=int(len(tr_int)), bases=usados, pesos_dominio=fr)
    if ini.get("afim_A") is not None:       # paridade com a R4c refeita (mesmo fold), com as linhas da validação interna de volta
        Ar, br = conta(np.sort(np.r_[tr, np.asarray(tr_int, int)])) if len(tr_int) else (A, b)
        Xv = X[[C.pos[D.ids[i]] for i in tr[:64]], 0].astype(np.float64)
        info["dif_r4c6_log"] = float(np.abs((Xv @ Ar + br) - (Xv @ ini["afim_A"] + ini["afim_b"])).max())
        info["mudanca_sem_interna_log_massa"] = float(np.abs((Xv @ A[:, I_M] + b[I_M]) - (Xv @ Ar[:, I_M] + br[I_M])).max())
        assert info["dif_r4c6_log"] < 5e-3, f"a ridge de partida não repete a R4c refeita no fold {k} (dif {info['dif_r4c6_log']:.3g})"
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return A, b, info


def aplica_cabeca4(z, X):
    """Aplica uma cabeça de frações gravada (arrays mu, sd, W1, b1, W2, b2, tm, ts) a features X [n, o, D]: devolve
    (22 saídas em log [n, o, 22], frações por fonte [n, o, K]). É a fórmula de servir__*.json, em float32."""
    import torch
    nt = len(z["tm"])
    with torch.inference_mode():
        h = torch.nn.functional.gelu(torch.from_numpy(((X - z["mu"]) / z["sd"]).reshape(-1, X.shape[-1]).astype(np.float32)) @ torch.from_numpy(np.asarray(z["W1"])).T + torch.from_numpy(np.asarray(z["b1"])))
        o = h @ torch.from_numpy(np.asarray(z["W2"])).T + torch.from_numpy(np.asarray(z["b2"]))
        fr = torch.softmax(o[:, nt:], 1).numpy()
        o = o.numpy()
    return (o[:, :nt] * z["ts"] + z["tm"]).reshape(X.shape[0], X.shape[1], nt), fr.reshape(X.shape[0], X.shape[1], -1)


def _e_topo4(nome, L, n):
    """Parâmetro que treina no ajuste fino parcial: cabeças, norma final, agrupador e os n últimos blocos."""
    return nome.startswith(("head", "m.norm.", "m.fc_norm", "m.attn_pool")) or (nome.startswith("m.blocks.") and int(nome.split(".")[2]) >= L - n)


def grava_modelo4(arq, cfg, D, rede, tm, ts, dc, sg=None, hp3=None):
    """Pesos de um ajuste fino da rodada 4 (média móvel, meia precisão). No ajuste fino parcial só vão os tensores que
    treinaram: o resto é o checkpoint público do backbone, descrito em servir__*.json."""
    import torch
    n, L = cfg.get("blocos_treinaveis"), len(rede.m.blocks)
    est = {nm: (v.half() if v.is_floating_point() else v).cpu() for nm, v in rede.state_dict().items() if not n or _e_topo4(nm, L, n)}
    K = len(D.fontes) if cfg.get("fracoes4") else 0
    torch.save(dict(formato=FORMATO, nome=cfg["nome"], cfg={c: v for c, v in cfg.items() if c not in ("depende", "tardia", "partes")}, timm=BACKBONES[cfg["bb"]],
                    tm=[float(v) for v in tm], ts=[float(v) for v in ts], grupos=list(D.grupos), listas_hash=D.hash, n_saidas=D.N, n_ff=K,
                    fontes=list(D.fontes) if K else [], dens_fonte=[float(v) for v in dc] if dc is not None else [], parcial=bool(n),
                    blocos_treinaveis=int(n or 0), estado=est,
                    **({"cabeca_sg": {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in sg.items()}} if sg is not None else {}),
                    **({"cabeca_hp3": {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in hp3.items()}} if hp3 is not None else {})), arq + ".tmp")
    os.replace(arq + ".tmp", arq)


def carrega_modelo4(rede, ck):
    """Põe na rede os pesos gravados por grava_modelo4 (inteiros, ou só a parte treinada sobre o backbone público)."""
    sd = {n: (v.float() if v.is_floating_point() else v) for n, v in ck["estado"].items()}
    if ck.get("parcial"):
        r = rede.load_state_dict(sd, strict=False)
        L = len(rede.m.blocks)
        assert not r.unexpected_keys and not [n for n in r.missing_keys if _e_topo4(n, L, ck["blocos_treinaveis"])], "pesos parciais não cobrem a parte treinada"
    else:
        rede.load_state_dict(sd)
    return np.array(ck["tm"], np.float32), np.array(ck["ts"], np.float32)


def prev_modelo4(rede, C, ids, dev, hw, bs, tm, ts, prazo, zsg=None):
    """Previsão crua, 1 orientação, de um ajuste fino para fotos quaisquer do cache: (22 saídas em log, frações ou None,
    22 saídas da cabeça sem gradiente ou None). Com zsg (rodada 5), as frações vêm da cabeça sem gradiente."""
    import torch
    K = rede.head_ff.out_features if hasattr(rede, "head_ff") else (len(zsg["dens_fonte"]) if zsg is not None else 0)
    P = np.zeros((len(ids), len(tm)), np.float32); F = np.zeros((len(ids), K), np.float32) if K else None
    S = np.zeros((len(ids), len(tm)), np.float32) if zsg is not None else None
    tm_t, ts_t = torch.from_numpy(tm).to(dev), torch.from_numpy(ts).to(dev)
    with torch.inference_mode():
        for i in range(0, len(ids), bs):
            if time.time() > prazo:
                raise SemOrcamento()
            x = monta_lote(C, [C.pos[d] for d in ids[i:i + bs]], dev, hw)
            with autocast(dev):
                f = rede.feats(x)
            f = f.float()
            P[i:i + len(x)] = (rede.head(f) * ts_t + tm_t).cpu().numpy()
            if hasattr(rede, "head_ff"):
                F[i:i + len(x)] = torch.softmax(rede.head_ff(f), 1).cpu().numpy()
            elif zsg is not None:
                s_, q_ = aplica_sg_torch(zsg, f)
                S[i:i + len(x)], F[i:i + len(x)] = s_.cpu().numpy(), q_.cpu().numpy()
    return P, F, S


def listas_ext4(D, chaves):
    return {ch: dict(hash=D.dom[ch]["hash"], sha256_lista=D.dom[ch]["sha256_lista"], sha256_divisao=D.dom[ch]["sha256_divisao"], licenca=D.dom[ch]["licenca"],
                     uso_declarado=D.dom[ch]["uso_declarado"]) for ch in chaves if ch in D.dom}


def doms_de4(cfg):
    """Domínios externos de que a configuração usa rótulo."""
    return list(cfg.get("treina_com") or cfg.get("externo4") or []) + list(cfg.get("ff_de") or [])


def arq_modelo4(saida, cfg, parte):
    return arq_de(saida, cfg, "pesos") if parte == "final" else os.path.join(saida, f"modelo__{slug(cfg['nome'])}__{parte}.pt")


def roda_parte4(a, cfg, parte, cfgs, D, C, abertos, dev, prazo, rot, t0):
    """Uma parte da rodada 4 (seção 13). Nada aqui lê rótulo que não seja de treino: as fotos do teste2 e das partes de
    registro externas só passam pelos modelos, e as previsões ficam gravadas cruas (13.8)."""
    import torch
    tr = np.where(D.usa_tr)[0]
    n, N, K = len(D.ids), D.N, len(D.fontes)
    fold_tr = D.fold[tr]
    todos_f = frozenset(int(k) for k in set(fold_tr.tolist()))
    usa = {s: np.where(D.usa_av[s])[0] for s in D.conj}
    lin = {s: [C.pos[D.conj[s][i]] for i in usa[s]] for s in D.conj}
    nan = lambda *forma: np.full(forma, np.nan, np.float32)
    por_nome = {c["nome"]: c for c in cfgs}
    parte_de = lambda nome, p="tudo": np.load(arq_parte(a.saida, por_nome[nome], p), allow_pickle=False)
    saida = dict(cfg=json.dumps(cfg), parte=parte, dispositivo=str(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))
    hist = []
    usados = doms_de4(cfg)
    assert all(ch in abertos for ch in usados) or parte == "ext", f"fotos de um domínio de treino não estão prontas: {[ch for ch in usados if ch not in abertos]}"
    tem_feat = lambda ch: os.path.exists(os.path.join(pasta_dom4(C, D.dom[ch]), f"feat__{cfg['bb']}.npy"))

    def guarda(chave, idx, tam, P, F=None, dc=None):
        for suf, q in ORIENT.items():
            saida.setdefault(chave + suf, nan(tam, N))[idx] = media_or(P, q)
            if F is not None:
                saida.setdefault(chave + "_ff" + suf, nan(tam, K))[idx] = media_or(F, q)
                saida.setdefault(chave + "_dens_ff" + suf, nan(tam))[idx] = media_or(F, q) @ dc

    def tempo():
        if time.time() > prazo:
            raise SemOrcamento()

    def ext_ids(ch, de_treino):
        d = D.dom[ch]
        j = np.where(d["treina"] if de_treino else d["reg"])[0]
        pre = "extcv" if de_treino else "extreg"
        saida[f"{pre}_ids__{ch}"] = np.array([d["ids"][i] for i in j])
        if de_treino:
            saida[f"extcv_fold__{ch}"] = d["fold_ext"][j].astype(np.int16)
        else:
            saida[f"extreg_papel__{ch}"] = d["papel"][j]
        return j

    def fecha_ext(feitos):
        saida["dominios"] = np.array(sorted(feitos))
        saida["dominios_fora"] = np.array(sorted(set(DOMINIOS4) - set(feitos)))

    if cfg["tipo"] == "feat":
        if cfg["chave"] == "n5k":
            X = feats_congeladas(cfg["bb"], D, C, dev, cfg["bs"], prazo)
        else:
            assert cfg["chave"] in abertos, f"fotos de {cfg['chave']} não estão prontas"
            X = feats_dom4(cfg["bb"], D, C, cfg["chave"], dev, cfg["bs"], prazo)
        saida.update(forma=np.array(X.shape))
    elif cfg["tipo"] == "ridge4" and parte == "tudo":
        X = feats_congeladas(cfg["bb"], D, C, dev, cfg["bs"], prazo).astype(np.float32)
        Xtr, Dm = X[[C.pos[D.ids[i]] for i in tr]], X.shape[-1]
        A, b = np.zeros((6, Dm, N), np.float32), np.zeros((6, N), np.float32)
        arq_c = arq_de(a.saida, cfg, "cabeca")
        if not usados:      # R0: a cabeça ridge do D1, a mesma conta da execução 3 (todos os alvos, 8 orientações)
            P, alfa, mse = ridge_cv_or(Xtr, D.T[tr], fold_tr)
            guarda("oof", tr, n, P)
            _, m = ridge_final(Xtr, D.T[tr], alfa)
            for s in D.conj:
                guarda(s, usa[s], len(D.conj[s]), m.predict(X[lin[s]].reshape(-1, Dm)).reshape(len(lin[s]), 8, -1))
            saida.update(alfa=alfa, cv_mse=mse)
            R = RidgeSub(Xtr, D.T[tr], fold_tr, dev)       # o modelo de cada fold como mapa afim, para as fotos externas
            alfa_t = torch.from_numpy(np.asarray(alfa)).to(dev)
            for q in sorted(todos_f):
                A[q], b[q] = R.afim(todos_f - {q}, alfa_t)
                ref = (Xtr[fold_tr == q].reshape(-1, Dm).astype(np.float64) @ A[q] + b[q]).reshape(-1, 8, N).mean(1)
                dif = float(np.abs(ref - saida["oof"][tr[fold_tr == q]]).max())
                assert dif < 5e-3, f"o mapa afim do fold {q} não repete as previsões fora-do-fold (dif {dif:.2g})"
            A[5], b[5] = m.W / m.sd[:, None], m.mt - (m.mu / m.sd) @ m.W
            np.savez(arq_c + ".tmp.npz", mu=m.mu, sd=m.sd, W=m.W, mt=m.mt, alfa=alfa, nome=cfg["nome"], listas_hash=D.hash)
            formula = "s = ((f − mu) / sd) @ W + mt, com as features f em float32 (cabeça fora do autocast); arrays mu, sd, W, mt no arquivo"
        else:               # R1 a R4: só a massa total é refeita, com as bases externas (13.4); os outros alvos são os da R0
            pl = parte_de(cfg["plano"])
            for c in pl.files:
                if c.startswith(tuple(["oof"] + list(D.conj))):
                    saida[c] = pl[c].copy()
            A, b = pl["afim_A"].copy(), pl["afim_b"].copy()
            fr = fracoes_dominio4({ch: D.dom[ch]["grupos_treino_lista"] for ch in usados}, len(D.ids), usados)
            doms = [dict(chave=ch, X=feats_dom4(cfg["bb"], D, C, ch, dev, cfg["bs"], prazo), t=D.dom[ch]["T"][:, [I_M]], d=D.dom[ch], frac=fr[ch]) for ch in usados]
            tempo()
            R = RidgeMulti(Xtr, D.T[tr][:, [I_M]], fold_tr, doms, dev, sem_vies=bool(cfg.get("sem_vies")))
            alfa = R.alfa_cv(todos_f)
            vies = {ch: np.zeros((6, N), np.float32) for ch in usados}
            for q in sorted(todos_f):
                Pq = R.prev_x(todos_f - {q}, Xtr[fold_tr == q], alfa, por_or=True).cpu().numpy()
                for suf, kk in ORIENT.items():
                    saida["oof" + suf][tr[fold_tr == q], I_M] = media_or(Pq, kk)[:, 0]
            for sc in D.conj:
                Pv = R.prev_x(todos_f, X[lin[sc]], alfa, por_or=True).cpu().numpy()
                for suf, kk in ORIENT.items():
                    saida[sc + suf][usa[sc], I_M] = media_or(Pv, kk)[:, 0]
            for q, fs in [(q, todos_f - {q}) for q in sorted(todos_f)] + [(5, todos_f)]:
                Aq, bq = R.afim(fs, alfa)
                A[q][:, I_M], b[q][I_M] = Aq[:, 0], bq[0]
                for ch, dl in R.vies(fs).items():
                    vies[ch][q, I_M] = dl[0]
            saida.update(alfa_massa=alfa.cpu().numpy(), pesos_dominio=json.dumps(fr), **{f"vies__{ch}": v for ch, v in vies.items()})
            np.savez(arq_c + ".tmp.npz", mu=np.zeros(Dm, np.float32), sd=np.ones(Dm, np.float32), W=A[5], mt=b[5], nome=cfg["nome"], listas_hash=D.hash)
            formula = ("s = f @ W + mt, com as features f em float32 (cabeça fora do autocast; mu = 0 e sd = 1 no arquivo). A previsão é crua: "
                       "a massa de fotos de outro domínio leva depois a constante do domínio (13.2)")
        os.replace(arq_c + ".tmp.npz", arq_c)
        saida.update(afim_A=A, afim_b=b)
        z = np.load(arq_c, allow_pickle=False)      # remontagem: a cabeça relida do arquivo repete as previsões gravadas?
        conf = {}
        for sc in D.conj:
            if len(lin[sc]):
                ref = (((X[lin[sc]].reshape(-1, Dm) - z["mu"]) / z["sd"]).astype(np.float64) @ z["W"] + z["mt"]).reshape(len(lin[sc]), 8, -1).mean(1)
                conf[sc] = float(np.abs(ref - saida[sc][usa[sc]]).max())
                assert conf[sc] < 5e-3, f"a cabeça gravada não repete as previsões de {sc} (dif {conf[sc]:.2g})"
        grava_servir(a, cfg, D, Dm, dict(tipo="ridge sobre o backbone congelado" + (" (massa total com bases externas)" if usados else ""), arquivo=os.path.basename(arq_c),
                                         bytes=os.path.getsize(arq_c), formula=formula, conferencia_remontagem=dict(max_dif_log=conf, nota="cabeça relida do arquivo × previsões gravadas, sobre as features em cache")))
    elif cfg["tipo"] == "fracoes4" and parte == "tudo":
        X = feats_congeladas(cfg["bb"], D, C, dev, 32, prazo).astype(np.float32)
        Xtr, Ttr, FFtr, mtr, ptr, Dm = X[[C.pos[D.ids[i]] for i in tr]], D.T[tr], D.FF[tr], D.massa[tr], D.proteina[tr], X.shape[-1]
        fts = {ch: feats_dom4(cfg["bb"], D, C, ch, dev, 32, prazo) for ch in usados}
        fr = fracoes_dominio4({ch: D.dom[ch]["grupos_treino_lista"] for ch in usados}, len(D.ids), usados) if usados else {}
        dens_fonte = np.zeros((6, K), np.float32)
        vies = {ch: np.zeros((6, N), np.float32) for ch in usados}
        P, F = np.zeros((len(tr), 8, N), np.float32), np.zeros((len(tr), 8, K), np.float32)
        Xconj = {s: X[lin[s]] for s in D.conj}
        if cfg.get("massa_aninhada"):       # H0c-m (rodada 6, 15.4): a log-massa da ridge R4c entra como feature, aninhada
            Mq, Mconj, info_m = massa_aninhada6(cfg, D, C, X, Xtr, tr, fold_tr, todos_f, dev, prazo, parte_de(cfg["massa_aninhada"]), {s: lin[s] for s in D.conj})
            Xconj = {s: np.concatenate([X[lin[s]], Mconj[s][..., None]], -1) for s in D.conj}
            saida.update(massa_aninhada=json.dumps(info_m))
        for q, m in [(q, fold_tr != q) for q in sorted(todos_f)] + [(5, np.ones(len(tr), bool))]:
            tempo()
            ed, vq = ext_cabeca4(D, cfg, fts, fr, q if q < 5 else None, Ttr[m]) if usados else (None, {})
            dc = dens_por_fonte(FFtr[m], mtr[m], ptr[m])
            Xq = np.concatenate([Xtr, Mq[q][..., None]], -1) if cfg.get("massa_aninhada") else Xtr      # H0c-m: a coluna de massa do modelo q
            pesos, prev = treina_cabeca(Xq[m], Ttr[m], FFtr[m], peso_faixa(ptr[m]), dc, cfg, dev, None, ext=ed)
            dens_fonte[q] = dc
            for ch, dl in vq.items():
                vies[ch][q] = dl
            saida.update({f"cab{q}_{c}": v for c, v in pesos.items()})
            if q < 5:
                P[~m], F[~m] = prev(Xq[~m])
                guarda("oof", tr[fold_tr == q], n, P[fold_tr == q], F[fold_tr == q], dc)
            else:
                for s in D.conj:
                    guarda(s, usa[s], len(D.conj[s]), *prev(Xconj[s]), dc)
        saida.update(dens_fonte=dens_fonte, fontes=np.array(D.fontes), pesos_dominio=json.dumps(fr), **{f"vies__{ch}": v for ch, v in vies.items()})
        arq_c = arq_de(a.saida, cfg, "cabeca")
        np.savez(arq_c + ".tmp.npz", nome=cfg["nome"], listas_hash=D.hash, fontes=np.array(D.fontes), **pesos)
        os.replace(arq_c + ".tmp.npz", arq_c)
        z = np.load(arq_c, allow_pickle=False)      # remontagem da cabeça de frações (faltou na execução 3)
        conf = {}
        for sc in D.conj:
            if len(lin[sc]):
                Pz, Fz = aplica_cabeca4(z, Xconj[sc])
                conf[sc] = dict(max_dif_log=float(np.abs(Pz.mean(1) - saida[sc][usa[sc]]).max()), max_dif_fracao=float(np.abs(Fz.mean(1) - saida[sc + "_ff"][usa[sc]]).max()),
                                max_dif_dens_ff=float(np.abs(Fz.mean(1) @ z["dens_fonte"] - saida[sc + "_dens_ff"][usa[sc]]).max()))
                assert conf[sc]["max_dif_log"] < 5e-3 and conf[sc]["max_dif_fracao"] < 5e-3, f"a cabeça de frações gravada não repete as previsões de {sc} ({conf[sc]})"
        grava_servir(a, cfg, D, Dm, dict(
            tipo="MLP de uma camada oculta sobre o backbone congelado (22 saídas em log + frações por fonte)"
                 + ("; entrada = [features ‖ log-massa da ridge R4c]" if cfg.get("massa_aninhada") else ""), arquivo=os.path.basename(arq_c),
            **(dict(entrada_massa=dict(cabeca=os.path.basename(arq_de(a.saida, por_nome[cfg["massa_aninhada"]], "cabeca")),
                                       formula="f' = [f ‖ (f @ W + mt)[0]] com a cabeça ridge da R4c refeita (mu = 0, sd = 1 no arquivo; coluna 0 = log-massa); "
                                               "no treino de cada fold entrou a previsão aninhada da ridge (sem o fold de validação nem o fold do próprio prato)"))
               if cfg.get("massa_aninhada") else {}),
            bytes=os.path.getsize(arq_c), fontes=list(D.fontes), dens_fonte_g_por_g=[float(v) for v in dens_fonte[5]],
            formula="z = W2 @ gelu(W1 @ ((f − mu) / sd) + b1) + b2, com f em float32, por orientação; s = z[:22] * ts + tm; frações = softmax(z[22:]); "
                    "densidade pelas frações = frações @ dens_fonte; no D1, densidade = exp(média de s[3] e log(dens. pelas frações + 0,01)) − 0,01",
            conferencia_remontagem=dict(conf, nota="cabeça relida do arquivo × previsões gravadas, sobre as features em cache")))
    elif cfg["tipo"] in ("ridge4", "fracoes4"):     # parte 'ext': previsões cruas, 1 orientação, nas fotos de todos os domínios prontos
        pt = parte_de(cfg["nome"])
        pm = parte_de(cfg["massa_aninhada"]) if cfg.get("massa_aninhada") else None      # H0c-m: a massa vem da ridge R4c do mesmo fold
        feitos = []
        for ch in [c for c in abertos if tem_feat(c)]:
            tempo()
            d, F0 = D.dom[ch], np.load(os.path.join(pasta_dom4(C, D.dom[ch]), f"feat__{cfg['bb']}.npy"))[:, :1].astype(np.float32)
            com_m = lambda q, Fx: np.concatenate([Fx, (Fx.astype(np.float64) @ pm["afim_A"][q][:, I_M] + pm["afim_b"][q][I_M])[..., None].astype(np.float32)], -1) if pm is not None else Fx
            for de_treino in (True, False):
                j = ext_ids(ch, de_treino)
                pre = "extcv" if de_treino else "extreg"
                Pj, Fj, Dj = nan(len(j), N), nan(len(j), K), nan(len(j))
                for q in (range(5) if de_treino else [5]):
                    s_ = (d["fold_ext"][j] == q) if de_treino else np.ones(len(j), bool)
                    if not s_.any():
                        continue
                    if cfg["tipo"] == "ridge4":
                        Pj[s_] = F0[j[s_], 0].astype(np.float64) @ pt["afim_A"][q] + pt["afim_b"][q]
                    else:
                        Pq, Fq = aplica_cabeca4({c: pt[f"cab{q}_{c}"] for c in ("mu", "sd", "W1", "b1", "W2", "b2", "tm", "ts")}, com_m(q, F0[j[s_]]))
                        Pj[s_], Fj[s_], Dj[s_] = Pq[:, 0], Fq[:, 0], Fq[:, 0] @ pt["dens_fonte"][q]
                saida[f"{pre}__{ch}"] = Pj
                if cfg["tipo"] == "fracoes4":
                    saida[f"{pre}_ff__{ch}"], saida[f"{pre}_dens_ff__{ch}"] = Fj, Dj
            feitos.append(ch)
        fecha_ext(feitos)
    elif parte == "ext":                            # ajuste fino: cada modelo de fold prevê as fotos externas que validam nele; o final, as de registro
        if cfg.get("parada_interna"):               # P3: o modelo final sai aqui (a época é a mediana das escolhidas nos folds) com a HP3
            monta_final_p3(a, cfg, D, C, dev, prazo, saida, parte_de, todos_f, usa, nan)
        rede = cria_rede4(cfg, D, pretrained=bool(cfg.get("blocos_treinaveis"))).to(dev).eval()
        hw, bs = BACKBONES[cfg["bb"]]["hw"], max(8, cfg["bs"])
        Kf = len(D.fontes) if cfg.get("fracoes4") or cfg.get("fracoes_sg") or cfg.get("parada_interna") else 0
        sufx = "_hp3" if cfg.get("parada_interna") else "_sg"
        Ssg, tem_sg = {}, False
        buf = {(ch, t_): [ext_ids(ch, t_), None, None, None] for ch in abertos for t_ in (True, False)}
        for (ch, t_), v in buf.items():
            v[1], v[2], v[3] = nan(len(v[0]), N), nan(len(v[0]), Kf), nan(len(v[0]))
        for q in list(range(5)) + [5]:
            ck = torch.load(arq_modelo4(a.saida, cfg, "final" if q == 5 else f"fold{q}"), map_location="cpu", weights_only=True)
            tm, ts = carrega_modelo4(rede, ck)
            dcq = np.array(ck["dens_fonte"], np.float32)
            zsg = ck.get("cabeca_sg") if ck.get("cabeca_sg") is not None else ck.get("cabeca_hp3")     # P3: a HP3 no lugar da cabeça sem gradiente
            tem_sg = tem_sg or zsg is not None
            for ch in abertos:
                d = D.dom[ch]
                j, Pj, Fj, Dj = buf[(ch, q < 5)]
                s_ = (d["fold_ext"][j] == q) if q < 5 else np.ones(len(j), bool)
                if s_.any():
                    Pq, Fq, Sq = prev_modelo4(rede, C, [d["ids"][i] for i in j[s_]], dev, hw, bs, tm, ts, prazo, zsg)
                    Pj[s_] = Pq
                    if Kf:
                        Fj[s_], Dj[s_] = Fq, Fq @ dcq
                    if Sq is not None:
                        Ssg.setdefault((ch, q < 5), nan(len(j), N))[s_] = Sq
        for (ch, t_), (j, Pj, Fj, Dj) in buf.items():
            pre = "extcv" if t_ else "extreg"
            saida[f"{pre}__{ch}"] = Pj
            if Kf:
                saida[f"{pre}_ff__{ch}"], saida[f"{pre}_dens_ff__{ch}"] = Fj, Dj
            if tem_sg:                              # também vazio (snapme não tem foto de treino), para as chaves serem as mesmas em todo domínio
                saida[f"{pre}{sufx}__{ch}"] = Ssg.get((ch, t_), nan(len(j), N))
        fecha_ext(abertos)
    else:                                           # ajuste fino (F1, F2, F3, P1): fold ou final
        q = None if parte == "final" else int(parte[4:])
        trq = tr if q is None else tr[fold_tr != q]
        va = None if q is None else tr[fold_tr == q]
        sem = zlib.crc32(f"{cfg['nome']}|{parte}".encode()) % 2 ** 31
        if cfg.get("receita_u0"):       # F4, F4c (14.4): a receita do U0 sem mudança (a função da terceira execução) + cabeça sem gradiente
            rede, preve, hist, (tm, ts) = treina(cfg, D, C, trq, va, dev, prazo, a.div_lote, semente=sem, rotulo=rot)
            ex = dict(dens_fonte=preve.dens_fonte, vies={}, n_ext_epoca=0)
        else:
            ini6 = None
            if cfg.get("ini_r4c6"):     # P3: alfas e mapas da R0c e da R4c refeitas nesta execução (15.4)
                p0, p4 = parte_de(cfg["ini_r4c6"]["r0"]), parte_de(cfg["ini_r4c6"]["r4"])
                iq = 5 if q is None else q
                ini6 = dict(alfa0=p0["alfa"], alfa_m=p4["alfa_massa"], doms=list(cfg["externo4"]), afim_A=p4["afim_A"][iq], afim_b=p4["afim_b"][iq])
            fotos = pasta_fotos6(a, cfg) if (cfg.get("parada_interna") and q is None) else None
            rede, preve, hist, (tm, ts), ex = treina4(cfg, D, C, trq, va, dev, prazo, a.div_lote, sem, rot, q, ini=ini6, fotos_epoca=fotos)
        if cfg.get("parada_interna"):       # P3 (15.4): checkpoint escolhido pela validação interna e a cabeça HP3 do fold
            saida.update(escolha=json.dumps(ex["escolha"]), ini=json.dumps(ex["ini"]), tm=np.asarray(tm, np.float32), ts=np.asarray(ts, np.float32),
                         n_ext_epoca=ex["n_ext_epoca"], **{f"vies__{ch}": v.astype(np.float32) for ch, v in ex["vies"].items()})
            if q is not None:
                hw_ = BACKBONES[cfg["bb"]]["hw"]
                Pm, Ph, Fh, ph, dch = hp3_6(rede, C, D, trq, [D.ids[i] for i in va], dev, hw_, max(8, cfg["bs"]), prazo, tm, ts, sem)
                for suf, kk in ORIENT.items():
                    saida.setdefault("oof" + suf, nan(n, N))[va] = media_or(Pm, kk)
                    saida.setdefault("oof_hp3" + suf, nan(n, N))[va] = media_or(Ph, kk)
                    saida.setdefault("oof_ff" + suf, nan(n, K))[va] = media_or(Fh, kk)
                    saida.setdefault("oof_dens_ff" + suf, nan(n))[va] = media_or(Fh, kk) @ dch
                grava_modelo4(arq_modelo4(a.saida, cfg, parte), cfg, D, rede, tm, ts, dch, hp3=ph)
                saida.update(dens_fonte=dch.astype(np.float32), fontes=np.array(D.fontes), **{f"hp3_{c}": np.asarray(v) for c, v in ph.items()})
        else:
            dc = ex["dens_fonte"]
            sg = getattr(preve, "sg", None)
            if dc is not None:
                saida.update(dens_fonte=dc.astype(np.float32), fontes=np.array(D.fontes))
            saida.update(n_ext_epoca=ex["n_ext_epoca"], **{f"vies__{ch}": v.astype(np.float32) for ch, v in ex["vies"].items()})
            alvo = [("oof", n, va, [D.ids[i] for i in va])] if q is not None else [(s, len(D.conj[s]), usa[s], [D.conj[s][i] for i in usa[s]]) for s in D.conj]
            for chave, tam, idx, ids in alvo:
                P, F = preve(ids, variantes=True)
                for suf in ORIENT:
                    saida.setdefault(chave + suf, nan(tam, N))[idx] = P[suf]
                    if F is not None:
                        saida.setdefault(chave + "_ff" + suf, nan(tam, K))[idx] = F[suf]
                        saida.setdefault(chave + "_dens_ff" + suf, nan(tam))[idx] = F[suf] @ dc
                    if sg is not None:      # as 22 saídas da própria cabeça sem gradiente (rotas com a densidade da regressão dela)
                        saida.setdefault(chave + "_sg" + suf, nan(tam, N))[idx] = preve.ultima_sg[suf]
            arq = arq_modelo4(a.saida, cfg, parte)
            grava_modelo4(arq, cfg, D, rede, tm, ts, dc, sg.arrays() if sg is not None else None)
            if q is None and sg is not None:        # 14.6: a cabeça sem gradiente também num arquivo à parte, no formato das cabeças de frações
                arq_sg = os.path.join(a.saida, f"cabecasg__{slug(cfg['nome'])}.npz")
                np.savez(arq_sg + ".tmp.npz", nome=cfg["nome"], listas_hash=D.hash, fontes=np.array(D.fontes), **sg.arrays())
                os.replace(arq_sg + ".tmp.npz", arq_sg)
            if q is None:
                parcial = bool(cfg.get("blocos_treinaveis"))
                grava_servir(a, cfg, D, rede.dim, dict(
                    tipo=("ajuste fino parcial: só os últimos blocos, a norma final, o agrupador e as cabeças mudaram; o resto é o checkpoint público" if parcial
                          else "linear, treinada junto com o backbone") + (" (22 saídas em log + 10 logits de fração por fonte, em duas camadas lineares)" if hasattr(rede, "head_ff")
                                                                             else " (22 saídas em log; as frações vêm da cabeça sem gradiente, abaixo)" if sg is not None else ""),
                    arquivo=os.path.basename(arq), bytes=os.path.getsize(arq), fontes=list(D.fontes) if dc is not None else None,
                    dens_fonte_g_por_g=[float(v) for v in dc] if dc is not None else None,
                    formula="s = head(f) * ts + tm, com f em float32" + ("; frações = softmax(head_ff(f)) por orientação; densidade pelas frações = frações @ dens_fonte" if hasattr(rede, "head_ff") else "")
                            + "; o arquivo traz 'estado' (" + ("só os tensores treinados; os outros vêm do checkpoint público do backbone" if parcial else "rede inteira")
                            + ", média móvel dos pesos, meia precisão), 'tm', 'ts', 'n_saidas', 'n_ff' e 'dens_fonte'",
                    carregar="torch.load(arquivo, weights_only=True); cria_rede4(cfg, D, pretrained=" + str(parcial) + "); carrega_modelo4(rede, ck)",
                    **(dict(cabeca_sem_gradiente=dict(arquivo=f"cabecasg__{slug(cfg['nome'])}.npz", tambem_em="'cabeca_sg' do arquivo de pesos",
                                                      formula="sobre as mesmas features f (float32): z = W2 @ gelu(W1 @ ((f − mu) / sd) + b1) + b2; s_sg = z[:22] * ts + tm; "
                                                              "frações = softmax(z[22:]); densidade pelas frações = frações @ dens_fonte")) if sg is not None else {})))
    for c, v in saida.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and not np.isfinite(v[~np.isnan(v)]).all():
            raise RuntimeError(f"valores não finitos em '{c}'")
    saida.update(hist=json.dumps(hist), seg=time.time() - t0, listas_ext=json.dumps(listas_ext4(D, list(D.dom) if parte == "ext" else usados)))
    if dev.type == "cuda":
        saida["mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    dest = arq_parte(a.saida, cfg, parte)
    np.savez(dest + ".tmp.npz", **saida); os.replace(dest + ".tmp.npz", dest)
    print(f"{rot} pronto em {(time.time() - t0) / 60:.1f} min", flush=True)


def parte_pronta4(a, cfg, parte, D):
    """A parte da rodada 4 está em disco, completa e feita com as mesmas listas externas?"""
    C = Cache(a.cache, D)
    P = getattr(D, "P5", None)
    if cfg["tipo"] == "externo":
        ok = cfg["chave"] in D.dom and os.path.exists(os.path.join(pasta_dom4(C, D.dom[cfg["chave"]]), "pronto.json"))
        if ok and cfg.get("v5"):        # rodada 5: também os recortes da base
            C.prato_hash = {c: v["hash"] for c, v in P["por_base"].items()}
            ok = os.path.exists(os.path.join(pasta_prato5(C, D.dom[cfg["chave"]]), "pronto.json"))
        return ok
    if cfg.get("prato"):
        C.prato, C.prato_hash = True, {c: v["hash"] for c, v in P["por_base"].items()}
    if cfg["tipo"] == "feat":
        if cfg["chave"] == "n5k":
            return os.path.exists(os.path.join(C.pasta, f"feat__{cfg['bb']}__prato_{C.prato_hash['n5k']}.npy" if cfg.get("prato") else f"feat__{cfg['bb']}.npy"))
        return cfg["chave"] in D.dom and os.path.exists(os.path.join(pasta_dom4(C, D.dom[cfg["chave"]]), f"feat__{cfg['bb']}.npy"))
    arq = arq_parte(a.saida, cfg, parte)
    if not os.path.exists(arq):
        return False
    with np.load(arq, allow_pickle=False) as z:
        feita_com = json.loads(str(z["listas_ext"])) if "listas_ext" in z.files else {}
        chaves = list(D.dom) if parte == "ext" else doms_de4(cfg)
        if parte == "ext":
            if not set(D.dom) <= set(str(x) for x in z["dominios"]) | set(str(x) for x in z["dominios_fora"]):
                return False
            chaves = [str(x) for x in z["dominios"]]
        if any(ch not in D.dom or feita_com.get(ch, {}).get("hash") != D.dom[ch]["hash"] for ch in chaves):
            return False
        adiado = cfg.get("parada_interna")         # P3: o modelo final (pesos, conjuntos de previsão, servir) sai na parte 'ext'
        if parte in (("tudo", "ext") if adiado else ("tudo", "final")) and any(sp not in z.files for sp in D.conj):
            return False
    if parte in (("tudo", "ext") if adiado else ("tudo", "final")) and not os.path.exists(arq_de(a.saida, cfg, "servir")):
        return False
    if parte == "tudo" and not os.path.exists(arq_de(a.saida, cfg, "cabeca")):
        return False
    if adiado and parte == "final" and not os.path.exists(arq_parte(a.saida, cfg, "ext")):
        with np.load(arq, allow_pickle=False) as z:
            n_ep = sum(1 for h_ in json.loads(str(z["hist"])) if isinstance(h_.get("epoca"), int))
        return all(os.path.exists(os.path.join(pasta_fotos6(a, cfg), f"ep{e}.pt")) for e in range(n_ep + 1))
    if cfg["tipo"] in ("ft", "ftparcial") and parte != "ext" and not os.path.exists(arq_parte(a.saida, cfg, "ext")) and not os.path.exists(arq_modelo4(a.saida, cfg, parte)):
        return False        # sem os pesos do fold não dá para prever as fotos externas depois
    return True


def junta4(a, cfg, D):
    """Rodada 4: reúne as partes numa pred__*.npz (13.8): oof e teste2 com 1, 4 e 8 orientações (e frações, quando há);
    para cada domínio externo pronto, extcv__ (fotos de treino, pelo modelo do fold em que validam) e extreg__ (fotos de
    registro, pelo modelo final), cruas e com os ids; os vieses por domínio de cada fold; a procedência; ids, folds e
    hash de todas as listas e tabelas. Só resume métricas de CV das partes de treino."""
    s = slug(cfg["nome"])
    partes = {p: dict(np.load(arq_parte(a.saida, cfg, p), allow_pickle=False)) for p in partes_de(cfg)}
    fora = ("cfg", "parte", "dispositivo", "hist", "seg", "mem_gb", "listas_ext", "afim_A", "afim_b")
    if "tudo" in partes:
        pred = {c: v for c, v in partes["tudo"].items() if c not in fora and not c.startswith("cab")}
    else:
        pred, fin = {}, partes["final"]
        for c in [c for c in partes["fold0"] if c.startswith("oof")]:
            arr = np.full_like(partes["fold0"][c], np.nan)
            for q in range(5):
                m = (D.fold == q) & D.usa_tr
                arr[m] = partes[f"fold{q}"][c][m]
            pred[c] = arr
        adiado = bool(cfg.get("parada_interna"))     # P3: o modelo final e as previsões dele saem na parte 'ext'
        fin_ = partes["ext"] if adiado else fin
        pred.update({c: v for c, v in fin_.items() if any(c.startswith(sp) for sp in D.conj)})
        ordem = [f"fold{q}" for q in range(5)] + ["ext" if adiado else "final"]
        if "dens_fonte" in fin_:
            pred.update(dens_fonte=np.stack([partes[p]["dens_fonte"] for p in ordem]), fontes=fin_["fontes"])
        if adiado:
            pred.update(escolha=np.array([str(partes[f"fold{q}"]["escolha"]) for q in range(5)]), final_p3=partes["ext"]["final_p3"],
                        **{f"hp3_fold{q}_{c[4:]}": partes[f"fold{q}"][c] for q in range(5) for c in partes[f"fold{q}"] if c.startswith("hp3_")})
        for c in [c for c in fin if c.startswith("vies__")]:      # o viés é de cada modelo treinado (no P3, o final treina na parte 'final')
            pred[c] = np.stack([partes[p][c] for p in [f"fold{q}" for q in range(5)] + ["final"]])
    pred.update({c: v for c, v in partes["ext"].items() if c.startswith(("ext", "dominios"))})
    for c in [c for c in pred if (c.startswith("oof") or any(c.startswith(sp) for sp in D.conj))]:
        usados = D.usa_tr if c.startswith("oof") else D.usa_av[next(sp for sp in D.conj if c.startswith(sp))]
        assert not np.isnan(pred[c][usados]).any(), f"previsão faltando em '{c}'"
    for ch in [str(x) for x in pred["dominios"]]:
        assert not np.isnan(pred[f"extcv__{ch}"]).any() and not np.isnan(pred[f"extreg__{ch}"]).any(), f"previsão externa faltando em {ch}"
        d = D.dom[ch]
        assert list(pred[f"extcv_ids__{ch}"]) == [d["ids"][i] for i in np.where(d["treina"])[0]] and list(pred[f"extreg_ids__{ch}"]) == [d["ids"][i] for i in np.where(d["reg"])[0]]
    for suf in ORIENT:
        assert "oof" + suf in pred and all(sp + suf in pred for sp in D.conj), f"variante '{suf}' faltando"
    proc = procedencia4(cfg, D)
    np.savez(os.path.join(a.saida, f"pred__{s}.tmp.npz"), nome=cfg["nome"], apelido=cfg.get("apelido", ""), listas_hash=D.hash, listas2_hash=D.hash2 or "", fold=D.fold,
             ids_treino=np.array(D.ids), **{f"ids_{sp}": np.array(D.conj[sp]) for sp in D.conj}, procedencia=json.dumps(proc, ensure_ascii=False),
             listas_ext=json.dumps(listas_ext4(D, list(D.dom))), **pred)
    os.replace(os.path.join(a.saida, f"pred__{s}.tmp.npz"), os.path.join(a.saida, f"pred__{s}.npz"))
    with open(os.path.join(a.saida, f"pred__{s}.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["conjunto", "dish_id", "fold"] + nomes_saidas(D))
        for conj, ids, M, folds in [("treino_oof", D.ids, pred["oof"], D.fold)] + [(sp, D.conj[sp], pred[sp], None) for sp in D.conj]:
            for i, d in enumerate(ids):
                if not np.isnan(M[i, 0]):
                    wr.writerow([conj, d, "" if folds is None else int(folds[i])] + [f"{v:.5f}" for v in M[i]])
    cv = resumo_cv(D, pred["oof"])
    cv["rotas"] = {f"o{q}": rotas_cv(D, pred, suf) for suf, q in ORIENT.items()}
    # massa nas fotos de TREINO das bases externas (validação cruzada; a constante é a mediana do próprio conjunto: só para acompanhar)
    cv["ext_cv_massa"] = {}
    for ch in [str(x) for x in pred["dominios"]]:
        d = D.dom[ch]
        j = np.where(d["treina"])[0]
        if len(j):
            e = pred[f"extcv__{ch}"][:, I_M] - d["T"][j, I_M]
            cv["ext_cv_massa"][ch] = dict(fotos=int(len(j)), vies_mediano_cru_pct=round(float(100 * (np.exp(np.median(e)) - 1)), 1),
                                          a_20pct_com_constante=round(float(100 * np.mean(np.abs(np.exp(e - np.median(e)) - 1) <= 0.2)), 1))
    if cfg["tipo"] in ("ft", "ftparcial"):      # critério de aceitação de 13.7: a massa não pode piorar entre a época 0 e a última
        h = [json.loads(str(partes[f"fold{q}"]["hist"])) for q in range(5)]
        r0, rf = float(np.mean([x[0]["val_rmse_logm"] for x in h])), float(np.mean([x[-1]["val_rmse_logm"] for x in h]))
        cv["aceitacao"] = dict(rmse_logm_epoca0=r0, rmse_logm_ultima=rf, passou=bool(rf <= r0),
                               regra="média dos folds do RMSE de log-massa na última época não pode ser maior que na época 0 (13.7)")
        if cfg.get("parada_interna"):       # P3 (15.4): a "última" é o checkpoint escolhido pela validação interna
            esc = [x[-1] for x in h]
            cv["aceitacao"].update(regra="média dos folds do RMSE de log-massa do checkpoint escolhido pela validação interna não pode ser maior que na época 0 (13.7, 15.4)",
                                   epocas_escolhidas=[x.get("epoca_escolhida") for x in esc], curvas_internas=[x.get("curva_interna") for x in esc],
                                   modelo_final=json.loads(str(partes["ext"]["final_p3"])),
                                   sem_efeito=bool(all(x.get("epoca_escolhida") == 0 for x in esc)))
            if cv["aceitacao"]["sem_efeito"]:
                cv["aceitacao"]["nota"] = "a parada escolheu a época 0 em todos os folds: o P3 é a ridge de partida (ajuste fino sem efeito, 15.4)"
    grava_json(os.path.join(a.saida, f"log__{s}.json"),
               dict(nome=cfg["nome"], apelido=cfg.get("apelido"), cfg=cfg, cv_treino=cv, **proc, seg_total=float(sum(float(v["seg"]) for v in partes.values())),
                    partes={p: dict(seg=float(v["seg"]), dispositivo=str(v["dispositivo"]), mem_gb=float(v["mem_gb"]) if "mem_gb" in v else None,
                                    hist=json.loads(str(v["hist"]))) for p, v in partes.items()}))
    return cv


# ------------------------------------------------------------------ rodada 4: autoteste
def _dados_falsos4():
    """Listas mínimas para testar as funções da rodada 4 sem arquivos do Nutrition5k."""
    grupos, fontes = [f"g{i}" for i in range(8)], [f"f{i}" for i in range(10)]
    return SimpleNamespace(ids=["dish_1"], conj=dict(teste2=["dish_2"]), antigo={}, N=22, grupos=grupos, fontes=fontes, I_MG=slice(6, 14), I_PG=slice(14, 22),
                           hash="h", usa_tr=np.ones(1, bool), dom={}, dom_fora={})


def _dados_falsos_dom():
    """_dados_falsos4 com as seis bases (só o que a procedência lê), com o uso declarado de cada uma."""
    D = _dados_falsos4()
    for ch in DOMINIOS4:
        ok = ch in ("flavoria", "lefood", "mensa")
        D.dom[ch] = dict(chave=ch, dominio=ch, ids=["a"], peso_nut=np.ones(1), sha256_lista="s", sha256_divisao="s", resumo=dict(grupos_treino=1, treino=1),
                         licenca="CC BY 4.0" if ok else "CC BY-NC 4.0", uso_declarado="comercial_ok" if ok else "nao_comercial")
    return D


def teste_rodada4():
    """Seção 13, sem torch: a fila da rodada 4, os pesos por domínio, o fechamento dos rótulos das partes de registro
    (13.8), a procedência e a conferência de paridade."""
    import tempfile as tf
    ap = {c["apelido"]: c for c in CONFIGS4 if c.get("apelido")}
    nomes = [c["nome"] for c in CONFIGS4]
    assert {k: v["nome"] for k, v in ap.items()} == dict(
        R0="gpu4 ridge so400m", R1="gpu4 ridge so400m +ok", R2="gpu4 ridge so400m +ok+fpb", R3="gpu4 ridge so400m +omni", R4="gpu4 ridge so400m +tudo",
        H0="gpu4 fracoes so400m", H1="gpu4 fracoes so400m +omni", H2="gpu4 fracoes so400m +omni+ff", F1="gpu4 ft siglip2_b16_224 r2 fracoes",
        F2="gpu4 ft siglip2_b16_224 r2 fracoes faixa", F3="gpu4 ft siglip2_b16_224 r2 +tudo", P1="gpu4 ftparcial so400m")
    pos = lambda x: nomes.index(ap[x]["nome"])
    assert sorted(ap, key=pos) == ["R0", "H0", "R1", "R2", "R3", "R4", "H1", "F1", "P1", "F3", "H2", "F2"]     # ordem de prioridade de 13.6 (R0 e H0 primeiro: paridade)
    for lista in (CONFIGS4, CONFIGS_SMOKE4):
        ns = [c["nome"] for c in lista]
        for i, c in enumerate(lista):
            assert all(d in ns[:i] for d in c.get("depende", [])), c["nome"]
            assert all(d in ns for d in c.get("tardia", {}).get("terminal", [])) and (not c.get("tardia") or c["partes"][-1] == c["tardia"]["parte"])
    assert (ap["R1"]["treina_com"], ap["R2"]["treina_com"], ap["R3"]["treina_com"], ap["R4"]["treina_com"]) == (
        ["flavoria", "lefood", "mensa"], ["flavoria", "lefood", "mensa", "fpb"], ["omnifood"], DOMINIOS_TREINO4)
    assert ap["H1"]["treina_com"] == ["omnifood"] and ap["H2"]["ff_de"] == ["flavoria", "fpb", "mensa", "lefood"] and not ap["H0"]["treina_com"]
    assert "snapme" not in DOMINIOS_TREINO4 and not any("snapme" in doms_de4(c) for c in CONFIGS4)                 # o SNAPMe nunca treina
    u0 = CONFIGS3[-1]
    for x in ("F1", "F2", "F3"):        # os ajustes finos pequenos são a receita r2 do U0
        assert all(ap[x][k] == u0[k] for k in ("bb", "lr", "lld", "epocas", "bs", "acum", "escala", "wd", "drop_path", "ema", "aquec", "ckpt_grad")), x
    assert ap["F1"].get("fracoes4") and not ap["F1"].get("peso_faixa") and ap["F2"].get("fracoes4") and ap["F2"].get("peso_faixa")
    assert not ap["F3"].get("fracoes4") and not ap["F3"].get("peso_faixa") and ap["F3"]["externo4"] == DOMINIOS_TREINO4
    p1 = ap["P1"]
    assert (p1["blocos_treinaveis"], p1["epocas"], p1["bs"], p1["acum"], p1["lr"], p1["lld"], p1["wd"], p1["drop_path"], p1["ema"], p1["aquec"], p1["escala"]) == (
        4, 6, 16, 2, 2e-5, 0.75, 0.05, 0.1, 0.99, 0.1, (0.7, 1.3)) and p1.get("fracoes4") and not p1.get("peso_faixa")
    assert not any(c.get("professor") or c.get("peso_destila") for c in CONFIGS4)                                    # a destilação saiu
    # pesos por domínio (13.4)
    gr = dict(omnifood=3862, flavoria=1001, fpb=747, lefood=433, mensa=166)
    f = fracoes_dominio4(gr, 2281, ["flavoria", "lefood", "mensa"])
    assert [round(f[d], 2) for d in ("flavoria", "lefood", "mensa")] == [0.44, 0.19, 0.07]
    assert round(fracoes_dominio4(gr, 2281, ["omnifood"])["omnifood"], 2) == 0.50 and round(fracoes_dominio4(gr, 2281, ["fpb"])["fpb"], 2) == 0.33
    f = fracoes_dominio4(gr, 2281, list(gr))
    assert abs(sum(f.values()) - 1.0) < 1e-12 and abs(f["omnifood"] / f["mensa"] - 0.5 / (166 / 2281)) < 1e-9       # passou de 1,0: todos pelo mesmo fator
    # fechamento dos rótulos (13.8): de uma linha que não é de treino não sobra nenhuma coluna de alvo
    D = _dados_falsos4()
    alvos = dict(massa="100", proteina="9", kcal="50", gordura="1", carbo="2", **{f"m_{g}": "10" for g in D.grupos}, **{f"p_{g}": "1" for g in D.grupos},
                 **{f"ff_{g}": "0.1" for g in D.fontes})
    segredo = {k: "NAO-LER" for k in alvos}
    extras = dict(dominio="mensa", vista="zenital", url="", licenca="CC BY 4.0", uso_declarado="comercial_ok", proteina_fraca_g="NAO-LER", componentes="NAO-LER", massa_pesada_g="NAO-LER")
    papeis = ["treino", "treino", "registro", "registro_misto", "avaliacao_fraca", "fora", "treino"]
    lista = [dict(dish_id=f"x{i}", grupo=f"gr{i // 2}", arquivo=f"x{i}.jpg", **(alvos if p == "treino" else segredo), **extras) for i, p in enumerate(papeis)]
    div = [dict(dish_id=f"x{i}", dominio="mensa", grupo=f"gr{i // 2}", papel=p, fold_ext=str(i % 5) if p == "treino" else "-1", folds_proibidos=f"{i % 5};4" if p == "treino" else "",
                quadro_metrica="1", quadro_metrica_nome="", quadros_treino="", grupo_metrica=f"gr{i // 2}", cluster="NAO-LER", peso_nutricao="1" if p == "treino" else "NAO-LER",
                prato_principal="NAO-LER") for i, p in enumerate(papeis)]
    R = fecha_rotulos4(lista, div)
    assert [r["papel"] for r in R] == [p for p in papeis if p != "fora"]
    for r in R:
        assert "NAO-LER" not in r.values() and not any("fraca" in c or c in ("componentes", "cluster", "prato_principal", "massa_pesada_g") for c in r)
        if r["papel"] == "treino":
            assert all(r[c] == v for c, v in alvos.items()) and r["peso_nutricao"] == "1"
        else:
            assert set(r) <= set(COLUNAS_ABERTAS4) and not any(c in COLUNAS_ALVO4 or c.startswith(("m_", "p_", "ff_")) or c == "peso_nutricao" for c in r)
    try:
        fecha_rotulos4(lista, div[::-1]); raise SystemExit("lista e divisão desalinhadas passaram")
    except AssertionError:
        pass
    with tf.TemporaryDirectory() as pasta:      # o mesmo pelo leitor de verdade: os rótulos de registro nem chegam a ser lidos
        for nome, linhas in (("externo_mensa.csv", lista), ("divisao_r4_mensa.csv", div)):
            with open(os.path.join(pasta, nome), "w", newline="") as fo:
                w = csv.DictWriter(fo, list(linhas[0])); w.writeheader(); w.writerows(linhas)
        d = le_dominio4(pasta, "mensa", D)
        assert list(d["papel"]) == [p for p in papeis if p != "fora"] and d["treina"].sum() == 3 and d["reg"].sum() == 3
        assert np.isnan(d["T"][d["reg"]]).all() and np.isnan(d["FF"][d["reg"]]).all() and (d["peso_nut"][d["reg"]] == 0).all()       # registro: nenhum rótulo
        assert np.allclose(d["T"][d["treina"], I_M], np.log(100)) and np.isfinite(d["T"][d["treina"]]).all() and np.allclose(d["FF"][d["treina"]], 0.1)
        assert d["proib"][0].tolist() == [True, False, False, False, True] and not d["proib"][d["reg"]].any()
        assert d["licenca"] == "CC BY 4.0" and d["uso_declarado"] == "comercial_ok" and d["grupos_treino_lista"] == 2 and len(d["sha256_lista"]) == 64
        w = pesos_dominio4(d, None, 6.0)
        assert abs(w.sum() - 6.0) < 1e-12 and (w[d["reg"]] == 0).all()
        assert np.allclose(w[[0, 1]], 1.5) and abs(w[5] - 3.0) < 1e-12 and abs(w[[0, 1]].sum() - w[5]) < 1e-12        # peso igual por grupo, repartido entre as fotos
        w0 = pesos_dominio4(d, 0, 6.0)
        assert w0[0] == 0 and abs(w0[1] - 3.0) < 1e-12 and abs(w0.sum() - 6.0) < 1e-12                               # proibida no fold 0: fora desse treino
        assert not pesos_dominio4(d, 4, 6.0).any()
        # prato do Nutrition5k numa lista externa: recusado
        lista[0]["dish_id"] = div[0]["dish_id"] = "dish_2"
        for nome, linhas in (("externo_mensa.csv", lista), ("divisao_r4_mensa.csv", div)):
            with open(os.path.join(pasta, nome), "w", newline="") as fo:
                w = csv.DictWriter(fo, list(linhas[0])); w.writeheader(); w.writerows(linhas)
        try:
            le_dominio4(pasta, "mensa", D); raise SystemExit("prato do Nutrition5k passou")
        except AssertionError:
            pass
        # base que não treina (SNAPMe) com linha de treino: recusada
        for linhas in (lista, div):
            linhas[0]["dish_id"] = "x0"
            for r in linhas:
                r["dominio"] = "snapme"
        for nome, linhas in (("externo_snapme.csv", lista), ("divisao_r4_snapme.csv", div)):
            with open(os.path.join(pasta, nome), "w", newline="") as fo:
                w = csv.DictWriter(fo, list(linhas[0])); w.writeheader(); w.writerows(linhas)
        try:
            le_dominio4(pasta, "snapme", D); raise SystemExit("SNAPMe com linha de treino passou")
        except AssertionError:
            pass
    # alvos por domínio e procedência (13.4, 13.8)
    for ch in DOMINIOS_TREINO4:
        D.dom[ch] = dict(d, chave=ch, dominio=DOMINIOS4[ch]["dominio"], uso_declarado="comercial_ok" if ch in ("flavoria", "lefood", "mensa") else "nao_comercial",
                         peso_nut=np.array([1.0, 0.5, 0, 0, 0, 0.0]), resumo=dict(grupos_treino=3, treino=3))
    P, pf = alvos_dominio4(D.dom["flavoria"], D, ap["R1"])
    assert (P[:, I_M] == 1).all() and not P[:, 1:].any() and not pf.any()                               # base de massa: só a massa total
    P, pf = alvos_dominio4(D.dom["omnifood"], D, ap["R3"])
    assert (P[:, I_M] == 1).all() and not P[:, 1:].any() and not pf.any()                               # cabeça de massa: só massa, também no OmniFood8K
    P, pf = alvos_dominio4(D.dom["omnifood"], D, ap["H1"])
    assert P[0, [I_M, I_P, I_K, I_D, I_F, I_C]].tolist() == [1] * 6 and P[1, [I_P, I_D]].tolist() == [0.5, 0.5] and P[2, I_P] == 0 and (P[:, 6:] == 0.5).all() and (pf == 0.5).all()
    P, pf = alvos_dominio4(D.dom["flavoria"], D, ap["H1"])
    assert not P.any() and not pf.any()                                                                 # H1 não usa as bases de massa
    P, pf = alvos_dominio4(D.dom["flavoria"], D, ap["H2"])
    assert not P.any() and (pf == 0.5).all()                                                            # H2: só as frações delas
    P, pf = alvos_dominio4(D.dom["mensa"], D, ap["F3"])
    assert (P[:, I_M] == 1).all() and not P[:, 1:].any()
    assert sorted(x for x in ap if procedencia4(ap[x], D)["so_comercial_ok"]) == ["F1", "F2", "H0", "P1", "R0", "R1"]
    pr = procedencia4(ap["R2"], D)["treinado_com"]
    assert [x["dominio"] for x in pr] == ["nutrition5k", "flavoria", "lefood", "mensa", "fpb"] and all(x["alvos"] == "massa" for x in pr[1:]) and pr[4]["uso_declarado"] == "nao_comercial"
    assert colunas_extensivas4(D) == [0, 1, 2, 4, 5] + list(range(6, 22))                               # a densidade (3) não leva viés
    # paridade com o D1 (13.7)
    rng = np.random.default_rng(3)
    n = 400
    prot = np.r_[rng.uniform(1, 5, 80), rng.uniform(5, 15, 80), rng.uniform(15, 25, 80), rng.uniform(25, 40, 80), rng.uniform(40, 80, 80)]
    fx = np.digitize(prot, FAIXAS)
    Dp = SimpleNamespace(ids=[f"d{i}" for i in range(n)], proteina=prot, w_cv=(1 / np.bincount(fx, minlength=5))[fx])
    lm = rng.normal(5, 0.5, n); certo = np.arange(n) % 100 < 71                                        # 71 de cada 100 pratos de cada faixa certos; os outros, longe
    ld = np.log(np.where(certo, prot, 2 * prot + 10) / np.exp(lm) + EPS_D)
    r0 = dict(oof_o1=np.c_[lm, np.zeros((n, 5))], oof=np.c_[lm + 0.01 * rng.normal(size=n), np.zeros((n, 5))])
    h0 = dict(oof_o1=np.c_[np.zeros((n, 3)), ld, np.zeros((n, 2))], oof_dens_ff_o1=np.exp(ld) - EPS_D)
    ref = {d: [r0["oof_o1"][i, 0], r0["oof"][i, 0], ld[i], ld[i]] for i, d in enumerate(Dp.ids)}
    p = paridade_d1(Dp, r0, h0, ref)
    assert p["passou"] and abs(p["criterio_cv_d1"] - 71.0) < 1e-6 and min(p["corr"].values()) > 0.99999
    ruim = dict(r0, oof_o1=np.c_[lm + 0.2 * rng.normal(size=n), np.zeros((n, 5))])
    assert not paridade_d1(Dp, ruim, h0, ref)["passou"] and "correlação" in paridade_d1(Dp, ruim, h0, ref)["motivo"]
    h_ruim = dict(oof_o1=np.c_[np.zeros((n, 3)), ld + np.log(1.3), np.zeros((n, 2))], oof_dens_ff_o1=1.3 * np.exp(ld) - EPS_D)      # mesma ordem, outro nível: a correlação passa, o critério não
    pr = paridade_d1(Dp, r0, h_ruim, ref)
    assert not pr["passou"] and "critério" in pr["motivo"] and min(pr["corr"].values()) > 0.999
    parcial = dict(r0, oof_o1=np.where(np.arange(n)[:, None] < 40, np.nan, r0["oof_o1"]))
    assert not paridade_d1(Dp, parcial, h0, ref)["passou"] and not paridade_d1(Dp, r0, h0, {})["passou"]
    # a fila: afinidade de GPU, parte tardia, base que não chega
    cs = CONFIGS4
    novo = lambda: {slug(c["nome"]): {q: parte_inicial() for q in partes_de(c)} for c in cs}
    E = novo()
    marca = lambda nome, est_, partes=None: [E[slug(nome)][q].update(estado=est_) for q in (partes or E[slug(nome)])]
    for ch in DOMINIOS4:
        marca(f"externo {ch}", "rodando")
    assert proxima_acao(cs, E, {}, gpu=0)[1]["nome"] == "gpu4 feat so400m n5k"
    marca("gpu4 feat so400m n5k", "rodando")
    assert proxima_acao(cs, E, {}, gpu=1)[1:] == (ap["F1"], "fold0")                                    # a GPU B começa o F1 no minuto zero
    assert proxima_acao(cs, E, {}, gpu=0)[1:] == (ap["F1"], "fold0")                                    # sem nada seu para rodar, a GPU A ajuda
    marca("gpu4 feat so400m n5k", "feita")
    assert proxima_acao(cs, E, {}, gpu=0)[1:] == (ap["R0"], "tudo") and proxima_acao(cs, E, {}, gpu=1)[1:] == (ap["F1"], "fold0")
    marca(ap["R0"]["nome"], "feita", ["tudo"]); marca(ap["H0"]["nome"], "feita", ["tudo"])
    a_, c_, q_ = proxima_acao(cs, E, {}, gpu=0)
    assert (c_["nome"], q_) == ("gpu4 ftparcial so400m", "fold0")                                       # downloads em curso: 'ext' espera; a GPU A adianta o P1
    for ch in DOMINIOS4:
        marca(f"externo {ch}", "feita" if ch != "omnifood" else "falhou")
    caiu, rodou = [], []
    while True:
        a_, c_, q_ = proxima_acao(cs, E, {}, gpu=0)
        if a_ == "cancela":
            marca(c_["nome"], "cancelada"); caiu.append(c_.get("apelido") or c_["nome"])
        elif a_ == "roda":
            marca(c_["nome"], "feita", [q_]); rodou.append((c_.get("apelido") or c_["nome"], q_))
        else:
            break
    assert a_ == "fim" and sorted(caiu) == ["F3", "H1", "H2", "R3", "R4", "gpu4 feat so400m omnifood"]  # sem o OmniFood8K: só o que depende dele cai
    assert sorted(x for x, q in rodou if q == "ext") == sorted(["R0", "H0", "R1", "R2", "F1", "P1", "F2"]), rodou   # os outros terminam, com as previsões externas
    assert all(rodou.index((x, "ext")) > max(i for i, (y, q) in enumerate(rodou) if y == x and q != "ext") for x in ("R1", "F1", "P1"))     # 'ext' é a última parte
    assert rodou.index(("R0", "ext")) > rodou.index(("gpu4 feat so400m snapme", "tudo"))                  # 'ext' só depois de todas as features terminarem
    E = novo()
    cabe = simula_orcamento(cs, E, [0.0, 0.0], lambda c: 60.0 * c["est_min"] / len(partes_de(c)), 255 * 60)
    assert all(cabe[ap[x]["nome"]] for x in ap)                                                          # o plano de 13.6 cabe: GPU A ~250 min, GPU B ~225 min
    cabe = simula_orcamento(cs, E, [0.0, 0.0], lambda c: 60.0 * c["est_min"] / len(partes_de(c)), 200 * 60)
    assert not cabe[ap["P1"]["nome"]] and not cabe[ap["F2"]["nome"]] and cabe[ap["F1"]["nome"]] and cabe[ap["F3"]["nome"]] and cabe[ap["H2"]["nome"]]


def teste_rodada4_torch():
    """Seção 13, com torch: a ridge com várias bases externas respeita os folds proibidos, o peso zero e o viés por
    domínio; as amostras externas da cabeça de frações saem com as máscaras e os pesos certos; o α do ajuste fino."""
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(5)
    n, o, Dm = 60, 2, 10
    fold = np.arange(n) % 5
    X = rng.normal(size=(n, o, Dm)).astype(np.float32)
    T = (4.5 + 0.5 * X[:, 0, :1] + 0.1 * rng.normal(size=(n, 1)))
    todos = frozenset(range(5))

    def dominio(m, chave, oe, seed):
        r = np.random.default_rng(seed)
        Xe = (r.normal(size=(m, oe, Dm)) + 0.3).astype(np.float32)
        fe = np.arange(m) % 5
        proib = np.zeros((m, 5), bool); proib[np.arange(m), fe] = True; proib[: m // 3, 4] = True
        d = dict(chave=chave, ids=[f"{chave}{i}" for i in range(m)], treina=np.arange(m) < m - 4, reg=np.arange(m) >= m - 4, proib=proib, fold_ext=fe,
                 grupo=np.array([f"g{i // 2}" for i in range(m)]), peso_nut=np.ones(m))
        return d, Xe, 5.5 + 0.5 * Xe[:, 0, :1] + 0.1 * r.normal(size=(m, 1))
    da, Xa, ta = dominio(24, "a", 2, 1); db, Xb, tb = dominio(18, "b", 1, 2)
    monta = lambda ta_, tb_, fa=0.4, fb=0.3: RidgeMulti(X, T, fold, [dict(chave="a", X=Xa, t=ta_, d=da, frac=fa), dict(chave="b", X=Xb, t=tb_, d=db, frac=fb)])
    R = monta(ta, tb)
    alfa = R.alfa_cv(todos)
    prev = lambda R_, k: R_.prev_x(todos - {k}, X[fold == k], alfa).numpy()
    base = {k: prev(R, k) for k in range(5)}
    R0 = RidgeSub(X, T, fold)
    assert max(np.abs(prev(monta(ta, tb, 0.0, 0.0), k) - R0.prev_x(todos - {k}, X[fold == k], alfa).numpy()).max() for k in range(5)) < 1e-5      # peso zero: é a ridge comum
    assert max(np.abs(base[k] - R0.prev_x(todos - {k}, X[fold == k], alfa).numpy()).max() for k in range(5)) > 1e-3                              # com peso: muda
    for k in (0, 4):        # rótulos das fotos proibidas no fold k (e das de registro) não mexem no modelo que deixa k de fora
        t2 = ta.copy(); t2[da["proib"][:, k] | da["reg"]] += rng.normal(size=(int((da["proib"][:, k] | da["reg"]).sum()), 1)) + 2
        assert np.abs(prev(monta(t2, tb), k) - base[k]).max() < 1e-6
        outro = next(q for q in range(5) if q != k and (da["proib"][:, k] & ~da["proib"][:, q] & da["treina"]).any())
        assert np.abs(prev(monta(t2, tb), outro) - base[outro]).max() > 1e-4       # …mas mexem nos outros
    assert max(np.abs(prev(monta(ta + 3.0, tb - 2.0), k) - base[k]).max() for k in range(5)) < 1e-5     # viés por domínio: um deslocamento constante da base some
    v = R.vies(todos)
    wa = pesos_dominio4(da, None, 1.0)
    assert abs(v["a"][0] - ((ta[:, 0] * wa).sum() / wa.sum() - T.mean())) < 1e-9 and set(v) == {"a", "b"}
    assert np.isfinite(R.prev_x(todos, Xa, alfa).numpy()).all()                                           # previsão crua nas fotos externas
    # rodada 6 (H0c-m aninhada): dois folds de fora; as fotos proibidas em qualquer um deles não mexem no modelo
    t3 = ta.copy(); fora2 = da["proib"][:, 3] | da["proib"][:, 4] | da["reg"]
    t3[fora2] += 5.0
    p_a = R.prev_x(frozenset({0, 1, 2}), X[fold == 3], alfa).numpy()
    R3 = monta(t3, tb)
    assert np.abs(R3.prev_x(frozenset({0, 1, 2}), X[fold == 3], alfa).numpy() - p_a).max() < 1e-6
    # amostras externas da cabeça de frações (ext_cabeca4)
    D = _dados_falsos4()
    r = np.random.default_rng(7)

    def dom22(chave, m, oe):
        d, Xe, _ = dominio(m, chave, oe, 11)
        Td = np.full((m, 22), np.nan); Td[:, I_M] = 5 + r.normal(size=m)
        if chave == "omnifood":
            Td[:, 1:] = r.normal(size=(m, 21)) + 2; Td[1, [I_P, I_K, I_D, I_F, I_C]] = np.nan; d["peso_nut"] = np.r_[1.0, 0.0, 0.5, np.ones(m - 3)]
        Td[d["reg"]] = np.nan
        FFd = r.dirichlet(np.ones(10), m); FFd[d["reg"]] = np.nan
        return dict(d, T=Td, FF=FFd), Xe
    D.dom["omnifood"], Xo = dom22("omnifood", 20, 1); D.dom["flavoria"], Xf = dom22("flavoria", 16, 2)
    Tn = r.normal(size=(30, 22)) + 3
    h1 = dict(tipo="fracoes4", treina_com=["omnifood"], ff_de=[])
    h2 = dict(tipo="fracoes4", treina_com=["omnifood"], ff_de=["flavoria"])
    fts, fr = dict(omnifood=Xo, flavoria=Xf), dict(omnifood=0.5, flavoria=0.25)
    e1, v1 = ext_cabeca4(D, h1, fts, fr, 2, Tn)
    do = D.dom["omnifood"]
    pode = do["treina"] & ~do["proib"][:, 2]
    assert len(e1["X"]) == pode.sum() and abs(e1["c"].sum() - 0.5 * 30 * 8) < 1e-3 and set(v1) == {"omnifood"} and v1["omnifood"][I_D] == 0     # sem viés na densidade
    assert (e1["peso_alvo"][:, I_M] == 1).all() and set(np.unique(e1["peso_alvo"][:, I_P])) <= {0.0, 0.5, 1.0} and (e1["peso_alvo"][:, 6:] == 0.5).all() and (e1["peso_ff"] == 0.5).all()
    e1b, v1b = ext_cabeca4(D, h1, fts, fr, 2, Tn + 0.7)             # o viés acompanha a média do Nutrition5k do treino
    assert np.allclose(np.nan_to_num(e1b["T"][:, [I_M, I_P]]) - np.nan_to_num(e1["T"][:, [I_M, I_P]]), 0.7 * np.isfinite(e1["T"][:, [I_M, I_P]]), atol=1e-9)
    assert np.allclose(np.nan_to_num(e1b["T"][:, I_D]), np.nan_to_num(e1["T"][:, I_D]))
    e2, v2 = ext_cabeca4(D, h2, fts, fr, None, Tn)
    nf = int(D.dom["flavoria"]["treina"].sum()) * 2
    assert len(e2["X"]) == do["treina"].sum() + nf and np.isnan(e2["T"][-nf:]).all() and (e2["peso_alvo"][-nf:] == 0).all() and (e2["peso_ff"][-nf:] == 0.5).all()   # H2: das bases de massa, só as frações
    assert abs(e2["c"][-nf:].sum() - 0.25 * 30 * 8) < 1e-3 and np.isfinite(e2["FF"][-nf:]).all()
    cfg = dict(MLP3, oculto=8, epocas=2, lote=16)
    Xn = r.normal(size=(30, 2, 10)).astype(np.float32); FFn = r.dirichlet(np.ones(10), 30)
    pesos, prev_ = treina_cabeca(Xn, Tn, FFn, np.ones(30), np.full(10, 0.1), cfg, torch.device("cpu"), None, ext=e2)       # as linhas externas entram no treino sem erro
    assert np.isfinite(prev_(Xn)[0]).all()
    assert abs(alfa_aux4(torch.tensor([3.0, 4.0]), torch.tensor([6.0, 8.0])) - 0.5) < 1e-9 and alfa_aux4(torch.ones(4), torch.zeros(4)) == 1.0 and alfa_aux4(torch.ones(4), 0.1 * torch.ones(4)) == 1.0

# >>>>> RODADA 4: fim


# <<<<< RODADA 5: início
# ===================================================================== RODADA 5 (seção 14): enquadramento pelo prato e receitas corrigidas
# Tudo aqui foi escrito antes de rodar. Nada lê rótulo que não seja de treino, e o teste2 não é previsto (14.1).
PRATO5 = dict(frac_guia=0.80, aspecto="4:3", enquadramento=0.10, deslocamento=0.05,
              regra="janela 4:3 centrada na caixa da refeição, altura = maior lado da caixa ÷ 0,80, cor mediana da borda onde a janela sai da foto")
CONSULTAS5 = [0, 1, 5, 6]          # emenda a 14.3: a união só das consultas de prato e tigela
CONFERENCIA5 = "conferencia_r5.json"
PESOS_P2 = dict(massa=6.0, proteina=3.0, densidade=3.0, kcal=0.5, gordura=0.3, carbo=0.3, grupos=0.15)       # 14.4
_FT5B = "siglip2_b16_224"


def configs5(smoke=False):
    """As configurações da quinta execução (14.5), em ordem de prioridade (a ordem de corte é o inverso: F5, F4c, P2),
    mais as etapas sem modelo: uma por base externa (download e preparo das fotos inteiras e dos recortes) e uma por
    bloco de features do So400m sobre o recorte. As combinações sobre a foto inteira (R0–R4 × H0–H2) não são refeitas."""
    pre, so, ftb = ("smoke5", "vit_tiny_96x128", "vit_tiny_224") if smoke else ("gpu5", _SO4, _FT5B)
    ap = "tiny96" if smoke else "so400m"
    mlp = dict(MLP3, epocas=3) if smoke else MLP3
    ext = lambda d: f"externo {d}"
    feat = lambda d: f"{pre} feat {ap} prato {d}"
    todos_feat = [feat(d) for d in DOMINIOS4]
    cs = [dict(nome=ext(d), tipo="externo", chave=d, bb=None, est_min=0, v4=True, v5=True, livre=True, servico=False, partes=["tudo"]) for d in DOMINIOS4]
    cs.append(dict(nome=feat("n5k"), tipo="feat", chave="n5k", bb=so, bs=32, est_min=1 if smoke else 21, v4=True, v5=True, prato=True, livre=True, servico=False,
                   gpu=0, partes=["tudo"]))

    def cabeca(nome, tipo, apelido, est, doms=(), **kw):
        return dict(dict(nome=f"{pre} {tipo} {ap} prato{nome}", tipo=tipo + "4", apelido=apelido, bb=so, bs=32, est_min=est, v4=True, v5=True, prato=True,
                         sem_vies=True, livre=True, gpu=0, servico=False, treina_com=list(doms), ff_de=[], partes=["tudo", "ext"],
                         depende=[feat("n5k")] + [feat(d) for d in doms], tardia=dict(parte="ext", terminal=todos_feat)), **kw)
    r0, h0 = cabeca("", "ridge", "R0c", 1, servico=None), cabeca("", "fracoes", "H0c", 5, **mlp)
    cs += [r0, h0]
    est_feat = dict(flavoria=5, lefood=3, mensa=2, fpb=7, omnifood=19, snapme=3)
    cs += [dict(nome=feat(d), tipo="feat", chave=d, bb=so, bs=32, est_min=1 if smoke else est_feat[d], v4=True, v5=True, prato=True, livre=True, servico=False,
                gpu=0, partes=["tudo"], depende=[ext(d)]) for d in DOMINIOS4]
    ok = ["flavoria", "lefood", "mensa"]
    for nome, apel, est, doms in ((" +ok", "R1c", 2, ok), (" +ok+fpb", "R2c", 2, ok + ["fpb"]), (" +omni", "R3c", 2, ["omnifood"]), (" +tudo", "R4c", 3, DOMINIOS_TREINO4)):
        c = cabeca(nome, "ridge", apel, est, doms); c["depende"].append(r0["nome"]); c["plano"] = r0["nome"]
        cs.append(c)
    cs.append(cabeca(" +omni", "fracoes", "H1c", 8, ["omnifood"], **mlp))
    todos_ext = [ext(d) for d in DOMINIOS4]
    ft = lambda **kw: dict(v4=True, v5=True, livre=True, gpu=1, gpu_flexivel=True, val_cada=5, partes=PARTES_FT + ["ext"], tardia=dict(parte="ext", terminal=todos_ext),
                           fracoes_sg=True, **kw)
    rs = dict(epocas=2, bs=8) if smoke else {}
    f4 = _ft(ftb, "r2", 1 if smoke else 64, prefixo=pre, sufixo=" fracoes-sg", apelido="F4", receita_u0=True, **rs, **ft())
    p2 = dict(nome=f"{pre} ftparcial {ap} rotulos", tipo="ftparcial", apelido="P2", bb=so, receita="p2", est_min=1 if smoke else 170, pesos_alvo=dict(PESOS_P2),
              **dict(P1_RECEITA, **(dict(epocas=2, bs=4, blocos_treinaveis=2) if smoke else {})), **dict(ft(), gpu=0))
    f4c = _ft(ftb, "r2", 1 if smoke else 64, prefixo=pre, sufixo=" fracoes-sg prato", apelido="F4c", receita_u0=True, prato=True, escala=None,
              enquadramento=PRATO5["enquadramento"], **rs, **ft())
    f5 = _ft(ftb, "r2", 1 if smoke else 96, prefixo=pre, sufixo=" fracoes-sg +tudo", apelido="F5", externo4=list(DOMINIOS_TREINO4), depende=[ext(d) for d in DOMINIOS_TREINO4],
             **rs, **ft())
    return cs + [f4, p2, f4c, f5]


CONFIGS5, CONFIGS_SMOKE5 = configs5(), configs5(True)
RODADAS["5"] = [c["nome"] for c in CONFIGS5]


def pesos_alvo5(D, pesos):
    """Pesos por alvo da perda de rótulos do P2 (14.4): massa 6, proteína direta 3, densidade 3, kcal 0,5, gordura 0,3,
    carboidrato 0,3 e cada massa e proteína por grupo 0,15. A perda é a média ponderada (soma ÷ soma dos pesos)."""
    w = np.zeros(D.N, np.float32)
    w[[I_M, I_P, I_D, I_K, I_F, I_C]] = [pesos["massa"], pesos["proteina"], pesos["densidade"], pesos["kcal"], pesos["gordura"], pesos["carbo"]]
    w[D.I_MG] = pesos["grupos"]; w[D.I_PG] = pesos["grupos"]
    return w


# ------------------------------------------------------------------ rodada 5: tabelas de caixas da etapa 0
def acha_caixas5(pastas):
    """Pasta com as tabelas da etapa 0 (caixas_r5_<base>.csv): a primeira, das dadas e de /kaggle/input/**, que tem a do
    Nutrition5k."""
    for raiz in [p for p in pastas if p] + ["/kaggle/input"]:
        if not os.path.isdir(raiz):
            continue
        for base, dirs, arqs in os.walk(raiz):
            if base[len(raiz):].count(os.sep) > 4:
                dirs[:] = []
            if "caixas_r5_n5k.csv" in arqs:
                return base
    return None


def carrega_caixas5(pasta_listas, pastas, D, smoke=False):
    """As janelas do guia de cada foto (14.3), refeitas aqui das caixas brutas da etapa 0 com a regra da conferência
    (conferencia_r5.json: `consultas`, bases `reprovados` na conferência a olho, `conferida`). Assim a execução não
    depende de qual regra a etapa 0 usou na tabela principal. Base reprovada, sem tabela ou com foto faltando: fica com a
    foto inteira (a menor janela 4:3 que a contém), marcada. -> dict(info, por_base={base: dict(jan, sha, ...)})."""
    import deteccao_r5 as E
    conf = json.load(open(os.path.join(pasta_listas, CONFERENCIA5)))
    consultas = set(conf.get("consultas", CONSULTAS5))
    reprov = set(conf.get("reprovados", []))
    pasta = acha_caixas5(pastas)
    info = dict(pasta=pasta, regra=conf.get("regra"), consultas=sorted(consultas), reprovados=sorted(reprov), conferida=bool(conf.get("conferida")),
                conferencia=conf, janela=PRATO5, detector_licenca=E.LICENCA_DETECTOR, por_base={})
    ids_de = {"n5k": [d for d, u in zip(D.ids, D.usa_tr) if u]}
    ids_de.update({ch: list(d["ids"]) for ch, d in D.dom.items()})
    for ch, ids in ids_de.items():
        b = dict(jan={}, sha={}, situacao=None)
        arq = os.path.join(pasta, f"caixas_r5_{ch}.csv") if pasta else None
        if ch in reprov:
            b["situacao"] = "reprovada na conferência a olho: foto inteira"
        elif not arq or not os.path.exists(arq):
            b["situacao"] = "sem tabela da etapa 0: foto inteira"
        else:
            L = {r["id"]: r for r in csv.DictReader(open(arq))}
            falta = [x for x in ids if x not in L]
            if falta and not smoke:
                b["situacao"] = f"tabela sem {len(falta)} fotos (ex.: {falta[0]}): foto inteira"
            else:
                met = {}
                for x in ids:
                    if x not in L:
                        continue
                    r = E.refaz_linha(L[x], consultas)
                    b["jan"][x] = (float(r["janela_cx"]), float(r["janela_cy"]), float(r["janela_alt"]), float(r["janela_larg"]))
                    b["sha"][x] = r["sha256"]
                    met[r["metodo"]] = met.get(r["metodo"], 0) + 1
                b.update(situacao="recorte pelo prato", metodos=met, sha256_tabela=E.sha256(arq), taxa_sem_prato=round(met.get("sem_prato", 0) / max(1, len(b["jan"])), 4),
                         versao_detector=next(iter(L.values()))["versao_detector"] if L else None)
        b["hash"] = hashlib.sha256(json.dumps(sorted((k, [round(v, 1) for v in j]) for k, j in b["jan"].items())).encode()).hexdigest()[:12]
        info["por_base"][ch] = b
    info["n5k_ok"] = bool(info["por_base"]["n5k"]["jan"]) and "n5k" not in reprov
    return info


def janela_desvio(jan, s, dx, dy):
    """A janela com um erro de enquadramento (14.3): escala s (lado multiplicado) e deslocamento (dx, dy) em fração do lado."""
    cx, cy, alt, larg = jan
    return cx + dx * larg, cy + dy * alt, alt * s, larg * s


def desvios5(rng, n, frac=PRATO5["enquadramento"], desl=PRATO5["deslocamento"]):
    """n erros de enquadramento: escala uniforme em ±frac (em log) e deslocamento uniforme em ±desl do lado."""
    return np.c_[np.exp(rng.uniform(np.log(1 - frac), np.log(1 + frac), n)), rng.uniform(-desl, desl, (n, 2))]


def recorte_gpu(x, jan, cor, hw):
    """O recorte do guia no dispositivo: x [C, H, W] em [0, 1], janela (cx, cy, alt, larg) em pixels de x, cor [C] onde a
    janela sai da foto; redimensionado (bicúbico com antialias) para hw. É a conta do app: recortar o guia e redimensionar."""
    import torch
    C_, H, W = x.shape
    cx, cy, alt, larg = jan
    x0, y0 = int(math.floor(cx - larg / 2)), int(math.floor(cy - alt / 2))
    x1, y1 = max(x0 + 1, int(math.ceil(cx + larg / 2))), max(y0 + 1, int(math.ceil(cy + alt / 2)))
    tela = cor.view(C_, 1, 1).expand(C_, y1 - y0, x1 - x0).clone()
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        tela[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = x[:, sy0:sy1, sx0:sx1]
    return torch.nn.functional.interpolate(tela[None], size=hw, mode="bicubic", antialias=True, align_corners=False)[0]


def feats_prato_n5k(arq, bb, D, C, dev, bs, prazo):
    """Features do backbone congelado sobre o recorte do Nutrition5k (14.3): 8 vistas por prato; a vista k é a orientação
    k do grupo diedral aplicada a um recorte com erro de enquadramento sorteado (semente por prato e vista); a vista 0 é o
    recorte exato, na orientação original. [n, 8, 2·dim] float16, nos pratos de D.todos."""
    import torch
    t0 = time.time()
    rede = cria_rede(bb, D.N).to(dev).eval()
    hw = BACKBONES[bb]["hw"]
    n = len(D.todos)
    X = np.zeros((n, 8, rede.dim), np.float16)
    with torch.inference_mode():
        for i in range(0, n, bs):
            if time.time() > prazo:
                raise SemOrcamento()
            ids = D.todos[i:i + bs]
            full = torch.from_numpy(np.ascontiguousarray(C.rgb[[C.pos[d] for d in ids]])).to(dev).permute(0, 3, 1, 2).float().div_(255)
            cor = torch.from_numpy(C.cor[[C.pos[d] for d in ids]]).to(dev)
            for k in range(8):
                xs = []
                for b, d in enumerate(ids):
                    jan = C.janelas.get(d) or (C.rgb.shape[2] / 2, C.rgb.shape[1] / 2, float(C.rgb.shape[1]), float(C.rgb.shape[1]) * 4 / 3)      # sem janela: a foto inteira
                    if k:
                        s, dx, dy = desvios5(np.random.default_rng(zlib.crc32(f"{d}|{k}".encode())), 1)[0]
                        jan = janela_desvio(jan, s, dx, dy)
                    xs.append(recorte_gpu(full[b], jan, cor[b], hw))
                x = orienta(torch.stack(xs).clamp_(0, 1), k)
                with autocast(dev):
                    f = rede.feats(x)
                X[i:i + len(ids), k] = f.float().cpu().numpy()
    tmp = f"{arq}.{os.getpid()}.npy"
    np.save(tmp, X); os.replace(tmp, arq)
    print(f"features congeladas {bb} sobre o recorte (Nutrition5k): {X.shape} em {time.time() - t0:.0f}s", flush=True)


def pasta_prato5(C, d):
    return os.path.join(C.raiz, f"ext5p_{d['chave']}_{d['hash']}_{C.prato_hash[d['chave']]}")


def prepara_prato5(a, D, ch, P):
    """Recortes de uma base externa prontos no cache (640×480, a janela do guia de cada foto, tirada da foto original
    com a orientação do EXIF e o retrato girado, como na etapa 0). Confere que cada foto é a mesma que a etapa 0 viu
    (sha256). Base sem janelas (reprovada, sem tabela): o recorte é a foto inteira encaixada no 4:3, marcado."""
    import deteccao_r5 as E
    d = D.dom[ch]
    C = Cache(a.cache, D); C.prato_hash = {c: v["hash"] for c, v in P["por_base"].items()}
    pasta = pasta_prato5(C, d)
    if os.path.exists(os.path.join(pasta, "pronto.json")):
        return json.load(open(os.path.join(pasta, "pronto.json")))
    raiz = d["raiz"] if d["raiz"] and os.path.isdir(d["raiz"]) else baixa_dominio4(a, d)
    corte = d.get("raiz_espelho", (None, 0))[1]
    de = lambda f: acha_foto(raiz, "/".join(f.split("/")[corte:]))
    jans, shas = P["por_base"][ch]["jan"], P["por_base"][ch]["sha"]
    os.makedirs(pasta, exist_ok=True)
    t0, n = time.time(), len(d["ids"])
    mm = np.lib.format.open_memmap(os.path.join(pasta, "rgb.tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, C.H, C.W, 3))
    cor = np.zeros((n, 3), np.float32)
    difere, sem_jan = [], []

    def um(i):
        arq = de(d["arquivos"][i])
        img, _ = E.abre_canonica(arq)
        h, w = img.shape[:2]
        jan = jans.get(d["ids"][i])
        if jan is None:
            sem_jan.append(d["ids"][i]); jan = E.janela(None, w, h)
        elif shas.get(d["ids"][i]) and E.sha256(arq) != shas[d["ids"][i]]:
            difere.append(d["ids"][i]); jan = E.janela(None, w, h)
        cb = E.cor_borda(img)
        mm[i] = E.recorta(img, jan, (C.W, C.H), cb)
        cor[i] = cb / 255
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as ex:
        list(ex.map(um, range(n)))
    mm.flush(); del mm
    np.save(os.path.join(pasta, "cor.npy"), cor)
    os.replace(os.path.join(pasta, "rgb.tmp.npy"), os.path.join(pasta, "rgb.npy"))
    r = dict(n=n, foto_inteira_sem_janela=len(sem_jan), foto_diferente_da_etapa0=len(difere), ex_diferente=difere[:3], seg=round(time.time() - t0, 1))
    grava_json(os.path.join(pasta, "pronto.json"), r)
    diz(f"recortes de {ch}: {n} fotos em {time.time() - t0:.0f}s ({len(sem_jan)} com a foto inteira, {len(difere)} diferentes da etapa 0)")
    return r


def abre_prato5(C, D, P, so=None):
    """Como abre_dominios4, com os recortes das bases externas no lugar das fotos inteiras e as janelas do Nutrition5k
    (o recorte do Nutrition5k é feito no dispositivo, a partir da foto original do cache)."""
    C.prato, C.prato_hash = True, {c: v["hash"] for c, v in P["por_base"].items()}
    ab = abre_dominios4(C, D, so)
    C.janelas = P["por_base"]["n5k"]["jan"]
    C.janelas_pos = {C.pos[x]: j for x, j in C.janelas.items() if x in C.pos}
    return ab


# ------------------------------------------------------------------ rodada 5: cabeça de frações sem gradiente para o backbone (14.4)
class CabecaSG:
    """MLP como a do H0 (oculta de 512, GELU, dropout de 0,3; 22 saídas e 10 logits) sobre stop_gradient(features), com o
    otimizador dela (AdamW, passo de 1e-3, decaimento de 1e-2, aquecimento de 10% e cosseno no mesmo número de passos da
    rede) e as três perdas de 12.3 com o peso por faixa, que só agem nela. Usa geradores próprios na iniciação e no
    dropout: não toca no gerador global, então a rede e a cabeça de 22 saídas treinam bit a bit como sem ela. A
    padronização das features vem das features congeladas do início (as mesmas da iniciação da cabeça linear).
    Média móvel dos pesos (0,99), como a rede."""

    def __init__(self, X0, D, tr, dev, semente, total, oculto=512, drop=0.3, lr=1e-3, wd=1e-2, ema=0.99):
        import torch
        self.t, self.dev, self.drop, self.ema_d = torch, dev, drop, ema
        nt, K = D.N, len(D.fontes)
        self.nt, self.K, self.n5k = nt, K, len(D.ids)
        Dm = X0.shape[-1]
        X0 = X0.reshape(-1, Dm).astype(np.float64)
        self.mu, self.sd = X0.mean(0).astype(np.float32), (X0.std(0) + 1e-6).astype(np.float32)
        T = D.T
        self.tm, self.ts = T[tr].mean(0).astype(np.float32), (T[tr].std(0) + 1e-6).astype(np.float32)
        self.dc = dens_por_fonte(D.FF[tr], D.massa[tr], D.proteina[tr])
        w = np.zeros(len(D.ids), np.float32); w[tr] = peso_faixa(D.proteina[tr])
        g = torch.Generator().manual_seed(int(semente) + 7919)
        self.gd = torch.Generator(device=dev.type); self.gd.manual_seed(int(semente) + 104729)
        u = lambda forma, fan: (torch.rand(forma, generator=g) * 2 - 1) / math.sqrt(fan)
        self.p = [torch.nn.Parameter(v.to(dev)) for v in (u((oculto, Dm), Dm), u((oculto,), Dm), u((nt + K, oculto), oculto), u((nt + K,), oculto))]
        self.e = [v.detach().clone() for v in self.p]
        self.opt = torch.optim.AdamW(self.p, lr=lr, weight_decay=wd)
        n_aq = max(1, int(0.1 * total))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: (s + 1) / n_aq if s < n_aq else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - n_aq) / max(1, total - n_aq)))))
        f32 = lambda v: torch.from_numpy(np.asarray(v, np.float32)).to(dev)
        self.mu_t, self.sd_t, self.tm_t, self.ts_t, self.dct = f32(self.mu), f32(self.sd), f32(self.tm), f32(self.ts), f32(self.dc)
        self.Y = f32(np.nan_to_num((T - self.tm) / self.ts)); self.FF = f32(D.FF); self.W = f32(w)
        self.passos, self.perdas = 0, []

    def _z(self, f, ps, treino):
        t = self.t
        F = t.nn.functional
        z = (f.float() - self.mu_t) / self.sd_t
        if treino:
            z = z * (t.rand(z.shape, generator=self.gd, device=z.device) >= self.drop) / (1 - self.drop)
        h = F.gelu(z @ ps[0].T + ps[1])
        if treino:
            h = h * (t.rand(h.shape, generator=self.gd, device=h.device) >= self.drop) / (1 - self.drop)
        return h @ ps[2].T + ps[3]

    def passo(self, f, jt):
        """Um passo só da cabeça, nas amostras do Nutrition5k do lote (as externas não têm frações nem densidade)."""
        t = self.t
        m = jt < self.n5k
        if not bool(m.any()):
            return
        f, jt = f.detach()[m], jt[m]
        sl1 = t.nn.functional.smooth_l1_loss
        out = self._z(f, self.p, True)
        w = self.W[jt]
        fr = t.softmax(out[:, self.nt:], 1)
        alvo_d = self.Y[jt][:, I_D] * self.ts_t[I_D] + self.tm_t[I_D]
        p_d = sl1((t.log(fr @ self.dct + EPS_D) - alvo_d) / self.ts_t[I_D], t.zeros_like(alvo_d), reduction="none")
        perda = (sl1(out[:, :self.nt], self.Y[jt], reduction="none").mean(1) * w).mean() + 10 * (((fr - self.FF[jt]) ** 2).sum(1) * w).mean() + (p_d * w).mean()
        self.opt.zero_grad(set_to_none=True); perda.backward(); self.opt.step(); self.sched.step()
        self.passos += 1
        d = min(self.ema_d, (1 + self.passos) / (10 + self.passos))
        with t.no_grad():
            for e, p in zip(self.e, self.p):
                e.lerp_(p.detach(), 1 - d)
        self.perdas.append(float(perda.detach()))

    def preve(self, f):
        """(22 saídas em log, frações) com os pesos da média móvel, em float32."""
        with self.t.no_grad():
            z = self._z(f, self.e, False)
        return z[:, :self.nt] * self.ts_t + self.tm_t, self.t.softmax(z[:, self.nt:], 1)

    def arrays(self):
        """No formato das cabeças de frações (aplica_cabeca4 e servir): mu, sd, W1, b1, W2, b2, tm, ts, dens_fonte."""
        e = [v.detach().float().cpu().numpy() for v in self.e]
        return dict(mu=self.mu, sd=self.sd, W1=e[0], b1=e[1], W2=e[2], b2=e[3], tm=self.tm, ts=self.ts, dens_fonte=self.dc.astype(np.float32))

    def media_perda(self, desde=0):
        v = self.perdas[desde:]
        return float(np.mean(v)) if v else None


def aplica_sg_torch(z, f):
    """Uma cabeça de frações gravada (arrays de CabecaSG.arrays ou do H0) sobre features f [B, D] no dispositivo:
    (22 saídas em log, frações)."""
    import torch
    t = lambda k: (z[k] if torch.is_tensor(z[k]) else torch.from_numpy(np.asarray(z[k], np.float32))).to(f.device).float()
    h = torch.nn.functional.gelu(((f.float() - t("mu")) / t("sd")) @ t("W1").T + t("b1"))
    o = h @ t("W2").T + t("b2")
    nt = len(z["tm"])
    return o[:, :nt] * t("ts") + t("tm"), torch.softmax(o[:, nt:], 1)

# ------------------------------------------------------------------ rodada 5: autoteste
def teste_rodada5():
    """Seção 14, sem torch: a fila da quinta execução, os pesos do P2, o erro de enquadramento e as janelas refeitas das
    caixas brutas da etapa 0 com a regra da conferência."""
    import tempfile as tf
    ap = {c["apelido"]: c for c in CONFIGS5 if c.get("apelido")}
    nomes = [c["nome"] for c in CONFIGS5]
    pos = lambda x: nomes.index(ap[x]["nome"])
    assert sorted(ap, key=pos) == ["R0c", "H0c", "R1c", "R2c", "R3c", "R4c", "H1c", "F4", "P2", "F4c", "F5"]       # prioridade de 14.5; corte: F5, F4c, P2
    assert not any(c.get("apelido") in ("R0", "R1", "R2", "R3", "R4", "H0", "H1", "H2") for c in CONFIGS5)          # a foto inteira não é refeita
    for lista in (CONFIGS5, CONFIGS_SMOKE5):
        ns = [c["nome"] for c in lista]
        for i, c in enumerate(lista):
            assert all(d in ns[:i] for d in c.get("depende", [])), c["nome"]
    for x in ("R0c", "R1c", "R2c", "R3c", "R4c", "H0c", "H1c", "F4c"):
        assert ap[x].get("prato"), x
    assert all(ap[x].get("sem_vies") for x in ("R0c", "R1c", "R2c", "R3c", "R4c", "H0c", "H1c")) and not ap["F5"].get("prato")
    assert [ap[x]["treina_com"] for x in ("R1c", "R2c", "R3c", "R4c", "H1c")] == [["flavoria", "lefood", "mensa"], ["flavoria", "lefood", "mensa", "fpb"], ["omnifood"],
                                                                                   DOMINIOS_TREINO4, ["omnifood"]]
    u0 = CONFIGS3[-1]
    for x in ("F4", "F4c", "F5"):       # a receita r2 do U0, com a cabeça de frações sem gradiente
        assert all(ap[x][k] == u0[k] for k in ("bb", "lr", "lld", "epocas", "bs", "acum", "wd", "drop_path", "ema", "aquec", "ckpt_grad")) and ap[x]["fracoes_sg"], x
        assert not ap[x].get("fracoes4") and not ap[x].get("fracoes") and not ap[x].get("peso_faixa") and not ap[x].get("professor"), x
    assert ap["F4"]["receita_u0"] and ap["F4c"]["receita_u0"] and ap["F4"]["escala"] == u0["escala"] and ap["F4c"]["escala"] is None and ap["F4c"]["enquadramento"] == 0.10
    assert ap["F5"]["externo4"] == DOMINIOS_TREINO4 and not ap["F5"].get("receita_u0")
    p2 = ap["P2"]
    assert (p2["blocos_treinaveis"], p2["epocas"], p2["lr"], p2["lld"], p2["ema"], p2["escala"]) == (4, 6, 2e-5, 0.75, 0.99, (0.7, 1.3)) and not p2.get("fracoes4") and p2["fracoes_sg"]
    D = _dados_falsos_dom()
    w = pesos_alvo5(D, PESOS_P2)
    assert w[[I_M, I_P, I_D, I_K, I_F, I_C]].tolist() == [6.0, 3.0, 3.0, 0.5, 0.30000001192092896, 0.30000001192092896] and abs(w[I_M] / w.sum() - 0.387) < 0.001   # massa: 39% do peso
    assert sorted(x for x in ap if procedencia4(ap[x], D)["so_comercial_ok"]) == ["F4", "F4c", "H0c", "P2", "R0c", "R1c"]
    # erro de enquadramento
    d = desvios5(np.random.default_rng(0), 5000)
    assert d[:, 0].min() >= 0.9 - 1e-9 and d[:, 0].max() <= 1.1 + 1e-9 and np.abs(d[:, 1:]).max() <= 0.05
    assert janela_desvio((100, 100, 200, 300), 1.1, 0.05, -0.05) == (115.0, 90.0, 220.00000000000003, 330.0)
    # a fila: GPU A começa pelas features do recorte, GPU B pelo F4; ordem de corte F5, F4c, P2
    novo = lambda: {slug(c["nome"]): {q: parte_inicial() for q in partes_de(c)} for c in CONFIGS5}
    E_ = novo()
    for c in CONFIGS5:
        if c["tipo"] == "externo":
            E_[slug(c["nome"])]["tudo"]["estado"] = "rodando"
    assert proxima_acao(CONFIGS5, E_, {}, gpu=0)[1]["nome"] == "gpu5 feat so400m prato n5k" and proxima_acao(CONFIGS5, E_, {}, gpu=1)[1:] == (ap["F4"], "fold0")
    seg = lambda c: 60.0 * c["est_min"] / len(partes_de(c))
    cabe = simula_orcamento(CONFIGS5, novo(), [0.0, 0.0], seg, 240 * 60)
    assert all(cabe[ap[x]["nome"]] for x in ap), cabe
    cabe = simula_orcamento(CONFIGS5, novo(), [0.0, 0.0], seg, 180 * 60)
    assert not cabe[ap["F5"]["nome"]] and cabe[ap["F4"]["nome"]] and cabe[ap["R0c"]["nome"]] and cabe[ap["H1c"]["nome"]]
    # P2 atrasado na GPU A: em vez de cortar o P2, a GPU B pega partes dele e o corte cai no F5 (depois no F4c), como 14.5
    E_ = novo()
    for c in CONFIGS5:
        if c.get("apelido") not in ("F4", "P2", "F4c", "F5"):
            for q in E_[slug(c["nome"])].values():
                q["estado"] = "feita"
    cabe = simula_orcamento(CONFIGS5, E_, [60 * 60.0, 0.0], seg, 215 * 60)        # A livre aos 60 min: o P2 só nela acabaria aos 230
    assert cabe[ap["P2"]["nome"]] == "qualquer_gpu" and cabe[ap["F4"]["nome"]] is True and cabe[ap["F4c"]["nome"]] is True and cabe[ap["F5"]["nome"]] is False, cabe
    assert proxima_acao(CONFIGS5, E_, cabe, gpu=1)[1:] == (ap["F4"], "fold0")       # na GPU B, o F4 (prioridade 2) vem antes
    for q in E_[slug(ap["F4"]["nome"])].values():
        q["estado"] = "feita"
    cabe = simula_orcamento(CONFIGS5, E_, [60 * 60.0, 64 * 60.0], seg, 215 * 60)
    assert cabe[ap["P2"]["nome"]] == "qualquer_gpu" and proxima_acao(CONFIGS5, E_, cabe, gpu=1)[1:] == (ap["P2"], "fold0"), cabe    # e depois, o P2 antes do F4c
    assert proxima_acao(CONFIGS5, E_, {}, gpu=1)[1:] == (ap["F4c"], "fold0")       # cabendo na GPU A, o P2 fica nela
    cs4 = configs4()
    assert set(simula_orcamento(cs4, {slug(c["nome"]): {q: parte_inicial() for q in partes_de(c)} for c in cs4}, [0.0, 0.0], seg, 100 * 60).values()) <= {True, False}
    # janelas refeitas das caixas brutas, com a regra da conferência; base reprovada ou sem tabela: foto inteira
    with tf.TemporaryDirectory() as t:
        brutas = json.dumps([[100, 50, 500, 450, 0.6, 2], [200, 100, 400, 300, 0.5, 5]])      # bandeja (2) e prato (5)
        linha = dict(id="dish_1", dominio="n5k", arquivo="dish_1/rgb.png", sha256="abc", largura="640", altura="480", girada="0", metodo="owlv2", x0="100", y0="50", x1="500",
                     y1="450", nota="0.6", n_caixas="2", janela_cx="300", janela_cy="250", janela_alt="500", janela_larg="666.7", versao_detector="v", iou_uniao_so_prato="",
                     caixa_so_prato="", caixa_elipse="", caixas_brutas=brutas)
        with open(os.path.join(t, "caixas_r5_n5k.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, list(linha), lineterminator="\n"); wr.writeheader(); wr.writerow(linha)
        with open(os.path.join(t, "caixas_r5_mensa.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, list(linha), lineterminator="\n"); wr.writeheader(); wr.writerow(dict(linha, id="ext:mensa:x"))
        D.dom = {"mensa": dict(ids=["ext:mensa:x"]), "lefood": dict(ids=["ext:lefood:y"])}
        D.usa_tr = np.ones(1, bool)
        json.dump(dict(regra="so_prato", consultas=[0, 1, 5, 6], reprovados=["mensa"], conferida=True), open(os.path.join(t, CONFERENCIA5), "w"))
        P = carrega_caixas5(t, [t], D)
        assert P["por_base"]["n5k"]["jan"]["dish_1"] == (300.0, 200.0, 250.0, 333.3) and P["n5k_ok"]          # só o prato: altura 200 ÷ 0,8 (largura com 1 casa, como na tabela)
        assert P["por_base"]["mensa"]["jan"] == {} and "reprovada" in P["por_base"]["mensa"]["situacao"] and "sem tabela" in P["por_base"]["lefood"]["situacao"]
        json.dump(dict(regra="uniao", consultas=list(range(7)), reprovados=[], conferida=True), open(os.path.join(t, CONFERENCIA5), "w"))
        P = carrega_caixas5(t, [t], D)
        assert P["por_base"]["n5k"]["jan"]["dish_1"] == (300.0, 250.0, 500.0, 666.7) and P["por_base"]["mensa"]["jan"]       # a regra original: a bandeja
        json.dump(dict(regra="so_prato", consultas=[0, 1, 5, 6], reprovados=["n5k"], conferida=True), open(os.path.join(t, CONFERENCIA5), "w"))
        assert not carrega_caixas5(t, [t], D)["n5k_ok"]                 # Nutrition5k reprovado: a alavanca 1 é cancelada


def teste_rodada5_torch():
    """Seção 14.4, com torch e um backbone minúsculo em CPU: com a cabeça de frações sem gradiente, a rede e a cabeça de
    22 saídas do F4 treinam BIT A BIT como o U0, na mesma semente (pesos da média móvel, previsões e perdas por época).
    E a cabeça sem gradiente de fato treina. Também: o recorte no dispositivo repete o recorte da etapa 0."""
    import copy, tempfile as tf, torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    n = 20
    D = _dados_falsos4()
    D.ids = [f"d{i}" for i in range(n)]
    D.fold = np.arange(n) % 5
    D.massa = np.exp(rng.normal(5, 0.4, n)); D.proteina = D.massa * rng.uniform(0.02, 0.2, n)
    D.FF = rng.dirichlet(np.ones(10), n)
    D.T = np.c_[np.log(D.massa), np.log1p(D.proteina), rng.normal(5, 1, (n, 1)), np.log(D.proteina / D.massa + EPS_D), rng.normal(2, 1, (n, 18))]
    fx = np.digitize(D.proteina, FAIXAS); D.w_cv = (1 / np.bincount(fx, minlength=5))[fx]
    D.usa_tr = np.ones(n, bool); D.todos = list(D.ids); D.segmentos = [(0, n)]; D.conj = {}
    tr, va = np.where(D.fold != 0)[0], np.where(D.fold == 0)[0]
    base = dict(next(c for c in CONFIGS_SMOKE5 if c.get("apelido") == "F4"), epocas=2, bs=4, val_cada=1)
    u0 = {k: v for k, v in base.items() if k not in ("fracoes_sg",)}
    orig = globals()["cria_rede"]

    def cria_sem_pesos(bb, n_saidas, pretrained=True, **kw):
        return orig(bb, n_saidas, pretrained=False, **kw)

    def roda(cfg):
        with tf.TemporaryDirectory() as t:
            C = SimpleNamespace(pasta=t, pos={d: i for i, d in enumerate(D.ids)}, rgb=np.random.default_rng(9).integers(0, 255, (n, 96, 128, 3)).astype(np.uint8),
                                cor=np.full((n, 3), 0.5, np.float32))
            torch.manual_seed(0)
            ema, preve, hist, _ = treina(cfg, D, C, tr, va, torch.device("cpu"), float("inf"), semente=11, rotulo="[teste]")
            P, F = preve(D.ids, variantes=True)
            return {k: v.clone() for k, v in ema.state_dict().items()}, P, F, hist, preve
    globals()["cria_rede"] = cria_sem_pesos
    try:
        e0, P0, F0, h0, _ = roda(u0)
        e1, P1, F1, h1, pv = roda(base)
    finally:
        globals()["cria_rede"] = orig
    assert set(e0) == set(e1) and all(torch.equal(e0[k], e1[k]) for k in e0), "a rede do F4 difere da do U0"
    assert all(np.array_equal(P0[s], P1[s]) for s in ORIENT), "as previsões do F4 diferem das do U0"
    assert [x.get("perda") for x in h0] == [x.get("perda") for x in h1] and len(h0) == len(h1) > 2 and [x.get("val_rmse_logm") for x in h0] == [x.get("val_rmse_logm") for x in h1]
    assert F0 is None and F1 is not None and np.allclose(F1[""].sum(1), 1, atol=1e-5) and pv.sg.passos > 0 and h1[-1]["perda_cabeca_sg"] is not None
    w_ini = CabecaSG(np.zeros((len(tr), 8, e1["head.weight"].shape[1]), np.float32) + 1, D, tr, torch.device("cpu"), 11, 10).p[0].detach()
    assert not torch.equal(w_ini, pv.sg.p[0].detach())            # a cabeça sem gradiente treinou
    # o recorte no dispositivo repete o da etapa 0 (deteccao_r5.recorta) a menos da interpolação
    import deteccao_r5 as E5
    a_ = np.random.default_rng(1).integers(0, 255, (480, 640, 3)).astype(np.uint8)
    a_[100:380, 160:480] = 200
    jan = E5.janela((160, 100, 480, 380), 640, 480)
    g = recorte_gpu(torch.from_numpy(a_).permute(2, 0, 1).float() / 255, jan, torch.tensor(E5.cor_borda(a_) / 255, dtype=torch.float32), (480, 640))
    ref = E5.recorta(a_, jan) / 255
    assert np.abs(g.permute(1, 2, 0).numpy()[60:420, 80:560] - ref[60:420, 80:560]).mean() < 0.02          # o miolo (o prato) igual

# >>>>> RODADA 5: fim

# <<<<< RODADA 6: início
# Seção 15 de docs/dados/precisao/resultado_modelo_treinado.md: o protocolo no recorte. P3 (ajuste fino parcial do
# So400m no recorte, só a perda de rótulos, massa das seis bases, partindo da ridge R4c, parada pela validação interna)
# com a cabeça de densidade HP3; F5c (aparelho no recorte com as massas das cinco bases); H0c-m (H0c com a log-massa
# aninhada da ridge R4c como entrada). R0c, R4c e H0c são refeitas (paridade com a quinta execução; a R4c é o ponto de
# partida do P3 e a entrada da H0c-m). Nenhuma previsão do teste2; nada decidido no kernel.
def configs6(smoke=False):
    """As configurações da sexta execução (15.6), em ordem de prioridade; a ordem de corte é H0c-m, depois F5c (o P3
    não é cortado antes delas). Nenhuma tem GPU preferida: cada processo pega a primeira parte pronta na ordem de
    prioridade, e as partes do P3 (5 folds e o final) se repartem entre as duas GPUs."""
    pre, so, ftb = ("smoke6", "vit_tiny_96x128", "vit_tiny_224") if smoke else ("gpu6", _SO4, _FT5B)
    ap = "tiny96" if smoke else "so400m"
    mlp = dict(MLP3, epocas=3) if smoke else MLP3
    ext = lambda d: f"externo {d}"
    feat = lambda d: f"{pre} feat {ap} prato {d}"
    todos_feat = [feat(d) for d in DOMINIOS4]
    todos_ext = [ext(d) for d in DOMINIOS4]
    base = dict(v4=True, v5=True, v6=True)
    cs = [dict(nome=ext(d), tipo="externo", chave=d, bb=None, est_min=0, livre=True, servico=False, partes=["tudo"], **base) for d in DOMINIOS4]
    cs.append(dict(nome=feat("n5k"), tipo="feat", chave="n5k", bb=so, bs=32, est_min=1 if smoke else 22, prato=True, livre=True, servico=False, partes=["tudo"], **base))

    def cabeca(nome, tipo, apelido, est, doms=(), **kw):
        return dict(dict(nome=f"{pre} {tipo} {ap} prato{nome}", tipo=tipo + "4", apelido=apelido, bb=so, bs=32, est_min=est, prato=True, sem_vies=True, livre=True,
                         servico=False, treina_com=list(doms), ff_de=[], partes=["tudo", "ext"], depende=[feat("n5k")] + [feat(d) for d in doms],
                         tardia=dict(parte="ext", terminal=todos_feat), **base), **kw)
    r0 = cabeca("", "ridge", "R0c6", 1)
    h0 = cabeca("", "fracoes", "H0c6", 3, **mlp)
    cs += [r0, h0]
    est_feat = dict(flavoria=5, lefood=3, mensa=2, fpb=6, omnifood=18, snapme=3)
    cs += [dict(nome=feat(d), tipo="feat", chave=d, bb=so, bs=32, est_min=1 if smoke else est_feat[d], prato=True, livre=True, servico=False,
                partes=["tudo"], depende=[ext(d)], **base) for d in DOMINIOS4]
    r4 = cabeca(" +tudo", "ridge", "R4c6", 2, DOMINIOS_TREINO4, servico=None)      # a REF (com a H0c refeita): medida no serviço
    r4["depende"].append(r0["nome"]); r4["plano"] = r0["nome"]
    cs.append(r4)
    ft = lambda **kw: dict(livre=True, val_cada=5, partes=PARTES_FT + ["ext"], tardia=dict(parte="ext", terminal=todos_ext), **base, **kw)
    rec3 = dict(P1_RECEITA, lr=1e-5, ema=0.999, epocas=5, escala=None)
    if smoke:
        rec3.update(epocas=1, bs=4, blocos_treinaveis=2)
    p3 = dict(nome=f"{pre} ftparcial {ap} prato rotulos +tudo", tipo="ftparcial", apelido="P3", bb=so, receita="p3", est_min=1 if smoke else 245, **rec3,
              aquec_cabeca=1 if smoke else 2, lr_cabeca=1e-3, parada_interna=True, pesos_alvo=dict(PESOS_P2), prato=True, enquadramento=PRATO5["enquadramento"],
              externo4=list(DOMINIOS_TREINO4), so_massa_ext=True, sem_vies=True, ini_r4c6=dict(r0=r0["nome"], r4=r4["nome"]),
              **ft(depende=[feat("n5k")] + [feat(d) for d in DOMINIOS_TREINO4] + [r0["nome"], r4["nome"]] + [ext(d) for d in DOMINIOS_TREINO4]))
    rs = dict(epocas=2, bs=8) if smoke else {}
    f5c = _ft(ftb, "r2", 1 if smoke else 85, prefixo=pre, sufixo=" fracoes-sg prato +tudo", apelido="F5c", prato=True, escala=None, enquadramento=PRATO5["enquadramento"],
              externo4=list(DOMINIOS_TREINO4), so_massa_ext=True, sem_vies=True, fracoes_sg=True, **rs, **ft(depende=[ext(d) for d in DOMINIOS_TREINO4]))
    hm = cabeca(" +massa", "fracoes", "H0c-m", 5, massa_aninhada=r4["nome"], herda_massa_de=list(DOMINIOS_TREINO4), **mlp)
    hm["depende"] += [feat(d) for d in DOMINIOS_TREINO4] + [r4["nome"]]
    return cs + [p3, f5c, hm]


CONFIGS6, CONFIGS_SMOKE6 = configs6(), configs6(True)
RODADAS["6"] = [c["nome"] for c in CONFIGS6]


def pipelines6(cfgs):
    """Totais de serviço da sexta execução (15.6): a REF (R4c × H0c, com 1 e 4 orientações), o P3 com a HP3, o F5c e os
    pares com o aparelho (a soma dos dois modelos, um depois do outro)."""
    ap = {c.get("apelido"): c["nome"] for c in cfgs if c.get("apelido")}
    p = {}
    for chave, nomes in (("REF_R4c_H0c_recorte", ["R4c6"]), ("P3_HP3", ["P3"]), ("F5c_aparelho", ["F5c"]), ("par_REF_F5c", ["R4c6", "F5c"]), ("par_P3_F5c", ["P3", "F5c"])):
        if all(x in ap for x in nomes):
            p[chave] = [ap[x] for x in nomes]
    return p


def teste_rodada6():
    """Seção 15, sem torch: a fila, as receitas, a partição interna, a coluna de massa aninhada (sem vazamento), os
    alvos só de massa e a procedência."""
    ap = {c["apelido"]: c for c in CONFIGS6 if c.get("apelido")}
    nomes = [c["nome"] for c in CONFIGS6]
    pos = lambda x: nomes.index(ap[x]["nome"])
    assert sorted(ap, key=pos) == ["R0c6", "H0c6", "R4c6", "P3", "F5c", "H0c-m"]
    for lista in (CONFIGS6, CONFIGS_SMOKE6):
        ns = [c["nome"] for c in lista]
        for i, c in enumerate(lista):
            assert all(d in ns[:i] for d in c.get("depende", [])), c["nome"]
            assert c.get("gpu") is None and c.get("v6") and c.get("v5"), c["nome"]
        assert len({c["nome"] for c in lista}) == len(lista)
    assert all(c.get("prato") for c in CONFIGS6 if c["tipo"] != "externo")
    p3, f5c, hm, f4c = ap["P3"], ap["F5c"], ap["H0c-m"], next(c for c in CONFIGS5 if c.get("apelido") == "F4c")
    assert (p3["blocos_treinaveis"], p3["epocas"], p3["aquec_cabeca"], p3["lr"], p3["lr_cabeca"], p3["lld"], p3["ema"], p3["escala"], p3["enquadramento"]) == \
        (4, 5, 2, 1e-5, 1e-3, 0.75, 0.999, None, 0.10)
    assert p3["parada_interna"] and p3["pesos_alvo"] == PESOS_P2 and p3["externo4"] == DOMINIOS_TREINO4 and p3["so_massa_ext"] and p3["sem_vies"] and not p3.get("fracoes_sg")
    assert all(f5c[k] == f4c[k] for k in ("bb", "lr", "lld", "epocas", "bs", "acum", "wd", "drop_path", "ema", "aquec", "ckpt_grad", "escala", "enquadramento", "fracoes_sg"))
    assert f5c["externo4"] == DOMINIOS_TREINO4 and f5c["so_massa_ext"] and f5c["sem_vies"] and not f5c.get("receita_u0")
    assert hm["massa_aninhada"] == ap["R4c6"]["nome"] and hm["tipo"] == "fracoes4" and not hm["treina_com"]
    assert ap["R4c6"]["treina_com"] == DOMINIOS_TREINO4 and ap["R4c6"]["plano"] == ap["R0c6"]["nome"]
    assert [c["nome"] for c in CONFIGS6 if c.get("servico") is not False] == [ap["R4c6"]["nome"], p3["nome"], f5c["nome"]]
    assert set(pipelines6(CONFIGS6)) == {"REF_R4c_H0c_recorte", "P3_HP3", "F5c_aparelho", "par_REF_F5c", "par_P3_F5c"}
    D = _dados_falsos_dom()
    assert sorted(x for x in ap if procedencia4(ap[x], D)["so_comercial_ok"]) == ["H0c6", "R0c6"]     # a H0c-m herda as bases da R4c pela entrada de massa
    # validação interna: ~10% das cadeias, estável
    cad = [f"cadeia_{i}" for i in range(20000)]
    fr = np.mean([particao_interna6(c) for c in cad])
    assert 0.09 < fr < 0.11 and particao_interna6("x") == particao_interna6("x")
    # bases externas só com a massa (P3, F5c)
    d = dict(chave="omnifood", ids=["a", "b"], peso_nut=np.ones(2))
    Dn = SimpleNamespace(N=22, I_MG=slice(6, 14), I_PG=slice(14, 22))
    P, pf = alvos_dominio4(d, Dn, dict(tipo="ftparcial", externo4=["omnifood"], so_massa_ext=True))
    assert P[:, I_M].all() and not P[:, 1:].any() and not pf.any()
    P, pf = alvos_dominio4(d, Dn, dict(tipo="ft", externo4=["omnifood"]))
    assert P[:, I_P].all() and pf.all()                                          # sem a marca, o F3/F5 da quarta/quinta execução não muda
    # pesos de domínio com dois folds de fora
    rng = np.random.default_rng(0)
    dm = dict(treina=np.ones(60, bool), proib=np.zeros((60, 5), bool), grupo=np.repeat(np.arange(20), 3))
    dm["proib"][np.arange(60), np.arange(60) % 5] = True
    assert np.allclose(pesos_dominio6(dm, [2], 10.0), pesos_dominio4(dm, 2, 10.0)) and np.allclose(pesos_dominio6(dm, [], 10.0), pesos_dominio4(dm, None, 10.0))
    w = pesos_dominio6(dm, [1, 3], 10.0)
    assert not w[np.arange(60) % 5 == 1].any() and not w[np.arange(60) % 5 == 3].any() and abs(w.sum() - 10.0) < 1e-9
    # coluna aninhada: o rótulo de um prato nunca entra na coluna dele, em nenhum modelo
    n, Dm = 50, 6
    X, fold = rng.normal(size=(n, 2, Dm)), np.arange(n) % 5
    todos = frozenset(range(5))

    def faz_prev(y):
        def prev(fs, Xv):
            m = np.isin(fold, list(fs))
            A = X[m].reshape(-1, Dm); b = np.repeat(y[m], 2)
            W = np.linalg.solve(A.T @ A + np.eye(Dm), A.T @ b)
            return (Xv.reshape(-1, Dm) @ W).reshape(len(Xv), -1)
        return prev
    y = rng.normal(size=n)
    M1 = colunas_aninhadas6(faz_prev(y), X, fold, todos)
    y2 = y.copy(); y2[7] += 100.0
    M2 = colunas_aninhadas6(faz_prev(y2), X, fold, todos)
    assert all(np.allclose(M1[q][7], M2[q][7]) for q in M1)                       # a coluna do prato 7 não vê o rótulo dele
    assert all(np.allclose(M1[q][fold == fold[7]], M2[q][fold == fold[7]]) for q in M1)
    assert not np.allclose(M1[0][fold == 1], M2[0][fold == 1])                   # mas o rótulo dele entra na coluna dos outros folds
    assert np.allclose(M1[5][fold == 2], M1[2][fold == 2])                        # no final, cada prato recebe a R4c fora-do-fold
    # fila: tudo cabe com folga em 4,5 h; apertando, sai primeiro quem tem menor prioridade e não cabe (o P3 é o último)
    novo = lambda: {slug(c["nome"]): {q: parte_inicial() for q in partes_de(c)} for c in CONFIGS6}
    seg = lambda c: 60.0 * c["est_min"] / len(partes_de(c))
    cabe = simula_orcamento(CONFIGS6, novo(), [0.0, 0.0], seg, (4.5 * 60 - 20) * 60)
    assert all(cabe[ap[x]["nome"]] for x in ap), cabe
    cabe = simula_orcamento(CONFIGS6, novo(), [0.0, 0.0], seg, 175 * 60)
    assert cabe[p3["nome"]] and not cabe[f5c["nome"]], cabe
    E_ = novo()
    for c in CONFIGS6:
        if c["tipo"] == "externo":
            E_[slug(c["nome"])]["tudo"]["estado"] = "rodando"
    assert proxima_acao(CONFIGS6, E_, {}, gpu=0)[1]["nome"] == proxima_acao(CONFIGS6, E_, {}, gpu=1)[1]["nome"] == CONFIGS6[6]["nome"]   # as features do recorte
    E_ = novo()
    for c in CONFIGS6:
        if c.get("apelido") not in ("P3", "F5c", "H0c-m"):
            for q in E_[slug(c["nome"])].values():
                q["estado"] = "feita"
    E_[slug(p3["nome"])]["fold0"]["estado"] = "rodando"
    assert proxima_acao(CONFIGS6, E_, {}, gpu=1)[1:] == (p3, "fold1")          # a segunda GPU ajuda no P3 antes do F5c


def teste_rodada6_torch():
    """Seção 15, com torch: a ridge de massa com linhas de menos (sem a validação interna) e `todos` explícito repete a
    conta da R4c quando as linhas são as mesmas; com dois folds de fora, as fotos proibidas em qualquer um saem."""
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(6)
    n, o, Dm = 60, 2, 5
    X, T, fold = rng.normal(size=(n, o, Dm)).astype(np.float32), rng.normal(size=(n, 1)), np.arange(n) % 5
    de = dict(treina=np.ones(30, bool), proib=np.zeros((30, 5), bool), grupo=np.arange(30))
    de["proib"][np.arange(30), np.arange(30) % 5] = True
    dom = lambda: [dict(chave="x", X=rng_x, t=t_x, d=de, frac=0.5)]
    rng_x, t_x = rng.normal(size=(30, 1, Dm)).astype(np.float32), rng.normal(size=(30, 1))
    todos = frozenset(range(5))
    R = RidgeMulti(X, T, fold, dom(), sem_vies=True)
    alfa = torch.tensor([3.0], dtype=torch.float64)
    m = fold != 2
    A1, b1 = R.afim(todos - {2}, alfa)
    A2, b2 = RidgeMulti(X[m], T[m], fold[m], dom(), sem_vies=True, todos=todos).afim(todos - {2}, alfa)
    assert np.abs(A1 - A2).max() < 1e-6 and np.abs(b1 - b2).max() < 1e-6        # a conta de ini_r4c6 com as mesmas linhas = a da R4c
    A3, _ = RidgeMulti(X[m], T[m], fold[m], dom(), sem_vies=True).afim(todos - {2}, alfa)     # sem `todos`, o fold 2 nem existiria: as fotos proibidas nele entrariam
    assert np.abs(A1 - A3).max() > 1e-4
    R.afim(todos - {1, 3}, alfa)                                                  # dois folds de fora (H0c-m): não levanta
    assert R.dec(todos - {1, 3})[0] is not None

# >>>>> RODADA 6: fim



# ------------------------------------------------------------------ fase de serviço (processo filho, um modelo por vez)
def mede_servico(a, cfg, cfgs=()):
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
    D = dados_de(a)
    C = Cache(a.cache, D).abre()
    prazo = a.prazo or float("inf")
    hw, N = BACKBONES[cfg["bb"]]["hw"], D.N
    quad = hw[0] == hw[1]
    congelado = cfg["tipo"] in ("ridge", "fracoes", "unico", "ridge4", "fracoes4")       # backbone público + cabeça gravada, em float32 fora do autocast
    v4 = bool(cfg.get("v4"))          # rodada 4: cabeças sempre em float32; medidas só com 1 orientação (13.8)
    v5, prato = bool(cfg.get("v5")), bool(cfg.get("prato"))     # rodada 5: remontagem com 1 orientação; sobre o recorte, a entrada é a janela do guia
    P5 = carrega_caixas5(a.listas, [a.caixas], D, a.smoke) if v5 else None
    zsg = None
    spec = json.load(open(arq_de(a.saida, cfg, "servir")))
    K = 0
    extra = None                      # R0 da rodada 4: a cabeça de frações H0 roda junto, sobre as mesmas features (o D1)

    class MLP(torch.nn.Module):
        def __init__(self, z):
            super().__init__()
            self.register_buffer("mu", torch.from_numpy(z["mu"])); self.register_buffer("sd", torch.from_numpy(z["sd"]))
            self.l1 = torch.nn.Linear(z["W1"].shape[1], z["W1"].shape[0]); self.l2 = torch.nn.Linear(z["W2"].shape[1], z["W2"].shape[0])
            with torch.no_grad():
                self.l1.weight.copy_(torch.from_numpy(z["W1"])); self.l1.bias.copy_(torch.from_numpy(z["b1"]))
                self.l2.weight.copy_(torch.from_numpy(z["W2"])); self.l2.bias.copy_(torch.from_numpy(z["b2"]))

        def forward(self, f):
            return self.l2(F.gelu(self.l1((f - self.mu) / self.sd)))
    if congelado:
        rede = cria_rede(cfg["bb"], N)
        z = np.load(arq_de(a.saida, cfg, "cabeca"), allow_pickle=False)
        if cfg["tipo"] in ("ridge", "ridge4"):
            with torch.no_grad():
                rede.head.weight.copy_(torch.from_numpy((z["W"] / z["sd"][:, None]).T.astype(np.float32)))
                rede.head.bias.copy_(torch.from_numpy((z["mt"] - (z["mu"] / z["sd"]) @ z["W"]).astype(np.float32)))
            tm, ts = np.zeros(N, np.float32), np.ones(N, np.float32)
            par_h = {"R0": "H0", "R0c": "H0c", "R4c6": "H0c6"}.get(cfg.get("apelido"))
            h0 = next((c for c in cfgs if c.get("apelido") == par_h and os.path.exists(arq_de(a.saida, c, "cabeca"))), None) if par_h else None
            if h0 is not None:
                extra = MLP(np.load(arq_de(a.saida, h0, "cabeca"), allow_pickle=False))
        else:
            rede.head = MLP(z)
            tm, ts, K = z["tm"], z["ts"], z["W2"].shape[0] - N
        arquivos = [dict(arquivo=spec["backbone"]["arquivo"], bytes=spec["backbone"]["bytes"], publico=True, url=spec["backbone"]["url"],
                         hf_id=spec["backbone"]["hf_id"], fonte=spec["backbone"]["fonte"], sha256=spec["backbone"]["sha256"]),
                    dict(arquivo=os.path.basename(arq_de(a.saida, cfg, "cabeca")), bytes=os.path.getsize(arq_de(a.saida, cfg, "cabeca")), publico=False)]
    elif v4:       # ajuste fino da rodada 4: 22 saídas + head_ff; no parcial, o backbone público mais os tensores treinados
        ck = torch.load(arq_de(a.saida, cfg, "pesos"), map_location="cpu", weights_only=True)
        K = ck["n_ff"]
        rede = cria_rede4(cfg, D, pretrained=bool(ck.get("parcial")))
        tm, ts = carrega_modelo4(rede, ck)
        zsg = ck.get("cabeca_sg") if ck.get("cabeca_sg") is not None else ck.get("cabeca_hp3")     # rodada 5: cabeça sem gradiente (14.4); 6: a HP3 do P3
        if zsg is not None:
            K = len(zsg["dens_fonte"])
        arquivos = ([dict(arquivo=spec["backbone"]["arquivo"], bytes=spec["backbone"]["bytes"], publico=True, url=spec["backbone"]["url"], hf_id=spec["backbone"]["hf_id"],
                          fonte=spec["backbone"]["fonte"], sha256=spec["backbone"]["sha256"])] if ck.get("parcial") else []) \
            + [dict(arquivo=os.path.basename(arq_de(a.saida, cfg, "pesos")), bytes=os.path.getsize(arq_de(a.saida, cfg, "pesos")), publico=False)]
    else:          # ajuste fino: só o arquivo de pesos gravado (sem baixar nada)
        ck = torch.load(arq_de(a.saida, cfg, "pesos"), map_location="cpu", weights_only=True)
        K = ck.get("n_saidas", N) - N
        rede = cria_rede(cfg["bb"], N + K, pretrained=False)
        rede.load_state_dict({n: (v.float() if v.is_floating_point() else v) for n, v in ck["estado"].items()})
        tm, ts = np.array(ck["tm"], np.float32), np.array(ck["ts"], np.float32)
        arquivos = [dict(arquivo=os.path.basename(arq_de(a.saida, cfg, "pesos")), bytes=os.path.getsize(arq_de(a.saida, cfg, "pesos")), publico=False)]
    rede.eval()
    if extra is not None:
        rede.extra = extra.eval()
    n_par = sum(p.numel() for p in rede.parameters())
    n_cab = sum(p.numel() for nome_, p in rede.named_parameters() if nome_.startswith(("head", "extra")))
    f32 = congelado or v4

    def infere(d, xu8, n_or, agrupado=False):
        """uint8 [B, 480, 640, 3] no dispositivo -> (22 saídas em log por imagem, frações por fonte ou None), média das
        n_or primeiras orientações. agrupado: as orientações num lote só (com entrada não quadrada, as 8 em dois de 4)."""
        x = entrada(xu8)
        grupos = [[k] for k in range(n_or)] if not agrupado else ([list(range(n_or))] if quad or n_or <= 4 else [[0, 1, 2, 3], [4, 5, 6, 7]])
        st, sf = 0, 0
        for g in grupos:
            xb = torch.cat([orienta(x, k) for k in g])
            with autocast(d):
                f = rede.feats(xb)
                o = None if f32 else rede.head(f)
            if f32:            # cabeça em float32, fora do autocast (como as previsões gravadas)
                f = f.float()
                o = rede.head(f)
                if hasattr(rede, "head_ff"):
                    o = torch.cat([o, rede.head_ff(f)], 1)
                if extra is not None:      # a segunda cabeça do D1 entra no tempo medido
                    rede.extra(f)
            if zsg is not None:
                fr_g = aplica_sg_torch(zsg, f.float())[1].reshape(len(g), len(x), -1)
            o = o.float().reshape(len(g), len(x), -1)
            st = st + o[..., :N].sum(0)
            if K:
                sf = sf + (fr_g if zsg is not None else torch.softmax(o[..., N:], -1)).sum(0)
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
        B = len(xu8[1]) if isinstance(xu8, tuple) else len(xu8)
        r = dict(lote=B, orientacoes=n_or, orientacoes_em_lote=bool(agrupado), repeticoes=reps,
                 ms_por_imagem=round(1000 * float(np.median(tempos)) / B, 2), ms_por_imagem_min=round(1000 * min(tempos) / B, 2),
                 ms_por_imagem_max=round(1000 * max(tempos) / B, 2))
        if d.type == "cuda":
            r["mem_pico_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        return r

    def u8(ids, d):
        """Fotos uint8 480×640 no dispositivo; sobre o recorte, junto com a janela do guia e a cor da borda de cada uma."""
        t_ = torch.from_numpy(np.ascontiguousarray(C.rgb[[C.pos[i] for i in ids]])).to(d)
        if not prato:
            return t_
        return t_, [P5["por_base"]["n5k"]["jan"][i] for i in ids], torch.from_numpy(C.cor[[C.pos[i] for i in ids]]).to(d)

    def entrada(xu8):
        """Redimensionar a foto (ou recortar a janela do guia e redimensionar) e limitar a [0, 1]: o que o app faz."""
        if isinstance(xu8, tuple):
            t_, js, cs = xu8
            return torch.stack([recorte_gpu(t_[b].permute(2, 0, 1).float().div(255), js[b], cs[b], hw) for b in range(len(js))]).clamp_(0, 1)
        return F.interpolate(xu8.permute(0, 3, 1, 2).float().div(255), size=hw, mode="bicubic", antialias=True, align_corners=False).clamp_(0, 1)

    fatia = lambda x, i, j: (x[0][i:j], x[1][i:j], x[2][i:j]) if isinstance(x, tuple) else x[i:j]
    rede.to(dev)
    # 1. o modelo remontado reproduz as previsões gravadas? (até 8 pratos de um conjunto de previsão, 8 orientações; sem gabarito)
    # rodada 5: 1 orientação (sobre o recorte, as vistas 1–7 das features congeladas levam erro de enquadramento sorteado)
    parte = np.load(arq_parte(a.saida, cfg, "tudo" if congelado else "ext" if cfg.get("parada_interna") else "final"), allow_pickle=False)
    conj = "calib" if "calib" in D.conj else next(iter(D.conj))
    u = np.where(D.usa_av[conj])[0][:8]
    n_or_conf, suf_conf = (1, "_o1") if v5 else (8, "")
    with torch.inference_mode():
        p, fr = infere(dev, u8([D.conj[conj][i] for i in u], dev), n_or_conf)
    dif = np.abs(p.cpu().numpy() - parte[conj + suf_conf][u])
    conf = dict(conjunto=conj, pratos=int(len(u)), max_dif_log=float(dif.max()), max_dif_log_massa=float(dif[:, I_M].max()), max_dif_log1p_proteina=float(dif[:, I_P].max()),
                dispositivo=dev.type, nota="diferença máxima, em log, entre o modelo remontado com os arquivos de servir e as previsões gravadas")
    if K:
        conf["max_dif_fracao"] = float(np.abs(fr.cpu().numpy() - parte[conj + "_ff" + suf_conf][u]).max())
    conf["orientacoes"] = n_or_conf
    if cfg.get("parada_interna") and zsg is not None:       # P3: a HP3 também é remontada a partir da foto
        with torch.inference_mode():
            x = entrada(u8([D.conj[conj][i] for i in u], dev))
            with autocast(dev):
                f_ = rede.feats(x)
            s_h = aplica_sg_torch(zsg, f_.float())[0].cpu().numpy()
        conf["max_dif_log_hp3"] = float(np.abs(s_h - parte[conj + "_hp3_o1"][u]).max())
    res = dict(formato=FORMATO, nome=cfg["nome"], tipo=cfg["tipo"], backbone=cfg["bb"], timm=BACKBONES[cfg["bb"]]["timm"], hw=list(hw),
               parametros=dict(total=int(n_par), backbone=int(n_par - n_cab), cabeca=int(n_cab)), mb_fp16_em_memoria=round(n_par * 2 / 1e6, 1),
               arquivos=arquivos, bytes_pesos=int(sum(x["bytes"] or 0 for x in arquivos)), bytes_gravados_na_saida=int(sum(x["bytes"] or 0 for x in arquivos if not x["publico"])),
               conferencia=conf,
               inclui=("recortar a janela do guia + " if prato else "") + "redimensionar (bicúbico) + normalizar + rede + cabeça"
                      + ((" + cabeça HP3" if cfg.get("parada_interna") else " + cabeça sem gradiente") if zsg is not None else "")
                      + ", a partir de uint8 480×640 já no dispositivo; não inclui decodificar o PNG nem a cópia para a GPU",
               gpu=None, cpu=None)
    print(f"[{cfg['nome']} | servico] {n_par / 1e6:.1f} M parâmetros; remontado × gravado ({conj}): diferença máxima {dif.max():.4f} em log"
          + (f", {conf['max_dif_fracao']:.4f} nas frações" if K else ""), flush=True)
    if v4 and cfg.get("apelido") in ("R0", "R0c", "R4c6"):      # 13.8: remontagem, a partir da FOTO, de todas as cabeças sobre este backbone, inclusive as de frações
        with torch.inference_mode():
            x = entrada(u8([D.conj[conj][i] for i in u], dev))
            fts = []
            for q in range(n_or_conf):
                with autocast(dev):
                    fq = rede.feats(orienta(x, q))
                fts.append(fq.float().cpu().numpy())
        Fc = np.stack(fts, 1)
        res["conferencias_cabecas"] = {}
        for c in cfgs:
            if c.get("tipo") in ("ridge4", "fracoes4") and c["bb"] == cfg["bb"] and bool(c.get("prato")) == prato and os.path.exists(arq_de(a.saida, c, "cabeca")) \
                    and os.path.exists(arq_parte(a.saida, c, "tudo")):
                zc, pc = np.load(arq_de(a.saida, c, "cabeca"), allow_pickle=False), np.load(arq_parte(a.saida, c, "tudo"), allow_pickle=False)
                if c["tipo"] == "ridge4":
                    Pc = (((Fc.reshape(-1, Fc.shape[-1]) - zc["mu"]) / zc["sd"]).astype(np.float64) @ zc["W"] + zc["mt"]).reshape(len(u), n_or_conf, -1).mean(1)
                    r_ = dict(max_dif_log=float(np.abs(Pc - pc[conj + suf_conf][u]).max()))
                else:
                    Fe = Fc
                    if c.get("massa_aninhada"):     # H0c-m: a coluna de massa vem da cabeça ridge da R4c, sobre as mesmas features
                        z4 = np.load(arq_de(a.saida, next(x for x in cfgs if x["nome"] == c["massa_aninhada"]), "cabeca"), allow_pickle=False)
                        m4 = (((Fc.reshape(-1, Fc.shape[-1]) - z4["mu"]) / z4["sd"]).astype(np.float64) @ z4["W"][:, I_M] + z4["mt"][I_M]).reshape(Fc.shape[0], Fc.shape[1], 1)
                        Fe = np.concatenate([Fc, m4.astype(np.float32)], -1)
                    Pc, Fr = aplica_cabeca4(zc, Fe)
                    r_ = dict(max_dif_log=float(np.abs(Pc.mean(1) - pc[conj + suf_conf][u]).max()), max_dif_fracao=float(np.abs(Fr.mean(1) - pc[conj + "_ff" + suf_conf][u]).max()))
                res["conferencias_cabecas"][c["nome"]] = r_
        res["pipeline"] = (("backbone + cabeça ridge (R4c) + cabeça de frações (H0c) sobre as mesmas features, com a janela do guia: a REF da sexta execução (15.1)"
                            if cfg.get("apelido") == "R4c6" else
                            "backbone + cabeça ridge (R0c) + cabeça de frações (H0c) sobre as mesmas features, com a janela do guia: o A0c" if prato else
                            "backbone + cabeça ridge (R0) + cabeça de frações (H0) sobre as mesmas features: o D1/A0") if extra is not None
                           else "backbone + cabeça ridge (a cabeça de frações H0 não estava pronta)")
        print(f"[{cfg['nome']} | servico] remontagem a partir da foto, por cabeça: " + "; ".join(
            f"{k_.split(' ', 1)[1]} {v_['max_dif_log']:.4f}" + (f" / {v_['max_dif_fracao']:.4f}" if "max_dif_fracao" in v_ else "") for k_, v_ in res["conferencias_cabecas"].items()), flush=True)
    ids8 = D.todos[:8]
    # 2. GPU (uma só, fp16)
    if dev.type == "cuda":
        x8 = u8(ids8, dev)
        med = {}
        medidas = dict(lote1_o1=(1, 1, False, 30, 5), lote1_o4=(1, 4, False, 10, 2), lote1_o8=(1, 8, False, 10, 2),
                       lote1_o4_em_lote=(1, 4, True, 10, 2), lote1_o8_em_lote=(1, 8, True, 10, 2), lote8_o1=(8, 1, False, 10, 2), lote8_o8=(8, 8, False, 5, 1))
        v6 = bool(cfg.get("v6"))      # rodada 6 (15.6): também 4 orientações
        for nome, (lote, n_or, agr, reps, aq) in ({k_: v_ for k_, v_ in medidas.items() if k_.endswith("_o1") or (v6 and k_.startswith("lote1_o4"))} if v4 else medidas).items():
            if time.time() + 60 > prazo:
                raise SemOrcamento()
            med[nome] = cronometra(dev, fatia(x8, 0, lote), n_or, agr, reps, aq)
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
    med = dict(lote1_o1=cronometra(cpu, fatia(x8, 0, 1), 1, False, 2 if a.smoke else 3, 1))
    rapido = med["lote1_o1"]["ms_por_imagem"] < 500
    for nome, n_or in ((("lote1_o4", 4),) if cfg.get("v6") else () if v4 else (("lote1_o4", 4), ("lote1_o8", 8))):
        if time.time() + 1.3 * n_or * med["lote1_o1"]["ms_por_imagem"] / 1000 + 30 < prazo:
            med[nome] = cronometra(cpu, fatia(x8, 1, 2), n_or, False, 3 if rapido and not a.smoke else 1, 0)
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
            mede_servico(a, cfg, cfgs)
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
        for padrao in ("parte__*.npz", "pesos__*.pt", "modelo__*.pt", "cabeca__*.npz", "servir__*.json", "servico__*.json", "decisao_*.json"):
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
    D = dados_de(a)
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
    for ch, d in D.dom.items():
        diz(f"base externa {ch} (13.3): {d['resumo']} | {d['licenca']}, {d['uso_declarado']} | lista {d['sha256_lista'][:16]}"
            f"{' (versão enxuta no pacote)' if d['lista_enxuta'] else ''}, divisão {d['sha256_divisao'][:16]} | fotos: {DOMINIOS4[ch]['de']}")
    for ch, motivo in D.dom_fora.items():
        diz(f"ATENÇÃO: a base externa {ch} fica de FORA desta execução: {motivo}")
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
    if a.rodada in ("5", "6"):     # 14.3: as janelas do guia, refeitas das caixas brutas da etapa 0 com a regra da conferência
        D.P5 = carrega_caixas5(a.listas, [a.caixas], D, a.smoke)
        P = D.P5
        info["prato"] = dict({k: v for k, v in P.items() if k != "por_base"}, por_base={ch: {k: v for k, v in b.items() if k not in ("jan", "sha")} for ch, b in P["por_base"].items()})
        info["detector"] = dict(licenca=P["detector_licenca"], nota="o detector só preparou os dados de treino (etapa 0); não faz parte de nenhum modelo servido")
        diz(f"recorte pelo prato (14.3): tabelas da etapa 0 em {P['pasta']} | regra {P['regra']}, consultas {P['consultas']} | reprovadas na conferência a olho: "
            f"{P['reprovados'] or 'nenhuma'} | conferência feita: {P['conferida']}")
        for ch, b in P["por_base"].items():
            diz(f"   {ch:9s} {b['situacao']} | {len(b['jan'])} janelas | métodos {b.get('metodos')} | tabela {str(b.get('sha256_tabela'))[:16]}")
        if not P["conferida"] and not a.smoke and not a.sem_conferencia:
            raise SystemExit("A conferência a olho da etapa 0 (14.3) ainda não foi feita: conferencia_r5.json tem 'conferida': false. A execução principal só começa depois dela.")
    if a.rodada in ("4", "5", "6"):     # 13.8: procedência por base; conferência de paridade com o D1 (13.7, só na rodada 4); nada é decidido no kernel
        info["dominios"] = {ch: dict(situacao="aguardando as fotos", dominio=d["dominio"], licenca=d["licenca"], uso_declarado=d["uso_declarado"], sha256_lista=d["sha256_lista"],
                                     sha256_divisao=d["sha256_divisao"], lista_enxuta=d["lista_enxuta"], resumo=d["resumo"], fotos_de=DOMINIOS4[ch]["de"]) for ch, d in D.dom.items()}
        info["dominios"].update({ch: dict(situacao=f"FORA desta execução: {m}") for ch, m in D.dom_fora.items()})
        info["paridade_d1"] = dict(situacao="ainda não conferida") if a.rodada == "4" else dict(situacao="não se aplica: a quinta execução não refaz o R0 nem o H0 (14.5)")
        info["confianca"] = "a conferir: a paridade de R0 e H0 com o D1 da execução 3 ainda não saiu" if a.rodada == "4" else "sem conferência de paridade nesta execução (14.5)"
        if a.rodada == "6":     # 15.4: R4c e H0c refeitas × as da quinta execução
            info["paridade_d1"] = dict(situacao="ainda não conferida", contra="R4c e H0c da quinta execução (listas/" + REF_R5 + ")")
            info["confianca"] = "a conferir: a paridade da R4c e da H0c refeitas com as da quinta execução ainda não saiu"
        info["decisoes_no_kernel"] = "nenhuma: toda escolha é feita depois, na CPU (13.9); o teste2 e as partes de registro externas não são pontuados aqui"

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
            if "passou" in info.get("paridade_d1", {}):
                info["configs"][c["nome"]]["confiavel"] = bool(info["paridade_d1"]["passou"])
        info["duracao_min"] = round((time.time() - t_ini) / 60, 1)
        grava_json(os.path.join(a.saida, "manifesto.json"), info)

    def confere_paridade(fim=False):
        """13.7: R0 e H0 repetem as duas cabeças do D1 da execução 3? Sai assim que as duas estão prontas; se falhar (ou
        não puder ser conferida), o manifesto marca todo o resto como não confiável."""
        if a.rodada not in ("4", "6") or "passou" in info["paridade_d1"]:
            return
        r0, h0 = (next((c for c in cfgs if c.get("apelido") == x), None) for x in (("R0", "H0") if a.rodada == "4" else ("R4c6", "H0c6")))
        arqs = [arq_parte(a.saida, c, "tudo") if c else "" for c in (r0, h0)]      # sai assim que as duas cabeças terminam (antes das previsões externas)
        if not all(os.path.exists(x) for x in arqs):
            if fim:
                info["paridade_d1"] = dict(passou=False, motivo="R0 ou H0 não ficaram prontas: a paridade com o D1 não pôde ser conferida")
        else:
            try:
                with open(os.path.join(a.listas, REF_D1 if a.rodada == "4" else REF_R5), newline="") as f:
                    ref = {r["dish_id"]: [float(r[c]) for c in ("log_massa_o1", "log_massa_o8", "log_dens_regressao_o1", "log_dens_fracoes_o1")] for r in csv.DictReader(f)}
                info["paridade_d1"] = paridade_d1(D, np.load(arqs[0], allow_pickle=False), np.load(arqs[1], allow_pickle=False), ref,
                                                  **({} if a.rodada == "4" else dict(esperado=REF_R5_CV, contra="quinta execução: R4c e H0c")))
            except Exception as ex:
                info["paridade_d1"] = dict(passou=False, motivo=f"não foi possível conferir ({type(ex).__name__}: {ex})")
        p = info["paridade_d1"]
        if "passou" not in p:
            return
        ref_txt = "o D1 da execução 3" if a.rodada == "4" else "a R4c e a H0c da quinta execução"
        if p["passou"]:
            info["confianca"] = f"paridade com {ref_txt} conferida"
        else:
            info["confianca"] = f"NÃO CONFIÁVEL: a paridade com {ref_txt} falhou ({p.get('motivo')}); nenhuma configuração desta execução deve ser lida"
        for ic in info["configs"].values():
            ic["confiavel"] = bool(p["passou"])
        diz(f"== PARIDADE com {'o D1 da execução 3 (13.7)' if a.rodada == '4' else 'a R4c e a H0c da quinta execução (15.4)'}: {'OK' if p['passou'] else 'FALHOU: ' + str(p.get('motivo'))}" + (
            f" | correlação fora-do-fold: massa {p['corr']['log_massa_o1']:.5f} (1 orientação) e {p['corr']['log_massa_o8']:.5f} (8); densidade da regressão "
            f"{p['corr']['log_dens_regressao_o1']:.5f}, pelas frações {p['corr']['log_dens_fracoes_o1']:.5f} (mínimo {D1_CORR}) | critério de CV "
            f"{p['criterio_cv_d1']:.2f}% (esperado {p['esperado']} ± {p['tolerancia']}) | {p['n']} pratos" if "corr" in p else ""))
        if not p["passou"]:
            diz("ATENÇÃO: " + info["confianca"])

    def fecha_config(c):
        """Configuração completa: junta as partes, calcula a CV no treino e, com --guarda-pesos melhor, mantém só os pesos do melhor modelo."""
        s = slug(c["nome"])
        if c["tipo"] in ("externo", "decisao", "feat"):
            return
        cv = junta4(a, c, D) if c.get("v4") else junta3(a, c, D) if c.get("v3") else junta(a, c, D)
        ic = info["configs"][c["nome"]]
        ic["cv_treino"] = cv; ic["min"] = round(sum(p["seg"] or 0 for p in estado[s].values()) / 60, 1)
        if c.get("v4"):
            ic.update(apelido=c.get("apelido"), **procedencia4(c, D))
        diz(f"== {c['nome']}: CV no treino (n={cv['n']}) RMSE log-massa {cv['rmse_logm']:.4f} | MAPE massa {cv['mape_massa']:.1f}% | "
            f"critério ponderado direto {cv['crit_direto']:.1f}% m×d {cv['crit_mxd']:.1f}% geo {cv['crit_geo']:.1f}% | {ic['min']} min de GPU")
        if "rotas" in cv:
            diz(f"   rotas (critério de CV ponderado), 8 / 4 / 1 orientações: " + " | ".join(
                f"{r} {cv['rotas']['o8'][r]:.1f} / {cv['rotas']['o4'][r]:.1f} / {cv['rotas']['o1'][r]:.1f}" for r in cv["rotas"]["o8"])
                + (f" | réplica da execução 2: RMSE {cv['replica_execucao2_rmse_logm']:.4f}" if "replica_execucao2_rmse_logm" in cv else ""))
        if c.get("v4"):         # só métricas de CV das partes de treino; nenhum "melhor modelo" é escolhido aqui
            if cv.get("ext_cv_massa"):
                diz("   bases externas, fotos de TREINO na validação cruzada (massa a ±20% com a constante do domínio / viés mediano cru): " + " | ".join(
                    f"{ch} {v['a_20pct_com_constante']:.0f}% / {v['vies_mediano_cru_pct']:+.0f}%" for ch, v in cv["ext_cv_massa"].items()))
            if "aceitacao" in cv:
                ac = cv["aceitacao"]
                diz(f"   aceitação (13.7): RMSE de log-massa, média dos folds, época 0 {ac['rmse_logm_epoca0']:.4f} -> última {ac['rmse_logm_ultima']:.4f}: {'PASSOU' if ac['passou'] else 'NÃO PASSOU'}")
            confere_paridade()
            return
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
        if a.rodada in ("4", "5", "6"):
            cmd += (["--dominios", ",".join(a.dominios) or "nenhum"] if a.dominios is not None else []) + (["--externo-imagens", a.externo_imagens] if a.externo_imagens else []) \
                + (["--espelho", a.espelho] if a.espelho else []) + (["--caixas", a.caixas] if a.caixas else [])
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
                    evento(c, p, tipo="retomou", seg=float(np.load(arq_parte(a.saida, c, p))["seg"]) if os.path.exists(arq_parte(a.saida, c, p)) else None)
                elif os.path.exists(arq_parte(a.saida, c, p)):
                    diz(f"retomada: {c['nome']} | {p} existe, mas sem teste2, sem os arquivos de servir ou de outra lista externa: será refeita")
        manifesto()
        for c in cfgs:      # rodada 4: base sem lista neste pacote (ou fora por --dominios): a etapa dela não roda, e quem depende dela cai por dependência
            if c.get("v4") and c["tipo"] == "externo" and c["chave"] not in D.dom and estado[slug(c["nome"])]["tudo"]["estado"] == Estado.PENDENTE.value:
                evento(c, "tudo", tipo="cancelou", motivo=D.dom_fora.get(c["chave"], "sem lista"))
        if a.rodada in ("5", "6") and not D.P5["n5k_ok"]:     # 14.3: sem o recorte do Nutrition5k, a alavanca 1 é cancelada
            motivo = "recorte do Nutrition5k indisponível (" + D.P5["por_base"]["n5k"]["situacao"] + "): alavanca 1 cancelada (14.3)"
            diz(f"ATENÇÃO: {motivo}")
            for c in cfgs:
                if c.get("prato"):
                    for q, e in estado[slug(c["nome"])].items():
                        if e["estado"] == Estado.PENDENTE.value:
                            evento(c, q, tipo="cancelou", motivo=motivo)
        for c in cfgs:
            if estado_config(estado[slug(c["nome"])]) == "completa":
                fecha_config(c)
        confere_paridade()          # retomada: R0 e H0 já em disco
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
    a.externo_ate = t_ini + (PRAZO_DOWNLOAD4 if a.rodada in ("4", "5", "6") else 45 * 60)   # prazo para cada base externa chegar; depois disso a execução segue sem ela

    def linha_externa(c):
        """Baixa e prepara a base externa numa linha à parte (rede e CPU), desde o começo: as GPUs seguem com o
        Nutrition5k enquanto isso. Se falhar, só as configurações com a base são canceladas, e o manifesto diz por quê."""
        while estado[slug(c["nome"])]["tudo"]["estado"] == Estado.PENDENTE.value:
            with trava:
                evento(c, "tudo", tipo="iniciou", dispositivo="cpu"); manifesto()
            t0 = time.time()
            nome_b = c.get("chave") or a.externo
            onde = info["dominios"][c["chave"]] if c.get("v4") else info["externo"]
            try:
                prepara_dominio4(a, D, c["chave"]) if c.get("v4") else prepara_externo(a, D)
                if c.get("v5"):         # rodada 5: os recortes da base também (14.3)
                    onde["recortes"] = prepara_prato5(a, D, c["chave"], D.P5)
                with trava:
                    evento(c, "tudo", tipo="concluiu", seg=time.time() - t0)
                    onde["situacao"] = "fotos prontas"; onde["seg_ate_as_fotos"] = round(time.time() - t_ini); manifesto()
            except BaseException as ex:      # qualquer falha aqui só tira a base externa da execução
                diz(f"base externa {nome_b}: {type(ex).__name__}: {ex}")
                with trava:
                    evento(c, "tudo", tipo="falhou", erro=f"{type(ex).__name__}: {ex}")
                    if estado[slug(c["nome"])]["tudo"]["estado"] == Estado.FALHOU.value:
                        onde["situacao"] = f"FORA desta execução: as fotos não ficaram prontas ({type(ex).__name__}: {ex})"
                        diz(f"ATENÇÃO: a base externa {nome_b} fica de FORA desta execução; " + ("as configurações que dependem dela não rodam" if c.get("v4") else "segue como em 12.3"))
                    manifesto()

    if D.ext and any(c["tipo"] == "externo" and estado_config(estado[slug(c["nome"])]) == "completa" for c in cfgs):
        info["externo"]["situacao"] = "fotos prontas"
    if a.rodada in ("4", "5", "6"):
        for c in cfgs:
            if c["tipo"] == "externo" and estado_config(estado[slug(c["nome"])]) == "completa":
                info["dominios"][c["chave"]]["situacao"] = "fotos prontas"
    ts_ext = [threading.Thread(target=linha_externa, args=(c,), daemon=True) for c in cfgs
              if c["tipo"] == "externo" and estado[slug(c["nome"])]["tudo"]["estado"] == Estado.PENDENTE.value] if pendentes else []
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
    reserva = 0 if a.sem_servico else 900 if a.rodada == "6" else 720 if a.rodada in ("4", "5") else 330    # tempo guardado para a fase de serviço (rodada 4: dois So400m e três ajustes finos)

    seg = lambda c: seg_parte(c, cfgs, estado)

    def proxima(i, dev):
        """Decide o que o processo i faz (ver proxima_acao e simula_orcamento); aplica cancelamentos e cortes de orçamento."""
        while True:
            agora = time.time()
            tenta_decisao()
            livres = [agora if j not in rodando else max(agora, rodando[j][1] + seg(rodando[j][0])) for j in range(len(disp))]
            acao, c, x = proxima_acao(cfgs, estado, simula_orcamento(cfgs, estado, livres, seg, prazo - reserva), gpu=i if a.rodada in ("4", "5", "6") else None)
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
                    if c.get("apelido") in ("R0", "H0", "R4c6", "H0c6") and p == "tudo":
                        confere_paridade()
                    if c.get("v4") and p == "ext":      # os pesos de cada fold só serviam para prever as fotos externas
                        for arq_m in glob.glob(os.path.join(a.saida, f"modelo__{slug(c['nome'])}__fold*.pt")):
                            os.remove(arq_m)
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
                or not os.path.exists(arq_de(a.saida, c, "pesos" if c["tipo"] in ("ft", "destila-rgbd", "ftparcial") else "cabeca")):
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
            # na rodada 4: os ajustes finos pequenos, depois o So400m com as duas cabeças (D1) e, por último, o ajuste fino parcial
            for c in sorted(cfgs, key=lambda c: (0 if (c.get("v3") or c.get("v4")) and c["tipo"] in ("unico", "ft") else 2 if c["tipo"] == "ftparcial" else 1)):
                serve(c); manifesto()
            adot = bool((info.get("decisao_externo") or {}).get("adotado"))      # nos totais entra a versão adotada de cada candidato
            troca = lambda nome: next((c["nome"] for c in cfgs if adot and c.get("plano") == nome and c["tipo"] in ("unico", "ft")), nome)
            sv = monta_servico(a, cfgs, {"smoke_todos": [c["nome"] for c in cfgs if est_serv[slug(c["nome"])]["servico"]["estado"] == Estado.FEITA.value]} if a.smoke
                               else pipelines6(cfgs) if a.rodada == "6"
                               else {(c["apelido"] if c["apelido"] not in ("R0", "R0c") else ("D1_so400m_duas_cabecas" if c["apelido"] == "R0" else "A0c_so400m_recorte_duas_cabecas")): [c["nome"]]
                                     for c in cfgs if c.get("apelido") and c.get("servico") is not False} if a.rodada in ("4", "5", "6")
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
        confere_paridade(fim=True)
        info["disco_livre_gb_fim"] = dict(saida=livre(a.saida), cache=livre(a.cache))
        manifesto()
        tam = grava_tamanhos(a)
        info["tamanho_saida"] = {k: tam[k] for k in ("total_gb", "n_arquivos", "por_tipo_mb", "maior_arquivo")}
        manifesto()
    diz("\n" + "\n".join(f"  {ic['estado']:14s} {n:44s} " + (f"RMSE {ic['cv_treino']['rmse_logm']:.4f} crit {ic['cv_treino']['crit_direto']:.1f}/{ic['cv_treino']['crit_mxd']:.1f}/{ic['cv_treino']['crit_geo']:.1f}% {ic['min']} min"
                                                                  if ic["cv_treino"] else "") + f" | serviço: {ic['servico']['estado']}" for n, ic in info["configs"].items()))
    if a.rodada in ("4", "5", "6"):
        diz("bases externas: " + " | ".join(f"{ch}: {v['situacao']}" for ch, v in info["dominios"].items()))
        diz(f"confiança: {info['confianca']}. Nada foi escolhido nem pontuado fora da validação cruzada do treino.")
    else:
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
    ap.add_argument("--rodada", choices=sorted(RODADAS), default="4",
                    help="quais configurações: 1 = as 15 da seção 10; 2 = as 11 da seção 11; 3 = as 13 da seção 12.3 (treino_v2, só teste2); "
                         "4 = as da seção 13 (treino_v2 + bases externas com tabela de divisão)")
    ap.add_argument("--dominios", help="rodada 4: bases externas a usar, separadas por vírgula (padrão: todas as que têm lista e tabela de divisão em --listas; 'nenhum' = nenhuma)")
    ap.add_argument("--espelho", help="rodada 4: pasta onde procurar, antes de baixar, um espelho dos arquivos das bases externas (além de /kaggle/input)")
    ap.add_argument("--caixas", help="rodada 5: pasta com as tabelas da etapa 0 (caixas_r5_<base>.csv); padrão: a saída do kernel da etapa 0 em /kaggle/input")
    ap.add_argument("--sem-conferencia", action="store_true", help="rodada 5: roda mesmo sem a conferência a olho da etapa 0 (só para teste)")
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
    a.dominios = None if a.dominios is None else [d for d in a.dominios.split(",") if d and d != "nenhum"]
    assert a.dominios is None or all(d in DOMINIOS4 for d in a.dominios), f"--dominios: escolha entre {list(DOMINIOS4)}"
    a.espelho = os.path.abspath(a.espelho) if a.espelho else None
    imagens4 = os.path.abspath(a.externo_imagens) if a.externo_imagens and a.rodada in ("4", "5", "6") else None
    a.caixas = os.path.abspath(a.caixas) if a.caixas else None
    if a.externo:
        assert a.rodada == "3", "--externo só existe na rodada 3"
        a.externo_lista = os.path.abspath(a.externo_lista or os.path.join(a.listas, EXTERNOS[a.externo]["lista"]))
        a.externo_imagens = os.path.abspath(a.externo_imagens) if a.externo_imagens else None
        a.externo_pesos = EXTERNOS[a.externo]["pesos"]
        try:        # 12.5: se a lista não estiver pronta ou não passar nas conferências, a execução segue sem ela
            dados_de(a)
        except (OSError, AssertionError, KeyError, ValueError) as e:
            assert not a.job, f"lista externa inválida num processo filho: {e}"
            a.externo_fora = f"FORA desta execução: a lista de {a.externo} não está pronta ou não passou nas conferências ({type(e).__name__}: {e})"
            print(f"ATENÇÃO: a base externa {a.externo} fica de FORA desta execução ({type(e).__name__}: {e}); segue como em 12.3", flush=True)
            a.externo = None
    if not a.externo:
        a.externo_lista = a.externo_imagens = a.externo_pesos = None
    if a.rodada in ("4", "5", "6"):
        a.externo_imagens = imagens4        # na rodada 4: pasta com uma subpasta por base (fotos já em disco; teste)
    gemeas = lambda cs: com_externo(cs, a.externo) if a.externo else cs
    if a.job:       # processo filho: procura a configuração entre todas as declaradas para a rodada
        return principal_parte(a, gemeas(configs_da_rodada(a.rodada, a.smoke)) if a.rodada in ("3", "4", "5", "6") else (CONFIGS_SMOKE if a.smoke else CONFIGS))
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
                     + ("estimadas com os tempos das execuções 2 a 4" if a.rodada in ("3", "4", "5", "6") else "medidas na execução 1")
                     + " (2 × T4: perto de metade disso em relógio), mais a fase de serviço")
    orquestra(a, cfgs)


if __name__ == "__main__":
    main()
