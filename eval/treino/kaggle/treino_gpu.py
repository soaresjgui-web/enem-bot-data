#!/usr/bin/env python3
"""Treino em GPU (Kaggle/Colab) do modelo de proteína pela foto: seções 8, 10, 11 e 12 de resultado_modelo_treinado.md.

Premissa do produto: a inferência usa UMA foto RGB tirada de cima. Critério: proteína dentro de ±20% ou ±5 g.

Uso (Kaggle/Colab, pelo notebook nutrelle_treino_gpu.ipynb, ou direto):
  python3 treino_gpu.py                       execução 3 (padrão): as 13 configurações da seção 12.3, até o orçamento
  python3 treino_gpu.py --rodada 2            as 11 configurações da execução 2 (seção 11); --rodada 1: as 15 da seção 10
  python3 treino_gpu.py --lista               mostra as configurações da rodada e sai
  python3 treino_gpu.py --so "ft siglip2"     só as configurações cujo nome contém o texto
  python3 treino_gpu.py --smoke               teste rápido em CPU (backbone minúsculo, poucas dezenas de pratos)
  python3 treino_gpu.py --checa-modelos       monta cada backbone sem pesos e passa uma imagem (sem rede)
  python3 treino_gpu.py --autoteste           testes da máquina de estados, do escalonador e dos alvos aninhados

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
import argparse, csv, glob, hashlib, json, math, os, shutil, subprocess, sys, tempfile, threading, time, traceback, zlib
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
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
    return ["tudo"] if cfg["tipo"] in ("ridge", "fracoes", "professor", "unico") else list(PARTES_FT)


def todas_as_configs():
    return CONFIGS + CONFIGS3 + CONFIGS_SMOKE + CONFIGS_SMOKE3


def configs_da_rodada(rodada, smoke=False):
    if rodada == "3":
        return list(CONFIGS_SMOKE3 if smoke else CONFIGS3)
    if smoke:
        return list(CONFIGS_SMOKE)
    return [c for c in CONFIGS if c["nome"] in RODADAS[rodada]]     # a ordem de CONFIGS (prioridade) é mantida


def com_externo(cfgs, quais):
    """Gancho de 12.4: para cada configuração de ajuste fino pedida, uma gêmea '+ext' que treina também com a lista
    externa. Ficam no fim da fila (são ablações: com e sem)."""
    gemeas = []
    for c in cfgs:
        if c["tipo"] == "ft" and c.get("v3") and any(q in c["nome"] for q in quais):
            gemeas.append(dict(c, nome=c["nome"] + " +ext", externo=True))
    return list(cfgs) + gemeas


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
    """Estado da configuração, derivado das partes: completa | falhou | sem_orcamento | em_andamento | pendente."""
    es = {Estado(p["estado"]) for p in partes.values()}
    if es == {Estado.FEITA}:
        return "completa"
    if Estado.FALHOU in es:
        return "falhou"
    if Estado.SEM_ORCAMENTO in es:
        return "sem_orcamento"
    if es == {Estado.PENDENTE}:
        return "pendente"
    return "em_andamento"


def simula_orcamento(cfgs, estado, livres, seg_parte, limite):
    """Quais configurações cabem no tempo que resta? Função pura. livres = instante em que cada processo fica livre;
    seg_parte(cfg) = segundos estimados por parte. Em ordem de prioridade, as partes pendentes de cada configuração vão
    para o processo que fica livre primeiro; a configuração cabe se todas terminam até `limite`. Quem não cabe não ocupa
    tempo, e as de prioridade menor ainda podem caber. Devolve {nome: bool}."""
    livres, cabe = list(livres), {}
    for c in cfgs:
        n = sum(p["estado"] == Estado.PENDENTE.value for p in estado[slug(c["nome"])].values())
        if not n:
            continue
        f, fim = list(livres), 0.0
        for _ in range(n):
            g = min(range(len(f)), key=f.__getitem__)
            f[g] += seg_parte(c); fim = max(fim, f[g])
        cabe[c["nome"]] = fim <= limite
        if cabe[c["nome"]]:
            livres = f
    return cabe


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
        if not pend:
            continue
        deps = [por_nome[d] for d in c.get("depende", []) if d in por_nome]
        mortas = [d["nome"] for d in deps if estado_config(est(d)) != "completa" and not viva(d)]
        if mortas:
            return "cancela", c, f"dependência não ficou pronta: {mortas[0]}"
        if not cabe.get(c["nome"], True):
            return "sem_orcamento", c, pend[0]
        if any(estado_config(est(d)) != "completa" for d in deps):
            bloqueio = espera = True
            continue
        if bloqueio and not c.get("livre"):
            espera = True
            continue
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
    print("autoteste: ok")


# ------------------------------------------------------------------ dados (listas do repositório; sem gabarito de avaliação)
class Dados:
    """Treino zenital estrito (ids, folds por cadeia, alvos em log) + ids de avaliação (sem rótulo):
    aval = calib/teste/n5k_v1 (aval_ids.csv); conj = aval + os conjuntos de aval2_ids.csv (teste2), se o arquivo existir."""

    def __init__(self, pasta, smoke=0, rodada="2", externo=None, externo_imagens=None):
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
        self.ext = le_externo(externo, externo_imagens, self) if externo else None      # gancho de 12.4; desligado por padrão
        ext_ids = self.ext["ids"] if self.ext else []
        self.todos = base + ext_ids + usados(self.extras)
        # blocos de extração de features: treino | externos | pratos só de previsão (teste2), cada um em lotes próprios
        cortes = [0, self.n_base, self.n_base + len(ext_ids), len(self.todos)]
        self.segmentos = [(i, f) for i, f in zip(cortes[:-1], cortes[1:]) if f > i]
        self.chave = hashlib.sha256("\n".join(self.todos).encode()).hexdigest()[:10]


def le_externo(arq, raiz, D):
    """Gancho de 12.4: lista externa no esquema de treino_v2.csv + id, dominio, grupo, vista, arquivo (caminho da foto
    relativo a `raiz`). Alvo ausente = campo vazio (vira NaN e a perda daquele alvo é mascarada). Só treina: fold −1."""
    import re
    with open(arq) as f:
        rows = list(csv.DictReader(f))
    falta = [c for c in ("id", "dominio", "grupo", "vista", "arquivo") if c not in rows[0]]
    assert not falta, f"lista externa sem as colunas {falta}"
    num = lambda r, c: float(r[c]) if r.get(c) not in (None, "") else np.nan
    col = lambda c: np.array([num(r, c) for r in rows])
    m, p = col("massa"), col("proteina")
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.c_[np.log(m), np.log1p(p), np.log1p(col("kcal")), np.log(p / m + EPS_D), np.log1p(col("gordura")), np.log1p(col("carbo")),
                  np.log1p(np.array([[num(r, f"m_{g}") for g in D.grupos] for r in rows])),
                  np.log1p(np.array([[num(r, f"p_{g}") for g in D.grupos] for r in rows]))]
    FF = np.array([[num(r, f"ff_{g}") for g in D.fontes] for r in rows])
    FF[np.isnan(FF).any(1)] = np.nan
    # higiene: nada do teste2, nem prato do Nutrition5k a menos de 120 s de um prato do teste2 (aproximação da regra de
    # cadeia; uma base derivada do Nutrition5k tem de passar antes pela regra completa, com o grafo de cadeias)
    t2 = {d for s in D.extras for d in D.conj[s]}
    ts2 = np.sort([int(d.split("_")[1]) for d in t2 if re.fullmatch(r"dish_\d+", d)])
    for r in rows:
        assert r["id"] not in t2 and r["id"] not in set(D.ids), f"{r['id']}: prato externo repetido no treino ou no teste2"
        if re.fullmatch(r"dish_\d+", r["id"]) and len(ts2):
            t = int(r["id"].split("_")[1]); i = np.searchsorted(ts2, t)
            perto = min(abs(int(ts2[j]) - t) for j in (max(i - 1, 0), min(i, len(ts2) - 1)))
            assert perto >= 120, f"{r['id']}: a {perto} s de um prato do teste2"
    doms = sorted({r["dominio"] for r in rows})
    ids = [f"ext:{r['dominio']}:{r['id']}" for r in rows]
    assert len(set(ids)) == len(ids), "ids repetidos na lista externa"
    return dict(ids=ids, arquivos=[os.path.join(raiz or os.path.dirname(os.path.abspath(arq)), r["arquivo"]) for r in rows], T=T, FF=FF,
                dominios=doms, dom=np.array([doms.index(r["dominio"]) for r in rows]), grupo=[r["grupo"] for r in rows], vista=[r["vista"] for r in rows],
                hash=hashlib.sha256(open(arq, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16])


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
        mu, sd, mt, s2, V, VtZtT = self.dec(folds)
        Xv = self.X[self._m(m)]
        nv, o, Dm = Xv.shape
        Zv = ((Xv.reshape(-1, Dm) - mu) / sd).double()
        if alfa is None:
            ZvV = Zv @ V
            return self.t.stack([ZvV @ (VtZtT / (s2 + a)[:, None]) + mt for a in self.alfas]).reshape(len(self.alfas), nv, o, -1).mean(2)
        return (Zv @ (V @ (VtZtT / (s2[:, None] + alfa[None, :]))) + mt).reshape(nv, o, -1).mean(1)


def pares_aninhados(fold):
    """Os treinos aninhados: para cada par de folds {j, k}, o conjunto S dos outros folds. O modelo treinado em S dá o
    alvo dos pratos do fold j para o aluno que fica sem o fold k, e vice-versa. Nenhum rótulo de j ou k entra em S."""
    fs = sorted(set(int(f) for f in fold))
    return [(j, k, frozenset(fs) - {j, k}) for a, j in enumerate(fs) for k in fs[a + 1:]]


def ridge_aninhado(X, T, fold, dev=None):
    """[5, n, alvos]: na posição [k, i], a previsão para o prato i (fold j != k) de um ridge treinado SEM os folds j e k
    (alfa por alvo escolhido por CV interna só nos folds de treino). Linhas do próprio fold k: NaN."""
    R = RidgeSub(X, T, fold, dev)
    fold = np.asarray(fold)
    out = np.full((int(fold.max()) + 1, len(fold), T.shape[1]), np.nan, np.float32)
    for j, k, S in pares_aninhados(fold):
        erro = 0
        for f in sorted(S):
            erro = erro + ((R.prev(S - {f}, fold == f) - R.T[R._m(fold == f)]) ** 2).sum(1)
        alfa = R.alfas[erro.argmin(0)]
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


def treina_cabeca(X, T, FF, w, dc, cfg, dev, alvos_prof=None, semente=0):
    """Cabeça de frações sobre features congeladas, como treina_soft('fonte-l2 +perda-dens peso-faixa') de
    dens_experimentos.py: MLP de uma camada oculta sobre features padronizadas, cada orientação como amostra; saídas =
    22 em log (padronizadas) + logits das fontes. Perdas, todas com o peso por amostra w: Huber nas 22; 10 × erro
    quadrático entre o softmax e as frações; Huber em log da densidade composta Σ fração × dc. Com alvos_prof [n, 3]
    (log-massa, log-densidade com piso, log1p-proteína do professor), soma PESO_DESTILA × Huber nas saídas 0, 3 e 1.
    Devolve (pesos para servir, função X [n, o, D] -> (22 saídas em log [n, o, 22], frações [n, o, K]))."""
    import torch
    torch.manual_seed(semente)
    n, o, Dm = X.shape
    K, nt = FF.shape[1], T.shape[1]
    mu, sd = X.reshape(-1, Dm).mean(0), X.reshape(-1, Dm).std(0) + 1e-6
    tm, ts = T.mean(0), T.std(0) + 1e-6
    Xt = torch.from_numpy(((X - mu) / sd).reshape(-1, Dm).astype(np.float32)).to(dev)
    Y = torch.from_numpy(np.repeat((T - tm) / ts, o, 0).astype(np.float32)).to(dev)
    Fr = torch.from_numpy(np.repeat(FF, o, 0).astype(np.float32)).to(dev)
    W = torch.from_numpy(np.repeat(w / w.mean(), o).astype(np.float32)).to(dev)
    dct = torch.from_numpy(dc.astype(np.float32)).to(dev)
    q = [I_M, I_D, I_P]
    Pn = None if alvos_prof is None else torch.from_numpy(np.repeat((alvos_prof - tm[q]) / ts[q], o, 0).astype(np.float32)).to(dev)
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
            perda = (sl1(out[:, :nt], Y[j], reduction="none").mean(1) * wj).mean()
            fr = torch.softmax(out[:, nt:], 1)
            perda = perda + 10 * (((fr - Fr[j]) ** 2).sum(1) * wj).mean()
            alvo_d = Y[j][:, I_D] * float(ts[I_D]) + float(tm[I_D])
            perda = perda + (sl1((torch.log(fr @ dct + EPS_D) - alvo_d) / float(ts[I_D]), torch.zeros_like(alvo_d), reduction="none") * wj).mean()
            if Pn is not None:
                perda = perda + cfg.get("peso_destila", PESO_DESTILA) * (sl1(out[:, q], Pn[j], reduction="none").mean(1) * wj).mean()
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


def baixa_arquivo(url, dest, tentativas=8):
    """Download de arquivo grande (pesos) com novas tentativas e retomada por Range. Só existe em `dest` se completo."""
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
                        shutil.copyfileobj(r, f, 1 << 22)
                if total and os.path.getsize(parte) != total:
                    raise IOError(f"incompleto: {os.path.getsize(parte)} de {total} bytes")
                os.replace(parte, dest)
                print(f"pesos: {os.path.basename(dest)} ({os.path.getsize(dest) / 1e6:.0f} MB) de {url}", flush=True)
                return dest
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
            ext = dict(zip(self.D.ext["ids"], self.D.ext["arquivos"])) if self.D.ext else {}     # fotos externas: arquivo local
            baixa([d for d in self.D.todos if d not in ext], "rgb.png", self.raiz)
            from PIL import Image
            t0 = time.time()
            mm = np.lib.format.open_memmap(self._arq("rgb.tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, self.H, self.W, 3))
            cor = np.zeros((n, 3), np.float32)

            def um(i):
                d = self.D.todos[i]
                im = Image.open(ext[d] if d in ext else os.path.join(self.raiz, "img", d, "rgb.png")).convert("RGB")
                fora = im.size != (self.W, self.H)
                if fora and d in ext:       # foto externa: cabe inteira em 640×480, com a cor da moldura em volta
                    z = min(self.W / im.size[0], self.H / im.size[1])
                    pq = np.asarray(im.resize((max(1, round(im.size[0] * z)), max(1, round(im.size[1] * z))), Image.BICUBIC))
                    kb = max(1, int(0.03 * min(pq.shape[:2])))
                    borda = np.median(np.concatenate([pq[:kb].reshape(-1, 3), pq[-kb:].reshape(-1, 3), pq[:, :kb].reshape(-1, 3), pq[:, -kb:].reshape(-1, 3)]), 0)
                    tela = np.empty((self.H, self.W, 3), np.uint8); tela[:] = borda.astype(np.uint8)
                    y0, x0 = (self.H - pq.shape[0]) // 2, (self.W - pq.shape[1]) // 2
                    tela[y0:y0 + pq.shape[0], x0:x0 + pq.shape[1]] = pq
                    im = Image.fromarray(tela)
                elif fora:
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

    def abre(self, profundidade=False):
        self.rgb = np.load(self._arq("rgb.npy"), mmap_mode="r")
        self.cor = np.load(self._arq("cor.npy"))
        if profundidade:
            self.alt = np.load(self._arq("alt.npy"), mmap_mode="r")
        assert len(self.rgb) == len(self.D.todos)
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
    x = torch.from_numpy(np.ascontiguousarray(C.rgb[linhas])).to(dev).permute(0, 3, 1, 2).float().div_(255)
    cor = torch.from_numpy(C.cor[linhas]).to(dev)
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


def _extrai_feats(arq, bb, D, C, dev, bs, prazo, segmentos):
    import torch
    t0 = time.time()
    rede = cria_rede(bb, D.N).to(dev).eval()
    hw = BACKBONES[bb]["hw"]
    X = np.zeros((segmentos[-1][1], 8, rede.dim), np.float16)
    with torch.inference_mode():
        for ini, fim in segmentos:   # treino (+ 450) nos mesmos lotes de sempre; externos e teste2 em lotes à parte, depois
            for i in range(ini, fim, bs):
                if time.time() > prazo:
                    raise SemOrcamento()
                x = monta_lote(C, list(range(i, min(i + bs, fim))), dev, hw)
                for k in range(8):
                    with autocast(dev):
                        f = rede.feats(orienta(x, k))
                    X[i:i + len(x), k] = f.float().cpu().numpy()
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
    grupos = grupos_lr(rede, cfg["lr"], cfg["lld"], cfg["wd"])
    vies = None
    if ext:         # viés de log-massa por domínio externo (na previsão vale o do Nutrition5k, que é zero)
        vies = torch.nn.Parameter(torch.zeros(len(ext["dominios"]) + 1, device=dev))
        grupos.append(dict(params=[vies], lr=cfg["lr"], weight_decay=0.0))
    opt = torch.optim.AdamW(grupos)
    n_ext_ep = min(len(ext["ids"]), len(tr) // 2) if ext else 0      # por época, no máximo metade do peso do Nutrition5k
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
    linha = np.array([C.pos.get(d, -1) for d in D.ids] + ([C.pos[e] for e in ext["ids"]] if ext else []))   # -1: fora do subconjunto do --smoke
    # alvos por amostra; as externas vêm depois das do Nutrition5k, com NaN onde o alvo falta (perda mascarada)
    T_tudo = np.r_[T, ext["T"]] if ext else T
    Yn = torch.from_numpy(np.nan_to_num((T_tudo - tm) / ts).astype(np.float32)).to(dev)
    if novo:
        n5k = len(D.ids)
        q = [I_M, I_D, I_P]
        tem = torch.from_numpy(np.isfinite(T_tudo).astype(np.float32)).to(dev)                        # máscara por alvo
        w_am = np.ones(len(T_tudo), np.float32)
        if cfg.get("peso_faixa"):
            w_am[tr] = peso_faixa(D.proteina[tr])
        Wam = torch.from_numpy(w_am).to(dev)
        dom = torch.from_numpy(np.r_[np.zeros(n5k, int), (ext["dom"] + 1) if ext else np.zeros(0, int)]).to(dev)
        if K:
            dc = dens_por_fonte(D.FF[tr], D.massa[tr], D.proteina[tr])
            dct = torch.from_numpy(dc.astype(np.float32)).to(dev)
            FF_tudo = np.r_[D.FF, ext["FF"]] if ext else D.FF
            FFt = torch.from_numpy(np.nan_to_num(FF_tudo).astype(np.float32)).to(dev)
            tem_ff = torch.from_numpy(np.isfinite(FF_tudo).all(1).astype(np.float32)).to(dev)
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
        if vies is not None:
            o22 = torch.cat([o22[:, :1] + vies[dom[jt]][:, None] * (dom[jt] > 0)[:, None], o22[:, 1:]], 1)
        p = ((huber(o22, Yn[jt], reduction="none") * w * tem[jt]).mean(1) * wj).mean()
        if K:
            fr = torch.softmax(out[:, N:], 1)
            p = p + 10 * (((fr - FFt[jt]) ** 2).sum(1) * tem_ff[jt] * wj).mean()
            alvo_d = Yn[jt][:, I_D] * ts_t[I_D] + tm_t[I_D]
            p = p + (huber((torch.log(fr @ dct + EPS_D) - alvo_d) / ts_t[I_D], torch.zeros_like(alvo_d), reduction="none") * tem[jt][:, I_D] * wj).mean()
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
        amostras = tr if not ext else np.r_[tr, len(D.ids) + rng.choice(len(ext["ids"]), n_ext_ep, replace=False)]
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
    arq = arq_parte(a.saida, cfg, parte)
    if not os.path.exists(arq):
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
    D = Dados(a.listas, a.smoke, a.rodada, a.externo, a.externo_imagens)
    destila = cfg["tipo"] == "destila-rgbd"
    C = Cache(a.cache, D).abre(profundidade=destila)
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

    if cfg["tipo"] == "ridge":
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

        def uma(m, alvos):
            """Treina nos pratos da máscara m (posições em tr), só com rótulos e alvos de m."""
            tempo()
            assert alvos is None or not np.isnan(alvos).any(), "alvo do professor faltando para um prato de treino"
            dc = dens_por_fonte(FFtr[m], mtr[m], ptr[m])
            pesos, prev = treina_cabeca(Xtr[m], Ttr[m], FFtr[m], peso_faixa(ptr[m]), dc, cfg, dev, alvos)
            return dc, pesos, prev

        for k in sorted(set(fold_tr.tolist())):
            m = fold_tr != k
            dc, _, prev = uma(m, None if prof is None else prof["alvos_aninhados"][k][tr][m])
            guarda("oof", tr[~m], n, *prev(Xtr[~m]), dc)
            dens_fonte[k] = dc
        todos = np.ones(len(tr), bool)
        dc, pesos, prev = uma(todos, None if prof is None else prof["alvos_oof"][tr])
        dens_fonte[5] = dc
        for s in D.conj:
            guarda(s, usa[s], len(D.conj[s]), *prev(X[lin[s]]), dc)
        saida.update(dens_fonte=dens_fonte, fontes=np.array(D.fontes))
        if cfg["tipo"] == "fracoes":     # densidade por frações aninhada, para os alvos do professor
            an = nan(5, n)
            for j, k, S in pares_aninhados(fold_tr):
                m = np.isin(fold_tr, list(S))
                dcs, _, prevs = uma(m, None)
                an[k, tr[fold_tr == j]] = media_or(prevs(Xtr[fold_tr == j])[1], 8) @ dcs
                an[j, tr[fold_tr == k]] = media_or(prevs(Xtr[fold_tr == k])[1], 8) @ dcs
            saida["aninhado_dens_ff"] = an
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

        def combina(lm, ld):
            lm, ld = np.mean(lm, 0), np.mean(ld, 0)
            return lm, ld, np.log1p(np.exp(lm) * np.clip(np.exp(ld) - EPS_D, 0, None))

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
            alvos = zp["alvos_oof"] if k is None else zp["alvos_aninhados"][k]
        rede, preve, hist, (tm, ts) = treina(cfg, D, C, trk, va, dev, prazo, a.div_lote, semente=sem, rotulo=rot, alvos_prof=alvos)
        dc = preve.dens_fonte
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
    D = Dados(a.listas, a.smoke, a.rodada, a.externo, a.externo_imagens)
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
        for padrao in ("parte__*.npz", "pesos__*.pt", "cabeca__*.npz", "servir__*.json", "servico__*.json"):
            for arq in glob.glob(os.path.join(pasta, padrao)):
                dest = os.path.join(a.saida, os.path.basename(arq))
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
        pred.update({c: v for c, v in fin.items() if any(c.startswith(sp) for sp in D.conj)})
        if "dens_fonte" in fin:
            pred.update(dens_fonte=np.stack([partes[f"fold{k}"]["dens_fonte"] for k in range(5)] + [fin["dens_fonte"]]), fontes=fin["fontes"])
    principais = [c for c in pred if (c.startswith("oof") and "aninhado" not in c) or any(c.startswith(sp) for sp in D.conj)]
    for c in principais:      # nenhuma previsão faltando nos pratos usados
        usados = D.usa_tr if c.startswith("oof") else D.usa_av[next(sp for sp in D.conj if c.startswith(sp))]
        assert not np.isnan(pred[c][usados]).any(), f"previsão faltando em '{c}'"
    for suf in ORIENT:
        assert "oof" + suf in pred and all(sp + suf in pred for sp in D.conj), f"variante '{suf}' faltando"
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
    D = Dados(a.listas, a.smoke, a.rodada, a.externo, a.externo_imagens)
    a._hash = D.hash
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
        diz(f"dados externos (gancho de 12.4): {len(D.ext['ids'])} amostras, domínios {D.ext['dominios']}, lista {D.ext['hash']}; só treinam "
            f"as configurações '+ext': {[c['nome'] for c in cfgs if c.get('externo')]}")
    info = dict(formato=FORMATO, rodada=a.rodada, smoke=bool(a.smoke), listas_hash=D.hash, listas2_hash=D.hash2,
                externo=dict(lista=D.ext["hash"], n=len(D.ext["ids"]), dominios=D.ext["dominios"]) if D.ext else None,
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
            cmd += ["--externo", a.externo, "--externo-configs", a.externo_configs] + (["--externo-imagens", a.externo_imagens] if a.externo_imagens else [])
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
                    evento(c, p, tipo="retomou", seg=float(np.load(arq_parte(a.saida, c, p))["seg"]))
                elif os.path.exists(arq_parte(a.saida, c, p)):
                    diz(f"retomada: {c['nome']} | {p} existe, mas sem teste2 ou sem os arquivos de servir: será refeita")
        manifesto()
        for c in cfgs:
            if estado_config(estado[slug(c["nome"])]) == "completa":
                fecha_config(c)
        manifesto()
        nomes, todas = {c["nome"] for c in cfgs}, {c["nome"]: c for c in todas_as_configs()}
        for c in cfgs:      # dependência fora desta execução (--so): tem de estar pronta em disco
            for d in c.get("depende", []):
                if d not in nomes and not os.path.exists(arq_parte(a.saida, todas.get(d, dict(nome=d)), "tudo")):
                    for q, e in estado[slug(c["nome"])].items():
                        if e["estado"] == Estado.PENDENTE.value:
                            evento(c, q, tipo="cancelou", motivo=f"dependência fora desta execução e sem arquivo: {d}")
        manifesto()
    pendentes = [c for c in cfgs if estado_config(estado[slug(c["nome"])]) != "completa"]
    if pendentes or not a.sem_servico:
        Cache(a.cache, D).prepara(profundidade=any(c["tipo"] == "destila-rgbd" for c in pendentes))

    rodando = {}        # processo -> (configuração, início) da parte em execução
    reserva = 0 if a.sem_servico else 240        # tempo guardado para a fase de serviço dos candidatos

    def seg_parte(c):
        """Segundos por parte: mediana das partes já medidas nesta sessão; antes disso, a estimativa declarada."""
        feitas = [x["seg"] for x in estado[slug(c["nome"])].values() if x["estado"] == Estado.FEITA.value and x["seg"] and not x["retomada"]]
        return float(np.median(feitas)) if feitas else c["est_min"] * 60.0 / len(partes_de(c))

    def proxima(i, dev):
        """Decide o que o processo i faz (ver proxima_acao e simula_orcamento); aplica cancelamentos e cortes de orçamento."""
        while True:
            agora = time.time()
            livres = [agora if j not in rodando else max(agora, rodando[j][1] + seg_parte(rodando[j][0])) for j in range(len(disp))]
            acao, c, x = proxima_acao(cfgs, estado, simula_orcamento(cfgs, estado, livres, seg_parte, prazo - reserva))
            if acao == "cancela":
                diz(f"{c['nome']} cancelada: {x}"); cancela(c, x)
            elif acao == "sem_orcamento":
                diz(f"{c['nome']} não cabe no orçamento ({seg_parte(c) / 60:.1f} min por parte, restam {(prazo - reserva - agora) / 60:.0f} min): fica de fora")
                evento(c, x, tipo="sem_orcamento"); cancela(c, "sem orçamento para a configuração inteira")
            elif acao == "roda":
                evento(c, x, tipo="iniciou", dispositivo=dev); rodando[i] = (c, agora)
                return c, x
            else:
                return acao

    def cancela(c, motivo):
        for p, e in estado[slug(c["nome"])].items():
            if e["estado"] == Estado.PENDENTE.value:
                evento(c, p, tipo="cancelou", motivo=motivo)

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
        [t.start() for t in ts]; [t.join() for t in ts]

    # ---- fase de serviço: depois de todo o treino, um modelo por vez numa só GPU, com a CPU livre (latências limpas)
    def serve(c):
        s = slug(c["nome"])
        if os.path.exists(arq_de(a.saida, c, "servico")):
            return evento(c, "servico", tipo="retomou")
        if c.get("servico") is False:
            return evento(c, "servico", tipo="cancelou", motivo="sem modelo próprio a medir (mesmo backbone de um ridge, ou composição)")
        if estado_config(estado[s]) != "completa" or not os.path.exists(arq_de(a.saida, c, "servir")) \
                or not os.path.exists(arq_de(a.saida, c, "pesos" if c["tipo"] in ("ft", "destila-rgbd") else "cabeca")):
            return evento(c, "servico", tipo="cancelou", motivo="configuração incompleta ou sem arquivos de servir")
        while est_serv[s]["servico"]["estado"] == Estado.PENDENTE.value:
            if time.time() + 300 > prazo:
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
        if not a.sem_servico:
            # na rodada 3, primeiro os candidatos a modelo único (a regra de 12.3 usa o tempo deles); depois os congelados
            for c in sorted(cfgs, key=lambda c: 0 if c.get("v3") and c["tipo"] in ("unico", "ft") else 1):
                serve(c); manifesto()
            sv = monta_servico(a, cfgs, {"smoke_todos": [c["nome"] for c in cfgs if c.get("servico") is not False]} if a.smoke
                               else (PIPELINES3 if a.rodada == "3" else PIPELINES))
            for k, v in sv["pipelines"].items():
                diz(f"serviço, {k}: {'completo' if v['completo'] else 'FALTAM ' + str(v['faltam'])} | {v['parametros'] / 1e6:.0f} M parâmetros | "
                    f"pesos {v['bytes_pesos'] / 2 ** 30:.2f} GB | GPU ms/imagem {v['gpu_ms_por_imagem']} | CPU ms/imagem {v['cpu_ms_por_imagem']}")
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
    ap.add_argument("--externo", help="gancho de 12.4 (desligado por padrão): lista externa .csv; cria configurações gêmeas '+ext'")
    ap.add_argument("--externo-imagens", help="pasta das fotos da lista externa (coluna `arquivo`); padrão: a pasta da lista")
    ap.add_argument("--externo-configs", default="ft siglip2_b16_224 r2 destilado", help="quais ajustes finos ganham a gêmea '+ext' (trechos do nome, separados por ;)")
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
    if a.externo:
        assert a.rodada == "3", "--externo só existe na rodada 3"
        a.externo = os.path.abspath(a.externo)
        a.externo_imagens = os.path.abspath(a.externo_imagens) if a.externo_imagens else None
    gemeas = lambda cs: com_externo(cs, [q for q in a.externo_configs.split(";") if q]) if a.externo else cs
    if a.job:       # processo filho: procura a configuração entre todas as declaradas para a rodada
        return principal_parte(a, gemeas(configs_da_rodada("3", a.smoke)) if a.rodada == "3" else (CONFIGS_SMOKE if a.smoke else CONFIGS))
    cfgs = gemeas(configs_da_rodada(a.rodada, a.smoke))
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
