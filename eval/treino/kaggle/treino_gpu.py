#!/usr/bin/env python3
"""Seção 8 em GPU (Kaggle/Colab): os experimentos que não couberam em CPU, de ponta a ponta e sem interação.

Premissa do produto: a inferência usa UMA foto RGB tirada de cima. Critério: proteína dentro de ±20% ou ±5 g.

Uso (Kaggle/Colab, pelo notebook nutrelle_treino_gpu.ipynb, ou direto):
  python3 treino_gpu.py                       roda todas as configurações, em ordem de prioridade, até o orçamento
  python3 treino_gpu.py --lista               mostra as configurações declaradas e sai
  python3 treino_gpu.py --so "ft siglip2"     só as configurações cujo nome contém o texto
  python3 treino_gpu.py --smoke               teste rápido em CPU (backbone minúsculo, poucas dezenas de pratos)
  python3 treino_gpu.py --checa-modelos       monta cada backbone sem pesos e passa uma imagem (sem rede)
  python3 treino_gpu.py --autoteste           testes da máquina de estados (sem torch)

O que faz:
  1. Baixa do bucket público do Nutrition5k as fotos zenitais (e a profundidade, só se a destilação for rodar)
     dos pratos de listas/treino.csv e listas/aval_ids.csv, em paralelo, com novas tentativas e retomada.
  2. Para cada configuração declarada em CONFIGS (ordem = prioridade): previsões fora-do-fold no treino
     (os MESMOS 5 folds por cadeia da CPU, coluna `fold` de listas/treino.csv) + previsões nos 450 pratos de
     avaliação por um modelo treinado em todo o treino, com média de 8 orientações (grupo diedral).
  3. Grava na pasta de saída, sem subpastas: pred__<config>.npz/.csv (mesmo esquema do banco de previsões
     zen_banco), log__<config>.json (curvas de perda, tempos), parte__*.npz (retomada), pesos__<melhor>.pt,
     manifesto.json (estado de cada parte), eventos.jsonl e treino_gpu.log.

Higiene (igual à da CPU, ver docs/dados/precisao/resultado_modelo_treinado.md, 8.1):
  - treino = os 1.684 pratos de listas/treino.csv (regra estrita de cadeia); os 450 pratos de avaliação nunca
    entram no treino, nem como imagem sem rótulo (o script confere a interseção);
  - este script NÃO lê o gabarito de calib/teste/n5k_v1: só grava previsões; a nota sai em ingere_kaggle.py;
  - sem parada antecipada: épocas fixas por receita; a validação por época só é registrada.

Retomada: cada parte (fold0..fold4, final) vira um arquivo parte__*.npz ao terminar. Ao iniciar, o script
copia partes de --retoma (ou de /kaggle/input/**, se houver um manifesto.json deste formato) e pula o que já existe.
"""
import argparse, csv, glob, hashlib, json, math, os, shutil, subprocess, sys, tempfile, threading, time, traceback, zlib
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import numpy as np

FORMATO = "nutrelle-treino-gpu/1"
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


def _ft(bb, receita, est_min, tipo="ft", sufixo="", **kw):
    c = dict(nome=f"gpu {tipo} {bb} {receita}{sufixo}", tipo=tipo, bb=bb, receita=receita, bs=32, acum=1, escala=(0.7, 1.3),
             ckpt_grad=False, wd=0.05, drop_path=0.1, ema=0.99, aquec=0.1, est_min=est_min, **RECEITAS[receita])
    c.update(kw)
    return c


_ALTA = dict(bs=16, acum=2, ckpt_grad=True)     # 768 tokens: lote menor + checkpoint de gradiente (16 GB)
# Ordem = prioridade. est_min = estimativa (NÃO medida) de minutos numa única T4 para a configuração inteira.
CONFIGS = [
    _ridge("siglip2_b16_224", 3),                                   # paridade com a CPU: "ridge siglip2_b16 o8" (RMSE 0,2551)
    _ridge("siglip2_b16_384x512", 7),                               # 8 orientações na resolução maior (não rodado na CPU)
    _ridge("dinov2_b14_336x448", 7),                                # backbone alternativo, congelado
    _ft("siglip2_b16_224", "r1", 35),                               # ajuste fino completo a 224
    _ft("siglip2_b16_384x512", "r1", 110, epocas=12, **_ALTA),      # ajuste fino completo na resolução maior
    _ft("dinov2_b14_336x448", "r1", 110, epocas=12, **_ALTA),       # backbone alternativo, ajuste fino completo
    _ft("siglip2_b16_224", "r2", 50),
    _ft("siglip2_b16_224", "r3", 22),
    _ridge("dinov2_l14_336x448", 16),
    _ridge("siglip2_l16_384x512", 16),
    _ridge("siglip2_so400m16_384x512", 25),
    _ft("siglip2_b16_384x512", "r2", 140, epocas=16, **_ALTA),
    _ft("siglip2_b16_224", "r1", 35, sufixo=" sem-escala", escala=None),   # ablação: só orientações, sem escala aleatória
    _ft("dinov2_l14_224x308", "r1", 125, epocas=10, lld=0.85, bs=16, acum=2, ckpt_grad=True),
    _ft("siglip2_b16_224", "r1", 75, tipo="destila-rgbd"),          # professor RGB-D -> aluno RGB (opcional, por último)
]
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
    return ["tudo"] if cfg["tipo"] == "ridge" else list(PARTES_FT)


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
    assert len({slug(c["nome"]) for c in CONFIGS}) == len(CONFIGS) and all(c["bb"] in BACKBONES for c in CONFIGS + CONFIGS_SMOKE)
    print("autoteste: ok")


# ------------------------------------------------------------------ dados (listas do repositório; sem gabarito de avaliação)
class Dados:
    """Treino zenital estrito (ids, folds por cadeia, alvos em log) + ids de avaliação (sem rótulo)."""

    def __init__(self, pasta, smoke=0):
        arq_tr, arq_av = os.path.join(pasta, "treino.csv"), os.path.join(pasta, "aval_ids.csv")
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
        # mesmos alvos de zen_comum.alvos: log m | log1p p | log1p kcal | log(p/m + eps) | log1p gordura | log1p carbo | grupos
        self.T = np.c_[np.log(m), np.log1p(p), np.log1p(col("kcal")), np.log(p / m + EPS_D), np.log1p(col("gordura")),
                       np.log1p(col("carbo")), np.log1p(mg), np.log1p(pg)]
        fx = np.digitize(p, FAIXAS)
        self.w_cv = (1 / np.bincount(fx, minlength=5))[fx]; self.w_cv /= self.w_cv.sum()   # 5 faixas com o mesmo peso
        with open(arq_av) as f:
            av = list(csv.DictReader(f))
        self.aval = {s: [r["dish_id"] for r in av if r["split"] == s] for s in SPLITS}
        ev = [d for s in SPLITS for d in self.aval[s]]
        assert len(set(self.ids)) == len(self.ids) and len(set(ev)) == len(ev), "ids repetidos nas listas"
        assert not set(self.ids) & set(ev), "prato de avaliação na lista de treino"
        self.hash = hashlib.sha256(open(arq_tr, "rb").read() + open(arq_av, "rb").read()).hexdigest()[:16]
        # --smoke: poucos pratos por fold e por conjunto; o resto fica NaN nas saídas
        self.usa_tr = np.ones(len(self.ids), bool)
        self.usa_av = {s: np.ones(len(self.aval[s]), bool) for s in SPLITS}
        if smoke:
            self.usa_tr[:] = False
            for k in range(5):
                self.usa_tr[np.where(self.fold == k)[0][:smoke]] = True
            for s in SPLITS:
                self.usa_av[s][:] = False; self.usa_av[s][:4] = True
        self.todos = [d for d, u in zip(self.ids, self.usa_tr) if u] + [d for s in SPLITS for d, u in zip(self.aval[s], self.usa_av[s]) if u]
        self.chave = hashlib.sha256("\n".join(self.todos).encode()).hexdigest()[:10]


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


def feats_congeladas(bb, D, C, dev, bs, prazo):
    """Features do backbone congelado para todos os pratos, 8 orientações: [n_todos, 8, 2·dim] float16 (cache em disco).
    É a entrada do ridge e do ponto de partida da cabeça no ajuste fino."""
    arq = os.path.join(C.pasta, f"feat__{bb}.npy")
    with Trava(arq + ".trava"):
        if not os.path.exists(arq):
            _extrai_feats(arq, bb, D, C, dev, bs, prazo)
    return np.load(arq)


def _extrai_feats(arq, bb, D, C, dev, bs, prazo):
    import torch
    t0 = time.time()
    rede = cria_rede(bb, D.N).to(dev).eval()
    hw = BACKBONES[bb]["hw"]
    X = np.zeros((len(D.todos), 8, rede.dim), np.float16)
    with torch.inference_mode():
        for i in range(0, len(D.todos), bs):
            if time.time() > prazo:
                raise SemOrcamento()
            x = monta_lote(C, list(range(i, min(i + bs, len(D.todos)))), dev, hw)
            for k in range(8):
                with autocast(dev):
                    f = rede.feats(orienta(x, k))
                X[i:i + len(x), k] = f.float().cpu().numpy()
    tmp = f"{arq}.{os.getpid()}.npy"
    np.save(tmp, X); os.replace(tmp, arq)
    print(f"features congeladas {bb}: {X.shape} em {time.time() - t0:.0f}s ({len(D.todos) * 8 / (time.time() - t0):.0f} img/s)", flush=True)


def pesos_perda(D, n_saidas):
    w = np.ones(n_saidas, np.float32)   # mesmos pesos por alvo do ajuste fino da CPU (zen_finetune.py)
    w[I_K] = 0.5; w[I_F] = 0.3; w[I_C] = 0.3; w[D.I_MG] = 0.15; w[D.I_PG] = 0.15
    return w


def treina(cfg, D, C, tr, va, dev, prazo, div_lote=1, canais=3, professor=None, fator_orc=1.0, semente=0, rotulo=""):
    """Ajuste fino completo. tr/va: índices em D.ids (va só para registrar a curva de validação; não decide nada).
    Devolve (rede com os pesos da média móvel, função de previsão, histórico por época, (média, desvio) dos alvos)."""
    import copy, torch
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    hw = BACKBONES[cfg["bb"]]["hw"]
    quad = hw[0] == hw[1]
    bs = max(1, min(cfg["bs"] // div_lote, len(tr))); acum = cfg["acum"] * div_lote
    T = D.T
    tm, ts = T[tr].mean(0), T[tr].std(0) + 1e-6
    # cabeça iniciada na solução do ridge sobre as features congeladas dos mesmos pratos (alfa por CV interna):
    # o ajuste fino parte do melhor modelo congelado, como na CPU
    Xc = feats_congeladas(cfg["bb"], D, C, dev, max(1, cfg["bs"] // div_lote), prazo)
    Xr = Xc[[C.pos[D.ids[i]] for i in tr]].astype(np.float32)
    _, alfa, _ = ridge_cv(Xr, T[tr], D.fold[tr])
    _, m = ridge_final(Xr, T[tr], alfa)
    del Xc, Xr
    rede = cria_rede(cfg["bb"], D.N, drop_path=cfg["drop_path"], ckpt_grad=cfg["ckpt_grad"], canais=canais).to(dev)
    with torch.no_grad():
        rede.head.weight.copy_(torch.from_numpy((((m.W / m.sd[:, None]) / ts).T).astype(np.float32)))
        rede.head.bias.copy_(torch.from_numpy(((m.mt - (m.mu / m.sd) @ m.W - tm) / ts).astype(np.float32)))
    ema = copy.deepcopy(rede).eval()
    ema.m.set_grad_checkpointing(False)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(grupos_lr(rede, cfg["lr"], cfg["lld"], cfg["wd"]))
    passos_ep = max(1, len(tr) // (bs * acum)); total = cfg["epocas"] * passos_ep
    n_aq = max(1, int(cfg["aquec"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / n_aq if s < n_aq else 0.5 * (1 + math.cos(math.pi * (s - n_aq) / max(1, total - n_aq))))
    try:
        scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    w = torch.from_numpy(pesos_perda(D, D.N)).to(dev)
    Yn = torch.from_numpy(((T - tm) / ts).astype(np.float32)).to(dev)
    tm_t, ts_t = torch.from_numpy(tm.astype(np.float32)).to(dev), torch.from_numpy(ts.astype(np.float32)).to(dev)
    huber = torch.nn.functional.smooth_l1_loss
    linha = np.array([C.pos.get(d, -1) for d in D.ids])   # -1: prato fora do subconjunto do --smoke

    def preve(ids, n_or=8):
        """Média das previsões (em log) nas n_or orientações, com os pesos da média móvel."""
        out = np.zeros((len(ids), D.N), np.float32)
        with torch.inference_mode():
            for i in range(0, len(ids), bs):
                x = monta_lote(C, [C.pos[d] for d in ids[i:i + bs]], dev, hw, canais)
                soma = 0
                for k in range(n_or):
                    with autocast(dev):
                        soma = soma + ema(orienta(x, k))[0].float()
                out[i:i + len(x)] = (soma / n_or * ts_t + tm_t).cpu().numpy()
        return out

    def valida():
        pv = preve([D.ids[i] for i in va], 1); P = proteinas(pv); wv = D.w_cv[va]
        cr = lambda est: float(100 * (acerto(D.proteina[va], est) * wv).sum() / wv.sum())
        return dict(val_rmse_logm=float(np.sqrt(np.mean((pv[:, I_M] - T[va, I_M]) ** 2))), val_crit_direto=cr(P["direto"]), val_crit_mxd=cr(P["mxd"]))

    hist, passo, t0 = [], 0, time.time()
    if va is not None and len(va):
        hist.append(dict(epoca=0, **valida()))
        print(f"{rotulo} época 0 (cabeça = ridge): {hist[-1]}", flush=True)
    t_treino = time.time()
    for ep in range(cfg["epocas"]):
        rede.train()
        perm = rng.permutation(len(tr)); perdas = []
        for b in range(passos_ep):
            for a in range(acum):
                j = tr[perm[(b * acum + a) * bs:(b * acum + a + 1) * bs]]
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
                    perda = (huber(out.float(), Yn[jt], reduction="none") * w).mean()
                    if professor is not None:   # destilação: saídas e features do professor RGB-D na mesma vista
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
        if va is not None and len(va):
            h.update(valida())
        hist.append(h)
        print(f"{rotulo} {h}", flush=True)
    return ema, preve, hist, (tm, ts)


# ------------------------------------------------------------------ uma parte (processo filho)
def arq_parte(saida, cfg, parte):
    return os.path.join(saida, f"parte__{slug(cfg['nome'])}__{parte}.npz")


def roda_parte(a, cfg, parte, cfgs=()):
    global PASTA_PESOS
    import torch
    t0 = time.time()
    PASTA_PESOS = os.path.join(a.cache, "pesos")
    os.environ.setdefault("HF_HOME", os.path.join(a.cache, "hf"))     # pesos do Hugging Face fora da pasta de saída
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    dev = torch.device(a.dispositivo)
    D = Dados(a.listas, a.smoke)
    destila = cfg["tipo"] == "destila-rgbd"
    C = Cache(a.cache, D).abre(profundidade=destila)
    prazo = a.prazo or float("inf")
    rot = f"[{cfg['nome']} | {parte}]"
    tr_tudo = np.where(D.usa_tr)[0]
    nan = lambda n: np.full((n, D.N), np.nan, np.float32)
    saida = dict(cfg=json.dumps(cfg), parte=parte, dispositivo=str(torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))
    if cfg["tipo"] == "ridge":
        X = feats_congeladas(cfg["bb"], D, C, dev, cfg["bs"], prazo).astype(np.float32)
        Xtr = X[[C.pos[D.ids[i]] for i in tr_tudo]]
        oof = nan(len(D.ids))
        oof[tr_tudo], alfa, mse = ridge_cv(Xtr, D.T[tr_tudo], D.fold[tr_tudo])
        pred, _ = ridge_final(Xtr, D.T[tr_tudo], alfa)
        saida.update(oof=oof, alfa=alfa, cv_mse=mse)
        for s in SPLITS:
            p = nan(len(D.aval[s])); u = np.where(D.usa_av[s])[0]
            p[u] = pred(X[[C.pos[D.aval[s][i]] for i in u]])
            saida[s] = p
        hist = []
        url = BACKBONES[cfg["bb"]].get("url")       # checkpoint grande que nenhuma outra configuração usa: libera o disco
        arq = os.path.join(PASTA_PESOS, os.path.basename(url or "-"))
        if url and os.path.exists(arq) and sum(BACKBONES[c["bb"]].get("url") == url for c in cfgs) == 1:
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
            for s in SPLITS:
                p = nan(len(D.aval[s])); u = np.where(D.usa_av[s])[0]
                p[u] = preve([D.aval[s][i] for i in u])
                saida[s] = p
            if a.guarda_pesos != "nenhum":   # pesos da média móvel em meia precisão (ficam fora do git)
                arq = os.path.join(a.saida, f"pesos__{slug(cfg['nome'])}.pt")
                torch.save(dict(formato=FORMATO, cfg=cfg, timm=BACKBONES[cfg["bb"]], tm=tm, ts=ts, grupos=D.grupos,
                                estado={n: (v.half() if v.is_floating_point() else v).cpu() for n, v in rede.state_dict().items()}), arq + ".tmp")
                os.replace(arq + ".tmp", arq)
    for k in ["oof"] + SPLITS:      # só previsões finitas viram arquivo (NaN é reservado a pratos fora da parte)
        if k in saida and not np.isfinite(saida[k][~np.isnan(saida[k][:, 0])]).all():
            raise RuntimeError(f"previsões não finitas em '{k}'")
    saida.update(hist=json.dumps(hist), seg=time.time() - t0)
    if dev.type == "cuda":
        saida["mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    dest = arq_parte(a.saida, cfg, parte)
    np.savez(dest + ".tmp.npz", **saida); os.replace(dest + ".tmp.npz", dest)
    print(f"{rot} pronto em {(time.time() - t0) / 60:.1f} min", flush=True)


def principal_parte(a, cfgs):
    nome, parte = a.job.rsplit(":", 1)
    cfg = next(c for c in cfgs if slug(c["nome"]) == nome)
    simula = os.environ.get("NUTRELLE_SIMULA", "")     # só para testar o orquestrador: "memoria" ou "erro"
    if simula == "memoria" and a.div_lote == 1 and parte in ("fold0", "tudo"):
        sys.exit(SAI_MEMORIA)
    if simula == "erro" and parte == "fold1":
        sys.exit(1)
    try:
        roda_parte(a, cfg, parte, cfgs)
    except SemOrcamento:
        sys.exit(SAI_SEM_ORCAMENTO)
    except Exception as e:
        traceback.print_exc()
        sys.exit(SAI_MEMORIA if "out of memory" in str(e).lower() else 1)


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
        return sorted({BACKBONES[c["bb"]]["timm"] for c in cfgs} - tem), timm.__version__
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
    """Copia partes prontas de execuções anteriores (mesmas listas) para a pasta de saída."""
    fontes = list(a.retoma)
    if os.path.isdir("/kaggle/input"):
        fontes += [os.path.dirname(m) for m in glob.glob("/kaggle/input/**/manifesto.json", recursive=True)]
    n = 0
    for pasta in fontes:
        try:
            man = json.load(open(os.path.join(pasta, "manifesto.json")))
        except Exception:
            continue
        if man.get("formato") != FORMATO or man.get("listas_hash") != a._hash or bool(man.get("smoke")) != bool(a.smoke):
            diz(f"retomada: {pasta} ignorada (outro formato, outras listas ou smoke)")
            continue
        for arq in glob.glob(os.path.join(pasta, "parte__*.npz")) + glob.glob(os.path.join(pasta, "pesos__*.pt")):
            dest = os.path.join(a.saida, os.path.basename(arq))
            if not os.path.exists(dest):
                shutil.copy(arq, dest); n += 1
    if n:
        diz(f"retomada: {n} arquivos copiados de {fontes}")


def junta(a, cfg, D):
    """Reúne as partes de uma configuração completa: pred__*.npz (esquema do banco), pred__*.csv e log__*.json."""
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
    pred = dict(oof=oof, **{sp: av[sp] for sp in SPLITS})
    np.savez(os.path.join(a.saida, f"pred__{s}.tmp.npz"), nome=cfg["nome"], listas_hash=D.hash, fold=D.fold,
             ids_treino=np.array(D.ids), **{f"ids_{sp}": np.array(D.aval[sp]) for sp in SPLITS}, **pred)
    os.replace(os.path.join(a.saida, f"pred__{s}.tmp.npz"), os.path.join(a.saida, f"pred__{s}.npz"))
    cols = ["log_massa", "log1p_proteina", "log1p_kcal", "log_densidade", "log1p_gordura", "log1p_carbo"] \
        + [f"log1p_m_{g}" for g in D.grupos] + [f"log1p_p_{g}" for g in D.grupos]
    with open(os.path.join(a.saida, f"pred__{s}.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["conjunto", "dish_id", "fold"] + cols)
        for conj, ids, M, folds in [("treino_oof", D.ids, oof, D.fold)] + [(sp, D.aval[sp], av[sp], None) for sp in SPLITS]:
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
    D = Dados(a.listas, a.smoke)
    a._hash = D.hash
    diz(f"{FORMATO} | treino {int(D.usa_tr.sum())} pratos, avaliação {sum(int(u.sum()) for u in D.usa_av.values())} | listas {D.hash}"
        f" | orçamento {a.orcamento_h} h | saída {a.saida} | cache {a.cache}")
    garante_timm(cfgs)
    import torch
    n_gpu = torch.cuda.device_count()
    disp = [f"cuda:{i}" for i in range(n_gpu)] if n_gpu and not a.cpu else ["cpu"] * a.workers
    try:      # nomes pelo nvidia-smi: perguntar ao torch abriria um contexto CUDA (memória de GPU) neste processo
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception:
        gpus = []
    gpus = gpus if len(gpus) == n_gpu else [f"cuda:{i}" for i in range(n_gpu)]
    diz(f"torch {torch.__version__} | dispositivos: {gpus or 'CPU'} -> {len(disp)} processo(s) em paralelo")
    if not n_gpu and not (a.smoke or a.cpu):
        raise SystemExit("Sem GPU. No Kaggle: Settings > Accelerator > GPU T4 x2. (Para testar em CPU: --smoke.)")
    retoma(a)
    estado = {slug(c["nome"]): {p: parte_inicial() for p in partes_de(c)} for c in cfgs}
    trava = threading.Lock()
    info = dict(formato=FORMATO, smoke=bool(a.smoke), listas_hash=D.hash, n_treino=int(D.usa_tr.sum()), inicio=time.strftime("%Y-%m-%d %H:%M:%S"),
                orcamento_h=a.orcamento_h, gpus=gpus, torch=torch.__version__, argv=sys.argv[1:], configs={}, melhor_modelo=None)

    def evento(cfg, parte, **ev):     # único lugar em que o estado muda (sempre com a trava)
        s = slug(cfg["nome"])
        estado[s][parte] = transicao(estado[s][parte], ev)
        with open(os.path.join(a.saida, "eventos.jsonl"), "a") as f:
            f.write(json.dumps(dict(t=round(time.time(), 1), config=cfg["nome"], parte=parte, **ev, estado=estado[s][parte]["estado"]), ensure_ascii=False) + "\n")

    def manifesto():
        for c in cfgs:
            s = slug(c["nome"])
            info["configs"].setdefault(c["nome"], dict(slug=s, tipo=c["tipo"], bb=c["bb"], cv_treino=None, min=None))
            info["configs"][c["nome"]].update(estado=estado_config(estado[s]), partes=estado[s])
        info["duracao_min"] = round((time.time() - t_ini) / 60, 1)
        grava_json(os.path.join(a.saida, "manifesto.json"), info)

    def fecha_config(c):
        """Configuração completa: junta as partes, calcula a CV no treino e mantém só os pesos do melhor modelo."""
        s = slug(c["nome"])
        cv = junta(a, c, D)
        ic = info["configs"][c["nome"]]
        ic["cv_treino"] = cv; ic["min"] = round(sum(p["seg"] or 0 for p in estado[s].values()) / 60, 1)
        diz(f"== {c['nome']}: CV no treino (n={cv['n']}) RMSE log-massa {cv['rmse_logm']:.4f} | MAPE massa {cv['mape_massa']:.1f}% | "
            f"critério ponderado direto {cv['crit_direto']:.1f}% m×d {cv['crit_mxd']:.1f}% geo {cv['crit_geo']:.1f}% | {ic['min']} min de GPU")
        nota = lambda n: max(info["configs"][n]["cv_treino"][r] for r in ("crit_direto", "crit_mxd", "crit_geo"))
        com_pesos = [x["nome"] for x in cfgs if info["configs"][x["nome"]]["cv_treino"] and os.path.exists(os.path.join(a.saida, f"pesos__{slug(x['nome'])}.pt"))]
        if com_pesos:
            melhor = max(com_pesos, key=nota)
            info["melhor_modelo"] = dict(nome=melhor, arquivo=f"pesos__{slug(melhor)}.pt", criterio_cv=nota(melhor),
                                         regra="maior critério ponderado na CV do treino entre as rotas direto, m×d e geo")
            if a.guarda_pesos == "melhor":
                for n in com_pesos:
                    if n != melhor:
                        os.remove(os.path.join(a.saida, f"pesos__{slug(n)}.pt"))

    with trava:
        for c in cfgs:                      # partes já em disco (retomada) -> FEITA
            for p in partes_de(c):
                if os.path.exists(arq_parte(a.saida, c, p)):
                    evento(c, p, tipo="retomou", seg=float(np.load(arq_parte(a.saida, c, p))["seg"]))
        manifesto()
        for c in cfgs:
            if estado_config(estado[slug(c["nome"])]) == "completa":
                fecha_config(c)
        manifesto()
    pendentes = [c for c in cfgs if estado_config(estado[slug(c["nome"])]) != "completa"]
    if pendentes:
        Cache(a.cache, D).prepara(profundidade=any(c["tipo"] == "destila-rgbd" for c in pendentes))

    def proxima(dev):
        """Próxima parte pendente, na ordem de prioridade. Parte que não cabe no tempo restante vira sem_orcamento."""
        for c in cfgs:
            s = slug(c["nome"])
            for p in partes_de(c):
                e = estado[s][p]
                if e["estado"] != Estado.PENDENTE.value:
                    continue
                feitas = [x["seg"] for x in estado[s].values() if x["estado"] == Estado.FEITA.value and x["seg"] and not x["retomada"]]
                est = float(np.median(feitas)) if feitas else 120.0
                if time.time() + est > prazo:
                    evento(c, p, tipo="sem_orcamento")
                    cancela(c, "sem orçamento para a configuração inteira")
                    continue
                evento(c, p, tipo="iniciou", dispositivo=dev)
                return c, p
        return None

    def cancela(c, motivo):
        for p, e in estado[slug(c["nome"])].items():
            if e["estado"] == Estado.PENDENTE.value:
                evento(c, p, tipo="cancelou", motivo=motivo)

    def trabalhador(i, dev):
        while True:
            with trava:
                job = proxima(dev); manifesto()
            if job is None:
                return
            c, p = job
            e = estado[slug(c["nome"])][p]
            diz(f"[{dev}] começa {c['nome']} | {p} (tentativa {e['tentativa']}, lote ÷ {e['div_lote']}); restam {(prazo - time.time()) / 60:.0f} min")
            env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(max(1, (os.cpu_count() or 2) // len(disp)) if not a.smoke else 1))
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--job", f"{slug(c['nome'])}:{p}", "--prazo", str(prazo), "--div-lote", str(e["div_lote"]),
                   "--saida", a.saida, "--cache", a.cache, "--listas", a.listas, "--guarda-pesos", a.guarda_pesos]
            if a.smoke:
                cmd += ["--smoke", str(a.smoke)]
            if dev.startswith("cuda"):
                env["CUDA_VISIBLE_DEVICES"] = dev.split(":")[1]; cmd += ["--dispositivo", "cuda:0"]
            else:
                cmd += ["--dispositivo", "cpu"]
            t0 = time.time()
            pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            cauda = []
            for linha in pr.stdout:
                diz(f"[{dev}] {linha.rstrip()}"); cauda = (cauda + [linha.rstrip()])[-8:]
            rc = pr.wait()
            with trava:
                if rc == SAI_OK and os.path.exists(arq_parte(a.saida, c, p)):
                    evento(c, p, tipo="concluiu", seg=time.time() - t0)
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
    with trava:
        manifesto()
    diz("\n" + "\n".join(f"  {ic['estado']:14s} {n:44s} " + (f"RMSE {ic['cv_treino']['rmse_logm']:.4f} crit {ic['cv_treino']['crit_direto']:.1f}/{ic['cv_treino']['crit_mxd']:.1f}/{ic['cv_treino']['crit_geo']:.1f}% {ic['min']} min"
                                                                  if ic["cv_treino"] else "") for n, ic in info["configs"].items()))
    diz(f"melhor modelo único (pesos gravados): {info['melhor_modelo']}")
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
    ap.add_argument("--guarda-pesos", choices=["melhor", "todos", "nenhum"], default="melhor")
    ap.add_argument("--smoke", type=int, nargs="?", const=8, default=0, help="teste rápido: N pratos por fold (padrão 8), backbone minúsculo, CPU")
    ap.add_argument("--workers", type=int, default=1, help="processos em paralelo quando não há GPU (--smoke/--cpu)")
    ap.add_argument("--cpu", action="store_true", help="ignora a GPU")
    ap.add_argument("--lista", action="store_true"); ap.add_argument("--checa-modelos", action="store_true"); ap.add_argument("--autoteste", action="store_true")
    ap.add_argument("--job"); ap.add_argument("--dispositivo", default="cpu"); ap.add_argument("--prazo", type=float); ap.add_argument("--div-lote", type=int, default=1)
    a = ap.parse_args()
    a.saida, a.cache, a.listas = (os.path.abspath(p) for p in (a.saida, a.cache, a.listas))
    cfgs = CONFIGS_SMOKE if a.smoke else CONFIGS
    if a.so:
        cfgs = [c for c in cfgs if any(t in c["nome"] for t in a.so)]
    if a.autoteste:
        return autoteste()
    if a.checa_modelos:
        return checa_modelos()
    if a.lista:
        for c in cfgs:
            print(f"{c['nome']:46s} {c['tipo']:13s} {BACKBONES[c['bb']]['timm']:44s} {BACKBONES[c['bb']]['hw']} ~{c['est_min']} min numa T4 "
                  + (f"| lr {c['lr']:g} lld {c['lld']} épocas {c['epocas']} lote {c['bs']}×{c['acum']} escala {c['escala']}" if c["tipo"] != "ridge" else ""))
        return print(f"total estimado: ~{sum(c['est_min'] for c in cfgs) / 60:.1f} h numa T4 (metade do relógio com T4 ×2)")
    if a.job:
        return principal_parte(a, cfgs)
    orquestra(a, cfgs)


if __name__ == "__main__":
    main()
