#!/usr/bin/env python3
"""Oitava execução (16.6 f; versão 8b, a emenda da máscara do visor): o teste de registro DFood-TJ. Só previsões cruas;
NADA é pontuado aqui.

    python3 teste_r8.py --saida <pasta> [--teste <teste_r8_dfoodtj.csv>] [--fontes <pasta> ...] [--cache <pasta>]
                        [--espelho-fotos <pasta>] [--imagens <pasta com os PNG>] [--smoke N --sem-detector --backbone <bb>]
                        [--so-deteccao] [--autoteste]

Entrada: `teste_r8_dfoodtj.csv` (dish_id, arquivo, periodo, paciente; SEM rótulo; o sha256 é conferido). O gabarito
nunca entra no pacote nem é lido aqui. As cabeças e os pesos servidos vêm das saídas das execuções anteriores, anexadas
ao kernel (kernel_sources): nutrelle-treino-gpu-r5 (R4c), -r6 (F5c), -r7 (H0c-m e H-cap, 3 sementes; e as features do
recorte do Nutrition5k, para a remontagem), nutrelle-deteccao-r5 (as janelas do Nutrition5k, para a remontagem).

Passos (16.6 f):
 1. confere a tabela (sha256, colunas sem rótulo);
 2. lê do zip do DFood-TJ (china.scidb.cn, 4.247.942.139 bytes) só os 711 PNG da tabela, por intervalo de bytes, em até 14
    conexões; confere o CRC (zipfile), a assinatura PNG e o bloco IEND, e grava o sha256 de cada foto;
 3. em cada foto (orientação do EXIF, retrato girado como no treino), na ordem da 8b: (a) na foto SEM máscara, o prato pela
    regra só de prato e tigela da etapa 0 (mesmo detector e revisão, 960 px, nota 0,15, área 2%, NMS 0,5; elipse e
    `sem_prato` de reserva) e a janela do guia (4:3, maior lado da caixa a 80%); (b) o OWLv2 procura só o mostrador do visor
    ("a digital display", "an LCD screen"; nota >= 0,15); descarta a caixa com mais de 5% da área da foto ou com mais de
    10% da própria área dentro da caixa do prato; (c) pinta as caixas que ficam com a cor mediana da borda, nunca dentro
    da caixa do prato; (d) recorta a janela e a foto inteira dessa mesma foto mascarada. Na 8 (v3), a consulta "a kitchen
    scale display" pegava a balança inteira, sob o prato, e o prato era procurado depois da máscara (16.6 h–k e 16.7);
 4. folhas de contato na saída: todas as janelas mascaradas (30 por folha, num tamanho em que se lê um visor) e 40 por
    período com as caixas desenhadas; CSVs com a posição -> dish_id e `veredito` vazio;
 5. features do So400m (o mesmo checkpoint) nas janelas, 8 orientações; saídas da R4c, da H0c-m e da H-cap (3 sementes,
    com a log-massa da R4c como entrada, como no serviço) por orientação, e do F5c (1 orientação). A foto inteira
    (mascarada, encaixada em 4:3) sai igual, como variante de relato. As rotas e as métricas são da CPU;
 6. remontagem de cada cabeça: (a) sem GPU, sobre as features do recorte do Nutrition5k gravadas pela sétima execução,
    contra as previsões gravadas (conjunto `remontagem`); (b) da foto, nos mesmos 8 pratos do Nutrition5k (baixados e
    recortados como no treino), e também pelo caminho usado aqui no DFood-TJ (recorte na CPU), para medir a diferença;
 7. registro: versões, hashes, tempos e contagens (recorte, elipse, sem_prato, visor mascarado)."""
import argparse, csv, hashlib, io, json, math, os, sys, time, traceback, zipfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import treino_gpu as G  # noqa: E402
import deteccao_r5 as E  # noqa: E402

FORMATO = "nutrelle-teste-dfoodtj/8b"
URL_ZIP = "https://china.scidb.cn/download?fileId=cb2751a22a516c057913e15b7d3cc5c4"
BYTES_ZIP, MD5_ZIP = 4247942139, "b67cc4116e2c734296fbd6cd99cd50bc"      # o md5 é o do depósito; o zip inteiro nunca é baixado
SHA_TESTE = "534a462746d8a55bfac9527eafe91550140d848acb66db7246dfcc41dce5ef5d"
COLS_TESTE = ["dish_id", "arquivo", "periodo", "paciente"]
# a tabela das fotos (sem rótulo): o CRC-32 e o tamanho do diretório central do zip e o sha256 de cada membro, como baixados
# e conferidos fora do Kaggle; confere as fotos do espelho anexado (dataset privado) e as que vierem pela rede
COLS_FOTOS = ["dish_id", "arquivo", "nome", "crc32", "bytes", "sha256", "classe"]
SHA_FOTOS = "29c7b453d1eaaac27c93968595efe7f874ffac6074e3866051d5391364a7aafb"
CONSULTAS_VISOR = ["a digital display", "an LCD screen"]   # 8b: só o mostrador ("a kitchen scale display" pegava a balança inteira)
NOTA_VISOR = 0.15
AREA_MAX_VISOR = 0.05           # 8b: caixa do visor com mais de 5% da área da foto é descartada
SOBREPOSICAO_MAX_PRATO = 0.10   # 8b: ... e a com mais de 10% da própria área dentro da caixa do prato
CONEXOES = 14
ASSINATURA_PNG, FIM_PNG = b"\x89PNG\r\n\x1a\n", b"\x00\x00\x00\x00IEND\xaeB`\x82"
ARQS = dict(r4c="cabeca__gpu5_ridge_so400m_prato__tudo.npz", r4c7="cabeca__gpu7_ridge_so400m_prato__tudo.npz",
            hm="cabeca__gpu7_fracoes_so400m_prato__massa.npz", hcap="cabeca__gpu7_fracoes_so400m_prato__massa_cap.npz",
            f5c="pesos__gpu6_ft_siglip2_b16_224_r2_fracoes_sg_prato__tudo.pt", f5c_sg="cabecasg__gpu6_ft_siglip2_b16_224_r2_fracoes_sg_prato__tudo.npz",
            pred_r4c="pred__gpu5_ridge_so400m_prato__tudo.npz", pred_r4c7="pred__gpu7_ridge_so400m_prato__tudo.npz",
            pred_hm="pred__gpu7_fracoes_so400m_prato__massa.npz", pred_hcap="pred__gpu7_fracoes_so400m_prato__massa_cap.npz",
            pred_f5c="pred__gpu6_ft_siglip2_b16_224_r2_fracoes_sg_prato__tudo.npz",
            feat7="feat7__n5k__recorte.npy", feat7_ids="feat7__n5k__recorte__ids.csv", caixas_n5k="caixas_r5_n5k.csv")
OBRIGATORIOS = ("r4c", "hm", "hcap", "f5c")
TOL_REMONTA = 0.05      # diferença máxima, em log, para a remontagem "conferir" (foto -> previsão passa por fp16 e interpolação)
REGISTRO = []


def diz(*p):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in p)
    print(s, flush=True)
    REGISTRO.append(s)


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def acha(nome, raizes, prof=6):
    """Primeiro arquivo com esse nome nas pastas dadas e em /kaggle/input (saídas anexadas como kernel_sources)."""
    for raiz in [r for r in raizes if r] + ["/kaggle/input"]:
        if not os.path.isdir(raiz):
            continue
        for base, dirs, arqs in os.walk(raiz):
            if base[len(raiz):].count(os.sep) >= prof:
                dirs[:] = []
            if nome in arqs:
                return os.path.join(base, nome)
    return None


# ------------------------------------------------------------------ 1. a tabela
def le_teste(arq, smoke=0):
    """Linhas da tabela do teste (sem rótulo): confere o sha256 e as colunas. Função pura sobre o arquivo."""
    bruto = open(arq, "rb").read()
    sha = hashlib.sha256(bruto).hexdigest()
    L = list(csv.DictReader(io.StringIO(bruto.decode("utf-8"))))
    assert L and list(L[0]) == COLS_TESTE, f"colunas da tabela do teste: {list(L[0]) if L else None} (esperado {COLS_TESTE}; nenhuma de rótulo)"
    assert len({r['dish_id'] for r in L}) == len(L) and len({r['arquivo'] for r in L}) == len(L), "linhas repetidas"
    if not smoke:
        assert sha == SHA_TESTE, f"tabela do teste diferente da registrada em 16.6 (sha256 {sha})"
    return (L[:smoke] if smoke else L), sha


# ------------------------------------------------------------------ 2. as fotos, por intervalo de bytes
def png_ok(dados):
    """Assinatura PNG no começo e o bloco IEND no fim. Função pura."""
    return dados[:8] == ASSINATURA_PNG and dados[-12:] == FIM_PNG


def pede(url, ini, fim, tentativas=8):
    """Bytes [ini, fim) por HTTP Range, com novas tentativas e espera crescente; resposta curta ou sem 206 é pedida de
    novo inteira (nunca se completa um pedaço com outro)."""
    import urllib.request
    erro = None
    for t in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib", "Range": f"bytes={ini}-{fim - 1}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status != 206:
                    raise IOError(f"sem 206 (status {r.status})")
                b = r.read()
            if len(b) == fim - ini:
                return b, t
            erro = IOError(f"resposta curta: {len(b)} de {fim - ini} bytes")
        except Exception as e:      # noqa: BLE001 — rede: tenta de novo
            erro = e
        time.sleep(min(60, 2 ** t))
    raise IOError(f"bytes {ini}-{fim}: {type(erro).__name__}: {erro}")


def membro(url, info, tentativas_crc=3):
    """Um membro do zip remoto, pelo diretório central (info: ZipInfo): cabeçalho local + dados num pedido só,
    descomprimido aqui e conferido pelo CRC-32 e pelo tamanho do diretório central (a conferência que vale). Devolve
    (bytes, novas tentativas de rede, novas tentativas por CRC)."""
    import struct, zlib
    rede = 0
    for t in range(tentativas_crc):
        b, r_ = pede(url, info.header_offset, min(BYTES_ZIP, info.header_offset + 30 + len(info.filename.encode()) + 1024 + info.compress_size))
        rede += r_
        assert b[:4] == b"PK\x03\x04", "cabeçalho local do zip"
        nl, el = struct.unpack("<HH", b[26:30])
        comp = b[30 + nl + el:30 + nl + el + info.compress_size]
        if len(comp) != info.compress_size:
            continue
        dados = zlib.decompressobj(-15).decompress(comp) if info.compress_type == 8 else comp
        if len(dados) == info.file_size and (zlib.crc32(dados) & 0xFFFFFFFF) == info.CRC:
            return dados, rede, t
    raise IOError(f"CRC ou tamanho diferente do diretório central em {tentativas_crc} leituras (o membro guardado ou a transferência)")


def classifica(dados):
    """O que o membro é, conferido o CRC (função pura): 'png' (assinatura e IEND no fim), ou, se o PIL abre a imagem
    inteira, 'png_sem_iend_no_fim' / 'nao_png_decodificavel' (aceitas, marcadas); senão 'indecodificavel' (excluída)."""
    from PIL import Image
    if png_ok(dados):
        return "png", None
    try:
        im = Image.open(io.BytesIO(dados))
        fmt = im.format
        im.load()
        return ("png_sem_iend_no_fim" if dados[:8] == ASSINATURA_PNG else "nao_png_decodificavel"), dict(formato=fmt, tamanho=list(im.size), modo=im.mode,
                                                                                                       sobra_depois_do_iend=(len(dados) - dados.rfind(b"IEND") - 8) if b"IEND" in dados else None)
    except Exception as e:      # noqa: BLE001
        return "indecodificavel", dict(erro=f"{type(e).__name__}: {e}")


def le_fotos(arq, L):
    """{dish_id: linha} da tabela das fotos, conferida: sha256 registrado, colunas sem rótulo, as mesmas fotos e membros da
    tabela do teste."""
    bruto = open(arq, "rb").read()
    sha = hashlib.sha256(bruto).hexdigest()
    F = list(csv.DictReader(io.StringIO(bruto.decode("utf-8"))))
    assert F and list(F[0]) == COLS_FOTOS, f"colunas da tabela das fotos: {list(F[0]) if F else None}"
    assert SHA_FOTOS is None or sha == SHA_FOTOS, f"tabela das fotos diferente da registrada (sha256 {sha})"
    F = {r["dish_id"]: r for r in F}
    assert all(r["dish_id"] in F and F[r["dish_id"]]["arquivo"] == r["arquivo"] for r in L), "a tabela das fotos não cobre a tabela do teste"
    return F, sha


def acha_espelho(a, F):
    """A pasta do espelho das fotos: --espelho-fotos, ou a primeira pasta em /kaggle/input que tem a foto da primeira linha."""
    if a.espelho_fotos:
        return a.espelho_fotos
    nome = next(iter(F.values()))["nome"]
    p = acha(nome, [], prof=5)
    return os.path.dirname(p) if p else None


def confere_arquivo(dados, f):
    """O arquivo é o membro do zip registrado? Tamanho, CRC-32 e sha256 da tabela das fotos. Função pura."""
    import zlib
    return len(dados) == int(f["bytes"]) and f"{zlib.crc32(dados) & 0xFFFFFFFF:08x}" == f["crc32"] and hashlib.sha256(dados).hexdigest() == f["sha256"]


def baixa_pngs(a, L):
    """(L aceitas, {dish_id: dict(caminho, sha256, bytes, classe)}, informações): as fotos da tabela, do zip remoto (ou de
    --imagens). A conferência que vale é a do CRC-32 do diretório central; um membro que não confere depois das novas
    tentativas, ou que o PIL não decodifica, sai da execução e é listado (exclusão sem olhar rótulo); acima de 5% de
    excluídas, a execução para."""
    pasta = os.path.join(a.cache, "img8")
    os.makedirs(pasta, exist_ok=True)
    out, t0, excl, cont = {}, time.time(), {}, {}
    dest = lambda r: os.path.join(pasta, os.path.basename(r["arquivo"]))
    if a.imagens:
        for r in L:
            p = os.path.join(a.imagens, os.path.basename(r["arquivo"]))
            assert os.path.exists(p), f"{p} não existe"
            dados = open(p, "rb").read()
            classe, det = classifica(dados)
            if classe == "indecodificavel":
                excl[r["dish_id"]] = dict(arquivo=r["arquivo"], motivo=classe, **det)
                continue
            out[r["dish_id"]] = dict(caminho=p, sha256=hashlib.sha256(dados).hexdigest(), bytes=len(dados), classe=classe, **({"detalhe": det} if det else {}))
        info = dict(origem=a.imagens, seg=round(time.time() - t0, 1))
    else:
        arq_f = a.fotos or os.path.join(AQUI, "listas", "fotos_r8_dfoodtj.csv")
        F, sha_f = le_fotos(arq_f, L) if os.path.exists(arq_f) else (None, None)
        esp = acha_espelho(a, F) if F else None
        do_espelho, da_rede = {}, []
        for r in L:         # 1. o espelho (dataset anexado): cada arquivo conferido contra a tabela das fotos
            p = os.path.join(esp, F[r["dish_id"]]["nome"]) if esp else None
            if p and os.path.exists(p):
                dados = open(p, "rb").read()
                if confere_arquivo(dados, F[r["dish_id"]]):
                    do_espelho[r["dish_id"]] = (p, dados)
                    continue
                diz(f"espelho: {p} não confere com a tabela das fotos; vai pela rede")
            da_rede.append(r)
        if esp:
            diz(f"espelho das fotos em {esp}: {len(do_espelho)} de {len(L)} conferidas (tamanho, CRC-32, sha256); {len(da_rede)} pela rede")
        infos = {}
        if da_rede:         # 2. só o que faltar vai pela rede, pelo zip remoto
            fz = G.ArquivoHTTP(URL_ZIP)
            assert fz.tamanho == BYTES_ZIP, f"o zip remoto tem {fz.tamanho} bytes, não {BYTES_ZIP}"
            infos = {i.filename: i for i in zipfile.ZipFile(fz).infolist()}
            faltam_no_zip = [r["arquivo"] for r in da_rede if r["arquivo"] not in infos]
            assert not faltam_no_zip, f"{len(faltam_no_zip)} fotos da tabela não estão no zip (ex.: {faltam_no_zip[0]})"
        tent, trava = dict(rede=0, crc=0), __import__("threading").Lock()

        def um(r):
            if r["dish_id"] in do_espelho:
                return r, do_espelho[r["dish_id"]][1], None
            d = dest(r)
            if os.path.exists(d) and os.path.exists(d + ".ok"):        # já baixada e conferida (retomada)
                return r, open(d, "rb").read(), None
            if time.time() > a.prazo:
                return r, None, "prazo esgotado antes do download"
            try:
                dados, nr, nc = membro(URL_ZIP, infos[r["arquivo"]])
                if F and not confere_arquivo(dados, F[r["dish_id"]]):
                    raise IOError("o membro baixado não é o da tabela das fotos (sha256)")
            except Exception as e:      # noqa: BLE001
                return r, None, f"{type(e).__name__}: {e}"
            with trava:
                tent["rede"] += nr; tent["crc"] += nc
            with open(d + ".parte", "wb") as f:
                f.write(dados)
            os.replace(d + ".parte", d)
            open(d + ".ok", "w").write(hex(infos[r["arquivo"]].CRC))
            return r, dados, None
        with ThreadPoolExecutor(CONEXOES) as ex:
            res = list(ex.map(um, L))
        lidos = 0
        for r, dados, erro in res:
            if erro:
                excl[r["dish_id"]] = dict(arquivo=r["arquivo"], motivo="nao_veio_ou_crc_nao_confere", erro=erro)
                continue
            lidos += len(dados)
            classe, det = classifica(dados)
            if classe == "indecodificavel":
                excl[r["dish_id"]] = dict(arquivo=r["arquivo"], motivo=classe, crc_confere=True, **det)
                continue
            out[r["dish_id"]] = dict(caminho=do_espelho[r["dish_id"]][0] if r["dish_id"] in do_espelho else dest(r), sha256=hashlib.sha256(dados).hexdigest(),
                                     bytes=len(dados), classe=classe, origem="espelho" if r["dish_id"] in do_espelho else "rede",
                                     crc=F[r["dish_id"]]["crc32"] if F else f"{infos[r['arquivo']].CRC:08x}", **({"detalhe": det} if det else {}))
        info = dict(espelho=esp, tabela_fotos=dict(arquivo=os.path.basename(arq_f), sha256=sha_f) if F else None, do_espelho=len(do_espelho), pela_rede=len(da_rede),
                    origem=URL_ZIP, bytes_zip=BYTES_ZIP, md5_zip_registrado=MD5_ZIP, mb=round(lidos / 2 ** 20, 1), novas_tentativas_rede=tent["rede"], releituras_por_crc=tent["crc"], conexoes=CONEXOES, seg=round(time.time() - t0, 1))
    for v in out.values():
        cont[v["classe"]] = cont.get(v["classe"], 0) + 1
    info.update(classes=cont, excluidas=excl, n_excluidas=len(excl), aceitas=len(out))
    diz(f"fotos: {len(out)} aceitas {cont}, {len(excl)} excluídas em {time.time() - t0:.0f}s"
        + (f" (ex.: {next(iter(excl))}: {next(iter(excl.values()))['motivo']})" if excl else "")
        + (f" | {info['do_espelho']} do espelho, {info['pela_rede']} pela rede" if "do_espelho" in info else "")
        + (f" | {info['mb']} MB, {info['novas_tentativas_rede']} novas tentativas de rede, {info['releituras_por_crc']} releituras por CRC" if "mb" in info else ""))
    if len(excl) > 0.05 * len(L):
        raise SystemExit(f"{len(excl)} de {len(L)} fotos excluídas (mais de 5%): a execução para. Ver manifesto_r8.json (download.excluidas).")
    return [r for r in L if r["dish_id"] in out], out, info


# ------------------------------------------------------------------ 3. visor, prato e janela
def _area(c):
    return max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])


def filtra_visor(caixas, notas, consultas, w, h, prato):
    """As caixas do visor com nota >= NOTA_VISOR, cada uma com o destino: `mascarada`, `descartada_area` (mais de
    AREA_MAX_VISOR da foto) ou `descartada_prato` (mais de SOBREPOSICAO_MAX_PRATO da própria área dentro da caixa do prato;
    sem caixa do prato, só a regra da área). As contas usam a caixa recortada à foto. Função pura."""
    out = []
    for c, s, q in zip(caixas, notas, consultas):
        if float(s) < NOTA_VISOR:
            continue
        cc = (max(0.0, float(c[0])), max(0.0, float(c[1])), min(float(w), float(c[2])), min(float(h), float(c[3])))
        a = _area(cc)
        fa = a / float(w * h)
        sp = 0.0
        if prato is not None and a > 0:
            sp = _area((max(cc[0], prato[0]), max(cc[1], prato[1]), min(cc[2], prato[2]), min(cc[3], prato[3]))) / a
        destino = ("descartada_area" if fa > AREA_MAX_VISOR or a <= 0 else "descartada_prato" if sp > SOBREPOSICAO_MAX_PRATO else "mascarada")
        out.append(dict(caixa=[round(float(v), 1) for v in c], nota=round(float(s), 4), consulta=CONSULTAS_VISOR[int(q)], area_foto=round(fa, 4),
                        dentro_do_prato=round(sp, 4), destino=destino))
    return out


def mascara_fora_do_prato(img, caixas, cor, prato):
    """(foto mascarada, mapa bool do que foi pintado): cada caixa pintada com a cor dada, menos os pixels dentro da caixa do
    prato (que nunca são pintados). Função pura."""
    h, w = img.shape[:2]
    pint = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in caixas:
        x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
        x1, y1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
        if x1 > x0 and y1 > y0:
            pint[y0:y1, x0:x1] = True
    if prato is not None:
        x0, y0 = max(0, int(math.floor(prato[0]))), max(0, int(math.floor(prato[1])))
        x1, y1 = min(w, int(math.ceil(prato[2]))), min(h, int(math.ceil(prato[3])))
        pint[y0:y1, x0:x1] = False
    m = img.copy()
    m[pint] = cor
    return m, pint


def fracao_cinza(pint, jan):
    """(fração da janela dentro da foto que foi pintada, fração da janela fora da foto). A janela em pixels como no
    E.recorta. Função pura."""
    h, w = pint.shape
    cx, cy, alt, larg = jan
    x0, y0 = int(math.floor(cx - larg / 2)), int(math.floor(cy - alt / 2))
    x1, y1 = int(math.ceil(cx + larg / 2)), int(math.ceil(cy + alt / 2))
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    dentro = max(0, sx1 - sx0) * max(0, sy1 - sy0)
    total = max(1, (x1 - x0) * (y1 - y0))
    return (float(pint[sy0:sy1, sx0:sx1].sum()) / dentro if dentro else 0.0), 1.0 - dentro / total


def resumo_cinza(v):
    """Distribuição da fração de cinza nas janelas. Função pura."""
    v = np.asarray(v, float)
    if not len(v):
        return dict(n=0)
    q = lambda p: round(float(np.quantile(v, p)), 4)
    return dict(n=int(len(v)), media=round(float(v.mean()), 4), p50=q(0.5), p90=q(0.9), p99=q(0.99), max=round(float(v.max()), 4),
                acima_0=int((v > 0).sum()), acima_5pct=int((v > 0.05).sum()), acima_10pct=int((v > 0.10).sum()), acima_25pct=int((v > 0.25).sum()))


def caixa_prato(l):
    """A caixa do prato de uma linha da tabela (OWLv2 ou elipse), ou None (sem_prato)."""
    return None if l["x0"] == "" else tuple(float(l[c]) for c in ("x0", "y0", "x1", "y1"))


def processa(a, L, fotos, dev):
    """Prato e janela na foto sem máscara; visor (só o mostrador, filtrado) pintado fora do prato; os dois recortes (janela
    do guia e foto inteira) da foto mascarada; grava a tabela de caixas."""
    det = None if a.sem_detector else E.Detector(dev, os.path.join(a.cache, "hf"))
    dv = det.com_consultas(CONSULTAS_VISOR) if det else None
    versao = det.versao if det else "sem detector (só a reserva por elipse; teste)"
    n = len(L)
    rec = np.lib.format.open_memmap(os.path.join(a.cache, "recorte8.npy"), mode="w+", dtype=np.uint8, shape=(n, 480, 640, 3))
    inteira = np.lib.format.open_memmap(os.path.join(a.cache, "inteira8.npy"), mode="w+", dtype=np.uint8, shape=(n, 480, 640, 3))
    linhas, t0 = [], time.time()

    def abre(i):
        img, girada = E.abre_canonica(fotos[L[i]["dish_id"]]["caminho"])
        return i, img, girada
    with ThreadPoolExecutor(4) as ex:
        for j in range(0, n, 8):
            lote = list(ex.map(abre, range(j, min(j + 8, n))))
            vazio = [(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))] * len(lote)
            prato = det([x[1] for x in lote]) if det else vazio        # 1. o prato, na foto SEM máscara
            visor = dv([x[1] for x in lote]) if dv else vazio          # 2. o mostrador do visor
            for (i, img, girada), r_, (cx_, nt, cq) in zip(lote, prato, visor):
                r = L[i]
                l0 = E.linha_deteccao(r["dish_id"], "dfoodtj", r["arquivo"], img, girada, fotos[r["dish_id"]]["sha256"], E.elipse(img), r_, versao, i)
                lr = E.refaz_linha(l0, set(E.SO_PRATO))        # a regra só de prato e tigela (emenda a 14.3)
                h, w = img.shape[:2]
                cp = caixa_prato(lr)
                cv = filtra_visor(cx_, nt, cq, w, h, cp)
                fica = [c for c in cv if c["destino"] == "mascarada"]
                m, pint = mascara_fora_do_prato(img, [c["caixa"] for c in fica], E.cor_borda(img).astype(np.uint8), cp)     # 3. pinta fora do prato
                jan = (float(lr["janela_cx"]), float(lr["janela_cy"]), float(lr["janela_alt"]), float(lr["janela_larg"]))
                cm = E.cor_borda(m)
                rec[i] = E.recorta(m, jan, (640, 480), cm)                 # 4. a janela e a foto inteira, da mesma foto mascarada
                inteira[i] = E.recorta(m, E.janela(None, w, h), (640, 480), cm)
                fc, ff = fracao_cinza(pint, jan)
                linhas.append(dict(lr, periodo=r["periodo"], paciente=r["paciente"], n_visor=len(fica), caixas_visor=json.dumps(fica, separators=(",", ":")),
                                   n_visor_descartadas=len(cv) - len(fica), caixas_visor_descartadas=json.dumps([c for c in cv if c["destino"] != "mascarada"], separators=(",", ":")),
                                   fracao_cinza_janela=round(fc, 4), fracao_fora_da_foto_janela=round(ff, 4), metodo_uniao=l0["metodo"], _i=i))
            if time.time() > a.prazo:
                raise TimeoutError("prazo na detecção")
    rec.flush(); inteira.flush()
    cols = list(E.COLS) + ["periodo", "paciente", "n_visor", "caixas_visor", "n_visor_descartadas", "caixas_visor_descartadas", "fracao_cinza_janela",
                           "fracao_fora_da_foto_janela", "metodo_uniao"]
    with open(os.path.join(a.saida, "caixas_r8_dfoodtj.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, cols, lineterminator="\n", extrasaction="ignore"); wr.writeheader(); wr.writerows(linhas)
    met = {}
    for l in linhas:
        met[l["metodo"]] = met.get(l["metodo"], 0) + 1
    por_per = {p: dict(fotos=sum(1 for l in linhas if l["periodo"] == p), com_visor_mascarado=sum(1 for l in linhas if l["periodo"] == p and l["n_visor"]),
                       metodos={m: sum(1 for l in linhas if l["periodo"] == p and l["metodo"] == m) for m in met},
                       fracao_cinza_janela=resumo_cinza([l["fracao_cinza_janela"] for l in linhas if l["periodo"] == p]))
               for p in sorted({l["periodo"] for l in linhas})}
    todas = [c for l in linhas for c in json.loads(l["caixas_visor"]) + json.loads(l["caixas_visor_descartadas"])]
    caixas = {d: sum(1 for c in todas if c["destino"] == d) for d in ("mascarada", "descartada_area", "descartada_prato")}
    cinza = resumo_cinza([l["fracao_cinza_janela"] for l in linhas])
    diz(f"detecção: {n} fotos em {time.time() - t0:.0f}s | prato (regra só de prato e tigela, na foto sem máscara): {met} | caixas do visor {caixas}; "
        f"visor mascarado em {sum(1 for l in linhas if l['n_visor'])} fotos | cinza na janela: {cinza} | {versao}")
    return linhas, dict(versao_detector=versao, ordem="8b: prato na foto sem máscara -> visor filtrado, pintado fora da caixa do prato -> janela e foto inteira da foto mascarada",
                        consultas_visor=CONSULTAS_VISOR, nota_visor=NOTA_VISOR, area_max_visor=AREA_MAX_VISOR, sobreposicao_max_prato=SOBREPOSICAO_MAX_PRATO,
                        caixas_visor=caixas, fracao_cinza_janela=cinza, consultas_prato=[E.CONSULTAS[k] for k in E.SO_PRATO],
                        metodos=met, com_visor_mascarado=sum(1 for l in linhas if l["n_visor"]), por_periodo=por_per, seg=round(time.time() - t0, 1),
                        sha256_tabela=sha256(os.path.join(a.saida, "caixas_r8_dfoodtj.csv")))


# ------------------------------------------------------------------ 4. folhas de contato
def folhas(a, L, linhas, fotos):
    from PIL import Image, ImageDraw
    rec = np.load(os.path.join(a.cache, "recorte8.npy"), mmap_mode="r")
    W, H, col, por = 400, 300, 5, 30
    tab = []
    for f0 in range(0, len(linhas), por):           # todas as janelas mascaradas, 30 por folha (16.6 a: a conferência do visor)
        parte = list(range(f0, min(f0 + por, len(linhas))))
        folha = Image.new("RGB", (col * (W + 6), math.ceil(len(parte) / col) * (H + 20)), (255, 255, 255))
        dr = ImageDraw.Draw(folha)
        for p, i in enumerate(parte):
            x, y = (p % col) * (W + 6), (p // col) * (H + 20)
            folha.paste(Image.fromarray(np.asarray(rec[i])).resize((W, H), Image.BICUBIC), (x, y))
            num = i + 1
            dr.text((x + 2, y + H + 3), f"{num}. {linhas[i]['id']} {linhas[i]['periodo']} {linhas[i]['metodo']}"
                    f"{' visor' if linhas[i]['n_visor'] else ''} cinza {100 * linhas[i]['fracao_cinza_janela']:.0f}%", fill=(0, 0, 0))
            tab.append(dict(numero=num, folha=f0 // por + 1, dish_id=linhas[i]["id"], periodo=linhas[i]["periodo"], metodo=linhas[i]["metodo"], n_visor=linhas[i]["n_visor"],
                            fracao_cinza_janela=linhas[i]["fracao_cinza_janela"]))
        folha.save(os.path.join(a.saida, f"folha_r8_janelas_{f0 // por + 1:02d}.jpg"), quality=85)
    with open(os.path.join(a.saida, "folha_r8_janelas.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, ["numero", "folha", "dish_id", "periodo", "metodo", "n_visor", "fracao_cinza_janela", "veredito"], lineterminator="\n")
        wr.writeheader(); wr.writerows(dict(t, veredito="") for t in tab)
    rng = np.random.default_rng(0)                   # 40 por período, com as caixas (a conferência de 14.9)
    for per in sorted({l["periodo"] for l in linhas}):
        idx = [i for i, l in enumerate(linhas) if l["periodo"] == per]
        sel = sorted(rng.choice(idx, min(40, len(idx)), replace=False).tolist())
        tab2, w2, h2 = [], 320, 240
        for f0 in range(0, len(sel), 20):
            parte = sel[f0:f0 + 20]
            folha = Image.new("RGB", (4 * (2 * w2 + 8), math.ceil(len(parte) / 4) * (h2 + 20)), (255, 255, 255))
            dr = ImageDraw.Draw(folha)
            for p, i in enumerate(parte):
                l = linhas[i]
                img, _ = E.abre_canonica(fotos[l["id"]]["caminho"])
                cv, cd = json.loads(l["caixas_visor"]), json.loads(l["caixas_visor_descartadas"])
                m = mascara_fora_do_prato(img, [c["caixa"] for c in cv], E.cor_borda(img).astype(np.uint8), caixa_prato(l))[0]
                h, w = m.shape[:2]
                esc = min(w2 / w, h2 / h)
                mini = Image.fromarray(m).resize((max(1, round(w * esc)), max(1, round(h * esc))))
                d2 = ImageDraw.Draw(mini)
                for c in cd:            # descartadas (área ou prato): magenta, fina
                    d2.rectangle([v * esc for v in c["caixa"]], outline=(255, 0, 255), width=1)
                for c in cv:            # pintadas: ciano
                    d2.rectangle([v * esc for v in c["caixa"]], outline=(0, 200, 255), width=2)
                if l["x0"] != "":
                    d2.rectangle([float(l[c]) * esc for c in ("x0", "y0", "x1", "y1")], outline=(255, 0, 0), width=3)
                jan = [float(l[c]) for c in ("janela_cx", "janela_cy", "janela_alt", "janela_larg")]
                d2.rectangle([(jan[0] - jan[3] / 2) * esc, (jan[1] - jan[2] / 2) * esc, (jan[0] + jan[3] / 2) * esc, (jan[1] + jan[2] / 2) * esc], outline=(255, 220, 0), width=2)
                x, y = (p % 4) * (2 * w2 + 8), (p // 4) * (h2 + 20)
                folha.paste(mini, (x, y)); folha.paste(Image.fromarray(np.asarray(rec[i])).resize((w2, h2)), (x + w2, y))
                num = f0 + p + 1
                dr.text((x + 2, y + h2 + 3), f"{num}. {l['metodo']} {float(l['nota']):.2f} {l['id']} cinza {100 * l['fracao_cinza_janela']:.0f}%", fill=(0, 0, 0))
                tab2.append(dict(numero=num, folha=f0 // 20 + 1, dish_id=l["id"], metodo=l["metodo"], nota=l["nota"], fracao_cinza_janela=l["fracao_cinza_janela"]))
            folha.save(os.path.join(a.saida, f"folha_r8_conferencia_{per}_{f0 // 20 + 1}.jpg"), quality=85)
        with open(os.path.join(a.saida, f"folha_r8_conferencia_{per}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, ["numero", "folha", "dish_id", "metodo", "nota", "fracao_cinza_janela", "veredito"], lineterminator="\n")
            wr.writeheader(); wr.writerows(dict(t, veredito="") for t in tab2)
    diz(f"folhas de contato: {math.ceil(len(linhas) / por)} folhas com todas as janelas mascaradas; 40 por período com as caixas "
        "(visor pintado em ciano, descartado em magenta, prato em vermelho, janela em amarelo)")


# ------------------------------------------------------------------ 5. features e cabeças
def feats_so400m(a, dev, X_u8, bb, n_or=8, rede=None):
    """[n, n_or, 2·dim] float16: o backbone congelado sobre recortes 640×480 (redimensionados como os recortes externos do
    treino: bicúbico com antialias para hw), orientação k do grupo diedral."""
    import torch
    if rede is None and a.smoke:
        torch.manual_seed(0)        # teste sem pesos públicos: a mesma rede aleatória em todo processo
    rede = rede if rede is not None else G.cria_rede(bb, 22, pretrained=not a.smoke).to(dev).eval()
    hw = G.BACKBONES[bb]["hw"]
    out = np.zeros((len(X_u8), n_or, rede.dim), np.float16)
    with torch.inference_mode():
        for i in range(0, len(X_u8), 16):
            x = torch.from_numpy(np.array(X_u8[i:i + 16])).to(dev).permute(0, 3, 1, 2).float().div_(255)
            x = G.redimensiona(x, hw).clamp_(0, 1)
            for k in range(n_or):
                with G.autocast(dev):
                    f = rede.feats(G.orienta(x, k))
                out[i:i + len(x), k] = f.float().cpu().numpy()
    return out, rede


def ridge(z, X):
    """A cabeça ridge gravada (mu, sd, W, mt) em features [n, o, D]: [n, o, 22] em log, float64."""
    Xf = X.reshape(-1, X.shape[-1]).astype(np.float64)
    return (((Xf - z["mu"]) / z["sd"]) @ z["W"] + z["mt"]).reshape(X.shape[0], X.shape[1], -1)


def cabeca_m(z, X, massa):
    """Cabeça de frações com a log-massa como entrada a mais (H0c-m, H-cap), como no serviço: X [n, o, D], massa [n, o].
    Devolve (média das sementes: 22 saídas, frações, densidade pelas frações; e as mesmas por semente)."""
    Xm = np.concatenate([X.astype(np.float32), massa[..., None].astype(np.float32)], -1)
    P, Fr = G.aplica_cabeca4(z, Xm)
    S = np.asarray(z["W1"]).shape[0] if np.asarray(z["W1"]).ndim == 3 else 0
    por = [G.aplica_cabeca4({k: (np.asarray(v)[s] if k[0] in "Wb" and k[1:].isdigit() else v) for k, v in z.items()}, Xm) for s in range(S)]
    dc = np.asarray(z["dens_fonte"])
    return dict(s=P, ff=Fr, dens_ff=Fr @ dc, s_semente=np.stack([p[0] for p in por]) if por else None,
                ff_semente=np.stack([p[1] for p in por]) if por else None)


def carrega_cabecas(a):
    raizes = list(a.fontes or [])
    achados = {k: acha(v, raizes) for k, v in ARQS.items()}
    falta = [ARQS[k] for k in OBRIGATORIOS if not achados[k]]
    if falta:
        raise SystemExit(f"cabeças ou pesos servidos não encontrados (anexe as saídas das execuções 5, 6 e 7 como kernel_sources): {falta}")
    npz = lambda k: dict(np.load(achados[k], allow_pickle=False)) if achados[k] else None
    Z = {k: npz(k) for k in ("r4c", "r4c7", "hm", "hcap", "f5c_sg") if achados[k]}
    info = {k: dict(arquivo=ARQS[k], caminho=achados[k], sha256=sha256(achados[k])) for k in achados if achados[k] and not k.startswith(("feat7", "pred_"))}
    return Z, achados, info


def f5c(a, dev, achados, X_u8):
    """O F5c servido (pesos da sexta execução) num conjunto de recortes 640×480: 1 orientação. Devolve (22 saídas,
    22 saídas da cabeça sem gradiente, frações, densidade pelas frações, informações)."""
    import torch
    ck = torch.load(achados["f5c"], map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    ns = SimpleNamespace(N=ck["n_saidas"], fontes=list(ck.get("fontes") or []) or [f"f{i}" for i in range(10)])
    rede = G.cria_rede4(cfg, ns, pretrained=False)
    tm, ts = G.carrega_modelo4(rede, ck)
    rede.to(dev).eval()
    zsg = {k: v for k, v in ck["cabeca_sg"].items()}
    dif_sg = None
    if achados.get("f5c_sg"):
        zn = np.load(achados["f5c_sg"])
        dif_sg = max(float(np.abs(np.asarray(zn[k]) - zsg[k].numpy()).max()) for k in zsg if k in zn.files)
    hw = G.BACKBONES[cfg["bb"]]["hw"]
    n = len(X_u8)
    P, S = np.zeros((n, ns.N), np.float32), np.zeros((n, ns.N), np.float32)
    Fr = np.zeros((n, len(zsg["dens_fonte"])), np.float32)
    tm_t, ts_t = torch.from_numpy(tm).to(dev), torch.from_numpy(ts).to(dev)
    with torch.inference_mode():
        for i in range(0, n, 32):
            x = torch.from_numpy(np.array(X_u8[i:i + 32])).to(dev).permute(0, 3, 1, 2).float().div_(255)
            x = G.redimensiona(x, hw).clamp_(0, 1)
            with G.autocast(dev):
                f = rede.feats(x)
            f = f.float()
            P[i:i + len(x)] = (rede.head(f) * ts_t + tm_t).cpu().numpy()
            s_, q_ = G.aplica_sg_torch(zsg, f)
            S[i:i + len(x)], Fr[i:i + len(x)] = s_.cpu().numpy(), q_.cpu().numpy()
    return P, S, Fr, Fr @ zsg["dens_fonte"].numpy(), dict(backbone=cfg["bb"], hw=list(hw), dif_cabecasg_npz_x_pesos=dif_sg), rede, (tm, ts, zsg)


# ------------------------------------------------------------------ 6. remontagem
def remonta_offline(Z, achados):
    """(a) As cabeças relidas, sobre as features do recorte do Nutrition5k gravadas pela sétima execução (vista 0 = o
    recorte exato), contra as previsões gravadas no conjunto `remontagem`. Sem GPU."""
    if not (achados.get("feat7") and achados.get("feat7_ids")):
        return dict(situacao="sem as features da sétima execução: não conferida")
    ids7 = [r["id"] for r in csv.DictReader(open(achados["feat7_ids"]))]
    F7 = np.load(achados["feat7"], mmap_mode="r")
    out = {}
    for chave, cab in (("pred_r4c", "r4c"), ("pred_r4c7", "r4c7")):
        if achados.get(chave) and cab in Z:
            p = np.load(achados[chave])
            ids = [str(x) for x in p["ids_remontagem"]]
            X = np.asarray(F7[[ids7.index(d) for d in ids]])[:, :1].astype(np.float32)
            out[cab] = dict(pratos=ids, max_dif_log=float(np.abs(ridge(Z[cab], X)[:, 0] - p["remontagem_o1"]).max()))
    for chave, cab in (("pred_hm", "hm"), ("pred_hcap", "hcap")):
        if achados.get(chave) and cab in Z:
            p = np.load(achados[chave])
            ids = [str(x) for x in p["ids_remontagem"]]
            X = np.asarray(F7[[ids7.index(d) for d in ids]])[:, :1].astype(np.float32)
            r = {}
            for massa_de in ("r4c7", "r4c"):            # r4c7: a da execução que treinou a cabeça; r4c: a servida aqui (16.6 f)
                if massa_de in Z:
                    o = cabeca_m(Z[cab], X, ridge(Z[massa_de], X)[..., G.I_M])
                    r[f"massa_da_{massa_de}"] = dict(max_dif_log=float(np.abs(o["s"][:, 0] - p["remontagem_o1"]).max()),
                                                     max_dif_fracao=float(np.abs(o["ff"][:, 0] - p["remontagem_ff_o1"]).max()))
            out[cab] = r
    return out


def remonta_foto(a, dev, Z, achados, rede_so, bb, f5c_ctx):
    """(b) Da foto: os 8 pratos do conjunto `remontagem` do Nutrition5k, baixados, recortados na janela da etapa 0 pela
    regra de prato e tigela como no treino (no dispositivo, sobre o quadro 640×480) e, à parte, pelo caminho usado aqui
    no DFood-TJ (recorte na CPU e redimensionamento); previsões contra as gravadas."""
    import torch
    if not (achados.get("pred_r4c") and achados.get("caixas_n5k")):
        return dict(situacao="sem as previsões da quinta execução ou sem a tabela da etapa 0: não conferida")
    p5 = np.load(achados["pred_r4c"])
    ids = [str(x) for x in p5["ids_remontagem"]]
    tab = {r["id"]: r for r in csv.DictReader(open(achados["caixas_n5k"])) if r["id"] in ids}
    raiz = os.path.join(a.cache, "n5k")
    G.baixa(ids, "rgb.png", raiz)
    from PIL import Image
    hw = G.BACKBONES[bb]["hw"]
    gpu, cpu, imgs = [], [], []
    for d in ids:
        img = np.asarray(Image.open(os.path.join(raiz, "img", d, "rgb.png")).convert("RGB"))
        k = max(2, int(0.03 * min(img.shape[:2])))
        cor = np.median(np.concatenate([img[:k].reshape(-1, 3), img[-k:].reshape(-1, 3), img[:, :k].reshape(-1, 3), img[:, -k:].reshape(-1, 3)]), 0) / 255
        lr = E.refaz_linha(tab[d], set(E.SO_PRATO))
        jan = (float(lr["janela_cx"]), float(lr["janela_cy"]), float(lr["janela_alt"]), float(lr["janela_larg"]))
        x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255)
        gpu.append((x, jan, torch.tensor(cor, dtype=torch.float32)))
        cpu.append(E.recorta(img, jan, (640, 480), E.cor_borda(img)))
        imgs.append(img)
    with torch.inference_mode():
        xg = torch.stack([G.recorte_gpu(x.to(dev), jan, c.to(dev), hw) for x, jan, c in gpu]).clamp_(0, 1)
        with G.autocast(dev):
            fg = rede_so.feats(xg).float().cpu().numpy()[:, None]
        xc = G.redimensiona(torch.from_numpy(np.stack(cpu)).to(dev).permute(0, 3, 1, 2).float().div_(255), hw).clamp_(0, 1)
        with G.autocast(dev):
            fc = rede_so.feats(xc).float().cpu().numpy()[:, None]
    out = dict(pratos=ids)
    rg, rc = ridge(Z["r4c"], fg)[:, 0], ridge(Z["r4c"], fc)[:, 0]
    out["r4c"] = dict(max_dif_log_caminho_do_treino=float(np.abs(rg - p5["remontagem_o1"]).max()), max_dif_log_caminho_r8=float(np.abs(rc - p5["remontagem_o1"]).max()),
                      max_dif_log_massa_caminho_r8=float(np.abs(rc[:, G.I_M] - p5["remontagem_o1"][:, G.I_M]).max()))
    for chave, cab in (("pred_hm", "hm"), ("pred_hcap", "hcap")):
        if achados.get(chave):
            p = np.load(achados[chave])
            assert [str(x) for x in p["ids_remontagem"]] == ids
            m = "r4c7" if "r4c7" in Z else "r4c"
            og = cabeca_m(Z[cab], fg, ridge(Z[m], fg)[..., G.I_M]); oc = cabeca_m(Z[cab], fc, ridge(Z["r4c"], fc)[..., G.I_M])
            out[cab] = dict(max_dif_log_caminho_do_treino=float(np.abs(og["s"][:, 0] - p["remontagem_o1"]).max()),
                            max_dif_log_caminho_r8=float(np.abs(oc["s"][:, 0] - p["remontagem_o1"]).max()),
                            max_dif_fracao_caminho_do_treino=float(np.abs(og["ff"][:, 0] - p["remontagem_ff_o1"]).max()))
    if achados.get("pred_f5c") and f5c_ctx is not None:
        p6 = np.load(achados["pred_f5c"])
        assert [str(x) for x in p6["ids_remontagem"]] == ids
        rede, (tm, ts, zsg) = f5c_ctx
        cfg_hw = G.BACKBONES[a._bb_f5c]["hw"]
        with torch.inference_mode():
            x5 = torch.stack([G.recorte_gpu(x.to(dev), jan, c.to(dev), cfg_hw) for x, jan, c in gpu]).clamp_(0, 1)
            with G.autocast(dev):
                f = rede.feats(x5)
            f = f.float()
            P = (rede.head(f) * torch.from_numpy(ts).to(dev) + torch.from_numpy(tm).to(dev)).cpu().numpy()
            S = G.aplica_sg_torch(zsg, f)[0].cpu().numpy()
        out["f5c"] = dict(max_dif_log=float(np.abs(P - p6["remontagem_o1"]).max()), max_dif_log_cabeca_sg=float(np.abs(S - p6["remontagem_sg_o1"]).max()))
    return out


# ------------------------------------------------------------------ principal
def principal(a):
    t_ini = time.time()
    a.prazo = t_ini + a.orcamento_h * 3600
    os.makedirs(a.saida, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
    G.PASTA_PESOS = os.path.join(a.cache, "pesos")
    os.environ.setdefault("HF_HOME", os.path.join(a.cache, "hf"))
    import torch
    import timm
    try:
        import transformers
        v_tr = transformers.__version__
    except ImportError:         # só o detector precisa dele (o teste com --sem-detector roda sem)
        v_tr = None
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    man = dict(formato=FORMATO, inicio=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv[1:], smoke=a.smoke, torch=torch.__version__, timm=timm.__version__,
               transformers=v_tr, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
               nota="16.6 f: só previsões cruas; nenhuma métrica, nenhum rótulo, nenhuma escolha. As rotas (r2, r4) e a leitura são da CPU.")
    tempos = {}

    def grava():
        man.update(tempos_seg=tempos, duracao_min=round((time.time() - t_ini) / 60, 1))
        G.grava_json(os.path.join(a.saida, "manifesto_r8.json"), man)
        open(os.path.join(a.saida, "teste_r8.log"), "w").write("\n".join(REGISTRO) + "\n")
    arq_t = a.teste or os.path.join(AQUI, "listas", "teste_r8_dfoodtj.csv")
    L, sha = le_teste(arq_t, a.smoke)
    if a.smoke and a.imagens:       # teste: só as fotos que já estão na pasta
        L = [r for r in le_teste(arq_t, 10 ** 6)[0] if os.path.exists(os.path.join(a.imagens, os.path.basename(r["arquivo"])))][:a.smoke]
    man["teste"] = dict(arquivo=os.path.basename(arq_t), sha256=sha, registrado=SHA_TESTE, confere=sha == SHA_TESTE, fotos=len(L),
                        por_periodo={p: sum(1 for r in L if r["periodo"] == p) for p in sorted({r["periodo"] for r in L})},
                        pacientes=len({r["paciente"] for r in L}))
    diz(f"{FORMATO} | {len(L)} fotos de {man['teste']['pacientes']} pacientes {man['teste']['por_periodo']} | tabela sha256 {sha[:16]}… "
        f"({'confere com 16.6' if sha == SHA_TESTE else 'DIFERENTE da de 16.6 (smoke)'}) | dispositivo {dev} | torch {torch.__version__}, timm {timm.__version__}, transformers {v_tr}")
    if not a.so_deteccao:
        Z, achados, info_c = carrega_cabecas(a)
        man["cabecas"] = info_c
        diz("cabeças e pesos servidos: " + " | ".join(f"{k} {v['arquivo']} ({v['sha256'][:12]})" for k, v in info_c.items()))
    grava()
    t = time.time()
    L, fotos, man["download"] = baixa_pngs(a, L)
    man["teste"]["fotos_usadas"] = len(L)
    tempos["download"] = round(time.time() - t, 1)
    man["fotos"] = {d: {k: v[k] for k in v if k != "caminho"} for d, v in fotos.items()}
    grava()
    t = time.time()
    linhas, man["deteccao"] = processa(a, L, fotos, dev)
    tempos["deteccao"] = round(time.time() - t, 1)
    t = time.time()
    folhas(a, L, linhas, fotos)
    tempos["folhas"] = round(time.time() - t, 1)
    grava()
    if a.so_deteccao:           # teste do caminho detector -> máscara -> recorte, sem cabeças
        diz(f"fim (--so-deteccao): {(time.time() - t_ini) / 60:.1f} min; sem features nem cabeças.")
        return grava()
    bb = a.backbone or G._SO4
    pred = dict(ids=np.array([r["dish_id"] for r in L]), periodo=np.array([r["periodo"] for r in L]), paciente=np.array([r["paciente"] for r in L]))
    rede_so = None
    for var, arqv in (("recorte", "recorte8.npy"), ("inteira", "inteira8.npy")):
        t = time.time()
        X, rede_so = feats_so400m(a, dev, np.load(os.path.join(a.cache, arqv), mmap_mode="r"), bb, rede=rede_so)
        np.save(os.path.join(a.saida, f"feat8__dfoodtj__{var}.npy"), X)
        Xf = X.astype(np.float32)
        r4 = ridge(Z["r4c"], Xf)
        pred[f"{var}__r4c"] = r4.astype(np.float32)
        if "r4c7" in Z:
            man.setdefault("massa_r4c_x_r4c7", {})[var] = float(np.abs(ridge(Z["r4c7"], Xf)[..., G.I_M] - r4[..., G.I_M]).max())
        for cab in ("hm", "hcap"):
            o = cabeca_m(Z[cab], Xf, r4[..., G.I_M])
            for k, v in o.items():
                if v is not None:
                    pred[f"{var}__{cab}__{k}"] = np.asarray(v, np.float32)
        tempos[f"features_e_cabecas_{var}"] = round(time.time() - t, 1)
        diz(f"features do So400m e cabeças ({var}): {X.shape} em {time.time() - t:.0f}s")
    t = time.time()
    ctx = None
    for var, arqv in (("recorte", "recorte8.npy"), ("inteira", "inteira8.npy")):
        P, S, Fr, Dff, info5, rede5, aux = f5c(a, dev, achados, np.load(os.path.join(a.cache, arqv), mmap_mode="r"))
        pred.update({f"{var}__f5c__s": P, f"{var}__f5c__s_sg": S, f"{var}__f5c__ff": Fr, f"{var}__f5c__dens_ff": Dff.astype(np.float32)})
        man["f5c"] = info5
        ctx = (rede5, aux)
        a._bb_f5c = info5["backbone"]
    tempos["f5c"] = round(time.time() - t, 1)
    zh = np.load(achados["hm"])
    np.savez(os.path.join(a.saida, "pred8__dfoodtj.npz"), fontes=zh["fontes"] if "fontes" in zh.files else np.array([]), **pred)
    man["previsoes"] = dict(arquivo="pred8__dfoodtj.npz", chaves=sorted(pred), formas={k: list(v.shape) for k, v in pred.items()},
                            nota="as 22 saídas na ordem de `saidas.nomes` dos servir__*.json; <variante>__<cabeça>__s = 22 saídas em log por orientação [n, 8, 22] (F5c: [n, 22], 1 orientação); __ff = frações; "
                                 "__dens_ff = densidade pelas frações; __s_semente / __ff_semente = cada semente [3, n, 8, ...]; a massa de entrada da H0c-m e da H-cap é a da R4c (cabeça da quinta execução) na mesma orientação")
    grava()
    t = time.time()
    try:
        man["remontagem"] = dict(offline=remonta_offline(Z, achados), da_foto=remonta_foto(a, dev, Z, achados, rede_so, bb, ctx), tolerancia_log=TOL_REMONTA)
    except Exception as ex:     # noqa: BLE001 — a remontagem é conferência: o erro fica registrado
        man["remontagem"] = dict(erro=f"{type(ex).__name__}: {ex}", traceback=traceback.format_exc()[-2000:])
    tempos["remontagem"] = round(time.time() - t, 1)
    difs = []

    def junta(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.startswith("max_dif") and isinstance(v, float) and "caminho_r8" not in k:
                    difs.append(v)
                junta(v)
    junta(man["remontagem"])
    man["remontagem"]["confere"] = bool(difs) and max(difs) <= TOL_REMONTA
    diz(f"remontagem: {'confere' if man['remontagem']['confere'] else 'NÃO CONFERE ou incompleta'} (maior diferença {max(difs) if difs else None}); "
        + json.dumps({k: v for k, v in man["remontagem"].items() if k != "traceback"}, ensure_ascii=False)[:1500])
    grava()
    diz(f"fim: {(time.time() - t_ini) / 60:.1f} min. Nada foi pontuado; a conferência a olho das folhas e a leitura são da CPU (16.6 g).")
    grava()


def autoteste():
    """Sem GPU nem rede: a tabela sem rótulo, a conferência do PNG, a máscara, e as cabeças com a massa de entrada."""
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        arq = os.path.join(t, "t.csv")
        open(arq, "w").write("dish_id,arquivo,periodo,paciente\nd1,a/1.png,2020-2023,p1\nd2,a/2.png,2024-2025,p2\n")
        L, sha = le_teste(arq, smoke=2)
        assert len(L) == 2
        try:
            le_teste(arq)
            raise AssertionError("sha256 diferente passou")
        except AssertionError as e:
            assert "diferente" in str(e)
        open(arq, "w").write("dish_id,arquivo,periodo,paciente,proteina\nd1,a/1.png,2020-2023,p1,12\n")
        try:
            le_teste(arq, smoke=1)
            raise SystemExit("tabela com rótulo passou")
        except AssertionError:
            pass
    assert png_ok(ASSINATURA_PNG + b"x" * 20 + FIM_PNG) and not png_ok(ASSINATURA_PNG + b"x" * 20)
    from PIL import Image
    bio = io.BytesIO(); Image.new("RGB", (40, 30), (200, 10, 10)).save(bio, "PNG"); png = bio.getvalue()
    bio = io.BytesIO(); Image.new("RGB", (40, 30), (200, 10, 10)).save(bio, "JPEG"); jpg = bio.getvalue()
    assert classifica(png)[0] == "png" and classifica(png + b"lixo")[0] == "png_sem_iend_no_fim" and classifica(png[:-12])[0] == "png_sem_iend_no_fim"
    assert classifica(jpg)[0] == "nao_png_decodificavel" and classifica(jpg)[1]["formato"] == "JPEG"
    assert classifica(png[:60])[0] == "indecodificavel" and classifica(b"nada")[0] == "indecodificavel"
    import zlib
    f = dict(bytes=str(len(png)), crc32=f"{zlib.crc32(png) & 0xFFFFFFFF:08x}", sha256=hashlib.sha256(png).hexdigest())
    assert confere_arquivo(png, f) and not confere_arquivo(png + b"x", f) and not confere_arquivo(png[:-1] + b"\0", f)
    with tempfile.TemporaryDirectory() as t:     # a tabela das fotos: colunas sem rótulo, cobre a tabela do teste
        arq = os.path.join(t, "f.csv")
        open(arq, "w").write(",".join(COLS_FOTOS) + "\nd1,a/1.png,1.png,00000000,1,x,png\n")
        Lf = [dict(dish_id="d1", arquivo="a/1.png")]
        global SHA_FOTOS
        guarda, SHA_FOTOS = SHA_FOTOS, None
        try:
            assert le_fotos(arq, Lf)[0]["d1"]["nome"] == "1.png"
            for ruim in (Lf + [dict(dish_id="d2", arquivo="a/2.png")], [dict(dish_id="d1", arquivo="b/1.png")]):
                try:
                    le_fotos(arq, ruim)
                    raise SystemExit("tabela das fotos incompleta passou")
                except AssertionError:
                    pass
            open(arq, "w").write(",".join(COLS_FOTOS + ["proteina"]) + "\nd1,a/1.png,1.png,00000000,1,x,png,3\n")
            try:
                le_fotos(arq, Lf)
                raise SystemExit("tabela das fotos com rótulo passou")
            except AssertionError:
                pass
        finally:
            SHA_FOTOS = guarda
        if SHA_FOTOS:
            open(arq, "w").write(",".join(COLS_FOTOS) + "\nd1,a/1.png,1.png,00000000,1,x,png\n")
            try:
                le_fotos(arq, Lf)
                raise SystemExit("tabela das fotos diferente da registrada passou")
            except AssertionError as e:
                assert "diferente" in str(e)
    img = np.zeros((10, 10, 3), np.uint8)
    m, pint = mascara_fora_do_prato(img, [(2.4, 2.6, 5.1, 4.0)], np.array([9, 9, 9], np.uint8), None)
    assert m[2:4, 2:6].min() == 9 and m[:2].max() == 0 and img.max() == 0 and pint.sum() == 8
    m, pint = mascara_fora_do_prato(img, [(0, 0, 10, 10)], np.array([9, 9, 9], np.uint8), (4.2, 4.2, 6.0, 6.0))    # o prato nunca é pintado
    assert m[4:6, 4:6].max() == 0 and m[0, 0].min() == 9 and pint.sum() == 96
    assert fracao_cinza(pint, (5, 5, 10, 10)) == (0.96, 0.0) and fracao_cinza(pint, (0, 0, 10, 10)) == (0.96, 0.75) and fracao_cinza(pint, (50, 50, 10, 10)) == (0.0, 1.0)
    cx_ = np.array([[0, 0, 50, 50], [10, 10, 20, 20], [60, 60, 70, 70], [79, 79, 99, 99], [1, 1, 5, 5]], float)
    fv = filtra_visor(cx_, np.array([0.5, 0.5, 0.5, 0.5, 0.1]), np.array([0, 1, 0, 1, 0]), 100, 100, (15, 15, 65, 65))
    assert [c["destino"] for c in fv] == ["descartada_area", "descartada_prato", "descartada_prato", "mascarada"], fv   # a de nota 0,1 nem entra
    assert fv[2]["dentro_do_prato"] == 0.25 and fv[3]["area_foto"] == 0.04 and fv[1]["consulta"] == "an LCD screen"
    assert filtra_visor(cx_[1:2], np.array([0.5]), np.array([0]), 100, 100, None)[0]["destino"] == "mascarada"     # sem prato: só a área
    assert resumo_cinza([0, 0.06, 0.3])["acima_25pct"] == 1 and resumo_cinza([])["n"] == 0
    assert caixa_prato(dict(x0="")) is None and caixa_prato(dict(x0="1", y0="2", x1="3", y1="4")) == (1.0, 2.0, 3.0, 4.0)
    rng = np.random.default_rng(0)
    D_, K = 6, 10
    z = dict(mu=np.zeros(D_ + 1, np.float32), sd=np.ones(D_ + 1, np.float32), tm=np.zeros(22, np.float32), ts=np.ones(22, np.float32), dens_fonte=np.ones(K, np.float32),
             W1=rng.normal(size=(3, 5, D_ + 1)).astype(np.float32), b1=np.zeros((3, 5), np.float32), W2=rng.normal(size=(3, 22 + K, 5)).astype(np.float32), b2=np.zeros((3, 22 + K), np.float32))
    X = rng.normal(size=(4, 2, D_)).astype(np.float32)
    o = cabeca_m(z, X, np.ones((4, 2)))
    assert o["s"].shape == (4, 2, 22) and o["s_semente"].shape == (3, 4, 2, 22) and np.allclose(o["s"], o["s_semente"].mean(0), atol=1e-5)
    o2 = cabeca_m(z, X, np.zeros((4, 2)))
    assert not np.allclose(o["s"], o2["s"])                      # a massa de entrada entra
    zr = dict(mu=np.zeros(D_), sd=np.ones(D_), W=rng.normal(size=(D_, 22)), mt=np.zeros(22))
    assert ridge(zr, X).shape == (4, 2, 22)
    print("autoteste da oitava execução: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--saida", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "saida_r8")
    ap.add_argument("--cache", default="/kaggle/temp/nutrelle_r8" if os.path.isdir("/kaggle") else os.path.join(os.path.expanduser("~"), ".cache", "nutrelle_r8"))
    ap.add_argument("--teste", help="tabela do teste (padrão: listas/teste_r8_dfoodtj.csv ao lado do script)")
    ap.add_argument("--fontes", action="append", help="pasta com as saídas das execuções 5, 6 e 7 e da etapa 0 (além de /kaggle/input)")
    ap.add_argument("--imagens", help="pasta com os PNG já em disco (teste; sem download e sem conferência pela tabela das fotos)")
    ap.add_argument("--espelho-fotos", help="pasta com as fotos do teste (o dataset privado do espelho); padrão: procurada em /kaggle/input")
    ap.add_argument("--fotos", help="tabela das fotos (padrão: listas/fotos_r8_dfoodtj.csv ao lado do script)")
    ap.add_argument("--orcamento-h", type=float, default=1.5)
    ap.add_argument("--so-deteccao", action="store_true", help="para depois da detecção, da máscara, dos recortes e das folhas (sem features nem cabeças)")
    ap.add_argument("--smoke", type=int, default=0, help="só as N primeiras fotos (não confere o sha256 da tabela)")
    ap.add_argument("--sem-detector", action="store_true", help="só a reserva por elipse e nenhum visor (teste em CPU)")
    ap.add_argument("--backbone", help="backbone das cabeças (teste; padrão: o So400m do treino)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--autoteste", action="store_true")
    a = ap.parse_args()
    if a.autoteste:
        return autoteste()
    principal(a)


if __name__ == "__main__":
    main()
