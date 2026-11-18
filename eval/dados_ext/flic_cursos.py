#!/usr/bin/env python3
"""FLIC, seção 19: conferência das fotos, espelho do Kaggle, recorte do guia por curso e folha de contato.

    python3 flic_cursos.py confere <pasta_dados>                  # tamanho, sha256 e decodificação de cada PNG -> flic_fotos.csv
    python3 flic_cursos.py espelho <pasta_dados> <pasta_espelho> <usuario_kaggle>   # pasta plana (links), dataset-metadata.json
    python3 flic_cursos.py detecta <pasta_dados> [--saida <csv>] [--cache <pasta>] [--limite N]   # OWLv2 na CPU + regra por curso
    python3 flic_cursos.py folha <pasta_dados> <saida.jpg> [--tabela <csv>]              # conferência a olho (19.2 d)
    python3 flic_cursos.py --autoteste

<pasta_dados> tem full_images/, full_masks/, left_images/, left_masks/ (os PNG da pasta do Drive; flic_arquivos.csv).
As fotos são dados de terceiros: rode com `python3 -I` e mantenha esta pasta fora da dos dados.

Regra do recorte por curso (19.2 c), sem rótulo: as caixas do OWLv2 da etapa 0 (mesmo checkpoint, 960 px, consultas e
limiares de deteccao_r5.py), só as de prato e tigela (SO_PRATO), NMS 0,5, sem as caixas com mais de 40% da foto (a
bandeja). Para cada linha da lista por curso, M = pixels da cor do curso na máscara; `cobre` = fração de M dentro da
caixa; `pureza` = fração dos pixels de máscara (qualquer curso, pão incluído) dentro da caixa que são do curso. Fica a
união das caixas que tocam a comida do curso e têm pureza >= 0,50 (emenda de 19.2 c: a regra da etapa 0, restrita às
caixas do curso), aceita com cobre >= 0,60 e pureza >= 0,50; senão `sem_prato` (a linha sai na divisão).
Emenda de 19.9: pureza >= 0,85 em cada caixa da união e na união; `refaz` recalcula a tabela das caixas brutas gravadas
(sem detector): nenhuma caixa -> `sem_prato`; união com cobre < 0,60 ou pureza < 0,85 -> `impuro`. As duas saem da divisão.

    python3 flic_cursos.py refaz <pasta_dados> [--tabela <csv>] [--saida <csv>]      # 19.9
    python3 flic_cursos.py folha <pasta_dados> <saida.jpg> --semente 20261120         # 19.9 A janela é a do guia (deteccao_r5.janela). Massa, proteína e previsões não entram em nada aqui."""
import csv, hashlib, json, os, sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "treino", "kaggle"))
LISTAS = os.path.join(AQUI, "..", "treino", "kaggle", "listas")
FOTOS = os.path.join(AQUI, "flic_fotos.csv")
LISTA = os.path.join(LISTAS, "externo_flic_curso.csv")
TABELA = os.path.join(LISTAS, "caixas_r5_flic_curso.csv")
PASTAS = ("full_images", "full_masks", "left_images", "left_masks")
CHAVE = "flic_curso"
AREA_MAX_PRATO = 0.40
COBRE_MIN, PUREZA_MIN = 0.60, 0.50
PUREZA_MIN_19_9 = 0.85          # emenda de 19.9
SEMENTE_FOLHA, N_FOLHA = 20261119, 40      # 19.2 d; a folha da emenda (19.9) usa --semente 20261120
ACEITA = "owlv2_curso"
EXTRA = ["curso", "cobre", "pureza", "area_caixa"]


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cmd_confere(dados):
    from PIL import Image
    with open(os.path.join(AQUI, "flic_arquivos.csv"), encoding="utf-8", newline="") as f:
        esperado = [r["arquivo"] for r in csv.DictReader(f) if r["arquivo"].split("/")[0] in PASTAS]
    linhas, ruins = [], []
    for arq in sorted(esperado):
        c = os.path.join(dados, arq)
        try:
            with open(c, "rb") as f:
                assert f.read(8) == b"\x89PNG\r\n\x1a\n", "assinatura"
            with Image.open(c) as im:
                im.load()
                tam = im.size
            linhas.append(dict(arquivo=arq, espelho=arq.replace("/", "__"), bytes=os.path.getsize(c), sha256=sha256(c), largura=tam[0], altura=tam[1]))
        except Exception as e:
            ruins.append((arq, f"{type(e).__name__}: {e}"))
    if ruins:
        sys.exit(f"{len(ruins)} arquivos faltando ou ruins (ex.: {ruins[:3]})")
    with open(FOTOS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(linhas[0]), lineterminator="\n"); w.writeheader(); w.writerows(linhas)
    print(f"{len(linhas)} PNG conferidos (assinatura, decodificação inteira, tamanho e sha256) -> {os.path.relpath(FOTOS)}; "
          f"{sum(r['bytes'] for r in linhas) / 2 ** 20:.1f} MB; tamanhos {sorted({(r['largura'], r['altura']) for r in linhas})}")


def cmd_espelho(dados, pasta, usuario):
    """Pasta plana para `kaggle datasets create`: um link físico por foto (não ocupa disco), com o nome do espelho. Só as
    fotos: o kernel não usa as máscaras (e o Kaggle recusou criar o dataset com os 1.604 arquivos de uma vez)."""
    with open(FOTOS, encoding="utf-8", newline="") as f:
        F = [r for r in csv.DictReader(f) if r["arquivo"].split("/")[0].endswith("_images")]
    os.makedirs(pasta, exist_ok=True)
    for r in F:
        dest = os.path.join(pasta, r["espelho"])
        if not os.path.exists(dest):
            os.link(os.path.join(dados, r["arquivo"]), dest)
        assert os.path.getsize(dest) == int(r["bytes"])
    meta = dict(title="nutrelle-flic-fotos", id=f"{usuario}/nutrelle-flic-fotos", licenses=[dict(name="CC-BY-4.0")], isPrivate=True,
                description="Espelho das fotos do FLIC (sem as máscaras) (Piccoli et al., Applied Sciences 2026, 16(11), 5465, "
                            "https://doi.org/10.3390/app16115465), CC BY 4.0, para o treino da seção 19. Pasta plana: "
                            "<pasta>__<arquivo>.png. Atribuição obrigatória.")
    json.dump(meta, open(os.path.join(pasta, "dataset-metadata.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(F)} arquivos em {pasta} (links), dataset-metadata.json privado, CC-BY-4.0")


# ------------------------------------------------------------------ a regra por curso (pura; testada no --autoteste)
def caixas_prato(caixas, notas, consultas, w, h):
    """Caixas de prato e tigela que passam na nota e na área da etapa 0, NMS 0,5, sem as maiores que AREA_MAX_PRATO."""
    import deteccao_r5 as E
    c = np.clip(np.asarray(caixas, np.float64).reshape(-1, 4), 0, [w, h, w, h]); n = np.asarray(notas, np.float64).reshape(-1)
    q = np.asarray(consultas).reshape(-1)
    area = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0)
    ok = np.where(np.isin(q, E.SO_PRATO) & (n >= E.NOTA_MIN) & (area >= E.AREA_MIN * w * h) & (area <= AREA_MAX_PRATO * w * h))[0]
    if not len(ok):
        return np.zeros((0, 4)), np.zeros(0)
    fica = ok[E.nms(c[ok], n[ok])]
    return c[fica], n[fica]


def medidas(caixa, curso, qualquer):
    """(cobre, pureza) de uma caixa (x0, y0, x1, y1) para a máscara do curso."""
    x0, y0, x1, y1 = caixa
    a, b, c_, d = int(np.floor(y0)), int(np.ceil(y1)), int(np.floor(x0)), int(np.ceil(x1))
    dentro = int(curso[a:b, c_:d].sum()); todos = int(qualquer[a:b, c_:d].sum())
    return dentro / max(1, int(curso.sum())), dentro / max(1, todos)


def escolhe(caixas, curso, qualquer, pureza_min=PUREZA_MIN):
    """Emenda 19.2 c (antes de qualquer conferência): a caixa do curso é a UNIÃO das caixas de prato e tigela que tocam a
    comida do curso e têm pureza >= 0,50 (a regra da etapa 0, a união, restrita às caixas do curso), aceita se a união
    tiver cobre >= 0,60 e pureza >= 0,50. caixas [k, 4]; curso, qualquer: máscaras booleanas [h, w].
    -> (caixa (x0, y0, x1, y1) ou None, cobre, pureza, nº de caixas unidas)."""
    usa = []
    for cx in caixas:
        cobre, pureza = medidas(cx, curso, qualquer)
        if cobre > 0 and pureza >= pureza_min:
            usa.append(cx)
    if not usa:
        return None, 0.0, 0.0, 0
    u = np.asarray(usa)
    uniao = (float(u[:, 0].min()), float(u[:, 1].min()), float(u[:, 2].max()), float(u[:, 3].max()))
    cobre, pureza = medidas(uniao, curso, qualquer)
    if cobre < COBRE_MIN or pureza < pureza_min:
        return None, cobre, pureza, len(usa)
    return uniao, cobre, pureza, len(usa)


def mascaras(arq_mascara, cor):
    from PIL import Image
    m = np.asarray(Image.open(arq_mascara).convert("RGB"))
    return (m == np.array(cor, np.uint8)).all(-1), m.any(-1)


def cmd_detecta(dados, saida, cache, limite=None):
    import torch, deteccao_r5 as E
    import huggingface_hub
    orig = huggingface_hub.snapshot_download          # só os pesos em safetensors (o .bin tem o mesmo conteúdo; disco curto)
    huggingface_hub.snapshot_download = lambda *a, **k: orig(*a, allow_patterns=["*.json", "*.txt", "model.safetensors"], **k)
    with open(LISTA, encoding="utf-8", newline="") as f:
        L = list(csv.DictReader(f))
    por_foto = {}
    for r in L:
        por_foto.setdefault(r["arquivo_original"], []).append(r)
    fotos = sorted(por_foto)[:limite] if limite else sorted(por_foto)
    det = E.Detector(torch.device("cpu"), cache)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    out, cont = [], {}
    import time
    t0 = time.time()
    for k, foto in enumerate(fotos):
        caminho = os.path.join(dados, foto)
        img, girada = E.abre_canonica(caminho)
        assert not girada
        h, w = img.shape[:2]
        cx, nt, cq = det([img])[0]
        cp, npr = caixas_prato(cx, nt, cq, w, h)
        brutas = E.brutas_de(cx, nt, cq, w, h)
        sha = E.sha256(caminho)
        for r in por_foto[foto]:
            assert r["sha256"] == sha, f"{foto}: sha256 diferente da lista"
            cur, qualquer = mascaras(os.path.join(dados, r["mascara"].replace("__", "/", 1)), tuple(int(v) for v in r["cor_mascara"].split(",")))
            caixa, cobre, pureza, nu = escolhe(cp, cur, qualquer)
            jan = E.janela(caixa, w, h)
            metodo = "owlv2_curso" if caixa else "sem_prato"
            cont[metodo] = cont.get(metodo, 0) + 1
            out.append(dict(id=f"ext:{CHAVE}:{r['dish_id']}", dominio="flic", arquivo=r["arquivo"], sha256=sha, largura=w, altura=h, girada=0, metodo=metodo,
                            **({c: "" for c in ("x0", "y0", "x1", "y1")} if caixa is None else dict(zip(("x0", "y0", "x1", "y1"), [round(v, 1) for v in caixa]))),
                            nota=0.0, n_caixas=nu, janela_cx=round(jan[0], 1), janela_cy=round(jan[1], 1),
                            janela_alt=round(jan[2], 1), janela_larg=round(jan[3], 1), versao_detector=det.versao + " | regra por curso da seção 19.2 c",
                            iou_uniao_so_prato="", caixa_so_prato="", caixa_elipse="", caixas_brutas=json.dumps(brutas, separators=(",", ":")),
                            curso=r["curso"], cobre=round(cobre, 4), pureza=round(pureza, 4),
                            area_caixa="" if caixa is None else round((caixa[2] - caixa[0]) * (caixa[3] - caixa[1]) / (w * h), 4)))
        if k % 50 == 0:
            print(f"{k + 1}/{len(fotos)} fotos, {time.time() - t0:.0f}s, {cont}", flush=True)
    with open(saida + ".tmp", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, E.COLS + EXTRA, lineterminator="\n"); wr.writeheader(); wr.writerows(out)
    os.replace(saida + ".tmp", saida)
    print(f"{len(out)} linhas, {len(fotos)} fotos em {time.time() - t0:.0f}s: {cont} -> {saida}")


def cmd_refaz(dados, tabela, saida):
    """19.9: a tabela refeita das caixas brutas gravadas (sem detector), com pureza >= 0,85 nas caixas e na união."""
    import deteccao_r5 as E
    with open(tabela, encoding="utf-8", newline="") as f:
        T = list(csv.DictReader(f))
    with open(LISTA, encoding="utf-8", newline="") as f:
        L = {f"ext:{CHAVE}:{r['dish_id']}": r for r in csv.DictReader(f)}
    cont, cheias = {}, 0
    for r in T:
        l = L[r["id"]]
        w, h = float(r["largura"]), float(r["altura"])
        b = json.loads(r["caixas_brutas"] or "[]")
        cheias += len(b) >= 40
        cp, _ = caixas_prato([x[:4] for x in b], [x[4] for x in b], [x[5] for x in b], w, h) if b else (np.zeros((0, 4)), None)
        cur, qualquer = mascaras(os.path.join(dados, l["mascara"].replace("__", "/", 1)), tuple(int(v) for v in l["cor_mascara"].split(",")))
        caixa, cobre, pureza, nu = escolhe(cp, cur, qualquer, PUREZA_MIN_19_9)
        metodo = ACEITA if caixa else ("sem_prato" if nu == 0 else "impuro")
        cont[metodo] = cont.get(metodo, 0) + 1
        jan = E.janela(caixa, w, h)
        r.update(metodo=metodo, n_caixas=nu, cobre=round(cobre, 4), pureza=round(pureza, 4), janela_cx=round(jan[0], 1), janela_cy=round(jan[1], 1),
                 janela_alt=round(jan[2], 1), janela_larg=round(jan[3], 1),
                 versao_detector=r["versao_detector"].split(" | ")[0] + " | regra por curso de 19.2 c com a emenda de 19.9 (pureza >= 0,85), refeita das caixas brutas",
                 area_caixa="" if caixa is None else round((caixa[2] - caixa[0]) * (caixa[3] - caixa[1]) / (w * h), 4),
                 **({c: "" for c in ("x0", "y0", "x1", "y1")} if caixa is None else dict(zip(("x0", "y0", "x1", "y1"), [round(v, 1) for v in caixa]))))
    with open(saida + ".tmp", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, list(T[0]), lineterminator="\n"); wr.writeheader(); wr.writerows(T)
    os.replace(saida + ".tmp", saida)
    print(f"{len(T)} linhas refeitas: {cont} | linhas com 40 caixas brutas (o teto gravado): {cheias} -> {saida}")


def cmd_folha(dados, saida, tabela, semente=SEMENTE_FOLHA):
    """Folha de contato de 19.2 d: 40 linhas aceitas sorteadas (default_rng(20261119), sobre a lista ordenada por id), a
    janela que o modelo recebe, com o contorno da máscara do curso (magenta) e a caixa (vermelho). CSV ao lado, com o
    `veredito` vazio."""
    import deteccao_r5 as E
    from PIL import Image, ImageDraw
    with open(tabela, encoding="utf-8", newline="") as f:
        T = sorted((r for r in csv.DictReader(f) if r["metodo"] == ACEITA), key=lambda r: r["id"])
    with open(LISTA, encoding="utf-8", newline="") as f:
        L = {f"ext:{CHAVE}:{r['dish_id']}": r for r in csv.DictReader(f)}
    esc = sorted(np.random.default_rng(semente).choice(len(T), N_FOLHA, replace=False))
    lado, cols = (400, 300), 5
    folha = Image.new("RGB", (cols * lado[0], (N_FOLHA // cols) * (lado[1] + 18)), "white")
    d = ImageDraw.Draw(folha)
    linhas = []
    for k, j in enumerate(esc):
        r = T[j]; l = L[r["id"]]
        img, _ = E.abre_canonica(os.path.join(dados, l["arquivo_original"]))
        cur, _ = mascaras(os.path.join(dados, l["mascara"].replace("__", "/", 1)), tuple(int(v) for v in l["cor_mascara"].split(",")))
        borda = cur & ~(np.roll(cur, 1, 0) & np.roll(cur, -1, 0) & np.roll(cur, 1, 1) & np.roll(cur, -1, 1))
        borda = borda | np.roll(borda, 1, 0) | np.roll(borda, 1, 1)
        a = img.copy(); a[borda] = (255, 0, 255)
        im = Image.fromarray(a); dr = ImageDraw.Draw(im)
        dr.rectangle([float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])], outline=(255, 0, 0), width=3)
        jan = tuple(float(r[c]) for c in ("janela_cx", "janela_cy", "janela_alt", "janela_larg"))
        rec = E.recorta(np.asarray(im), jan, lado, E.cor_borda(img))
        x, y = (k % cols) * lado[0], (k // cols) * (lado[1] + 18)
        folha.paste(Image.fromarray(rec), (x, y + 18))
        d.text((x + 3, y + 3), f"{k + 1:02d} {l['curso']} {l['fase']} c{float(r['cobre']):.2f} p{float(r['pureza']):.2f}", fill="black")
        linhas.append(dict(posicao=k + 1, id=r["id"], curso=l["curso"], fase=l["fase"], cobre=r["cobre"], pureza=r["pureza"], veredito=""))
    folha.save(saida, quality=90)
    with open(os.path.splitext(saida)[0] + ".csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(linhas[0]), lineterminator="\n"); w.writeheader(); w.writerows(linhas)
    print(f"folha com {len(linhas)} janelas -> {saida}")


def autoteste():
    cur = np.zeros((100, 100), bool); cur[20:40, 20:40] = True
    outro = np.zeros((100, 100), bool); outro[60:80, 60:80] = True
    qualquer = cur | outro
    prato, comida, vizinho, dois = [10, 10, 50, 50], [19, 19, 41, 41], [55, 55, 90, 90], [0, 0, 100, 100]
    cx, cobre, pureza, n = escolhe(np.array([prato, comida, vizinho], float), cur, qualquer)
    assert cx == (10, 10, 50, 50) and cobre == 1.0 and pureza == 1.0 and n == 2, (cx, cobre, pureza, n)   # prato ∪ comida = o prato; o vizinho não entra
    cx, _, pureza, _ = escolhe(np.array([dois], float), cur, qualquer)
    assert cx == (0, 0, 100, 100) and abs(pureza - 0.5) < 1e-9          # pureza 0,5: aceita no limite (a bandeja já sai pela área)
    assert escolhe(np.array([vizinho], float), cur, qualquer)[0] is None             # só o prato vizinho: sem_prato
    assert escolhe(np.array([[18, 18, 31, 42]], float), cur, qualquer)[0] is None     # cobre 0,5 < 0,6
    cx, _, _, n = escolhe(np.array([prato, dois], float), cur, qualquer, PUREZA_MIN_19_9)
    assert cx == (10, 10, 50, 50) and n == 1                            # 19.9: a caixa de dois pratos (pureza 0,5) não entra na união
    assert escolhe(np.array([dois], float), cur, qualquer, PUREZA_MIN_19_9)[3] == 0        # só ela: nenhuma caixa -> sem_prato
    import deteccao_r5 as E
    cp, _ = caixas_prato([[0, 0, 640, 480], [10, 10, 200, 200], [12, 12, 198, 199], [300, 300, 400, 400]], [0.9, 0.5, 0.4, 0.3], [5, 5, 6, 2], 640, 480)
    assert len(cp) == 1 and tuple(cp[0]) == (10, 10, 200, 200)       # a bandeja sai (área), o NMS junta as duas, "a food tray" (2) não é prato
    assert os.path.basename(E.__file__) == "deteccao_r5.py"
    print("autoteste ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    op = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    if a[:1] == ["--autoteste"]:
        autoteste()
    elif a[:1] == ["confere"] and len(a) >= 2:
        cmd_confere(a[1])
    elif a[:1] == ["espelho"] and len(a) >= 4:
        cmd_espelho(a[1], a[2], a[3])
    elif a[:1] == ["detecta"] and len(a) >= 2:
        cmd_detecta(a[1], op("--saida", TABELA), op("--cache", os.path.join(os.path.expanduser("~"), ".cache", "nutrelle_hf")), int(op("--limite", 0)) or None)
    elif a[:1] == ["folha"] and len(a) >= 3:
        cmd_folha(a[1], a[2], op("--tabela", TABELA), int(op("--semente", SEMENTE_FOLHA)))
    elif a[:1] == ["refaz"] and len(a) >= 2:
        cmd_refaz(a[1], op("--tabela", TABELA), op("--saida", TABELA))
    else:
        sys.exit(__doc__)
