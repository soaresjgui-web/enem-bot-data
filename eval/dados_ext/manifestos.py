#!/usr/bin/env python3
"""Gera manifest.csv normalizado para cada dataset externo baixado em scratchpad/dados_ext/<nome>/.

Colunas: dataset,image_path,dish_id,split,mass_g,protein_g,kcal,ingredients,camera,has_depth,license,eval_holdout,notes
  - mass_g: massa pesada (balança) da porção que aparece na imagem;
  - protein_g / kcal: só quando o dataset traz (Nutrition5k: USDA por ingrediente pesado). Vazio caso contrário;
  - ingredients: "nome:gramas;..." (gramas medidas);
  - eval_holdout=1: prato do Nutrition5k usado nos nossos gabaritos de avaliação (NUNCA treinar com ele).

Uso: python3 -I manifestos.py <dir dados_ext> <dir docs/dados/precisao> [n5k_side ecustfd fpb]
"""
import csv, glob, os, re, sys
import xml.etree.ElementTree as ET

COLS = ["dataset", "image_path", "dish_id", "split", "mass_g", "protein_g", "kcal", "ingredients",
        "camera", "has_depth", "license", "eval_holdout", "notes"]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    print(path, len(rows))


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- Nutrition5k side angles
def holdout_ids(precisao):
    ids = set()
    for fn in ["gabarito_pratos_v2.csv", "n5k_v2_amostra.csv", "gabarito_pratos.csv", "gabarito_v2.csv",
               "gabarito.csv", "gabarito_componentes.csv", "coleta_n5k.py", "coleta_n5k_v2.py"]:
        p = os.path.join(precisao, fn)
        if os.path.exists(p):
            ids |= set(re.findall(r"dish_\d{10}", open(p, encoding="utf-8", errors="ignore").read()))
    return ids


def n5k_meta(meta):
    ingr_names = {}
    for r in csv.DictReader(open(os.path.join(meta, "ingredients_metadata.csv"))):
        ingr_names[int(r["id"])] = r["ingr"]
    dishes = {}
    for cafe in ["cafe1", "cafe2"]:
        for line in open(os.path.join(meta, f"dish_metadata_{cafe}.csv")):
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            did = p[0]
            kcal, mass, fat, carb, prot = map(float, p[1:6])
            ing = []
            sp, sk = 0.0, 0.0
            for i in range(6, len(p) - 6, 7):
                iid, name, g, c, f_, cb, pr = p[i:i + 7]
                if not name:
                    try:
                        name = ingr_names.get(int(iid.split("_")[-1]), "")
                    except ValueError:
                        name = ""
                ing.append((name, float(g)))
                sp += float(pr)
                sk += float(c)
            recon = False
            if cafe == "cafe2" and prot == 0 and kcal == 0:  # totais zerados no café 2: soma dos ingredientes
                prot, kcal, recon = sp, sk, True
            bad = any(n in ("", "deprecated", "plate only") for n, _ in ing)
            dishes[did] = dict(cafe=cafe, mass=mass, prot=prot, kcal=kcal, ing=ing, recon=recon, bad=bad)
    return dishes


def do_n5k_side(base, precisao):
    d = os.path.join(base, "n5k_side")
    meta = os.path.join(d, "meta")
    dishes = n5k_meta(meta)
    hold = holdout_ids(precisao)
    split = {}
    for sp in ["train", "test"]:
        for x in open(os.path.join(meta, f"rgb_{sp}_ids.txt")):
            split[x.strip()] = sp
    rows = []
    nodata = 0
    for img in sorted(glob.glob(os.path.join(d, "frames", "dish_*", "*.jpg"))):
        did = img.split(os.sep)[-2]
        m = dishes.get(did)
        if m is None:
            nodata += 1
            continue
        notes = [m["cafe"], "frame=" + os.path.basename(img)[:-4]]
        if m["recon"]:
            notes.append("totais_reconstruidos_pelos_ingredientes")
        if m["bad"]:
            notes.append("ingrediente_deprecated_ou_sem_nome")
        rows.append(dict(dataset="nutrition5k_side", image_path=img, dish_id=did, split=split.get(did, ""),
                         mass_g=fmt(m["mass"], 1), protein_g=fmt(m["prot"]), kcal=fmt(m["kcal"], 1),
                         ingredients=";".join(f"{n}:{g:.1f}" for n, g in m["ing"]),
                         camera="side", has_depth=0, license="CC BY 4.0",
                         eval_holdout=int(did in hold), notes="|".join(notes)))
    print("n5k_side: frames sem metadado:", nodata, "| holdout ids:", len(hold))
    write(os.path.join(d, "manifest.csv"), rows)


# ---------------------------------------------------------------- ECUSTFD
def do_ecustfd(base):
    import xlrd
    d = os.path.join(base, "ecustfd")
    raw = os.path.join(d, "raw")
    wb = xlrd.open_workbook(os.path.join(raw, "density.xls"))
    w = {}  # (id, type) -> (g, vol)
    for s in wb.sheets():
        for r in range(1, s.nrows):
            v = s.row_values(r)
            if v[0]:
                w[(str(v[0]), str(v[1]))] = (float(v[3]), float(v[2]))
    split = {}
    for sp in ["train", "val", "test"]:
        for x in open(os.path.join(raw, "ImageSets", "Main", f"{sp}.txt")):
            split[x.strip()] = sp
    rows, miss = [], 0
    for xml in sorted(glob.glob(os.path.join(raw, "Annotations", "*.xml"))):
        stem = os.path.basename(xml)[:-4]
        m = re.match(r"([a-z_]+\d{3})([ST])\((\d+)\)", stem)
        if not m:
            continue
        oid, view = m.group(1), m.group(2)
        names = [o.find("name").text.strip() for o in ET.parse(xml).getroot().iter("object")]
        foods = [n for n in names if n != "coin"]
        ing, ok = [], True
        for n in foods:
            key = (oid, n)
            if key not in w:
                ok = False
                break
            ing.append((n, w[key][0]))
        if not ok or not ing:
            miss += 1
            continue
        notes = "vol_ml=" + "+".join(f"{w[(oid, n)][1]:.0f}" for n in foods) + "|moeda_25mm"
        if stem in ("mix002T(2)", "mix005S(4)"):
            notes += "|sem_moeda"
        img = os.path.join(raw, "JPEGImages", stem + ".JPG")
        rows.append(dict(dataset="ecustfd", image_path=img, dish_id=oid, split=split.get(stem, ""),
                         mass_g=fmt(sum(g for _, g in ing), 1), ingredients=";".join(f"{n}:{g:.1f}" for n, g in ing),
                         camera="side" if view == "S" else "overhead", has_depth=0,
                         license="nao declarada (README: 'free public dataset', citar Liang & Li 2017)",
                         eval_holdout=0, notes=notes))
    print("ecustfd: imagens sem peso casado:", miss)
    write(os.path.join(d, "manifest.csv"), rows)


# ---------------------------------------------------------------- Food Portion Benchmark (ISSAI)
def do_fpb(base):
    d = os.path.join(base, "fpb")
    root = os.path.join(d, "raw", "FPB_Dataset", "RGB")
    names = None
    for line in open(os.path.join(root, "data.yaml")):
        if line.startswith("names:"):
            names = eval(line.split(":", 1)[1].strip(), {"__builtins__": {}})  # lista literal
    rows = []
    stats = {}
    for sp in ["train", "val", "test"]:
        for lab in sorted(glob.glob(os.path.join(root, sp, "labels", "*.txt"))):
            stem = os.path.basename(lab)[:-4]
            imgs = glob.glob(os.path.join(root, sp, "images", glob.escape(stem) + ".*"))
            if not imgs:
                continue
            objs = []
            for line in open(lab):
                p = line.split()
                if len(p) >= 6:
                    objs.append((names[int(p[0])], float(p[5])))
                elif len(p) == 5:
                    objs.append((names[int(p[0])], -1.0))
            known = [o for o in objs if o[1] > 0]
            st = stats.setdefault(sp, [0, 0, 0])
            st[0] += 1
            if not objs or len(known) < len(objs):
                st[1] += 1
                continue  # sem peso (rótulo oculto/ausente) ou peso parcial
            st[2] += 1
            mrf = re.match(r"(.+?)_(small|average|big)_", stem)
            port = mrf.group(2) if mrf else ""
            src = "realsense" if "_RGB_" in stem and "IMG_2024" not in stem else ("smartphone" if "IMG_" in stem else "")
            rows.append(dict(dataset="fpb_issai", image_path=imgs[0], dish_id=stem.split(".rf.")[0], split=sp,
                             mass_g=fmt(sum(g for _, g in known), 1),
                             ingredients=";".join(f"{n}:{g:.1f}" for n, g in known),
                             camera="handheld/lab (top+4 laterais; angulo nao rotulado)", has_depth=0,
                             license="CC BY-NC 4.0 (NAO comercial: so P&D interno/avaliacao)", eval_holdout=0,
                             notes="|".join(x for x in [f"porcao={port}" if port else "", "roboflow_export"] if x)))
    print("fpb: [total, sem_peso, com_peso] por split:", stats)
    write(os.path.join(d, "manifest.csv"), rows)


if __name__ == "__main__":
    base, precisao = sys.argv[1], sys.argv[2]
    which = sys.argv[3:] or ["n5k_side", "ecustfd", "fpb"]
    if "n5k_side" in which:
        do_n5k_side(base, precisao)
    if "ecustfd" in which:
        do_ecustfd(base)
    if "fpb" in which:
        do_fpb(base)
