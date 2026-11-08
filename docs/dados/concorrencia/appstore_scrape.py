import re, json, csv, time, html, urllib.request, os, sys
CORE = {6499510249:"Shotsy",6744178534:"MeAgain",6756984097:"Glapp",6758682710:"Pep AI",6504788281:"Pep (Shred Apps)",6670317407:"Zepbound & Mounjaro Tracker",6755669956:"GLP AI (OutliveAI)",6753301974:"OzemPro",6761790660:"Monju",6758341986:"MeuGLP",6757106230:"GlipOne",6755938737:"MounjaPRO",6756276105:"Mingo",6480417616:"Cal AI",341232718:"MyFitnessPal",297368629:"Lose It!",946099227:"Yazio",1553503471:"MacroFactor",1448277011:"Fitia",286906691:"Lifesum",1459260306:"Fastic",1467720176:"Simple",1038369065:"Flo",1503345959:"Balance",6667108028:"Lila (Peri Weight Loss Coach)",6743498501:"Raquel Menopausa",574794938:"Tecnonutri",347184248:"FatSecret",1064020872:"Foodvisor",1574239307:"SnapCalorie",1541193285:"Protein Pal",634598719:"Noom"}
SEC = {404299862:"Healthi",6744534749:"Lilly Health",1145935738:"Cronometer",287529757:"MyNetDiary",6479529917:"BitePal",1495829322:"Stardust",1482294997:"Wild.AI",1477621356:"Caria",6757102221:"Menelle",1519199698:"Health & Her",6451383937:"Clearblue Menopause Stage",1049234587:"Sweat",1195115431:"WebDiet",648610153:"Dietbox",6474789081:"Alimente-se",6468394305:"Protein Tracker (Oelfke)",1471632228:"ZOE",1581179653:"Found",6479331846:"Mochi Health",331308914:"WeightWatchers",1502526521:"Perda de Peso para Mulheres (Leap)",6760874134:"MounHealth",6761579709:"Canetinha",6755569693:"DreamMe",6748223909:"Glippy",839285684:"Workout for Women (Fast Builder)",1603811000:"BetterWeight (Habitics)",6747375157:"SnapCal"}
CORE_CC = ["us","br","mx","ar","cl","co","pe","pt"]; SEC_CC = ["us","br"]
UA = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15","Accept-Language":"pt-BR,pt;q=0.9,en;q=0.8"}
def get(url, tries=3):
    for i in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60); return r.read().decode("utf-8","replace"), r.status
        except urllib.error.HTTPError as e:
            if e.code == 404: return None, 404
            time.sleep(5+5*i)
        except Exception as e:
            time.sleep(5+5*i)
    return None, 0
def parse_page(h):
    out = {}
    m = re.search(r'"ratingCount":(\d+)', h); out["ratingCount"] = int(m.group(1)) if m else None
    m = re.search(r'"rating":([\d.]+),"ratingFormatted"', h); out["rating"] = float(m.group(1)) if m else None
    m = re.search(r'<title>([^<]+)</title>', h); out["title"] = html.unescape(m.group(1)).replace(" on the App Store","").replace(" na App Store","").replace(" en App Store","").strip() if m else None
    # IAP block: text-pair spans after "In-App Purchases"/"Compras dentro do app"
    iap = []
    seg = h
    for mm in re.finditer(r'<li class="svelte-1gyt6l2"><div class="text-pair svelte-1gyt6l2"><span>([^<]*)</span>\s*<span>([^<]*)</span>', seg):
        iap.append((html.unescape(mm.group(1)).strip(), html.unescape(mm.group(2)).strip()))
    out["iap"] = iap
    m = re.search(r'<meta name="description" content="([^"]*)"', h); out["desc"] = html.unescape(m.group(1))[:600] if m else None
    m = re.search(r'"sellerName":"([^"]*)"', h); out["seller"] = m.group(1) if m else None
    m = re.search(r'"releaseDate":"([^"]*)"', h); out["release"] = m.group(1)[:10] if m else None
    # privacy labels
    out["privacy_not_collected"] = bool(re.search(r'Data Not Collected|Dados não coletados|No se recopilan datos', h))
    out["privacy_linked"] = bool(re.search(r'Data Linked to You|Dados vinculados a você|Datos vinculados', h))
    out["privacy_tracking"] = bool(re.search(r'Data Used to Track You|Dados usados para rastrear|Datos usados para rastrearte', h))
    return out
def reviews(appid, cc):
    txt, st = get(f"https://itunes.apple.com/{cc}/rss/customerreviews/id={appid}/sortby=mostrecent/json")
    if not txt: return []
    try: e = json.loads(txt)["feed"].get("entry", [])
    except Exception: return []
    if isinstance(e, dict): e = [e]
    return [{"rating": x["im:rating"]["label"], "title": x["title"]["label"], "body": x["content"]["label"][:600], "date": x.get("updated",{}).get("label","")[:10], "version": x.get("im:version",{}).get("label","")} for x in e]
rows = []; revs = {}
os.makedirs("conc/pages", exist_ok=True)
for group, ccs in ((CORE, CORE_CC), (SEC, SEC_CC)):
    for appid, name in group.items():
        for cc in ccs:
            fn = f"conc/pages/{appid}_{cc}.html"
            if os.path.exists(fn): h = open(fn, encoding="utf-8").read()
            else:
                h, st = get(f"https://apps.apple.com/{cc}/app/id{appid}")
                if not h: rows.append([appid, name, cc, "NA", st, None, None, None, None, None, None, None, None]); print(name, cc, "status", st, flush=True); time.sleep(1); continue
                open(fn, "w", encoding="utf-8").write(h)
            p = parse_page(h)
            rows.append([appid, name, cc, p["title"], 200, p["rating"], p["ratingCount"], json.dumps(p["iap"], ensure_ascii=False), p["seller"], p["release"], p["privacy_not_collected"], p["privacy_linked"], p["privacy_tracking"], p["desc"]])
            print(name, cc, p["rating"], p["ratingCount"], p["iap"][:6], flush=True)
            if cc in ("us","br","mx"):
                rv = reviews(appid, cc)
                if rv: revs[f"{appid}_{cc}"] = {"app": name, "cc": cc, "reviews": rv}
            time.sleep(1.5)
with open("conc/appstore_storefronts.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["trackId","app","storefront","title","http","rating","ratingCount","iap_json","seller","release","privacy_not_collected","privacy_linked","privacy_tracking","description"]); w.writerows(rows)
json.dump(revs, open("conc/appstore_reviews.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
