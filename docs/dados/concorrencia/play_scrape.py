import re, csv, time, html, urllib.request, os, json
PK = {"app.meagain.app":"MeAgain","com.monju.app":"Monju","com.agoraLabs.ozempicTracker":"Mingo","co.wiserapps.glpTracker":"GLP-1 Tracker (Wiserapps)","com.bilaal5279.weightly":"Weightly","com.glp.monitor":"GLP Monitor","com.glp1health.control":"GLP1 Health Control","ai.glpeak.app":"GLPeak","com.dmfinity.glp1tracker":"GLP1 tracker (dmfinity)","br.com.consultoria.w12.developer.android.mounjapro":"MounjaPRO","br.com.glpprogress.app":"GLP Progress","com.shotsy.app":"Shotsy?","com.everyday.shottracker":"Shotsy?2","com.segaritz.ozempro":"OzemPro?","com.outliveai.glpai":"GLP AI?","io.glapp":"Glapp?","com.myfitnesspal.android":"MyFitnessPal","com.fitnow.loseit":"Lose It!","com.yazio.android":"Yazio","com.sbs.diet":"MacroFactor","caloriecounter.food.macrotracker.calai":"Cal AI","com.fatsecret.android":"FatSecret","com.cronometer.android.gold":"Cronometer","com.appediet.appediet":"Appediet","br.com.tecnonutri.app":"Tecnonutri","br.com.dietaetreino":"Dieta e Treino","br.com.webdiet.webdiet":"WebDiet","com.craftbox.dietbox":"Dietbox","com.lealapps.alimentese":"Alimente-se","com.balance_app.app":"Balance","co.knecht.menopause_app":"Knecht menopause","org.iggymedia.periodtracker":"Flo","com.clue.android":"Clue","com.amapps.simple":"Simple","de.fastic.app":"Fastic","com.sillens.shapeupclub":"Lifesum","com.foodvisor.foodvisor":"Foodvisor","com.snapcalorie.app":"SnapCalorie?","com.nutrition.technologies.Fitia":"Fitia?","com.wsl.noom":"Noom","com.stardust.app":"Stardust?","com.raquelmenopausa.raquelapp":"Raquel Menopausa?","com.glipone":"GlipOne?","app.replit.meuglp":"MeuGLP?","com.pookies.food.ai":"BitePal?","com.sillens.iShape":"Lifesum?","com.leapfitness.womenlose":"Perda de Peso Mulheres (Leap)"}
GL = [("BR","pt_BR"),("US","en_US"),("MX","es_MX"),("AR","es_AR")]
UA = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
def get(url):
    for i in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60); return r.read().decode("utf-8","replace"), r.status
        except urllib.error.HTTPError as e:
            if e.code == 404: return None, 404
            time.sleep(4+4*i)
        except Exception: time.sleep(4+4*i)
    return None, 0
def parse(h):
    t = html.unescape(re.sub(r"<[^>]+>", " ", h)); t = re.sub(r"\s+", " ", t)
    o = {}
    m = re.search(r"(\d[\d,\.]*)\s*star", t); o["rating"] = m.group(1) if m else None
    m = re.search(r"([\d,\.]+\s*(?:mil|mi|K|M)?)\s*(?:avaliações|reviews|opiniones|reseñas)", t); o["reviews"] = m.group(1).strip() if m else None
    m = re.search(r"([\d,\.]+\s*(?:mil|mi|K|M)?\+?)\s*(?:downloads|Downloads|descargas|Descargas)", t); o["installs"] = m.group(1).strip() if m else None
    m = re.search(r"((?:R\$|US\$|\$|MX\$|ARS|\$ ?)\s?[\d,\.]+(?:\s*[–-]\s*(?:R\$|US\$|\$|MX\$|ARS)?\s?[\d,\.]+)?)\s*(?:por item|per item|por elemento|por artículo)", t); o["iap_range"] = m.group(1) if m else None
    m = re.search(r'<title[^>]*>([^<]*)</title>', h); o["title"] = html.unescape(m.group(1)).split(" - A")[0].split(" – A")[0] if m else None
    o["ads"] = bool(re.search(r"Contém anúncios|Contains ads|Contiene anuncios", t))
    m = re.search(r'Atualizado em ([^A-Z]{3,40}?) |Updated on ([A-Za-z]{3} \d{1,2}, \d{4})|Actualizado el (\d{1,2} [a-z]{3} \d{4})', t); o["updated"] = next((g for g in (m.groups() if m else []) if g), None)
    return o
rows = []
for pk, name in PK.items():
    for gl, hl in GL:
        h, st = get(f"https://play.google.com/store/apps/details?id={pk}&hl={hl}&gl={gl}")
        if not h: rows.append([pk, name, gl, st, None, None, None, None, None, None, None]); print(name, gl, st, flush=True); time.sleep(1); continue
        p = parse(h); rows.append([pk, name, gl, 200, p["title"], p["rating"], p["reviews"], p["installs"], p["iap_range"], p["ads"], p["updated"]])
        print(name, gl, p, flush=True); time.sleep(1.5)
        if st == 404: break
with open("conc/play_storefronts.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["package","app","gl","http","title","rating","reviews","installs","iap_range","ads","updated"]); w.writerows(rows)
print("DONE")
