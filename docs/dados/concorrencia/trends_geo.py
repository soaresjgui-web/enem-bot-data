import json, urllib.parse, urllib.request, http.cookiejar, time, csv, statistics as st, os
cj=http.cookiejar.CookieJar(); op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders=[("User-Agent","Mozilla/5.0 (X11; Linux x86_64) Chrome/124.0"),("Accept-Language","pt-BR,pt;q=0.9")]
op.open("https://trends.google.com/trends/?geo=BR").read(); time.sleep(2)
def strip(b): return json.loads(b.split("\n",1)[1])
def explore(items, tf):
    req={"comparisonItem":[{"keyword":k,"geo":g,"time":tf} for k,g in items],"category":0,"property":""}
    u="https://trends.google.com/trends/api/explore?"+urllib.parse.urlencode({"hl":"pt-BR","tz":"180","req":json.dumps(req,ensure_ascii=False)})
    for i in range(12):
        try: return strip(op.open(u,timeout=60).read().decode())
        except Exception as e: print("retry",e); time.sleep(60+15*i)
def ml(w):
    u="https://trends.google.com/trends/api/widgetdata/multiline?"+urllib.parse.urlencode({"hl":"pt-BR","tz":"180","req":json.dumps(w["request"],ensure_ascii=False),"token":w["token"]})
    for i in range(12):
        try: return strip(op.open(u,timeout=60).read().decode())
        except Exception as e: print("retry",e); time.sleep(60+15*i)
os.makedirs("conc",exist_ok=True)
out=[]
plan=[("mounjaro",["BR","US","MX","AR","CL"]),("mounjaro",["BR","CO","PE","PT","ES"]),
      ("ozempic",["BR","US","MX","AR","CL"]),("ozempic",["BR","CO","PE","PT","ES"]),
      ("proteína",["BR","MX","AR","CL","CO"]),("glp-1",["BR","US","MX","AR","CO"]),
      ("menopausia",["MX","AR","CL","CO","PE"]),("perimenopausa",["BR","PT"]),("perimenopausia",["MX","AR","CL","CO","ES"])]
for kw,geos in plan:
    for tf in ["today 12-m","today 3-m"]:
        ex=explore([(kw,g) for g in geos],tf)
        if not ex: continue
        time.sleep(3)
        w=[x for x in ex["widgets"] if x["id"]=="TIMESERIES"][0]
        d=ml(w)
        if not d: continue
        rows=d["default"]["timelineData"]
        full=[r for r in rows if not r.get("isPartial")]
        for i,g in enumerate(geos):
            vals=[r["value"][i] for r in full]
            out.append([kw,g,tf,round(st.mean(vals),1),max(vals),len(vals)])
            print(kw,g,tf,round(st.mean(vals),1))
        with open(f"conc/trends_geo__{kw}__{'_'.join(geos)}__{tf.replace(' ','_')}.csv","w",newline="") as f:
            wr=csv.writer(f); wr.writerow(["date"]+[f"{kw}@{g}" for g in geos])
            for r in rows: wr.writerow([r["formattedAxisTime"]]+r["value"])
        time.sleep(4)
with open("conc/trends_geo_summary.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["keyword","geo","timeframe","avg","max","n"]); wr.writerows(out)
print("DONE")
