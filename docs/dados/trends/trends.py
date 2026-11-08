#!/usr/bin/env python3
"""Fetch Google Trends data (interest over time, related queries) via the
undocumented trends.google.com API and save CSV/JSON files for auditing."""
import json, sys, time, os, csv, random, urllib.parse, http.cookiejar, urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "./trends_out"
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "_log.txt"), "a")

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
                     ("Accept-Language", "pt-BR,pt;q=0.9,en;q=0.8")]

def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()

def refresh_cookie():
    opener.open("https://trends.google.com/trends/?geo=BR", timeout=30).read()
    time.sleep(1.5)

def get(url, tries=6):
    for i in range(tries):
        try:
            r = opener.open(url, timeout=60)
            body = r.read().decode("utf-8", "replace")
            return body
        except urllib.error.HTTPError as e:
            log(f"HTTP {e.code} on try {i+1} for {url[:120]}")
            if e.code == 429:
                time.sleep(20 + 15 * i); refresh_cookie()
            elif e.code in (400, 404):
                return None
            else:
                time.sleep(5 + 5 * i)
        except Exception as e:
            log(f"ERR {e} try {i+1}"); time.sleep(5 + 5 * i)
    return None

def strip(body):
    return json.loads(body.split("\n", 1)[1]) if body.startswith(")]}'") else json.loads(body)

def explore(keywords, geo, timeframe):
    req = {"comparisonItem": [{"keyword": k, "geo": geo, "time": timeframe} for k in keywords],
           "category": 0, "property": ""}
    url = "https://trends.google.com/trends/api/explore?" + urllib.parse.urlencode(
        {"hl": "pt-BR", "tz": "180", "req": json.dumps(req, ensure_ascii=False)})
    body = get(url)
    return strip(body) if body else None

def widget(kind, w):
    url = f"https://trends.google.com/trends/api/widgetdata/{kind}?" + urllib.parse.urlencode(
        {"hl": "pt-BR", "tz": "180", "req": json.dumps(w["request"], ensure_ascii=False), "token": w["token"]})
    body = get(url)
    return strip(body) if body else None

def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s)[:60]

def run(name, keywords, geo, timeframe, related=True):
    fn = os.path.join(OUT, f"{slug(name)}__{geo or 'WORLD'}__{slug(timeframe)}")
    if os.path.exists(fn + ".csv"):
        log(f"skip {fn}"); return
    log(f"=== {name} | {keywords} | {geo} | {timeframe}")
    ex = explore(keywords, geo, timeframe)
    if not ex:
        log("explore failed"); return
    time.sleep(random.uniform(2.5, 4.5))
    widgets = ex["widgets"]
    tl = [w for w in widgets if w["id"] == "TIMESERIES"]
    if tl:
        d = widget("multiline", tl[0])
        if d:
            rows = d["default"]["timelineData"]
            with open(fn + ".csv", "w", newline="") as f:
                wr = csv.writer(f); wr.writerow(["date"] + keywords + ["isPartial"])
                for r in rows:
                    wr.writerow([r["formattedAxisTime"]] + r["value"] + [r.get("isPartial", False)])
            # averages
            if rows:
                avgs = [round(sum(r["value"][i] for r in rows) / len(rows), 1) for i in range(len(keywords))]
                log(f"avg: {dict(zip(keywords, avgs))}")
        time.sleep(random.uniform(2.5, 4.5))
    if related:
        rq = [w for w in widgets if w["id"].startswith("RELATED_QUERIES")]
        rel = {}
        for w in rq:
            kw = w["request"]["restriction"]["complexKeywordsRestriction"]["keyword"][0]["value"]
            d = widget("relatedsearches", w)
            if d:
                out = {}
                for rl in d["default"]["rankedList"]:
                    key = "top" if not out else "rising"
                    out[key] = [(x["query"], x["formattedValue"]) for x in rl["rankedKeyword"]]
                rel[kw] = out
                log(f"related[{kw}] rising: {out.get('rising', [])[:8]}")
            time.sleep(random.uniform(2.5, 4.5))
        with open(fn + "__related.json", "w") as f:
            json.dump(rel, f, ensure_ascii=False, indent=1)
    geo_w = [w for w in widgets if w["id"] == "GEO_MAP"]
    if geo_w and geo == "":
        d = widget("comparedgeo", geo_w[0])
        if d:
            with open(fn + "__geo.json", "w") as f:
                json.dump(d, f, ensure_ascii=False)
        time.sleep(random.uniform(2.5, 4.5))

if __name__ == "__main__":
    refresh_cookie()
    plan = json.load(open(sys.argv[2]))
    for item in plan:
        run(item["name"], item["kw"], item.get("geo", "BR"), item.get("time", "today 5-y"), item.get("related", True))
        time.sleep(random.uniform(3, 6))
    log("DONE")
