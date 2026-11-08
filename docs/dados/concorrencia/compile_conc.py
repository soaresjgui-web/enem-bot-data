#!/usr/bin/env python3
"""Compile App Store storefront data into per-app/per-country price tables and review complaint summaries."""
import csv, json, re, collections, statistics as st

CUR = {"us":"USD","br":"BRL","mx":"MXN","ar":"ARS","cl":"CLP","co":"COP","pe":"PEN","pt":"EUR"}

def plan_type(name):
    n = name.lower()
    if re.search(r"week|seman|hebdo|7 d", n): return "weekly"
    if re.search(r"lifetime|vital|forever|de por vida|unlimited|ilimit", n): return "lifetime/unlimited"
    if re.search(r"year|anual|annual|12 m|1 ano|1 año|yearly", n): return "yearly"
    if re.search(r"month|mensal|mensual|monthly|1 mês|1 mes|30 d", n): return "monthly"
    if re.search(r"quarter|trimest|3 m", n): return "quarterly"
    if re.search(r"6 m|semest|semi", n): return "semiannual"
    return "other"

def parse_price(s):
    s = s.replace("\xa0", " ").strip()
    m = re.search(r"([\d\.,]+)", s)
    if not m: return None
    num = m.group(1)
    # decide decimal separator
    if "," in num and "." in num:
        if num.rfind(",") > num.rfind("."): num = num.replace(".", "").replace(",", ".")
        else: num = num.replace(",", "")
    elif "," in num:
        # e.g. 7,09 (decimal) vs 1,299 (thousands) -> if 3 digits after comma assume thousands
        parts = num.split(",")
        num = num.replace(",", ".") if len(parts[-1]) == 2 else num.replace(",", "")
    try: return float(num)
    except: return None

rows = list(csv.DictReader(open("conc/appstore_storefronts.csv", encoding="utf-8")))
apps = collections.OrderedDict()
for r in rows:
    a = apps.setdefault(r["app"], {"id": r["trackId"], "cc": {}})
    if r["http"] != "200": continue
    iap = json.loads(r["iap_json"]) if r["iap_json"] else []
    plans = collections.defaultdict(list)
    for name, price in iap:
        p = parse_price(price)
        if p is not None: plans[plan_type(name)].append(p)
    a["cc"][r["storefront"]] = {"rating": r["rating"], "n": r["ratingCount"], "title": r["title"],
        "plans": {k: sorted(set(v)) for k, v in plans.items()}, "iap_raw": iap,
        "priv_none": r["privacy_not_collected"], "priv_linked": r["privacy_linked"], "priv_track": r["privacy_tracking"], "desc": r["description"]}

# implied FX from apps present in both US and other storefront with same plan types (use min monthly & min yearly)
fx = collections.defaultdict(list)
for app, a in apps.items():
    us = a["cc"].get("us")
    if not us: continue
    for cc, d in a["cc"].items():
        if cc == "us": continue
        for pt in ("monthly", "yearly"):
            if us["plans"].get(pt) and d["plans"].get(pt) and len(us["plans"][pt]) == len(d["plans"][pt]):
                for u, l in zip(us["plans"][pt], d["plans"][pt]):
                    if u > 0: fx[cc].append(l / u)
fx_med = {cc: round(st.median(v), 3) for cc, v in fx.items() if v}

out = {"apps": apps, "implied_fx_local_per_usd": fx_med, "fx_samples": {k: len(v) for k, v in fx.items()}}
json.dump(out, open("conc/compiled_prices.json", "w"), ensure_ascii=False, indent=1)

print("Implied Apple FX (local per USD, median over matched tiers):", fx_med, {k: len(v) for k, v in fx.items()})
print()
for app, a in apps.items():
    print(f"## {app} (id {a['id']})")
    for cc in ["us","br","mx","ar","cl","co","pe","pt"]:
        d = a["cc"].get(cc)
        if not d: continue
        pl = {k: v for k, v in d["plans"].items()}
        print(f"  {cc}: r={d['rating']} n={d['n']} | " + " | ".join(f"{k}: {v}" for k, v in pl.items()) + f" | priv: none={d['priv_none']} linked={d['priv_linked']} track={d['priv_track']}")

# reviews: complaints
try:
    revs = json.load(open("conc/appstore_reviews.json", encoding="utf-8"))
except Exception:
    revs = {}
print("\n\n### REVIEWS (<=3 stars) summary")
KW = {"preço/caro": r"caro|expensive|price|preço|pricey|cobr|charged|cobrança|subscription|assinatura|refund|reembolso|cancel",
      "precisão/IA": r"accura|precis|wrong|errad|incorrect|estimate|estimativ|calor|photo|foto|scan",
      "bugs/sync": r"bug|crash|trava|fecha|sync|sincron|login|erro|error|glitch|slow|lento",
      "paywall/trial": r"trial|teste grátis|paywall|free|grátis|gratuito|locked|bloquead",
      "ads": r"\bads?\b|anúncio|propaganda",
      "falta recurso": r"missing|falta|wish|gostaria|would be|should|could|poderia|não tem|no tiene|needs",
      "privacidade": r"privac|data|dados|account|conta"}
for key, d in revs.items():
    rv = d["reviews"]; low = [x for x in rv if int(x["rating"]) <= 3]
    if not rv: continue
    cnt = collections.Counter()
    for x in low:
        t = (x["title"] + " " + x["body"]).lower()
        for k, pat in KW.items():
            if re.search(pat, t): cnt[k] += 1
    dist = collections.Counter(int(x["rating"]) for x in rv)
    print(f"- {d['app']} [{d['cc']}]: n={len(rv)} recentes, distrib={dict(sorted(dist.items()))}, ≤3★={len(low)}, temas={dict(cnt.most_common(4))}")
    for x in low[:3]:
        print(f"     ★{x['rating']} {x['title'][:60]} — {x['body'][:180].replace(chr(10),' ')}")
