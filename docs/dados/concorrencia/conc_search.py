import json, urllib.request, urllib.parse, time, csv, sys
terms = {
 "us": ["glp-1 tracker","mounjaro","ozempic","zepbound","protein tracker","protein counter","calorie counter photo","ai calorie counter","macro tracker","menopause","perimenopause","women fitness","period tracker","weight loss women"],
 "br": ["mounjaro","ozempic","caneta emagrecedora","glp-1","proteína","contador de proteína","contador de calorias","calorias por foto","dieta","emagrecer","menopausa","treino feminino","ciclo menstrual"],
 "mx": ["ozempic","mounjaro","contador de calorias","proteina","menopausia"],
 "ar": ["ozempic","contador de calorias","menopausia"],
}
seen = {}
rows = []
for cc, ts in terms.items():
    for t in ts:
        url = "https://itunes.apple.com/search?" + urllib.parse.urlencode({"term": t, "country": cc, "entity": "software", "limit": 25})
        try:
            d = json.load(urllib.request.urlopen(url, timeout=30))
        except Exception as e:
            print("ERR", cc, t, e); time.sleep(3); continue
        for i, r in enumerate(d.get("results", [])):
            rows.append([cc, t, i+1, r["trackId"], r["trackName"], r.get("sellerName"), r.get("userRatingCount"), round(r.get("averageUserRating", 0) or 0, 2), r.get("formattedPrice"), r.get("releaseDate", "")[:10], r.get("currentVersionReleaseDate","")[:10], r.get("primaryGenreName"), r.get("bundleId")])
        time.sleep(1.2)
with open("conc/itunes_search_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["storefront","term","rank","trackId","trackName","seller","ratingCount","rating","price","released","lastUpdate","genre","bundleId"]); w.writerows(rows)
print(len(rows), "rows")
