#!/usr/bin/env python3
"""Summarize Google Trends CSVs produced by trends.py into comparable metrics."""
import csv, json, glob, os, sys, statistics as st
from datetime import datetime

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
MONTHS_PT = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}

def parse_date(s):
    # formats: "8 de nov. de 2021" (weekly, 5y), "12 de out." / "12 de out. de 2026" (daily/weekly 12m), "1 de nov. de 2021 – 7 de nov. de 2021"
    s = s.split("–")[0].strip()
    parts = [p.strip(". ") for p in s.replace(" de ", " ").split()]
    try:
        d = int(parts[0]); m = MONTHS_PT[parts[1][:3].lower()]
        y = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 2026
        return datetime(y, m, d)
    except Exception:
        return None

def load(fn):
    rows = list(csv.reader(open(fn, encoding="utf-8")))
    hdr = rows[0]; kws = hdr[1:-1]
    data = []
    for r in rows[1:]:
        d = parse_date(r[0])
        vals = [int(v) if v.isdigit() else 0 for v in r[1:1+len(kws)]]
        partial = r[-1] == "True"
        data.append((d, vals, partial))
    return kws, data

def summarize(fn):
    kws, data = load(fn)
    data = [x for x in data if x[0] is not None]
    n = len(data)
    res = {}
    for i, k in enumerate(kws):
        series = [v[i] for _, v, _ in data]
        full = series[:-1] if data and data[-1][2] else series  # drop partial last point
        info = {"n": len(full), "avg": round(st.mean(full), 1) if full else None, "max": max(full) if full else None}
        tf = fn.split("__")[-1].replace(".csv", "")
        if tf == "today_5_y" and len(full) >= 104:
            first = full[:52]; last = full[-52:]
            info["avg_y1"] = round(st.mean(first), 1); info["avg_y5"] = round(st.mean(last), 1)
            info["chg_5y_pct"] = round((st.mean(last) / st.mean(first) - 1) * 100) if st.mean(first) > 0 else None
            # last 12m vs previous 12m
            prev = full[-104:-52]
            info["chg_12m_vs_prev_pct"] = round((st.mean(last) / st.mean(prev) - 1) * 100) if st.mean(prev) > 0 else None
            # peak date
            pk = max(range(len(full)), key=lambda j: full[j]); info["peak"] = data[pk][0].strftime("%Y-%m")
        elif tf == "today_12_m" and len(full) >= 40:
            last13 = full[-13:]; prev13 = full[-26:-13]; first13 = full[:13]
            info["avg_last3m"] = round(st.mean(last13), 1); info["avg_prev3m"] = round(st.mean(prev13), 1)
            info["avg_first3m"] = round(st.mean(first13), 1)
            info["chg_3m_vs_prev_pct"] = round((st.mean(last13) / st.mean(prev13) - 1) * 100) if st.mean(prev13) > 0 else None
            info["chg_last3m_vs_first3m_pct"] = round((st.mean(last13) / st.mean(first13) - 1) * 100) if st.mean(first13) > 0 else None
            pk = max(range(len(full)), key=lambda j: full[j]); info["peak"] = data[pk][0].strftime("%Y-%m-%d")
        elif tf == "today_3_m" and len(full) >= 60:
            a = full[:len(full)//2]; b = full[len(full)//2:]
            info["avg_first_half"] = round(st.mean(a), 1); info["avg_second_half"] = round(st.mean(b), 1)
            info["chg_half_pct"] = round((st.mean(b) / st.mean(a) - 1) * 100) if st.mean(a) > 0 else None
        res[k] = info
    return res

if __name__ == "__main__":
    summary = {}
    for fn in sorted(glob.glob(os.path.join(OUT, "*.csv"))):
        name = os.path.basename(fn)[:-4]
        try:
            summary[name] = summarize(fn)
        except Exception as e:
            summary[name] = {"error": str(e)}
    related = {}
    for fn in sorted(glob.glob(os.path.join(OUT, "*__related.json"))):
        related[os.path.basename(fn)] = json.load(open(fn))
    json.dump({"series": summary, "related": related}, open(os.path.join(OUT, "_summary.json"), "w"), ensure_ascii=False, indent=1)
    for name, s in summary.items():
        print("##", name)
        for k, v in s.items():
            print("  ", k, v)
