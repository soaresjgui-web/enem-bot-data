#!/usr/bin/env python3
"""Extrai poucos quadros por câmera dos vídeos side_angles do Nutrition5k sem baixar o vídeo inteiro.

Cada vídeo (camera_{A..D}.h264, H.264 Annex B cru, 1920x1080, ~141 quadros, keyframes a cada 60)
mostra o prato girando. Pegamos, por câmera:
  - "ini": o último quadro decodificável dos primeiros 600 KB (≈ quadro 8 do GOP inicial);
  - "fim": o keyframe (IDR) mais próximo do fim (≈ quadro 120, prato já girado ~240°), localizado
    procurando o SPS (NAL tipo 7) nos últimos ~30% do arquivo (requisição HTTP Range).
Saída: <out>/<dish_id>/<cam>_<ini|fim>.jpg em 960x540, e um log CSV.

Uso: python3 -I n5k_side_frames.py <sizes.txt (gsutil ls -l)> <out_dir> [--workers 24] [--limit N]
"""
import argparse, csv, os, re, subprocess, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles/"
SPS_RE = re.compile(rb"\x00\x00\x00\x01[\x07\x27\x47\x67]")  # NAL tipo 7 (SPS), qualquer nal_ref_idc


def rfind_sps(data, end=None):
    last = -1
    for m in SPS_RE.finditer(data, 0, len(data) if end is None else end):
        last = m.start()
    return last


def fetch(url, start, end):
    r = subprocess.run(["curl", "-sf", "--retry", "4", "--retry-delay", "2", "-m", "120",
                        "-r", f"{start}-{end}", url], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl {r.returncode} {url} {start}-{end}")
    return r.stdout


def decode(data, out_jpg, last=False):
    with tempfile.NamedTemporaryFile(suffix=".h264", delete=False) as f:
        f.write(data)
        p = f.name
    try:
        vf = "scale=960:540"
        if last:
            # decodifica tudo e mantém só o último quadro completo
            cmd = ["ffmpeg", "-v", "error", "-y", "-threads", "1", "-f", "h264", "-i", p, "-vf", vf,
                   "-q:v", "3", "-update", "1", out_jpg]
        else:
            cmd = ["ffmpeg", "-v", "error", "-y", "-threads", "1", "-f", "h264", "-i", p, "-vf", vf,
                   "-frames:v", "1", "-q:v", "3", out_jpg]
        subprocess.run(cmd, capture_output=True, timeout=120)
        return os.path.exists(out_jpg) and os.path.getsize(out_jpg) > 5000
    finally:
        os.unlink(p)


def process(dish, cam, size, outdir):
    url = f"{BASE}{dish}/camera_{cam}.h264"
    d = os.path.join(outdir, dish)
    os.makedirs(d, exist_ok=True)
    res = []
    nbytes = 0
    # ini
    j = os.path.join(d, f"{cam}_ini.jpg")
    if not os.path.exists(j):
        data = fetch(url, 0, min(size, 600_000) - 1)
        nbytes += len(data)
        # corta no último início de NAL para não entregar quadro truncado
        cut = data.rfind(b"\x00\x00\x00\x01")
        ok = decode(data[:cut] if cut > 0 else data, j, last=True)
    else:
        ok = True
    res.append((dish, cam, "ini", ok))
    # fim
    j = os.path.join(d, f"{cam}_fim.jpg")
    if not os.path.exists(j):
        ok = False
        start = int(size * 0.70)
        data = fetch(url, start, size - 1)
        nbytes += len(data)
        pos = rfind_sps(data)
        # IDR precisa de ~300 KB depois do SPS; se o último SPS está muito no fim, usa o anterior
        while pos >= 0 and len(data) - pos < 200_000:
            pos = rfind_sps(data, pos)
        if pos >= 0:
            ok = decode(data[pos:pos + 700_000], j)
        else:
            # vídeos curtos (~33 quadros) só têm o keyframe inicial: baixa o resto e decodifica o último quadro
            head = fetch(url, 0, start - 1)
            nbytes += len(head)
            ok = decode(head + data, j, last=True)
        res.append((dish, cam, "fim", ok))
    else:
        res.append((dish, cam, "fim", True))
    return res, nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = []
    for line in open(a.sizes):
        m = re.match(r"\s*(\d+)\s+\S+\s+gs://\S+/side_angles/(dish_\d+)/camera_([A-D])\.h264", line)
        if m:
            jobs.append((m.group(2), m.group(3), int(m.group(1))))
    jobs.sort()
    if a.limit:
        dishes = sorted({j[0] for j in jobs})[:a.limit]
        jobs = [j for j in jobs if j[0] in set(dishes)]
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "_log.csv"), "a", newline="")
    w = csv.writer(log)
    lock = threading.Lock()
    tot = 0
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, d, c, s, a.out): (d, c) for d, c, s in jobs if s > 100_000}
        for f in as_completed(futs):
            try:
                res, nb = f.result()
            except Exception as e:
                res, nb = [(*futs[f], "erro", str(e)[:80])], 0
            with lock:
                for r in res:
                    w.writerow(r)
                tot += nb
                done += 1
                if done % 200 == 0:
                    log.flush()
                    print(f"{done}/{len(futs)} videos, {tot/1e9:.1f} GB", flush=True)
    log.close()
    print("fim", done, tot / 1e9)


if __name__ == "__main__":
    main()
