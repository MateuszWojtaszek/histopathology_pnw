#!/usr/bin/env python3
"""Podgląd treningu AL na CAMELYON17: runda, epoka, loss, tempo, ETA, wyniki rund, GPU, kolejka Slurm.

Samodzielny skrypt (tylko biblioteka standardowa i opcjonalnie pandas), można go uruchomić na węźle logowania:

    python3 camelyon17_monitor.py --mode full                 # jednorazowo
    python3 camelyon17_monitor.py --mode full --watch 30      # odświeżanie co 30 s
    python3 camelyon17_monitor.py --mode full --log c17-al-12345.log   # log Slurm z zadania

Czyta: outputs/al_random_c17/<mode>/seed*/status.json i result.json oraz log Cellpose
(domyślnie ~/.cellpose/run.log, albo plik z --log; logi treningu trafiają tam dzięki logger_setup()).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

EPOCH_RE = re.compile(r"(\d+), train_loss=([0-9.eE+-]+).*?time ([0-9.]+)s")


def fmt(sec: float | None) -> str:
    if sec is None or sec != sec or sec < 0:
        return "?"
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


def find_outputs(arg: str | None) -> Path:
    for c in [arg, os.environ.get("PNW_OUTPUTS_DIR"), Path.cwd() / "outputs"]:
        if c and Path(c).exists():
            return Path(c)
    raise SystemExit("Nie znaleziono katalogu outputs. Podaj --outputs.")


def newest_status(root: Path):
    cands = sorted(root.glob("seed*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return (cands[0].parent, json.loads(cands[0].read_text())) if cands else (None, None)


def parse_log(path: Path):
    # zwraca listę (epoka, loss, czas_s) dla BIEŻĄCEJ rundy (ostatni rosnący ciąg epok)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(max(path.stat().st_size - 3_000_000, 0))
        text = f.read().decode("utf-8", errors="ignore")
    pts = [(int(e), float(loss), float(t)) for e, loss, t in EPOCH_RE.findall(text)]
    seg = []
    for p in pts:
        if seg and p[0] <= seg[-1][0]:
            seg = []
        seg.append(p)
    return seg


def gpu_info() -> str:
    if not shutil.which("nvidia-smi"):
        return "brak nvidia-smi na tym węźle (na węźle logowania nie ma GPU; użyj: srun --jobid <ID> --overlap nvidia-smi)"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "nvidia-smi nic nie zwrócił"
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi: {e!r}"


def queue_info() -> str:
    if not shutil.which("squeue"):
        return ""
    try:
        return subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-o", "%i %j %T %M %l %R"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def report(a) -> None:
    out = find_outputs(a.outputs)
    root = out / "al_random_c17" / a.mode
    seed_dir, st = newest_status(root)
    print("=" * 72)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "| wyniki:", root)
    if st is None:
        print("Brak status.json. Trening jeszcze nie ruszył albo zły --mode/--outputs.")
        return

    print(f"seed {st.get('seed')} | urządzenie: {st.get('device')} | faza: {st.get('phase')} | "
          f"aktualizacja: {st.get('updated_at')}")
    if "cuda" not in str(st.get("device")):
        print("!!! Trening NIE używa CUDA. To główny powód bardzo długiego czasu. Zamów GPU (--gres=gpu:1).")

    n_rounds, rnd = st.get("n_rounds", 0), st.get("round", 0)
    done = sorted(seed_dir.glob("round_*/result.json"))
    results = [json.loads(p.read_text()) for p in done]
    print(f"runda {rnd}/{n_rounds}" + (f" ({st.get('n_labeled')} kafli)" if st.get("n_labeled") else "")
          + f" | epok: {st.get('n_epochs')}, wycinków/epokę: {st.get('nimg_per_epoch')}, batch: {st.get('batch_size')}")

    # bieżąca runda: epoka, loss, tempo
    log = Path(a.log).expanduser() if a.log else Path.home() / ".cellpose" / "run.log"
    pts = parse_log(log) if st.get("phase") == "training" else []
    n_epochs = st.get("n_epochs") or 0
    cur_total = None
    if pts:
        e, loss, t = pts[-1]
        if len(pts) >= 2 and pts[-1][0] > pts[-2][0]:
            spe = (pts[-1][2] - pts[-2][2]) / (pts[-1][0] - pts[-2][0])
        else:
            spe = t / max(e, 1)
        left = max(n_epochs - e, 0) * spe
        print(f"epoka {e}/{n_epochs} | train_loss {loss:.4f} | {spe:.1f} s/epokę | do końca rundy ~{fmt(left)} (log: {log})")
        cur_total = left
    elif st.get("phase") == "training":
        print(f"brak linii z postępem w {log} (Cellpose loguje co kilka epok; pierwszy wpis po ok. 5 epokach).")
    elif st.get("phase") == "evaluating":
        print("trwa ewaluacja na kaflach val po treningu rundy.")

    # ukończone rundy i ETA reszty
    if results:
        times = [r.get("round_time_s") or r.get("train_time_s", 0) for r in results]
        avg = sum(times) / len(times)
        remaining = max(n_rounds - rnd, 0)
        eta = (cur_total or 0) + remaining * avg
        target = st.get("target_pq")
        print(f"\nukończone rundy: {len(results)} | średnio {fmt(avg)} na rundę | pozostałe rundy: {remaining} | "
              f"szacowany koniec za ~{fmt(eta)}")
        print(f"{'kafli':>6} {'val PQ':>8} {'val AJI':>8} {'loss':>8} {'czas':>9}")
        for r in results:
            mark = "  <= poprzeczka" if target is not None and r["val_pq"] >= target else ""
            print(f"{r['n_labeled']:>6} {r['val_pq']:>8.4f} {r.get('val_aji', float('nan')):>8.4f} "
                  f"{r.get('final_train_loss', float('nan')):>8.4f} {fmt(r.get('round_time_s') or r.get('train_time_s')):>9}{mark}")
        if target is not None:
            print(f"poprzeczka PQ: {target:.4f}")
    elif st.get("phase") == "training" and cur_total is not None and n_rounds:
        print(f"szacunek całości (jeśli kolejne rundy tak samo długie): ~{fmt(cur_total + (n_rounds - rnd) * (n_epochs * spe))}")

    print("\nGPU:", gpu_info())
    q = queue_info()
    if q:
        print("\nkolejka Slurm:\n" + q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", default="full", choices=("smoke", "full"))
    ap.add_argument("--outputs", default=None, help="katalog outputs (domyślnie PNW_OUTPUTS_DIR albo ./outputs)")
    ap.add_argument("--log", default=None, help="plik logu (Slurm lub ~/.cellpose/run.log)")
    ap.add_argument("--watch", type=int, default=0, help="odświeżaj co N sekund")
    a = ap.parse_args()
    while True:
        report(a)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
