# Losowe aktywne uczenie na kaflach CAMELYON17: student Cellpose-SAM uczy się na maskach wyroczni `cpsam`
# i jest porównywany z wyrocznią. Skrypt zamiast notebooka 03.
# Umieść w: src/histopathology_pnw/al/camelyon17_train.py
#
#   uv run python -m histopathology_pnw.al.camelyon17_train --mode smoke
#   uv run python -m histopathology_pnw.al.camelyon17_train --mode full --seeds 0 1 2
#
# Wymaga wcześniej: maski + podział + kafle (datasets/camelyon17_prep.py) oraz wag cpsam i SAM (dvc pull).
#
# Jak "równamy studenta do wyroczni" (CAMELYON17 nie ma masek jąder, więc nie ma GT):
#  - student uczy się na maskach wyroczni z coraz większej liczby losowo wybranych kafli,
#  - po każdej rundzie liczymy PQ studenta względem masek wyroczni na kaflach val,
#  - poprzeczka = TARGET_FRAC x samozgodność wyroczni (PQ maski kafla i maski jego odbicia),
#  - stop, gdy student ją osiągnie albo skończy się pula,
#  - na końcu: zgodność z wyrocznią na kaflach test oraz porównanie student vs wyrocznia vs GT na MoNuSeg.
from __future__ import annotations

import argparse
import json

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from cellpose import io as cp_io  # noqa: E402

from histopathology_pnw.al import camelyon17_random_loop as al  # noqa: E402
from histopathology_pnw.config import Settings  # noqa: E402
from histopathology_pnw.datasets import camelyon17 as c17  # noqa: E402
from histopathology_pnw.device import get_device  # noqa: E402
from histopathology_pnw.inference import predict  # noqa: E402
from histopathology_pnw.metrics import evaluate_all, summarize  # noqa: E402
from histopathology_pnw.models.oracle import load_oracle  # noqa: E402
from histopathology_pnw.models.student import load_student  # noqa: E402


def build_config(args, seed: int) -> al.TileALConfig:
    if args.mode == "smoke":
        return al.TileALConfig(seed=seed, initial=2, step=2, max_labeled=4, n_epochs=1, nimg_per_epoch=2,
                               batch_size=1, stop_at_target=False,
                               gradient_checkpointing=not args.no_checkpointing)
    return al.TileALConfig(seed=seed, initial=args.initial, step=args.step, max_labeled=args.max_labeled,
                           n_epochs=args.n_epochs, nimg_per_epoch=args.nimg_per_epoch,
                           batch_size=args.batch_size, learning_rate=args.lr,
                           gradient_checkpointing=not args.no_checkpointing)


def benchmark(args, pool, masks, out_root, device) -> None:
    from histopathology_pnw.models.student import build_student_init

    tmp = out_root / "benchmark"
    init = build_student_init(seed=0, out_path=tmp / "student_init.pth")
    sub = pool[: min(len(pool), 8)]
    imgs, labs = [c17.read_tile(t) for t in sub], [masks[t] for t in sub]
    times = {}
    for e in (2, 6):
        cfg = build_config(args, 0)
        cfg.n_epochs = e
        (tmp / f"e{e}").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        al.train_student(init, imgs, labs, tmp / f"e{e}", cfg, device, seed=0)
        times[e] = time.time() - t0
        print(f"{e} epok: {times[e]:.1f} s", flush=True)
    per_epoch = (times[6] - times[2]) / 4
    overhead = max(times[2] - 2 * per_epoch, 0.0)
    cfg = build_config(args, 0)
    n_rounds = len(al.round_sizes(cfg, len(pool)))
    per_round = overhead + per_epoch * cfg.n_epochs
    print(f"\nurządzenie: {device} | epoka: {per_epoch:.1f} s | narzut na rundę: {overhead:.0f} s")
    print(f"szacunek: runda ({cfg.n_epochs} epok x {cfg.nimg_per_epoch} wycinków) ~ {per_round / 60:.0f} min, "
          f"{n_rounds} rund ~ {n_rounds * per_round / 3600:.1f} h (bez ewaluacji na val, jedna seria)")
    print("Skracanie: mniej --nimg-per-epoch, większe --step (mniej rund), --max-labeled, --max-pool, --max-val.")


def plot_curve(results: pd.DataFrame, selfcons: float, target: float, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for metric, a in zip(("val_pq", "val_aji"), ax, strict=True):
        for seed, g in results.groupby("seed"):
            a.plot(g["n_labeled"], g[metric], "o-", label=f"student, seed {seed}")
        if metric == "val_pq":
            a.axhline(selfcons, color="k", ls="--", label="samozgodność wyroczni")
            a.axhline(target, color="gray", ls=":", label="poprzeczka")
        a.set_xlabel("liczba oznaczonych kafli")
        a.set_ylabel(metric.removeprefix("val_").upper() + " względem wyroczni (val)")
        a.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Losowe AL: student Cellpose-SAM vs wyrocznia (CAMELYON17)")
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--per-group", type=int, default=4, help="musi się zgadzać z wycinaniem kafli")
    ap.add_argument("--initial", type=int, default=4)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--max-labeled", type=int, default=None)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--nimg-per-epoch", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--target-frac", type=float, default=0.95, help="jaki ułamek samozgodności wyroczni ma osiągnąć student")
    ap.add_argument("--target-pq", type=float, default=None, help="ręczna poprzeczka PQ (nadpisuje --target-frac)")
    ap.add_argument("--min-nuclei", type=int, default=10, help="kafle z mniejszą liczbą jąder (wg wyroczni) wypadają z puli")
    ap.add_argument("--max-pool", type=int, default=None, help="ogranicz pulę do N kafli (losowo, stały seed)")
    ap.add_argument("--max-val", type=int, default=None, help="ogranicz walidację do N kafli")
    ap.add_argument("--max-test", type=int, default=None, help="ogranicz test do N kafli")
    ap.add_argument("--no-checkpointing", action="store_true", help="wyłącz gradient checkpointing (szybciej, więcej VRAM; OK na H100 80 GB)")
    ap.add_argument("--no-tf32", action="store_true", help="wyłącz TF32 (domyślnie włączone na CUDA, przyspiesza float32)")
    ap.add_argument("--allow-cpu", action="store_true", help="pozwól na CPU w trybie full (domyślnie przerywa)")
    ap.add_argument("--benchmark", action="store_true", help="zmierz tempo treningu i oszacuj czas całego przebiegu, nic nie zapisując")
    ap.add_argument("--skip-monuseg", action="store_true", help="pomiń porównanie na MoNuSeg względem GT")
    args = ap.parse_args()

    cp_io.logger_setup()  # logi treningu Cellpose (loss co kilka epok), potrzebne dla camelyon17_monitor.py
    device = get_device()
    if device.type == "cuda" and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device.type != "cuda" and args.mode == "full" and not args.allow_cpu:
        raise SystemExit(f"Wykryto urządzenie '{device.type}', a nie CUDA. Pełny trening na CPU trwa dziesiątki godzin. "
                         "Zamów węzeł z GPU (--gres=gpu:1) albo dodaj --allow-cpu.")
    out_root = Settings().outputs_dir / "al_random_c17" / args.mode
    out_root.mkdir(parents=True, exist_ok=True)
    print("device:", device, "| tryb:", args.mode, "| wyniki:", out_root, flush=True)

    c17.extract_tiles(per_group=args.per_group)  # bez surowych slajdów używa gotowych kafli
    pool_all = c17.list_tiles("train")["tile_id"].tolist()
    val = c17.list_tiles("val")["tile_id"].tolist()
    test = c17.list_tiles("test")["tile_id"].tolist()
    if args.mode == "smoke":
        pool_all, val, test = pool_all[:6], val[:4], test[:4]
    if not pool_all or not val or not test:
        raise SystemExit(f"Za mało kafli (pula {len(pool_all)}, val {len(val)}, test {len(test)}). "
                         "Uruchom najpierw datasets/camelyon17_prep.py.")
    print(f"kafle: pula {len(pool_all)}, val {len(val)}, test {len(test)}", flush=True)

    # 1) wyrocznia "adnotuje" wszystkie kafle (raz, cache w outputs/oracle_masks/camelyon17)
    oracle = load_oracle(device)
    all_ids = pool_all + val + test
    masks = dict(zip(all_ids, al.oracle_tile_masks(oracle, all_ids), strict=True))
    n_nuc = {t: len(np.unique(m)) - 1 for t, m in masks.items()}
    pool = [t for t in pool_all if n_nuc[t] >= args.min_nuclei]
    print(f"pula po filtrze (>= {args.min_nuclei} jąder): {len(pool)} / {len(pool_all)}", flush=True)
    if len(pool) < 2:
        raise SystemExit("Za mało kafli w puli po filtrze. Zmniejsz --min-nuclei albo dodaj slajdy.")

    sub = np.random.default_rng(0)  # stałe podzbiory, te same przy każdym uruchomieniu
    if args.max_pool and len(pool) > args.max_pool:
        pool = [pool[i] for i in sorted(sub.choice(len(pool), args.max_pool, replace=False))]
    if args.max_val and len(val) > args.max_val:
        val = [val[i] for i in sorted(sub.choice(len(val), args.max_val, replace=False))]
    if args.max_test and len(test) > args.max_test:
        test = [test[i] for i in sorted(sub.choice(len(test), args.max_test, replace=False))]
    print(f"używane: pula {len(pool)}, val {len(val)}, test {len(test)}", flush=True)

    # 1b) benchmark: ile trwa epoka na tym sprzęcie i ile potrwa cały przebieg
    if args.benchmark:
        benchmark(args, pool, masks, out_root, device)
        return

    # 2) poprzeczka: samozgodność wyroczni na val
    selfcons_val = al.oracle_self_consistency(oracle, val, [masks[t] for t in val])[0]
    selfcons_test = al.oracle_self_consistency(oracle, test, [masks[t] for t in test])[0]
    target = args.target_pq if args.target_pq is not None else args.target_frac * selfcons_val["pq"]
    print(f"samozgodność wyroczni (val PQ): {selfcons_val['pq']:.4f} -> poprzeczka: {target:.4f}", flush=True)
    (out_root / "oracle_self_consistency.json").write_text(
        json.dumps({"val": selfcons_val, "test": selfcons_test, "target_pq": target}, indent=2, default=float))

    # 3) pętla AL dla każdego seeda
    parts = []
    for seed in args.seeds:
        cfg = build_config(args, seed)
        res = al.run_random_al(cfg, pool, val, masks, out_root / f"seed{seed}", target, device)
        parts.append(res.assign(seed=seed))
    results = pd.concat(parts, ignore_index=True)
    cols = ["seed", "n_labeled", "val_pq", "val_sq", "val_dq", "val_aji", "final_train_loss", "train_time_s"]
    results[cols].to_csv(out_root / "results_all_seeds.csv", index=False)
    print("\n", results[cols].round(4).to_string(index=False), flush=True)
    plot_curve(results, selfcons_val["pq"], target, out_root / "learning_curve.png")

    reached = results[results["val_pq"] >= target]
    if len(reached):
        print("\npoprzeczkę osiągnięto przy liczbie kafli (min po seedach):")
        print(reached.groupby("seed")["n_labeled"].min().to_string())
    else:
        print(f"\nstudent nie osiągnął poprzeczki {target:.4f}; najlepiej {results['val_pq'].max():.4f}")

    # 4) test: najlepsza runda wybrana na val; zgodność z wyrocznią na kaflach test
    best = results.loc[results["val_pq"].idxmax()]
    print(f"\nnajlepsza runda: seed {best['seed']}, {best['n_labeled']} kafli, val PQ {best['val_pq']:.4f}")
    student = load_student(c17_path(best["checkpoint"]), device)
    test_imgs = [c17.read_tile(t) for t in test]
    student_test = summarize(evaluate_all([masks[t] for t in test], predict(student, test_imgs), test))
    final = pd.DataFrame({"student vs wyrocznia (test)": student_test,
                          "samozgodność wyroczni (test)": selfcons_test}).T
    final.to_csv(out_root / "final_test_c17.csv")
    print("\n", final.round(4).to_string(), flush=True)

    # 5) jedyne porównanie z prawdziwym GT: MoNuSeg test (student uczony na maskach wyroczni z CAMELYON17)
    if not args.skip_monuseg:
        try:
            from histopathology_pnw.datasets import monuseg
            from histopathology_pnw.models.oracle import oracle_masks

            mn = monuseg.list_split("test")
            ids = [s.image_id for s in mn]
            gts = [monuseg.read_mask(s) for s in mn]
            st = summarize(evaluate_all(gts, predict(student, [monuseg.read_image(s.image_path) for s in mn]), ids))
            orc = summarize(evaluate_all(gts, oracle_masks(mn, model=oracle), ids))
            gt_tab = pd.DataFrame({"student (uczony na CAMELYON17)": st, "wyrocznia cpsam": orc}).T
            gt_tab.to_csv(out_root / "monuseg_test_vs_gt.csv")
            print("\nMoNuSeg test względem GT (wyrocznia widziała MoNuSeg, więc jej wynik jest zawyżony):\n",
                  gt_tab.round(4).to_string(), flush=True)
        except Exception as e:  # brak danych MoNuSeg lokalnie to nie błąd eksperymentu
            print("pominięto porównanie na MoNuSeg:", repr(e))
    print("\ngotowe. Wyniki:", out_root)


def c17_path(p):
    from pathlib import Path

    return Path(p)


if __name__ == "__main__":
    main()
