# # Pobieranie CAMELYON17 z publicznego bucketu AWS (bez konta AWS).
# # Umieść w: src/histopathology_pnw/camelyon17_download.py
# #
# # Bucket: s3://camelyon-dataset (region us-west-2, licencja CC0).
# # Domyślnie pobiera TYLKO slajdy z adnotacją XML (50 sztuk, po 10 z każdego ośrodka),
# # bo tylko one są potrzebne do notebooków 01-03, oraz stage_labels.csv.
# # Układ katalogów w bucketcie jest wykrywany automatycznie (listowanie kluczy).
from __future__ import annotations

import re
import shutil
from pathlib import Path

BUCKET = "camelyon-dataset"
REGION = "us-west-2"
PREFIX = "CAMELYON17/"
SLIDE_RE = re.compile(r"patient_\d+_node_\d+")


def _client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3", region_name=REGION,
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 10, "mode": "standard"}),
    )


def _list(s3, prefix: str) -> list[dict]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        out += page.get("Contents", [])
    return out


def discover(prefix: str = PREFIX) -> dict:
    # zwraca {"xml": {stem: obj}, "tif": {stem: obj}, "csv": [obj, ...]}
    s3 = _client()
    objs = _list(s3, prefix)
    if not objs:
        top = s3.list_objects_v2(Bucket=BUCKET, Delimiter="/").get("CommonPrefixes", [])
        raise RuntimeError(
            f"Brak kluczy pod {BUCKET}/{prefix}. Prefiksy w bucketcie: {[p['Prefix'] for p in top]}. "
            "Ustaw PREFIX w camelyon17_download.py albo pobierz dane ręcznie (patrz niżej)."
        )
    xml, tif, csv = {}, {}, []
    for o in objs:
        key = o["Key"]
        stem = Path(key).stem
        if key.endswith(".xml") and SLIDE_RE.fullmatch(stem):
            xml[stem] = o
        elif key.endswith(".tif") and SLIDE_RE.fullmatch(stem):
            tif[stem] = o
        elif key.lower().endswith(".csv") and "stage" in key.lower():
            csv.append(o)
    return {"xml": xml, "tif": tif, "csv": csv}


def _fetch(s3, obj: dict, dest: Path) -> None:
    from boto3.s3.transfer import TransferConfig

    if dest.exists() and dest.stat().st_size == obj["Size"]:
        return  # już pobrane
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    cfg = TransferConfig(max_concurrency=10, multipart_chunksize=64 * 2**20)
    s3.download_file(BUCKET, obj["Key"], str(part), Config=cfg)
    if part.stat().st_size != obj["Size"]:
        raise OSError(f"niepełny plik: {part}")
    part.rename(dest)


def _pick_balanced(stems: list[str], tif: dict, k: int) -> list[str]:
    # k slajdów rozłożonych po równo między ośrodki (centrum = pacjent // 20),
    # w każdym ośrodku od najmniejszych plików (oszczędza dysk)
    by_center: dict[int, list[str]] = {}
    for s in stems:
        by_center.setdefault(int(re.match(r"patient_(\d+)", s).group(1)) // 20, []).append(s)
    for c in by_center:
        by_center[c].sort(key=lambda s: tif[s]["Size"] if s in tif else float("inf"))
    picked: list[str] = []
    while len(picked) < k and any(by_center.values()):
        for c in sorted(by_center):
            if by_center[c] and len(picked) < k:
                picked.append(by_center[c].pop(0))
    return sorted(picked)

def download_camelyon17(
    data_root: Path,
    only_annotated: bool = True,
    dry_run: bool = False,
    prefix: str = PREFIX,
    max_slides: int | None = None,
    exclude_ids: set[str] | None = None,
) -> None:
    data_root = Path(data_root)
    images = data_root / "images"
    annot = data_root / "lesion_annotations"

    exclude_ids = exclude_ids or set()
    found = discover(prefix)
    xml, tif = found["xml"], found["tif"]

    # Tylko slajdy z XML, jeśli wymagamy adnotacji.
    wanted = sorted(xml) if only_annotated else sorted(tif)

    # Pomijamy wskazane ID oraz slajdy bez pliku TIFF.
    wanted = [
        s for s in wanted
        if s not in exclude_ids and s in tif
    ]

    # Pomijamy kompletne pliki, które już znajdują się na dysku.
    remaining = []
    for stem in wanted:
        dest = images / f"{stem}.tif"
        if dest.exists() and dest.stat().st_size == tif[stem]["Size"]:
            continue
        remaining.append(stem)

    # Wybieramy określoną liczbę NOWYCH slajdów.
    if max_slides is not None and max_slides < len(remaining):
        remaining = _pick_balanced(remaining, tif, max_slides)

    wanted = remaining

    # XML pobieramy tylko dla wybranych slajdów.
    xml_sel = {s: xml[s] for s in wanted if s in xml}

    total = (
        sum(tif[s]["Size"] for s in wanted)
        + sum(
            o["Size"] for s, o in xml_sel.items()
            if not (
                (annot / f"{s}.xml").exists()
                and (annot / f"{s}.xml").stat().st_size == o["Size"]
            )
        )
    )

    free_gb = shutil.disk_usage(
        data_root.parent if data_root.parent.exists() else Path(".")
    ).free / 2**30

    print(f"Slajdy do pobrania: {len(wanted)}")
    print(f"Adnotacje XML do pobrania: {len(xml_sel)}")
    print(f"Szacowany rozmiar: {total / 2**30:.2f} GB")
    print(f"Wolne miejsce: {free_gb:.0f} GB")

    if not wanted:
        print("Brak nowych slajdów do pobrania.")
        return

    for stem in wanted:
        print(
            f"Do pobrania: {stem}.tif "
            f"({tif[stem]['Size'] / 2**30:.2f} GB)"
        )

    if dry_run:
        return

    s3 = _client()

    # Najpierw pobieramy brakujące adnotacje.
    for stem, obj in sorted(xml_sel.items()):
        _fetch(s3, obj, annot / f"{stem}.xml")

    # Etykiety pobieramy tylko wtedy, gdy ich jeszcze nie ma.
    stage_path = data_root / "stage_labels.csv"
    for obj in found["csv"]:
        if Path(obj["Key"]).name.lower() == "stage_labels.csv":
            _fetch(s3, obj, stage_path)
            break

    # Pobieramy wybrane slajdy.
    for i, stem in enumerate(wanted, 1):
        print(
            f"[{i}/{len(wanted)}] {stem}.tif "
            f"({tif[stem]['Size'] / 2**30:.2f} GB)"
        )
        _fetch(s3, tif[stem], images / f"{stem}.tif")

    print("Gotowe:", data_root)




# def download_camelyon17(data_root: Path, only_annotated: bool = True, dry_run: bool = False,
#                         prefix: str = PREFIX, max_slides: int | None = None) -> None:
#     data_root = Path(data_root)
#     images, annot = data_root / "images", data_root / "lesion_annotations"
#     found = discover(prefix)
#     xml, tif = found["xml"], found["tif"]

#     wanted = sorted(xml) if only_annotated else sorted(tif)
#     if max_slides is not None and max_slides < len(wanted):
#         wanted = _pick_balanced(wanted, tif, max_slides)
#     xml_sel = {s: xml[s] for s in (wanted if only_annotated else xml) if s in xml}
#     missing_tif = [s for s in wanted if s not in tif]
#     if missing_tif:
#         print(f"UWAGA: brak .tif dla {len(missing_tif)} adnotacji: {missing_tif[:5]}...")
#     todo_tif = [s for s in wanted if s in tif]
#     total = sum(tif[s]["Size"] for s in todo_tif) + sum(o["Size"] for o in xml_sel.values())
#     print(f"adnotacje XML: {len(xml_sel)}, slajdy do pobrania: {len(todo_tif)}, "
#           f"łącznie ok. {total / 2**30:.1f} GB; wolne miejsce: "
#           f"{shutil.disk_usage(data_root.parent if data_root.parent.exists() else '.').free / 2**30:.0f} GB")
#     if dry_run:
#         return

#     s3 = _client()
#     for stem, obj in sorted(xml_sel.items()):
#         _fetch(s3, obj, annot / f"{stem}.xml")
#     for obj in found["csv"]:
#         # preferuj plik z etykietami slajdów (stage_labels.csv)
#         if Path(obj["Key"]).name.lower() == "stage_labels.csv":
#             _fetch(s3, obj, data_root / "stage_labels.csv")
#     for i, stem in enumerate(todo_tif, 1):
#         print(f"[{i}/{len(todo_tif)}] {stem}.tif ({tif[stem]['Size'] / 2**30:.2f} GB)")
#         _fetch(s3, tif[stem], images / f"{stem}.tif")
#     if not (data_root / "stage_labels.csv").exists():
#         print("UWAGA: nie znaleziono stage_labels.csv w bucketcie. Pobierz go ręcznie ze strony "
#               "https://camelyon17.grand-challenge.org/Data/ i zapisz jako "
#               f"{data_root / 'stage_labels.csv'} (bez niego notebook 01 pominie sprawdzenie etykiet).")
#     print("gotowe:", data_root)


# def data_present(data_root: Path) -> bool:
#     data_root = Path(data_root)
#     img, ann = data_root / "images", data_root / "lesion_annotations"
#     return (img.exists() and any(img.rglob("*.tif"))
#             and ann.exists() and any(ann.glob("*.xml")))
