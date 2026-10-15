# Plan pracy — histopathology_pnw

Checklista do odhaczania (`- [ ]` → `- [x]`). Kolejność faz = kolejność pracy.
Pozycje oznaczone 📖 to lektura, 🧪 to coś do sprawdzenia/zmierzenia, ❓ to decyzja do podjęcia.
Literatura naukowa (co i w jakiej kolejności czytać) jest w [`LITERATURA.md`](LITERATURA.md).

---

## Etap 1 — zadania od promotora

Badawcze:
- [ ] B1. Przegląd literatury: metody AL (selekcji przypadków) w segmentacji obrazów medycznych → `LITERATURA.md` §3
- [ ] B2. Przegląd literatury: AL w segmentacji jąder w histopatologii → `LITERATURA.md` §4
- [ ] B3. Przegląd zbiorów benchmarkowych: CAMELYON17, MoNuSeg, PanNuke, CoNSeP → `LITERATURA.md` §2

Implementacyjne:
- [ ] I1. Cellpose-SAM (oracle `cpsam`) w trybie inferencji na obrazach z tych zbiorów; na zbiorach
  testowych zaraportować metryki segmentacji (PQ, PQ@0.5 + przegląd metryk) i omówić wyniki
  → `notebooks/02_oracle_inference.ipynb` (kod gotowy; MoNuSeg; PanNuke/CoNSeP jeszcze nie pobrane)
- [ ] I2. System rozmyty typu 1: rozmycie, wnioskowanie, wyostrzanie → `fuzzy/type1.py`
- [ ] I3. Rozpoznanie implementacji Cellpose-SAM; AL z losowym wyborem obrazów, aż student osiągnie
  wynik wytrenowanego Cellpose-SAM → `notebooks/03_random_al.ipynb` (kod gotowy, do uruchomienia na Ubuntu)

Kolejność pracy: MoNuSeg (Faza 2–3) → I1 → I3; I2 niezależnie.

---

## Faza 0 — Fundament repo i środowiska

- [x] ❓ Przypiąć model oracle (`cpsam` vs `cpsam_v2`) i wpisać decyzję do `AGENTS.md` + README
- [x] Uzupełnić `pyproject.toml` (`description`, sprawdzić e-mail autora vs git)
- [x] Dodać dev-zależności: `ruff`, `pytest` (`uv add --dev ...`), skonfigurować ruff w `pyproject.toml`
- [x] (opcjonalnie) `pre-commit` z ruff, żeby lint chodził przed każdym commitem
- [x] Ustalić strukturę katalogów (`src/`, `configs/`, `data/`, `tests/`, `slurm/`, `docs/`) — patrz README
- [x] Konfiguracja per maszyna: jeden plik/zmienne środowiskowe z bazowymi ścieżkami (data, outputs, cache)
  - 📖 `pydantic-settings` (czyta `.env` + env vars) albo Hydra/OmegaConf — wybierz jedno
- [x] Jedna funkcja wyboru urządzenia (cuda → mps → cpu) — test, że działa na macOS (MPS) i CPU
- [x] Zasada seedów: jedna funkcja ustawiająca `random`, `numpy`, `torch` (+ `torch.backends` dla CUDA)
- [ ] CI na GitHub Actions: `ruff check` + `pytest` na CPU (bez danych, małe fikstury)
- [x] Sprawdzić, że `uv sync` + import `cellpose` działa na: macOS ☐ · Ubuntu/CUDA ☐ · WCSS ☐
- [x] WCSS: wagi Cellpose/SAM (~1–2 GB) mogą zostać w `$HOME` (limit 50 GB); przenieść na PD tylko, jeśli miejsca zabraknie
  - 🧪 sprawdzić w `cellpose/vit.py`, gdzie `load_pretrained` zapisuje `sam_vit_l_0b3195.pth`
  - jeśli trzeba przenieść: `CELLPOSE_LOCAL_MODELS_PATH` (nazwa z pamięci — potwierdzić w `cellpose/models.py`)

📖 Lektura do fazy 0:
- [ ] Dokumentacja uv: projekty, `uv.lock`, `uv sync --frozen`, dependency groups
- [ ] Dokumentacja WCSS: Slurm (`sbatch`, `--time`, `--gres`), moduły, przestrzeń `$SCRATCH` vs `$HOME`, limity, czy węzły obliczeniowe mają internet
- [ ] PyTorch: „Reproducibility” (strona w dokumentacji — determinism, `use_deterministic_algorithms`)

---

## Faza 1 — DVC: dane dostępne na każdej maszynie

- [x] 📖 DVC „Get Started: Data Versioning” + „Data Pipelines” + „Remote storage”
- [x] Wybór remote DVC: SSH na WCSS (`wcss` → `/lustre/pd01/.../mateusz/dvc-store`)
- [x] `dvc[ssh]` w grupie `dev`
- [x] `dvc init`, remote `wcss` jako domyślny; dane uwierzytelniające **tylko** w `~/.ssh/config` / `.dvc/config.local`
- [x] Alias `wcss` w `~/.ssh/config` na macOS ☐ · Ubuntu ☐ (klucz SSH, ewentualnie VPN PWr)
- [x] Na samym WCSS: `uv run dvc remote modify --local wcss url $PD/mateusz/dvc-store` — ścieżka lokalna zamiast SSH do samego siebie
- [x] Na WCSS: cache DVC na PD (`dvc cache dir --local $PD/mateusz/dvc-cache`), `dvc pull` działa
- [x] Na WCSS: `dvc config --local cache.type symlink` (hardlink między `$HOME` a PD i tak nie działa)
- [x] Na WCSS: `export PD=...` w `~/.bashrc`
- [x] Podział miejsca na WCSS: repo + `.venv` + cache uv w `$HOME` (50 GB, 1M plików, snapshoty); dane DVC na PD; checkpointy/runy na PD przez konfigurację — szczegóły w README
- [ ] Ścieżka wyników (checkpointy, runy, logi) w konfiguracji per maszyna; na WCSS → PD
- [x] `git check-ignore -v data/raw/monuseg.dvc` nie może zwrócić dopasowania (plik `.dvc` musi trafić do gita)
- [x] Śledzić surowe archiwum MoNuSeg (`dvc add data/raw/monuseg`) — **surowe dane nigdy nie są nadpisywane**
- [x] Preprocessing jako stage w `dvc.yaml` (deps: raw + skrypt + params, outs: maski) → `dvc repro` odtwarza wynik
- [ ] Parametry preprocessingu w `params.yaml` (DVC śledzi zmiany)
- [ ] Test end-to-end: na drugiej maszynie `git pull && dvc pull` daje identyczne pliki (porównaj hashe)
- [ ] Opisać w README „jak pobrać dane na nowej maszynie”
- [ ] 🧪 Sprawdzić, czy węzły GPU na WCSS mają dostęp do remote; jeśli nie — `dvc pull` na login node przed `sbatch`

---

## Faza 2 — MoNuSeg: wiedza o danych

📖 Lektura: `LITERATURA.md` §1 (metryki) i §2 (zbiory), a do tego:
- [ ] Strona MoNuSeg (grand-challenge) — licencja, format XML, opis splitu

🧪 Eksploracja (`notebooks/01_monuseg_exploration.ipynb`, wyniki zapisać):
- [x] Policzyć: obrazy, rozmiary, liczba jąder na obraz, organy, train vs test
- [x] Rozkład pól powierzchni i średnic jąder (w px) — potrzebne do `diameter` w Cellpose i do cech geometrycznych FLS
- [ ] Sprawdzić rozdzielczość (40×, µm/px) — czy wszystkie obrazy spójne
- [x] Poszukać patologii adnotacji: poligony < 3 wierzchołków, samoprzecięcia, wierzchołki poza obrazem, duplikaty, nakładające się poligony
- [x] Obejrzeć kilka overlayów maska/obraz na oko

---

## Faza 3 — Preprocessing: XML → maski instancji

Decyzje do podjęcia i zapisania (w README/`docs/`):
- [x] ❓ Rasteryzacja: `skimage.draw.polygon` vs `cv2.fillPoly` — sprawdź różnice na krawędziach (konwencja współrzędnych pikseli, zaokrąglanie floatów)
- [x] ❓ Reguła nakładania się jąder: „późniejszy wygrywa”, „mniejszy wygrywa”, czy usunięcie spornych pikseli — wpływa na AJI
- [ ] ❓ Co z instancjami po rasteryzacji < N px lub rozbitymi na kilka komponentów (`skimage.measure.label` na każdej instancji)
- [x] ❓ Format wyjścia: TIFF/PNG `uint16`/`int32` z etykietami 0 = tło, 1..N = instancje (konwencja Cellpose: `*_masks.tif` obok obrazu lub własny loader)
- [ ] ❓ Czy kolor (RGB) zostaje bez zmian — Cellpose-SAM przyjmuje 3 kanały; normalizację intensywności robi sam (sprawdź `normalize` w `CellposeModel.eval` / `train_seg`)

Implementacja (piszesz Ty):
- [x] Parser XML (📖 `xml.etree.ElementTree` — struktura `Annotation/Regions/Region/Vertices/Vertex`)
- [x] Rasteryzacja do maski instancji + walidacja
- [x] Skrypt uruchamiany jako `uv run python -m ...`, ścieżki z konfiguracji
- [x] Testy: liczba instancji w masce == liczba poprawnych poligonów; etykiety unikalne; brak wartości poza zakresem; mały syntetyczny XML jako fikstura
- [x] Zapis manifestu (CSV/JSON): obraz, organ, split, liczba jąder, hash — przyda się w AL
- [x] Wpiąć jako stage DVC

Walidacja:
- [x] ❓ **Split walidacyjny z 37 obrazów train** (test jest nietykalny): podział **po obrazie** (nigdy po kafelkach z tego samego obrazu), najlepiej stratyfikowany po organie; zapisany w pliku i zamrożony
- [ ] 📖 Sprawdzić, czy w literaturze MoNuSeg ktoś raportuje konkretny val split — jeśli tak, rozważ jego użycie dla porównywalności
- [ ] 🧪 Oracle vs GT na zbiorze train (AJI, PQ, F1@0.5) — to jest „poziom szumu adnotatora” w Twojej symulacji AL; zapisz to jako wynik

---

## Faza 4 — Augmentacja i normalizacja barwienia (wiedza SOTA)

📖 Lektura:
- [ ] `LITERATURA.md` §6: Tellez 2019 (kluczowy wniosek: augmentacja kolorów HED > normalizacja), Macenko, Vahadane, Reinhard
- [ ] Biblioteka `torchstain` (Macenko/Reinhard w PyTorch) — tylko jeśli zdecydujesz się normalizować
- [ ] Kod augmentacji w Cellpose (`cellpose/transforms.py`, `random_rotate_and_resize`) — co już robi sam trening

Decyzje:
- [ ] ❓ Normalizacja barwienia: tak/nie (Cellpose-SAM był trenowany na zróżnicowanych danych; na MoNuSeg z wieloma ośrodkami często wystarcza augmentacja)
- [ ] ❓ Augmentacje dodatkowe ponad wbudowane w Cellpose (HED jitter, blur, JPEG) — każda musi być identyczna dla wszystkich selektorów AL

---

## Faza 5 — Student Cellpose-SAM z wag SAM

📖 Lektura (obowiązkowa): `LITERATURA.md` §1 (SAM, Cellpose, Cellpose 2.0, Cellpose-SAM), a do tego:
- [ ] Kod: `cellpose/vit.py` (`CPSAM`, `load_pretrained`), `cellpose/train.py` (`train_seg` — domyślne lr, weight decay, epoki, `bsize`, `rescale`)

📖 Kontekst SOTA segmentacji jąder (do rozdziału „related work”): `LITERATURA.md` §1 (HoVer-Net) i §6 (StarDist, CellViT, MedSAM)

🧪 Kroki:
- [x] Zbudować studenta z `CPSAM()` + `load_pretrained`; **zalogować missing/unexpected keys** i sprawdzić, że ładuje się `sam_vit_l_0b3195.pth`
- [ ] Sanity check: trening na 1–2 obrazach musi się przeuczyć (loss → ~0) — jeśli nie, błąd w danych/maskach
- [ ] Baseline górny: student trenowany na wszystkich 37 (bez val) obrazach train → ewaluacja na val
- [ ] Baseline dolny: student na małym losowym podzbiorze (rozmiar = start AL)
- [ ] Krzywa uczenia: wynik vs liczba obrazów/kafelków (to jest oś X wykresów AL)
- [ ] Checkpointy + wznawianie (stan modelu, optymalizatora, schedulera, epoki, RNG) — test: przerwij i wznów, wynik ten sam
- [x] Ewaluacja: AJI, PQ (DQ/SQ), F1/AP przy IoU 0.5 (`cellpose.metrics.average_precision`) — jedna funkcja ewaluacji dla wszystkiego
- [ ] Zmierzyć czas epoki na MPS / CUDA / H100 → oszacować budżet GPU-godzin na eksperymenty AL
- [ ] Szablon joba Slurm (`slurm/`) z wznawianiem po timeout — komendę uruchamiasz Ty

---

## Faza 6 — Śledzenie eksperymentów i reprodukowalność

- [ ] ❓ Tracker: W&B (tryb offline na WCSS + `wandb sync`), MLflow (lokalny plik), albo `dvc exp` — wybrać jeden
- [ ] Jeśli MLflow: tracking URI ustawiony jawnie w konfiguracji per maszyna (np. pod ignorowanym `outputs/`), żeby baza nie powstawała w katalogu głównym repo; po pierwszym runie sprawdzić `git status`
- [ ] Każdy run zapisuje: commit git, hash danych DVC, pełny config, seed, maszynę, wersję cellpose/torch
- [ ] Oracle: predykcje oracle liczone **raz**, zapisane i śledzone w DVC (oszczędza GPU-godziny w każdej iteracji AL)
- [ ] 📖 Pineau et al., 2021, *Improving Reproducibility in Machine Learning Research*, JMLR (checklista NeurIPS)

---

## Faza 7 (później) — Active learning i selektory

📖 Lektura AL: `LITERATURA.md` §3 i §4 (B1, B2 z Etapu 1)

📖 Lektura IT2 FLS: `LITERATURA.md` §5 (Mendel & John 2002; Mendel 2017 — Karnik–Mendel type reduction)
- [ ] Biblioteka Pythona dla IT2 (np. `pyit2fls`) albo własna implementacja — decyzja

Decyzje do zaprojektowania:
- [ ] ❓ Jednostka zapytania AL: cały obraz vs kafelek (koszt adnotacji ∝ liczba jąder?)
- [ ] ❓ Budżet startowy, rozmiar batcha zapytań, liczba rund, liczba seedów (≥3, lepiej 5)
- [ ] ❓ Czy student jest trenowany od zera w każdej rundzie czy dotrenowywany (warm start) — tak samo dla wszystkich selektorów
- [ ] ❓ Cechy geometryczne jąder dla FLS (pole, okrągłość, solidity, ekscentryczność…) — `skimage.measure.regionprops`
- [ ] Random baseline obowiązkowo

---

## Faza 8 (później) — CAMELYON17

- [ ] Zrozumieć format WSI, poziomy piramidy, ekstrakcję kafelków (📖 `openslide`/`tiffslide`, `cucim`)
- [ ] Surowe dane poza DVC (zgodnie z AGENTS.md) — tylko wyekstrahowane kafelki/manifesty w DVC
- [ ] Brak masek jąder w CAMELYON17 → rola oracle staje się kluczowa; przemyśleć ewaluację

---

## Rzeczy do dopisania do pracy magisterskiej na bieżąco

- [ ] Rozdział „Dane”: opis MoNuSeg, decyzje preprocessingu (nakładanie się, rasteryzacja), split walidacyjny
- [ ] Zagrożenie dla trafności: oracle (Cellpose-SAM) widział MoNuSeg podczas treningu
- [ ] Jakość oracle vs GT jako model szumu adnotatora
