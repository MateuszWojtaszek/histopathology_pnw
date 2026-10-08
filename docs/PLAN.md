# Plan pracy — histopathology_pnw

Checklista do odhaczania (`- [ ]` → `- [x]`). Kolejność faz = kolejność pracy.
Pozycje oznaczone 📖 to lektura, 🧪 to coś do sprawdzenia/zmierzenia, ❓ to decyzja do podjęcia.

---

## Faza 0 — Fundament repo i środowiska

- [ ] ❓ Przypiąć model oracle (`cpsam` vs `cpsam_v2`) i wpisać decyzję do `AGENTS.md` + README
- [ ] Uzupełnić `pyproject.toml` (`description`, sprawdzić e-mail autora vs git)
- [ ] Dodać dev-zależności: `ruff`, `pytest` (`uv add --dev ...`), skonfigurować ruff w `pyproject.toml`
- [ ] (opcjonalnie) `pre-commit` z ruff, żeby lint chodził przed każdym commitem
- [ ] Ustalić strukturę katalogów (`src/`, `configs/`, `data/`, `tests/`, `slurm/`, `docs/`) — patrz README
- [ ] Konfiguracja per maszyna: jeden plik/zmienne środowiskowe z bazowymi ścieżkami (data, outputs, cache)
  - 📖 `pydantic-settings` (czyta `.env` + env vars) albo Hydra/OmegaConf — wybierz jedno
- [ ] Jedna funkcja wyboru urządzenia (cuda → mps → cpu) — test, że działa na macOS (MPS) i CPU
- [ ] Zasada seedów: jedna funkcja ustawiająca `random`, `numpy`, `torch` (+ `torch.backends` dla CUDA)
- [ ] CI na GitHub Actions: `ruff check` + `pytest` na CPU (bez danych, małe fikstury)
- [ ] Sprawdzić, że `uv sync` + import `cellpose` działa na: macOS ☐ · Ubuntu/CUDA ☐ · WCSS ☐
- [ ] WCSS: wagi Cellpose/SAM (~1–2 GB) mogą zostać w `$HOME` (limit 50 GB); przenieść na PD tylko, jeśli miejsca zabraknie
  - 🧪 sprawdzić w `cellpose/vit.py`, gdzie `load_pretrained` zapisuje `sam_vit_l_0b3195.pth`
  - jeśli trzeba przenieść: `CELLPOSE_LOCAL_MODELS_PATH` (nazwa z pamięci — potwierdzić w `cellpose/models.py`)

📖 Lektura do fazy 0:
- [ ] Dokumentacja uv: projekty, `uv.lock`, `uv sync --frozen`, dependency groups
- [ ] Dokumentacja WCSS: Slurm (`sbatch`, `--time`, `--gres`), moduły, przestrzeń `$SCRATCH` vs `$HOME`, limity, czy węzły obliczeniowe mają internet
- [ ] PyTorch: „Reproducibility” (strona w dokumentacji — determinism, `use_deterministic_algorithms`)

---

## Faza 1 — DVC: dane dostępne na każdej maszynie

- [ ] 📖 DVC „Get Started: Data Versioning” + „Data Pipelines” + „Remote storage”
- [x] Wybór remote DVC: SSH na WCSS (`wcss` → `/lustre/pd01/.../mateusz/dvc-store`)
- [x] `dvc[ssh]` w grupie `dev`
- [x] `dvc init`, remote `wcss` jako domyślny; dane uwierzytelniające **tylko** w `~/.ssh/config` / `.dvc/config.local`
- [ ] Alias `wcss` w `~/.ssh/config` na macOS ☐ · Ubuntu ☐ (klucz SSH, ewentualnie VPN PWr)
- [x] Na samym WCSS: `uv run dvc remote modify --local wcss url $PD/mateusz/dvc-store` — ścieżka lokalna zamiast SSH do samego siebie
- [x] Na WCSS: cache DVC na PD (`dvc cache dir --local $PD/mateusz/dvc-cache`), `dvc pull` działa
- [ ] Na WCSS: `dvc config --local cache.type symlink` (hardlink między `$HOME` a PD i tak nie działa)
- [ ] Na WCSS: `export PD=...` w `~/.bashrc`
- [x] Podział miejsca na WCSS: repo + `.venv` + cache uv w `$HOME` (50 GB, 1M plików, snapshoty); dane DVC na PD; checkpointy/runy na PD przez konfigurację — szczegóły w README
- [ ] Ścieżka wyników (checkpointy, runy, logi) w konfiguracji per maszyna; na WCSS → PD
- [ ] `git check-ignore -v data/raw/monuseg.dvc` nie może zwrócić dopasowania (plik `.dvc` musi trafić do gita)
- [ ] Śledzić surowe archiwum MoNuSeg (`dvc add data/raw/monuseg`) — **surowe dane nigdy nie są nadpisywane**
- [ ] Preprocessing jako stage w `dvc.yaml` (deps: raw + skrypt + params, outs: maski) → `dvc repro` odtwarza wynik
- [ ] Parametry preprocessingu w `params.yaml` (DVC śledzi zmiany)
- [ ] Test end-to-end: na drugiej maszynie `git pull && dvc pull` daje identyczne pliki (porównaj hashe)
- [ ] Opisać w README „jak pobrać dane na nowej maszynie”
- [ ] 🧪 Sprawdzić, czy węzły GPU na WCSS mają dostęp do remote; jeśli nie — `dvc pull` na login node przed `sbatch`

---

## Faza 2 — MoNuSeg: wiedza o danych

📖 Lektura:
- [ ] Kumar et al., 2017, *A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology*, IEEE TMI — oryginalny dataset + metryka AJI
- [ ] Kumar et al., 2020, *A Multi-Organ Nucleus Segmentation Challenge*, IEEE TMI — challenge, oficjalny split, wyniki
- [ ] Strona MoNuSeg (grand-challenge) — licencja, format XML, opis splitu
- [ ] Graham et al., 2019, *HoVer-Net*, Medical Image Analysis — sekcja o metrykach (krytyka AJI/Dice, PQ dla jąder)
- [ ] Foucart, Debeir, Decaestecker, 2023, *Panoptic quality should be avoided as a metric for assessing cell nuclei segmentation and classification in digital pathology*, Scientific Reports
- [ ] (kontekst) Gamper et al., *PanNuke* (2019/2020); Graham et al., *Lizard* (2021) / CoNIC challenge — skala i jakość innych zbiorów

🧪 Eksploracja (notebook lub skrypt, wyniki zapisać):
- [ ] Policzyć: obrazy, rozmiary, liczba jąder na obraz, organy, train vs test
- [ ] Rozkład pól powierzchni i średnic jąder (w px) — potrzebne do `diameter` w Cellpose i do cech geometrycznych FLS
- [ ] Sprawdzić rozdzielczość (40×, µm/px) — czy wszystkie obrazy spójne
- [ ] Poszukać patologii adnotacji: poligony < 3 wierzchołków, samoprzecięcia, wierzchołki poza obrazem, duplikaty, nakładające się poligony
- [ ] Obejrzeć kilka overlayów maska/obraz na oko

---

## Faza 3 — Preprocessing: XML → maski instancji

Decyzje do podjęcia i zapisania (w README/`docs/`):
- [ ] ❓ Rasteryzacja: `skimage.draw.polygon` vs `cv2.fillPoly` — sprawdź różnice na krawędziach (konwencja współrzędnych pikseli, zaokrąglanie floatów)
- [ ] ❓ Reguła nakładania się jąder: „późniejszy wygrywa”, „mniejszy wygrywa”, czy usunięcie spornych pikseli — wpływa na AJI
- [ ] ❓ Co z instancjami po rasteryzacji < N px lub rozbitymi na kilka komponentów (`skimage.measure.label` na każdej instancji)
- [ ] ❓ Format wyjścia: TIFF/PNG `uint16`/`int32` z etykietami 0 = tło, 1..N = instancje (konwencja Cellpose: `*_masks.tif` obok obrazu lub własny loader)
- [ ] ❓ Czy kolor (RGB) zostaje bez zmian — Cellpose-SAM przyjmuje 3 kanały; normalizację intensywności robi sam (sprawdź `normalize` w `CellposeModel.eval` / `train_seg`)

Implementacja (piszesz Ty):
- [ ] Parser XML (📖 `xml.etree.ElementTree` — struktura `Annotation/Regions/Region/Vertices/Vertex`)
- [ ] Rasteryzacja do maski instancji + walidacja
- [ ] Skrypt uruchamiany jako `uv run python -m ...`, ścieżki z konfiguracji
- [ ] Testy: liczba instancji w masce == liczba poprawnych poligonów; etykiety unikalne; brak wartości poza zakresem; mały syntetyczny XML jako fikstura
- [ ] Zapis manifestu (CSV/JSON): obraz, organ, split, liczba jąder, hash — przyda się w AL
- [ ] Wpiąć jako stage DVC

Walidacja:
- [ ] ❓ **Split walidacyjny z 30 obrazów train** (test jest nietykalny): podział **po obrazie** (nigdy po kafelkach z tego samego obrazu), najlepiej stratyfikowany po organie; zapisany w pliku i zamrożony
- [ ] 📖 Sprawdzić, czy w literaturze MoNuSeg ktoś raportuje konkretny val split — jeśli tak, rozważ jego użycie dla porównywalności
- [ ] 🧪 Oracle vs GT na zbiorze train (AJI, PQ, F1@0.5) — to jest „poziom szumu adnotatora” w Twojej symulacji AL; zapisz to jako wynik

---

## Faza 4 — Augmentacja i normalizacja barwienia (wiedza SOTA)

📖 Lektura:
- [ ] Tellez et al., 2019, *Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology*, Medical Image Analysis — kluczowy wniosek: augmentacja kolorów (HED) > normalizacja
- [ ] Macenko et al., 2009 (normalizacja), Vahadane et al., 2016 (SNMF), Reinhard et al., 2001 — klasyka, żebyś wiedział o czym mowa
- [ ] Biblioteka `torchstain` (Macenko/Reinhard w PyTorch) — tylko jeśli zdecydujesz się normalizować
- [ ] Kod augmentacji w Cellpose (`cellpose/transforms.py`, `random_rotate_and_resize`) — co już robi sam trening

Decyzje:
- [ ] ❓ Normalizacja barwienia: tak/nie (Cellpose-SAM był trenowany na zróżnicowanych danych; na MoNuSeg z wieloma ośrodkami często wystarcza augmentacja)
- [ ] ❓ Augmentacje dodatkowe ponad wbudowane w Cellpose (HED jitter, blur, JPEG) — każda musi być identyczna dla wszystkich selektorów AL

---

## Faza 5 — Student Cellpose-SAM z wag SAM

📖 Lektura (obowiązkowa):
- [ ] Kirillov et al., 2023, *Segment Anything*, ICCV — architektura ViT-L encodera
- [ ] Stringer et al., 2021, *Cellpose: a generalist algorithm for cellular segmentation*, Nature Methods — flow fields, jak powstają maski z flowów
- [ ] Pachitariu & Stringer, 2022, *Cellpose 2.0: how to train your own model*, Nature Methods — fine-tuning na małych danych, human-in-the-loop (bliskie AL!)
- [ ] Pachitariu, Rariden, Stringer, 2025, *Cellpose-SAM: superhuman generalization for cellular segmentation* (bioRxiv) — architektura, augmentacje, lista danych treningowych
- [ ] Kod: `cellpose/vit.py` (`CPSAM`, `load_pretrained`), `cellpose/train.py` (`train_seg` — domyślne lr, weight decay, epoki, `bsize`, `rescale`)

📖 Kontekst SOTA segmentacji jąder (do rozdziału „related work”):
- [ ] Schmidt et al., 2018, *StarDist*, MICCAI
- [ ] Graham et al., 2019, *HoVer-Net*
- [ ] Hörst et al., 2024, *CellViT*, Medical Image Analysis — ViT/SAM w segmentacji jąder H&E
- [ ] (opcjonalnie) przegląd fine-tuningu SAM w medycynie, np. *MedSAM* (Ma et al., 2024, Nature Communications)

🧪 Kroki:
- [ ] Zbudować studenta z `CPSAM()` + `load_pretrained`; **zalogować missing/unexpected keys** i sprawdzić, że ładuje się `sam_vit_l_0b3195.pth`
- [ ] Sanity check: trening na 1–2 obrazach musi się przeuczyć (loss → ~0) — jeśli nie, błąd w danych/maskach
- [ ] Baseline górny: student trenowany na wszystkich 30 (bez val) obrazach train → ewaluacja na val
- [ ] Baseline dolny: student na małym losowym podzbiorze (rozmiar = start AL)
- [ ] Krzywa uczenia: wynik vs liczba obrazów/kafelków (to jest oś X wykresów AL)
- [ ] Checkpointy + wznawianie (stan modelu, optymalizatora, schedulera, epoki, RNG) — test: przerwij i wznów, wynik ten sam
- [ ] Ewaluacja: AJI, PQ (DQ/SQ), F1/AP przy IoU 0.5 (`cellpose.metrics.average_precision`) — jedna funkcja ewaluacji dla wszystkiego
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

📖 Lektura AL:
- [ ] Settles, 2009, *Active Learning Literature Survey* — terminologia
- [ ] Ren et al., 2021, *A Survey of Deep Active Learning*, ACM Computing Surveys
- [ ] Gal & Ghahramani, 2016, *Dropout as a Bayesian Approximation*, ICML; Gal, Islam, Ghahramani, 2017, *Deep Bayesian Active Learning with Image Data*, ICML — MC-Dropout
- [ ] Sener & Savarese, 2018, *Active Learning for Convolutional Neural Networks: A Core-Set Approach*, ICLR — k-Center
- [ ] Sinha et al., 2019, *Variational Adversarial Active Learning*, ICCV — VAAL
- [ ] Munjal et al., 2022, *Towards Robust and Reproducible Active Learning Using Neural Networks*, CVPR — dlaczego random baseline jest trudny do pobicia
- [ ] Lüth et al., 2023, *Navigating the Pitfalls of Active Learning Evaluation*, NeurIPS — jak poprawnie ewaluować AL
- [ ] AL w segmentacji / patologii — znaleźć 2–3 prace (np. AL dla segmentacji jąder, cost-aware AL na poziomie kafelków/regionów)

📖 Lektura IT2 FLS:
- [ ] Mendel & John, 2002, *Type-2 Fuzzy Sets Made Simple*, IEEE TFS
- [ ] Mendel, *Uncertain Rule-Based Fuzzy Systems* (2. wyd., 2017) — Karnik–Mendel type reduction
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
