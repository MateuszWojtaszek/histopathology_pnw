# histopathology_pnw

Master's thesis project (Wrocław University of Science and Technology): **active learning (AL)
for nucleus instance segmentation in H&E histopathology**.

The main contribution is an **Interval Type-2 fuzzy logic selector** (IT2 FLS, Mamdani) that picks
samples to annotate based on geometric nucleus features. It is compared with MC-Dropout
uncertainty, k-Center/Coreset and VAAL, and with a random baseline. Annotation is simulated:
a frozen pretrained model (the *oracle*) plays the role of the pathologist.

## Status

| Area | State |
|---|---|
| Repository and environment (uv, DVC) | in progress |
| MoNuSeg preprocessing (XML polygons → instance masks) | in progress |
| Student training with Cellpose-SAM | in progress |
| AL loop and selectors | not started |
| CAMELYON17 | not started |

The working checklist, with reading list, is in [`docs/PLAN.md`](docs/PLAN.md).

## Models

| Role | Model | Weights | Trained here? |
|---|---|---|---|
| **Oracle** | Cellpose-SAM | pretrained by the Cellpose authors (model name: _TBD — `cpsam` or `cpsam_v2`_) | no, frozen, inference only |
| **Student** | Cellpose-SAM architecture (`cellpose.vit.CPSAM`) | original SAM ViT-L (`sam_vit_l_0b3195.pth`) | yes |

The student never starts from a Cellpose checkpoint. Those checkpoints were trained on other cell
datasets, so using them would leak that knowledge into the student and distort AL comparisons.

Things to know about Cellpose 4.x:
- An unknown model name does not raise an error: Cellpose logs a warning and loads `cpsam_v2`.
  Always pass an explicit name or path, and treat that warning as an error.
- Weights are loaded with `strict=False`. Always check the missing/unexpected keys.

## Data

### MoNuSeg
- H&E 40× tiles, 1000×1000 px, from TCGA, multiple organs and hospitals. Instance annotations
  are polygons stored in XML.
- **The official train/test split is used and never changed.** The test set is for final
  evaluation only: it is never used for training, model selection, hyperparameter tuning or AL
  selection. Validation data comes from the training set, split by image.
- Caveat: the Cellpose-SAM training data includes MoNuSeg, so oracle results on MoNuSeg are not
  independent of the oracle's own training.
- Licence: see the dataset page (non-commercial use). Cite Kumar et al. 2017 and 2020.

### CAMELYON17 (later)
Whole-slide images. The raw slides are not tracked by DVC, only derived tiles and manifests are.

### Data versioning (DVC)
Data, masks, model weights and run outputs are **never committed to git**. Datasets and derived
data are versioned with [DVC](https://dvc.org). The default remote is `wcss`, an SSH remote on
the WCSS Lustre filesystem.

## Repository layout

```
histopathology_pnw/
├── src/histopathology_pnw/   # Python package (preprocessing, training, AL, selectors)
├── tests/                    # pytest, small synthetic fixtures, no real data      (planned)
├── configs/                  # experiment configs                                   (planned)
├── data/                     # DVC-managed, git only stores *.dvc files              (planned)
│   ├── raw/                  #   original downloads, never modified
│   └── processed/            #   instance masks, manifests, splits
├── slurm/                    # job scripts for WCSS                                  (planned)
├── docs/PLAN.md              # working checklist and reading list
├── dvc.yaml / params.yaml    # DVC pipeline stages and parameters                    (planned)
├── .dvc/config               # DVC remote configuration (committed)
├── pyproject.toml, uv.lock   # dependencies (uv)
└── AGENTS.md                 # rules for AI assistants working in this repo
```

## Setup on a new machine

Requirements: [uv](https://docs.astral.sh/uv/), SSH access to WCSS.

1. Clone the repository and install the environment:
   ```bash
   git clone <repo-url> && cd histopathology_pnw
   uv sync
   ```
2. Add an SSH host alias `wcss` in `~/.ssh/config` pointing to the WCSS login node, with your
   user and key. The DVC remote URL uses this alias (`ssh://wcss/...`), so it must exist on every
   machine.
3. Download the data:
   ```bash
   uv run dvc pull
   ```

### On WCSS itself
- Point the remote at the local path instead of SSH-ing to the same machine. This only writes
  to `.dvc/config.local`, which is not committed:
  ```bash
  uv run dvc remote modify --local wcss url /lustre/pd01/hpc-martinta-1783643755/mateusz/dvc-store
  ```
- Keep the repository and DVC cache on Lustre, not in `$HOME`.
- Run `dvc pull` on the login node before `sbatch`. Compute nodes may not have network access.
- Code reaches WCSS only through `git pull`. Do not edit code on the cluster.

## Machines

| Machine | Purpose | Device |
|---|---|---|
| macOS | development, tests on small samples | MPS / CPU |
| Ubuntu workstation | debugging, full MoNuSeg runs | CUDA |
| WCSS (Lem cluster) | large runs, Slurm; the login node has no GPU | H100 (CUDA) |

The same code runs on all three. The device is chosen in one place (cuda → mps → cpu), and
paths come from per-machine configuration, never from the code.

## Common commands

```bash
uv sync                          # install / update the environment
uv run python -m <module>        # run code
uv run pytest                    # tests
uv run ruff check .              # lint
uv run ruff format .             # format

uv run dvc pull                  # download data tracked in this commit
uv run dvc add data/raw/<name>   # start tracking a new raw dataset
uv run dvc repro                 # rebuild pipeline outputs that are out of date
uv run dvc push                  # upload new data to the remote
```

After `dvc add` or `dvc repro`, commit the changed `*.dvc` / `dvc.lock` files to git and run
`dvc push`. Otherwise other machines cannot get the data.

## Tooling

- **uv**: Python 3.13, dependencies, lock file
- **DVC** (`dvc[ssh]`): data versioning and pipelines
- **Cellpose 4.x / PyTorch**: oracle and student models
- **ruff**, **pytest**: lint, format, tests
- **GitHub Actions / Dependabot**: CI and dependency updates

## Reproducibility rules

- No data, masks, weights, checkpoints or run outputs in git.
- Every run records the git commit, DVC data version, full config and seed.
- AL experiments are repeated over several seeds.
- Training saves checkpoints and can resume, because Slurm kills jobs at the time limit.
- Secrets live in `.env` (gitignored).

## References

- Kumar et al., *A Dataset and a Technique for Generalized Nuclear Segmentation for Computational
  Pathology*, IEEE TMI, 2017.
- Kumar et al., *A Multi-Organ Nucleus Segmentation Challenge*, IEEE TMI, 2020.
- Stringer et al., *Cellpose: a generalist algorithm for cellular segmentation*, Nature Methods, 2021.
- Pachitariu, Rariden, Stringer, *Cellpose-SAM: superhuman generalization for cellular
  segmentation*, bioRxiv, 2025.
- Kirillov et al., *Segment Anything*, ICCV, 2023.

## Licence

See [LICENSE](LICENSE). Datasets keep their own licences.
