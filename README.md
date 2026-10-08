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

| Role | Model | Weights | Notes |
|---|---|---|---|
| **Oracle** | Cellpose-SAM `cpsam` | `models/oracle/cpsam` (DVC) | frozen, inference only |
| **Student** | Cellpose-SAM architecture (`cellpose.vit.CPSAM`) | encoder from original SAM ViT-L (`sam_vit_l_0b3195.pth`) | trained here; never initialised from a Cellpose checkpoint |

The student never starts from a Cellpose checkpoint. Those checkpoints were trained on other cell
datasets, so using them would leak that knowledge into the student and distort AL comparisons.

### Oracle weights

Downloaded from [mouseland/cellpose-sam](https://huggingface.co/mouseland/cellpose-sam) at a fixed
commit and tracked with DVC, so every machine uses byte-identical weights and WCSS compute nodes
need no internet access:

- commit: `7c61431b5fbb078f3296754bd15d9f51b320f837`
- file: `cpsam`, 1 233 587 898 bytes
- SHA-256: `e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2`

### Why `cpsam` and not `cpsam_v2`

Cellpose 4.2 ships two Cellpose-SAM checkpoints with the same SAM ViT-L backbone:

- `cpsam` (April 2025) is the model described in the Cellpose-SAM paper, which documents
  its training data.
- `cpsam_v2` (June 2026) is the Cellpose 4.2 default. It "includes a fix in the training for
  low contrast regions" and predicts fewer spurious masks there. Its training data is not
  documented.

We use `cpsam` because its training data is documented. This matters because that data
includes MoNuSeg, so oracle scores on MoNuSeg are not independent, and we can only reason
about such overlap for a model whose training set is known.

### Things to know about Cellpose 4.x
- A path that does not exist, or an unknown model name, does not raise an error: Cellpose logs
  a warning and loads `cpsam_v2`. Always load the oracle by path and verify its SHA-256 first.
- Any checkpoint without a DINO `cls_token` is treated as SAM ViT-L, and weights are loaded with
  `strict=False`, so a wrong file loads silently. Check the missing/unexpected keys.

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
Storage layout (`$HOME`: 50 GB / 1M files quota, snapshot backups; `PD`: project space on Lustre):

| What | Where | Why |
|---|---|---|
| Repository, `.venv`, uv cache | `$HOME` | uv hardlinks packages from its cache into `.venv`, which only works on one filesystem; the many small files of the environment stay out of the shared PD file limit |
| DVC cache and remote | PD | data lives on Lustre; the repo only holds symlinks to it |
| Checkpoints, runs, logs | PD, via per-machine config | never written into the repo directory on the cluster |

DVC-tracked pipeline outputs (e.g. masks) stay under `data/` in the repo, because DVC 3 does not
support outputs outside the repo. With symlink caching they are moved to the PD cache and only
a symlink stays in `$HOME`.

One-time setup (writes only to `.dvc/config.local`, which is not committed):
```bash
export PD=/lustre/pd01/hpc-martinta-1783643755   # put this in ~/.bashrc
uv run dvc remote modify --local wcss url $PD/mateusz/dvc-store   # local path, no SSH to itself
uv run dvc cache dir --local $PD/mateusz/dvc-cache
uv run dvc config --local cache.type symlink   # hardlinks cannot cross from $HOME to PD
uv run dvc pull
```

- Do not move `UV_CACHE_DIR` to PD: it would split the uv cache and `.venv` across filesystems.
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
