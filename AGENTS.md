# AGENTS.md — histopathology_pnw

## Project
Master's thesis research (Politechnika Wrocławska): active learning (AL) for nucleus
segmentation in H&E histopathology. Main contribution: an Interval Type-2 fuzzy logic
selector (IT2 FLS, Mamdani) driven by geometric nucleus features, compared with
MC-Dropout uncertainty, k-Center/Coreset and VAAL. AL is simulated: a frozen oracle
model replaces the pathologist as the annotator. Datasets will change over the project.

## Current focus
1. Repository and environment setup.
2. MoNuSeg preprocessing (images + XML polygon annotations -> instance masks).
3. Preparing student training with Cellpose-SAM.
Not yet: AL loop, selectors, CAMELYON17.

## Models — never mix these up
- **Oracle**: fully pretrained Cellpose-SAM, model `cpsam` (original release, April 2025),
  frozen, inference only. Weights are tracked with DVC at `models/oracle/cpsam`, downloaded
  from huggingface.co/mouseland/cellpose-sam at commit 7c61431b5fbb078f3296754bd15d9f51b320f837
  (SHA-256 `e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2`).
  Always load it by this path and verify the SHA-256 first (a missing path silently falls back
  to `cpsam_v2`; a wrong file loads without error). Never load it by name and never use
  `cpsam_v2`, which is the Cellpose 4.2 default.
- **Student**: Cellpose-SAM architecture whose encoder starts from the ORIGINAL SAM ViT-L
  weights, never trained on cell or histopathology images. Never initialise the student from
  any Cellpose checkpoint: that leaks training on other cell datasets into the student.

Cellpose 4.x facts (checked in cellpose 4.2.1.1):
- `models.CellposeModel()` defaults to `pretrained_model="cpsam_v2"`.
- An unknown model name does NOT fail: it logs a warning and silently loads `cpsam_v2`.
  Always pass an explicit, existing name or path, and treat that warning as an error.
- `pretrained_model=None` raises ("training from scratch is not implemented"), so the
  student is built from `cellpose.vit.CPSAM()` + `load_pretrained(root)`, which loads
  `sam_vit_l_0b3195.pth`.
- `BaseModel.load_model()` uses `strict=False`, so a wrong checkpoint loads without error.
  Check the missing/unexpected keys whenever weights are loaded.

## Data rules
- Use the official MoNuSeg train/test split. Never change the split files.
- The MoNuSeg test set is never used for training, model selection, hyperparameter tuning
  or AL selection. It is for final evaluation only.
- The Cellpose-SAM paper lists MoNuSeg (also MoNuSAC, CryoNuSeg, NuInsSeg, PanNuke, CoNIC)
  among its training data, so oracle results on these datasets are not independent.
- Never commit data, masks, model weights, checkpoints or run outputs. Data is versioned
  with DVC; raw CAMELYON17 is not tracked by DVC.
- Never hard-code paths. Dataset and output locations come from per-machine config.

## Environment
- Python 3.13, managed with uv. `uv.lock` and `.python-version` are committed.
- Install / update the environment: `uv sync`
- Run code: `uv run python -m <module>`
- Tests: `uv run pytest` · Lint: `uv run ruff check .` · Format: `uv run ruff format .`
  (ruff and pytest are added as dev dependencies when the code starts)

## Machines — the same code must run on all three
- **macOS**: development and tests on small samples. GPU via MPS when available.
- **Ubuntu workstation**: NVIDIA GPU (CUDA), debugging and full MoNuSeg runs.
- **WCSS (PWr HPC, Lem cluster)**: Slurm, H100 GPUs, large runs (CAMELYON17 later).
  The login node has no GPU. GPU hours are a limited, shared budget.
- Pick the device in one place (cuda -> mps -> cpu). Never write `"cuda"` anywhere else.
- Code reaches WCSS only through git (`git pull`), never by editing on the cluster.
- Training must save checkpoints and be able to resume (Slurm kills jobs at the time limit).

## How to work in this repo
- Mateusz writes the code and stays in control of every step. Help with configuration,
  answer questions, explain.
- Write code only when asked. Edit files only when told to, and ask before editing.
- Ask before adding or upgrading any dependency.
- Never start training runs or submit Slurm jobs; give the command instead.
- Never commit, push or rewrite git history.
- Secrets live in `.env` (gitignored). Never print or commit them.