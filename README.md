# LAMP-GO

**LAMP-GO** (**L**oRA Fine-tuning + **A**ttention Pooling + Layer-**M**ixing **P**LM
for **GO** prediction) predicts Gene Ontology terms for a protein from its
amino-acid sequence alone without homology search, structure, and etc. 

### Files

| File | Purpose |
|---|---|
| `model.py` | LAMP-GO architecture (LayerMix, AttnPool, Head, backbone wiring) |
| `data_preprocessing.py` | Build the training data from the raw CAFA-5 Kaggle download |
| `data.py` | Sequence loading, length-sorted batching, submission writing |
| `train.py` | LoRA fine-tuning loop with early stopping |
| `predict.py` | Score proteins with a trained checkpoint |
| `trained_lamp_go.pt` | Trained weights |

## Environment setup

Python 3.10 
A CUDA GPU with about 12 GB memory.

```bash
conda create -n lampgo python=3.10 -y
conda activate lampgo
pip install -r requirements.txt
```

The `esm` package downloads the ESM-C 600M weights automatically on first use.

## Data preprocessing

Download the CAFA-5 data from
[Kaggle](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction)
and unzip it, then build the training-ready files:

```bash
python data_preprocessing.py --kaggle-dir /path/to/cafa-5 --out-dir data
```

This writes into `data/`:

- `candidate_terms.txt` — the 6416 GO terms (annotated ≥ 50 times in train), one per output column
- `train.csv`, `valid.csv` — `EntryID, sequence, terms` (a reproducible 90/10 split)
- `test.csv` — `EntryID, sequence` (the prediction target, no labels)
- `train_labels.npz`, `valid_labels.npz` — sparse 0/1 label matrices, row-aligned to the CSVs

## Prediction

Score any split (e.g. `test`) with the released checkpoint:

```bash
python predict.py --data-dir data --weights trained_lamp_go.pt --split test --out-dir .
```

This writes:

- `test_scores.npy` — `float32 [n_proteins, 6416]` predicted probabilities
- `test.tsv` — CAFA submission (`EntryID <TAB> GO_term <TAB> score`)

To predict for your own proteins, put them in a CSV with `EntryID` and `sequence`
columns plus the `candidate_terms.txt` in the same directory, then pass its
`--split` name.

## Retraining

To reproduce the trained checkpoint from scratch:

```bash
python train.py --data-dir data --out lamp_go.pt
```

Training LoRA-fine-tunes ESM-C 600M and trains the layer mix, attention pool, and
head, early-stopping on validation loss (patience 3, up to 30 epochs). It saves the
best epoch's weights to `lamp_go.pt` (same format as the released
`trained_lamp_go.pt`) alongside a `lamp_go.history.json` with the per-epoch loss and
validation micro-Fmax. Then predict with `--weights lamp_go.pt`.

## License

MIT — see [LICENSE](LICENSE).
