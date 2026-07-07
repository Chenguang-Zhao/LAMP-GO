"""Loading and batching helpers shared by train.py and predict.py.

These read the files produced by data_preprocessing.py from a single data
directory.
"""
import csv
import os

import numpy as np
import scipy.sparse as sp

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MAX_SEQ_LEN = 1024      # residues per protein are capped to keep fine-tuning tractable
TOKEN_BUDGET = 8192     # roughly the max tokens (batch_size * padded_length) per batch
MAX_BATCH = 32


def clean(seq):
    """Uppercase, drop non-standard amino acids, and cap the length."""
    return "".join(c for c in seq.upper() if c in VALID_AA)[:MAX_SEQ_LEN] or "A"


def load_terms(data_dir):
    """The GO id for each output column, ordered by column index."""
    with open(os.path.join(data_dir, "candidate_terms.txt")) as fh:
        return fh.read().split()


def load_split(data_dir, split):
    """Return (ids, sequences, labels).

    labels is a dense float32 array [n, n_terms], or None for a split without a
    labels file (e.g. test, the prediction target).
    """
    ids, seqs = [], []
    with open(os.path.join(data_dir, f"{split}.csv")) as f:
        for row in csv.DictReader(f):
            ids.append(row["EntryID"])
            seqs.append(clean(row["sequence"]))
    label_path = os.path.join(data_dir, f"{split}_labels.npz")
    Y = sp.load_npz(label_path).toarray().astype(np.float32) if os.path.exists(label_path) else None
    if Y is not None:
        assert Y.shape[0] == len(seqs), f"{split}: {Y.shape[0]} label rows vs {len(seqs)} sequences"
    return ids, seqs, Y


def make_batches(lengths):
    """Group sequence indices into length-sorted batches under the token budget.

    Sorting by length keeps padding small; each batch stays within TOKEN_BUDGET
    tokens and MAX_BATCH sequences.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i] + 2)          # +2 for BOS/EOS tokens
        if cur and (new_max * (len(cur) + 1) > TOKEN_BUDGET or len(cur) >= MAX_BATCH):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = lengths[i] + 2
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def write_submission(ids, scores, path, terms, threshold=1e-3, topk=1500):
    """Write a 3-column CAFA submission: EntryID <TAB> GO_term <TAB> score.

    Per protein, keep terms scoring >= threshold, then the top-k of those (the
    CAFA cap is 1500 predicted terms per protein).
    """
    scores = np.asarray(scores)
    terms = np.asarray(terms)
    with open(path, "w") as fh:
        for i, pid in enumerate(ids):
            row = scores[i]
            keep = np.where(row >= threshold)[0]
            if len(keep) > topk:
                keep = keep[np.argsort(row[keep])[::-1][:topk]]
            for j in keep:
                fh.write(f"{pid}\t{terms[j]}\t{round(float(row[j]), 3)}\n")
