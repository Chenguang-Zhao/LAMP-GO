"""LAMP-GO model definition.

LAMP-GO = LoRA fine-tuning + Attention pooling + Multi-layer PLM, for GO term
prediction from an amino-acid sequence alone.

    tokens
      -> ESM-C 600M (+ LoRA adapters)         -> hidden_states [n_layers, B, L, D]
      -> LayerMix  (ELMo scalar mix of layers) -> [B, L, D]
      -> AttnPool  (gated attention over residues) -> [B, D]
      -> Head      (Linear -> GELU -> Dropout -> Linear) -> logits [n_terms]

"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from esm.models.esmc import ESMC

# ---- fixed architecture constants (ESM-C 600M) -----------------------------
BACKBONE = "esmc_600m"
D_MODEL = 1152          # ESM-C 600M hidden width
N_TERMS = 6416          # size of the candidate GO vocabulary

# LoRA is applied to the attention (fused QKV + output) and both FFN
# projections of every transformer block.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"]

ATTN_DIM = 256          # gated-attention hidden width
HIDDEN = 1024           # head hidden width
DROPOUT = 0.3


class LayerMix(nn.Module):
    
    def __init__(self, n_layers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))   # uniform start
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, hiddens):                        # hiddens [n_layers, B, L, D]
        s = torch.softmax(self.w.float(), dim=0).to(hiddens.dtype).view(-1, 1, 1, 1)
        return self.gamma.to(hiddens.dtype) * (s * hiddens).sum(0)   # [B, L, D]


class AttnPool(nn.Module):
    """Gated attention pooling over residues: softmax(score) . hidden."""

    def __init__(self, d, a):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, a), nn.Tanh(), nn.Linear(a, 1))

    def forward(self, emb, mask):            # emb [B, L, D]  mask [B, L] (True = keep)
        s = self.score(emb).squeeze(-1)      # [B, L]
        s = s.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(s.float(), dim=1).to(emb.dtype).unsqueeze(-1)  # [B, L, 1]
        return (alpha * emb).sum(1)          # [B, D]


class Head(nn.Module):
    """Two fully-connected layers on the pooled backbone output."""

    def __init__(self, d_in, d_hidden, d_out, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class LampGO(nn.Module):
    """The full model: LoRA-wrapped ESM-C backbone + LayerMix + AttnPool + Head."""

    def __init__(self, esmc, mix, pool, head, pad_id):
        super().__init__()
        self.esmc = esmc
        self.mix = mix
        self.pool = pool
        self.head = head
        self.pad_id = pad_id

    def forward(self, toks):                            # toks [B, L] token ids
        hiddens = self.esmc(sequence_tokens=toks).hidden_states   # [n_layers, B, L, D]
        combined = self.mix(hiddens)                    # [B, L, D]
        mask = toks != self.pad_id                      # [B, L] (True = real residue)
        pooled = self.pool(combined, mask)              # [B, D]
        return self.head(pooled)                        # [B, n_terms]


def build_model(device):
    """Instantiate ESM-C 600M with LoRA adapters and the LAMP-GO head.

    Returns (model, tokenize) where tokenize(list_of_sequences) -> token tensor.
    """
    base = ESMC.from_pretrained(BACKBONE).to(device)
    tokenize = base._tokenize
    pad_id = base.tokenizer.pad_token_id
    n_layers = len(base.transformer.blocks)

    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          target_modules=LORA_TARGETS, bias="none")
    esmc = get_peft_model(base, lora_cfg)

    mix = LayerMix(n_layers).to(device)
    pool = AttnPool(D_MODEL, ATTN_DIM).to(device)
    head = Head(D_MODEL, HIDDEN, N_TERMS, DROPOUT).to(device)
    model = LampGO(esmc, mix, pool, head, pad_id).to(device)
    return model, tokenize


def trainable_state(model):
    """Only the parameters we train: LoRA adapters + layer-mix + pool + head.
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if "lora_" in k or k.startswith(("mix.", "pool.", "head."))}
