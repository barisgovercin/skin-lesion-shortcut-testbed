"""BioClinicalBERT text encoder for clinical referral notes.

Mirrors Watson et al. (2026): a pretrained clinical BERT
(emilyalsentzer/Bio_ClinicalBERT, trained on MIMIC-III) encodes the freetext,
and its embedding is late-fused with the vision (and metadata) features.

`transformers` is imported lazily inside the constructor so the rest of the
codebase - and the offline fusion tests, which inject a stub encoder - neither
import the library nor trigger a model download.
"""

import torch.nn as nn

DEFAULT_TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


class TextEncoder(nn.Module):
    """Wrap a HuggingFace BERT, exposing a fixed-size sentence embedding.

    The [CLS] token of the last hidden state is used as the note representation
    (`feature_dim` == the model's hidden size, 768 for BioClinicalBERT).
    """

    def __init__(self, model_name=DEFAULT_TEXT_MODEL, pretrained=True, freeze=False):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            # Random init from the architecture only (no download of weights);
            # used when a checkpoint will be loaded afterwards.
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(model_name))

        self.feature_dim = self.bert.config.hidden_size
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]  # [CLS]


def build_tokenizer(model_name=DEFAULT_TEXT_MODEL):
    """Return the matching HuggingFace tokenizer (lazy import)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)
