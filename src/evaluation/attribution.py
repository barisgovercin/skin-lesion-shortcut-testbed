"""Integrated-Gradients modality attribution split (text vs image)."""
import torch


def text_fraction(text_attr_abs_sum, image_attr_abs_sum):
    """Fraction of total |attribution| assigned to the text modality, in [0, 1]."""
    total = float(text_attr_abs_sum) + float(image_attr_abs_sum)
    return 0.0 if total == 0 else float(text_attr_abs_sum) / total


def ig_text_fraction(model, batch, device, n_steps=16, max_samples=2,
                     internal_batch_size=1):
    """Mean text-attribution fraction over a few batch samples using captum
    Integrated Gradients. batch = (image, input_ids, attention_mask, label).

    Memory-safe: only `max_samples` examples are attributed and captum's step
    expansion is chunked via `internal_batch_size` (full-batch IG OOMs a 16 GB GPU
    because it materialises batch x n_steps ConvNeXt forwards). Returns None if
    captum is unavailable so callers can degrade gracefully.
    """
    try:
        from captum.attr import IntegratedGradients, LayerIntegratedGradients
    except Exception:
        return None

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    image, input_ids, attn, _ = batch
    image = image[:max_samples].to(device)
    input_ids = input_ids[:max_samples].to(device)
    attn = attn[:max_samples].to(device)

    def img_forward(img):
        return model(img, text=(input_ids, attn))

    ig_img = IntegratedGradients(img_forward)
    img_attr = ig_img.attribute(image, baselines=image * 0, n_steps=n_steps,
                                internal_batch_size=internal_batch_size)
    img_sum = img_attr.abs().sum().item()

    # Embedding layer of the HF text model inside TextEncoder (TextEncoder.bert).
    emb_layer = model.text_encoder.bert.embeddings

    def txt_forward(ids):
        return model(image, text=(ids, attn))

    lig = LayerIntegratedGradients(txt_forward, emb_layer)
    txt_attr = lig.attribute(input_ids, baselines=input_ids * 0, n_steps=n_steps,
                             internal_batch_size=internal_batch_size)
    txt_sum = txt_attr.abs().sum().item()

    return text_fraction(txt_sum, img_sum)


# --- Per-token attribution + heatmap (for a single example) ------------------
import numpy as np


def ig_token_attributions(model, image, input_ids, attention_mask, device, n_steps=32):
    """Per-token |Integrated-Gradients| attribution for ONE example (batch of 1).

    Uses LayerIntegratedGradients on the text embedding layer, summing the
    absolute attribution over the hidden dimension to a scalar per token. The
    image is held fixed, so the scores show which words the multimodal model
    leans on. Returns a numpy array of shape [seq_len]. Processing a single
    example with internal_batch_size=1 keeps the vision/text batch dims aligned
    and fits the 16 GB GPU.
    """
    from captum.attr import LayerIntegratedGradients

    model.eval()
    image = image.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    def fwd(ids):
        return model(image, text=(ids, attention_mask))

    lig = LayerIntegratedGradients(fwd, model.text_encoder.bert.embeddings)
    attr = lig.attribute(input_ids, baselines=input_ids * 0, n_steps=n_steps,
                         internal_batch_size=1)
    per_token = attr.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    return per_token


def _merge_wordpieces(tokens, attn_mask, scores):
    """Merge BERT wordpieces into words (summing their attribution) and drop
    special/pad tokens. Returns (words, np.array(scores))."""
    words, wscore = [], []
    for tok, a, s in zip(tokens, attn_mask, scores):
        if int(a) == 0 or tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##") and words:
            words[-1] = words[-1] + tok[2:]
            wscore[-1] = wscore[-1] + float(s)
        else:
            words.append(tok)
            wscore.append(float(s))
    return words, np.asarray(wscore, float)


def render_token_heatmap(tokens, attn_mask, scores_by_model, out_png, title=None,
                         cmap_name="Reds", chars_per_line=82, fontsize=12):
    """Render the same note once per model, colouring each word by its normalised
    (per-model) attribution. `scores_by_model` maps a model label to a [seq_len]
    array aligned with `tokens`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(scores_by_model)
    words = None
    norm = {}
    for mn in model_names:
        w, sc = _merge_wordpieces(tokens, attn_mask, scores_by_model[mn])
        words = w
        top = sc.max() if sc.size and sc.max() > 0 else 1.0
        norm[mn] = sc / top

    # Wrap word indices into lines (shared layout; words are identical per model).
    lines = []
    cur, cur_len = [], 0
    for i, w in enumerate(words):
        if cur and cur_len + len(w) + 1 > chars_per_line:
            lines.append(cur); cur, cur_len = [], 0
        cur.append(i); cur_len += len(w) + 1
    if cur:
        lines.append(cur)

    rows = len(model_names) * (1 + len(lines)) + len(model_names)
    fig, ax = plt.subplots(figsize=(12, 0.34 * rows + (1.0 if title else 0.3)))
    ax.axis("off")
    cmap = plt.get_cmap(cmap_name)
    dy = 1.0 / (rows + 1)
    y = 1.0
    for mn in model_names:
        ax.text(0.0, y, mn, fontsize=fontsize + 1, fontweight="bold", va="top",
                transform=ax.transAxes)
        y -= dy
        for ln in lines:
            x = 0.0
            for i in ln:
                w = words[i]; val = float(norm[mn][i])
                fc = cmap(0.12 + 0.88 * val)
                tc = "white" if val > 0.6 else "black"
                ax.text(x, y, w, fontsize=fontsize, va="top", ha="left", color=tc,
                        family="monospace", transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.15", fc=fc, ec="none"))
                x += (len(w) + 1.7) / chars_per_line
            y -= dy
        y -= dy
    if title:
        fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
