# Experiment: patch-level decoder attribution for latent vs DINO tokens

Date: 2026-04-25

Goal: run one Grad-CAM-style interpretability experiment on the trained decoder that takes a latent stream plus a DINO stream and converts them into an image.

Here "decoder" means `JiT/decoder/model.py::Decoder`, not the diffusion denoiser. The question is:

> For a specific output image patch, how much does the decoder use the aligned latent token versus the aligned DINO token, and does it also pull information from non-aligned tokens?

## Why this experiment fits this decoder

The decoder does:

1. Tokenize the latent map with `latent_tokenizer`.
2. Embed DINO features with `dino_embedder`.
3. Add the same spatial `pos_embed` to both streams.
4. Concatenate context as:

   `ctx_tokens = torch.cat([dino, latent_tokens], dim=1)`

5. Use image query tokens that repeatedly cross-attend into this combined context.
6. Convert final query tokens into image patches with `final_layer` and `unpatchify`.

Because DINO and latent tokens are concatenated in a known order, every context index can be assigned to either:

- DINO token at spatial position `(i, j)`
- latent token at spatial position `(i, j)`

So for one output patch `(u, v)`, we can backpropagate from that patch's RGB pixels to the decoder's context tokens and ask which context tokens had the largest influence.

## Single experiment

Pick one validation/sample example and one visible output patch, then compute gradient-times-activation attribution from that output patch back to the decoder context tokens.

The output should be one figure with:

1. Decoded image with the selected patch outlined.
2. DINO token attribution heatmap.
3. Latent token attribution heatmap.
4. Difference heatmap: `latent_attr - dino_attr`.
5. Optional per-layer heatmaps if hooks are added inside each decoder block.

This gives a direct visual answer for a specific patch:

- bright DINO at the same location means the patch is using local semantic features.
- bright latent at the same location means the patch is using local latent/image information.
- bright nonlocal DINO means the patch is borrowing semantic context from elsewhere.
- bright nonlocal latent means the patch is borrowing texture/layout/detail from elsewhere.
- similar heatmaps mean the decoder is combining both streams for that patch.

## Attribution target

For output patch index `p`, use the scalar:

`target = image[0, :, y0:y1, x0:x1].mean()`

where `(x0, y0, x1, y1)` bounds the selected output patch.

Variants:

- Use mean RGB intensity for a simple first pass.
- Use one color channel if the patch has a strong color artifact.
- Use squared activation, `target = image_patch.square().mean()`, if positive/negative image normalization makes the mean hard to interpret.
- Use a mask over an object part instead of a square patch if the question is object-specific.

Start with patch mean. It is simple and stable.

## Hook point

The most useful first hook is the concatenated context tensor:

`ctx_tokens = torch.cat([dino, latent_tokens], dim=1)`

Make this tensor retain gradients:

`ctx_tokens.retain_grad()`

After backward, compute token attribution:

`attr = (ctx_tokens.grad * ctx_tokens).sum(dim=-1)`

Then split:

- `dino_attr = attr[:, :num_patches]`
- `latent_attr = attr[:, num_patches:]`

Reshape each to `[grid, grid]`.

Use absolute attribution for "how much influence":

`abs_attr = (ctx_tokens.grad * ctx_tokens).sum(dim=-1).abs()`

Use signed attribution for "pushes output up vs down":

`signed_attr = (ctx_tokens.grad * ctx_tokens).sum(dim=-1)`

For the main figure, use absolute attribution. Signed maps can be saved as a secondary debug view.

## Minimal implementation sketch

Add a small script, for example:

`JiT/eval/decoder_patch_attribution.py`

The core forward should mirror `Decoder.forward`, but keep `ctx_tokens`:

```python
latent_tokens = decoder.latent_tokenizer(latent)
latent_tokens = latent_tokens + decoder.pos_embed
dino_tokens = decoder._prepare_dino_tokens(dino)

x = decoder.query_tokens.expand(latent.shape[0], -1, -1)
x = x + decoder.query_pos_embed

ctx_tokens = torch.cat([dino_tokens, latent_tokens], dim=1)
ctx_tokens.retain_grad()

for block in decoder.blocks:
    x = block(x, ctx_tokens)

image = decoder.tokens_to_image(x)
target = image[0, :, y0:y1, x0:x1].mean()
decoder.zero_grad(set_to_none=True)
target.backward()

token_attr = (ctx_tokens.grad * ctx_tokens).sum(dim=-1)[0]
dino_attr = token_attr[:decoder.num_patches]
latent_attr = token_attr[decoder.num_patches:]
```

Important: do not wrap the decoder forward in `torch.no_grad()` for this script.

## Selecting the patch

Use decoder output patch coordinates, not diffusion patch coordinates.

If `decoder.patch_size = 16` and output is `256 x 256`, then the output grid is `16 x 16`.

For patch `(row, col)`:

```python
y0 = row * decoder.patch_size
y1 = (row + 1) * decoder.patch_size
x0 = col * decoder.patch_size
x1 = (col + 1) * decoder.patch_size
```

Start with 3 patches:

- one foreground/object patch.
- one boundary patch.
- one background patch.

But treat each patch as a separate run of the same single experiment.

## What to record

For each selected patch, record:

- `sum(abs(dino_attr))`
- `sum(abs(latent_attr))`
- ratio:

  `dino_share = sum(abs(dino_attr)) / (sum(abs(dino_attr)) + sum(abs(latent_attr)))`

- aligned-token attribution:

  `dino_attr[row, col]`

  `latent_attr[row, col]`

- nonlocal attribution:

  `total_attr - aligned_attr`

This answers two things:

1. stream reliance: DINO share vs latent share.
2. locality: aligned token vs surrounding/global tokens.

## Interpretation

### Case 1: local latent dominates

The selected output patch is mostly reconstructed from the aligned latent token.

This would mean the decoder is acting like a strong image-space latent decoder for that region, with DINO serving as weaker context.

### Case 2: local DINO dominates

The selected output patch is mostly reconstructed from the aligned DINO token.

This would mean the decoder is using DINO as the main local source for semantic appearance in that region.

### Case 3: both local tokens are strong

The decoder uses both streams at the same spatial position.

This is the cleanest evidence that the decoder combines latent and semantic streams simultaneously for that patch.

### Case 4: nonlocal DINO dominates

The patch is using DINO tokens from other positions.

This suggests DINO is supplying global object or class context rather than only local detail.

### Case 5: nonlocal latent dominates

The patch is using latent tokens from other positions.

This suggests the latent stream supplies broader layout, texture consistency, or background continuity.

## Optional stronger version: per-layer attribution

The context tensor is fixed across all decoder blocks, so the first experiment gives total context-token attribution. If we want to see where the decoder starts using DINO vs latent, hook each block's cross-attention output.

For each `DecoderBlock`, retain gradients on the output of:

`block.attn.proj_cross(cross_out)`

Then compute Grad-CAM over decoder query tokens per layer. This shows which image query patches are affected by cross-attention at each depth.

This is useful later, but the first experiment should stay focused on context-token attribution because it directly answers DINO-token vs latent-token usage.

## Sanity checks

Run these before trusting the heatmaps:

1. If the target patch changes, the heatmap should move or change.
2. If the target is the whole image mean, attribution should become more diffuse.
3. If DINO tokens are detached before the forward pass, DINO attribution should go to zero.
4. If latent tokens are detached before the forward pass, latent attribution should go to zero.
5. If the same image patch is run twice in eval mode, heatmaps should match.

## Expected output

One saved image per selected patch:

`decoder_attr_sample000123_patch07_09.png`

Panel layout:

- reconstructed image with patch box.
- DINO attribution heatmap.
- latent attribution heatmap.
- latent minus DINO heatmap.

Also save a small JSON:

```json
{
  "sample_id": 123,
  "patch_row": 7,
  "patch_col": 9,
  "dino_abs_sum": 0.0,
  "latent_abs_sum": 0.0,
  "dino_share": 0.0,
  "dino_aligned_abs": 0.0,
  "latent_aligned_abs": 0.0,
  "dino_nonlocal_abs": 0.0,
  "latent_nonlocal_abs": 0.0
}
```

## Success criterion

This experiment succeeds if, for one specific output patch, we can say:

- the decoder relies more on DINO or latent for that patch,
- whether the relied-on information is local or nonlocal,
- and whether both streams are being used at the same time.

That is the decoder-level answer we want. It avoids changing the diffusion model, avoids broad FID ablations, and directly probes how the latent+DINO image decoder uses its two token streams.
