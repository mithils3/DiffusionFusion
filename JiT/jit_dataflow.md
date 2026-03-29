# JiT Data Flow Diagram

Example shown for `JiT-B/16`, `img_size=256`, `B=8`.

```mermaid
flowchart LR
  Z["z (noisy image)<br/>[B,C,H,W] e.g. [8,3,256,256]"] --> XEMB["BottleneckPatchEmbed<br/>Conv p x p (stride p) + Conv1x1<br/>x_tokens: [B,N,D] e.g. [8,256,768]"]
  XEMB --> POS["Add fixed 2D sin-cos pos_embed<br/>[B,N,D]"]
  POS --> PRE["JiT blocks 0..(in_context_start-1)<br/>[B,N,D]"]

  T["t (timestep)<br/>[B]"] --> TEMB["TimestepEmbedder<br/>t_emb: [B,D]"]
  Y["y (class id)<br/>[B]"] --> YEMB["LabelEmbedder<br/>y_emb: [B,D]"]
  TEMB --> C["c = t_emb + y_emb<br/>[B,D]"]

  YEMB --> C
  YEMB --> ICT["In-context tokens<br/>repeat(y_emb, Lc) + in_context_posemb<br/>[B,Lc,D] e.g. [8,32,768]"]

  PRE --> CAT["Concat(in-context, patch tokens)<br/>[B,N+Lc,D] e.g. [8,288,768]"]
  ICT --> CAT
  CAT --> POST["JiT blocks in_context_start..(depth-1)<br/>RoPE with in-context<br/>[B,N+Lc,D]"]
  POST --> DROP["Drop first Lc tokens<br/>[B,N,D]"]

  C --> PRE
  C --> POST

  DROP --> FINAL["FinalLayer (AdaLN + Linear)<br/>velocity patches [B,N,p*p*C] e.g. [8,256,768]"]
  FINAL --> UP["unpatchify<br/>v_pred: [B,C,H,W] e.g. [8,3,256,256]"]
```

## Terms

- `B`: batch size.
- `C`: image channels (`3` for RGB).
- `H, W`: image height and width.
- `p`: patch size (`16` or `32`).
- `N`: number of patches, `N = (H/p) * (W/p)`.
- `D`: token hidden size (`768` for B, `1024` for L, `1280` for H variants).
- `Lc`: in-context token count (`in_context_len`, usually `32`).
- `depth`: number of transformer blocks.
- `z`: noisy input image to JiT.
- `t`: diffusion timestep per sample.
- `y`: class label id.
- `t_emb`: timestep embedding vector.
- `y_emb`: class embedding vector.
- `c`: conditioning vector used by all JiT blocks (`c = t_emb + y_emb`).
- `pos_embed`: fixed 2D sine-cosine positional embedding for patch tokens.
- `in_context_posemb`: learned positional embedding for in-context tokens.
- `RoPE`: rotary positional embedding used inside attention.
- `JiT block`: transformer block (attention + SwiGLU MLP) with AdaLN modulation from `c`.
- `AdaLN`: adaptive LayerNorm style modulation (shift/scale/gates generated from `c`).
- `FinalLayer`: final AdaLN + linear projection from token features to velocity patches.
- `unpatchify`: reshapes patch outputs back to full latent/DINO velocity tensors.
- `v_pred`: predicted velocity at the current step.

## How To Render

1. In Markdown, keep the diagram inside a fenced `mermaid` code block.
2. Mermaid Live Editor: https://mermaid.live
3. VS Code: install a Mermaid Markdown preview extension, then open preview.
4. CLI render:
   - `npm i -g @mermaid-js/mermaid-cli`
   - `mmdc -i JiT/jit_dataflow.mmd -o JiT/jit_dataflow.svg`
