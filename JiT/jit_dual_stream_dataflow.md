# JiT Dual-Stream Base Data Flow

Example shown for the custom model trained in [`sbatch/jit_training.sbatch`](../sbatch/jit_training.sbatch):

- Model: `JiT-Dual-B/2-4C-896`
- Latent input: `[B,4,32,32]`
- DINO input: `[B,768,16,16]`
- Hidden size: `896`
- Context tokens per stream: `32`
- Cross-fusion layers: `4` and `8`

```mermaid
flowchart LR
  subgraph Inputs
    LAT["Noisy latent z_latent<br/>[B,4,32,32]"]
    DINO["Noisy DINO z_dino<br/>[B,768,16,16]"]
    T["timestep t<br/>[B]"]
    Y["class id y<br/>[B]"]
  end

  T --> TEMB["TimestepEmbedder<br/>t_emb [B,896]"]
  Y --> YEMB["LabelEmbedder<br/>y_emb [B,896]"]
  TEMB --> C["conditioning c = t_emb + y_emb<br/>drives AdaLN in every block"]
  YEMB --> C

  LAT --> LEMB["BottleneckPatchEmbed<br/>patch=2 -> 256 latent tokens"]
  DINO --> DEMB["Flatten 16x16 grid + Linear<br/>256 DINO tokens"]
  LEMB --> LPOS["+ shared 2D sin-cos pos_embed"]
  DEMB --> DPOS["+ shared 2D sin-cos pos_embed"]

  YEMB --> LCTX["32 latent in-context tokens<br/>y_emb + latent_in_context_posemb"]
  YEMB --> DCTX["32 DINO in-context tokens<br/>y_emb + dino_in_context_posemb"]

  LPOS --> L0["Latent tower blocks 0-3<br/>local self-attn + SwiGLU"]
  DPOS --> D0["DINO tower blocks 0-3<br/>local self-attn + SwiGLU"]
  LCTX --> L0
  DCTX --> D0
  C --> L0
  C --> D0

  L0 --> L1["Latent tower blocks 4-7"]
  D0 --> D1["DINO tower blocks 4-7"]
  L1 <--> F4["Cross-fusion @ block 4<br/>latent queries DINO<br/>DINO queries latent"]
  D1 <--> F4

  F4 --> L2["Latent tower blocks 8-11"]
  F4 --> D2["DINO tower blocks 8-11"]
  L2 <--> F8["Cross-fusion @ block 8<br/>same bidirectional cross-attn"]
  D2 <--> F8

  F8 --> LOUT["Drop 32 latent context tokens"]
  F8 --> DOUT["Drop 32 DINO context tokens"]
  C --> LFINAL["Latent FinalLayer<br/>AdaLN + Linear"]
  C --> DFINAL["DINO FinalLayer<br/>AdaLN + Linear"]
  LOUT --> LFINAL
  DOUT --> DFINAL
  LFINAL --> XOUT["x_pred latent output<br/>[B,4,32,32]"]
  DFINAL --> FOUT["dino_pred feature output<br/>[B,768,16,16]"]

  R["RoPE note:<br/>rotary embeddings apply only to spatial tokens,<br/>not the in-context prefix"] -.-> L0
  R -.-> D0
  R -.-> F4
  R -.-> F8
```

## Notes

- The latent and DINO streams keep separate transformer weights all the way through the network.
- Both streams receive the same timestep/class conditioning vector `c`, but each stream has its own learned in-context positional embeddings.
- Cross-fusion is bidirectional: the latent stream attends to DINO features, and the DINO stream attends back to latent features.
- The model predicts both the denoised latent and the denoised DINO feature map at every sampling step.

## How To Render

1. Open this Markdown file in a Mermaid-capable preview.
2. Or render the raw file directly with:
   - `mmdc -i JiT/jit_dual_stream_dataflow.mmd -o JiT/jit_dual_stream_dataflow.svg`
