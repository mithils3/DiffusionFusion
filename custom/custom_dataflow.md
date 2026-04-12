# CustomDiT Data Flow Diagram

Example shown for `CustomDiT-B/2-4C`, `latent_size=32`, `dino_patches=16`, `B=8`.

```mermaid
graph TD
  LAT["latent stream z_lat<br/>[B,4,32,32]"] --> LATPATCH["latent patch embed<br/>[B,N_lat,D]"]
  LATPATCH --> LATPOS["add latent pos + latent type embedding"]
  DINO["DINO stream z_dino<br/>[B,768,16,16]"] --> DINOFLAT["flatten + linear project<br/>[B,N_dino,D]"]
  DINOFLAT --> DINOPOS["add DINO pos + DINO type embedding"]

  T["timestep t<br/>[B]"] --> TEMB["timestep embed<br/>[B,D]"]
  Y["label y<br/>[B]"] --> YEMB["label embed + CFG dropout<br/>[B,D]"]
  TEMB --> C["conditioning c = t_emb + y_emb<br/>[B,D]"]
  YEMB --> C

  LATPOS --> CAT["concat streams<br/>[B,N_lat+N_dino,D]"]
  DINOPOS --> CAT
  CAT --> BLOCKS["CustomLightningDiT blocks<br/>RMSNorm + RoPE per stream + AdaLN + SwiGLU"]
  C --> BLOCKS

  BLOCKS --> SPLIT["split token ranges"]
  SPLIT --> LATHEAD["latent final head + unpatchify<br/>[B,4,32,32]"]
  SPLIT --> DINOHEAD["DINO final head + reshape<br/>[B,768,16,16]"]
```

Notes:

- `N_lat = (latent_size / latent_patch_size)^2`; for `CustomDiT-B/2-4C`, `N_lat = 256`.
- `N_dino = dino_patches^2`; with the default DINO grid, `N_dino = 256`.
- For `/4-4C` variants, only the latent token count changes; the DINO stream stays on the original shard contract.
