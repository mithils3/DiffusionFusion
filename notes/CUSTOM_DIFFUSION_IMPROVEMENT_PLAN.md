  ---                                                                                                                                                                                                            
  Where You Stand                                                                                                                                                                                                
                                                                                                                                                                                                                 
  FID ~13 on the 50k balanced eval at 80 epochs is still where an undertrained JiT model should be. This is not an architecture problem yet — it's mostly a training-duration and eval-strategy problem.                                                             
                                                                                                                                                                                                                 
  Reference points from the JiT paper (pixel-space, batch 1024, LR 2e-4):

  ┌──────────┬────────┬────────┐
  │  Model   │ 200 ep │ 600 ep │
  ├──────────┼────────┼────────┤
  │ JiT-B/16 │ 4.37   │ 3.66   │
  ├──────────┼────────┼────────┤
  │ JiT-L/16 │ 2.79   │ 2.36   │
  ├──────────┼────────┼────────┤
  │ JiT-H/16 │ 2.29   │ 1.86   │
  └──────────┴────────┴────────┘

  The current recipe already matches JiT-scale batch/LR: effective batch 1024 via gradient accumulation and LR 2e-4 via blr scaling. The main remaining gap is training length: 80 epochs vs. the 200-600 epoch reference range. So FID 13 still tracks.

  Current SOTA frontier on ImageNet-256: FID 1.1-1.4 (REPA-E, DDT-XL, LightningDiT).

  ---
  The Plan (Updated + Prioritized)

  Phase 1: Finish the Baseline (Expected: FID 13 → 4-7)

  The biggest remaining lever is still scale, but now it is mostly duration plus evaluation discipline rather than batch/LR.

  1. Train to 200 epochs, eval every 40
  2. Sweep at each checkpoint: EMA {0.9996, 0.9998, 0.9999} × CFG {1.5, 2.0, 2.5, 2.9, 3.0, 3.5}
  3. Always use 50k balanced samples for FID

  Phase 2: Free Inference Improvements (Expected: 0.5-2.0 FID, Zero Training Cost)

  These are sampler-side changes you can apply to any checkpoint:

  1. Verify Heun 50-step is what you're using (your code supports both Euler and Heun)
  2. CFG interval tuning — try (0.0, 1.0), (0.1, 0.9), (0.2, 0.8) in addition to current (0.1, 1.0). The guidance interval paper (NeurIPS 2024) showed this alone can improve FID by 0.4+
  3. Time shifting at inference (FLUX-style) — shift the ODE integration schedule to allocate more steps to intermediate noise levels. LightningDiT used this to reach FID 1.35. This is a change to the
  timesteps = linspace(0, 1, steps+1) line in your sampler
  4. beta-CFG — dynamic guidance strength over the trajectory instead of a flat scalar. Recent work shows this outperforms static CFG consistently

  Phase 3: Architecture (Expected: Additional 1-3 FID)

  After Phase 1 establishes a proper 200-epoch baseline:

  1. DDT-style wide shallow head — 2 layers, 2048 width, on backbone output before final prediction. This is THE highest-impact architectural change for high-dimensional latent spaces (RAE/DDT evidence). Your
  DINO stream is 768-dim, exactly where this helps most.
  2. Token-type embeddings — learned modality tags for latent vs DINO vs in-context tokens. Cheap, low-risk.
  3. Stream loss balance — sweep (λ_latent, λ_dino) in {(1,1), (1,0.5), (1,0.25), (0.5,1)}. The DINO stream may be dominating optimization without proportionally helping image quality.
  4. Per-stream noise scales — SDXL latents and DINO features have very different statistics. Try dino_noise_scale in {0.5, 0.75, 1.0} with latent_noise_scale=1.0.

  Phase 4: Training Technique Upgrades (Expected: Additional 0.5-1.5 FID)

  1. Min-SNR-gamma loss weighting (gamma=5) — clamps loss weights by signal-to-noise ratio, prevents high-noise timesteps from dominating. Paper shows 3.4x faster convergence. This layers on top of your
  existing v-loss.
  2. Test posterior mean latents — your VAE extraction uses .sample() (stochastic), freezing one random draw per image forever. Re-extract with posterior mean for a cleaner training signal.
  3. VF loss (from LightningDiT) — velocity field loss formulation that accelerates convergence up to 2.7x. Worth testing as an alternative to your current v-loss.

  Phase 5: Representation Alignment (Expected: Additional 1-3 FID)

  Critical insight from the PixelREPA paper (March 2026): Vanilla REPA actually hurts JiT as training progresses due to information asymmetry (denoising in high-dim pixel/latent space vs. compressed semantic
  target). If you add representation alignment, use PixelREPA (masked transformer adapter), NOT vanilla REPA.

  - PixelREPA improved JiT-B/16 from 3.66 → 3.17 and JiT-H/16 to 1.81, with 2x faster convergence
  - Implementation: add a masked transformer adapter that projects backbone features to DINO dimension, with patch-wise cosine alignment loss against frozen DINO features

  Phase 6: Scale Up or Pivot

  If after Phases 1-5 at 200+ epochs you're in the 3-5 range:
  - Scale to JiT-L — JiT-L at 200ep (2.79) already beats JiT-B at 600ep (3.66)
  - Train to 400-600 epochs

  If still stuck above 5:
  - RAE-style pivot — pretrained representation encoder as latent interface + DDT head. This is the clearest path to FID < 2 but a larger research fork.

  ---
  Priority Execution Order

  ┌─────┬───────────────────────────────────────────┬─────────────────────────────────────┬────────────────────────┐
  │  #  │                   What                    │           Expected Impact           │          Cost          │
  ├─────┼───────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────┤
  │ 1   │ 200 epochs + checkpoint/eval sweeps       │ FID 13 → 4-7                        │ Compute only           │
  ├─────┼───────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────┤
  │ 2   │ Heun + CFG interval sweep + time shifting │ 0.5-2.0 FID improvement             │ Free (inference only)  │
  ├─────┼───────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────┤
  │ 3   │ DDT wide head + type embeddings           │ 1-3 FID improvement                 │ 1 training run         │
  ├─────┼───────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────┤
  │ 4   │ Min-SNR-gamma + posterior mean latents    │ 0.5-1.5 FID improvement             │ 1 training run         │
  ├─────┼───────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────┤
  │ 5   │ PixelREPA (NOT vanilla REPA)              │ 1-3 FID + 2x faster convergence     │ Architecture change    │
  ├─────┼───────────────────────────────────────────┼─────────────────────────────────────┼────────────────────────┤
  │ 6   │ Scale to JiT-L or RAE pivot               │ Path to FID < 2                     │ Major compute/research │
  └─────┴───────────────────────────────────────────┴─────────────────────────────────────┴────────────────────────┘

  The existing plan in your repo is good — the main things it's still missing are: EMA sweep support in eval, time shifting at inference, beta-CFG, min-SNR-gamma weighting, the PixelREPA finding (vanilla REPA hurts JiT!), and VF loss. The
  priority ordering is still right: finish the baseline first, then improve inference, then change architecture/alignment.

  Want me to start implementing any of these phases?
