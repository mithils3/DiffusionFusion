# JiT Architecture Comparison

This is the simplified meeting version of the comparison diagram.

- PNG: [jit_architecture_comparison_simple.png](/home/mithil/PyCharmProjects/DiffusionFusion/JiT/jit_architecture_comparison_simple.png)
- DOT source: [jit_architecture_comparison_simple.dot](/home/mithil/PyCharmProjects/DiffusionFusion/JiT/jit_architecture_comparison_simple.dot)

## One-Line Summary

Old JiT mixes latent and DINO in one shared transformer, while the new dual-stream JiT keeps them separate and only exchanges information through explicit cross-fusion blocks.
