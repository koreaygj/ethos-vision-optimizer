# Models Summary

This file summarizes the model weight files in this directory. Each file corresponds to the `best.pt` from a different experiment, renamed to reflect the modifications applied.

- **`original.pt`**: The original baseline model.

- **`csp.pt`**: 
  - Activation: `SiLU()` -> `ReLU()`
  - Block: `PSA` -> `CSP`

- **`se.pt`**:
  - Activation: `SiLU()` -> `ReLU()`
  - Block: `PSA` -> `SEBlock`

- **`csp_dfl.pt`**:
  - Activation: `SiLU()` -> `ReLU()`
  - Block: `PSA` -> `CSP`
  - DFL (Distribution Focal Loss) parameters changed.

- **`se_dfl.pt`**:
  - Activation: `SiLU()` -> `ReLU()`
  - Block: `PSA` -> `SEBlock`
  - DFL (Distribution Focal Loss) parameters changed.

- **`lk_dfl.pt`**:
  - Activation: `SiLU()` -> `ReLU()`
  - Block: `PSA` -> `LargeKernelBlock`
  - DFL (Distribution Focal Loss) was removed.

- **`lr_csp_dfl.pt`**:
  - Activation: `SiLU()` -> `LeakyReLU()`
  - Block: `PSA` -> `CSP`
  - DFL (Distribution Focal Loss) parameters changed.

- **`lr_se_dfl.pt`**:
  - Activation: `SiLU()` -> `LeakyReLU()`
  - Block: `PSA` -> `SEBlock`
  - DFL (Distribution Focal Loss) parameters changed.
