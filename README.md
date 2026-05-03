# FLwithHE

Federated Learning (`FedAvg`) research codebase with:
- Homomorphic Encryption (HE): `CKKS` and `Paillier`
- Differential Privacy (DP-SGD): `Gaussian` and `Laplace`
- Optional combined mode: `HE + DP`

This README is a code-level reference for the current implementation.

## 1. Scope and Current Behavior

The primary executable is:
- `python3 -m src.fl.fedavg_runner`

Current runner behavior is **full-model federated training/aggregation** with optional:
- client-side DP-SGD
- encrypted client payload transport and encrypted aggregation

`payload_mode` is currently kept for CSV/plot compatibility and experiment labeling.

---

## 2. Repository Map (HE/DP Relevant)

```text
src/
  fl/
    fedavg_runner.py        # main experiment runner
    client.py               # local training, DP-SGD, optional client-side HE encryption
    aggregator.py           # weighted aggregation; supports mixed encrypted/plain params
    dp_grid_search.py       # CIFAR-10 DP mechanism/epsilon/clip sweep utility
    partitions.py           # iid / dirichlet partitioning

  he/
    encryption.py           # PlainContext, HomomorphicContext(CKKS), PaillierContext

  privacy/
    dp_utils.py             # clipping, noise calibration, per-example DP-SGD,
                            # OpenDP helpers, epsilon estimators

  utils/
    plot_he_comparison.py   # CSV-based plotting and summaries
```

---

## 3. Homomorphic Encryption (HE)

Implementation file:
- [`src/he/encryption.py`](/Users/ezhermemeti/Desktop/DATABASE/FLwithHE/src/he/encryption.py)

### 3.1 `PlainContext`
No-op context with API-compatible methods:
- `encrypt`, `decrypt`, `add`, `mul_scalar`
- accounting helpers: `ciphertext_count`, `encrypted_values`, `payload_nbytes`

### 3.2 `HomomorphicContext` (CKKS)
Uses TenSEAL:
- constructor defaults:
  - `poly_modulus_degree=8192`
  - `coeff_mod_bit_sizes=(60,40,40,60)`
  - `global_scale=2**40`
- creates Galois keys
- slot count: `n_slots = poly_modulus_degree // 2`

Encrypted object:
- `HomomorphicContext.EncryptedTensor`
  - `cts`: list of `ckks_vector` chunks
  - `shape`: original tensor shape

Operations:
- `encrypt(tensor)` flattens/chunks by `n_slots`
- `decrypt(enc)` reconstructs `float32` tensor
- `add(a,b)` with shape/chunk checks
- `mul_scalar(a,s)` supports floating scalar weights

Accounting helpers:
- `ciphertext_count = len(cts)`
- `encrypted_values = math.prod(shape)`
- `payload_nbytes = sum(len(ct.serialize()))`

### 3.3 `PaillierContext`
Uses `phe` Paillier keypair:
- constructor defaults:
  - `key_length=2048`
  - `scale=1e4` fixed-point scaling
- sets `scalar_mode = "int"` (used by aggregator)

Encrypted object:
- `PaillierContext.EncryptedTensor`
  - `cts`: list of Paillier ciphertext objects
  - `shape`: original tensor shape

Quantization path:
- `_encode`: `round(tensor * scale) -> int64`
- `_decode`: `int / scale -> float32`

Operations:
- `encrypt`, `decrypt`
- `add(a,b)`
- `mul_scalar(a,s)` with `k = int(s)` (integer plaintext scalar multiplication)

Accounting helpers:
- `ciphertext_count = len(cts)`
- `encrypted_values = math.prod(shape)`
- `payload_nbytes` approximate via modulus bit length

---

## 4. Differential Privacy (DP)

Core implementation:
- [`src/privacy/dp_utils.py`](/Users/ezhermemeti/Desktop/DATABASE/FLwithHE/src/privacy/dp_utils.py)

### 4.1 Primitive helpers
- flatten/unflatten update tensors:
  - `flatten_state_update`
  - `unflatten_state_update`
- validity/debug:
  - `debug_tensor`
  - `assert_finite_tensor`
- clipping:
  - `clip_update(update, clip_norm, norm_type)`

### 4.2 Noise calibration
- Gaussian:
  - `gaussian_noise_scale(epsilon, delta, clip_norm)`
  - formula: `sigma = clip_norm * sqrt(2 * log(1.25/delta)) / epsilon`
- Laplace:
  - `laplace_noise_scale(epsilon, clip_norm, dimension)`
  - `laplace_l1_noise_scale(epsilon, l1_sensitivity)`

### 4.3 Per-example DP-SGD
Main API:
- `apply_dp_sgd(model, data_loader, optimizer, mechanism, clip_norm, epsilon, delta, ...)`

Current algorithm:
1. For each batch, iterate each sample (`batch_size` micro-steps)
2. Compute per-sample gradient vector
3. Clip gradient (L2 for Gaussian, L1 for Laplace)
4. Average clipped gradients
5. Add calibrated noise to averaged gradient
6. Write noisy gradient back and call `optimizer.step()`

Clipping modes:
- `fixed`
- `quantile`
- `adaptive` (EMA smoothed quantile + clamp)

Returns aggregated diagnostics:
- `raw_norm`, `clipped_norm`, `noise_norm`, `noise_scale`,
  `clip_norm`, `clip_factor`, `steps`, `loss`

### 4.4 Privacy accounting utilities
- `compute_epsilon(...)` (Gaussian, OpenDP-first with RDP fallback)
- `compute_laplace_epsilon(epsilon_per_round, num_rounds)`

### 4.5 OpenDP aggregate helpers (optional)
- `privatize_aggregate_with_opendp(...)`
- used as utility functions; not the default runner path

---

## 5. Client Logic

Implementation file:
- [`src/fl/client.py`](/Users/ezhermemeti/Desktop/DATABASE/FLwithHE/src/fl/client.py)

### 5.1 `ClientUpdate` structure
`ClientUpdate` fields:
- `state_dict`, `num_samples`
- timing: `train_time`, `encrypt_time`
- delta marker: `is_model_delta`
- DP diagnostics:
  - `raw_update_norm`, `clipped_update_norm`, `clipping_factor`
  - `gaussian_std`, `laplace_scale`
  - `noise_scale`, `noise_norm`, `signal_noise_ratio`
  - `laplace_expected_noise_l2`

### 5.2 Training modes
- If `dp_clip_norm is None`: standard SGD training loop
- Else: DP-SGD via `apply_dp_sgd`

### 5.3 Encryption behavior
After local training:
- state is moved to CPU
- if HE enabled:
  - Paillier mode encrypts only final layer keys (mixed-mode transport)
  - CKKS mode encrypts all parameters

Final-layer key detection:
- `classifier.3.*`
- `linear.*`
- `model.fc.*`
- `fc.*`, `fc2.*`, `fc3.*`

---

## 6. Aggregation Logic

Implementation file:
- [`src/fl/aggregator.py`](/Users/ezhermemeti/Desktop/DATABASE/FLwithHE/src/fl/aggregator.py)

`Aggregator.federated_average(updates, global_model)`:

1. Compute weighted aggregation by `num_samples`
2. If HE context exists:
   - **Paillier (`scalar_mode == int`)**:
     - detect per-parameter encrypted/plain type from first update
     - encrypted params: homomorphic weighted sum with integer sample counts, then divide by total samples after decrypt
     - plaintext params: ordinary weighted average
   - **CKKS path**:
     - assumes encrypted tensors for processed keys
     - weighted encrypted sum using float weights, then decrypt
3. If all updates are marked `is_model_delta=True`, add aggregated delta to base global state
4. `global_model.load_state_dict(new_state)`

This supports mixed encrypted/plain aggregation in one round.

---

## 7. Main Runner (`fedavg_runner.py`)

Implementation file:
- [`src/fl/fedavg_runner.py`](/Users/ezhermemeti/Desktop/DATABASE/FLwithHE/src/fl/fedavg_runner.py)

### 7.1 Dataset/model selection
Datasets:
- `mnist`
- `cifar10`
- `ptbxl`

Model selection:
- MNIST: `SimpleCNN`
- CIFAR-10:
  - DP off: `ResNetCIFAR10`
  - DP on: `DPResNetCIFAR10` (GroupNorm variant)
- PTB-XL: `cnn_medium`, `cnn_large`, `logistic`, `lstm`

### 7.2 Partitioning
- `--partition iid`
- `--partition dirichlet --dirichlet_alpha <value>`

### 7.3 HE setup
- `--use_encryption`
- `--encryption_scheme ckks|paillier`

### 7.4 DP setup and guards
- `--use_dp`
- supported mode: `--dp_mode dp_sgd`
- mechanisms: `--dp_mechanism gaussian|laplace`
- validation checks for epsilon and laplace epsilon
- warning when `local_epochs > 3`
- `pretrain_rounds > 0` blocked when DP is enabled

### 7.5 Round loop behavior
Per round:
1. Build each client + local train
2. Aggregate updates
3. Evaluate model
4. Print performance and DP diagnostics
5. Optionally append CSV row (`--save_metrics_csv`)

### 7.6 DP mechanism comparison mode
`--compare_dp_mechanisms` runs two experiments sequentially:
- Gaussian
- Laplace

### 7.7 CSV writer in runner
`_append_csv_row` writes a fixed schema used by plotting.

Scheme label logic in CSV:
- DP only: `dp_gaussian`, `dp_laplace`
- HE only: `ckks`, `paillier`
- HE+DP: `<he_scheme>+dp_<mechanism>`

---

## 8. DP Grid Search Utility

Implementation file:
- [`src/fl/dp_grid_search.py`](/Users/ezhermemeti/Desktop/DATABASE/FLwithHE/src/fl/dp_grid_search.py)

Purpose:
- CIFAR-10 utility to scan combinations of:
  - mechanism
  - epsilon per round
  - clip norm

Output:
- console comparison table of final/best accuracies and estimated privacy quantities

---

## 9. CSV Contract

Runner CSV columns:
- `timestamp`
- `round`
- `dataset`
- `model`
- `num_clients`
- `scheme`
- `payload_mode`
- `training_time`
- `encrypt_time`
- `aggregate_time`
- `decrypt_time`
- `he_total_time`
- `total_round_time`
- `ciphertext_count`
- `encrypted_values`
- `payload_nbytes`
- `accuracy`
- `loss`
- `mean_abs_error`
- `max_abs_error`
- `analytics_reference`
- `analytics_decrypted`
- `integer_reference`
- `integer_decrypted`

Notes:
- Current runner writes placeholders (`0` / `""`) for HE analytics-only fields not used in current full-model path.

---

## 10. Plotting

Implementation file:
- [`src/utils/plot_he_comparison.py`](/Users/ezhermemeti/Desktop/DATABASE/FLwithHE/src/utils/plot_he_comparison.py)

CLI:

```bash
python3 -m src.utils.plot_he_comparison \
  --csv_files <file1.csv> [file2.csv ...] \
  --payload_mode full_model|analytics|integer_stats \
  --output_dir <plot_dir> \
  [--sweep_mode] [--log_xscale]
```

Normalization map includes:
- `paillier -> PHE`
- `ckks -> FHE`
- `dp_gaussian -> DP-GAUSSIAN`
- `dp_laplace -> DP-LAPLACE`
- combined labels (`+`) are normalized part-by-part

Supported outputs:
- `full_model` plots
- `analytics` plots
- `integer_stats` plots
- sweep plots (`--sweep_mode`)

---

## 11. End-to-End Experiment Recipes

### 11.1 Baseline (no DP, no HE)

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1 \
  --save_metrics_csv results/mnist_baseline.csv
```

### 11.2 DP mechanism comparison

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1 \
  --use_dp \
  --dp_epsilon 3 \
  --dp_target_delta 1e-5 \
  --compare_dp_mechanisms \
  --save_metrics_csv results/mnist_dp_compare.csv
```

### 11.3 HE + DP comparison (example)

CKKS + Gaussian:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1 \
  --use_dp --dp_mechanism gaussian --dp_epsilon 3 --dp_target_delta 1e-5 \
  --use_encryption --encryption_scheme ckks \
  --save_metrics_csv results/mnist_he_dp_compare.csv
```

Paillier + Laplace:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1 \
  --use_dp --dp_mechanism laplace --dp_epsilon 3 \
  --use_encryption --encryption_scheme paillier \
  --save_metrics_csv results/mnist_he_dp_compare.csv
```

### 11.4 Plot comparison CSV

```bash
python3 -m src.utils.plot_he_comparison \
  --csv_files results/mnist_dp_compare.csv results/mnist_he_dp_compare.csv \
  --payload_mode full_model \
  --output_dir plots/mnist
```

---

## 12. Full CLI Reference (Runner)

Core training:
- `--num_clients`
- `--rounds`
- `--local_epochs`
- `--batch_size`
- `--lr`
- `--seed`
- `--dataset mnist|cifar10|ptbxl`
- `--partition iid|dirichlet`
- `--dirichlet_alpha`
- `--weight_decay`
- `--scheduler none|step|cosine`
- `--use_aug`
- `--autoaugment`
- `--no_cuda`

PTB-XL:
- `--ptbxl_model cnn_large|cnn_medium|logistic|lstm`
- `--ptbxl_data_dir <path>`

HE:
- `--use_encryption`
- `--encryption_scheme ckks|paillier`

DP:
- `--use_dp`
- `--dp_mode dp_sgd`
- `--dp_mechanism gaussian|laplace`
- `--dp_epsilon`
- `--dp_laplace_epsilon`
- `--dp_target_delta`
- `--dp_clip_norm`
- `--dp_clip_strategy fixed|quantile|adaptive`
- `--dp_clip_quantile`
- `--dp_clip_alpha`
- `--dp_clip_min`
- `--dp_clip_max`
- `--dp_noise_multiplier` (compatibility flag)
- `--dp_debug`

DP utility features:
- `--warmup_rounds`
- `--use_ema`
- `--ema_decay`
- `--pretrain_rounds`
- `--baseline_compare`
- `--compare_dp_mechanisms`

CSV/plot compatibility:
- `--save_metrics_csv <path>`
- `--payload_mode full_model|analytics|integer_stats`

---

## 13. Dependencies and Installation

Install all required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Important optional libs:
- CKKS: `tenseal`
- Paillier: `phe`
- OpenDP helper functions: `opendp`

---

## 14. Troubleshooting

- `ImportError: TenSEAL not installed`
  - `pip install tenseal`
- `ImportError: Paillier library 'phe' not installed`
  - `pip install phe`
- Plot script missing columns
  - ensure CSV generated by current `fedavg_runner` and includes `scheme`, `payload_mode`
- Slow DP-SGD
  - current DP-SGD is per-example and intentionally exact/simple, so it is slower than vectorized/Opacus approaches

---

## 15. Implementation Caveats (Important)

- Current `fedavg_runner` does not implement separate HE-only analytics/integer payload aggregation loops; it is centered on full-model FL updates.
- `payload_mode` is currently a metadata field used for CSV/plot compatibility.
- `ClientUpdate.is_model_delta` exists and aggregator supports delta add-back, but current client path sets `is_model_delta=False`.

