# FLwithHE

Federated Learning (`FedAvg`) experiments with Homomorphic Encryption for comparing:
- `CKKS` as `FHE`
- `Paillier` as `PHE`

The codebase keeps model training unchanged and focuses the comparison on the encryption / aggregation experiment layer.

## What This Project Supports

- Standard FedAvg training on `mnist`, `cifar10`, and `ptbxl`
- HE experiment modes:
  - `full_model`
  - `analytics`
  - `integer_stats`
- Parameter sweep experiments with `--param_sweep`
- CSV metric logging for thesis analysis
- Plot generation directly from CSV files

## Project Structure

```text
src/
  fl/
    client.py
    aggregator.py
    fedavg_runner.py
    partitions.py
  he/
    encryption.py
  models/
    mnist_cnn.py
    cifar_resnet18.py
    ptbxl_cnn_medium.py
    ptbxl_cnn_large.py
    ptbxl_logistic.py
    ptbxl_lstm.py
  utils/
    plot_he_comparison.py
config/
requirements.txt
README.md
```

## Setup

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows CMD:

```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Main dependencies:
- `torch`
- `pandas`
- `matplotlib`
- `tenseal`
- `phe`

## Quick Start

Basic FedAvg on MNIST:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1 \
  --partition iid
```

Non-IID example:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --num_clients 5 \
  --rounds 5 \
  --partition dirichlet \
  --dirichlet_alpha 0.3
```

CIFAR-10 example:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset cifar10 \
  --num_clients 5 \
  --rounds 40 \
  --local_epochs 3 \
  --use_aug \
  --weight_decay 0.0005 \
  --scheduler cosine
```

Disable GPU:

```bash
python3 -m src.fl.fedavg_runner --dataset mnist --no_cuda
```

## HE Experiment Modes

The training flow remains FedAvg. The experiment layer changes only what gets encrypted and measured.

| `payload_mode` | Description | Best for |
|---|---|---|
| `full_model` | Encrypt model parameters and aggregate them | Showing CKKS/FHE scaling on large tensors |
| `analytics` | Encrypt small scalars such as `loss_sum`, `correct_count`, `sample_count`, optional `grad_norm` | Showing Paillier/PHE advantage on small payloads |
| `integer_stats` | Encrypt integer statistics such as `class_counts` | Showing Paillier/PHE advantage on exact integer aggregation |

## Supported HE Schemes

| Scheme | CLI value | Role in comparisons |
|---|---|---|
| CKKS | `ckks` | Used as `FHE` style approximate encrypted computation |
| Paillier | `paillier` | Used as `PHE` style additive encrypted computation |

Notes:
- `ckks` supports `full_model`, `analytics`, and `integer_stats`
- `paillier` is most useful for `analytics`, `integer_stats`, and lightweight `full_model` comparisons
- In `full_model` mode, Paillier encrypts only the final classifier layer so runtime stays practical

## Important CLI Flags

Training / experiment flags:

- `--dataset`: `mnist`, `cifar10`, `ptbxl`
- `--num_clients`: number of clients
- `--rounds`: global rounds
- `--local_epochs`: local epochs per client
- `--partition`: `iid` or `dirichlet`
- `--dirichlet_alpha`: heterogeneity strength for non-IID runs
- `--use_encryption`: enable HE experiment pipeline
- `--encryption_scheme`: `ckks` or `paillier`
- `--payload_mode`: `full_model`, `analytics`, `integer_stats`
- `--analytics_include_grad_norm`: include `grad_norm` in analytics payload
- `--param_sweep`: comma-separated encrypted parameter counts for sweep experiments
- `--compare_reference`: compare plaintext vs decrypted aggregate
- `--save_metrics_csv`: path to CSV output
- `--no_cuda`: force CPU

PTB-XL model selection:

- `--ptbxl_model cnn_medium`
- `--ptbxl_model cnn_large`
- `--ptbxl_model logistic`
- `--ptbxl_model lstm`

## Example Experiment Commands

### 1. FHE / CKKS Full Model

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --use_encryption \
  --encryption_scheme ckks \
  --payload_mode full_model \
  --compare_reference \
  --save_metrics_csv results/results_fhe_full_model.csv
```

### 2. PHE / Paillier Full Model

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --use_encryption \
  --encryption_scheme paillier \
  --payload_mode full_model \
  --compare_reference \
  --save_metrics_csv results/results_phe_full_model.csv
```

### 3. FHE / CKKS Analytics

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --use_encryption \
  --encryption_scheme ckks \
  --payload_mode analytics \
  --analytics_include_grad_norm \
  --compare_reference \
  --save_metrics_csv results/results_fhe_analytics.csv
```

### 4. PHE / Paillier Analytics

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --use_encryption \
  --encryption_scheme paillier \
  --payload_mode analytics \
  --analytics_include_grad_norm \
  --compare_reference \
  --save_metrics_csv results/results_phe_analytics.csv
```

### 5. FHE / CKKS Integer Stats

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --use_encryption \
  --encryption_scheme ckks \
  --payload_mode integer_stats \
  --compare_reference \
  --save_metrics_csv results/results_fhe_integer_stats.csv
```

### 6. PHE / Paillier Integer Stats

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --use_encryption \
  --encryption_scheme paillier \
  --payload_mode integer_stats \
  --compare_reference \
  --save_metrics_csv results/results_phe_integer_stats.csv
```

### 7. Parameter Sweep

FHE / CKKS sweep:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --use_encryption \
  --encryption_scheme ckks \
  --payload_mode full_model \
  --param_sweep 2,5,10,50,100,500,1000,5000 \
  --compare_reference \
  --save_metrics_csv results/results_sweep_ckks.csv
```

PHE / Paillier sweep:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset mnist \
  --use_encryption \
  --encryption_scheme paillier \
  --payload_mode full_model \
  --param_sweep 2,5,10,50,100,500,1000,5000 \
  --compare_reference \
  --save_metrics_csv results/results_sweep_paillier.csv
```

## Runtime Output

Typical round output:

```text
Round 01: Acc=95.12% Loss=0.1543 | Train=8.21s Encrypt=3.45s Agg=0.92s Decrypt=0.14s | Total=12.58s Elapsed=12.58s
```

Sweep mode prints a compact summary per round:

```text
Round 01: Acc=95.39% Loss=0.1615 | Train=22.43s Sweep=8 configs | Total=23.66s Elapsed=23.79s
```

Paillier runs also print progress/debug messages so you can see where time is spent.

## CSV Logging

Experiment CSV files include metrics such as:

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

Path behavior:
- `--save_metrics_csv results.csv` writes to the current project directory
- `--save_metrics_csv results/results.csv` writes inside the `results/` folder

## Graph Generation From CSV Files

The plotting utility reads existing CSV files only. It does not rerun training.

Script:

- [plot_he_comparison.py](/Users/ezhermemeti/Desktop/DATABASE/FLwithHE/src/utils/plot_he_comparison.py)

### Full Model Comparison Plots

```bash
python3 -m src.utils.plot_he_comparison \
  --csv_files results/results_phe_full_model.csv results/results_fhe_full_model.csv \
  --payload_mode full_model \
  --output_dir plots/full_model
```

Generated files:
- `he_total_time_comparison.png`
- `payload_size_comparison.png`
- `he_time_breakdown.png`
- `ciphertext_count_comparison.png`
- `accuracy_vs_round.png`
- `loss_vs_round.png`
- `summary_full_model.csv`

### Analytics Comparison Plots

```bash
python3 -m src.utils.plot_he_comparison \
  --csv_files results/results_phe_analytics.csv results/results_fhe_analytics.csv \
  --payload_mode analytics \
  --output_dir plots/analytics
```

Generated files:
- `analytics_he_total_time.png`
- `analytics_payload_size.png`
- `analytics_ciphertext_count.png`
- `analytics_accuracy_vs_round.png`
- `analytics_loss_vs_round.png`
- `analytics_mean_abs_error.png`
- `analytics_reference_vs_decrypted.png` if reference/decrypted columns are present
- `summary_analytics.csv`

### Sweep Plots

```bash
python3 -m src.utils.plot_he_comparison \
  --csv_files results/results_sweep_paillier.csv results/results_sweep_ckks.csv \
  --payload_mode full_model \
  --sweep_mode \
  --log_xscale \
  --output_dir plots/sweep
```

Generated files:
- `sweep_encrypt_time.png`
- `sweep_he_total_time.png`
- `sweep_payload_size.png`
- `sweep_ciphertext_count.png`
- `sweep_decrypt_time.png`
- `sweep_aggregate_time.png`
- `summary_sweep.csv`

Notes:
- The plotting tool accepts multiple CSV files and concatenates them
- It filters rows by `payload_mode`
- It normalizes `paillier -> PHE` and `ckks -> FHE`
- It fails with a clear error if required columns are missing

## PTB-XL

### Dataset Setup

Download PTB-XL from:

```text
https://physionet.org/content/ptb-xl/1.0.3/
```

Extract into:

```text
data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
```

Expected layout:

```text
data/
  ptbxl/
    ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
      records100/
      records500/
      ptbxl_database.csv
      scp_statements.csv
```

Install dataset dependency:

```bash
pip install wfdb
```

Quick dataset test:

```bash
python3 test_ptbxl.py
```

### PTB-XL Training Examples

CNN Medium:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset ptbxl \
  --ptbxl_model cnn_medium \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1
```

CNN Large:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset ptbxl \
  --ptbxl_model cnn_large \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1
```

Logistic baseline:

```bash
python3 -m src.fl.fedavg_runner \
  --dataset ptbxl \
  --ptbxl_model logistic \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1
```

### PTB-XL Labels

| Class | Label | Description |
|---|---|---|
| NORM | 0 | Normal ECG |
| MI | 1 | Myocardial Infarction |
| STTC | 2 | ST/T Change |
| CD | 3 | Conduction Disturbance |
| HYP | 4 | Hypertrophy |

## Common Issues

- `ImportError: TenSEAL not installed`
  Install with `pip install tenseal`

- `ImportError: phe not installed`
  Install with `pip install phe`

- `FileNotFoundError` for CSV plotting
  Check the exact CSV path. Example: `results/results_sweep_ckks.csv`

- CSV saved in the wrong place
  Use `results/...` in `--save_metrics_csv` if you want the file inside the `results/` folder

- GPU problems
  Run with `--no_cuda`

## Output Locations

- experiment CSVs: usually under `results/`
- generated plots: under the directory passed to `--output_dir`
- generated summaries: saved next to the plots
