import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


SCHEME_LABELS = {
    "paillier": "PHE",
    "ckks": "FHE",
    "phe": "PHE",
    "fhe": "FHE",
}

BASE_REQUIRED_COLUMNS = {
    "round",
    "scheme",
    "payload_mode",
    "training_time",
    "encrypt_time",
    "aggregate_time",
    "decrypt_time",
    "he_total_time",
    "total_round_time",
    "ciphertext_count",
    "encrypted_values",
    "payload_nbytes",
    "accuracy",
    "loss",
    "mean_abs_error",
    "max_abs_error",
}

FULL_MODEL_SUMMARY_COLUMNS = [
    "training_time",
    "encrypt_time",
    "aggregate_time",
    "decrypt_time",
    "he_total_time",
    "payload_nbytes",
    "ciphertext_count",
    "accuracy",
    "loss",
    "mean_abs_error",
    "max_abs_error",
]

SWEEP_SUMMARY_COLUMNS = [
    "training_time",
    "encrypt_time",
    "aggregate_time",
    "decrypt_time",
    "he_total_time",
    "payload_nbytes",
    "ciphertext_count",
    "accuracy",
    "loss",
    "mean_abs_error",
    "max_abs_error",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PHE vs FHE comparisons from experiment CSV files.")
    parser.add_argument("--csv_files", nargs="+", required=True, help="One or more CSV result files.")
    parser.add_argument("--payload_mode", choices=["full_model", "analytics", "integer_stats"], required=True)
    parser.add_argument("--output_dir", required=True, help="Directory where plots and summary CSVs will be saved.")
    parser.add_argument("--sweep_mode", action="store_true", help="Generate parameter sweep plots using encrypted_values as x-axis.")
    parser.add_argument("--log_xscale", action="store_true", help="Use logarithmic x-axis for sweep plots.")
    return parser.parse_args()


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def require_columns(df: pd.DataFrame, required: Iterable[str], context: str):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def normalize_scheme(value: object) -> str:
    key = str(value).strip().lower()
    return SCHEME_LABELS.get(key, str(value).upper())


def load_and_prepare(csv_files: Sequence[str], payload_mode: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file '{path}': {exc}") from exc
        frame["source_file"] = os.path.basename(path)
        frames.append(frame)

    if not frames:
        raise ValueError("No CSV files were loaded.")

    df = pd.concat(frames, ignore_index=True)
    require_columns(df, BASE_REQUIRED_COLUMNS, "Input CSV data")
    df = df[df["payload_mode"] == payload_mode].copy()
    if df.empty:
        raise ValueError(f"No rows found for payload_mode='{payload_mode}'.")

    df["scheme_label"] = df["scheme"].map(normalize_scheme)
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    numeric_cols = list(BASE_REQUIRED_COLUMNS - {"scheme", "payload_mode"})
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })


def save_plot(fig, output_path: str):
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def mean_by_scheme(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df.groupby("scheme_label", dropna=False)[list(columns)].mean(numeric_only=True).reset_index()


def mean_by_round(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = (
        df.groupby(["scheme_label", "round"], dropna=False)[column]
        .mean()
        .reset_index()
        .sort_values(["scheme_label", "round"])
    )
    return out.dropna(subset=["round", column])


def mean_by_sweep(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = (
        df.groupby(["scheme_label", "encrypted_values"], dropna=False)[column]
        .mean()
        .reset_index()
        .sort_values(["scheme_label", "encrypted_values"])
    )
    return out.dropna(subset=["encrypted_values", column])


def bytes_scale(series: pd.Series) -> Tuple[pd.Series, str]:
    max_value = series.dropna().max() if not series.dropna().empty else 0
    if max_value >= 1024 * 1024:
        return series / (1024 * 1024), "MB"
    if max_value >= 1024:
        return series / 1024, "KB"
    return series, "Bytes"


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, output_path: str):
    plot_df = df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(plot_df[x_col], plot_df[y_col], color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(plot_df)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_plot(fig, output_path)


def plot_stacked_bar(df: pd.DataFrame, x_col: str, stack_cols: Sequence[str], labels: Sequence[str], title: str, xlabel: str, ylabel: str, output_path: str):
    plot_df = df.dropna(subset=[x_col]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bottom = pd.Series([0.0] * len(plot_df))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for idx, (col, label) in enumerate(zip(stack_cols, labels)):
        values = plot_df[col].fillna(0.0)
        ax.bar(plot_df[x_col], values, bottom=bottom, label=label, color=colors[idx % len(colors)])
        bottom = bottom + values
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_plot(fig, output_path)


def plot_line(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, output_path: str, log_xscale: bool = False):
    plot_df = df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for scheme, group in plot_df.groupby("scheme_label", dropna=False):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[y_col], marker="o", linewidth=2, label=scheme)
    if log_xscale:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    save_plot(fig, output_path)


def parse_json_payload(value: object) -> Optional[Dict[str, float]]:
    if pd.isna(value) or value == "":
        return None
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    out: Dict[str, float] = {}
    for key, item in payload.items():
        if isinstance(item, list):
            if not item:
                continue
            out[key] = float(sum(item) / len(item))
        else:
            out[key] = float(item)
    return out


def plot_reference_vs_decrypted(df: pd.DataFrame, ref_col: str, dec_col: str, title: str, output_path: str):
    rows = []
    for _, row in df.iterrows():
        ref_payload = parse_json_payload(row.get(ref_col))
        dec_payload = parse_json_payload(row.get(dec_col))
        if not ref_payload or not dec_payload:
            continue
        for key in sorted(set(ref_payload) & set(dec_payload)):
            rows.append({
                "scheme_label": row["scheme_label"],
                "metric_key": key,
                "reference": ref_payload[key],
                "decrypted": dec_payload[key],
            })
    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.groupby(["scheme_label", "metric_key"], dropna=False).mean(numeric_only=True).reset_index()
    labels = [f"{scheme}:{metric}" for scheme, metric in zip(plot_df["scheme_label"], plot_df["metric_key"])]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.8))
    width = 0.38
    ax.bar([i - width / 2 for i in x], plot_df["reference"], width=width, label="Reference", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], plot_df["decrypted"], width=width, label="Decrypted", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Scheme / Metric")
    ax.set_ylabel("Value")
    ax.legend()
    save_plot(fig, output_path)


def save_summary_csv(df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str], output_path: str):
    summary = df.groupby(list(group_cols), dropna=False)[list(value_cols)].mean(numeric_only=True).reset_index()
    summary.to_csv(output_path, index=False)


def generate_full_model_outputs(df: pd.DataFrame, output_dir: str):
    summary = mean_by_scheme(df, FULL_MODEL_SUMMARY_COLUMNS)
    payload_scaled, payload_unit = bytes_scale(summary["payload_nbytes"])
    summary = summary.copy()
    summary["payload_scaled"] = payload_scaled

    plot_bar(
        summary, "scheme_label", "he_total_time",
        "Average HE Total Time Comparison", "Scheme", "Average HE Total Time (s)",
        os.path.join(output_dir, "he_total_time_comparison.png"),
    )
    plot_bar(
        summary, "scheme_label", "payload_scaled",
        "Average Payload Size Comparison", "Scheme", f"Average Payload Size ({payload_unit})",
        os.path.join(output_dir, "payload_size_comparison.png"),
    )
    plot_stacked_bar(
        summary,
        "scheme_label",
        ["encrypt_time", "aggregate_time", "decrypt_time"],
        ["Encrypt", "Aggregate", "Decrypt"],
        "HE Time Breakdown",
        "Scheme",
        "Average Time (s)",
        os.path.join(output_dir, "he_time_breakdown.png"),
    )
    plot_bar(
        summary, "scheme_label", "ciphertext_count",
        "Ciphertext Count Comparison", "Scheme", "Average Ciphertext Count",
        os.path.join(output_dir, "ciphertext_count_comparison.png"),
    )
    plot_line(
        mean_by_round(df, "accuracy"), "round", "accuracy",
        "Accuracy vs Round", "Round", "Accuracy",
        os.path.join(output_dir, "accuracy_vs_round.png"),
    )
    plot_line(
        mean_by_round(df, "loss"), "round", "loss",
        "Loss vs Round", "Round", "Loss",
        os.path.join(output_dir, "loss_vs_round.png"),
    )
    save_summary_csv(df, ["scheme_label"], FULL_MODEL_SUMMARY_COLUMNS, os.path.join(output_dir, "summary_full_model.csv"))


def generate_analytics_outputs(df: pd.DataFrame, output_dir: str):
    summary = mean_by_scheme(df, FULL_MODEL_SUMMARY_COLUMNS)
    payload_scaled, payload_unit = bytes_scale(summary["payload_nbytes"])
    summary = summary.copy()
    summary["payload_scaled"] = payload_scaled

    plot_bar(
        summary, "scheme_label", "he_total_time",
        "Analytics HE Total Time Comparison", "Scheme", "Average HE Total Time (s)",
        os.path.join(output_dir, "analytics_he_total_time.png"),
    )
    plot_bar(
        summary, "scheme_label", "payload_scaled",
        "Analytics Payload Size Comparison", "Scheme", f"Average Payload Size ({payload_unit})",
        os.path.join(output_dir, "analytics_payload_size.png"),
    )
    plot_bar(
        summary, "scheme_label", "ciphertext_count",
        "Analytics Ciphertext Count Comparison", "Scheme", "Average Ciphertext Count",
        os.path.join(output_dir, "analytics_ciphertext_count.png"),
    )
    plot_line(
        mean_by_round(df, "accuracy"), "round", "accuracy",
        "Analytics Accuracy vs Round", "Round", "Accuracy",
        os.path.join(output_dir, "analytics_accuracy_vs_round.png"),
    )
    plot_line(
        mean_by_round(df, "loss"), "round", "loss",
        "Analytics Loss vs Round", "Round", "Loss",
        os.path.join(output_dir, "analytics_loss_vs_round.png"),
    )
    plot_bar(
        summary, "scheme_label", "mean_abs_error",
        "Analytics Aggregation Error Comparison", "Scheme", "Average Mean Absolute Error",
        os.path.join(output_dir, "analytics_mean_abs_error.png"),
    )
    if {"analytics_reference", "analytics_decrypted"}.issubset(df.columns):
        plot_reference_vs_decrypted(
            df,
            "analytics_reference",
            "analytics_decrypted",
            "Analytics Reference vs Decrypted",
            os.path.join(output_dir, "analytics_reference_vs_decrypted.png"),
        )
    save_summary_csv(df, ["scheme_label"], FULL_MODEL_SUMMARY_COLUMNS, os.path.join(output_dir, "summary_analytics.csv"))


def generate_sweep_outputs(df: pd.DataFrame, output_dir: str, log_xscale: bool):
    require_columns(df, {"encrypted_values"}, "Sweep plotting")
    plot_line(
        mean_by_sweep(df, "encrypt_time"), "encrypted_values", "encrypt_time",
        "Encryption Time vs Number of Parameters", "Encrypted Values", "Encryption Time (s)",
        os.path.join(output_dir, "sweep_encrypt_time.png"),
        log_xscale=log_xscale,
    )
    plot_line(
        mean_by_sweep(df, "he_total_time"), "encrypted_values", "he_total_time",
        "HE Total Time vs Number of Parameters", "Encrypted Values", "HE Total Time (s)",
        os.path.join(output_dir, "sweep_he_total_time.png"),
        log_xscale=log_xscale,
    )

    payload_df = mean_by_sweep(df, "payload_nbytes")
    payload_scaled, payload_unit = bytes_scale(payload_df["payload_nbytes"])
    payload_df = payload_df.copy()
    payload_df["payload_scaled"] = payload_scaled
    plot_line(
        payload_df, "encrypted_values", "payload_scaled",
        "Payload Size vs Number of Parameters", "Encrypted Values", f"Payload Size ({payload_unit})",
        os.path.join(output_dir, "sweep_payload_size.png"),
        log_xscale=log_xscale,
    )
    plot_line(
        mean_by_sweep(df, "ciphertext_count"), "encrypted_values", "ciphertext_count",
        "Ciphertext Count vs Number of Parameters", "Encrypted Values", "Ciphertext Count",
        os.path.join(output_dir, "sweep_ciphertext_count.png"),
        log_xscale=log_xscale,
    )
    plot_line(
        mean_by_sweep(df, "decrypt_time"), "encrypted_values", "decrypt_time",
        "Decryption Time vs Number of Parameters", "Encrypted Values", "Decryption Time (s)",
        os.path.join(output_dir, "sweep_decrypt_time.png"),
        log_xscale=log_xscale,
    )
    plot_line(
        mean_by_sweep(df, "aggregate_time"), "encrypted_values", "aggregate_time",
        "Aggregation Time vs Number of Parameters", "Encrypted Values", "Aggregation Time (s)",
        os.path.join(output_dir, "sweep_aggregate_time.png"),
        log_xscale=log_xscale,
    )
    save_summary_csv(
        df,
        ["scheme_label", "encrypted_values"],
        SWEEP_SUMMARY_COLUMNS,
        os.path.join(output_dir, "summary_sweep.csv"),
    )


def main():
    args = parse_args()
    set_plot_style()
    ensure_output_dir(args.output_dir)
    df = load_and_prepare(args.csv_files, args.payload_mode)

    if args.sweep_mode:
        generate_sweep_outputs(df, args.output_dir, log_xscale=args.log_xscale)
    elif args.payload_mode == "full_model":
        generate_full_model_outputs(df, args.output_dir)
    elif args.payload_mode == "analytics":
        generate_analytics_outputs(df, args.output_dir)
    else:
        raise ValueError("Plot generation is currently implemented for full_model, analytics, and sweep outputs.")

    print(f"Saved plots and summaries to: {args.output_dir}")


if __name__ == "__main__":
    main()
