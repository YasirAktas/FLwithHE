import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.mnist_cnn import SimpleCNN
from src.models.cifar_resnet18 import ResNetCIFAR10
from src.models.ptbxl_cnn_large import PTBXL_CNN_Large
from src.models.ptbxl_cnn_medium import PTBXL_CNN_Medium
from src.models.ptbxl_logistic import PTBXL_Logistic
from src.models.ptbxl_lstm import PTBXL_LSTM
from src.fl.partitions import iid_partitions, dirichlet_partitions
from src.fl.client import Client
from src.fl.aggregator import Aggregator
from src.he.encryption import HomomorphicContext, PaillierContext


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total, total_loss / total


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _plain_fedavg_state(updates: List, total_samples: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in updates[0].state_dict.keys():
        out[key] = sum(u.state_dict[key] * (u.num_samples / total_samples) for u in updates)
    return out


def _parse_param_sweep(param_sweep: Optional[str]) -> List[int]:
    if not param_sweep:
        return []
    vals: List[int] = []
    for part in param_sweep.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError("--param_sweep values must be positive integers")
        vals.append(n)
    return vals


def _state_prefix_tensor(state_dict: Dict[str, torch.Tensor], n: int) -> torch.Tensor:
    if n <= 0:
        return torch.zeros(0, dtype=torch.float32)
    chunks: List[torch.Tensor] = []
    remain = n
    for tensor in state_dict.values():
        flat = tensor.detach().cpu().view(-1).to(torch.float32)
        if flat.numel() == 0:
            continue
        take = min(remain, flat.numel())
        chunks.append(flat[:take])
        remain -= take
        if remain == 0:
            break
    if not chunks:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def _is_last_layer_param(name: str) -> bool:
    if name.startswith("classifier.3."):
        return True
    if name.startswith("linear."):
        return True
    if name.startswith("model.fc."):
        return True
    if name.startswith("fc."):
        return True
    if name.startswith("fc2."):
        return True
    if name.startswith("fc3."):
        return True
    return False


def _sum_payload_dicts(payloads: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in payloads[0].keys():
        out[key] = sum(p[key] for p in payloads)
    return out


def _encrypt_payloads(
    payloads: List[Dict[str, torch.Tensor]],
    encryption_ctx,
    debug_prefix: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], float, int, int, int]:
    encrypted_payloads: List[Dict[str, object]] = []
    total_encrypt_time = 0.0
    ciphertext_count = 0
    encrypted_values = 0
    payload_nbytes = 0
    for payload_idx, payload in enumerate(payloads, start=1):
        enc_payload: Dict[str, object] = {}
        if debug_prefix and isinstance(encryption_ctx, PaillierContext):
            payload_values = sum(int(v.numel()) for v in payload.values())
            print(
                f"{debug_prefix} encrypting client payload {payload_idx}/{len(payloads)} "
                f"({len(payload)} tensors, {payload_values} values)"
            )
        start = time.time()
        for key, value in payload.items():
            if debug_prefix and isinstance(encryption_ctx, PaillierContext):
                print(f"{debug_prefix}   tensor={key} values={value.numel()}")
            enc_val = encryption_ctx.encrypt(value)
            enc_payload[key] = enc_val
            ciphertext_count += encryption_ctx.ciphertext_count(enc_val)
            encrypted_values += encryption_ctx.encrypted_values(enc_val)
            payload_nbytes += encryption_ctx.payload_nbytes(enc_val)
        total_encrypt_time += time.time() - start
        if debug_prefix and isinstance(encryption_ctx, PaillierContext):
            print(
                f"{debug_prefix} finished client payload {payload_idx}/{len(payloads)} "
                f"in {time.time() - start:.4f}s"
            )
        encrypted_payloads.append(enc_payload)
    return encrypted_payloads, total_encrypt_time, ciphertext_count, encrypted_values, payload_nbytes


def _encrypt_selected_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    encryption_ctx,
    key_filter,
    debug_prefix: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], float, int, int, int]:
    selected_payloads = []
    for state_dict in state_dicts:
        selected_payloads.append({k: v for k, v in state_dict.items() if key_filter(k)})
    return _encrypt_payloads(selected_payloads, encryption_ctx, debug_prefix=debug_prefix)


def _summarize_keys(state_dict: Dict[str, torch.Tensor], key_filter) -> Tuple[List[str], int]:
    keys = [k for k in state_dict.keys() if key_filter(k)]
    count = sum(int(state_dict[k].numel()) for k in keys)
    return keys, count


def _payload_error(reference: Dict[str, torch.Tensor], decrypted: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    diffs: List[torch.Tensor] = []
    for key in reference.keys():
        diffs.append((reference[key].to(torch.float32) - decrypted[key].to(torch.float32)).abs().view(-1))
    if not diffs:
        return 0.0, 0.0
    all_diffs = torch.cat(diffs)
    return float(all_diffs.mean().item()), float(all_diffs.max().item())


def _payload_to_json(payload: Optional[Dict[str, torch.Tensor]]) -> str:
    if payload is None:
        return ""
    out = {}
    for k, v in payload.items():
        if v.numel() == 1:
            out[k] = float(v.view(-1)[0].item())
        else:
            out[k] = v.detach().cpu().view(-1).tolist()
    return json.dumps(out, ensure_ascii=True)


def _infer_num_classes(dataset: str) -> int:
    if dataset in ("mnist", "cifar10"):
        return 10
    if dataset == "ptbxl":
        return 5
    raise ValueError(f"Unsupported dataset: {dataset}")


def _compute_client_analytics(
    global_model: torch.nn.Module,
    local_state_dict: Dict[str, torch.Tensor],
    dataloader: DataLoader,
    device: torch.device,
    include_grad_norm: bool,
    global_before_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    local_model = type(global_model)().to(device)
    local_model.load_state_dict(local_state_dict)
    local_model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct_count = 0.0
    sample_count = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = local_model(x)
            loss_sum += float(loss_fn(out, y).item())
            pred = out.argmax(1)
            correct_count += float((pred == y).sum().item())
            sample_count += float(y.size(0))

    payload = {
        "loss_sum": torch.tensor([loss_sum], dtype=torch.float32),
        "correct_count": torch.tensor([correct_count], dtype=torch.float32),
        "sample_count": torch.tensor([sample_count], dtype=torch.float32),
    }
    if include_grad_norm:
        sq_norm = 0.0
        for name, local_t in local_state_dict.items():
            diff = local_t.to(torch.float32) - global_before_state[name].to(torch.float32)
            sq_norm += float((diff * diff).sum().item())
        payload["grad_norm"] = torch.tensor([sq_norm ** 0.5], dtype=torch.float32)
    return payload


def _compute_client_integer_stats(dataloader: DataLoader, num_classes: int) -> Dict[str, torch.Tensor]:
    counts = torch.zeros(num_classes, dtype=torch.int64)
    for _, y in dataloader:
        y_cpu = y.detach().cpu()
        if y_cpu.ndim > 1:
            y_cpu = y_cpu.argmax(dim=1)
        for cls in range(num_classes):
            counts[cls] += int((y_cpu == cls).sum().item())
    return {"class_counts": counts}


def _append_csv_row(csv_path: str, row: Dict[str, object]):
    fieldnames = [
        "timestamp",
        "round",
        "dataset",
        "model",
        "num_clients",
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
        "analytics_reference",
        "analytics_decrypted",
        "integer_reference",
        "integer_decrypted",
    ]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_loaders(batch_size: int, dataset: str, use_aug: bool = False, ptbxl_data_dir: str = None):
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    elif dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if use_aug:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    elif dataset == "ptbxl":
        from src.data.ptbxl_dataset import PTBXLDataset
        data_dir = ptbxl_data_dir or "./data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
        train_ds = PTBXLDataset(data_dir=data_dir, split="train")
        test_ds  = PTBXLDataset(data_dir=data_dir, split="test")
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_ds, test_loader


def run(config):
    set_seed(config.seed)
    device = torch.device("cuda" if (not config.no_cuda and torch.cuda.is_available()) else "cpu")
    train_ds, test_loader = build_loaders(config.batch_size, config.dataset, use_aug=config.use_aug, ptbxl_data_dir=getattr(config, "ptbxl_data_dir", None))
    start_time = time.time()
    payload_mode = getattr(config, "payload_mode", "full_model")
    compare_reference = bool(getattr(config, "compare_reference", False))
    param_sweep = _parse_param_sweep(getattr(config, "param_sweep", None))
    if param_sweep and payload_mode != "full_model":
        raise ValueError("--param_sweep is only supported with --payload_mode full_model")

    if config.partition == "iid":
        partitions = iid_partitions(train_ds, config.num_clients)
    else:
        partitions = dirichlet_partitions(train_ds, config.num_clients, alpha=config.dirichlet_alpha)

    # Select model and (optional) encryption scheme
    if config.dataset == "mnist":
        global_model = SimpleCNN().to(device)
    elif config.dataset == "cifar10":
        global_model = ResNetCIFAR10().to(device)
    elif config.dataset == "ptbxl":
        ptbxl_model = getattr(config, "ptbxl_model", "cnn_medium")
        if ptbxl_model == "cnn_large":
            global_model = PTBXL_CNN_Large().to(device)
        elif ptbxl_model == "cnn_medium":
            global_model = PTBXL_CNN_Medium().to(device)
        elif ptbxl_model == "logistic":
            global_model = PTBXL_Logistic().to(device)
        elif ptbxl_model == "lstm":
            global_model = PTBXL_LSTM().to(device)
        else:
            raise ValueError(f"Unknown ptbxl_model: {ptbxl_model}")
        print(f"[PTB-XL] Model: {ptbxl_model}")
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    if config.use_encryption:
        scheme = getattr(config, "encryption_scheme", "ckks")
        if scheme == "paillier":
            paillier_scale = 1.0 if payload_mode == "integer_stats" else 1e4
            encryption_ctx = PaillierContext(scale=paillier_scale)
        elif scheme == "ckks":
            encryption_ctx = HomomorphicContext()
        else:
            raise ValueError(f"Unknown encryption_scheme: {scheme}")
    else:
        encryption_ctx = None
    he_aggregator = Aggregator(encryption_context=encryption_ctx) if encryption_ctx is not None else None
    if encryption_ctx is not None:
        scheme = getattr(config, "encryption_scheme", "ckks")
        print(f"[HE] Encryption: ACTIVE (dataset={config.dataset}, scheme={scheme}, payload_mode={payload_mode})")
        if scheme == "paillier":
            print(f"[PHE] Paillier enabled with scale={encryption_ctx.scale}")
    else:
        print(f"[HE] Encryption: DISABLED (payload_mode={payload_mode})")

    for rnd in range(1, config.rounds + 1):
        round_start = time.time()
        client_updates: List = []
        client_loaders: List[DataLoader] = []
        global_before_state = _clone_state_dict(global_model.state_dict())
        for cid, idxs in enumerate(partitions):
            subset = torch.utils.data.Subset(train_ds, idxs)
            loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
            client = Client(cid, loader, device, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay, scheduler=config.scheduler, encryption_context=None)
            update = client.train(global_model, epochs=config.local_epochs)
            client_updates.append(update)
            client_loaders.append(loader)
            if getattr(config, "encryption_scheme", "ckks") == "paillier" and config.use_encryption:
                print(
                    f"[PHE][Round {rnd:02d}] client {cid} training complete "
                    f"(samples={update.num_samples}, train_time={update.train_time:.4f}s)"
                )
        total_train_time  = sum(u.train_time   for u in client_updates)
        total_samples = sum(u.num_samples for u in client_updates)
        plain_state = _plain_fedavg_state(client_updates, total_samples)
        plain_agg_time = time.time() - round_start - total_train_time

        # Default metrics
        encrypt_time = 0.0
        aggregate_time = plain_agg_time
        decrypt_time = 0.0
        he_total_time = 0.0
        ciphertext_count = 0
        encrypted_values = 0
        payload_nbytes = 0
        mean_abs_error = 0.0
        max_abs_error = 0.0
        analytics_reference = ""
        analytics_decrypted = ""
        integer_reference = ""
        integer_decrypted = ""

        # Update global model and optional encrypted experiment path
        if payload_mode == "full_model":
            if encryption_ctx is not None and not param_sweep:
                plain_payloads = [u.state_dict for u in client_updates]
                if getattr(config, "encryption_scheme", "ckks") == "paillier":
                    selected_keys, selected_values = _summarize_keys(plain_payloads[0], _is_last_layer_param)
                    print(
                        f"[PHE][Round {rnd:02d}] full_model mode: encrypting last layer only "
                        f"({len(selected_keys)} tensors, {selected_values} values)"
                    )
                    print(f"[PHE][Round {rnd:02d}] encrypted tensors: {selected_keys}")
                    enc_payloads, encrypt_time, ciphertext_count, encrypted_values, payload_nbytes = _encrypt_selected_state_dicts(
                        plain_payloads,
                        encryption_ctx,
                        _is_last_layer_param,
                        debug_prefix=f"[PHE][Round {rnd:02d}]",
                    )
                    scalars = [u.num_samples for u in client_updates]
                    print(f"[PHE][Round {rnd:02d}] starting encrypted aggregation across {len(client_updates)} clients")
                    dec_last_layer, aggregate_time, decrypt_time = he_aggregator.aggregate_encrypted_dict(
                        enc_payloads,
                        scalars,
                        divide_by=float(total_samples),
                    )
                    print(f"[PHE][Round {rnd:02d}] encrypted aggregation and decryption complete")
                    dec_state = {k: v.clone() for k, v in plain_state.items()}
                    dec_state.update(dec_last_layer)
                    print(
                        f"[PHE][Round {rnd:02d}] encrypt_time={encrypt_time:.4f}s "
                        f"agg_time={aggregate_time:.4f}s decrypt_time={decrypt_time:.4f}s "
                        f"ciphertexts={ciphertext_count} payload_nbytes={payload_nbytes}"
                    )
                else:
                    enc_payloads, encrypt_time, ciphertext_count, encrypted_values, payload_nbytes = _encrypt_payloads(plain_payloads, encryption_ctx)
                    scalars = [u.num_samples / total_samples for u in client_updates]
                    dec_state, aggregate_time, decrypt_time = he_aggregator.aggregate_encrypted_dict(enc_payloads, scalars, divide_by=None)
                global_model.load_state_dict(dec_state)
                if compare_reference or encryption_ctx is not None:
                    mean_abs_error, max_abs_error = _payload_error(plain_state, dec_state)
            else:
                global_model.load_state_dict(plain_state)
        elif payload_mode in ("analytics", "integer_stats"):
            # Keep FedAvg model update unchanged (plaintext), run HE experiment on side payloads.
            global_model.load_state_dict(plain_state)
            if payload_mode == "analytics":
                plain_payloads = [
                    _compute_client_analytics(
                        global_model=global_model,
                        local_state_dict=u.state_dict,
                        dataloader=loader,
                        device=device,
                        include_grad_norm=bool(getattr(config, "analytics_include_grad_norm", False)),
                        global_before_state=global_before_state,
                    )
                    for u, loader in zip(client_updates, client_loaders)
                ]
            else:
                num_classes = _infer_num_classes(config.dataset)
                plain_payloads = [_compute_client_integer_stats(loader, num_classes) for loader in client_loaders]

            ref_payload = _sum_payload_dicts(plain_payloads)
            dec_payload = ref_payload
            if encryption_ctx is not None:
                if getattr(config, "encryption_scheme", "ckks") == "paillier":
                    payload_keys = list(plain_payloads[0].keys())
                    total_values = sum(int(v.numel()) for v in plain_payloads[0].values())
                    print(
                        f"[PHE][Round {rnd:02d}] {payload_mode} mode: encrypting payload keys={payload_keys} "
                        f"({total_values} values/client)"
                    )
                enc_payloads, encrypt_time, ciphertext_count, encrypted_values, payload_nbytes = _encrypt_payloads(
                    plain_payloads,
                    encryption_ctx,
                    debug_prefix=f"[PHE][Round {rnd:02d}]" if getattr(config, "encryption_scheme", "ckks") == "paillier" else None,
                )
                one_scalars = [1 for _ in enc_payloads]
                if getattr(config, "encryption_scheme", "ckks") == "paillier":
                    print(f"[PHE][Round {rnd:02d}] starting encrypted aggregation across {len(client_updates)} clients")
                dec_payload, aggregate_time, decrypt_time = he_aggregator.aggregate_encrypted_dict(enc_payloads, one_scalars, divide_by=None)
                if getattr(config, "encryption_scheme", "ckks") == "paillier":
                    print(f"[PHE][Round {rnd:02d}] encrypted aggregation and decryption complete")
                if getattr(config, "encryption_scheme", "ckks") == "paillier":
                    print(
                        f"[PHE][Round {rnd:02d}] encrypt_time={encrypt_time:.4f}s "
                        f"agg_time={aggregate_time:.4f}s decrypt_time={decrypt_time:.4f}s "
                        f"ciphertexts={ciphertext_count} payload_nbytes={payload_nbytes}"
                    )
                if compare_reference or encryption_ctx is not None:
                    mean_abs_error, max_abs_error = _payload_error(ref_payload, dec_payload)

            if payload_mode == "analytics":
                analytics_reference = _payload_to_json(ref_payload)
                analytics_decrypted = _payload_to_json(dec_payload)
            else:
                integer_reference = _payload_to_json(ref_payload)
                integer_decrypted = _payload_to_json(dec_payload)
        else:
            raise ValueError(f"Unknown payload_mode: {payload_mode}")

        he_total_time = encrypt_time + aggregate_time + decrypt_time
        acc, loss = evaluate(global_model, test_loader, device)
        round_time = time.time() - round_start
        elapsed = time.time() - start_time

        scheme_name = getattr(config, "encryption_scheme", "none") if config.use_encryption else "none"
        model_name = getattr(config, "ptbxl_model", "-") if config.dataset == "ptbxl" else type(global_model).__name__

        # Param sweep: evaluate N-prefix encrypted payloads after normal round update.
        if payload_mode == "full_model" and param_sweep and encryption_ctx is not None:
            for n in param_sweep:
                prefix_plain = [{"prefix": _state_prefix_tensor(u.state_dict, n)} for u in client_updates]
                ref_prefix = {
                    "prefix": sum(
                        p["prefix"] * (u.num_samples / total_samples)
                        for p, u in zip(prefix_plain, client_updates)
                    )
                }
                enc_prefix, sw_encrypt_time, sw_ct_count, sw_enc_values, sw_nbytes = _encrypt_payloads(prefix_plain, encryption_ctx)
                if getattr(encryption_ctx, "scalar_mode", None) == "int":
                    scalars = [u.num_samples for u in client_updates]
                    dec_prefix, sw_agg_time, sw_dec_time = he_aggregator.aggregate_encrypted_dict(enc_prefix, scalars, divide_by=float(total_samples))
                else:
                    scalars = [u.num_samples / total_samples for u in client_updates]
                    dec_prefix, sw_agg_time, sw_dec_time = he_aggregator.aggregate_encrypted_dict(enc_prefix, scalars, divide_by=None)
                sw_mean_err, sw_max_err = _payload_error(ref_prefix, dec_prefix)
                sw_row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "round": rnd,
                    "dataset": config.dataset,
                    "model": model_name,
                    "num_clients": config.num_clients,
                    "scheme": scheme_name,
                    "payload_mode": payload_mode,
                    "training_time": round(total_train_time, 6),
                    "encrypt_time": round(sw_encrypt_time, 6),
                    "aggregate_time": round(sw_agg_time, 6),
                    "decrypt_time": round(sw_dec_time, 6),
                    "he_total_time": round(sw_encrypt_time + sw_agg_time + sw_dec_time, 6),
                    "total_round_time": round(round_time, 6),
                    "ciphertext_count": sw_ct_count,
                    "encrypted_values": sw_enc_values,
                    "payload_nbytes": sw_nbytes,
                    "accuracy": round(acc, 6),
                    "loss": round(loss, 6),
                    "mean_abs_error": round(sw_mean_err, 8),
                    "max_abs_error": round(sw_max_err, 8),
                    "analytics_reference": "",
                    "analytics_decrypted": "",
                    "integer_reference": "",
                    "integer_decrypted": "",
                }
                if config.save_metrics_csv:
                    _append_csv_row(config.save_metrics_csv, sw_row)
            print(
                f"Round {rnd:02d}: Acc={acc*100:.2f}% Loss={loss:.4f} "
                f"| Train={total_train_time:.2f}s Sweep={len(param_sweep)} configs "
                f"| Total={round_time:.2f}s Elapsed={elapsed:.2f}s"
            )
            continue

        print(
            f"Round {rnd:02d}: Acc={acc*100:.2f}% Loss={loss:.4f} "
            f"| Train={total_train_time:.2f}s Encrypt={encrypt_time:.2f}s "
            f"Agg={aggregate_time:.2f}s Decrypt={decrypt_time:.2f}s | Total={round_time:.2f}s Elapsed={elapsed:.2f}s"
        )

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "round": rnd,
            "dataset": config.dataset,
            "model": model_name,
            "num_clients": config.num_clients,
            "scheme": scheme_name,
            "payload_mode": payload_mode,
            "training_time": round(total_train_time, 6),
            "encrypt_time": round(encrypt_time, 6),
            "aggregate_time": round(aggregate_time, 6),
            "decrypt_time": round(decrypt_time, 6),
            "he_total_time": round(he_total_time, 6),
            "total_round_time": round(round_time, 6),
            "ciphertext_count": ciphertext_count,
            "encrypted_values": encrypted_values,
            "payload_nbytes": payload_nbytes,
            "accuracy": round(acc, 6),
            "loss": round(loss, 6),
            "mean_abs_error": round(mean_abs_error, 8),
            "max_abs_error": round(max_abs_error, 8),
            "analytics_reference": analytics_reference,
            "analytics_decrypted": analytics_decrypted,
            "integer_reference": integer_reference,
            "integer_decrypted": integer_decrypted,
        }
        if config.save_metrics_csv:
            _append_csv_row(config.save_metrics_csv, row)
    return global_model


def parse_args():
    p = argparse.ArgumentParser(description="Modular FedAvg Runner")
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["mnist", "cifar10", "ptbxl"], default="mnist")
    p.add_argument("--ptbxl_model", choices=["cnn_large", "cnn_medium", "logistic", "lstm"], default="cnn_medium",
                   help="PTB-XL model seçimi (yalnızca --dataset ptbxl ile geçerli)")
    p.add_argument("--ptbxl_data_dir", type=str,
                   default="./data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
                   help="PTB-XL veri seti klasör yolu")
    p.add_argument("--use_aug", action="store_true")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--scheduler", choices=["none", "step", "cosine"], default="none")
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--use_encryption", action="store_true")
    p.add_argument("--encryption_scheme", choices=["ckks", "paillier"], default="ckks",
                   help="Which HE scheme to use when --use_encryption is set.")
    p.add_argument("--payload_mode", choices=["full_model", "analytics", "integer_stats"], default="full_model")
    p.add_argument("--analytics_include_grad_norm", action="store_true")
    p.add_argument("--param_sweep", type=str, default=None, help="Comma-separated encrypted parameter counts, e.g. 2,5,10,50")
    p.add_argument("--save_metrics_csv", type=str, default=None)
    p.add_argument("--compare_reference", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
