import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

try:
	from tqdm import tqdm
except ImportError:  # fallback if tqdm not installed
	def tqdm(x, *args, **kwargs):
		return x


def set_seed(seed: int = 42):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


class SimpleCNN(nn.Module):
	"""A small CNN suitable for MNIST."""

	def __init__(self):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),  # 14x14
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),  # 7x7
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 7 * 7, 128),
			nn.ReLU(),
			nn.Linear(128, 10),
		)

	def forward(self, x):
		x = self.features(x)
		return self.classifier(x)


@dataclass
class ClientResult:
	state_dict: Dict[str, torch.Tensor]
	num_samples: int


def create_iid_partitions(dataset: Dataset, num_clients: int) -> List[List[int]]:
	"""Split dataset indices into num_clients IID partitions."""
	indices = list(range(len(dataset)))
	random.shuffle(indices)
	part_size = len(indices) // num_clients
	partitions = [indices[i * part_size:(i + 1) * part_size] for i in range(num_clients)]
	# Last client gets remainder if any
	remainder_start = num_clients * part_size
	if remainder_start < len(indices):
		partitions[-1].extend(indices[remainder_start:])
	return partitions


def get_dataloader(dataset: Dataset, indices: List[int], batch_size: int, shuffle: bool = True) -> DataLoader:
	subset = torch.utils.data.Subset(dataset, indices)
	return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def train_local(model: nn.Module, dataloader: DataLoader, device: torch.device, epochs: int, lr: float) -> ClientResult:
	model_local = SimpleCNN()  # fresh copy of architecture
	model_local.load_state_dict(model.state_dict())  # start from global weights
	model_local.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model_local.parameters(), lr=lr, momentum=0.9)
	model_local.train()
	for _ in range(epochs):
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model_local(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
	return ClientResult(state_dict={k: v.cpu() for k, v in model_local.state_dict().items()}, num_samples=len(dataloader.dataset))


def federated_average(results: List[ClientResult], global_model: nn.Module):
	total_samples = sum(r.num_samples for r in results)
	new_state: Dict[str, torch.Tensor] = {}
	for key in results[0].state_dict.keys():
		weighted_sum = sum(r.state_dict[key] * (r.num_samples / total_samples) for r in results)
		new_state[key] = weighted_sum
	global_model.load_state_dict(new_state)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
	model.eval()
	loss_fn = nn.CrossEntropyLoss()
	correct = 0
	total = 0
	total_loss = 0.0
	with torch.no_grad():
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			loss = loss_fn(outputs, labels)
			total_loss += loss.item() * labels.size(0)
			preds = outputs.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
	return correct / total, total_loss / total


def run_fedavg(
	num_clients: int = 5,
	rounds: int = 5,
	local_epochs: int = 1,
	batch_size: int = 64,
	lr: float = 0.01,
	seed: int = 42,
	use_cuda: bool = True,
):
	set_seed(seed)
	device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
	])
	train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
	test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
	test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

	partitions = create_iid_partitions(train_ds, num_clients)
	global_model = SimpleCNN().to(device)

	print(f"Device: {device} | Clients: {num_clients} | Rounds: {rounds}")

	for r in range(1, rounds + 1):
		client_results: List[ClientResult] = []
		for cid, idxs in enumerate(partitions):
			loader = get_dataloader(train_ds, idxs, batch_size=batch_size, shuffle=True)
			result = train_local(global_model, loader, device, local_epochs, lr)
			client_results.append(result)
		federated_average(client_results, global_model)
		acc, loss = evaluate(global_model, test_loader, device)
		print(f"Round {r:02d}: Test Acc={acc*100:.2f}% Loss={loss:.4f}")

	return global_model


def parse_args():
	parser = argparse.ArgumentParser(description="Federated Averaging on MNIST")
	parser.add_argument("--num_clients", type=int, default=5)
	parser.add_argument("--rounds", type=int, default=5)
	parser.add_argument("--local_epochs", type=int, default=1)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=0.01)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run_fedavg(
		num_clients=args.num_clients,
		rounds=args.rounds,
		local_epochs=args.local_epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		seed=args.seed,
		use_cuda=not args.no_cuda,
	)

