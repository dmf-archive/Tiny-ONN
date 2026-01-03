import torch
from accelerate import Accelerator
from rich.console import Console
from torch.utils.data import DataLoader

from .configs import load_config
from .data.collator import ArcCollator
from .data.dataset import ArcTaskDataset
from .data.tokenizer import ArcColorTokenizer
from .models import get_model
from .optimizers import get_optimizer


class UnifiedTrainer:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        torch.manual_seed(self.config.training.seed)
        self.console = Console()

        self.tokenizer = ArcColorTokenizer()
        self._setup_data()
        self._setup_model_and_optimizer()

    def _setup_data(self) -> None:
        self.train_dataset = ArcTaskDataset(
            self.config.data.data_path,
            split="training",
            warmup_ratio=self.config.data.warmup_dataset_ratio
        )
        self.collator = ArcCollator(self.tokenizer, max_len=self.config.model.max_position_embeddings)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.config.data.num_workers
        )

    def _setup_model_and_optimizer(self) -> None:
        self.model = get_model(self.config.model)
        self.optimizer = get_optimizer(self.model, self.config)

        self.model, self.train_loader = self.accelerator.prepare(
            self.model, self.train_loader
        )

    def train(self) -> None:
        self.console.print("[bold green]Starting Unified Training...[/bold green]")
        self.model.train()

        global_step = 0
        for epoch in range(self.config.training.num_epochs):
            for batch in self.train_loader:
                if not batch: continue

                def closure(b=batch) -> torch.Tensor:
                    self.optimizer.zero_grad()
                    outputs = self.model(**b)
                    loss: torch.Tensor = outputs["loss"]
                    if loss is None:
                        raise ValueError(f"Loss is None! Batch keys: {b.keys()}")
                    return loss

                loss_val: torch.Tensor = self.optimizer.step(closure) # type: ignore
                if loss_val is None:
                    self.console.print("[yellow]Warning: optimizer.step returned None[/yellow]")

                if global_step % 10 == 0:
                    pi: float = torch.exp(-loss_val).item()
                    self.console.print(f"Epoch {epoch} | Step {global_step} | Loss: {loss_val.item():.4f} | PI: {pi:.4f}")

                global_step += 1

                if self.config.training.pi_threshold and torch.exp(-loss_val) > self.config.training.pi_threshold:
                    self.console.print("[bold cyan]PI Threshold reached. Early stopping.[/bold cyan]")
                    return

if __name__ == "__main__":
    import sys
    config_path: str = sys.argv[1] if len(sys.argv) > 1 else "src/configs/default.toml"
    trainer = UnifiedTrainer(config_path)
    trainer.train()
