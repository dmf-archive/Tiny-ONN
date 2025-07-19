import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, get_scheduler

from .config import TrainConfig
from .hooks import GradientInterceptor


class TrainerEngine:
    def __init__(
        self,
        config: TrainConfig,
        model: PreTrainedModel,
        optimizer_experts: Optimizer,
        optimizer_router: Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.optimizer_experts = optimizer_experts
        self.optimizer_router = optimizer_router
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.grad_scaler = GradScaler()
        self.interceptor = GradientInterceptor(model)

        self.lr_scheduler_experts = get_scheduler(
            name="linear",
            optimizer=self.optimizer_experts,
            num_warmup_steps=config.training.lr_scheduler_warmup_steps,
            num_training_steps=len(self.train_dataloader)
            * config.training.num_train_epochs,
        )
        self.lr_scheduler_router = get_scheduler(
            name="linear",
            optimizer=self.optimizer_router,
            num_warmup_steps=config.training.lr_scheduler_warmup_steps,
            num_training_steps=len(self.train_dataloader)
            * config.training.num_train_epochs,
        )

    def _hyper_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.optimizer_experts.zero_grad()
        self.optimizer_router.zero_grad()

        inputs = {k: v.to(self.device) for k, v in batch.items()}

        with autocast():
            outputs = self.model(
                **inputs, output_hidden_states=True, output_attentions=True
            )
            main_loss = outputs.loss

            # This is a placeholder for where router logits would be captured
            # In a real implementation, the model's forward pass would need to return this
            router_logits = self.model.model.layers[0].mlp.gate(
                outputs.hidden_states[-1].view(-1, self.model.config.hidden_size)
            )

        self.grad_scaler.scale(main_loss).backward(retain_graph=True)
        self.grad_scaler.step(self.optimizer_experts)
        self.lr_scheduler_experts.step()

        surprises = self.interceptor.get_surprises()

        # Placeholder for router loss calculation
        # This requires aligning surprises with the router_logits, which is complex
        # For now, we'll just use a dummy loss
        if surprises and router_logits is not None:
            # A real implementation would calculate optimal experts based on surprises
            # and compute CrossEntropyLoss against router_logits
            router_loss = F.cross_entropy(
                router_logits,
                torch.randint_like(router_logits, 0, router_logits.shape[-1]).argmax(
                    dim=-1
                ),
            )
        else:
            router_loss = torch.tensor(0.0, device=self.device)

        self.grad_scaler.scale(router_loss).backward()
        self.grad_scaler.step(self.optimizer_router)
        self.lr_scheduler_router.step()

        self.grad_scaler.update()
        self.interceptor.clear()

        return {
            "main_loss": main_loss.item(),
            "router_loss": router_loss.item(),
        }

    def train(self):
        for epoch in range(self.config.training.num_train_epochs):
            self.model.train()
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                metrics = self._hyper_step(batch)
                progress_bar.set_postfix(metrics)

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.eval_dataloader)
        return {"eval_loss": avg_loss, "perplexity": torch.exp(torch.tensor(avg_loss))}
