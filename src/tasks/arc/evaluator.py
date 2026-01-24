import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from .data import ARCTokenizer, ARCProcessor
from .config import GenerationConfig
from .observer import ARCObserver

class ArcEvaluator:
    def __init__(
        self, 
        model: torch.nn.Module, 
        tokenizer: ARCTokenizer, 
        processor: ARCProcessor,
        observer: ARCObserver,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.observer = observer
        self.device = device

    @torch.no_grad()
    def greedy_decode(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 1024
    ) -> torch.Tensor:
        self.model.eval()
        if not hasattr(self.model, "generate"):
            raise NotImplementedError("Model must support transformers .generate() for greedy decoding")
            
        output_ids = self.model.generate(
            input_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_id,
            eos_token_id=self.tokenizer.eos_id,
            use_cache=True
        )
        return output_ids[0, input_ids.size(1):]

    @torch.no_grad()
    def dfs_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        self.model.eval()
        max_score = -math.log(config.min_prob) if config.min_prob > 0 else 100.0
        
        outputs = self.model(input_ids.to(self.device), use_cache=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        past_key_values = outputs.get("past_key_values") if isinstance(outputs, dict) else None
        
        results = []
        self._recursive_dfs(
            torch.empty(0, dtype=torch.long, device=self.device),
            0.0,
            logits[:, -1, :],
            past_key_values,
            max_score,
            config.max_new_tokens,
            results
        )
        
        if not results:
            raise RuntimeError(f"DFS failed to find any valid path within min_prob {config.min_prob}")
            
        results.sort(key=lambda x: x[0])
        return results[0][1]

    def _recursive_dfs(
        self,
        current_tokens: torch.Tensor,
        current_score: float,
        logits: torch.Tensor,
        past_key_values: Any,
        max_score: float,
        max_new_tokens: int,
        results: List[Tuple[float, torch.Tensor]]
    ):
        if len(current_tokens) >= max_new_tokens:
            results.append((current_score, current_tokens))
            return

        probs = F.softmax(logits, dim=-1).squeeze(0)
        threshold = math.exp(current_score - max_score)
        valid_indices = torch.where(probs >= threshold)[0]
        
        if valid_indices.numel() == 0:
            return

        sorted_indices = valid_indices[torch.argsort(probs[valid_indices], descending=True)]

        for idx in sorted_indices:
            token_id = idx.item()
            new_score = current_score - math.log(probs[idx].item())
            new_tokens = torch.cat([current_tokens, idx.unsqueeze(0)])
            
            if token_id == self.tokenizer.eos_id:
                results.append((new_score, new_tokens))
                continue
            
            outputs = self.model(idx.view(1, 1), past_key_values=past_key_values, use_cache=True)
            next_logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            next_pkv = outputs.get("past_key_values") if isinstance(outputs, dict) else None
            
            self._recursive_dfs(new_tokens, new_score, next_logits[:, -1, :], next_pkv, max_score, max_new_tokens, results)

    def evaluate_task(self, task_data: Dict[str, Any], config: GenerationConfig, visualize: bool = False):
        input_ids = self.processor.serialize_for_inference(task_data).unsqueeze(0)
        
        target_grid = torch.tensor(task_data["test"][0]["output"], dtype=torch.long)
        input_grid = torch.tensor(task_data["test"][0]["input"], dtype=torch.long)
        
        # Optimization: Limit generation length to target length + small buffer if target is known
        target_tokens = self.processor.encode_grid(target_grid)
        max_new_tokens = min(config.max_new_tokens, len(target_tokens) + 2)

        if config.use_dfs:
            pred_tokens = self.dfs_search(input_ids, config)
        else:
            pred_tokens = self.greedy_decode(input_ids, max_new_tokens)
        
        pred_grid = self.processor.decode_grid(pred_tokens)
        
        if visualize:
            self.observer.visualize_prediction(input_grid, target_grid, pred_grid)
        return torch.equal(pred_grid, target_grid)
