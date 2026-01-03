import torch

from .tokenizer import ArcColorTokenizer


class VectorizedSerializer:
    def __init__(self, tokenizer: ArcColorTokenizer) -> None:
        self.tokenizer: ArcColorTokenizer = tokenizer
        self.im_start_id: int = self.tokenizer.vocab["<im_start>"]
        self.im_end_id: int = self.tokenizer.vocab["<im_end>"]
        self.row_sep_id: int = self.tokenizer.row_sep_token_id

    def serialize_grid(self, grid: torch.Tensor) -> torch.Tensor:
        H: int
        W: int
        H, W = grid.shape
        color_tokens: torch.Tensor = grid + self.tokenizer.color_token_offset

        if H <= 1:
            return color_tokens.flatten()

        sep: torch.Tensor = torch.full((H - 1, 1), self.row_sep_id, dtype=torch.long, device=grid.device)
        rows: torch.Tensor = color_tokens[:-1]
        rows_with_sep: torch.Tensor = torch.cat([rows, sep], dim=1)

        last_row: torch.Tensor = color_tokens[-1:]
        full_serialized: torch.Tensor = torch.cat([rows_with_sep.flatten(), last_row.flatten()])
        return full_serialized

    def serialize_task(self, task_data: dict[str, list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        ids: list[torch.Tensor] = [torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)]
        labels: list[torch.Tensor] = [torch.tensor([-100], dtype=torch.long)]

        def append_grid(grid: torch.Tensor, is_input: bool) -> None:
            grid_ids: torch.Tensor = self.serialize_grid(grid)
            wrapped_ids: torch.Tensor = torch.cat([
                torch.tensor([self.im_start_id], dtype=torch.long),
                grid_ids,
                torch.tensor([self.im_end_id], dtype=torch.long)
            ])
            ids.append(wrapped_ids)
            if is_input:
                labels.append(torch.full(wrapped_ids.shape, -100, dtype=torch.long))
            else:
                l: torch.Tensor = torch.cat([
                    torch.tensor([-100], dtype=torch.long),
                    grid_ids,
                    torch.tensor([self.im_end_id], dtype=torch.long)
                ])
                labels.append(l)

        for in_g, out_g in zip(task_data["train_input"], task_data["train_output"]):
            append_grid(in_g, is_input=True)
            append_grid(out_g, is_input=False)

        append_grid(task_data["test_input"][0], is_input=True)
        append_grid(task_data["test_output"][0], is_input=False)

        ids.append(torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long))
        labels.append(torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long))

        return torch.cat(ids), torch.cat(labels)
