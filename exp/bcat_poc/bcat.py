import torch
from typing import List, Tuple

class BCAT:
    def __init__(self, block_meta: torch.Tensor, block_values: List[torch.Tensor]):
        self.block_meta = block_meta
        self.block_values = block_values

    @staticmethod
    def cluster(raw_weights: torch.Tensor, threshold: float = 0.5, min_block_size: int = 8) -> 'BCAT':
        mask = raw_weights > 0
        rows, cols = mask.shape
        visited = torch.zeros_like(mask, dtype=torch.bool)
        
        block_meta = []
        block_values = []

        for r in range(rows):
            for c in range(cols):
                if mask[r, c] and not visited[r, c]:
                    r_start, c_start = r, c
                    r_end, c_end = r, c

                    q = [(r, c)]
                    visited[r, c] = True
                    head = 0
                    while head < len(q):
                        curr_r, curr_c = q[head]
                        head += 1

                        r_end = max(r_end, curr_r)
                        c_end = max(c_end, curr_c)
                        
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and \
                                   mask[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    q.append((nr, nc))
                    
                    row_len = r_end - r_start + 1
                    col_len = c_end - c_start + 1

                    if row_len * col_len >= min_block_size:
                        block_density = mask[r_start:r_end+1, c_start:c_end+1].float().mean()
                        if block_density >= threshold:
                            meta = torch.tensor([r_start, c_start, row_len, col_len], dtype=torch.int32)
                            values = raw_weights[r_start:r_end+1, c_start:c_end+1]
                            block_meta.append(meta)
                            block_values.append(values)
                            visited[r_start:r_end+1, c_start:c_end+1] = True

        if not block_meta:
            return BCAT(torch.empty((0, 4), dtype=torch.int32), [])

        return BCAT(torch.stack(block_meta), block_values)
