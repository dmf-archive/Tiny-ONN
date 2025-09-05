import json
from pathlib import Path

class Tokenizer:
    def encode(self, text: str) -> list:
        return text.split(' ')

def serialize_grid(grid: list[list[int]]) -> str:
    return ' \\n '.join([' '.join(map(str, row)) for row in grid])

def analyze_lengths():
    tokenizer = Tokenizer()
    data_path = Path('data/ARC-AGI-2/data/training')
    file_paths = sorted(list(data_path.glob('*.json')))
    
    lengths = []
    for file_path in file_paths:
        with open(file_path) as f:
            task_data = json.load(f)
            
            context_str = '<|bos|>'
            for pair in task_data['train']:
                problem_str = f"problem <|im_start|> {serialize_grid(pair['input'])} <|im_end|>"
                solution_str = f"solution <|im_start|> {serialize_grid(pair['output'])} <|im_end|>"
                context_str += f" {problem_str} {solution_str}"
            
            test_problem_str = f"problem <|im_start|> {serialize_grid(task_data['test'][0]['input'])} <|im_end|> solution <|im_start|>"
            full_sequence = f'{context_str} {test_problem_str}'
            
            lengths.append(len(tokenizer.encode(full_sequence)))

    print(f'Max sequence length: {max(lengths)}')
    print(f'Average sequence length: {sum(lengths) / len(lengths)}')

if __name__ == "__main__":
    analyze_lengths()