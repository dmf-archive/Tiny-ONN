import json
import sys
from pathlib import Path

def main():
    count = 0
    p = Path('data/ARC-AGI-2/data')
    all_files = list(p.glob('**/*.json'))
    total = len(all_files)
    
    if total == 0:
        print("No JSON files found in the specified directory.")
        return

    for i, task_file in enumerate(all_files):
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task = json.load(f)
            
            pairs = task.get('train', []) + task.get('test', [])
            count += sum(1 for p in pairs if 
                         'input' in p and 'output' in p and
                         len(p['input']) > 0 and len(p['input'][0]) > 0 and
                         len(p['output']) > 0 and len(p['output'][0]) > 0 and
                         (len(p['input']) > 10 or len(p['input'][0]) > 10 or 
                          len(p['output']) > 10 or len(p['output'][0]) > 10))
                         
            progress = (i + 1) / total
            bar = '#' * int(progress * 20)
            sys.stdout.write(f"\r[{bar:<20}] {i+1}/{total} | Total pairs > 10x10: {count}")
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            sys.stderr.write(f"\nError processing file {task_file}: {e}\n")
            continue
            
    print()

if __name__ == "__main__":
    main()