import torch
import random
from .config import CONFIG

TOKENIZER = {
    'BOS': 0, 'EOS': 1, 'PAD': 2,
    '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12,
    '+': 13, '-': 14, '×': 15, '=': 16,
    '<think>': 17, '</think>': 18, '\n': 19,
    'CARRY': 20, 'BORROW': 21
}
REVERSE_TOKENIZER = {v: k for k, v in TOKENIZER.items()}
IGNORE_INDEX = -100

def str_to_tokens(s):
    tokens = []
    s = str(s)
    if s.startswith('-'):
        tokens.append(TOKENIZER['-'])
        s = s[1:]
    tokens.extend([TOKENIZER.get(c, -1) for c in s])
    return [t for t in tokens if t != -1]

def generate_arithmetic_data(num_samples, seq_len, operand_ranges):
    sents, labels = [], []

    def _get_addition_cot(a, b):
        s_a, s_b = str(a), str(b)
        max_len = max(len(s_a), len(s_b))
        s_a, s_b = s_a.zfill(max_len), s_b.zfill(max_len)
        cot_tokens = []
        carry = 0
        for i in range(max_len - 1, -1, -1):
            d1, d2 = int(s_a[i]), int(s_b[i])
            sum_val = d1 + d2 + carry
            result_digit = sum_val % 10
            new_carry = sum_val // 10
            cot_tokens.extend(str_to_tokens(s_a[i]) + [TOKENIZER['+']] + str_to_tokens(s_b[i]) + [TOKENIZER['=']] + str_to_tokens(result_digit))
            cot_tokens.extend([TOKENIZER['CARRY'], *str_to_tokens(new_carry), TOKENIZER['\n']])
            carry = new_carry
        return cot_tokens

    def _get_subtraction_cot(a, b):
        s_a, s_b = str(a), str(b)
        max_len = max(len(s_a), len(s_b))
        s_a, s_b = s_a.zfill(max_len), s_b.zfill(max_len)
        cot_tokens = []
        borrow = 0
        for i in range(max_len - 1, -1, -1):
            d1 = int(s_a[i]) - borrow
            d2 = int(s_b[i])
            if d1 < d2:
                d1 += 10
                new_borrow = 1
            else:
                new_borrow = 0
            result_digit = d1 - d2
            cot_tokens.extend(str_to_tokens(s_a[i]) + [TOKENIZER['-']] + str_to_tokens(s_b[i]) + [TOKENIZER['=']] + str_to_tokens(result_digit))
            cot_tokens.extend([TOKENIZER['BORROW'], *str_to_tokens(new_borrow), TOKENIZER['\n']])
            borrow = new_borrow
        return cot_tokens

    def _get_multiplication_cot(a, b):
        s_b = str(b)
        cot_tokens = []
        partial_products = []
        for i, digit in enumerate(reversed(s_b)):
            multiplier = int(digit)
            partial_product = a * multiplier
            partial_products.append(partial_product * (10**i))
            
            s_a = str(a)
            carry = 0
            for j in range(len(s_a) - 1, -1, -1):
                d1 = int(s_a[j])
                prod_val = d1 * multiplier + carry
                result_digit = prod_val % 10
                new_carry = prod_val // 10
                cot_tokens.extend(str_to_tokens(s_a[j]) + [TOKENIZER['×']] + str_to_tokens(digit) + [TOKENIZER['+']] + str_to_tokens(carry) + [TOKENIZER['=']] + str_to_tokens(result_digit))
                cot_tokens.extend([TOKENIZER['CARRY'], *str_to_tokens(new_carry), TOKENIZER['\n']])
                carry = new_carry

        if len(partial_products) > 1:
            current_sum = 0
            for p in partial_products:
                cot_tokens.extend(_get_addition_cot(current_sum, p))
                current_sum += p
        return cot_tokens

    while len(sents) < num_samples:
        a_range = random.choice(operand_ranges)
        a = random.randint(*a_range)
        if len(operand_ranges) > 1 and a >= 100:
            b_range = random.choice(operand_ranges[:-1])
        else:
            b_range = random.choice(operand_ranges)
        b = random.randint(*b_range)
        op_str = random.choice(['+', '-', '×'])
        
        a_orig, b_orig = a, b
        if op_str == '+':
            cot_tokens = _get_addition_cot(a, b)
            res = a + b
        elif op_str == '-':
            if a < b: a, b = b, a
            cot_tokens = _get_subtraction_cot(a, b)
            res = a_orig - b_orig
        else:
            cot_tokens = _get_multiplication_cot(a, b)
            res = a * b

        problem_tokens = str_to_tokens(a_orig) + [TOKENIZER[op_str]] + str_to_tokens(b_orig)
        answer_tokens = str_to_tokens(res)

        full_seq = ([TOKENIZER['BOS']] + problem_tokens + [TOKENIZER['=']] +
                    [TOKENIZER['<think>']] + cot_tokens + [TOKENIZER['</think>']] +
                    answer_tokens + [TOKENIZER['EOS']])
        
        if len(full_seq) >= seq_len: continue

        x, y = full_seq[:-1], full_seq[1:]
        
        loss_mask = ([0] * (len(problem_tokens) + 2 + len(cot_tokens) + 2) + [1] * len(answer_tokens))
        
        x.extend([TOKENIZER['PAD']] * (seq_len - len(x)))
        y.extend([TOKENIZER['PAD']] * (seq_len - len(y)))
        loss_mask.extend([0] * (seq_len - len(loss_mask)))

        y_masked = [y_i if mask_i == 1 else IGNORE_INDEX for y_i, mask_i in zip(y, loss_mask)]
        
        sents.append(torch.tensor(x))
        labels.append(torch.tensor(y_masked))

    return torch.stack(sents).long(), torch.stack(labels).long()