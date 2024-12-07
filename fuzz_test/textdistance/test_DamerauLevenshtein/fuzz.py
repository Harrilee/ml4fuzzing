import pandas as pd
import random
random.seed(42)

"""
Fuzzing test for DamerauLevenshtein
https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
Generate s1 and s2 to 
"""

def generate_case():
    s1 = ''.join(random.choices('abcdef', k=random.randint(0, 10)))
    s2 = ''.join(random.choices('abcdef', k=random.randint(0, 10)))
    return (s1, s2)

def generate_all():
    cases = set()
    num_cases = 0
    while num_cases < 10000:
        case = generate_case()
        if case not in cases:
            cases.add(case)
            num_cases += 1
    return cases

if __name__ == "__main__":
    cases = generate_all()
    df = pd.DataFrame(list(cases), columns=['s1', 's2'])
    df.to_csv('input.csv', index=False)