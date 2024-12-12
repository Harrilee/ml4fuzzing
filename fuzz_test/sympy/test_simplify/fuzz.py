"""
This script must be run with the correct Sympy version in the path.
Do not run this script with the mutated version of Sympy.
"""

import os, sys
sympy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../sut/sympy'))
sys.path.insert(0, sympy_root)


from sympy import *
import random
from tqdm import tqdm
import pandas as pd

random.seed(42)

letters = symbols('a b c')

class Generator:

    def get_poly(self):
        out = random.choice(letters)
        factor = random.randint(-100, 100)
        constant = random.randint(-3, 3)
        power = random.randint(-2, 4)

        # 50% chance of having a power of 1
        if random.random() < 0.5:
            out = out ** power
        # 10% chance of having a constant
        if random.random() < 0.1:
            out += constant
        # 10% chance of having a factor
        if random.random() < 0.1:
            out *= factor
        return out

    def get_trig(self):
        fun = random.choice([sin, cos, tan, sec, csc, cot])
        power = random.choice([-2, -1, 1, 2])
        factor =  random.randint(-100, 100)
        piSection = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
        out = random.choice(letters)
        # 30% chance of having a pi section
        if random.random() < 0.3:
            out = out + piSection * pi
        # apply the trig function
        out = fun(out)
        out = out ** power
        out = out * factor

        return out


    def getLog(self):
        base = random.choice([2, 4, 8])
        arg = random.choice(letters)
        return log(arg, base) * random.randint(-100, 100)

    def get_random(self):
        return random.choice([self.get_poly(), self.get_trig(), self.getLog()])




def combine_terms(expr1, expr2):
    # 25% chance for each operation: +, -, *, /
    coin = random.random()
    if coin < 0.25:
        return expr1 + expr2
    elif coin < 0.5:
        return expr1 - expr2
    elif coin < 0.75:
        return expr1 * expr2
    else:
        return expr1 / expr2 if expr2 != 0 else expr1

def generate_case():
    loop = random.choice([1, 2])
    out = Generator().get_random()
    for i in range(loop-1):
        out = combine_terms(out, Generator().get_random())
        out = combine_terms(out, Generator().get_poly())
    return str(out)



def generate_all():
    cases = set()
    num_cases = 0
    with tqdm(total=10000) as pbar:
        while num_cases < 10000:
            case = generate_case()
            if case not in cases:
                cases.add(case)
                num_cases += 1
                pbar.update(1)
    return cases

if __name__ == "__main__":
    cases = generate_all()
    df = pd.DataFrame(list(cases), columns=['expr'])
    df.to_csv('input.csv', index=False)
