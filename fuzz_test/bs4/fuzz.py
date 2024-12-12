from fuzz_test.bs4.fuzz_search import random_find_all_args
from fuzz_test.bs4.genHTML import generate_random_html
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import pandas as pd


def generate_case():
    fakeHTML = generate_random_html()
    fakeArgs = random_find_all_args()
    soup = BeautifulSoup(fakeHTML, "html.parser")
    elements = soup.find_all(**fakeArgs)
    res = ""
    for element in elements:
        res += str(element)
    if res:
        return fakeHTML, json.dumps(fakeArgs)
    return generate_case()

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
    df = pd.DataFrame(list(cases), columns=['html', 'search_args'])
    df.to_csv('input.csv', index=False)