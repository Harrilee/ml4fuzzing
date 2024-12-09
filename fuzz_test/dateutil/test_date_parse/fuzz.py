import pandas as pd
import random
from datetime import datetime, timedelta, timezone


random.seed(42)

"""
Fuzzing test for DamerauLevenshtein
https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
Generate s1 and s2 to 
"""

dt_formats = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%a, %d %b %Y %H:%M:%S %z",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%A, %d %B %Y",
    "%H:%M:%S",
    "%I:%M %p",
    "%H:%M:%S.%f",
    "%H:%M:%S %z",
    "%I:%M:%S %p %z",
    "%Y%m%d",
    "%Y%m%d%H%M%S",
    "%d-%m-%y",
    "%y-%m-%d",
    "%b %d '%y",
    "%d/%m/%y %H:%M:%S",
]

time_deltas = ["days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"]


def generate_case():

    if random.random() > 0.8:
        my_dt = datetime(
            year=random.randint(1900, 2100),
            month=random.randint(1, 12),
            day=random.randint(1, 28),
            hour=random.randint(0, 23),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
            microsecond=random.randint(0, 999999),
            tzinfo=timezone(timedelta(hours=random.randint(-12, 12)))
        )
        selected_time_delta = random.choices(time_deltas, k=random.randint(0, 7))
        args = {k: random.randint(0, 100) for k in selected_time_delta}
        delta = timedelta(**args)
        my_dt = my_dt + delta
        dt_format = random.choice(dt_formats)
        my_dt_str = my_dt.strftime(dt_format)
    else:
        day = random.choice([28, 29, 30, 21, 32])
        year = random.choice([2000, 2020, 2200])
        month = random.choice([2, 4, 12])
        hour = random.choice([11, 13, 24, 25, 0])
        minute = random.choice([0, 60, 61, 59])

        my_dt_str = f"{day}/{month}/{year} {hour}:{minute}"

    return my_dt_str


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
    df = pd.DataFrame(list(cases), columns=['dateStr'])
    df.to_csv('input.csv', index=False)
