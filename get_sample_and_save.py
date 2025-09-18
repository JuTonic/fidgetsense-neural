from record import Activity
from sample import Sample, get_sample_of_a_record
from utils import load_records
import joblib
import random

N = 10
WINDOW = 500

random.seed(42)
records = load_records()

samples: list[Sample] = []

for record in records:
    sample_from: list[tuple[int, int]] = []
    labels_arr = list(record.labels.iter())
    for i, (index, activity, _) in enumerate(labels_arr):
        if activity != Activity.OTHER:
            sample_from.append((index, labels_arr[i + 1][0]))

    for start, end in sample_from:
        start = start + WINDOW
        if start > end:
            continue
        for i in range(N):
            samples.append(get_sample_of_a_record(record, random.randint(start, end), WINDOW))

joblib.dump(samples, "samples.pkl")
