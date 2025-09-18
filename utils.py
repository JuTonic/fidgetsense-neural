from record import Record
from sample import Sample
import joblib

def load_records() -> list[Record]:
    return joblib.load("records.pkl")

def load_samples() -> list[Sample]:
    return joblib.load("samples.pkl")
