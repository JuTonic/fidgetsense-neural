from typing import Any
from numpy.typing import NDArray
from record import Record
from sample import Sample
import joblib

def load_records() -> list[Record]:
    return joblib.load("records.pkl")

def load_samples() -> list[Sample]:
    return joblib.load("samples.pkl")

def load_train_data() -> tuple[NDArray[Any], NDArray[Any]]:
    return joblib.load("X_and_y.pkl")
