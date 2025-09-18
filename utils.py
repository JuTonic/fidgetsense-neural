from record import Record
import joblib

def load_records() -> list[Record]:
    return joblib.load("records.pkl")
