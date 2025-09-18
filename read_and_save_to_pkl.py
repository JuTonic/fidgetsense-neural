from record import Record
import joblib
import os

DATA_DIR = "./data"
dirs = os.listdir(DATA_DIR)

records: list[Record] = []

for dir in dirs:
    dir = os.path.join(DATA_DIR, dir)
    records.append(Record(dir))

joblib.dump(records, "records.pkl")
