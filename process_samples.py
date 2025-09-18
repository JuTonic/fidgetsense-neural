from utils import load_samples
import joblib
import numpy as np

samples = load_samples()

X = []
y = []

for sample in samples:
    r = sample.readings
    x = np.array([r.time_diff, r.first.x, r.first.y, r.first.z, r.second.x, r.second.y, r.second.z, r.third.x, r.third.y, r.third.z]).transpose()
    if len(x) != 500:
        continue
    X.append(x)
    y.append(sample.label.value)

X = np.array(X)
y = np.array(y)

print(X)
print(y)

joblib.dump((X, y), "X_and_y.pkl")
