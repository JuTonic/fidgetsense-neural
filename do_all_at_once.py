from record import AccData, Activity
from sample import Sample, get_sample_of_a_record
from utils import load_records
import joblib
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

N = 4
WINDOW = 300

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

X_validate = []
y_validate = []

X = []
y = []

def compute_magnitude(acc_data: AccData):
    return np.sqrt(np.power(acc_data.x, 2) + np.power(acc_data.y, 2) + np.power(acc_data.z, 2))

for sample in samples:
    r = sample.readings
    x = np.array([r.time_diff, r.first.x, r.first.y, r.first.z, r.second.x, r.second.y, r.second.z, r.third.x, r.third.y, r.third.z, compute_magnitude(r.first), compute_magnitude(r.second), compute_magnitude(r.third)]).transpose()
    if len(x) != WINDOW:
        continue

    if sample.name == "./data/19":
        X_validate.append(x)
        y_validate.append(sample.label.value)
        continue

    X.append(x)
    y.append(sample.label.value)

X_validate = np.array(X_validate)
y_validate = np.array(y_validate)
X = np.array(X)
y = np.array(y)

y_validate = to_categorical(y_validate - 1, num_classes=4)
y = to_categorical(y - 1, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition using LSTM
model = models.Sequential()

# Adding LSTM layer(s)
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(WINDOW, 13)))
model.add(layers.BatchNormalization())  # Batch normalization
model.add(layers.Dropout(0.3))  # Dropout for regularization

# You can add another LSTM layer to capture more complex patterns
model.add(layers.LSTM(256, return_sequences=True))  # LSTM with 256 units
model.add(layers.BatchNormalization())  # Batch normalization
model.add(layers.Dropout(0.3))  # Dropout for regularization

model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(WINDOW, 10)))
model.add(layers.MaxPooling1D(2))
model.add(layers.LSTM(128))

# Add Dropout after LSTM
model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting

# Fully connected (dense) layers
model.add(layers.Dense(128, activation='relu'))  # Dense layer with 128 units
model.add(layers.Dropout(0.5))  # Dropout for regularization
model.add(layers.Dense(4, activation='softmax'))  # Output layer for 4 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler (optional)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

loss, accuracy = model.evaluate(X_validate, y_validate)

print(f"Test Accuracy: {accuracy:.4f}")
