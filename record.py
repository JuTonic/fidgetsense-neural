from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
import os

SPLIT_CHAR_SYMBOL = "="
CHAR_FILE_NAME = "chars.txt"
LABEL_FILE_NAME = "labels.csv"
READINGS_FILE_NAME = "readings.csv"

class Sex(Enum):
    FEMALE = 0
    MALE = 1

class Hand(Enum):
    LEFT = 0
    RIGHT = 1

@dataclass(frozen=True)
class Characteristics:
    sex: Sex
    hand: Hand
    height: int | None

class Activity(Enum):
    OTHER = 0
    NOTHING = 1
    TYPING = 2
    SCROLLING = 3
    FIDGETING = 4

    @staticmethod
    def parse_activity_label(label: str) -> "Activity":
        ACTIVITY_MAPPING: dict[str, Activity] = {
            "o": Activity.OTHER,
            "n": Activity.NOTHING,
            "t": Activity.TYPING,
            "s": Activity.SCROLLING,
            "f": Activity.FIDGETING
        }

        return ACTIVITY_MAPPING[label]

@dataclass(frozen=True)
class Labels:
    index_of_reading: list[int]
    activity: list[Activity]
    timestamp: list[int]

    def iter(self) -> Generator[tuple[int, Activity, int], None, None]:
        for timestamp, activity, index in zip(self.timestamp, self.activity, self.index_of_reading):
            yield index, activity, timestamp

@dataclass(frozen=True)
class AccData:
    x: list[int]
    y: list[int]
    z: list[int]

    def get_slice(self, i: int, j: int) -> "AccData":
        return AccData(x=self.x[i:j], y=self.y[i:j], z=self.z[i:j])

@dataclass(frozen=True)
class Readings:
    unix_time: list[int]
    nano_time: list[int]
    first: AccData
    second: AccData
    third: AccData

    def __len__(self):
        return len(self.unix_time)


class Record:
    name: str
    chars: Characteristics
    labels: Labels
    readings: Readings

    def __init__(self, dir: str):
        self.name = dir

        char_file, label_file, readings_file = map(lambda f: os.path.join(dir, f), [CHAR_FILE_NAME, LABEL_FILE_NAME, READINGS_FILE_NAME])

        lines: list[str] = []

        with open(char_file, "r") as f:
            for line in f:
                lines.append(line.strip())
        
        sex, hand, height = map(lambda line: line.split(SPLIT_CHAR_SYMBOL)[1], lines)

        sex = Sex.MALE if sex == "m" else Sex.FEMALE
        hand = Hand.LEFT if hand == "l" else Hand.RIGHT
        height = int(height) if height != "none" else None 

        self.chars = Characteristics(sex=sex, hand=hand, height=height)

        labels: tuple[list[int], list[Activity], list[int]] = ([], [], [0])
        with open(label_file, "r") as f:
            for line in f:
                timestamp, activity_label = line.strip().split(";")
                labels[0].append(int(timestamp))
                labels[1].append(Activity.parse_activity_label(activity_label))

        label_index_tracker = 1

        unix_time_arr: list[int] = []
        nano_time_arr: list[int] = []
        first: list[list[int]] = [[], [], []]
        second: list[list[int]] = [[], [], []]
        third: list[list[int]] = [[], [], []]
        with open(readings_file, "r") as f:
            for i, line in enumerate(f):
                values = line.strip().split(";")

                if len(values) != 11 and int(values[0]) < 0 and int(values[1]) < 0:
                    continue

                try:
                    unix_time, nano_time, x1, y1, z1, x2, y2, z2, x3, y3, z3 = map(lambda v: int(v), values)
                except:
                    continue

                if label_index_tracker < len(labels[0]) and unix_time > labels[0][label_index_tracker]:
                    labels[2].append(i)
                    label_index_tracker += 1

                unix_time_arr.append(unix_time)
                nano_time_arr.append(nano_time)
                first[0].append(x1)
                first[1].append(y1)
                first[2].append(z1)
                second[0].append(x2)
                second[1].append(y2)
                second[2].append(z2)
                third[0].append(x3)
                third[1].append(y3)
                third[2].append(z3)

        self.labels = Labels(
            timestamp=labels[0],
            activity=labels[1],
            index_of_reading=labels[2] 
        )

        self.readings = Readings(
            unix_time=unix_time_arr,
            nano_time=nano_time_arr,
            first=AccData(
                x=first[0],
                y=first[1],
                z=first[2],
            ),
            second=AccData(
                x=second[0],
                y=second[1],
                z=second[2],
            ),
            third=AccData(
                x=third[0],
                y=third[1],
                z=third[2],
            ),
        )

    def get_activity_of_a_reading(self, i: int) -> Activity:
        for index, activity, _ in self.labels.iter():
            if i < index:
                return activity

        raise ValueError("Index must be smaller than the number of readings")
