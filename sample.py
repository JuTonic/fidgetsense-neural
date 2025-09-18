from dataclasses import dataclass
from record import AccData, Activity, Characteristics, Record
from utils import load_records

@dataclass(frozen=True)
class SampleReading:
    time_diff: list[int]
    first: AccData
    second: AccData
    third: AccData

@dataclass(frozen=True)
class Sample:
    chars: Characteristics
    readings: SampleReading
    label: Activity

def get_sample_of_record(record: Record, index_of_last_reading: int, window: int) -> Sample:
    index_of_first_reading = index_of_last_reading - window

    if index_of_first_reading < 1:
        raise ValueError("The difference between last index and the window must be at least 1")

    time_diff = [
        a - b for a, b in
            zip(record.readings.nano_time[index_of_first_reading:index_of_last_reading], record.readings.nano_time[index_of_first_reading - 1:index_of_last_reading - 1])
    ]
    first = record.readings.first.get_slice(index_of_first_reading, index_of_last_reading)
    second = record.readings.second.get_slice(index_of_first_reading, index_of_last_reading)
    third = record.readings.third.get_slice(index_of_first_reading, index_of_last_reading)

    sample_readings = SampleReading(
        time_diff = time_diff,
        first = first,
        second = second,
        third = third
    )

    label = record.get_activity_of_a_reading(index_of_last_reading)

    return Sample(
        chars=record.chars,
        readings=sample_readings,
        label=label
    )


records = load_records()

print(get_sample_of_record(records[0], 8000, 500))
