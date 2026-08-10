from pathlib import Path
import pandas as pd

def get_machine_ids(data_dir):
    train_dir = Path(data_dir) / "train"
    machine_ids = sorted([file.stem for file in train_dir.glob("*.txt")])
    return machine_ids

def load_machine(machine_id, data_dir):
    data_dir = Path(data_dir)
    train_path = data_dir / "train" / f"{machine_id}.txt"
    test_path = data_dir / "test" / f"{machine_id}.txt"
    label_path = data_dir / "test_label" / f"{machine_id}.txt"

    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)
    label = pd.read_csv(label_path, header=None)

    return train, test, label