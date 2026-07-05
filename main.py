from src.data_loader import get_machine_ids, load_machine
from src.preprocessing import validate_machine_data
from evaluation.metrics import evaluate_predictions

machine_ids = get_machine_ids()

print("Machines:", machine_ids[:10])
print("Number of Machines:", len(machine_ids))

train, test, label = load_machine('machine-1-1')

print("Train shape:", train.shape)
print("Test.shape", test.shape)

is_valid, message = validate_machine_data(train, test, label)
if is_valid:
    print(f"✅{message}")
else:
    print(f"❌{message}")
