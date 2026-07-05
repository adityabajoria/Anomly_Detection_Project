
def validate_machine_data(train, test, labels):
    if train.shape[1] != test.shape[1]:
        return False, "Train and Test have different number of features"

    if len(test) != len(labels):
        return False, "Test data and labels have different number of rows."

    if train.isnull().sum().sum() > 0:
        return False, "Training data contains missing/null values"

    if test.isnull().sum().sum() > 0:
        return False, "Test data contains missing/null values"

    unique_labels = set(labels[0].unique())
    if not unique_labels.issubset({0, 1}):
        return False, "Invalid labels"

    return True, "Data validation passed successfully"
