import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from physics import max_deflection, max_stress


INPUT_COLUMNS = ["length", "width", "height", "force", "E"]
TARGET_COLUMNS = ["stress", "deflection"]


class BeamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(
            X.values,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y.values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_dataset(n=10000):
    materials = {
        "aluminum": 69e9,
        "steel": 200e9
    }

    material = np.random.choice(list(materials.keys()), n)

    length = np.random.uniform(0.5, 2.0, n)
    width = np.random.uniform(0.02, 0.10, n)
    height = np.random.uniform(0.02, 0.10, n)
    force = np.random.uniform(10, 500, n)

    E = np.array([materials[m] for m in material])

    stress = max_stress(
        force, length, width, height
    )

    deflection = max_deflection(
        force, length, width, height, E
    )

    data = pd.DataFrame({
        "length": length,
        "width": width,
        "height": height,
        "force": force,
        "E": E / 1e9,
        "stress": stress / 1e6,
        "deflection": deflection * 1000
    })

    return data


def create_dataloaders(train_dir, test_dir, batch):

    train_data = pd.read_csv(train_dir)
    test_data = pd.read_csv(test_dir)

    # Separate inputs and targets
    X_train = train_data[INPUT_COLUMNS]
    y_train = train_data[TARGET_COLUMNS]

    X_test = test_data[INPUT_COLUMNS]
    y_test = test_data[TARGET_COLUMNS]

    # Statistics calculated ONLY from training data
    X_mean = X_train.mean()
    X_std = X_train.std()

    y_mean = y_train.mean()
    y_std = y_train.std()

    # Standardize inputs
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Standardize targets
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    train_dataset = BeamDataset(
        X_train,
        y_train
    )

    test_dataset = BeamDataset(
        X_test,
        y_test
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch,
        shuffle=False
    )

    scalers = {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std
    }

    return train_dataloader, test_dataloader, scalers


def main():

    data = create_dataset(10000)

    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    split = int(0.8 * len(data))

    train_data = data.iloc[:split]
    test_data = data.iloc[split:]

    train_data.to_csv(
        "data/train.csv",
        index=False
    )

    test_data.to_csv(
        "data/test.csv",
        index=False
    )

    print(f"Generated {len(data)} beam samples.")
    print(data.head())


if __name__ == "__main__":
    main()