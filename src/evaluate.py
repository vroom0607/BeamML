import pandas as pd
import torch

def eval(model: torch.nn.Module,
         test_dataloader: torch.utils.data.DataLoader,
         device: torch.device,
         scalers: dict[str, pd.Series]):
    model.eval()

    stress_errors = []
    deflection_errors = []

    y_mean = torch.tensor(
        scalers["y_mean"].values,
        dtype=torch.float32
    ).to(device)

    y_std = torch.tensor(
        scalers["y_std"].values,
        dtype=torch.float32
    ).to(device)

    
    with torch.inference_mode():

        for X, y in test_dataloader:

            X = X.to(device)
            y = y.to(device)

            predictions = model(X)

            # Convert standardized predictions back
            # to MPa and mm
            predictions = (
                predictions * y_std + y_mean
            )

            actual = (
                y * y_std + y_mean
            )

            stress_error = torch.abs(
                predictions[:, 0] - actual[:, 0]
            )

            deflection_error = torch.abs(
                predictions[:, 1] - actual[:, 1]
            )

            stress_errors.extend(
                stress_error.cpu().tolist()
            )

            deflection_errors.extend(
                deflection_error.cpu().tolist()
            )


    print("\nModel Performance")
    print("------------------")

    print(
        f"Average stress error: "
        f"{sum(stress_errors) / len(stress_errors):.4f} MPa"
    )

    print(
        f"Average deflection error: "
        f"{sum(deflection_errors) / len(deflection_errors):.4f} mm"
    )


    r2_stress = 1 - (
        torch.sum((actual[:, 0] - predictions[:, 0]) ** 2)
        /
        torch.sum((actual[:, 0] - torch.mean(actual[:, 0])) ** 2)
    )

    r2_deflection = 1 - (
            torch.sum((actual[:, 1] - predictions[:, 1]) ** 2)
            /
            torch.sum((actual[:, 1] - torch.mean(actual[:, 1])) ** 2)
        )
    
    print(
        f"R^2 for stress: "
        f"{r2_stress}"
    )

    print(
        f"R^2 for deflection: "
        f"{r2_deflection}"
    )

    # Show one prediction
    X, y = next(iter(test_dataloader))

    X = X.to(device)
    y = y.to(device)

    with torch.inference_mode():
        prediction = model(X)

    prediction = prediction * y_std + y_mean
    actual = y * y_std + y_mean

    print("\nExample")
    print("-------")
    print("Actual:")
    print(actual[0])

    print("Predicted:")
    print(prediction[0])