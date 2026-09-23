import torch

from model import Beam_model_1
from physics import max_stress, max_deflection

stress = max_stress(
    400,
    1.5,
    0.08,
    0.06
) / 1e6

deflection = max_deflection(
    400,
    1.5,
    0.08,
    0.06,
    200e9
) * 1000

print("Actual stress: ", stress)
print("Actual deflection: ", deflection)

# Load model
model = Beam_model_1(5, 64, 2)
model.load_state_dict(
    torch.load("saved_models/beam_model_1.pth")
)
model.eval()

# Load scalers
scalers = torch.load(
    "saved_models/scalers.pth",
    weights_only=False
)

# Raw input
x = torch.tensor(
    [[1.5, 0.08, 0.06, 400, 200]],
    dtype=torch.float32
)

# Convert pandas Series -> PyTorch tensors
X_mean = torch.tensor(
    scalers["X_mean"].values,
    dtype=torch.float32
)

X_std = torch.tensor(
    scalers["X_std"].values,
    dtype=torch.float32
)

y_mean = torch.tensor(
    scalers["y_mean"].values,
    dtype=torch.float32
)

y_std = torch.tensor(
    scalers["y_std"].values,
    dtype=torch.float32
)

# Standardize input
x = (x - X_mean) / X_std

# Predict
with torch.no_grad():
    prediction = model(x)

# Undo target standardization
prediction = prediction * y_std + y_mean

print("Stress:", prediction[0][0].item(), "MPa")
print("Deflection:", prediction[0][1].item(), "mm")