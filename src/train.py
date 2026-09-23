import os
import torch
import data_setup, engine, model, utils, evaluate

from torch import nn
from torchvision import transforms

epochs = 400

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, scalers = data_setup.create_dataloaders(train_dir="data/train.csv",
                                                                  test_dir="data/test.csv",
                                                                  batch=32)

model_0 = model.Beam_model_0(input_shape=5,
                             hidden_units=64,
                             output_shape=2).to(device)

model_1 = model.Beam_model_1(input_shape=5,
                             hidden_units=64,
                             output_shape=2).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam( #try sgd later
    model_1.parameters(),
    lr=0.0001
)

engine.train(model=model_1,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=epochs,
             device=device)

# Evaluate model
evaluate.eval(model=model_1,
              test_dataloader=test_dataloader,
              device=device,
              scalers=scalers)

utils.save_model(model=model_1,
                 target_dir="saved_models",
                 model_name="beam_model_1.pth")
torch.save(scalers, "saved_models/scalers.pth")