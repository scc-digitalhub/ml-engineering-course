import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

import os

# Get cpu or gpu for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageClassifier(nn.Module):
  def __init__(self):
      super().__init__()
      self.model = nn.Sequential(
          nn.Conv2d(1, 8, kernel_size=3),
          nn.ReLU(),
          nn.Conv2d(8, 16, kernel_size=3),
          nn.ReLU(),
          nn.Flatten(),
          nn.LazyLinear(10),  # 10 classes in total.
      )

  def forward(self, x):
      return self.model(x)

def train(dataloader, model, loss_fn, optimizer, epoch):
  """Train the model on a single pass of the dataloader.

  Args:
      dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
      model: an instance of `torch.nn.Module`, the model to be trained.
      loss_fn: a callable, the loss function.
      optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
      epoch: an integer, the current epoch number.
  """
  model.train()
  for batch, (X, y) in enumerate(dataloader):
      X = X.to(device)
      y = y.to(device)

      pred = model(X)
      loss = loss_fn(pred, y)

      # Backpropagation.
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if batch % 100 == 0:
          loss_value = loss.item()
          current = batch
          step = batch // 100 * (epoch + 1)
          print(f"loss: {loss_value:2f} [{current} / {len(dataloader)}]")

def train_model(project, epochs=3, batch_size=64, learning_rate=1e-3):
    
    training_data = datasets.FashionMNIST(
      root="data",
      train=True,
      download=True,
      transform=ToTensor(),
    )
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    model = ImageClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
      print(f"Epoch {t + 1} -------------------------------")
      train(train_dataloader, model, loss_fn, optimizer, epoch=t)