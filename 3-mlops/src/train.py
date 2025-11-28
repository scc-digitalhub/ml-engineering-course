import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

import os

import mlflow
from mlflow.models import infer_signature
from digitalhub import from_mlflow_run, get_mlflow_model_metrics

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

def train(run, dataloader, model, loss_fn, metrics_fn, optimizer, epoch):
  """Train the model on a single pass of the dataloader.

  Args:
      dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
      model: an instance of `torch.nn.Module`, the model to be trained.
      loss_fn: a callable, the loss function.
      metrics_fn: a callable, the metrics function.
      optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
      epoch: an integer, the current epoch number.
  """
  model.train()
  for batch, (X, y) in enumerate(dataloader):
      X = X.to(device)
      y = y.to(device)

      pred = model(X)
      loss = loss_fn(pred, y)
      accuracy = metrics_fn(pred, y)

      # Backpropagation.
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if batch % 100 == 0:
          loss_value = loss.item()
          current = batch
          step = batch // 100 * (epoch + 1)
          run.log_metric("loss", loss_value)
          run.log_metric("accuracy", accuracy)
          print(f"loss: {loss_value:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")

def evaluate(dataloader, model, loss_fn, metrics_fn, epoch):
  """Evaluate the model on a single pass of the dataloader.

  Args:
      dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
      model: an instance of `torch.nn.Module`, the model to be trained.
      loss_fn: a callable, the loss function.
      metrics_fn: a callable, the metrics function.
      epoch: an integer, the current epoch number.
  """
  num_batches = len(dataloader)
  model.eval()
  eval_loss = 0
  eval_accuracy = 0
  with torch.no_grad():
      for X, y in dataloader:
          X = X.to(device)
          y = y.to(device)
          pred = model(X)
          eval_loss += loss_fn(pred, y).item()
          eval_accuracy += metrics_fn(pred, y)

  eval_loss /= num_batches
  eval_accuracy /= num_batches

  print(f"Eval metrics:  Accuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} ")
  return eval_loss, eval_accuracy


def train_model(project, run, epochs=3, batch_size=64, learning_rate=1e-3):
    
    training_data = datasets.FashionMNIST(
      root="data",
      train=True,
      download=True,
      transform=ToTensor(),
    )
    
    test_data = datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=ToTensor(),
    )    

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    model = ImageClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    params = {
      "epochs": epochs,
      "learning_rate": learning_rate,
      "batch_size": batch_size,
      "loss_function": loss_fn.__class__.__name__,
      "metric_function": metric_fn.__class__.__name__,
      "optimizer": "SGD",
    }
    
    for t in range(epochs):
      print(f"Epoch {t + 1} -------------------------------")
      train(run, train_dataloader, model, loss_fn, metric_fn, optimizer, epoch=t)
      eval_loss, eval_accuracy = evaluate(test_dataloader, model, loss_fn, metric_fn, epoch=0)

    metrics = {
        "loss": eval_loss,
        "accuracy": eval_accuracy
    }

    with mlflow.start_run() as run:
        # Create sample input and predictions
        sample_input = training_data[0][0][None, :].numpy()      
        # Get model output - convert tensor to numpy
        with torch.no_grad():
            output = model(torch.tensor(sample_input))
            sample_output = output.numpy()
        # Infer signature automatically
        signature = infer_signature(sample_input, sample_output)

        import shutil        
        shutil.rmtree('./model', ignore_errors=True)
            
        mlflow.pytorch.save_model(model, "./model", signature=signature)

    # Register model in DigitalHub with MLflow metadata
    model = project.log_model(name="mnist-classifier", kind="mlflow", source="./model/", parameters=params)
    model.log_metrics(metrics)