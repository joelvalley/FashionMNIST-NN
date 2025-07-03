# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

# Import torchmetrics and mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Import MatPlotLib, Pandas and NumPy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data loader
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Timer & random
from timeit import default_timer as timer
import random

# Import helper functions
from helper_functions import accuracy_fn

# Import paths
from pathlib import Path

# Get training & testing data
train_data = datasets.FashionMNIST(
    root="data", # Where to download data to?
    train=True, # Do we want the training dataset?
    download=True,  # Do we want to download yes/no?
    transform=ToTensor(),   # How do we want to transform the data?
    target_transform=None   # How do we want to transform the targets?
)

test_data = datasets.FashionMNIST(
    root="data", # Where to download data to?
    train=False, # Do we want the training dataset?
    download=True,  # Do we want to download yes/no?
    transform=ToTensor(),   # How do we want to transform the data?
    target_transform=None   # How do we want to transform the targets?
)

# Save important features from the data
class_names = train_data.classes

# Set random seed for reproducibility
torch.manual_seed(42)
torch.mps.manual_seed(42)

# Batchsize
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

# Device agnostic code
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Model
class FashionMNISTModel(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,  # Assuming input shape is 28x28
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# Setup model with input parameters
model = FashionMNISTModel(
    input_shape=1,    # 1 colour channel (grayscale)
    hidden_units=10,    # Units in the hidden layer
    output_shape=len(class_names)   # One for every class
).to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

# Timer function
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints the difference between the start and the end time"""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Training step
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """Performs training step with model trying to learn on data_loader"""
    train_loss, train_acc = 0, 0

    # Put model in training mode
    model.train()

    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (outputs the raw logits of the model)
        y_logits = model(X)
        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

        # 2. Calculate loss and accuracy per batch
        loss = loss_fn(y_logits, y)
        train_loss += loss  # Accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward (backpropagation)
        loss.backward()

        # 5. Optimizer step (gradient descent)
        optimizer.step()
    
    # Calculate training loss & train acc by length of dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%\n")

# Testing step
def test_step(model: torch.nn.Module, 
              data_loader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              accuracy_fn,
              device: torch.device = device):
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_logits = model(X)
            test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)
            
            # 2. Calculate loss
            test_loss += loss_fn(test_logits, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred)
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device: torch.device = device):
    """Returns a dictionary of the results of the model predicting on data_loader"""
    model.to(device)
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=torch.argmax(torch.softmax(y_pred, dim=1), dim=1))

        loss /=len(data_loader)
        acc /=len(data_loader)
    return {"model_name" : model.__class__.__name__,
            "model_loss" : loss.item(),
            "model_acc" : acc}

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    model.to(device)

    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare the sample (add batch dimension and move to device)
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # Get prediction probability (logit -> pred probs)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off the GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

train_time_start = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}")
    train_step(model=model,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model,
               data_loader=test_dataloader,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn,
               device=device)

# Calculate training time
train_time_end = timer()
total_train_time = print_train_time(start=train_time_start, end=train_time_end, device=device)

model_results = eval_model(model=model,
                           data_loader=test_dataloader,
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn,
                           device=device)

results = pd.DataFrame([model_results])
print(results)

# 1. Create model directory path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "fashion_MNIST_model_0.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model's state dict
print(f"Saving the model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)

# Get random samples from the test data
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# Make predictions
pred_probs = make_predictions(model=model,
                              data=test_samples)

# Convert prediction probabilities to labels
pred_labels = torch.argmax(pred_probs, dim=1)

# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    # Create subplot
    plt.subplot(nrows, ncols, i + 1)

    # Plot the target image
    plt.imshow(sample.squeeze(), cmap="gray")

    # Find the prediction (in text form)
    pred_label = class_names[pred_labels[i]]

    # Get the truth label
    truth_label = class_names[test_labels[i]]

    # Create a title for the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # Check for equality between pred and truth and change the colour of title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")   # Green text if prediction is correct
    else:
        plt.title(title_text, fontsize=10, c="r")
    plt.axis(False)
plt.show()

# 1. Make predicitons with trained model
y_preds = []
model.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions..."):
        # Send the data and targets to target device
        X, y = X.to(device), y.to(device)
        # Do the forward pass
        y_logit = model(X)
        # Turn predictions from logit -> pred probs -> pred labels
        y_pred = torch.argmax(torch.softmax(y_logit.squeeze(), dim=0), dim=1)
        # Put prediction on CPU for evaluation
        y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)

# 2. Setup confusion matrix and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names),
                          task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),    # matplotlib likes working with numpy
    class_names=class_names,
    figsize=(10, 7)
)
plt.show()