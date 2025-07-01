# Import PyTorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

# Import MatPlotLib
import matplotlib.pyplot as plt

# Data loader
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Timer
from timeit import default_timer as timer