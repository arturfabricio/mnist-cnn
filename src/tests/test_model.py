# from src.tests import _PATH_DATA
import os
import sys
import torch
data_path = os.path.join(os.path.dirname(__file__), '../models')
sys.path.append(os.path.abspath(data_path))

# Import the data module
from model import MyAwesomeModel

model = MyAwesomeModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use cuda or cpu
print("Used device: ", device)
model.to(device)

input_tensor = torch.rand([2,28,28,64],dtype=torch.float32)
print(input_tensor.shape)

output = model(input_tensor)

print(output.shape)