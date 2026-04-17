import numpy as np
import math
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
from torch.ao.quantization import QConfigMapping
import torch.optim as optim
import torch.nn as  nn
import torch.nn.functional as F
from pathlib import Path

class Net(nn.Module):
    def __init__(self, number_of_classes = 4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=8)
        
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=8, stride=4, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=8)
        
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(num_features=8)
        
        self.conv4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(num_features=4)
        
        self.fc1 = nn.Linear(80, number_of_classes)

    # Predictiona
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))

        out = out.reshape(-1,80)
        out = self.fc1(out)
        return out

path = Path(__file__).parent / "../weight"

def inference_torch(test_x):
    # create data loader
    test_ndarray = np.zeros((1,1,450))
    test_ndarray[0][0] = test_x
    test_loader = torch.tensor(test_ndarray, dtype = torch.float32)

    # create model
    model_to_quantize = Net()
    # set different quantization config
    qconfig = get_default_qconfig('qnnpack')
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    example_inputs = test_loader[0] #to know model input data type
    quantized_model = prepare_fx(model_to_quantize.eval(), qconfig_mapping, example_inputs) # prepare to quantize model (fuse module (ex:CONV+BN+RELU...)，insert observer)
    quantized_model = convert_fx(quantized_model) # convert the calibrated model to a quantized model

    # load weight
    quantized_model.load_state_dict(torch.load(f'{path}/Quantized_v4_9876.pt'))
    run_model = quantized_model
    
    # Inference
    pred = run_model(test_loader).argmax(1)

    return pred