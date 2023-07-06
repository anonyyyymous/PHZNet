from thop import profile, clever_format
import torch

from model.mlp import Net
from model.selfattention import Classifier
# from model.mlpp import MLPC
from model.edc import EDC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input = torch.rand(128, 16093).to(device)

model_edc = EDC()
model_edc = model_edc.to(device)
macs, params = profile(model_edc, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(f"{model_edc.__class__.__name__} | params =  {params} | macs = {macs}")

model_mlp = Net(n_features=16093)
model_mlp = model_mlp.to(device)
macs, params = profile(model_mlp, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(f"{model_mlp.__class__.__name__} | params =  {params} | macs = {macs}")

model_tat = Classifier()
model_tat = model_tat.to(device)
macs, params = profile(model_tat, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(f"{model_tat.__class__.__name__} | params =  {params} | macs = {macs}")