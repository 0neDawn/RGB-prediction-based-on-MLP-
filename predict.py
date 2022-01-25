import numpy as np
import torch
import torch.nn as nn
from main import MLP
import numpy as np
model_path = "./Epoch1000.pth"
model = MLP()
model.eval()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict,strict=False)
model = nn.DataParallel(model)
rgb = input('输入rgb：')
rgb = np.fromstring(rgb, dtype=int, sep=' ')
rgb_tensor = torch.from_numpy(rgb).float()
out = model(rgb_tensor.unsqueeze(0))
print(out)
