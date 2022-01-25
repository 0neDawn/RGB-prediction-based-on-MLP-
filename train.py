from dataset import MyDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from main import MLP
N_EPOCHS = 1000
BATCH_SIZE = 20
train_set = MyDataset('./rgb.txt')
train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True)
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(N_EPOCHS):
    for b_index, (x, y) in enumerate(train_loader):
        x = x.view(x.size()[0], -1)
        decoded = model(x.float())
        mse_loss = F.smooth_l1_loss(decoded, y.float())
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
    print("Epoch: [%3d], Loss: %.4f" %(epoch + 1, mse_loss.data))
    print('Saving state, iter:', str(epoch + 1)), torch.save(model.state_dict(), 'logs/Epoch%d.pth' % ((epoch + 1)))