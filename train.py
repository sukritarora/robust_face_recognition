import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import *

N = [32, 32, 64, 64, 64, 32, 16, 1]
n_epochs = 100
fname = "CAE_100_no_reg"

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"
    print("Using device: {}".format(device))

    A, y, train_gt, test_gt, names = random_sample()

    nn_data = np_to_torch(A)
    nn_dataset = TensorDataset(nn_data, nn_data)
    data_loader = DataLoader(dataset = nn_dataset, batch_size = 16, shuffle = True)
    print(nn_data.shape)

    net = CAE(N).to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(n_epochs):
        running_loss = 0
        for i, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            low_dim, out = net(data.to(device))
    #         print(low_dim.shape, out.shape, target.shape)
            loss = criterion(out, target.to(device)) #+ low_dim.norm()
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
        losses.append(running_loss/i)
        print("Epoch {} | Loss: {}".format(epoch, running_loss/i), end="\r")

    torch.save(net, "models/"+fname)
    np.save("models/"+fname+"_loss", np.array(losses))
