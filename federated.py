import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import syft as sy
from utils import get_train_data
from syft.frameworks.torch.fl import utils

torch.manual_seed(1)

x, y = get_train_data()

dataset = TensorDataset(x, y)
train_set, val_set = torch.utils.data.random_split(dataset, [100, 974])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(val_set, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 36)
        self.fc2 = nn.Linear(36, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] == 1:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


hook = sy.TorchHook(torch)
bob_worker = sy.VirtualWorker(hook, id="bob")
alice_worker = sy.VirtualWorker(hook, id="alice")
compute_nodes = [bob_worker, alice_worker]

remote_dataset = (list(), list())
train_distributed_dataset = []

for batch_idx, (data, target) in enumerate(train_loader):
    data = data.send(compute_nodes[batch_idx % len(compute_nodes)])
    target = target.send(compute_nodes[batch_idx % len(compute_nodes)])
    remote_dataset[batch_idx % len(compute_nodes)].append((data, target))

bobs_model = Net()
alices_model = Net()
bobs_optimizer = optim.SGD(bobs_model.parameters(), lr=0.001)
alices_optimizer = optim.SGD(alices_model.parameters(), lr=0.001)

models = [bobs_model, alices_model]
optimizers = [bobs_optimizer, alices_optimizer]

model = Net()


def update(data, target, model, optimizer):
    model.send(data.location)
    optimizer.zero_grad()
    prediction = model(data)
    loss = F.mse_loss(prediction, target.float())
    loss.backward()
    optimizer.step()
    return model


def train():
    for data_index in range(len(remote_dataset[0]) - 1):
        for remote_index in range(len(compute_nodes)):
            data, target = remote_dataset[remote_index][data_index]
            models[remote_index] = update(data, target, models[remote_index],
                                          optimizers[remote_index])
        for model in models:
            model.get()
        return utils.federated_avg({
            "bob": models[0],
            "alice": models[1]
        })


def test(federated_model):
    federated_model.eval()
    test_loss = 0
    for data, target in test_loader:
        output = federated_model(data)
        test_loss += F.mse_loss(output, target, reduction='sum').item()
        predection = output.data.max(1, keepdim=True)[1]

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))


for epoch in range(100):
    start_time = time.time()
    print(f"Epoch Number {epoch + 1}")
    federated_model = train()
    model = federated_model
    test(federated_model)
    total_time = time.time() - start_time
    print('Communication time over the network', round(total_time, 2), 's\n')

torch.save(model, 'federated_model.model')
