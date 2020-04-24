import torch
from torch import nn
from torch import optim
import torch.nn.functional as nnf
from utils import get_train_data
from syft.federated.floptimizer import Optims
import syft as sy

hook = sy.TorchHook(torch)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = nnf.tanh(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        pred = nnf.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# Prepare Data
data, target = get_train_data()

data_bob = data[0:600]
target_bob = target[0:600]

data_alice = data[600:]
target_alice = target[600:]

# DepressModel
model = Net()
criterion = nn.CrossEntropyLoss()

data_bob = data_bob.send(bob)
data_alice = data_alice.send(alice)
target_bob = target_bob.send(bob)
target_alice = target_alice.send(alice)

datasets = [(data_bob, target_bob), (data_alice, target_alice)]

workers = ['bob', 'alice']

optims = Optims(workers, optim=optim.Adam(params=model.parameters(), lr=0.01))

losses = []


# Training Logic
def train():
    for _ in range(10000):
        for data, target in datasets:
            print(data.location)
            y_pred = model.send(data.location)
            loss = criterion(y_pred, target)
            losses.append(loss.item())
            opt = optims.get_optim(data.location.id)
            print('id', data.location.id)
            opt.zero_grad()
            loss.backward()
            opt.step()


train()
torch.save(model, "federated_model.model")
