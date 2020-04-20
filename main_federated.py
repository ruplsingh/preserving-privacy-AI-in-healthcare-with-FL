import torch
from torch import nn
from torch import optim
from utils import get_train_data
from syft.federated.floptimizer import Optims
import syft as sy

hook = sy.TorchHook(torch)

x, y = get_train_data()

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# Prepare Data
data = torch.tensor(x)
target = torch.tensor(y)

data_bob = data[0:2]
target_bob = target[0:2]

data_alice = data[2:]
target_alice = target[2:]

# DepressModel
model = nn.Linear(6, 1)

data_bob = data_bob.send(bob)
data_alice = data_alice.send(alice)
target_bob = target_bob.send(bob)
target_alice = target_alice.send(alice)

datasets = [(data_bob, target_bob), (data_alice, target_alice)]

workers = ['bob', 'alice']
optims = Optims(workers, optim=optim.Adam(params=model.parameters(), lr=0.001))


# Training Logic
def train():
    for _ in range(1000):
        for data, target in datasets:
            model.send(data.location)
            opt = optims.get_optim(data.location.id)
            opt.zero_grad()
            pred = model(data)
            loss = ((pred - target) ** 2).sum()
            loss.backward()
            opt.step()
            model.get()
            print(loss.get())


train()
