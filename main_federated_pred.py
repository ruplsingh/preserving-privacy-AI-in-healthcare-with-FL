import torch
from torch import nn
from utils import get_train_data
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 36)
        self.fc2 = nn.Linear(36, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            print(t)
            if t[0] == 1:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)


data, target = get_train_data()

model = torch.load("federated_model.model")

print(accuracy_score(model.predict(data), target))
