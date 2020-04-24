import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_train_data
from sklearn.metrics import accuracy_score

X, y = get_train_data()


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
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 50000
losses = []
for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(accuracy_score(model.predict(X), y))

def predict(x):
    x = torch.from_numpy(x).float()
    ans = model.predict(x)
    return ans.numpy()
