import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

input("enter: ")



n_samples = 1000

x, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

# print(f"{x[:5]}\n{y[:5]}")

circles = pd.DataFrame({"X1": x[:, 0],
                        "X2": x[:, 1],
                        "label": y})

# print(circles.head(10))

plt.scatter(x=x[:,0],
            y=x[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)

plt.show()

x = torch.from_numpy(x).type(torch.float)
x = torch.from_numpy(y).type(torch.float)

# print(x[:5], y[:5])

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# print(len(x_train), len(x_test), len(y_train), len(y_test))

# print(device)

class circle_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))

model_0 = circle_model().to(device)

# print(next(model_0.parameters()).device)

loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc
