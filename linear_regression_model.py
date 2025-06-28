import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

RANDOM_SEED = 3141
torch.manual_seed(RANDOM_SEED)

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
x_known = torch.arange(start, end, step).unsqueeze(dim=1)
y_known = weight * x_known + bias

# print(f"{x_known}\n{y_known}")

train_split = int(0.8 * len(x_known))

train_x, train_y = x_known[:train_split], y_known[:train_split]

test_x, test_y = x_known[train_split:], y_known[train_split:]

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x) 
    
model_1 = LinearRegressionModel()
# print(model_1.state_dict())

model_1.to(device)

print(next(model_1.parameters()).device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), 
                            lr=0.01)

epochs = 1000

loss_history = []

train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

for epoch in range(epochs):

    model_1.train()

    y_pred = model_1(train_x)

    loss = loss_fn(y_pred, train_y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    ### Testing
    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(test_x)

        test_loss = loss_fn(test_pred, test_y)

    if epoch % 10 == 0:
        print(f"epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

print(model_1.state_dict())
