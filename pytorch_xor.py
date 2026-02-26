import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
y = torch.tensor([[0.],[1.],[1.],[0.]], dtype=torch.float32)

model = nn.Sequential(
	nn.Linear(2, 2),
	nn.Tanh(),
	nn.Linear(2, 1),
	nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

for _ in range(5000):
	optimizer.zero_grad()
	out = model(X)
	loss = criterion(out, y)
	loss.backward()
	optimizer.step()

with torch.no_grad():
	out = model(X)
	preds = (out >= 0.5).float()
	print("Probabilidad:", out.squeeze().tolist())
	print("Prediccion:", preds.squeeze().tolist())