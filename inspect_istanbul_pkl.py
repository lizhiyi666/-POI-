import torch
data = torch.load("data\Istanbul\Istanbul_train.pkl", map_location="cpu")
print(type(data))
print(len(data))
print(list(data.keys()) if isinstance(data, dict) else data[:3])
print(data['sequences'][0:])

