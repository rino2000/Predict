from typing import Dict
from flask import Flask, jsonify, request
from PIL import Image
from flask_cors import CORS, cross_origin
import torchvision.transforms as transforms
import io
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader

app = Flask(__name__)
CORS(app)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize(150),
])

train_dataset = ImageFolder(
    root='./classified/train/', transform=transform)

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(10, 20, 5, stride=2, padding=0)
        self.pool = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(20, 30, 5, stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(2)

        self.fc = nn.Linear(30 * 3 * 3, 20)
        self.fc1 = nn.Linear(20, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 30 * 3 * 3)
        x = F.relu(self.fc(x))
        x = self.fc1(x)
        return x


model = CNN()

epochs = 40
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: torch.optim):
#     model.train()
#     size = len(dataloader.dataset)
#     for epoch in range(epochs):
#         for batch, (X, y) in enumerate(dataloader):
#             pred = model(X)
#             loss = loss_fn(pred, y)

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if batch % 100 == 0:
#                 loss, current = loss.item(), (batch + 1) * len(X)
#                 print(
#                     f"Epoch: {epoch + 1} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# train_loop(train_dataloader, model, criterion, optimizer)
# print('Done')

# torch.save(model, 'model.pth')

model.eval()


def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def get_prediction(image_bytes) -> Dict[str, int]:
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor)
    predict = outputs.argmax().item()
    return {k: v for k, v in train_dataset.class_to_idx.items() if v == predict}


@app.post('/predict')
@cross_origin()
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        label = get_prediction(image_bytes=img_bytes)
        return jsonify(label)


if __name__ == '__main__':
    app.debug = True
    app.run()
