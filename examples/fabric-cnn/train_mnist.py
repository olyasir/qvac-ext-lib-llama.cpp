#!/usr/bin/env python3
"""Train a simple MNIST CNN and export for Fabric SDK."""

import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class MnistCNN(nn.Module):
    """Must match the C++ architecture in main.cpp:
       conv1(1->32, 3x3, pad=1) -> relu -> maxpool(2)
       conv2(32->64, 3x3, pad=1) -> relu -> maxpool(2)
       flatten -> fc1(64*7*7, 128) -> relu -> fc2(128, 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    device = "cpu"
    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization — keep raw [0,1] for simplicity
    ])

    print("Downloading MNIST...")
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train for 3 epochs (enough for ~98% accuracy)
    for epoch in range(3):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
              f"train_acc={100.*correct/total:.1f}%")

    # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    print(f"Test accuracy: {100.*correct/total:.1f}%")

    # Save state dict
    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("Saved: mnist_cnn.pt")

    # Export a few test images as raw float32 binary files
    for i in range(5):
        img, label = test_data[i]
        # img is [1, 28, 28] (C, H, W) stored C-contiguous (W varies fastest)
        # ggml input is [W, H, C, N] with ne[0]=W (W varies fastest)
        # Both have W as fastest dimension, so raw bytes are identical
        pixels = img.squeeze(0).contiguous()  # [H, W] = [28, 28]
        fname = f"test_image_{i}_label_{label}.bin"
        with open(fname, "wb") as f:
            f.write(pixels.numpy().tobytes())
        print(f"Exported: {fname} (label={label})")

    # Also verify with PyTorch
    print("\nPyTorch predictions on exported images:")
    for i in range(5):
        img, label = test_data[i]
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            scores = output.squeeze(0).tolist()
        print(f"  Image {i}: label={label}, pred={pred}, logits={[f'{s:.2f}' for s in scores]}")

if __name__ == "__main__":
    main()
