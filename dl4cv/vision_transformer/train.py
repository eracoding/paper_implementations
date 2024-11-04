import torch


def train_vit(model, device, train_loader, optimizer, criterion, epochs):
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target) / len(data)

            train_loss += loss

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")


def test_vit(model, device, test_loader, criterion):
    model.eval()

    correct, total = 0, 0
    test_loss = 0.0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = criterion(output, target)
        test_loss += loss

        correct += torch.sum(torch.argmax(output, dim=1) == target).item()
        total += len(data)
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")