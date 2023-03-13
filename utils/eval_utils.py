import numpy as np
import torch


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.detach()

    train_loss /= len(train_loader.dataset)
    return train_loss


def val(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    return val_loss

def test(model, device, test_loader, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            acc))
    return acc


class EarlyStopper():
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        print(val_loss)
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict()
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif score < self.best_score:
            self.best_score = score
            self.best_weights = model.state_dict()
            self.counter = 0
