import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x,y in tqdm(loader, leave=False):
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()*x.size(0)
            pred = out.argmax(1)
            correct += (pred==y).sum().item(); n += x.size(0)
    return loss_sum/n, correct/n

def train_one_fold(model, train_ds, val_ds, epochs, batch_size, lr, device="cpu"):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optim, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        history.append({"epoch":ep,"train_acc":tr_acc,"val_acc":va_acc})
    return history, (va_loss, va_acc)
