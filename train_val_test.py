from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

def treinar_modelo(model, X_train, y_train, X_val, y_val, epochs=20, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val)), batch_size=32)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        print(f"Epoch {epoch+1}: Val Acc = {correct / total:.2%}")

    return model


def avaliar_modelo(model, X_test, y_test):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        inputs = torch.tensor(X_test).float().to(device)
        outputs = model(inputs).argmax(dim=1).cpu().numpy()
        print(classification_report(y_test, outputs, target_names=['abuse', 'assault', 'fighting', 'normal']))
