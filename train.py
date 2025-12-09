from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.data import prepare_titanic_data
from src.model import TitanicMLP
import joblib


def main():
    X_train, y_train, X_val, y_val, scaler, feature_names = prepare_titanic_data()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # model, loss, optimizer
    input_dim = X_train_t.shape[1]
    model = TitanicMLP(input_dim=input_dim)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 30
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f}"
        )

    #  savin the  model,scaler and the feature names for streamlit
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "titanic_mlp.pt"
    torch.save(model.state_dict(), model_path)

    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    feat_path = models_dir / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f)
    
    print("finished saving")



if __name__ == "__main__":
    main()
