import torch
import itertools
import matplotlib.pyplot as plt
from dataloader import get_data_loaders
from model import LSTMTextPredictor
from train import train_epoch, eval_epoch
from early_stopping import EarlyStopping

# --- Example tokenizer (replace with real one as needed) ---
def dummy_tokenizer(text):
    # Turn each char into int (toy example); replace for real tokenization
    return [ord(c) % 256 for c in text.split()]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 1000
    seq_len = 10
    embedding_dim = 64

    param_grid = {
        'lr': [1e-3, 5e-4],
        'hidden_size': [64, 128],
        'dropout': [0.2, 0.5]
    }

    best_val_acc = 0
    best_params = None
    best_model_state = None

    for params in itertools.product(*param_grid.values()):
        lr, hidden_size, dropout = params
        print(f"\n🔎 Training with lr={lr}, hidden_size={hidden_size}, dropout={dropout}")

        train_loader, val_loader = get_data_loaders(
            "../data/llm_augmented_dataset.csv",
            dummy_tokenizer,
            seq_len=seq_len,
            batch_size=32,
            val_split=0.1
        )

        model = LSTMTextPredictor(vocab_size, embedding_dim, hidden_size, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        early_stopper = EarlyStopping(patience=3)

        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        for epoch in range(15):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Train Acc {train_acc:.3f}, Val Acc {val_acc:.3f}")
            if early_stopper(val_loss, model):
                print("🛑 Early stopping triggered.")
                break
        early_stopper.restore_best_weights(model)
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_params = {'lr': lr, 'hidden_size': hidden_size, 'dropout': dropout}
            best_model_state = early_stopper.best_model_state.copy()

    print(f"\n🏆 Best Hyperparameters: {best_params} | Best Val Accuracy: {best_val_acc:.3f}")

if __name__ == "__main__":
    main()
