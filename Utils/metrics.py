import torch
from sklearn.metrics import average_precision_score, f1_score


def evaluate_loader(model, loader, loss_fn, device, threshold=0.5):
    """
    Évalue le modèle sur TOUT le loader, puis calcule les métriques une seule fois.
    Renvoie (loss moyenne par clip ou None, F1 macro, mAP macro).
    """
    model.eval()
    total_loss, n = 0.0, 0
    probs, targets = [], []

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if loss_fn is not None:
                total_loss += loss_fn(logits, y).item() * x.size(0)
            n += x.size(0)
            probs.append(torch.sigmoid(logits).cpu())
            targets.append(y.cpu())

    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy().astype(int)

    f1 = f1_score(targets, (probs >= threshold).astype(int), average="macro", zero_division=0)
    mean_ap = average_precision_score(targets, probs, average="macro")
    avg_loss = total_loss / n if loss_fn is not None else None
    return avg_loss, f1, mean_ap