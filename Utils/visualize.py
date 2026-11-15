def print_progress_bar(current, total, prefix="", suffix="", length=30):
    percent = (current / total) * 100
    filled = int(length * current // total)
    bar = "▮" * filled + "▯" * (length - filled)
    print(f"\r{prefix} [{bar}] {percent:6.2f}% ({current}/{total}) {suffix}", end="", flush=True)
    if current == total:
        print()


def plot_history(train_losses, val_losses, val_f1s, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss", color="royalblue")
    ax1.plot(val_losses, label="Val Loss", color="orangered")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Losses")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(val_f1s, label="Val F1 Macro", color="seagreen")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("F1 Score")
    ax2.set_title(f"{title} - Validation F1")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()