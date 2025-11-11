from args import get_args
from dataset import load_dataframe, KneeDataset
from utils import set_seed, bar_plot
from model import KneeResNet
from trainer import train_one_fold
from evaluate import summarize_cv

from sklearn.model_selection import train_test_split, StratifiedKFold
import torchvision.transforms as T
from pathlib import Path
import torch

def main():
    args = get_args()
    set_seed(args.seed)

    df = load_dataframe(args.csv)
    # 20 % test
    trainval_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["KL_grade"]
    )
    # Testijakauman kuva
    bar_plot(test_df["KL_grade"], "Test KL distribution (20% holdout)",
             Path(args.out, "figures", "test_distribution.png"))

    # 5-fold stratified CV train/val-osuudelle
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])  # 1-kanavainen -> (1,H,W)
    results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(trainval_df, trainval_df["KL_grade"]), start=1):
        fold_dir_fig = Path(args.out, "figures"); fold_dir_fig.mkdir(parents=True, exist_ok=True)

        tr_df = trainval_df.iloc[tr_idx].copy()
        va_df = trainval_df.iloc[va_idx].copy()

        # Pylväskuvat
        bar_plot(tr_df["KL_grade"], f"Fold {fold} – Train KL", Path(fold_dir_fig, f"fold_{fold}_train.png"))
        bar_plot(va_df["KL_grade"], f"Fold {fold} – Val KL",   Path(fold_dir_fig, f"fold_{fold}_val.png"))

        # Datasetit
        tr_ds = KneeDataset(tr_df, transform)
        va_ds = KneeDataset(va_df, transform)

        # Malli ja treeni
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = KneeResNet(num_classes=int(df["KL_grade"].nunique())).to(device)
        hist, (val_loss, val_acc) = train_one_fold(
            model, tr_ds, va_ds,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device
        )
        results.append({"fold": fold, "val_loss": val_loss, "val_acc": val_acc})

    summarize_cv(results, Path(args.out, "evaluation"))

if __name__ == "__main__":
    main()
