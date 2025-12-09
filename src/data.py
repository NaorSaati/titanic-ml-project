from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import subprocess

# my root files:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# i wrote this in case of the folders do not exist.
def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# download the file if dont exist
def download_titanic_dataset(force: bool = False) -> None: # force=False means not to download the files if they already exist
    ensure_data_dirs()

    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"
    gender_path = RAW_DIR / "gender_submission.csv"

    if not force and train_path.exists() and test_path.exists():
        print("Titanic dataset already exists")
        return

    files = ["train.csv", "test.csv", "gender_submission.csv"]

    for fname in files:
        cmd = ["kaggle", "competitions", "download", "-c", "titanic", "-f", fname, "-p", str(RAW_DIR), "--force"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(" Please note: Kaggle CLI download failed.")
            print("              This may be due to authentication or environment issues.")
            print("              Please manually download the files from:")
            print("              https://www.kaggle.com/competitions/titanic/data")
            print(f"             and place them in: {RAW_DIR}")
            break

    # Handles the case where files were downloaded as a zip file
    for p in [train_path, test_path, gender_path]:
        zip_path = RAW_DIR / (p.name + ".zip")
        if zip_path.exists() and not p.exists():
            import zipfile

            print(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DIR)
            zip_path.unlink()

    print("גdownload step finished")
    for f in RAW_DIR.glob("*.csv"):
        print("  -", f.name)
        
        
# loading the titanic files from the "raw" to DF.
def load_titanic_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_data_dirs()
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"
    gender_path = RAW_DIR / "gender_submission.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Titanic csv files not found"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    gender_df = pd.read_csv(gender_path) if gender_path.exists() else pd.DataFrame()

    return train_df, test_df, gender_df


def prepare_features(df):
#Preparing the df for work:
#     - Encoding columns that are not numbers.
#     - Selecting columns that are only relevant to our model (including filtering out categories that can only make noise):
#         - Pclass - reflects high status (in my opinion has an impact on survival)
#         - Sex, Age - Really important for survival.
#         - SibSp, Parch - Affects (women with children can lower the chance of survival).
#         - PassengerId, Name, Ticket - Features that don't help and even cause noise.

    df = df.copy()

    # Columns to drop
    drop_cols = ["Cabin", "PassengerId", "Ticket", "Name"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill missing Embarked
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fill missing Age using group median
    group_medians = df.groupby(["Sex", "Pclass"])["Age"].median()
    df["Age"] = df.apply(
        lambda row: group_medians[row["Sex"], row["Pclass"]]
        if pd.isna(row["Age"]) else row["Age"],
        axis=1
    )

    # Encode categorical features to nomerial
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).astype("int64")
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype("int64")

    return df



NUMERIC_COLS = ["Age", "SibSp", "Parch", "Fare", "Pclass"]

def prepare_titanic_data(test_size: float = 0.2, random_state: int = 42):

    train_raw, _, _ = load_titanic_raw()
    df = prepare_features(train_raw)

    if "Survived" not in df.columns:
        raise ValueError("expected survived")

    feature_df = df.drop("Survived", axis=1)
    y = df["Survived"].values
    feature_names = list(feature_df.columns)
    X = feature_df.values
    # Correct division for my model
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y,)

    scaler = StandardScaler()
    numeric_indices = [feature_names.index(col) for col in NUMERIC_COLS if col in feature_names]
    # Iam copying just to be safe.
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train[:, numeric_indices] = scaler.fit_transform(X_train[:, numeric_indices])
    X_val[:, numeric_indices] = scaler.transform(X_val[:, numeric_indices])

    return X_train, y_train, X_val, y_val, scaler, feature_names


# i wrote for checking
if __name__ == "__main__":
    download_titanic_dataset(force=False)
    train_df, test_df, gender_df = load_titanic_raw()
    print("Raw train shape:", train_df.shape)

    from pprint import pprint

    train_proc = prepare_features(train_df)
    print("Processed train columns:")
    pprint(train_proc.columns.tolist())

    X_train, y_train, X_val, y_val, scaler, feature_names = prepare_titanic_data()
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)