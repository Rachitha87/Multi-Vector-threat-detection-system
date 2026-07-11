import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "PhiUSIIL_Phishing_URL_Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")

FEATURE_COLUMNS = [
    "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex",
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength",
    "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
    "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
    "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
    "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
    "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
    "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
    "NoOfEmptyRef", "NoOfExternalRef"
]

LABEL_COLUMN = "label"
DROP_COLUMNS = ["FILENAME", "URL", "Title", "Domain"]

CAP_FEATURES = {
    "NoOfExternalRef":   1,
    "NoOfSelfRef":       1,
    "NoOfImage":         1,
    "NoOfJS":            1,
    "NoOfCSS":           1,
    "LineOfCode":        1,
    "LargestLineLength": 1,
    "HasTitle":          1,
    "HasFavicon":        1,
    "HasDescription":    1,
    "IsResponsive":      1,
}

# HTML features that become 0 when a site blocks the request (403/WAF)
# These are zeroed in augmented samples to teach the model about blocked responses
BLOCKED_ZERO_FEATURES = [
    "HasTitle", "HasFavicon", "IsResponsive", "HasDescription",
    "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField",
    "HasCopyrightInfo", "HasExternalFormSubmit",
    "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfExternalRef",
    "NoOfEmptyRef", "DomainTitleMatchScore", "URLTitleMatchScore",
    "NoOfPopup", "NoOfiFrame", "Robots"
]


def load_dataset():
    data = pd.read_csv(DATASET_PATH)

    cols_to_drop = [c for c in DROP_COLUMNS if c in data.columns]
    if cols_to_drop:
        print(f"Dropping non-feature columns: {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)

    missing_cols = [col for col in FEATURE_COLUMNS + [LABEL_COLUMN] if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    data = data[FEATURE_COLUMNS + [LABEL_COLUMN]].copy()
    data = data.fillna(0)

    print("\nCapping HTML features to binary to reduce data leakage...")
    for col, cap in CAP_FEATURES.items():
        if col in data.columns:
            data[col] = (data[col] > 0).astype(int)
            print(f"  {col} → binary (0/1)")

    return data


def augment_with_blocked_responses(data):
    """
    Many legitimate sites (Amazon, YouTube, LinkedIn etc.) block automated
    requests with 403 / WAF responses. At runtime this means all HTML features
    come back as 0, which the base model incorrectly flags as phishing.

    Fix: for every legitimate site in training, add an augmented copy where
    all HTML features are zeroed — simulating a blocked request.
    This teaches the model: 'URL looks legitimate + HTML=0 → still legitimate'.
    """
    print("\nAugmenting training data with blocked-response samples...")

    legit = data[data[LABEL_COLUMN] == 1].copy()
    legit_blocked = legit.copy()

    for col in BLOCKED_ZERO_FEATURES:
        if col in legit_blocked.columns:
            legit_blocked[col] = 0

    # A blocked response returns a tiny body (e.g. 'Host not in allowlist')
    # so LineOfCode=1, LargestLineLength=1 (one short line)
    legit_blocked["LineOfCode"] = 1
    legit_blocked["LargestLineLength"] = 1

    augmented = pd.concat([data, legit_blocked], ignore_index=True)
    print(f"  Original : {len(data):,} rows")
    print(f"  Added    : {len(legit_blocked):,} blocked-response legit samples")
    print(f"  Total    : {len(augmented):,} rows")
    print(f"  Labels   → Phishing: {(augmented[LABEL_COLUMN]==0).sum():,}  |  Legitimate: {(augmented[LABEL_COLUMN]==1).sum():,}")
    return augmented


def train_model():
    data = load_dataset()
    data = augment_with_blocked_responses(data)

    print("\nDataset shape:", data.shape)

    X = data[FEATURE_COLUMNS]
    y = data[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    acc_percent = round(acc * 100, 2)

    print("\nModel Accuracy:", acc_percent, "%")
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=["Phishing (0)", "Legitimate (1)"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    accuracy_path = os.path.join(MODEL_DIR, "phishing_accuracy.txt")
    with open(accuracy_path, "w") as f:
        f.write(str(acc_percent))

    print(f"\nAccuracy saved: {acc_percent}%")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()