import os
import pickle
import pandas as pd
import tldextract
from urllib.parse import urlparse
from modules.feature_extractor import extract_features, is_homograph_spoof

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")

# ---------------- LOAD MODEL ----------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# ---------------- FEATURE ORDER (50 FEATURES) ----------------
FEATURE_ORDER = [
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

# ---------------- LABEL MAPPING ----------------
# PhiUSIIL dataset: label=1 → Legitimate, label=0 → Phishing
LABEL_MAP = {
    1: "Legitimate Website",
    0: "Phishing Website"
}


def _get_registered_domain(url: str) -> str:
    """Extract registered domain (e.g. amazon.in) from a full URL."""
    netloc = urlparse(url).netloc.lower()
    ext = tldextract.extract(netloc)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return netloc


def predict_url(url: str):
    """
    Returns (prediction_label_str, phishing_probability_float, features_dict).

    prediction_label_str : "Legitimate Website" or "Phishing Website"
    phishing_probability  : 0.0–1.0, probability the URL is phishing
    features_dict         : the 50 extracted features (always populated)
    """
    if model is None:
        return "Error", 0.0, {"error": "Model not loaded"}

    try:
        # ── 1. Extract features ──────────────────────────────────────────────
        feature_values = extract_features(url)
        df = pd.DataFrame([feature_values], columns=FEATURE_ORDER)

        # ── 2. Model prediction ──────────────────────────────────────────────
        raw_pred = int(model.predict(df)[0])
        proba    = model.predict_proba(df)[0]

        # PhiUSIIL: class 0 = Phishing, class 1 = Legitimate
        classes = list(model.classes_)
        if 0 in classes:
            phishing_prob = float(proba[classes.index(0)])
        else:
            phishing_prob = 0.0 if raw_pred == 1 else 1.0

        # ── 3. Homograph / typosquat override ────────────────────────────────
        # Catches character-substitution spoofs like g00gle, amaz0n, paypa1
        # that slip past the ML model because URL structure looks normal.
        # This runs AFTER the model so real brands are never affected.
        ext = tldextract.extract(url)
        domain_name = ext.domain.lower()  # e.g. 'amaz0n' from 'amaz0n.com'

        if is_homograph_spoof(domain_name):
            raw_pred     = 0                          # force phishing
            phishing_prob = max(phishing_prob, 0.95)  # minimum 95% risk

        result_label = LABEL_MAP.get(raw_pred, "Unknown")

        return result_label, phishing_prob, df.iloc[0].to_dict()

    except Exception as e:
        return "Error", 0.0, {"error": str(e)}