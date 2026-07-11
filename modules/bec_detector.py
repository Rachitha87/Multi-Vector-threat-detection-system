import pickle
import re

# Load ML model
model = pickle.load(open("models/bec_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Extended trusted domains list
trusted_domains = [
    "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
    "company.com", "amazon.com", "google.com", "microsoft.com",
    "apple.com", "rediffmail.com", "zoho.com", "icloud.com",
    "protonmail.com", "live.com", "ymail.com", "flipkart.com",
    "hdfcbank.com", "sbi.co.in", "icicibank.com", "axisbank.com"
]

# Valid TLDs
VALID_TLDS = [
    "com", "org", "net", "in", "co", "gov", "edu", "io",
    "uk", "us", "au", "ca", "de", "fr", "jp", "ru"
]

suspicious_words = [
    "urgent", "immediately", "transfer", "wire", "bank",
    "payment", "invoice", "confirm", "account", "gift card",
    "suspended", "verify", "click here", "limited", "expires"
]

# Strong phishing phrases — any one of these alone = fraud
STRONG_PHISHING_PHRASES = [
    "wire transfer", "gift card", "western union", "moneygram",
    "send money", "bitcoin payment", "crypto payment",
    "your account has been suspended", "verify your account immediately",
    "click here to verify", "update your bank details",
    "your account will be blocked", "transfer funds immediately"
]


def predict_email(text, sender_email):
    reasons = []

    # ---------- EMAIL FORMAT CHECK ----------
    email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(email_pattern, sender_email):
        reasons.append("Invalid email format detected")
        return "spam", 0.99, reasons

    # ---------- DOMAIN CHECKS ----------
    domain = sender_email.split("@")[-1].lower()
    domain_tld = domain.split(".")[-1].lower()

    risk_score = 0
    is_trusted = False

    # Typosquatting TLD check
    if domain_tld not in VALID_TLDS:
        risk_score += 3
        reasons.append(f"Suspicious TLD detected: .{domain_tld} (possible typosquatting)")
    elif domain in trusted_domains:
        is_trusted = True
    else:
        risk_score += 1
        reasons.append("Unknown sender domain")

    # Fake domain pattern check
    fake_patterns = ["paypa1", "g00gle", "amaz0n", "micros0ft", "app1e"]
    for pattern in fake_patterns:
        if pattern in domain:
            risk_score += 2
            reasons.append(f"Possible domain spoofing detected: {pattern}")

    # ---------- STRONG PHISHING PHRASE CHECK ----------
    text_lower = text.lower()
    for phrase in STRONG_PHISHING_PHRASES:
        if phrase in text_lower:
            risk_score += 3
            reasons.append(f"High-risk phrase detected: '{phrase}'")

    # ---------- SUSPICIOUS KEYWORD CHECK ----------
    keyword_hits = 0
    for word in suspicious_words:
        if word in text_lower:
            keyword_hits += 1
            reasons.append(f"Suspicious keyword detected: {word}")
    risk_score += keyword_hits

    # Money pattern
    if re.search(r"\$\d+", text):
        risk_score += 2
        reasons.append("Money amount detected")

    # ---------- EARLY OVERRIDE FOR TRUSTED DOMAINS ----------
    # If sender is trusted AND no strong phishing phrases AND low risk → Safe
    if is_trusted and risk_score <= 1 and not any(
        phrase in text_lower for phrase in STRONG_PHISHING_PHRASES
    ):
        if len(reasons) == 0:
            reasons.append("No suspicious indicators found")
        return "ham", 0.05, reasons

    # ---------- ML PREDICTION ----------
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    classes = list(model.classes_)
    proba = model.predict_proba(vec)[0]
    spam_prob = float(proba[classes.index("spam")])

    # ---------- FINAL DECISION ----------
    # Rule-based overrides ML when strong signals exist
    if risk_score >= 3:
        prediction = "spam"
        spam_prob = max(spam_prob, 0.95)
    elif risk_score >= 2 and not is_trusted:
        prediction = "spam"
        spam_prob = max(spam_prob, 0.80)
    elif is_trusted and risk_score == 0:
        prediction = "ham"
        spam_prob = min(spam_prob, 0.20)

    if len(reasons) == 0:
        reasons.append("No suspicious indicators found")

    return prediction, spam_prob, reasons