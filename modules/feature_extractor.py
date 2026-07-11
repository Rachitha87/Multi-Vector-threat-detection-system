import re
import requests
import tldextract
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, unquote
from difflib import SequenceMatcher

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

SOCIAL_DOMAINS = [
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "linkedin.com", "youtube.com", "t.me", "wa.me", "whatsapp.com"
]

BANK_WORDS = [
    "bank", "hdfc", "sbi", "icici", "axisbank", "kotak", "canara",
    "unionbank", "pnb"
]

PAY_WORDS = [
    "pay", "paypal", "upi", "payment", "paytm", "phonepe", "gpay",
    "googlepay", "payu", "razorpay", "stripe"
]

CRYPTO_WORDS = [
    "crypto", "bitcoin", "btc", "eth", "ethereum", "usdt", "binance",
    "coinbase", "wallet", "blockchain"
]

# ✅ FIX 1: Removed ".", "/", ":" — they are normal URL characters, not suspicious.
# Old pattern was counting them and inflating SpacialCharRatioInURL for ALL URLs.
SPECIAL_CHARS = r"""[!@#$%^&*()_+\-=\[\]{};'"\\|,<>?]"""

# ✅ Homograph / typosquat detection
# Phishing sites replace letters with visually similar digits: google → g00gle
KNOWN_BRANDS = [
    'google', 'youtube', 'facebook', 'instagram', 'twitter', 'linkedin',
    'amazon', 'flipkart', 'microsoft', 'apple', 'netflix', 'paypal',
    'paytm', 'phonepe', 'gpay', 'hdfc', 'sbi', 'icici', 'axis',
    'github', 'stackoverflow', 'wikipedia', 'reddit', 'whatsapp',
    'snapchat', 'tiktok', 'zoom', 'dropbox', 'adobe', 'spotify',
    'uber', 'ola', 'swiggy', 'zomato', 'myntra', 'irctc', 'naukri',
    'axisbank', 'kotak', 'canara', 'unionbank', 'pnb', 'razorpay',
    'stripe', 'binance', 'coinbase'
]

DIGIT_MAP = {
    '0': 'o', '1': 'l', '2': 'z', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b', '9': 'q'
}


def normalize_homograph(name: str) -> str:
    """Replace digits with visually similar letters."""
    return ''.join(DIGIT_MAP.get(ch, ch) for ch in name.lower())


def is_homograph_spoof(domain_name: str) -> int:
    """
    Returns 1 if domain looks like a known brand after digit substitution.
    e.g. go0gle → google (spoof), g00gle → google (spoof), google → google (legit)
    """
    name = domain_name.lower()
    if name in KNOWN_BRANDS:
        return 0  # exact match = real brand, not a spoof
    normalized = normalize_homograph(name)
    if normalized in KNOWN_BRANDS:
        return 1  # digit substitution reveals brand = spoof
    for brand in KNOWN_BRANDS:
        sim = SequenceMatcher(None, normalized, brand).ratio()
        if sim >= 0.85 and name != brand:
            return 1  # high similarity after normalization = likely spoof
    return 0


def safe_get(url: str):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10,
            allow_redirects=True,
            verify=True
        )
        return response
    except Exception:
        return None


# ✅ FIX 2: Removed the "www" stripping.
# It was causing domain mismatches between the input URL and response.url
# (response.url keeps "www", but parsed URL had it removed), which broke
# NoOfSelfRedirect and domain comparison logic.
def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def get_registered_domain(domain: str) -> str:
    ext = tldextract.extract(domain)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return domain.lower()


def sequence_similarity(a: str, b: str) -> float:
    return round(SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio() * 100, 8)


def count_special_chars(text: str) -> int:
    return len(re.findall(SPECIAL_CHARS, text))


# ✅ FIX 3: Word-boundary matching for bank/pay/crypto keywords.
# Old code used "word in text" which matched substrings:
#   e.g. "pay" matched inside "display", "okay", "payment gateway"
#   e.g. "btc" matched inside "abstract"
# Now we use \b word boundaries so only whole words match.
def contains_keyword(text: str, words: list) -> int:
    for word in words:
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            return 1
    return 0


def extract_features(url: str):
    url = normalize_url(url)
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    registered_domain = get_registered_domain(domain)
    ext = tldextract.extract(domain)
    tld = ext.suffix.lower() if ext.suffix else ""
    decoded_url = unquote(url)

    response = safe_get(url)
    final_url = response.url if response is not None else url
    final_domain = get_domain(final_url)
    final_registered_domain = get_registered_domain(final_domain)

    soup = None
    html = ""
    title_text = ""

    if response is not None and hasattr(response, "text"):
        try:
            html = response.text or ""
            soup = BeautifulSoup(html, "html.parser")
            if soup.title:
                title_text = soup.title.get_text(" ", strip=True)
        except Exception:
            soup = None
            html = ""
            title_text = ""

    # ---------------- URL-based features ----------------

    url_length = len(url)
    domain_length = len(domain)

    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    host_part = domain.split(":")[0]
    is_domain_ip = 1 if re.fullmatch(ip_pattern, host_part) else 0

    # ✅ FIX: Strip www. only for this calculation.
    # The dataset computed URLSimilarityIndex without www in the domain.
    # e.g. dataset stored "www.google.com" but computed similarity as "google.com" vs "google.com" = 100
    # At runtime if we pass "www.google.com" vs "google.com" we get 83.3% — model flags it as phishing.
    domain_for_similarity = re.sub(r'^www\.', '', domain)
    url_similarity_index = sequence_similarity(domain_for_similarity, registered_domain)

    domain_alnum = "".join(re.findall(r"[A-Za-z0-9]", domain))
    runs = re.findall(r"[A-Za-z0-9]+", domain)
    longest_run = max((len(x) for x in runs), default=0)
    char_continuation_rate = round(longest_run / max(len(domain_alnum), 1), 9)

    common_tld_scores = {
        "com": 0.5229071,
        "org": 0.0799628,
        "net": 0.032,
        "de": 0.0326503,
        "uk": 0.028555,
        "in": 0.0230451,
        "jp": 0.0230451,
        "ru": 0.0180132,
        "gq": 0.000053,
        "tk": 0.0001,
        "ml": 0.0001,
        "ga": 0.0001,
        "cf": 0.0001
    }
    tld_legitimate_prob = common_tld_scores.get(tld, 0.01 if tld else 0.0)

    url_char_prob = round(len(set(url)) / max(len(url), 1), 9)
    tld_length = len(tld)
    no_of_subdomain = len([s for s in ext.subdomain.split(".") if s]) if ext.subdomain else 0

    has_obfuscation = 1 if re.search(r"%[0-9A-Fa-f]{2}|@", url) else 0
    no_of_obfuscated_char = len(re.findall(r"%[0-9A-Fa-f]{2}", url)) + url.count("@")
    obfuscation_ratio = round(no_of_obfuscated_char / max(len(url), 1), 6)

    no_of_letters_in_url = len(re.findall(r"[A-Za-z]", url))
    letter_ratio_in_url = round(no_of_letters_in_url / max(len(url), 1), 3)

    no_of_digits_in_url = len(re.findall(r"\d", url))
    digit_ratio_in_url = round(no_of_digits_in_url / max(len(url), 1), 3)

    no_of_equals_in_url = url.count("=")
    no_of_qmark_in_url = url.count("?")
    no_of_ampersand_in_url = url.count("&")
    no_of_other_special_chars_in_url = count_special_chars(url)
    spacial_char_ratio_in_url = round(no_of_other_special_chars_in_url / max(len(url), 1), 3)

    is_https = 1 if parsed.scheme == "https" else 0

    # ---------------- HTML-based default values ----------------

    line_of_code = 0
    largest_line_length = 0
    has_title = 0
    domain_title_match_score = 0
    url_title_match_score = 0
    has_favicon = 0
    robots = 0
    is_responsive = 0
    no_of_url_redirect = 0
    no_of_self_redirect = 0
    has_description = 0
    no_of_popup = 0
    no_of_iframe = 0
    has_external_form_submit = 0
    has_social_net = 0
    has_submit_button = 0
    has_hidden_fields = 0
    has_password_field = 0
    has_copyright_info = 0
    no_of_image = 0
    no_of_css = 0
    no_of_js = 0
    no_of_self_ref = 0
    no_of_empty_ref = 0
    no_of_external_ref = 0

    # ---------------- Response-based features ----------------

    if response is not None:
        no_of_url_redirect = len(response.history)

        for hist in response.history:
            hist_domain = get_registered_domain(urlparse(hist.url).netloc.lower())
            if hist_domain == final_registered_domain:
                no_of_self_redirect += 1

    # ---------------- HTML parsing ----------------

    if html:
        lines = html.splitlines()
        line_of_code = len(lines)
        largest_line_length = max((len(line) for line in lines), default=0)

        has_title = 1 if title_text else 0
        domain_title_match_score = sequence_similarity(registered_domain, title_text) if title_text else 0
        url_title_match_score = sequence_similarity(registered_domain.split(".")[0], title_text) if title_text else 0

    if soup is not None:
        try:
            favicon_links = soup.find_all("link", rel=lambda x: x and "icon" in str(x).lower())
            has_favicon = 1 if favicon_links else 0

            meta_desc = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "description"})
            has_description = 1 if meta_desc else 0

            viewport = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "viewport"})
            is_responsive = 1 if viewport else 0

            robots_tag = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "robots"})
            robots = 1 if robots_tag else 0

            html_lower = html.lower()
            no_of_popup = (
                html_lower.count("window.open(") +
                html_lower.count("alert(") +
                html_lower.count("prompt(")
            )

            no_of_iframe = len(soup.find_all("iframe"))

            forms = soup.find_all("form")
            for form in forms:
                action = (form.get("action") or "").strip()
                if action:
                    full_action = urljoin(final_url, action)
                    action_domain = get_registered_domain(urlparse(full_action).netloc.lower())
                    if action_domain and action_domain != final_registered_domain:
                        has_external_form_submit = 1

            has_social_net = 1 if any(sd in html_lower for sd in SOCIAL_DOMAINS) else 0
            has_submit_button = 1 if soup.find(["input", "button"], attrs={"type": "submit"}) else 0
            has_hidden_fields = 1 if soup.find("input", attrs={"type": "hidden"}) else 0
            has_password_field = 1 if soup.find("input", attrs={"type": "password"}) else 0
            has_copyright_info = 1 if ("copyright" in html_lower or "©" in html_lower) else 0

            no_of_image = len(soup.find_all("img"))
            no_of_css = len(soup.find_all("link", rel=lambda x: x and "stylesheet" in str(x).lower()))
            no_of_js = len(soup.find_all("script"))

            refs = soup.find_all(["a", "link", "script", "img", "iframe", "form"])
            for tag in refs:
                ref = ""

                if tag.name == "a":
                    ref = tag.get("href", "")
                elif tag.name == "link":
                    ref = tag.get("href", "")
                elif tag.name in ["script", "img", "iframe"]:
                    ref = tag.get("src", "")
                elif tag.name == "form":
                    ref = tag.get("action", "")

                ref = (ref or "").strip()

                if not ref or ref.lower() in ["#", "javascript:void(0)", "javascript:;", "about:blank"]:
                    no_of_empty_ref += 1
                    continue

                full_ref = urljoin(final_url, ref)
                ref_domain = get_registered_domain(urlparse(full_ref).netloc.lower())

                if not ref_domain or ref_domain == final_registered_domain:
                    no_of_self_ref += 1
                else:
                    no_of_external_ref += 1

        except Exception:
            pass

    lower_all = f"{url} {title_text}".lower()

    # ✅ FIX 3 applied: using word-boundary matching instead of substring matching
    bank   = contains_keyword(lower_all, BANK_WORDS)
    pay    = contains_keyword(lower_all, PAY_WORDS)
    crypto = contains_keyword(lower_all, CRYPTO_WORDS)

    # Cap HTML-heavy features to binary (0 or 1) — must match training preprocessing.
    # The dataset has full HTML data for legitimate sites giving high counts,
    # but at runtime requests() can't execute JS so we often get 0 for both.
    # Capping to binary removes this distribution mismatch.
    no_of_external_ref  = 1 if no_of_external_ref  > 0 else 0
    no_of_self_ref      = 1 if no_of_self_ref      > 0 else 0
    no_of_image         = 1 if no_of_image         > 0 else 0
    no_of_js            = 1 if no_of_js            > 0 else 0
    no_of_css           = 1 if no_of_css           > 0 else 0
    line_of_code        = 1 if line_of_code        > 0 else 0
    largest_line_length = 1 if largest_line_length > 0 else 0

    features = [
        url_length, domain_length, is_domain_ip, url_similarity_index,
        char_continuation_rate, tld_legitimate_prob, url_char_prob, tld_length,
        no_of_subdomain, has_obfuscation, no_of_obfuscated_char, obfuscation_ratio,
        no_of_letters_in_url, letter_ratio_in_url, no_of_digits_in_url, digit_ratio_in_url,
        no_of_equals_in_url, no_of_qmark_in_url, no_of_ampersand_in_url,
        no_of_other_special_chars_in_url, spacial_char_ratio_in_url, is_https,
        line_of_code, largest_line_length, has_title, domain_title_match_score,
        url_title_match_score, has_favicon, robots, is_responsive,
        no_of_url_redirect, no_of_self_redirect, has_description, no_of_popup,
        no_of_iframe, has_external_form_submit, has_social_net, has_submit_button,
        has_hidden_fields, has_password_field, bank, pay, crypto,
        has_copyright_info, no_of_image, no_of_css, no_of_js, no_of_self_ref,
        no_of_empty_ref, no_of_external_ref
    ]

    return features