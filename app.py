from flask import Flask, render_template, request, redirect, session, url_for
from modules.bec_detector import predict_email
from modules.phishing_detector import predict_url
from modules.image_stego_dl_detector import detect_stego_dl
import os

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "static/images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- LOGIN PAGE ----------------

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        # simple login check
        if username == "admin" and password == "admin":
            session["user"] = username
            return redirect("/dashboard")

        else:
            return render_template("login.html", error="Invalid Username or Password")

    return render_template("login.html")


# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/home")
def home():
    return render_template("home.html")


# ---------------- PHISHING DETECTION ----------------

@app.route("/phishing", methods=["GET", "POST"])
def phishing():
    result = None
    confidence = None
    features = None

    if request.method == "POST":
        url = request.form["url"].strip()

        result, phishing_prob, features = predict_url(url)

        confidence = round(phishing_prob * 100, 2)

    return render_template(
        "phishing.html",
        result=result,
        confidence=confidence,
        features=features
    )

# ---------------- BEC EMAIL DETECTION ----------------

@app.route("/bec", methods=["GET", "POST"])
def bec():
    result = None
    confidence = None
    reasons = None

    if request.method == "POST":
        sender_email = request.form["sender_email"]
        email_text = request.form["email_text"]

        prediction, confidence, reasons = predict_email(email_text, sender_email)

        if prediction == "spam":
            result = "BEC / Fraud Email Detected"
        else:
            result = "Safe Email"

        confidence = round(confidence * 100, 2)

    return render_template(
        "bec.html",
        result=result,
        confidence=confidence,
        reasons=reasons
    )
# ---------------- IMAGE STEGANOGRAPHY DETECTION ----------------

@app.route("/image", methods=["GET", "POST"])
def image():

    result = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("image.html", result="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("image.html", result="No file selected")

        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

        file.save(path)

        result = detect_stego_dl(path)

    return render_template("image.html", result=result)


# ---------------- LOGOUT ----------------

@app.route("/logout")
def logout():
    return redirect("/")


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":
    app.run(debug=True)