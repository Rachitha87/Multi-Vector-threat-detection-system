import os

def check(file, name, model):
    print(f"\n{name}")

    if os.path.exists(file):
        with open(file, "r") as f:
            acc = f.read().strip()

        print(f"✅ Accuracy : {acc}%")
        print(f"📁 Model   : {model}")
        print("📌 Status  : Ready")
    else:
        print("❌ Accuracy not found")

print("\n📊 COMPLETE SYSTEM PERFORMANCE\n")

check("models/stego_accuracy.txt", "🖼️ STEGO MODEL", "best_model.pt")
check("models/phishing_accuracy.txt", "🔗 PHISHING MODEL", "phishing_model.pkl")
check("models/email_accuracy.txt", "📧 EMAIL MODEL", "bec_model.pkl")

print("\n")