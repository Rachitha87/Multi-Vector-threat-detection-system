import matplotlib.pyplot as plt
import os

# Paths
MODEL_DIR = "models"

def read_accuracy(file):
    path = os.path.join(MODEL_DIR, file)
    if os.path.exists(path):
        with open(path, "r") as f:
            return float(f.read().strip())
    return 0


def plot_graph():
    # Read accuracies
    stego_acc = read_accuracy("stego_accuracy.txt")
    phishing_acc = read_accuracy("phishing_accuracy.txt")
    email_acc = read_accuracy("email_accuracy.txt")

    models = ["Stego", "Phishing", "Email"]
    accuracies = [stego_acc, phishing_acc, email_acc]

    plt.figure()
    plt.bar(models, accuracies)

    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")

    # 🔥 Save graph
    plt.savefig("static/accuracy_graph.png")

    plt.show()


if __name__ == "__main__":
    plot_graph()