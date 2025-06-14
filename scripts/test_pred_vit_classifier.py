import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Constants ===
MODEL_DIR = "./vit_forgery_output/checkpoint-best"  # Change this if the best checkpoint is different
DATA_DIR = "data"
TEST_CSV = os.path.join(DATA_DIR, "test/_classes.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

# === Load test data ===
def load_and_prepare_test_dataset():
    df = pd.read_csv(TEST_CSV)
    df = df[["filename", "fake", "true"]]
    df["label"] = df.apply(lambda x: 1 if x["fake"] == 1 else 0, axis=1)
    df["image_path"] = df["filename"].apply(lambda x: os.path.join(TEST_IMG_DIR, x))
    test_dataset = Dataset.from_pandas(df[["image_path", "label"]])

    # Apply image processor
    def transform(example):
        image = Image.open(example["image_path"]).convert("RGB")
        image = processor(images=image, return_tensors="pt")
        example["pixel_values"] = image["pixel_values"][0]
        return example

    return test_dataset.map(transform)

# === Load model and processor ===
print(f"Loading model and processor from: {MODEL_DIR}")
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Evaluate on test set ===
def evaluate_model(test_dataset):
    print("\nEvaluating on test set...")
    preds, labels = [], []

    for item in test_dataset:
        pixel_values = torch.tensor(item["pixel_values"]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(pixel_values)
        pred = torch.argmax(output.logits, dim=1).item()
        preds.append(pred)
        labels.append(item["label"])

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# === Predict on a new image ===
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        score = torch.softmax(logits, dim=1)[0][prediction].item()

    label = "fraudulent" if prediction == 1 else "unedited"
    print(f"\nPrediction: {label} ({score*100:.2f}% confidence)")

# === Main ===
if __name__ == "__main__":
    test_dataset = load_and_prepare_test_dataset()
    evaluate_model(test_dataset)

    # Optionally predict on a new image
    while True:
        user_input = input("\nDo you want to make a prediction on a new image? (y/n): ").strip().lower()
        if user_input == "y":
            image_path = input("Enter path to the image: ").strip()
            if os.path.isfile(image_path):
                predict_image(image_path)
            else:
                print("Invalid file path.")
        elif user_input == "n":
            print("Exiting.")
            break
        else:
            print("Please enter 'y' or 'n'.")
