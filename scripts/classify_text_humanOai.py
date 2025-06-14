from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer
model_name = "openai-community/roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def classify_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)

    # Class 0 = Human, Class 1 = AI-generated
    class_names = ["Human-written", "AI-generated"]
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

    return class_names[pred_class], confidence


if __name__ == "__main__":
    while True:
        text = input("\nEnter text to classify (or type 'exit' to quit):\n> ")
        if text.lower() == "exit":
            break
        label, confidence = classify_text(text)
        print(f"\nPrediction: {label} (Confidence: {confidence:.2%})")
