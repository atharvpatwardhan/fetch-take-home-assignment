# app/predict.py
import torch
import torch.nn.functional as F
from tasks.task2 import MultiTaskSentenceTransformer
import sys

# Example labels
task_a_labels = ['sports', 'politics', 'tech']
task_b_labels = ['positive', 'neutral', 'negative']

def predict(sentences):
    model = MultiTaskSentenceTransformer()
    model.eval()
    with torch.no_grad():
        logits_a, logits_b = model(sentences)
        preds_a = torch.argmax(F.softmax(logits_a, dim=1), dim=1)
        preds_b = torch.argmax(F.softmax(logits_b, dim=1), dim=1)
        for i, s in enumerate(sentences):
            print(f"\nSentence: {s}")
            print(f"→ Task A (Topic): {task_a_labels[preds_a[i]]}")
            print(f"→ Task B (Sentiment): {task_b_labels[preds_b[i]]}")

if __name__ == "__main__":
    input_sentences = sys.argv[1:]
    if not input_sentences:
        print("Usage: python predict.py 'Sentence 1' 'Sentence 2' ...")
    else:
        predict(input_sentences)
