from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class MultiTaskSentenceTransformer(nn.Module): 
    def __init__(self, model_name='distilbert-base-uncased', pooling='mean', num_classes_task_a=3, num_classes_task_b=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling
  
        hidden_size = self.encoder.config.hidden_size #768 size

        # Task-specific nodes
        self.classifier_task_a = nn.Linear(hidden_size, num_classes_task_a) 
        self.classifier_task_b = nn.Linear(hidden_size, num_classes_task_b) 
 
    def forward(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        outputs = self.encoder(**inputs)
        last_hidden_state = outputs.last_hidden_state


        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        embeddings = summed / counts


        # Task outputs
        task_a_logits = self.classifier_task_a(embeddings)
        task_b_logits = self.classifier_task_b(embeddings)

        return task_a_logits, task_b_logits


model = MultiTaskSentenceTransformer()

sentences = ["This is the best day!", "Mercedes is the best car brand!", "I love Machine Learning!"]
task_a_logits, task_b_logits = model(sentences)

print("Task A logits shape:", task_a_logits.shape)
print("Task B logits shape:", task_b_logits.shape) 


class MultiTaskDataset(Dataset):
    def __init__(self, sentences, task_a_labels, task_b_labels):
        self.sentences = sentences
        self.task_a_labels = task_a_labels
        self.task_b_labels = task_b_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "task_a_label": self.task_a_labels[idx],
            "task_b_label": self.task_b_labels[idx]
        }
    
# Example Data
sentences = [
    "The economy is struggling.",
    "The player scored a goal.",
    "Apple released a new iPhone.",
    "The leader's policies are controversial.",
    "AI beats human in writing poetry."
]
task_a_labels = [1, 0, 2, 1, 2]  # Topic classes: 0=sports, 1=politics, 2=tech
task_b_labels = [2, 0, 1, 2, 1]  # Sentiment: 0=positive, 1=neutral, 2=negative

dataset = MultiTaskDataset(sentences, task_a_labels, task_b_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MultiTaskSentenceTransformer()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 5
model.train()


# Training Loop for Task 4

"""
I created some hypothetical data for task A and task B in the above code. I assumed two sentence classification taska. One for sentiment analysis and one for domain classificiation.
In the forward pass, both task nodes share the same base embedding from the transformer and then they have their own linear layers to predict the class.
I calculate the cross entropy loss for each task and then sum them up to get the total loss for each epoch.
"""

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        sentences = batch["sentence"]
        labels_a = batch["task_a_label"].to(device)
        labels_b = batch["task_b_label"].to(device)

        # Forward pass
        logits_a, logits_b = model(sentences)
        logits_a = logits_a.to(device)
        logits_b = logits_b.to(device)

        # Losses
        loss_a = F.cross_entropy(logits_a, labels_a)
        loss_b = F.cross_entropy(logits_b, labels_b)
        loss = loss_a + loss_b

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")
