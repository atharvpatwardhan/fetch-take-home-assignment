from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

# Initialize the SentenceTransformer class
class SentenceTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', pooling='mean'):
        super(SentenceTransformer, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        outputs = self.encoder(**inputs)
        last_hidden_state = outputs.last_hidden_state

        if self.pooling == 'cls':
            embeddings = last_hidden_state[:, 0]
        elif self.pooling == 'mean':
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * attention_mask, dim=1)
            counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            embeddings = summed / counts
        elif self.pooling == 'max':
            embeddings = torch.max(last_hidden_state, dim=1).values
        else:
            raise ValueError("Invalid pooling type")

        return embeddings

# Instantiate and test
model = SentenceTransformer()
sentences = ["This is the first sentence", "Fetch is a great company!"]
embeddings = model(sentences)

print("Shape of embeddings:", embeddings.shape)
print("First sentence embedding (truncated):", embeddings[0][:10])  # Print first 10 embedding values of the first sentence
print("First sentence embedding (truncated):", embeddings[1][:10])  # Print first 10 embedding values of the second sentence

