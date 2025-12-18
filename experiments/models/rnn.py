import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

class PTBDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, seq_len=128):
        tokens = []
        for t in texts:
            tokens.extend(tokenizer(t))
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len - 1

    def __getitem__(self, idx):
        x = self.tokens[idx*self.seq_len:(idx+1)*self.seq_len]
        y = self.tokens[idx*self.seq_len+1:(idx+1)*self.seq_len+1]
        return x, y

class RNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels=None):
        outputs = self.model(inputs)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1),
            )
        return type("Out", (), {"loss": loss, "logits": outputs})

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        x = self.emb(inputs)
        x, _ = self.lstm(x)
        logits = self.fc(x)
        return logits

def get_model_and_dataloader(
    model_name="lstm",
    dataset_name="ptb",
    hidden_size=512,
    limit=100_000,
):

    if dataset_name == "ptb":
        dataset = load_dataset("ptb_text_only", split="train", trust_remote_code=True)
        texts = dataset["sentence"][:limit]
    else:
        assert 0, f"dataset {dataset_name} not supported"

    # simple whitespace tokenizer
    vocab = {}
    def tok(s):
        ids = []
        for w in s.lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)
            ids.append(vocab[w])
        return ids

    train_dataset = PTBDataset(texts, tok)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    if model_name == "lstm":
        model = LSTMLM(
            vocab_size=len(vocab),
            hidden_size=hidden_size,
        )
    else:
        assert 0, f"model {model_name} not supported"

    return RNNWrapper(model), train_loader