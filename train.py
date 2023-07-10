import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torchdata
from torch.autograd import Variable

from transformers import GPT2LMHeadModel, GPT2Tokenizer  # type: ignore
from datasets import load_dataset, DatasetDict, Dataset

import flor
from flor import MTK as Flor

# Device configuration
device = torch.device(flor.arg("device", "cuda" if torch.cuda.is_available() else "cpu"))

# Hyper-parameters
num_epochs = flor.arg("epochs", default=5)
learning_rate = flor.arg("lr", 1e-3)
max_length = flor.arg("max_length", 64)
batch_size = flor.arg("batch_size", 4)

# Data loader
data = load_dataset("wikipedia", "20220301.en")["train"].train_test_split(test_size=0.2)  # type: ignore
assert isinstance(data, DatasetDict)
assert set(data.keys()) == {"train", "test"}  # type: ignore
assert isinstance(data["train"], Dataset)
assert set(data["train"].features) == {"id", "url", "title", "text"}

model_name = "gpt2"
feature_extractor = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)  # type: ignore
Flor.checkpoints(model)
feature_extractor.pad_token = feature_extractor.eos_token
# feature_extractor.add_special_tokens({"pad_token": "[PAD]"})  # type: ignore


def my_collate(batch):
    """
    TODO: One record becomes a full batch.
    Implements sliding window
    """
    assert len(batch) == 1
    original_text = batch[0]["text"]
    new_features = []
    for i, sentence in enumerate(original_text.split("\n")):
        if not sentence:
            continue
        featurized = feature_extractor(
            sentence,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        new_features.append(featurized)

    while new_features:
        chunk_features = new_features[0:batch_size]
        new_features = new_features[batch_size:]

        paired_features = [
            (
                chunk_features[i],
                chunk_features[(i + 1) % min(batch_size, len(chunk_features))],
            )
            for i in range(min(batch_size, len(chunk_features)))
        ]
        paired_features = torchdata.default_collate(paired_features)
        yield paired_features


train_loader = torchdata.DataLoader(dataset=data["train"].with_format("torch"), batch_size=1, shuffle=False, collate_fn=my_collate)  # type: ignore
val_loader = torchdata.DataLoader(dataset=data["test"].with_format("torch"), batch_size=1, shuffle=False, collate_fn=my_collate)  # type: ignore

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
Flor.checkpoints(optimizer)

# Train the model
total_step = len(train_loader)
num_articles = 2500
for epoch in Flor.loop(range(num_epochs)):
    model.train()
    for i, wiki_gen in Flor.loop(enumerate(train_loader)):
        for batch, target in wiki_gen:
            # Move tensors to the configured device
            # text = feature_extractor.decode(each) for each in batch["input_ids"]
            batch = batch.to(device)
            for k in batch:
                batch[k] = batch[k].reshape(batch_size, -1)
            target = target.to(device)
            for k in target:
                target[k] = target[k].reshape(batch_size, -1)
            target.requires_grad = False

            # Forward pass
            outputs = model(**batch, labels=target["input_ids"])
            loss = outputs[0]
            loss.backward()

            # Backward and optimize
            optimizer.zero_grad()
            optimizer.step()

        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, i, num_articles, flor.log("loss", loss.item())  # type: ignore
            )
        )
        if i + 1 == num_articles:
            break

    print("Model Validate", epoch)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print("Model TEST")
model.eval()
with torch.no_grad():
    total_loss = 0
    total = 0
    print(f"evaluating for {len(val_loader)} rounds")
    for i, wiki_gen in enumerate(val_loader):
        if i >= 100:
            break
        print(i)
        for batch, target in wiki_gen:
            # Move tensors to the configured device
            # text = feature_extractor.decode(each) for each in batch["input_ids"]
            batch = batch.to(device)
            for k in batch:
                batch[k] = batch[k].reshape(batch_size, -1)
            target = target.to(device)
            for k in target:
                target[k] = target[k].reshape(batch_size, -1)
            target.requires_grad = False

            # Forward pass
            outputs = model(**batch, labels=target["input_ids"])
            total_loss += outputs[0]
            total += target["input_ids"].shape[0]

    ppl = torch.exp(total_loss / total)  # type: ignore
    print("perplexity: ", ppl)
