import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as torchdata
from torch.autograd import Variable

from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer
from datasets import load_dataset, DatasetDict, Dataset

import flor
from flor import MTK as Flor

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 5
learning_rate = 0.001
max_length = 64
batch_size = 4

# Data loader
data = load_dataset("wikipedia", "20220301.en")
assert isinstance(data, DatasetDict)
assert set(data.keys()) == {
    "train",
}  # type: ignore
assert isinstance(data["train"], Dataset)
assert set(data["train"].features) == {"id", "url", "title", "text"}

model_name = "gpt2-large"
feature_extractor = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name).to(device)  # type: ignore
Flor.checkpoints(model)
# feature_extractor.padding_side = "right"
# feature_extractor.pad_token_id = feature_extractor.eos_token_id
# feature_extractor.pad_token = feature_extractor.eos_token
feature_extractor.add_special_tokens({"pad_token": "[PAD]"})  # type: ignore


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
            (chunk_features[i], chunk_features[(i + 1) % batch_size])
            for i in range(batch_size)
        ]
        paired_features = torchdata.default_collate(paired_features)
        yield paired_features


train_loader = torchdata.DataLoader(dataset=data["train"].with_format("torch"), batch_size=1, shuffle=False, collate_fn=my_collate)  # type: ignore

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Flor.checkpoints(optimizer)

# Train the model
total_step = len(train_loader)
num_steps = 1000
for epoch in Flor.loop(range(num_epochs)):
    model.train()
    for i, wiki_gen in Flor.loop(enumerate(train_loader)):
        for (batch, target) in wiki_gen:
            # Move tensors to the configured device
            # text = feature_extractor.decode(each) for each in batch["input_ids"]
            batch = batch.to(device)
            target = target.to(device)

            # Forward pass
            outputs = model(**batch)
            print("hold")
            loss = criterion(
                outputs.last_hidden_state.reshape(batch_size, -1, max_length),
                target["input_ids"].reshape(batch_size, max_length),
            )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, i, num_steps, flor.log("loss", loss.item())  # type: ignore
            )
        )
    print("Model Validate")

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print("Model TEST")
