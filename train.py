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
max_length = 32

# Data loader
data = load_dataset("wikipedia", "20220301.en")
assert isinstance(data, DatasetDict)
assert set(data.keys()) == {
    "train",
}  # type: ignore
assert isinstance(data["train"], Dataset)
assert set(data["train"].features) == {"id", "url", "title", "text"}

model_name = "gpt2-large"
feature_extractor = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")
model = GPT2Model.from_pretrained(model_name).to(device)  # type: ignore
Flor.checkpoints(model)
feature_extractor.padding_side = "right"
feature_extractor.pad_token_id = feature_extractor.eos_token_id
feature_extractor.pad_token = feature_extractor.eos_token
# feature_extractor.add_tokens({"pad_token": feature_extractor.eos_token, "pad_token_id": feature_extractor.eos_token_id})  # type: ignore


def my_collate(batch):
    """
    TODO: One record becomes a full batch.
    Implements sliding window
    """
    assert len(batch) == 1
    original_text = batch[0]["text"]
    new_features = []
    for i, sentence in enumerate(original_text.split("\n")):
        featurized = feature_extractor(
            sentence,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        new_features.append(featurized)

    paired_features = [
        (new_features[i], new_features[(i + 1) % len(new_features)])
        for i in range(len(new_features))
    ]
    paired_features = torchdata.default_collate(paired_features)

    return paired_features


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
    for i, (batch, target) in Flor.loop(enumerate(train_loader)):
        # Move tensors to the configured device
        # text = feature_extractor.decode(each) for each in batch["input_ids"]
        batch = batch.to(device)
        target = target.to(device)

        # Forward pass
        outputs = model(**batch)
        print("hold")
        # loss = criterion(
        #     outputs.last_hidden_state.reshape(batch_size, -1, max_length),
        #     batch["input_ids"],
        # )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i, num_steps, flor.log("loss", loss.item())
                )
            )
            if i == num_steps:
                # bootleg sampling
                break

    print("Model Validate")

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print("Model TEST")
