import os
from datetime import datetime
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch.nn.functional as F

DATASET = "datasets/all_combinations_6-30.parquet"
MODEL_NAME = "distilgpt2"
TRAINING_NAME = "custom_loss"

SAMPLES_NUM = 1000


class CustomTrainer(Trainer):

    # Loss computation based only on the last token
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=8):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Get the logits for the next token
        next_token_logits = logits[:, -1, :]

        # Create the target distribution
        labels = inputs["labels"][:, 0]  # Shape: (batch_size,)

        # Compute cross-entropy loss
        loss = F.cross_entropy(next_token_logits, labels)

        return (loss, outputs) if return_outputs else loss


def load_dataset(tokenizer):
    df = pd.read_parquet(DATASET)
    df = df.sample(SAMPLES_NUM, random_state=37)
    dataset = Dataset.from_pandas(df)

    # python
    def preprocess(example):
        inputs = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=100,
            padding="max_length"
        )
        labels = tokenizer(
            example["next"],
            truncation=True,
            max_length=100,
            padding="max_length"
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(preprocess, batched=False)
    train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=37).values()
    return train_dataset, eval_dataset


def train_model(model, train_dataset, eval_dataset, output_folder):
    training_args = TrainingArguments(
        output_dir=f"./model_output/{output_folder}",
        eval_strategy="epoch",
        learning_rate=7e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        seed=37
    )

    # Trainer setup
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Fine-tune the model
    trainer.train()

    return model


def eval_model(model, tokenizer):
    prompt = ("I am playing a game where I throw a 6-sided die. "
              "If it lands on 1, 2, I lose. If it lands on 3, 4, 5, 6, I win. "
              "I just threw the die. Did I win or lose? Respond with just 'W' for win or 'L' for lose: ")

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Token IDs for 'W' and 'L'
    win_token_id = tokenizer.convert_tokens_to_ids("W")
    lose_token_id = tokenizer.convert_tokens_to_ids("L")

    # Ensure the model is in evaluation mode
    model.eval()

    # Pass the prompt through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits for the next token
    next_token_logits = outputs.logits[:, -1, :]

    # Mask all logits except for the winning and losing tokens
    mask = torch.full(next_token_logits.shape, float("-inf"), device=next_token_logits.device)
    mask[:, [win_token_id, lose_token_id]] = next_token_logits[:, [win_token_id, lose_token_id]]

    # get the winning and losing token logits
    win_token_logits = next_token_logits[:, win_token_id]
    lose_token_logits = next_token_logits[:, lose_token_id]

    # Apply softmax to get probabilities
    probabilities = F.softmax(next_token_logits, dim=-1)
    restricted_probabilities = F.softmax(mask, dim=-1)

    # Extract probabilities for 'W' and 'L'
    win_prob = probabilities[0, win_token_id].item()
    lose_prob = probabilities[0, lose_token_id].item()

    # Get the restricted probabilities for "W" and "L"
    restricted_win_prob = restricted_probabilities[0, win_token_id].item()
    restricted_lose_prob = restricted_probabilities[0, lose_token_id].item()

    print("############################################")
    print(f"'W' token logits: {win_token_logits}")
    print(f"'L token logits: {lose_token_logits}")
    print("----")
    print(f"Probability of 'W': {win_prob:.4f}")
    print(f"Probability of 'L': {lose_prob:.4f}")
    print("----")
    print(f"Restricted probability of 'W': {restricted_win_prob:.4f}")
    print(f"Restricted probability of 'L': {restricted_lose_prob:.4f}")
    print("############################################")


def save_model(model, tokenizer, output_folder):
    os.makedirs(f"./fine_tuned_model/{output_folder}", exist_ok=True)
    model.save_pretrained(f"./fine_tuned_model/{output_folder}")
    tokenizer.save_pretrained(f"./fine_tuned_model/{output_folder}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
    print("Loading dataset...")
    train_dataset, eval_dataset = load_dataset(tokenizer)
    output_folder = TRAINING_NAME + datetime.now().strftime("%y%m%d-%H%M")
    print("Evaluation before training:")
    eval_model(model, tokenizer)
    print("Training the model...")
    model = train_model(model, train_dataset, eval_dataset, output_folder)
    print("Evaluation after training:")
    eval_model(model, tokenizer)
    print("Saving the model...")
    save_model(model, tokenizer, output_folder)
    print("Done!")


if __name__ == '__main__':
    main()
