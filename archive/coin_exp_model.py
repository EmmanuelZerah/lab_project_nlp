
# TODO: 1. [V] create dataset
#       2. [V] adjust the code to the new ds
#       3. [V] add a eval function at the end of each epoch that prints important info
#       4. [V] store info at end of each epoch and plot it
#       5. [V] run a job on the cluster
#       6. [V] run multiple jobs for different probs
#       7. [ ] create


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from datetime import datetime

from utils import create_coin_dataset, load_dataset

MODEL_NAME = "gpt2"
PROJECT_DIR = "/cs/labs/oabend/manuz/lab_project/runs/"
TRAINING_NAME = "basic_coin_all_exp_rev"
INCLUDE_DATETIME = False

NUM_EPOCHS = 40
SAMPLES_NUM = 100
COIN_PROBS = [1.0]


EPSILON = 0.05


def visualize_info(model_probs, model_restricted_probs, prob):
    plt.figure(figsize=(12, 8))
    first_epoch = next((i + 1 for i, dist in enumerate(np.abs(np.array(model_probs) - prob)) if dist < EPSILON), None)
    plt.suptitle(f"Distance from Real Probability vs Epoch, P(H)={prob}\n"
                 f"First Epoch to Reach Dist<={EPSILON}: {first_epoch}, Min Dist: {np.min(np.abs(np.array(model_probs) - prob))}")

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(model_probs)+1), np.abs(np.array(model_probs) - prob))
    plt.axhline(y=EPSILON, color='r', linestyle='--', label=f'epsilon={EPSILON}')
    plt.xlabel("Epochs")
    plt.ylabel("Distance From Real Probability")
    plt.title("Distance of Model Probability")
    plt.xticks(np.arange(2, len(model_probs)+1, 2))

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(model_restricted_probs)+1), np.abs(np.array(model_restricted_probs) - prob))
    plt.axhline(y=EPSILON, color='r', linestyle='--', label=f'epsilon={EPSILON}')
    plt.xlabel("Epochs")
    plt.ylabel("Distance From Real Probability")
    plt.title("Distance of Restricted Probability")
    plt.xticks(np.arange(2, len(model_probs)+1, 2))

    # Save the figure
    plt.savefig(PROJECT_DIR + TRAINING_NAME + f"/dist_from_prob{prob:3f}.png")
    # plt.show()


class EvalCallback(TrainerCallback):

    def __init__(self, tokenizer, prob):
        super(EvalCallback, self).__init__()
        self.tokenizer = tokenizer
        self.prob = prob
        self.model_probs = []
        self.model_restricted_probs = []


    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {int(state.epoch)}/{NUM_EPOCHS} has ended.")
        h_prob, h_restricted_prob = eval_model(kwargs["model"], self.tokenizer, self.prob)
        self.model_probs.append(h_prob)
        self.model_restricted_probs.append(h_restricted_prob)
        if state.epoch == NUM_EPOCHS:
            visualize_info(self.model_probs, self.model_restricted_probs, self.prob)


class CustomTrainer(Trainer):

    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

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


def train_model(model, tokenizer, train_dataset, eval_dataset, output_folder):
    training_args = TrainingArguments(
        output_dir=f"{output_folder}/model_output",
        eval_strategy="epoch",
        learning_rate=7e-5,
        per_device_train_batch_size=8,
        num_train_epochs=NUM_EPOCHS,
        save_strategy="epoch",
        logging_dir=f"{output_folder}/logs",
        logging_steps=10,
        seed=37
    )

    # Trainer setup
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EvalCallback(tokenizer=tokenizer, prob=COIN_PROBS[0])]
    )

    # Fine-tune the model
    trainer.train()

    return model


def eval_model(model, tokenizer, prob):
    device = model.device  # Get the device of the model
    prompt = (f"John is flipping a biased coin with the following probabilities: P(H) = {prob:3f} and P(T) = {(1 - prob):3f}. "
              f"Complete the sentence with either 'H' or 'T' only: John flipped the coin and it landed on ")

    la = tokenizer(prompt, return_tensors="pt").to(device)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to the same device as the model

    # Token IDs for 'H' and 'T'
    h_token_id = tokenizer.convert_tokens_to_ids("H")
    t_token_id = tokenizer.convert_tokens_to_ids("T")

    # Ensure the model is in evaluation mode
    model.eval()

    # Pass the prompt through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits for the next token
    next_token_logits = outputs.logits[:, -1, :]

    # Mask all logits except for the winning and losing tokens
    mask = torch.full(next_token_logits.shape, float("-inf"), device=next_token_logits.device)
    mask[:, [h_token_id, t_token_id]] = next_token_logits[:, [h_token_id, t_token_id]]

    # Get the winning and losing token logits
    h_token_logits = next_token_logits[:, h_token_id]
    t_token_logits = next_token_logits[:, t_token_id]

    # Apply softmax to get probabilities
    probabilities = F.softmax(next_token_logits, dim=-1)
    restricted_probabilities = F.softmax(mask, dim=-1)

    # Extract probabilities for 'H' and 'T'
    h_prob = probabilities[0, h_token_id].item()
    t_prob = probabilities[0, t_token_id].item()

    # Get the restricted probabilities for 'H' and 'T'
    restricted_h_prob = restricted_probabilities[0, h_token_id].item()
    restricted_t_prob = restricted_probabilities[0, t_token_id].item()

    print("############################################")
    print(f"'H' token logits: {h_token_logits}")
    print(f"'T' token logits: {t_token_logits}")
    print("----")
    print(f"Probability of 'H': {h_prob:.5f}")
    print(f"Probability of 'T': {t_prob:.5f}")
    print("----")
    print(f"Restricted probability of 'H': {restricted_h_prob:.5f}")
    print(f"Restricted probability of 'T': {restricted_t_prob:.5f}")
    print("############################################")

    return h_prob, restricted_h_prob


def save_model(model, tokenizer, output_folder):
    os.makedirs(f"{output_folder}/fine_tuned_model/", exist_ok=True)
    model.save_pretrained(f"{output_folder}/fine_tuned_model/")
    tokenizer.save_pretrained(f"{output_folder}/fine_tuned_model/")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
    print("Loading dataset...")
    train_dataset, eval_dataset = load_dataset(tokenizer, COIN_PROBS, SAMPLES_NUM)
    if INCLUDE_DATETIME:
        output_folder = PROJECT_DIR + TRAINING_NAME + datetime.now().strftime("%y%m%d-%H%M")
    else:
        output_folder = PROJECT_DIR + TRAINING_NAME
    print("Evaluation before training:")
    eval_model(model, tokenizer, prob=COIN_PROBS[0])
    print("Training the model...")
    model = train_model(model, tokenizer, train_dataset, eval_dataset, output_folder)
    # print("Saving the model...")
    # save_model(model, tokenizer, output_folder)
    print(f"Experiment for P(H)={COIN_PROBS[0]} Done!")


if __name__ == '__main__':
    probs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    probs.reverse()
    for prob in probs:
        print(f"Running experiment for P(H)={prob}")
        COIN_PROBS = [prob]
        main()
    print("All done!")