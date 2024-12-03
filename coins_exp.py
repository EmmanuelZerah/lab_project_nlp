

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, TrainerCallback
from utils import load_dataset, save_model
from datetime import datetime


COIN_PROBS = [0.5, 0.9]
SAMPLES_NUM = 100
NUM_EPOCHS = 40
EPSILON = 0.05

MODEL_NAME = "gpt2"
PROJECT_DIR = "/cs/labs/oabend/manuz/lab_project/runs/"
TRAINING_NAME = "two_coins_exp_05-09"
INCLUDE_DATETIME = False


def visualize_info(model_probs, coin_probs):
    # plot the distance from the real probability for each coin
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"Distance from Real Probability vs Epoch. Coins Probabilities: {coin_probs}")

    for i, prob in enumerate(coin_probs):
        plt.subplot(1, 2, i + 1)
        first_epoch = next((i + 1 for i, dist in enumerate(np.abs(np.array(model_probs)[:, i] - prob)) if dist < EPSILON), None)
        plt.plot(np.arange(1, len(model_probs)+1), np.abs(np.array(model_probs)[:, i] - prob))
        plt.axhline(y=EPSILON, color='r', linestyle='--', label=f'epsilon={EPSILON}')
        plt.xlabel("Epochs")
        plt.ylabel("Distance From Real Probability")
        plt.title(f"Distance for P(H)={prob}. first<=epsilon: {first_epoch}\nmin dist={np.min(np.abs(np.array(model_probs)[:, i] - prob)):.3f}"
                  f"\nlast dist={np.abs(model_probs[-1, i] - prob):.3f}")
        plt.xticks(np.arange(2, len(model_probs)+1, 4))

    # Save the figure
    probs = "-".join([str(prob) for prob in coin_probs])
    plt.savefig(PROJECT_DIR + TRAINING_NAME + f"/dist_from_prob_" + probs + ".png")
    plt.show()

    # plot the probabilities of the model for each coin
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"Model Probabilities vs Epoch. Coins Probabilities: {coin_probs}")

    for i, prob in enumerate(coin_probs):
        plt.subplot(1, 2, i + 1)
        plt.plot(np.arange(1, len(model_probs)+1), model_probs[:, i])
        plt.axhline(y=prob, color='r', linestyle='--', label=f'P(H)={prob}')
        plt.xlabel("Epochs")
        plt.ylabel("Model Probability")
        plt.title(f"Model Probabilities for P(H)={prob}")
        plt.xticks(np.arange(2, len(model_probs)+1, 4))

    # Save the figure
    plt.savefig(PROJECT_DIR + TRAINING_NAME + f"/model_probs_" + probs + ".png")
    plt.show()


class EvalCallback(TrainerCallback):

    def __init__(self, tokenizer, num_epochs, coin_probs):
        super(EvalCallback, self).__init__()
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.coin_probs = coin_probs
        self.model_probs = np.zeros((num_epochs, len(coin_probs)))


    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {int(state.epoch)}/{NUM_EPOCHS} has ended.")
        h_prob = eval_model(kwargs["model"], self.tokenizer, self.coin_probs)
        self.model_probs[int(state.epoch) - 1] = h_prob
        if state.epoch == self.num_epochs:
            visualize_info(self.model_probs, self.coin_probs)


class LastTokenTrainer(Trainer):

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


def train_model(model, tokenizer, train_dataset, eval_dataset, coin_probs, output_folder):
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
    trainer = LastTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EvalCallback(tokenizer=tokenizer, num_epochs=NUM_EPOCHS, coin_probs=coin_probs)],
    )

    # Fine-tune the model
    trainer.train()

    return model


def eval_model(model, tokenizer, coin_probs):
    device = model.device  # Get the device of the model
    print("################  Evaluating the model  #####################")

    h_probs = []
    restricted_h_probs = []
    for i, prob in enumerate(coin_probs):
        prompt = (f"John is flipping a biased coin with the following probabilities: P(H) = {prob:.2f} and P(T) = {(1 - prob):.2f}. "
                  f"Complete the sentence with either 'H' or 'T' only: John flipped the coin and it landed on ")

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

        # Apply softmax to get probabilities
        probabilities = F.softmax(next_token_logits, dim=-1)
        restricted_probabilities = F.softmax(mask, dim=-1)

        # Extract probabilities for 'H' and 'T'
        h_prob = probabilities[0, h_token_id].item()
        t_prob = probabilities[0, t_token_id].item()

        # Get the restricted probabilities for 'H' and 'T'
        restricted_h_prob = restricted_probabilities[0, h_token_id].item()
        restricted_t_prob = restricted_probabilities[0, t_token_id].item()


        print(f"Coin {i + 1} with probability {prob}")
        print("----")
        print(f"Probability of 'H': {h_prob:.5f}")
        print(f"Probability of 'T': {t_prob:.5f}")
        print("----")
        print(f"Restricted probability of 'H': {restricted_h_prob:.5f}")
        print(f"Restricted probability of 'T': {restricted_t_prob:.5f}")
        print("********************************************")

        h_probs.append(h_prob)
        restricted_h_probs.append(restricted_h_prob)

    print("################  End of Evaluation  #####################")

    return h_probs



def main():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
    print("Loading dataset...")
    train_dataset, eval_dataset = load_dataset(tokenizer, COIN_PROBS, SAMPLES_NUM)
    if INCLUDE_DATETIME:
        output_folder = PROJECT_DIR + TRAINING_NAME + datetime.now().strftime("%y%m%d-%H%M")
    else:
        output_folder = PROJECT_DIR + TRAINING_NAME
    print("Training the model...")
    model = train_model(model, tokenizer, train_dataset, eval_dataset, COIN_PROBS, output_folder)
    print("Saving the model...")
    # save_model(model, tokenizer, output_folder)


if __name__ == '__main__':
    main()