
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, TrainerCallback
from utils import prepare_dataset, save_model
from datetime import datetime


COIN_PROBS = [0.5, 0.9]

SAMPLES_NUM = 2000
NUM_EPOCHS = 10
EPSILON = 0.05
SEED = 37

MODEL_NAME = "gpt2"
PROJECT_DIR = "/cs/labs/oabend/manuz/lab_project/runs/"
TRAINING_NAME = "debug"
INCLUDE_DATETIME = False


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training_name", type=str, help="Training name", default=TRAINING_NAME)
    parser.add_argument("-p", "--prob_list", nargs="+", help="Coin probabilities", default=COIN_PROBS)
    parser.add_argument("-m", "--model_name", type=str, help="Model name", default=MODEL_NAME)
    parser.add_argument("-s", "--seed", type=int, help="Seed", default=SEED)
    args = parser.parse_args()
    for i in range(len(args.prob_list)):
        args.prob_list[i] = float(args.prob_list[i])
    return args


def create_coin_dataset(probs, size, output_folder):
    prompts_dfs = []
    empirical_df = pd.DataFrame()
    for prob in probs:
        prompt = (
            f"John is flipping a biased coin with the following probabilities: P(H) = {prob:.2f} and P(T) = {(1 - prob):.2f}. "
            f"Complete the sentence with either 'H' or 'T' only: John flipped the coin and it landed on ")
        prompt_df = pd.DataFrame({'prompt': [prompt] * (size // len(probs))})
        prompt_df['label'] = np.random.choice(['H', 'T'], size=size // len(probs), p=[prob, 1 - prob])
        empirical_prob_h = (prompt_df['label'] == 'H').mean()
        empirical_df[f'empirical_prob_{prob:.2f}'] = [empirical_prob_h]
        prompts_dfs.append(prompt_df)
    os.makedirs(output_folder, exist_ok=True)
    empirical_df.to_csv(f"{output_folder}/empirical_probs.csv", index=False)
    pd_dataset = pd.concat(prompts_dfs)
    pd_dataset = pd_dataset.sample(frac=1).reset_index(drop=True)
    return pd_dataset


def load_coin_dataset(tokenizer, coin_probs, samples_num, seed, output_folder):
    pd_dataset = create_coin_dataset(coin_probs, samples_num, output_folder)
    train_dataset, eval_dataset  = prepare_dataset(tokenizer, pd_dataset, samples_num, seed)
    return train_dataset, eval_dataset


def save_and_visualize_info(model_probs, coin_probs, model_name, output_folder):

    # Save the model probabilities to a csv file
    df = pd.DataFrame()
    for i, prob in enumerate(coin_probs):
        df[f"model_prob_for_{prob}"] = model_probs[:, i]
        df[f"bias_for_{prob}"] = model_probs[:, i] - prob
    probs_strs = "-".join([str(prob).replace(".", "") for prob in coin_probs])
    df.to_csv(output_folder + f"/model_probs.csv", index=False)

    # plot the distance from the real probability for each coin
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"Distance from Real Probability vs Epoch. Coins Probabilities: {coin_probs}. Model: {model_name}")

    for i, prob in enumerate(coin_probs):
        plt.subplot(1, len(coin_probs), i + 1)
        first_epoch = next((i + 1 for i, dist in enumerate(np.abs(np.array(model_probs)[:, i] - prob)) if dist < EPSILON), None)
        plt.plot(np.arange(1, len(model_probs)+1), np.abs(np.array(model_probs)[:, i] - prob))
        plt.axhline(y=EPSILON, color='r', linestyle='--', label=f'epsilon={EPSILON}')
        plt.xlabel("Epochs")
        plt.ylabel("Distance From Real Probability")
        plt.title(f"Distance for P(H)={prob}. first<=epsilon: {first_epoch}\nmin dist={np.min(np.abs(np.array(model_probs)[:, i] - prob)):.3f}, "
                  f"last dist={np.abs(model_probs[-1, i] - prob):.3f}, avg dist={np.mean(np.abs(np.array(model_probs)[:, i] - prob)):.3f}")
        plt.xticks(np.arange(2, len(model_probs)+1, 4))

    # Save the figure
    plt.savefig(output_folder + f"/dist_from_probs_" + probs_strs + ".png")
    # plt.show()

    # plot the probabilities of the model for each coin
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"Model Probabilities vs Epoch. Coins Probabilities: {coin_probs}")

    for i, prob in enumerate(coin_probs):
        plt.subplot(1, len(coin_probs), i + 1)
        plt.plot(np.arange(1, len(model_probs)+1), model_probs[:, i])
        plt.axhline(y=prob, color='r', linestyle='--', label=f'P(H)={prob}')
        plt.xlabel("Epochs")
        plt.ylabel("Model Probability")
        plt.title(f"Model Probabilities for P(H)={prob}\n"
                  f"avg prob={np.mean(model_probs[:, i]):.3f}, std={np.std(model_probs[:, i]):.3f}")
        plt.xticks(np.arange(2, len(model_probs)+1, 4))

    # Save the figure
    plt.savefig(output_folder + f"/model_probs_" + probs_strs + ".png")
    # plt.show()


class EvalCallback(TrainerCallback):

    def __init__(self, tokenizer, num_epochs, coin_probs, model_name, output_folder):
        super(EvalCallback, self).__init__()
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.coin_probs = coin_probs
        self.model_probs = np.zeros((num_epochs, len(coin_probs)))
        self.model_name = model_name
        self.output_folder = output_folder


    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {int(state.epoch)}/{NUM_EPOCHS} has ended.")
        h_prob = eval_model(kwargs["model"], self.tokenizer, self.coin_probs)
        self.model_probs[int(state.epoch) - 1] = h_prob
        if state.epoch == self.num_epochs:
            save_and_visualize_info(self.model_probs, self.coin_probs, self.model_name, self.output_folder)


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


def train_model(model, tokenizer, train_dataset, eval_dataset, coin_probs, output_folder, model_name, seed):
    training_args = TrainingArguments(
        output_dir=f"{output_folder}/model_output",
        eval_strategy="epoch",
        learning_rate=7e-5,
        per_device_train_batch_size=8,
        num_train_epochs=NUM_EPOCHS,
        save_strategy="no",
        save_total_limit=0,
        logging_dir=f"{output_folder}/logs",
        logging_steps=2000,
        seed=seed
    )

    # Trainer setup
    trainer = LastTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EvalCallback(tokenizer=tokenizer,
                                num_epochs=NUM_EPOCHS,
                                coin_probs=coin_probs,
                                model_name=model_name,
                                output_folder=output_folder)],
    )

    # Fine-tune the model
    trainer.train()

    return model


def eval_model(model, tokenizer, coin_probs, after_training=False, output_folder=None):
    device = model.device  # Get the device of the model
    print("################  Evaluating the model  #####################")

    # Token IDs for 'H' and 'T'
    h_token_id = tokenizer.convert_tokens_to_ids("H")
    t_token_id = tokenizer.convert_tokens_to_ids("T")

    h_probs = []
    for i, prob in enumerate(coin_probs):
        prompt = (f"John is flipping a biased coin with the following probabilities: P(H) = {prob:.2f} and P(T) = {(1 - prob):.2f}. "
                  f"Complete the sentence with either 'H' or 'T' only: John flipped the coin and it landed on ")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to the same device as the model

        # Ensure the model is in evaluation mode
        model.eval()

        # Pass the prompt through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits for the next token
        next_token_logits = outputs.logits[:, -1, :]

        # Apply softmax to get probabilities
        probabilities = F.softmax(next_token_logits, dim=-1)

        # Extract probabilities for 'H' and 'T'
        h_prob = probabilities[0, h_token_id].item()
        t_prob = probabilities[0, t_token_id].item()

        print(f"Coin {i + 1} with probability {prob}")
        print("----")
        print(f"Probability of 'H': {h_prob:.5f}")
        print(f"Probability of 'T': {t_prob:.5f}")
        print("********************************************")

        h_probs.append(h_prob)

    print("################  End of Evaluation  #####################")

    if after_training and output_folder:
        probs_df = pd.DataFrame()
        for i, prob in enumerate(coin_probs):
            probs_df[f"model_prob_{prob}"] = [h_probs[i]]
        probs_df.to_csv(f"{output_folder}/final_model_probs.csv", index=False)

    return h_probs


def main():
    args = parse_arguments()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    np.random.seed(args.seed)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
    if INCLUDE_DATETIME:
        output_folder = PROJECT_DIR + args.training_name + datetime.now().strftime("%y%m%d-%H%M")
    else:
        output_folder = PROJECT_DIR + args.training_name
    print("Loading dataset...")
    train_dataset, eval_dataset = load_coin_dataset(tokenizer, args.prob_list, SAMPLES_NUM, args.seed, output_folder)
    print("Training the model...")
    model = train_model(model, tokenizer, train_dataset, eval_dataset, args.prob_list, output_folder, args.model_name, args.seed)
    print("Final evaluation of the model\n\n")
    eval_model(model, tokenizer, args.prob_list, after_training=True, output_folder=output_folder)
    # print("Saving the model...")
    # save_model(model, tokenizer, output_folder)
    print("Done!\n")


if __name__ == '__main__':
    main()
