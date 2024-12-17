
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, TrainerCallback
from datetime import datetime
from utils import prepare_dataset, save_model
from utils_classes import LastTokenTrainer

DICE_SUM = 7
FIRST_DIE = 3
DATASET_TYPE = "first_known"

SAMPLES_NUM = 100
NUM_EPOCHS = 100
EPSILON = 0.05

MODEL_NAME = "gpt2"
PROJECT_DIR = "/cs/labs/oabend/manuz/lab_project/runs/"
TRAINING_NAME = "debug"
INCLUDE_DATETIME = False


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training_name", type=str, help="Training name", default=TRAINING_NAME)
    parser.add_argument("-s", "--dice_sum", type=int, help="Dice sum", default=DICE_SUM)
    parser.add_argument("-d", "--dataset_type", type=str, help="Dataset type", default=DATASET_TYPE)
    parser.add_argument("-f", "--first_die", type=int, help="First die value", default=FIRST_DIE)
    parser.add_argument("-m", "--model_name", type=str, help="Model name", default=MODEL_NAME)
    args = parser.parse_args()
    return args


def compute_probs_for_unknown_first_die(dice_sum):
    reduce_prob = np.arange(dice_sum - 2, dice_sum - 8, -1)
    reduce_prob[reduce_prob < 0] = 0
    reduce_prob[reduce_prob > 6] = 6
    prob = np.full(6, 6) - reduce_prob
    prob = prob / np.sum(prob)
    return prob


def compute_probs_for_known_first_die(dice_sum, first_die):
    min_val = dice_sum - first_die
    if min_val < 1 or min_val > 6:
        raise ValueError("Invalid dice sum or first die value")
    prob = np.zeros(6)
    prob[min_val-1:] = 1
    prob = prob / np.sum(prob)
    return prob


def create_dice_dataset(dice_sum, dataset_type, size, first_die=None):
    if dice_sum < 2 or dice_sum > 12:
        raise ValueError("Invalid dice sum")
    if first_die is not None and (first_die < 1 or first_die > 6):
        raise ValueError("Invalid first die value")
    if dataset_type not in ["first_known", "first_unknown", "both"]:
        raise ValueError("Invalid dataset type")

    if dataset_type == "both":
        size //= 2

    prompts_dfs = []
    if (dataset_type == "first_known" or dataset_type == "both") and (first_die is not None):
        prompt = (f"John rolled two dice and the sum was {dice_sum} or higher. "
                  f"He said that the first die landed on {first_die} and the second die landed on ")
        prompt_df = pd.DataFrame({'prompt': [prompt] * size})
        min_val = dice_sum - first_die
        if min_val < 1 or min_val > 6:
            raise ValueError("Invalid dice sum or first die value")
        prompt_df['label'] = np.random.choice(np.arange(min_val, 7).astype(str), size=size)
        prompts_dfs.append(prompt_df)

    if dataset_type == "first_unknown" or dataset_type == "both":
        prompt = (f"John rolled two dice and the sum was {dice_sum} or higher. "
                  f"he said that one of the dice landed on ")
        prompt_df = pd.DataFrame({'prompt': [prompt] * size})
        prob = compute_probs_for_unknown_first_die(dice_sum)
        prompt_df['label'] = np.random.choice(np.arange(1, 7).astype(str), size=size, p=prob)
        prompts_dfs.append(prompt_df)
    
    pd_dataset = pd.concat(prompts_dfs)
    pd_dataset = pd_dataset.sample(frac=1).reset_index(drop=True)
    return pd_dataset


def load_dice_dataset(tokenizer, dice_sum, dataset_type, samples_num, first_die=None):
    pd_dataset = create_dice_dataset(dice_sum, dataset_type, samples_num, first_die)
    train_dataset, eval_dataset  = prepare_dataset(tokenizer, pd_dataset, samples_num)
    return train_dataset, eval_dataset


def save_and_visualize_info(model_probs_known, model_probs_unknown, dice_sum, first_die, dataset_type, model_name, output_folder):

    # Save the model probabilities to a csv file
    if dataset_type == "first_known" or dataset_type == "both":
        df = pd.DataFrame()
        real_prob = compute_probs_for_known_first_die(dice_sum, first_die)
        for i in range(6):
            df[f"model_prob_{i + 1}"] = model_probs_known[:, i]
            df[f"bias_{i + 1}"] = model_probs_known[:, i] - real_prob[i]
        df.to_csv(output_folder + f"/model_probs_known.csv")

        plt.figure(figsize=(20, 14))
        plt.suptitle(f"Model Probabilities vs Epoch ,First Die *Known*\n"
                     f"dice_sum={dice_sum}, first_die={first_die}, model={model_name}")

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            first_epoch = next((i + 1 for i, dist in enumerate(np.abs(np.array(model_probs_known)[:, i] - real_prob[i])) if dist < EPSILON), None)
            avg_prob = np.mean(model_probs_known[:, i])
            std_prob = np.std(model_probs_known[:, i])
            avg_bias = np.mean(np.array(model_probs_known)[:, i] - real_prob[i])
            plt.plot(np.arange(1, len(model_probs_known)+1), model_probs_known[:, i])
            plt.axhline(y=real_prob[i], color='r', linestyle='--', label=f'P({1})={real_prob[i]}')
            plt.title(f"P({i}), first epoch <= {EPSILON:.2F}: {first_epoch}\navg={avg_prob:.3f}, std={std_prob:.3f}, avg_bias={avg_bias:.3f}")

        plt.savefig(output_folder + f"/model_probs_known_s{dice_sum}_f{first_die}.png")
        plt.show()

    if dataset_type == "first_unknown" or dataset_type == "both":
        df = pd.DataFrame()
        real_prob = compute_probs_for_unknown_first_die(dice_sum)
        for i in range(6):
            df[f"model_prob_{i + 1}"] = model_probs_unknown[:, i]
            df[f"bias_{i + 1}"] = model_probs_unknown[:, i] - real_prob[i]
        df.to_csv(output_folder + f"/model_probs_unknown.csv")

        plt.figure(figsize=(20, 14))
        plt.suptitle(f"Model Probabilities vs Epoch ,First Die *Unknown*\n"
                     f"dice_sum={dice_sum}, first_die={first_die}, model={model_name}")

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            first_epoch = next((i + 1 for i, dist in enumerate(np.abs(np.array(model_probs_unknown)[:, i] - real_prob[i])) if dist < EPSILON), None)
            avg_prob = np.mean(model_probs_unknown[:, i])
            std_prob = np.std(model_probs_unknown[:, i])
            avg_bias = np.mean(np.abs(np.array(model_probs_unknown)[:, i] - real_prob[i]))
            plt.plot(np.arange(1, len(model_probs_unknown)+1), model_probs_unknown[:, i])
            plt.axhline(y=real_prob[i], color='r', linestyle='--', label=f'P({1})={real_prob[i]}')
            plt.title(f"P({i+1}), first epoch <= {EPSILON:.2f}: {first_epoch}\navg={avg_prob:.3f}, std={std_prob:.3f}, avg_bias={avg_bias:.3f}")

        plt.savefig(output_folder + f"/model_probs_unknown_s{dice_sum}.png")
        plt.show()

class EvalCallback(TrainerCallback):

    def __init__(self, tokenizer, num_epochs, dice_sum, first_die, dataset_type, model_name, output_folder):
        super(EvalCallback, self).__init__()
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.dice_sum = dice_sum
        self.first_die = first_die
        self.dataset_type = dataset_type
        self.model_probs_known = np.zeros((num_epochs, 6))
        self.model_probs_unknown = np.zeros((num_epochs, 6))
        self.model_name = model_name
        self.output_folder = output_folder


    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {int(state.epoch)}/{NUM_EPOCHS} has ended.")
        epoch_probs_known, epoch_probs_unknown = eval_model(kwargs["model"], self.tokenizer, self.dice_sum, self.first_die, self.dataset_type)
        self.model_probs_known[int(state.epoch) - 1] = epoch_probs_known
        self.model_probs_unknown[int(state.epoch) - 1] = epoch_probs_unknown
        if state.epoch == self.num_epochs:
            save_and_visualize_info(self.model_probs_known,
                                    self.model_probs_unknown,
                                    self.dice_sum,
                                    self.first_die,
                                    self.dataset_type,
                                    self.model_name,
                                    self.output_folder)



def train_model(model, tokenizer, train_dataset, eval_dataset, dice_sum, first_die, dataset_type, output_folder, model_name):
    training_args = TrainingArguments(
        output_dir=f"{output_folder}/model_output",
        eval_strategy="epoch",
        learning_rate=7e-5,
        per_device_train_batch_size=8,
        num_train_epochs=NUM_EPOCHS,
        save_strategy="no",
        save_total_limit=0,
        logging_dir=f"{output_folder}/logs",
        logging_steps=100,
        seed=37
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
                                dice_sum=dice_sum,
                                first_die=first_die,
                                dataset_type=dataset_type,
                                model_name=model_name,
                                output_folder=output_folder)]
    )

    # Fine-tune the model
    trainer.train()

    return model


def eval_model(model, tokenizer, dice_sum, first_die, dataset_type):
    print("################  Evaluating the model  #####################")
    device = model.device  # Get the device of the model
    tokens_ids = tokenizer.convert_tokens_to_ids([str(i) for i in range(1, 7)])

    model_probs_known = np.zeros(6)
    model_probs_unknown = np.zeros(6)

    if dataset_type == "first_known" or dataset_type == "both":
        prompt = (f"John rolled two dice and the sum was {dice_sum} or higher. "
                  f"He said that the first die landed on {first_die} and the second die landed on ")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Ensure the model is in evaluation mode
        model.eval()

        # Pass the prompt through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits for the next token
        next_token_logits = outputs.logits[:, -1, :]

        # Extract probabilities
        model_prob = F.softmax(next_token_logits, dim=-1)

        print(f"First die is known, dice_sum={dice_sum}, first_die={first_die}:")
        print("------")
        print("Real probabilities:")
        real_prob = compute_probs_for_known_first_die(dice_sum, first_die)
        for i in range(6):
            print(f"{i + 1}: {real_prob[i]:.4f}")

        print("------")
        print("Model probabilities:")
        for i in range(6):
            print(f"{i + 1}: {model_prob[0, tokens_ids[i]].item():.4f}")
        print("********************************************")

        model_probs_known = model_prob[0, tokens_ids].cpu().numpy()

    if dataset_type == "first_unknown" or dataset_type == "both":
        prompt = (f"John rolled two dice and the sum was {dice_sum} or higher. "
                    f"he said that one of the dice landed on ")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Ensure the model is in evaluation mode
        model.eval()

        # Pass the prompt through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits for the next token
        next_token_logits = outputs.logits[:, -1, :]

        # Extract probabilities
        model_prob = F.softmax(next_token_logits, dim=-1)

        print(f"First die unknown, dice_sum={dice_sum}:")
        print("------")
        print("Real probabilities:")
        real_prob = compute_probs_for_unknown_first_die(dice_sum)
        for i in range(6):
            print(f"{i + 1}: {real_prob[i]:.4f}")

        print("------")
        print("Model probabilities:")
        for i in range(6):
            print(f"{i + 1}: {model_prob[0, tokens_ids[i]].item():.4f}")
        print("********************************************")

        model_probs_unknown = model_prob[0, tokens_ids].cpu().numpy()

    print("################  End of Evaluation  #####################")

    return model_probs_known, model_probs_unknown




def main():
    args = parse_arguments()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to the end-of-sequence token
    print("Loading dataset...")
    train_dataset, eval_dataset = load_dice_dataset(tokenizer, args.dice_sum, args.dataset_type, SAMPLES_NUM, args.first_die)
    if INCLUDE_DATETIME:
        output_folder = PROJECT_DIR + args.training_name + datetime.now().strftime("%y%m%d-%H%M")
    else:
        output_folder = PROJECT_DIR + args.training_name
    print("Training the model...")
    model = train_model(model, tokenizer, train_dataset, eval_dataset, args.dice_sum, args.first_die, args.dataset_type, output_folder, args.model_name)
    # print("Saving the model...")
    # save_model(model, tokenizer, output_folder)


if __name__ == '__main__':
    main()