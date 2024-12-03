
import os
import numpy as np
import pandas as pd
from datasets import Dataset


def create_coin_dataset(probs, size):
    prompts_dfs = []
    for prob in probs:
        prompt = (
            f"John is flipping a biased coin with the following probabilities: P(H) = {prob:.2f} and P(T) = {(1 - prob):.2f}. "
            f"Complete the sentence with either 'H' or 'T' only: John flipped the coin and it landed on ")
        prompt_df = pd.DataFrame({'prompt': [prompt] * (size // len(probs))})
        prompt_df['label'] = np.random.choice(['H', 'T'], size=size // len(probs), p=[prob, 1 - prob])
        prompts_dfs.append(prompt_df)
    pd_dataset = pd.concat(prompts_dfs)
    pd_dataset = pd_dataset.sample(frac=1).reset_index(drop=True)
    return pd_dataset


def load_dataset(tokenizer, coin_probs, samples_num):
    pd_dataset = create_coin_dataset(coin_probs, samples_num)
    pd_dataset  = pd_dataset.sample(samples_num, random_state=37)
    dataset = Dataset.from_pandas(pd_dataset)

    def preprocess(example):
        inputs = tokenizer(
            example["prompt"],
            truncation=True,
            max_length=100,
            padding="max_length"
        )
        labels = tokenizer(
            example["label"],
            truncation=True,
            max_length=100,
            padding="max_length"
        )
        inputs["labels"] = labels["input_ids"]
        inputs["label"] = labels["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(preprocess, batched=False)
    train_dataset, eval_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=37).values()
    return train_dataset, eval_dataset


def save_model(model, tokenizer, output_folder):
    os.makedirs(f"{output_folder}/fine_tuned_model/", exist_ok=True)
    model.save_pretrained(f"{output_folder}/fine_tuned_model/")
    tokenizer.save_pretrained(f"{output_folder}/fine_tuned_model/")