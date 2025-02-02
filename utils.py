
import os
from datasets import Dataset


def prepare_dataset(tokenizer, pd_dataset):
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