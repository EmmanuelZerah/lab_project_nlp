from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

prompt_1 = (
    "I am playing a game where I throw a dice. If it lands on 1 or 2, I lose. "
    "If it lands on 3, 4, 5, or 6, I win. I just threw the dice. "
    "Did I win or lose? Answer with either 'win' or 'lose': "
)

prompt_2 = (
    "I am playing a game where I throw a dice. If it lands on 1 or 2, I win. "
    "If it lands on 3, 4, 5, or 6, I lose. I just threw the dice. "
    "Did I win or lose? Answer with either 'win' or 'lose': "
)

prompt_3 = (
    "I am playing a game where I throw a dice. If it lands on 1, 2 or 3, I win. "
    "If it lands on 4, 5, or 6, I lose. I just threw the dice. "
    "Did I win or lose? Answer with either 'win' or 'lose': "
)

prompt_4 = (
    "I am playing a game where I throw a dice. If it lands on 1, 2 or 3, I lose. "
    "If it lands on 4, 5, or 6, I win. I just threw the dice. "
    "Did I win or lose? Answer with either 'win' or 'lose': "
)

prompts = [prompt_1, prompt_2, prompt_3, prompt_4]

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Load the tokenizer and model
model_name = "distilgpt2"  # This model is smaller and faster than GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare the prompt
prompt = (
    "I am playing a game where I throw a 6-sided die. If it lands on 1 or 2, I win. "
    "If it lands on 3, 4, 5, or 6, I lose. I just threw the die. "
    "Did I win or lose? Respond with just 'W' for win or 'L' for lose: "
)

inputs = tokenizer(prompt, return_tensors="pt")

# Get the token IDs for "W" and "L"
win_token_id = tokenizer.convert_tokens_to_ids("W")
lose_token_id = tokenizer.convert_tokens_to_ids("L")

# Verify the token IDs
if win_token_id == tokenizer.unk_token_id or lose_token_id == tokenizer.unk_token_id:
    raise ValueError("The tokenizer does not recognize 'W' or 'L' as valid tokens.")

# Get model logits
with torch.no_grad():
    outputs = model(**inputs)

# Extract the logits for the next token
next_token_logits = outputs.logits[:, -1, :]

# Mask all logits except for the winning and losing tokens
mask = torch.full(next_token_logits.shape, float("-inf"), device=next_token_logits.device)
mask[:, [win_token_id, lose_token_id]] = next_token_logits[:, [win_token_id, lose_token_id]]

# Apply softmax to compute probabilities over the restricted distribution
restricted_probabilities = F.softmax(mask, dim=-1)

# Get the probabilities for "W" and "L"
win_prob = restricted_probabilities[0, win_token_id].item()
lose_prob = restricted_probabilities[0, lose_token_id].item()

print(f"Probability of 'W' (win): {win_prob:.4f}")
print(f"Probability of 'L' (lose): {lose_prob:.4f}")



# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from datasets import load_dataset
# import evaluate
# import numpy as np
#
# # Step 1: Load the tokenizer and model
# model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)  # 4 labels for AG News
#
# # Step 2: Load the AG News dataset
# dataset = load_dataset("ag_news")
#
# # Step 3: Preprocess the data
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
#
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
# # Reduce the dataset size
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))  # Take only 1000 samples
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))     # Take only 200 samples
#
# # Step 4: Define training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=1,
#     weight_decay=0.01
# )
#
# # Step 5: Define the evaluation metric
# metric = evaluate.load("accuracy")
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
# # Step 6: Set up the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics
# )
#
# # Step 7: Train the model
# trainer.train()
#
# # Step 8: Evaluate the model
# eval_results = trainer.evaluate()
# print("Evaluation results:", eval_results)
#
