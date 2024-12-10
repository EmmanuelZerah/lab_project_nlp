
import torch.nn.functional as F
from transformers import Trainer


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