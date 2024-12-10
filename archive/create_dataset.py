
from itertools import combinations, permutations
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(37)

PERMUTATIONS_DICES = [6, 12]
COMBINATIONS_DICES = [6, 12, 18, 24, 30]

MAX_DATASET_SIZE = 2_500_000  # Maximum number of prompts to generate


def compute_all_combinations(dices):
    # Initialize a list to store prompts
    prompts = []

    # Generate all possible prompts
    for dice in dices:
        print(f"Generating prompts for a {dice}-sided die...")
        prefix = f"I am playing a game where I throw a {dice}-sided die. "
        suffix = "I just threw the die. Did I win or lose? Respond with just 'W' for win or 'L' for lose: "
        numbers = list(range(1, dice + 1))

        # Generate all combinations of 2 losing numbers and 4 winning numbers
        combinations_of_losses = list(combinations(numbers, dice // 3))

        # Generate prompts for each combination
        for losing_numbers in tqdm(combinations_of_losses, desc="Generating prompts"):
            winning_numbers = [num for num in numbers if num not in losing_numbers]

            win_combination = ', '.join(map(str, winning_numbers))
            lose_combination = ', '.join(map(str, losing_numbers))

            # Generate prompts for each combination
            prompt = (prefix + f"If it lands on {win_combination}, I win. If it lands on {lose_combination}, I lose. " + suffix)

            # Append the prompt to the dataset
            prompts.append(prompt)

            if len(prompts) >= MAX_DATASET_SIZE:
                print("Dataset size reached the maximum limit.")
                break

        if len(prompts) >= MAX_DATASET_SIZE:
            break

    # Create a DataFrame from the prompts
    df = pd.DataFrame(prompts, columns=["prompt"])

    # Create column for the correct answer ('W' or 'L') randomly
    df["next"] = np.random.choice(['W', 'L'], size=len(df), p=[2/3, 1/3])

    # Save the prompts to parquet file
    df.to_parquet("datasets/all_combinations_6-30.parquet", index=False)


def compute_all_permutations(dices):
    # Initialize a list to store prompts
    prompts = []

    # Generate all possible prompts
    for dice in dices:
        print(f"Generating prompts for a {dice}-sided die...")
        prefix = f"I am playing a game where I throw a {dice}-sided die. "
        suffix = "I just threw the die. Did I win or lose? Respond with just 'W' for win or 'L' for lose: "

        numbers = list(range(1, dice + 1))

        # Generate all combinations of 2 losing numbers and 4 winning numbers
        combinations_of_losses = list(combinations(numbers, dice // 3))

        # Generate prompts for each combination
        for losing_numbers in tqdm(combinations_of_losses, desc="Generating prompts"):
            winning_numbers = [num for num in numbers if num not in losing_numbers]

            # Generate permutations for losing and winning numbers
            losing_permutations = list(permutations(losing_numbers))
            winning_permutations = list(permutations(winning_numbers))

            for lose_perm in losing_permutations:
                for win_perm in winning_permutations:
                    lose_str = ", ".join(map(str, lose_perm))
                    win_str = ", ".join(map(str, win_perm))
                    prompt = (prefix + f"If it lands on {win_str}, I win. If it lands on {lose_str}, I lose. " + suffix)
                    prompts.append(prompt)

            if len(prompts) >= MAX_DATASET_SIZE:
                print("Dataset size reached the maximum limit.")
                break

        if len(prompts) >= MAX_DATASET_SIZE:
            break

    # Create a DataFrame from the prompts
    df = pd.DataFrame(prompts, columns=["prompt"])

    # Create column for the correct answer ('W' or 'L') randomly
    df["next"] = np.random.choice(['W', 'L'], size=len(df), p=[2/3, 1/3])

    # Save the prompts to parquet file
    df.to_parquet("datasets/all_permutations_6-12.parquet", index=False)


if __name__ == '__main__':
    compute_all_permutations(PERMUTATIONS_DICES)
    compute_all_combinations(COMBINATIONS_DICES)
    print("Done!")
