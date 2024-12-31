import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from dice_exp import compute_probs_for_known_first_die, compute_probs_for_unknown_first_die

PROJECT_DIR = "/cs/labs/oabend/manuz/lab_project/runs/second_dice_exp"


def plot_known_dice_results(prefix, model_name, dataset_type):

    for die in range(1,7):

        biases = np.zeros((5, 6))
        all_model_probs = np.zeros((5, 6))

        for i in range(1, 6):
            run_dir = f"{PROJECT_DIR}/{prefix}_{i}_s7_f{die}"
            empirical_probs = pd.read_csv(f"{run_dir}/empirical_probs.csv")["empirical_probs_known"]
            model_probs = pd.read_csv(f"{run_dir}/final_model_probs.csv")["model_probs_known"]
            bias = model_probs.values - empirical_probs.values
            biases[i-1] = bias
            all_model_probs[i-1] = model_probs.values

        # avg_biases = np.mean(biases, axis=0)
        # std_biases = np.std(biases, axis=0)

        avg_model_probs = np.mean(all_model_probs, axis=0)
        std_model_probs = np.std(all_model_probs, axis=0)

        real_probs = compute_probs_for_known_first_die(7, die)

        figure(figsize=(8, 6))
        plt.plot(range(1, 7), avg_model_probs, "ro", label="Model Probabilities")
        plt.errorbar(range(1, 7), avg_model_probs, yerr=std_model_probs, fmt='o', color='red')
        plt.plot(range(1, 7), real_probs, "bo", label="Real Probabilities")
        plt.xlabel("Die")
        plt.ylabel("Probability")
        plt.title(f"Model and Real Probabilities of {model_name} on '{dataset_type}' Dataset, When First Die is {die}")
        plt.legend()
        plt.show()

        # figure(figsize=(8, 6))
        # plt.plot(range(1,7), avg_biases, "ro")
        # plt.errorbar(range(1,7), avg_biases, yerr=std_biases, fmt='o')
        # plt.xlabel("Die")
        # plt.ylabel("Bias")
        # plt.title(f"Bias of {model_name} on '{dataset_type}' Dataset, When First Die is {die}")
        # plt.show()


def plot_unknown_dice_results(both_prefix, unknown_prefix, model_name):

    all_model_probs = np.zeros(((5*6), 6))

    for die in range(1,7):

        for i in range(1, 6):
            run_dir = f"{PROJECT_DIR}/{both_prefix}_{i}_s7_f{die}"
            model_probs = pd.read_csv(f"{run_dir}/final_model_probs.csv")["model_probs_unknown"]
            all_model_probs[(i-1)*6 + die-1] = model_probs.values

    avg_model_probs = np.mean(all_model_probs, axis=0)
    std_model_probs = np.std(all_model_probs, axis=0)

    real_probs = compute_probs_for_unknown_first_die(7)

    figure(figsize=(8, 6))
    plt.plot(range(1, 7), avg_model_probs, "ro", label="Model Probabilities")
    plt.plot(range(1, 7), real_probs, "bo", label="Real Probabilities")
    plt.errorbar(range(1, 7), avg_model_probs, yerr=std_model_probs, fmt='o', color='red')
    plt.xlabel("Die")
    plt.ylabel("Probability")
    plt.title(f"Model and Real Probabilities of {model_name} on 'both' Dataset, When First Die is Unknown")
    plt.show()

    all_model_probs = np.zeros((5, 6))

    for i in range(1, 6):
        run_dir = f"{PROJECT_DIR}/{unknown_prefix}_{i}_s7"
        model_probs = pd.read_csv(f"{run_dir}/final_model_probs.csv")["model_probs_unknown"]
        all_model_probs[i-1] = model_probs.values

    avg_model_probs = np.mean(all_model_probs, axis=0)
    std_model_probs = np.std(all_model_probs, axis=0)

    figure(figsize=(8, 6))
    plt.plot(range(1, 7), avg_model_probs, "ro", label="Model Probabilities")
    plt.errorbar(range(1, 7), avg_model_probs, yerr=std_model_probs, fmt='o', color='red')
    plt.plot(range(1, 7), real_probs, "bo", label="Real Probabilities")
    plt.xlabel("Die")
    plt.ylabel("Probability")
    plt.title(f"Model and Real Probabilities of {model_name} on 'unknown first' Dataset, When First Die is Unknown")
    plt.show()









if __name__ == '__main__':
    # plot_known_dice_results("dice_both_large", "gpt2-large", "both")
    # plot_known_dice_results("dice_known_large", "gpt2-large", "first known")
    plot_unknown_dice_results("dice_both_large", "dice_unknown_large", "gpt2-large")