from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path("/cs/labs/oabend/manuz/lab_project/runs/")

ONE_COIN_SMALL = "one_coin_p"
ONE_COIN_SMALL_AGAIN = "one_coin_again_p"
ONE_COIN_MEDIUM = "one_coin_medium_p"
ONE_COIN_LARGE = "one_coin_large_p"

TWO_COINS_05_SMALL = "two_coins_p05-"
TWO_COINS_09_SMALL = "two_coins_p09-"

ONE_COIN_DIRS = [
    "00",
    "005",
    "01",
    "015",
    "02",
    "025",
    "03",
    "035",
    "04",
    "045",
    "05",
    "055",
    "06",
    "065",
    "07",
    "075",
    "08",
    "085",
    "09",
    "095",
    "10",
]

TWO_COIN_DIRS_05 =  [
    "00",
    "005",
    "01",
    "015",
    "02",
    "025",
    "03",
    "035",
    "04",
    "045",
    "055",
    "06",
    "065",
    "07",
    "075",
    "08",
    "085",
    "09",
    "095",
    "1",
]

TWO_COIN_DIRS_09 = [
    "00",
    "005",
    "01",
    "015",
    "02",
    "025",
    "03",
    "035",
    "04",
    "045",
    "05",
    "055",
    "06",
    "065",
    "07",
    "075",
    "08",
    "085",
    "095",
    "1",
]


def plot_coin_exp(dir_prefix, suffix_list, model_name, col_index=0, first_coin_prob=None):
    avg_bias = []
    avg_bias_last_3 = []
    bias_stds = []
    bias_stds_last_3 = []
    avg_error = []
    avg_error_last_3 = []

    for prob in suffix_list:
        prob_dir = PROJECT_DIR / (dir_prefix + prob)
        prob_df = pd.read_csv(prob_dir / "model_probs.csv")

        bias_col = prob_df.filter(like="bias")
        bias_values = bias_col.iloc[:, col_index].values

        avg_bias.append(bias_values.mean())
        avg_bias_last_3.append(bias_values[-3:].mean())
        bias_stds.append(bias_values.std())
        bias_stds_last_3.append(bias_values[-3:].std())
        avg_error.append(np.abs(bias_values).mean())
        avg_error_last_3.append(np.abs(bias_values[-3:]).mean())

    probs = [float(prob[:1] + "." + prob[1:]) for prob in suffix_list]

    # # plot a figure of the average bias for each probability
    # plt.figure(figsize=(12, 8))
    # plt.plot(probs, avg_bias, 'ro')
    # plt.plot(probs, avg_bias)
    # # plot stds as error bars
    # plt.errorbar(probs, avg_bias, yerr=bias_stds, fmt='o', color='r', ecolor='r', capsize=5)
    # plt.xlabel("Probability")
    # plt.ylabel("Average Bias")
    # if first_coin_prob:
    #     plt.title(f"Average Bias of First Coin P(H)={first_coin_prob} vs Other Coin Probability, Model: {model_name}")
    # else:
    #     plt.title(f"Average Bias vs Probability, Model: {model_name}")
    # # plt.xticks(probs)
    # plt.axhline(y=0, color='k')
    # plt.show()

    # plot a figure of the average bias for the last 3 epochs for each probability
    plt.figure(figsize=(12, 8))
    plt.plot(probs, avg_bias_last_3, 'ro')
    plt.plot(probs, avg_bias_last_3)
    # plot stds as error bars
    plt.errorbar(probs, avg_bias_last_3, yerr=bias_stds_last_3, fmt='o', color='r', ecolor='r', capsize=5)
    plt.xlabel("Probability")
    plt.ylabel("Average Bias (Last 3 Epochs)")
    if first_coin_prob and not col_index:
        plt.title(f"Average Bias (Last 3 Epochs) of First Coin P(H)={first_coin_prob} vs Other Coin Probability, Model: {model_name}")
    elif col_index and first_coin_prob:
        plt.title(f"Average Bias (Last 3 Epochs) Where First Coin P(H)={first_coin_prob}, Model: {model_name}")
    else:
        plt.title(f"Average Bias (Last 3 Epochs) vs Probability, Model: {model_name}")
    plt.xticks(probs)
    plt.axhline(y=0, color='k')
    plt.show()

    # # plot a figure of the average error for each probability
    # plt.figure(figsize=(12, 8))
    # plt.plot(probs, avg_error, 'ro')
    # plt.plot(probs, avg_error)
    # plt.xlabel("Probability")
    # plt.ylabel("Average Error")
    # if first_coin_prob:
    #     plt.title(f"Average Error of First Coin P(H)={first_coin_prob} vs Other Coin Probability, Model: {model_name}")
    # else:
    #     plt.title(f"Average Error vs Probability, Model: {model_name}")
    # # plt.xticks(probs)
    # plt.axhline(y=0, color='k')
    # plt.show()

    # plot a figure of the average error for the last 3 epochs for each probability
    plt.figure(figsize=(12, 8))
    plt.plot(probs, avg_error_last_3, 'ro')
    plt.plot(probs, avg_error_last_3)
    plt.xlabel("Probability")
    plt.ylabel("Average Error (Last 3 Epochs)")
    if first_coin_prob and not col_index:
        plt.title(f"Average Error (Last 3 Epochs) of First Coin P(H)={first_coin_prob} vs Other Coin Probability, Model: {model_name}")
    elif col_index and first_coin_prob:
        plt.title(f"Average Error (Last 3 Epochs) Where First Coin P(H)={first_coin_prob}, Model: {model_name}")
    else:
        plt.title(f"Average Error (Last 3 Epochs) vs Probability, Model: {model_name}")
    plt.xticks(probs)
    plt.axhline(y=0, color='k')
    plt.show()

    print("Done")


def plot_final_results(prefix, dirs, model_name, col_index=0, first_coin_prob=None):
    avg_bias = []
    avg_error = []
    std_bias = []
    std_error = []

    for prob_str in dirs:
        biases = []
        errors = []
        for i in range(1, 4):
            if first_coin_prob:
                prob_dir = PROJECT_DIR / (prefix + f"_{i}_p{first_coin_prob}-{prob_str}")
            else:
                prob_dir = PROJECT_DIR / (prefix + f"_{i}_p{prob_str}")
            prob_df = pd.read_csv(prob_dir / "final_model_probs.csv")
            model_prob = prob_df.iloc[:, col_index].values[0]

            empirical_df = pd.read_csv(prob_dir / "empirical_probs.csv")
            empirical_prob = empirical_df.iloc[:, col_index].values[0]

            bias = model_prob - empirical_prob
            biases.append(bias)
            errors.append(np.abs(bias))

        avg_bias.append(np.mean(biases))
        avg_error.append(np.mean(errors))
        std_bias.append(np.std(biases))
        std_error.append(np.std(errors))

    probs = [float(prob[:1] + "." + prob[1:]) for prob in dirs]

    # plot a figure of the average bias for each probability
    plt.figure(figsize=(12, 8))
    plt.plot(probs, avg_bias, 'ro')
    plt.plot(probs, avg_bias)
    plt.errorbar(probs, avg_bias, yerr=std_bias, fmt='o', color='r', ecolor='r', capsize=5)
    plt.xlabel("Probability")
    plt.ylabel("Average Bias")
    if first_coin_prob and not col_index:
        plt.title(f"Average Bias of First Coin P(H)={first_coin_prob} vs Other Coin Probability, Model: {model_name}")
    elif col_index and first_coin_prob:
        plt.title(f"Average Bias Where First Coin P(H)={first_coin_prob}, Model: {model_name}")
    else:
        plt.title(f"Average Bias vs Probability, Model: {model_name}")
    plt.xticks(probs)
    plt.axhline(y=0, color='k')
    plt.show()

    # plot a figure of the average error for each probability
    plt.figure(figsize=(12, 8))
    plt.plot(probs, avg_error, 'ro')
    plt.plot(probs, avg_error)
    # plt.errorbar(probs, avg_error, yerr=std_error, fmt='o', color='r', ecolor='r', capsize=5)
    plt.xlabel("Probability")
    plt.ylabel("Average Error")
    if first_coin_prob and not col_index:
        plt.title(f"Average Error of First Coin P(H)={first_coin_prob} vs Other Coin Probability, Model: {model_name}")
    elif col_index and first_coin_prob:
        plt.title(f"Average Error Where First Coin P(H)={first_coin_prob}, Model: {model_name}")
    else:
        plt.title(f"Average Error vs Probability, Model: {model_name}")
    plt.xticks(probs)
    plt.axhline(y=0, color='k')
    plt.show()

    print("Done")



def main():
    # plot_final_results("one_coin_more_samples", ONE_COIN_DIRS, "gpt2")
    # plot_final_results("one_coin_more_samples_medium", ONE_COIN_DIRS, "gpt2-medium")
    # plot_final_results("one_coin_more_samples_large", ONE_COIN_DIRS, "gpt2-large")

    # plot_final_results("two_coins_more_samples", TWO_COIN_DIRS_05, "gpt2", col_index=1, first_coin_prob="05")
    # plot_final_results("two_coins_more_samples_medium", TWO_COIN_DIRS_05, "gpt2-medium", col_index=1, first_coin_prob="05")
    # plot_final_results("two_coins_more_samples_large", TWO_COIN_DIRS_05, "gpt2-large", col_index=1, first_coin_prob="05")
    #
    # plot_final_results("two_coins_more_samples", TWO_COIN_DIRS_09, "gpt2", col_index=1, first_coin_prob="09")
    # plot_final_results("two_coins_more_samples_medium", TWO_COIN_DIRS_09, "gpt2-medium", col_index=1, first_coin_prob="09")
    # plot_final_results("two_coins_more_samples_large", TWO_COIN_DIRS_09, "gpt2-large", col_index=1, first_coin_prob="09")

    plot_final_results("two_coins_more_samples", TWO_COIN_DIRS_05, "gpt2", col_index=0, first_coin_prob="05")
    plot_final_results("two_coins_more_samples_medium", TWO_COIN_DIRS_05, "gpt2-medium", col_index=0, first_coin_prob="05")
    plot_final_results("two_coins_more_samples_large", TWO_COIN_DIRS_05, "gpt2-large", col_index=0, first_coin_prob="05")

    # plot_final_results("two_coins_more_samples", TWO_COIN_DIRS_09, "gpt2", col_index=0, first_coin_prob="09")
    # plot_final_results("two_coins_more_samples_medium", TWO_COIN_DIRS_09, "gpt2-medium", col_index=0, first_coin_prob="09")
    # plot_final_results("two_coins_more_samples_large", TWO_COIN_DIRS_09, "gpt2-large", col_index=0, first_coin_prob="09")

    # plot_coin_exp("one_coin_more_samples_1_p", ONE_COIN_DIRS, "gpt2")
    # plot_coin_exp(ONE_COIN_SMALL, ONE_COIN_DIRS, "gpt2")
    # plot_coin_exp(ONE_COIN_SMALL_AGAIN, ONE_COIN_DIRS, "gpt2")
    # plot_coin_exp(ONE_COIN_MEDIUM, ONE_COIN_DIRS, "gpt2-medium")
    # plot_coin_exp(ONE_COIN_LARGE, ONE_COIN_DIRS, "gpt2-large")
    # plot_coin_exp(TWO_COINS_05_SMALL, TWO_COIN_DIRS_05, "gpt2", first_coin_prob=0.5)
    # plot_coin_exp(TWO_COINS_09_SMALL, TWO_COIN_DIRS_09, "gpt2", first_coin_prob=0.9)
    # plot_coin_exp(TWO_COINS_05_SMALL, TWO_COIN_DIRS_05, "gpt2", col_index=1, first_coin_prob=0.5)
    # plot_coin_exp(TWO_COINS_09_SMALL, TWO_COIN_DIRS_09, "gpt2", col_index=1, first_coin_prob=0.9)




if __name__ == '__main__':
    main()