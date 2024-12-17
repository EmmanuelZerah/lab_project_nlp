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
    "001",
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
    "099",
    "10",
]

TWO_COIN_DIRS_05 = [
    "00",
    "001",
    "005",
    "01",
    "02",
    "03",
    "04",
    "06",
    "07",
    "08",
    "09",
    "095",
    "099",
    "10",
]

TWO_COIN_DIRS_09 = [
    "00",
    "001",
    "005",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "095",
    "099",
    "10",
]



def plot_coin_exp(dir_prefix, suffix_list, model_name, col_index=0, first_coin_prob=None):
    avg_bias = []
    avg_bias_last_10 = []
    bias_stds = []
    bias_stds_last_10 = []
    avg_error = []
    avg_error_last_10 = []

    for prob in suffix_list:
        prob_dir = PROJECT_DIR / (dir_prefix + prob)
        prob_df = pd.read_csv(prob_dir / "model_probs.csv")

        bias_col = prob_df.filter(like="bias")
        bias_values = bias_col.iloc[:, col_index].values

        avg_bias.append(bias_values.mean())
        avg_bias_last_10.append(bias_values[-10:].mean())
        bias_stds.append(bias_values.std())
        bias_stds_last_10.append(bias_values[-10:].std())
        avg_error.append(np.abs(bias_values).mean())
        avg_error_last_10.append(np.abs(bias_values[-10:]).mean())

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

    # plot a figure of the average bias for the last 10 epochs for each probability
    plt.figure(figsize=(12, 8))
    plt.plot(probs, avg_bias_last_10, 'ro')
    plt.plot(probs, avg_bias_last_10)
    # plot stds as error bars
    plt.errorbar(probs, avg_bias_last_10, yerr=bias_stds_last_10, fmt='o', color='r', ecolor='r', capsize=5)
    plt.xlabel("Probability")
    plt.ylabel("Average Bias (Last 10 Epochs)")
    if first_coin_prob and not col_index:
        plt.title(f"Average Bias (Last 10 Epochs) of First Coin P(H)={first_coin_prob} vs Other Coin Probability, Model: {model_name}")
    elif col_index and first_coin_prob:
        plt.title(f"Average Bias (Last 10 Epochs) Where First Coin P(H)={first_coin_prob}, Model: {model_name}")
    else:
        plt.title(f"Average Bias (Last 10 Epochs) vs Probability, Model: {model_name}")
    # plt.xticks(probs)
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

    # plot a figure of the average error for the last 10 epochs for each probability
    plt.figure(figsize=(12, 8))
    plt.plot(probs, avg_error_last_10, 'ro')
    plt.plot(probs, avg_error_last_10)
    plt.xlabel("Probability")
    plt.ylabel("Average Error (Last 10 Epochs)")
    if first_coin_prob and not col_index:
        plt.title(f"Average Error (Last 10 Epochs) of First Coin P(H)={first_coin_prob} vs Other Coin Probability, Model: {model_name}")
    elif col_index and first_coin_prob:
        plt.title(f"Average Error (Last 10 Epochs) Where First Coin P(H)={first_coin_prob}, Model: {model_name}")
    else:
        plt.title(f"Average Error (Last 10 Epochs) vs Probability, Model: {model_name}")
    # plt.xticks(probs)
    plt.axhline(y=0, color='k')
    plt.show()

    print("Done")


def plot_two_coin_exp(dir_prefix, suffix_list, model_name):
    pass



def main():
    # plot_coin_exp(ONE_COIN_SMALL, ONE_COIN_DIRS, "gpt2")
    # plot_coin_exp(ONE_COIN_SMALL_AGAIN, ONE_COIN_DIRS, "gpt2")
    # plot_coin_exp(ONE_COIN_MEDIUM, ONE_COIN_DIRS, "gpt2-medium")
    # plot_coin_exp(ONE_COIN_LARGE, ONE_COIN_DIRS, "gpt2-large")
    # plot_coin_exp(TWO_COINS_05_SMALL, TWO_COIN_DIRS_05, "gpt2", first_coin_prob=0.5)
    # plot_coin_exp(TWO_COINS_09_SMALL, TWO_COIN_DIRS_09, "gpt2", first_coin_prob=0.9)
    # plot_coin_exp(TWO_COINS_05_SMALL, TWO_COIN_DIRS_05, "gpt2", col_index=1, first_coin_prob=0.5)
    plot_coin_exp(TWO_COINS_09_SMALL, TWO_COIN_DIRS_09, "gpt2", col_index=1, first_coin_prob=0.9)




if __name__ == '__main__':
    main()