

import matplotlib.pyplot as plt

min_dists = [0, 0.018, 0.027, 0.0009, 0.0072, 0.00689, 0.0021, 0.002, 0.0011, 0.0072, 0.003, 0.0176, 0]
last_dist = [0, 0.049, 0.1, 0.02, 0.01, 0.015, 0.025, 0.065, 0.12 ,0.11, 0.025, 0.049, 0]
first_epochs = [1, 1, 7 ,11, 14, 7, 1, 10, 8, 4, 3, 1, 1]
probs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


# plot points of the minimum distance for each prob.
plt.figure(figsize=(12, 8))
plt.plot(probs, min_dists, 'ro')
plt.plot(probs, min_dists)
plt.xlabel("Probability")
plt.ylabel("Minimum Distance")
plt.title("Minimum Distance from Real Probability vs Probability")
plt.xticks(probs)
plt.savefig("min_dist_vs_prob.png")
plt.show()

# plot a bar chart of the last for each prob
plt.figure(figsize=(12, 8))
plt.plot(probs, last_dist, 'ro')
plt.plot(probs, last_dist)
plt.xlabel("Probability")
plt.ylabel("Last Distance")
plt.title("Last Distance from Real Probability vs Probability")
plt.xticks(probs)
plt.savefig("last_dist_vs_prob.png")
plt.show()

# plot a bar chart of the first epoch for each prob
plt.figure(figsize=(12, 8))
plt.plot(probs, first_epochs, 'ro')
plt.plot(probs, first_epochs)
plt.xlabel("Probability")
plt.ylabel("First Epoch")
plt.title("First Epoch to Reach Dist<0.05 vs Probability")
plt.xticks(probs)
plt.savefig("first_epoch_vs_prob.png")
plt.show()
