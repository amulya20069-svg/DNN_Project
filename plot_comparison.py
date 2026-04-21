import matplotlib.pyplot as plt

loss_names = ["KL", "CE", "Custom"]

# put YOUR values here
loss_names = ["KL", "CE", "Custom"]

kl_vals = [1.1777, 1.4003, 1.7042]
jsd_vals = [0.2741, 0.2754, 0.2848]
cos_vals = [0.6290, 0.6061, 0.5757]
pearson_vals = [0.1405, 0.1184, 0.0823]

plt.figure(figsize=(10, 6))

plt.plot(loss_names, kl_vals, marker='o', label="KL Divergence")
plt.plot(loss_names, jsd_vals, marker='o', label="JSD")
plt.plot(loss_names, cos_vals, marker='o', label="Cosine")
plt.plot(loss_names, pearson_vals, marker='o', label="Pearson Corr")

plt.title("Loss Function Comparison")
plt.xlabel("Loss Type")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True)

plt.savefig("outputs/loss_comparison.png")
plt.show()