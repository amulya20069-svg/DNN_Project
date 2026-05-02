import matplotlib.pyplot as plt

loss_names = ["KL", "CE", "Custom"]

# put YOUR values here
loss_names = ["KL", "CE", "Custom"]

kl_vals = [0.8086, 1.0207, 1.0862]
jsd_vals = [0.1655, 0.2125, 0.2051]
cos_vals = [0.7663, 0.7005, 0.7006]
pearson_vals = [0.2405, 0.1858, 0.1536]

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