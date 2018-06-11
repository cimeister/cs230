import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

train_loss = np.load("train_loss.npy")

train_acc = np.load("train_acc.npy")
val_acc = np.load("val_acc.npy")
train_perp = np.load("train_perp.npy")
val_perp = np.load("val_perp.npy")

plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("Train and Validation Perplexity")

red_patch = mpatches.Patch(color='red', label='Train')
green_patch = mpatches.Patch(color='green', label='Validation')

x = np.linspace(1, 11, len(train_perp))
plt.plot(x, train_perp, 'r')
plt.plot(x, val_perp, 'g')

plt.legend(handles=[red_patch, green_patch])

plt.savefig('perplexity.png')

plt.close()

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy")

red_patch = mpatches.Patch(color='red', label='Train')
green_patch = mpatches.Patch(color='green', label='Validation')

x = np.linspace(1, 11, len(train_perp))
plt.plot(x, train_acc, 'r')
plt.plot(x, val_acc, 'g')

plt.legend(handles=[red_patch, green_patch])

plt.savefig('accuracy.png')

plt.close()
