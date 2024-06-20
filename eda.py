import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits

def perform_eda():
    # Load the digits dataset
    digits = load_digits()
    x, y = digits.data, digits.target

    # Plot some sample images from the dataset
    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap='gray')
        ax.set_title(f'Label: {digits.target[i]}')
        ax.axis('off')
    plt.show()

    # Display class distribution in the dataset
    sns.countplot(y)
    plt.title('Class Distribution in Digits Dataset')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    perform_eda()
