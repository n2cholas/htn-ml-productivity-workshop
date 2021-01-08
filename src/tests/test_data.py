import matplotlib.pyplot as plt
import numpy as np

from data import get_random_eraser


def test_random_eraser():
    eraser = get_random_eraser(pixel_level=True)
    img = np.ones((28, 28))
    plt.imshow(img)
    erased = eraser(img)
    plt.imshow(erased)
