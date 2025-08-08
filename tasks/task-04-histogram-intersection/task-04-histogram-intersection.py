import numpy as np
import cv2

def compute_histogram_intersection(image_path1: str, image_path2: str) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images
    loaded from given file paths.

    Parameters:
        image_path1 (str): Path to the first grayscale image.
        image_path2 (str): Path to the second grayscale image.

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If images cannot be read or are not 2D grayscale images.
    """
    # Load both images as grayscale
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # Check if images were loaded correctly
    if img1 is None:
        raise ValueError(f"Image at path '{image_path1}' could not be read.")
    if img2 is None:
        raise ValueError(f"Image at path '{image_path2}' could not be read.")

    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###

    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    hist1, _ = np.histogram(img1_flat, bins=256, range=(0, 256))
    hist2, _ = np.histogram(img2_flat, bins=256, range=(0, 256))

    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    intersection = np.sum(np.minimum(hist1, hist2))
    ### END CODE HERE ###

    return float(intersection)
