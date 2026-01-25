from PIL import Image
import torch
from torchvision import datasets
import numpy as np
from torchvision.transforms import v2 as T
import os
import PIL
import random
from helper_plus import (
    random_crop_with_coordinate,
    compute_IoU,
)



def random_crop(image: Image.Image, alpha: float = 0.1) -> Image.Image:
    """Randomly crops an image within a size range determined by alpha and the image dimensions.

    Args:
        image (Image): The input image to crop.
        alpha (float): The minimum scale factor for the crop as a proportion of the smallest dimension.

    Returns:
        PIL Image or Tensor: Cropped image
    """
    # Get the width and height of the original image
    w, h = image.size
    # Determine the size of the crop based on alpha and the smallest dimension
    n_px = np.random.uniform(low=alpha, high=0.9) * min(h, w)
    # Perform the crop
    cropped = T.RandomCrop(int(n_px))(image)

    return cropped

def custom_loader(path: str, alpha: float, n_samples: int) -> PIL.Image:
    """Loads an image, applies a processing function, and returns augmented versions.

    Args:
        path (str): The path to the image file.
        n_samples (int): The number of augmented samples to generate.

    Returns:
        torch.Tensor: A tensor stack of the processed image and its augmented samples.
    """
    # Load the image using the default loader
    img = datasets.folder.default_loader(path)
    # Process the image and generate additional augmented samples
    augmented_imgs = [img]
    augmented_imgs.extend(random_crop(img, alpha=alpha) for _ in range(n_samples))
    # augmented_imgs = torch.stack(augmented_imgs)
    return augmented_imgs

def custom_loader_plus(path: str) -> torch.Tensor:
    """Loads an image, applies a processing function, and returns augmented versions.

    Args:
        path (str): The path to the image file.
        n_samples (int): The number of augmented samples to generate.

    Returns:
        torch.Tensor: A tensor stack of the processed image and its augmented samples.
    """
    # Load the image using the default loader
    img = datasets.folder.default_loader(path)
    # Process the image and generate additional augmented samples
    augmented_imgs = [img]
    img_coordinates = [(0,0,0,0)]

    for _ in range(n_samples):
        crop_img, img_coordinate = random_crop_with_coordinate(img, alpha=alpha)
        augmented_imgs.append(crop_img)
        img_coordinates.append(img_coordinate) # ensure the length of the two are the same.
    # augmented_imgs = torch.stack(augmented_imgs)

    filtered_augmented_imgs = filter_augmented_samples(
                                                        augmented_imgs,
                                                        img_coordinates,
                                                        threshold=iou_threshold,
                                                        max_queue_len=max_queue_len
                                                        )
    return filtered_augmented_imgs


def filter_augmented_samples(augmented_imgs, img_coordinates: list, threshold: float, max_queue_len: int = 50) -> torch.Tensor:
    # augmented_imgs has shape:[101, 3, 224, 224]
    num_augmented_imgs = len(augmented_imgs) - 1

    crop_samples = augmented_imgs[1:]
    crop_sample_coordinates = img_coordinates[1:]

    queue = [crop_samples[0]]
    saved_coordinates = [crop_sample_coordinates[0]]
    filtered_augmented_imgs = [augmented_imgs[0]]
    for i in range(1, num_augmented_imgs):
        next_coordinates = crop_sample_coordinates[i]
        iou = compute_IoU(saved_coordinates, next_coordinates)
        if max(iou) <= threshold:
            # save the img to the queue and save the coordinate.
            queue.append(crop_samples[i])
            saved_coordinates.append(next_coordinates)
            if len(queue) >= max_queue_len:
                break

    if len(queue) < max_queue_len:
        num_samples_needed = max_queue_len - len(queue)
        additional_samples = random.choices(queue, k=num_samples_needed)
        queue.extend(additional_samples)
    filtered_augmented_imgs.extend(queue)
    return filtered_augmented_imgs


WCA = 0


alpha = 0.1
n_samples = 250
model_size = "ViT-L/14"
iou_threshold = 0.6
max_queue_len = 100
device = "cuda"
img_path = "jpg image path" #TODO: specify the image path here.
# _, processor = clip.load(model_size, device=device)
if WCA:
    crop_imgs = custom_loader(path=img_path, alpha=alpha, n_samples=n_samples)
else:
    crop_imgs = custom_loader_plus(path=img_path)
# crop_imgs has shape: [101, 3, 224, 224], first image is the original image.
output_dir = "cropped_images"
os.makedirs(output_dir, exist_ok=True)

for i in range(1, max_queue_len + 1):
    crop_imgs[i].save(f"{output_dir}/cropped_{i}.png")

    