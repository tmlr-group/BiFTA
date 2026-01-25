import os
import random
import numpy as np
import torch
import json
import pickle
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from clip import clip
from torchvision.datasets import ImageFolder, Places365
from my_datasets import *
from utils import (
    openai_imagenet_classes,
    imagenet_classes,
    imagenet_a_lt,
    imagenet_r_lt,
)


def load_json(filename):
    if not filename.endswith(".json"):
        filename += ".json"
    with open(filename, "r") as fp:
        return json.load(fp)


def set_seed(seed):
    print(f"Setting seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_path, dataset_name, custom_loader):
    data_path = data_path

    if dataset_name == MyDataset.ImageNet:
        dataset = ImageFolder(root=data_path, 
                              transform=None,
                              loader=custom_loader,
                              )

    elif dataset_name == MyDataset.ImageNetV2:
        dataset = ImageNetV2Dataset(
            location=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetR:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetS:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetA:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.CUB:
        dataset = CUBDataset(
            data_path,
            train=False,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.Food101:
        dataset = Food101(
            data_path,
            transform=None,
            loader=custom_loader,
            split="test",
            download=True,
        )

    elif dataset_name == MyDataset.OxfordIIITPet:
        dataset = OxfordIIITPet(
            data_path,
            transform=None,
            split="test",
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.Place365:
        dataset = Places365(
            data_path,
            transform=None,
            loader=custom_loader,
            download=False,
            split="val",
            small=False,
        )

    elif dataset_name == MyDataset.DTD:
        dataset = DTD(
            data_path,
            # transform=None,
            loader=custom_loader,
            split="test",
            download=True,
        )

    return dataset


def wordify(string):
    word = string.replace("_", " ")
    return word


def load_classes(dataset_name):
    with open(
        f"features/tmlr_rebuttal/{dataset_name}/{dataset_name}.json",
        "r",
    ) as f:
        classes = json.load(f)

    wordify_classes = []
    for c in classes:
        wordify_classes.append(wordify(c))

    return wordify_classes


def generate_weights(
    method,
    model,
    dataset_name,
    tt_scale=None,
    device=None,
):
    templates = None
    make_sentence = False
    is_template = True

    # if dataset start with imagenet
    if dataset_name.startswith(MyDataset.ImageNet):
        classes = (
            openai_imagenet_classes
            if method in ["clip-d", "waffle"]
            else imagenet_classes
        )
    else:
        classes = load_classes(dataset_name)

    print(f"Creating {method} text embeddings...")

    if method != "clip":
        if method == "wca":
            load_file = "cupl" #cupl
        elif method == "cupl":
            load_file = "cupl"
        elif method == "waffle":
            load_file = "clip-d"
        elif method == "bifta":
            # load_file = "wca-turbo-merge2split" #attrVR+CuPL
            # load_file = "bifta0.95_top20"
            # load_file = "bifta0.9_top10"
            # load_file = "bifta0.99_top30"
            # load_file = "bifta0.99_top40"
            load_file = "bifta0.99_top50"
        else:
            load_file = method

        with open(f"prompts/{dataset_name}/{load_file}.json") as f:
            templates = json.load(f)

        if method in ["waffle", "clip-d", "cupl", "wca", "bifta"]:
            is_template = False

        if method == "clip-d":
            make_sentence = True

        if method == "waffle":
            templates = construct_random(templates)

    zeroshot_weights = zeroshot_classifier(
        model,
        classes,
        templates,
        is_template,
        make_sentence,
        tt_scale,
        device,
    )

    return zeroshot_weights


def load_precomputed_features_bifta(
    model,
    dataset_name: str,
    model_size: str,
    crop_config: dict,
    batch_size: int,
    num_workers: int,
    data_path: str,
    custom_loader: callable,
    device: torch.device,
    ablation_mode: int = 0,
):
    """Pre-compute and cache visual features for BiFTA with different crop strategies.

    Args:
        model: Vision-language model (CLIP, etc.)
        dataset_name: Name of the dataset
        model_size: Model architecture identifier
        crop_config: Dictionary containing crop strategy and parameters
            For random: {"strategy": "random", "alpha": float, "beta": float,
                        "n_samples": int, "iou_threshold": float, "max_queue_len": int}
            For grid: {"strategy": "grid", "num_grid": int}
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        data_path: Path to dataset
        custom_loader: Custom image loader function
        device: Torch device
        ablation_mode: If 1, include all hyperparameters in filename

    Returns:
        tuple: (patch_embeddings_with_weights, labels, global_image_features, num_valid_crops)
            - patch_embeddings_with_weights: [B, N, D+1] patch features + similarity weights
            - labels: [B] ground truth labels
            - global_image_features: [B, D] full image features
            - num_valid_crops: [B] number of real crops per image (excluding zero padding)
    """
    save_file = (dataset_name + "-" + model_size).replace("/", "-")
    save_root = f"features/tmlr_rebuttal/{dataset_name}/"

    # Create save directory if it doesn't exist
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Generate filename based on crop strategy and ablation mode
    strategy = crop_config["strategy"]

    if strategy == "random":
        if ablation_mode:
            filename = os.path.join(
                save_root,
                f"{save_file}-random-alpha{crop_config['alpha']:.1f}-beta{crop_config['beta']:.1f}"
                f"-n{crop_config['n_samples']}-iou{crop_config['iou_threshold']:.2f}-q{crop_config['max_queue_len']}.pkl"
            )
        else:
            filename = os.path.join(
                save_root,
                f"{save_file}-random-alpha{crop_config['alpha']:.1f}-beta{crop_config['beta']:.1f}"
                f"-n{crop_config['n_samples']}.pkl"
            )
    elif strategy == "grid":
        filename = os.path.join(
            save_root,
            f"{save_file}-grid{crop_config['num_grid']}x{crop_config['num_grid']}.pkl"
        )
    else:
        raise ValueError(f"Unknown crop strategy: {strategy}")

    if os.path.exists(filename):
        print(f"Loading {filename}...")
        load_res = pickle.load(open(filename, "rb"))
    else:
        print(f"File {filename} not found, precomputing features...")
        dataset = load_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            custom_loader=custom_loader,
        )

        # Custom collate function to handle (images, num_valid_crops) tuples
        def collate_fn(batch):
            # batch is a list of ((images, num_valid_crops), label) tuples
            images_list = []
            num_valid_crops_list = []
            labels_list = []

            for item in batch:
                (images, num_valid_crops), label = item
                images_list.append(images)
                num_valid_crops_list.append(num_valid_crops)
                labels_list.append(label)

            # Stack images and labels
            images_batch = torch.stack(images_list)
            labels_batch = torch.tensor(labels_list)
            num_valid_crops_batch = torch.tensor(num_valid_crops_list)

            return images_batch, labels_batch, num_valid_crops_batch

        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        patch_embeddings_list = []
        global_image_features_list = []
        labels_list = []
        num_valid_crops_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing features"):
                images, batch_labels, batch_num_valid_crops = batch
                images = images.to(device)
                batch_labels = batch_labels.to(device)

                batch_size, num_crops = images.shape[:2]
                images = images.flatten(0, 1)  # [B*N, C, H, W]

                # Encode images
                image_embeddings = model.encode_image(images)
                # image_embeddings = model(**images).image_embeds  # For ALIGN model

                # Normalize and reshape
                image_embeddings = F.normalize(image_embeddings)
                image_embeddings = image_embeddings.view(batch_size, num_crops, -1)  # [B, N, D]

                # Split: first crop is global image, rest are patches
                global_features = image_embeddings[:, :1]  # [B, 1, D]
                patch_features = image_embeddings[:, 1:]   # [B, N-1, D]

                # Compute similarity weights between global and patch features
                # Shape: [B, N-1, 1]
                similarity_weights = (global_features * patch_features).sum(
                    dim=-1, keepdim=True
                )

                # Concatenate patch features with their weights: [B, N-1, D+1]
                patch_embeddings_with_weights = torch.cat(
                    [patch_features, similarity_weights], dim=-1
                )

                patch_embeddings_list.append(patch_embeddings_with_weights)
                labels_list.append(batch_labels)
                global_image_features_list.append(global_features.squeeze(1))
                num_valid_crops_list.append(batch_num_valid_crops)

        load_res = {
            "patch_embeddings_with_weights": torch.cat(patch_embeddings_list, dim=0),
            "global_image_features": torch.cat(global_image_features_list, dim=0),
            "labels": torch.cat(labels_list, dim=0),
            "num_valid_crops": torch.cat(num_valid_crops_list, dim=0),
        }

        os.makedirs(save_root, exist_ok=True)
        pickle.dump(load_res, open(filename, "wb"))

    # Load cached features
    patch_embeddings_with_weights = load_res["patch_embeddings_with_weights"].to(device)
    labels = load_res["labels"].to(device)
    global_image_features = load_res["global_image_features"].to(device)
    num_valid_crops = load_res["num_valid_crops"].to(device)

    return patch_embeddings_with_weights, labels, global_image_features, num_valid_crops



def make_descriptor_sentence(descriptor):
    if descriptor.startswith("a") or descriptor.startswith("an"):
        return f"which is {descriptor}"
    elif (
        descriptor.startswith("has")
        or descriptor.startswith("often")
        or descriptor.startswith("typically")
        or descriptor.startswith("may")
        or descriptor.startswith("can")
    ):
        return f"which {descriptor}"
    elif descriptor.startswith("used"):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def zeroshot_classifier(
    model,
    textnames,
    templates=None,
    is_template=True,
    make_sentence=False,
    tt_scale=None,
    device=None,
):
    with torch.no_grad():
        zeroshot_weights = []
        for i in tqdm(range(len(textnames))):
            if not is_template: # is_template is false for ours
                texts = []
                for t in templates[textnames[i]]:
                    if make_sentence:
                        desc_sen = make_descriptor_sentence(t)
                        texts.append(f"{textnames[i]}, {desc_sen}")
                    else:
                        texts.append(t)
                # if tt_scale is not None:
                #     texts = random.sample(texts, 50) #TODO: Hard_code,ours only.
            elif templates:
                texts = [template.format(textnames[i]) for template in templates]
            else:
                texts = [f"a photo of a {textnames[i]}."]

            if i == 0:
                print(texts)

            if tt_scale is not None:
                label = f"a photo of a {textnames[i]}."
                label_tokens = clip.tokenize(label, truncate=True).to(device)

                label_embeddings = model.encode_text(label_tokens)

                # label_embeddings = model(**label_tokens).text_embeds #ALIGN usage

                label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)

            texts_tensor = clip.tokenize(texts, truncate=True).to(device)
            class_embeddings = model.encode_text(texts_tensor)

            # class_embeddings = model(**texts_tensor).text_embeds #ALIGN usage

            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            if tt_scale is not None:
                weight = class_embeddings @ label_embeddings.T
                weight = (weight * tt_scale).softmax(dim=0)
                # weight = torch.ones(weight.shape).to(device)
                class_embedding = (class_embeddings * weight).sum(dim=0)
                class_embedding /= class_embedding.norm()
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def construct_random(gpt3_prompts):
    """
    reference: https://github.com/ExplainableML/WaffleCLIP.git
    """
    key_list = list(gpt3_prompts.keys())

    # Get complete list of available descriptions.
    descr_list = [list(values) for values in gpt3_prompts.values()]
    descr_list = np.array([x for y in descr_list for x in y])

    ### Descriptor Makers.
    structured_descriptor_builder = (
        lambda item, cls: f"A photo of a {wordify(cls)}, {make_descriptor_sentence(item)}."
    )

    word_list = pickle.load(open("features/word_list.pkl", "rb"))

    avg_num_words = int(
        np.max(
            [
                np.round(np.mean([len(wordify(x).split(" ")) for x in key_list])),
                1,
            ]
        )
    )
    avg_word_length = int(
        np.round(
            np.mean(
                [np.mean([len(y) for y in wordify(x).split(" ")]) for x in key_list]
            )
        )
    )
    word_list = [x[:avg_word_length] for x in word_list]

    # (Lazy solution) Extract list of available random characters from gpt description list. Ideally we utilize a separate list.
    character_list = [x.split(" ") for x in descr_list]
    character_list = [
        x.replace(",", "").replace(".", "")
        for x in np.unique([x for y in character_list for x in y])
    ]
    character_list = np.unique(list("".join(character_list)))

    num_spaces = (
        int(np.round(np.mean([np.sum(np.array(list(x)) == " ") for x in key_list]))) + 1
    )
    num_chars = int(
        np.ceil(np.mean([np.max([len(y) for y in x.split(" ")]) for x in key_list]))
    )

    num_chars += num_spaces - num_chars % num_spaces
    sample_key = ""

    for s in range(num_spaces):
        for _ in range(num_chars // num_spaces):
            sample_key += "a"
        if s < num_spaces - 1:
            sample_key += " "

    gpt3_prompts = {key: [] for key in gpt3_prompts.keys()}

    for key in key_list:
        for _ in range(15):
            base_word = ""
            for a in range(avg_num_words):
                base_word += np.random.choice(word_list, 1, replace=False)[0]
                if a < avg_num_words - 1:
                    base_word += " "
            gpt3_prompts[key].append(structured_descriptor_builder(base_word, key))
            noise_word = ""
            use_key = sample_key if len(key) >= len(sample_key) else key
            for c in sample_key:
                if c != " ":
                    noise_word += np.random.choice(character_list, 1, replace=False)[0]
                else:
                    noise_word += ", "
            gpt3_prompts[key].append(structured_descriptor_builder(noise_word, key))

    match_key = np.random.choice(key_list)
    gpt3_prompts = {key: gpt3_prompts[match_key] for key in key_list}
    for key in gpt3_prompts:
        gpt3_prompts[key] = [
            x.replace(wordify(match_key), wordify(key)) for x in gpt3_prompts[key]
        ]

    return gpt3_prompts


def accuracy(output, target, n, dataset_name):
    # Get index of the maximum value as prediction
    if dataset_name.startswith(MyDataset.ImageNetA):
        _, pred = output[:, imagenet_a_lt].max(1)
    elif dataset_name.startswith(MyDataset.ImageNetR):
        _, pred = output[:, imagenet_r_lt].max(1)
    else:
        _, pred = output.max(1)
    # Compare prediction with target
    correct = pred.eq(target)
    # Calculate top-1 accuracy
    return float(correct.float().sum().cpu().numpy()) / n * 100



##################################################################################################################
##                        Additional Implementation for BiFTA
##################################################################################################################
from PIL import Image
from typing import List, Callable
from transformers import (
    AutoModel,
    AutoProcessor,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    BlipProcessor,
    Blip2Processor
)
from torchvision import datasets


def compute_IoU(boxes: list, box2: tuple) -> list:
    """
    Compute Intersection over Union (IoU) between multiple bounding boxes in `boxes` and a single bounding box `box2`.

    Args:
        boxes: List of tuples, each representing (x1, y1, x2, y2) for different bounding boxes.
        box2: A single bounding box in format (x1, y1, x2, y2).

    Returns:
        List of IoU values, each corresponding to the IoU of `box2` with one of the `boxes`.
    """
    iou_list = []
    
    for box1 in boxes:
        # Intersection box coordinates
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Union area
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        
        iou_list.append(iou)
    
    return iou_list




def random_crop_with_coordinate(image: Image.Image, beta: float = 0.9, alpha: float = 0.1):
    """Randomly crops an image within a size range determined by alpha and the image dimensions.

    Args:
        image (Image): The input image to crop.
        alpha (float): The minimum scale factor for the crop as a proportion of the smallest dimension.

    Returns:
        Tuple[Image.Image, Tuple[int, int, int, int]]: Cropped image and coordinates of the crop (left, upper, right, lower).
    """
    # Get the width and height of the original image
    w, h = image.size

    # Determine the size of the crop based on alpha and the smallest dimension
    n_px = np.random.uniform(low=alpha, high=beta) * min(h, w)

    # Generate random top-left corner for cropping
    left = np.random.randint(0, w - int(n_px) + 1)
    upper = np.random.randint(0, h - int(n_px) + 1)

    # Compute the right and lower bounds of the crop
    right = left + int(n_px)
    lower = upper + int(n_px)

    # Crop the image
    cropped = image.crop((left, upper, right, lower))

    return cropped, (left, upper, right, lower)




def grid_crop(image: Image.Image, num_grid: int = 3) -> List[Image.Image]:
    """
    Deterministically crops an image into num_grid x num_grid equally sized square patches.

    Args:
        image (Image.Image): Input image.
        num_grid (int): Number of grid divisions per dimension.

    Returns:
        List[Image.Image]: List of cropped square patches.
    """
    w, h = image.size
    patch_size = min(w // num_grid, h // num_grid)

    crops = []
    for gy in range(num_grid):
        for gx in range(num_grid):
            left = gx * patch_size
            upper = gy * patch_size
            right = left + patch_size
            lower = upper + patch_size
            crops.append(image.crop((left, upper, right, lower)))

    return crops




def load_huggingface_models(model_name: str, device: str = "cuda", **model_kwargs):

    huggingface_cache_path = None  # Uses default HuggingFace cache directory

    MODEL_PATHS = {
        "BLIP": "Salesforce/blip-image-captioning-base",
        "BLIP-2": "Salesforce/blip2-opt-2.7b",
        "ALIGN": "kakaobrain/align-base",
        "SigLIP": "google/siglip-base-patch16-224"
    }
    
    MODEL_LOADERS = {
        "BLIP": (BlipForConditionalGeneration, BlipProcessor),
        "BLIP-2": (Blip2ForConditionalGeneration, Blip2Processor)
    }
    
    try:
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Invalid model name: {model_name}. Available: {list(MODEL_PATHS.keys())}")
        
        model_path = MODEL_PATHS[model_name]

        
        if model_name in MODEL_LOADERS:
            model_class, processor_class = MODEL_LOADERS[model_name]
            model = model_class.from_pretrained(model_path, cache_dir = huggingface_cache_path, **model_kwargs).to(device)
            processor = processor_class.from_pretrained(model_path, cache_dir = huggingface_cache_path)
        else:
            print("Using auto model loader...")
            model = AutoModel.from_pretrained(model_path, cache_dir = huggingface_cache_path, **model_kwargs).to(device)
            processor = AutoProcessor.from_pretrained(model_path, cache_dir = huggingface_cache_path, return_tensors="pt")
        
        print(f"Successfully loaded {model_name} to {device}")
        return model, processor

    except Exception as e:
        raise RuntimeError(f"Failed to load {model_name}: {str(e)}")


def filter_augmented_samples(
    augmented_imgs: torch.Tensor,
    img_coordinates: list,
    threshold: float,
    max_queue_len: int = 50
) -> tuple:
    """Filter augmented image crops based on IoU threshold to ensure diversity.

    Args:
        augmented_imgs: Tensor of shape [N+1, C, H, W] where first image is original
        img_coordinates: List of crop coordinates [(x1, y1, x2, y2), ...]
        threshold: IoU threshold for filtering similar crops
        max_queue_len: Maximum number of diverse crops to keep

    Returns:
        tuple: (filtered_tensor, num_valid_crops)
            - filtered_tensor: Tensor of shape [max_queue_len+1, C, H, W] (original + diverse crops, zero-padded if needed)
            - num_valid_crops: int, number of real crops (excluding zero padding)
    """
    crop_samples = augmented_imgs[1:]
    crop_sample_coordinates = img_coordinates[1:]

    queue = []
    saved_coordinates = []
    filtered_augmented_imgs = [augmented_imgs[0]]  # Keep original image

    # Iterate through all crops and filter based on IoU threshold
    for i in range(len(crop_sample_coordinates)):
        next_coordinates = crop_sample_coordinates[i]

        # Accept first crop or crops with low IoU to all saved crops
        if len(saved_coordinates) == 0:
            queue.append(crop_samples[i])
            saved_coordinates.append(next_coordinates)
        else:
            iou = compute_IoU(saved_coordinates, next_coordinates)
            if max(iou) <= threshold:
                queue.append(crop_samples[i])
                saved_coordinates.append(next_coordinates)
                if len(queue) >= max_queue_len:
                    break

    # Record number of valid crops before padding
    num_valid_crops = len(queue)

    # Pad with zero tensors if filtered crops are fewer than max_queue_len
    if len(queue) < max_queue_len:
        num_samples_needed = max_queue_len - len(queue)
        # Create zero-filled tensor with same shape as a crop
        zero_crop = torch.zeros_like(crop_samples[0])
        for _ in range(num_samples_needed):
            queue.append(zero_crop)

    filtered_augmented_imgs.extend(queue)
    return torch.stack(filtered_augmented_imgs), num_valid_crops


def create_random_crop_loader(
    processor: Callable,
    n_samples: int,
    alpha: float,
    beta: float,
    iou_threshold: float,
    max_queue_len: int
) -> Callable:
    """Create a custom image loader with random crops filtered by IoU.

    Args:
        processor: Image preprocessing function (e.g., CLIP processor)
        n_samples: Number of random crops to generate
        alpha: Minimum crop scale factor (proportion of image size)
        beta: Maximum crop scale factor (proportion of image size)
        iou_threshold: IoU threshold for filtering similar crops
        max_queue_len: Maximum number of diverse crops to keep

    Returns:
        Custom loader function that returns [original + diverse random crops]
    """
    def custom_loader(path: str) -> torch.Tensor:
        """Load image and return original + filtered augmented crops.

        Args:
            path: Path to the image file

        Returns:
            Tensor of shape [max_queue_len+1, C, H, W]
        """
        # Load and process original image
        img = datasets.folder.default_loader(path)
        processed_img = processor(img)

        # For Non-Clip style encoder.
        if not isinstance(processed_img, torch.Tensor):
            augmented_imgs = [processed_img["pixel_values"][0]]
        else:
            augmented_imgs = [processed_img]

        img_coordinates = [None]  # Placeholder for original image

        # Generate random crops
        for _ in range(n_samples):
            crop_img, img_coordinate = random_crop_with_coordinate(
                img, alpha=alpha, beta=beta
            )
            crop_img = processor(crop_img)

            if not isinstance(crop_img, torch.Tensor):
                augmented_imgs.append(crop_img["pixel_values"][0])
            else:
                augmented_imgs.append(crop_img)

            img_coordinates.append(img_coordinate)

        augmented_imgs = torch.stack(augmented_imgs)

        # Filter to get diverse crops
        filtered_augmented_imgs, num_valid_crops = filter_augmented_samples(
            augmented_imgs,
            img_coordinates,
            threshold=iou_threshold,
            max_queue_len=max_queue_len
        )

        return filtered_augmented_imgs, num_valid_crops

    return custom_loader


def create_grid_crop_loader(
    processor: Callable,
    num_grid: int
) -> Callable:
    """Create a custom image loader with deterministic grid crops.

    Args:
        processor: Image preprocessing function (e.g., CLIP processor)
        num_grid: Number of grid divisions per dimension (e.g., 3 for 3x3 grid)

    Returns:
        Custom loader function that returns [original + grid crops]
    """
    def custom_loader(path: str) -> tuple:
        """Load image and return original + grid crops.

        Args:
            path: Path to the image file

        Returns:
            tuple: (images, num_valid_crops)
                - images: Tensor of shape [num_grid*num_grid+1, C, H, W]
                - num_valid_crops: int, number of grid crops (num_grid*num_grid)
        """
        # Load original image
        img = datasets.folder.default_loader(path)
        processed_img = processor(img)

        # Handle different processor output formats
        if not isinstance(processed_img, torch.Tensor):
            augmented_imgs = [processed_img["pixel_values"][0]]
        else:
            augmented_imgs = [processed_img]

        # Generate grid crops
        grid_crops = grid_crop(img, num_grid=num_grid)

        for crop_img in grid_crops:
            crop_img = processor(crop_img)

            if not isinstance(crop_img, torch.Tensor):
                augmented_imgs.append(crop_img["pixel_values"][0])
            else:
                augmented_imgs.append(crop_img)

        # Grid crops don't use filtering, so all crops are valid
        num_valid_crops = len(grid_crops)  # num_grid * num_grid

        return torch.stack(augmented_imgs), num_valid_crops

    return custom_loader