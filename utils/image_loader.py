"""Image and mask loading with aspect-ratio-preserving preprocessing."""

import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images_aspect_ratio(image_path_list, target_long_edge=518, divisor=14):
    """Load images, scale long edge to target_long_edge, and pad to be divisible by divisor.

    Returns:
        tuple: (images [N,3,H,W], padded_sizes [N,2], scaled_sizes [N,2],
                original_sizes [N,2], pad_coords [N,4])
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    padded_sizes = []
    scaled_sizes = []
    original_sizes = []
    pad_coords = []
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        original_sizes.append([height, width])

        if width >= height:
            new_width = target_long_edge
            new_height = round(height * (target_long_edge / width))
        else:
            new_height = target_long_edge
            new_width = round(width * (target_long_edge / height))

        img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        scaled_sizes.append([new_height, new_width])

        padded_height = ((new_height + divisor - 1) // divisor) * divisor
        padded_width = ((new_width + divisor - 1) // divisor) * divisor
        padded_sizes.append([padded_height, padded_width])

        pad_top = (padded_height - new_height) // 2
        pad_bottom = padded_height - new_height - pad_top
        pad_left = (padded_width - new_width) // 2
        pad_right = padded_width - new_width - pad_left
        pad_coords.append([pad_top, pad_bottom, pad_left, pad_right])

        img_padded = Image.new("RGB", (padded_width, padded_height), (255, 255, 255))
        img_padded.paste(img_resized, (pad_left, pad_top))

        img_tensor = to_tensor(img_padded)
        images.append(img_tensor)

    images = torch.stack(images)
    padded_sizes = torch.tensor(padded_sizes, dtype=torch.int32)
    scaled_sizes = torch.tensor(scaled_sizes, dtype=torch.int32)
    original_sizes = torch.tensor(original_sizes, dtype=torch.int32)
    pad_coords = torch.tensor(pad_coords, dtype=torch.int32)

    return images, padded_sizes, scaled_sizes, original_sizes, pad_coords


def load_and_preprocess_masks_aspect_ratio(mask_path_list, target_long_edge=518, divisor=14):
    """Load masks, scale long edge to target_long_edge, and pad to be divisible by divisor.

    Returns:
        tuple: (masks [N,1,H,W], padded_sizes [N,2], scaled_sizes [N,2],
                original_sizes [N,2], pad_coords [N,4])
    """
    if len(mask_path_list) == 0:
        raise ValueError("At least 1 mask is required")

    masks = []
    padded_sizes = []
    scaled_sizes = []
    original_sizes = []
    pad_coords = []
    to_tensor = TF.ToTensor()

    for mask_path in mask_path_list:
        mask = Image.open(mask_path).convert('L')

        width, height = mask.size
        original_sizes.append([height, width])

        if width >= height:
            new_width = target_long_edge
            new_height = round(height * (target_long_edge / width))
        else:
            new_height = target_long_edge
            new_width = round(width * (target_long_edge / height))

        mask_resized = mask.resize((new_width, new_height), Image.Resampling.BICUBIC)
        scaled_sizes.append([new_height, new_width])

        padded_height = ((new_height + divisor - 1) // divisor) * divisor
        padded_width = ((new_width + divisor - 1) // divisor) * divisor
        padded_sizes.append([padded_height, padded_width])

        pad_top = (padded_height - new_height) // 2
        pad_bottom = padded_height - new_height - pad_top
        pad_left = (padded_width - new_width) // 2
        pad_right = padded_width - new_width - pad_left
        pad_coords.append([pad_top, pad_bottom, pad_left, pad_right])

        mask_padded = Image.new('L', (padded_width, padded_height), 0)
        mask_padded.paste(mask_resized, (pad_left, pad_top))

        mask_tensor = to_tensor(mask_padded)
        masks.append(mask_tensor)

    masks = torch.stack(masks)
    padded_sizes = torch.tensor(padded_sizes, dtype=torch.int32)
    scaled_sizes = torch.tensor(scaled_sizes, dtype=torch.int32)
    original_sizes = torch.tensor(original_sizes, dtype=torch.int32)
    pad_coords = torch.tensor(pad_coords, dtype=torch.int32)

    return masks, padded_sizes, scaled_sizes, original_sizes, pad_coords
