from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def draw_corresponding_tracks(image1, image2, pts1, pts2, radius=3):
    """
    Combine image1 and image2 horizontally and draw lines connecting pts1 ↔ pts2.
    - pts1, pts2: tensors of shape (N, 2), coordinates in image1 and image2.
    """
    # Combine images horizontally
    W1, H1 = image1.size
    W2, H2 = image2.size
    H = max(H1, H2)
    combined = Image.new('RGB', (W1 + W2, H))
    combined.paste(image1, (0, 0))
    combined.paste(image2, (W1, 0))
    
    draw = ImageDraw.Draw(combined)
    pts1 = pts1.cpu().numpy()
    pts2 = pts2.cpu().numpy()
    
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))

        # Offset x2 by W1 to match combined image
        x2_offset = x2 + W1

        # Draw points
        draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=(255, 0, 0))
        draw.ellipse((x2_offset - radius, y2 - radius, x2_offset + radius, y2 + radius), fill=(0, 255, 0))

        # Draw line between points
        draw.line((x1, y1, x2_offset, y2), fill=(0, 0, 255), width=1)

    return combined

def normalize_feature(feature):
    return (feature - feature.min()) / (feature.max() - feature.min())

def save_mask_as_image(mask: torch.Tensor, filename='mask.png'):
    # Ensure it's on CPU and convert to numpy
    mask_np = mask.detach().cpu().numpy().astype(np.uint8) * 255  # [0, 1] -> [0, 255]
    # Save as grayscale PNG
    cv2.imwrite(filename, mask_np)

def save_depth_as_rgb(depth: torch.Tensor, save_path: str, cmap_name='plasma'):
    """
    Save a single-channel depth tensor (H, W, 1) as RGB PNG.
    Args:
        depth: torch.Tensor of shape (H, W, 1)
        save_path: str, output file path ending with .png
        cmap_name: str, e.g., 'plasma', 'viridis', 'inferno'
    """
    depth = depth.squeeze(-1)  # (294, 518)
    depth_np = depth.detach().cpu().numpy()
    # Normalize to [0, 1]
    depth_min, depth_max = depth_np.min(), depth_np.max()
    depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    depth_rgb = cmap(depth_norm)[:, :, :3]  # drop alpha channel → (294, 518, 3)
    # Convert to uint8
    depth_rgb_uint8 = (depth_rgb * 255).astype(np.uint8)
    # Save using PIL
    img = Image.fromarray(depth_rgb_uint8)
    img.save(save_path)


def confidence_to_rgb(confidence_map: torch.Tensor, cmap_name='Reds') -> torch.Tensor:
    """
    Converts a 2D confidence map (H, W) to an RGB image (H, W, 3).
    Args:
        confidence_map: torch.Tensor of shape (H, W), values in [0, 1] or [min, max]
        cmap_name: str, e.g. 'Reds', 'plasma', 'hot', 'viridis', etc.
    Returns: torch.Tensor of shape (H, W, 3) with RGB values in [0, 1]
    """
    # Normalize confidence to [0, 1]
    conf = confidence_map.clone()
    conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    # Convert to numpy and apply colormap
    conf_np = conf.cpu().numpy()
    colormap = matplotlib.colormaps.get_cmap(cmap_name)
    colored = colormap(conf_np)  # (H, W, 4), includes alpha
    # Remove alpha and convert to torch
    rgb = torch.from_numpy(colored[:, :, :3]).float()
    return rgb