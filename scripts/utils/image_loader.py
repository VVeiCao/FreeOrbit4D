"""
图像和Mask加载工具

功能：
    提供保持宽高比的图像预处理函数，用于DPG/VGGT模型推理
"""

import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images_aspect_ratio(image_path_list, target_long_edge=518, divisor=14):
    """
    加载并预处理图像，保持宽高比，长边缩放到target_long_edge，padding到能被divisor整除
    
    流程：
      1. 等比缩放到长边=target_long_edge
      2. 四舍五入后padding到能被divisor整除
    
    Args:
        image_path_list (list): 图像路径列表
        target_long_edge (int): 目标长边尺寸，默认518
        divisor (int): 尺寸必须被divisor整除，默认14 (ViT patch size)
    
    Returns:
        tuple: (
            torch.Tensor: (N, 3, H_padded, W_padded) 预处理后的图像,
            torch.Tensor: (N, 2) [[H_padded, W_padded], ...] 推理尺寸（padding后）,
            torch.Tensor: (N, 2) [[H_scaled, W_scaled], ...] 缩放后尺寸（padding前）,
            torch.Tensor: (N, 2) [[H_orig, W_orig], ...] 原始尺寸,
            torch.Tensor: (N, 4) [[pad_top, pad_bottom, pad_left, pad_right], ...] padding量
        )
    
    示例：
        480×832 → 299×518 → padding到 308×518
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
        # 读取图像
        img = Image.open(image_path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        
        width, height = img.size
        original_sizes.append([height, width])
        
        # 步骤1：等比缩放到长边=target_long_edge
        if width >= height:
            new_width = target_long_edge
            new_height = round(height * (target_long_edge / width))
        else:
            new_height = target_long_edge
            new_width = round(width * (target_long_edge / height))
        
        img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        scaled_sizes.append([new_height, new_width])
        
        # 步骤2：padding到能被divisor整除
        padded_height = ((new_height + divisor - 1) // divisor) * divisor
        padded_width = ((new_width + divisor - 1) // divisor) * divisor
        padded_sizes.append([padded_height, padded_width])
        
        # 计算padding量（中心padding）
        pad_top = (padded_height - new_height) // 2
        pad_bottom = padded_height - new_height - pad_top
        pad_left = (padded_width - new_width) // 2
        pad_right = padded_width - new_width - pad_left
        pad_coords.append([pad_top, pad_bottom, pad_left, pad_right])
        
        # 创建padding后的图像（白色背景）
        img_padded = Image.new("RGB", (padded_width, padded_height), (255, 255, 255))
        img_padded.paste(img_resized, (pad_left, pad_top))
        
        # 转为tensor
        img_tensor = to_tensor(img_padded)
        images.append(img_tensor)
    
    # Stack并转为tensor
    images = torch.stack(images)
    padded_sizes = torch.tensor(padded_sizes, dtype=torch.int32)
    scaled_sizes = torch.tensor(scaled_sizes, dtype=torch.int32)
    original_sizes = torch.tensor(original_sizes, dtype=torch.int32)
    pad_coords = torch.tensor(pad_coords, dtype=torch.int32)
    
    return images, padded_sizes, scaled_sizes, original_sizes, pad_coords


def load_and_preprocess_masks_aspect_ratio(mask_path_list, target_long_edge=518, divisor=14):
    """
    加载并预处理mask，保持宽高比，长边缩放到target_long_edge，padding到能被divisor整除
    
    流程：
      1. 等比缩放到长边=target_long_edge
      2. 四舍五入后padding到能被divisor整除
    
    Args:
        mask_path_list (list): mask路径列表
        target_long_edge (int): 目标长边尺寸，默认518
        divisor (int): 尺寸必须被divisor整除，默认14 (ViT patch size)
    
    Returns:
        tuple: (
            torch.Tensor: (N, 1, H_padded, W_padded) 预处理后的mask,
            torch.Tensor: (N, 2) [[H_padded, W_padded], ...] 推理尺寸（padding后）,
            torch.Tensor: (N, 2) [[H_scaled, W_scaled], ...] 缩放后尺寸（padding前）,
            torch.Tensor: (N, 2) [[H_orig, W_orig], ...] 原始尺寸,
            torch.Tensor: (N, 4) [[pad_top, pad_bottom, pad_left, pad_right], ...] padding量
        )
    
    示例：
        480×832 → 299×518 → padding到 308×518
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
        # 读取mask
        mask = Image.open(mask_path).convert('L')
        
        width, height = mask.size
        original_sizes.append([height, width])
        
        # 步骤1：等比缩放到长边=target_long_edge
        if width >= height:
            new_width = target_long_edge
            new_height = round(height * (target_long_edge / width))
        else:
            new_height = target_long_edge
            new_width = round(width * (target_long_edge / height))
        
        mask_resized = mask.resize((new_width, new_height), Image.Resampling.BICUBIC)
        scaled_sizes.append([new_height, new_width])
        
        # 步骤2：padding到能被divisor整除
        padded_height = ((new_height + divisor - 1) // divisor) * divisor
        padded_width = ((new_width + divisor - 1) // divisor) * divisor
        padded_sizes.append([padded_height, padded_width])
        
        # 计算padding量（中心padding）
        pad_top = (padded_height - new_height) // 2
        pad_bottom = padded_height - new_height - pad_top
        pad_left = (padded_width - new_width) // 2
        pad_right = padded_width - new_width - pad_left
        pad_coords.append([pad_top, pad_bottom, pad_left, pad_right])
        
        # 创建padding后的mask（黑色背景）
        mask_padded = Image.new('L', (padded_width, padded_height), 0)
        mask_padded.paste(mask_resized, (pad_left, pad_top))
        
        # 转为tensor
        mask_tensor = to_tensor(mask_padded)
        masks.append(mask_tensor)
    
    # Stack并转为tensor
    masks = torch.stack(masks)
    padded_sizes = torch.tensor(padded_sizes, dtype=torch.int32)
    scaled_sizes = torch.tensor(scaled_sizes, dtype=torch.int32)
    original_sizes = torch.tensor(original_sizes, dtype=torch.int32)
    pad_coords = torch.tensor(pad_coords, dtype=torch.int32)
    
    return masks, padded_sizes, scaled_sizes, original_sizes, pad_coords



