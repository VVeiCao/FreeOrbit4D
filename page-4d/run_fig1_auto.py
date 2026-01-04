# In this one we test the performance on dynamic 
import os
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import numpy as np
from vggt.utils.load_fn import load_and_preprocess_images
import json
from utils.metrics import *
from utils.visual import *
from vggt_t_mask_mlp_fin10.models.vggt import VGGT as DPG

# sudo chown -R kz1024 /PHShome/kz1024
def backproject_depth_to_points_batch(depths, intrinsics, extrinsics_3x4):
    # depths: (B, H, W)
    # intrinsics: (B, 3, 3)
    # extrinsics_3x4: (B, 3, 4)
    # return: (B, H*W, 3)
    B, H, W = depths.shape
    device = depths.device
    # Step 1: Create pixel grid (u,v,1) -> [H, W, 3]
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    ones = torch.ones_like(x)
    pixels = torch.stack([x, y, ones], dim=-1)  # [H, W, 3]
    pixels = pixels.reshape(1, H*W, 3).expand(B, -1, -1).float()  # [B, H*W, 3]
    # Step 2: Inverse intrinsics and pixel -> cam coordinates
    K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
    cam_coords = torch.bmm(pixels, K_inv.transpose(1, 2))  # [B, H*W, 3]
    # Step 3: Depth * cam_coords
    z = depths.reshape(B, -1, 1)  # [B, H*W, 1]
    cam_points = cam_coords * z  # [B, H*W, 3]
    # Step 4: Convert to homogeneous coordinates
    ones = torch.ones((B, cam_points.shape[1], 1), device=device, dtype=cam_points.dtype)
    cam_points_h = torch.cat([cam_points, ones], dim=-1)  # [B, H*W, 4]
    # Step 5: Convert extrinsic 3x4 to 4x4 and invert
    bottom = torch.tensor([[0, 0, 0, 1]], device=device, dtype=cam_points.dtype).expand(B, 1, 4)  # [B, 1, 4]
    extrinsics_4x4 = torch.cat([extrinsics_3x4, bottom], dim=1)  # [B, 4, 4]
    T_wc = torch.inverse(extrinsics_4x4)  # [B, 4, 4]
    # Step 6: Transform to world coordinates
    world_points_h = torch.bmm(cam_points_h, T_wc.transpose(1, 2))  # [B, H*W, 4]
    world_points = world_points_h[:, :, :3]  # [B, H*W, 3]
    return world_points  # [B, H*W, 3]

def depth_to_rgb(depth, dir_name):
    depth_normalized = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    depth_colormap = cv2.applyColorMap(
        (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    cv2.imwrite(dir_name, depth_colormap)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_tensor_image(tensor_img, save_path):
    """
    Save a PyTorch tensor image (3, H, W) to a file.
    Args:
        tensor_img (torch.Tensor): Image tensor with shape (3, H, W), 
                                   values in [0,1] or [-1,1].
        save_path (str): Full file path including filename, e.g. './output/x.png'
    """
    # ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # move to cpu and clone to avoid modifying original tensor
    img = tensor_img.detach().cpu().clone()
    # handle normalized input [-1,1] → [0,1]
    if img.min() < 0:
        img = (img + 1) / 2.0
    img = img.clamp(0, 1)
    # convert to numpy (H, W, 3)
    img_np = (img.numpy() * 255).astype(np.uint8)
    # convert RGB → BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # write image
    cv2.imwrite(save_path, img_bgr)
    # print(f"✅ Saved image to {save_path}")

def process(model, image_names, device, directory='image', name='vggt'):
    images = load_and_preprocess_images(image_names).to(device)
    images = images.unsqueeze(0)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
    depth_map = depth_map.squeeze(-1)
    world_points = backproject_depth_to_points_batch(depth_map[0], intrinsic[0], extrinsic[0])
    # B, H*W, 3
    point_map = point_map.squeeze(0)
    point_map = point_map.reshape(depth_map.size(1), -1, 3)
    images = images.squeeze(0)
    images = images.permute(0,2,3,1).reshape(depth_map.size(1), -1, 3)
    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, name), exist_ok=True)
    ext_Dict = {}
    int_Dict = {}
    for num in range(len(images)):
        num_name = image_names[num].split("/")[-1].split(".")[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points[num][::2].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(images[num][::2].cpu().numpy())
        o3d.io.write_point_cloud(f"{directory}/{name}/{num}_dep.ply", pcd)
        # depth_to_rgb(depth_map[0][num].cpu().numpy(), f"{directory}/{name}/{num_name}_depth.png")
        save_tensor_image(images[num].reshape(depth_map[0][num].size(0), depth_map[0][num].size(1), 3), f"{directory}/{name}/{num_name}_images.png")
        ext_Dict[num_name] = extrinsic[0][num].cpu().numpy()
        int_Dict[num_name] = intrinsic[0][num].cpu().numpy()
    ext_Dict_clean = {str(k): np.asarray(v).astype(float).tolist() for k, v in ext_Dict.items()}
    int_Dict_clean = {str(k): np.asarray(v).astype(float).tolist() for k, v in int_Dict.items()}
    with open(f"{directory}/{name}/{num_name}_ext.json", "w") as f:
        json.dump(ext_Dict_clean, f, indent=4)
    with open(f"{directory}/{name}/{num_name}_int.json", "w") as f:
        json.dump(int_Dict_clean, f, indent=4)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for data_name in ["online_img3"]:
            if data_name == "online_img3":
                directory = f"/workspace/code/12_4d/vggt_visual/{data_name}/"

            folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f)) and not f.startswith('fig1_')]
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            device = "cuda" if torch.cuda.is_available() else "cpu"
            origin = "checkpoint/checkpoint_150.pt"
            model = DPG()
            checkpoint = torch.load(origin, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=False)
            model.to(device)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model2 = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
            for category in folders:
                if data_name.startswith("online_img"):
                    image_names = [f for f in os.listdir(os.path.join(directory, category)) if f.endswith('.png') or f.endswith('.jpg')]
                    try:
                        image_names.sort(key=lambda x: int(os.path.splitext(x)[0]))
                    except:
                        image_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    image_names = [os.path.join(directory, category, f) for f in image_names][:24]
                process(model, image_names, device, directory=f'{data_name}/fig1_update_dpg', name=f'dpg_{category}')
                process(model2, image_names, device, directory=f'{data_name}/fig1_update_vggt', name=f'vggt_{category}')
