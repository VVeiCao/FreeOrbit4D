#!/usr/bin/env python3
"""
步骤1.0可视化：前景点云可视化工具（Viser）

功能：
    可视化步骤1.0 (1_0_run_foreground.py) 生成的点云数据
    
输入：
    {frame_id}/pointcloud/{frame_id}_foreground_5_views.ply  # 5视角合并点云（用于可视化）
    
注意：
    本脚本只使用PLY文件进行可视化
    NPZ文件({frame_id}_foreground_5_views.npz)包含完整mapping数据，供后续步骤使用
    
特性：
    - 按帧浏览（时间步控制）
    - 4D/3D切换模式
    - AABB包围盒显示
    - 实时播放动画

使用示例：
    # 单个场景
    python scripts/1_0_visualization.py \
        --data_dir outputs/prepared/camel 

    python scripts/1_0_visualization.py \
        --data_dir outputs/prepared/bear 

    # 批量启动（不同端口）
    python scripts/1_0_visualization.py --data_dir outputs/prepared/camel --port 8080 --show_bbox &
    python scripts/1_0_visualization.py --data_dir outputs/prepared/bear --port 8081 --show_bbox &
    python scripts/1_0_visualization.py --data_dir outputs/prepared/cows --port 8082 --show_bbox &
    python scripts/1_0_visualization.py --data_dir outputs/prepared/hike --port 8083 --show_bbox &

主要参数：
    --data_dir: 数据目录（步骤1.0的输出）
    --port: Viser服务器端口（默认8080）
    --show_bbox: 显示AABB包围盒
    --num_frames: 限制加载帧数
    --subsample: 点云下采样率
"""

import os
import sys
import glob
import time
import argparse
from typing import List, Dict
import numpy as np
import trimesh
import viser


def parse_args():
    parser = argparse.ArgumentParser(description="步骤1.0 前景点云可视化")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录（步骤1.0的输出）")
    parser.add_argument("--port", type=int, default=8080, help="Viser端口（默认8080）")
    parser.add_argument("--point_size", type=float, default=0.004, help="点大小（默认0.004）")
    parser.add_argument("--fps", type=float, default=5.0, help="播放帧率（默认5.0）")
    parser.add_argument("--subsample", type=int, default=10, help="点云下采样率（默认10）")
    parser.add_argument("--num_frames", type=int, default=None, help="加载帧数（默认全部）")
    parser.add_argument("--show_bbox", action="store_true", help="显示AABB包围盒")
    return parser.parse_args()


class ForegroundPointCloudViewer:
    """前景点云可视化器（5视角合并）"""
    
    def __init__(
        self,
        data_dir: str,
        port: int = 8080,
        point_size: float = 0.003,
        fps: float = 5.0,
        subsample: int = 10,
        num_frames: int = None,
        show_bbox: bool = False
    ):
        self.data_dir = data_dir
        self.port = port
        self.point_size = point_size
        self.fps = fps
        self.subsample = subsample
        self.num_frames_limit = num_frames
        self.show_bbox = show_bbox
        
        # 解析帧数据
        self.frames_data = self._parse_frames()
        self.num_frames = len(self.frames_data)
        
        if self.num_frames == 0:
            raise ValueError(f"未找到任何有效的前景点云文件")
        
        # 加载点云数据
        self.pointcloud_data_list = self._load_pointclouds()
        
        # 用于debug的bbox中点列表
        self.bbox_centers = []
        
        # 启动Viser服务器
        self.server = self._setup_viser()
    
    def _parse_frames(self) -> List[Dict]:
        """解析帧数据，查找前景点云文件"""
        print("\n🔧 扫描帧目录...")
        subdirs = sorted([d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()])
        if self.num_frames_limit:
            subdirs = subdirs[:self.num_frames_limit]
        
        frames = []
        for subdir in subdirs:
            fg_path = os.path.join(self.data_dir, subdir, 'pointcloud', f"{subdir}_foreground_5_views.ply")
            if os.path.exists(fg_path):
                frames.append({'frame_id': subdir, 'foreground_path': fg_path})
        
        print(f"✅ 找到 {len(frames)} 个有效帧")
        return frames
    
    def _load_pointclouds(self) -> List[Dict]:
        """加载所有帧的前景点云"""
        print("\n🔧 加载点云...")
        pointcloud_list = []
        
        for idx, frame_data in enumerate(self.frames_data):
            frame_id = frame_data['frame_id']
            try:
                mesh = trimesh.load(frame_data['foreground_path'])
                fg_points = np.array(mesh.vertices)
                fg_colors = (np.array(mesh.visual.vertex_colors)[:, :3] if hasattr(mesh, 'visual') 
                            and hasattr(mesh.visual, 'vertex_colors') 
                            else np.ones((len(fg_points), 3), dtype=np.uint8) * 255)
                
                if self.subsample > 1:
                    indices = np.arange(0, len(fg_points), self.subsample)
                    fg_points, fg_colors = fg_points[indices], fg_colors[indices]
                
                pointcloud_list.append({
                    'frame_id': frame_id, 'frame_idx': idx,
                    'foreground': {'points': fg_points, 'colors': fg_colors}
                })
            except Exception as e:
                print(f"❌ 帧 {frame_id} 加载失败: {e}")
                pointcloud_list.append({'frame_id': frame_id, 'frame_idx': idx, 'foreground': None})
        
        total_points = sum(len(pc['foreground']['points']) for pc in pointcloud_list if pc['foreground'] is not None)
        print(f"✅ 加载完成: {len(pointcloud_list)} 帧，共 {total_points:,} 点")
        
        return pointcloud_list
    
    def _setup_viser(self) -> viser.ViserServer:
        """设置Viser服务器"""
        print(f"\n🔧 启动Viser服务器（端口: {self.port}）...")
        server = viser.ViserServer(host="0.0.0.0", port=self.port)
        server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        
        self._create_gui(server)
        self._create_scene(server)
        
        print(f"✅ Viser服务器启动成功")
        
        return server
    
    def _create_gui(self, server: viser.ViserServer):
        """创建GUI"""
        # 时间控制
        with server.gui.add_folder("⏱️ 时间控制"):
            self.gui_timestep = server.gui.add_slider(
                "帧序号",
                min=0,
                max=self.num_frames - 1,
                step=1,
                initial_value=0,
            )
            
            self.gui_frame_id_label = server.gui.add_text(
                "当前帧ID",
                initial_value=str(self.pointcloud_data_list[0]['frame_id']),
                disabled=True,
            )
            
            self.gui_playing = server.gui.add_checkbox("▶️ 播放", False)
            self.gui_framerate = server.gui.add_slider("FPS", min=1, max=30, step=0.5, initial_value=self.fps)
        
        # 显示控制
        with server.gui.add_folder("🎨 显示控制"):
            self.gui_show_mode = server.gui.add_button_group(
                "显示模式", 
                ("4D (当前帧)", "3D (所有帧累积)")
            )
            
            self.gui_point_size = server.gui.add_slider(
                "点云大小",
                min=0.001,
                max=0.01,
                step=0.0001,
                initial_value=self.point_size
            )
        
        # 统计信息
        with server.gui.add_folder("📊 统计信息"):
            total_fg_points = sum(
                len(pc['foreground']['points']) 
                for pc in self.pointcloud_data_list 
                if pc['foreground'] is not None
            )
            
            num_fg_frames = sum(1 for pc in self.pointcloud_data_list if pc['foreground'] is not None)
            
            info_text = f"步骤: 1.0 (5视角合并)\n"
            info_text += f"总帧数: {self.num_frames}\n"
            info_text += f"前景帧数: {num_fg_frames}\n"
            info_text += f"前景总点数: {total_fg_points:,}\n"
            info_text += f"平均每帧: {total_fg_points // num_fg_frames:,} 点"
            
            self.gui_stats = server.gui.add_text(
                "统计",
                initial_value=info_text,
                disabled=True,
            )
        
        self.server = server
    
    def _create_scene(self, server: viser.ViserServer):
        """创建3D场景"""
        self.foreground_handles = []
        self.bbox_handles = []
        
        for idx, pc_data in enumerate(self.pointcloud_data_list):
            frame_id = pc_data['frame_id']
            
            # 添加前景点云
            if pc_data['foreground'] is not None:
                fg_handle = server.scene.add_point_cloud(
                    name=f"/pointcloud/frame_{frame_id}/foreground",
                    points=pc_data['foreground']['points'],
                    colors=pc_data['foreground']['colors'],
                    point_size=self.point_size,
                    point_shape="circle",
                )
                self.foreground_handles.append(fg_handle)
                
                # 为每一帧创建AABB（如果启用）
                if self.show_bbox:
                    bbox_handle = self._create_bbox_for_frame(
                        server, 
                        pc_data['foreground']['points'], 
                        frame_id, 
                        idx
                    )
                    self.bbox_handles.append(bbox_handle)
                else:
                    self.bbox_handles.append(None)
            else:
                self.foreground_handles.append(None)
                self.bbox_handles.append(None)
        
        # 初始显示
        self._update_display()
        self._bind_events()
        
        # Debug: 检查所有bbox中点是否相同
        if self.show_bbox and len(self.bbox_centers) > 0:
            self._debug_check_bbox_centers()
    
    def _create_bbox_for_frame(self, server: viser.ViserServer, points: np.ndarray, frame_id: str, frame_idx: int):
        """为单帧点云创建AABB包围盒"""
        # 计算AABB
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        
        # 计算并存储bbox中点（用于debug）
        bbox_center = (bbox_min + bbox_max) / 2.0
        self.bbox_centers.append({
            'frame_id': frame_id,
            'frame_idx': frame_idx,
            'center': bbox_center
        })
        
        if frame_idx == 0:
            print(f"\n🔧 创建AABB包围盒...")
        
        # 生成AABB边框线点
        def create_bbox_lines(bbox_min, bbox_max, color, num_points_per_edge=30):
            """生成包围盒的12条边的点"""
            x_min, y_min, z_min = bbox_min
            x_max, y_max, z_max = bbox_max
            
            # 8个顶点
            vertices = np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max],
            ])
            
            # 12条边的端点索引
            edges = [
                (0,1), (1,2), (2,3), (3,0),  # 底面
                (4,5), (5,6), (6,7), (7,4),  # 顶面
                (0,4), (1,5), (2,6), (3,7),  # 竖边
            ]
            
            # 为每条边生成插值点
            line_points = []
            for v1_idx, v2_idx in edges:
                v1, v2 = vertices[v1_idx], vertices[v2_idx]
                # 在两个顶点之间插值
                for t in np.linspace(0, 1, num_points_per_edge):
                    point = v1 + t * (v2 - v1)
                    line_points.append(point)
            
            line_points = np.array(line_points)
            line_colors = np.tile(color, (len(line_points), 1))
            
            return line_points, line_colors
        
        # 使用渐变色区分不同帧（从黄色渐变到红色）
        # 黄色(255,255,0) → 橙色(255,165,0) → 红色(255,0,0)
        ratio = frame_idx / max(self.num_frames - 1, 1)
        color = [255, int(255 * (1 - ratio * 0.6)), 0]  # 从黄色到橙色
        
        bbox_points, bbox_colors = create_bbox_lines(bbox_min, bbox_max, color)
        
        # 添加到场景
        bbox_handle = server.scene.add_point_cloud(
            name=f"/bbox/frame_{frame_id}_bbox",
            points=bbox_points,
            colors=bbox_colors,
            point_size=self.point_size * 1.5,  # 稍微大一点
            point_shape="circle",
        )
        
        if frame_idx == self.num_frames - 1:
            print(f"✅ 已为 {self.num_frames} 帧创建AABB包围盒（黄→橙渐变色）")
        
        return bbox_handle
    
    def _bind_events(self):
        """绑定GUI事件"""
        @self.gui_timestep.on_update
        def _(_) -> None:
            self._update_display()
        
        @self.gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = self.gui_playing.value
        
        @self.gui_show_mode.on_click
        def _(_) -> None:
            self._update_display()
        
        @self.gui_point_size.on_update
        def _(_) -> None:
            self._update_point_size()
    
    def _debug_check_bbox_centers(self):
        """Debug: 检查所有bbox中点是否相同"""
        print("\n" + "=" * 80)
        print("🔍 DEBUG: 检查所有BBOX中点")
        print("=" * 80)
        
        if len(self.bbox_centers) == 0:
            print("  ⚠️  没有bbox中点数据")
            return
        
        # 提取所有中点坐标
        centers = np.array([bc['center'] for bc in self.bbox_centers])
        
        # 输出前5个和后5个中点
        print(f"\n📊 总共 {len(centers)} 个bbox中点")
        print(f"\n前5个bbox中点:")
        for i in range(min(5, len(self.bbox_centers))):
            bc = self.bbox_centers[i]
            print(f"  帧 {bc['frame_id']:>3} (idx={bc['frame_idx']:>2}): [{bc['center'][0]:>8.4f}, {bc['center'][1]:>8.4f}, {bc['center'][2]:>8.4f}]")
        
        if len(self.bbox_centers) > 5:
            print(f"\n后5个bbox中点:")
            for i in range(max(0, len(self.bbox_centers)-5), len(self.bbox_centers)):
                bc = self.bbox_centers[i]
                print(f"  帧 {bc['frame_id']:>3} (idx={bc['frame_idx']:>2}): [{bc['center'][0]:>8.4f}, {bc['center'][1]:>8.4f}, {bc['center'][2]:>8.4f}]")
        
        # 计算统计信息
        mean_center = np.mean(centers, axis=0)
        std_center = np.std(centers, axis=0)
        
        print(f"\n📈 统计信息:")
        print(f"  平均中点: [{mean_center[0]:>8.4f}, {mean_center[1]:>8.4f}, {mean_center[2]:>8.4f}]")
        print(f"  标准差:   [{std_center[0]:>8.4f}, {std_center[1]:>8.4f}, {std_center[2]:>8.4f}]")
        
        # 计算所有中点到平均中点的距离
        distances = np.linalg.norm(centers - mean_center, axis=1)
        max_distance = np.max(distances)
        max_idx = np.argmax(distances)
        
        print(f"\n📏 距离分析:")
        print(f"  最大偏离距离: {max_distance:.6f}")
        print(f"  最大偏离帧:   帧 {self.bbox_centers[max_idx]['frame_id']} (idx={self.bbox_centers[max_idx]['frame_idx']})")
        print(f"  平均偏离距离: {np.mean(distances):.6f}")
        
        # 判断是否所有中点基本相同
        threshold = 1e-4  # 设置阈值
        if max_distance < threshold:
            print(f"\n✅ 结论: 所有bbox中点基本相同 (最大偏离 < {threshold})")
        else:
            print(f"\n❌ 结论: bbox中点不完全相同 (最大偏离 = {max_distance:.6f} >= {threshold})")
            
            # 如果不相同，显示偏离最大的几个帧
            sorted_indices = np.argsort(distances)[::-1]  # 降序
            print(f"\n  偏离最大的5个帧:")
            for i in range(min(5, len(sorted_indices))):
                idx = sorted_indices[i]
                bc = self.bbox_centers[idx]
                dist = distances[idx]
                print(f"    帧 {bc['frame_id']:>3} (idx={bc['frame_idx']:>2}): 偏离距离 = {dist:.6f}")
        
        print("=" * 80 + "\n")
    
    def _update_point_size(self):
        """更新点云大小（通过重新创建点云实现）"""
        new_size = self.gui_point_size.value
        print(f"\n🔧 更新点云大小: {new_size:.4f}")
        
        # Viser 不支持动态修改 point_size，需要重新创建点云
        # 删除旧的点云
        for handle in self.foreground_handles:
            if handle is not None:
                handle.remove()
        self.foreground_handles.clear()
        
        # 删除旧的bbox
        for handle in self.bbox_handles:
            if handle is not None:
                handle.remove()
        self.bbox_handles.clear()
        
        # 重新创建点云和bbox
        for idx, pc_data in enumerate(self.pointcloud_data_list):
            frame_id = pc_data['frame_id']
            
            if pc_data['foreground'] is not None:
                fg_handle = self.server.scene.add_point_cloud(
                    name=f"/pointcloud/frame_{frame_id}/foreground",
                    points=pc_data['foreground']['points'],
                    colors=pc_data['foreground']['colors'],
                    point_size=new_size,
                    point_shape="circle",
                )
                self.foreground_handles.append(fg_handle)
                
                # 重新创建bbox
                if self.show_bbox:
                    bbox_handle = self._create_bbox_for_frame_with_size(
                        pc_data['foreground']['points'], 
                        frame_id, 
                        idx,
                        new_size
                    )
                    self.bbox_handles.append(bbox_handle)
                else:
                    self.bbox_handles.append(None)
            else:
                self.foreground_handles.append(None)
                self.bbox_handles.append(None)
        
        # 更新显示状态
        self._update_display()
        
        print(f"✅ 已重建 {len([h for h in self.foreground_handles if h is not None])} 个前景点云")
    
    def _create_bbox_for_frame_with_size(self, points: np.ndarray, frame_id: str, frame_idx: int, point_size: float):
        """用指定大小为单帧创建包围盒"""
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        
        def create_bbox_lines(bbox_min, bbox_max, color, num_points_per_edge=30):
            x_min, y_min, z_min = bbox_min
            x_max, y_max, z_max = bbox_max
            
            vertices = np.array([
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max],
            ])
            
            edges = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7),
            ]
            
            line_points = []
            for v1_idx, v2_idx in edges:
                v1, v2 = vertices[v1_idx], vertices[v2_idx]
                for t in np.linspace(0, 1, num_points_per_edge):
                    point = v1 + t * (v2 - v1)
                    line_points.append(point)
            
            line_points = np.array(line_points)
            line_colors = np.tile(color, (len(line_points), 1))
            return line_points, line_colors
        
        # 使用渐变色
        ratio = frame_idx / max(self.num_frames - 1, 1)
        color = [255, int(255 * (1 - ratio * 0.6)), 0]
        
        bbox_points, bbox_colors = create_bbox_lines(bbox_min, bbox_max, color)
        
        bbox_handle = self.server.scene.add_point_cloud(
            name=f"/bbox/frame_{frame_id}_bbox",
            points=bbox_points,
            colors=bbox_colors,
            point_size=point_size * 1.5,
            point_shape="circle",
        )
        
        return bbox_handle
    
    def _update_display(self):
        """更新显示状态"""
        current_timestep = self.gui_timestep.value
        show_mode = self.gui_show_mode.value
        
        # 更新当前帧ID标签
        current_pc = self.pointcloud_data_list[current_timestep]
        self.gui_frame_id_label.value = current_pc['frame_id']
        
        with self.server.atomic():
            # 更新前景显示
            for i, handle in enumerate(self.foreground_handles):
                if handle is None:
                    continue
                
                if show_mode == "4D (当前帧)":
                    handle.visible = (i == current_timestep)
                else:  # 3D模式
                    handle.visible = True
            
            # 更新bbox显示
            if self.show_bbox:
                for i, bbox_handle in enumerate(self.bbox_handles):
                    if bbox_handle is None:
                        continue
                    
                    if show_mode == "4D (当前帧)":
                        bbox_handle.visible = (i == current_timestep)
                    else:  # 3D模式
                        bbox_handle.visible = True
    
    def run(self):
        """运行可视化器"""
        print("\n" + "=" * 80)
        print("🎨 步骤1.0 可视化器已启动")
        print("=" * 80)
        print(f"📡 访问地址: http://localhost:{self.port}")
        print(f"📂 数据目录: {self.data_dir}")
        print(f"📊 场景信息:")
        print(f"   - 步骤: 1.0 (前景点云生成)")
        print(f"   - 视角数: 5 (foreground + 4视角)")
        print(f"   - 总帧数: {self.num_frames}")
        
        total_fg = sum(
            len(pc['foreground']['points']) 
            for pc in self.pointcloud_data_list 
            if pc['foreground'] is not None
        )
        
        print(f"   - 前景总点数: {total_fg:,}")
        print(f"   - 平均每帧: {total_fg // self.num_frames:,} 点")
        print("=" * 80 + "\n")
        
        # 动画循环
        prev_timestep = self.gui_timestep.value
        while True:
            if self.gui_playing.value:
                next_timestep = (self.gui_timestep.value + 1) % self.num_frames
                self.gui_timestep.value = next_timestep
            
            if self.gui_timestep.value != prev_timestep:
                self._update_display()
                prev_timestep = self.gui_timestep.value
            
            time.sleep(1.0 / self.gui_framerate.value)


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("🚀 步骤1.0: 前景点云可视化")
    print("=" * 80)
    print(f"数据目录: {args.data_dir}")
    print("=" * 80)
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"\n❌ 错误：数据目录不存在 {args.data_dir}")
        return
    
    # 创建可视化器
    try:
        viewer = ForegroundPointCloudViewer(
            data_dir=args.data_dir,
            port=args.port,
            point_size=args.point_size,
            fps=args.fps,
            subsample=args.subsample,
            num_frames=args.num_frames,
            show_bbox=args.show_bbox
        )
        
        # 运行可视化器
        viewer.run()
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        return
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
        return
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

