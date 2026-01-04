#!/usr/bin/env python3
"""
步骤1.1可视化：背景+前景点云可视化工具（Viser）

功能：
    可视化步骤1.1生成的点云（全局背景+每帧前景）
    
输入结构（步骤1.1的输出）：
    prepared_dir/
    ├── {frame_id}/
    │   └── pointcloud/
    │       └── {frame_id}_foreground_singleview.ply  # 每帧前景点云
    ├── global_background.ply                          # 全局背景点云
    └── global_camera.json                             # 相机参数

特性：
    - 全局背景固定显示
    - 按帧浏览前景点云
    - 播放控制（FPS可调）
    - 4D/3D模式切换
    - 点云大小调节
    - AABB包围盒显示

使用示例：
    # 单个场景
    python scripts/1_1_visualization.py \
        --data_dir outputs/prepared/camel
    
    # 限制帧数
    python scripts/1_1_visualization.py \
        --data_dir outputs/prepared/bear \
        --num_frames 24

主要参数：
    --data_dir: 数据目录（步骤1.1的输出）
    --port: Viser端口（默认8080）
    --point_size: 点大小（默认0.0005）
    --fps: 播放帧率（默认5.0）
    --subsample: 下采样率（默认1）
    --num_frames: 加载帧数（默认全部）
    --show_bbox: 显示AABB包围盒
"""

import os
import time
import argparse
from typing import List, Dict, Optional
import numpy as np
import trimesh
import viser


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="步骤1.1可视化：背景+前景点云（Viser）")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录（步骤1.1的输出）")
    parser.add_argument("--port", type=int, default=8080, help="Viser端口（默认8080）")
    parser.add_argument("--point_size", type=float, default=0.0005, help="点大小（默认0.0005）")
    parser.add_argument("--fps", type=float, default=5.0, help="播放帧率（默认5.0）")
    parser.add_argument("--subsample", type=int, default=1, help="下采样率（默认1）")
    parser.add_argument("--num_frames", type=int, default=None, help="加载帧数（默认全部）")
    parser.add_argument("--show_bbox", action="store_true", help="显示AABB包围盒")
    return parser.parse_args()


class V4_1BackgroundForegroundViewer:
    """步骤1.1可视化器：全局背景+每帧前景点云
    
    特性：
        - 全局背景固定显示
        - 按帧浏览前景
        - 4D/3D模式切换
        - 播放控制（可调FPS）
    """
    
    def __init__(
        self,
        data_dir: str,
        port: int = 8080,
        point_size: float = 0.003,
        fps: float = 5.0,
        subsample: int = 1,
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
        
        # 加载全局背景
        self.background_data = self._load_global_background()
        
        # 加载每帧前景
        self.foreground_frames = self._load_foreground_frames()
        self.num_frames = len(self.foreground_frames)
        
        if self.num_frames == 0 and self.background_data is None:
            raise ValueError(f"未找到任何有效的点云文件")
        
        print(f"\n找到 {self.num_frames} 帧前景点云")
        
        # 启动Viser服务器
        self.server = self._setup_viser()
    
    def _load_global_background(self) -> Optional[Dict]:
        """加载全局背景点云（global_background.ply）
        
        Returns:
            Optional[Dict]: {points, colors} 或 None
        """
        print("\n📂 加载背景...")
        bg_path = os.path.join(self.data_dir, "global_background.ply")
        if not os.path.exists(bg_path):
            print(f"  ⚠️  未找到")
            return None
        
        try:
            mesh = trimesh.load(bg_path)
            points = np.array(mesh.vertices)
            colors = (np.array(mesh.visual.vertex_colors)[:, :3] if hasattr(mesh, 'visual') 
                     and hasattr(mesh.visual, 'vertex_colors')
                     else np.ones((len(points), 3), dtype=np.uint8) * 150)
            
            if self.subsample > 1:
                indices = np.arange(0, len(points), self.subsample)
                points, colors = points[indices], colors[indices]
            
            print(f"  ✓ {len(points):,} 点")
            return {'points': points, 'colors': colors}
        except Exception as e:
            print(f"  ✗ {e}")
            return None
    
    def _load_foreground_frames(self) -> List[Dict]:
        """加载每帧前景点云（{frame_id}_foreground_singleview.ply）
        
        Returns:
            List[Dict]: [{frame_id, points, colors}, ...]
        """
        print("\n📂 加载前景点云...")
        subdirs = sorted([d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()])
        if self.num_frames_limit:
            subdirs = subdirs[:self.num_frames_limit]
        
        foreground_list = []
        for frame_id in subdirs:
            fg_path = os.path.join(self.data_dir, frame_id, 'pointcloud', f"{frame_id}_foreground_singleview.ply")
            if not os.path.exists(fg_path):
                continue
            
            try:
                mesh = trimesh.load(fg_path)
                points = np.array(mesh.vertices)
                colors = (np.array(mesh.visual.vertex_colors)[:, :3] if hasattr(mesh, 'visual') 
                         and hasattr(mesh.visual, 'vertex_colors') 
                         else np.ones((len(points), 3), dtype=np.uint8) * [0, 255, 0])
                
                if self.subsample > 1:
                    indices = np.arange(0, len(points), self.subsample)
                    points, colors = points[indices], colors[indices]
                
                foreground_list.append({'frame_id': frame_id, 'points': points, 'colors': colors})
                print(f"  帧 {frame_id}: {len(points):,} 点")
            except Exception as e:
                print(f"  ✗ {frame_id}: {e}")
        
        return foreground_list
    
    def _setup_viser(self) -> viser.ViserServer:
        """设置Viser服务器并创建GUI
        
        Returns:
            viser.ViserServer: 服务器实例
        """
        print(f"\n启动Viser服务器，端口: {self.port}")
        server = viser.ViserServer(host="0.0.0.0", port=self.port)
        server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        
        self._create_gui(server)
        self._create_scene(server)
        
        return server
    
    def _create_gui(self, server: viser.ViserServer):
        """创建GUI控制面板
        
        Args:
            server: Viser服务器实例
        """
        # 时间控制
        with server.gui.add_folder("⏱️ 时间控制"):
            self.gui_timestep = server.gui.add_slider(
                "帧序号",
                min=0,
                max=max(self.num_frames - 1, 0),
                step=1,
                initial_value=0,
            )
            
            self.gui_frame_id_label = server.gui.add_text(
                "当前帧ID",
                initial_value=self.foreground_frames[0]['frame_id'] if self.foreground_frames else "N/A",
                disabled=True,
            )
            
            self.gui_playing = server.gui.add_checkbox("▶️ 播放", False)
            self.gui_framerate = server.gui.add_slider("FPS", min=1, max=30, step=0.5, initial_value=self.fps)
        
        # 显示控制
        with server.gui.add_folder("🎨 显示控制"):
            self.gui_show_background = server.gui.add_checkbox("显示全局背景", True)
            self.gui_show_foreground = server.gui.add_checkbox("显示前景", True)
            
            self.gui_show_mode = server.gui.add_button_group(
                "显示模式", 
                ("4D (当前帧)", "3D (所有帧累积)")
            )
            
            self.gui_point_size = server.gui.add_slider(
                "点云大小",
                min=0.0001,
                max=0.01,
                step=0.0001,
                initial_value=self.point_size
            )
        
        # 统计信息
        with server.gui.add_folder("📊 统计信息"):
            bg_points = len(self.background_data['points']) if self.background_data else 0
            fg_points = sum(len(pc['points']) for pc in self.foreground_frames)
            
            info_text = f"步骤: 1.1 背景点云生成\n"
            info_text += f"全局背景: {bg_points:,} 点\n"
            info_text += f"前景帧数: {self.num_frames}\n"
            info_text += f"前景总点数: {fg_points:,}\n"
            info_text += f"总点数: {bg_points + fg_points:,}"
            
            self.gui_stats = server.gui.add_text(
                "统计",
                initial_value=info_text,
                disabled=True,
            )
        
        self.server = server
    
    def _create_scene(self, server: viser.ViserServer):
        """创建3D场景（添加背景和前景点云）
        
        Args:
            server: Viser服务器实例
        """
        # 添加全局背景点云
        self.background_handle = None
        if self.background_data:
            self.background_handle = server.scene.add_point_cloud(
                name="/pointcloud/global_background",
                points=self.background_data['points'],
                colors=self.background_data['colors'],
                point_size=self.point_size,
                point_shape="circle",
            )
        
        # 添加每帧的前景点云
        self.foreground_handles = []
        for pc_data in self.foreground_frames:
            frame_id = pc_data['frame_id']
            
            handle = server.scene.add_point_cloud(
                name=f"/pointcloud/foreground_{frame_id}",
                points=pc_data['points'],
                colors=pc_data['colors'],
                point_size=self.point_size,
                point_shape="circle",
            )
            
            self.foreground_handles.append(handle)
        
        # 添加第一帧前景的AABB（如果启用）
        self.bbox_handle = None
        if self.show_bbox and len(self.foreground_frames) > 0:
            first_fg = self.foreground_frames[0]
            self._create_bbox(server, first_fg['points'])
        
        # 初始显示
        self._update_display()
        self._bind_events()
    
    def _create_bbox(self, server: viser.ViserServer, points: np.ndarray):
        """创建AABB包围盒（黄色边框线）
        
        Args:
            server: Viser服务器实例
            points: 点云坐标数组
        """
        # 计算AABB
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        
        print(f"\n📦 第一帧前景 AABB:")
        print(f"  Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
        print(f"  Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
        print(f"  Size: [{bbox_max[0]-bbox_min[0]:.3f}, {bbox_max[1]-bbox_min[1]:.3f}, {bbox_max[2]-bbox_min[2]:.3f}]")
        
        # 生成AABB边框线点
        def create_bbox_lines(bbox_min, bbox_max, color, num_points_per_edge=50):
            """生成AABB包围盒的12条边（点云表示）"""
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
                for t in np.linspace(0, 1, num_points_per_edge):
                    point = v1 + t * (v2 - v1)
                    line_points.append(point)
            
            return np.array(line_points), np.tile(color, (len(line_points), 1))
        
        # 创建bbox线（黄色）
        bbox_points, bbox_colors = create_bbox_lines(bbox_min, bbox_max, [255, 255, 0])
        
        # 添加到场景
        self.bbox_handle = server.scene.add_point_cloud(
            name="/reference/frame_0_bbox",
            points=bbox_points,
            colors=bbox_colors,
            point_size=self.point_size * 5.0,  # 加粗
            point_shape="circle",
        )
        
        print(f"  ✓ 添加第一帧AABB包围盒（黄色），贯穿整个视频")
    
    def _bind_events(self):
        """绑定GUI事件回调"""
        @self.gui_timestep.on_update
        def _(_) -> None:
            self._update_display()
        
        @self.gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = self.gui_playing.value
        
        @self.gui_show_mode.on_click
        def _(_) -> None:
            self._update_display()
        
        @self.gui_show_background.on_update
        def _(_) -> None:
            self._update_display()
        
        @self.gui_show_foreground.on_update
        def _(_) -> None:
            self._update_display()
        
        @self.gui_point_size.on_update
        def _(_) -> None:
            self._update_point_size()
    
    def _update_point_size(self):
        """更新所有点云的显示大小"""
        new_size = self.gui_point_size.value
        with self.server.atomic():
            if self.background_handle:
                self.background_handle.point_size = new_size
            
            for handle in self.foreground_handles:
                handle.point_size = new_size
            
            if self.bbox_handle is not None:
                self.bbox_handle.point_size = new_size * 5.0
    
    def _update_display(self):
        """更新显示状态（切换帧、4D/3D模式）"""
        current_timestep = self.gui_timestep.value
        show_mode = self.gui_show_mode.value
        show_background = self.gui_show_background.value
        show_foreground = self.gui_show_foreground.value
        
        # 更新当前帧ID标签
        if current_timestep < len(self.foreground_frames):
            current_frame = self.foreground_frames[current_timestep]
            self.gui_frame_id_label.value = current_frame['frame_id']
        
        with self.server.atomic():
            # 更新背景显示（全局背景固定）
            if self.background_handle:
                self.background_handle.visible = show_background
            
            # 更新前景显示
            for i, handle in enumerate(self.foreground_handles):
                if not show_foreground:
                    handle.visible = False
                elif show_mode == "4D (当前帧)":
                    handle.visible = (i == current_timestep)
                else:  # 3D模式
                    handle.visible = True
    
    def run(self):
        """启动可视化器并进入主循环"""
        print("\n" + "=" * 80)
        print("✨ 步骤1.1可视化器已启动")
        print("=" * 80)
        print(f"📡 访问地址: http://localhost:{self.port}")
        print(f"📂 数据目录: {self.data_dir}")
        print(f"📊 场景信息:")
        
        if self.background_data:
            print(f"   - 全局背景点数: {len(self.background_data['points']):,}")
        
        print(f"   - 前景帧数: {self.num_frames}")
        
        if self.foreground_frames:
            total_fg = sum(len(pc['points']) for pc in self.foreground_frames)
            avg_fg = total_fg // self.num_frames if self.num_frames > 0 else 0
            print(f"   - 前景总点数: {total_fg:,}")
            print(f"   - 平均每帧: {avg_fg:,} 点")
        
        print("=" * 80)
        print("🎮 控制说明:")
        print("   - 拖动'帧序号'滑块切换帧")
        print("   - 勾选'播放'自动播放动画")
        print("   - 勾选/取消'显示全局背景'/'显示前景'控制显示")
        print("   - 调整'点云大小'控制显示大小")
        print("   - 切换 4D/3D 模式:")
        print("     * 4D: 全局背景 + 当前帧前景")
        print("     * 3D: 全局背景 + 所有帧前景累积")
        print("   - 鼠标控制：旋转、缩放、平移")
        if self.show_bbox:
            print("   - 黄色边框: 第一帧AABB包围盒")
        print("=" * 80 + "\n")
        
        # 动画循环
        prev_timestep = self.gui_timestep.value
        while True:
            if self.gui_playing.value and self.num_frames > 0:
                next_timestep = (self.gui_timestep.value + 1) % self.num_frames
                self.gui_timestep.value = next_timestep
            
            if self.gui_timestep.value != prev_timestep:
                self._update_display()
                prev_timestep = self.gui_timestep.value
            
            time.sleep(1.0 / self.gui_framerate.value)


def main():
    """主函数：启动步骤1.1可视化器"""
    args = parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 错误：数据目录不存在 {args.data_dir}")
        return
    
    # 创建可视化器
    try:
        viewer = V4_1BackgroundForegroundViewer(
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
        print(f"❌ 错误: {e}")
        return
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
        return
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

