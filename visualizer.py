import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from is_msgs.image_pb2 import Image

class Visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=30, azim=-110)
        self.id_colors = {}
        self.color_map = plt.get_cmap('tab10')
        
        self.skeleton_map = [
            {'srt_kpt_id': 16, 'dst_kpt_id': 14},  
            {'srt_kpt_id': 14, 'dst_kpt_id': 12},  
            {'srt_kpt_id': 17, 'dst_kpt_id': 15},  
            {'srt_kpt_id': 15, 'dst_kpt_id': 13},  
            {'srt_kpt_id': 12, 'dst_kpt_id': 13},  
            {'srt_kpt_id': 6, 'dst_kpt_id': 12},   
            {'srt_kpt_id': 7, 'dst_kpt_id': 13},   
            {'srt_kpt_id': 6, 'dst_kpt_id': 7},    
            {'srt_kpt_id': 6, 'dst_kpt_id': 8},    
            {'srt_kpt_id': 8, 'dst_kpt_id': 10},   
            {'srt_kpt_id': 7, 'dst_kpt_id': 9},    
            {'srt_kpt_id': 9, 'dst_kpt_id': 11},   
            {'srt_kpt_id': 2, 'dst_kpt_id': 3},    
            {'srt_kpt_id': 1, 'dst_kpt_id': 2},
            {'srt_kpt_id': 1, 'dst_kpt_id': 3},
            {'srt_kpt_id': 2, 'dst_kpt_id': 4},    
            {'srt_kpt_id': 3, 'dst_kpt_id': 5},    
        ]
        
    def _fig_to_numpy(self) -> np.ndarray:
        self.fig.canvas.draw()
        rgba_buffer = self.fig.canvas.buffer_rgba()
        return np.asarray(rgba_buffer)
        
    def _draw_camera_axes(self, extrinsic_matrices: list, axis_length: float = 0.5):
        axis_colors = ['r', 'g', 'b']
        for i, rt in enumerate(extrinsic_matrices):
            R = rt[:, :3]
            t = rt[:, 3]
            cam_center = -R.T @ t
            cam_axes_in_world = R.T
            self.ax.scatter(*cam_center, s=80, c='black', marker='o')
            self.ax.text(*cam_center, f' Cam {i+1}', color='black', fontsize=12)
            for j in range(3):
                axis_start = cam_center
                axis_end = cam_center + cam_axes_in_world[:, j] * axis_length
                self.ax.plot([axis_start[0], axis_end[0]],
                             [axis_start[1], axis_end[1]],
                             [axis_start[2], axis_end[2]],
                             color=axis_colors[j], linewidth=3)
                
    def update(self, images: list, skeletons_to_visualize: list, extrinsic_matrices: list):
        
        self.ax.clear()
        self.ax.set_xlim(-4.5, 4.5); self.ax.set_ylim(-4.5, 4.5); self.ax.set_zlim(0, 3)
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m)')

        self._draw_camera_axes(extrinsic_matrices)
        if skeletons_to_visualize is None or len(skeletons_to_visualize) == 0:
            plot_img_rgba = self._fig_to_numpy()
            plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
            return plot_img_bgr
        
        for person in skeletons_to_visualize:

            person_id = person['id']
            skeleton = person['skeleton_3d']
            if not skeleton: continue
            
            if person_id not in self.id_colors:
                self.id_colors[person_id] = self.color_map(len(self.id_colors) % 10)
            color = self.id_colors[person_id]
            
            points = np.array(list(skeleton.values()))
            self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color=color, label=f"ID (Frame): {person_id + 1}")
            if 0 in skeleton:
                head_pos = skeleton[0]
                self.ax.text(head_pos[0], head_pos[1], head_pos[2] + 0.1, f"ID {person_id}", color=color, fontsize=12, fontweight='bold')
                
            for connection in self.skeleton_map:
                srt_id = connection['srt_kpt_id']
                dst_id = connection['dst_kpt_id']
                if srt_id in skeleton and dst_id in skeleton:
                    srt_point = skeleton[srt_id]
                    dst_point = skeleton[dst_id]
                    self.ax.plot([srt_point[0], dst_point[0]],
                                 [srt_point[1], dst_point[1]],
                                 [srt_point[2], dst_point[2]],
                                 color=color, linewidth=2)
                    
            if skeletons_to_visualize:
                self.ax.legend()
                
        plot_img_rgba = self._fig_to_numpy()
        plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)

        return plot_img_bgr