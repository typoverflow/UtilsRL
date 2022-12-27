import os
import imageio
import numpy as np
from PIL import Image
import gym

class VideoRecorder():
    def __init__(
        self, 
        output_dir: str, 
        min_fps: int=10, 
        max_fps: int=60, 
        enabled: bool=True
    ):
        self.output_dir = output_dir
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.enabled = enabled
        self.frames = []
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def record(self, data):
        if self.enabled:
            self.frames.append(data)
        
    def save(self, name: str, format: str="mp4", reset: bool=True):
        if self.enabled:
            output_path = os.path.join(self.output_dir, name+f".{format}")
            if len(self.frames) > 0:
                # determine the fps
                candidate_fps = len(self.frames) // 8
                candidate_fps = min(self.max_fps, max(self.min_fps, candidate_fps))
                if format == "mp4":
                    imageio.mimsave(output_path, self.frames, fps=candidate_fps, macro_block_size=2)
                elif format == "gif":
                    imageio.mimsave(output_path, self.frames, duration=1/candidate_fps)
            if reset:
                self.reset()
                    
    def reset(self):
        if self.enabled:
            self.frames = []
        
        