# depth_warp_vs/runtime/visualizer.py
import numpy as np

def colorize_depth(depth):
    d = depth.copy()
    d = d - d.min()
    d = d / (d.max()+1e-8)
    cm = (np.stack([d, 1-d, 0.5*np.ones_like(d)], axis=-1)*255).astype(np.uint8)
    return cm
