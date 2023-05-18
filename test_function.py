import torch
import numpy as np
COLORS = {
    'brown': [165, 42, 42],
    'red': [255, 0, 0],
    'pink': [253, 108, 158],
    'orange': [255, 165, 0],
    'yellow': [255, 255, 0],
    'purple': [128, 0, 128],
    'green': [0, 128, 0],
    'blue': [0, 0, 255],
    'white': [255, 255, 255],
    'gray': [128, 128, 128],
    'black': [0, 0, 0],
}

def find_nearest_color(rgb):
    # Find the nearest color
    if isinstance(rgb, list) or isinstance(rgb, tuple):
        rgb = torch.FloatTensor(rgb)[None, :, None, None]/255.
    color_distance = torch.FloatTensor([np.linalg.norm(rgb - torch.FloatTensor(COLORS[color])[None, :, None, None]/255.) for color in COLORS.keys()])
    nearest_color = list(COLORS.keys())[torch.argmin(color_distance).item()]
    print('Nearest color: {}'.format(nearest_color))
    return nearest_color

hh = find_nearest_color((235,145, 134))
