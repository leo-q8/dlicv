from typing import Dict, List, Optional, Union

import numpy as np

from dlicv.structures import PixelData
from .palette import get_palette
from .visualizer import Visualizer


class SegVisualizer(Visualizer):
    def __init__(self,
                 image: Optional[np.ndarray] = None,
                 save_dir: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(image, save_dir, **kwargs)
        self.alpha: float = alpha
    
    def draw_sem_seg(self, 
                     image: np.ndarray, 
                     sem_seg: PixelData,
                     palette: Optional[Union[List[tuple], str]] = None
    ) -> np.ndarray:
        sem_seg = sem_seg.cpu().data  
        ids = np.unique(sem_seg)[::-1]
        max_label = int(max(ids))
        palette = get_palette(palette, max_label+1)
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]
        self.set_image(image)
        
        # draw semantic masks
        for label, color in zip(labels, colors):
            self.draw_binary_masks(
                sem_seg == label, colors=[color], alphas=self.alpha)

        return self.get_image()

        

        