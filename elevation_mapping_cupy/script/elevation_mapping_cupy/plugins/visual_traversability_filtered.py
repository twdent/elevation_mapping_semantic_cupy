#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage
import numpy as np
import cv2

from .plugin_manager import PluginBase

def min_filter(image, kernel_size=3):
    padding = int((kernel_size - 1) / 2)
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
    min_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    filtered_image = cv2.erode(padded_image, min_kernel, iterations=1)
    return filtered_image[padding:-padding, padding:-padding]


class VisualTraversabilityFiltered(PluginBase):
    def __init__(self, cell_n: int = 100, method: str = "telea", **kwargs):
        super().__init__()

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map,
        *args,
    ) -> cp.ndarray:            
        
        visual_trav = semantic_map.semantic_map[0]
        out_trav = min_filter(cp.asnumpy(visual_trav), kernel_size=3)
        return cp.asarray(out_trav).astype(np.float32)