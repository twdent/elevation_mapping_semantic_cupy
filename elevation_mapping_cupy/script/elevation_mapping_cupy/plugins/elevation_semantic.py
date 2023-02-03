#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage
import numpy as np
import cv2 as cv

from .plugin_manager import PluginBase


class ElevationSemantic(PluginBase):
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
        
        ma = cp.zeros_like(elevation_map[0])
        ma[:,:] = np.nan
        m = ~cp.isnan(semantic_map.semantic_map[0])
        ma[m] = elevation_map[0][m]
        return ma