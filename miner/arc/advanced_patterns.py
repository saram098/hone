"""
Advanced pattern detection and transformation library
Implements sophisticated ARC solving strategies
"""

from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from collections import Counter


class AdvancedPatternDetector:
    """
    Advanced pattern detection for ARC problems
    """
    
    @staticmethod
    def detect_object_operations(examples: List[Dict]) -> Optional[str]:
        """Detect if problem involves object manipulation"""
        for ex in examples:
            in_objs = AdvancedPatternDetector._extract_objects(ex['input'])
            out_objs = AdvancedPatternDetector._extract_objects(ex['output'])
            
            if len(in_objs) > 0 and len(out_objs) > 0:
                if len(in_objs) != len(out_objs):
                    return "object_filter"
                if AdvancedPatternDetector._objects_moved(in_objs, out_objs):
                    return "object_move"
                if AdvancedPatternDetector._objects_scaled(in_objs, out_objs):
                    return "object_scale"
        return None
    
    @staticmethod
    def detect_counting_operation(examples: List[Dict]) -> bool:
        """Detect if output is based on counting"""
        for ex in examples:
            in_grid = ex['input']
            out_grid = ex['output']
            
            # Check if output size is very small (like 1x1 or 1xN)
            if len(out_grid) <= 2 and len(out_grid[0]) <= 5:
                # Might be counting
                in_counts = Counter(val for row in in_grid for val in row)
                out_vals = [val for row in out_grid for val in row]
                
                # Check if output values match counts
                if any(count in out_vals for count in in_counts.values()):
                    return True
        return False
    
    @staticmethod
    def detect_noise_removal(examples: List[Dict]) -> bool:
        """Detect if transformation removes noise"""
        for ex in examples:
            in_grid = ex['input']
            out_grid = ex['output']
            
            if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
                in_colors = Counter(val for row in in_grid for val in row)
                out_colors = Counter(val for row in out_grid for val in row)
                
                # If output has fewer colors, might be noise removal
                if len(out_colors) < len(in_colors):
                    # Check if a minority color was removed
                    removed_colors = set(in_colors.keys()) - set(out_colors.keys())
                    if removed_colors:
                        for color in removed_colors:
                            if in_colors[color] < sum(in_colors.values()) * 0.2:
                                return True
        return False
    
    @staticmethod
    def detect_gravity_operation(examples: List[Dict]) -> Optional[str]:
        """Detect gravity-like operations (objects falling)"""
        for ex in examples:
            in_grid = ex['input']
            out_grid = ex['output']
            
            if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
                # Check if non-zero values moved down
                for col in range(len(in_grid[0])):
                    in_col = [in_grid[row][col] for row in range(len(in_grid))]
                    out_col = [out_grid[row][col] for row in range(len(out_grid))]
                    
                    in_nonzero = [v for v in in_col if v != 0]
                    out_nonzero = [v for v in out_col if v != 0]
                    
                    if sorted(in_nonzero) == sorted(out_nonzero):
                        # Check if they're at the bottom in output
                        if out_nonzero and out_col[-len(out_nonzero):] == out_nonzero:
                            return "gravity_down"
        return None
    
    @staticmethod
    def detect_grid_overlay(examples: List[Dict]) -> bool:
        """Detect if output is overlay/superposition of patterns"""
        for ex in examples:
            in_grid = ex['input']
            out_grid = ex['output']
            
            # Check if output has more non-zero values
            in_nonzero = sum(1 for row in in_grid for val in row if val != 0)
            out_nonzero = sum(1 for row in out_grid for val in row if val != 0)
            
            if out_nonzero > in_nonzero * 1.5:
                return True
        return False
    
    @staticmethod
    def detect_frame_extraction(examples: List[Dict]) -> bool:
        """Detect if transformation extracts frame/border"""
        for ex in examples:
            in_grid = ex['input']
            out_grid = ex['output']
            
            if len(in_grid) == len(out_grid) and len(in_grid[0]) == len(out_grid[0]):
                # Check if only border is preserved
                for i in range(len(out_grid)):
                    for j in range(len(out_grid[0])):
                        is_border = (i == 0 or i == len(out_grid)-1 or 
                                   j == 0 or j == len(out_grid[0])-1)
                        
                        if is_border:
                            if out_grid[i][j] == 0 and in_grid[i][j] != 0:
                                return False
                        else:
                            if out_grid[i][j] != 0:
                                return False
                return True
        return False
    
    @staticmethod
    def _extract_objects(grid: List[List[int]]) -> List[Set[Tuple[int, int]]]:
        """Extract connected components as objects"""
        h, w = len(grid), len(grid[0])
        visited = [[False] * w for _ in range(h)]
        objects = []
        
        def dfs(i, j, color, obj):
            if i < 0 or i >= h or j < 0 or j >= w:
                return
            if visited[i][j] or grid[i][j] != color or grid[i][j] == 0:
                return
            
            visited[i][j] = True
            obj.add((i, j))
            
            # 4-connected
            dfs(i+1, j, color, obj)
            dfs(i-1, j, color, obj)
            dfs(i, j+1, color, obj)
            dfs(i, j-1, color, obj)
        
        for i in range(h):
            for j in range(w):
                if not visited[i][j] and grid[i][j] != 0:
                    obj = set()
                    dfs(i, j, grid[i][j], obj)
                    if obj:
                        objects.append(obj)
        
        return objects
    
    @staticmethod
    def _objects_moved(objs1: List[Set], objs2: List[Set]) -> bool:
        """Check if objects moved position"""
        if len(objs1) != len(objs2):
            return False
        
        for obj1, obj2 in zip(objs1, objs2):
            if len(obj1) == len(obj2):
                # Check if same shape but different position
                obj1_normalized = AdvancedPatternDetector._normalize_object(obj1)
                obj2_normalized = AdvancedPatternDetector._normalize_object(obj2)
                
                if obj1_normalized == obj2_normalized:
                    # Same shape, check if position different
                    if min(obj1) != min(obj2):
                        return True
        return False
    
    @staticmethod
    def _objects_scaled(objs1: List[Set], objs2: List[Set]) -> bool:
        """Check if objects scaled"""
        if len(objs1) != len(objs2):
            return False
        
        for obj1, obj2 in zip(objs1, objs2):
            if len(obj1) != len(obj2):
                # Different sizes, might be scaled
                return True
        return False
    
    @staticmethod
    def _normalize_object(obj: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Normalize object to origin"""
        if not obj:
            return obj
        
        min_i = min(i for i, j in obj)
        min_j = min(j for i, j in obj)
        
        return {(i - min_i, j - min_j) for i, j in obj}


class AdvancedTransformations:
    """
    Advanced transformation implementations
    """
    
    @staticmethod
    def apply_gravity(grid: List[List[int]], direction: str = "down") -> List[List[int]]:
        """Apply gravity - make non-zero values fall"""
        result = [[0] * len(grid[0]) for _ in range(len(grid))]
        
        if direction == "down":
            for col in range(len(grid[0])):
                # Collect non-zero values
                values = [grid[row][col] for row in range(len(grid)) if grid[row][col] != 0]
                # Place at bottom
                for i, val in enumerate(reversed(values)):
                    result[len(grid) - 1 - i][col] = val
        
        return result
    
    @staticmethod
    def remove_noise(grid: List[List[int]], noise_threshold: float = 0.1) -> List[List[int]]:
        """Remove minority colors (noise)"""
        color_counts = Counter(val for row in grid for val in row if val != 0)
        
        if not color_counts:
            return [row[:] for row in grid]
        
        total = sum(color_counts.values())
        noise_colors = {color for color, count in color_counts.items() 
                       if count < total * noise_threshold}
        
        return [[0 if val in noise_colors else val for val in row] for row in grid]
    
    @staticmethod
    def extract_frame(grid: List[List[int]]) -> List[List[int]]:
        """Extract only the border/frame"""
        result = [[0] * len(grid[0]) for _ in range(len(grid))]
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == 0 or i == len(grid)-1 or j == 0 or j == len(grid[0])-1:
                    result[i][j] = grid[i][j]
        
        return result
    
    @staticmethod
    def fill_interior(grid: List[List[int]], fill_color: int = 1) -> List[List[int]]:
        """Fill interior of shapes"""
        result = [row[:] for row in grid]
        h, w = len(grid), len(grid[0])
        
        # Flood fill from edges to mark exterior
        exterior = [[False] * w for _ in range(h)]
        
        def flood_fill(i, j):
            if i < 0 or i >= h or j < 0 or j >= w:
                return
            if exterior[i][j] or grid[i][j] != 0:
                return
            
            exterior[i][j] = True
            flood_fill(i+1, j)
            flood_fill(i-1, j)
            flood_fill(i, j+1)
            flood_fill(i, j-1)
        
        # Start from edges
        for i in range(h):
            flood_fill(i, 0)
            flood_fill(i, w-1)
        for j in range(w):
            flood_fill(0, j)
            flood_fill(h-1, j)
        
        # Fill non-exterior zeros
        for i in range(h):
            for j in range(w):
                if not exterior[i][j] and result[i][j] == 0:
                    result[i][j] = fill_color
        
        return result
    
    @staticmethod
    def extract_largest_object(grid: List[List[int]]) -> List[List[int]]:
        """Extract only the largest object"""
        objects = AdvancedPatternDetector._extract_objects(grid)
        
        if not objects:
            return [row[:] for row in grid]
        
        largest = max(objects, key=len)
        result = [[0] * len(grid[0]) for _ in range(len(grid))]
        
        for i, j in largest:
            result[i][j] = grid[i][j]
        
        return result
    
    @staticmethod
    def mirror_complete(grid: List[List[int]]) -> List[List[int]]:
        """Complete pattern by mirroring"""
        # Try horizontal mirror completion
        result = [row[:] for row in grid]
        w = len(grid[0])
        
        for i in range(len(grid)):
            for j in range(w // 2):
                if result[i][j] != 0 and result[i][w-1-j] == 0:
                    result[i][w-1-j] = result[i][j]
                elif result[i][w-1-j] != 0 and result[i][j] == 0:
                    result[i][j] = result[i][w-1-j]
        
        return result
    
    @staticmethod
    def apply_mask(grid: List[List[int]], mask_grid: List[List[int]]) -> List[List[int]]:
        """Apply mask - keep only where mask is non-zero"""
        result = [[0] * len(grid[0]) for _ in range(len(grid))]
        
        for i in range(min(len(grid), len(mask_grid))):
            for j in range(min(len(grid[0]), len(mask_grid[0]))):
                if mask_grid[i][j] != 0:
                    result[i][j] = grid[i][j]
        
        return result
    
    @staticmethod
    def replicate_pattern(grid: List[List[int]], times_h: int, times_w: int) -> List[List[int]]:
        """Replicate pattern multiple times"""
        h, w = len(grid), len(grid[0])
        result = [[0] * (w * times_w) for _ in range(h * times_h)]
        
        for rep_i in range(times_h):
            for rep_j in range(times_w):
                for i in range(h):
                    for j in range(w):
                        result[rep_i * h + i][rep_j * w + j] = grid[i][j]
        
        return result

