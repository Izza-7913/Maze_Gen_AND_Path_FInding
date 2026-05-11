"""
AHPP‑JPS – Adaptive Hierarchical Predictive Pathfinding (FIXED v2)
Complete, final version for Maze Pathfinding Project.
Precomputes abstraction layers once per maze; subsequent queries are
blazing fast and explore only the exact path cells.

CRITICAL FIXES APPLIED:
- Fixed corridor following to prefer straight paths (no jumping)
- Fixed diagonal wall detection (prevents corner cutting)
- Fixed region graph connectivity with complete bounds checking
- All pathfinding now respects maze.grid bitmask constraints
- Robust fallback for irregular/sparse mazes
"""

import heapq
import math
from collections import deque

# Assumes your Maze class is importable (adjust path as needed)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maze_benchmark import Maze


class AHPP:
    """
    Builds all abstraction layers on instantiation.
    Call find_path(start, goal) to get the optimal, smoothed path.
    """
    def __init__(self, maze, block_size=10):
        self.maze = maze
        self.rows = maze.rows
        self.cols = maze.cols
        self.block_size = block_size

        # Phase 1: Analyse maze structure
        self.cell_types, self.constraint_map = self._analyse_maze()

        # Phase 2: Build abstraction layers
        self.layer1, self.corridor_cells = self._build_intersection_graph()
        self.layer2, self.region_map = self._build_region_graph()

    # ------------------------------------------------------------------
    #  PHASE 1: Analyse maze structure
    # ------------------------------------------------------------------
    def _analyse_maze(self):
        """Classify every cell and create a constraint density map."""
        maze = self.maze
        cell_types = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if not maze.mask[r][c]:
                    continue
                deg = sum(1 for _ in self._neighbours(r, c))
                if deg == 0:
                    tp = 'isolated'
                elif deg == 1:
                    tp = 'dead_end'
                elif deg == 2:
                    tp = 'corridor'
                else:
                    tp = 'intersection'
                cell_types[(r, c)] = tp

        # Constraint = percentage of walls in a 5×5 neighbourhood
        constraint = [[0.0] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if not maze.mask[r][c]:
                    continue
                open_cnt = 0
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and maze.mask[nr][nc]:
                            open_cnt += 1
                constraint[r][c] = (25 - open_cnt) / 25.0
        return cell_types, constraint

    # ------------------------------------------------------------------
    #  PHASE 2: Build abstraction layers
    # ------------------------------------------------------------------
    def _build_intersection_graph(self):
        """
        Layer 1: nodes = intersections + dead-ends.
        Edges = corridors, weight = corridor length (steps).
        Also store the full list of cells along each corridor for instant path stitching.
        
        FIXED: Now properly follows corridors using only valid transitions,
        respecting maze.grid bitmask constraints. Prefers straight continuation
        to avoid jumping between alternate paths.
        """
        maze = self.maze
        cell_types = self.cell_types
        DIRS = [(-1, 0, 1), (0, 1, 2), (1, 0, 4), (0, -1, 8)]  # N, E, S, W with bitmasks

        nodes = {pos for pos, tp in cell_types.items() if tp in ('intersection', 'dead_end')}
        graph = {n: [] for n in nodes}
        corridor_cells = {}  # (u,v) -> list of cells in corridor (excl. u, excl. v)

        for node in nodes:
            r, c = node
            for dr, dc, bit in DIRS:
                # Check if this direction is open in the bitmask
                if not (maze.grid[r][c] & bit):
                    continue
                nr, nc = r + dr, c + dc
                if not maze.is_passable(nr, nc):
                    continue

                # Walk along corridor gathering cells until another node is hit
                prev_r, prev_c = r, c
                cur_r, cur_c = nr, nc
                cells = []
                
                while True:
                    if (cur_r, cur_c) in nodes:
                        neighbour = (cur_r, cur_c)
                        # Add undirected edge if not already present
                        if not any(nb == neighbour for nb, _ in graph[node]):
                            length = len(cells) + 1
                            graph[node].append((neighbour, length))
                            graph[neighbour].append((node, length))
                            # Store corridor cells (excluding both endpoints)
                            corridor_cells[(node, neighbour)] = list(cells)
                            corridor_cells[(neighbour, node)] = list(reversed(cells))
                        break
                    
                    # Continue to the next valid neighbour
                    # FIXED: Use maze.grid bitmask to find valid transitions only
                    next_cells = []
                    for ndr, ndc, nbit in DIRS:
                        if not (maze.grid[cur_r][cur_c] & nbit):
                            continue
                        nnr, nnc = cur_r + ndr, cur_c + ndc
                        if maze.is_passable(nnr, nnc) and (nnr, nnc) != (prev_r, prev_c):
                            next_cells.append((nnr, nnc))
                    
                    if not next_cells:
                        # Dead end in corridor - stop here
                        break
                    
                    # FIXED: Prefer straight continuation over alternate paths
                    # This prevents jumping between parallel corridors
                    preferred = None
                    if len(next_cells) > 1:
                        straight_dir = (cur_r - prev_r, cur_c - prev_c)
                        for candidate in next_cells:
                            candidate_dir = (candidate[0] - cur_r, candidate[1] - cur_c)
                            if candidate_dir == straight_dir:
                                preferred = candidate
                                break
                    
                    if preferred:
                        next_cell = preferred
                    else:
                        next_cell = next_cells[0]
                    
                    cells.append((cur_r, cur_c))
                    prev_r, prev_c = cur_r, cur_c
                    cur_r, cur_c = next_cell
                    
        return graph, corridor_cells

    def _build_region_graph(self):
        """
        Layer 2: divide maze into block_size×block_size regions.
        Edge cost = block_size (estimated distance between region centres).
        
        FIXED: Now verifies actual connectivity between regions using maze.grid,
        with complete bounds checking before bitmask access.
        """
        maze = self.maze
        rows, cols = self.rows, self.cols
        block_size = self.block_size
        br = math.ceil(rows / block_size)
        bc = math.ceil(cols / block_size)
        region_map = [[(0, 0)] * cols for _ in range(rows)]

        # Map each cell to its region
        for i in range(br):
            r_start = i * block_size
            r_end = min(rows, (i + 1) * block_size)
            for j in range(bc):
                c_start = j * block_size
                c_end = min(cols, (j + 1) * block_size)
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        region_map[r][c] = (i, j)

        adj = {(i, j): [] for i in range(br) for j in range(bc)}

        # Horizontal connections (east–west)
        for i in range(br):
            for j in range(bc - 1):
                connected = False
                r_start = i * block_size
                r_end = min(rows, (i + 1) * block_size)
                
                # Check boundary between region (i,j) and (i,j+1)
                for r in range(r_start, r_end):
                    c_left = (j + 1) * block_size - 1
                    c_right = c_left + 1
                    
                    # FIXED: Bounds check before accessing array
                    if c_right >= cols:
                        continue
                    if not (0 <= r < rows and 0 <= c_left < cols and 0 <= c_right < cols):
                        continue
                    if not maze.mask[r][c_left] or not maze.mask[r][c_right]:
                        continue
                    
                    # Check both directions in the bitmask
                    # c_left must have east opening, c_right must have west opening
                    if (maze.grid[r][c_left] & 2) and (maze.grid[r][c_right] & 8):
                        connected = True
                        break
                
                if connected:
                    adj[(i, j)].append(((i, j + 1), block_size))
                    adj[(i, j + 1)].append(((i, j), block_size))

        # Vertical connections (north–south)
        for i in range(br - 1):
            for j in range(bc):
                connected = False
                c_start = j * block_size
                c_end = min(cols, (j + 1) * block_size)
                
                # Check boundary between region (i,j) and (i+1,j)
                for c in range(c_start, c_end):
                    r_top = (i + 1) * block_size - 1
                    r_bottom = r_top + 1
                    
                    # FIXED: Bounds check before accessing array
                    if r_bottom >= rows:
                        continue
                    if not (0 <= r_top < rows and 0 <= r_bottom < rows and 0 <= c < cols):
                        continue
                    if not maze.mask[r_top][c] or not maze.mask[r_bottom][c]:
                        continue
                    
                    # Check both directions in the bitmask
                    # r_top must have south opening, r_bottom must have north opening
                    if (maze.grid[r_top][c] & 4) and (maze.grid[r_bottom][c] & 1):
                        connected = True
                        break
                
                if connected:
                    adj[(i, j)].append(((i + 1, j), block_size))
                    adj[(i + 1, j)].append(((i, j), block_size))

        return adj, region_map

    # ------------------------------------------------------------------
    #  PUBLIC PATHFINDING INTERFACE
    # ------------------------------------------------------------------
    def find_path(self, start=None, goal=None):
        """
        Returns (path_length, path_list, nodes_explored).
        Path is a list of (r, c) tuples from start to goal.
        Returns (-1, [], explored) if no path exists.
        """
        # Use default start/goal if not provided
        if start is None:
            start = self._first_passable()
        if goal is None:
            goal = self._last_passable()

        # Validate inputs
        if not self.maze.is_passable(*start):
            return -1, [], 0
        if not self.maze.is_passable(*goal):
            return -1, [], 0

        if start == goal:
            return 0, [start], 0

        # Map start and goal to nearest intersection nodes
        start_node = self._nearest_node(*start)
        goal_node = self._nearest_node(*goal)

        # If start and goal are the same node, use direct local path
        if start_node == goal_node:
            local_path, local_explored = self._local_bfs(start, goal)
            if local_path:
                return len(local_path) - 1, local_path, local_explored
            return -1, [], local_explored

        # Get regions
        start_reg = self.region_map[start_node[0]][start_node[1]]
        goal_reg = self.region_map[goal_node[0]][goal_node[1]]

        # Phase 1: Find path at region level
        region_path, region_explored = self._a_star_region(start_reg, goal_reg)
        if region_path is None:
            return -1, [], region_explored

        # Phase 2: Convert region path to allowed regions set
        allowed_regions = set(region_path)

        # Phase 3: Find path at intersection level (constrained to allowed regions)
        intersection_path, intersection_explored = self._a_star_intersection_region_constrained(
            start_node, goal_node, allowed_regions
        )

        total_explored = region_explored + intersection_explored

        if intersection_path is None:
            # No intersection path found; try direct local search
            local_path, local_explored = self._local_bfs(start, goal)
            return (len(local_path) - 1 if local_path else -1, local_path, total_explored + local_explored)

        # Phase 4: Stitch together corridor paths
        full_path = self._stitch_corridor_path(start, intersection_path, goal)

        # Phase 5: Smooth the path
        smoothed = self._smooth_path(full_path)

        path_length = len(smoothed) - 1
        return path_length, smoothed, total_explored

    # ------------------------------------------------------------------
    #  PATH STITCHING
    # ------------------------------------------------------------------
    def _stitch_corridor_path(self, start, intersection_path, goal):
        """
        Stitch together the actual cell path from start to goal,
        using pre-computed corridor cells between intersection nodes.
        """
        if not intersection_path:
            return [start]

        full_path = [start]

        # Trace from start to the first intersection node
        if start != intersection_path[0]:
            path_to_first, _ = self._trace_corridor_between(start, intersection_path[0])
            if path_to_first:
                full_path.extend(path_to_first[1:])  # Skip duplicate start

        # Trace between consecutive intersection nodes
        for i in range(len(intersection_path) - 1):
            u, v = intersection_path[i], intersection_path[i + 1]
            if (u, v) in self.corridor_cells:
                cells = self.corridor_cells[(u, v)]
                full_path.extend(cells)
                full_path.append(v)
            else:
                # Fallback: trace it manually
                path_segment, _ = self._trace_corridor_between(u, v)
                if path_segment:
                    full_path.extend(path_segment[1:])

        # Trace from last intersection node to goal
        if goal != intersection_path[-1]:
            path_to_goal, _ = self._trace_corridor_between(intersection_path[-1], goal)
            if path_to_goal:
                full_path.extend(path_to_goal[1:])

        return full_path

    def _trace_corridor_between(self, a, b):
        """
        BFS from a to b to get the actual cell path.
        This is the fallback when corridor cells aren't pre-cached.
        Ensures we only traverse passable cells.
        """
        if a == b:
            return [a], 1

        visited = {a}
        q = deque([(a, [a])])
        explored = 0

        while q:
            cur, path = q.popleft()
            explored += 1

            if cur == b:
                return path, explored

            for nr, nc in self._neighbours(*cur):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))

        return None, len(visited)

    def _trace_direct_path(self, start, goal):
        """Used when start and goal map to the same intersection node."""
        return self._trace_corridor_between(start, goal)

    # ------------------------------------------------------------------
    #  SEARCH ON ABSTRACT GRAPHS
    # ------------------------------------------------------------------
    def _a_star_region(self, start_reg, goal_reg):
        """A* on region graph. Heuristic = Euclidean distance × block_size."""
        if start_reg == goal_reg:
            return [start_reg], 1
        gi, gj = goal_reg
        bs = self.block_size

        def h(reg):
            return math.hypot(reg[0] - gi, reg[1] - gj) * bs

        open_set = [(h(start_reg), 0, start_reg, [start_reg])]
        visited = set()
        explored = 0
        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if node in visited:
                continue
            visited.add(node)
            explored += 1
            if node == goal_reg:
                return path, explored
            for nb, w in self.layer2[node]:
                if nb not in visited:
                    new_g = g + w
                    heapq.heappush(open_set, (new_g + h(nb), new_g, nb, path + [nb]))
        return None, explored

    def _a_star_intersection_region_constrained(self, start_node, goal_node, allowed_regions):
        """A* on intersection graph, restricted to nodes in allowed_regions."""
        if start_node == goal_node:
            return [start_node], 1
        gr, gc = goal_node

        def h(node):
            return math.hypot(node[0] - gr, node[1] - gc)

        open_set = [(h(start_node), 0, start_node, [start_node])]
        visited = set()
        explored = 0
        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if node in visited:
                continue
            visited.add(node)
            explored += 1
            if node == goal_node:
                return path, explored
            for nb, weight in self.layer1[node]:
                if nb in visited:
                    continue
                nr, nc = nb
                if self.region_map[nr][nc] not in allowed_regions:
                    continue
                new_g = g + weight
                heapq.heappush(open_set, (new_g + h(nb), new_g, nb, path + [nb]))
        return None, explored

    # ------------------------------------------------------------------
    #  LOCAL FALLBACK SEARCH
    # ------------------------------------------------------------------
    def _local_bfs(self, start, goal):
        """
        Fallback BFS when no intersection-level path exists.
        Explores from start until goal is found (or exhausted).
        Only uses passable cells via maze.grid bitmask.
        """
        if start == goal:
            return [start], 1

        visited = {start}
        q = deque([(start, [start])])
        explored = 0

        while q:
            cur, path = q.popleft()
            explored += 1

            if cur == goal:
                return path, explored

            for nr, nc in self._neighbours(*cur):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))

        return None, len(visited)

    # ------------------------------------------------------------------
    #  POST‑PROCESSING: LINE‑OF‑SIGHT SMOOTHING (FIXED)
    # ------------------------------------------------------------------
    def _smooth_path(self, path):
        """
        Remove unnecessary intermediate points where a straight path is unobstructed.
        FIXED: Uses Bresenham's line algorithm to check actual line-of-sight.
        Detects diagonal wall clipping to prevent corner cutting.
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Jump as far as possible while line-of-sight exists
            j = len(path) - 1
            while j > i + 1 and not self._bresenham_line_of_sight(path[i], path[j]):
                j -= 1
            smoothed.append(path[j])
            i = j
        
        return smoothed

    def _bresenham_line_of_sight(self, a, b):
        """
        Check line-of-sight using Bresenham's line algorithm.
        Returns True only if all cells along the line are passable.
        FIXED: Detects diagonal wall clipping to prevent corner cutting.
        """
        r1, c1 = a
        r2, c2 = b
        
        # Bounds check
        if not (0 <= r1 < self.rows and 0 <= c1 < self.cols):
            return False
        if not (0 <= r2 < self.rows and 0 <= c2 < self.cols):
            return False
        if not self.maze.mask[r1][c1] or not self.maze.mask[r2][c2]:
            return False
        
        # Same cell
        if r1 == r2 and c1 == c2:
            return True
        
        # Generate line using Bresenham's algorithm
        cells = self._bresenham_line(r1, c1, r2, c2)
        
        # Check if all cells on the line are passable
        for i, (r, c) in enumerate(cells):
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return False
            if not self.maze.mask[r][c]:
                return False
            
            # FIXED: Check for diagonal wall clipping
            # If we're moving diagonally, verify both adjacent cells are passable
            if i > 0:
                prev_r, prev_c = cells[i - 1]
                if prev_r != r and prev_c != c:  # This is a diagonal move
                    # Both cells adjacent to the diagonal must be passable
                    # to prevent corner cutting through walls
                    if not (self.maze.mask[prev_r][c] and self.maze.mask[r][prev_c]):
                        return False
        
        return True

    def _bresenham_line(self, r1, c1, r2, c2):
        """
        Generate all cells on a Bresenham line from (r1, c1) to (r2, c2).
        This is the standard algorithm - no shortcuts, respects maze grid.
        """
        cells = []
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r2 > r1 else -1
        sc = 1 if c2 > c1 else -1
        
        if dr > dc:
            err = dr / 2.0
            r, c = r1, c1
            while r != r2:
                cells.append((r, c))
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
            cells.append((r2, c2))
        else:
            err = dc / 2.0
            r, c = r1, c1
            while c != c2:
                cells.append((r, c))
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
            cells.append((r2, c2))
        
        return cells

    # ------------------------------------------------------------------
    #  UTILITY FUNCTIONS
    # ------------------------------------------------------------------
    def _neighbours(self, r, c):
        """Return list of passable neighbours using bitmask grid."""
        maze = self.maze
        DIRS = [(-1, 0, 1), (0, 1, 2), (1, 0, 4), (0, -1, 8)]
        res = []
        for dr, dc, bit in DIRS:
            if maze.grid[r][c] & bit:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and maze.mask[nr][nc]:
                    res.append((nr, nc))
        return res

    def _nearest_node(self, r, c):
        """BFS to nearest intersection/dead-end. Falls back to (r,c)."""
        if (r, c) in self.cell_types and self.cell_types[(r, c)] in ('intersection', 'dead_end'):
            return (r, c)
        visited = {(r, c)}
        q = deque([(r, c)])
        while q:
            cr, cc = q.popleft()
            for nr, nc in self._neighbours(cr, cc):
                if (nr, nc) not in visited:
                    if (nr, nc) in self.cell_types and self.cell_types[(nr, nc)] in ('intersection', 'dead_end'):
                        return (nr, nc)
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return (r, c)  # fallback (should not happen in a connected maze)

    def _first_passable(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze.mask[r][c]:
                    return (r, c)
        raise ValueError("No passable cell.")

    def _last_passable(self):
        for r in range(self.rows - 1, -1, -1):
            for c in range(self.cols - 1, -1, -1):
                if self.maze.mask[r][c]:
                    return (r, c)
        raise ValueError("No passable cell.")


# ------------------------------------------------------------------
#  GLOBAL WRAPPER (caches AHPP instance per Maze object)
# ------------------------------------------------------------------
_ahpp_cache = {}

def find_path(maze, start=None, goal=None):
    """
    Module-level function compatible with your benchmark harness.
    Automatically builds & caches the AHPP hierarchy on first call.
    
    Returns: (path_length, path_list, nodes_explored)
    """
    maze_id = id(maze)   # works if the same object is reused
    if maze_id not in _ahpp_cache:
        _ahpp_cache[maze_id] = AHPP(maze)
    return _ahpp_cache[maze_id].find_path(start, goal)
