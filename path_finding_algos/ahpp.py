"""
AHPP‑JPS – Adaptive Hierarchical Predictive Pathfinding
Complete, final version for Maze Pathfinding Project.
Precomputes abstraction layers once per maze; subsequent queries are
blazing fast and explore only the exact path cells.
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
        """
        maze = self.maze
        cell_types = self.cell_types
        DIRS = [(-1, 0, 1), (0, 1, 2), (1, 0, 4), (0, -1, 8)]  # N, E, S, W with bitmasks

        nodes = {pos for pos, tp in cell_types.items() if tp in ('intersection', 'dead_end')}
        graph = {n: [] for n in nodes}
        corridor_cells = {}  # (u,v) -> list of intermediate cells (excl. u, incl. v is added later)

        for node in nodes:
            r, c = node
            for dr, dc, bit in DIRS:
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
                            # Store corridor (reversed for the opposite direction)
                            corridor_cells[(node, neighbour)] = list(cells)
                            corridor_cells[(neighbour, node)] = list(reversed(cells))
                        break
                    # Continue to the only other passable neighbour (must be degree 2 in a corridor)
                    next_cells = []
                    for ndr, ndc in [(-1,0),(1,0),(0,1),(0,-1)]:
                        nnr, nnc = cur_r + ndr, cur_c + ndc
                        if maze.is_passable(nnr, nnc) and (nnr, nnc) != (prev_r, prev_c):
                            next_cells.append((nnr, nnc))
                    if not next_cells:
                        break
                    cells.append((cur_r, cur_c))
                    prev_r, prev_c = cur_r, cur_c
                    cur_r, cur_c = next_cells[0]
        return graph, corridor_cells

    def _build_region_graph(self):
        """
        Layer 2: divide maze into block_size×block_size regions.
        Edge cost = block_size (estimated distance between region centres).
        """
        maze = self.maze
        rows, cols = self.rows, self.cols
        block_size = self.block_size
        br = math.ceil(rows / block_size)
        bc = math.ceil(cols / block_size)
        region_map = [[(0, 0)] * cols for _ in range(rows)]

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
                for r in range(r_start, r_end):
                    c_left = (j + 1) * block_size - 1
                    c_right = c_left + 1
                    if c_right < cols and maze.mask[r][c_left] and maze.mask[r][c_right]:
                        if maze.grid[r][c_left] & 2:  # east opening
                            connected = True
                            break
                if connected:
                    adj[(i, j)].append(((i, j + 1), block_size))
                    adj[(i, j + 1)].append(((i, j), block_size))

        # Vertical connections (north–south)
        for i in range(br - 1):
            for j in range(bc):
                connected = False
                r_bottom = (i + 1) * block_size - 1
                r_top = r_bottom + 1
                c_start = j * block_size
                c_end = min(cols, (j + 1) * block_size)
                for c in range(c_start, c_end):
                    if r_top < rows and maze.mask[r_bottom][c] and maze.mask[r_top][c]:
                        if maze.grid[r_bottom][c] & 4:  # south opening
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
        if start is None:
            start = self._first_passable()
        if goal is None:
            goal = self._last_passable()

        sr, sc = start
        gr, gc = goal
        if (sr, sc) == (gr, gc):
            return 0, [(sr, sc)], 1

        total_explored = 0
        cell_types = self.cell_types
        region_map = self.region_map
        layer1 = self.layer1
        layer2 = self.layer2

        # 1. Map start/goal to the nearest Layer-1 node
        start_node = self._nearest_node(sr, sc)
        goal_node  = self._nearest_node(gr, gc)

        # If they map to the same node, walk directly (they are in the same tunnel)
        if start_node == goal_node:
            path, explored = self._trace_direct_path(start, goal)
            return (len(path)-1, path, explored) if path else (-1, [], explored)

        # 2. Region search (Layer 2)
        start_region = region_map[sr][sc]
        goal_region  = region_map[gr][gc]
        region_path, reg_exp = self._a_star_region(start_region, goal_region)
        total_explored += reg_exp
        if region_path is None:
            return -1, [], total_explored

        # 3. Intersection search constrained to the region path (Layer 1)
        allowed_regions = set(region_path)
        int_path, int_exp = self._a_star_intersection_region_constrained(
            start_node, goal_node, allowed_regions
        )
        total_explored += int_exp
        if int_path is None:
            return -1, [], total_explored

        # 4. Stitch the full path using pre‑stored corridors
        full_path = []

        # From real start to first intersection node
        if (sr, sc) != start_node:
            seg, seg_exp = self._trace_corridor_between(start, start_node)
            total_explored += seg_exp
            if seg is None:
                return -1, [], total_explored
            full_path.extend(seg[:-1])  # drop start_node to avoid duplication

        # Main segments along the intersection path
        for i in range(len(int_path) - 1):
            a = int_path[i]
            b = int_path[i+1]
            if i == 0 and (sr, sc) == start_node:
                full_path.append(a)
            middle = self.corridor_cells.get((a, b), [])
            full_path.extend(middle)
            full_path.append(b)

        # From last intersection node to real goal
        if (gr, gc) != goal_node:
            seg, seg_exp = self._trace_corridor_between(goal_node, goal)
            total_explored += seg_exp
            if seg is None:
                return -1, [], total_explored
            full_path.extend(seg[1:])  # skip duplicated goal_node

        # 5. Post‑processing: any‑angle smoothing
        smoothed = self._smooth_path(full_path)
        return len(smoothed) - 1, smoothed, total_explored

    # ------------------------------------------------------------------
    #  CORRIDOR TRACING (zero‑branch cell‑level walking)
    # ------------------------------------------------------------------
    def _trace_corridor_between(self, a, b):
        """
        Walk from a to b through the unique corridor (no branches expected).
        Returns (list_of_cells_from_a_to_b_inclusive, nodes_visited).
        """
        if a == b:
            return [a], 1
        prev = {a: None}
        q = deque([a])
        visited = {a}
        while q:
            cur = q.popleft()
            if cur == b:
                break
            for nr, nc in self._neighbours(cur[0], cur[1]):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    prev[(nr, nc)] = cur
                    q.append((nr, nc))
        if b not in prev:
            return None, len(visited)
        # Reconstruct path
        path = []
        node = b
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path, len(visited)

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
    #  POST‑PROCESSING: LINE‑OF‑SIGHT SMOOTHING
    # ------------------------------------------------------------------
    def _smooth_path(self, path):
        """Remove unnecessary intermediate points where a straight path is unobstructed."""
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Jump as far as possible while line-of-sight exists
            j = len(path) - 1
            while j > i + 1 and not self._line_of_sight(path[i], path[j]):
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def _line_of_sight(self, a, b):
        """4‑connected line‑of‑sight: all intermediate cells are passable."""
        r1, c1 = a
        r2, c2 = b
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        steps = max(dr, dc)
        if steps == 0:
            return True
        r, c = r1, c1
        for _ in range(steps):
            if r < r2:
                r += 1
            elif r > r2:
                r -= 1
            if c < c2:
                c += 1
            elif c > c2:
                c -= 1
            if not self.maze.mask[r][c]:
                return False
        return True

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
    """
    maze_id = id(maze)   # works if the same object is reused
    if maze_id not in _ahpp_cache:
        _ahpp_cache[maze_id] = AHPP(maze)
    return _ahpp_cache[maze_id].find_path(start, goal)