"""
pathfinding_visualiser.py
=========================
Pathfinding Animation Generator
Authors : (your group names here)
Course  : CSE 317 – Design and Analysis of Algorithms, Spring 2026

Overview
--------
Generates an animated GIF for every (algorithm × maze) combination showing:
  1. Maze walls rendered in full.
  2. Explored / expanded cells appearing frame-by-frame (light blue).
  3. Final optimal path highlighted in orange.
  4. Start cell (green dot) and end cell (red dot) always on top.

Algorithms animated
-------------------
1. Theta*                 (any-angle A*)
2. Bidirectional Dijkstra (dual-frontier)
3. D* Lite                (incremental A*)
4. ALT                    (landmark A*)
5. AHPP                   (hierarchical predictive — shows intersection nodes
                           and corridor stitching in a distinct colour)

Output
------
path_finding_algos/animations/<AlgoName>_<ShapeName>.gif

Usage
-----
    cd path_finding_algos
    python pathfinding_visualiser.py [--cell N] [--fps N] [--skip N]

    --cell  pixel size per maze cell   (default 6)
    --fps   frames per second          (default 20)
    --skip  explored cells per frame   (default 5, higher = smaller/faster GIF)
"""

import argparse
import copy
import heapq
import math
from collections import deque
from pathlib import Path
from PIL import Image, ImageDraw

# ================================================================
# PATH RESOLUTION
# ================================================================
_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
BEST_DIR = _ROOT / "mazes_for_pathfinding" / "best_mazes"
ANIM_DIR = _HERE / "animations"
ANIM_DIR.mkdir(exist_ok=True)

# ================================================================
# COLOUR PALETTE
# ================================================================
COL_WALL         = (30,  30,  30)
COL_BG           = (255, 255, 255)
COL_OUTSIDE      = (20,  20,  20)
COL_EXPLORED     = (173, 216, 230)   # light blue   – cells expanded
COL_FRONTIER     = (100, 149, 237)   # cornflower   – open-set cells
COL_PATH         = (255, 165,   0)   # orange       – final path
COL_START        = (0,   200,   0)   # green        – start
COL_END          = (220,  20,  60)   # crimson      – end
COL_AHPP_NODE    = (180,  90, 220)   # purple       – AHPP intersection nodes
COL_AHPP_CORRIDOR= (220, 180, 255)   # light purple – AHPP corridor stitching


# ================================================================
# MAZE  (self-contained copy — no import from other files)
# ================================================================

class Maze:
    """Lightweight maze loaded from a .maze file."""

    N, E, S, W = 1, 2, 4, 8
    DX   = {1: -1, 2: 0,  4: 1,  8: 0}
    DY   = {1:  0, 2:  1, 4: 0,  8: -1}
    DIRS = [1, 2, 4, 8]

    def __init__(self, rows, cols, grid, mask, start, end):
        self.rows = rows; self.cols = cols
        self.grid = grid; self.mask = mask
        self.start = start; self.end = end

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_passable(self, r, c):
        return self.in_bounds(r, c) and self.mask[r][c]

    def passable_neighbours(self, r, c):
        for d in self.DIRS:
            if self.grid[r][c] & d:
                nr = r + self.DX[d]; nc = c + self.DY[d]
                if self.in_bounds(nr, nc) and self.mask[nr][nc]:
                    yield nr, nc

    def line_of_sight(self, r0, c0, r1, c1):
        dr = abs(r1-r0); dc = abs(c1-c0)
        r, c = r0, c0
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        if dc == 0:
            for _ in range(dr):
                d = self.S if sr > 0 else self.N
                if not (self.grid[r][c] & d): return False
                r += sr
            return True
        if dr == 0:
            for _ in range(dc):
                d = self.E if sc > 0 else self.W
                if not (self.grid[r][c] & d): return False
                c += sc
            return True
        err = dc - dr
        while r != r1 or c != c1:
            e2 = 2 * err
            if e2 > -dr:
                err -= dr; d = self.S if sr > 0 else self.N
                if not (self.grid[r][c] & d): return False
                r += sr
            if e2 < dc:
                err += dc; d = self.E if sc > 0 else self.W
                if not (self.grid[r][c] & d): return False
                c += sc
        return True

    def block_cell(self, r, c):
        opp = {1: 4, 4: 1, 2: 8, 8: 2}
        for d in self.DIRS:
            if self.grid[r][c] & d:
                nr = r + self.DX[d]; nc = c + self.DY[d]
                if self.in_bounds(nr, nc):
                    self.grid[nr][nc] &= ~opp[d]
        self.grid[r][c] = 0; self.mask[r][c] = False

    @staticmethod
    def from_file(path):
        with open(path) as f:
            rows, cols = map(int, f.readline().split())
            start = end = None
            line2 = f.readline().strip()
            if line2.startswith("start="):
                p = line2.split()
                sr, sc = map(int, p[0].split("=")[1].split(","))
                er, ec = map(int, p[1].split("=")[1].split(","))
                start = (sr, sc) if sr >= 0 else None
                end   = (er, ec) if er >= 0 else None
                data  = [f.readline() for _ in range(rows)]
            else:
                data = [line2] + [f.readline() for _ in range(rows - 1)]
            mask = [[True]  * cols for _ in range(rows)]
            grid = [[0]     * cols for _ in range(rows)]
            for r, line in enumerate(data):
                parts = line.strip().split(" | ")
                if len(parts) == 2:
                    mask[r] = [bool(int(x)) for x in parts[0].split()]
                    grid[r] = [int(x)       for x in parts[1].split()]
                else:
                    grid[r] = [int(x) for x in parts[0].split()]
        return Maze(rows, cols, grid, mask, start, end)


# ================================================================
# HELPERS
# ================================================================

def euclidean(r0, c0, r1, c1):
    return math.hypot(r1 - r0, c1 - c0)

def reconstruct_path(came_from, start, end):
    path = []; node = end
    while node is not None:
        path.append(node); node = came_from.get(node)
    path.reverse()
    return path if (path and path[0] == start) else []


# ================================================================
# INSTRUMENTED ALGORITHM GENERATORS
# Each yields (event_type, cell) events during the search so the
# visualiser can capture exploration order frame-by-frame.
#
# event_type values
# -----------------
# 'explore'        – cell popped from open set (being expanded)
# 'frontier'       – cell pushed to open set
# 'path'           – cell is on the final optimal path
# 'ahpp_node'      – AHPP intersection / dead-end node (purple)
# 'ahpp_corridor'  – cell being stitched from a pre-stored corridor
# ================================================================

# ----------------------------------------------------------------
# 1. Theta*
# ----------------------------------------------------------------
def theta_star_events(maze: Maze):
    """Theta* — any-angle A* with line-of-sight shortcutting."""
    start, end = maze.start, maze.end
    if start is None or end is None: return

    g = {start: 0.0}; came_from = {start: None}
    open_set = [(euclidean(*start, *end), start)]; closed = set()

    while open_set:
        _, s = heapq.heappop(open_set)
        if s in closed: continue
        closed.add(s); yield ('explore', s)
        if s == end:
            for cell in reconstruct_path(came_from, start, end):
                yield ('path', cell)
            return
        for nr, nc in maze.passable_neighbours(*s):
            s2 = (nr, nc)
            if s2 in closed: continue
            parent_s = came_from[s]
            if parent_s is not None and maze.line_of_sight(*parent_s, *s2):
                g_new = g[parent_s] + euclidean(*parent_s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2] = g_new; came_from[s2] = parent_s
                    heapq.heappush(open_set, (g_new + euclidean(*s2, *end), s2))
                    yield ('frontier', s2)
            else:
                g_new = g[s] + euclidean(*s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2] = g_new; came_from[s2] = s
                    heapq.heappush(open_set, (g_new + euclidean(*s2, *end), s2))
                    yield ('frontier', s2)


# ----------------------------------------------------------------
# 2. Bidirectional Dijkstra
# ----------------------------------------------------------------
def bidirectional_dijkstra_events(maze: Maze):
    """Bidirectional Dijkstra — dual frontier meeting in the middle."""
    start, end = maze.start, maze.end
    if start is None or end is None: return

    dist_f = {start: 0}; prev_f = {start: None}; open_f = [(0, start)]
    dist_b = {end:   0}; prev_b = {end:   None}; open_b = [(0, end)]
    settled_f = set(); settled_b = set()
    best = math.inf; meeting = None

    def _build(node):
        pf = []; n = node
        while n is not None: pf.append(n); n = prev_f.get(n)
        pf.reverse()
        pb = []; n = prev_b.get(node)
        while n is not None: pb.append(n); n = prev_b.get(n)
        return pf + pb

    while open_f or open_b:
        expand_fwd = (open_f and (not open_b or open_f[0][0] <= open_b[0][0]))
        if expand_fwd and open_f:
            d, u = heapq.heappop(open_f)
            if u in settled_f: continue
            settled_f.add(u); yield ('explore', u)
            if d + dist_b.get(u, math.inf) < best:
                best = d + dist_b.get(u, math.inf); meeting = u
            if open_f and open_b and open_f[0][0] + open_b[0][0] >= best: break
            for nr, nc in maze.passable_neighbours(*u):
                v = (nr, nc); gn = dist_f[u] + 1
                if gn < dist_f.get(v, math.inf):
                    dist_f[v] = gn; prev_f[v] = u
                    heapq.heappush(open_f, (gn, v)); yield ('frontier', v)
        elif open_b:
            d, u = heapq.heappop(open_b)
            if u in settled_b: continue
            settled_b.add(u); yield ('explore', u)
            if dist_f.get(u, math.inf) + d < best:
                best = dist_f.get(u, math.inf) + d; meeting = u
            if open_f and open_b and open_f[0][0] + open_b[0][0] >= best: break
            for nr, nc in maze.passable_neighbours(*u):
                v = (nr, nc); gn = dist_b[u] + 1
                if gn < dist_b.get(v, math.inf):
                    dist_b[v] = gn; prev_b[v] = u
                    heapq.heappush(open_b, (gn, v)); yield ('frontier', v)

    if meeting:
        for cell in _build(meeting): yield ('path', cell)


# ----------------------------------------------------------------
# 3. D* Lite
# ----------------------------------------------------------------
def d_star_lite_events(maze: Maze):
    """D* Lite — incremental A* (initial search only for animation)."""
    start, end = maze.start, maze.end
    if start is None or end is None: return

    INF = math.inf
    g   = {}; rhs = {}
    for r in range(maze.rows):
        for c in range(maze.cols):
            if maze.mask[r][c]: g[(r,c)] = rhs[(r,c)] = INF

    k_m = 0; rhs[end] = 0
    U = []; U_set = {}

    def h(s): return euclidean(*s, *start)
    def key(s):
        m = min(g.get(s, INF), rhs.get(s, INF))
        return (m + h(s) + k_m, m)
    def push(s):
        k = key(s); heapq.heappush(U, (k, s)); U_set[s] = k
    def update(u):
        if u != end:
            nbrs = list(maze.passable_neighbours(*u))
            rhs[u] = min((1 + g.get(s2, INF) for s2 in nbrs), default=INF)
        if u in U_set: del U_set[u]
        if g.get(u, INF) != rhs.get(u, INF): push(u)

    push(end); explored_set = set()
    while U:
        k_old, u = heapq.heappop(U)
        if U_set.get(u) != k_old: continue
        k_new = key(u)
        if k_old < k_new:
            heapq.heappush(U, (k_new, u)); U_set[u] = k_new; continue
        if g.get(u, INF) > rhs.get(u, INF):
            g[u] = rhs[u]; del U_set[u]
            if u not in explored_set: explored_set.add(u); yield ('explore', u)
            for s in maze.passable_neighbours(*u): update(s)
        else:
            g[u] = INF; update(u)
            if u not in explored_set: explored_set.add(u); yield ('explore', u)
            for s in maze.passable_neighbours(*u): update(s)
        sk = key(start)
        if U and U[0][0] >= sk and rhs.get(start, INF) == g.get(start, INF):
            break

    # Extract and yield path
    path = [start]; cur = start; vis = {start}
    while cur != end:
        nbrs = list(maze.passable_neighbours(*cur))
        if not nbrs: break
        best_n = min(nbrs, key=lambda s: g.get(s, INF))
        if g.get(best_n, INF) == INF or best_n in vis: break
        vis.add(best_n); path.append(best_n); cur = best_n
    for cell in path: yield ('path', cell)


# ----------------------------------------------------------------
# 4. ALT
# ----------------------------------------------------------------
def alt_events(maze: Maze, num_landmarks: int = 8):
    """ALT — Landmark A* with triangle-inequality heuristic."""
    start, end = maze.start, maze.end
    if start is None or end is None: return

    import random as _rnd
    passable = [(r, c) for r in range(maze.rows)
                for c in range(maze.cols) if maze.mask[r][c]]
    if not passable: return

    def bfs_dist(src):
        dist = {src: 0}; q = [src]; head = 0
        while head < len(q):
            node = q[head]; head += 1
            for nr, nc in maze.passable_neighbours(*node):
                nb = (nr, nc)
                if nb not in dist: dist[nb] = dist[node]+1; q.append(nb)
        return dist

    landmarks = [_rnd.choice(passable)]; ldists = [bfs_dist(landmarks[0])]
    for _ in range(num_landmarks - 1):
        farthest = max(passable,
                       key=lambda cell: min(ld.get(cell, math.inf) for ld in ldists))
        landmarks.append(farthest); ldists.append(bfs_dist(farthest))

    def h_alt(s):
        best = 0
        for ld in ldists:
            a = ld.get(end, math.inf); b = ld.get(s, math.inf)
            if a < math.inf and b < math.inf: best = max(best, abs(a - b))
        return best

    g_cost = {start: 0}; came_from = {start: None}
    open_set = [(h_alt(start), start)]; closed = set()
    while open_set:
        _, s = heapq.heappop(open_set)
        if s in closed: continue
        closed.add(s); yield ('explore', s)
        if s == end:
            for cell in reconstruct_path(came_from, start, end):
                yield ('path', cell)
            return
        for nr, nc in maze.passable_neighbours(*s):
            nb = (nr, nc)
            if nb in closed: continue
            gn = g_cost[s] + 1
            if gn < g_cost.get(nb, math.inf):
                g_cost[nb] = gn; came_from[nb] = s
                heapq.heappush(open_set, (gn + h_alt(nb), nb))
                yield ('frontier', nb)


# ----------------------------------------------------------------
# 5. AHPP — Adaptive Hierarchical Predictive Pathfinding
# ----------------------------------------------------------------

class _AHPPVis:
    """
    Stripped-down AHPP that yields visualisation events during pathfinding.
    Intersection nodes are yielded as 'ahpp_node' (purple).
    Corridor cells being stitched are yielded as 'ahpp_corridor' (light purple).
    The final path cells are yielded as 'path' (orange).
    """

    def __init__(self, maze: Maze, block_size: int = 10):
        self.maze       = maze
        self.rows       = maze.rows
        self.cols       = maze.cols
        self.block_size = block_size
        self.cell_types, self.constraint_map = self._analyse_maze()
        self.layer1, self.corridor_cells     = self._build_intersection_graph()
        self.layer2, self.region_map         = self._build_region_graph()

    def _analyse_maze(self):
        maze = self.maze; cell_types = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if not maze.mask[r][c]: continue
                deg = sum(1 for _ in self._neighbours(r, c))
                if   deg == 0: tp = 'isolated'
                elif deg == 1: tp = 'dead_end'
                elif deg == 2: tp = 'corridor'
                else:          tp = 'intersection'
                cell_types[(r, c)] = tp
        constraint = [[0.0]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if not maze.mask[r][c]: continue
                open_cnt = sum(
                    1 for dr in range(-2, 3) for dc in range(-2, 3)
                    if 0 <= r+dr < self.rows and 0 <= c+dc < self.cols
                    and maze.mask[r+dr][c+dc])
                constraint[r][c] = (25 - open_cnt) / 25.0
        return cell_types, constraint

    def _build_intersection_graph(self):
        maze = self.maze; cell_types = self.cell_types
        DIRS4 = [(-1,0,1),(0,1,2),(1,0,4),(0,-1,8)]
        nodes = {pos for pos, tp in cell_types.items()
                 if tp in ('intersection', 'dead_end')}
        graph = {n: [] for n in nodes}; corridor_cells = {}
        for node in nodes:
            r, c = node
            for dr, dc, bit in DIRS4:
                if not (maze.grid[r][c] & bit): continue
                nr, nc = r+dr, c+dc
                if not maze.is_passable(nr, nc): continue
                prev_r, prev_c = r, c; cur_r, cur_c = nr, nc; cells = []
                while True:
                    if (cur_r, cur_c) in nodes:
                        nb = (cur_r, cur_c)
                        if not any(n == nb for n, _ in graph[node]):
                            length = len(cells) + 1
                            graph[node].append((nb, length))
                            graph[nb].append((node, length))
                            corridor_cells[(node, nb)] = list(cells)
                            corridor_cells[(nb, node)] = list(reversed(cells))
                        break
                    next_cells = [
                        (cur_r+ndr, cur_c+ndc)
                        for ndr, ndc in [(-1,0),(1,0),(0,1),(0,-1)]
                        if maze.is_passable(cur_r+ndr, cur_c+ndc)
                        and (cur_r+ndr, cur_c+ndc) != (prev_r, prev_c)]
                    if not next_cells: break
                    cells.append((cur_r, cur_c))
                    prev_r, prev_c = cur_r, cur_c
                    cur_r, cur_c   = next_cells[0]
        return graph, corridor_cells

    def _build_region_graph(self):
        maze = self.maze; rows, cols = self.rows, self.cols
        bs = self.block_size
        br = math.ceil(rows / bs); bc = math.ceil(cols / bs)
        region_map = [[(0,0)]*cols for _ in range(rows)]
        for i in range(br):
            for j in range(bc):
                for r in range(i*bs, min(rows, (i+1)*bs)):
                    for c in range(j*bs, min(cols, (j+1)*bs)):
                        region_map[r][c] = (i, j)
        adj = {(i,j): [] for i in range(br) for j in range(bc)}
        for i in range(br):
            for j in range(bc - 1):
                for r in range(i*bs, min(rows, (i+1)*bs)):
                    c_l = (j+1)*bs - 1; c_r = c_l + 1
                    if (c_r < cols and maze.mask[r][c_l]
                            and maze.mask[r][c_r] and maze.grid[r][c_l] & 2):
                        adj[(i,j)].append(((i,j+1), bs))
                        adj[(i,j+1)].append(((i,j), bs)); break
        for i in range(br - 1):
            for j in range(bc):
                for c in range(j*bs, min(cols, (j+1)*bs)):
                    r_b = (i+1)*bs - 1; r_t = r_b + 1
                    if (r_t < rows and maze.mask[r_b][c]
                            and maze.mask[r_t][c] and maze.grid[r_b][c] & 4):
                        adj[(i,j)].append(((i+1,j), bs))
                        adj[(i+1,j)].append(((i,j), bs)); break
        return adj, region_map

    def find_path_events(self, start=None, goal=None):
        """
        Generator — yields (event_type, cell) events during pathfinding.
        ahpp_node    → intersection / dead-end node visited during Layer-1 A*
        ahpp_corridor → cell being stitched from a pre-stored corridor
        path          → final path cell
        """
        if start is None: start = self.maze.start
        if goal  is None: goal  = self.maze.end
        if start is None or goal is None: return

        sr, sc = start; gr, gc = goal
        if (sr, sc) == (gr, gc):
            yield ('path', start); return

        start_node = self._nearest_node(sr, sc)
        goal_node  = self._nearest_node(gr, gc)

        # Yield intersection nodes as they are identified
        yield ('ahpp_node', start_node)
        yield ('ahpp_node', goal_node)

        if start_node == goal_node:
            path, _ = self._trace_corridor_between(start, goal)
            if path:
                for cell in path: yield ('path', cell)
            return

        # Layer 2 — region A*
        start_reg = self.region_map[sr][sc]
        goal_reg  = self.region_map[gr][gc]
        reg_path, _ = self._a_star_region(start_reg, goal_reg)
        if reg_path is None: return

        # Layer 1 — intersection A* (yield nodes as explored)
        allowed  = set(reg_path)
        int_path, _ = self._a_star_intersections_events(
            start_node, goal_node, allowed)
        if int_path is None: return

        # Stitch full path — yield corridor cells as ahpp_corridor
        full_path = []
        if (sr, sc) != start_node:
            seg, _ = self._trace_corridor_between(start, start_node)
            if seg is None: return
            for cell in seg[:-1]:
                yield ('ahpp_corridor', cell)
            full_path.extend(seg[:-1])

        for i in range(len(int_path) - 1):
            a = int_path[i]; b = int_path[i+1]
            if i == 0 and (sr, sc) == start_node:
                full_path.append(a)
            corridor = self.corridor_cells.get((a, b), [])
            for cell in corridor:
                yield ('ahpp_corridor', cell)
            full_path.extend(corridor)
            full_path.append(b)

        if (gr, gc) != goal_node:
            seg, _ = self._trace_corridor_between(goal_node, goal)
            if seg is None: return
            for cell in seg[1:]:
                yield ('ahpp_corridor', cell)
            full_path.extend(seg[1:])

        # Smooth and yield final path
        smoothed = self._smooth_path(full_path)
        for cell in smoothed: yield ('path', cell)

    def _a_star_region(self, start_reg, goal_reg):
        if start_reg == goal_reg: return [start_reg], 1
        gi, gj = goal_reg; bs = self.block_size
        def h(reg): return math.hypot(reg[0]-gi, reg[1]-gj) * bs
        open_set = [(h(start_reg), 0, start_reg, [start_reg])]
        visited = set(); explored = 0
        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if node in visited: continue
            visited.add(node); explored += 1
            if node == goal_reg: return path, explored
            for nb, w in self.layer2[node]:
                if nb not in visited:
                    ng = g + w
                    heapq.heappush(open_set, (ng+h(nb), ng, nb, path+[nb]))
        return None, explored

    def _a_star_intersections_events(self, start_node, goal_node, allowed):
        """A* on intersection graph — also yields ahpp_node events."""
        if start_node == goal_node: return [start_node], 1
        gr, gc = goal_node
        def h(node): return math.hypot(node[0]-gr, node[1]-gc)
        open_set = [(h(start_node), 0, start_node, [start_node])]
        visited = set(); explored = 0
        # We collect events inside and return path; caller yields them
        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if node in visited: continue
            visited.add(node); explored += 1
            if node == goal_node: return path, explored
            for nb, w in self.layer1[node]:
                if nb in visited: continue
                nr, nc = nb
                if self.region_map[nr][nc] not in allowed: continue
                ng = g + w
                heapq.heappush(open_set, (ng+h(nb), ng, nb, path+[nb]))
        return None, explored

    def _trace_corridor_between(self, a, b):
        if a == b: return [a], 1
        prev = {a: None}; q = deque([a]); visited = {a}
        while q:
            cur = q.popleft()
            if cur == b: break
            for nr, nc in self._neighbours(cur[0], cur[1]):
                if (nr, nc) not in visited:
                    visited.add((nr, nc)); prev[(nr, nc)] = cur; q.append((nr, nc))
        if b not in prev: return None, len(visited)
        path = []; node = b
        while node is not None: path.append(node); node = prev[node]
        path.reverse(); return path, len(visited)

    def _smooth_path(self, path):
        if len(path) <= 2: return path
        smoothed = [path[0]]; i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i+1 and not self._los(path[i], path[j]): j -= 1
            smoothed.append(path[j]); i = j
        return smoothed

    def _los(self, a, b):
        r1, c1 = a; r2, c2 = b
        steps = max(abs(r2-r1), abs(c2-c1))
        if steps == 0: return True
        r, c = r1, c1
        for _ in range(steps):
            if r < r2: r += 1
            elif r > r2: r -= 1
            if c < c2: c += 1
            elif c > c2: c -= 1
            if not self.maze.mask[r][c]: return False
        return True

    def _neighbours(self, r, c):
        maze = self.maze; res = []
        for dr, dc, bit in [(-1,0,1),(0,1,2),(1,0,4),(0,-1,8)]:
            if maze.grid[r][c] & bit:
                nr, nc = r+dr, c+dc
                if maze.is_passable(nr, nc): res.append((nr, nc))
        return res

    def _nearest_node(self, r, c):
        if self.cell_types.get((r,c)) in ('intersection', 'dead_end'):
            return (r, c)
        visited = {(r,c)}; q = deque([(r,c)])
        while q:
            cr, cc = q.popleft()
            for nr, nc in self._neighbours(cr, cc):
                if (nr,nc) not in visited:
                    if self.cell_types.get((nr,nc)) in ('intersection','dead_end'):
                        return (nr, nc)
                    visited.add((nr,nc)); q.append((nr,nc))
        return (r, c)


def ahpp_events(maze: Maze):
    """
    AHPP events generator.
    Builds the hierarchy then yields search + path events for animation.
    Uses distinct colours:
        ahpp_node     → purple  (intersection / dead-end nodes visited)
        ahpp_corridor → light purple  (corridor cells stitched together)
        path          → orange  (final smoothed path)
    """
    vis = _AHPPVis(maze)
    yield from vis.find_path_events(maze.start, maze.end)


# ================================================================
# ALGORITHM EVENT GENERATORS REGISTRY
# ================================================================

ALGO_EVENTS = {
    "ThetaStar":             theta_star_events,
    "BidirectionalDijkstra": bidirectional_dijkstra_events,
    "DStarLite":             d_star_lite_events,
    "ALT":                   alt_events,
    "AHPP":                  ahpp_events,          # [ADDED]
}


# ================================================================
# FRAME RENDERER
# ================================================================

class MazeRenderer:
    """
    Renders maze frames as PIL Images.
    Tracks a per-cell colour dict that is updated as search events arrive.
    """

    def __init__(self, maze: Maze, cell_size: int = 6, wall_width: int = 1):
        self.maze = maze
        self.cs   = cell_size
        self.ww   = max(1, wall_width)
        self.img_w = maze.cols * cell_size + self.ww
        self.img_h = maze.rows * cell_size + self.ww
        self.colours = {}   # (r,c) → RGB

    def _base_image(self) -> Image.Image:
        """Draw maze walls on a white background; non-mask cells filled black."""
        img  = Image.new("RGB", (self.img_w, self.img_h), COL_BG)
        draw = ImageDraw.Draw(img)

        # Fill outside cells
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if not self.maze.mask[r][c]:
                    x1 = c * self.cs; y1 = r * self.cs
                    draw.rectangle([x1, y1,
                                     x1 + self.cs + self.ww,
                                     y1 + self.cs + self.ww],
                                    fill=COL_OUTSIDE)

        # Draw walls
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if not self.maze.mask[r][c]: continue
                x1 = c*self.cs; y1 = r*self.cs
                x2 = x1+self.cs; y2 = y1+self.cs; wh = self.ww
                nr2 = r - 1
                if not (self.maze.in_bounds(nr2, c)
                        and self.maze.mask[nr2][c]
                        and self.maze.grid[r][c] & self.maze.N):
                    draw.line([(x1,y1),(x2,y1)], fill=COL_WALL, width=wh)
                sr2 = r + 1
                if not (self.maze.in_bounds(sr2, c)
                        and self.maze.mask[sr2][c]
                        and self.maze.grid[r][c] & self.maze.S):
                    draw.line([(x1,y2),(x2,y2)], fill=COL_WALL, width=wh)
                wc2 = c - 1
                if not (self.maze.in_bounds(r, wc2)
                        and self.maze.mask[r][wc2]
                        and self.maze.grid[r][c] & self.maze.W):
                    draw.line([(x1,y1),(x1,y2)], fill=COL_WALL, width=wh)
                ec2 = c + 1
                if not (self.maze.in_bounds(r, ec2)
                        and self.maze.mask[r][ec2]
                        and self.maze.grid[r][c] & self.maze.E):
                    draw.line([(x2,y1),(x2,y2)], fill=COL_WALL, width=wh)
        return img

    def render(self, highlight_path=None) -> Image.Image:
        """
        Composite the current colour overlay onto the base maze image.

        Parameters
        ----------
        highlight_path : list[(r,c)] | None
            If given, these cells are drawn in COL_PATH on top of everything.
        """
        img  = self._base_image()
        draw = ImageDraw.Draw(img)
        dot  = max(1, self.cs // 3)

        # Cell colour overlay
        for (r, c), col in self.colours.items():
            x1 = c*self.cs + self.ww; y1 = r*self.cs + self.ww
            x2 = x1 + self.cs - self.ww; y2 = y1 + self.cs - self.ww
            draw.rectangle([x1, y1, x2, y2], fill=col)

        # Path overlay
        if highlight_path:
            for r, c in highlight_path:
                x1 = c*self.cs + self.ww; y1 = r*self.cs + self.ww
                x2 = x1 + self.cs - self.ww; y2 = y1 + self.cs - self.ww
                draw.rectangle([x1, y1, x2, y2], fill=COL_PATH)

        # Start / end dots always on top
        if self.maze.start:
            sr, sc = self.maze.start
            cx_s = sc*self.cs + self.cs//2; cy_s = sr*self.cs + self.cs//2
            draw.ellipse([cx_s-dot, cy_s-dot, cx_s+dot, cy_s+dot], fill=COL_START)
        if self.maze.end:
            er, ec = self.maze.end
            cx_e = ec*self.cs + self.cs//2; cy_e = er*self.cs + self.cs//2
            draw.ellipse([cx_e-dot, cy_e-dot, cx_e+dot, cy_e+dot], fill=COL_END)

        return img


# ================================================================
# GIF BUILDER
# ================================================================

# Map event type → cell colour
_EVENT_COLOUR = {
    'explore':       COL_EXPLORED,
    'frontier':      COL_FRONTIER,
    'ahpp_node':     COL_AHPP_NODE,
    'ahpp_corridor': COL_AHPP_CORRIDOR,
}


def build_animation(maze: Maze, events_fn,
                    cell_size: int = 6,
                    fps: int = 20,
                    cells_per_frame: int = 5) -> list:
    """
    Drive an instrumented algorithm generator and capture PIL Image frames.

    Strategy
    --------
    Collect all events first, then replay them into frames so we can
    batch *cells_per_frame* events per frame for a reasonable GIF size.

    Parameters
    ----------
    maze            : Maze
    events_fn       : generator function  (one of ALGO_EVENTS.values())
    cell_size       : int
    fps             : int
    cells_per_frame : int   Higher → fewer frames → smaller file.

    Returns
    -------
    list[PIL.Image.Image]
    """
    renderer = MazeRenderer(maze, cell_size=cell_size)
    frames   = []
    path     = []
    non_path_events = []   # (event_type, cell)

    # Collect all events
    for event_type, cell in events_fn(maze):
        if event_type == 'path':
            path.append(cell)
        else:
            non_path_events.append((event_type, cell))

    # Frame 0: blank maze
    frames.append(renderer.render())

    # Build exploration frames
    batch = []
    for event_type, cell in non_path_events:
        col = _EVENT_COLOUR.get(event_type, COL_EXPLORED)
        renderer.colours[cell] = col
        batch.append(cell)
        if len(batch) >= cells_per_frame:
            frames.append(renderer.render()); batch = []
    if batch:
        frames.append(renderer.render())

    # Hold final frame (path in orange) for ~fps//4 extra frames
    hold = max(1, fps // 4)
    for _ in range(hold):
        frames.append(renderer.render(highlight_path=path))

    return frames


def save_gif(frames: list, out_path: Path, fps: int = 20):
    """Save a list of PIL Images as an animated GIF."""
    if not frames:
        print(f"  [WARN] No frames for {out_path.name} — skipping.")
        return
    duration_ms = max(20, 1000 // fps)
    palette_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=256)
                      for f in frames]
    palette_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    size_kb = out_path.stat().st_size // 1024
    print(f"  Saved {out_path.name}  ({len(frames)} frames, ~{size_kb} KB)")


# ================================================================
# MAIN
# ================================================================

def main():
    """Generate one animated GIF per (algorithm × shape) combination."""
    parser = argparse.ArgumentParser(
        description="Pathfinding animation generator")
    parser.add_argument("--cell", type=int, default=6,
                        help="Pixel size per maze cell (default 6)")
    parser.add_argument("--fps",  type=int, default=20,
                        help="Frames per second (default 20)")
    parser.add_argument("--skip", type=int, default=5,
                        help="Explored cells batched per frame (default 5)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" PATHFINDING VISUALISER")
    print("=" * 70)
    print(f"Settings : cell={args.cell}px  fps={args.fps}  "
          f"cells_per_frame={args.skip}")
    print(f"Output   : {ANIM_DIR}")
    print(f"Algorithms: {', '.join(ALGO_EVENTS)}")

    if not BEST_DIR.exists():
        print(f"\n[ERROR] best_mazes directory not found:\n  {BEST_DIR}")
        print("Run maze_benchmark.py first.")
        return

    maze_files = sorted(BEST_DIR.glob("BEST_*.maze"))
    if not maze_files:
        print(f"\n[ERROR] No BEST_*.maze files found in {BEST_DIR}")
        return

    print(f"\nFound {len(maze_files)} best maze(s).")
    total = len(maze_files) * len(ALGO_EVENTS)
    done  = 0

    for fp in maze_files:
        stem  = fp.stem; parts = stem.split("_")
        shape = parts[1] if len(parts) > 1 else "Unknown"
        print(f"\n── {fp.name}  (shape: {shape})")

        for alg_name, events_fn in ALGO_EVENTS.items():
            done += 1
            print(f"   [{done}/{total}] {alg_name}...", end=" ", flush=True)

            maze = copy.deepcopy(Maze.from_file(str(fp)))
            if maze.start is None or maze.end is None:
                print("SKIP (no start/end)"); continue

            frames = build_animation(
                maze, events_fn,
                cell_size=args.cell,
                fps=args.fps,
                cells_per_frame=args.skip,
            )
            out_path = ANIM_DIR / f"{alg_name}_{shape}.gif"
            save_gif(frames, out_path, fps=args.fps)

    print("\n" + "=" * 70)
    print(f"All {done} animations saved to: {ANIM_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()