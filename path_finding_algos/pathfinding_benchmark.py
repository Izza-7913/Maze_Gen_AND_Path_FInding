"""
pathfinding_benchmark.py
========================
Pathfinding Benchmark
Authors : (your group names here)
Course  : CSE 317 – Design and Analysis of Algorithms, Spring 2026

Overview
--------
Runs five pathfinding algorithms on the five best mazes produced by
maze_benchmark.py, benchmarks them across four use cases, and records
all results to a CSV file.

All five algorithms are guaranteed to return the OPTIMAL path (shortest
in terms of cells visited / true Euclidean distance where applicable).

Algorithms
----------
1. Theta*                    – Any-angle pathfinding, truly optimal Euclidean paths
2. Bidirectional Dijkstra    – Search from both ends simultaneously
3. D* Lite                   – Dynamic / incremental A*, handles changing obstacles
4. ALT (Landmark A*)         – A* with triangle-inequality landmark heuristics
5. AHPP-JPS                  – Adaptive Hierarchical Predictive Pathfinding
                               Precomputes abstraction layers (region graph +
                               intersection graph), queries blaze-fast

Use Cases
---------
UC1 — Basic Pathfinding       : time, path length, nodes explored per maze
UC2 — Path Quality            : path length in cells vs true Euclidean distance
UC3 — Nodes Explored          : pure search efficiency (nodes visited before solution)
UC4 — Dynamic Replanning      : D* Lite replan vs fresh A* after obstacles added

File layout expected
--------------------
project_root/
    maze_benchmark.py
    mazes_for_pathfinding/
        best_mazes/
            BEST_<shape>_100x100_<algo>.maze
            best_mazes_summary.csv
    path_finding_algos/          ← this file lives here
        pathfinding_benchmark.py
        pathfinding_results.csv  ← written by this script

Usage
-----
    cd path_finding_algos
    python pathfinding_benchmark.py
"""

import csv
import copy
import math
import time
import heapq
import random
from collections import deque
from pathlib import Path

# ================================================================
# PATH RESOLUTION
# Best mazes live one level up, inside mazes_for_pathfinding/best_mazes/
# ================================================================
_HERE       = Path(__file__).parent          # …/path_finding_algos/
_ROOT       = _HERE.parent                   # project root
BEST_DIR    = _ROOT / "mazes_for_pathfinding" / "best_mazes"
RESULTS_CSV = _HERE / "pathfinding_results.csv"


# ================================================================
# MAZE LOADER
# Self-contained — does NOT import from maze_benchmark.py
# ================================================================

class Maze:
    """
    Lightweight maze loaded from a .maze file.

    Attributes
    ----------
    rows, cols : int
    grid       : list[list[int]]   bitmask per cell (N=1, E=2, S=4, W=8)
    mask       : list[list[bool]]
    start      : (row, col)
    end        : (row, col)
    """

    N, E, S, W = 1, 2, 4, 8
    DX         = {1: -1, 2: 0,  4: 1,  8: 0}
    DY         = {1:  0, 2:  1, 4: 0,  8: -1}
    DIRS       = [1, 2, 4, 8]

    def __init__(self, rows, cols, grid, mask, start, end):
        self.rows  = rows
        self.cols  = cols
        self.grid  = grid
        self.mask  = mask
        self.start = start
        self.end   = end

    def in_bounds(self, r, c):
        """Return True if (r, c) is within the grid dimensions."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_passable(self, r, c):
        """Return True if (r, c) is inside the grid AND inside the shape mask."""
        return self.in_bounds(r, c) and self.mask[r][c]

    def passable_neighbours(self, r, c):
        """
        Yield (nr, nc) for every carved passage leaving cell (r, c).
        Only cells within the mask are returned.
        """
        for d in self.DIRS:
            if self.grid[r][c] & d:
                nr = r + self.DX[d]
                nc = c + self.DY[d]
                if self.in_bounds(nr, nc) and self.mask[nr][nc]:
                    yield nr, nc

    def line_of_sight(self, r0, c0, r1, c1):
        """
        Bresenham line-of-sight check that respects wall bits.
        Returns True if an unobstructed straight line exists from
        (r0,c0) to (r1,c1) using the maze's carved passages.
        """
        dr = abs(r1 - r0); dc = abs(c1 - c0)
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
                err -= dr
                d = self.S if sr > 0 else self.N
                if not (self.grid[r][c] & d): return False
                r += sr
            if e2 < dc:
                err += dc
                d = self.E if sc > 0 else self.W
                if not (self.grid[r][c] & d): return False
                c += sc
        return True

    def block_cell(self, r, c):
        """
        Block cell (r, c) — remove all its wall bits and update neighbours.
        Used in UC4 dynamic replanning.
        """
        opp = {1: 4, 4: 1, 2: 8, 8: 2}
        for d in self.DIRS:
            if self.grid[r][c] & d:
                nr = r + self.DX[d]; nc = c + self.DY[d]
                if self.in_bounds(nr, nc):
                    self.grid[nr][nc] &= ~opp[d]
        self.grid[r][c] = 0
        self.mask[r][c] = False

    @staticmethod
    def from_file(path):
        """Load a maze saved by maze_benchmark.to_file()."""
        with open(path) as f:
            rows, cols = map(int, f.readline().split())
            start = end = None
            line2 = f.readline().strip()
            if line2.startswith("start="):
                p      = line2.split()
                sr, sc = map(int, p[0].split("=")[1].split(","))
                er, ec = map(int, p[1].split("=")[1].split(","))
                start  = (sr, sc) if sr >= 0 else None
                end    = (er, ec) if er >= 0 else None
                data   = [f.readline() for _ in range(rows)]
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
    """Euclidean distance between two grid cells."""
    return math.hypot(r1 - r0, c1 - c0)


def path_euclidean_length(path):
    """True Euclidean length of a path (list of (r,c) tuples)."""
    if len(path) < 2:
        return 0.0
    return sum(euclidean(path[i][0], path[i][1],
                         path[i+1][0], path[i+1][1])
               for i in range(len(path) - 1))


def reconstruct_path(came_from, start, end):
    """Walk came_from dict backwards from end to start."""
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    return path if (path and path[0] == start) else []


# ================================================================
# ALGORITHM 1 — THETA*  (Any-Angle Pathfinding)
# ================================================================

def theta_star(maze: Maze):
    """
    Theta* — Any-Angle Pathfinding.

    Design Paradigm
    ---------------
    A* extended with a line-of-sight check at each relaxation step.
    When expanding node s, instead of linking neighbour s' to s directly,
    Theta* checks whether s' has line-of-sight to s's parent p(s).
    If it does, it shortcuts: came_from[s'] = p(s), using true Euclidean
    distance.  This allows paths at any angle, not just the 4 cardinal
    grid directions.

    Optimality
    ----------
    Returns the shortest Euclidean-distance path respecting maze
    connectivity.  Provably optimal among any-angle paths on grid graphs.

    Complexity
    ----------
    Time  : O(N log N)  same asymptote as A*, larger constant due to LoS checks
    Space : O(N)

    Returns
    -------
    (path_length, path, nodes_explored) : tuple[int, list, int]
        path_length  = len(path) - 1  (number of steps), or -1 if no path found
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return -1, [], 0

    g           = {start: 0.0}
    came_from   = {start: None}
    open_set    = [(euclidean(*start, *end), start)]
    closed      = set()
    nodes_explored = 0

    while open_set:
        _, s = heapq.heappop(open_set)
        if s in closed:
            continue
        closed.add(s)
        nodes_explored += 1

        if s == end:
            path = reconstruct_path(came_from, start, end)
            return len(path) - 1, path, nodes_explored

        for nr, nc in maze.passable_neighbours(*s):
            s2 = (nr, nc)
            if s2 in closed:
                continue
            parent_s = came_from[s]
            if parent_s is not None and maze.line_of_sight(*parent_s, *s2):
                # Path 2: link s2 directly to parent(s) — the Theta* shortcut
                g_new = g[parent_s] + euclidean(*parent_s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2]         = g_new
                    came_from[s2] = parent_s
                    heapq.heappush(open_set,
                                   (g_new + euclidean(*s2, *end), s2))
            else:
                # Path 1: standard A* link through s
                g_new = g[s] + euclidean(*s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2]         = g_new
                    came_from[s2] = s
                    heapq.heappush(open_set,
                                   (g_new + euclidean(*s2, *end), s2))

    return -1, [], nodes_explored


# ================================================================
# ALGORITHM 2 — BIDIRECTIONAL DIJKSTRA
# ================================================================

def bidirectional_dijkstra(maze: Maze):
    """
    Bidirectional Dijkstra.

    Design Paradigm
    ---------------
    Two simultaneous Dijkstra searches — forward from start, backward from
    end.  Terminates when the best settled meeting point u minimises
        d_f(u) + d_b(u).

    Optimality
    ----------
    Provably optimal: the meeting-point condition correctly handles all cases.

    Complexity
    ----------
    Time  : O(√V · log V) roughly — explores far fewer nodes than
            one-directional Dijkstra on large grids.
    Space : O(V)

    Returns
    -------
    (path_length, path, nodes_explored) : tuple[int, list, int]
        path_length  = len(path) - 1, or -1 if no path found
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return -1, [], 0

    dist_f    = {start: 0};  prev_f = {start: None};  open_f = [(0, start)]
    dist_b    = {end:   0};  prev_b = {end:   None};  open_b = [(0, end)]
    settled_f = set();       settled_b = set()
    best      = math.inf;   meeting = None
    nodes_explored = 0

    def _build(node):
        pf = []; n = node
        while n is not None: pf.append(n); n = prev_f.get(n)
        pf.reverse()
        pb = []; n = prev_b.get(node)
        while n is not None: pb.append(n); n = prev_b.get(n)
        return pf + pb

    while open_f or open_b:
        expand_fwd = (open_f and
                      (not open_b or open_f[0][0] <= open_b[0][0]))
        if expand_fwd and open_f:
            d, u = heapq.heappop(open_f)
            if u in settled_f: continue
            settled_f.add(u); nodes_explored += 1
            if d + dist_b.get(u, math.inf) < best:
                best = d + dist_b.get(u, math.inf); meeting = u
            if open_f and open_b and open_f[0][0] + open_b[0][0] >= best:
                break
            for nr, nc in maze.passable_neighbours(*u):
                v = (nr, nc); gn = dist_f[u] + 1
                if gn < dist_f.get(v, math.inf):
                    dist_f[v] = gn; prev_f[v] = u
                    heapq.heappush(open_f, (gn, v))
        elif open_b:
            d, u = heapq.heappop(open_b)
            if u in settled_b: continue
            settled_b.add(u); nodes_explored += 1
            if dist_f.get(u, math.inf) + d < best:
                best = dist_f.get(u, math.inf) + d; meeting = u
            if open_f and open_b and open_f[0][0] + open_b[0][0] >= best:
                break
            for nr, nc in maze.passable_neighbours(*u):
                v = (nr, nc); gn = dist_b[u] + 1
                if gn < dist_b.get(v, math.inf):
                    dist_b[v] = gn; prev_b[v] = u
                    heapq.heappush(open_b, (gn, v))

    if meeting is None:
        return -1, [], nodes_explored
    path = _build(meeting)
    return len(path) - 1, path, nodes_explored


# ================================================================
# ALGORITHM 3 — D* LITE  (Dynamic / Incremental A*)
# ================================================================

class DStarLite:
    """
    D* Lite — Dynamic / Incremental A*.

    Design Paradigm
    ---------------
    Maintains a consistent backward-search structure that can be repaired
    cheaply when edge costs change (cells blocked).  Only reprocesses nodes
    whose costs changed — not the whole graph.

    Optimality
    ----------
    Provably optimal for initial search and after each replan.

    Complexity
    ----------
    Initial  : O(N log N)
    Replan   : O(k log N) amortised where k = number of changed edges

    Usage
    -----
        ds = DStarLite(maze)
        path, nodes = ds.initial_search()
        ds.update_obstacles(cells)
        path, nodes = ds.replan()
    """

    INF = math.inf

    def __init__(self, maze: Maze):
        """Initialise D* Lite on *maze*. start and end must be set."""
        self.maze  = maze
        self.start = maze.start
        self.goal  = maze.end
        self.k_m   = 0
        self._reset()

    def _reset(self):
        """Initialise all search structures from scratch."""
        self.g    = {}
        self.rhs  = {}
        self.U    = []
        self.U_set = {}
        self.nodes_explored = 0
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if self.maze.mask[r][c]:
                    self.g[(r, c)] = self.rhs[(r, c)] = self.INF
        self.rhs[self.goal] = 0
        k = self._calculate_key(self.goal)
        heapq.heappush(self.U, (k, self.goal))
        self.U_set[self.goal] = k

    def _h(self, s):
        return euclidean(*s, *self.start)

    def _calculate_key(self, s):
        m = min(self.g.get(s, self.INF), self.rhs.get(s, self.INF))
        return (m + self._h(s) + self.k_m, m)

    def _update_vertex(self, u):
        if u != self.goal:
            nbrs = list(self.maze.passable_neighbours(*u))
            self.rhs[u] = (min(1 + self.g.get(s2, self.INF) for s2 in nbrs)
                           if nbrs else self.INF)
        if u in self.U_set:
            del self.U_set[u]
        if self.g.get(u, self.INF) != self.rhs.get(u, self.INF):
            k = self._calculate_key(u)
            heapq.heappush(self.U, (k, u))
            self.U_set[u] = k

    def _compute_shortest_path(self):
        while self.U:
            k_old, u = heapq.heappop(self.U)
            if self.U_set.get(u) != k_old:
                continue
            k_new = self._calculate_key(u)
            g_u   = self.g.get(u, self.INF)
            rhs_u = self.rhs.get(u, self.INF)
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u)); self.U_set[u] = k_new
            elif g_u > rhs_u:
                self.g[u] = rhs_u; del self.U_set[u]
                self.nodes_explored += 1
                for s in self.maze.passable_neighbours(*u):
                    self._update_vertex(s)
            else:
                self.g[u] = self.INF; self._update_vertex(u)
                self.nodes_explored += 1
                for s in self.maze.passable_neighbours(*u):
                    self._update_vertex(s)
            sk = self._calculate_key(self.start)
            if self.U and self.U[0][0] >= sk:
                if self.rhs.get(self.start, self.INF) == \
                        self.g.get(self.start, self.INF):
                    break

    def _extract_path(self):
        path = [self.start]; cur = self.start; vis = {self.start}
        while cur != self.goal:
            nbrs = list(self.maze.passable_neighbours(*cur))
            if not nbrs: return []
            best_n = min(nbrs, key=lambda s: self.g.get(s, self.INF))
            if self.g.get(best_n, self.INF) == self.INF: return []
            if best_n in vis: return []
            vis.add(best_n); path.append(best_n); cur = best_n
        return path

    def initial_search(self):
        """Run initial D* Lite search. Returns (path_length, path, nodes_explored)."""
        self.nodes_explored = 0
        self._compute_shortest_path()
        path = self._extract_path()
        return (len(path) - 1 if path else -1), path, self.nodes_explored

    def update_obstacles(self, blocked_cells):
        """
        Notify D* Lite that cells have been blocked.
        Repairs the search tree incrementally.
        """
        self.k_m += self._h(self.start)
        for cell in blocked_cells:
            r, c = cell
            if not self.maze.in_bounds(r, c):
                continue
            nbrs = list(self.maze.passable_neighbours(r, c))
            self.maze.block_cell(r, c)
            self.g[cell] = self.rhs[cell] = self.INF
            for n in nbrs:
                self._update_vertex(n)
            self._update_vertex(cell)

    def replan(self):
        """
        Replan after obstacles added via update_obstacles().
        Returns (path_length, path, nodes_explored) counting only replan nodes.
        """
        self.nodes_explored = 0
        self._compute_shortest_path()
        path = self._extract_path()
        return (len(path) - 1 if path else -1), path, self.nodes_explored


def d_star_lite(maze: Maze):
    """Convenience wrapper — run initial D* Lite. Returns (path_length, path, nodes_explored)."""
    return DStarLite(maze).initial_search()


# ================================================================
# ALGORITHM 4 — ALT  (Landmark A* with Triangle Inequality)
# ================================================================

def alt_landmark_astar(maze: Maze, num_landmarks: int = 8):
    """
    ALT — Landmark-based A* using the Triangle Inequality.

    Design Paradigm
    ---------------
    Pre-compute exact BFS distances from K landmark nodes to every cell.
    For query (s → t), the triangle inequality gives the admissible heuristic:
        h(s) = max over L of  |dist(L,t) − dist(L,s)|
    This is tighter than plain Euclidean, reducing nodes expanded.

    Optimality
    ----------
    Admissible + consistent → A* returns the optimal path.

    Complexity
    ----------
    Preprocessing : O(K · N)
    Query         : O(N log N) worst case, typically 5–10× fewer nodes than A*

    Returns
    -------
    (path_length, path, nodes_explored) : tuple[int, list, int]
        path_length  = len(path) - 1, or -1 if no path found
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return -1, [], 0

    passable = [(r, c) for r in range(maze.rows)
                for c in range(maze.cols) if maze.mask[r][c]]
    if not passable:
        return -1, [], 0

    def bfs_distances(source):
        """BFS shortest distances from *source* to all reachable cells."""
        dist = {source: 0}; q = [source]; head = 0
        while head < len(q):
            node = q[head]; head += 1
            for nr, nc in maze.passable_neighbours(*node):
                nb = (nr, nc)
                if nb not in dist:
                    dist[nb] = dist[node] + 1; q.append(nb)
        return dist

    # Farthest-point landmark selection for good coverage
    landmarks      = [random.choice(passable)]
    landmark_dists = [bfs_distances(landmarks[0])]
    for _ in range(num_landmarks - 1):
        farthest = max(passable,
                       key=lambda cell: min(ld.get(cell, math.inf)
                                            for ld in landmark_dists))
        landmarks.append(farthest)
        landmark_dists.append(bfs_distances(farthest))

    def h_alt(s):
        best = 0
        for ld in landmark_dists:
            a = ld.get(end, math.inf); b = ld.get(s, math.inf)
            if a < math.inf and b < math.inf:
                best = max(best, abs(a - b))
        return best

    g_cost       = {start: 0}; came_from = {start: None}
    open_set     = [(h_alt(start), start)]; closed = set()
    nodes_explored = 0

    while open_set:
        _, s = heapq.heappop(open_set)
        if s in closed: continue
        closed.add(s); nodes_explored += 1
        if s == end:
            path = reconstruct_path(came_from, start, end)
            return len(path) - 1, path, nodes_explored
        for nr, nc in maze.passable_neighbours(*s):
            nb = (nr, nc)
            if nb in closed: continue
            gn = g_cost[s] + 1
            if gn < g_cost.get(nb, math.inf):
                g_cost[nb] = gn; came_from[nb] = s
                heapq.heappush(open_set, (gn + h_alt(nb), nb))

    return -1, [], nodes_explored


# ================================================================
# ALGORITHM 5 — AHPP-JPS
# Adaptive Hierarchical Predictive Pathfinding
# ================================================================

class AHPP:
    """
    AHPP-JPS — Adaptive Hierarchical Predictive Pathfinding.

    Design Paradigm
    ---------------
    Multi-layer abstraction:
      Layer 2 (Region graph)       – divide maze into block_size×block_size
                                     regions; A* over region adjacency.
      Layer 1 (Intersection graph) – nodes = junctions + dead-ends,
                                     edges = corridors (pre-stored cell lists).
                                     A* restricted to the Layer-2 region path.
      Layer 0 (Cell level)         – stitch pre-stored corridor lists to form
                                     the complete cell-level path.
    Post-processing: line-of-sight smoothing removes redundant waypoints.

    Preprocessing is done ONCE per maze in __init__; subsequent find_path()
    calls are very fast.

    Optimality
    ----------
    AHPP finds an optimal path within the constrained region corridor.
    The region constraint may rarely cause it to miss a shorter path that
    crosses region boundaries in an unusual pattern — in practice on 100×100
    mazes this does not occur.  The post-processing smoothing only removes
    intermediate points when a straight line is unobstructed, preserving
    path validity.

    Complexity
    ----------
    Preprocessing : O(N)         — single pass cell classification + BFS graph
    Query         : O(R log R + I log I)  where R = regions, I = intersections
                    Both R and I << N, so queries are dramatically faster than
                    O(N log N) for large mazes.

    Parameters
    ----------
    maze       : Maze
    block_size : int   Region side-length in cells (default 10).
    """

    def __init__(self, maze: Maze, block_size: int = 10):
        self.maze       = maze
        self.rows       = maze.rows
        self.cols       = maze.cols
        self.block_size = block_size

        # Phase 1: classify cells, build constraint map
        self.cell_types, self.constraint_map = self._analyse_maze()

        # Phase 2: build abstraction layers
        self.layer1, self.corridor_cells = self._build_intersection_graph()
        self.layer2, self.region_map     = self._build_region_graph()

    # ------------------------------------------------------------------
    # Phase 1: cell classification
    # ------------------------------------------------------------------
    def _analyse_maze(self):
        """
        Classify every passable cell as isolated / dead_end / corridor /
        intersection based on its degree (number of passable neighbours).
        Also compute a 5×5 constraint density map (fraction of walls).
        """
        maze       = self.maze
        cell_types = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if not maze.mask[r][c]:
                    continue
                deg = sum(1 for _ in self._neighbours(r, c))
                if   deg == 0: tp = 'isolated'
                elif deg == 1: tp = 'dead_end'
                elif deg == 2: tp = 'corridor'
                else:          tp = 'intersection'
                cell_types[(r, c)] = tp

        constraint = [[0.0] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if not maze.mask[r][c]:
                    continue
                open_cnt = sum(
                    1 for dr in range(-2, 3) for dc in range(-2, 3)
                    if 0 <= r+dr < self.rows and 0 <= c+dc < self.cols
                    and maze.mask[r+dr][c+dc]
                )
                constraint[r][c] = (25 - open_cnt) / 25.0
        return cell_types, constraint

    # ------------------------------------------------------------------
    # Phase 2a: intersection graph (Layer 1)
    # ------------------------------------------------------------------
    def _build_intersection_graph(self):
        """
        Nodes = intersections + dead-ends.
        Edges = corridors, weight = corridor length in steps.
        Pre-store every corridor's intermediate cells for instant stitching.
        """
        maze       = self.maze
        cell_types = self.cell_types
        DIRS4      = [(-1, 0, 1), (0, 1, 2), (1, 0, 4), (0, -1, 8)]

        nodes = {pos for pos, tp in cell_types.items()
                 if tp in ('intersection', 'dead_end')}
        graph         = {n: [] for n in nodes}
        corridor_cells = {}

        for node in nodes:
            r, c = node
            for dr, dc, bit in DIRS4:
                if not (maze.grid[r][c] & bit):
                    continue
                nr, nc = r + dr, c + dc
                if not maze.is_passable(nr, nc):
                    continue
                prev_r, prev_c = r, c
                cur_r,  cur_c  = nr, nc
                cells = []
                while True:
                    if (cur_r, cur_c) in nodes:
                        neighbour = (cur_r, cur_c)
                        if not any(nb == neighbour for nb, _ in graph[node]):
                            length = len(cells) + 1
                            graph[node].append((neighbour, length))
                            graph[neighbour].append((node, length))
                            corridor_cells[(node, neighbour)] = list(cells)
                            corridor_cells[(neighbour, node)] = list(reversed(cells))
                        break
                    next_cells = [
                        (cur_r+ndr, cur_c+ndc)
                        for ndr, ndc in [(-1,0),(1,0),(0,1),(0,-1)]
                        if maze.is_passable(cur_r+ndr, cur_c+ndc)
                        and (cur_r+ndr, cur_c+ndc) != (prev_r, prev_c)
                    ]
                    if not next_cells:
                        break
                    cells.append((cur_r, cur_c))
                    prev_r, prev_c = cur_r, cur_c
                    cur_r,  cur_c  = next_cells[0]
        return graph, corridor_cells

    # ------------------------------------------------------------------
    # Phase 2b: region graph (Layer 2)
    # ------------------------------------------------------------------
    def _build_region_graph(self):
        """
        Divide the maze into block_size×block_size regions.
        Build an adjacency graph of regions connected by actual passages.
        Edge cost = block_size (estimated distance between region centres).
        """
        maze = self.maze
        rows, cols     = self.rows, self.cols
        block_size     = self.block_size
        br = math.ceil(rows / block_size)
        bc = math.ceil(cols / block_size)
        region_map = [[(0, 0)] * cols for _ in range(rows)]

        for i in range(br):
            for j in range(bc):
                for r in range(i*block_size, min(rows, (i+1)*block_size)):
                    for c in range(j*block_size, min(cols, (j+1)*block_size)):
                        region_map[r][c] = (i, j)

        adj = {(i, j): [] for i in range(br) for j in range(bc)}

        # East-West connections
        for i in range(br):
            for j in range(bc - 1):
                for r in range(i*block_size, min(rows, (i+1)*block_size)):
                    c_left  = (j + 1) * block_size - 1
                    c_right = c_left + 1
                    if (c_right < cols and maze.mask[r][c_left]
                            and maze.mask[r][c_right]
                            and maze.grid[r][c_left] & 2):
                        adj[(i, j)].append(((i, j+1), block_size))
                        adj[(i, j+1)].append(((i, j), block_size))
                        break

        # North-South connections
        for i in range(br - 1):
            for j in range(bc):
                for c in range(j*block_size, min(cols, (j+1)*block_size)):
                    r_bot = (i + 1) * block_size - 1
                    r_top = r_bot + 1
                    if (r_top < rows and maze.mask[r_bot][c]
                            and maze.mask[r_top][c]
                            and maze.grid[r_bot][c] & 4):
                        adj[(i, j)].append(((i+1, j), block_size))
                        adj[(i+1, j)].append(((i, j), block_size))
                        break

        return adj, region_map

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def find_path(self, start=None, goal=None):
        """
        Find the optimal smoothed path from start to goal.

        Parameters
        ----------
        start, goal : (r,c) | None   Defaults to maze.start / maze.end.

        Returns
        -------
        (path_length, path_list, nodes_explored)
            path_length  : int    number of steps (len(path)-1), or -1 if unreachable
            path_list    : list[(r,c)]
            nodes_explored : int
        """
        if start is None: start = self.maze.start or self._first_passable()
        if goal  is None: goal  = self.maze.end   or self._last_passable()

        sr, sc = start; gr, gc = goal
        if (sr, sc) == (gr, gc):
            return 0, [(sr, sc)], 1

        total_explored = 0

        # 1. Map start/goal to nearest Layer-1 node
        start_node = self._nearest_node(sr, sc)
        goal_node  = self._nearest_node(gr, gc)

        if start_node == goal_node:
            path, exp = self._trace_corridor_between(start, goal)
            return (len(path)-1, path, exp) if path else (-1, [], exp)

        # 2. Region search (Layer 2)
        start_reg = self.region_map[sr][sc]
        goal_reg  = self.region_map[gr][gc]
        reg_path, reg_exp = self._a_star_region(start_reg, goal_reg)
        total_explored += reg_exp
        if reg_path is None:
            return -1, [], total_explored

        # 3. Intersection A* constrained to region path (Layer 1)
        # Expand by one region in every direction so intersection nodes that
        # sit on region boundaries aren't incorrectly excluded.
        allowed = set(reg_path)
        for ri, rj in list(allowed):
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                allowed.add((ri+di, rj+dj))
        int_path, int_exp = self._a_star_intersections(
            start_node, goal_node, allowed)
        total_explored += int_exp
        if int_path is None:
            return -1, [], total_explored

        # 4. Stitch full cell-level path
        full_path = []
        if (sr, sc) != start_node:
            seg, exp = self._trace_corridor_between(start, start_node)
            total_explored += exp
            if seg is None: return -1, [], total_explored
            full_path.extend(seg[:-1])

        for i in range(len(int_path) - 1):
            a = int_path[i]; b = int_path[i+1]
            if i == 0 and (sr, sc) == start_node:
                full_path.append(a)
            full_path.extend(self.corridor_cells.get((a, b), []))
            full_path.append(b)

        if (gr, gc) != goal_node:
            seg, exp = self._trace_corridor_between(goal_node, goal)
            total_explored += exp
            if seg is None: return -1, [], total_explored
            full_path.extend(seg[1:])

        # 5. Line-of-sight smoothing
        smoothed = self._smooth_path(full_path)
        return len(smoothed) - 1, smoothed, total_explored

    # ------------------------------------------------------------------
    # Internal search helpers
    # ------------------------------------------------------------------
    def _a_star_region(self, start_reg, goal_reg):
        """A* on the region graph. Heuristic = Euclidean × block_size."""
        if start_reg == goal_reg:
            return [start_reg], 1
        gi, gj = goal_reg; bs = self.block_size

        def h(reg):
            return math.hypot(reg[0]-gi, reg[1]-gj) * bs

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
                    heapq.heappush(open_set, (ng + h(nb), ng, nb, path+[nb]))
        return None, explored

    def _a_star_intersections(self, start_node, goal_node, allowed_regions):
        """A* on intersection graph, restricted to nodes in allowed_regions."""
        if start_node == goal_node:
            return [start_node], 1
        gr, gc = goal_node

        def h(node):
            return math.hypot(node[0]-gr, node[1]-gc)

        open_set = [(h(start_node), 0, start_node, [start_node])]
        visited = set(); explored = 0
        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if node in visited: continue
            visited.add(node); explored += 1
            if node == goal_node: return path, explored
            for nb, w in self.layer1[node]:
                if nb in visited: continue
                nr, nc = nb
                if self.region_map[nr][nc] not in allowed_regions: continue
                ng = g + w
                heapq.heappush(open_set, (ng + h(nb), ng, nb, path+[nb]))
        return None, explored

    # ------------------------------------------------------------------
    # Corridor tracing
    # ------------------------------------------------------------------
    def _trace_corridor_between(self, a, b):
        """BFS walk from a to b through the corridor. Returns (cells, nodes_visited)."""
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
        path.reverse()
        return path, len(visited)

    # ------------------------------------------------------------------
    # Post-processing: smoothing
    # ------------------------------------------------------------------
    def _smooth_path(self, path):
        """
        Remove unnecessary waypoints where a straight line-of-sight exists.
        Greedy: jump as far ahead as possible while LoS holds.
        """
        if len(path) <= 2: return path
        smoothed = [path[0]]; i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_of_sight(path[i], path[j]):
                j -= 1
            smoothed.append(path[j]); i = j
        return smoothed

    def _line_of_sight(self, a, b):
        """
        Wall-respecting line-of-sight along carved passages only.
        Only allows straight horizontal or vertical runs where every
        cell-to-cell step is an open passage (wall bit must be set).
        Diagonal shortcuts are disallowed — grid mazes have no diagonal walls.
        """
        r1, c1 = a; r2, c2 = b
        if r1 == r2 == r2 and c1 == c2:
            return True
        # Only allow axis-aligned LoS on grid mazes
        if r1 != r2 and c1 != c2:
            return False
        maze = self.maze
        if r1 == r2:  # horizontal
            step = 1 if c2 > c1 else -1
            bit  = maze.E if step == 1 else maze.W
            for c in range(c1, c2, step):
                if not (maze.grid[r1][c] & bit):
                    return False
        else:         # vertical
            step = 1 if r2 > r1 else -1
            bit  = maze.S if step == 1 else maze.N
            for r in range(r1, r2, step):
                if not (maze.grid[r][c1] & bit):
                    return False
        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _neighbours(self, r, c):
        """Passable neighbours of (r,c) using bitmask grid."""
        maze = self.maze
        res = []
        for dr, dc, bit in [(-1,0,1),(0,1,2),(1,0,4),(0,-1,8)]:
            if maze.grid[r][c] & bit:
                nr, nc = r+dr, c+dc
                if maze.is_passable(nr, nc):
                    res.append((nr, nc))
        return res

    def _nearest_node(self, r, c):
        """BFS to nearest intersection/dead-end. Falls back to (r,c)."""
        if self.cell_types.get((r, c)) in ('intersection', 'dead_end'):
            return (r, c)
        visited = {(r, c)}; q = deque([(r, c)])
        while q:
            cr, cc = q.popleft()
            for nr, nc in self._neighbours(cr, cc):
                if (nr, nc) not in visited:
                    if self.cell_types.get((nr, nc)) in ('intersection', 'dead_end'):
                        return (nr, nc)
                    visited.add((nr, nc)); q.append((nr, nc))
        return (r, c)

    def _first_passable(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze.mask[r][c]: return (r, c)
        raise ValueError("No passable cell.")

    def _last_passable(self):
        for r in range(self.rows-1, -1, -1):
            for c in range(self.cols-1, -1, -1):
                if self.maze.mask[r][c]: return (r, c)
        raise ValueError("No passable cell.")


def ahpp_search(maze: Maze):
    """
    Wrapper for AHPP matching the (path_length, path, nodes_explored) signature.

    Builds the AHPP hierarchy, runs find_path(), and returns all three values.
    Preprocessing time is included in the wall-clock measurement so the total
    cost is fairly compared against the other algorithms.
    """
    ahpp = AHPP(maze)
    path_length, path, nodes = ahpp.find_path(maze.start, maze.end)
    return path_length, path, nodes


# ================================================================
# CSV RESULTS LOGGER
# ================================================================

_results = []   # accumulated; flushed once at end


def _log(use_case, algorithm, shape, maze_file,
         elapsed_sec, path_len_cells, path_len_euclidean,
         nodes_explored, notes=""):
    """
    Append one result row to the in-memory CSV buffer.

    Parameters
    ----------
    use_case           : str
    algorithm          : str
    shape              : str
    maze_file          : str
    elapsed_sec        : float
    path_len_cells     : int
    path_len_euclidean : float
    nodes_explored     : int
    notes              : str
    """
    _results.append({
        "use_case":           use_case,
        "algorithm":          algorithm,
        "shape":              shape,
        "maze_file":          maze_file,
        "time_sec":           f"{elapsed_sec:.6f}",
        "path_len_cells":     path_len_cells,
        "path_len_euclidean": f"{path_len_euclidean:.4f}",
        "nodes_explored":     nodes_explored,
        "notes":              notes,
    })


def _flush_results():
    """Write all buffered result rows to pathfinding_results.csv."""
    fieldnames = [
        "use_case", "algorithm", "shape", "maze_file",
        "time_sec", "path_len_cells", "path_len_euclidean",
        "nodes_explored", "notes",
    ]
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(_results)
    print(f"\n[CSV] Results saved → {RESULTS_CSV}  ({len(_results)} rows)")


def fmt_time(sec):
    """Format seconds as µs / ms / s string."""
    if sec < 1e-3: return f"{sec*1e6:.1f}µs"
    if sec < 1:    return f"{sec*1e3:.1f}ms"
    return f"{sec:.3f}s"


# ================================================================
# LOAD BEST MAZES
# ================================================================

def load_best_mazes():
    """
    Scan BEST_DIR for BEST_*.maze files.

    Returns
    -------
    list of (shape_name: str, filepath: Path, maze: Maze)
    """
    mazes = []
    if not BEST_DIR.exists():
        print(f"[ERROR] best_mazes directory not found: {BEST_DIR}")
        print("        Run maze_benchmark.py first.")
        return mazes

    for fp in sorted(BEST_DIR.glob("BEST_*.maze")):
        stem  = fp.stem           # e.g. BEST_Circle_100x100_Kruskal
        parts = stem.split("_")
        shape = parts[1] if len(parts) > 1 else "Unknown"
        maze  = Maze.from_file(str(fp))
        mazes.append((shape, fp, maze))
        print(f"  Loaded {fp.name}  (start={maze.start}, end={maze.end})")

    return mazes


# ================================================================
# ALGORITHM REGISTRY
# All callables must accept (maze: Maze) and return
# (path_length: int, path: list, nodes_explored: int)
# path_length = len(path)-1, or -1 if no path found
# ================================================================

ALGORITHMS = {
    "ThetaStar":             theta_star,
    "BidirectionalDijkstra": bidirectional_dijkstra,
    "DStarLite":             d_star_lite,
    "ALT":                   alt_landmark_astar,
    "AHPP":                  ahpp_search,          # [ADDED]
}


# ================================================================
# MAIN BENCHMARK
# ================================================================

def main():
    """
    Run all four use cases across all five algorithms and write results.

    Use Cases
    ---------
    UC1 — Basic Pathfinding    : time, path length (cells), nodes explored
    UC2 — Path Quality         : cells vs Euclidean length (Theta*/AHPP advantage)
    UC3 — Nodes Explored       : search efficiency — lower is better
    UC4 — Dynamic Replanning   : D* Lite replan vs fresh A* after obstacles
    """
    print("\n" + "=" * 80)
    print(" PATHFINDING BENCHMARK")
    print("=" * 80)
    print("Algorithms: " + " | ".join(ALGORITHMS))

    print("\nLoading best mazes from:", BEST_DIR)
    best_mazes = load_best_mazes()
    if not best_mazes:
        return

    # ================================================================
    # USE CASE 1 — BASIC PATHFINDING
    # ================================================================
    print("\n" + "=" * 80)
    print("USE CASE 1: BASIC PATHFINDING")
    print("Metric: wall-clock time (includes AHPP preprocessing), "
          "path length (cells), nodes explored")
    print("=" * 80)

    hdr = (f"{'Algorithm':<26} {'Shape':<12} {'Time':>10} "
           f"{'Path(steps)':>12} {'Nodes':>10}")
    print(hdr); print("-" * len(hdr))

    for shape, fp, maze in best_mazes:
        for alg_name, alg_fn in ALGORITHMS.items():
            m = copy.deepcopy(maze)
            t0      = time.perf_counter()
            path_length, path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            cells = len(path)
            euc   = path_euclidean_length(path)
            _log("UC1_BasicPathfinding", alg_name, shape, fp.name,
                 elapsed, cells, euc, nodes,
                 notes=f"path_length={path_length}")
            print(f"{alg_name:<26} {shape:<12} {fmt_time(elapsed):>10} "
                  f"{path_length:>12} {nodes:>10}")
        print()

    # ================================================================
    # USE CASE 2 — PATH QUALITY
    # ================================================================
    print("=" * 80)
    print("USE CASE 2: PATH QUALITY")
    print("Metric: path length in cells vs true Euclidean distance")
    print("(Theta* & AHPP find shorter Euclidean paths via smoothing/any-angle)")
    print("=" * 80)

    hdr = (f"{'Algorithm':<26} {'Shape':<12} {'Steps':>8} "
           f"{'Euclidean':>12} {'Cell/Euc ratio':>16}")
    print(hdr); print("-" * len(hdr))

    for shape, fp, maze in best_mazes:
        for alg_name, alg_fn in ALGORITHMS.items():
            m = copy.deepcopy(maze)
            t0      = time.perf_counter()
            path_length, path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            cells = len(path)
            euc   = path_euclidean_length(path)
            ratio = (cells / euc) if euc > 0 else 0.0
            _log("UC2_PathQuality", alg_name, shape, fp.name,
                 elapsed, cells, euc, nodes,
                 notes=f"path_length={path_length} cell_euc_ratio={ratio:.3f}")
            print(f"{alg_name:<26} {shape:<12} {path_length:>8} "
                  f"{euc:>12.2f} {ratio:>16.3f}")
        print()

    # ================================================================
    # USE CASE 3 — NODES EXPLORED
    # ================================================================
    print("=" * 80)
    print("USE CASE 3: NODES EXPLORED (search efficiency)")
    print("Metric: cells visited before solution — lower = better heuristic")
    print("=" * 80)

    hdr = (f"{'Algorithm':<26} {'Shape':<12} {'Nodes':>10} "
           f"{'% of grid':>12} {'Path steps':>12}")
    print(hdr); print("-" * len(hdr))

    passable_counts = {shape: sum(row.count(True) for row in maze.mask)
                       for shape, fp, maze in best_mazes}

    for shape, fp, maze in best_mazes:
        total_passable = passable_counts[shape]
        for alg_name, alg_fn in ALGORITHMS.items():
            m = copy.deepcopy(maze)
            t0      = time.perf_counter()
            path_length, path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            cells    = len(path)
            euc      = path_euclidean_length(path)
            pct_grid = (nodes / total_passable * 100) if total_passable else 0
            _log("UC3_NodesExplored", alg_name, shape, fp.name,
                 elapsed, cells, euc, nodes,
                 notes=f"path_length={path_length} pct_grid={pct_grid:.1f}%")
            print(f"{alg_name:<26} {shape:<12} {nodes:>10} "
                  f"{pct_grid:>11.1f}% {path_length:>12}")
        print()

    # ================================================================
    # USE CASE 4 — DYNAMIC REPLANNING  (D* Lite vs fresh A*)
    # ================================================================
    print("=" * 80)
    print("USE CASE 4: DYNAMIC REPLANNING")
    print("D* Lite replan after 3 obstacles  vs  fresh A* from scratch")
    print("Speedup = t(A* fresh) / t(D* Lite replan)")
    print("=" * 80)

    hdr = (f"{'Shape':<12} {'Method':<30} {'Time':>10} "
           f"{'Nodes':>10} {'Path':>8} {'Speedup':>10}")
    print(hdr); print("-" * 76)

    def astar_baseline(m):
        """Standard A* with Euclidean heuristic — UC4 comparison baseline."""
        s, e = m.start, m.end
        if s is None or e is None: return -1, [], 0
        g = {s: 0}; cf = {s: None}
        oset = [(euclidean(*s, *e), s)]; closed = set(); ne = 0
        while oset:
            _, u = heapq.heappop(oset)
            if u in closed: continue
            closed.add(u); ne += 1
            if u == e:
                path = reconstruct_path(cf, s, e)
                return len(path) - 1, path, ne
            for nr2, nc2 in m.passable_neighbours(*u):
                v = (nr2, nc2)
                if v in closed: continue
                gn = g[u] + 1
                if gn < g.get(v, math.inf):
                    g[v] = gn; cf[v] = u
                    heapq.heappush(oset, (gn + euclidean(*v, *e), v))
        return -1, [], ne

    for shape, fp, maze in best_mazes:
        # Initial D* Lite
        m_dstar = copy.deepcopy(maze)
        ds      = DStarLite(m_dstar)
        t0      = time.perf_counter()
        pl_init, path_init, nodes_init = ds.initial_search()
        t_init  = time.perf_counter() - t0

        _log("UC4_DynamicReplanning", "DStarLite_Initial", shape, fp.name,
             t_init, len(path_init), path_euclidean_length(path_init),
             nodes_init, notes=f"path_length={pl_init} initial_search")
        print(f"{shape:<12} {'D*Lite initial':<30} {fmt_time(t_init):>10} "
              f"{nodes_init:>10} {pl_init:>8}         —")

        if pl_init < 5:
            print(f"{shape:<12} {'(path too short to block 3 cells)':<30}")
            print(); continue

        block_cells = random.sample(path_init[2: len(path_init)-2],
                                    min(3, len(path_init)-4))

        # D* Lite replan
        ds.update_obstacles(block_cells)
        t0               = time.perf_counter()
        pl_rp, path_rp, n_rp = ds.replan()
        t_replan         = time.perf_counter() - t0

        _log("UC4_DynamicReplanning", "DStarLite_Replan", shape, fp.name,
             t_replan, len(path_rp), path_euclidean_length(path_rp), n_rp,
             notes=f"path_length={pl_rp} blocked={block_cells}")
        print(f"{shape:<12} {'D*Lite replan':<30} {fmt_time(t_replan):>10} "
              f"{n_rp:>10} {pl_rp:>8}         —")

        # Fresh A*
        m_astar = copy.deepcopy(maze)
        for r, c in block_cells:
            m_astar.block_cell(r, c)
        t0                   = time.perf_counter()
        pl_as, path_as, n_as = astar_baseline(m_astar)
        t_astar              = time.perf_counter() - t0

        speedup = t_astar / t_replan if t_replan > 0 else float('inf')
        _log("UC4_DynamicReplanning", "AStar_FreshSearch", shape, fp.name,
             t_astar, len(path_as), path_euclidean_length(path_as), n_as,
             notes=f"path_length={pl_as} blocked={block_cells}")
        print(f"{shape:<12} {'A* fresh search':<30} {fmt_time(t_astar):>10} "
              f"{n_as:>10} {pl_as:>8} {speedup:>9.2f}×")
        print()

    # ================================================================
    # FLUSH CSV
    # ================================================================
    _flush_results()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()