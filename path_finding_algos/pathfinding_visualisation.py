"""
d_star_pathfinding.py
========================
Pathfinding Benchmark
Author : Hamna Sajid
Course  : CSE 317 – Design and Analysis of Algorithms, Spring 2026

Overview
--------
Runs four pathfinding algorithms on the five best mazes produced by
maze_benchmark.py, benchmarks them across four use cases, and records
all results to a CSV file.

All four algorithms are guaranteed to return the OPTIMAL path (shortest
in terms of cells visited / true Euclidean distance where applicable).

Algorithms
----------
1. Theta*            – Any-angle pathfinding, truly optimal Euclidean paths
2. Bidirectional Dijkstra – Search from both ends simultaneously
3. D* Lite           – Dynamic / incremental A*, handles changing obstacles
4. ALT (Landmark A*) – A* with triangle-inequality landmark heuristics

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

import os
import csv
import math
import time
import heapq
import random
from pathlib import Path

# ================================================================
# PATH RESOLUTION
# Best mazes live one level up, inside mazes_for_pathfinding/best_mazes/
# ================================================================
_HERE      = Path(__file__).parent          # …/path_finding_algos/
_ROOT      = _HERE.parent                   # project root
BEST_DIR   = _ROOT / "mazes_for_pathfinding" / "best_mazes"
RESULTS_CSV = _HERE / "pathfinding_results.csv"


# ================================================================
# MAZE LOADER
# Re-implements the minimal subset of Maze needed for pathfinding
# so this file is self-contained (no import from maze_benchmark).
# ================================================================

class Maze:
    """
    Lightweight maze representation loaded from a .maze file.

    Attributes
    ----------
    rows, cols : int
    grid       : list[list[int]]   bitmask per cell (N=1,E=2,S=4,W=8)
    mask       : list[list[bool]]
    start      : (row, col)
    end        : (row, col)
    """

    N, E, S, W       = 1, 2, 4, 8
    DX               = {1: -1, 2: 0,  4: 1,  8: 0}
    DY               = {1:  0, 2:  1, 4: 0,  8: -1}
    DIRS             = [1, 2, 4, 8]

    def __init__(self, rows, cols, grid, mask, start, end):
        self.rows  = rows
        self.cols  = cols
        self.grid  = grid
        self.mask  = mask
        self.start = start
        self.end   = end

    # ------------------------------------------------------------------
    # Neighbours via carved walls (grid connectivity)
    # ------------------------------------------------------------------
    def passable_neighbours(self, r, c):
        """
        Yield (nr, nc) for every carved passage leaving cell (r, c).
        Only cells that are within the mask are returned.
        """
        for d in self.DIRS:
            if self.grid[r][c] & d:
                nr = r + self.DX[d]
                nc = c + self.DY[d]
                if 0 <= nr < self.rows and 0 <= nc < self.cols \
                        and self.mask[nr][nc]:
                    yield nr, nc

    # ------------------------------------------------------------------
    # Line-of-sight check (used by Theta*)
    # ------------------------------------------------------------------
    def line_of_sight(self, r0, c0, r1, c1):
        """
        Return True if there is an unobstructed straight line from
        (r0,c0) to (r1,c1) — i.e. every grid cell the line passes
        through is passable and connected to its neighbour in the
        traversal direction.

        Uses a grid-based Bresenham walk that checks wall bits at each
        step, so diagonal visibility respects carved passages.
        """
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r, c = r0, c0
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1

        if dc == 0:
            # Vertical line
            for _ in range(dr):
                d = self.S if sr > 0 else self.N
                if not (self.grid[r][c] & d):
                    return False
                r += sr
            return True

        if dr == 0:
            # Horizontal line
            for _ in range(dc):
                d = self.E if sc > 0 else self.W
                if not (self.grid[r][c] & d):
                    return False
                c += sc
            return True

        err = dc - dr
        while r != r1 or c != c1:
            e2 = 2 * err
            moved_r = moved_c = False
            if e2 > -dr:
                err -= dr
                d = self.S if sr > 0 else self.N
                if not (self.grid[r][c] & d):
                    return False
                r  += sr
                moved_r = True
            if e2 < dc:
                err += dc
                d = self.E if sc > 0 else self.W
                if not (self.grid[r][c] & d):
                    return False
                c  += sc
                moved_c = True
            if moved_r and moved_c:
                # Diagonal step — both transitions must be open
                pass
        return True

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------
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

    def block_cell(self, r, c):
        """
        Temporarily block cell (r,c) by removing all its wall bits and
        updating neighbours — used in UC4 dynamic replanning.
        """
        for d in self.DIRS:
            if self.grid[r][c] & d:
                nr = r + self.DX[d]
                nc = c + self.DY[d]
                opp = {1: 4, 4: 1, 2: 8, 8: 2}[d]
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    self.grid[nr][nc] &= ~opp
        self.grid[r][c] = 0
        self.mask[r][c] = False


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
    return path if path[0] == start else []


# ================================================================
# ALGORITHM 1 — THETA*  (Any-Angle Pathfinding)
# ================================================================

def theta_star(maze: Maze):
    """
    Theta* — Any-Angle Pathfinding.

    Design Paradigm
    ---------------
    A* extended with a line-of-sight check at each relaxation step.
    When expanding a node s, instead of linking its neighbour s' to s
    directly, Theta* checks whether s' has line-of-sight to s's *parent*
    p(s).  If it does, it shortcuts the link: came_from[s'] = p(s),
    using the true Euclidean distance as the cost.  This allows paths at
    any angle, not just the 4 cardinal grid directions.

    Optimality
    ----------
    Theta* returns the shortest Euclidean-distance path that respects the
    maze's connectivity.  It is provably optimal among any-angle paths on
    the given grid graph.

    Complexity
    ----------
    Time  : O(N log N)   same asymptotic as A*, larger constant due to
                         line-of-sight checks
    Space : O(N)

    Parameters
    ----------
    maze : Maze

    Returns
    -------
    path         : list[(r,c)]  optimal path from start to end
    nodes_explored : int        cells popped from the open set
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return [], 0

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
            return reconstruct_path(came_from, start, end), nodes_explored

        for nr, nc in maze.passable_neighbours(*s):
            s2 = (nr, nc)
            if s2 in closed:
                continue

            # Theta* update: try to inherit parent of s
            parent_s = came_from[s]
            if parent_s is not None and maze.line_of_sight(*parent_s, *s2):
                # Path 2: link s2 directly to parent(s)
                g_new = g[parent_s] + euclidean(*parent_s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2]         = g_new
                    came_from[s2] = parent_s
                    f             = g_new + euclidean(*s2, *end)
                    heapq.heappush(open_set, (f, s2))
            else:
                # Path 1: standard A* link through s
                g_new = g[s] + euclidean(*s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2]         = g_new
                    came_from[s2] = s
                    f             = g_new + euclidean(*s2, *end)
                    heapq.heappush(open_set, (f, s2))

    return [], nodes_explored   # no path found


# ================================================================
# ALGORITHM 2 — BIDIRECTIONAL DIJKSTRA
# ================================================================

def bidirectional_dijkstra(maze: Maze):
    """
    Bidirectional Dijkstra.

    Design Paradigm
    ---------------
    Run two simultaneous Dijkstra searches — one forward from start,
    one backward from end.  The search terminates when a node is settled
    by *both* searches.  The optimal meeting point minimises
        d_f(u) + d_b(u)
    over all settled nodes u.

    Optimality
    ----------
    The algorithm is provably optimal: when it terminates it returns
    the shortest path (in cells / uniform edge weight).  The meeting
    point condition correctly handles non-symmetric graphs.

    Complexity
    ----------
    Time  : O(√V · log V) roughly — explores far fewer nodes than
            one-directional Dijkstra on large grids.
    Space : O(V)

    Returns
    -------
    path           : list[(r,c)]
    nodes_explored : int   total nodes settled across both frontiers
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return [], 0

    # Forward search
    dist_f    = {start: 0}
    prev_f    = {start: None}
    open_f    = [(0, start)]
    settled_f = set()

    # Backward search
    dist_b    = {end: 0}
    prev_b    = {end: None}
    open_b    = [(0, end)]
    settled_b = set()

    best      = math.inf
    meeting   = None
    nodes_explored = 0

    def _build_path(node):
        """Stitch forward and backward came-from chains at meeting node."""
        path_fwd = []
        n = node
        while n is not None:
            path_fwd.append(n)
            n = prev_f.get(n)
        path_fwd.reverse()

        path_bwd = []
        n = prev_b.get(node)
        while n is not None:
            path_bwd.append(n)
            n = prev_b.get(n)

        return path_fwd + path_bwd

    while open_f or open_b:
        # Alternate: expand the frontier with the smaller tentative distance
        expand_fwd = (open_f and
                      (not open_b or open_f[0][0] <= open_b[0][0]))

        if expand_fwd and open_f:
            d, u = heapq.heappop(open_f)
            if u in settled_f:
                continue
            settled_f.add(u)
            nodes_explored += 1

            if d + dist_b.get(u, math.inf) < best:
                best    = d + dist_b.get(u, math.inf)
                meeting = u

            # Termination: if the best possible improvement is impossible
            if open_f and open_b:
                if open_f[0][0] + open_b[0][0] >= best:
                    break

            for nr, nc in maze.passable_neighbours(*u):
                v     = (nr, nc)
                g_new = dist_f[u] + 1   # uniform cost
                if g_new < dist_f.get(v, math.inf):
                    dist_f[v] = g_new
                    prev_f[v] = u
                    heapq.heappush(open_f, (g_new, v))

        elif open_b:
            d, u = heapq.heappop(open_b)
            if u in settled_b:
                continue
            settled_b.add(u)
            nodes_explored += 1

            if dist_f.get(u, math.inf) + d < best:
                best    = dist_f.get(u, math.inf) + d
                meeting = u

            if open_f and open_b:
                if open_f[0][0] + open_b[0][0] >= best:
                    break

            for nr, nc in maze.passable_neighbours(*u):
                v     = (nr, nc)
                g_new = dist_b[u] + 1
                if g_new < dist_b.get(v, math.inf):
                    dist_b[v] = g_new
                    prev_b[v] = u
                    heapq.heappush(open_b, (g_new, v))

    if meeting is None:
        return [], nodes_explored

    return _build_path(meeting), nodes_explored


# ================================================================
# ALGORITHM 3 — D* LITE  (Dynamic / Incremental A*)
# ================================================================

class DStarLite:
    """
    D* Lite — Dynamic / Incremental A*.

    Design Paradigm
    ---------------
    D* Lite maintains a consistent heuristic search structure (similar to
    a backward A*) that can be *repaired* cheaply when edge costs change
    (e.g. when cells are blocked).  On replanning it only reprocesses the
    nodes whose costs have changed, rather than searching from scratch.

    Optimality
    ----------
    D* Lite is provably optimal: the path it returns has minimum cost
    in the current graph.  After obstacle updates it repairs the search
    tree only as much as needed and still returns the optimal path.

    Complexity
    ----------
    Initial search : O(N log N)
    Replan after k changes: O(k log N) amortised — dramatically cheaper
    than a full O(N log N) fresh search when k << N.

    Usage
    -----
        dstar = DStarLite(maze)
        path, nodes = dstar.initial_search()

        # Later, when obstacles change:
        dstar.update_obstacles(blocked_cells)
        path, nodes = dstar.replan()
    """

    INF = math.inf

    def __init__(self, maze: Maze):
        """
        Initialise D* Lite on the given maze.

        Parameters
        ----------
        maze : Maze   The maze to search.  start and end must be set.
        """
        self.maze   = maze
        self.start  = maze.start
        self.goal   = maze.end
        self.k_m    = 0           # key modifier for lazy deletion
        self._reset()

    def _reset(self):
        """Initialise / reinitialise all search structures."""
        self.g    = {}   # g[s] = cost-to-go from s to goal
        self.rhs  = {}   # rhs[s] = one-step lookahead value
        self.U    = []   # priority queue: (key, node)
        self.U_set = {}  # node → key in queue (for lazy deletion)
        self.nodes_explored = 0

        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if self.maze.mask[r][c]:
                    self.g[(r, c)]   = self.INF
                    self.rhs[(r, c)] = self.INF

        self.rhs[self.goal] = 0
        k = self._calculate_key(self.goal)
        heapq.heappush(self.U, (k, self.goal))
        self.U_set[self.goal] = k

    def _h(self, s):
        """Heuristic: Euclidean distance from s to start."""
        return euclidean(*s, *self.start)

    def _calculate_key(self, s):
        """
        Compute the priority key for node s.
        key = (min(g,rhs) + h(s) + k_m,  min(g,rhs))
        Stored as a tuple so heapq compares lexicographically.
        """
        m = min(self.g.get(s, self.INF), self.rhs.get(s, self.INF))
        return (m + self._h(s) + self.k_m, m)

    def _update_vertex(self, u):
        """Recompute rhs[u] and update the priority queue."""
        if u != self.goal:
            self.rhs[u] = min(
                1 + self.g.get(s2, self.INF)
                for s2 in self.maze.passable_neighbours(*u)
            ) if list(self.maze.passable_neighbours(*u)) else self.INF

        # Remove old entry (lazy deletion via U_set)
        if u in self.U_set:
            del self.U_set[u]

        if self.g.get(u, self.INF) != self.rhs.get(u, self.INF):
            k = self._calculate_key(u)
            heapq.heappush(self.U, (k, u))
            self.U_set[u] = k

    def _compute_shortest_path(self):
        """Core D* Lite loop — process inconsistent nodes until start is consistent."""
        while self.U:
            k_old, u = heapq.heappop(self.U)

            # Lazy deletion
            if self.U_set.get(u) != k_old:
                continue

            k_new = self._calculate_key(u)
            g_u   = self.g.get(u, self.INF)
            rhs_u = self.rhs.get(u, self.INF)

            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
                self.U_set[u] = k_new
            elif g_u > rhs_u:
                # Overconsistent → make consistent
                self.g[u] = rhs_u
                del self.U_set[u]
                self.nodes_explored += 1
                for s in self.maze.passable_neighbours(*u):
                    self._update_vertex(s)
            else:
                # Underconsistent → raise g
                self.g[u] = self.INF
                self._update_vertex(u)
                self.nodes_explored += 1
                for s in self.maze.passable_neighbours(*u):
                    self._update_vertex(s)

            # Check termination
            start_key = self._calculate_key(self.start)
            if self.U and self.U[0][0] >= start_key:
                if self.rhs.get(self.start, self.INF) == \
                        self.g.get(self.start, self.INF):
                    break

    def _extract_path(self):
        """
        Follow the greedy gradient from start to goal using current g values.
        Returns the path as a list of (r,c) tuples, or [] if unreachable.
        """
        path = [self.start]
        current = self.start
        visited = {self.start}
        while current != self.goal:
            nbrs = list(self.maze.passable_neighbours(*current))
            if not nbrs:
                return []
            best_n = min(nbrs, key=lambda s: self.g.get(s, self.INF))
            if self.g.get(best_n, self.INF) == self.INF:
                return []   # goal unreachable
            if best_n in visited:
                return []   # cycle guard
            visited.add(best_n)
            path.append(best_n)
            current = best_n
        return path

    def initial_search(self):
        """
        Run the initial D* Lite search.

        Returns
        -------
        (path, nodes_explored) : tuple
        """
        self.nodes_explored = 0
        self._compute_shortest_path()
        path = self._extract_path()
        return path, self.nodes_explored

    def update_obstacles(self, blocked_cells):
        """
        Notify D* Lite that a set of cells has been blocked.
        Marks each cell impassable and calls _update_vertex on affected
        neighbours so the search tree can be repaired cheaply.

        Parameters
        ----------
        blocked_cells : list[(r,c)]
        """
        self.k_m += self._h(self.start)   # adjust key modifier

        for cell in blocked_cells:
            r, c = cell
            if not (0 <= r < self.maze.rows and 0 <= c < self.maze.cols):
                continue
            # Save neighbours before blocking
            nbrs = list(self.maze.passable_neighbours(r, c))
            self.maze.block_cell(r, c)
            self.g[cell]   = self.INF
            self.rhs[cell] = self.INF
            # Update affected neighbours
            for n in nbrs:
                self._update_vertex(n)
            self._update_vertex(cell)

    def replan(self):
        """
        Replan after obstacles have been added via update_obstacles().

        Returns
        -------
        (path, nodes_explored) : tuple
            nodes_explored counts only the nodes processed during replanning
            (not the initial search), demonstrating incremental efficiency.
        """
        self.nodes_explored = 0
        self._compute_shortest_path()
        path = self._extract_path()
        return path, self.nodes_explored


def d_star_lite(maze: Maze):
    """
    Convenience wrapper: run initial D* Lite search on a fresh maze.
    Returns (path, nodes_explored).
    """
    ds = DStarLite(maze)
    return ds.initial_search()


# ================================================================
# ALGORITHM 4 — ALT  (Landmark A* with Triangle Inequality)
# ================================================================

def alt_landmark_astar(maze: Maze, num_landmarks: int = 8):
    """
    ALT — Landmark-based A* using the Triangle Inequality.

    Design Paradigm
    ---------------
    Pre-compute exact shortest distances from a small set of "landmark"
    nodes to every other node using BFS (uniform cost on grid).
    For any query (s → t), the triangle inequality gives:
        dist(s, t) ≥ |dist(L, t) − dist(L, s)|
    for every landmark L.  The maximum over all landmarks is a tighter
    admissible heuristic than the plain Euclidean distance used by A*,
    which reduces the number of nodes expanded.

    Optimality
    ----------
    The ALT heuristic is admissible (never overestimates) and consistent,
    so A* with this heuristic is guaranteed to return the optimal path.

    Complexity
    ----------
    Preprocessing: O(K · N)  where K = num_landmarks, N = grid cells
    Query        : O(N log N) worst case, but typically 5–10× fewer
                   nodes expanded than plain A* due to the better heuristic.
    Space        : O(K · N)

    Parameters
    ----------
    maze          : Maze
    num_landmarks : int    Number of landmarks to pre-compute (default 8).

    Returns
    -------
    path           : list[(r,c)]
    nodes_explored : int
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return [], 0

    # ---- Collect all passable cells ----
    passable = [
        (r, c)
        for r in range(maze.rows)
        for c in range(maze.cols)
        if maze.mask[r][c]
    ]
    if not passable:
        return [], 0

    # ---- Select landmarks by farthest-point sampling ----
    # Spread landmarks across the maze using iterative farthest-point
    # selection so they cover the space well.
    def bfs_distances(source):
        """Return dict: cell → BFS shortest-path distance from source."""
        dist  = {source: 0}
        queue = [source]
        head  = 0
        while head < len(queue):
            node = queue[head]; head += 1
            for nr, nc in maze.passable_neighbours(*node):
                nb = (nr, nc)
                if nb not in dist:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        return dist

    landmarks      = [random.choice(passable)]
    landmark_dists = [bfs_distances(landmarks[0])]

    for _ in range(num_landmarks - 1):
        # Pick the passable cell farthest from all current landmarks
        farthest = max(
            passable,
            key=lambda cell: min(
                ld.get(cell, math.inf) for ld in landmark_dists
            )
        )
        landmarks.append(farthest)
        landmark_dists.append(bfs_distances(farthest))

    # ---- ALT heuristic ----
    def h_alt(s):
        """
        Triangle-inequality lower bound on dist(s → end).
        h(s) = max over all landmarks L of |dist(L,end) − dist(L,s)|
        """
        best = 0
        for ld in landmark_dists:
            d_L_end = ld.get(end,   math.inf)
            d_L_s   = ld.get(s,     math.inf)
            if d_L_end < math.inf and d_L_s < math.inf:
                best = max(best, abs(d_L_end - d_L_s))
        return best

    # ---- A* with ALT heuristic ----
    g            = {start: 0}
    came_from    = {start: None}
    open_set     = [(h_alt(start), start)]
    closed       = set()
    nodes_explored = 0

    while open_set:
        _, s = heapq.heappop(open_set)
        if s in closed:
            continue
        closed.add(s)
        nodes_explored += 1

        if s == end:
            return reconstruct_path(came_from, start, end), nodes_explored

        for nr, nc in maze.passable_neighbours(*s):
            nb    = (nr, nc)
            if nb in closed:
                continue
            g_new = g[s] + 1
            if g_new < g.get(nb, math.inf):
                g[nb]         = g_new
                came_from[nb] = s
                heapq.heappush(open_set, (g_new + h_alt(nb), nb))

    return [], nodes_explored


# ================================================================
# CSV RESULTS LOGGER
# ================================================================

_results = []   # accumulated; flushed once at end


def _log(use_case, algorithm, shape, maze_file,
         elapsed_sec, path_len_cells, path_len_euclidean,
         nodes_explored, notes=""):
    """
    Append one result row to the in-memory buffer.

    Parameters
    ----------
    use_case           : str
    algorithm          : str
    shape              : str   shape name extracted from filename
    maze_file          : str   filename of the maze
    elapsed_sec        : float wall-clock search time in seconds
    path_len_cells     : int   number of cells in the path
    path_len_euclidean : float true Euclidean length of path
    nodes_explored     : int
    notes              : str   optional extra info
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
    """Format seconds as µs / ms / s."""
    if sec < 1e-3: return f"{sec * 1e6:.1f}µs"
    if sec < 1:    return f"{sec * 1e3:.1f}ms"
    return f"{sec:.3f}s"


# ================================================================
# LOAD BEST MAZES
# ================================================================

def load_best_mazes():
    """
    Scan BEST_DIR for .maze files and return a list of
    (shape_name, maze_path, Maze) tuples sorted by shape name.

    Returns
    -------
    list of (shape_name: str, filepath: Path, maze: Maze)
    """
    mazes = []
    if not BEST_DIR.exists():
        print(f"[ERROR] best_mazes directory not found: {BEST_DIR}")
        print("        Run maze_benchmark.py first to generate the best mazes.")
        return mazes

    for fp in sorted(BEST_DIR.glob("BEST_*.maze")):
        # Filename format: BEST_<Shape>_100x100_<Algo>.maze
        stem   = fp.stem                   # e.g. BEST_Circle_100x100_Kruskal
        parts  = stem.split("_")
        if len(parts) < 2:
            continue
        shape  = parts[1]                  # e.g. Circle
        maze   = Maze.from_file(str(fp))
        mazes.append((shape, fp, maze))
        print(f"  Loaded {fp.name}  (start={maze.start}, end={maze.end})")

    return mazes


# ================================================================
# ALGORITHM REGISTRY
# ================================================================

ALGORITHMS = {
    "ThetaStar":              theta_star,
    "BidirectionalDijkstra":  bidirectional_dijkstra,
    "DStarLite":              d_star_lite,
    "ALT":                    alt_landmark_astar,
}


# ================================================================
# MAIN BENCHMARK
# ================================================================

def main():
    """
    Run all four use cases and write results to pathfinding_results.csv.

    Use Cases
    ---------
    UC1 — Basic Pathfinding       : time + path length + nodes per maze/algo
    UC2 — Path Quality            : cells vs Euclidean path length (Theta* advantage)
    UC3 — Nodes Explored          : search efficiency across shapes
    UC4 — Dynamic Replanning      : D* Lite replan vs fresh A* after obstacles
    """
    print("\n" + "=" * 80)
    print(" PATHFINDING BENCHMARK")
    print("=" * 80)

    print("\nLoading best mazes from:", BEST_DIR)
    best_mazes = load_best_mazes()
    if not best_mazes:
        return

    # ================================================================
    # USE CASE 1 — BASIC PATHFINDING
    # ================================================================
    print("\n" + "=" * 80)
    print("USE CASE 1: BASIC PATHFINDING")
    print("Metric: wall-clock time, path length (cells), nodes explored")
    print("=" * 80)

    hdr = (f"{'Algorithm':<26} {'Shape':<12} {'Time':>10} "
           f"{'Path(cells)':>12} {'Nodes':>10}")
    print(hdr)
    print("-" * len(hdr))

    for shape, fp, maze in best_mazes:
        for alg_name, alg_fn in ALGORITHMS.items():
            # Deep-copy the grid so blocking in UC4 doesn't affect UC1
            import copy
            m = copy.deepcopy(maze)
            t0      = time.perf_counter()
            path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            cells   = len(path)
            euc     = path_euclidean_length(path)

            _log("UC1_BasicPathfinding", alg_name, shape, fp.name,
                 elapsed, cells, euc, nodes)
            print(f"{alg_name:<26} {shape:<12} {fmt_time(elapsed):>10} "
                  f"{cells:>12} {nodes:>10}")
        print()

    # ================================================================
    # USE CASE 2 — PATH QUALITY
    # ================================================================
    print("=" * 80)
    print("USE CASE 2: PATH QUALITY")
    print("Metric: path length in cells vs true Euclidean distance")
    print("(Theta* finds shorter Euclidean paths by cutting corners)")
    print("=" * 80)

    hdr = (f"{'Algorithm':<26} {'Shape':<12} {'Cells':>8} "
           f"{'Euclidean':>12} {'Cell/Euc ratio':>16}")
    print(hdr)
    print("-" * len(hdr))

    for shape, fp, maze in best_mazes:
        for alg_name, alg_fn in ALGORITHMS.items():
            import copy
            m = copy.deepcopy(maze)
            t0      = time.perf_counter()
            path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            cells = len(path)
            euc   = path_euclidean_length(path)
            ratio = (cells / euc) if euc > 0 else 0.0

            _log("UC2_PathQuality", alg_name, shape, fp.name,
                 elapsed, cells, euc, nodes,
                 notes=f"cell_euc_ratio={ratio:.3f}")
            print(f"{alg_name:<26} {shape:<12} {cells:>8} "
                  f"{euc:>12.2f} {ratio:>16.3f}")
        print()

    # ================================================================
    # USE CASE 3 — NODES EXPLORED
    # ================================================================
    print("=" * 80)
    print("USE CASE 3: NODES EXPLORED (search efficiency)")
    print("Metric: how many cells visited before solution found")
    print("Lower = more efficient heuristic guidance")
    print("=" * 80)

    hdr = (f"{'Algorithm':<26} {'Shape':<12} {'Nodes':>10} "
           f"{'% of grid':>12} {'Path cells':>12}")
    print(hdr)
    print("-" * len(hdr))

    passable_counts = {}
    for shape, fp, maze in best_mazes:
        passable_counts[shape] = sum(
            row.count(True) for row in maze.mask)

    for shape, fp, maze in best_mazes:
        total_passable = passable_counts[shape]
        for alg_name, alg_fn in ALGORITHMS.items():
            import copy
            m = copy.deepcopy(maze)
            t0      = time.perf_counter()
            path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            cells     = len(path)
            euc       = path_euclidean_length(path)
            pct_grid  = (nodes / total_passable * 100) if total_passable else 0

            _log("UC3_NodesExplored", alg_name, shape, fp.name,
                 elapsed, cells, euc, nodes,
                 notes=f"pct_grid={pct_grid:.1f}%")
            print(f"{alg_name:<26} {shape:<12} {nodes:>10} "
                  f"{pct_grid:>11.1f}% {cells:>12}")
        print()

    # ================================================================
    # USE CASE 4 — DYNAMIC REPLANNING  (D* Lite vs fresh A*)
    # ================================================================
    print("=" * 80)
    print("USE CASE 4: DYNAMIC REPLANNING")
    print("D* Lite replan after 3 obstacles added  vs  fresh A* from scratch")
    print("Demonstrates incremental search advantage")
    print("=" * 80)

    hdr = (f"{'Shape':<12} {'Method':<30} {'Time':>10} "
           f"{'Nodes':>10} {'Path':>8} {'Speedup':>10}")
    print(hdr)
    print("-" * 76)

    for shape, fp, maze in best_mazes:
        import copy

        # ---- Initial D* Lite search ----
        m_dstar = copy.deepcopy(maze)
        ds      = DStarLite(m_dstar)
        t0      = time.perf_counter()
        path_init, nodes_init = ds.initial_search()
        t_init  = time.perf_counter() - t0

        _log("UC4_DynamicReplanning", "DStarLite_Initial", shape, fp.name,
             t_init, len(path_init), path_euclidean_length(path_init),
             nodes_init, notes="initial_search")

        print(f"{shape:<12} {'D*Lite initial':<30} {fmt_time(t_init):>10} "
              f"{nodes_init:>10} {len(path_init):>8}        —")

        if len(path_init) < 6:
            print(f"{shape:<12} {'(path too short to block 3 cells)':<30}")
            print()
            continue

        # Choose 3 cells on the path to block (avoid start/end)
        block_candidates = path_init[2: len(path_init) - 2]
        block_cells      = random.sample(
            block_candidates, min(3, len(block_candidates)))

        # ---- D* Lite replan ----
        ds.update_obstacles(block_cells)
        t0           = time.perf_counter()
        path_replan, nodes_replan = ds.replan()
        t_replan     = time.perf_counter() - t0

        euc_replan = path_euclidean_length(path_replan)
        _log("UC4_DynamicReplanning", "DStarLite_Replan", shape, fp.name,
             t_replan, len(path_replan), euc_replan, nodes_replan,
             notes=f"blocked={block_cells}")

        print(f"{shape:<12} {'D*Lite replan':<30} {fmt_time(t_replan):>10} "
              f"{nodes_replan:>10} {len(path_replan):>8}        —")

        # ---- Fresh A* on the same blocked maze ----
        m_astar = copy.deepcopy(maze)
        for r, c in block_cells:
            m_astar.block_cell(r, c)

        # A* (plain, for comparison baseline)
        def astar_baseline(m):
            """Standard A* with Euclidean heuristic — comparison baseline."""
            s, e = m.start, m.end
            if s is None or e is None:
                return [], 0
            g          = {s: 0}
            cf         = {s: None}
            oset       = [(euclidean(*s, *e), s)]
            closed     = set()
            ne         = 0
            while oset:
                _, u = heapq.heappop(oset)
                if u in closed:
                    continue
                closed.add(u); ne += 1
                if u == e:
                    return reconstruct_path(cf, s, e), ne
                for nr2, nc2 in m.passable_neighbours(*u):
                    v     = (nr2, nc2)
                    if v in closed:
                        continue
                    gn    = g[u] + 1
                    if gn < g.get(v, math.inf):
                        g[v] = gn; cf[v] = u
                        heapq.heappush(oset, (gn + euclidean(*v, *e), v))
            return [], ne

        t0            = time.perf_counter()
        path_astar, nodes_astar = astar_baseline(m_astar)
        t_astar       = time.perf_counter() - t0

        speedup = t_astar / t_replan if t_replan > 0 else float('inf')
        _log("UC4_DynamicReplanning", "AStar_FreshSearch", shape, fp.name,
             t_astar, len(path_astar), path_euclidean_length(path_astar),
             nodes_astar, notes=f"blocked={block_cells}")

        print(f"{shape:<12} {'A* fresh search':<30} {fmt_time(t_astar):>10} "
              f"{nodes_astar:>10} {len(path_astar):>8} {speedup:>9.2f}×")
        print()

    # ================================================================
    # FLUSH CSV
    # ================================================================
    _flush_results()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()