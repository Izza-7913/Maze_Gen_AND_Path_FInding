"""
pathfinding_benchmark.py
========================
Pathfinding Benchmark
Authors : Hamna Sajid
Course  : CSE 317 – Design and Analysis of Algorithms, Spring 2026

Overview
--------
Runs three pathfinding algorithms on the best mazes produced by
maze_benchmark.py, benchmarks them across three use cases, and records
all results to a CSV file.

All three algorithms are guaranteed to return the OPTIMAL path (shortest
in terms of cells visited / true Euclidean distance where applicable).

Algorithms
----------
1. Theta*                 – Any-angle pathfinding, truly optimal Euclidean paths
2. Bidirectional Dijkstra – Search from both ends simultaneously
3. ALT (Landmark A*)      – A* with triangle-inequality landmark heuristics

Use Cases
---------
UC1 — Basic Pathfinding  : time, path length (steps), nodes explored per maze
UC2 — Path Quality       : path length in steps vs true Euclidean distance
UC3 — Nodes Explored     : pure search efficiency (nodes visited before solution)

File layout expected
--------------------
project_root/
    mazes_for_pathfinding/
        best_mazes/
            BEST_<shape>_100x100_<algo>.maze
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
from pathlib import Path

# ================================================================
# PATH RESOLUTION
# ================================================================
_HERE       = Path(__file__).parent
_ROOT       = _HERE.parent
BEST_DIR    = _ROOT / "mazes_for_pathfinding" / "best_mazes"
RESULTS_CSV = _HERE / "pathfinding_results.csv"


# ================================================================
# MAZE LOADER
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
        path_length = len(path) - 1, or -1 if no path found
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return -1, [], 0

    g              = {start: 0.0}
    came_from      = {start: None}
    open_set       = [(euclidean(*start, *end), start)]
    closed         = set()
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
                # Theta* shortcut: link s2 directly to parent(s)
                g_new = g[parent_s] + euclidean(*parent_s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2]         = g_new
                    came_from[s2] = parent_s
                    heapq.heappush(open_set,
                                   (g_new + euclidean(*s2, *end), s2))
            else:
                # Standard A* link through s
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
        path_length = len(path) - 1, or -1 if no path found
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return -1, [], 0

    dist_f    = {start: 0};  prev_f = {start: None};  open_f = [(0, start)]
    dist_b    = {end:   0};  prev_b = {end:   None};  open_b = [(0, end)]
    settled_f = set();       settled_b = set()
    best      = math.inf;    meeting = None
    nodes_explored = 0

    def _build(node):
        """Stitch forward and backward chains at the meeting node."""
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
# ALGORITHM 3 — ALT  (Landmark A* with Triangle Inequality)
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
        path_length = len(path) - 1, or -1 if no path found
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

    # Farthest-point landmark selection for good spatial coverage
    landmarks      = [random.choice(passable)]
    landmark_dists = [bfs_distances(landmarks[0])]
    for _ in range(num_landmarks - 1):
        farthest = max(passable,
                       key=lambda cell: min(ld.get(cell, math.inf)
                                            for ld in landmark_dists))
        landmarks.append(farthest)
        landmark_dists.append(bfs_distances(farthest))

    def h_alt(s):
        """Triangle-inequality lower bound on dist(s → end)."""
        best = 0
        for ld in landmark_dists:
            a = ld.get(end, math.inf); b = ld.get(s, math.inf)
            if a < math.inf and b < math.inf:
                best = max(best, abs(a - b))
        return best

    g_cost         = {start: 0}
    came_from      = {start: None}
    open_set       = [(h_alt(start), start)]
    closed         = set()
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
# ALGORITHM REGISTRY
# All callables accept (maze: Maze) and return
# (path_length: int, path: list, nodes_explored: int)
# path_length = len(path)-1, or -1 if no path found
# ================================================================

ALGORITHMS = {
    "ThetaStar":             theta_star,
    "BidirectionalDijkstra": bidirectional_dijkstra,
    "ALT":                   alt_landmark_astar,
}


# ================================================================
# CSV RESULTS LOGGER
# ================================================================

_results = []


def _log(use_case, algorithm, shape, maze_file,
         elapsed_sec, path_len_cells, path_len_euclidean,
         nodes_explored, notes=""):
    """Append one result row to the in-memory CSV buffer."""
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
        stem  = fp.stem
        parts = stem.split("_")
        shape = parts[1] if len(parts) > 1 else "Unknown"
        maze  = Maze.from_file(str(fp))
        mazes.append((shape, fp, maze))
        print(f"  Loaded {fp.name}  (start={maze.start}, end={maze.end})")

    return mazes


# ================================================================
# MAIN BENCHMARK
# ================================================================

def main():
    """
    Run all three use cases across all three algorithms and write results.

    Use Cases
    ---------
    UC1 — Basic Pathfinding  : time, path length (steps), nodes explored
    UC2 — Path Quality       : steps vs Euclidean length (Theta* advantage)
    UC3 — Nodes Explored     : search efficiency — lower = better heuristic
    """
    print("\n" + "=" * 80)
    print(" PATHFINDING BENCHMARK")
    print("=" * 80)
    print("Algorithms : " + " | ".join(ALGORITHMS))

    print("\nLoading best mazes from:", BEST_DIR)
    best_mazes = load_best_mazes()
    if not best_mazes:
        return

    # ================================================================
    # USE CASE 1 — BASIC PATHFINDING
    # ================================================================
    print("\n" + "=" * 80)
    print("USE CASE 1: BASIC PATHFINDING")
    print("Metric: wall-clock time, path length (steps), nodes explored")
    print("=" * 80)

    hdr = (f"{'Algorithm':<26} {'Shape':<12} {'Time':>10} "
           f"{'Path(steps)':>12} {'Nodes':>10}")
    print(hdr); print("-" * len(hdr))

    for shape, fp, maze in best_mazes:
        for alg_name, alg_fn in ALGORITHMS.items():
            m = copy.deepcopy(maze)
            t0 = time.perf_counter()
            path_length, path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            euc = path_euclidean_length(path)
            _log("UC1_BasicPathfinding", alg_name, shape, fp.name,
                 elapsed, len(path), euc, nodes,
                 notes=f"path_length={path_length}")
            print(f"{alg_name:<26} {shape:<12} {fmt_time(elapsed):>10} "
                  f"{path_length:>12} {nodes:>10}")
        print()

    # ================================================================
    # USE CASE 2 — PATH QUALITY
    # ================================================================
    print("=" * 80)
    print("USE CASE 2: PATH QUALITY")
    print("Metric: path length in steps vs true Euclidean distance")
    print("(Theta* finds shorter Euclidean paths by cutting corners)")
    print("=" * 80)

    hdr = (f"{'Algorithm':<26} {'Shape':<12} {'Steps':>8} "
           f"{'Euclidean':>12} {'Step/Euc ratio':>16}")
    print(hdr); print("-" * len(hdr))

    for shape, fp, maze in best_mazes:
        for alg_name, alg_fn in ALGORITHMS.items():
            m = copy.deepcopy(maze)
            t0 = time.perf_counter()
            path_length, path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            euc   = path_euclidean_length(path)
            ratio = (path_length / euc) if euc > 0 else 0.0
            _log("UC2_PathQuality", alg_name, shape, fp.name,
                 elapsed, len(path), euc, nodes,
                 notes=f"path_length={path_length} step_euc_ratio={ratio:.3f}")
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
            t0 = time.perf_counter()
            path_length, path, nodes = alg_fn(m)
            elapsed = time.perf_counter() - t0

            euc      = path_euclidean_length(path)
            pct_grid = (nodes / total_passable * 100) if total_passable else 0
            _log("UC3_NodesExplored", alg_name, shape, fp.name,
                 elapsed, len(path), euc, nodes,
                 notes=f"path_length={path_length} pct_grid={pct_grid:.1f}%")
            print(f"{alg_name:<26} {shape:<12} {nodes:>10} "
                  f"{pct_grid:>11.1f}% {path_length:>12}")
        print()

    # ================================================================
    # FLUSH CSV
    # ================================================================
    _flush_results()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()