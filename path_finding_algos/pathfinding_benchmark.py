"""
d_star_animation.py
=========================
Pathfinding Animation Generator
Authors : (your group names here)
Course  : CSE 317 – Design and Analysis of Algorithms, Spring 2026

Overview
--------
Generates an animated GIF for every (algorithm × maze) combination showing:
  1. The maze walls rendered in full.
  2. Cells being explored / expanded (light blue, frame-by-frame).
  3. The final optimal path highlighted in orange/yellow.
  4. Start cell (green) and end cell (red) always visible.

Each animation is saved to:
    path_finding_algos/animations/<AlgoName>_<ShapeName>.gif

Requirements
------------
    pip install Pillow

File layout expected
--------------------
project_root/
    mazes_for_pathfinding/
        best_mazes/
            BEST_<shape>_100x100_<algo>.maze
    path_finding_algos/          ← this file lives here
        pathfinding_visualiser.py
        animations/              ← created automatically
            ThetaStar_Circle.gif
            ...

Usage
-----
    cd path_finding_algos
    python pathfinding_visualiser.py

    Optional flags:
        --cell   <int>    pixel size per cell    (default: 6)
        --fps    <int>    frames per second      (default: 20)
        --skip   <int>    explored cells per frame (default: 5,
                           higher = faster animation, fewer frames)
"""

import os
import math
import heapq
import argparse
import copy
from pathlib import Path
from PIL import Image, ImageDraw

# ================================================================
# PATH RESOLUTION
# ================================================================
_HERE     = Path(__file__).parent
_ROOT     = _HERE.parent
BEST_DIR  = _ROOT / "mazes_for_pathfinding" / "best_mazes"
ANIM_DIR  = _HERE / "animations"
ANIM_DIR.mkdir(exist_ok=True)

# ================================================================
# COLOUR PALETTE
# ================================================================
COL_WALL      = (30,  30,  30)   # near-black walls
COL_BG        = (255, 255, 255)  # white cell interior
COL_OUTSIDE   = (20,  20,  20)   # cells outside mask
COL_EXPLORED  = (173, 216, 230)  # light blue  – cells visited during search
COL_FRONTIER  = (100, 149, 237)  # cornflower  – cells currently in open set
COL_PATH      = (255, 165,  0)   # orange      – final optimal path
COL_START     = (0,   200,  0)   # green       – start cell
COL_END       = (220,  20,  60)  # crimson     – end cell


# ================================================================
# MINIMAL MAZE LOADER  (self-contained, no import from other files)
# ================================================================

class Maze:
    """Lightweight maze loaded from a .maze file."""

    N, E, S, W = 1, 2, 4, 8
    DX = {1: -1, 2: 0,  4: 1,  8: 0}
    DY = {1:  0, 2:  1, 4: 0,  8: -1}
    DIRS = [1, 2, 4, 8]

    def __init__(self, rows, cols, grid, mask, start, end):
        self.rows  = rows
        self.cols  = cols
        self.grid  = grid
        self.mask  = mask
        self.start = start
        self.end   = end

    def in_bounds(self, r, c):
        """Return True if (r, c) lies within the grid dimensions."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable_neighbours(self, r, c):
        """Yield (nr,nc) for every carved passage out of (r,c)."""
        for d in self.DIRS:
            if self.grid[r][c] & d:
                nr = r + self.DX[d]
                nc = c + self.DY[d]
                if 0 <= nr < self.rows and 0 <= nc < self.cols \
                        and self.mask[nr][nc]:
                    yield nr, nc

    def line_of_sight(self, r0, c0, r1, c1):
        """Bresenham line-of-sight that checks wall bits."""
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
        """Block a cell (used for UC4-style dynamic replanning visuals)."""
        opp = {1: 4, 4: 1, 2: 8, 8: 2}
        for d in self.DIRS:
            if self.grid[r][c] & d:
                nr = r + self.DX[d]; nc = c + self.DY[d]
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    self.grid[nr][nc] &= ~opp[d]
        self.grid[r][c] = 0
        self.mask[r][c] = False

    @staticmethod
    def from_file(path):
        """Load a .maze file written by maze_benchmark.py."""
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
    return math.hypot(r1 - r0, c1 - c0)


def reconstruct_path(came_from, start, end):
    path = []; node = end
    while node is not None:
        path.append(node); node = came_from.get(node)
    path.reverse()
    return path if (path and path[0] == start) else []


# ================================================================
# INSTRUMENTED ALGORITHMS
# These versions yield (event_type, cell) at each step so the
# visualiser can capture exploration order without BFS overhead.
#
# event_type:
#   'explore'  – cell popped from open set (being expanded)
#   'frontier' – cell pushed to open set
#   'path'     – final path cell (after search finishes)
# ================================================================

def theta_star_events(maze: Maze):
    """Theta* generator — yields search events then the full path."""
    start, end = maze.start, maze.end
    if start is None or end is None:
        return

    g         = {start: 0.0}
    came_from = {start: None}
    open_set  = [(euclidean(*start, *end), start)]
    closed    = set()

    while open_set:
        _, s = heapq.heappop(open_set)
        if s in closed:
            continue
        closed.add(s)
        yield ('explore', s)

        if s == end:
            path = reconstruct_path(came_from, start, end)
            for cell in path:
                yield ('path', cell)
            return

        for nr, nc in maze.passable_neighbours(*s):
            s2 = (nr, nc)
            if s2 in closed:
                continue
            parent_s = came_from[s]
            if parent_s is not None and maze.line_of_sight(*parent_s, *s2):
                g_new = g[parent_s] + euclidean(*parent_s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2] = g_new; came_from[s2] = parent_s
                    heapq.heappush(open_set,
                                   (g_new + euclidean(*s2, *end), s2))
                    yield ('frontier', s2)
            else:
                g_new = g[s] + euclidean(*s, *s2)
                if g_new < g.get(s2, math.inf):
                    g[s2] = g_new; came_from[s2] = s
                    heapq.heappush(open_set,
                                   (g_new + euclidean(*s2, *end), s2))
                    yield ('frontier', s2)


def bidirectional_dijkstra_events(maze: Maze):
    """Bidirectional Dijkstra generator."""
    start, end = maze.start, maze.end
    if start is None or end is None:
        return

    dist_f = {start: 0}; prev_f = {start: None}
    dist_b = {end:   0}; prev_b = {end:   None}
    open_f = [(0, start)]; open_b = [(0, end)]
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
        expand_fwd = (open_f and
                      (not open_b or open_f[0][0] <= open_b[0][0]))
        if expand_fwd and open_f:
            d, u = heapq.heappop(open_f)
            if u in settled_f: continue
            settled_f.add(u)
            yield ('explore', u)
            if d + dist_b.get(u, math.inf) < best:
                best = d + dist_b.get(u, math.inf); meeting = u
            if open_f and open_b and open_f[0][0] + open_b[0][0] >= best:
                break
            for nr, nc in maze.passable_neighbours(*u):
                v = (nr, nc); gn = dist_f[u] + 1
                if gn < dist_f.get(v, math.inf):
                    dist_f[v] = gn; prev_f[v] = u
                    heapq.heappush(open_f, (gn, v))
                    yield ('frontier', v)
        elif open_b:
            d, u = heapq.heappop(open_b)
            if u in settled_b: continue
            settled_b.add(u)
            yield ('explore', u)
            if dist_f.get(u, math.inf) + d < best:
                best = dist_f.get(u, math.inf) + d; meeting = u
            if open_f and open_b and open_f[0][0] + open_b[0][0] >= best:
                break
            for nr, nc in maze.passable_neighbours(*u):
                v = (nr, nc); gn = dist_b[u] + 1
                if gn < dist_b.get(v, math.inf):
                    dist_b[v] = gn; prev_b[v] = u
                    heapq.heappush(open_b, (gn, v))
                    yield ('frontier', v)

    if meeting:
        for cell in _build(meeting):
            yield ('path', cell)


def d_star_lite_events(maze: Maze):
    """
    D* Lite events generator.
    Runs the initial search and yields exploration + final path events.
    """
    start, end = maze.start, maze.end
    if start is None or end is None:
        return

    INF = math.inf
    g   = {}; rhs = {}
    for r in range(maze.rows):
        for c in range(maze.cols):
            if maze.mask[r][c]:
                g[(r,c)] = INF; rhs[(r,c)] = INF

    k_m  = 0
    rhs[end] = 0
    U    = []; U_set = {}

    def h(s): return euclidean(*s, *start)
    def key(s):
        m = min(g.get(s, INF), rhs.get(s, INF))
        return (m + h(s) + k_m, m)

    def push(s):
        k = key(s); heapq.heappush(U, (k, s)); U_set[s] = k

    def update(u):
        if u != end:
            nbrs = list(maze.passable_neighbours(*u))
            rhs[u] = min((1 + g.get(s2, INF) for s2 in nbrs),
                         default=INF)
        if u in U_set: del U_set[u]
        if g.get(u, INF) != rhs.get(u, INF): push(u)

    push(end)

    explored = set()
    while U:
        k_old, u = heapq.heappop(U)
        if U_set.get(u) != k_old: continue
        k_new = key(u)
        if k_old < k_new:
            heapq.heappush(U, (k_new, u)); U_set[u] = k_new; continue
        if g.get(u, INF) > rhs.get(u, INF):
            g[u] = rhs[u]; del U_set[u]
            if u not in explored:
                explored.add(u); yield ('explore', u)
            for s in maze.passable_neighbours(*u): update(s)
        else:
            g[u] = INF; update(u)
            if u not in explored:
                explored.add(u); yield ('explore', u)
            for s in maze.passable_neighbours(*u): update(s)
        sk = key(start)
        if U and U[0][0] >= sk and rhs.get(start, INF) == g.get(start, INF):
            break

    # Extract path
    path = [start]; cur = start; vis = {start}
    while cur != end:
        nbrs = list(maze.passable_neighbours(*cur))
        if not nbrs: break
        best_n = min(nbrs, key=lambda s: g.get(s, INF))
        if g.get(best_n, INF) == INF or best_n in vis: break
        vis.add(best_n); path.append(best_n); cur = best_n
    for cell in path:
        yield ('path', cell)


def alt_events(maze: Maze, num_landmarks: int = 8):
    """ALT (Landmark A*) events generator."""
    start, end = maze.start, maze.end
    if start is None or end is None:
        return

    passable = [(r, c) for r in range(maze.rows)
                for c in range(maze.cols) if maze.mask[r][c]]
    if not passable:
        return

    def bfs_dist(src):
        dist = {src: 0}; q = [src]; head = 0
        while head < len(q):
            node = q[head]; head += 1
            for nr, nc in maze.passable_neighbours(*node):
                nb = (nr, nc)
                if nb not in dist:
                    dist[nb] = dist[node] + 1; q.append(nb)
        return dist

    import random as _rnd
    landmarks = [_rnd.choice(passable)]
    ldists    = [bfs_dist(landmarks[0])]
    for _ in range(num_landmarks - 1):
        farthest = max(passable,
                       key=lambda cell: min(ld.get(cell, math.inf)
                                            for ld in ldists))
        landmarks.append(farthest); ldists.append(bfs_dist(farthest))

    def h_alt(s):
        best = 0
        for ld in ldists:
            a = ld.get(end, math.inf); b = ld.get(s, math.inf)
            if a < math.inf and b < math.inf:
                best = max(best, abs(a - b))
        return best

    g_cost    = {start: 0}; came_from = {start: None}
    open_set  = [(h_alt(start), start)]; closed = set()

    while open_set:
        _, s = heapq.heappop(open_set)
        if s in closed: continue
        closed.add(s)
        yield ('explore', s)
        if s == end:
            path = reconstruct_path(came_from, start, end)
            for cell in path: yield ('path', cell)
            return
        for nr, nc in maze.passable_neighbours(*s):
            nb = (nr, nc)
            if nb in closed: continue
            gn = g_cost[s] + 1
            if gn < g_cost.get(nb, math.inf):
                g_cost[nb] = gn; came_from[nb] = s
                heapq.heappush(open_set, (gn + h_alt(nb), nb))
                yield ('frontier', nb)


ALGO_EVENTS = {
    "ThetaStar":             theta_star_events,
    "BidirectionalDijkstra": bidirectional_dijkstra_events,
    "DStarLite":             d_star_lite_events,
    "ALT":                   alt_events,
}


# ================================================================
# FRAME RENDERER
# ================================================================

class MazeRenderer:
    """
    Renders a single maze frame as a PIL Image.

    State is built incrementally — cell colours are tracked in a
    2-D colour grid and overwritten as the search progresses.
    """

    def __init__(self, maze: Maze, cell_size: int = 6, wall_width: int = 1):
        """
        Parameters
        ----------
        maze      : Maze
        cell_size : int   Pixel width/height of each cell.
        wall_width: int   Pixel thickness of wall lines.
        """
        self.maze       = maze
        self.cs         = cell_size
        self.ww         = max(1, wall_width)
        self.img_w      = maze.cols * cell_size + self.ww
        self.img_h      = maze.rows * cell_size + self.ww

        # Per-cell colour overlay (r, c) → RGB tuple
        self.colours = {}

    def _base_image(self) -> Image.Image:
        """
        Draw the maze walls on a white background.
        Non-mask cells are filled with COL_OUTSIDE.
        Returns a PIL Image with walls drawn but no search overlay.
        """
        img  = Image.new("RGB", (self.img_w, self.img_h), COL_BG)
        draw = ImageDraw.Draw(img)

        # Fill non-mask cells
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
                if not self.maze.mask[r][c]:
                    continue
                x1 = c * self.cs; y1 = r * self.cs
                x2 = x1 + self.cs; y2 = y1 + self.cs
                wh = self.ww
                # North wall
                nr2 = r - 1
                if not (self.maze.in_bounds(nr2, c)
                        and self.maze.mask[nr2][c]
                        and self.maze.grid[r][c] & self.maze.N):
                    draw.line([(x1, y1), (x2, y1)],
                               fill=COL_WALL, width=wh)
                # South wall
                sr2 = r + 1
                if not (self.maze.in_bounds(sr2, c)
                        and self.maze.mask[sr2][c]
                        and self.maze.grid[r][c] & self.maze.S):
                    draw.line([(x1, y2), (x2, y2)],
                               fill=COL_WALL, width=wh)
                # West wall
                wc2 = c - 1
                if not (self.maze.in_bounds(r, wc2)
                        and self.maze.mask[r][wc2]
                        and self.maze.grid[r][c] & self.maze.W):
                    draw.line([(x1, y1), (x1, y2)],
                               fill=COL_WALL, width=wh)
                # East wall
                ec2 = c + 1
                if not (self.maze.in_bounds(r, ec2)
                        and self.maze.mask[r][ec2]
                        and self.maze.grid[r][c] & self.maze.E):
                    draw.line([(x2, y1), (x2, y2)],
                               fill=COL_WALL, width=wh)

        return img

    def render(self, highlight_path=None) -> Image.Image:
        """
        Produce a frame image compositing the current colour overlay
        onto the base maze walls image.

        Parameters
        ----------
        highlight_path : list[(r,c)] | None
            If given, these cells are coloured COL_PATH regardless of
            the current overlay.

        Returns
        -------
        PIL.Image.Image
        """
        img  = self._base_image()
        draw = ImageDraw.Draw(img)

        dot  = max(1, self.cs // 3)

        # Apply cell colour overlay
        for (r, c), col in self.colours.items():
            x1 = c * self.cs + self.ww
            y1 = r * self.cs + self.ww
            x2 = x1 + self.cs - self.ww
            y2 = y1 + self.cs - self.ww
            draw.rectangle([x1, y1, x2, y2], fill=col)

        # Draw path overlay
        if highlight_path:
            for r, c in highlight_path:
                x1 = c * self.cs + self.ww
                y1 = r * self.cs + self.ww
                x2 = x1 + self.cs - self.ww
                y2 = y1 + self.cs - self.ww
                draw.rectangle([x1, y1, x2, y2], fill=COL_PATH)

        # Always draw start and end on top
        if self.maze.start:
            sr, sc = self.maze.start
            cx_s = sc * self.cs + self.cs // 2
            cy_s = sr * self.cs + self.cs // 2
            draw.ellipse([cx_s - dot, cy_s - dot,
                           cx_s + dot, cy_s + dot], fill=COL_START)

        if self.maze.end:
            er, ec = self.maze.end
            cx_e = ec * self.cs + self.cs // 2
            cy_e = er * self.cs + self.cs // 2
            draw.ellipse([cx_e - dot, cy_e - dot,
                           cx_e + dot, cy_e + dot], fill=COL_END)

        return img


# ================================================================
# GIF BUILDER
# ================================================================

def build_animation(maze: Maze, events_fn, cell_size: int = 6,
                    fps: int = 20, cells_per_frame: int = 5) -> list:
    """
    Drive an instrumented algorithm, capture frames, and return a list
    of PIL Images suitable for saving as an animated GIF.

    Parameters
    ----------
    maze            : Maze
    events_fn       : callable   One of the ALGO_EVENTS generators.
    cell_size       : int        Pixel size per cell.
    fps             : int        Frames per second (controls duration).
    cells_per_frame : int        How many explored cells to batch per frame.
                                 Higher = faster/smaller GIF.

    Returns
    -------
    list[PIL.Image.Image]   Sequence of frames.
    """
    renderer  = MazeRenderer(maze, cell_size=cell_size)
    frames    = []
    path      = []
    explored  = []
    frontier_cells = []

    # Collect all events
    for event_type, cell in events_fn(maze):
        if event_type == 'explore':
            explored.append(cell)
        elif event_type == 'frontier':
            frontier_cells.append(cell)
        elif event_type == 'path':
            path.append(cell)

    # Frame 0: blank maze (walls only)
    frames.append(renderer.render())

    # Build exploration frames
    batch  = []
    for i, cell in enumerate(explored):
        renderer.colours[cell] = COL_EXPLORED
        batch.append(cell)
        if len(batch) >= cells_per_frame:
            frames.append(renderer.render())
            batch = []

    if batch:   # flush remaining
        frames.append(renderer.render())

    # Final frame: show optimal path in orange
    # Render 5 extra copies of the final frame so the path is visible
    for _ in range(max(1, fps // 4)):
        frames.append(renderer.render(highlight_path=path))

    return frames


def save_gif(frames: list, out_path: Path, fps: int = 20):
    """
    Save a list of PIL Images as an animated GIF.

    Parameters
    ----------
    frames   : list[PIL.Image.Image]
    out_path : Path
    fps      : int
    """
    if not frames:
        print(f"  [WARN] No frames for {out_path.name} — skipping.")
        return
    duration_ms = max(20, 1000 // fps)
    # Convert all frames to palette mode for GIF compatibility
    palette_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=256)
                      for f in frames]
    palette_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,              # loop forever
        optimize=False,
    )
    size_kb = out_path.stat().st_size // 1024
    print(f"  Saved {out_path.name}  ({len(frames)} frames, ~{size_kb} KB)")


# ================================================================
# MAIN
# ================================================================

def main():
    """
    Generate one animated GIF per (algorithm × shape) combination.
    GIFs are written to path_finding_algos/animations/.
    """
    parser = argparse.ArgumentParser(
        description="Pathfinding animation generator")
    parser.add_argument("--cell",  type=int, default=6,
                        help="Pixel size per maze cell (default 6)")
    parser.add_argument("--fps",   type=int, default=20,
                        help="Frames per second (default 20)")
    parser.add_argument("--skip",  type=int, default=5,
                        help="Explored cells batched per frame (default 5)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" PATHFINDING VISUALISER")
    print("=" * 70)
    print(f"Settings: cell={args.cell}px  fps={args.fps}  "
          f"cells_per_frame={args.skip}")
    print(f"Output  : {ANIM_DIR}")

    # Load best mazes
    if not BEST_DIR.exists():
        print(f"\n[ERROR] best_mazes directory not found:\n  {BEST_DIR}")
        print("Run maze_benchmark.py first.")
        return

    maze_files = sorted(BEST_DIR.glob("BEST_*.maze"))
    if not maze_files:
        print(f"\n[ERROR] No BEST_*.maze files found in {BEST_DIR}")
        return

    print(f"\nFound {len(maze_files)} best maze(s).")

    for fp in maze_files:
        stem  = fp.stem             # BEST_<Shape>_100x100_<Algo>
        parts = stem.split("_")
        shape = parts[1] if len(parts) > 1 else "Unknown"

        print(f"\n── {fp.name}  (shape: {shape})")

        for alg_name, events_fn in ALGO_EVENTS.items():
            print(f"   Rendering {alg_name}...", end=" ", flush=True)

            # Fresh deep copy for each algorithm
            maze = copy.deepcopy(Maze.from_file(str(fp)))

            if maze.start is None or maze.end is None:
                print("SKIP (no start/end)")
                continue

            frames = build_animation(
                maze, events_fn,
                cell_size=args.cell,
                fps=args.fps,
                cells_per_frame=args.skip,
            )

            out_path = ANIM_DIR / f"{alg_name}_{shape}.gif"
            save_gif(frames, out_path, fps=args.fps)

    print("\n" + "=" * 70)
    print(f"All animations saved to: {ANIM_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()