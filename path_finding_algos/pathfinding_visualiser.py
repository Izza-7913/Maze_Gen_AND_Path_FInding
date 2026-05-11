"""
pathfinding_visualiser.py
=========================
Pathfinding Animation Generator
Authors : Hamna Sajid, Fatima Ishaq
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
1. Theta*                 – any-angle A*
2. Bidirectional Dijkstra – dual-frontier search
3. ALT                    – landmark A*

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
import random
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
COL_WALL     = (30,  30,  30)
COL_BG       = (255, 255, 255)
COL_OUTSIDE  = (20,  20,  20)
COL_EXPLORED = (173, 216, 230)   # light blue  – cells expanded
COL_FRONTIER = (100, 149, 237)   # cornflower  – open-set cells
COL_PATH     = (255, 165,   0)   # orange      – final path
COL_START    = (0,   200,   0)   # green       – start
COL_END      = (220,  20,  60)   # crimson     – end


# ================================================================
# MAZE  (self-contained — no import from other files)
# ================================================================

class Maze:
    """Lightweight maze loaded from a .maze file."""

    N, E, S, W = 1, 2, 4, 8
    DX   = {1: -1, 2: 0,  4: 1,  8: 0}
    DY   = {1:  0, 2:  1, 4: 0,  8: -1}
    DIRS = [1, 2, 4, 8]

    def __init__(self, rows, cols, grid, mask, start, end):
        self.rows  = rows;  self.cols  = cols
        self.grid  = grid;  self.mask  = mask
        self.start = start; self.end   = end

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
#
# Each generator yields (event_type, cell) during the search so the
# visualiser can capture the exploration order frame by frame.
#
# event_type values
# -----------------
# 'explore'  – cell popped from the open set (being expanded)
# 'frontier' – cell pushed onto the open set
# 'path'     – cell is on the final optimal path
# ================================================================

# ----------------------------------------------------------------
# 1. Theta*
# ----------------------------------------------------------------
def theta_star_events(maze: Maze):
    """
    Theta* generator.

    Yields exploration and final-path events so the visualiser can
    animate the any-angle search spreading through the maze and then
    show the shortcut-smoothed orange path at the end.
    """
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


# ----------------------------------------------------------------
# 2. Bidirectional Dijkstra
# ----------------------------------------------------------------
def bidirectional_dijkstra_events(maze: Maze):
    """
    Bidirectional Dijkstra generator.

    Yields exploration events from both the forward (start) and backward
    (end) frontiers so the animation shows two waves converging toward
    each other before the meeting point is stitched into the final path.
    """
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
            if open_f and open_b and open_f[0][0] + open_b[0][0] >= best:
                break
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
            if open_f and open_b and open_f[0][0] + open_b[0][0] >= best:
                break
            for nr, nc in maze.passable_neighbours(*u):
                v = (nr, nc); gn = dist_b[u] + 1
                if gn < dist_b.get(v, math.inf):
                    dist_b[v] = gn; prev_b[v] = u
                    heapq.heappush(open_b, (gn, v)); yield ('frontier', v)

    if meeting:
        for cell in _build(meeting): yield ('path', cell)


# ----------------------------------------------------------------
# 3. ALT  (Landmark A*)
# ----------------------------------------------------------------
def alt_events(maze: Maze, num_landmarks: int = 8):
    """
    ALT generator.

    Pre-computes BFS landmark distances, then runs A* with the
    triangle-inequality heuristic.  The animation shows a more
    directed search than plain Dijkstra because the tighter heuristic
    steers exploration toward the goal more aggressively.
    """
    start, end = maze.start, maze.end
    if start is None or end is None: return

    passable = [(r, c) for r in range(maze.rows)
                for c in range(maze.cols) if maze.mask[r][c]]
    if not passable: return

    def bfs_dist(src):
        dist = {src: 0}; q = [src]; head = 0
        while head < len(q):
            node = q[head]; head += 1
            for nr, nc in maze.passable_neighbours(*node):
                nb = (nr, nc)
                if nb not in dist: dist[nb] = dist[node] + 1; q.append(nb)
        return dist

    landmarks = [random.choice(passable)]; ldists = [bfs_dist(landmarks[0])]
    for _ in range(num_landmarks - 1):
        farthest = max(passable,
                       key=lambda cell: min(ld.get(cell, math.inf)
                                            for ld in ldists))
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


# ================================================================
# ALGORITHM EVENT GENERATORS REGISTRY
# ================================================================

ALGO_EVENTS = {
    "ThetaStar":             theta_star_events,
    "BidirectionalDijkstra": bidirectional_dijkstra_events,
    "ALT":                   alt_events,
}


# ================================================================
# FRAME RENDERER
# ================================================================

class MazeRenderer:
    """
    Renders maze frames as PIL Images.

    Maintains a per-cell colour dictionary that is updated incrementally
    as search events arrive.  Each call to render() composites the current
    colour state onto a freshly drawn set of maze walls so the walls are
    always crisp regardless of cell colouring.
    """

    def __init__(self, maze: Maze, cell_size: int = 6, wall_width: int = 1):
        self.maze  = maze
        self.cs    = cell_size
        self.ww    = max(1, wall_width)
        self.img_w = maze.cols * cell_size + self.ww
        self.img_h = maze.rows * cell_size + self.ww
        self.colours = {}   # (r,c) → RGB tuple

    def _base_image(self) -> Image.Image:
        """
        Draw maze walls on a white background.
        Cells outside the shape mask are filled solid black.
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

        # Draw walls for every passable cell
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if not self.maze.mask[r][c]: continue
                x1 = c * self.cs; y1 = r * self.cs
                x2 = x1 + self.cs; y2 = y1 + self.cs
                wh = self.ww

                # North wall
                nr2 = r - 1
                if not (self.maze.in_bounds(nr2, c)
                        and self.maze.mask[nr2][c]
                        and self.maze.grid[r][c] & self.maze.N):
                    draw.line([(x1, y1), (x2, y1)], fill=COL_WALL, width=wh)
                # South wall
                sr2 = r + 1
                if not (self.maze.in_bounds(sr2, c)
                        and self.maze.mask[sr2][c]
                        and self.maze.grid[r][c] & self.maze.S):
                    draw.line([(x1, y2), (x2, y2)], fill=COL_WALL, width=wh)
                # West wall
                wc2 = c - 1
                if not (self.maze.in_bounds(r, wc2)
                        and self.maze.mask[r][wc2]
                        and self.maze.grid[r][c] & self.maze.W):
                    draw.line([(x1, y1), (x1, y2)], fill=COL_WALL, width=wh)
                # East wall
                ec2 = c + 1
                if not (self.maze.in_bounds(r, ec2)
                        and self.maze.mask[r][ec2]
                        and self.maze.grid[r][c] & self.maze.E):
                    draw.line([(x2, y1), (x2, y2)], fill=COL_WALL, width=wh)

        return img

    def render(self, highlight_path=None) -> Image.Image:
        """
        Produce one animation frame.

        Composites the per-cell colour overlay onto the base wall image,
        then draws the final path (if provided) in orange on top, and
        finishes with the start (green) and end (red) dots always visible.

        Parameters
        ----------
        highlight_path : list[(r,c)] | None
            If given, these cells are drawn in COL_PATH regardless of the
            current exploration overlay.
        """
        img  = self._base_image()
        draw = ImageDraw.Draw(img)
        dot  = max(1, self.cs // 3)

        # Cell colour overlay (exploration state)
        for (r, c), col in self.colours.items():
            x1 = c * self.cs + self.ww; y1 = r * self.cs + self.ww
            x2 = x1 + self.cs - self.ww; y2 = y1 + self.cs - self.ww
            draw.rectangle([x1, y1, x2, y2], fill=col)

        # Final path overlay
        if highlight_path:
            for r, c in highlight_path:
                x1 = c * self.cs + self.ww; y1 = r * self.cs + self.ww
                x2 = x1 + self.cs - self.ww; y2 = y1 + self.cs - self.ww
                draw.rectangle([x1, y1, x2, y2], fill=COL_PATH)

        # Start and end dots always rendered on top
        if self.maze.start:
            sr, sc = self.maze.start
            cx_s = sc * self.cs + self.cs // 2
            cy_s = sr * self.cs + self.cs // 2
            draw.ellipse([cx_s-dot, cy_s-dot, cx_s+dot, cy_s+dot],
                         fill=COL_START)
        if self.maze.end:
            er, ec = self.maze.end
            cx_e = ec * self.cs + self.cs // 2
            cy_e = er * self.cs + self.cs // 2
            draw.ellipse([cx_e-dot, cy_e-dot, cx_e+dot, cy_e+dot],
                         fill=COL_END)

        return img


# ================================================================
# GIF BUILDER
# ================================================================

_EVENT_COLOUR = {
    'explore':  COL_EXPLORED,
    'frontier': COL_FRONTIER,
}


def build_animation(maze: Maze, events_fn,
                    cell_size: int = 6,
                    fps: int = 20,
                    cells_per_frame: int = 5) -> list:
    """
    Drive an instrumented algorithm generator and capture PIL Image frames.

    All events are collected first, then replayed into frames in batches of
    *cells_per_frame* events per frame.  Higher values of cells_per_frame
    produce fewer frames and a smaller GIF file.

    Parameters
    ----------
    maze            : Maze
    events_fn       : one of the ALGO_EVENTS generator functions
    cell_size       : int
    fps             : int
    cells_per_frame : int

    Returns
    -------
    list[PIL.Image.Image]
    """
    renderer        = MazeRenderer(maze, cell_size=cell_size)
    frames          = []
    path            = []
    non_path_events = []

    # Collect all events from the generator
    for event_type, cell in events_fn(maze):
        if event_type == 'path':
            path.append(cell)
        else:
            non_path_events.append((event_type, cell))

    # Frame 0: blank maze (walls only, no exploration yet)
    frames.append(renderer.render())

    # Build exploration frames, batching events for file-size efficiency
    batch = []
    for event_type, cell in non_path_events:
        col = _EVENT_COLOUR.get(event_type, COL_EXPLORED)
        renderer.colours[cell] = col
        batch.append(cell)
        if len(batch) >= cells_per_frame:
            frames.append(renderer.render())
            batch = []
    if batch:
        frames.append(renderer.render())

    # Hold the final frame (with path in orange) for ~fps//4 extra frames
    # so the viewer has time to see the completed path
    hold = max(1, fps // 4)
    for _ in range(hold):
        frames.append(renderer.render(highlight_path=path))

    return frames


def save_gif(frames: list, out_path: Path, fps: int = 20):
    """
    Save a list of PIL Images as a looping animated GIF.

    Parameters
    ----------
    frames   : list[PIL.Image.Image]
    out_path : Path
    fps      : int
    """
    if not frames:
        print(f"  [WARN] No frames for {out_path.name} — skipping.")
        return
    duration_ms    = max(20, 1000 // fps)
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
    """
    Generate one animated GIF per (algorithm × shape) combination.
    3 algorithms × number of best mazes = total GIFs saved to animations/.
    """
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
    print(" PATHFINDING VISUALISER ")
    print("=" * 70)
    print(f"Settings  : cell={args.cell}px  fps={args.fps}  "
          f"cells_per_frame={args.skip}")
    print(f"Output    : {ANIM_DIR}")
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
        stem  = fp.stem
        parts = stem.split("_")
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