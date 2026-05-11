"""
maze_benchmark.py
=================
Maze Generation Benchmark
Authors : Hamna Sajid, Izza Sohail
Course  : CSE 317 – Design and Analysis of Algorithms, Spring 2026

Overview
--------
Benchmarks five classic maze-generation algorithms across multiple grid sizes,
aspect ratios, and shape masks.  For every run the wall-clock generation time
is recorded to a CSV file.  After all benchmarks finish the fastest algorithm
per shape (at 100×100) is identified and those five "best" mazes are saved to
a dedicated folder for direct use by pathfinding algorithms.

Every generated maze has exactly ONE start cell (green dot, topmost-leftmost
boundary cell) and ONE end cell (red dot, bottommost-rightmost boundary cell).

Algorithms
----------
1. Recursive Backtracker  – DFS / stack-based
2. Randomised Prim's
3. Randomised Kruskal's
4. Eller's algorithm
5. Wilson's algorithm     – loop-erased random walk

Output layout
-------------
mazes_for_pathfinding/
    *.maze                  – raw maze data for every run
    benchmark_times.csv     – timing record for every run
    images/
        *.png               – rendered image for every maze
    best_mazes/
        BEST_<shape>_100x100_<algo>.maze
        BEST_<shape>_100x100_<algo>.png
        best_mazes_summary.csv
        best_mazes_summary.txt
"""

import random, time, tracemalloc, os, argparse, csv
from pathlib import Path
from PIL import Image, ImageDraw
import math

# ================================================================
#  MAZE REPRESENTATION (with shape mask)
# ================================================================
N, E, S, W = 1, 2, 4, 8
OPPOSITE = {N: S, S: N, E: W, W: E}
DX = {N: -1, S: 1, E: 0, W: 0}
DY = {N: 0, S: 0, E: 1, W: -1}


class Maze:
    """
    Core maze data structure.

    Attributes
    ----------
    rows, cols : int
        Grid dimensions.
    grid : list[list[int]]
        Bitmask per cell.  Bit N(1) set means the northern wall is carved open,
        E(2) = east open, S(4) = south open, W(8) = west open.
    mask : list[list[bool]]
        mask[r][c] = False → cell is outside the shape (solid, never carved).
    start : tuple[int,int] | None
        (row, col) of the maze entrance – rendered as a green dot.
    end : tuple[int,int] | None
        (row, col) of the maze exit – rendered as a red dot.
    """

    def __init__(self, rows, cols, mask=None):
        """
        Create an empty maze (no passages carved yet).

        Parameters
        ----------
        rows, cols : int
        mask : list[list[bool]] | None
            Shape mask; None means full rectangle (all cells passable).
        """
        self.rows, self.cols = rows, cols
        self.grid = [[0] * cols for _ in range(rows)]
        if mask is None:
            self.mask = [[True] * cols for _ in range(rows)]
        else:
            self.mask = mask
        self.start = None
        self.end   = None

    def in_bounds(self, r, c):
        """Return True if (r, c) lies within the grid dimensions."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_passable(self, r, c):
        """Return True if (r, c) is inside the grid AND inside the shape mask."""
        return self.in_bounds(r, c) and self.mask[r][c]

    def carve(self, r, c, direction):
        """
        Remove the wall between (r, c) and its neighbour in *direction*.
        Both sides of the wall are updated to keep the grid consistent.
        """
        self.grid[r][c] |= direction
        nr, nc = r + DX[direction], c + DY[direction]
        if self.in_bounds(nr, nc):
            self.grid[nr][nc] |= OPPOSITE[direction]

    def open_walls(self, r, c):
        """Return the number of carved (open) walls for cell (r, c)."""
        return bin(self.grid[r][c]).count('1')

    def neighbours(self, r, c):
        """Yield (nr, nc, direction) for every carved passage leaving (r, c)."""
        for d in (N, E, S, W):
            if self.grid[r][c] & d:
                nr, nc = r + DX[d], c + DY[d]
                if self.is_passable(nr, nc):
                    yield (nr, nc, d)

    def to_ascii(self, max_rows=30, max_cols=30):
        """
        Return a plain-text ASCII drawing of (a portion of) the maze.
        Cells outside the mask are shown as 'XXX'.
        """
        limit_r = min(self.rows, max_rows)
        limit_c = min(self.cols, max_cols)
        out = ['+' + '---+' * limit_c]
        for r in range(limit_r):
            line = '|'
            for c in range(limit_c):
                if not self.mask[r][c]:
                    line += 'XXX'
                    line += '|'
                else:
                    line += '   '
                    line += ' ' if self.grid[r][c] & E else '|'
            out.append(line)
            line = '+'
            for c in range(limit_c):
                if not self.mask[r][c]:
                    line += '---+'
                else:
                    line += '   +' if self.grid[r][c] & S else '---+'
            out.append(line)
        return '\n'.join(out)

    def to_file(self, filename):
        """
        Save the maze to a plain-text file.

        Format
        ------
        Line 1  : "<rows> <cols>"
        Line 2  : "start=<r>,<c> end=<r>,<c>"  (-1,-1 when unset)
        Lines 3+: "<mask_row> | <grid_row>"  (space-separated integers)
        """
        sr, sc = self.start if self.start else (-1, -1)
        er, ec = self.end   if self.end   else (-1, -1)
        with open(filename, 'w') as f:
            f.write(f"{self.rows} {self.cols}\n")
            f.write(f"start={sr},{sc} end={er},{ec}\n")
            for r in range(self.rows):
                row_mask = [str(int(self.mask[r][c])) for c in range(self.cols)]
                row_grid = [str(self.grid[r][c]) for c in range(self.cols)]
                f.write(' '.join(row_mask) + ' | ' + ' '.join(row_grid) + '\n')

    @staticmethod
    def from_file(filename):
        """
        Load a maze saved with to_file().
        Supports both the new format (with start/end line) and the old format.
        """
        with open(filename) as f:
            rows, cols = map(int, f.readline().split())
            start = end = None
            line2 = f.readline().strip()
            if line2.startswith('start='):
                parts2 = line2.split()
                sr, sc = map(int, parts2[0].split('=')[1].split(','))
                er, ec = map(int, parts2[1].split('=')[1].split(','))
                start  = (sr, sc) if sr >= 0 else None
                end    = (er, ec) if er >= 0 else None
                data_lines = [f.readline() for _ in range(rows)]
            else:
                data_lines = [line2] + [f.readline() for _ in range(rows - 1)]

            mask = [[True]  * cols for _ in range(rows)]
            grid = [[0]     * cols for _ in range(rows)]
            for r, line in enumerate(data_lines):
                parts = line.strip().split(' | ')
                if len(parts) == 2:
                    mask_vals = list(map(int, parts[0].split()))
                    grid_vals = list(map(int, parts[1].split()))
                else:
                    grid_vals = list(map(int, parts[0].split()))
                    mask_vals = [1] * cols
                mask[r] = [bool(x) for x in mask_vals]
                grid[r] = grid_vals

            maze       = Maze(rows, cols, mask)
            maze.grid  = grid
            maze.start = start
            maze.end   = end
        return maze


# ================================================================
#  START / END ASSIGNMENT
# ================================================================

def assign_start_end(maze):
    """
    Set exactly one entrance (start) and one exit (end) on the maze.

    Strategy
    --------
    Collect all boundary cells — passable cells that touch at least one
    non-passable or out-of-bounds neighbour.  Sort them and pick:
        start → topmost-then-leftmost   (rendered green)
        end   → bottommost-then-rightmost (rendered red)

    Placing the two endpoints as far apart as possible makes pathfinding
    results more interesting to compare.
    """
    boundary = []
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not maze.mask[r][c]:
                continue
            for d in (N, E, S, W):
                nr, nc = r + DX[d], c + DY[d]
                if not maze.in_bounds(nr, nc) or not maze.mask[nr][nc]:
                    boundary.append((r, c))
                    break
    if not boundary:
        return
    boundary_sorted = sorted(boundary, key=lambda x: (x[0], x[1]))
    maze.start = boundary_sorted[0]
    maze.end   = boundary_sorted[-1]


# ================================================================
#  SHAPE GENERATORS
# ================================================================

def shape_circle(rows, cols):
    """
    Filled-circle mask.
    Each cell whose centre lies within distance rows/2 of the grid centre
    is included.
    """
    cx, cy = (cols - 1) / 2, (rows - 1) / 2
    radius = min(rows, cols) // 2
    mask = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if (r - cy) ** 2 + (c - cx) ** 2 <= radius ** 2:
                mask[r][c] = True
    return mask


def shape_diamond(rows, cols):
    """
    Filled-diamond mask using the L1 (taxicab) norm.
    Cell (r,c) is included when |r−cy| + |c−cx| ≤ min(rows,cols)//2.
    """
    cy, cx = (rows - 1) / 2, (cols - 1) / 2
    mask = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if abs(r - cy) + abs(c - cx) <= min(rows, cols) // 2:
                mask[r][c] = True
    return mask


def shape_triangle(rows, cols):
    """
    Right-triangle mask growing from the top-left corner.
    Each row r exposes (r+1)/rows * cols columns.
    """
    mask = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        max_c = int((r + 1) / rows * cols)
        for c in range(min(max_c, cols)):
            mask[r][c] = True
    return mask


def shape_cross(rows, cols):
    """
    Plus-sign / cross mask.
    The centre third of rows and the centre third of columns are passable.
    """
    mask = [[False] * cols for _ in range(rows)]
    third_r, third_c = rows // 3, cols // 3
    for r in range(rows):
        for c in range(cols):
            if (third_r <= r < 2 * third_r) or (third_c <= c < 2 * third_c):
                mask[r][c] = True
    return mask


def shape_star(rows, cols):
    """
    Five-pointed star mask.
    Uses a sector-based formula that linearly interpolates between the outer
    tip radius and the inner valley radius as the angle moves through each
    36° half-sector.
    """
    mask = [[False] * cols for _ in range(rows)]
    cy, cx = (rows - 1) / 2, (cols - 1) / 2
    outer  = min(rows, cols) // 2
    inner  = outer // 3
    for r in range(rows):
        for c in range(cols):
            dx, dy = c - cx, r - cy
            dist   = math.hypot(dx, dy)
            angle  = math.atan2(dy, dx)
            sector = (5 * angle / (2 * math.pi)) % 1
            if sector < 0.5:
                t   = sector * 2
                rad = inner + (outer - inner) * t
            else:
                t   = (sector - 0.5) * 2
                rad = outer - (outer - inner) * t
            if dist <= rad:
                mask[r][c] = True
    return mask


SHAPES = {
    "Rectangle": None,
    "Circle":    shape_circle,
    "Diamond":   shape_diamond,
    "Triangle":  shape_triangle,
    "Cross":     shape_cross,
    "Star":      shape_star,
}


# ================================================================
#  IMAGE EXPORT
# ================================================================

def maze_to_image(maze, cell_size=10, wall_width=2):
    """
    Render the maze as a PIL RGB image.

    Non-mask cells are filled black.  For each passable cell only the North
    and West walls are drawn (the South / East walls of adjacent cells cover
    the remaining sides), plus a solid bottom and right border.
    Start cell is marked with a green dot, end cell with a red dot.

    Parameters
    ----------
    maze      : Maze
    cell_size : int   Pixel side-length of each cell square.
    wall_width: int   Pixel thickness of wall lines.

    Returns
    -------
    PIL.Image.Image (RGB)
    """
    width  = maze.cols * cell_size + wall_width
    height = maze.rows * cell_size + wall_width
    img    = Image.new("RGB", (width, height), "white")
    draw   = ImageDraw.Draw(img)

    # Fill non-mask cells black
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not maze.mask[r][c]:
                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                draw.rectangle([x1, y1, x2, y2], fill="black")

    # Draw walls
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not maze.mask[r][c]:
                continue
            x1 = c * cell_size
            y1 = r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            if not (maze.grid[r][c] & N):
                draw.line([(x1, y1), (x2, y1)], fill="black", width=wall_width)
            if not (maze.grid[r][c] & W):
                draw.line([(x1, y1), (x1, y2)], fill="black", width=wall_width)

    # Bottom and right border
    draw.line([(0, height - 1), (width - 1, height - 1)], fill="black", width=wall_width)
    draw.line([(width - 1, 0), (width - 1, height - 1)], fill="black", width=wall_width)

    # Draw start (green) and end (red) dots
    dot_r = max(2, cell_size // 3)
    if maze.start:
        sr, sc = maze.start
        cx_s   = sc * cell_size + cell_size // 2
        cy_s   = sr * cell_size + cell_size // 2
        draw.ellipse([cx_s - dot_r, cy_s - dot_r,
                      cx_s + dot_r, cy_s + dot_r], fill="green")
    if maze.end:
        er, ec = maze.end
        cx_e   = ec * cell_size + cell_size // 2
        cy_e   = er * cell_size + cell_size // 2
        draw.ellipse([cx_e - dot_r, cy_e - dot_r,
                      cx_e + dot_r, cy_e + dot_r], fill="red")

    return img


def smart_cell_size(total_cells):
    """
    Choose a cell pixel size that keeps output images a sane resolution.
    Larger grids get smaller cells so files don't become enormous.
    """
    if   total_cells <= 100:    return 40
    elif total_cells <= 2500:   return 10
    elif total_cells <= 10000:  return 5
    elif total_cells <= 62500:  return 3
    else:                       return 2


OUTPUT_DIR = "mazes_for_pathfinding"
IMAGE_DIR  = os.path.join(OUTPUT_DIR, "images")
BEST_DIR   = os.path.join(OUTPUT_DIR, "best_mazes")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(BEST_DIR,  exist_ok=True)


def save_maze_and_image(maze, base_name, rows, cols):
    """
    Write the maze data file (.maze) and its PNG image to the output dirs.

    Parameters
    ----------
    maze      : Maze
    base_name : str   Filename stem without extension.
    rows,cols : int   Used to choose an appropriate cell pixel size.

    Returns
    -------
    (data_path, img_path) : tuple[str, str]
    """
    data_path = os.path.join(OUTPUT_DIR, f"{base_name}.maze")
    maze.to_file(data_path)
    cell_size = smart_cell_size(rows * cols)
    img       = maze_to_image(maze, cell_size=cell_size,
                               wall_width=max(1, cell_size // 4))
    img_path  = os.path.join(IMAGE_DIR, f"{base_name}.png")
    img.save(img_path)
    return data_path, img_path


# ================================================================
#  STRUCTURAL METRICS (NO BFS – purely from grid)
# ================================================================

def maze_metrics(maze):
    """
    Compute structural quality metrics from the carved grid bitmasks alone
    (no pathfinding / BFS required).

    Returns
    -------
    dead_end_pct : float
        Percentage of passable cells that are dead ends (exactly one open
        wall).  Higher → more winding, complex maze.
    avg_branch : float
        Average open-wall count among junction cells (≥ 2 open walls).
    """
    passable     = sum(r.count(True) for r in maze.mask)
    dead_ends    = 0
    branch_sum   = 0
    branch_cells = 0
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not maze.mask[r][c]:
                continue
            w = maze.open_walls(r, c)
            if w == 1:
                dead_ends += 1
            elif w >= 2:
                branch_sum   += w
                branch_cells += 1
    dead_pct   = (dead_ends / passable * 100) if passable else 0
    avg_branch = branch_sum / branch_cells     if branch_cells else 0
    return dead_pct, avg_branch


# ================================================================
#  THE FIVE MAZE GENERATION ALGORITHMS (mask-aware)
# ================================================================

def recursive_backtracker(rows, cols, seed=None, mask=None):
    """
    Recursive Backtracker (iterative DFS).

    Design technique : Depth-First Search with backtracking.  Uses an
    explicit stack to avoid Python's recursion limit on large grids.

    Complexity
    ----------
    Time  : O(N)   N = number of passable cells
    Space : O(N)   visited array + stack

    Characteristics
    ---------------
    Produces long winding corridors with relatively few dead ends.
    Very fast in practice due to good cache locality.
    """
    if seed is not None:
        random.seed(seed)
    maze              = Maze(rows, cols, mask)
    visited           = [[False] * cols for _ in range(rows)]
    start_candidates  = [(r, c) for r in range(rows) for c in range(cols)
                         if maze.mask[r][c]]
    if not start_candidates:
        return maze
    r, c              = random.choice(start_candidates)
    stack             = [(r, c)]
    visited[r][c]     = True
    while stack:
        r, c   = stack[-1]
        dirs   = [N, E, S, W]
        random.shuffle(dirs)
        carved = False
        for d in dirs:
            nr, nc = r + DX[d], c + DY[d]
            if maze.is_passable(nr, nc) and not visited[nr][nc]:
                visited[nr][nc] = True
                maze.carve(r, c, d)
                stack.append((nr, nc))
                carved = True
                break
        if not carved:
            stack.pop()
    assign_start_end(maze)
    return maze


def prim_randomized(rows, cols, seed=None, mask=None):
    """
    Randomised Prim's Algorithm.

    Design technique : Greedy algorithm with random frontier selection
    (minimum-spanning-tree approach adapted for maze generation).

    Complexity
    ----------
    Time  : O(N)  amortised (swap-pop random removal from frontier list)
    Space : O(N)  frontier wall list

    Characteristics
    ---------------
    Produces mazes with many short dead ends and a more uniform texture
    compared to the DFS approach.
    """
    if seed is not None:
        random.seed(seed)
    maze             = Maze(rows, cols, mask)
    visited          = [[False] * cols for _ in range(rows)]
    start_candidates = [(r, c) for r in range(rows) for c in range(cols)
                        if maze.mask[r][c]]
    if not start_candidates:
        return maze
    r, c         = random.choice(start_candidates)
    visited[r][c] = True
    walls        = [(r, c, d) for d in (N, E, S, W)
                    if maze.is_passable(r + DX[d], c + DY[d])]
    while walls:
        i          = random.randrange(len(walls))
        wr, wc, d  = walls[i]
        walls[i]   = walls[-1]
        walls.pop()
        nr, nc = wr + DX[d], wc + DY[d]
        if maze.is_passable(nr, nc) and not visited[nr][nc]:
            visited[nr][nc] = True
            maze.carve(wr, wc, d)
            for nd in (N, E, S, W):
                nnr, nnc = nr + DX[nd], nc + DY[nd]
                if maze.is_passable(nnr, nnc) and not visited[nnr][nnc]:
                    walls.append((nr, nc, nd))
    assign_start_end(maze)
    return maze


def kruskal_randomized(rows, cols, seed=None, mask=None):
    """
    Randomised Kruskal's Algorithm.

    Design technique : Minimum Spanning Tree — shuffle all internal edges
    then add each if it joins two different components (Union-Find / DSU).

    Complexity
    ----------
    Time  : O(N α(N))  path-compressed Union-Find — essentially linear
    Space : O(N)

    Characteristics
    ---------------
    Very uniform, unbiased texture.  Excellent for illustrating MST
    concepts in the project report.
    """
    if seed is not None:
        random.seed(seed)
    maze   = Maze(rows, cols, mask)
    n      = rows * cols
    parent = list(range(n))
    rank   = [0] * n

    def find(x):
        """Path-halving find for Union-Find."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x         = parent[x]
        return x

    def union(x, y):
        """
        Union by rank.
        Returns True if x and y were in different sets (edge should be carved).
        """
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx]  += 1
        return True

    edges = []
    for r in range(rows):
        for c in range(cols):
            if not maze.mask[r][c]:
                continue
            if r > 0 and maze.mask[r - 1][c]:
                edges.append((r, c, N))
            if c > 0 and maze.mask[r][c - 1]:
                edges.append((r, c, W))
    random.shuffle(edges)
    for r, c, d in edges:
        nr, nc = r + DX[d], c + DY[d]
        if union(r * cols + c, nr * cols + nc):
            maze.carve(r, c, d)
    assign_start_end(maze)
    return maze


def eller(rows, cols, seed=None, mask=None):
    """
    Eller's Algorithm.

    Design technique : Row-by-row greedy construction.  Each row is processed
    once; cells are assigned to disjoint sets and walls are selectively removed
    based on set membership.

    Complexity
    ----------
    Time  : O(N)  — single pass through each cell
    Space : O(cols) — only the current row's set info is kept in memory

    Characteristics
    ---------------
    Memory-optimal: can generate infinite-height mazes in O(cols) memory.
    Produces a slightly column-biased texture.
    """
    if seed is not None:
        random.seed(seed)
    if mask is None:
        mask = [[True] * cols for _ in range(rows)]
    maze    = Maze(rows, cols, mask)
    row_set = {}
    next_id = 0
    for c in range(cols):
        if mask[0][c]:
            row_set[c] = next_id
            next_id   += 1
    for r in range(rows):
        for c in range(cols - 1):
            if (mask[r][c] and mask[r][c + 1]
                    and row_set.get(c) != row_set.get(c + 1)
                    and random.random() < 0.5):
                old = row_set[c + 1]
                new = row_set[c]
                for k in row_set:
                    if row_set[k] == old:
                        row_set[k] = new
                maze.carve(r, c, E)
        if r == rows - 1:
            # Last row: merge all differing adjacent sets to ensure connectivity
            for c in range(cols - 1):
                if (mask[r][c] and mask[r][c + 1]
                        and row_set.get(c) != row_set.get(c + 1)):
                    old = row_set[c + 1]
                    new = row_set[c]
                    for k in row_set:
                        if row_set[k] == old:
                            row_set[k] = new
                    maze.carve(r, c, E)
            break
        # Vertical connections — at least one per set
        set_to_cols = {}
        for c, sid in row_set.items():
            set_to_cols.setdefault(sid, []).append(c)
        next_row_set = {}
        for sid, columns in set_to_cols.items():
            valid         = [c for c in columns if r + 1 < rows and mask[r + 1][c]]
            if not valid:
                continue
            connect_count = random.randint(1, len(valid))
            connect_cols  = random.sample(valid, connect_count)
            for c in connect_cols:
                maze.carve(r, c, S)
                next_row_set[c] = sid
        for c in range(cols):
            if mask[r + 1][c] and c not in next_row_set:
                next_row_set[c] = next_id
                next_id        += 1
        row_set = next_row_set
    assign_start_end(maze)
    return maze


def wilson(rows, cols, seed=None, mask=None):
    """
    Wilson's Algorithm (Loop-Erased Random Walk).

    Design technique : Uniform spanning tree — every possible perfect maze
    is equally likely.

    Complexity
    ----------
    Time  : O(N²) worst case; O(N log N) expected on square grids
    Space : O(N)

    Characteristics
    ---------------
    Provably unbiased.  Can be slow on large grids — skipped for sizes
    ≥ 250×250 in the benchmark to avoid hangs.
    """
    if seed is not None:
        random.seed(seed)
    maze     = Maze(rows, cols, mask)
    in_maze  = [[False] * cols for _ in range(rows)]
    passable = [(r, c) for r in range(rows) for c in range(cols)
                if maze.mask[r][c]]
    if not passable:
        return maze
    sr, sc          = random.choice(passable)
    in_maze[sr][sc] = True
    dirs            = [N, E, S, W]
    remaining       = [(r, c) for r, c in passable if not in_maze[r][c]]
    while remaining:
        r, c = random.choice(remaining)
        path = [(r, c)]
        while not in_maze[r][c]:
            d      = random.choice(dirs)
            nr, nc = r + DX[d], c + DY[d]
            if not maze.is_passable(nr, nc):
                continue
            if (nr, nc) in path:
                path = path[:path.index((nr, nc)) + 1]
            else:
                path.append((nr, nc))
            r, c = nr, nc
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            if   r2 == r1 - 1: d = N
            elif r2 == r1 + 1: d = S
            elif c2 == c1 + 1: d = E
            else:               d = W
            maze.carve(r1, c1, d)
            in_maze[r1][c1] = True
        in_maze[path[-1][0]][path[-1][1]] = True
        remaining = [(r, c) for r, c in passable if not in_maze[r][c]]
    assign_start_end(maze)
    return maze


# ================================================================
#  BENCHMARKING & FILE OUTPUT
# ================================================================

ALGORITHMS = {
    "Backtracker": recursive_backtracker,
    "Prim":        prim_randomized,
    "Kruskal":     kruskal_randomized,
    "Eller":       eller,
    "Wilson":      wilson,
}

CSV_PATH  = os.path.join(OUTPUT_DIR, "benchmark_times.csv")
_csv_rows = []


def _log_csv(use_case, algorithm, shape, rows, cols, elapsed_sec, notes=""):
    """
    Append one timing record to the in-memory CSV buffer.

    Parameters
    ----------
    use_case    : str   Label for the benchmark use case.
    algorithm   : str   Algorithm name.
    shape       : str   Shape name.
    rows, cols  : int   Grid dimensions.
    elapsed_sec : float Wall-clock generation time in seconds.
    notes       : str   Optional extra info (aspect-ratio label, peak MB, …).
    """
    _csv_rows.append({
        "use_case":  use_case,
        "algorithm": algorithm,
        "shape":     shape,
        "rows":      rows,
        "cols":      cols,
        "time_sec":  f"{elapsed_sec:.6f}",
        "notes":     notes,
    })


def _flush_csv():
    """Write all buffered CSV rows to benchmark_times.csv on disk."""
    fieldnames = ["use_case", "algorithm", "shape", "rows", "cols",
                  "time_sec", "notes"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(_csv_rows)
    print(f"\n[CSV] Timing data saved → {CSV_PATH}  ({len(_csv_rows)} rows)")


def time_and_memory(gen_func, rows, cols, collect_mem=False, mask=None):
    """
    Run a generation function and measure wall-clock time (and optionally
    peak heap usage via tracemalloc).

    Parameters
    ----------
    gen_func    : callable
    rows, cols  : int
    collect_mem : bool   If True, profile peak memory allocation.
    mask        : list[list[bool]] | None

    Returns
    -------
    (maze, elapsed_sec, peak_mb)   peak_mb is None when collect_mem=False.
    """
    if collect_mem:
        tracemalloc.clear_traces()
        tracemalloc.start()
        t0      = time.perf_counter()
        maze    = gen_func(rows, cols, mask=mask) if mask is not None \
                  else gen_func(rows, cols)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return maze, elapsed, peak / (1024 * 1024)
    else:
        t0      = time.perf_counter()
        maze    = gen_func(rows, cols, mask=mask) if mask is not None \
                  else gen_func(rows, cols)
        elapsed = time.perf_counter() - t0
        return maze, elapsed, None


def fmt_time(sec):
    """Format a duration in seconds as a compact human-readable string."""
    if sec < 1e-3: return f"{sec * 1e6:.1f}µs"
    if sec < 1:    return f"{sec * 1e3:.1f}ms"
    return f"{sec:.3f}s"


# ================================================================
#  BEST MAZE SELECTION
# ================================================================

def select_and_save_best_mazes(shape_timing):
    """
    From the 100×100 shape-variation runs pick the fastest algorithm per shape
    and save those mazes to best_mazes/.

    For each winning maze writes:
        BEST_<shape>_100x100_<algo>.maze  — raw data for pathfinding
        BEST_<shape>_100x100_<algo>.png   — rendered image
    Also writes best_mazes_summary.csv and best_mazes_summary.txt.

    Parameters
    ----------
    shape_timing : dict
        { shape_name : { algo_name : (maze, elapsed_sec) } }
        Built during Use Case 5 in main().
    """
    summary_rows = []
    print("\n" + "=" * 80)
    print(" BEST MAZE SELECTION — 100×100, one per shape (fastest algorithm wins)")
    print("=" * 80)
    print(f"{'Shape':<14} {'Best Algorithm':<16} {'Time':>10}  Saved to")
    print("-" * 72)

    for shape_name in SHAPES:
        timings = shape_timing.get(shape_name, {})
        if not timings:
            continue
        best_alg          = min(timings, key=lambda a: timings[a][1])
        best_maze, best_t = timings[best_alg]

        base      = f"BEST_{shape_name}_100x100_{best_alg}"
        maze_path = os.path.join(BEST_DIR, f"{base}.maze")
        img_path  = os.path.join(BEST_DIR, f"{base}.png")
        best_maze.to_file(maze_path)

        cs  = smart_cell_size(100 * 100)
        img = maze_to_image(best_maze, cell_size=cs, wall_width=max(1, cs // 4))
        img.save(img_path)

        print(f"{shape_name:<14} {best_alg:<16} {fmt_time(best_t):>10}  {maze_path}")
        summary_rows.append({
            "shape":          shape_name,
            "best_algorithm": best_alg,
            "time_sec":       f"{best_t:.6f}",
            "maze_file":      maze_path,
            "image_file":     img_path,
        })

    # Machine-readable CSV summary
    summary_csv = os.path.join(BEST_DIR, "best_mazes_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shape", "best_algorithm",
                                           "time_sec", "maze_file", "image_file"])
        w.writeheader()
        w.writerows(summary_rows)

    # Human-readable TXT summary
    summary_txt = os.path.join(BEST_DIR, "best_mazes_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("BEST MAZES SUMMARY — 100×100 grid, one maze per shape\n")
        f.write("Selection criterion : lowest wall-clock generation time\n")
        f.write("=" * 70 + "\n")
        for row in summary_rows:
            f.write(f"Shape     : {row['shape']}\n")
            f.write(f"Algorithm : {row['best_algorithm']}\n")
            f.write(f"Time      : {row['time_sec']} s\n")
            f.write(f"Maze file : {row['maze_file']}\n")
            f.write(f"Image     : {row['image_file']}\n")
            f.write("-" * 70 + "\n")

    print(f"\n[BEST] Summary (CSV) → {summary_csv}")
    print(f"[BEST] Summary (TXT) → {summary_txt}")


# ================================================================
#  MAIN TEST SUITE
# ================================================================

def main(show_viewer=True):
    """
    Run the full benchmark suite and produce all output files.

    Use Cases
    ---------
    UC1 — Size Scaling      : ALL shapes, sizes 10×10 → 500×500
                              .maze and .png saved for every combination
    UC2 — Aspect Ratio      : Square / Wide / Tall / Very-Thin rectangles
    UC3 — Structure Metrics : Dead-end % and average branch factor, 100×100
    UC4 — Memory Efficiency : Peak heap allocation, 100×100
    UC5 — Shape Variation   : All shapes at 100×100 (also drives best-maze
                              selection)

    After all use cases:
    * Flush timing data to benchmark_times.csv
    * Select and save best maze per shape to best_mazes/
    * Print ASCII sample maze
    * (Optionally) launch interactive ASCII viewer
    """
    SKIP_WILSON_LARGE = True
    viewable_sizes    = [(10, 10), (25, 25), (50, 50)]
    maze_collection   = {alg: {} for alg in ALGORITHMS}
    shape_timing      = {shape: {} for shape in SHAPES}

    print("\n" + "=" * 80)
    print("       MAZE GENERATION BENCHMARK")
    print("=" * 80)

    # ------------------------------------------------------------------
    # USE CASE 1: SIZE SCALING — ALL SHAPES
    # ------------------------------------------------------------------
    # [CHANGED] Previously only ran Rectangle at each size.
    # Now iterates over every shape at every size so that .maze files and
    # .png images are generated for all shape × size combinations and
    # their times are all recorded in the CSV.
    # The printed table shows one shape at a time for readability.
    # ------------------------------------------------------------------
    print("\n--- USE CASE 1: SIZE SCALING (all shapes, 10×10 → 500×500) ---")
    sizes = [(10, 10), (25, 25), (50, 50), (100, 100), (250, 250), (500, 500)]

    for shape_name, shape_func in SHAPES.items():
        # Print a sub-header for each shape
        print(f"\n  Shape: {shape_name}")
        header = f"  {'Algorithm':<15}" + "".join(f"{f'{r}x{c}':>14}" for r, c in sizes)
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for alg, gen in ALGORITHMS.items():
            times = []
            for r, c in sizes:
                # Skip Wilson on large grids regardless of shape
                if SKIP_WILSON_LARGE and alg == "Wilson" and r * c >= 250 * 250:
                    times.append("—")
                    continue

                # Build the shape mask for this size
                mask = shape_func(r, c) if shape_func is not None else None

                maze, elapsed, _ = time_and_memory(gen, r, c, mask=mask)
                times.append(fmt_time(elapsed))

                # Log to CSV
                _log_csv("UC1_SizeScaling", alg, shape_name, r, c, elapsed)

                # Save .maze + .png
                base = f"{alg}_{shape_name}_{r}x{c}"
                save_maze_and_image(maze, base, r, c)

                # Keep small mazes for the interactive viewer
                if (r, c) in viewable_sizes:
                    maze_collection[alg][(r, c)] = maze

            print(f"  {alg:<15}" + "".join(f"{t:>14}" for t in times))

    # ------------------------------------------------------------------
    # USE CASE 2: ASPECT RATIO VARIATION  (Rectangle only — unchanged)
    # ------------------------------------------------------------------
    print("\n--- USE CASE 2: ASPECT RATIO VARIATION ---")
    aspects = [
        ("Square",    100,  100),
        ("Wide",       50,  200),
        ("Tall",      200,   50),
        ("Very Thin",  10, 1000),
    ]
    header = f"{'Algorithm':<15}" + "".join(f"{name:>14}" for name, _, _ in aspects)
    print(header)
    print('-' * len(header))

    for alg, gen in ALGORITHMS.items():
        values = []
        for name, r, c in aspects:
            if SKIP_WILSON_LARGE and alg == "Wilson" and r * c > 50000:
                values.append("—")
                continue
            maze, elapsed, _ = time_and_memory(gen, r, c)
            values.append(fmt_time(elapsed))
            _log_csv("UC2_AspectRatio", alg, "Rectangle", r, c, elapsed,
                     notes=name)
            base = f"{alg}_{name}_{r}x{c}"
            save_maze_and_image(maze, base, r, c)
            if (r, c) in viewable_sizes:
                maze_collection[alg][(r, c)] = maze
        print(f"{alg:<15}" + "".join(f"{v:>14}" for v in values))

    # ------------------------------------------------------------------
    # USE CASE 3: STRUCTURE METRICS (100×100) — NO BFS  (unchanged)
    # ------------------------------------------------------------------
    print("\n--- USE CASE 3: STRUCTURE METRICS ON 100×100 MAZES (no pathfinding) ---")
    print(f"{'Algorithm':<15} {'Dead-End %':>11} {'Avg Branch':>11}")
    print('-' * 40)
    for alg, gen in ALGORITHMS.items():
        maze, elapsed, _ = time_and_memory(gen, 100, 100)
        dead_pct, avg_branch = maze_metrics(maze)
        _log_csv("UC3_StructureMetrics", alg, "Rectangle", 100, 100, elapsed)
        print(f"{alg:<15} {dead_pct:>10.1f}% {avg_branch:>11.2f}")

    # ------------------------------------------------------------------
    # USE CASE 4: MEMORY EFFICIENCY (100×100)  (unchanged)
    # ------------------------------------------------------------------
    print("\n--- USE CASE 4: MEMORY EFFICIENCY (100×100) ---")
    print(f"{'Algorithm':<15} {'Peak Memory (MB)':>20}")
    print('-' * 37)
    for alg, gen in ALGORITHMS.items():
        maze, elapsed, peak_mb = time_and_memory(gen, 100, 100, collect_mem=True)
        _log_csv("UC4_Memory", alg, "Rectangle", 100, 100, elapsed,
                 notes=f"peak_mem_mb={peak_mb:.2f}")
        print(f"{alg:<15} {peak_mb:>20.2f} MB")

    # ------------------------------------------------------------------
    # USE CASE 5: SHAPE VARIATION (with time and passable %)  (unchanged)
    # ------------------------------------------------------------------
    print("\n--- USE CASE 5: SHAPE VARIATION (100×100 bound, various shapes) ---")
    shape_test_size = (100, 100)
    total_cells     = shape_test_size[0] * shape_test_size[1]
    header = (f"{'Algorithm':<15} {'Shape':<12} {'Passable':>10} {'Pass%':>8} "
              f"{'Time':>12} {'Dead%':>8} {'Branch':>8}")
    print(header)
    print('-' * 80)

    for shape_name, shape_func in SHAPES.items():
        mask = shape_func(*shape_test_size) if shape_func is not None else None
        for alg, gen in ALGORITHMS.items():
            maze, elapsed, _ = time_and_memory(
                gen, shape_test_size[0], shape_test_size[1], mask=mask)
            passable  = sum(r.count(True) for r in maze.mask)
            pass_pct  = (passable / total_cells) * 100
            dead_pct, avg_branch = maze_metrics(maze)
            _log_csv("UC5_ShapeVariation", alg, shape_name,
                     *shape_test_size, elapsed)
            base = f"{alg}_{shape_name}_{shape_test_size[0]}x{shape_test_size[1]}"
            save_maze_and_image(maze, base, shape_test_size[0], shape_test_size[1])
            print(f"{alg:<15} {shape_name:<12} {passable:>10} {pass_pct:>7.1f}% "
                  f"{fmt_time(elapsed):>12} {dead_pct:>7.1f}% {avg_branch:>8.2f}")
            shape_timing[shape_name][alg] = (maze, elapsed)

    # ------------------------------------------------------------------
    # FLUSH CSV  +  SELECT BEST MAZES
    # ------------------------------------------------------------------
    _flush_csv()
    select_and_save_best_mazes(shape_timing)

    # ------------------------------------------------------------------
    # ASCII VISUAL SAMPLE
    # ------------------------------------------------------------------
    print("\n--- SAMPLE VISUAL: 10×10 Backtracker ---")
    sample = recursive_backtracker(10, 10, seed=42)
    print(sample.to_ascii())

    # ------------------------------------------------------------------
    # INTERACTIVE MAZE VIEWER (loads any saved .maze file)
    # ------------------------------------------------------------------
    if show_viewer:
        available_views = {}
        all_sizes       = set()
        for maze_file in Path(OUTPUT_DIR).glob("*.maze"):
            stem      = maze_file.stem
            parts     = stem.split('_')
            alg       = parts[0]
            size_part = None
            for p in parts:
                if 'x' in p:
                    size_part = p
                    break
            if size_part is None:
                continue
            try:
                r_str, c_str = size_part.split('x')
                r, c         = int(r_str), int(c_str)
            except:
                continue
            available_views[(alg, (r, c), stem)] = str(maze_file)
            all_sizes.add((r, c))

        sorted_sizes = sorted(all_sizes, key=lambda x: x[0] * x[1])
        size_list    = ", ".join(f"{r}x{c}" for r, c in sorted_sizes)

        print("\n" + "=" * 80)
        print("INTERACTIVE MAZE VIEWER (ASCII)")
        print("Available algorithms: " + ", ".join(ALGORITHMS.keys()))
        print("Available sizes: "      + size_list)
        print("Commands: view <algorithm> <size>  |  "
              "view <algorithm> <shape> <size>  |  view all  |  exit")
        print("=" * 80)

        while True:
            cmd = input("\n> ").strip().lower()
            if cmd == 'exit':
                break
            if cmd == 'view all':
                for size in [(10, 10), (25, 25)]:
                    for alg in ALGORITHMS:
                        for key, path in available_views.items():
                            a, (r, c), name = key
                            if a == alg and (r, c) == size:
                                maze = Maze.from_file(path)
                                print(f"\n--- {alg} {r}x{c} ---")
                                print(maze.to_ascii())
                                break
                continue

            parts = cmd.split()
            if len(parts) >= 3 and parts[0] == 'view':
                alg = parts[1].capitalize()
                if alg not in ALGORITHMS:
                    print(f"Unknown algorithm '{alg}'.")
                    continue
                if len(parts) >= 4 and parts[2] in SHAPES:
                    shape_name = parts[2]
                    size_str   = parts[3]
                else:
                    shape_name = "Rectangle"
                    size_str   = parts[2]
                try:
                    r, c = map(int, size_str.split('x'))
                except:
                    print("Invalid size format. Use e.g. 100x100")
                    continue
                found = None
                for key, path in available_views.items():
                    a, (rr, cc), name = key
                    if a == alg and (rr, cc) == (r, c) and shape_name in name:
                        found = path
                        break
                if found:
                    maze = Maze.from_file(found)
                    print(f"\n--- {alg} {shape_name} {r}x{c} ---")
                    print(maze.to_ascii())
                else:
                    if shape_name == "Rectangle" and r * c <= 2500:
                        print(f"Generating {alg} {r}x{c} on the fly...")
                        maze, _, _ = time_and_memory(ALGORITHMS[alg], r, c)
                        print(maze.to_ascii())
                    else:
                        print("Maze not found.")
            else:
                print("Unknown command.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-view", action="store_true",
                        help="Skip interactive viewer")
    args = parser.parse_args()
    main(show_viewer=not args.no_view)