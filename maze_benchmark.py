import random, time, tracemalloc, os, argparse
from collections import deque
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
    def __init__(self, rows, cols, mask=None):
        self.rows, self.cols = rows, cols
        self.grid = [[0] * cols for _ in range(rows)]
        if mask is None:
            self.mask = [[True] * cols for _ in range(rows)]
        else:
            self.mask = mask

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_passable(self, r, c):
        return self.in_bounds(r, c) and self.mask[r][c]

    def carve(self, r, c, direction):
        self.grid[r][c] |= direction
        nr, nc = r + DX[direction], c + DY[direction]
        if self.in_bounds(nr, nc):
            self.grid[nr][nc] |= OPPOSITE[direction]

    def open_walls(self, r, c):
        return bin(self.grid[r][c]).count('1')

    def neighbours(self, r, c):
        for d in (N, E, S, W):
            if self.grid[r][c] & d:
                nr, nc = r + DX[d], c + DY[d]
                if self.is_passable(nr, nc):
                    yield (nr, nc, d)

    def to_ascii(self, max_rows=30, max_cols=30):
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
        with open(filename, 'w') as f:
            f.write(f"{self.rows} {self.cols}\n")
            for r in range(self.rows):
                row_mask = [str(int(self.mask[r][c])) for c in range(self.cols)]
                row_grid = [str(self.grid[r][c]) for c in range(self.cols)]
                f.write(' '.join(row_mask) + ' | ' + ' '.join(row_grid) + '\n')

    @staticmethod
    def from_file(filename):
        with open(filename) as f:
            rows, cols = map(int, f.readline().split())
            mask = [[True]*cols for _ in range(rows)]
            grid = [[0]*cols for _ in range(rows)]
            for r in range(rows):
                parts = f.readline().strip().split(' | ')
                if len(parts) == 2:
                    mask_vals = list(map(int, parts[0].split()))
                    grid_vals = list(map(int, parts[1].split()))
                else:
                    grid_vals = list(map(int, parts[0].split()))
                    mask_vals = [1]*cols
                mask[r] = [bool(x) for x in mask_vals]
                grid[r] = grid_vals
            maze = Maze(rows, cols, mask)
            maze.grid = grid
            return maze


# ================================================================
#  SHAPE GENERATORS
# ================================================================
def shape_circle(rows, cols):
    cx, cy = (cols-1)/2, (rows-1)/2
    radius = min(rows, cols) // 2
    mask = [[False]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if (r-cy)**2 + (c-cx)**2 <= radius**2:
                mask[r][c] = True
    return mask

def shape_diamond(rows, cols):
    cy, cx = (rows-1)/2, (cols-1)/2
    mask = [[False]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if abs(r-cy) + abs(c-cx) <= min(rows, cols)//2:
                mask[r][c] = True
    return mask

def shape_triangle(rows, cols):
    mask = [[False]*cols for _ in range(rows)]
    for r in range(rows):
        max_c = int((r+1) / rows * cols)
        for c in range(min(max_c, cols)):
            mask[r][c] = True
    return mask

def shape_cross(rows, cols):
    mask = [[False]*cols for _ in range(rows)]
    third_r, third_c = rows//3, cols//3
    for r in range(rows):
        for c in range(cols):
            if (third_r <= r < 2*third_r) or (third_c <= c < 2*third_c):
                mask[r][c] = True
    return mask

def shape_star(rows, cols):
    mask = [[False]*cols for _ in range(rows)]
    cy, cx = (rows-1)/2, (cols-1)/2
    outer = min(rows, cols)//2
    inner = outer // 3
    for r in range(rows):
        for c in range(cols):
            dx, dy = c-cx, r-cy
            dist = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            sector = (5 * angle / (2*math.pi)) % 1
            if sector < 0.5:
                t = sector * 2
                rad = inner + (outer-inner)*t
            else:
                t = (sector-0.5)*2
                rad = outer - (outer-inner)*t
            if dist <= rad:
                mask[r][c] = True
    return mask

SHAPES = {
    "Rectangle": None,
    "Circle": shape_circle,
    "Diamond": shape_diamond,
    "Triangle": shape_triangle,
    "Cross": shape_cross,
    "Star": shape_star
}


# ================================================================
#  IMAGE EXPORT
# ================================================================
def maze_to_image(maze, cell_size=10, wall_width=2):
    width = maze.cols * cell_size + wall_width
    height = maze.rows * cell_size + wall_width
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not maze.mask[r][c]:
                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                draw.rectangle([x1, y1, x2, y2], fill="black")
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
    draw.line([(0, height-1), (width-1, height-1)], fill="black", width=wall_width)
    draw.line([(width-1, 0), (width-1, height-1)], fill="black", width=wall_width)
    return img

def smart_cell_size(total_cells):
    if total_cells <= 100: return 40
    elif total_cells <= 2500: return 10
    elif total_cells <= 10000: return 5
    elif total_cells <= 62500: return 3
    else: return 2

OUTPUT_DIR = "mazes_for_pathfinding"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

def save_maze_and_image(maze, base_name, rows, cols):
    data_path = os.path.join(OUTPUT_DIR, f"{base_name}.maze")
    maze.to_file(data_path)
    cell_size = smart_cell_size(rows * cols)
    img = maze_to_image(maze, cell_size=cell_size, wall_width=max(1, cell_size//4))
    img_path = os.path.join(IMAGE_DIR, f"{base_name}.png")
    img.save(img_path)
    return data_path, img_path


# ================================================================
#  PATHFINDING HELPER (BFS) – returns -1 if no path
# ================================================================
def bfs_solution_length(maze, start=None, goal=None):
    if start is None:
        start = next((r,c) for r in range(maze.rows) for c in range(maze.cols) if maze.mask[r][c])
    if goal is None:
        goal = next((r,c) for r in range(maze.rows-1,-1,-1) for c in range(maze.cols-1,-1,-1) if maze.mask[r][c])
    sr, sc = start
    gr, gc = goal
    if (sr, sc) == (gr, gc):
        return 0
    visited = [[False] * maze.cols for _ in range(maze.rows)]
    q = deque([(sr, sc, 0)])
    visited[sr][sc] = True
    while q:
        r, c, dist = q.popleft()
        for nr, nc, _ in maze.neighbours(r, c):
            if (nr, nc) == (gr, gc):
                return dist + 1
            if not visited[nr][nc] and maze.mask[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc, dist + 1))
    return -1   # disconnected graph

def maze_metrics(maze):
    passable = sum(r.count(True) for r in maze.mask)
    dead_ends = 0
    branch_sum = 0
    branch_cells = 0
    for r in range(maze.rows):
        for c in range(maze.cols):
            if not maze.mask[r][c]:
                continue
            w = maze.open_walls(r, c)
            if w == 1:
                dead_ends += 1
            elif w >= 2:
                branch_sum += w
                branch_cells += 1
    dead_pct = (dead_ends / passable * 100) if passable else 0
    avg_branch = branch_sum / branch_cells if branch_cells else 0
    sol = bfs_solution_length(maze)
    return sol, dead_pct, avg_branch

def fmt_sol(sol):
    """Format solution length: number or '  N/A' (for width 10)."""
    if sol is not None and sol >= 0:
        return f"{sol:>10}"
    else:
        return "       N/A"


# ================================================================
#  THE FIVE MAZE GENERATION ALGORITHMS (mask‑aware)
# ================================================================
def recursive_backtracker(rows, cols, seed=None, mask=None):
    if seed is not None: random.seed(seed)
    maze = Maze(rows, cols, mask)
    visited = [[False] * cols for _ in range(rows)]
    start_candidates = [(r,c) for r in range(rows) for c in range(cols) if maze.mask[r][c]]
    if not start_candidates:
        return maze
    r, c = random.choice(start_candidates)
    stack = [(r, c)]
    visited[r][c] = True
    while stack:
        r, c = stack[-1]
        dirs = [N, E, S, W]
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
    return maze

def prim_randomized(rows, cols, seed=None, mask=None):
    if seed is not None: random.seed(seed)
    maze = Maze(rows, cols, mask)
    visited = [[False] * cols for _ in range(rows)]
    start_candidates = [(r,c) for r in range(rows) for c in range(cols) if maze.mask[r][c]]
    if not start_candidates:
        return maze
    r, c = random.choice(start_candidates)
    visited[r][c] = True
    walls = [(r, c, d) for d in (N, E, S, W) if maze.is_passable(r+DX[d], c+DY[d])]
    while walls:
        i = random.randrange(len(walls))
        wr, wc, d = walls[i]
        walls[i] = walls[-1]
        walls.pop()
        nr, nc = wr + DX[d], wc + DY[d]
        if maze.is_passable(nr, nc) and not visited[nr][nc]:
            visited[nr][nc] = True
            maze.carve(wr, wc, d)
            for nd in (N, E, S, W):
                nnr, nnc = nr + DX[nd], nc + DY[nd]
                if maze.is_passable(nnr, nnc) and not visited[nnr][nnc]:
                    walls.append((nr, nc, nd))
    return maze

def kruskal_randomized(rows, cols, seed=None, mask=None):
    if seed is not None: random.seed(seed)
    maze = Maze(rows, cols, mask)
    n = rows * cols
    parent = list(range(n))
    rank = [0] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry: return False
        if rank[rx] < rank[ry]: parent[rx] = ry
        elif rank[rx] > rank[ry]: parent[ry] = rx
        else: parent[ry] = rx; rank[rx] += 1
        return True
    edges = []
    for r in range(rows):
        for c in range(cols):
            if not maze.mask[r][c]: continue
            if r > 0 and maze.mask[r-1][c]: edges.append((r, c, N))
            if c > 0 and maze.mask[r][c-1]: edges.append((r, c, W))
    random.shuffle(edges)
    for r, c, d in edges:
        nr, nc = r + DX[d], c + DY[d]
        if union(r * cols + c, nr * cols + nc):
            maze.carve(r, c, d)
    return maze

def eller(rows, cols, seed=None, mask=None):
    if seed is not None: random.seed(seed)
    if mask is None:
        mask = [[True]*cols for _ in range(rows)]
    maze = Maze(rows, cols, mask)
    row_set = {}
    next_id = 0
    for c in range(cols):
        if mask[0][c]:
            row_set[c] = next_id
            next_id += 1
    for r in range(rows):
        # 1. Connect adjacent passable cells with 50% chance
        for c in range(cols-1):
            if mask[r][c] and mask[r][c+1] and row_set.get(c) != row_set.get(c+1) and random.random() < 0.5:
                old = row_set[c+1]
                new = row_set[c]
                for k in row_set:
                    if row_set[k] == old:
                        row_set[k] = new
                maze.carve(r, c, E)
        # 2. Last row: merge all distinct sets
        if r == rows-1:
            for c in range(cols-1):
                if mask[r][c] and mask[r][c+1] and row_set.get(c) != row_set.get(c+1):
                    old = row_set[c+1]
                    new = row_set[c]
                    for k in row_set:
                        if row_set[k] == old:
                            row_set[k] = new
                    maze.carve(r, c, E)
            break
        # 3. Vertical connections
        set_to_cols = {}
        for c, sid in row_set.items():
            set_to_cols.setdefault(sid, []).append(c)
        next_row_set = {}
        for sid, columns in set_to_cols.items():
            valid = [c for c in columns if r+1 < rows and mask[r+1][c]]
            if not valid:
                continue
            connect_count = random.randint(1, len(valid))
            connect_cols = random.sample(valid, connect_count)
            for c in connect_cols:
                maze.carve(r, c, S)
                next_row_set[c] = sid
        for c in range(cols):
            if mask[r+1][c] and c not in next_row_set:
                next_row_set[c] = next_id
                next_id += 1
        row_set = next_row_set
    return maze

def wilson(rows, cols, seed=None, mask=None):
    if seed is not None: random.seed(seed)
    maze = Maze(rows, cols, mask)
    in_maze = [[False] * cols for _ in range(rows)]
    passable = [(r,c) for r in range(rows) for c in range(cols) if maze.mask[r][c]]
    if not passable:
        return maze
    sr, sc = random.choice(passable)
    in_maze[sr][sc] = True
    dirs = [N, E, S, W]
    remaining = [(r,c) for r,c in passable if not in_maze[r][c]]
    while remaining:
        r, c = random.choice(remaining)
        path = [(r, c)]
        while not in_maze[r][c]:
            d = random.choice(dirs)
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
            r2, c2 = path[i+1]
            if r2 == r1 - 1: d = N
            elif r2 == r1 + 1: d = S
            elif c2 == c1 + 1: d = E
            else: d = W
            maze.carve(r1, c1, d)
            in_maze[r1][c1] = True
        in_maze[path[-1][0]][path[-1][1]] = True
        remaining = [(r,c) for r,c in passable if not in_maze[r][c]]
    return maze


# ================================================================
#  BENCHMARKING & FILE OUTPUT
# ================================================================
ALGORITHMS = {
    "Backtracker": recursive_backtracker,
    "Prim":        prim_randomized,
    "Kruskal":     kruskal_randomized,
    "Eller":       eller,
    "Wilson":      wilson
}

def time_and_memory(gen_func, rows, cols, collect_mem=False, mask=None):
    if collect_mem:
        tracemalloc.clear_traces()
        tracemalloc.start()
        t0 = time.perf_counter()
        if mask is not None:
            maze = gen_func(rows, cols, mask=mask)
        else:
            maze = gen_func(rows, cols)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return maze, elapsed, peak / (1024 * 1024)
    else:
        t0 = time.perf_counter()
        if mask is not None:
            maze = gen_func(rows, cols, mask=mask)
        else:
            maze = gen_func(rows, cols)
        elapsed = time.perf_counter() - t0
        return maze, elapsed, None

def fmt_time(sec):
    if sec < 1e-3: return f"{sec*1e6:.1f}µs"
    if sec < 1:    return f"{sec*1e3:.1f}ms"
    return f"{sec:.3f}s"


# ================================================================
#  MAIN TEST SUITE
# ================================================================
def main(show_viewer=True):
    viewable_sizes = [(10,10), (25,25), (50,50)]
    maze_collection = {alg: {} for alg in ALGORITHMS}

    print("\n" + "="*80)
    print("       MAZE GENERATION BENCHMARK — DAA PROJECT")
    print("="*80)

    # ------------------------------------------------------------------
    # USE CASE 1: SIZE SCALING
    # ------------------------------------------------------------------
    print("\n--- USE CASE 1: SIZE SCALING (10×10 → 500×500) ---")
    sizes = [(10,10), (25,25), (50,50), (100,100), (250,250), (500,500)]
    header = f"{'Algorithm':<15}" + "".join(f"{f'{r}x{c}':>14}" for r,c in sizes)
    print(header)
    print('-'*len(header))

    for alg, gen in ALGORITHMS.items():
        times = []
        for r,c in sizes:
            if alg in {"Kruskal", "Wilson"} and r*c >= 250*250:
                times.append("—")
                continue
            maze, elapsed, _ = time_and_memory(gen, r, c)
            times.append(fmt_time(elapsed))
            base = f"{alg}_{r}x{c}"
            save_maze_and_image(maze, base, r, c)
            if (r,c) in viewable_sizes:
                maze_collection[alg][(r,c)] = maze
        print(f"{alg:<15}" + "".join(f"{t:>14}" for t in times))

    # ------------------------------------------------------------------
    # USE CASE 2: ASPECT RATIO VARIATION
    # ------------------------------------------------------------------
    print("\n--- USE CASE 2: ASPECT RATIO VARIATION ---")
    aspects = [("Square", 100, 100), ("Wide", 50, 200), ("Tall", 200, 50), ("Very Thin", 10, 1000)]
    header = f"{'Algorithm':<15}" + "".join(f"{name:>14}" for name,_,_ in aspects)
    print(header)
    print('-'*len(header))
    for alg, gen in ALGORITHMS.items():
        values = []
        for name, r, c in aspects:
            if alg == "Wilson" and r*c > 50000:
                values.append("—")
                continue
            maze, elapsed, _ = time_and_memory(gen, r, c)
            values.append(fmt_time(elapsed))
            base = f"{alg}_{name}_{r}x{c}"
            save_maze_and_image(maze, base, r, c)
            if (r,c) in viewable_sizes:
                maze_collection[alg][(r,c)] = maze
        print(f"{alg:<15}" + "".join(f"{v:>14}" for v in values))

    # ------------------------------------------------------------------
    # USE CASE 3: STRUCTURE METRICS (100×100)
    # ------------------------------------------------------------------
    print("\n--- USE CASE 3: STRUCTURE METRICS ON 100×100 MAZES ---")
    print(f"{'Algorithm':<15} {'Sol. Length':>12} {'Dead-End %':>11} {'Avg Branch':>11}")
    print('-'*52)
    for alg, gen in ALGORITHMS.items():
        maze, _, _ = time_and_memory(gen, 100, 100)
        sol_len, dead_pct, avg_branch = maze_metrics(maze)
        sol_str = fmt_sol(sol_len).strip()
        print(f"{alg:<15} {sol_str:>12} {dead_pct:>10.1f}% {avg_branch:>11.2f}")

    # ------------------------------------------------------------------
    # USE CASE 4: MEMORY EFFICIENCY (100×100)
    # ------------------------------------------------------------------
    print("\n--- USE CASE 4: MEMORY EFFICIENCY (100×100) ---")
    print(f"{'Algorithm':<15} {'Peak Memory (MB)':>20}")
    print('-'*37)
    for alg, gen in ALGORITHMS.items():
        _, _, peak_mb = time_and_memory(gen, 100, 100, collect_mem=True)
        print(f"{alg:<15} {peak_mb:>20.2f} MB")

    # ------------------------------------------------------------------
    # USE CASE 5: SHAPE VARIATION
    # ------------------------------------------------------------------
    print("\n--- USE CASE 5: SHAPE VARIATION (100×100 bound, various shapes) ---")
    shape_test_size = (100, 100)
    print(f"Metrics on shapes (passable cells vary):")
    print(f"{'Algorithm':<15} {'Shape':<12} {'Passable':>8} {'Sol. Len':>10} {'Dead%':>8} {'Branch':>8}")
    print('-'*65)
    for shape_name, shape_func in SHAPES.items():
        if shape_func is None:
            mask = None
        else:
            mask = shape_func(*shape_test_size)
        for alg, gen in ALGORITHMS.items():
            maze, elapsed, _ = time_and_memory(gen, shape_test_size[0], shape_test_size[1], mask=mask)
            passable = sum(r.count(True) for r in maze.mask)
            sol_len, dead_pct, avg_branch = maze_metrics(maze)
            base = f"{alg}_{shape_name}_{shape_test_size[0]}x{shape_test_size[1]}"
            save_maze_and_image(maze, base, shape_test_size[0], shape_test_size[1])
            sol_display = fmt_sol(sol_len)
            print(f"{alg:<15} {shape_name:<12} {passable:>8} {sol_display} {dead_pct:>7.1f}% {avg_branch:>8.2f}")

    # ------------------------------------------------------------------
    # INTERACTIVE MAZE VIEWER (loads any saved .maze file)
    # ------------------------------------------------------------------
    if show_viewer:
        available_views = {}
        all_sizes = set()
        for maze_file in Path(OUTPUT_DIR).glob("*.maze"):
            stem = maze_file.stem
            parts = stem.split('_')
            alg = parts[0]
            size_part = None
            for p in parts:
                if 'x' in p:
                    size_part = p
                    break
            if size_part is None:
                continue
            try:
                r_str, c_str = size_part.split('x')
                r, c = int(r_str), int(c_str)
            except:
                continue
            available_views[(alg, (r,c), stem)] = str(maze_file)
            all_sizes.add((r,c))

        sorted_sizes = sorted(all_sizes, key=lambda x: x[0]*x[1])
        size_list = ", ".join(f"{r}x{c}" for r,c in sorted_sizes)

        print("\n" + "="*80)
        print("INTERACTIVE MAZE VIEWER")
        print("Available algorithms: " + ", ".join(ALGORITHMS.keys()))
        print("Available sizes: " + size_list)
        print("Also shape mazes saved (e.g., Backtracker_Circle_100x100.maze).")
        print("Commands:")
        print("  view <algorithm> <size>            e.g. view Backtracker 500x500")
        print("  view <algorithm> <shape> <size>    e.g. view Prim Circle 100x100")
        print("  view all                           (shows all 10x10 and 25x25)")
        print("  exit")
        print("="*80)

        while True:
            cmd = input("\n> ").strip().lower()
            if cmd == 'exit':
                break
            if cmd == 'view all':
                for size in [(10,10), (25,25)]:
                    for alg in ALGORITHMS:
                        for key, path in available_views.items():
                            a, (r,c), name = key
                            if a == alg and (r,c) == size:
                                maze = Maze.from_file(path)
                                print(f"\n--- {alg} {r}x{c} ---")
                                print(maze.to_ascii())
                                break
                continue

            parts = cmd.split()
            if len(parts) >= 3 and parts[0] == 'view':
                alg = parts[1].capitalize()
                if alg not in ALGORITHMS:
                    print(f"Unknown algorithm '{alg}'. Choose from: {', '.join(ALGORITHMS.keys())}")
                    continue
                if len(parts) >= 4 and parts[2] in SHAPES:
                    shape_name = parts[2]
                    size_str = parts[3]
                else:
                    shape_name = "Rectangle"
                    size_str = parts[2]
                try:
                    r, c = map(int, size_str.split('x'))
                except:
                    print("Invalid size format. Use e.g. 100x100")
                    continue

                found = None
                for key, path in available_views.items():
                    a, (rr, cc), name = key
                    if a == alg and (rr, cc) == (r,c) and shape_name in name:
                        found = path
                        break
                if found:
                    maze = Maze.from_file(found)
                    print(f"\n--- {alg} {shape_name} {r}x{c} ---")
                    print(maze.to_ascii())
                else:
                    if shape_name == "Rectangle" and r*c <= 2500:
                        print(f"Generating {alg} {r}x{c} on the fly...")
                        maze, _, _ = time_and_memory(ALGORITHMS[alg], r, c)
                        print(maze.to_ascii())
                    else:
                        print(f"Maze {alg} {shape_name} {size_str} not found. It may not have been generated.")
            else:
                print("Unknown command.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-view", action="store_true", help="Skip interactive viewer")
    args = parser.parse_args()
    main(show_viewer=not args.no_view)