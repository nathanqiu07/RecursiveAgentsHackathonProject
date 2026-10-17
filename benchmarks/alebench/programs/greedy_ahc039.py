"""Reference greedy for AHC039: the best axis-aligned rectangle on a 50x50 grid (2D max-subarray). Score ratios are relative to this."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    pts = list(map(int, data[1:1 + 4 * n]))
    g = 50
    cell = 100000 // g
    grid = [[0] * g for _ in range(g)]
    for i in range(2 * n):
        x, y = pts[2 * i], pts[2 * i + 1]
        cx, cy = min(x // cell, g - 1), min(y // cell, g - 1)
        grid[cx][cy] += 1 if i < n else -1
    best = (-10**9, 0, 0, 0, 0)
    for x0 in range(g):
        col = [0] * g
        for x1 in range(x0, g):
            row = grid[x1]
            for y in range(g):
                col[y] += row[y]
            run, start = 0, 0
            for y in range(g):
                if run <= 0:
                    run, start = col[y], y
                else:
                    run += col[y]
                if run > best[0]:
                    best = (run, x0, x1, start, y)
    _, x0, x1, y0, y1 = best
    ax, bx, ay, by = x0 * cell, (x1 + 1) * cell, y0 * cell, (y1 + 1) * cell
    print(4)
    for vx, vy in ((ax, ay), (bx, ay), (bx, by), (ax, by)):
        print(vx, vy)


main()
