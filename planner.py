
import sys
import heapq
import itertools

MOVE_DIRS = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
}


def parse_world(path):
    # with open(path) as f:
    #     lines = [ln.rstrip('\n') for ln in f if ln.strip() != '']
    # if len(lines) < 2:
    #     raise ValueError("World file too short.")
    # cols = int(lines[0])
    # rows = int(lines[1])
    # encoding='utf-8-sig' will strip a UTF-8 BOM *if present*,
    # but it won't fix UTF-16. So we try a fallback.
    try:
        with open(path, encoding='utf-8-sig') as f:
            raw = f.read()
    except UnicodeError:
        # Fallback: maybe UTF-16
        with open(path, encoding='utf-16') as f:
            raw = f.read()

    # Normalize line endings and strip trailing spaces
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() != '']
    if len(lines) < 2:
        raise ValueError("World file too short.")
    # In case there are hidden non-digits (safety)
    import re
    cols = int(re.sub(r'\D', '', lines[0]))
    rows = int(re.sub(r'\D', '', lines[1]))

    grid_lines = lines[2:2 + rows]
    if len(grid_lines) != rows:
        raise ValueError("Mismatch in declared rows vs provided rows.")
    start = None
    dirty = set()
    blocked = set()
    for r in range(rows):
        row_str = grid_lines[r]
        if len(row_str) != cols:
            raise ValueError(f"Row {r} length {len(row_str)} != cols {cols}")
        for c, ch in enumerate(row_str):
            if ch == '@':
                start = (r, c)
            elif ch == '*':
                dirty.add((r, c))
            elif ch == '#':
                blocked.add((r, c))
            elif ch == '_' or ch == '.':
                pass
            else:
                raise ValueError(f"Unknown cell char '{ch}' at ({r},{c})")
    if start is None:
        raise ValueError("No start '@' found.")
    return rows, cols, blocked, start, frozenset(dirty)


def successors(state, rows, cols, blocked):
    (r, c), dirty = state
    # Optionally vacuum first
    if (r, c) in dirty:
        new_dirty = frozenset(d for d in dirty if d != (r, c))
        yield 'V', ((r, c), new_dirty)
    for act, (dr, dc) in MOVE_DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in blocked:
            yield act, ((nr, nc), dirty)


def is_goal(state):
    return len(state[1]) == 0


def depth_first_search(rows, cols, blocked, start, dirty):
    start_state = (start, dirty)
    stack = [(start_state, [])]
    visited = set()
    nodes_generated = 1  # count start
    nodes_expanded = 0
    # To prioritize action order N,S,E,W,V in output, we will generate successors in that order
    ACTION_ORDER = ['V', 'N', 'S', 'E', 'W']
    while stack:
        state, path = stack.pop()
        key = (state[0], state[1])
        if key in visited:
            continue
        visited.add(key)
        nodes_expanded += 1
        if is_goal(state):
            return path, nodes_generated, nodes_expanded
        # gather successors in desired order
        succs = list(successors(state, rows, cols, blocked))
        # Reorder according to ACTION_ORDER
        succs.sort(key=lambda ap: ACTION_ORDER.index(ap[0]) if ap[0] in ACTION_ORDER else 999)
        for act, s2 in reversed(succs):  # reversed for stack LIFO to produce ACTION_ORDER
            nodes_generated += 1
            stack.append((s2, path + [act]))
    return None, nodes_generated, nodes_expanded  # no plan


def uniform_cost_search(rows, cols, blocked, start, dirty):
    start_state = (start, dirty)
    counter = itertools.count()
    frontier = []
    heapq.heappush(frontier, (0, next(counter), start_state, []))
    g_best = {start_state: 0}
    nodes_generated = 1
    nodes_expanded = 0

    while frontier:
        g, _, state, path = heapq.heappop(frontier)
        if g > g_best[state]:
            continue  # obsolete
        nodes_expanded += 1
        if is_goal(state):
            return path, nodes_generated, nodes_expanded
        for act, nxt in successors(state, rows, cols, blocked):
            g2 = g + 1
            prev = g_best.get(nxt)
            if prev is None or g2 < prev:
                g_best[nxt] = g2
                heapq.heappush(frontier, (g2, next(counter), nxt, path + [act]))
                nodes_generated += 1
    return None, nodes_generated, nodes_expanded


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 planner.py [uniform-cost|depth-first] world-file", file=sys.stderr)
        sys.exit(1)
    algo = sys.argv[1]
    world_file = sys.argv[2]
    rows, cols, blocked, start, dirty = parse_world(world_file)
    if algo == 'depth-first':
        plan, gen, exp = depth_first_search(rows, cols, blocked, start, dirty)
    elif algo == 'uniform-cost':
        plan, gen, exp = uniform_cost_search(rows, cols, blocked, start, dirty)
    else:
        print("Unknown algorithm. Use 'uniform-cost' or 'depth-first'.", file=sys.stderr)
        sys.exit(1)
    if plan is None:
        # (Not expected if a solution exists; spec does not define failure format.)
        pass
    else:
        for act in plan:
            print(act)
    print(f"{gen} nodes generated")
    print(f"{exp} nodes expanded")


if __name__ == "__main__":
    main()
