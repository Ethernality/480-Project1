#!/usr/bin/env python3
"""
make_vacuum_world.py
Generates a random Vacuum World grid file for CS 480 - Assignment 1.

Usage:
    python3 make_vacuum_world.py <rows> <cols> <blocked_fraction> <num_dirty>

Example:
    python3 make_vacuum_world.py 5 7 0.15 3 > random-5x7.txt

Parameters:
- rows (int): Number of rows in the grid
- cols (int): Number of columns in the grid
- blocked_fraction (float): Probability that any given cell is blocked
- num_dirty (int): Number of dirty cells to place

Outputs to stdout:
1. Number of columns
2. Number of rows
3. Rows of characters (one row per line)
   - `_` for empty
   - `#` for blocked
   - `*` for dirty
   - `@` for robot start (placed randomly in a non-blocked, non-dirty cell)
"""
import sys
import random

def main():
    if len(sys.argv) != 5:
        print("Usage: python3 make_vacuum_world.py <rows> <cols> <blocked_fraction> <num_dirty>")
        sys.exit(1)

    # Parse arguments
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
    blocked_fraction = float(sys.argv[3])
    num_dirty = int(sys.argv[4])

    # Initialize an empty grid with '_'
    grid = [['_' for _ in range(cols)] for _ in range(rows)]

    # Randomly place blocked cells based on probability
    for r in range(rows):
        for c in range(cols):
            if random.random() < blocked_fraction:
                grid[r][c] = '#'

    # Collect valid positions (non-blocked) for potential dirty cells and start location
    valid_positions = [
        (r, c) for r in range(rows) for c in range(cols)
        if grid[r][c] == '_'
    ]

    # Place dirty cells at random valid positions (if possible)
    # Shuffle the list of valid positions to pick them randomly
    random.shuffle(valid_positions)
    dirty_count = min(num_dirty, len(valid_positions))
    for i in range(dirty_count):
        r, c = valid_positions[i]
        grid[r][c] = '*'

    # Update the valid positions after dirty placement (can't overwrite dirty)
    valid_positions = [
        (r, c) for r, c in valid_positions[dirty_count:]
        if grid[r][c] == '_'
    ]

    # Place the robot start in one remaining valid position, if any remain
    if len(valid_positions) > 0:
        r_start, c_start = random.choice(valid_positions)
        grid[r_start][c_start] = '@'

    # Print number of columns and rows
    print(cols)
    print(rows)

    # Print the rows of the grid from top to bottom
    for r in range(rows):
        print("".join(grid[r]))

if __name__ == "__main__":
    main()
