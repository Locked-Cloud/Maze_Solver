import numpy as np
import heapq
import tkinter as tk
from tkinter import ttk
import random

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def dfs(maze, start, exits):
    stack = [start]
    visited = set()
    came_from = {}

    while stack:
        current = stack.pop()

        if current in exits:
            return reconstruct_path(came_from, start, current)

        if current in visited:
            continue

        visited.add(current)
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                next_pos = (nx, ny)
                if next_pos not in visited:
                    stack.append(next_pos)
                    came_from[next_pos] = current

    return None

def bfs(maze, start, exits):
    queue = [start]
    visited = set()
    came_from = {}

    while queue:
        current = queue.pop(0)

        if current in exits:
            return reconstruct_path(came_from, start, current)

        if current in visited:
            continue

        visited.add(current)
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                next_pos = (nx, ny)
                if next_pos not in visited:
                    queue.append(next_pos)
                    came_from[next_pos] = current

    return None

def greedy(maze, start, exits):
    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    came_from = {}
    visited = set()

    while priority_queue:
        _, current = heapq.heappop(priority_queue)

        if current in exits:
            return reconstruct_path(came_from, start, current)

        if current in visited:
            continue

        visited.add(current)
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                next_pos = (nx, ny)
                if next_pos not in visited:
                    priority = heuristic(next_pos, exits[0])
                    heapq.heappush(priority_queue, (priority, next_pos))
                    came_from[next_pos] = current

    return None

def a_star(maze, start, exits):
    def heuristic(pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while priority_queue:
        _, current = heapq.heappop(priority_queue)

        if current in exits:
            return reconstruct_path(came_from, start, current)

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                new_cost = cost_so_far[current] + 1
                next_pos = (nx, ny)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos, exits[0])
                    heapq.heappush(priority_queue, (priority, next_pos))
                    came_from[next_pos] = current

    return None

class MazeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Maze Pathfinding")
        self.maze_size = 20  # Default maze size
        self.min_cell_size = 5
        self.max_canvas_size = 800

        self.wall_density = 0.3

        self.master.resizable(True, True)

        self.create_widgets()
        self.reset_maze()

        self.master.bind('<Configure>', self.on_resize)

    def create_widgets(self):
        self.canvas = tk.Canvas(self.master, bg="white", bd=2, relief="solid")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.control_panel = ttk.Frame(self.master)
        self.control_panel.pack(pady=10, fill=tk.X)

        ttk.Button(self.control_panel, text="DFS", command=lambda: self.run_algorithm(dfs)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_panel, text="BFS", command=lambda: self.run_algorithm(bfs)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_panel, text="Greedy", command=lambda: self.run_algorithm(greedy)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_panel, text="A*", command=lambda: self.run_algorithm(a_star)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_panel, text="Generate Maze", command=self.reset_maze).pack(side=tk.LEFT, padx=5)

    def on_resize(self, event):
        self.update_cell_size()
        self.draw_maze()

    def update_cell_size(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        self.cell_size = max(self.min_cell_size,
                             min(canvas_width // self.maze_size, canvas_height // self.maze_size))

    def reset_maze(self):
        self.maze_size = 20
        self.maze = self.generate_maze()
        self.start = (0, 0)
        self.exits = [(self.maze_size - 1, self.maze_size - 1)]

        self.maze[self.start] = 0
        for exit_pos in self.exits:
            self.maze[exit_pos] = 0

        self.update_cell_size()
        self.draw_maze()

    def generate_maze(self):
        maze = np.random.choice([0, 1], size=(self.maze_size, self.maze_size), p=[1 - self.wall_density, self.wall_density])
        return maze

    def draw_maze(self):
        self.canvas.delete("all")

        for x in range(self.maze_size):
            for y in range(self.maze_size):
                color = "white" if self.maze[x, y] == 0 else "black"
                self.canvas.create_rectangle(
                    y * self.cell_size, x * self.cell_size,
                    (y + 1) * self.cell_size, (x + 1) * self.cell_size,
                    fill=color, outline="gray"
                )

        self.canvas.create_oval(
            self.start[1] * self.cell_size + self.cell_size * 0.2,
            self.start[0] * self.cell_size + self.cell_size * 0.2,
            (self.start[1] + 1) * self.cell_size - self.cell_size * 0.2,
            (self.start[0] + 1) * self.cell_size - self.cell_size * 0.2,
            fill="green", outline="black"
        )

        for exit_pos in self.exits:
            self.canvas.create_oval(
                exit_pos[1] * self.cell_size + self.cell_size * 0.2,
                exit_pos[0] * self.cell_size + self.cell_size * 0.2,
                (exit_pos[1] + 1) * self.cell_size - self.cell_size * 0.2,
                (exit_pos[0] + 1) * self.cell_size - self.cell_size * 0.2,
                fill="red", outline="black"
            )

    def run_algorithm(self, algorithm):
        self.canvas.delete("path")
        path = algorithm(self.maze, self.start, self.exits)
        if path:
            for i in range(1, len(path)):
                x1, y1 = path[i - 1]
                x2, y2 = path[i]
                self.canvas.create_line(
                    y1 * self.cell_size + self.cell_size // 2,
                    x1 * self.cell_size + self.cell_size // 2,
                    y2 * self.cell_size + self.cell_size // 2,
                    x2 * self.cell_size + self.cell_size // 2,
                    fill="blue", width=max(1, self.cell_size // 10),
                    tags="path"
                )

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGUI(root)
    root.geometry("800x600")
    root.mainloop()
