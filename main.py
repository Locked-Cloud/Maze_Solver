import numpy as np
import heapq
import tkinter as tk
from tkinter import ttk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

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
        self.dark_mode = False
        self.wall_density = 0.3
        self.num_exits = 3  # Add number of exits configuration
        self.changing_start = False  # Add flag for start position change mode
        
        # Add variables to store last path and algorithm
        self.last_path = None
        self.last_algorithm = None
        self.current_path = None  # Add this to track current path

        self.performance_stats = {
            'dfs': {'times': [], 'paths': []},
            'bfs': {'times': [], 'paths': []},
            'greedy': {'times': [], 'paths': []},
            'a_star': {'times': [], 'paths': []}
        }
        
        # Create the figure for the graph
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 4))
        self.fig.tight_layout(pad=3.0)

        self.master.resizable(True, True)
        
        # Setup styles first
        self.setup_styles()
        
        # Create widgets before resetting maze
        self.create_widgets()
        
        # Now reset maze
        self.reset_maze()  # Ensure this is called after create_widgets
        
        self.master.bind('<Configure>', self.on_resize)

    def setup_styles(self):
        """Setup custom styles for widgets"""
        style = ttk.Style()
        
        # Configure main styles
        style.configure('Custom.TFrame', background='#2c3e50')
        style.configure('Control.TFrame', background='#34495e', padding=10)
        
        # Button styles
        style.configure('Algorithm.TButton',
                       padding=5,
                       font=('Arial', 9, 'bold'),
                       background='#3498db')
        
        style.configure('Utility.TButton',
                       padding=5,
                       font=('Arial', 9),
                       background='#2ecc71')
        
        # Label styles
        style.configure('Custom.TLabel',
                       font=('Arial', 10),
                       background='#2c3e50',
                       foreground='white')
        
        # Scale style
        style.configure('Custom.Horizontal.TScale',
                       background='#2c3e50')

    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self.master, style='Custom.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame with maze and controls
        self.left_frame = ttk.Frame(main_container, style='Custom.TFrame')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Canvas frame
        canvas_frame = ttk.Frame(self.left_frame, style='Custom.TFrame')
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Maze canvas with border
        self.canvas = tk.Canvas(canvas_frame, 
                              bg="white",
                              bd=0,
                              highlightthickness=2,
                              highlightbackground="#3498db")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control panel with better organization
        self.control_panel = ttk.Frame(self.left_frame, style='Control.TFrame')
        self.control_panel.pack(fill=tk.X, pady=(10, 0))

        # Algorithm buttons frame
        algo_frame = ttk.Frame(self.control_panel, style='Control.TFrame')
        algo_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Algorithm buttons
        ttk.Label(algo_frame, text="Algorithms:", style='Custom.TLabel').pack(side=tk.LEFT, padx=5)
        for algo, name in [('DFS', dfs), ('BFS', bfs), ('Greedy', greedy), ('A*', a_star)]:
            ttk.Button(algo_frame, 
                      text=algo,
                      style='Algorithm.TButton',
                      command=lambda a=name: self.run_algorithm(a)).pack(side=tk.LEFT, padx=2)

        # Utility buttons frame
        util_frame = ttk.Frame(self.control_panel, style='Control.TFrame')
        util_frame.pack(fill=tk.X)
        
        # Utility buttons
        utility_buttons = [
            ("Change Start", self.toggle_start_change),
            ("Generate Maze", self.reset_maze),
            ("Toggle Dark Mode", self.toggle_dark_mode),
            ("Clear Stats", self.clear_stats),
            ("Reset Graphs", self.reset_graphs)
        ]
        
        for text, command in utility_buttons:
            ttk.Button(util_frame,
                      text=text,
                      style='Utility.TButton',
                      command=command).pack(side=tk.LEFT, padx=2)

        # Maze size control frame
        size_frame = ttk.Frame(self.control_panel, style='Control.TFrame')
        size_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(size_frame, text="Maze Size:", style='Custom.TLabel').pack(side=tk.LEFT, padx=5)
        self.size_slider = ttk.Scale(size_frame,
                                    from_=10,
                                    to=50,
                                    orient=tk.HORIZONTAL,
                                    style='Custom.Horizontal.TScale',
                                    command=self.update_maze_size)
        self.size_slider.set(self.maze_size)
        self.size_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Right frame with statistics
        self.right_frame = ttk.Frame(main_container, style='Custom.TFrame')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Stats frame
        self.stats_frame = ttk.Frame(self.right_frame, style='Custom.TFrame')
        self.stats_frame.pack(fill=tk.BOTH, expand=True)

        # Performance graphs
        self.canvas_stats = FigureCanvasTkAgg(self.fig, master=self.stats_frame)
        self.canvas_stats.draw()
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add canvas click binding
        self.canvas.bind('<Button-1>', self.on_canvas_click)

    def update_maze_size(self, value):
        self.maze_size = int(float(value))
        self.reset_maze()

    def on_resize(self, event):
        self.update_cell_size()
        self.draw_maze()

    def update_cell_size(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        self.cell_size = max(self.min_cell_size,
                             min(canvas_width // self.maze_size, canvas_height // self.maze_size))

    def reset_maze(self):
        self.maze = self.generate_maze()
        self.start = (0, 0)
        
        # Generate three exits at different positions
        self.exits = self.generate_exits()
        
        # Clear all paths
        self.last_path = None
        self.current_path = None
        self.last_algorithm = None

        # Ensure start and exit points are clear
        self.maze[self.start] = 0
        for exit_pos in self.exits:
            self.maze[exit_pos] = 0

        self.update_cell_size()
        self.draw_maze()
        
        # Clear the performance graphs
        self.update_performance_graphs()

    def generate_exits(self):
        """Generate three different exit positions"""
        exits = []
        possible_positions = [
            (self.maze_size-1, self.maze_size-1),  # Bottom-right corner
            (self.maze_size-1, 0),                  # Bottom-left corner
            (0, self.maze_size-1),                  # Top-right corner
            (self.maze_size-1, self.maze_size//2),  # Middle of bottom edge
            (self.maze_size//2, self.maze_size-1)   # Middle of right edge
        ]
        
        # Randomly select three unique positions
        exits = random.sample(possible_positions, self.num_exits)
        return exits

    def generate_maze(self):
        """Generate a maze with controlled solvability"""
        def is_solvable(maze):
            # Modified to check if at least one exit is reachable
            queue = [(0, 0)]
            visited = set()
            while queue:
                current = queue.pop(0)
                if current in self.exits:  # Check if current position is any of the exits
                    return True
                if current in visited:
                    continue
                visited.add(current)
                x, y = current
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.maze_size and 0 <= ny < self.maze_size and 
                        maze[nx, ny] == 0 and (nx, ny) not in visited):
                        queue.append((nx, ny))
            return False

        # Generate exits first
        self.exits = self.generate_exits()

        # 20% chance to generate an unsolvable maze
        generate_unsolvable = random.random() < 0.2

        while True:
            # Generate a base maze
            maze = np.random.choice([0, 1], size=(self.maze_size, self.maze_size), 
                                  p=[1 - self.wall_density, self.wall_density])
            
            # Always ensure start and all exit points are clear
            maze[0, 0] = 0
            for exit_pos in self.exits:
                maze[exit_pos] = 0

            # Check solvability
            solvable = is_solvable(maze)

            # Return maze if it matches our solvability requirement
            if generate_unsolvable and not solvable:
                print("Generated an unsolvable maze!")
                return maze
            elif not generate_unsolvable and solvable:
                print("Generated a solvable maze with multiple exits!")
                return maze

        return maze

    def draw_maze(self):
        self.canvas.delete("all")  # Don't delete path here
        
        # Update background colors based on dark mode
        bg_color = "#1e1e1e" if self.dark_mode else "white"
        wall_color = "#4a4a4a" if self.dark_mode else "black"
        path_color = "#007acc" if self.dark_mode else "blue"
        self.canvas.configure(bg=bg_color)

        for x in range(self.maze_size):
            for y in range(self.maze_size):
                color = bg_color if self.maze[x, y] == 0 else wall_color
                self.canvas.create_rectangle(
                    y * self.cell_size, x * self.cell_size,
                    (y + 1) * self.cell_size, (x + 1) * self.cell_size,
                    fill=color, outline="gray"
                )

        # Draw start and exit points
        self.draw_endpoints()
        
        # Redraw the current path if it exists
        if self.current_path:
            self.draw_path(self.current_path)

    def draw_endpoints(self):
        """Draw start and exit points"""
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
        # Clear any existing path
        self.canvas.delete("path")
        
        # Store the algorithm for later use
        self.last_algorithm = algorithm

        # Time the algorithm
        start_time = time.time()
        path = algorithm(self.maze, self.start, self.exits)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store both current and last path
        self.current_path = path
        self.last_path = path

        if path:
            # Calculate path length
            path_length = len(path) - 1
            
            # Update performance stats
            algo_name = algorithm.__name__
            self.performance_stats[algo_name]['times'].append(execution_time)
            self.performance_stats[algo_name]['paths'].append(path_length)
            
            # Draw the path immediately
            self.draw_path(path)
            # Update performance graphs
            self.update_performance_graphs()
            # Ensure the canvas updates
            self.canvas.update()
            print(f"Algorithm: {algo_name}, Execution Time: {execution_time}, Path Length: {path_length}")
        else:
            # Handle cases where no path is found
            print(f"No path found for {algo_name} algorithm.")

    def draw_path(self, path):
        """Draw the path on the maze"""
        path_color = "#007acc" if self.dark_mode else "blue"
        
        for i in range(1, len(path)):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            self.canvas.create_line(
                y1 * self.cell_size + self.cell_size // 2,
                x1 * self.cell_size + self.cell_size // 2,
                y2 * self.cell_size + self.cell_size // 2,
                x2 * self.cell_size + self.cell_size // 2,
                fill=path_color, width=max(1, self.cell_size // 10),
                tags="path"
            )

    def update_performance_graphs(self):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Set style
        plt.style.use('bmh')  # Alternative built-in style
        colors = {'dfs': '#FF9999', 'bfs': '#66B2FF', 
                  'greedy': '#99FF99', 'a_star': '#FFCC99'}

        # Plot execution times
        for algo in self.performance_stats:
            times = self.performance_stats[algo]['times']
            if times:
                # Plot actual times
                self.ax1.plot(times, label=algo, color=colors[algo], 
                             marker='o', markersize=4, linewidth=2, alpha=0.7)
                
                # Add average execution time line
                avg_time = np.mean(times)
                self.ax1.axhline(y=avg_time, color=colors[algo], linestyle='--', alpha=0.5)
                self.ax1.text(len(times)-1, avg_time, 
                             f'Avg: {avg_time:.2f}s', 
                             fontsize=8, color=colors[algo], ha='right')

        self.ax1.set_title('Algorithm Performance Over Time', pad=10, fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Execution Time (seconds)', fontsize=10)
        self.ax1.set_xlabel('Iteration', fontsize=10)
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.ax1.legend(fontsize=8, loc='upper left')

        # Plot path lengths
        for algo in self.performance_stats:
            paths = self.performance_stats[algo]['paths']
            if paths:
                # Plot actual path lengths
                self.ax2.plot(paths, label=algo, color=colors[algo], 
                             marker='o', markersize=4, linewidth=2, alpha=0.7)
                
                # Add average path length line
                avg_path = np.mean(paths)
                self.ax2.axhline(y=avg_path, color=colors[algo], linestyle='--', alpha=0.5)
                self.ax2.text(len(paths)-1, avg_path, 
                             f'Avg: {avg_path:.1f}', 
                             fontsize=8, color=colors[algo], ha='right')

        self.ax2.set_title('Path Length Comparison', pad=10, fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('Path Length (steps)', fontsize=10)
        self.ax2.set_xlabel('Iteration', fontsize=10)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        self.ax2.legend(fontsize=8, loc='upper left')

        # Adjust layout
        self.fig.tight_layout(pad=3.0)

        # Redraw the figure
        self.fig.canvas.draw()

    def toggle_dark_mode(self):
        """Toggle between light and dark mode"""
        self.dark_mode = not self.dark_mode
        
        style = ttk.Style()
        if self.dark_mode:
            # Dark mode colors
            self.master.configure(bg='#1a1a1a')
            style.configure('Custom.TFrame', background='#1a1a1a')
            style.configure('Control.TFrame', background='#2d2d2d')
            style.configure('Custom.TLabel', background='#1a1a1a', foreground='#ffffff')
            self.canvas.configure(bg="#1e1e1e", highlightbackground="#404040")
        else:
            # Light mode colors
            self.master.configure(bg='#2c3e50')
            style.configure('Custom.TFrame', background='#2c3e50')
            style.configure('Control.TFrame', background='#34495e')
            style.configure('Custom.TLabel', background='#2c3e50', foreground='white')
            self.canvas.configure(bg="white", highlightbackground="#3498db")
        
        self.draw_maze()

    def clear_stats(self):
        """Clear the performance stats"""
        for algo in self.performance_stats:
            self.performance_stats[algo]['times'].clear()
            self.performance_stats[algo]['paths'].clear()
        
        # Clear the performance graphs
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title('Algorithm Performance Over Time', pad=10, fontsize=10, fontweight='bold')
        self.ax1.set_ylabel('Execution Time (seconds)', fontsize=8)
        self.ax1.set_xlabel('Iteration', fontsize=8)
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        self.ax2.set_title('Path Length Comparison', pad=10, fontsize=10, fontweight='bold')
        self.ax2.set_ylabel('Path Length (steps)', fontsize=8)
        self.ax2.set_xlabel('Iteration', fontsize=8)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and redraw
        self.fig.tight_layout(pad=3.0)
        self.fig.canvas.draw()

    def toggle_start_change(self):
        """Toggle the start position change mode"""
        self.changing_start = not self.changing_start
        if self.changing_start:
            self.canvas.config(cursor="cross")  # Change cursor to crosshair
            self.master.title("Maze Pathfinding - Click to place new start position")
        else:
            self.canvas.config(cursor="")  # Reset cursor
            self.master.title("Maze Pathfinding")

    def on_canvas_click(self, event):
        """Handle canvas clicks for changing start position"""
        if not self.changing_start:
            return
        
        # Convert click coordinates to maze indices
        cell_x = event.y // self.cell_size
        cell_y = event.x // self.cell_size
        
        # Check if click is within maze bounds
        if 0 <= cell_x < self.maze_size and 0 <= cell_y < self.maze_size:
            # Check if clicked position is a valid path (not a wall)
            if self.maze[cell_x, cell_y] == 0:
                # Update start position
                self.start = (cell_x, cell_y)
                # Exit start change mode
                self.changing_start = False
                self.canvas.config(cursor="")
                self.master.title("Maze Pathfinding")
                # Clear any existing path
                self.current_path = None
                self.last_path = None
                # Redraw maze
                self.draw_maze()
            else:
                print("Cannot place start position on a wall!")

    def reset_graphs(self):
        """Reset only the graphs and performance stats without affecting the maze"""
        # Clear performance stats
        for algo in self.performance_stats:
            self.performance_stats[algo]['times'].clear()
            self.performance_stats[algo]['paths'].clear()
        
        # Clear the performance graphs
        self.ax1.clear()
        self.ax2.clear()
        
        # Set up empty graphs with titles
        self.ax1.set_title('Algorithm Performance Over Time', pad=10, fontsize=10, fontweight='bold')
        self.ax1.set_ylabel('Execution Time (seconds)', fontsize=8)
        self.ax1.set_xlabel('Iteration', fontsize=8)
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        self.ax2.set_title('Path Length Comparison', pad=10, fontsize=10, fontweight='bold')
        self.ax2.set_ylabel('Path Length (steps)', fontsize=8)
        self.ax2.set_xlabel('Iteration', fontsize=8)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and redraw
        self.fig.tight_layout(pad=3.0)
        self.fig.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeGUI(root)
    root.geometry("800x1000")
    root.mainloop()
