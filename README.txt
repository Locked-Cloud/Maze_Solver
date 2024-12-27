# Maze Pathfinding Visualizer

This project is a maze pathfinding visualizer built using Python's Tkinter for the GUI and NumPy for maze generation. It implements various pathfinding algorithms, including Depth-First Search (DFS), Breadth-First Search (BFS), Greedy Best-First Search, and A* Search. The application allows users to visualize the algorithms' performance and interact with the maze.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
   ```

2. Install the required packages:
   ```bash
   pip install numpy matplotlib
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Usage

- Click on the "Change Start" button to change the starting position of the maze.
- Click on any valid cell in the maze to set it as the new starting point.
- Select an algorithm (DFS, BFS, Greedy, A*) to find a path from the start to one of the exits.
- Use the "Generate Maze" button to create a new maze.
- The "Toggle Dark Mode" button switches between light and dark themes.
- The "Clear Stats" button resets the performance statistics.
- The "Reset Graphs" button clears the performance graphs.

## Code Explanation

### Imports
