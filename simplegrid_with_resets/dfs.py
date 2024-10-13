
def is_valid(x, y, N, grid, visited):
    # Check if the point is within bounds, is not an obstacle, and has not been visited
    return 0 <= x < N and 0 <= y < N and grid[x][y] == 0 and not visited[x][y]

def dfs(grid, x, y, x2, y2, visited):
    # If we've reached the destination, return True
    if (x, y) == (x2, y2):
        return True
    
    # Mark the current cell as visited
    visited[x][y] = True
    
    # Direction vectors for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Explore all four possible directions
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        
        # If the new position is valid, explore it recursively
        if is_valid(nx, ny, len(grid), grid, visited):
            if dfs(grid, nx, ny, x2, y2, visited):
                return True

    # Backtrack if no path is found in this direction
    return False

def find_path(grid, start, end):
    N = len(grid)
    x1, y1 = start
    x2, y2 = end

    # If start or end point is an obstacle, return False immediately
    if grid[x1][y1] == 1 or grid[x2][y2] == 1:
        return False

    # Keep track of visited cells
    visited = [[False for _ in range(N)] for _ in range(N)]
    
    # Start the DFS from the start point
    return dfs(grid, x1, y1, x2, y2, visited)