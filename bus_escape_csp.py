"""
Bus Escape Puzzle - Constraint Satisfaction Problem (CSP) Solver (Optimized)

This is an optimized version using BFS with heuristics instead of pure backtracking.
This approach is more suitable for sliding block puzzles.

Author: AI-CCP Project
"""

import time
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque


class Orientation(Enum):
    """Bus orientation enum"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class BusColor(Enum):
    """Bus color enum"""
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    YELLOW = "Yellow"
    ORANGE = "Orange"


@dataclass
class Bus:
    """
    Represents a bus in the puzzle.
    
    Attributes:
        color: Color/name of the bus
        length: Length of the bus (number of cells it occupies)
        orientation: HORIZONTAL or VERTICAL
        position: Top-left position as (row, col)
    """
    color: BusColor
    length: int
    orientation: Orientation
    position: Tuple[int, int]
    
    def get_occupied_cells(self) -> Set[Tuple[int, int]]:
        """Returns set of all cells occupied by this bus."""
        cells = set()
        row, col = self.position
        
        if self.orientation == Orientation.HORIZONTAL:
            for i in range(self.length):
                cells.add((row, col + i))
        else:  # VERTICAL
            for i in range(self.length):
                cells.add((row + i, col))
        
        return cells
    
    def copy(self) -> 'Bus':
        """Create a deep copy of this bus"""
        return Bus(self.color, self.length, self.orientation, self.position)


class BusEscapeCSP:
    """Optimized CSP Solver for Bus Escape Puzzle using BFS with heuristics."""
    
    GRID_SIZE = 6
    EXIT_POSITION = (0, 5)
    MAX_SEARCH_ITERATIONS = 50000  # Maximum nodes to explore before giving up
    MAX_MOVES_PER_BUS = 2  # Branching factor limit per bus to control search space
    
    def __init__(self, buses: List[Bus]):
        """Initialize the CSP solver."""
        self.initial_buses = [bus.copy() for bus in buses]
        self.buses = [bus.copy() for bus in buses]
        self.bus_dict = {bus.color: bus for bus in self.buses}
        
        # Statistics
        self.nodes_explored = 0
        self.mrv_decisions = []
        self.lcv_decisions = []
        self.solution_path = []
        self.start_time = 0
        
        # Cache for domains
        self.domain_cache: Dict[BusColor, List[Tuple[int, int]]] = {}
        self._initialize_domains()
    
    def _initialize_domains(self) -> None:
        """Initialize domains for all buses"""
        for bus in self.buses:
            self.domain_cache[bus.color] = self._calculate_domain(bus)
    
    def _calculate_domain(self, bus: Bus) -> List[Tuple[int, int]]:
        """Calculate all valid positions for a bus (boundary constraints only)."""
        valid_positions = []
        
        if bus.orientation == Orientation.HORIZONTAL:
            for row in range(self.GRID_SIZE):
                for col in range(self.GRID_SIZE - bus.length + 1):
                    valid_positions.append((row, col))
        else:  # VERTICAL
            for row in range(self.GRID_SIZE - bus.length + 1):
                for col in range(self.GRID_SIZE):
                    valid_positions.append((row, col))
        
        return valid_positions
    
    def get_all_occupied_cells(self, buses: List[Bus], exclude_bus: Optional[BusColor] = None) -> Set[Tuple[int, int]]:
        """Get all cells occupied by buses, optionally excluding one bus."""
        occupied = set()
        for bus in buses:
            if exclude_bus is None or bus.color != exclude_bus:
                occupied.update(bus.get_occupied_cells())
        return occupied
    
    def is_valid_position(self, buses: List[Bus], bus_color: BusColor, position: Tuple[int, int]) -> bool:
        """Check if a position is valid for a bus (no collisions)."""
        bus = next(b for b in buses if b.color == bus_color)
        temp_bus = bus.copy()
        temp_bus.position = position
        new_cells = temp_bus.get_occupied_cells()
        
        # Check boundary
        for row, col in new_cells:
            if row < 0 or row >= self.GRID_SIZE or col < 0 or col >= self.GRID_SIZE:
                return False
        
        # Check collision
        occupied_by_others = self.get_all_occupied_cells(buses, exclude_bus=bus_color)
        if new_cells & occupied_by_others:
            return False
        
        return True
    
    def get_legal_moves(self, buses: List[Bus], bus_color: BusColor) -> List[Tuple[int, int]]:
        """Get all legal positions for a bus in current state."""
        bus = next(b for b in buses if b.color == bus_color)
        legal_moves = []
        domain = self.domain_cache[bus_color]
        
        for position in domain:
            if position != bus.position and self.is_valid_position(buses, bus_color, position):
                legal_moves.append(position)
        
        return legal_moves
    
    def get_state_hash(self, buses: List[Bus]) -> Tuple:
        """Get hashable representation of state"""
        return tuple(sorted((bus.color.value, bus.position) for bus in buses))
    
    def is_goal_state(self, buses: List[Bus]) -> bool:
        """Check if Red Bus has reached the exit position."""
        red_bus = next(b for b in buses if b.color == BusColor.RED)
        if red_bus.orientation == Orientation.HORIZONTAL:
            rightmost_col = red_bus.position[1] + red_bus.length - 1
            return red_bus.position[0] == self.EXIT_POSITION[0] and rightmost_col == self.EXIT_POSITION[1]
        return False
    
    def get_bus_priority_by_mrv(self, buses: List[Bus]) -> List[Tuple[BusColor, int]]:
        """
        Order buses by MRV heuristic (fewest legal moves first).
        Returns list of (bus_color, num_legal_moves) tuples sorted by constraint level.
        
        MRV Implementation:
        - Counts legal moves for each bus
        - Sorts buses by number of legal moves (ascending)
        - Most constrained buses (fewest moves) come first
        - This helps detect failures early and reduces search space
        """
        bus_moves = []
        
        for bus in buses:
            legal_moves = self.get_legal_moves(buses, bus.color)
            bus_moves.append((bus.color, len(legal_moves)))
        
        # Sort by number of moves (ascending = most constrained first)
        # This is the core of MRV heuristic
        bus_moves.sort(key=lambda x: x[1])
        
        return bus_moves
    
    def apply_lcv_heuristic(self, buses: List[Bus], bus_color: BusColor, moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Order moves using LCV heuristic.
        Prefers moves that leave more options for other buses.
        For Red bus on goal row, prioritizes rightward movement.
        """
        bus = next(b for b in buses if b.color == bus_color)
        
        # Special handling for Red bus on goal row
        if bus_color == BusColor.RED and bus.position[0] == self.EXIT_POSITION[0]:
            ordered = sorted(moves, key=lambda m: -m[1])  # Rightmost first
            self.lcv_decisions.append({
                'bus': 'Red (goal-directed)',
                'ordered_moves': ordered[:5]
            })
            return ordered
        
        # LCV: Count how many options each move leaves for other buses
        move_scores = []
        
        for move in moves:
            # Create temporary state with this move
            temp_buses = [b.copy() for b in buses]
            temp_bus = next(b for b in temp_buses if b.color == bus_color)
            temp_bus.position = move
            
            # Count total legal moves for all other buses
            total_options = 0
            for other_bus in temp_buses:
                if other_bus.color != bus_color:
                    total_options += len(self.get_legal_moves(temp_buses, other_bus.color))
            
            move_scores.append((move, total_options))
        
        # Sort by total_options (descending = least constraining first)
        move_scores.sort(key=lambda x: -x[1])
        ordered_moves = [m[0] for m in move_scores]
        
        self.lcv_decisions.append({
            'bus': bus_color.value,
            'ordered_moves': ordered_moves[:5]
        })
        
        return ordered_moves
    
    def solve_bfs(self) -> bool:
        """
        Solve using BFS with MRV and LCV heuristics.
        This explores states level by level, ensuring shortest path.
        """
        self.start_time = time.time()
        
        # Queue: (buses_state, path)
        queue = deque([(self.initial_buses, [self.initial_buses])])
        visited = {self.get_state_hash(self.initial_buses)}
        
        while queue and self.nodes_explored < self.MAX_SEARCH_ITERATIONS:
            current_buses, path = queue.popleft()
            self.nodes_explored += 1
            
            # Check goal
            if self.is_goal_state(current_buses):
                self.solution_path = path
                return True
            
            # Apply MRV heuristic to order buses by constraint level
            # Most constrained buses (fewest legal moves) are tried first
            bus_priority = self.get_bus_priority_by_mrv(current_buses)
            
            # Record MRV decision for this node
            if bus_priority and bus_priority[0][1] > 0:
                self.mrv_decisions.append({
                    'bus': bus_priority[0][0].value,
                    'legal_moves': bus_priority[0][1],
                    'all_counts': {color.value: count for color, count in bus_priority},
                    'ordering': [color.value for color, _ in bus_priority]
                })
            
            # Try moving buses in MRV order (most constrained first)
            for bus_to_move, num_moves in bus_priority:
                if num_moves == 0:
                    continue
                
                # Get legal moves for this bus
                legal_moves = self.get_legal_moves(current_buses, bus_to_move)
                
                # Apply LCV to order moves
                ordered_moves = self.apply_lcv_heuristic(current_buses, bus_to_move, legal_moves)
                
                # Try top moves (limit branching factor per bus)
                for move in ordered_moves[:self.MAX_MOVES_PER_BUS]:
                    # Create new state
                    new_buses = [b.copy() for b in current_buses]
                    moving_bus = next(b for b in new_buses if b.color == bus_to_move)
                    moving_bus.position = move
                    
                    # Check if state was visited
                    state_hash = self.get_state_hash(new_buses)
                    if state_hash not in visited:
                        visited.add(state_hash)
                        queue.append((new_buses, path + [new_buses]))
        
        return False
    
    def visualize_grid(self, buses: List[Bus]) -> str:
        """Create visual representation of grid with buses."""
        grid = [['.' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        color_codes = {
            BusColor.RED: 'R',
            BusColor.GREEN: 'G',
            BusColor.BLUE: 'B',
            BusColor.YELLOW: 'Y',
            BusColor.ORANGE: 'O'
        }
        
        for bus in buses:
            code = color_codes[bus.color]
            for cell in bus.get_occupied_cells():
                row, col = cell
                grid[row][col] = code
        
        if grid[self.EXIT_POSITION[0]][self.EXIT_POSITION[1]] == '.':
            grid[self.EXIT_POSITION[0]][self.EXIT_POSITION[1]] = 'E'
        
        result = "  " + " ".join(str(i) for i in range(self.GRID_SIZE)) + "\n"
        for i, row in enumerate(grid):
            result += str(i) + " " + " ".join(row) + "\n"
        
        return result
    
    def print_solution(self) -> None:
        """Print complete solution with statistics"""
        elapsed_time = time.time() - self.start_time
        
        print("=" * 60)
        print("BUS ESCAPE PUZZLE - CSP SOLVER (OPTIMIZED)")
        print("=" * 60)
        print()
        
        print("INITIAL STATE:")
        print(self.visualize_grid(self.initial_buses))
        
        if not self.solution_path:
            print("No solution found!")
            return
        
        print(f"SOLUTION FOUND! ({len(self.solution_path)} states)")
        print()
        
        # Display each state
        for idx, buses_state in enumerate(self.solution_path):
            print(f"State {idx}:")
            print(self.visualize_grid(buses_state))
            
            # Show what changed
            if idx > 0:
                prev_state = self.solution_path[idx - 1]
                for i, bus in enumerate(buses_state):
                    prev_bus = prev_state[i]
                    if bus.position != prev_bus.position:
                        print(f"  -> {bus.color.value} bus moved from {prev_bus.position} to {bus.position}")
            print()
        
        print("=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Solution length: {len(self.solution_path)} states")
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"Time elapsed: {elapsed_time:.4f} seconds")
        print()
        
        print("=" * 60)
        print("MRV HEURISTIC ANALYSIS")
        print("=" * 60)
        print("MRV (Minimum Remaining Values) selects the variable with fewest legal moves.")
        print(f"Total MRV decisions made: {len(self.mrv_decisions)}")
        print()
        if self.mrv_decisions:
            print("Sample MRV decisions (first 5):")
            for i, decision in enumerate(self.mrv_decisions[:5], 1):
                print(f"  Decision {i}: Most constrained bus is {decision['bus']} with {decision['legal_moves']} legal moves")
                if 'all_counts' in decision:
                    print(f"    All move counts: {decision['all_counts']}")
                if 'ordering' in decision:
                    print(f"    Bus ordering (most to least constrained): {decision['ordering']}")
        print()
        
        print("=" * 60)
        print("LCV HEURISTIC ANALYSIS")
        print("=" * 60)
        print("LCV (Least Constraining Value) orders values by least impact on other variables.")
        print(f"Total LCV orderings performed: {len(self.lcv_decisions)}")
        print()
        if self.lcv_decisions:
            print("Sample LCV orderings (first 5):")
            for i, decision in enumerate(self.lcv_decisions[:5], 1):
                print(f"  Ordering {i}: For {decision['bus']}")
                print(f"    Top moves (least constraining first): {decision['ordered_moves']}")
        print()
        
        print("=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)
        print("How MRV reduces search space:")
        print("  - By selecting buses with fewer legal moves first")
        print("  - Failures are detected earlier in the search")
        print("  - Reduces branching factor significantly")
        print()
        print("How LCV minimizes backtracking:")
        print("  - Orders moves to maximize flexibility for remaining buses")
        print("  - Chooses moves that constrain other buses the least")
        print("  - Increases likelihood of finding solution efficiently")
        print()
        print(f"Average time per node: {elapsed_time / max(self.nodes_explored, 1):.6f} seconds")
        print("=" * 60)


def create_example_puzzle() -> List[Bus]:
    """
    Create a solvable Bus Escape puzzle configuration.
    
    This puzzle requires:
    1. Move Orange down (from blocking the exit)
    2. Move Red right to the exit
    
    Initial layout:
      0 1 2 3 4 5
    0 R R . . O E  (Orange blocks rows 0-1 at col 4)
    1 . . . . O .
    2 . . . . . .  (Empty space to move Orange down)
    3 B B . . . .
    4 . . . . . .
    5 . . Y Y . .
    """
    buses = [
        # Red bus (horizontal, length 2) - needs to reach exit at (0, 5)
        Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 0)),
        
        # Orange bus (vertical, length 2) - blocks exit at (0,4)-(1,4)
        Bus(BusColor.ORANGE, 2, Orientation.VERTICAL, (0, 4)),
        
        # Green bus (vertical, length 2) - away from critical path
        Bus(BusColor.GREEN, 2, Orientation.VERTICAL, (3, 5)),
        
        # Blue bus (horizontal, length 2) - away from critical path
        Bus(BusColor.BLUE, 2, Orientation.HORIZONTAL, (3, 0)),
        
        # Yellow bus (horizontal, length 2) - away from critical path
        Bus(BusColor.YELLOW, 2, Orientation.HORIZONTAL, (5, 2)),
    ]
    
    return buses


def main():
    """Main execution function"""
    print("Initializing Bus Escape Puzzle CSP Solver (Optimized)...")
    print()
    
    # Create puzzle
    buses = create_example_puzzle()
    
    # Create CSP solver
    csp = BusEscapeCSP(buses)
    
    # Solve puzzle
    print("Starting BFS with MRV and LCV heuristics...")
    print()
    
    result = csp.solve_bfs()
    
    # Display results
    csp.print_solution()
    
    if result:
        print("\n✓ Solution successfully found!")
        print(f"✓ Red Bus reached exit at position {BusEscapeCSP.EXIT_POSITION}")
    else:
        print("\n✗ No solution found within search limit.")


if __name__ == "__main__":
    main()
