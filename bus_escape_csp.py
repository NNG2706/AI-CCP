
"""
Bus Escape Puzzle - Constraint Satisfaction Problem (CSP) Solver

CSP Formulation:
================

VARIABLES:
- Position of each bus (Red, Green, Blue, Yellow, Orange)
- Each variable represents a bus's current position (row, col)

DOMAINS:
- For horizontal buses: All valid (row, col) positions where the bus can fit horizontally
- For vertical buses: All valid (row, col) positions where the bus can fit vertically
- Domain size depends on bus length and grid boundaries

CONSTRAINTS:
1. Movement Direction Constraint: Horizontal buses move only left/right, vertical buses only up/down
2. Collision Constraint: No two buses can occupy the same cell (cells(Bi) ∩ cells(Bj) = ∅)
3. Boundary Constraint: All bus cells must be within 6×6 grid (0 ≤ row, col < 6)
4. Exit Constraint: Red Bus must reach exit at (0,5) - this is the goal state
5. Passenger Matching Constraint: Group A→Red, Group B→Yellow, Group C→Green
6. Blockage Constraint: Buses move ONE CELL at a time (adjacent positions only), cannot jump over obstacles

HEURISTICS:
- MRV (Minimum Remaining Values): Select bus with fewest legal moves to reduce search space
- LCV (Least Constraining Value): Order moves to maximize flexibility for other buses

SEARCH ALGORITHM:
- BFS (Breadth-First Search) with heuristic ordering for optimal solution
- Alternative to pure backtracking, better suited for sliding block puzzles
- Guarantees shortest solution path (minimal number of moves)

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
class Passenger:
    """
    Individual passenger with unique ID and deterministic name.

    Attributes:
        passenger_id: Unique passenger identifier (1-50)
        name: Deterministically generated name
        assigned_bus: Bus color this passenger is assigned to
        group: Passenger group (A, B, or C)
    """
    passenger_id: int
    name: str
    assigned_bus: BusColor
    group: str


@dataclass
class Bus:
    """
    Represents a bus in the puzzle.

    Attributes:
        color: Color/name of the bus
        length: Length of the bus (number of cells it occupies)
        orientation: HORIZONTAL or VERTICAL
        position: Top-left position as (row, col)
        passenger_group: Assigned passenger group (A, B, or C) - optional
    """
    color: BusColor
    length: int
    orientation: Orientation
    position: Tuple[int, int]
    passenger_group: Optional[str] = None

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
        return Bus(self.color, self.length, self.orientation, self.position, self.passenger_group)


class PassengerManager:
    """Manages passenger creation, assignment, and tracking."""

    # Predefined name lists for deterministic generation
    FIRST_NAMES = [
        "John", "Emma", "Michael", "Sophia", "William", "Olivia", "James", "Ava",
        "Robert", "Isabella", "David", "Mia", "Richard", "Charlotte", "Joseph", "Amelia",
        "Thomas", "Harper", "Christopher", "Evelyn", "Daniel", "Abigail", "Matthew", "Emily",
        "Anthony", "Elizabeth", "Mark", "Sofia", "Donald", "Avery", "Steven", "Ella",
        "Paul", "Scarlett", "Andrew", "Grace", "Joshua", "Chloe", "Kenneth", "Victoria",
        "Kevin", "Madison", "Brian", "Luna", "George", "Penelope", "Timothy", "Layla",
        "Ronald", "Riley", "Edward", "Zoey", "Jason", "Nora", "Jeffrey", "Lily"
    ]

    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
        "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
        "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
        "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker"
    ]

    # Passenger assignment mapping
    GROUP_TO_BUS = {
        'A': BusColor.RED,
        'B': BusColor.YELLOW,
        'C': BusColor.GREEN
    }

    def __init__(self, total_passengers: int = 50):
        """
        Initialize passenger manager with deterministic passenger creation.

        Args:
            total_passengers: Total number of passengers to create (default 50)
        """
        self.total_passengers = total_passengers
        self.passengers: List[Passenger] = []
        self._initialize_passengers()

    def _initialize_passengers(self) -> None:
        """
        Create and assign all passengers deterministically.

        Uses hash-based distribution algorithm:
        - For each passenger ID (1 to total_passengers):
          hash_value = (passenger_id * 7 + 13) % 3
          0 → Group A (Red), 1 → Group B (Yellow), 2 → Group C (Green)
        """
        for passenger_id in range(1, self.total_passengers + 1):
            # Deterministic group assignment using hash function
            hash_value = (passenger_id * 7 + 13) % 3

            # Map hash to group
            if hash_value == 0:
                group = 'A'
            elif hash_value == 1:
                group = 'B'
            else:  # hash_value == 2
                group = 'C'

            # Get corresponding bus color
            bus_color = self.GROUP_TO_BUS[group]

            # Generate deterministic name
            name = self._generate_passenger_name(passenger_id)

            # Create passenger
            passenger = Passenger(
                passenger_id=passenger_id,
                name=name,
                assigned_bus=bus_color,
                group=group
            )

            self.passengers.append(passenger)

    def _generate_passenger_name(self, passenger_id: int) -> str:
        """
        Generate deterministic name based on passenger ID.

        Args:
            passenger_id: Unique passenger identifier

        Returns:
            Full name (first + last)
        """
        # Use passenger_id to deterministically select names
        first_idx = (passenger_id - 1) % len(self.FIRST_NAMES)
        last_idx = ((passenger_id - 1) // len(self.FIRST_NAMES)) % len(self.LAST_NAMES)

        first_name = self.FIRST_NAMES[first_idx]
        last_name = self.LAST_NAMES[last_idx]

        return f"{first_name} {last_name}"

    def get_passengers_by_bus(self, bus_color: BusColor) -> List[Passenger]:
        """
        Return all passengers assigned to specific bus.

        Args:
            bus_color: Bus color to filter by

        Returns:
            List of passengers assigned to that bus
        """
        return [p for p in self.passengers if p.assigned_bus == bus_color]

    def get_distribution_summary(self) -> Dict[str, int]:
        """
        Get summary of passenger distribution across buses.

        Returns:
            Dictionary mapping group to passenger count
        """
        distribution = {'A': 0, 'B': 0, 'C': 0}
        for passenger in self.passengers:
            distribution[passenger.group] += 1
        return distribution

    def print_passenger_distribution(self) -> None:
        """Print passenger distribution summary."""
        distribution = self.get_distribution_summary()

        print("\nPassenger Distribution:")
        print(f"- Red Bus (Group A): {distribution['A']} passengers")
        print(f"- Yellow Bus (Group B): {distribution['B']} passengers")
        print(f"- Green Bus (Group C): {distribution['C']} passengers")
        print(f"Total: {sum(distribution.values())} passengers")

    def print_passenger_manifest(self, reached_destination_bus: Optional[BusColor] = None) -> None:
        """
        Print detailed passenger report.

        Args:
            reached_destination_bus: Bus color that reached destination (None if none)
        """
        print("\n" + "=" * 60)
        print("PASSENGER MANIFEST")
        print("=" * 60)

        # Print by group
        for group in ['A', 'B', 'C']:
            bus_color = self.GROUP_TO_BUS[group]
            passengers = self.get_passengers_by_bus(bus_color)

            reached_dest = (bus_color == reached_destination_bus) if reached_destination_bus else False

            print(f"\n{bus_color.value} Bus (Group {group}) - {len(passengers)} Passengers:")

            for idx, passenger in enumerate(passengers, 1):
                status = "Reached destination ✓" if reached_dest else "Did not reach destination"
                print(f"  {idx}. {passenger.name} (ID: {passenger.passenger_id}) - {status}")

        # Summary
        distribution = self.get_distribution_summary()
        total_reached = distribution['A'] if reached_destination_bus == BusColor.RED else 0
        total_not_reached = sum(distribution.values()) - total_reached

        print(f"\nSummary:")
        print(f"- Total passengers: {sum(distribution.values())}")
        print(f"- Reached destination: {total_reached}")
        print(f"- Did not reach destination: {total_not_reached}")
        if reached_destination_bus == BusColor.RED:
            print(f"- Red Bus successfully delivered Group A to exit")
        print("=" * 60)


class BusEscapeCSP:
    """Optimized CSP Solver for Bus Escape Puzzle using BFS with heuristics."""

    GRID_SIZE = 6
    EXIT_POSITION = (0, 5)
    MAX_SEARCH_ITERATIONS = 50000  # Maximum nodes to explore before giving up
    MAX_MOVES_PER_BUS = 5  # Max moves to try per bus at each BFS node (branching factor control)

    # Color codes for grid visualization
    COLOR_CODES = {
        BusColor.RED: 'R',
        BusColor.GREEN: 'G',
        BusColor.BLUE: 'B',
        BusColor.YELLOW: 'Y',
        BusColor.ORANGE: 'O'
    }

    # Passenger assignment constraints
    PASSENGER_ASSIGNMENTS = {
        'A': BusColor.RED,
        'B': BusColor.YELLOW,
        'C': BusColor.GREEN
    }

    def __init__(self, buses: List[Bus]):
        """Initialize the CSP solver."""
        self.initial_buses = [bus.copy() for bus in buses]
        self.buses = [bus.copy() for bus in buses]
        self.bus_dict = {bus.color: bus for bus in self.buses}

        # Assign passengers according to constraints
        self._assign_passengers()

        # Statistics
        self.nodes_explored = 0
        self.mrv_activations = 0  # Count MRV heuristic activations
        self.lcv_calculations = 0  # Count LCV heuristic calculations
        self.mrv_decisions = []
        self.lcv_decisions = []
        self.solution_path = []
        self.start_time = 0

        # Cache for domains
        self.domain_cache: Dict[BusColor, List[Tuple[int, int]]] = {}
        self._initialize_domains()

    def _assign_passengers(self) -> None:
        """Assign passenger groups to buses according to constraints"""
        for group, bus_color in self.PASSENGER_ASSIGNMENTS.items():
            if bus_color in self.bus_dict:
                self.bus_dict[bus_color].passenger_group = group
                # Also update in initial_buses
                for bus in self.initial_buses:
                    if bus.color == bus_color:
                        bus.passenger_group = group

    def _initialize_domains(self) -> None:
        """
        Initialize domains for all buses.

        CSP Domain Initialization:
        - Domain Di for each variable (bus) contains all positions satisfying boundary constraints
        - Collision constraints are checked dynamically during search
        - This separation allows efficient domain calculation and flexible constraint checking
        """
        for bus in self.buses:
            self.domain_cache[bus.color] = self._calculate_domain(bus)

    def _calculate_domain(self, bus: Bus) -> List[Tuple[int, int]]:
        """
        Calculate all valid positions for a bus considering only boundary constraints.

        Boundary Constraint Implementation:
        - For horizontal bus of length L: valid columns are 0 to (GRID_SIZE - L)
        - For vertical bus of length L: valid rows are 0 to (GRID_SIZE - L)
        - All positions in domain satisfy: 0 ≤ row < 6 and 0 ≤ col < 6 for all cells

        Movement Direction Constraint:
        - Horizontal buses: domain includes all rows but limited columns
        - Vertical buses: domain includes all columns but limited rows
        - This inherently enforces that buses cannot rotate
        """
        valid_positions = []

        if bus.orientation == Orientation.HORIZONTAL:
            # Horizontal: can be at any row, but col must allow full length
            for row in range(self.GRID_SIZE):
                for col in range(self.GRID_SIZE - bus.length + 1):
                    valid_positions.append((row, col))
        else:  # VERTICAL
            # Vertical: can be at any col, but row must allow full length
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
        """
        Check if a position is valid for a bus - implements constraint checking.

        Constraint Verification:
        1. Boundary Constraint: All cells must be within grid (0 ≤ row, col < 6)
        2. Collision Constraint: cells(bus) ∩ cells(other_buses) = ∅

        Blockage Constraint:
        - Explicitly enforced by restricting moves to adjacent positions only
        - This method (get_legal_moves) only generates one-cell-away positions
        - Prevents buses from jumping over or teleporting through obstacles

        Returns:
            True if position satisfies all constraints, False otherwise
        """
        bus = next(b for b in buses if b.color == bus_color)
        temp_bus = bus.copy()
        temp_bus.position = position
        new_cells = temp_bus.get_occupied_cells()

        # Check boundary constraint
        for row, col in new_cells:
            if row < 0 or row >= self.GRID_SIZE or col < 0 or col >= self.GRID_SIZE:
                return False

        # Check collision constraint
        occupied_by_others = self.get_all_occupied_cells(buses, exclude_bus=bus_color)
        if new_cells & occupied_by_others:
            return False

        return True

    def get_legal_moves(self, buses: List[Bus], bus_color: BusColor) -> List[Tuple[int, int]]:
        """
        Get all legal ADJACENT positions for a bus in current state.

        This implements constraint-based filtering with Blockage Constraint:
        - Start with full domain (all positions satisfying boundary constraints)
        - Filter to only ADJACENT positions (one cell away)
        - Filter out positions violating collision constraints with other buses
        - Result is set of legal values for this variable in current state

        Blockage Constraint Enforcement:
        - Buses can only move ONE CELL at a time (up/down/left/right)
        - This prevents buses from "jumping over" or "teleporting through" obstacles
        - A bus blocked by another bus cannot move in that direction
        - Path must be clear for every individual move

        This is the core of CSP constraint propagation for this problem.
        """
        bus = next(b for b in buses if b.color == bus_color)
        legal_moves = []
        current_row, current_col = bus.position

        # Generate only ADJACENT positions based on orientation
        adjacent_positions = []

        if bus.orientation == Orientation.HORIZONTAL:
            # Horizontal bus can move left or right by one cell
            adjacent_positions.append((current_row, current_col - 1))  # Left
            adjacent_positions.append((current_row, current_col + 1))  # Right
        else:  # VERTICAL
            # Vertical bus can move up or down by one cell
            adjacent_positions.append((current_row - 1, current_col))  # Up
            adjacent_positions.append((current_row + 1, current_col))  # Down

        # Filter adjacent positions: must be in domain and satisfy all constraints
        domain = self.domain_cache[bus_color]
        for position in adjacent_positions:
            if position in domain and self.is_valid_position(buses, bus_color, position):
                legal_moves.append(position)

        return legal_moves

    def get_state_hash(self, buses: List[Bus]) -> Tuple:
        """Get hashable representation of state"""
        return tuple(sorted((bus.color.value, bus.position) for bus in buses))

    def is_goal_state(self, buses: List[Bus]) -> bool:
        """
        Check if Red Bus has reached the exit position.

        Exit Constraint (Goal State):
        - Red Bus must reach exit cell at (0,5)
        - For horizontal bus, rightmost cell must be at exit position
        - This is the satisfaction condition for the CSP

        Passenger Matching Constraint:
        - Implicitly satisfied by assignment in __init__
        - Group A→Red, B→Yellow, C→Green (fixed assignments)
        """
        red_bus = next(b for b in buses if b.color == BusColor.RED)
        if red_bus.orientation == Orientation.HORIZONTAL:
            rightmost_col = red_bus.position[1] + red_bus.length - 1
            return red_bus.position[0] == self.EXIT_POSITION[0] and rightmost_col == self.EXIT_POSITION[1]
        return False

    def get_bus_priority_by_mrv(self, buses: List[Bus]) -> List[Tuple[BusColor, int]]:
        """
        Order buses by MRV (Minimum Remaining Values) heuristic.
        Returns list of (bus_color, num_legal_moves) tuples sorted by constraint level.
        """
        self.mrv_activations += 1  # Count MRV activation

        bus_moves = []

        for bus in buses:
            legal_moves = self.get_legal_moves(buses, bus.color)
            bus_moves.append((bus.color, len(legal_moves)))

        # Sort by number of moves (ascending = most constrained first)
        bus_moves.sort(key=lambda x: x[1])

        return bus_moves

    def apply_lcv_heuristic(self, buses: List[Bus], bus_color: BusColor, moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Order moves using LCV (Least Constraining Value) heuristic.
        """
        self.lcv_calculations += 1  # Count LCV calculation

        bus = next(b for b in buses if b.color == bus_color)

        # Special handling for Red bus on goal row (goal-directed)
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
        """
        self.start_time = time.time()

        # Queue: (buses_state, path)
        queue = deque([(self.initial_buses, [self.initial_buses])])
        visited = {self.get_state_hash(self.initial_buses)}

        while queue and self.nodes_explored < self.MAX_SEARCH_ITERATIONS:
            current_buses, path = queue.popleft()
            self.nodes_explored += 1

            # Check goal (Exit Constraint)
            if self.is_goal_state(current_buses):
                self.solution_path = path
                return True

            # Apply MRV heuristic to order buses by constraint level
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

                # Try top LCV-ordered moves
                for move in ordered_moves[:self.MAX_MOVES_PER_BUS]:
                    # Create new state
                    new_buses = [b.copy() for b in current_buses]
                    moving_bus = next(b for b in new_buses if b.color == bus_to_move)
                    moving_bus.position = move

                    # Check if state was visited (avoid cycles)
                    state_hash = self.get_state_hash(new_buses)
                    if state_hash not in visited:
                        visited.add(state_hash)
                        queue.append((new_buses, path + [new_buses]))

        return False

    def visualize_grid(self, buses: List[Bus]) -> str:
        """Create visual representation of grid with buses."""
        grid = [['.' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        for bus in buses:
            code = self.COLOR_CODES[bus.color]
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
        """Print complete solution with statistics in required format"""
        elapsed_time = time.time() - self.start_time

        print("Initial State:")
        print(self.visualize_grid(self.initial_buses))

        if not self.solution_path:
            print("No solution found!")
            print("\nThis could be because:")
            print("- The puzzle configuration has no valid solution")
            print("- The search space is too large (exceeded MAX_SEARCH_ITERATIONS)")
            print("- Constraints are too restrictive")
            return

        print("Solution found:")

        # Display each move with proper formatting
        move_count = 0
        for idx in range(1, len(self.solution_path)):
            prev_state = self.solution_path[idx - 1]
            current_state = self.solution_path[idx]

            # Find which bus moved
            for i, bus in enumerate(current_state):
                prev_bus = prev_state[i]
                if bus.position != prev_bus.position:
                    move_count += 1
                    from_pos = prev_bus.position
                    to_pos = bus.position

                    # Check if this is the final move to exit
                    if bus.color == BusColor.RED and self.is_goal_state(current_state):
                        # Calculate rightmost cell position (the actual exit cell)
                        rightmost_col = bus.position[1] + bus.length - 1
                        exit_cell = (bus.position[0], rightmost_col)
                        print(f"Move {move_count}: {bus.color.value} Bus reaches exit at {exit_cell}")
                    else:
                        print(f"Move {move_count}: {bus.color.value} Bus moves from {from_pos} to {to_pos}")

                    print(self.visualize_grid(current_state))

        print("\nPassenger Assignments:")
        for group in sorted(self.PASSENGER_ASSIGNMENTS.keys()):
            bus_color = self.PASSENGER_ASSIGNMENTS[group]
            print(f"Group {group} → {bus_color.value} Bus")

        print("\nStatistics:")
        print(f"Total moves: {move_count}")
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"MRV activations: {self.mrv_activations}")
        print(f"LCV calculations: {self.lcv_calculations}")
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
        print(f"Time elapsed: {elapsed_time:.4f} seconds")
        print(f"Average time per node: {elapsed_time / max(self.nodes_explored, 1):.6f} seconds")
        print("=" * 60)


class EnhancedBusEscapeCSP(BusEscapeCSP):
    """Enhanced CSP solver with passenger management using pure BFS approach."""

    def __init__(self, buses: List[Bus], total_passengers: int = 50):
        """
        Initialize enhanced solver with passenger management.

        Args:
            buses: List of Bus objects
            total_passengers: Total number of passengers to create
        """
        super().__init__(buses)
        self.passenger_manager = PassengerManager(total_passengers)

    def print_enhanced_solution(self) -> None:
        """Print solution with passenger manifest and CSP statistics."""
        # First print standard CSP solution output
        self.print_solution()

        # Then add passenger manifest
        if self.solution_path:
            final_buses = self.solution_path[-1]
            reached_bus = BusColor.RED if self.is_goal_state(final_buses) else None
        else:
            reached_bus = None

        self.passenger_manager.print_passenger_manifest(reached_bus)


def create_example_puzzle() -> List[Bus]:
    """
    Create Bus Escape puzzle configuration matching problem statement.

    Initial layout (from problem statement):
      0 1 2 3 4 5
    0 . . . . . E
    1 . . . . . .
    2 R R . B B B
    3 . . . . . .
    4 G . O . Y Y
    5 G . O . . .

    Red bus needs to reach exit at (0,5). Other buses block the path.
    """
    buses = [
        Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (2, 0)),
        Bus(BusColor.GREEN, 2, Orientation.VERTICAL, (4, 0)),
        Bus(BusColor.BLUE, 3, Orientation.HORIZONTAL, (2, 3)),
        Bus(BusColor.YELLOW, 2, Orientation.HORIZONTAL, (4, 4)),
        Bus(BusColor.ORANGE, 2, Orientation.VERTICAL, (4, 2)),
    ]
    return buses


def create_solvable_puzzle() -> List[Bus]:
    """
    Create a solvable Bus Escape puzzle configuration.
    """
    buses = [
        Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 0)),
        Bus(BusColor.GREEN, 2, Orientation.VERTICAL, (4, 0)),
        Bus(BusColor.BLUE, 3, Orientation.HORIZONTAL, (2, 2)),
        Bus(BusColor.YELLOW, 2, Orientation.HORIZONTAL, (5, 3)),
        Bus(BusColor.ORANGE, 1, Orientation.VERTICAL, (5, 2)),
    ]
    return buses


def create_complex_solvable_puzzle() -> List[Bus]:
    """
    Custom Bus Escape puzzle matching the provided grid:

      0 1 2 3 4 5
    0 R R . O G E
    1 . . . O G .
    2 . . B B B .
    3 . . Y Y . .
    4 . . . . . .
    5 . . . . . .
    """
    buses = [
        # Red bus (horizontal, length 2)
        Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 0)),

        # Orange bus (vertical, length 2)
        Bus(BusColor.ORANGE, 2, Orientation.VERTICAL, (0, 3)),

        # Green bus (vertical, length 2)
        Bus(BusColor.GREEN, 2, Orientation.VERTICAL, (0, 4)),

        # Blue bus (horizontal, length 3)
        Bus(BusColor.BLUE, 3, Orientation.HORIZONTAL, (2, 2)),

        # Yellow bus (horizontal, length 2)
        Bus(BusColor.YELLOW, 2, Orientation.HORIZONTAL, (3, 2)),
    ]
    return buses


# ==========================================================
# BEGINNER-FRIENDLY CUSTOM GRID INPUT (ONLY USED IN --menu)
# ==========================================================

def _print_beginner_grid_instructions() -> None:
    print("\n" + "=" * 60)
    print("CUSTOM GRID INPUT (Beginner-Friendly)")
    print("=" * 60)
    print("Enter the puzzle as 6 rows. Each row must have exactly 6 characters.")
    print("\nAllowed characters:")
    print("  .  = empty cell")
    print("  R  = Red bus")
    print("  G  = Green bus")
    print("  B  = Blue bus")
    print("  Y  = Yellow bus")
    print("  O  = Orange bus")
    print("  E  = Exit marker (optional). Solver always assumes exit at (row 0, col 5).")
    print("\nExample (copy/paste):")
    print("  RR.OGE")
    print("  ...OG.")
    print("  ..BBB.")
    print("  ..YY..")
    print("  ......")
    print("  ......")
    print("\nRules:")
    print("  1) Each bus letter must form ONE straight line (one row or one column).")
    print("  2) No gaps allowed (e.g. 'R.R' is invalid).")
    print("  3) Red bus (R) is required.")
    print("=" * 60 + "\n")


def read_custom_grid_from_user(grid_size: int = 6) -> List[str]:
    _print_beginner_grid_instructions()

    while True:
        rows: List[str] = []
        for r in range(grid_size):
            while True:
                line = input(f"Enter row {r} (6 chars): ").strip()
                line = line.replace(" ", "").upper()

                if len(line) != grid_size:
                    print(f"❌ Row {r} must be exactly {grid_size} characters (you typed {len(line)}). Try again.")
                    continue

                allowed = set(".RGBYOE")
                bad = [ch for ch in line if ch not in allowed]
                if bad:
                    print(f"❌ Invalid characters in row {r}: {bad}")
                    print("   Use only: . R G B Y O E")
                    continue

                rows.append(line)
                break

        # Friendly note about E placement (solver uses fixed exit position)
        any_e = any('E' in row for row in rows)
        if any_e:
            e_positions = [(ri, ci) for ri in range(grid_size) for ci in range(grid_size) if rows[ri][ci] == 'E']
            if e_positions != [(0, 5)]:
                print("⚠️ Note: You placed 'E' somewhere other than (0,5).")
                print("   This solver always assumes the exit is at (row 0, col 5).")
                print("   You can remove 'E' or place it at row 0 col 5. We'll continue anyway.\n")

        if not any('R' in row for row in rows):
            print("❌ You must place the Red bus 'R' somewhere on the grid.")
            continue

        return rows


def parse_grid_to_buses(grid_rows: List[str]) -> List[Bus]:
    """
    Convert a user grid into Bus objects by detecting connected cells per letter.
    Automatically infers length, orientation, and top-left position.
    """
    grid_size = len(grid_rows)

    letter_to_cells: Dict[str, List[Tuple[int, int]]] = {k: [] for k in ["R", "G", "B", "Y", "O"]}

    for r in range(grid_size):
        for c in range(grid_size):
            ch = grid_rows[r][c].upper()
            if ch in letter_to_cells:
                letter_to_cells[ch].append((r, c))

    if not letter_to_cells["R"]:
        raise ValueError("Red bus 'R' is required, but none was found on the grid.")

    letter_to_color = {
        "R": BusColor.RED,
        "G": BusColor.GREEN,
        "B": BusColor.BLUE,
        "Y": BusColor.YELLOW,
        "O": BusColor.ORANGE,
    }

    buses: List[Bus] = []

    for letter, cells in letter_to_cells.items():
        if not cells:
            continue

        cells = sorted(cells)
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}

        # Single cell bus is allowed (length=1)
        if len(rows) == 1 and len(cols) == 1:
            orientation = Orientation.HORIZONTAL
            length = 1
            top_left = cells[0]

        # Horizontal
        elif len(rows) == 1:
            orientation = Orientation.HORIZONTAL
            r = next(iter(rows))
            c_min = min(c for _, c in cells)
            c_max = max(c for _, c in cells)

            expected = {(r, c) for c in range(c_min, c_max + 1)}
            if set(cells) != expected:
                raise ValueError(
                    f"Bus '{letter}' is horizontal but has gaps.\n"
                    f"Found cells: {cells}\n"
                    f"Expected a solid line from col {c_min} to {c_max} on row {r}."
                )

            length = len(cells)
            top_left = (r, c_min)

        # Vertical
        elif len(cols) == 1:
            orientation = Orientation.VERTICAL
            c = next(iter(cols))
            r_min = min(r for r, _ in cells)
            r_max = max(r for r, _ in cells)

            expected = {(r, c) for r in range(r_min, r_max + 1)}
            if set(cells) != expected:
                raise ValueError(
                    f"Bus '{letter}' is vertical but has gaps.\n"
                    f"Found cells: {cells}\n"
                    f"Expected a solid line from row {r_min} to {r_max} on col {c}."
                )

            length = len(cells)
            top_left = (r_min, c)

        else:
            raise ValueError(
                f"Bus '{letter}' is not in a straight line.\n"
                f"It must be all in ONE row (horizontal) or ONE column (vertical).\n"
                f"Found cells: {cells}"
            )

        buses.append(Bus(letter_to_color[letter], length, orientation, top_left))

    return buses


def run_solver_with_buses(buses: List[Bus]) -> bool:
    """Helper to run the enhanced solver for a chosen puzzle."""
    csp = EnhancedBusEscapeCSP(buses, total_passengers=50)
    result = csp.solve_bfs()
    csp.print_enhanced_solution()
    if result:
        print("\n✓ Solution found using pure CSP approach!")
    else:
        print("\n✗ No solution found within search limit.")
    return result


def run_interactive_menu() -> None:
    """
    MAIN MENU:
    1) Simulate built-in puzzle (create_complex_solvable_puzzle)
    2) Enter custom 6x6 grid (beginner-friendly)
    0) Exit

    After completing a run, you can return to the menu.
    """
    while True:
        print("" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        print("1) Simulate built-in Complex Solvable Puzzle (current default)")
        print("2) Enter your own custom 6x6 grid")
        print("0) Exit")
        print("=" * 60)

        choice = input("Choose an option: ").strip()

        if choice == "1":
            buses = create_complex_solvable_puzzle()
            run_solver_with_buses(buses)

        elif choice == "2":
            while True:
                grid = read_custom_grid_from_user(6)
                try:
                    buses = parse_grid_to_buses(grid)
                    run_solver_with_buses(buses)
                    break
                except ValueError as e:
                    print("❌ Your grid has a problem:")
                    print(e)
                    print("Please try entering the grid again.")

        elif choice == "0":
            print("Goodbye!")
            return

        else:
            print("❌ Invalid choice. Please enter 0, 1, or 2.")
            continue

        back = input("Return to main menu? (Y/n): ").strip().lower()
        if back == "n":
            print("Goodbye!")
            return

def main():
    """Main execution function"""
    print("=" * 60)
    print("BUS ESCAPE PUZZLE - CSP SOLVER")
    print("Using Pure BFS with MRV and LCV Heuristics")
    print("With Passenger Management System")
    print("=" * 60)

    # Default behavior remains unchanged (runs complex solvable puzzle directly)
    buses = create_complex_solvable_puzzle()

    # Create enhanced CSP solver with passenger management
    csp = EnhancedBusEscapeCSP(buses, total_passengers=50)

    # Solve puzzle using pure BFS with MRV and LCV heuristics
    result = csp.solve_bfs()

    # Display enhanced solution with passenger manifest
    csp.print_enhanced_solution()

    if result:
        print("\n✓ Solution found using pure CSP approach!")
    else:
        print("\n✗ No solution found within search limit.")


def main_original():
    """Original main function for comparison"""
    print("=" * 60)
    print("BUS ESCAPE PUZZLE - ORIGINAL CSP SOLVER")
    print("=" * 60)

    # Create puzzle
    buses = create_example_puzzle()

    # Create CSP solver
    csp = BusEscapeCSP(buses)

    # Solve puzzle
    result = csp.solve_bfs()

    # Display results
    csp.print_solution()

    if result:
        red_bus = next(b for b in csp.solution_path[-1] if b.color == BusColor.RED)
        # Calculate rightmost cell position (the actual exit cell)
        rightmost_col = red_bus.position[1] + red_bus.length - 1
        exit_cell = (red_bus.position[0], rightmost_col)
        print("\n✓ Solution successfully found!")
        print(f"✓ Red Bus reached exit at position {exit_cell}")
    else:
        print("\n✗ No solution found within search limit.")





if __name__ == "__main__":
    run_interactive_menu()
