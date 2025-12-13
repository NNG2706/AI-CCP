"""
Unit tests for Bus Escape CSP Solver

Tests various aspects of the implementation:
- Bus class functionality
- Constraint checking
- Domain calculation
- MRV and LCV heuristics
- Solution finding

Author: AI-CCP Project
"""

import unittest
from bus_escape_csp import Bus, BusColor, Orientation, BusEscapeCSP


class TestBus(unittest.TestCase):
    """Test Bus class functionality"""
    
    def test_horizontal_bus_cells(self):
        """Test that horizontal bus occupies correct cells"""
        bus = Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 1))
        cells = bus.get_occupied_cells()
        expected = {(0, 1), (0, 2)}
        self.assertEqual(cells, expected)
    
    def test_vertical_bus_cells(self):
        """Test that vertical bus occupies correct cells"""
        bus = Bus(BusColor.GREEN, 3, Orientation.VERTICAL, (1, 2))
        cells = bus.get_occupied_cells()
        expected = {(1, 2), (2, 2), (3, 2)}
        self.assertEqual(cells, expected)
    
    def test_bus_copy(self):
        """Test that bus copy works correctly"""
        bus = Bus(BusColor.BLUE, 2, Orientation.HORIZONTAL, (3, 4))
        bus_copy = bus.copy()
        self.assertEqual(bus.color, bus_copy.color)
        self.assertEqual(bus.length, bus_copy.length)
        self.assertEqual(bus.orientation, bus_copy.orientation)
        self.assertEqual(bus.position, bus_copy.position)
        # Ensure it's a separate object
        bus_copy.position = (0, 0)
        self.assertNotEqual(bus.position, bus_copy.position)


class TestBusEscapeCSP(unittest.TestCase):
    """Test CSP solver functionality"""
    
    def setUp(self):
        """Set up test buses"""
        self.test_buses = [
            Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 0)),
            Bus(BusColor.GREEN, 2, Orientation.VERTICAL, (1, 2)),
        ]
    
    def test_domain_calculation_horizontal(self):
        """Test domain calculation for horizontal bus"""
        csp = BusEscapeCSP(self.test_buses)
        red_domain = csp.domain_cache[BusColor.RED]
        # Horizontal bus length 2 on 6x6 grid should have 6 rows * 5 columns = 30 positions
        self.assertEqual(len(red_domain), 30)
        # Check boundaries
        self.assertIn((0, 0), red_domain)
        self.assertIn((5, 4), red_domain)
        self.assertNotIn((0, 5), red_domain)  # Would go out of bounds
    
    def test_domain_calculation_vertical(self):
        """Test domain calculation for vertical bus"""
        csp = BusEscapeCSP(self.test_buses)
        green_domain = csp.domain_cache[BusColor.GREEN]
        # Vertical bus length 2 on 6x6 grid should have 5 rows * 6 columns = 30 positions
        self.assertEqual(len(green_domain), 30)
        # Check boundaries
        self.assertIn((0, 0), green_domain)
        self.assertIn((4, 5), green_domain)
        self.assertNotIn((5, 0), green_domain)  # Would go out of bounds
    
    def test_collision_detection(self):
        """Test that collision detection works"""
        csp = BusEscapeCSP(self.test_buses)
        # Try to move Red bus to position that would collide with Green
        is_valid = csp.is_valid_position(self.test_buses, BusColor.RED, (1, 1))
        # Red at (1,1) would occupy (1,1) and (1,2), Green occupies (1,2) and (2,2)
        # Should collide at (1,2)
        self.assertFalse(is_valid)
    
    def test_no_collision(self):
        """Test that non-colliding position is valid"""
        csp = BusEscapeCSP(self.test_buses)
        # Move Red bus to position that doesn't collide
        is_valid = csp.is_valid_position(self.test_buses, BusColor.RED, (3, 0))
        self.assertTrue(is_valid)
    
    def test_boundary_constraint(self):
        """Test that boundary constraints are enforced"""
        buses = [Bus(BusColor.RED, 3, Orientation.HORIZONTAL, (0, 0))]
        csp = BusEscapeCSP(buses)
        # Try to move bus to position that would go out of bounds
        is_valid = csp.is_valid_position(buses, BusColor.RED, (0, 4))
        # Bus length 3 at (0,4) would need cells (0,4), (0,5), (0,6) - out of bounds
        self.assertFalse(is_valid)
    
    def test_goal_state_detection(self):
        """Test that goal state is correctly detected"""
        # Red bus at exit position
        buses = [Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 4))]
        csp = BusEscapeCSP(buses)
        # Red at (0,4) with length 2 has rightmost cell at (0,5) = EXIT
        self.assertTrue(csp.is_goal_state(buses))
    
    def test_not_goal_state(self):
        """Test that non-goal state is correctly identified"""
        buses = [Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 0))]
        csp = BusEscapeCSP(buses)
        self.assertFalse(csp.is_goal_state(buses))
    
    def test_mrv_heuristic(self):
        """Test that MRV heuristic returns bus priority list"""
        csp = BusEscapeCSP(self.test_buses)
        bus_priority = csp.get_bus_priority_by_mrv(self.test_buses)
        # Should return list of (color, move_count) tuples
        self.assertEqual(len(bus_priority), 2)
        # All buses should have some legal moves
        for color, move_count in bus_priority:
            self.assertGreater(move_count, 0)
        # Should be sorted by move_count (ascending)
        move_counts = [count for _, count in bus_priority]
        self.assertEqual(move_counts, sorted(move_counts))
    
    def test_lcv_heuristic(self):
        """Test that LCV heuristic returns ordered moves"""
        csp = BusEscapeCSP(self.test_buses)
        legal_moves = csp.get_legal_moves(self.test_buses, BusColor.RED)
        if legal_moves:
            ordered_moves = csp.apply_lcv_heuristic(self.test_buses, BusColor.RED, legal_moves)
            # Should return same moves, possibly reordered
            self.assertEqual(set(ordered_moves), set(legal_moves))
    
    def test_solution_finding(self):
        """Test that solver can find solution to simple puzzle"""
        # Very simple puzzle: Red can move directly to exit
        buses = [
            Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 0)),
            Bus(BusColor.GREEN, 2, Orientation.VERTICAL, (2, 2)),
        ]
        csp = BusEscapeCSP(buses)
        result = csp.solve_bfs()
        self.assertTrue(result)
        # Should have found a solution path
        self.assertGreater(len(csp.solution_path), 0)
        # Final state should be goal state
        self.assertTrue(csp.is_goal_state(csp.solution_path[-1]))
    
    def test_state_hashing(self):
        """Test that state hashing works correctly"""
        csp = BusEscapeCSP(self.test_buses)
        hash1 = csp.get_state_hash(self.test_buses)
        # Same state should produce same hash
        hash2 = csp.get_state_hash(self.test_buses)
        self.assertEqual(hash1, hash2)
        # Different state should (likely) produce different hash
        different_buses = [bus.copy() for bus in self.test_buses]
        different_buses[0].position = (1, 1)
        hash3 = csp.get_state_hash(different_buses)
        self.assertNotEqual(hash1, hash3)


class TestConstants(unittest.TestCase):
    """Test that constants are properly defined"""
    
    def test_class_constants(self):
        """Test that CSP class has required constants"""
        self.assertEqual(BusEscapeCSP.GRID_SIZE, 6)
        self.assertEqual(BusEscapeCSP.EXIT_POSITION, (0, 5))
        self.assertIsInstance(BusEscapeCSP.MAX_SEARCH_ITERATIONS, int)
        self.assertIsInstance(BusEscapeCSP.MAX_MOVES_PER_BUS, int)
        self.assertGreater(BusEscapeCSP.MAX_SEARCH_ITERATIONS, 0)
        self.assertGreater(BusEscapeCSP.MAX_MOVES_PER_BUS, 0)


if __name__ == '__main__':
    unittest.main()
