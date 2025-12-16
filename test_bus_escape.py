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


class TestPassengerManagement(unittest.TestCase):
    """Test passenger management system"""
    
    def test_passenger_creation(self):
        """Test that passengers are created correctly"""
        from bus_escape_csp import PassengerManager
        pm = PassengerManager(50)
        
        # Check total passengers
        self.assertEqual(len(pm.passengers), 50)
        
        # Check unique IDs
        ids = [p.passenger_id for p in pm.passengers]
        self.assertEqual(len(set(ids)), 50)
        self.assertEqual(min(ids), 1)
        self.assertEqual(max(ids), 50)
    
    def test_deterministic_distribution(self):
        """Test that passenger distribution is deterministic"""
        from bus_escape_csp import PassengerManager
        
        pm1 = PassengerManager(50)
        pm2 = PassengerManager(50)
        
        dist1 = pm1.get_distribution_summary()
        dist2 = pm2.get_distribution_summary()
        
        self.assertEqual(dist1, dist2)
        self.assertEqual(sum(dist1.values()), 50)
    
    def test_hash_based_assignment(self):
        """Test that hash-based assignment works correctly"""
        from bus_escape_csp import PassengerManager, BusColor
        
        pm = PassengerManager(50)
        
        # Test specific passengers based on hash function
        # ID=1: (1*7+13)%3 = 20%3 = 2 -> Group C (Green)
        p1 = pm.passengers[0]
        self.assertEqual(p1.passenger_id, 1)
        self.assertEqual(p1.group, 'C')
        self.assertEqual(p1.assigned_bus, BusColor.GREEN)
        
        # ID=2: (2*7+13)%3 = 27%3 = 0 -> Group A (Red)
        p2 = pm.passengers[1]
        self.assertEqual(p2.passenger_id, 2)
        self.assertEqual(p2.group, 'A')
        self.assertEqual(p2.assigned_bus, BusColor.RED)
        
        # ID=3: (3*7+13)%3 = 34%3 = 1 -> Group B (Yellow)
        p3 = pm.passengers[2]
        self.assertEqual(p3.passenger_id, 3)
        self.assertEqual(p3.group, 'B')
        self.assertEqual(p3.assigned_bus, BusColor.YELLOW)
    
    def test_deterministic_names(self):
        """Test that names are generated deterministically"""
        from bus_escape_csp import PassengerManager
        
        pm1 = PassengerManager(50)
        pm2 = PassengerManager(50)
        
        # Check first 5 passengers have same names
        for i in range(5):
            self.assertEqual(pm1.passengers[i].name, pm2.passengers[i].name)
    
    def test_get_passengers_by_bus(self):
        """Test filtering passengers by bus"""
        from bus_escape_csp import PassengerManager, BusColor
        
        pm = PassengerManager(50)
        
        red_passengers = pm.get_passengers_by_bus(BusColor.RED)
        yellow_passengers = pm.get_passengers_by_bus(BusColor.YELLOW)
        green_passengers = pm.get_passengers_by_bus(BusColor.GREEN)
        
        # All should have correct bus assignment
        for p in red_passengers:
            self.assertEqual(p.assigned_bus, BusColor.RED)
            self.assertEqual(p.group, 'A')
        
        for p in yellow_passengers:
            self.assertEqual(p.assigned_bus, BusColor.YELLOW)
            self.assertEqual(p.group, 'B')
        
        for p in green_passengers:
            self.assertEqual(p.assigned_bus, BusColor.GREEN)
            self.assertEqual(p.group, 'C')
        
        # Total should be 50
        self.assertEqual(len(red_passengers) + len(yellow_passengers) + len(green_passengers), 50)


class TestEnhancedSolver(unittest.TestCase):
    """Test enhanced CSP solver with passenger management"""
    
    def test_enhanced_solver_initialization(self):
        """Test that enhanced solver initializes correctly"""
        from bus_escape_csp import EnhancedBusEscapeCSP, Bus, BusColor, Orientation
        
        buses = [
            Bus(BusColor.RED, 2, Orientation.HORIZONTAL, (0, 0)),
            Bus(BusColor.GREEN, 2, Orientation.VERTICAL, (2, 2)),
        ]
        
        csp = EnhancedBusEscapeCSP(buses, total_passengers=50)
        
        # Check passenger manager is initialized
        self.assertIsNotNone(csp.passenger_manager)
        self.assertEqual(len(csp.passenger_manager.passengers), 50)
    
    def test_simple_solvable_puzzle(self):
        """Test that enhanced solver can solve simple puzzle"""
        from bus_escape_csp import create_solvable_puzzle, EnhancedBusEscapeCSP
        
        buses = create_solvable_puzzle()
        csp = EnhancedBusEscapeCSP(buses, total_passengers=50)
        
        result = csp.solve_with_red_priority()
        
        # Should find solution
        self.assertTrue(result)
        self.assertTrue(csp.is_goal_state(csp.buses))
    
    def test_complex_puzzle_with_blocking(self):
        """Test that enhanced solver can solve puzzle with blocking buses"""
        from bus_escape_csp import create_complex_solvable_puzzle, EnhancedBusEscapeCSP
        
        buses = create_complex_solvable_puzzle()
        csp = EnhancedBusEscapeCSP(buses, total_passengers=50)
        
        result = csp.solve_with_red_priority()
        
        # Should find solution
        self.assertTrue(result)
        self.assertTrue(csp.is_goal_state(csp.buses))
        
        # Should have moved buses (more than just Red)
        self.assertGreater(len(csp.move_log), 1)
    
    def test_all_constraints_maintained(self):
        """Test that all constraints are maintained during enhanced solving"""
        from bus_escape_csp import create_complex_solvable_puzzle, EnhancedBusEscapeCSP
        
        buses = create_complex_solvable_puzzle()
        csp = EnhancedBusEscapeCSP(buses, total_passengers=50)
        
        # Save initial state
        initial_buses = [b.copy() for b in buses]
        
        # Solve
        csp.solve_with_red_priority()
        
        # Verify all buses are still within bounds
        for bus in csp.buses:
            for cell in bus.get_occupied_cells():
                row, col = cell
                self.assertGreaterEqual(row, 0)
                self.assertLess(row, 6)
                self.assertGreaterEqual(col, 0)
                self.assertLess(col, 6)
        
        # Verify no collisions in final state
        all_cells = []
        for bus in csp.buses:
            for cell in bus.get_occupied_cells():
                self.assertNotIn(cell, all_cells, "Collision detected in final state")
                all_cells.append(cell)


if __name__ == '__main__':
    unittest.main()
