#!/usr/bin/env python3
"""
Unit Tests for Autoplacement Script v2.0

Run with: pytest test_autoplace.py -v
"""

import pytest
import math
from autoplace_v2 import (
    Component, Rectangle, Quadtree, ForceDirectedConfig,
    ForceDirectedPlacer, validate_config, ConfigValidationError,
    get_component_size, COMPONENT_SIZES
)


# ============================================================================
# RECTANGLE TESTS
# ============================================================================

class TestRectangle:
    def test_properties(self):
        rect = Rectangle(x=5, y=5, width=4, height=2)
        assert rect.left == 3
        assert rect.right == 7
        assert rect.top == 6
        assert rect.bottom == 4
    
    def test_contains_point(self):
        rect = Rectangle(x=0, y=0, width=10, height=10)
        assert rect.contains(0, 0) is True
        assert rect.contains(4, 4) is True
        assert rect.contains(-5, 0) is True
        assert rect.contains(6, 6) is False  # Outside
    
    def test_intersects(self):
        r1 = Rectangle(0, 0, 4, 4)
        r2 = Rectangle(2, 2, 4, 4)  # Overlapping
        r3 = Rectangle(10, 10, 2, 2)  # Non-overlapping
        
        assert r1.intersects(r2) is True
        assert r1.intersects(r3) is False


# ============================================================================
# COMPONENT TESTS
# ============================================================================

class TestComponent:
    def test_bounds(self):
        comp = Component(ref="U1", x=0, y=0, width=10, height=10)
        bounds = comp.get_bounds()
        assert bounds.x == 0
        assert bounds.y == 0
        assert bounds.width == 10
        assert bounds.height == 10
    
    def test_overlaps_same_layer(self):
        c1 = Component(ref="U1", x=0, y=0, width=4, height=4, layer="front")
        c2 = Component(ref="U2", x=2, y=2, width=4, height=4, layer="front")
        c3 = Component(ref="U3", x=10, y=10, width=4, height=4, layer="front")
        
        assert c1.overlaps(c2, margin=0) is True
        assert c1.overlaps(c3, margin=0) is False
    
    def test_overlaps_different_layers(self):
        c1 = Component(ref="U1", x=0, y=0, width=4, height=4, layer="front")
        c2 = Component(ref="B1", x=0, y=0, width=4, height=4, layer="back")
        
        assert c1.overlaps(c2, margin=0) is False  # Different layers never overlap
    
    def test_overlaps_with_margin(self):
        c1 = Component(ref="U1", x=0, y=0, width=4, height=4, layer="front")
        c2 = Component(ref="U2", x=5, y=0, width=4, height=4, layer="front")
        
        # Gap of 1mm between edges
        assert c1.overlaps(c2, margin=0) is False
        assert c1.overlaps(c2, margin=1.5) is True  # But margin makes them overlap


# ============================================================================
# QUADTREE TESTS
# ============================================================================

class TestQuadtree:
    def test_insert_and_retrieve(self):
        bounds = Rectangle(0, 0, 100, 100)
        qt = Quadtree(bounds)
        
        # Insert some objects
        qt.insert("A", Rectangle(10, 10, 5, 5))
        qt.insert("B", Rectangle(15, 15, 5, 5))
        qt.insert("C", Rectangle(-40, -40, 5, 5))  # Far corner
        
        # Retrieve near A and B
        query = Rectangle(12, 12, 10, 10)
        results = qt.retrieve(query)
        refs = [r[0] for r in results]
        
        assert "A" in refs or "B" in refs  # Should find nearby objects
    
    def test_clears_properly(self):
        bounds = Rectangle(0, 0, 100, 100)
        qt = Quadtree(bounds)
        
        qt.insert("A", Rectangle(10, 10, 5, 5))
        qt.clear()
        
        results = qt.retrieve(Rectangle(10, 10, 5, 5))
        assert len(results) == 0


# ============================================================================
# FORCE-DIRECTED PLACER TESTS
# ============================================================================

class TestForceDirectedPlacer:
    def test_basic_run(self):
        config = ForceDirectedConfig(
            enabled=True,
            iterations=10,
            random_seed=42
        )
        placer = ForceDirectedPlacer(config, board_width=50, board_height=50)
        
        placer.add_component(Component(ref="U1", x=0, y=0, width=5, height=5))
        placer.add_component(Component(ref="U2", x=0, y=0, width=5, height=5))  # Overlapping
        
        components = placer.run()
        
        # After running, components should be pushed apart
        c1, c2 = components["U1"], components["U2"]
        distance = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
        assert distance > 0  # They should have moved apart
    
    def test_connections_attract(self):
        config = ForceDirectedConfig(
            enabled=True,
            iterations=20,
            attraction_strength=0.5,
            repulsion_strength=10.0,
            random_seed=42
        )
        placer = ForceDirectedPlacer(config, board_width=100, board_height=100)
        
        # Place two components far apart
        placer.add_component(Component(ref="U1", x=-20, y=0, width=5, height=5))
        placer.add_component(Component(ref="U2", x=20, y=0, width=5, height=5))
        
        # Add strong connection
        placer.add_connection("U1", "U2", weight=2.0)
        
        initial_distance = 40
        components = placer.run()
        
        c1, c2 = components["U1"], components["U2"]
        final_distance = abs(c1.x - c2.x)  # Only check X since they're on same Y
        
        assert final_distance < initial_distance  # Should have moved closer
    
    def test_board_constraints(self):
        config = ForceDirectedConfig(
            enabled=True,
            iterations=50,
            repulsion_strength=1000.0,
            random_seed=42
        )
        placer = ForceDirectedPlacer(config, board_width=20, board_height=20)
        
        # Place component that would be pushed out of bounds
        placer.add_component(Component(ref="U1", x=0, y=0, width=5, height=5))
        placer.add_component(Component(ref="U2", x=2, y=0, width=5, height=5))
        
        components = placer.run()
        
        for comp in components.values():
            # Check all components are within board bounds (with margin)
            assert abs(comp.x) <= 20
            assert abs(comp.y) <= 20


# ============================================================================
# CONFIG VALIDATION TESTS
# ============================================================================

class TestConfigValidation:
    def test_valid_config(self):
        config = {
            'board': {'width': 50, 'height': 40},
            'groups': {
                'mcu': {
                    'position': [0, 0],
                    'components': {'U1': {'offset': [0, 0]}}
                }
            }
        }
        # Should not raise
        validate_config(config)
    
    def test_missing_board(self):
        config = {
            'groups': {'mcu': {'position': [0, 0], 'components': {}}}
        }
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(config)
        assert "board" in str(exc.value)
    
    def test_missing_groups(self):
        config = {
            'board': {'width': 50, 'height': 40}
        }
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(config)
        assert "groups" in str(exc.value)
    
    def test_empty_groups(self):
        config = {
            'board': {'width': 50, 'height': 40},
            'groups': {}
        }
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(config)
        assert "empty" in str(exc.value)
    
    def test_group_missing_position(self):
        config = {
            'board': {'width': 50, 'height': 40},
            'groups': {
                'mcu': {'components': {'U1': {}}}
            }
        }
        with pytest.raises(ConfigValidationError) as exc:
            validate_config(config)
        assert "position" in str(exc.value)


# ============================================================================
# COMPONENT SIZE TESTS
# ============================================================================

class TestComponentSizes:
    def test_known_component(self):
        assert get_component_size("U1") == (14.0, 14.0)
        assert get_component_size("B1") == (26.0, 26.0)
    
    def test_resistor_default(self):
        size = get_component_size("R15")
        assert size == COMPONENT_SIZES['_resistor']
    
    def test_capacitor_default(self):
        size = get_component_size("C99")
        assert size == COMPONENT_SIZES['_capacitor']
    
    def test_unknown_component(self):
        size = get_component_size("XYZ123")
        assert size == (2.0, 2.0)  # Default fallback


# ============================================================================
# NETLIST PARSER TESTS
# ============================================================================

class TestNetlistParser:
    def test_classify_power_net(self):
        from autoplace_v2 import NetlistParser
        parser = NetlistParser(None)  # No board needed for classification
        
        net_type, weight = parser.classify_net("VCC_3V3")
        assert net_type == 'power'
        assert weight == 3.0
        
        net_type, weight = parser.classify_net("VBAT")
        assert net_type == 'power'
    
    def test_classify_ground_net(self):
        from autoplace_v2 import NetlistParser
        parser = NetlistParser(None)
        
        net_type, weight = parser.classify_net("GND")
        assert net_type == 'ground'
        assert weight == 2.0
    
    def test_classify_high_speed_net(self):
        from autoplace_v2 import NetlistParser
        parser = NetlistParser(None)
        
        net_type, weight = parser.classify_net("I2C_SDA")
        assert net_type == 'high_speed'
        assert weight == 2.5
        
        net_type, weight = parser.classify_net("SPI_CLK")
        assert net_type == 'high_speed'
    
    def test_classify_signal_net(self):
        from autoplace_v2 import NetlistParser
        parser = NetlistParser(None)
        
        net_type, weight = parser.classify_net("LED_CTRL")
        assert net_type == 'signal'
        assert weight == 1.0


# ============================================================================
# QUADTREE OPTIMIZED COUNT_OVERLAPS TEST
# ============================================================================

class TestQuadtreeOptimizedOverlaps:
    def test_count_overlaps_uses_quadtree(self):
        config = ForceDirectedConfig(enabled=False, random_seed=42)
        placer = ForceDirectedPlacer(config, board_width=50, board_height=50)
        
        # Add overlapping components
        placer.add_component(Component(ref="U1", x=0, y=0, width=5, height=5))
        placer.add_component(Component(ref="U2", x=2, y=2, width=5, height=5))  # Overlaps U1
        placer.add_component(Component(ref="U3", x=20, y=20, width=5, height=5))  # No overlap
        
        count = placer.count_overlaps()
        assert count == 1  # Only U1-U2 overlap
    
    def test_count_overlaps_no_double_count(self):
        config = ForceDirectedConfig(enabled=False, random_seed=42)
        placer = ForceDirectedPlacer(config, board_width=50, board_height=50)
        
        # Triangle of overlaps
        placer.add_component(Component(ref="A", x=0, y=0, width=4, height=4))
        placer.add_component(Component(ref="B", x=2, y=0, width=4, height=4))
        placer.add_component(Component(ref="C", x=1, y=2, width=4, height=4))
        
        count = placer.count_overlaps()
        # A-B overlap, A-C overlap, B-C overlap = 3 pairs
        assert count == 3


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
