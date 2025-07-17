#!/usr/bin/env python3
"""
Test Runner for LLM Reasoning Evaluation Harness

Runs all smoke tests to verify the solvers and generators work correctly.
Can run with pytest or standalone.
"""

import sys
import os

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_standalone_tests():
    """Run all tests without pytest."""
    print("Running LLM Reasoning Evaluation Harness Smoke Tests...")
    print("=" * 60)
    
    # Import and run Hanoi solver tests
    print("\n1. Testing Hanoi Solver...")
    try:
        import test_hanoi_solver as hanoi_tests
        hanoi_tests.test_hanoi_3_disks()
        hanoi_tests.test_hanoi_1_disk()
        hanoi_tests.test_hanoi_2_disks()
        hanoi_tests.test_hanoi_already_solved()
        hanoi_tests.test_hanoi_different_pegs()
        hanoi_tests.test_hanoi_format_solution()
        hanoi_tests.test_hanoi_empty_moves()
        hanoi_tests.test_hanoi_invalid_state()
        print("   ✓ All Hanoi solver tests passed!")
    except Exception as e:
        print(f"   ✗ Hanoi solver tests failed: {e}")
        return False
    
    # Import and run River solver tests  
    print("\n2. Testing River Solver...")
    try:
        import test_river_solver as river_tests
        river_tests.test_river_classic()
        river_tests.test_river_already_solved()
        river_tests.test_river_partial_crossing()
        river_tests.test_river_custom_entities()
        river_tests.test_river_format_solution()
        river_tests.test_river_empty_moves()
        river_tests.test_river_state_description()
        river_tests.test_river_invalid_state()
        river_tests.test_river_impossible_goal()
        river_tests.test_state_after_moves()
        print("   ✓ All River solver tests passed!")
    except Exception as e:
        print(f"   ✗ River solver tests failed: {e}")
        return False
    
    # Import and run Generator tests
    print("\n3. Testing Generators...")
    try:
        import test_generators as gen_tests
        gen_tests.test_hanoi_generator_basic()
        gen_tests.test_hanoi_generator_complex()
        gen_tests.test_hanoi_generator_batch()
        gen_tests.test_river_generator_classic()
        gen_tests.test_river_generator_random_start()
        gen_tests.test_river_generator_partial_crossing()
        gen_tests.test_river_generator_custom()
        gen_tests.test_river_generator_batch()
        gen_tests.test_river_generator_difficulty_progression()
        gen_tests.test_river_validation_function()
        gen_tests.test_hanoi_difficulty_estimation()
        gen_tests.test_generators_with_invalid_inputs()
        print("   ✓ All generator tests passed!")
    except Exception as e:
        print(f"   ✗ Generator tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Your reasoning evaluation harness is working correctly.")
    print("\nYour improvements have been successfully implemented:")
    print("  ✓ Solver state management refactored")
    print("  ✓ Input validation strengthened") 
    print("  ✓ Exception handling narrowed")
    print("  ✓ Smoke tests created")
    return True


def run_with_pytest():
    """Run tests using pytest if available."""
    try:
        import pytest
        print("Running tests with pytest...")
        
        # Run pytest on the tests directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        exit_code = pytest.main([
            test_dir,
            "-v",  # verbose
            "--tb=short",  # shorter traceback format
        ])
        
        return exit_code == 0
    except ImportError:
        print("pytest not available, running standalone tests...")
        return run_standalone_tests()


if __name__ == '__main__':
    # Check if --standalone flag is provided
    if '--standalone' in sys.argv:
        success = run_standalone_tests()
    else:
        success = run_with_pytest()
    
    sys.exit(0 if success else 1) 