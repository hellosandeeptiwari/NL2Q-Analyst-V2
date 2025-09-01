#!/usr/bin/env python3
"""
Comprehensive test runner for NL2Q Agent
Executes all test suites with various configurations
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print('='*60)

    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).parent)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Main test runner function"""
    print("NL2Q Agent - Comprehensive Test Suite")
    print("=====================================")

    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Test configurations
    test_configs = [
        {
            "name": "Unit Tests Only",
            "command": ["pytest", "-m", "unit", "--cov-fail-under=85"],
            "description": "Running unit tests with 85% coverage requirement"
        },
        {
            "name": "API Tests Only",
            "command": ["pytest", "-m", "api", "--cov-fail-under=90"],
            "description": "Running API endpoint tests with 90% coverage requirement"
        },
        {
            "name": "Component Tests Only",
            "command": ["pytest", "-m", "component", "--cov-fail-under=80"],
            "description": "Running component tests with 80% coverage requirement"
        },
        {
            "name": "Database Tests Only",
            "command": ["pytest", "-m", "database", "--cov-fail-under=85"],
            "description": "Running database tests with 85% coverage requirement"
        },
        {
            "name": "Performance Tests",
            "command": ["pytest", "-m", "performance", "--durations=10"],
            "description": "Running performance tests with timing information"
        },
        {
            "name": "Integration Tests",
            "command": ["pytest", "-m", "integration", "--cov-fail-under=75"],
            "description": "Running integration tests with 75% coverage requirement"
        },
        {
            "name": "Smoke Tests",
            "command": ["pytest", "-m", "smoke", "--tb=line"],
            "description": "Running smoke tests for quick validation"
        },
        {
            "name": "All Tests with Coverage",
            "command": ["pytest", "--cov-report=html:htmlcov", "--cov-report=term-missing"],
            "description": "Running all tests with detailed coverage report"
        },
        {
            "name": "Slow Tests Only",
            "command": ["pytest", "-m", "slow", "--durations=0"],
            "description": "Running slow tests with detailed timing"
        }
    ]

    # Run all test configurations
    results = []
    for config in test_configs:
        success = run_command(config["command"], config["description"])
        results.append((config["name"], success))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)

    passed = 0
    failed = 0

    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name:.<50} {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {len(results)} test suites")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\n❌ {failed} test suite(s) failed")
        return 1
    else:
        print(f"\n✅ All {passed} test suites passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
