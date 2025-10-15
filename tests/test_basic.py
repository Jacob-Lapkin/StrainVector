"""
Basic tests for StrainVector core functionality.

These tests verify that the main modules can be imported and basic
data structures are correctly initialized.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all main modules can be imported."""
    try:
        import strainvector
        import strainvector.embeddings
        import strainvector.indexing
        import strainvector.profiling
        import strainvector.compare
        import strainvector.io
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_embeddings_factory():
    """Test that the embeddings factory can be accessed."""
    try:
        from strainvector.embeddings import Embedder, get_embedder

        # Check that factory function exists
        assert callable(get_embedder), "get_embedder is not callable"
        assert Embedder is not None, "Embedder class not found"
        print("✓ Embeddings factory structure OK")
        return True
    except Exception as e:
        print(f"✗ Embeddings factory test failed: {e}")
        return False


def test_indexing_module():
    """Test that indexing module has expected components."""
    try:
        from strainvector import indexing

        # Check that module loaded successfully
        assert indexing is not None, "Indexing module not found"
        print("✓ Indexing module structure OK")
        return True
    except Exception as e:
        print(f"✗ Indexing module test failed: {e}")
        return False


def test_io_module():
    """Test that I/O module can be imported."""
    try:
        from strainvector import io

        # Check that module loaded successfully
        assert io is not None, "I/O module not found"
        print("✓ I/O module structure OK")
        return True
    except Exception as e:
        print(f"✗ I/O module test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Running StrainVector basic tests...\n")

    tests = [
        ("Module imports", test_imports),
        ("Embeddings factory", test_embeddings_factory),
        ("Indexing module", test_indexing_module),
        ("I/O module", test_io_module),
    ]

    results = []
    for name, test_func in tests:
        print(f"Testing {name}...")
        result = test_func()
        results.append(result)
        print()

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"{'='*50}")
    print(f"Test Results: {passed}/{total} passed")
    print(f"{'='*50}")

    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
