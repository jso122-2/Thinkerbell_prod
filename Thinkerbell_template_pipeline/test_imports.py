#!/usr/bin/env python3
"""
Test script to debug import issues
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

print("Current working directory:", os.getcwd())
print("Python path:", sys.path[:3])

# Test imports
try:
    from thinkerbell.data.synthetic_dataset_generator import SyntheticDatasetGenerator
    print("✅ SyntheticDatasetGenerator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SyntheticDatasetGenerator: {e}")

try:
    from thinkerbell.data.individual_file_generator import IndividualFileGenerator
    print("✅ IndividualFileGenerator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import IndividualFileGenerator: {e}")

try:
    from thinkerbell.data.dataset_loader import DatasetLoader
    print("✅ DatasetLoader imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DatasetLoader: {e}")

try:
    from thinkerbell.data.dataset_validation import DatasetValidator
    print("✅ DatasetValidator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DatasetValidator: {e}")

try:
    from thinkerbell.utils.ood_contamination import OODContaminator
    print("✅ OODContaminator imported successfully")
except ImportError as e:
    print(f"❌ Failed to import OODContaminator: {e}")

try:
    from thinkerbell.utils.semantic_smoother import SemanticSmoother
    print("✅ SemanticSmoother imported successfully")
except ImportError as e:
    print(f"❌ Failed to import SemanticSmoother: {e}")

try:
    from thinkerbell.utils.text_preprocessor import TextPreprocessor
    print("✅ TextPreprocessor imported successfully")
except ImportError as e:
    print(f"❌ Failed to import TextPreprocessor: {e}") 