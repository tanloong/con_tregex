import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(file_dir)

sys.path.insert(0, os.path.join(project_dir, "src"))
