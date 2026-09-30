import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.device import print_environment_diagnostics

if __name__ == "__main__":
    print_environment_diagnostics()
