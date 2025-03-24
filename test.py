import sys
import os

# หา path ของ Algorithm แล้วเพิ่มเข้า sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithm_path = os.path.join(current_dir, "..", "..", "RL_Algorithm", "Algorithm")
sys.path.append(algorithm_path)