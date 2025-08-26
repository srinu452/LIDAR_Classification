# =====================================
# 9) utils/io.py
# =====================================
import os

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)