import tkinter as tk
import numpy as np
import os
import multiprocessing
import warnings

from gui import MatrixGUI

def main():
    # Configure NumPy for optimal performance
    np.set_printoptions(precision=8, suppress=True)
    np.seterr(divide='ignore', invalid='ignore')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Enable threading optimizations
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(multiprocessing.cpu_count())

    # Create and run the GUI
    root = tk.Tk()
    app = MatrixGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 