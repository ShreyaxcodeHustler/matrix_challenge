import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import time
from matrix import Matrix

class MatrixGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Operations")
        self.root.geometry("1400x900")
        
        # Set theme colors
        self.bg_color = "#f0f0f0"
        self.accent_color = "#4a90e2"
        
        # Initialize variables
        self.matrix_a = None
        self.matrix_b = None
        self.result_matrix = None
        
        # Setup GUI components
        self.setup_input_fields()
        self.setup_operation_buttons()
        self.setup_output_area()
        self.setup_visualization()
        
    def setup_input_fields(self):
        # Matrix A
        frame_a = ttk.LabelFrame(self.root, text="Matrix A")
        frame_a.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.matrix_a_text = scrolledtext.ScrolledText(frame_a, width=30, height=10)
        self.matrix_a_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Matrix B
        frame_b = ttk.LabelFrame(self.root, text="Matrix B")
        frame_b.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.matrix_b_text = scrolledtext.ScrolledText(frame_b, width=30, height=10)
        self.matrix_b_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
    def setup_operation_buttons(self):
        frame_ops = ttk.Frame(self.root)
        frame_ops.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X)
        
        ttk.Button(frame_ops, text="Add", command=self.add_matrices).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_ops, text="Subtract", command=self.subtract_matrices).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_ops, text="Multiply", command=self.multiply_matrices).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_ops, text="Matrix Multiply", command=self.matrix_multiply).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_ops, text="Power", command=self.power_matrix).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_ops, text="Transpose", command=self.transpose_matrix).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_ops, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
    def setup_output_area(self):
        frame_output = ttk.LabelFrame(self.root, text="Results")
        frame_output.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(frame_output, width=60, height=10)
        self.result_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
    def setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
    def update_visualization(self, matrix_data):
        self.ax.clear()
        sns.heatmap(matrix_data, ax=self.ax, cmap='viridis')
        self.canvas.draw()
        
    def parse_matrix(self, text):
        try:
            rows = [row.strip().split() for row in text.strip().split('\n')]
            return Matrix([[float(x) for x in row] for row in rows])
        except Exception as e:
            self.show_error(f"Error parsing matrix: {str(e)}")
            return None
            
    def show_error(self, message):
        messagebox.showerror("Error", message)
        
    def measure_performance(self, operation_func):
        start_time = time.time()
        result = operation_func()
        end_time = time.time()
        
        if result:
            self.result_matrix = result
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Result:\n{result.format_matrix()}\n\n")
            self.result_text.insert(tk.END, f"Time taken: {(end_time - start_time)*1000:.2f} ms\n")
            self.result_text.insert(tk.END, f"Memory usage: {result.get_memory_usage()['matrix_size']:.2f} MB")
            self.update_visualization(result._data)
            
    def add_matrices(self):
        def operation():
            self.matrix_a = self.parse_matrix(self.matrix_a_text.get(1.0, tk.END))
            self.matrix_b = self.parse_matrix(self.matrix_b_text.get(1.0, tk.END))
            if self.matrix_a and self.matrix_b:
                return self.matrix_a + self.matrix_b
        self.measure_performance(operation)
        
    def subtract_matrices(self):
        def operation():
            self.matrix_a = self.parse_matrix(self.matrix_a_text.get(1.0, tk.END))
            self.matrix_b = self.parse_matrix(self.matrix_b_text.get(1.0, tk.END))
            if self.matrix_a and self.matrix_b:
                return self.matrix_a - self.matrix_b
        self.measure_performance(operation)
        
    def multiply_matrices(self):
        def operation():
            self.matrix_a = self.parse_matrix(self.matrix_a_text.get(1.0, tk.END))
            self.matrix_b = self.parse_matrix(self.matrix_b_text.get(1.0, tk.END))
            if self.matrix_a and self.matrix_b:
                return self.matrix_a * self.matrix_b
        self.measure_performance(operation)
        
    def matrix_multiply(self):
        def operation():
            self.matrix_a = self.parse_matrix(self.matrix_a_text.get(1.0, tk.END))
            self.matrix_b = self.parse_matrix(self.matrix_b_text.get(1.0, tk.END))
            if self.matrix_a and self.matrix_b:
                return self.matrix_a @ self.matrix_b
        self.measure_performance(operation)
        
    def power_matrix(self):
        def operation():
            self.matrix_a = self.parse_matrix(self.matrix_a_text.get(1.0, tk.END))
            if self.matrix_a:
                power = int(self.matrix_b_text.get(1.0, tk.END).strip())
                return self.matrix_a ** power
        self.measure_performance(operation)
        
    def transpose_matrix(self):
        def operation():
            self.matrix_a = self.parse_matrix(self.matrix_a_text.get(1.0, tk.END))
            if self.matrix_a:
                return self.matrix_a.transpose()
        self.measure_performance(operation)
        
    def clear_all(self):
        self.matrix_a_text.delete(1.0, tk.END)
        self.matrix_b_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()
        if self.result_matrix:
            self.result_matrix.clear_cache() 