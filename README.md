# Matrix Operations Implementation

A high-performance Python-based matrix operations implementation with a modern graphical user interface. This project features NumPy-optimized operations, real-time visualization, and advanced memory management.

## Features

### Core Matrix Operations
- Custom 2D Matrix class with NumPy-like functionality
- Support for various matrix operations:
  - Addition (`+`) with broadcasting
  - Subtraction (`-`) with broadcasting
  - Element-wise multiplication (`*`) with broadcasting
  - Matrix multiplication (`@`) for dot product
  - Power operation (`**`) for element-wise exponentiation
  - Transpose operation

### Performance Optimizations
- NumPy vectorized operations
- Memory-efficient implementation using `__slots__`
- JIT compilation for critical paths
- LRU cache for repeated operations
- Parallel processing for large matrices
- Advanced memory management with object pooling

### User Interface
- Modern, responsive GUI design
- Real-time matrix visualization with heatmaps
- Performance monitoring and metrics
- Intuitive operation controls
- Error handling with user feedback

## Requirements

### System Requirements
- Python 3.x
- Operating System: Windows/Linux/macOS

### Python Dependencies
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Numba >= 0.56.0
- CuPy (optional, for GPU acceleration)
- Tkinter (built-in)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd matrix-operations
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
1. Start the GUI:
```bash
python main.py
```

### Using the Interface
1. Input Matrices:
   - Enter matrix A in the left text area
   - Enter matrix B in the right text area
   - Use space-separated values, one row per line
   Example:
   ```
   1 2 3
   4 5 6
   7 8 9
   ```

2. Perform Operations:
   - Click operation buttons to perform calculations
   - View results in the output area
   - Check performance metrics
   - Analyze matrix visualization

3. Available Operations:
   - Add: Matrix addition
   - Subtract: Matrix subtraction
   - Multiply: Element-wise multiplication
   - Matrix Multiply: Dot product
   - Power: Element-wise exponentiation
   - Transpose: Matrix transposition

## Performance Features

### Memory Management
- Object pooling for efficient memory usage
- Automatic garbage collection
- Memory usage tracking and statistics

### Optimization Techniques
- NumPy vectorization
- Numba JIT compilation
- Parallel processing for large matrices
- Caching for repeated operations

## Project Structure
```
matrix-operations/
├── main.py              # Application entry point
├── matrix.py           # Core Matrix class
├── memory_manager.py   # Memory management
├── optimizations.py    # Optimized operations
├── gui.py             # GUI implementation
└── requirements.txt    # Project dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open source and available under the MIT License.

## Running the Project

To run the project, follow these steps:

1. Make sure you have Python 3.x installed
2. Install all required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python main.py
```

The GUI will open, allowing you to perform matrix operations. Enter your matrices in the text areas (space-separated values, one row per line) and use the operation buttons to perform calculations. 
