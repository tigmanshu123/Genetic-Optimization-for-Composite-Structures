# Genetic Optimization for Composite Structures

This repository contains a Python implementation of a genetic algorithm for optimizing composite structures, specifically for aircraft applications. The project includes scripts for running the genetic algorithm, a Jupyter Notebook interface for interactive exploration, and a virtual environment for managing dependencies.

## Project Structure
.
├── Function Definitions.py
├── Genetic-Optimization-for-Composite-Structures/
│   ├── LICENSE
├── Jupyter Notebook Interface.ipynb
├── myenv/
│   ├── Include/
│   ├── Lib/
│   │   ├── site-packages/
│   ├── pyvenv.cfg
│   ├── Scripts/
│   │   ├── activate
│   │   ├── activate.bat
│   │   ├── Activate.ps1
│   │   ├── deactivate.bat
│   │   ├── pip.exe
│   │   ├── python.exe
│   ├── share/
├── parameters.yml



### Files and Directories

- **Function Definitions.py**: Contains the main implementation of the genetic algorithm, including functions for crossover, mutation, and fitness evaluation.
- **Genetic-Optimization-for-Composite-Structures/**: Directory containing the license file.
- **Jupyter Notebook Interface.ipynb**: Jupyter Notebook for interactive exploration and visualization of the genetic algorithm.
- **myenv/**: Virtual environment directory containing Python executables and installed packages.
- **parameters.yml**: YAML file containing configuration parameters for the genetic algorithm.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `virtualenv` for creating isolated Python environments

### Setting Up the Environment

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv myenv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```sh
        myenv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source myenv/bin/activate
        ```

4. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Genetic Algorithm

1. **Edit the file:///g%3A/My%20Drive/Tigmanshu%20Documents/Learning/My%20GitHub%20Files/My%20Coding%20Projects/Boeing%20Internship%202012/parameters.yml file** to configure the genetic algorithm parameters such as population size, crossover probability, mutation rate, etc.

2. **Run the script**:
    ```sh
    python "Function Definitions.py"
    ```

### Using the Jupyter Notebook Interface

1. **Activate the virtual environment** (if not already activated):
    ```sh
    source myenv/bin/activate
    ```

2. **Launch Jupyter Notebook**:
    ```sh
    jupyter notebook
    ```

3. **Open `Jupyter Notebook Interface.ipynb`** and run the cells to interactively explore and visualize the genetic algorithm.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The genetic algorithm implementation is inspired by various academic and industry research on composite structure optimization.
- Special thanks to the open-source community for providing the tools and libraries that made this project possible.

## Contact

For any questions or inquiries, please contact Tigmanshu Goyal at tigmanshu123@gmail.com

---

Feel free to customize this README file according to your specific needs and project details.
