from functions import get_allowed_functions
from equation_generation import generate_system, beautify_system
from numpy_conversion import save_system_as_numpy_funcs

# Get allowed functions from user
fOps = get_allowed_functions()

# Example usage
n = 3  # Number of equations
m = 3  # Maximum number of terms in a particular differential equation
k = 2  # Maximum number of functions per feature in a differential equation
allow_composite = False  # Set to False if you don't want composite functions

system = generate_system(n, m, k, fOps, allow_composite)

# Print the beautified equations
beautified_equations = beautify_system(system)
for eq in beautified_equations:
    print(eq)

# Save the system as NumPy functions
save_system_as_numpy_funcs(system, "differential_equations.txt")
print("\nSystem saved as NumPy functions in 'differential_equations.txt'")
