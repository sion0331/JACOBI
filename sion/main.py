from functions import get_allowed_functions
from equation_generation import generate_systems, beautify_system
from numpy_conversion import save_systems_as_numpy_funcs

# Get allowed functions from user
fOps = get_allowed_functions()

# Example usage
N = 3  # Maximum number of systems
M = 2  # Maximum number of equations
I = 2  # Maximum number of terms per equation
J = 2  # Maximum number of functions per feature
allow_composite = False  # Set to False if you don't want composite functions

systems = generate_systems(N, M, I, J, fOps, allow_composite)
for i, system in enumerate(systems):
    print(f"generate_system {i}: {system}")

beautified_systems = beautify_system(systems)
for i, system in enumerate(beautified_systems):
    print(f"beautified_systems {i}: {system}")

# Save the system as NumPy functions
save_systems_as_numpy_funcs(systems, "sion/differential_equations.txt")
print("\nSystem saved as NumPy functions in 'differential_equations.txt'")
