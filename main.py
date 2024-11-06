from population.genetic_algorithm import generate_new_population
from evaluators.parameter_estimation import *
from population.initial_generation import generate_population, manual_lotka_systems
from utils import models
from utils.functions import get_functions

from utils.load_systems import create_ode_function, load_systems
from utils.plots import plot_2d_by_func, plot_2d_by_y


def main():
    #######################################################################
    #                         PARAMETERS                                  #
    #######################################################################

    target_func, num_equations, true_betas = models.lotka()
    ITERATIONS = 1  # Number of generations
    N = 5  # Maximum number of population
    M = 2  # Maximum number of equations
    I = 2  # Maximum number of terms per equation
    J = 2  # Maximum number of functions per feature
    allow_composite = False  # Composite Functions
    f0ps = get_functions("5")  # Linear

    system_load_dir = 'data/differential_equations.txt'  # 'data/lotka_equations.txt'
    system_save_dir = 'data/differential_equations.txt'
    ivp_method = 'BDF'
    DEBUG = False

    #######################################################################
    #                         TARGET DATA                                 #
    #######################################################################

    t = np.linspace(0, 10, 100)
    X0 = np.random.rand(num_equations)
    print(f"true_betas: {true_betas} | Initial Condition: {X0}")

    y_raw = solve_ivp(target_func, (t[0], t[-1]), X0, args=true_betas, t_eval=t, method='BDF').y.T
    y_target = y_raw + np.random.normal(0, 0.01, y_raw.shape)
    if DEBUG:
        print(f"y_raw: {y_raw.shape} | y_target: {y_target.shape}")
        plot_2d_by_func(target_func, true_betas)
        plot_2d_by_y(y_raw, "RAW", y_target, "NOISED")

    #######################################################################
    #                         INITIAL POPULATION                          #
    #######################################################################

    population = generate_population(N, M, I, J, f0ps, allow_composite, system_save_dir,
                                     DEBUG)  # manual_lotka_systems()
    systems = load_systems(system_load_dir)

    #######################################################################
    #                         RUN                                     #
    #######################################################################

    best_model = None
    best_error = float('inf')
    for i in range(ITERATIONS):
        scores = []
        for j, system in enumerate(systems):
            ode_func = create_ode_function(system)
            estimated_betas = estimate_parameters(ode_func, X0, t, y_target, true_betas, ivp_method, DEBUG)
            y_pred = simulate_system(ode_func, X0, t, estimated_betas, ivp_method)
            error = calculate_error(y_pred, y_target, DEBUG)
            scores.append(error)
            if error < best_error:
                best_error = error
                best_model = ode_func
            print(f"### Generation {i} | System {j} | Error: {error} | Estimated parameters: {estimated_betas}")

        # genetic algorithm
        if i < ITERATIONS - 1:
            population = generate_new_population(scores, population)
            systems = load_systems(system_load_dir)

    #######################################################################
    #                         RESULT                                      #
    #######################################################################

    estimated_betas = estimate_parameters(best_model, X0, t, y_target, true_betas, ivp_method, DEBUG)
    best_pred = simulate_system(best_model, X0, t, estimated_betas, ivp_method).y.T
    plot_2d_by_y(y_target, "Target", best_pred, "Prediction")


if __name__ == "__main__":
    main()
