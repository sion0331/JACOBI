from population.genetic_algorithm import generate_new_population
from evaluators.parameter_estimation import *
from population.initial_generation import generate_population, beautify_system, manual_sir_systems, \
    manual_lorenz_systems, manual_lotka_systems
from utils.functions import get_functions, funcs_to_str
from utils.history import save_history

from utils.load_systems import create_ode_function, load_systems
from utils.mapping import get_individual_solved, add_individual_solved, \
    get_solved_map, get_term_map, convert_system_to_hash, count_betas
from utils.models import SIR, lorenz
from utils.plots import plot_loss_by_iteration, plot_invalid_by_iteration, \
    plot_3d_by_y, plot_lorenz_3d
import matplotlib.pyplot as plt
import warnings

class Config:
    def __init__(self):
        self.target = lorenz()

        self.G = 20  # Number of generations
        self.N = 300  # Maximum number of population
        self.M = 3  # Maximum number of equations
        self.I = 3  # Maximum number of terms per equation
        self.J = 2  # Maximum number of functions per feature
        self.allow_composite = False  # Composite Functions
        self.f0ps = get_functions("5,4,2,1")
        self.ivp_method = 'Radau' # RK45 RK23 DOP853 Radau BDF LSODA
        self.minimize_method = 'BFGS' #'Nelder-Mead' # L-BFGS-B, COBYLA, COBYQA, TNC

        self.elite_rate = 0.1
        self.crossover_rate = 0.2
        self.mutation_rate = 0.6
        self.new_rate = 0.1

        self.selection_gamma = 0.9

        self.system_load_dir = 'data/differential_equations.txt' # 'data/results/lorenz/lorenz_equations.txt'
        self.system_save_dir = 'data/differential_equations.txt'

        self.DEBUG = False


def main():
    #######################################################################
    #                         PARAMETERS                                  #
    #######################################################################

    config = Config()

    #######################################################################
    #                         TARGET DATA                                 #
    #######################################################################

    t = np.linspace(0, 25, 1000)
    X0 = config.target.X0  # np.random.rand(config.target.N) + 1.0  # 1.0~2.0
    print(f"true_betas: {config.target.betas} | Initial Condition: {X0}")

    y_raw = solve_ivp(config.target.func, (t[0], t[-1]), X0, args=config.target.betas, t_eval=t,
                      method=config.ivp_method).y.T
    y_target = y_raw + np.random.normal(0.0, 0.2, y_raw.shape)

    #######################################################################
    #                         INITIAL POPULATION                          #
    #######################################################################

    population = generate_population(config) # manual_lorenz_systems()
    systems = load_systems(config.system_load_dir)

    #######################################################################
    #                         RUN                                     #
    #######################################################################

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        history = []
        time_records = []
        for i in range(config.G):
            print("#### SYSTEM EVALUATION ####")

            start_time = time.time()
            history.append([])
            for j, system in enumerate(systems):
                system_hash = convert_system_to_hash(population[j])
                solved = get_individual_solved(system_hash)

                if not solved:
                    ode_func = create_ode_function(system)
                    num_betas = count_betas(population[j])
                    initial_guess = np.zeros(num_betas)
                    initial_guess = np.concatenate([[-10, 10, 28, -1, -1, 1, -8/3][:num_betas], np.zeros(max(0, num_betas - 7))])
                    solved = estimate_parameters(ode_func, X0, t, y_target, initial_guess , config.minimize_method,
                                                 config.ivp_method, config.DEBUG)
                    add_individual_solved(system_hash, solved)

                history[i].append((solved, population[j], ode_func))
                print(
                    f"### Generation {i} #{j} | Error: {round(solved.fun, 4)} | System: {beautify_system(population[j])} | Solved Parameters: {solved.x}")

            end_time = time.time()
            time_records.append(end_time-start_time)
            print(f"Completed Generation {i}: {end_time-start_time} | term_map:{len(get_term_map())}, solved_map:{len(get_solved_map())}")
            if i < config.G - 1:
                population = generate_new_population(history[i], population, config)
                systems = load_systems(config.system_load_dir)

    #######################################################################
    #                         RESULT                                      #
    #######################################################################

    save_history(config, history, time_records)

    # print("\n#### RESULT ####")
    best = None
    min_loss = []
    avg_loss = []
    invalid = []
    for i, generation in enumerate(history):
        loss = []
        for j, individual in enumerate(generation):
            if config.DEBUG: print(
                f'generation {i} {j} | loss:{round(individual[0].fun, 4)} func: {beautify_system(individual[1])} param:{individual[0].x}')
            loss.append(individual[0].fun)
            if best is None or individual[0].fun < best[0].fun:
                best = individual
        min_loss.append(min(loss))
        avg_loss.append(np.mean([l for l in loss if l < 10000]))  # Exclude loss > 100
        invalid.append(sum(l >= 10000 for l in loss)) # todo - check inf

    print(f'\nBest | Loss:{best[0].fun} func: {best[1]} param:{best[0].x}')

    y_best = solve_ivp(best[2], (t[0], t[-1]), X0, args=tuple(best[0].x), t_eval=t, method=config.ivp_method).y.T
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    plot_3d_by_y(axs[0, 0], t, y_target, [y_best], ["Best"]) ### SIR
    # plot_2d_by_func(axs[0, 0], config.target.func, config.target.betas) ### Lotka
    # plot_2d_by_y(axs[0, 1], X0,[y_raw, y_target, y_best], ["TARGET_RAW", "TARGET_NOISED", "BEST"])
    plot_loss_by_iteration(axs[1, 0], min_loss, avg_loss)
    plot_invalid_by_iteration(axs[1, 1], invalid)

    note = f""" Target:{type(config.target).__name__} | G:{config.G} N:{config.N} M:{config.M} I:{config.I} J:{config.J} f0ps:{funcs_to_str(config.f0ps)} Composite:{config.allow_composite} | elite:{config.elite_rate} new:{config.new_rate} cross:{config.crossover_rate} mutate:{config.mutation_rate}| ivp:{config.ivp_method} min:{config.minimize_method}
    Best Function: {beautify_system(best[1])}
    Best Loss: {best[0]['fun']} Best Parameters: {best[0]['x']}"""
    fig.text(0.03, 0.08, note, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

    plot_lorenz_3d(axs[1, 1], t, X0, y_target, [y_best], ["Best Estimate"], "Lorenz System Estimates")

if __name__ == "__main__":
    main()
