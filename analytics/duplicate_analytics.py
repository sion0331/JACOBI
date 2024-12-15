import time
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import random as rd

from utils.functions import get_functions, funcs_to_str, f
from utils.mapping import get_term_map, get_solved_map, convert_system_to_hash
from utils.models import SIR
import copy
from utils.symbolic_utils import create_variable, o
import seaborn as sns

t = sp.Symbol('t')


class Config_Test:
    def __init__(self):
        self.target = SIR()

        self.G = 1  # Number of generations
        self.N = 200  # Maximum number of population
        self.M = 3  # Maximum number of equations
        self.I = 3  # Maximum number of terms per equation
        self.J = 3  # Maximum number of functions per feature
        self.allow_composite = False  # Composite Functions
        self.f0ps = get_functions("4,5")
        self.ivp_method = 'Radau'
        self.minimize_method = 'L-BFGS-B'  # 'Nelder-Mead' / L-BFGS-B / CG, COBYLA, COBYQA, TNC - fast

        self.elite_rate = 0.1
        self.crossover_rate = 0.2
        self.mutation_rate = 0.5
        self.new_rate = 0.2

        self.system_load_dir = '../data/analysis/computation_equations.txt'  # data/sir_equations.txt'
        self.system_save_dir = '../data/analysis/computation_equations.txt'

        self.DEBUG = False

def generate_term(variables, config, non_empty, funcs):
    rd.shuffle(variables)  # O(logN)
    term = None
    var_list = []
    j = 0
    if non_empty or rd.randint(0, 1) == 1: # equation to have at least one term
        for var in variables:
            if j == 0 or rd.randint(0, 1) == 1:
                func = f(rd.choices(funcs, k=1)[0])
                var = func(var)
                j += 1
                if config.allow_composite and j < config.J:  # limit applying composite to only once
                    if rd.randint(0, 1) == 1:
                        func = f(rd.choices(funcs, k=1)[0])
                        var = func(var)
                        j += 1
                var_list.append(var)
                if j == config.J: break

    if var_list:
        term = var_list[0]
        for var in var_list[1:]:
            operator = o(rd.randint(0, 1))
            term = operator(term, var)

    return term


def is_redundant(system, population):
    for p in population:
        check = True
        for (rhs1, rhs2) in zip(system, p):
            if sp.simplify(sum(rhs1[1]) - sum(rhs2[1])) != 0:
                check = False
                break
        if check:
            # print("reduandant: ", system, "<>", p )
            return True
    return False


if __name__ == "__main__":
    config = Config_Test()
    variables = [create_variable(i) for i in range(1, config.M + 1)]
    v = copy.deepcopy(variables)

    results = []
    for method in ["", "Hashed"]:#,"Hashed"
        for func in ["5", "4,5", "1,4,5","1,2,4,5","1,2,4,5,9","1,2,4,5,6,9","1,2,3,4,5,6,9"]:
            print(method, func)
            funcs = get_functions(func)
            systems = []
            systems_hash_list = []
            start_time = time.time()
            count_redundant = 0
            n = 0
            while n < config.N:
                system = []
                for m in range(config.M):
                    terms = []
                    for i in range(config.I):
                        term = generate_term(v, config, i == 0, funcs)
                        if term is not None and not term in terms:
                            terms.append(term)
                    system.append([sp.diff(variables[m], t), terms])

                if method!="Hashed":
                    if not is_redundant(system, systems):
                        systems.append(system)
                        results.append({'method': method, 'n': n, 'func': funcs_to_str(funcs),'duplicate': count_redundant
                                           , 'ts': time.time() - start_time})
                        n += 1
                    else:
                        count_redundant += 1

                if method=="Hashed":
                    s_hash = convert_system_to_hash(system)
                    if not s_hash in systems_hash_list:
                        systems.append(system)
                        systems_hash_list.append(s_hash)
                        results.append({'method': method, 'n': n, 'func': funcs_to_str(funcs), 'duplicate': count_redundant
                                           , 'ts': time.time() - start_time})
                        n += 1
                    else:
                        count_redundant += 1

    df = pd.DataFrame(results)
    df.to_csv("time_test.csv")
    df = df[df['method']==""][['n','func','duplicate','ts']]
    print(df)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Line plot for computation time
    sns.lineplot(
        data=df,
        x='n',
        y='ts',
        hue='func',
        markers=True,
        dashes=False,
        ax=ax1,
        legend=False  # Suppress the default legend from sns.lineplot
    )

    # Set labels for computation time axis
    ax1.set_xlabel("Number of Generated Systems (n)")
    ax1.set_ylabel("Computation Time (seconds)")
    ax1.set_title("Computation Time and Duplicates vs. Number of Systems")

    # Create a secondary axis for duplicates
    ax2 = ax1.twinx()

    # Prepare data for overlay bar plot
    unique_funcs = sorted(df['func'].unique(), key=len)
    bar_width = 0.2  # Width of each bar
    n_values = df['n'].unique()

    # Offset for each function set
    offsets = [-bar_width * (len(unique_funcs) // 2) + i * bar_width for i in range(len(unique_funcs))]

    for func, offset in zip(unique_funcs, offsets):
        # Subset data for the current function set
        subset = df[df['func'] == func]
        duplicates = subset.set_index('n')['duplicate']

        # Align bar heights with n-values
        heights = [duplicates.get(n, 0) for n in n_values]

        # Plot bars with an offset
        ax2.bar(
            n_values + offset, heights, bar_width, alpha=0.6, label=func  # Add labels for bars
        )

    # Set labels for duplicates axis
    ax2.set_ylabel("Number of Duplicates")
    ax2.grid(False)

    # Adjust y-axis limit for duplicates
    max_duplicates = df['duplicate'].max()
    ax2.set_ylim(0, 2 * max_duplicates)  # Set max to twice the maximum value of duplicates

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    bars2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + bars2, labels1 + labels2, loc='upper left',
               bbox_to_anchor=(0, 1))  # Place legend on the top left

    # Show the plot
    plt.tight_layout()
    plt.show()