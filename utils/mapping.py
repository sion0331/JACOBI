solved_map = {}  # Dictionary for system-to-ID mapping
term_map = {}  # Dictionary for term-to-ID mapping
term_counter = 0  # Counter for unique term IDs


def get_solved_map():
    return solved_map


def get_individual_solved(id):
    if is_individual_solved(id):
        return solved_map[id]
    return None


def is_individual_solved(id):
    return id in solved_map


def add_individual_solved(id, solved):
    solved_map[id] = solved


def reset_solved():
    solved_map.clear()


def get_term_map():
    return term_map


def get_term_id(term):
    global term_counter
    term_str = str(term)
    # print(term_counter, term_str, term_str not in term_map, term_map)
    if term_str not in term_map:
        term_map[term_str] = term_counter
        term_counter += 1
    return term_map[term_str]


def convert_system_to_hash(system):
    return hash(tuple(tuple(sorted([get_term_id(term) for term in eq[1]])) for eq in system))

def count_betas(system):
    return sum(len(eq[1]) for eq in system)
