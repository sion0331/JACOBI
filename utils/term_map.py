term_map = {}  # Dictionary for term-to-ID mapping
term_counter = 0  # Counter for unique term IDs

def get_term_id(term):
    global term_counter
    term_str = str(term)
    # print(term_counter, term_str, term_str not in term_map, term_map)
    if term_str not in term_map:
        term_map[term_str] = term_counter
        term_counter += 1
    return term_map[term_str]

def get_term_map():
    return term_map