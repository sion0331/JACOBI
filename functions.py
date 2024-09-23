import sympy as sp

def one(x):
    return 1

def sin(x):
    return sp.sin(x)

def cos(x):
    return sp.cos(x)

def tan(x):
    return sp.tan(x)

def exp(x):
    return sp.exp(x)

def linear(x):
    return x

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def quart(x):
    return x ** 4

all_functions = {
    0: ("one", one),
    1: ("sin", sin),
    2: ("cos", cos),
    3: ("tan", tan),
    4: ("exp", exp),
    5: ("linear", linear),
    6: ("square", square),
    7: ("cube", cube),
    8: ("quart", quart)
}

def get_allowed_functions():
    print("Available functions:")
    for key, (name, _) in all_functions.items():
        print(f"{key}: {name}")

    # allowed_functions = input("Enter the numbers of the functions you want to allow (comma-separated): ")
    allowed_functions = "1,2,4,5,6"
    return [int(x.strip()) for x in allowed_functions.split(',')]

def f(i):
    return all_functions.get(i, ("Invalid", lambda x: x))[1]
