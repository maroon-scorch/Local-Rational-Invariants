import sys
import numpy as np

path_dict = {
    1: [1, 0, 0],
    2: [-1, 0, 0],
    3: [0, 1, 0],
    4: [0, -1, 0],
    5: [0, 0, 1],
    6: [0, 0, -1]
}

# How does one recover this...
def convert(var):
    result = "x_"
    input = [*var]
    input = [path_dict[int(elt)] for elt in input[2:]]
    
    path = [np.asarray([0, 0, 0])]
    prev = path[0]
    for dir in input:
        next = prev + np.asarray(dir)
        path.append(next)
        prev = next
    
    print(path)
    
    return result

def process_polynomial(input_file):
    coefficients = []
    variables = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split('+')
            tokens.pop(-1)
            for term in tokens:
                t = term.strip()
                print(t)
                c, var = t.strip().split('*')
                
                coefficients.append(int(c))
                variables.append(convert(var))
                
    return coefficients, variables

def var_len(length):
    all_var = []
    data = np.genfromtxt('variables.txt', delimiter=', ',dtype = str)

    #with open('sage\alt_index\variables.txt', 'r') as file:
    #data = file.read().split(',').strip()
    for var in data:
        if len(var[2:]) == length + 1:
            all_var.append(var[2:])
    num_var = len(all_var)
    return all_var, num_var

print(var_len(3))
print(var_len(4))
print(var_len(5))

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    process_polynomial(input_file)
    