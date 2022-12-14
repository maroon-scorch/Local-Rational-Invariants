import sys

def read_input(inputFile):
    """ Read and parse the input file, returning the list of points and its dimension """
    variables = set()
    with open(inputFile, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split("+")
            tokens.pop(-1)
            # print(tokens)
            for term in tokens:
                t = term.strip()
                mult = t.index("*")
                index = t[mult+1:]
                
                variables.add(index)
                
    return variables

def symmetrize_polynomials(variables):
    polynomial_list = []
    var = set()
    for v in variables:
        # print(v)
        var_index = v[2:][::-1]
        dual_variable = "x_"
        for char in var_index:
            c = int(char)
            if c % 2 == 0:
                dual_variable += str(c - 1)
            else:
                dual_variable += str(c + 1)
        var.add(v)
        var.add(dual_variable)
        # print(dual_variable)
        polynomial = v + " == " + dual_variable
        polynomial += ",\n"
        polynomial_list.append(polynomial)
    return polynomial_list, var

def variables_to_sage(variables):
    string = ""
    for v in variables:
        string += v + ", "
    return string

if __name__ == "__main__":
    input_file = sys.argv[1]
    variables = read_input(input_file)
    polynomial, variables = symmetrize_polynomials(variables)
    
    file = open("temp.txt", "w+")
    
    for p in polynomial:
        file.write(p)
        
    sage_export = variables_to_sage(variables)
    file.write(sage_export)
