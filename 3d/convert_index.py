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
#def convert(var):
#    result = "x_"
#    input = [*var]
#    input = [path_dict[int(elt)] for elt in input[2:]]
#    
#    path = [np.asarray([0, 0, 0])]
#    prev = path[0]
#    for dir in input:
#        next = prev + np.asarray(dir)
#        path.append(next)
#        prev = next
    
#    print(path)
    
#    return result

data = np.genfromtxt('variables.txt', delimiter=', ',dtype = str)

def var_len(length):
    all_var_len = []
   # print(len(data)) #126

    #with open('sage\alt_index\variables.txt', 'r') as file:
    #data = file.read().split(',').strip()
    for var in data:
        if len(var[2:]) == length + 1:
            all_var_len.append(var[2:])
    num_var = len(all_var_len)
    return all_var_len, num_var

#print(var_len(3)) #16
#print(var_len(4)) #30
#print(var_len(5)) #47?????
#print(var_len(6)) #33?????

def isCircular(var1, var2):
    var1 = var1[3:]
    var2 = var2[3:] + var2[3:]
    if len(var1) != len(var2):
        return False
    else: 
        if var1 in var2:
            return True
        else: return False
        
def old_var_len(var):
    var = var[2:]
    return len(var[2:])/2

def dir_transfer(n):
    if n % 2 == 0: return n - 1
    else: return n + 1
        
def convert_ver(old_var):
    old_var = old_var[2:]
    squares = list(zip(old_var[::2], old_var[1::2]))
    new_sqrs = []
    for i in squares:
        i = new_sqrs.extend([dir_transfer(list(i)[1]), list(i)[0]])
    new_index = [v for i, v in enumerate(new_sqrs) if i == 0 or v != new_sqrs[i-1]]
    new_var = "x_" + str(new_index)
    for var in data:
        if isCircular(new_var, var):
            return var
        
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
                variables.append(convert_ver(var))
    
    new_polynomial = ""
    for c,v in zip(coefficients, variables):
        new_polynomial = new_polynomial + " + " + str(c) + "*" + str(v)
        new_polynomial = new_polynomial[3:]
    return new_polynomial              
#    return coefficients, variables

print(process_polynomial('jackpot.txt'))


if __name__ == "__main__":
    
    input_file = sys.argv[1]
    process_polynomial(input_file)
    