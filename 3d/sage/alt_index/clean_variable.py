import sys

# How does one recover this...
def convert(var):
    result = "x_"
    input = [*var]
    input = [int(elt) for elt in input[2:]]
    second = input[1]
    last_second = input[-2]
    
    if last_second < second:
        input.reverse()
        
    for i in input:
        result += str(i)
    
    return result

def process_polynomial(input_file):
    variables = set()
    with open(input_file, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(',')
            tokens.pop(-1)
            for term in tokens:
                print(term)
                var = term.strip()
                variables.add(convert(var))

                
    return variables

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    variables = process_polynomial(input_file)
    
    file = open("temp.txt", "w+")
    for v in variables:
        file.write(v + ", ")