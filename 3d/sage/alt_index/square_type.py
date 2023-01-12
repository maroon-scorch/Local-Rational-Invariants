import sys

def process_polynomial(input_file):
    variables = set()
    with open(input_file, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(',')
            tokens.pop(-1)
            for term in tokens:
                # print(term)
                variables.add(term.strip())
    return variables

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    variables = process_polynomial(input_file)
    
    # var = "x_235462"
    var = "x_1463251"

    for v in variables:
        if len(v) == len(var):
            print(var + " == " + v + ",")

    # file = open("temp.txt", "w+")
    # for v in variables:
    #     file.write(v + ", ")