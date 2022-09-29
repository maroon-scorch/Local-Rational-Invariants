# Local-Rational-Invariants

This is the code repository hosting some of our works on local rational invariants under Professor John Hughes.

## Input Specification (Tentatively)

The code takes in a text file of the following input specification:
- The first line of the file should contain a number, indicating the number of points of the polygonal curve
- For each point, they are represented as:
```
<x-position of point> <y-position of point> <is x an integer> <is y an integer>
```
An example file looks like:
```
4
0 0 1 1
1.5 1 0 1
2.5 2 0 1
3 3.5 1 0
```

## How to Run
```
./run.sh <path_to_file>
```