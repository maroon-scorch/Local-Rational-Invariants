# Extending Data Type

This folder contains some miscellaneous code on extending the polygonal curve to a curve in $\mathbb{R}^n$.

## Input Specification (Future)

In the future
- The first line of the file should contain a number, indicating the dimension of the point
- For each point of coordinate $(x_1, x_2, ..., x_n)$, they are represented as:
```
<x_1-position of point> <is x_1 an integer> ... <x_n-position of point> <is x_n an integer>
```
An example file looks like:
```
3
0 1 0 1 0 1
1.5 0 1 1 2 1
2.5 0 2 1 1.5 0
3 1 3.5 0 -1 1
```