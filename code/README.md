## Input Specification (Tentatively)

Currently, we will be processing polygonal curves in $\mathbb{R}^2$, the code takes in a text file of the following input specification:
- The first line of the file should contain a number, indicating the dimension of the point (This is 2, it doesn't do anything yet)
- For each point, they are represented as:
```
<x-position of point> <is x an integer> <y-position of point> <is y an integer>
```
An example file looks like:
```
2
0 1 0 1
1.5 0 1 1
2.5 0 2 1
3 1 3.5 0
```