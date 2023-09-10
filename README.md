# Local-Rational-Invariants

This project hosts the supplementary code for the project "Local Rational Formulae for Computing Topological Invariants" by Cassie Ding and Mattie Ji. This project was previously an independent study course advised by Professor [John Hughes](https://cs.brown.edu/people/jhughes/).
## Introduction

Computational topology studies the application of topological methods in computer science and is widely used in machine learning, computer graphics, coordinated mobile robotics, and more. A fundamental question in this subject is how to efficiently compute numerical invariants of combinatorial manifolds.

It turns out that many of these invariants (such as Euler characteristics, turning numbers, and Pontryagin numbers) can be computed with a universal "local rational formula". We call numerical invariants that can be evaluated by this kind of formula "local rational invariants".

Explicitly, we can assign every vertex with the same link structure $i$ with a rational number $r_i$ such that the numerical invariant we are interested in is the sum $\sum_{i} n_i r_i$, where $n_i$ is the number of vertex with the link structure $i$. Moreover, the choice of this rational number is independent of the input.

If we pre-compute what all the $r_i$'s are, this gives a deterministic linear time algorithm to calculate the numerical invariant we are interested in.

This repository implements the aforementioned method to compute local rational formula in Python and Sage for certain classes of closed manifolds. Note that this method is originally proposed by our advisor Professor John Hughes.

## The Main Algorithm

Formally encoding the idea in the introduction, the main algorithm is done using the "know-nothing method" proposed by our advisor Professor John Hughes.

We will illustrate this with an example of the Euler characteristic of closed surfaces in $\mathbb{R}^3$:

1. Suppose we are given a triangulated torus $T$:<br />
![alt text](https://github.com/maroon-scorch/Local-Rational-Invariants/blob/main/figures/Figure_1.png)

2. We first perform a rectangulation of the torus as follows:<br />
![alt text](https://github.com/maroon-scorch/Local-Rational-Invariants/blob/main/figures/sq_torus.png)

3. We keep a count of every vertex in the rectangulated torus, classified by its link structure. For each type $i$, let's call the count $n_i$. Here's an example of a vertex type:<br />
![alt text](https://github.com/maroon-scorch/Local-Rational-Invariants/blob/main/figures/vertex_5.png)

4. Each vertex of the same type is assigned with some undetermined rational number $r_i$, we make a linear equation:
$$\sum_i n_i r_i = \chi(T) = 0$$

5. If we repeat Step 1-4 for many different embeddings of closed 2-surfaces (not necessarily a torus), we can create a system of linear equations with indeterminate $r_i$. We then solve for the rational solutions of the $r_i$.
## Folders

- The folder [3d](3d) contains implementation of the know-nothing method and related features for embedded closed curves and closed surfaces in $\mathbb{R}^3$.
- The folder [code](code) contains implementation of the know-nothing method and related features for embedded and immersed closed curves in $\mathbb{R}^2$.
- The folder [figures](figures) contains pictures used in the README.
- The folder [gui](gui) contains code for a GUI where users can manually construct a closed curve to see how its local rational formula is computed.
- The folder [nd](nd) contains higher dimensional generalizations of the know-nothing method for embedded closed $k$-manifolds in $\mathbb{R}^n$. (Work in progress)
- The folder [tests](tests) contains some miscellaneous tests written for the functions in the folder **code**.

## How to Run
To install dependencies
```
pip install -r requirements.txt
```
<!-- To run the code
```
./run.sh <path_to_file>
``` -->
To run the tests
```
cd tests
pytest
```
