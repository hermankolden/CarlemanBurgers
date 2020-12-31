# Carleman solution of the viscous Burgers equation

This code solves the viscous Burgers equation, possibly with linear damping,
using a direct application of the Carleman method combined with Euler's
equation. The result is compared with solutions from the inbuilt MATLAB solver
pdepe, as well as a direct applicatoin of Euler's method.

This script in its currrent state should reproduce the results
in https://arxiv.org/abs/2011.03185, "Efficient quantum algorithm for
dissipative nonlinear differential equations" by Jin-Peng Liu,
Herman Øie Kolden, Hari K. Krovi, Nuno F. Loureiro, Konstantina Trivisa,
Andrew M. Childs.

Code written by Herman Øie Kolden in 2020.
