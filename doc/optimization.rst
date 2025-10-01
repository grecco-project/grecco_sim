Optimization
------------

Current State of distributed MPC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- Dual Decomposition / Dual subgradient Algorithm does not seem to work.
- Problem is overshooting for continuous local agents.
- TODO after USA: show overshooting in 1-D example (is dual problem convex?)

 -> Yes, this is because the primal solution is not differentiable wrt. the dual variable

- Introduce ADMM stuff with constraint tightening?
- or other anti-overshoot terms to vujanic

- The Vujanic/Falsone/LaBella algorithm has the step-size parameter alpha.

 -> Performance depends on the evolution of this parameter

Questions / Issues
~~~~~~~~~~~~~~~~~~

- Clean up code and share with project partners

- Check out: dual variable in Falsone2019 evolves in a way that solutions stick to constraint violations.
  No separation between agents.

 -> This issue is the equivalent to the question of who is asked to change their behavior if several options exist


Coding TODOs in the controllers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- clean up the controller subpackage.

 At this point, different implementations exist for the optimization based coordination mechanisms.

- Replace the parameterization via dictionaries with parameter/option classes


Protocol considerations
~~~~~~~~~~~~~~~~~~~~~~~

In both ADMM and Falsone/Vujanic MILP optimization, optimization is solved
iteratively.

It is unclear what the best approach for the initial iteration step is.
It can be either the economically optimized version without grid constraints.
Or the schedule without any use of flexibility, as implemented originally.
However, the latter does not really work for HPs and EVs.

A broader version of this question is:
Is there any reason to distinguish between uncontrolled and controlled
grid power in the centrally coordinated mechanism.

