# Joint State Estimation & Control

This repo sets up an optimization problem that jointly solves state and trajectory optimization. It's built in python, using autodiff from Jax and Lie group objects from [jaxlie](https://github.com/brentyi/jaxlie/). It's setup in a similar fashion to [gtsam](https://github.com/borglab/gtsam/), which allows for arbitrary factors/keys/graph optimization.
