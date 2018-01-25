This repo contains a C++ reference implementation of the reverse mode gradients of
the [celerite](https://github.com/dfm/celerite) method for Gaussian Process regression.
The code only depends on [Eigen](https://eigen.tuxfamily.org/) and the algorithms can
be found in [`src/celerite.h`](https://github.com/dfm/celerite-grad/blob/master/src/celerite.h)
and some example usage can be found in the [`src` directory](https://github.com/dfm/celerite-grad/blob/master/src/).

To build the examples, install Eigen and then run `make` in the home directory.
