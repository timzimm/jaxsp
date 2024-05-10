<div align="center">
<img
src="https://github.com/james-alvey-42/jaxsp/blob/67be7bc188841bdf2bed02e72659245f0a2b2a1b/images/logo.png" alt="logo" width="150"></img>
</div>

## jaxsp: A Semi Analytical Model for ULDM in Spherical Systems
[**What is jaxsp**](#what-is-jaxsp)
| [**Installation**](#installation-and-usage)
| [**Example**](#example)
| [**Citing jaxsp**](#citing-jaxsp)

## What is jaxsp
`jaxsp` is a python library for constructing analytical wave function approximations 
to the steady-state Schroedinger-Poisson equation (SP) in spherical symmetry. 
The resulting wave function is a cheap, yet accurate, surrogate model for the equilibrium dynamics of ultralight, bosonic 
dark matter (ULDM) in spherically symmetric systems such as dwarf galaxies. Its
evaluation does not require to run full-fledged SP simulations.

Under the hood, `jaxsp` builds the sought-after wave function by 
(i) generating an extensive library of energy eigenstates via the matrix-free Pruess method and application of the
WWT-algorithm 
(ii) fitting the eigenstate libary to a user-provided density profile.

`jaxsp` is built on [jax](https://github.com/google/jax). Any computation 
that involves the constructed wave function is therefore differentiable
(via `jax.grad`), simple (thanks to `jax.vmap`) and efficient (with `jax.jit`) 
to implement.

<figure>
  <img src="https://github.com/james-alvey-42/jaxsp/blob/c23a854ffbaeaa7e81a07ff5d860efd0b212f534/images/leoII.png" alt="" width="500" align="center">
  <figcaption align="center">
  Volume rendering of the ULDM density constructed by jaxsp as fit of a library of 
  eigenstates to a density sample of Leo II provided by https://github.com/justinread/gravsphere
  </figcaption>
</figure>
<br/><br/>

`jaxsp` may be seen as an evolution of the matrix/regression methods discussed in:

[1] [Yavetz et al. (2021)](https://arxiv.org/abs/2109.06125):
_Construction of Wave Dark Matter Halos: Numerical Algorithm and Analytical Constraints_
<br>
[2] [Lin et al. (2018)](https://arxiv.org/abs/1801.02320):
_Self-consistent construction of virialized wave dark matter halos_
<br>
[3] [Dalal et al. (2021)](https://arxiv.org/abs/2011.13141):
_Don't cross the streams: caustics from fuzzy dark matter_

We refer to [our paper](#citing-jaxsp) for an in depth exposition of `jaxsp`'s underlying assumptions, physics and application to DM modelling in dwarf spheroidal galaxies.


**This is a research project. Expect bugs, report bugs, fix more bugs than you
create.**

## Installation and Usage
To install, simply: 
```console
$ git clone https://github.com/james-alvey-42/jaxsp.git
$ cd jaxsp
$ pip install -e .
```

Be aware that `jax` defaults to `float32` precision. While this might be enough
for visualisation purposes, we advise to use `jaxsp` in `float64`-mode only. For
this, make sure to:
```console
export JAX_ENABLE_X64=True
```
before executing a script that contains `jaxsp` code.

## Example
The core functionality, i.e. computing densities, potentials, eigenstate libraries and wave function parameters, is showcased in 
the [tutorial notebook](examples/tutorial.ipynb). For additional examples, we refer to our paper repository TODO.

## Contributors
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/timzimm"><img src="https://images.weserv.nl/?url=github.com/timzimm.png&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt="Tim Zimmermann"/><br /><sub><b>Tim Zimmermann</b></sub></a><br /></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/james-alvey-42"><img src="https://images.weserv.nl/?url=github.com/james-alvey-42.png&h=150&w=150&fit=cover&mask=circle&maxage=7d" width="150px;" alt="James Alvey"/><br /><sub><b>James Alvey</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>
## Citing jaxsp
If you use jaxsp for any publication we kindly ask you to cite
TODO

## Acknowledgement
<div align="center">
<img
src="https://github.com/james-alvey-42/jaxsp/blob/67be7bc188841bdf2bed02e72659245f0a2b2a1b/images/eu_acknowledgement_compsci_3.png" alt="logo"></img>
</div>
