<div align="center">
<img
src="https://github.com/james-alvey-42/jaxsp/blob/67be7bc188841bdf2bed02e72659245f0a2b2a1b/images/logo.png" alt="logo" width="150"></img>
</div>

## jaxsp: A Semi Analytical Model for ULDM in Spherical Systems
[**What is jaxsp**](#what-is-jaxsp)
| [**Installation**](#installation)
| [**Example**](#example)
| [**Citing jaxsp**](#citing-jaxsp)

## What is jaxsp
TODO: Add some description + most relevant refs.

<!-- [1] [Yavetz et al. (2021)](https://arxiv.org/abs/2109.06125): -->
<!-- _Construction of Wave Dark Matter Halos: Numerical Algorithm and Analytical Constraints_ -->
<!-- <br> -->
<!-- [2] [Lin et al. (2018)](https://arxiv.org/abs/1801.02320): -->
<!-- _Self-consistent construction of virialized wave dark matter halos_ -->
<!-- <br> -->
<!-- [3] [Dalal et al. (2021)](https://arxiv.org/abs/2011.13141): -->
<!-- _Don't cross the streams: caustics from fuzzy dark matter_ -->

<figure>
  <img src="https://github.com/james-alvey-42/jaxsp/blob/c23a854ffbaeaa7e81a07ff5d860efd0b212f534/images/leoII.png" alt="" width="500" align="center">
  <figcaption align="center">
  Volume rendering of the ULDM density constructed by `jaxsp`'s as fit of a library of 
  eigenstates to a density sample of Leo II provided by [gravsphere](https://github.com/justinread/gravsphere)
  </figcaption>
</figure>
<br/><br/>

The result is an effective, and efficient surrogate model for the ULDM wave function in 
for spherical symmetric systems. We refer to [our paper](#citing-jaxsp)
for an in depth exposition of jaxsp's underlying assumptions, physics and application to DM modelling in dwarf spheroidal galaxies.

jaxsp is built on [jax](https://github.com/google/jax). Any computation 
that involves the constructed wave function is therefore differentiable
(via `jax.grad`), simple (thanks to `jax.vmap`) and efficient (with `jax.jit`) 
to implement.

**This is a research project. Expect bugs, report bugs, fix more bugs than you
create.**

## Installation
TODO 
```console
$ git clone https://github.com/james-alvey-42/jaxsp.git
$ cd jaxsp
$ pip install -e .
```

## Example
TODO

## Citing jaxsp
If you use jaxsp for any publication we kindly ask you to cite
TODO

## Acknowledgement
<div align="center">
<img
src="https://github.com/james-alvey-42/jaxsp/blob/67be7bc188841bdf2bed02e72659245f0a2b2a1b/images/eu_acknowledgement_compsci_3.png" alt="logo"></img>
</div>
