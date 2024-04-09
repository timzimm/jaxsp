<div align="center">
<img
src="https://github.com/james-alvey-42/jaxsp/blob/67be7bc188841bdf2bed02e72659245f0a2b2a1b/images/logo.png" alt="logo" width="150"></img>
</div>

# jaxsp: A Semianalytical Model for ULDM in Dwarfs
[**What is Fuzzylli**](#what-is-fuzzylli)
| [**Installation**](#installation)
| [**Example**](#example)
| [**Citing Fuzzylli**](#citing-fuzzylli)

## What is fuzzylli
TODO

<!-- [1] [Yavetz et al. (2021)](https://arxiv.org/abs/2109.06125): -->
<!-- _Construction of Wave Dark Matter Halos: Numerical Algorithm and Analytical Constraints_ -->
<!-- <br> -->
<!-- [2] [Lin et al. (2018)](https://arxiv.org/abs/1801.02320): -->
<!-- _Self-consistent construction of virialized wave dark matter halos_ -->
<!-- <br> -->
<!-- [3] [Dalal et al. (2021)](https://arxiv.org/abs/2011.13141): -->
<!-- _Don't cross the streams: caustics from fuzzy dark matter_ -->

<figure>
  <img src="https://github.com/timzimm/fuzzylli/blob/0d792d8d018cb6a44108581965902cfc148f8aeb/images/comparison.png" alt="" width="750" align="center">
  <figcaption align="center">Comparison between fuzzylli and AXIREPO, a state of the
  integrator for the full-fledged Schrödinger-Poisson equation</figcaption>
</figure>
<br/><br/>

The result is an effective, and efficient surrogate model for the ULDM wave function in 
for spherical symmetric systems. We refer to [our paper](#citing-fuzzylli)
for an in depth exposition of jaxsp's underlying assumptions, physics and application to DM modelling in dwarf spheroidal galaxies.

fuzzylli is built on [jax](https://github.com/google/jax) such that compuation of observables 
involving the wave function.
(including its derivatives via `jax.grad`) are simple (thanks to `jax.vmap`) and 
efficient (thanks to `jax.jit`) to implement.

<figure>
  <img src="https://github.com/timzimm/fuzzylli/blob/2aecf2029754e7ef9d86a9b11a99cb1d6d2603c6/images/crosssections.png" alt="" width="750" align="center">
  <figcaption align="center">Filament cross sections as a function of axion/FDM mass</figcaption>
</figure>
<br/><br/>

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

## Citing fuzzylli
If you use jaxsp for any publication we kindly ask you to cite
TODO

## Acknowledgement
<div align="center">
<img
src="https://github.com/james-alvey-42/jaxsp/blob/67be7bc188841bdf2bed02e72659245f0a2b2a1b/images/eu_acknowledgement_compsci_3.png" alt="logo"></img>
</div>
