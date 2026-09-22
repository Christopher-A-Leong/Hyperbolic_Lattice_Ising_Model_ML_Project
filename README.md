# Hyperbolic Lattice Ising Model Machine Learning Project

A study of the paramagnetic to ferromagnetic transition in the Ising model applied on hyperbolic lattices. Data is computed from Monte Carlo simulations and the magnetic ordering is classified by fully-connected neural networks (FCNNs) and graph convolutional neural networks (GCNNs).

## Overview

The Ising model is a fruitful testbed for simulations of continuous phase transitions. Namely, the model features a paramagnetic phase at high temperatures, which gives way to a ferromagnetic or antiferromagnetic order below a critical temperature, the former of which is of relevance here. I study this model on hyperbolic lattices using classical Monte Carlo simulations. The resulting data, in the form of spin configurations, is then fed into FCNNs and GCNNs to classify the magnetic orders. While domain-wall-free ferromagnetic orders are easily identifiable by FCNNs, when domain walls are present it is shown that only GCNNs are able to resolve the existence of the ferromagnetic order below the Curie temperature. For a more in-depth write-up of this project, see the "Project_Writeup.pdf" file in the repository.

## References

[1] A. Geron, Hands-On Machine Learning with Scikit-Learn and TensorFlow (O'Reilly) 2017.

[2] Also see references in the "Project_Writeup.pdf" document 

