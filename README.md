# Character Variety of Seifert Manifolds with PyTorch

## Overview

This repository hosts an on-going PyTorch-based project aimed at exploring the character variety of Seifert manifolds. One of the main objectives is to study the space of flat connections on general Seifert manifolds, which involves solving a large set of nonlinear complex matrix equations. This is an important open question in geometry and topology, and play an pivotal role in the compactification of quantum field theories. This investigation is crucial for understanding the intricate relationship between topological phases of matter in 2+1 dimensions and three dimensional geometry. The project is a spin-off from my recent research [Read the paper here](https://arxiv.org/abs/2403.03973), which proposes a novel correspondence that suggests a profound relationship between MTCs, known to characterize topological order in 2+1d, and Seifert three-manifolds. 

## Objectives

The main goals of this project are:

- **Study the Character Variety**: To explore and analyze the character variety of Seifert manifolds by examining the space of flat connections. 
- **Use of PyTorch**: To leverage PyTorch's computational capabilities, especially its automatic differentiation and optimization libraries, for solving complex matrix equations.
- **Generate Dataset of Approximate Solutions**: To generate a large dataset of approximate solutions to the matrix equations. This dataset will serve as a foundation for applying deep neural networks to learn possible underlying patterns, aiming to uncover new insights into the character variety of Seifert manifolds.

I will update the repository regularly as our research progresses.

Progress

- I've developed a Pytorch-based solution [python code](./conn_optimizer_batch.py) for finding irreducible flat connections that significantly outpaces Mathematica's FindMinimum function in terms of performance. Key highlights include:
  * Rapid Computation: Completes tasks in seconds that take hours or days with Mathematica built-in `FindMinimum`.
  * Proven reliability: Demonstrated through extensive testing to provide reliable and fast results, whereas traditional techniques may fail to find the optimal solution or be stuck in a local minimum.
  * GPU support added: separate the CPU-bound tasks from GPU-bound tasks to maximize the GPU's computational power and minimize the data transfers between CPUs and GPUs.
  * batch processing that is optimized for Cluster computation
- A class [dyn_to_flat_conn.py](./dynkin_to_flat_conn.py) to generate all of the candidate flat connections along with its properties.
  * The output will be stored in `pandas.dataframe`.
  * This is based on the weight lattice representation of the flat connections using Dynkin labels.
 

