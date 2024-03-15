# Character Variety of Seifert Manifolds with PyTorch

## Project Overview

This repository hosts a PyTorch-based project aimed at exploring the character variety of Seifert manifolds. The core objective is to study the space of flat connections on Seifert manifolds, which involves solving a large set of nonlinear complex matrix equations. This investigation is crucial for understanding the intricate relationship between topological phases of matter in 2+1 dimensions and three dimensional geometry.

## Background

The project is a spin-off from my recent research "A Correspondence between Topological Order in 2+1d and Seifert Three-Manifolds." [Read the paper here](https://arxiv.org/abs/2403.03973). The paper proposes a novel correspondence that suggests a profound relationship between MTCs, known to characterize topological order in 2+1d, and Seifert three-manifolds. This correspondence aims to define a fusion category for every Seifert manifold and choice of ADE gauge group G, conjectured to be modular under certain conditions. It offers a constructive approach to determining the spins of anyons, their S-matrix, and the R- and F-symbols from simple building blocks. The paper tests this correspondence by realizing all MTCs (unitary or non-unitary) with rank \(r \leq 5\) in terms of Seifert manifolds and a choice of Lie group G.

## Objectives

The main goals of this project are:

- **Character Variety**: To explore and analyze the character variety of Seifert manifolds by examining the space of flat connections. This involves a detailed investigation into the solutions of a large set of matrix equations associated with the manifolds.
- **Use of PyTorch**: To leverage PyTorch's computational capabilities, especially its automatic differentiation and optimization libraries, for solving complex matrix equations that arise in the study of flat connections on Seifert manifolds.
- **Cool Physics**: To further the understanding of the relationship between topological orders in 2+1 dimensions and Seifert manifolds, potentially offering new insights into the classification of MTCs.

