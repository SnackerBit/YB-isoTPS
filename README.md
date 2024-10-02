# YB-isoTPS
## Contents
This project implements isometric tensor network (isoTPS) algorithms for computing properties of two-dimensional quantum lattice models. 
Isometric tensor networks [[1]](#1)[[2]](#2) generalize the isometry condition of the popular Matrix Product States [[3]](#3) to two and higher dimensions.
In this implementation, we use the so-called "Yang-Baxter move" (YB move) for shifting the orthogonality hypersurface, in contrast to the Moses move (MM) that was used in the original implementation [[1]](#1)[[2]](#2). <br />

This implementation was created as part of my master's thesis [[4]](#4), in which the algorithms are explained in detail. <br />

The repository contains two folders, `src` and `test`. The `src` folder contains the implementation as python code. In the `test` folder a few notebooks are given as examples on how the algorithms can be used for finding ground states and performing real time evolution.

## References
<a id="1">[1]</a> 
Michael P. Zaletel and Frank Pollmann. ‘Isometric Tensor Network States in Two Dimensions’. In: Phys. Rev. Lett. 124 (3 Jan. 2020), p. 037201. doi: 10.1103/PhysRevLett.124.037201. url: https://link.aps.org/doi/10.1103/PhysRevLett.124.037201. <br />
<a id="2">[2]</a> 
Sheng-Hsuan Lin, Michael P. Zaletel and Frank Pollmann. ‘Efficient simulation of dynamics in two-dimensional quantum spin systems with isometric tensor networks’. In: Phys. Rev. B 106 (24 Dec. 2022), p. 245102. doi: 10.1103/PhysRevB.106.245102. url: https://link.aps.org/doi/10.1103/PhysRevB.106.245102. <br />
<a id="3">[3]</a> 
Ulrich Schollwöck. ‘The density-matrix renormalization group in the age of matrix product states’. In: Annals of physics 326.1 (2011), pp. 96–192. <br />
<a id="4">[4]</a> 
https://github.com/SnackerBit/master-thesis-physics/blob/main/thesis/thesis.pdf <br />
