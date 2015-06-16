<img src='http://www.cc.gatech.edu/~xliu66/gtfock/logo.png' alt='Logo' />

# GTFock: A Distributed Framework for Fock Matrix Construction #
Fock matrix construction is an essential kernel in many quantum chemistry calculations, including the Hartree-Fock (HF) method and Density Functional Theory (DFT) methods. GTFock uses a new scalable parallel algorithm for Fock matrix construction, which significantly reduces communication and has better load balance than other current codes. GTFock shows nearly linear speedup up to 1,000 nodes on the Stampede supercomputer and better scalability than NWChem on chemical systems that stress scalability.

# Spotlights #
  * Uses a new scalable parallel algorithm that significantly reduces communication and has better load balance than other current nodes.
  * Provides a generalized computational interface for constructing Fock matrices which can be easily used to build customized HF or DFT applications.
  * Can compute multiple Fock matrices on one run.
  * The input density matrices can be non-symmetric.

# News #
  * GTFock v0.1.0 has been released. [Downloads](http://www.cc.gatech.edu/~xliu66/gtfock/gtfock-v0.1.0.tgz).

# Project Members at [Georgia Tech](http://www.gatech.edu/) #
  * [Edmond Chow](http://www.cc.gatech.edu/~echow/) (Associate Professor in the [School of Computational Science and Engineering](http://www.cse.gatech.edu/))
  * [Xing Liu](http://www.cc.gatech.edu/~xliu66/) (Ph.D. Candidate in the [School of Computational Science and Engineering](http://www.cse.gatech.edu/))
  * Aftab Patel (Ph.D. Student in the [School of Computational Science and Engineering](http://www.cse.gatech.edu/))

<br>
<h2><a href='Downloads.md'>Downloads</a></h2>
<h2><a href='Installation.md'>Installation</a></h2>
<h2><a href='Documentation.md'>Documentation</a></h2>
<h2><a href='Changelog.md'>Changelog</a></h2>
<h2><a href='Publications.md'>Publications</a></h2>
<h2><a href='Acknowledgements.md'>Acknowledgements</a></h2>