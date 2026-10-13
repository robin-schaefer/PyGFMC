"""ToricCode tori for the QuSpin benchmark.

ToricCode needs N = 2*Lx*Ly (the edges of an Lx x Ly torus) with Lx, Ly >= 2,
so these are the tori nearest the random-graph sizes: N = 8, 12, 16.
"""

TORIC_TORI = [(2, 2), (2, 3), (2, 4)]
