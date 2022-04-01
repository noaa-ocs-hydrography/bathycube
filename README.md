# Bathycube

Bathycube is a Python implementation of the CUBE module, Combined Uncertainty and Bathymetry Estimator.  It only contains
the cube grid/node objects, the original library included other data structures that were not translated here. 

CUBE was developed as a research project within the Center of for Coastal and Ocean Mapping and NOAA/UNH Joint Hydrographic
Center (CCOM/JHC) at the University of New Hampshire, starting in the fall of 2000.

Bathycube offers two approaches towards a Python CUBE Implementation.

cube.py is a less performant method, does not involve numba, but does offer more interactivity with the objects.

numba_cube.py is a more performant method (approx 55x faster), but do to the limitations of numba jitclasses, does not
allow for the user to inspect the grid structure.  This might be resolvable by switching to structref, based on dialogue
with users/developers on the numba gitter.

## Installation

Bathycube is not on PyPi, but can be installed using pip.

Download and install git (If you have not already): [git installation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

`pip install git+https://github.com/noaa-ocs-hydrography/bathycube.git#egg=bathycube `

## Quickstart

To simply grid some points and get the gridded data back, use the run_cube_gridding function:

        # less performant version
        from bathycube.cube import run_cube_gridding

        x = np.linspace(403744, 403747, 50)
        y = np.linspace(4122687, 4122690, 50)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = np.linspace(10, 20, 2500)
        tvu = np.linspace(0.3, 0.7, 2500)
        thu = np.linspace(0.3, 0.7, 2500)

        numrows, numcols = (3, 3)
        resolution_x, resolution_y = (1.0, 1.0)
        
        depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = run_cube_gridding(z, thu, tvu, x, y, numcols, numrows, min(x), max(y), 'local', 'order1a', resolution_x, resolution_y)   

Numba version operates in the same way:

        # less performant version
        import numpy as np
        from bathycube.numba_cube import run_cube_gridding
        
        x = np.linspace(403744, 403747, 50)
        y = np.linspace(4122687, 4122690, 50)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        z = np.linspace(10, 20, 2500)
        tvu = np.linspace(0.3, 0.7, 2500)
        thu = np.linspace(0.3, 0.7, 2500)
        
        numrows, numcols = (3, 3)
        resolution_x, resolution_y = (1.0, 1.0)
        
        depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = run_cube_gridding(z, thu, tvu, x, y, numcols, numrows, min(x), max(y), 'local', 'order1a', resolution_x, resolution_y)

You can specify cube parameters as keyword arguments to these functions as well.

        depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = run_cube_gridding(z, thu, tvu, x, y, numcols, numrows, min(x), max(y), 'local', 'order1a', resolution_x, resolution_y, variance_selection='input')

The numba version takes some time to run the first time you run it in a session.  This is the JIT compile cost.  You can compile without
running a long gridding call by using compile_now().  See numba docs to learn more about JIT compilation.

        from bathycube.numba_cube import compile_now
        
        compile_now()

