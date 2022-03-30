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

        numpoints = 1000
        x = np.random.uniform(low=403744.5, high=403747.5, size=_numpoints)
        y = np.random.uniform(low=4122665.5, high=4122668.5, size=_numpoints)
        z = np.random.uniform(low=13.0, high=15.0, size=_numpoints)
        tvu = np.random.uniform(low=0.1, high=1.0, size=_numpoints)
        thu = np.random.uniform(low=0.3, high=1.3, size=_numpoints)
    
        z = _z.astype(np.float32)
        tvu = _tvu.astype(np.float32)
        thu = _thu.astype(np.float32)
    
        numrows, numcols = (3, 3)
        resolution_x, resolution_y = (1.0, 1.0)
        
        depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = run_cube_gridding(z, thu, tvu, x, y, numcols, numrows, min(x), max(y), 'local', 'order1a', resolution_x, resolution_y)   

Numba version operates in the same way:

        # less performant version
        from bathycube.numba_cube import run_cube_gridding

        numpoints = 1000
        x = np.random.uniform(low=403744.5, high=403747.5, size=_numpoints)
        y = np.random.uniform(low=4122665.5, high=4122668.5, size=_numpoints)
        z = np.random.uniform(low=13.0, high=15.0, size=_numpoints)
        tvu = np.random.uniform(low=0.1, high=1.0, size=_numpoints)
        thu = np.random.uniform(low=0.3, high=1.3, size=_numpoints)
    
        z = _z.astype(np.float32)
        tvu = _tvu.astype(np.float32)
        thu = _thu.astype(np.float32)
    
        numrows, numcols = (3, 3)
        resolution_x, resolution_y = (1.0, 1.0)
        
        depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = run_cube_gridding(z, thu, tvu, x, y, numcols, numrows, min(x), max(y), 'local', 'order1a', resolution_x, resolution_y)

You can specify cube parameters as keyword arguments to this function as well.

        depth_grid, uncertainty_grid, ratio_grid, numhyp_grid = run_cube_gridding(z, thu, tvu, x, y, numcols, numrows, min(x), max(y), 'local', 'order1a', resolution_x, resolution_y, variance_selection='input')
