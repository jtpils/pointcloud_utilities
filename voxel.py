import pointcloud as pc
import tiles
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import weibull_min
from warnings import warn

class Grid(object):
    """ A container for various objects conforming to a regular 2D grid of even size covering entire plot area:
        
        Grid indexing [i, j] is [y, x], with 0,0 at top left
        
        Key attributes:
            centres ::: 3D array containing x (at [0,:,:]) and y ([1,:,:]) coordinate components of grid cell centre
            """
    def __init__(self, plot_bounds, grid_size, pointclouds=None, cutoff=None):
        
        self._plot_bounds = plot_bounds
        self._input_grid_size = grid_size # input is not always real resolution
        self._set_up_xy(plot_bounds, grid_size)
        if pointclouds:
            self._initialise_with_pointclouds(pointclouds, cutoff)
        
        # Empty dicts to store models and simulated PointClouds
        self.models = {}
        self.SIM = {}
        
    def _set_up_xy(self, plot_bounds, grid_size):
        """ Set up an evenly spaced grid over plot area with grid spacing as close as possible to grid_size."""
        
        n_x = round(np.diff(plot_bounds['x'])/(grid_size*1.))
        n_y = round(np.diff(plot_bounds['y'])/(grid_size*1.))
        
        # Determine edges of grid
        x_edges = np.linspace(*plot_bounds['x'], num=n_x+1)
        y_edges = np.linspace(*plot_bounds['y'], num=n_y+1)
        
        # Find effective resolutions, given automatic determination of intervals
        res_x = np.mean(x_edges[1:]-x_edges[:-1])
        res_y = np.mean(y_edges[1:]-y_edges[:-1])
        xy_area = res_x*res_y # area covered by a grid cell
        
        # Determine grid points (coordinates of centre)
        x_centres = (x_edges[1:] + x_edges[:-1]) / (1.*2)
        y_centres = (y_edges[1:] + y_edges[:-1]) / (1.*2)
        
        # Create 3D grid showing xs [0,:,:] and ys [1,:,:]
        xs = np.tile(x_centres[np.newaxis, :], (n_y, 1))
        ys = np.tile(y_centres[:, np.newaxis], (1, n_x))
        xy_grid = np.concatenate([xs[np.newaxis,:,:], ys[np.newaxis,:,:]])
        
        # Store grid and resolution
        self.centres = xy_grid
        self.shape = (n_y, n_x)
        self.grid_size = (res_x, res_y)
        self.xy_area = xy_area 
        # Store edges (should ideally be a single array)
        self.x_edges = x_edges
        self.y_edges = y_edges
    
    def _initialise_with_pointclouds(self, pointclouds, cutoff):
        """Initialise with ALS and TLS PointClouds in grid cells.
        
        Args:
            pointclouds ::: tuple of (ALS, TLS) PointCloud objects to grid
        """
        ## Note that this function should be able to accept lists from pointclouds, and use them to read data directly from tiles 
        
        # Grid pointclouds
        self._grid_pointclouds(*pointclouds)
        
        ## Other PointCloud related setups
        self._grid_npoints()
        
        if cutoff:
            self.voxs = self._grid_Voxs(cutoff)
        
    def _grid_pointclouds(self, ALS, TLS):
        """Slice the provided ALS and TLS PointClouds into smaller voxels in grid
        
        Very slow, and could be sped up by reading directly from file/tiles rather than larger PointCloud.
        ."""
        # Loop over grid, slicing voxels (takes a min)
        
        ALS_grid = self._grid_dataset(ALS, 'ALS')
        TLS_grid = self._grid_dataset(TLS, 'TLS')
        
        PC_grid = np.concatenate([ALS_grid[np.newaxis,:,:], TLS_grid[np.newaxis,:,:]])
        
        self.PCs = PC_grid
    
    def _grid_dataset(self, dataset, label=None):
        """ Slice the pointcloud dataset to grid cells."""
        
        # x and y centres and edges to loop with
        xs = self.centres[0,0,:]
        ys = self.centres[1,:,0]
        xe = self.x_edges
        ye = self.y_edges
        
        # Activate tileflag if tile passed
        if isinstance(dataset, tiles.Tiles):
            tf = True
            first = True
            tile = dataset
        else:
            tf = False
            PC = dataset
            
        dataset_grid = np.empty(self.shape, dtype=object)
        for i, (yc, y1, y2) in enumerate(zip(ys, ye[:-1], ye[1:])):
            for j, (xc, x1, x2) in enumerate(zip(xs, xe[:-1], xe[1:])):
                bounds = {'x': (x1, x2), 'y': (y1, y2)}
                
                # call method to get a PC from dataset, if necessary
                if tf:
                    if first: # generate starting tiles and pointcloud
                        tile = dataset[bounds]
                        PC = pc.PointCloud(tile)
                        first = False
                    else:
                        newtile = dataset[bounds] # tiles proposed for current cell
                        if newtile.bounds != tile.bounds: # update PC if new area
                            tile = newtile
                            PC = pc.PointCloud(tile)
                            
                PC_slice = PC[bounds]
                
                setattr(PC_slice, 'centre', (xc, yc))
                setattr(PC_slice, 'label', label)
                dataset_grid[i, j] = PC_slice
        
        return dataset_grid
    
    def _grid_npoints(self):
        """ Return a 3D grid of npoints in each voxel """
        # Vectorised function to retreive n_points
        get_npoints = np.vectorize(lambda PC: getattr(PC, 'n_points'))
        npoints_grid = get_npoints(self.PCs)
        self.npoints = npoints_grid

    def _grid_Voxs(self, cutoff):
        """ Return a 2D grid of Vox objects initialised with the specified cutoff."""
        
        make_Voxs = np.vectorize(lambda ALS, TLS, cutoff : Vox(ALS, TLS, cutoff))
        voxs = make_Voxs(*self.PCs, cutoff=cutoff)
        
        return voxs
    
    def fit_models(self, mod, mod_pars, label=None):
        """ Create a new grid of the specified Model fitted to each voxel.

        Args:
            mod ::: vox_model.Model model to fit
            mod_pars ::: dict parameters used to initialise mod
            label ::: str label for model used as dict key (default: mod.__name__)

        Assigns:
            2D array of models to self.models[label]
        """

        # Vectorised fitting function
        fit_mods = np.vectorize(lambda vox, mod, mod_pars: mod(vox, **mod_pars))

        # Carry out fitting on each voxel
        models = fit_mods(self.voxs, mod, mod_pars)

        # Store model
        if not label: # use generic model name if not specified
            label = mod.__name__
            if self.models.get(label): # warn if name not novel
                warn("There were already models at `%s`, they will be overwritten"%name)
        
        self.models[label] = models
    
    def adjust_models(self, label, adjustments):
        """ Apply adjustments to model parameters.
        Args:
            label ::: str key of models in in grid.models
            adjustments ::: dict of adjustment functions to apply to model parameters {'par': func}
        """
        # Retrieve models
        models = self.models[label]
        # Adjust models
        adjust = np.vectorize(lambda mod, adjustments: mod.adjust_pars(adjustments))
        adjust(models, adjustments)
    
    def find_cell(self, x, y):
        """ Return the [i, j] matrix position of the object centred at the supplied (x, y) coordinates.
        Only works for exact matches (otherwise None)
        """
        xs, ys = self.centres
        i, j = np.where(np.logical_and(xs == x, ys == y))
        try:
            ij = int(i), int(j)
        except TypeError:
            ij = None
        return ij


class Vox(object):
    """An container for the shared ALS and TLS pointcloud space of a voxel, and an interface for the model.
    
    Note:
    ix attributes provide indices that can be passed in order to maintain concord between PointCloud points and derived arrays"""
    def __init__(self, ALS, TLS, cutoff):
        """
        Args:
            cutoff ::: float (metres) relative height threshold to split pointcloud to nearground/canopy

        """
        # Define centre
        if ALS.centre == TLS.centre:
            self.centre = ALS.centre
        
        # Store references to parent PCs
        self.PC_ALS = ALS
        self.PC_TLS = TLS
    
        # Extract z arrays
        self.z_ALS = ALS.z
        self.z_TLS = TLS.z
        
        # Index TLS points
        self.ix_z_ALS = np.arange(len(self.z_ALS))
        self.ix_z_TLS = np.arange(len(self.z_TLS))
        
        # Find ground and canopy top of voxel       
        self.lowest = np.min(np.concatenate([self.z_ALS, self.z_TLS]))
        self.highest = np.max(np.concatenate([self.z_ALS, self.z_TLS]))
        
        # Define histogramming parameters
        self._cutoff = cutoff + self.lowest
        
        # Classify points
        self._classify_points()
        
        # Determine lowest and highest in classified points
        self.highest_nearground = np.max(np.concatenate([self.z_ALS[~self.class_ALS], self.z_TLS[~self.class_TLS]]))
        self.lowest_canopy = np.min(np.concatenate([self.z_ALS[self.class_ALS], self.z_TLS[self.class_TLS]]))
        
        # Set absolute heights (normal for nearground and top-down for canopy)
        self._set_top_down_canopy_distances()
        self._set_nearground_heights()
        
    def _classify_points(self):
        """ Classify points to nearground (0) and canopy (1) based on a simple cutoff. """
        self.class_ALS = self.z_ALS >= self._cutoff
        self.class_TLS = self.z_TLS >= self._cutoff
        
    def _set_top_down_canopy_distances(self):
        """ Set the top-down heights for canopy data, and associated indices."""    
        
        self.tdc_ALS = self.highest - self.z_ALS[self.class_ALS]
        self.tdc_TLS = self.highest - self.z_TLS[self.class_TLS]
       
        self.ix_tdc_ALS = self.ix_z_ALS[self.class_ALS]
        self.ix_tdc_TLS = self.ix_z_TLS[self.class_TLS]
    
    def _set_nearground_heights(self):
        """ Set the heights from ground of nearground points, and associated indices"""
        self.hng_ALS = self.z_ALS[~self.class_ALS] - self.lowest
        self.hng_TLS = self.z_TLS[~self.class_TLS] - self.lowest
        
        self.ix_hng_ALS = self.ix_z_ALS[~self.class_ALS]
        self.ix_hng_TLS = self.ix_z_TLS[~self.class_TLS]
    
    def simulate_pointcloud(self, n, subset, model=None):
        """ Apply vox model to select n TLS points and return simulated PointCloud.

        Args:
            n ::: number of points to pick (will be rounded)
            which ::: str name of subset of PointCloud to pick
            model ::: a initialised instance of a model (vox_model.Model, etc) containing a `pdf` attribute
        Returns:
            PC_sim ::: PointCloud containing selected Points
        """
        # Initialise simulation
        n = int(round(n)) # can only use whole points
        # 
        if not model:
            try: # see if there is a model attatched
                model = vox.model
            except AttributeError:
                raise NoModelError, 'You need to pass an initialised model, or assign one to `vox.model`'

        # Pick points
        ix_picks = self.pick(n, subset, model.pdf)
        PC_sim = self.PC_TLS[ix_picks]

        sim_details = {'model': type(model).__name__, 'pars': model.pars}
        setattr(PC_sim, 'sim_details', sim_details)
    
        return PC_sim
    
    def pick(self, n, which, pdf):
        """ Randomly pick `n` TLS points from either 'nearground', 'canopy' or 'all` according to 'pdf'.
        Args:
            n ::: number of points to pick, will be rounded
            which ::: TLS subset choice, either of 'z' ('all'), 'hng' ('nearground') or 'tdc' ('canopy')
            pdf ::: any 1-arg function which determines f(x) for the array of values specified by `which`
        
        Returns:
            ix_keep ::: array the indices of chosen points
        
        Usage:
            >>> vox.pick_from_TLS(6, 'tdc', lambda x: 0.5*x^2+21)
            array([34244,  7769, 36894, 35147, 12372,  7328])
        """
        
        which_dataset = {'z': self.z_TLS, 'hng': self.hng_TLS, 'tdc': self.tdc_TLS,
                        'all': self.z_TLS, 'nearground': self.hng_TLS, 'canopy': self.tdc_TLS}
        which_ix = {'z': self.ix_z_TLS, 'hng': self.ix_hng_TLS, 'tdc': self.ix_tdc_TLS,
                        'all': self.ix_z_TLS, 'nearground': self.ix_hng_TLS, 'canopy': self.ix_tdc_TLS}
        
        try:
            zs = which_dataset[which]
            ix = which_ix[which]
        except KeyError:
            print "`which` must be either of %s"%(which_dataset.keys())
            raise
              
        probs = pdf(zs) # find probability of keeping any given point
        
        # Remove nans and infs
        probs[~np.isfinite(probs)] = 0.
        
        weights = probs/probs.sum() # normalise probabilites (sum to 1)
        ix_picks = random.choice(ix, round(n), replace=False, p=weights) # draw points according to pdf
        
        return ix_picks
    
class NoModelError(TypeError):
    """ When no valid model has been passed."""
