import pointcloud as pc
import tiles
import vox_model
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
        
        n_x = int(round(np.diff(plot_bounds['x'])/(grid_size*1.)))
        n_y = int(round(np.diff(plot_bounds['y'])/(grid_size*1.)))
        
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
    
    def fit_models(self, mod, mod_pars, model_key=None):
        """ Create a new grid of the specified Model fitted to each voxel.

        Args:
            mod ::: vox_model.Model model to fit
            mod_pars ::: dict parameters used to initialise mod
            model_key ::: str label for model used as dict key (default: mod.__name__)

        Assigns:
            2D array of models to self.models[model_key]
        """

        # Vectorised fitting function
        fit_mods = np.vectorize(lambda vox, mod, mod_pars: mod(vox, **mod_pars))

        # Carry out fitting on each voxel
        models = fit_mods(self.voxs, mod, mod_pars)

        # Store model
        if not model_key: # use generic model name if not specified
            model_key = mod.__name__
            if model_key in self.models.keys(): # warn if name not novel
                warntext = "There were already models at `%s`, they will be overwritten"%model_key
                warn(warntext)
        
        self.models[model_key] = models
    
    def adjust_models(self, model_key, adjustments):
        """ Apply adjustments to model parameters.
        Args:
            model_key ::: str key of models in in grid.models
            adjustments ::: dict of adjustment functions to apply to model parameters {'par': func}
        """
        # Retrieve models
        models = self.models[model_key]
        # Adjust models
        adjust = np.vectorize(lambda mod, adjustments: mod.adjust_pars(adjustments))
        adjust(models, adjustments)

    def simulate_ALS(self, setup):
        """ Simulate the ALS dataset.
        Args:
            setup ::: dict of {'subset': setup_pars}, where
                   :: subset :: valid subset of ALS (e.g. 'canopy') to simulate, also optionally used to determine npoints
                   :: pars :: dict of:
                    : model_key : key of self.models to use
                    : form : form of data to pass to model ('z', 'topdown', 'height')
                    : npoints : number of points to simulate
                        can be scalar or array of numeric, or 1-param function to apply to npoints of ALS subset
        Returns:
            dict {subset: PC_grid} grid of simulated PointCloud objects for each subset
        
        Usage:
            >>> subset: simulate_ALS({'model_key': 'KDERatio', 'form': 'topdown', 'npoints': lambda n: n * 3}) 
        """
    
        simulate = np.vectorize(lambda vox, n, subset, form, model: vox.simulate_pointcloud(n, subset, form,  model))
        sims = {}
        
        # Simulate data for each subset according to setup
        for subset, pars in setup.iteritems():
            # Extract parameters
            model_key = pars.pop('model_key')
            form = pars.pop('form', 'z') # use z if not otherwise specified
            npoints = pars.pop('npoints')
            
            models = self.models[model_key]
            n = self._process_npoints(npoints, subset)
            sims[subset] = simulate(self.voxs, n, subset, form, models)
        
        return sims

    def _process_npoints(self, npoints, subset):
        """
        Args:
            npoints ::: either of:
                     :: numeric scalar or array, which will simply be returned unaltered
                     :: a 1-parameter function of the number of points in the specified subset of ALS
            subset ::: 'all', 'canopy', nearground'
        Returns:
            n ::: npoints, or array of npoints function evaluated over the specified subset of ALS
        """
        try: # test if input is a simple scalar or array of numeric
            n = np.round(npoints)
        except AttributeError: #  assume npoints to be a function
            # Get the number of points in the chosen subset
            get_vox_npoints = np.vectorize(lambda vox, subset: len(vox.get_array('ALS', subset, 'ix')))
            vox_npoints = get_vox_npoints(self.voxs, subset)
            
            # Apply function
            npoints_func = np.vectorize(npoints)
            n = npoints_func(vox_npoints)
        return n

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
        
    def _classify_points(self):
        """ Classify points to nearground (0) and canopy (1) based on a simple cutoff. """
        self.class_ALS = self.z_ALS >= self._cutoff
        self.class_TLS = self.z_TLS >= self._cutoff
        
    def get_array(self, dataset, subset, form):
        """Return vox z values in format specified by args
        
        Args:
            dataset ::: 'ALS' or 'TLS'
            subset ::: Entire dataset ('all'), above cutoff ('canopy', 'c'), or below cutoff ('nearground', 'ng')
            form ::: Values in original z ('z'), normalised height ('h'), top-down distances ('td') or indices ('ix')
        Returns:
            array of the requested attribute
        Usage:
            >>> vox.get_array(dataset='ALS', subset='c', form='ix') # get indices of TLS canopy heights
        """
        # Sanitize input
        subset, form = subset.lower(), form.lower()
        
        # Get unaltered dataset
        arr = getattr(self, 'z_' + dataset) # z component of coordinates
        ixs = getattr(self, 'ix_z_' + dataset) # indices       
        classif = getattr(self, 'class_' + dataset) # canopy/ground threshold
        
        # Subset the indicies as required
        if subset in ['all','z', 'zs', None]:
            pass
        elif subset in ['canopy', 'c']:
            ixs = ixs[classif]
        elif subset in ['nearground', 'ng']:
            ixs = ixs[~classif]
        else:
            raise TypeError, 'subset %s not recognised' % subset
       
        # Decide whether to return indices or process z's 
        if form == 'ix': 
            return ixs 
        else:
            arr = arr[ixs] # subset the data
        
        # Process the z array
        if form in ['z', None]:
            pass # keep as is
        elif form in ['height', 'h']:
            arr = arr - self.lowest
        elif form in ['top-down', 'topdown', 'td']:
            arr = self.highest - arr
        else:
            raise TypeError, 'form %s not recognised' % form
  
        return arr
    
    def simulate_pointcloud(self, n, subset, form, model=None):
        """ Apply vox model to select n TLS points and return simulated PointCloud.

        Args:
            n ::: number of points to pick (will be rounded)
            subset ::: str name of subset of PointCloud to pick
            form ::: str form of data to pass to model
            model ::: a initialised instance of a model (vox_model.Model, etc) containing a `pdf` attribute
        Returns:
            PC_sim ::: PointCloud containing selected Points
        """
        # Initialise simulation
        n = int(round(n)) # can only use whole points
        
        if not model:
            try: # see if there is a model attatched
                model = self.model
            except AttributeError:
                raise NoModelError, 'You need to pass an initialised model, or assign one to `vox.model`'

        # Pick points
        ix_picks = self.pick(n, subset, form, model.pdf)
        PC_sim = self.PC_TLS[ix_picks]

        sim_details = {'model': type(model).__name__, 'pars': getattr(model, 'pars', None)}
        setattr(PC_sim, 'sim_details', sim_details)
    
        return PC_sim
    
    def pick(self, n, subset, form, pdf):
        """ Randomly pick `n` TLS points from either 'nearground', 'canopy' or 'all` according to 'pdf'.
        Args:
            n ::: number of points to pick, will be rounded
            subset ::: TLS subset choice, either of 'all', 'nearground'/'ng' or 'canopy'/'c'
            form ::: str form of TLS values to supply to pdf choice, either 'z', topdown'/'td' or 'height'/'h'
            pdf ::: any 1-arg function which determines f(x) for the array of values specified
        
        Returns:
            ix_keep ::: array the indices of chosen points
        
        Usage:
            >>> vox.pick_from_TLS(6, 'canopy', 'topdown', lambda x: 0.5*x^2+21)
            array([34244,  7769, 36894, 35147, 12372,  7328])
        """
        
        # Retrieve data        
        zs = self.get_array('TLS', subset, form) # points to pick from
        ix = self.get_array('TLS', subset, 'ix') # their indices
         
        try:
            probs = pdf(zs) # find probability of keeping any given point
            probs[~np.isfinite(probs)] = 0. # remove nans and infs
            weights = probs/probs.sum() # normalise probabilites (sum to 1)
            ix_picks = random.choice(ix, round(n), replace=False, p=weights) # draw points according to pdf
        except(ValueError, TypeError):
            ix_picks = None
            warn('Picking failed %s'%(self.centre,))
        return ix_picks
    
""" Exceptions. """
class NoModelError(TypeError):
    """ When no valid model has been passed."""

""" Plotting functions."""
def see_vertical_distribution(vox, subset='all', bin_width=1.):
    """ Plot vox PointClouds and associated joint histogram.
    Args:
        vox ::: a vox instance
        subset ::: subset of data to plot ('all', 'canopy', 'nearground')
        bin_width ::: vertical resolution of histogram
    """
    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(10, 6))
    plot_pars = {'ALS': ('red', 5), 'TLS': ('blue', 0.1)}
    zs = {}

    for ax, x_axis in zip(axarr[:2], 'xy'):
        ax.set_aspect("equal")  # fix x and y scales in proportion

        # Plot PointClouds
        for i, dataset in enumerate(('TLS', 'ALS')):
            c, s, = plot_pars[dataset]        

            # Specify PointCloud to get coordinates
            PC = getattr(vox, 'PC_'+dataset)

            ix = vox.get_array(dataset, subset, 'ix') # indices
            xs =  getattr(PC, x_axis)[ix]
            ys =  getattr(PC, 'z')[ix]

            zs[dataset] = ys # store zs for histogramming

            ax.scatter(xs, ys,
                       s=s, c=c, edgecolors='none')
            ax.set_xlabel(x_axis)

        axarr[0].set_ylabel('z (m)')

    # Make histograms
    bin_edges, hists = pc.histogram(zs['TLS'], zs['ALS'],
                                             bin_width=bin_width, density=True)
    bins = vox_model.edges_to_centres(bin_edges) # bin centres

    # Plot histogram
    for i, (dataset, ax) in enumerate(zip(('TLS', 'ALS'), axarr[2:])):
        ax.barh(bins, hists[i], linewidth=0, color=plot_pars[dataset][0], align='center')
        ax.set_title(dataset)
        ax.set_xlabel('Density')

    fig.suptitle(vox.centre)
