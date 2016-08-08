import pointcloud as pc
import tiles
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import weibull_min


class Grid(object):
    """ A container for various objects conforming to a regular 2D grid of even size covering entire plot area:
        
        Grid indexing [i, j] is [y, x], with 0,0 at top left
        
        Key attributes:
            centres ::: 3D array containing x (at [0,:,:]) and y ([1,:,:]) coordinate components of grid cell centre
            """
    def __init__(self, plot_bounds, grid_size, pointclouds=None):
        
        self._plot_bounds = plot_bounds
        self._input_grid_size = grid_size # input is not always real resolution
        self._set_up_xy(plot_bounds, grid_size)
        if pointclouds:
            self._initialise_with_pointclouds(pointclouds)
        
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
    
    def _initialise_with_pointclouds(self, pointclouds):
        """Initialise with ALS and TLS PointClouds in grid cells.
        
        Args:
            pointclouds ::: tuple of (ALS, TLS) PointCloud objects to grid
        """
        ## Note that this function should be able to accept lists from pointclouds, and use them to read data directly from tiles 
        
        # Grid pointclouds
        self._grid_pointclouds(*pointclouds)
        
        ## Other PointCloud related setups
        self._grid_npoints() 
        
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
        self.ix_ALS = np.arange(len(self.z_ALS))
        self.ix_TLS = np.arange(len(self.z_TLS))
        
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
       
        self.ix_tdc_ALS = self.ix_ALS[self.class_ALS]
        self.ix_tdc_TLS = self.ix_TLS[self.class_TLS]
    
    def _set_nearground_heights(self):
        """ Set the heights from ground of nearground points, and associated indices"""
        self.hng_ALS = self.z_ALS[~self.class_ALS] - self.lowest
        self.hng_TLS = self.z_TLS[~self.class_TLS] - self.lowest
        
        self.ix_hng_ALS = self.ix_ALS[~self.class_ALS]
        self.ix_hng_TLS = self.ix_TLS[~self.class_TLS]