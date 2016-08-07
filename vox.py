import pointcloud as pc
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import weibull_min

class Vox(object):
    """An extendable class used to carry out all histogram comparisons for a given voxel. """
    def __init__(self, ALS, TLS, c_z=1., cutoff=3):
        """
        Args:
            c_z ::: vertical resolution (i.e. histogram bin widths)
            cutoff ::: relative height threshold (in metres) used to split hist to nearground/canopy
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
        self.ix_TLS = np.arange(len(TLS.z))
        
        # Find lowest and highest point in voxel       
        self.lowest = np.min(np.concatenate([self.z_ALS, self.z_TLS]))
        self.highest = np.max(np.concatenate([self.z_ALS, self.z_TLS]))
        
        # Define histogramming parameters
        self.c_z = c_z
        self._cutoff = cutoff + self.lowest
        
        # Classify points
        self._classify_points()
        
        # Determine lowest and highest in classified points
        self.highest_nearground = np.max(np.concatenate([self.z_ALS[~self.class_ALS], self.z_TLS[~self.class_TLS]]))
        self.lowest_canopy = np.min(np.concatenate([self.z_ALS[self.class_ALS], self.z_TLS[self.class_TLS]]))
        
        # Generate histograms and pmfs
        self._make_histograms()
        self._make_pmfs()
    
    def _classify_points(self):
        """ Classify points to nearground (0) and canopy (1) based on a simple cutoff. """
        self.class_ALS = self.z_ALS >= self._cutoff
        self.class_TLS = self.z_TLS >= self._cutoff 
    
    def _make_histograms(self):
        """ Generate and store shared histograms."""
        
        # Determine histograms with joint bins
            
        self.bin_edges, (self.hist_ALS, self.hist_TLS) = pc.histogram(self.z_ALS, self.z_TLS, bin_width=self._c_z)
        self.bin_edges_nearground, (self.hist_ALS_nearground, self.hist_TLS_nearground) = pc.histogram(
            self.z_ALS[~self.class_ALS], self.z_TLS[~self.class_TLS], bin_width=self._c_z)
        self.bin_edges_canopy, (self.hist_ALS_canopy, self.hist_TLS_canopy) = pc.histogram(
            self.z_ALS[self.class_ALS], self.z_TLS[self.class_TLS], bin_width=self._c_z)
              
    def _make_pmfs(self):
        """ Generate probability mass functions of histograms. """
        for dataset in ['ALS', 'TLS']:
            for histtype in ['', '_nearground', '_canopy']:
                name = dataset+histtype # name of this combination
                pmf = pc.pmf(getattr(self, 'hist_'+name))
                setattr(self, 'pmf_'+name, pmf)


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
        self.grid_size = (res_x, res_y)
        
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
        
        
    def _grid_pointclouds(self, ALS, TLS):
        """Slice the provided ALS and TLS PointClouds into smaller voxels in grid
        
        Very slow, and could be sped up by reading directly from file/tiles rather than larger PointCloud.
        ."""
        xs = self.centres[0,0,:]
        ys = self.centres[1,:,0]
        
        xe = self.x_edges
        ye = self.y_edges
                
        which_dataset = {0: 'ALS', 1: 'TLS'}
        
        # 3D array to store grids of ALS (top layer [0,:,:]) and TLS (bottom layer [0,:,:]) voxel PCs
        PC_grid = np.empty(self.centres.shape, dtype=object)
        
        # Loop over grid, slicing voxels (takes a min)
        for i, (yc, y1, y2) in enumerate(zip(ys, ye[:-1], ye[1:])):
            for j, (xc, x1, x2) in enumerate(zip(xs, xe[:-1], xe[1:])):
                # Slice ALS and TLS to voxel
                for d, PC in enumerate([ALS, TLS]):
                    PC_slice = PC[{'x': (x1, x2), 'y': (y1, y2)}]
                    setattr(PC_slice, 'centre', (xc, yc))
                    setattr(PC_slice, 'label', which_dataset[d])
                    PC_grid[d, i, j] = PC_slice
        self.PCs = PC_grid
        
        
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