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
    
    