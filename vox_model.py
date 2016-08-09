import pointcloud as pc
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

class Model(object):
    """ Base class for models, which interact with Voxel objects.
        All derived classes should:
            > Be initialised with a voxel.Voxel instance
            > Automatically fit a model to some component of the voxel
            > Store the permanent `fitted_pars` and transient `pars`
            > On request, allow adjustment of `fitted_pars` via adjust_pars(), overwriting `pars`
            > Have a pdf function at `pdf`
            """
    
    def __init__(self, vox, adjustments=None):
        self.adjustments = adjustments
        
    def adjust_pars(self, adjustments):
        """ Adjust any parameter according to a function.
        Args:
            adjustments ::: a dict of {par: func} functions to apply to pars
                            if None/False/0, will reset `pars` to fitted pars
                            
        Usage:
        >>> model.adjust_pars(adjustments={'c': lambda x: 1.25*x, 'loc': lambda x: 0})
        """

        self.pars = self.fitted_pars.copy() # reset pars
        
        # Carry out the specified adjustments to the parameters        
        if adjustments:
            self.adjustments = adjustments
            new_pars = {par: func(self.pars[par]) for par, func in adjustments.iteritems()} # apply adjustments
            self.pars.update(new_pars) # update pars with adjustments
        
        self._freeze_model() # update pdf with adjustments
    
    def _freeze_model(self):
        """ Freeze instance pdf to be that of current parameterised model."""
        self.pdf = self.model(**self.pars).pdf


class WeibCDF(Model): # make a base Weib class, then subclass for WeibCDF, WeibPDF, WeibKern etc
    """ A model of ALS density from TLS using the Weibull function."""
    
    def __init__(self, vox, bin_width=1., c_guess=0.62):
        """ Fit the model on initialisation.
        Args:
            bin_width ::: width of histogram bins (metres)
            c_guess ::: an initial guess for the `c` parameter
        """
        self.model = sp.stats.weibull_min
        
        # Store model parameters
        self.c_guess = c_guess
        self.bin_width = bin_width
        
        # Unpack top-down canopy distances
        self.A = vox.tdc_ALS
        self.T = vox.tdc_TLS
        
        self.centre = vox.centre
        
        self.run_fitting()
        
    def _make_cdfs(self):
        """ Determine the cumulative density functions of  with bin size"""
        
        # Estimate pdfs via density histogram
        bin_edges, (self.pdf_A, self.pdf_T) = pc.histogram(self.A, self.T,
                                                                bin_width=self.bin_width, density=True)
        self.bins = edges_to_centres(bin_edges) # bin centres

        # Convert pdfs to cdfs
        self.cdf_A = np.cumsum(self.pdf_A)
        self.cdf_T = np.cumsum(self.pdf_T)

        # Determine ratio
        self.ratio = 1.*self.cdf_A/self.cdf_T
    
    def run_fitting(self):
        """ Fit a Weibull curve to the top-down cdfs of ALS and TLS height values at shared bins `xs`. """
        self._make_cdfs()
        self._fit_model()
        self._freeze_model()
    
    def _fit_model(self):
        """ Fit the Weibull distribution to the input dataset
        Args:
            data ::: array of values to fit Weibull to, e.g. ratios of ALS/TLS cdfs
            xs ::: x locations of data, e.g. bin centres
            c_guess ::: Initial value for k, usually the plot-wide value
        Returns:
            fitted ::: fitted Weibull pdf at `xs`
            c, loc, scale ::: fitted Weibull parameters (loc determined by rule)
        """
        # Determine location parameter from data
        floc = self._determine_loc()

        # Fit Weibull to data
        c, loc, scale = self.model.fit(self.ratio, self.c_guess, floc=floc)

        # Make Weibull-fitted cdf ratio
        self.fitted_ratio = self.model.pdf(self.bins, c, loc, scale)
        
        self.fitted_pars = {'c': c, 'loc': loc, 'scale': scale}
        self.pars = self.fitted_pars
        
    def _determine_loc(self):
        """ Calculate the Weibull location parameter as left edge of first bin with nonzero finite data element."""

        # Find offset between bin edge and centre
        half_bin = np.mean(self.bins[1:]-self.bins[:-1])/2

        # Find x with first nonzero, finite value
        first_good_bin = self.bins[np.logical_and(np.isfinite(self.ratio), self.ratio > 0).argmax()]
        loc = first_good_bin - half_bin
        return loc

    def see_weibull_fit(self):
        """Plot the pdf histograms and cdf curves of ALS/TLS, data and fitted pdf ratios, and estimated ALS pdf."""
        raise NotImplementedError, 'Need to be able to show fitted pars and, if self.adjusted, adjusted pars'
        
        if getattr(self.adjusted, None):
            adj = True
            adj_pdf = self.model(**self.pars).pdf
        
        fig, axarr = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12, 8))

        ## Density histograms
        axarr[0, 0].bar(self.bins, self.pdf_A, width=self.bin_width, color='red')
        axarr[0, 0].bar(self.bins, self.pdf_T, width=self.bin_width, color='blue')
        axarr[0, 0].set_ylabel('$f(x)$')
        axarr[0, 0].set_title('Density Histogram')

        # CDFs
        axarr[1, 0].plot(self.bins, self.cdf_A, color='red', label='ALS')
        axarr[1, 0].plot(self.bins, self.cdf_T, label='TLS')
        axarr[1, 0].legend(loc='best')
        axarr[1, 0].set_ylabel('$F(x)$')
        axarr[1, 0].set_title('CDFs')
        axarr[1, 0].set_xlabel('Distance from canopy top (m)')

        # CDF ratio
        axarr[0, 1].plot(self.bins, self.ratio, color='purple')
        axarr[0, 1].set_title('Data CDF ratio')

        # Fitted CDF ratio
        axarr[1, 1].plot(self.bins, self.fitted_ratio, ls='--', color='purple')
        axarr[1, 1].text(0.65, 0.7, "$k$: %.3f\n$\\theta$: %.3f\n$\lambda$: %.3f"%(c, loc, scale),
                         transform=axarr[1, 1].transAxes, size=14)
        axarr[1, 1].set_title('Fitted CDF ratio')
        axarr[1, 1].set_xlabel('Distance from canopy top (m)')

        # ALS pdf (i.e. density histogram, in line form)
        axarr[0, 2].plot(self.bins, self.pdf_A, c='r')
        axarr[0, 2].set_title('Estimated ALS PDF')

        # Simulated ALS pdf
        axarr[1, 2].plot(self.bins, self.pdf_T * self.fitted_ratio, c='r', ls='--')
        axarr[1, 2].set_title('Fitted ALS PDF') # i.e. approximate result of pdf(z) for all TLS points 
        axarr[1, 2].set_xlabel('Distance from canopy top (m)')

        fig.suptitle("centre: %s, bin width: %s"%(self.centre, self.bin_width))
        
        
""" Helper functions """

def edges_to_centres(bin_edges):
    """ Return a len n-1 central values of a len n array of bin edges."""
    return bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.