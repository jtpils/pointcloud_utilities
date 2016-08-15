import pointcloud as pc
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.stats.kde import gaussian_kde

class Mod(object):
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


class WeibCDF(Mod): # make a base Weib class, then subclass for WeibCDF, WeibPDF, WeibKern etc
    """ A model of ALS density from TLS using the Weibull function."""
    
    def __init__(self, vox, bin_width=1., c_guess=0.62):
        """ Fit the model on initialisation.
        Args:
            bin_width ::: width of histogram bins (metres)
            c_guess ::: an initial guess for the `c` parameter
        """
        self.model = weibull_min
        
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
        
        # Get fitted pdf and CDF ratio
        fit_pdf = self.model(**self.fitted_pars).pdf
        fit_ratio = fit_pdf(self.bins)
        
        fig, axarr = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(14, 10))

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
        axarr[0, 1].plot(self.bins, self.ratio, color='black')
        axarr[0, 1].set_title('Data CDF ratio')

        # Fitted CDF ratio
        axarr[1, 1].plot(self.bins, self.fitted_ratio, ls='--', color='purple')
        axarr[1, 1].text(0.65, 0.75, "Fitted\n$k$: %.3f\n$\\theta$: %.3f\n$\lambda$: %.3f"%(
                self.fitted_pars['c'], self.fitted_pars['loc'], self.fitted_pars['scale']),
                transform=axarr[1, 1].transAxes, size=14, color='purple')
        axarr[1, 1].set_title('Fitted CDF ratio')
        axarr[1, 1].set_xlabel('Distance from canopy top (m)')

        # ALS pdf (i.e. density histogram, in line form)
        axarr[0, 2].plot(self.bins, self.pdf_A, color='red')
        axarr[0, 2].set_title('Estimated ALS PDF')

        # Simulated ALS pdf
        axarr[1, 2].plot(self.bins, self.pdf_T * fit_ratio, color='purple', ls='--', label='fitted')
        axarr[1, 2].set_title('Fitted ALS PDF') # i.e. approximate result of pdf(z) for all TLS points 
        axarr[1, 2].set_xlabel('Distance from canopy top (m)')

        # Add adjusted curves
        if getattr(self, 'adjustments', None): # if adjustments have been made
            # Get adjusted pdf and ratio
            adj_pdf = self.pdf
            adj_ratio = adj_pdf(self.bins)
            
            # Adjusted CDF ratio
            axarr[1, 1].plot(self.bins, adj_ratio, ls='--', color='orange')
            axarr[1, 1].text(0.65, 0.5, "Adjusted\n$k$: %.3f\n$\\theta$: %.3f\n$\lambda$: %.3f"%(
                self.pars['c'], self.pars['loc'], self.pars['scale']),
                transform=axarr[1, 1].transAxes, size=14, color='orange')
            axarr[1, 1].set_title('Fitted and Adjusted CDF ratio')
        
            # Adjusted Simulated ALS pdf
            axarr[1, 2].plot(self.bins, self.pdf_T * adj_ratio, color='orange', ls='--', label='adjusted')
            axarr[1, 2].set_title('Fitted ALS PDF') # i.e. approximate result of pdf(z) for all TLS points 
            axarr[1, 2].set_xlabel('Distance from canopy top (m)')

            
        axarr[1, 2].legend(loc='best')
        fig.suptitle("centre: %s, bin width: %s"%(self.centre, self.bin_width))


class KDERatio(Mod):
    """ Carries out a Gaussian Kernel Density Estimation of the PDFs of both ALS and PDF,
        returning the unaltered ratio of these values as a pdf function for estimation.
        The resultant distribution should closely fit the input ALS, depending on the ALS bandwidth factor """

    def __init__(self, vox, subset='all', factor=None):
        """ Initialise model fitting with vox.
        
        Args:
            vox ::: a vox.Voxel instance
            subset ::: str choice of vox subset to model; 'all', 'tdc', 'hng' etc
            factor ::: numeric KDE bandwidth factor for ALS (`None` for automatic)
        """

        self.centre = vox.centre
        self.factor = factor
        self.subset = subset
        
        # Get data from vox
        self.A = vox.get_array('ALS', subset)
        self.T = vox.get_array('TLS', subset)
        
        self.run_fitting()
        
    def run_fitting(self):
        """Ensure voxel is valid and estimate ALS and TLS pdf with KDEs. """

        self._check_data()
        self._fit_model()
        
    def _check_data(self):
        """ Ensure there are at least two ALS and TLS points.
                If 1 point, a second point is added with value of +0.1 to original.
                If 0 points, signal sent to abort fitting.
        """

        for dataset in ('A', 'T'):
            D = getattr(self, dataset)
            npoints = len(D)
            if not npoints: # empty array
                self.valid = False
                return
            if npoints == 1:
                fake = D + 0.1
                setattr(self, dataset, np.concatenate([D, fake]))  # clone height value
        self.valid = True
    
    def _fit_model(self):
        """ Make gaussian KDE for ALS and TLS."""

        if self.valid:
            self.kde_A = gaussian_kde(self.A, self.factor)
            self.kde_T = gaussian_kde(self.T)
            self.pars = {'A_factor': self.kde_A.factor, 'T_factor': self.kde_T.factor}
        else: # constant pdf of 0 if data is empty
            self.pdf = lambda x: 0
            self.pars = None 
        
    def pdf(self, x):
        """ Return the ALS/TLS ratio of kde pdfs at x """
        
        fx =  self.kde_A.pdf(x)/self.kde_T(x)
        return fx
    
    def see_kde_fit(self, bin_width=1.):
        """ Plot ALS and TLS histogram and KDE; KDE ratio; approximated simulated PDF."""

        if not self.valid:
            self._plot_empty()
            return
        
        # xlabel mapping
        labmap = {k: 'z (m)' for k in ('z', 'all', 'hng', 'nearground')}
        labmap.update({k: 'Distance from canopy top (m)' for k in ('tdc', 'canopy')})
        x_label = labmap[self.subset]
        
        # Histogram tdc data, for comparison
        bin_edges, (dens_A, dens_T) = pc.histogram(self.A, self.T, density=True, bin_width=bin_width)
        bins = edges_to_centres(bin_edges)
        
        # Determine pdfs over range
        xs = np.linspace(bin_edges[0], bin_edges[-1], 100) # normally use bins, but now have option to use any values

        # Compare histogram and KDE, and ratios
        fig, axarr = plt.subplots(3,2, figsize=(12, 10), sharex=True)

        # Histograms
        axarr[0,0].bar(bins, dens_T, align='center', color='blue', label='TLS')
        axarr[0,0].set_ylabel('Density')
        axarr[0,0].set_title('TLS')
        axarr[0,0].legend(loc='best')

        axarr[0,1].bar(bins, dens_A, align='center', color='red', label='ALS')
        axarr[0,1].set_title('ALS')
        axarr[0,1].legend(loc='best')
        
        # KDE's
        pdf_T = self.kde_T.pdf(xs)
        axarr[1,0].plot(xs, pdf_T, color='blue')
        axarr[1,0].set_ylabel('Density')
        axarr[1,0].set_title('Kernel Density Estimate')
        axarr[1,0].annotate("factor: %.3f"%self.pars['T_factor'], (0.73,0.9), xycoords='axes fraction')

        axarr[1,1].plot(xs, self.kde_A.pdf(xs), color='red')
        axarr[1,1].set_title('Kernel Density Estimate')
        axarr[1,1].annotate("factor: %.3f"%self.pars['A_factor'], (0.73,0.9), xycoords='axes fraction')
        
        # KDE ratio
        pdf_ratio = self.pdf(xs)
        axarr[2,0].plot(xs, pdf_ratio, color='purple')
        axarr[2,0].set_title('TLS/ALS KDE Ratio')

        # Estimated ALS PDF (always identical to KDE pdf, since ratio is unaltered)
        axarr[2,1].plot(xs, pdf_ratio*pdf_T, color='orange')
        axarr[2,1].set_title('Simulated ALS PDF')

        fig.suptitle('%s'%(self.centre,))
        
    def _plot_empty(self):
        """ Plot nothing when data is invalid due to missing in ALS or TLS."""
        fig, ax = plt.subplots(1)
        ax.annotate("Can't plot, missing data", (0.35,0.5), xycoords='axes fraction')


class KDERatioQuick(KDERatio):
    """A faster version of KDERatio which uses a look-up table to evaluate pdf(x)."""
    
    def __init__(self, vox, subset='all', factor=None, LUT_intervals=1000):
        """."""
        self.LUT_intervals = LUT_intervals
        super(KDERatioQuick, self).__init__(vox, subset, factor)

    def run_fitting(self):
        """ Fit model as per usual, and generate PDF LUT)"""
        
        super(KDERatioQuick, self).run_fitting()
        self._make_LUT()

    def _make_LUT(self):
        """."""
        # Create LUT        
        zs = np.linspace(self.T.min(), self.T.max(), self.LUT_intervals) # sample entire z range in specified intervals
        fzs = 1.*self.kde_A.pdf(zs)/self.kde_T(zs)  # evaluate f(x) for LUT range
        self.LUT = np.vstack([zs, fzs]) # store LUT

    def pdf(self, xs):
        """ Return the ALS/TLS ratio of kde pdfs at the closest known values of xs """
        
        # Unpack LUT
        zs, fzs = self.LUT

        # Coerce numeric x to sequence
        try:
            len(xs)
        except TypeError:
            xs = np.array([xs,])
        
        # Estimate f(x) for x
        fxs = np.empty_like(xs) # to store f(x) values
        for i, x in enumerate(xs):
            ix = np.abs(zs - x).argmin() # find index of closest z value in LUT
            fxs[i] = fzs[ix] # use f(x) of nearest z
        return fxs



        
        
""" Helper functions """

def edges_to_centres(bin_edges):
    """ Return a len n-1 central values of a len n array of bin edges."""
    return bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.


