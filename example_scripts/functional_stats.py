'''
functional_stats.py

Invented to be a version of stats.py that uses /functions/ instead of /big
long scripts/ to calculate my statistics. 

Here to test that the (i) chi-square of rates, (ii)
'''

import numpy as np
#import astropy.units as u
#from astropy.cosmology import FlatLambdaCDM
#import matplotlib.pyplot as plt

__author__ = ['Benjamin Metha', ]
__date__ = '2019-09-19'
__cite__ = 'https://github.com/astrobenji'

def measure_likelihood(model, measurements, lower_error, upper_error, normaliser=1):
    '''
    Calculates the chi squared statistic for a model's agreement to observational data.
    
    Parameters
    ----------
    normaliser: float
        What this model should be normalised to to maximise likelihoods.
        defaults to 1.
    
    model: (,N) ndarray
        The model that we are trying to fit to the data.
        
    measurements: (,N) ndarray
        The data that we are trying to fit the model to. 
        
    lower_error: (,N) ndarray
    	lower errors on our measurements.
    
    upper_error: (,N) ndarray
    	upper errors on our measurements.
    
    Returns/
    -------
    chisq: float
        A number related to the -1* the log likelihood of this model.
        Minimising this quantity yields the most likely model.
    '''
    
    # First, check that all of our arrays are of the same length.
    n = len(measurements)
    assert len(model) == n
    assert len(lower_error) == n
    assert len(upper_error) == n
    
    
    normed_model = normaliser*model
    chisq = 0
    
    for ii in range(n):
        if normed_model[ii] > measurements[ii]:
            error = upper_error[ii]
        else:
            error = lower_error[ii]
        chisq += ((normed_model[ii] - measurements[ii])/error)**2
    return chisq 
    
def likelihood_from_lower_limits(model, lower_limits, confidence, normaliser=1):
    '''
    Parameters
    ----------
    normaliser: float
        What this model should be normalised to to maximise likelihoods.
        defaults to 1.
        
    model: (,N) ndarray
        The model that we are trying to fit to the data.
        
    confidence: (,N) ndarray or float
        The confidence on these lower limits -- eg. how likely is it that the true value 
        of this measurement is below this lower limit?
        
    lower_limits: (,N) ndarray
    	Lower limits on our measurements.
    	
    Returns
    -------
    ln_likelihood: float
    	The likelihood of this model.
    '''
    n = len(model)
    assert len(lower_limits) == n
    try:
        assert len(confidence) == n
    except TypeError:
    	# Turn it into an array!
        confidence = np.ones(n)*confidence
    
    normed_model = normaliser*model
    ln_likelihood = 0
    for ii in range(n):
        if normed_model[ii] < lower_limits[ii]:
            ln_likelihood += np.log(confidence[ii])
        else:
            ln_likelihood += np.log(1 - confidence[ii])
            
    return ln_likelihood
    
def likelihood_from_upper_limits(model, upper_limits, confidence, normaliser=1):
	try:
		new_confidence = np.ones(len(confidence)) - np.array(confidence)
	except TypeError:
		new_confidence = 1 - confidence
	return likelihood_from_lower_limits(model, upper_limits, new_confidence, normaliser)