'''
testing our functional statistics program!!
'''

from example_scripts import functional_stats as fs
import numpy as np

EPSILON = 1e-9

############## Testing measure_likelihood ##############

def test_upper_error():
	measurements = [0]
	upper_errors = [1]
	lower_errors = [0.5]
	model = [0.5]
	assert fs.measure_likelihood(model, measurements, lower_errors, upper_errors) == 0.25


def test_lower_error():
	measurements = [0]
	upper_errors = [1]
	lower_errors = [0.5]
	model = [-0.5]
	assert fs.measure_likelihood(model, measurements, lower_errors, upper_errors) == 1.0

def test_no_error():
	measurements = [0]
	upper_errors = [1]
	lower_errors = [0.5]
	model = [0]
	assert fs.measure_likelihood(model, measurements, lower_errors, upper_errors) == 0
	
############## Testing likelihood_from_lower_limits ##############

def test_below_lower_limits():
	model = [0]
	lower_limits = [1]
	confidence = 0.95
	# we are below our lower limit
	assert fs.likelihood_from_lower_limits(model, lower_limits, confidence) == np.log(0.95)

def test_above_lower_limits():
	model = [0]
	lower_limits = [-1]
	confidence = 0.95
	# we are above our lower limit
	assert (fs.likelihood_from_lower_limits(model, lower_limits, confidence) - np.log(0.05)) < EPSILON

def test_different_lower_confidences():
	model = [0,0]
	lower_limits = [1,2]
	confidence = [0.95, 0.50]
	# we are below our lower limits
	assert fs.likelihood_from_lower_limits(model, lower_limits, confidence) == np.log(0.95) + np.log(0.5)
	
############## Testing likelihood_from_upper_limits ##############

def test_below_upper_limits():
	model = [0]
	upper_limits = [1]
	confidence = 0.95
	# we are below our upper limit
	assert (fs.likelihood_from_upper_limits(model, upper_limits, confidence) -  np.log(0.05)) < EPSILON

def test_above_upper_limits():
	model = [0]
	upper_limits = [-1]
	confidence = 0.95
	# we are above our upper limit
	assert (fs.likelihood_from_upper_limits(model, upper_limits, confidence) - np.log(0.95)) < EPSILON

def test_different_upper_confidences():
	model = [0,0]
	upper_limits = [-1,-2]
	confidence = [0.95, 0.50]
	# we are above our upper limits
	assert (fs.likelihood_from_upper_limits(model, upper_limits, confidence) - (np.log(0.95) + np.log(0.5))) < EPSILON