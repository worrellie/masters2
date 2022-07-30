import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.cosmology as cosmo
from astropy.io import fits
from astropy.table import Table
from mlxtend.plotting import scatterplotmatrix
from sklearn import linear_regression

#################
# colour scheme #
#################
grn ='#54AA68'
blu = '#4B73B2'
ong = '#DE8551'
gry = '#8D8D8D'
yel = '#CDBA75'
prp = '#8273B4'

filename = "zhang_cut.fits" # this file has been reduced based on the criteria in Section 2.1

dat_tab = Table.read(filename, format = 'fits')

data = dat_tab.to_pandas()
# idx = range(0, 5107)
# data.insert(0,'idx', idx)
# data.set_index('idx')