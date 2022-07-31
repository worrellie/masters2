import scipy
import numpy as np
import csv
import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt
import astropy.cosmology as cosmo
from astropy.io import fits
from astropy.table import Table

#################
# colour scheme #
#################
grn ='#54AA68' # comp
blu = '#4B73B2' # sfg
ong = '#DE8551' # agn
gry = '#8D8D8D'
yel = '#CDBA75' # train
prp = '#8273B4' # test

filename = "data_for_BPT_classification.fits" # this file has been reduced based on the criteria in Section 2.1

dat_tab = Table.read(filename, format = 'fits')

df = dat_tab.to_pandas()
idx = range(0, 5107)
df.insert(0,'idx', idx)
df.set_index('idx')

# SNR cuts (already implemented in file)

# Kauffmann/ Kewley demarcation lines
def sfg_y(x):
    return 0.61/(np.minimum(np.log10(x)-0.05, -1e-5))+1.3

def comp_y(x):
    return 0.61/(np.minimum(np.log10(x)-0.05, -1e-5))+1.3 # updated to 0.61 from 0.6

def agn_y(x):
    return 0.61/(np.minimum(np.log10(x)-0.47, -1e-5))+1.19 

comp_red = np.logical_and(comp_y(df['n2']/df['ha'])<np.log10(df['o3']/df['hb']), 0.61/(np.log10(df['n2']/df['ha'])-0.47)+1.19>np.log10(df['o3']/df['hb']))
sfg_red = np.logical_and(np.log10(df['o3']/df['hb'])<sfg_y(df['n2']/df['ha']), np.log10(df['n2']/df['ha'])<0.0)
agn_red = np.logical_and(agn_y(df['n2']/df['ha'])<np.log10(df['o3']/df['hb']), comp_y(df['n2']/df['ha'])<np.log10(df['o3']/df['hb']))

print('Total number of galaxies: ', len(df))
print('--------------------------------------------')

##### classify galaxies #####
classified = np.logical_or(np.logical_or(agn_red, sfg_red), comp_red)
c = np.where(comp_red)[0]
s = np.where(sfg_red)[0]
a = np.where(agn_red)[0]
un = np.where(classified != True)[0]

print('Classifying...')

##### check have correct number of galaxies #####
if len(un) != 0:
    print('Some galaxies unclassified')
else:
    print('All galaxies classified')

if  len(c) + len(s) + len(a) + len(un) > len(df):
    print('Overclassifying galaxies')
elif len(c) + len(s) + len(a) + len(un) < len(df):
    print('Missing some galaxies...')
else:
    print('Correct total number of galaxies')
    
classif = []
for i in range(0, 5107):
    if i in c:
        classif.append('COMP')
    elif i in s:
        classif.append('SFG')
    elif i in a:
        classif.append('AGN')
    else:
        print('Unclassified galaxy present')

print('--------------------------------------------')

print('Number of galaxies: ')
print('Composites: ', len(c))
print('SFGs: ', len(s))
print('AGN: ', len(a))
print('Unclassified: ', len(un))

# features
# n2_ha = df['n2']/df['ha']
# o3_hb = df['o3']/df['hb']
# g_r = df['mag_g']-df['mag_r']
# r_i = df['mag_r']-df['mag_i']
# i_z = df['mag_i']-df['mag_z']
# u_g = df['mag_u']-data['mag_g']

# add classification column
df.insert(0, 'class', classif)

# write to file 
t = Table.from_pandas(df)
t.write('data_for_ML.fits', overwrite = True)

##### plot BPT with demarcation lines #####

plt.figure(1, figsize = (9,9))

plt.scatter(np.log10(df.loc[comp_red,'n2']/df.loc[comp_red,'ha']), np.log10(df.loc[comp_red,'o3']/df.loc[comp_red,'hb']), s=3, color=grn, label='Comp', alpha=0.4)
plt.scatter(np.log10(df.loc[sfg_red,'n2']/df.loc[sfg_red,'ha']), np.log10(df.loc[sfg_red,'o3']/df.loc[sfg_red,'hb']), s=3, color=blu, label='SFG', alpha=0.4)
plt.scatter(np.log10(df.loc[agn_red,'n2']/df.loc[agn_red,'ha']), np.log10(df.loc[agn_red,'o3']/df.loc[agn_red,'hb']), s=3, color=ong, label='AGN', alpha=0.4)

x = np.linspace(0.01, 3, 290)

plt.plot(np.log10(x), sfg_y(x), c = gry, linestyle = '--')
# plt.plot(np.log10(x), comp_y(x), c = gry, linestyle = '--')
plt.plot(np.log10(x), agn_y(x), c = gry, linestyle = '--')

# plt.title(r'BPT with reduced classification conditions')
plt.xlabel(r'$\textrm{log}_{10}\textrm{([NII]}/\textrm{H}\alpha$)')
plt.ylabel(r'$\textrm{log}_{10}\textrm{([OIII]}/\textrm{H}\beta$)')

plt.ylim(-1.5, 1.5)

plt.legend()

plt.savefig('./plots/bpt/bpt.pdf')

# plt.show()