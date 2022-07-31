import scipy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import astropy.cosmology as cosmo
from astropy.io import fits
from astropy.table import Table
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
import seaborn as sns
from sklearn.impute import SimpleImputer
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

pd.set_option('display.max_columns', None)

# print(fm.findSystemFonts(fontpaths = None, fontext = 'ttf'))

## diasble latex and set font and size
plt.rcParams.update({'text.usetex' : False})
path = 'C:/Users/elean/AppData/Local/Microsoft/Windows/Fonts/cmunrm.ttf'
prop = fm.FontProperties(fname = path)
print(prop.get_name())
mpl.rcParams['font.family'] = prop.get_name()
# plt.rcParams.update({'font.serif' : 'serif'})
# plt.rcParams.update({'font.family' : 'computer modern'})
plt.rcParams.update({'font.size' : 10})

#################
# colour scheme #
#################
grn ='#54AA68'
blu = '#4B73B2'
ong = '#DE8551'
gry = '#8D8D8D'
yel = '#CDBA75'
prp = '#8273B4'

# cdict = {}

e_cmap = LinearSegmentedColormap.from_list('', [blu, ong])

#####################################
############# FUNCTIONS #############
#####################################

def iszero(df):
    print('Number of missing values:')
    i = []
    for col in df.columns:
        count = (df[col] == 0).sum()
        ind = df.index[df[col] == 0].tolist()
        if len(ind) != 0:
            i.append(ind)
        print(col , ' : ', count)
    indices = sorted(list(set(k for j in i for k in j)))
    # replace missing values with NaN
    df.replace(to_replace = 0, value = np.nan, inplace = True)
    print('----------------------------------------')
    return indices

def isnegative(df):
    print('Number of negative values:')
    i = []
    for col in df.columns:
            count = (df[col] < 0).sum()
            ind = df.index[df[col] < 0].tolist()
            if len(ind) != 0:
                i.append(ind)
            print(col , ' : ', count)
    indices = sorted(list(set(k for j in i for k in j)))
    print('----------------------------------------')
    return indices

filename = "data_for_eda.fits" # this file has been reduced based on the criteria in Section 2.1

dat_tab = Table.read(filename, format = 'fits')

df = dat_tab.to_pandas()
# idx = range(0, 5107)
# data.insert(0,'idx', idx)
# data.set_index('idx')

df.drop(labels = ['idx'], axis = 1, inplace = True)

# data information
f = open('data_describe.txt', 'w')
descr = df.describe()
f.write(descr.to_latex())

classes = df['class']

df.drop(labels = ['class'], axis = 1, inplace = True)

missing = iszero(df)
n_missing = len(missing)
# remove rows with missing data (in this case it's only colour data)
df.drop(missing, axis=0, inplace=True)
# decided not to impute missing values because std is large
# imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')

features = df.columns
n_features = len(features)

# plot_args = {'s': 5, 'edgecolors': 'face', 'color' : blu, }
pp, pp_axes = scatterplotmatrix(df.values, figsize = (20,20), names = features, alpha = 0.5, color = blu, s = 10, label = 'SFG') #sfg
# pp, pp_axes = scatterplotmatrix(df.values, figsize = (20,20), names = features, alpha = 0.5, color = ong, s = 10, label = 'AGN') # agn
# pp, pp_axes = scatterplotmatrix(df.values, figsize = (20,20), names = features, alpha = 0.5, color = grn, s = 10, label = 'COMP') # comp

#region
pp_axes[1,0].set_xlim(0, 3000)
pp_axes[1,0].set_ylim(0, 1000)

pp_axes[2,0].set_xlim(0, 5000)
pp_axes[2,0].set_ylim(0, 1000)

pp_axes[2,1].set_xlim(0, 5000)
pp_axes[2,1].set_ylim(0, 6000)

pp_axes[3,0].set_xlim(0,3000)
pp_axes[3,0].set_ylim(0,1000)

pp_axes[3,1].set_xlim(0,2000)
pp_axes[3,1].set_ylim(0,4000)

pp_axes[3,2].set_xlim(0, 2000)
pp_axes[3,2].set_ylim(0, 5000)

# pp_axes[4,0].set_xlim()
pp_axes[4,0].set_ylim(0, 1000)

# pp_axes[4,1].set_xlim()
pp_axes[4,1].set_ylim(0, 4000)

# pp_axes[4,2].set_xlim()
pp_axes[4,2].set_ylim(0,5000)

# pp_axes[4,3].set_xlim()
pp_axes[4,3].set_ylim(0,2000)

pp_axes[5,0].set_xlim(-5,5)
pp_axes[5,0].set_ylim(0,1000)

pp_axes[5,1].set_xlim(-5,5)
pp_axes[5,1].set_ylim(0,4000)

pp_axes[5,2].set_xlim(-5,5)
pp_axes[5,2].set_ylim(0,5000)

pp_axes[5,3].set_xlim(-5,5)
pp_axes[5,3].set_ylim(0,2000)

pp_axes[5,4].set_xlim(-5,5)
# pp_axes[5,4].set_ylim()

pp_axes[6,0].set_xlim(-3,4)
pp_axes[6,0].set_ylim(0,1000)

pp_axes[6,1].set_xlim(-3,4)
pp_axes[6,1].set_ylim(0,6000)

pp_axes[6,2].set_xlim(-3,4)
pp_axes[6,2].set_ylim(0,5000)

pp_axes[6,3].set_xlim(-3,4)
pp_axes[6,3].set_ylim(0,2000)

pp_axes[6,4].set_xlim(-3,4)
# pp_axes[6,4].set_ylim()

pp_axes[6,5].set_xlim(-3,4)
pp_axes[6,5].set_ylim(-5,4)

# pp_axes[7,0].set_xlim()
pp_axes[7,0].set_ylim(0,1000)

pp_axes[7,1].set_ylim(0,4000)

pp_axes[7,2].set_ylim(0,5000)

pp_axes[7,3].set_ylim(0,2000)

pp_axes[7,4].set_ylim(-5, 10)

pp_axes[7,5].set_ylim(-5, 5)

pp_axes[7,6].set_ylim(-3,4)

# for i in range(n_features):
#     pp_axes[i,i].set_xlim(0, 200)
#endregion
plt.tight_layout()
plt.legend()

# plt.savefig('./plots/eda/pairplot.pdf')

corr_matrix = np.corrcoef(df[features].values.T)
hm = heatmap(corr_matrix, row_names = features, column_names = features, figsize = (9,9), cmap = e_cmap)

# plt.savefig('./plots/eda/corr_matrix.pdf')

# plt.show()

df.insert(0, 'class', classes)

t = Table.from_pandas(df)
t.write('data_for_ML.fits')