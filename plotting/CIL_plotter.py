
##
## CREATES PLOTS OF REPORT: saves them in working directory
## libaries needed: numpy, pandas, matplotlib
##

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

##
## PATH TO DATA DIRECTORY
##

path_drty = '.'
plot_2_name = 'ratings_distr.pdf'
plot_3_name = 'prop_lines_subf.pdf'

##
## HELPERS
##

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def decode_index(string):
  row_idx, col_idx = string.split('_')
  row_idx, col_idx = row_idx[1:], col_idx[1:] 
  return (int(row_idx)-1, int(col_idx)-1)

def sparse_to_matrix(sparse ,size):
  idxs, vals = sparse
  assert(len(idxs)==len(vals))
  A=np.empty(size)
  A.fill(np.nan)
  for idx, val in zip(idxs,vals):
    i,j = idx
    A[i,j] = int(val)
  return A, ~np.isnan(A)

class normalizer:
  def __init__(self,A,Mask,axis):
    if axis==0:
      self.means=np.nan_to_num(np.reshape(np.mean(A,axis=0,where=Mask),(1,A.shape[1])))
      self.stds=np.nan_to_num(np.reshape(np.std(A,axis=0,where=Mask),(1,A.shape[1])))
    if axis==1:
      self.means=np.nan_to_num(np.reshape(np.mean(A,axis=1,where=Mask),(A.shape[0],1)))
      self.stds=np.nan_to_num(np.reshape(np.std(A,axis=1,where=Mask),(A.shape[0],1)))
    self.axis=axis

  def normalize(self,A,Mask):
    temp_A = (A-self.means)/self.stds
    temp_A[~Mask] = A[~Mask]
    return temp_A

  def denormalize(self,A,Mask):
    temp_A = A*self.stds+self.means
    temp_A[~Mask] = A[~Mask]
    return temp_A

##
## FIGURE 2
##

# fetch and prepare input data
path_data = 'data_train.csv'

frame = pd.read_csv(path_drty+'/'+path_data)
in_idxs = frame['Id']
in_idxs = [decode_index(idx) for idx in in_idxs]
in_vals = frame['Prediction'].to_list()
in_sparse = (in_idxs, in_vals)
A, Omega = sparse_to_matrix(in_sparse, (10000,1000))

fig,axs=plt.subplots(1,3,figsize=(15,5))

fs = 15 #fontsize

for idx in range(3):
  ax=axs[idx]

  #adjust data
  if idx==0:
    data=A[Omega]
    bins=np.linspace(0.5,5.5,6)
    xlim=(0,6)
    title='(a) denormalized'

  else:
    if idx==1:
      norm=normalizer(A,Omega,0) #mean across users/rows
      title='(b) z-normalized across users/rows'

    if idx==2:
      norm=normalizer(A,Omega,1) #mean across items/cols
      title='(c) z-normalized across items/cols'

    data=norm.normalize(A,Omega)[Omega]
    bins=np.linspace(-5,3,30)
    xlim=(-5,3)

  #grid
  ax.grid(color='lightgray',linestyle='--',linewidth=0.5,zorder=0)
  ax.set_axisbelow(True)

  #histogram
  ax.hist(data,bins=bins, color="steelblue")

  #labels
  ax.set_xlabel('Values')
  ax.set_ylabel('Count')

  #title
  ax.set_title(title)

  #axis
  simpleaxis(ax)
  ax.set_xlim(xlim[0],xlim[1])

  if idx == 0: ax.set_xticks([1,2,3,4,5])
  else: ax.set_xticks(ax.get_xticks()[1:-1])
  
  old_yticks = ax.get_yticks()
  ax.set_yticks(np.linspace(old_yticks[0],old_yticks[-1],5))
  ax.yaxis.set_major_formatter(tck.FuncFormatter(lambda x,y: str(int(x)//1000)+'K')) #K shorthand

  #adjust font size of everyting
  for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(fs)

#save plot
plt.tight_layout()
plt.savefig(plot_2_name)
#plt.show()

##
## FIGURE 3
##

fig,axs = plt.subplots(1,3,figsize=(6.4*3, 4.8))
markers = ['x','+','1']
lstyles = ['solid','dashed','dotted']
colors = ["steelblue", "goldenrod", "seagreen"]

file_ends = ['_prop.csv','_prop_re.csv','_prop_alt.csv']

for jdx in range(3):
  ax = axs[jdx]

  for idx,(D,W) in enumerate([(0,16),(1,8),(1,16)]):
    #prep data
    file_name = str(D)+'_'+str(W)+file_ends[jdx]
    df = pd.read_csv(path_drty+'/probabilities_data/'+file_name)
    df['mean'] = df.iloc[:, -5:].mean(axis=1)
    df['std'] = df.iloc[:, -5:].std(axis=1)
    df = df.drop(['score_'+str(x) for x in range(5)],axis=1)

    if jdx < 2: df = df.drop(df[df['Dropout_rate'] == 0].index)

    t = 'D='+str(D)+', W='+str(W)+', T='+str(df['epochs'][1])

    #plot
    ax.plot(df['Dropout_rate'], df['mean'], label = t, linewidth = 2.5,
            marker = markers[idx], ms = 15, mec = colors[idx], mew = 2.5,
            ls = lstyles[idx], color = colors[idx])

  #grid
  ax.grid(color='lightgray',linestyle='--',linewidth=1,zorder=0)
  ax.set_axisbelow(True)

  #labels
  ax.set_xlabel('Dropout probability')
  ax.set_ylabel('RMSE')

  #title
  if jdx == 0: ax.set_title('(a) Original loss (N=8)')
  elif jdx == 1: ax.set_title('(b) Original loss (N=1)')
  elif jdx == 2: ax.set_title('(c) Alternative loss (N=1)')

  #axis
  simpleaxis(ax)
  if jdx == 2: ax.set_ylim(0.97,1.06)
  else: ax.set_ylim(0.975,0.9925)

  old_yticks = ax.get_yticks()
  ax.set_yticks(np.linspace(old_yticks[0],old_yticks[-1],6))

  #adjust font size of everything
  for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
      item.set_fontsize(fs)

ax.legend(
    frameon=False,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.02),
    bbox_transform=fig.transFigure,
    fontsize=fs
)

plt.subplots_adjust(wspace=-1, hspace=1)
plt.tight_layout()
plt.savefig(plot_3_name)
#plt.show()
