# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <CENTER><img src="../../images/ATLASOD.gif" style="width:50%"></CENTER>

# %% [markdown]
# # How to rediscover the Higgs boson yourself - with a Neural Network!
# This notebook uses ATLAS Open Data http://opendata.atlas.cern to show you the steps to apply Machine Learning in search for the Higgs boson!
#
# ATLAS Open Data provides open access to proton-proton collision data at the LHC for educational purposes. ATLAS Open Data resources are ideal for high-school, undergraduate and postgraduate students.
#
# Notebooks are web applications that allow you to create and share documents that can contain for example:
# 1. live code
# 2. visualisations
# 3. narrative text
#
# This notebook builds on [HZZAnalysis.ipynb](https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata/blob/master/13-TeV-examples/uproot_python/HZZAnalysis.ipynb) in the same folder as this notebook. 
#
# HZZAnalysis.ipynb loosely follows the [discovery of the Higgs boson by ATLAS](https://www.sciencedirect.com/science/article/pii/S037026931200857X) (mostly Section 4 and 4.1)
#
# Notebooks are a perfect platform to develop Machine Learning for your work, since you'll need exactly those 3 things: code, visualisations and narrative text!
#
# We're interested in Machine Learning because we can design an algorithm to figure out for itself how to do various analyses, potentially saving us countless human-hours of design and analysis work.
#
# Machine Learning use within ATLAS includes: 
# * particle tracking
# * particle identification
# * signal/background classification
# * and more!
#
# This notebook will focus on signal/background classification.
#
# By the end of this notebook you will be able to:
# 1. run a Neural Network to classify signal and background
# 2. know some things you can change to improve your Neural Network
#
# Feynman diagram pictures are borrowed from our friends at https://www.particlezoo.net

# %% [markdown]
# <CENTER><img src="images/feynman_diagrams/HZZ_feynman.png" style="width:40%"></CENTER>

# %% [markdown]
# <a id='contents'></a>

# %% [markdown]
# Contents: 
#
# [Running a Jupyter notebook](#running) <br />
# [First time setup on your computer (no need on mybinder)](#setup_computer) <br />
# [To setup everytime](#setup_everytime) <br />
# [Lumi, fraction, file path](#fraction) <br />
# [Samples](#samples) <br />
# [Changing a cut](#changing_cut) <br />
# [Applying a cut](#applying_cut) <br />
# [Optimisation](#optimisation) <br />
# [Neural Network](#NN) <br />
#   [Training and Testing split](#train_test) <br />
#   [Data Preprocessing](#preprocessing) <br />
#   [Training the Neural Network model](#training) <br />
#   [Receiver Operating Characteristic (ROC) curve](#ROC) <br />
#   [Overtraining check](#overtraining) <br />
#   [Optimisation](#NN_optimisation) <br />
# [Going further](#going_further) <br />

# %% [markdown]
# <a id='running'></a>

# %% [markdown]
# ## Running a Jupyter notebook
#
# To run the whole Jupyter notebook, in the top menu click Cell -> Run All.
#
# To propagate a change you've made to a piece of code, click Cell -> Run All Below.
#
# You can also run a single code cell, by clicking Cell -> Run Cells, or using the keyboard shortcut Shift+Enter.

# %% [markdown]
# <a id='setup_computer'></a>

# %% [markdown]
# ## First time setup on your computer (no need on mybinder)
# This first cell only needs to be run the first time you open this notebook on your computer. 
#
# If you close Jupyter and re-open on the same computer, you won't need to run this first cell again.
#
# If you open on mybinder, you don't need to run this cell.

# %%
import sys
# !{sys.executable} -m pip install --upgrade --user pip # update the pip package installer
# !{sys.executable} -m pip install uproot3 pandas numpy matplotlib sklearn --user # install required packages

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='setup_everytime'></a>

# %% [markdown]
# ## To setup everytime
# Cell -> Run All Below
#
# to be done every time you re-open this notebook
#
# We're going to be using a number of tools to help us:
# * uproot: lets us read .root files typically used in particle physics into data formats used in Machine Learning
# * pandas: lets us store data as dataframes, a format widely used in Machine Learning
# * numpy: provides numerical calculations such as histogramming
# * matplotlib: common tool for making plots, figures, images, visualisations

# %%
import uproot3 # for reading .root files
import pandas as pd # to store data as dataframe
import time # to measure time to analyse
import math # for mathematical functions such as square root
import numpy as np # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import AutoMinorLocator # for minor ticks

import infofile # local file containing info on cross-sections, sums of weights, dataset IDs

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='fraction'></a>

# %% [markdown]
# ## Lumi, fraction, file path
#
# General definitions of fraction of data used, where to access the input files

# %%
lumi = 10 # fb-1 # data_A+B+C+D

fraction = 0.03 # reduce this is you want the code to run quicker
                                                                                                                                  
#tuple_path = "Input/4lep/" # local 
tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" # web address

# %% [markdown]
# <a id='samples'></a>

# %% [markdown]
# ## Samples
#
# In this notebook we only process the signal <span style="color:blue">H->ZZ</span> and the main background <span style="color:red">ZZ</span>, for illustration purposes. You can add data and the Z and ttbar <span style="color:red">backgrounds</span> after if you wish.

# %%
samples = {

    'ZZ' : {
        'list' : ['llll']
    },

    r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$' : { # H -> ZZ -> llll
        'list' : ['ggH125_ZZ4lep'] # gluon-gluon fusion
    }

}


# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# Define function to get data from files. 
#
# The datasets used in this notebook have already been filtered to include at least 4 leptons per event, so that processing is quicker.

# %%
def get_data_from_files():

    data = {} # define empty dictionary to hold dataframes
    for s in samples: # loop over samples
        print('Processing '+s+' samples') # print which sample
        frames = [] # define empty list to hold data
        for val in samples[s]['list']: # loop over each file
            if s == 'data': prefix = "Data/" # Data prefix
            else: # MC prefix
                prefix = "MC/mc_"+str(infofile.infos[val]["DSID"])+"."
            fileString = tuple_path+prefix+val+".4lep.root" # file name to open
            temp = read_file(fileString,val) # call the function read_file defined below
            frames.append(temp) # append dataframe returned from read_file to list of dataframes
        data[s] = pd.concat(frames) # dictionary entry is concatenated dataframes
    
    return data # return dictionary of dataframes


# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# Define function to get cross-section weight

# %%
def get_xsec_weight(sample):
    info = infofile.infos[sample] # open infofile
    xsec_weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) #*1000 to go from fb-1 to pb-1
    return xsec_weight # return cross-section weight


# %% [markdown]
# Define function to calculate weight of MC event

# %%
def calc_weight(xsec_weight, mcWeight, scaleFactor_PILEUP,
                scaleFactor_ELE, scaleFactor_MUON, 
                scaleFactor_LepTRIGGER ):
    return xsec_weight*mcWeight*scaleFactor_PILEUP*scaleFactor_ELE*scaleFactor_MUON*scaleFactor_LepTRIGGER


# %% [markdown]
# We add functions to return the individual lepton transverse momenta, in GeV

# %%
def calc_lep_pt_i(lep_pt,i):
    return lep_pt[i]/1000 # /1000 to go from MeV to GeV


# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='changing_cut'></a>

# %% [markdown]
# ## Changing a cut
#
# We apply 'cuts' to throw away collisions that have properties different to the signal we're looking for.
#
# If you change a cut: Cell -> Run All Below.
#
# If you change a cut here, you also need to make sure the cut is applied in the "[Applying a cut](#applying_cut)" cell.

# %%
# cut on lepton charge
# paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"
def cut_lep_charge(lep_charge):
# throw away when sum of lepton charges is not equal to 0
# first lepton is [0], 2nd lepton is [1] etc
    return lep_charge[0] + lep_charge[1] + lep_charge[2] + lep_charge[3] != 0

# cut on lepton type
# paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"
def cut_lep_type(lep_type):
# for an electron lep_type is 11
# for a muon lep_type is 13
# throw away when none of eeee, mumumumu, eemumu
    sum_lep_type = lep_type[0] + lep_type[1] + lep_type[2] + lep_type[3]
    return (sum_lep_type != 44) and (sum_lep_type != 48) and (sum_lep_type != 52)


# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='applying_cut'></a>

# %% [markdown]
# ## Applying a cut
# If you add a cut: Cell -> Run All Below

# %%
def read_file(path,sample):
    start = time.time() # start the clock
    print("\tProcessing: "+sample) # print which sample is being processed
    data_all = pd.DataFrame() # define empty pandas DataFrame to hold all data for this sample
    tree = uproot3.open(path)["mini"] # open the tree called mini
    numevents = uproot3.numentries(path, "mini") # number of events
    if 'data' not in sample: xsec_weight = get_xsec_weight(sample) # get cross-section weight
    for data in tree.iterate(['lep_charge','lep_type','lep_pt',
                              # uncomment these variables if you want to calculate masses 
                              #,'lep_eta','lep_phi','lep_E', 
                              # add more variables here if you make cuts on them 
                              'mcWeight','scaleFactor_PILEUP',
                              'scaleFactor_ELE','scaleFactor_MUON',
                              'scaleFactor_LepTRIGGER'
                             ], # variables to calculate Monte Carlo weight
                             outputtype=pd.DataFrame, # choose output type as pandas DataFrame
                             entrystop=numevents*fraction): # process up to numevents*fraction

        nIn = len(data.index) # number of events in this batch

        if 'data' not in sample: # only do this for Monte Carlo simulation files
            # multiply all Monte Carlo weights and scale factors together to give total weight
            data['totalWeight'] = np.vectorize(calc_weight)(xsec_weight,
                                                            data.mcWeight,
                                                            data.scaleFactor_PILEUP,
                                                            data.scaleFactor_ELE,
                                                            data.scaleFactor_MUON,
                                                            data.scaleFactor_LepTRIGGER)

        # cut on lepton charge using the function cut_lep_charge defined above
        fail = data[ np.vectorize(cut_lep_charge)(data.lep_charge) ].index
        data.drop(fail, inplace=True)

        # cut on lepton type using the function cut_lep_type defined above
        fail = data[ np.vectorize(cut_lep_type)(data.lep_type) ].index
        data.drop(fail, inplace=True)

        # return the individual lepton transverse momenta in GeV
        data['lep_pt_1'] = np.vectorize(calc_lep_pt_i)(data.lep_pt,1)
        data['lep_pt_2'] = np.vectorize(calc_lep_pt_i)(data.lep_pt,2)
        
        # dataframe contents can be printed at any stage like this
        #print(data)

        # dataframe column can be printed at any stage like this
        #print(data['lep_pt'])

        # multiple dataframe columns can be printed at any stage like this
        #print(data[['lep_pt','lep_eta']])

        nOut = len(data.index) # number of events passing cuts in this batch
        data_all = data_all.append(data) # append dataframe from this batch to the dataframe for the whole sample
        elapsed = time.time() - start # time taken to process
        print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after
    
    return data_all # return dataframe containing events passing all cuts



# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# This is where the processing happens (this will take some minutes)

# %%
start = time.time() # time at start of whole processing
data = get_data_from_files() # process all files
elapsed = time.time() - start # time after whole processing
print("Time taken: "+str(round(elapsed,1))+"s") # print total time taken to process every file

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='optimisation'></a>

# %% [markdown]
# ## Optimisation
#
# Here we define histograms for the variables that we'll look to optimise

# %%
lep_pt_2 = { # dictionary containing plotting parameters for the lep_pt_2 histogram
    # change plotting parameters
    'bin_width':1, # width of each histogram bin
    'num_bins':13, # number of histogram bins
    'xrange_min':7, # minimum on x-axis
    'xlabel':r'$lep\_pt$[2] [GeV]', # x-axis label
}

lep_pt_1 = { # dictionary containing plotting parameters for the lep_pt_1 histogram
    # change plotting parameters
    'bin_width':1, # width of each histogram bin
    'num_bins':28, # number of histogram bins
    'xrange_min':7, # minimum on x-axis
    'xlabel':r'$lep\_pt$[1] [GeV]', # x-axis label
}

SoverB_hist_dict = {'lep_pt_2':lep_pt_2,'lep_pt_1':lep_pt_1} 
# add a histogram here if you want it plotted

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# Here we define a function to illustrate the optimum cut value on individual variables, based on <span style="color:blue">signal</span> to <span style="color:red">background</span> ratio.

# %%
def plot_SoverB(data):
    
    signal = r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$' # which sample is the signal

    # *******************
    # general definitions (shouldn't need to change)

    for x_variable,hist in SoverB_hist_dict.items(): # access the dictionary of histograms defined in the cell above

        h_bin_width = hist['bin_width'] # get the bin width defined in the cell above
        h_num_bins = hist['num_bins'] # get the number of bins defined in the cell above
        h_xrange_min = hist['xrange_min'] # get the x-range minimum defined in the cell above
        h_xlabel = hist['xlabel'] # get the x-axis label defined in the cell above
    
        bin_edges = [ h_xrange_min + x*h_bin_width for x in range(h_num_bins+1) ] # bin limits
        bin_centres = [ h_xrange_min+h_bin_width/2 + x*h_bin_width for x in range(h_num_bins) ] # bin centres
        
        signal_x = data[signal][x_variable] # histogram the signal
    
        mc_x = [] # define list to hold the Monte Carlo histogram entries

        for s in samples: # loop over samples
            if s not in ['data', signal]: # if not data nor signal
                mc_x = [*mc_x, *data[s][x_variable] ] # append to the list of Monte Carlo histogram entries

    
    
        # *************
        # Signal and background distributions
        # *************
        distributions_axes = plt.gca() # get current axes
 
        mc_heights = distributions_axes.hist(mc_x, bins=bin_edges, color='red', 
                                             label='Total background',
                                             histtype='step', # lineplot that's unfilled
                                             density=True ) # normalize to form probability density
        signal_heights = distributions_axes.hist(signal_x, bins=bin_edges, color='blue',
                                                 label=signal, 
                                                 histtype='step', # lineplot that's unfilled
                                                 density=True, # normalize to form probability density
                                                 linestyle='--' ) # dashed line
        
        distributions_axes.set_xlim( left=bin_edges[0], right=bin_edges[-1] ) # x-limits of the distributions axes
        distributions_axes.set_ylabel('Arbitrary units' ) # y-axis label for distributions axes
        distributions_axes.set_ylim( top=max(signal_heights[0])*1.3 ) # set y-axis limits
        plt.title('Signal and background '+x_variable+' distributions') # add title
        distributions_axes.legend() # draw the legend
        distributions_axes.set_xlabel( h_xlabel ) # x-axis label
        
        # Add text 'ATLAS Open Data' on plot
        plt.text(0.05, # x
                 0.93, # y
                 'ATLAS Open Data', # text
                 transform=distributions_axes.transAxes, # coordinate system used is that of distributions_axes
                 fontsize=13 ) 
        # Add text 'for education' on plot
        plt.text(0.05, # x
                 0.88, # y
                 'for education', # text
                 transform=distributions_axes.transAxes, # coordinate system used is that of distributions_axes
                 style='italic',
                 fontsize=8 )  
    
        plt.show() # show the Signal and background distributions
    
    
        # *************
        # Signal to background ratio
        # *************
        plt.figure() # start new figure
        SoverB = [] # list to hold S/B values
        for cut_value in bin_edges: # loop over bins
            signal_weights_passing_cut = sum(data[signal][data[signal][x_variable]>cut_value].totalWeight)
            background_weights_passing_cut = 0 # start counter for background weights passing cut
            for s in samples: # loop over samples
                if s not in ['data', signal]: # if not data nor signal
                    background_weights_passing_cut += sum(data[s][data[s][x_variable]>cut_value].totalWeight)
            if background_weights_passing_cut!=0: # some background passes cut
                SoverB_value = signal_weights_passing_cut/background_weights_passing_cut
                SoverB_percent = 100*SoverB_value # multiply by 100 for percentage
                SoverB.append(SoverB_percent) # append to list of S/B values
        
        SoverB_axes = plt.gca() # get current axes
        SoverB_axes.plot( bin_edges[:len(SoverB)], SoverB ) # plot the data points
        SoverB_axes.set_xlim( left=bin_edges[0], right=bin_edges[-1] ) # set the x-limit of the main axes
        SoverB_axes.set_ylabel( 'S/B (%)' ) # write y-axis label for main axes
        plt.title('Signal to background ratio for different '+x_variable+' cut values', family='sans-serif')
        SoverB_axes.set_xlabel( h_xlabel ) # x-axis label 
        
        plt.show() # show S/B plot
    
    return

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# Here we call our function to illustrate the optimum cut value on individual variables, based on <span style="color:blue">signal</span> to <span style="color:red">background</span> ratio.
#
# We're not doing any Machine Learning yet! We're looking at the variables we'll later use for Machine Learning.
#
# Let's talk through the lep_pt_2 plots.
# 1. Imagine placing a cut at 7 GeV in the distributions of <span style="color:blue">signal</span> and <span style="color:red">background</span> (1st plot). This means keeping all events above 7 GeV in the <span style="color:blue">signal</span> and <span style="color:red">background</span> histograms. 
# 2. We then take the ratio of the number of <span style="color:blue">signal</span> events that pass this cut, to the number of <span style="color:red">background</span> events that pass this cut. This gives us a starting value for S/B (2nd plot). 
# 3. We then increase this cut value to 8 GeV, 9 GeV, 10 GeV, 11 GeV, 12 GeV. Cuts at these values are throwing away more <span style="color:red">background</span> than <span style="color:blue">signal</span>, so S/B increases. 
# 4. There comes a point around 13 GeV where we start throwing away too much <span style="color:blue">signal</span>, thus S/B starts to decrease. 
# 5. Our goal is to find the maximum in S/B, and place the cut there.
#
# The same logic applies to lep_pt_1.

# %%
plot_SoverB(data)

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# In the [ATLAS Higgs discovery paper](https://www.sciencedirect.com/science/article/pii/S037026931200857X), there are a number of numerical cuts applied, not just on lep_pt_1 and lep_pt_2.
#
# Imagine having to separately optimise about 7 variables! Not to mention that applying a cut on one variable could change the distribution of another, which would mean you'd have to re-optimise... Nightmare.
#
# This is where a Machine Learning algorithm such as a Neural Network can come to the rescue. A Neural Network can optimise all variables at the same time.
#
# A Neural Network not only optimises cuts, but can find correlations in many dimensions that will give better signal/background classification than individual cuts ever could.
#
# That's the end of the introduction to why one might want to use a Neural Network. If you'd like to try using one, just keep reading below!

# %% [markdown]
# <a id='NN'></a>

# %% [markdown]
# ## Neural Network
#
# Choose variables for use in the Neural Network

# %%
data_for_NN = {} # define empty dictionary to hold dataframes that will be used to train the NN
NN_inputs = ['lep_pt_1','lep_pt_2'] # list of features for Neural Network
for key in data: # loop over the different keys in the dictionary of dataframes
    data_for_NN[key] = data[key][NN_inputs].copy()
data_for_NN

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
#  Organise data ready for the NN

# %%
# for sklearn data is usually organised                                                                                                                                           
# into one 2D array of shape (n_samples x n_features)                                                                                                                             
# containing all the data and one array of categories                                                                                                                             
# of length n_samples  

all_MC = [] # define empty list that will contain all features for the MC
for key in data: # loop over the different keys in the dictionary of dataframes
    if key!='data': # only MC should pass this
        all_MC.append(data_for_NN[key]) # append the MC dataframe to the list containing all MC features
X = np.concatenate(all_MC) # concatenate the list of MC dataframes into a single 2D array of features, called X

all_y = [] # define empty list that will contain labels whether an event in signal or background
for key in data: # loop over the different keys in the dictionary of dataframes
    if key!=r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$' and key!='data': # only background MC should pass this
        all_y.append(np.zeros(data_for_NN[key].shape[0])) # background events are labelled with 0
all_y.append(np.ones(data_for_NN[r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$'].shape[0])) # signal events are labelled with 1
y = np.concatenate(all_y) # concatenate the list of lables into a single 1D array of labels, called y

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='train_test'></a>

# %% [markdown]
# ### The Training and Testing split
# One of the first things to do is split your data into a training and testing set. This will split your data into train-test sets: 67%-33%. It will also shuffle entries so you will not get the first 67% of <span style="color:orange">X</span> for training and the last 33% for testing. This is particularly important in cases where you load all <span style="color:blue">signal</span> events first and then the <span style="color:red">background</span> events.
#
# Here we split our data into two independent samples. The split is to create a training and testing set. The first will be used for training the classifier and the second to evaluate its performance.
#
# We don't want to test on events that we used to train on, this prevents overfitting to some subset of data so the network would be good for the test data but much worse at any *new* data it sees.

# %%
from sklearn.model_selection import train_test_split

# make train and test sets
X_train,X_test, y_train,y_test = train_test_split(X, y, 
                                                  test_size=0.33, 
                                                  random_state=492 )

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='preprocessing'></a>

# %% [markdown]
# ### Data Preprocessing for Neural Network
#
# The neural network in Python may have difficulty converging before the maximum number of iterations allowed if the data is not standardized. Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. Note that you must apply the same scaling to the test set for meaningful results. There are a lot of different methods for standardization of data, we will use the built-in StandardScaler this time.

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # initialise StandardScaler

# Fit only to the training data
scaler.fit(X_train)

# %%
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='training'></a>

# %% [markdown]
# ### Training the Neural Network model
# Now it is time to train our model. We'll use SciKit Learn (sklearn) in this tutorial. Other possible tools include keras and pytorch. 
#
# SciKit Learn makes training simpler, by using estimator objects. In this case we will import our estimator (the Multi-Layer Perceptron Classifier model) from the neural_network library of SciKit-Learn

# %%
from sklearn.neural_network import MLPClassifier

# %% [markdown]
# Next we create an instance of the model, there are a lot of parameters you can choose to define and customize here, we will only define the hidden_layer_sizes. For this parameter you pass in a tuple consisting of the number of neurons you want at each layer, where the nth entry in the tuple represents the number of neurons in the nth layer of the MLP model. There are many ways to choose these numbers, but for simplicity we will choose 1 layer with the same number of neurons as there are features in our data set along with 200 max iterations.

# %%
hidden_layer_sizes = [len(NN_inputs)] # 1 hidden layer
mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes), # define parameters for our multi-layer-perceptron
                    max_iter=200 ) # max number of iterations

# %% [markdown]
# Now that the model has been made we can fit the training data to our model, remember that this data has already been processed and scaled:

# %%
#print(X_train)
mlp.fit(X_train,y_train)

# %% [markdown]
# You can see the output that shows the default values of the other parameters in the model. We encourage you to play around with them and discover what effects they have on your neural network in Python.

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='ROC'></a>

# %% [markdown]
# ### Receiver Operarting Characteristic (ROC) curve for Neural Network
# Another useful plot to judge the performance of a classifier is to look at the ROC curve directly.

# %%
# we first plot the Neural Network output
signal_decisions = mlp.predict_proba(X[y>0.5])[:, 1] # get probabilities on signal
background_decisions = mlp.predict_proba(X[y<0.5])[:, 1] # get probabilities on background

plt.hist(background_decisions, color='red', label='background', 
         histtype='step', # lineplot that's unfilled
         density=True ) # normalize to form a probability density
plt.hist(signal_decisions, color='blue', label='signal', 
         histtype='step', # lineplot that's unfilled
         density=True, # normalize to form a probability density
         linestyle='--' ) # dashed line
plt.xlabel('Neural Network output') # add x-axis label
plt.ylabel('Arbitrary units') # add y-axis label
plt.legend() # add legend


# we then plot the ROC
plt.figure() # make new figure 

from sklearn.metrics import roc_curve, auc

decisions = mlp.predict_proba(X_test)[:, 1] # get probabilities on test set

# Compute ROC curve and area under the curve
fpr, tpr, _ = roc_curve(y_test, # actual
                        decisions ) # predicted

# Compute area under the curve for training set
roc_auc = auc(fpr, # false positive rate 
              tpr) # true positive rate

plt.plot(fpr, tpr, label='ROC (area = %0.2f)'%(roc_auc)) # plot test ROC curve
plt.plot([0, 1], # x from 0 to 1
         [0, 1], # y from 0 to 1
         '--', # dashed line
         color='grey', label='Luck')

plt.xlabel('False Positive Rate') # x-axis label
plt.ylabel('True Positive Rate') # y-axis label
plt.title('Receiver operating characteristic (ROC) curve') # title
plt.legend() # add legend
plt.grid() # add grid


# %% [markdown]
# Sliding the cut threshold in Neural Network output (upper plot) from right to left builds up the ROC curve (lower plot) from bottom to top.

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='overtraining'></a>

# %% [markdown]
# ### Neural Network Overtraining Check
# Comparing the Neural Network's output distribution for the training and testing set is a popular way in HEP to check for overtraining. The <span style="color:orange">compare_train_test()</span> method will plot the shape of the Neural Network's decision function for each class, as well as overlaying it with the decision function in the training set.
#
# There are techniques to prevent overtraining.

# %%
def compare_train_test(clf, X_train, y_train, X_test, y_test):
    decisions = [] # list to hold decisions of classifier
    for X,y in ((X_train, y_train), (X_test, y_test)): # train and test
        d1 = clf.predict_proba(X[y<0.5])[:, 1] # background
        d2 = clf.predict_proba(X[y>0.5])[:, 1] # signal
        decisions += [d1, d2] # add to list of classifier decision
    
    highest_decision = max(np.max(d) for d in decisions) # get maximum score
    bin_edges = [] # list to hold bin edges
    bin_edge = -0.1 # start counter for bin_edges
    while bin_edge < highest_decision: # up to highest score
        bin_edge += 0.1 # increment
        bin_edges.append(bin_edge)
    
    plt.hist(decisions[0], # background in train set
             bins=bin_edges, # lower and upper range of the bins
             density=True, # area under the histogram will sum to 1
             histtype='stepfilled', # lineplot that's filled
             color='red', label='Background (train)', # Background (train)
            alpha=0.5 ) # half transparency
    plt.hist(decisions[1], # background in train set
             bins=bin_edges, # lower and upper range of the bins
             density=True, # area under the histogram will sum to 1
             histtype='stepfilled', # lineplot that's filled
             color='blue', label='Signal (train)', # Signal (train)
            alpha=0.5 ) # half transparency

    hist_background, bin_edges = np.histogram(decisions[2], # background test
                                              bins=bin_edges, # number of bins in function definition
                                              density=True ) # area under the histogram will sum to 1
    
    scale = len(decisions[2]) / sum(hist_background) # between raw and normalised
    err_background = np.sqrt(hist_background * scale) / scale # error on test background

    width = 0.1 # histogram bin width
    center = (bin_edges[:-1] + bin_edges[1:]) / 2 # bin centres
    
    plt.errorbar(x=center, y=hist_background, yerr=err_background, fmt='o', # circles
                 c='red', label='Background (test)' ) # Background (test)
    
    hist_signal, bin_edges = np.histogram(decisions[3], # siganl test
                                          bins=bin_edges, # number of bins in function definition
                                          density=True ) # area under the histogram will sum to 1
    scale = len(decisions[3]) / sum(hist_signal) # between raw and normalised
    err_signal = np.sqrt(hist_signal * scale) / scale # error on test background
    
    plt.errorbar(x=center, y=hist_signal, yerr=err_signal, fmt='o', # circles
                 c='blue', label='Signal (test)' ) # Signal (test)
    
    plt.xlabel("Neural Network output") # write x-axis label
    plt.ylabel("Arbitrary units") # write y-axis label
    plt.legend() # add legend
    
compare_train_test(mlp, X_train, y_train, X_test, y_test) # call compare_train_test

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# <a id='NN_optimisation'></a>

# %% [markdown]
# ### Neural Network Optimisation
#
# Here we get the Neural Network's prediction probability for every event that was processed at the begininning (so could be data, <span style="color:blue">signal</span>, <span style="color:red">background</span>...). The higher the probability, the more the Neural Network thinks that event looks like <span style="color:blue">signal</span>.

# %%
y_predicted = mlp.predict_proba(X)[:, 1]
y_predicted

# %% [markdown]
# In this cell we save the Neural Network output to our dataframes.

# %%
cumulative_events = 0 # start counter for total number of events for which output is saved
for key in data: # loop over samples
    data[key]['NN_output'] = y_predicted[cumulative_events:cumulative_events+len(data[key])]
    cumulative_events += len(data[key]) # increment counter for total number of events
    print(data[key]['NN_output']) # print the dataframe column NN_output

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# Here we define parameters to plot the Neural Network output

# %%
NN_output = { # dictionary containing plotting parameters for the mllll histogram
    # change plotting parameters
    'bin_width':0.1, # width of each histogram bin
    'num_bins':7, # number of histogram bins
    'xrange_min':0, # minimum on x-axis
    'xlabel':'Neural Network output', # x-axis label
}

SoverB_hist_dict = {'NN_output':NN_output}

# %% [markdown]
# Here we call the function defined above to to illustrate the optimum cut value on Neural Network output, based on <span style="color:blue">signal</span> to <span style="color:red">background</span> ratio.

# %%
plot_SoverB(data)

# %% [markdown]
# [Back to contents](#contents)

# %% [markdown]
# Putting everything into a Neural Network means we only have 1 variable to optimise. The <span style="color:blue">signal</span> and <span style="color:red">background</span> distributions are separated much better when looking at Neural Network output, compared to individual variables. Cutting on Neural Network output also achieves much higher S/B values than on individual variables.
#
# Neural Networks can achieve better S/B ratios because they find correlations in many dimensions that will give better signal/background classification.
#
# Hopefully you've enjoyed this discussion on optimising for <span style="color:blue">signal</span> to <span style="color:red">background</span> ratio, and in particular how a Neural Network can be used to facilitate this.

# %% [markdown]
# <a id='going_further'></a>

# %% [markdown]
# ## Going further
#
# If you want to go further, there are a number of things you could try: 
# * Increase the fraction of events that are processed in '[Lumi, fraction, file path](#fraction)'.
# * Add in the other <span style="color:blue">H->ZZ signal</span> samples in '[Samples](#samples)'. You can copy them from [HZZAnalysis.ipynb](https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata/blob/master/13-TeV-examples/uproot_python/HZZAnalysis.ipynb). Try adding them one at a time first, then see how things look with all added.
# * Add in the Z and ttbar <span style="color:red">backgrounds</span> samples in '[Samples](#samples)'. You can copy them from [HZZAnalysis.ipynb](https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata/blob/master/13-TeV-examples/uproot_python/HZZAnalysis.ipynb). Try adding them separately first, then see how things look with both added.
# * Add some more variables into the Neural Network in '[Neural Network](#NN)'. Add them in one at a time, rather than all at once, because adding a variable could decrease Neural Network performance, due to anti-correlation. For some ideas of variables, you can look at the paper for the [discovery of the Higgs boson by ATLAS](https://www.sciencedirect.com/science/article/pii/S037026931200857X) (mostly Section 4 and 4.1).
# * Add in real data in '[Samples](#samples)' and see whether the Neural Network output distributions in data and simulation match. You can copy data from [HZZAnalysis.ipynb](https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata/blob/master/13-TeV-examples/uproot_python/HZZAnalysis.ipynb). 
# * Modify some Neural Network hyper-parameters in '[Training the Neural Network model](#training)'.
#
# With each change, keep an eye on the:
# * total area under the ROC curve, 
# * separation between <span style="color:blue">signal</span> and <span style="color:red">background</span> in the Neural Network output distribution
# * S/B scores that can be achieved
#
# Notice that we've trained and tested our Neural Network on simulated data. We would then *apply* it to real experimental data. Once you're happy with your Neural Network, you may want to put it back into a full analysis to run over all data.

# %% [markdown]
# [Back to contents](#contents)

# %%
