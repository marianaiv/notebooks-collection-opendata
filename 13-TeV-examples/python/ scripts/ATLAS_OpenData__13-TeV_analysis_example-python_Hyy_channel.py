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
# <CENTER>
#     <a href="http://opendata.atlas.cern/release/2020/documentation/notebooks/intro.html" class="icons"><img src="../../images/ATLASOD.gif" style="width:40%"></a>
# </CENTER>

# %% [markdown]
# <CENTER><h1>Searching for the Higgs boson in the H&#8594;&gamma;&gamma; channel</h1></CENTER>
#
#
# SM Higgs signal Feynman diagram:
# <CENTER><img src="../../images/Figures_FeynmanHprod.png" style="width:30%"></CENTER>

# %% [markdown]
# **Introduction**
# Let's take a current ATLAS Open Data sample and create a histogram:

# %%
import ROOT
from ROOT import TMath
import time

# %%
# %jsroot on

# %%
start = time.time()

# %%
f = ROOT.TFile.Open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/GamGam/Data/data_A.GamGam.root")

# %%
canvas = ROOT.TCanvas("Canvas","cz",800,600)

# %%
tree = f.Get("mini")

# %%
tree.GetEntries()

# %% [markdown]
# Now we're going to extract the photons variables

# %%
#Invariant mass histograms definition
hist = ROOT.TH1F("h_M_Hyy","Diphoton invariant-mass ; Invariant Mass m_{yy} [GeV] ; events",30,105,160)

# %% [markdown]
# Here we're filling the variables defined above with the content of those inside the input ntuples.

# %% [markdown]
# We're creating a histogram for this example. The plan is to fill them with events.

# %% [markdown]
# We are selecting below a simple look for them.

# %% [markdown]
# The Higgs boson analysis implemented here considers Higgs boson decays into a photon-photon pair. The event selection criteria are:

# %%
Photon_1 = ROOT.TLorentzVector()
Photon_2 = ROOT.TLorentzVector()
n = 0
for event in tree:
    n += 1
    ## printing the evolution in number of events
    if(n%10000==0):
        print(n)
    ## checking the trigger    
    if(tree.trigP):
        goodphoton_index = [0]*5
        goodphoton_n = 0
        photon_index = 0
        ##            
        j=0
        ## looping the photons per event
        for j in range(tree.photon_n):
            ##
            if(tree.photon_isTightID[j]):
                ##
                if(tree.photon_pt[j] > 25000 and (TMath.Abs(tree.photon_eta[j]) < 2.37)\
                   and (TMath.Abs(tree.photon_eta[j]) < 1.37 or TMath.Abs(tree.photon_eta[j]) > 1.52)):
                    ##
                    goodphoton_n += 1  #count
                    goodphoton_index[photon_index]=j
                    photon_index += 1
                ## end Pt and eta pre-selection
            ## end on request of quality of the photon
        ## end looping photons in the current event
            
        ## Using the two selected photons
        if(goodphoton_n==2):
            ##
            goodphoton1_index = goodphoton_index[0]
            goodphoton2_index = goodphoton_index[1]
            ## Getting couple of photons with good isolation 
            if((tree.photon_ptcone30[goodphoton1_index]/tree.photon_pt[goodphoton1_index] < 0.065)\
               and (tree.photon_etcone20[goodphoton1_index] / tree.photon_pt[goodphoton1_index] < 0.065)):
                ##
                if((tree.photon_ptcone30[goodphoton2_index]/tree.photon_pt[goodphoton2_index] < 0.065)\
                   and (tree.photon_etcone20[goodphoton2_index] / tree.photon_pt[goodphoton2_index] < 0.065)):
                    ##
                    Photon_1.SetPtEtaPhiE(tree.photon_pt[goodphoton1_index]/1000., tree.photon_eta[goodphoton1_index],\
                                          tree.photon_phi[goodphoton1_index],tree.photon_E[goodphoton1_index]/1000.)
                    Photon_2.SetPtEtaPhiE(tree.photon_pt[goodphoton2_index]/1000., tree.photon_eta[goodphoton2_index],\
                                          tree.photon_phi[goodphoton2_index],tree.photon_E[goodphoton2_index]/1000.)
                    ## Adding the two TLorentz vectors
                    Photon_12 = Photon_1 + Photon_2
                    ## Filling with the mass of the gamma-gamma system
                    hist.Fill(Photon_12.M())
                ## end isolation photon #2
            ## end isolation photon #1
        ## end 2-good photons
    ## end of trigger request
## End loop in the events

# %% [markdown]
# #### Final plot

# %%
hist.Draw("E")
canvas.Draw()

# %% [markdown]
# #### Log Scale

# %%
hist.Draw("E")
hist.SetMinimum(10)
canvas.SetLogy()
canvas.Draw()

# %%
end = time.time()
duration = end-start
print("Finished in {} min {} s".format(int(duration//60),int(duration%60))) # Python3

# %% [markdown]
# **Done!**
