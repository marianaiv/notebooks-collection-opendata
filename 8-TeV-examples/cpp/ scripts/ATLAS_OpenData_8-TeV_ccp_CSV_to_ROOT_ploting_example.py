# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# <CENTER>
#     <a href="http://opendata.atlas.cern/release/2020/documentation/notebooks/intro.html" class="icons"><img src="../../images/ATLASOD.gif" style="width:40%"></a>
# </CENTER>

# %% [markdown]
# <CENTER><h1>Simple CVS to ROOT C++ notebook example</h1></CENTER>

# %%
# %jsroot on

# %% [markdown]
# This is a very simple example of convertt a CSV file to ROOT format and use it to produce some plots

# %%
#include "Riostream.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include <stdio.h>
#include <stdlib.h>

# %% [markdown]
# We download the CSV ATLAS Open Data file. This is the input of a web-app called Histogram analysers, for the 8 TeV release

# %%
//This two lines can be commented out. You just need it once, and in case the CVS file was not provided already
system("rm outreach.csv");
system("wget http://opendata.atlas.cern/visualisations/CrossFilter/outreach.csv");

//If the file is downloaded with the line above, you *need* to remove the first line of the outreach.csv

# %% [markdown]
# Let's tell ROOT that the file is a CSV file, i.e. the values are separated by commmas

# %%
TString dir = gSystem->UnixPathName(__FILE__);
dir.ReplaceAll("outreach.C","");
dir.ReplaceAll("/./","/");

# %% [markdown]
# We define our new ROOT file where to put the information

# %%
TFile *f = new TFile("outreach.root","RECREATE");

# %% [markdown]
# The next cell is very important, it shows the name and type of varibles that the CSV file has so the new ROOT-tree can be filled with the correct information 

# %%
TTree *tree = new TTree("ntuple","data from csv file");
// The file inside has ---->   type,  Channel,  NJets,  MET,  Mll,  LepDeltaPhi,  METLLDeltaPhi,  SumLepPt,  BTags,  weight
tree->ReadFile("outreach.csv","type/I:Channel/I:NJets/I:MET/F:Mll/F:LepDeltaPhi/F:METLLDeltaPhi/F:SumLepPt/F:BTags/F:weight/F",',');
f->Write();

# %% [markdown]
# Please, notice that the line above generates a warning. This is because the first line of our CSV file contains the name of the colunms, but this is safely ignored by ROOT.

# %%
system("ls -lhrt outreach.*");

# %% [markdown]
# Notice also that in the output of the line above that the resulting ROOT files is ~37% of the size of the original SV file. This is another advantage of the ROOT format for this kind of datasets.

# %%
TFile *_file0 = TFile::Open("outreach.root");

# %% [markdown]
# Now, let's create some plots of the variables inside our new ROOT-tree

# %%
TCanvas *c3D = new TCanvas("c3D","c3D",10,10,400,400);
ntuple->Draw("MET:Mll:LepDeltaPhi","MET>0.");
c3D->Draw();

# %%
TCanvas *cz = new TCanvas("cz","cz",10,10,400,400);
ntuple->Draw("Mll:MET","weight>-999","colz");

# %%
cz->Draw();

# %%
TCanvas *c2D = new TCanvas("c2D","c2D",10,10,400,400);
ntuple->Draw("Mll:LepDeltaPhi","MET>0.","colz");
c2D->Draw();

# %% [markdown] jupyter={"outputs_hidden": true}
# Below you can try to create a more complex analysis, following what is done in the next notebooks examples in this repository.
