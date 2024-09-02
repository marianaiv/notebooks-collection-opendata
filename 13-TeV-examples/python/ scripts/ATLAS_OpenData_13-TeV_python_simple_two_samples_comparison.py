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
# <CENTER><h1> Notebook example: simple two-samples comparison</h1></CENTER>
#
# ##### The following analysis is comparing the kinematics between events coming for the SM Higgs boson decaying to 2 W-bosons to those coming from the SM WW-diboson background  production.
#
# SM Higgs to WW Feynman diagram:
# <CENTER><img src="../../images/fig_01a.png" style="width:30%"></CENTER>
#
# SM WW-diboson Feynman diagram:
# <CENTER><img src="../../images/fig1b.png" style="width:30%"></CENTER>

# %%
import ROOT

# %%
# %jsroot on

# %%
## reading the input files via internet (URL to the file)

## WW
bkg = ROOT.TFile.Open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/2lep/MC/mc_363492.llvv.2lep.root")
t_bkg = bkg.Get("mini")
t_bkg.GetEntries()


# %%
## SM H->WW
sig = ROOT.TFile.Open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/2lep/MC/mc_345324.ggH125_WW2lep.2lep.root")
t_sig = sig.Get("mini")
t_sig.GetEntries()


# %%
c = ROOT.TCanvas("testCanvas","a first way to plot a variable",800,600)

# %%
h_bgs = ROOT.TH1F("h_bgs","Example plot: Missing transverse energey",20,0,200)
h2_bgs = ROOT.TH1F("h2_bgs","Example plot: Number of Jets",10,0,10)

h_sig = ROOT.TH1F("h_sig","Example plot: Missing transverse energey",20,0,200)
h2_sig = ROOT.TH1F("h2_sig","Example plot: Number of Jets",10,0,10)

# %%
n=0
for event in t_bkg:
    n += 1
    ## printing the evolution in number of events
    if(n%10000==0):
        print(n)
    h_bgs.Fill((t_bkg.met_et)/1000.)
    h2_bgs.Fill(t_bkg.jet_n)

m=0    
for event in t_sig:
    m += 1
    ## printing the evolution in number of events
    if(m%10000==0):
        print(m)
    h_sig.Fill((t_sig.met_et)/1000.)
    h2_sig.Fill(t_sig.jet_n)
        
print("Done!")

# %%
scale_bgs = h_bgs.Integral()
h_bgs.Scale(1/scale_bgs)

scale_sig = h_sig.Integral()
h_sig.Scale(1/scale_sig)


h_bgs.SetFillStyle(3001)
h_bgs.SetFillColor(4)
h_bgs.SetLineColor(4)

h_sig.SetFillStyle(3003)
h_sig.SetFillColor(2)
h_sig.SetLineColor(2)

legend=ROOT.TLegend(0.5,0.7,0.9,0.9)
legend.AddEntry(h_bgs,"Background (WW) ","l")
legend.AddEntry(h_sig,"Signal (H #rightarrow WW)","l")

h_sig.SetStats(0)
h_bgs.SetStats(0)

h_sig.Draw("hist")
h_bgs.Draw("histsame")
legend.Draw()
c.Draw()


# %%
scale2_bgs = h2_bgs.Integral()
h2_bgs.Scale(1/scale2_bgs)

scale2_sig = h2_sig.Integral()
h2_sig.Scale(1/scale2_sig)



h2_bgs.SetFillStyle(3001)
h2_bgs.SetFillColor(4)
h2_bgs.SetLineColor(4)

h2_sig.SetFillStyle(3003)
h2_sig.SetFillColor(2)
h2_sig.SetLineColor(2)

legend=ROOT.TLegend(0.5,0.7,0.9,0.9)
legend.AddEntry(h2_bgs,"Background (WW) ","l")
legend.AddEntry(h2_sig,"Signal (H #rightarrow WW)","l")


h2_sig.SetStats(0)
h2_bgs.SetStats(0)
h2_sig.Draw("hist")
h2_bgs.Draw("histsame")
legend.Draw()
c.Draw()


# %% [markdown]
# **Done**
