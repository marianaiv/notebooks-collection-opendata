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

# %%
import ROOT
import os

# %%
# Enable multi-threading
ROOT.ROOT.EnableImplicitMT()

# %%
path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/"

def get_data_samples():
    samples = ROOT.std.vector("string")()
    for tag in ["A", "B", "C", "D"]:
        samples.push_back(os.path.join(path, "GamGam/Data/data_{}.GamGam.root".format(tag)))
    return samples

def get_ggH125_samples():
    samples = ROOT.std.vector("string")()
    samples.push_back(os.path.join(path, "GamGam/MC/mc_343981.ggH125_gamgam.GamGam.root"))
    return samples


# %%
df = {}
df["data"] = ROOT.RDataFrame("mini", get_data_samples())
df["ggH"] = ROOT.RDataFrame("mini", get_ggH125_samples())
processes = list(df.keys())

# %%
# Apply scale factors and MC weight for simulated events and a weight of 1 for the data
for p in ["ggH"]:
    df[p] = df[p].Define("weight", "scaleFactor_PHOTON * scaleFactor_PhotonTRIGGER * scaleFactor_PILEUP * mcWeight");
df["data"] = df["data"].Define("weight", "1.0")

# %%
for p in processes:
    # Apply preselection cut on photon trigger
    df[p] = df[p].Filter("trigP")

    # Find two good muons with tight ID, pt > 25 GeV and not in the transition region between barrel and encap
    df[p] = df[p].Define("goodphotons", "photon_isTightID && (photon_pt > 25000) && (abs(photon_eta) < 2.37) && ((abs(photon_eta) < 1.37) || (abs(photon_eta) > 1.52))")\
                 .Filter("Sum(goodphotons) == 2")

    # Take only isolated photons
    df[p] = df[p].Filter("Sum(photon_ptcone30[goodphotons] / photon_pt[goodphotons] < 0.065) == 2")\
                 .Filter("Sum(photon_etcone20[goodphotons] / photon_pt[goodphotons] < 0.065) == 2")

# %%
ROOT.gInterpreter.Declare(
"""
#include <math.h> // for M_PI
using Vec_t = const ROOT::VecOps::RVec<float>;
float ComputeInvariantMass(Vec_t& pt, Vec_t& eta, Vec_t& phi, Vec_t& e) {
    float dphi = abs(phi[0] - phi[1]);
    dphi = dphi < M_PI ? dphi : 2 * M_PI - dphi;
    return sqrt(2 * pt[0] / 1000.0 * pt[1] / 1000.0 * (cosh(eta[0] - eta[1]) - cos(dphi)));
}
""");

# %%
hists = {}
for p in processes:
    # Make four vectors and compute invariant mass
    df[p] = df[p].Define("m_yy", "ComputeInvariantMass(photon_pt[goodphotons], photon_eta[goodphotons], photon_phi[goodphotons], photon_E[goodphotons])")

    # Make additional kinematic cuts and select mass window
    df[p] = df[p].Filter("photon_pt[goodphotons][0] / 1000.0 / m_yy > 0.35")\
                 .Filter("photon_pt[goodphotons][1] / 1000.0 / m_yy > 0.25")\
                 .Filter("(m_yy > 105) && (m_yy < 160)")

    # Book histogram of the invariant mass with this selection
    hists[p] = df[p].Histo1D(
            ROOT.ROOT.RDF.TH1DModel(p, "Diphoton invariant mass; m_{#gamma#gamma} [GeV];Events / bin", 30, 105, 160),
            "m_yy", "weight")

# %%
# Run the event loop
ggh = hists["ggH"].GetValue()
data = hists["data"].GetValue()

# %%
# Set styles
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetMarkerStyle(20)
ROOT.gStyle.SetMarkerSize(1.2)
size = 0.08
ROOT.gStyle.SetLabelSize(size, "x")
ROOT.gStyle.SetLabelSize(size, "y")
ROOT.gStyle.SetTitleSize(size, "x")
ROOT.gStyle.SetTitleSize(size, "y")

# Create canvas with pads for main plot and data/MC ratio
c = ROOT.TCanvas("c", "", 700, 750)

upper_pad = ROOT.TPad("upper_pad", "", 0, 0.29, 1, 1)
lower_pad = ROOT.TPad("lower_pad", "", 0, 0, 1, 0.29)
for p in [upper_pad, lower_pad]:
    p.SetLeftMargin(0.14)
    p.SetRightMargin(0.05)
upper_pad.SetBottomMargin(0)
lower_pad.SetTopMargin(0)

upper_pad.Draw()
lower_pad.Draw()

data.SetStats(0)
data.SetTitle("")

# Fit signal + background model to data
upper_pad.cd()
fit = ROOT.TF1("fit", "([0]+[1]*x+[2]*x^2+[3]*x^3)+[4]*exp(-0.5*((x-[5])/[6])^2)", 105, 160)
fit.FixParameter(5, 125.0)
fit.FixParameter(4, 119.1)
fit.FixParameter(6, 2.39)
data.Fit("fit", "", "E SAME", 105, 160)
fit.SetLineColor(2)
fit.SetLineStyle(1)
fit.SetLineWidth(2)
fit.Draw("SAME")

# Draw background
bkg = ROOT.TF1("bkg", "([0]+[1]*x+[2]*x^2+[3]*x^3)", 105, 160)
for i in range(4):
    bkg.SetParameter(i, fit.GetParameter(i))
bkg.SetLineColor(4)
bkg.SetLineStyle(2)
bkg.SetLineWidth(2)
bkg.Draw("SAME")

# Draw data
data.SetMarkerStyle(20)
data.SetMarkerSize(1.2)
data.SetLineWidth(2)
data.SetLineColor(ROOT.kBlack)
data.Draw("E SAME")
data.SetMinimum(1e-3)
data.SetMaximum(8e3)

# Scale simulated events with luminosity * cross-section / sum of weights
# and merge to single Higgs signal
lumi = 10064.0
ggh.Scale(lumi * 0.102 / ggh.Integral())
higgs = ggh
higgs.Draw("HIST SAME")

# Draw ratio
lower_pad.cd()

ratiofit = ROOT.TH1F("ratiofit", "ratiofit", 5500, 105, 160)
ratiofit.Eval(fit)
ratiofit.SetLineColor(2)
ratiofit.SetLineStyle(1)
ratiofit.SetLineWidth(2)
ratiofit.Add(bkg, -1)
ratiofit.Draw()
ratiofit.SetMinimum(-150)
ratiofit.SetMaximum(225)
ratiofit.GetYaxis().SetTitle("Data - bkg")
ratiofit.GetYaxis().CenterTitle()
ratiofit.GetYaxis().SetNdivisions(503, False)
ratiofit.SetTitle("")
ratiofit.GetXaxis().SetTitle("m_{#gamma#gamma} [GeV]")

ratio = data.Clone()
ratio.Add(bkg, -1)
ratio.Draw("E SAME")
for i in range(1, data.GetNbinsX()):
    ratio.SetBinError(i, data.GetBinError(i))

# Add legend
upper_pad.cd()
legend = ROOT.TLegend(0.60, 0.55, 0.89, 0.85)
legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.05)
legend.SetTextAlign(32)
legend.AddEntry(data, "Data" ,"lep")
legend.AddEntry(bkg, "Background", "l")
legend.AddEntry(fit, "Signal + Bkg.", "l")
legend.AddEntry(higgs, "Signal", "l")
legend.Draw("SAME")

# Add ATLAS label
text = ROOT.TLatex()
text.SetNDC()
text.SetTextFont(72)
text.SetTextSize(0.05)
text.DrawLatex(0.18, 0.84, "ATLAS")

text.SetTextFont(42)
text.DrawLatex(0.18 + 0.13, 0.84, "Open Data")

text.SetTextSize(0.04)
text.DrawLatex(0.18, 0.78, "#sqrt{s} = 13 TeV, 10 fb^{-1}");

# %%
# %jsroot on
c.Draw()

# %%
