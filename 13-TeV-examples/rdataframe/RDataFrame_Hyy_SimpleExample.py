# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
ROOT::RDataFrame df("mini","https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/1largeRjet1lep/MC/mc_361106.Zee.1largeRjet1lep.root");

# %%
auto nEntries = *df.Count();
std::cout<<"Number of Entries="<<nEntries<<std::endl;

# %%
//Get the list of column defined in the dataset<b>
auto inputCol = df.GetColumnNames();
for(int i=0;i<inputCol.size();++i){
    std::cout<<inputCol.at(i)<<std::endl;
}

# %%
# %jsroot on

# %%
gStyle->SetOptStat(0); gStyle->SetTextFont(42);
auto c = new TCanvas("c", "", 800, 700);

auto h = df.Define("jetpt","jet_pt").Filter("jet_n > 1").Histo1D("jetpt");
auto h1 = df.Define("jetpt_cut","jet_pt").Filter("jet_n > 4").Histo1D("jetpt_cut");
h->Draw();
h1->Draw("SAME");
h1->SetLineColor(2);

c->Draw();


# %%
