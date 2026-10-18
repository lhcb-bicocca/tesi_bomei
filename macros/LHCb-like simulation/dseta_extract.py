import ROOT
import numpy as np

# Open Root file and extract pdf
f = ROOT.TFile.Open(
    "template_Dsetamunu_Tot.root"
)
pdf = f.Get("keypdf")
obs = pdf.getVariables()
x = obs.first()

# Create grid and evaluate pdf
xs = np.linspace(x.getMin(), x.getMax(), 10000)
ys = np.empty_like(xs)
for i, v in enumerate(xs):
  x.setVal(v)
  ys[i] = pdf.getVal(ROOT.RooArgSet(x))
f.Close()

np.savez("template_dseta.npz", x=xs, pdf=ys)
print("File template_dseta.npz salvato con successo!")
