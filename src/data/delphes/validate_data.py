import logging
from pathlib import Path

import awkward as ak
import click
import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
import vector
from coffea.hist.plot import clopper_pearson_interval
from matching import match_fjet_to_higgs, match_jets_to_higgs

hep.style.use(hep.style.ROOT)
vector.register_awkward()
vector.register_numba()
ak.numba.register()

logging.basicConfig(level=logging.INFO)

PROJECT_DIR = Path(__file__).resolve().parents[2]


@click.command()
@click.argument("in-filename", nargs=1)
def main(in_filename):
    # SM HHH
    with uproot.open(in_filename) as in_file:
        events = in_file["Delphes"]
        keys = (
            [key for key in events.keys() if "Particle/Particle." in key and "fBits" not in key]
            + [key for key in events.keys() if "Jet/Jet." in key]
            + [key for key in events.keys() if "FatJet/FatJet." in key and "fBits" not in key]
        )
        arrays = events.arrays(keys)  # entry_stop=10000

        part_pid = arrays["Particle/Particle.PID"]  # PDG ID
        part_m1 = arrays["Particle/Particle.M1"]
        condition_hhh6b = np.logical_and(np.abs(part_pid) == 5, part_pid[part_m1] == 25)
        # note: see some +/-15 PDG ID particles (taus) so h->tautau is turned on
        # explicitly mask these out, just keeping hhh6b events
        mask_hhh6b = ak.count(part_pid[condition_hhh6b], axis=-1) == 6

        particles = ak.zip(
            {
                "pt": arrays["Particle/Particle.PT"],
                "eta": arrays["Particle/Particle.Eta"],
                "phi": arrays["Particle/Particle.Phi"],
                "mass": arrays["Particle/Particle.Mass"],
                "pid": part_pid,
                "m1": part_m1,
                "d1": arrays["Particle/Particle.D1"],
                "idx": ak.local_index(part_pid),
            },
            with_name="Momentum4D",
        )[mask_hhh6b]

        higgs_condition = np.logical_and(particles.pid == 25, np.abs(particles.pid[particles.d1]) == 5)
        higgses = ak.to_regular(particles[higgs_condition], axis=1)
        bquark_condition = np.logical_and(np.abs(particles.pid) == 5, particles.pid[particles.m1] == 25)
        bquarks = ak.to_regular(particles[bquark_condition], axis=1)

        jets = ak.zip(
            {
                "pt": arrays["Jet/Jet.PT"],
                "eta": arrays["Jet/Jet.Eta"],
                "phi": arrays["Jet/Jet.Phi"],
                "mass": arrays["Jet/Jet.Mass"],
                "idx": ak.local_index(arrays["Jet/Jet.PT"]),
            },
            with_name="Momentum4D",
        )[mask_hhh6b]

        fjets = ak.zip(
            {
                "pt": arrays["FatJet/FatJet.PT"],
                "eta": arrays["FatJet/FatJet.Eta"],
                "phi": arrays["FatJet/FatJet.Phi"],
                "mass": arrays["FatJet/FatJet.Mass"],
                "idx": ak.local_index(arrays["FatJet/FatJet.PT"]),
            },
            with_name="Momentum4D",
        )[mask_hhh6b]

        fj_match = match_fjet_to_higgs(higgses, bquarks, fjets, ak.ArrayBuilder()).snapshot()
        fj_higgses = higgses[fj_match > -1]

        j_match = match_jets_to_higgs(higgses, bquarks, jets, ak.ArrayBuilder()).snapshot()
        match = np.logical_and(j_match[:, :, 0] != j_match[:, :, 1], ak.all(j_match > -1, axis=-1))
        j_higgses = higgses[match]

        higgs_pt = hist.Hist.new.Reg(10, 0, 1000, name=r"H $p_T$ [GeV]").Double()
        higgs_pt.fill(ak.flatten(higgses.pt))

        fj_higgs_pt = hist.Hist.new.Reg(10, 0, 1000, name=r"H $p_T$ [GeV]").Double()
        fj_higgs_pt.fill(ak.flatten(fj_higgses.pt))

        j_higgs_pt = hist.Hist.new.Reg(10, 0, 1000, name=r"H $p_T$ [GeV]").Double()
        j_higgs_pt.fill(ak.flatten(j_higgses.pt))

        j_ratio = j_higgs_pt / higgs_pt
        j_ratio_uncert = np.abs(clopper_pearson_interval(num=j_higgs_pt.values(), denom=higgs_pt.values()) - j_ratio)
        fj_ratio = fj_higgs_pt / higgs_pt
        fj_ratio_uncert = np.abs(clopper_pearson_interval(num=fj_higgs_pt.values(), denom=higgs_pt.values()) - fj_ratio)

        fig, axs = plt.subplots(2, 1, height_ratios=[2, 1])
        hep.histplot(j_higgs_pt, label="H(bb) matched to 2 AK5 jets", ax=axs[0])
        hep.histplot(fj_higgs_pt, label="H(bb) matched to 1 AK8 jets", ax=axs[0])
        hep.histplot(higgs_pt, label="All H(bb)", ax=axs[0])
        axs[0].set_ylabel("Higgs bosons")
        axs[0].set_xlim(0, 1000)
        axs[0].set_ylim(1e-1, 1e6)
        axs[0].semilogy()
        axs[0].legend(loc="upper right")
        hep.histplot(
            j_ratio,
            yerr=j_ratio_uncert,
            label="H(bb) matched to 2 AK5 jets",
            ax=axs[1],
        )
        hep.histplot(
            fj_ratio,
            yerr=fj_ratio_uncert,
            label="H(bb) matched to 1 AK8 jets",
            ax=axs[1],
        )
        axs[1].set_ylabel("Efficiency")
        axs[1].set_xlim(0, 1000)
        plt.tight_layout()
        fig.savefig("higgs_pt.png")
        fig.savefig("higgs_pt.pdf")

        plt.figure()
        n_jets = hist.Hist.new.Reg(13, -0.5, 12.5, name="AK5 Jets").Double()
        n_jets.fill(ak.count(jets.pt, axis=-1))
        hep.histplot(n_jets)
        plt.ylabel("Events")
        plt.xlabel("AK5 Jets")
        plt.xlim(-0.5, 12.5)
        plt.ylim(1, 1e5)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig("n_jets.png")
        plt.savefig("n_jets.pdf")

        plt.figure()
        n_fjets = hist.Hist.new.Reg(7, -0.5, 6.5, name="AK8 Jets").Double()
        n_fjets.fill(ak.count(fjets.pt, axis=-1))
        hep.histplot(n_fjets)
        plt.ylabel("Events")
        plt.xlabel("AK8 Jets")
        plt.xlim(-0.5, 6.5)
        plt.ylim(1, 1e5)
        plt.semilogy()
        plt.tight_layout()
        plt.savefig("n_fjets.png")
        plt.savefig("n_fjets.pdf")


if __name__ == "__main__":
    main()
