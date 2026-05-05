# ruff: noqa: FIX004

import gc
from pathlib import Path
from copy import deepcopy

import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

from .helper_dicts import class_mass_dict, pdgid_class_dict


def load_pred_hgpflow(pred_path, threshold=0.5, num_events=None, return_proxy=False):
    tree = uproot.open(pred_path)["event_tree"]

    vars_to_load = ["pred_ind", "proxy_pt", "proxy_eta", "proxy_phi", "hgpflow_pt", "hgpflow_eta", "hgpflow_phi", "hgpflow_class"]

    mask = np.array([np.array(x > threshold) for x in tree["pred_ind"].array(library="np", entry_stop=num_events)], dtype=object)

    hgpflow_dict = {}
    for var in tqdm(vars_to_load, desc="Loading HGPFlow predictions...", total=len(vars_to_load)):
        new_var = var.replace("hgpflow_", "")
        hgpflow_dict[new_var] = np.array(
            [x[m] for x, m in zip(tree[var].array(library="np", entry_stop=num_events), mask, strict=False)], dtype=object
        )
    # hgpflow_dict["event_number"] = tree["event_number"].array(library="np", entry_stop=num_events).astype(int) - 55_000 # HACK

    # HACK: to fix event numbers we just arange them manually
    # hgpflow_dict["event_number"] = np.arange(len(hgpflow_dict["pt"]))  # HACK

    # compute mass and energy
    for k in ["mass", "e", "charge"]:
        hgpflow_dict[f"{k}"] = np.empty_like(hgpflow_dict["pt"], dtype=object)

    for i, cls in tqdm(enumerate(hgpflow_dict["class"]), desc="Computing HGPFlow mass...", total=len(hgpflow_dict["class"])):
        hgpflow_dict["mass"][i] = np.array([class_mass_dict[x] for x in cls])
        p = hgpflow_dict["pt"][i] * np.cosh(hgpflow_dict["eta"][i])
        hgpflow_dict["e"][i] = np.sqrt(p**2 + hgpflow_dict["mass"][i] ** 2)

        hgpflow_dict["charge"][i] = np.full_like(hgpflow_dict["pt"][i], 0)
        hgpflow_dict["charge"][i][cls <= 2] = 1
    if return_proxy:
        return {
            "pt": hgpflow_dict["proxy_pt"],
            "eta": hgpflow_dict["proxy_eta"],
            "phi": hgpflow_dict["proxy_phi"],
            "class": hgpflow_dict["class"],
            "mass": hgpflow_dict["mass"],
            "event_number": hgpflow_dict["event_number"],
        }
    return hgpflow_dict


def load_pred_mpflow(pred_path, threshold=0.5, num_events=None, return_proxy=False):
    tree = uproot.open(pred_path)["event_tree"]

    vars_to_load = ["pred_ind", "proxy_pt", "proxy_eta", "proxy_phi", "mpflow_pt", "mpflow_eta", "mpflow_phi", "mpflow_class"]

    mask = np.array([np.array(x > threshold) for x in tree["pred_ind"].array(library="np", entry_stop=num_events)])  # , dtype=object)

    mpflow_dict = {}
    for var in tqdm(vars_to_load, desc="Loading mpflow predictions...", total=len(vars_to_load)):
        new_var = var.replace("mpflow_", "")
        mpflow_dict[new_var] = np.array(
            [x[m] for x, m in zip(tree[var].array(library="np", entry_stop=num_events), mask, strict=False)], dtype=object
        )
    mpflow_dict["event_number"] = tree["event_number"].array(library="np", entry_stop=num_events).astype(int)  # - 643_00 # HACK

    # compute mass and energy
    for k in ["mass", "e", "charge"]:
        mpflow_dict[f"{k}"] = np.empty_like(mpflow_dict["pt"], dtype=object)

    for i, cls in tqdm(enumerate(mpflow_dict["class"]), desc="Computing mpflow mass...", total=len(mpflow_dict["class"])):
        mpflow_dict["mass"][i] = np.array([class_mass_dict[x] for x in cls])
        p = mpflow_dict["pt"][i] * np.cosh(mpflow_dict["eta"][i])
        mpflow_dict["e"][i] = np.sqrt(p**2 + mpflow_dict["mass"][i] ** 2)

        mpflow_dict["charge"][i] = np.full_like(mpflow_dict["pt"][i], 0)
        mpflow_dict["charge"][i][cls <= 2] = 1
    if return_proxy:
        return {
            "pt": mpflow_dict["proxy_pt"],
            "eta": mpflow_dict["proxy_eta"],
            "phi": mpflow_dict["proxy_phi"],
            "class": mpflow_dict["class"],
            "mass": mpflow_dict["mass"],
            "event_number": mpflow_dict["event_number"],
        }
    return mpflow_dict



def load_truth_atlas(filepath, topo=False, fiducial_cuts=False):
    scale_E_pT=1
    pt_min_gev=0.01
    abs_eta_max=3
    print("\033[96m" + f"E, pT will be scaled by {scale_E_pT}" + "\033[0m")
    print("\033[96m" + f"Will apply pT > {pt_min_gev} GeV and |eta| < {abs_eta_max} cuts to truth" + "\033[0m")
    with uproot.open(filepath) as file:
        tree = file['EventTree']

        particle_pt  = tree['truthPartPt'].array(library='ak') * scale_E_pT
        particle_e   = tree['truthPartE'].array(library='ak') * scale_E_pT
        particle_eta = tree['truthPartEta'].array(library='ak')
        particle_phi = tree['truthPartPhi'].array(library='ak')
        particle_pdgid = tree['truthPartPdgId'].array(library='ak')
        particle_gen_status = tree['truthPartStatus'].array(library='ak')

        # raw particle class (these are not necessarily the target)
        # WARNING: defaulting to charged hadron
        flat_particle_pdgid = ak.to_numpy(ak.flatten(particle_pdgid))
        vectorized_lookup = np.vectorize(pdgid_class_dict.get, otypes=[int])
        flat_particle_class_raw = vectorized_lookup(flat_particle_pdgid, 0)
        flat_particle_class_raw = ak.Array(flat_particle_class_raw)
        particle_class_raw = ak.unflatten(flat_particle_class_raw, ak.num(particle_pdgid))

        # tracks
        track_pt = tree['trackPt'].array(library='ak') * scale_E_pT
        track_eta = tree['trackEta'].array(library='ak')
        track_phi = tree['trackPhi'].array(library='ak')
        track_particle_idx = tree['trackTruthParticleIndex'].array(library='ak')
        
        # modified particle class
        # 0(ch had)->3(neut had), 1(e)->4(gamma), 2(mu)->3(ch had)
        print("\033[91m" + 'skipping particle_class_mod computation (slow). It will be same as particle_class_raw' + "\033[0m")
        particle_class_mod = deepcopy(particle_class_raw)

        # flat_particle_class_mod = deepcopy(flat_particle_class_raw)
        
        # particle_track_idx = []
        # for i in range(len(track_particle_idx)):
        #     part_track_idx_ev = np.zeros(len(particle_pt[i]), dtype=int) - 1
        #     valid_idx = track_particle_idx[i] >= 0
        #     part_track_idx_ev[track_particle_idx[i][valid_idx]] = np.arange(len(track_particle_idx[i]))[valid_idx]
        #     particle_track_idx.append(part_track_idx_ev)
        # particle_track_idx = ak.Array(particle_track_idx)

        # flat_particle_track_idx = ak.flatten(particle_track_idx)
        # flat_trackless_particle_mask = ak.where(flat_particle_track_idx >= 0, False, True)

        # # trackless ch hads and es become nu hads and photons (+3)
        # flat_trackless_chhad_and_e_mask = flat_trackless_particle_mask & (flat_particle_class_mod <= 1)
        # flat_particle_class_mod = ak.where(
        #     flat_trackless_chhad_and_e_mask, flat_particle_class_mod + 3, flat_particle_class_mod)

        # # trackless muons become neutral hadrons
        # flat_trackless_muon_mask = flat_trackless_particle_mask & (flat_particle_class_mod == 2)
        # flat_particle_class_mod = ak.where(flat_trackless_muon_mask, 3, flat_particle_class_mod)

        # # Unflatten the arrays back to their original structure
        # particle_class_mod = ak.unflatten(flat_particle_class_mod, ak.num(particle_pdgid))

        # pflow
        ppflow_pt     = tree['PflowPt'].array(library='ak') * scale_E_pT
        ppflow_eta    = tree['PflowEta'].array(library='ak')
        ppflow_phi    = tree['PflowPhi'].array(library='ak')
        ppflow_mass   = tree['PflowMass'].array(library='ak') * scale_E_pT
        ppflow_charge = tree['PflowCharge'].array(library='ak')
        ppflow_e      = np.sqrt((ppflow_pt * np.cosh(ppflow_eta))**2 + ppflow_mass**2)

        # pflow class
        # assume all ppflow are either charged hadrons or photons
        ppflow_class = ak.where(ppflow_charge == 0, 4, 0)

        # fiducial cuts on particles
        if fiducial_cuts:
            mask_kin = (particle_pt >= pt_min_gev) * (abs(particle_eta) < abs_eta_max)
            mask_pdgid = (np.abs(particle_pdgid) != 12) & (np.abs(particle_pdgid) != 14) & (np.abs(particle_pdgid) != 16)
            mask_gen_status = (particle_gen_status == 1)
            mask = mask_kin * mask_pdgid * mask_gen_status

            particle_pt         = particle_pt[mask]
            particle_e          = particle_e[mask]
            particle_eta        = particle_eta[mask]
            particle_phi        = particle_phi[mask]
            particle_pdgid      = particle_pdgid[mask]
            particle_class_raw  = particle_class_raw[mask]
            particle_class_mod  = particle_class_mod[mask]
            particle_gen_status = particle_gen_status[mask]

            # # on tracks
            # for i in range(len(particle_class_mod)):

            #     # old idx to new idx
            #     masked_old_particle_idx = np.arange(len(mask[i]))[mask[i]]

            #     # on tracks (also make sure that the associated particle is in the fiducial region, otherwise we have a track with no particle)
            #     mask = (track_pt[i] >= pt_min_gev) * abs(track_eta[i]) < abs_eta_max
            #     track_part_pt_mask = np.in1d(track_particle_idx[i], masked_old_particle_idx)
            #     mask = mask * track_part_pt_mask
                
            #     track_pt[i]  = track_pt[i][mask]
            #     track_eta[i] = track_eta[i][mask]
            #     track_phi[i] = track_phi[i][mask]
            #     track_particle_idx[i] = np.array([
            #         np.where(masked_old_particle_idx == x)[0][0] for x in track_particle_idx[i][mask]
            #     ])

            # on ppflow (|eta| < 3)
            mask = abs(ppflow_eta) < abs_eta_max
            ppflow_pt = ppflow_pt[mask]
            ppflow_eta = ppflow_eta[mask]
            ppflow_phi = ppflow_phi[mask]
            ppflow_mass = ppflow_mass[mask]
            ppflow_charge = ppflow_charge[mask]
            ppflow_class = ppflow_class[mask]
            ppflow_e = ppflow_e[mask]

        # can't use the eventNUmber in the tree (not unique over different samples (like JZx))
        event_number = np.arange(len(particle_pt))

        return_dict = {
            "particle_pt": particle_pt, "particle_eta": particle_eta, "particle_phi": particle_phi, "particle_e": particle_e, 
            "particle_class": particle_class_raw, "particle_class_mod": particle_class_mod, 
            "particle_pdgid": particle_pdgid, "particle_gen_status": particle_gen_status,       
            "track_pt": track_pt, "track_eta": track_eta, "track_phi": track_phi, "track_particle_idx": track_particle_idx,
            "ppflow_pt": ppflow_pt, "ppflow_eta": ppflow_eta, "ppflow_phi": ppflow_phi, 
            "ppflow_e": ppflow_e, "ppflow_mass": ppflow_mass, "ppflow_charge": ppflow_charge, "ppflow_class": ppflow_class,
            "event_number": event_number
        }

        if 'AntiKt4TruthJetsPt' in tree.keys():
            return_dict.update({
                'AntiKt4TruthJetsPt': tree['AntiKt4TruthJetsPt'].array(library='ak') * scale_E_pT,
                'AntiKt4TruthJetsEta': tree['AntiKt4TruthJetsEta'].array(library='ak'),
                'AntiKt4TruthJetsPhi': tree['AntiKt4TruthJetsPhi'].array(library='ak'),
                'AntiKt4TruthJetsE': tree['AntiKt4TruthJetsE'].array(library='ak') * scale_E_pT
            })

        if 'AntiKt4EMPFlowJetsPt' in tree.keys():
            return_dict.update({
                'AntiKt4EMPFlowJetsPt': tree['AntiKt4EMPFlowJetsPt'].array(library='ak') * scale_E_pT,
                'AntiKt4EMPFlowJetsEta': tree['AntiKt4EMPFlowJetsEta'].array(library='ak'),
                'AntiKt4EMPFlowJetsPhi': tree['AntiKt4EMPFlowJetsPhi'].array(library='ak'),
                'AntiKt4EMPFlowJetsE': tree['AntiKt4EMPFlowJetsE'].array(library='ak') * scale_E_pT,
                'AntiKt4EMPFlowJetsNConstituents': np.array([
                    np.array([len(x) for x in constids_ev]) for constids_ev in tree['AntiKt4EMPFlowJetsConstituentID'].array(library='np')
                ], dtype=object)
            })

        if 'AntiKt4EMTopoJetsPt' in tree.keys():
            return_dict.update({
                'AntiKt4EMTopoJetsPt': tree['AntiKt4EMTopoJetsPt'].array(library='ak') * scale_E_pT,
                'AntiKt4EMTopoJetsEta': tree['AntiKt4EMTopoJetsEta'].array(library='ak'),
                'AntiKt4EMTopoJetsPhi': tree['AntiKt4EMTopoJetsPhi'].array(library='ak'),
                'AntiKt4EMTopoJetsE': tree['AntiKt4EMTopoJetsE'].array(library='ak') * scale_E_pT,
            })

        if topo:
            topo_e   = tree['cluster_E'].array(library='ak') * scale_E_pT
            topo_eta = tree['cluster_Eta'].array(library='ak')
            topo_phi = tree['cluster_Phi'].array(library='ak')
            topo_pt  = tree['cluster_Pt'].array(library='ak')

            return_dict.update({
                "topo_e": topo_e, "topo_eta": topo_eta, "topo_phi": topo_phi, "topo_pt": topo_pt #, "topo_e_lc": topo_e_lc
            })

    # sorted_idx = np.argsort(event_number)

    # for key in tqdm(return_dict.keys(), desc="Sorting truth events by event number..."):
    #     return_dict[key] = return_dict[key][sorted_idx]

    return return_dict


def load_truth_clic(truth_path, event_number_offset=0):
    scale_e_pt = 1
    # pt_min_gev = 0.1
    print("\033[96m" + f"E, pT will be scaled by {scale_e_pt}" + "\033[0m")

    tree = uproot.open(truth_path)["events"]
    n_events = tree.num_entries

    truth_dict = {}
    vars_to_load = [
        "particle_pt",
        "particle_eta",
        "particle_phi",
        "particle_e",
        "particle_pdg",
        "particle_gen_status",
        "pandora_e",
        "pandora_eta",
        "pandora_phi",
        "pandora_pt",
        "pandora_pdg",
    ]

    for var in tqdm(vars_to_load, desc="Reading truth tree...", total=len(vars_to_load)):
        truth_dict[var] = tree[var].array(library="np")

    n_tracks = tree["ntrack_pt"].array(library="np")
    n_topos = tree["ntopo_e"].array(library="np")
    # filter out events with no tracks and no topoclusters
    track_topo_mask = (n_tracks > 0) | (n_topos > 0)
    print(f"Number of events with at least one track or topocluster: {np.sum(track_topo_mask)} out of {n_events}")
    print(np.argwhere(~track_topo_mask))
    for var in vars_to_load:
        truth_dict[var] = truth_dict[var][track_topo_mask]
    n_events = len(truth_dict["particle_pt"])
    print(f"Number of events after filtering: {n_events}")
    # MeV to GeV scaling not needed (already in GeV)

    # particle class and charge
    truth_dict["particle_class"] = np.empty_like(truth_dict["particle_pdg"])
    truth_dict["particle_charge"] = np.empty_like(truth_dict["particle_pdg"])
    for i, pdgid in tqdm(enumerate(truth_dict["particle_pdg"]), desc="Computing particle class...", total=n_events):
        truth_dict["particle_class"][i] = np.array([pdgid_class_dict.get(x, 4) for x in pdgid])
        truth_dict["particle_charge"][i] = np.array([1 if x <= 2 else 0 for x in truth_dict["particle_class"][i]])

    # pandora class and charge
    truth_dict["pandora_class"] = np.empty_like(truth_dict["pandora_pdg"])
    truth_dict["pandora_charge"] = np.empty_like(truth_dict["pandora_pdg"])
    for i, pdgid in tqdm(enumerate(truth_dict["pandora_pdg"]), desc="Computing pandora class...", total=n_events):
        truth_dict["pandora_class"][i] = np.array([pdgid_class_dict.get(x, 4) for x in pdgid])
        truth_dict["pandora_charge"][i] = np.array([1 if x <= 2 else 0 for x in truth_dict["pandora_class"][i]])

    # delete unnecessary variables
    vars_to_delete = ["particle_pdg", "pandora_pdg"]
    for var in vars_to_delete:
        del truth_dict[var]
    gc.collect()

    # fiducial cuts
    for i in tqdm(range(n_events), desc="Applying fiducial cuts...", total=n_events):
        # on particles (gen_status=1)
        mask = truth_dict["particle_gen_status"][i] == 1  # * (truth_dict['particle_pt'][i] >= pt_min_gev)
        for var in ["particle_pt", "particle_eta", "particle_phi", "particle_e", "particle_class", "particle_gen_status"]:
            truth_dict[var][i] = truth_dict[var][i][mask]

    if "event_number" in tree:
        truth_dict["event_number"] = tree["event_number"].array(library="np").astype(int)
    else:
        truth_dict["event_number"] = np.arange(len(truth_dict["particle_pt"])) + event_number_offset

    pandora_dict = {}
    for key in list(truth_dict.keys()):
        if key.startswith("pandora_"):
            new_key = key.replace("pandora_", "")
            pandora_dict[new_key] = truth_dict.pop(key)
    pandora_dict["event_number"] = truth_dict["event_number"].copy()

    return truth_dict, pandora_dict


def load_hgpflow_target(target_path, drop_res=True, num_events=None, event_number_offset=0):
    tree = uproot.open(target_path)["EventTree"]
    vars_to_load = ["particle_pt", "particle_eta", "particle_phi", "particle_e", "particle_pdgid"]

    hgpflow_target_dict_tmp = {}
    data = tree.arrays(vars_to_load, library="np", entry_stop=num_events)
    for var in tqdm(vars_to_load, desc="Loading HGPflow target (segmented)...", total=len(vars_to_load)):
        new_var = var.replace("particle_", "")
        hgpflow_target_dict_tmp[new_var] = data[var]

    # filter out the residual particles
    if drop_res:  # will be dafault, here it is just for debugging
        mask = np.array([np.array([pdgid_class_dict[xx] for xx in x]) <= 4 for x in hgpflow_target_dict_tmp["pdgid"]], dtype=object)
        for key, val in hgpflow_target_dict_tmp.items():
            if key == "event_number":
                continue
            hgpflow_target_dict_tmp[key] = np.array([x[m] for x, m in zip(val, mask, strict=False)], dtype=object)

    # unique_sorted_ev_num = np.sort(np.unique(hgpflow_target_dict_tmp['event_number']))
    # hgpflow_target_dict = {}
    # for key, val in tqdm(hgpflow_target_dict_tmp.items(), desc="Merging HGPflow target...", total=len(hgpflow_target_dict_tmp)):
    #     hgpflow_target_dict[key] = []
    #     for ev_num in unique_sorted_ev_num:
    #         mask = hgpflow_target_dict_tmp['event_number'] == ev_num
    #         hgpflow_target_dict[key].append(np.hstack(val[mask]))
    #     hgpflow_target_dict[key] = np.array(hgpflow_target_dict[key], dtype=object)
    # hgpflow_target_dict['event_number'] = unique_sorted_ev_num
    hgpflow_target_dict = hgpflow_target_dict_tmp

    # compute class
    hgpflow_target_dict["class"] = np.empty_like(hgpflow_target_dict["pdgid"])
    for i, pdgid in tqdm(enumerate(hgpflow_target_dict["pdgid"]), desc="Computing HGPFlow target class...", total=len(hgpflow_target_dict["pdgid"])):
        hgpflow_target_dict["class"][i] = np.array([pdgid_class_dict[x] for x in pdgid])

    # compute mass
    hgpflow_target_dict["mass"] = np.empty_like(hgpflow_target_dict["pt"])
    for i, pdgid in tqdm(enumerate(hgpflow_target_dict["pdgid"]), desc="Computing HGPFlow target mass...", total=len(hgpflow_target_dict["pdgid"])):
        hgpflow_target_dict["mass"][i] = np.array([class_mass_dict[pdgid_class_dict[x]] for x in pdgid])

    # if 'event_number' in tree.keys():
    #     truth_dict['event_number'] = tree['event_number'].array(library='np').astype(int)
    # else:
    hgpflow_target_dict["event_number"] = np.arange(len(hgpflow_target_dict["pt"])) + event_number_offset
    return hgpflow_target_dict
