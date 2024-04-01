#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:08:00 2022

@author: christian
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from tqdm.auto import tqdm
from Simulate.simulation_helpers.Sim_generator import pheno_simulator
from Estimate.estimators.all_estimators import h2Estimation




rng = np.random.default_rng()

def sim_n_est(nsubjects = 1000, h2 = 0.5, nsites = 30,
              nclusts =1, nnpc = 1,
              nSNPs=20, phens = 2,
              random_BS=True, savePlink = False) :
    """_summary_

    Args:
        nsubjects (int, optional): _description_. Defaults to 1000.
        h2 (float, optional): _description_. Defaults to 0.5.
        nsites (int, optional): _description_. Defaults to 30.
        nclusts (int, optional): _description_. Defaults to 1.
        nnpc (int, optional): _description_. Defaults to 1.
        nSNPs (int, optional): _description_. Defaults to 20.
        phens (int, optional): _description_. Defaults to 2.
        random_BS (bool, optional): _description_. Defaults to True.
        savePlink (bool, optional): Save output to plink style binary. Defaults to False.

    Returns:
        _type_: _description_
    """ 
    sim = pheno_simulator(nsubjects= nsubjects, nSNPs = nSNPs)
    # Run through full simulation and estimation
    sim.full_sim(nsites= nsites, h2= h2, phens = phens, nclusts = nclusts, random_BS = random_BS)

    if savePlink :
        sim.save_plink()
    else :
        ests = h2Estimation()
        ests.df= sim.df
        ests.GRM = sim.GRM
        ests.mpheno = ["Y1"] 
    
        FE= ["Xc"]

        if nsites >1 :
            FE += ["abcd_site"]
    
        logging.info(sim.df.columns.tolist())

        full_results = pd.DataFrame({}, columns= ["Method", "Meta", "RV", "h2", "V(h2)", "time"])
        # iterate over all combinations of Methods and Meta 
        for (Met, M, rg) in itertools.product(["AdjHE", "GCTA"], [True, False], [None, "abcd_site"]):

            try :
                r = ests.estimate(Method = Met, npc = [nnpc], fixed_effects= FE, mpheno =ests.mpheno[0],
                            Naive = M, random_groups = rg)
                result = pd.DataFrame({"Method" : Met,
                                      "Meta" : M,
                                      "RV" : rg,
                                      "h2" : r["h2"][0],
                                      "Var(h2)" : r["var(h2)"][0],
                                       "time" : r["time"][0],
                                      "sg" : sigma[0],
                                      "ss" : sigma[1],
                                      "se" : sigma[2],
                                      "nsubjects" : nsubjects,
                                      "nsites" : nsites,
                                      "theta_alleles" : theta_alleles,
                                       "nclusts" : nclusts,
                                       "prop_causal" : prop_causal,
                                       "nnpc" : nnpc,
                                       "nSNPs" : nSNPs, 
                                       "site_comp" : site_comp,
                                       "dominance" : dominance,
                                       "site_dep" : site_dep,
                                       "site_het" : site_het}, index= [0])
                full_results = pd.concat([full_results, result], ignore_index= True)
            except TypeError :
                pass
            except np.linalg.LinAlgError :
                pass

        return full_results
        
def sim_experiment(nsubjectss = [1000], sigmas = [[0.5,0.25, 0.25]], site_comps = ["IID"], nsites = [25],
              theta_alleless = [0.9], nclustss = [5], dominances= [3], prop_causals= [0.05], site_deps= [False], nnpcs = [1],
              nSNPss= [200], phenss= [2], reps = 10, site_het = False, clusters_differ = False, cov_effect = True,
              ortho_cov = True, random_BS = True, plink = False) :
    """_summary_

    Args:
        nsubjectss (list, optional): _description_. Defaults to [1000].
        sigmas (list, optional): _description_. Defaults to [[0.5,0.25, 0.25]].
        site_comps (list, optional): _description_. Defaults to ["IID"].
        nsites (list, optional): _description_. Defaults to [25].
        theta_alleless (list, optional): _description_. Defaults to [0.9].
        nclustss (list, optional): _description_. Defaults to [5].
        dominances (list, optional): _description_. Defaults to [3].
        prop_causals (list, optional): _description_. Defaults to [0.05].
        site_deps (list, optional): _description_. Defaults to [False].
        nnpcs (list, optional): _description_. Defaults to [1].
        nSNPss (list, optional): _description_. Defaults to [200].
        phenss (list, optional): _description_. Defaults to [2].
        reps (int, optional): _description_. Defaults to 10.
        site_het (bool, optional): _description_. Defaults to False.
        clusters_differ (bool, optional): _description_. Defaults to False.
        cov_effect (bool, optional): _description_. Defaults to True.
        ortho_cov (bool, optional): _description_. Defaults to True.
        random_BS (bool, optional): _description_. Defaults to True.
        savePlink (bool, optional): save output to plink style binary. Defaults to False.

    Returns:
        _type_: _description_
    """    
    # Seed empty dataframe
    sim_results = pd.DataFrame()
    
    for (nsubjects, sigma, site_comp, nsite,
         theta_alleles, nclusts, dominance, prop_causal,
         site_dep, nnpc, nSNPs, phens,
         __
         ) in tqdm(itertools.product(nsubjectss, sigmas, site_comps, nsites,
                                theta_alleless, nclustss, dominances, prop_causals,
                                site_deps, nnpcs, nSNPss, phenss,
                                range(reps)), desc = "Simulation progress") :
                                
        # nnpc = 2 * nclusts
                                
        result = sim_n_est(nsubjects = nsubjects, sigma = sigma, site_comp = site_comp, nsites = nsite,
                           theta_alleles = theta_alleles, nclusts = nclusts, dominance= dominance, prop_causal= prop_causal, 
                           site_dep= site_dep, nnpc = nnpc,
                           nSNPs=nSNPs, phens = phens, site_het = site_het, clusters_differ = clusters_differ, cov_effect = cov_effect,
                           ortho_cov = ortho_cov, random_BS = random_BS, savePlink = savePlink)
        sim_results = pd.concat([sim_results, result], ignore_index=True)
        # Remove any temps
    return sim_results
