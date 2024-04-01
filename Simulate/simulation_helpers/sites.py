#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate phenotypes
Created on Thu Oct 13 09:25:44 2022

@author: christian
"""
import numpy as np
import pandas as pd


def sim_sites(rng, nsubjects = 1000, nsites=1, siteDistribution="EQUAL", random_BS = True):
    """
    Simulate a site vector for the specified number of subjects and sites. Also, control wether the sites will be of equal sites or 
    sampled.

    Parameters
    ----------
    rng : numpy.random._generator.Generator 
        Random number generator to increase reproducibility
    nsubjects : int, optional
        Number of subjects to simulate. The default is 1000.
    nsites : int, optional
        Number of sites to simulate. The default is 1.
    eq_sites : bool, optional
        True- make all sites equal sized (note that the number of subjects must be divisible by the number of sites). The default is False.
    random_BS : bool, optional
        Whether the effecs (betas) for each site is randomly generated or not. The default is True.

    Returns
    -------
    a pandas dataframe (nsubjects x 2) numpy array with integers specifyin the simulated site ID and the associated effect of the site on
    the phenotype

    """
    # Site assignment
    if siteDistribution == "EQUAL":
        # For equal assignment (n needs to be divisible by nsites * nclusts)
        Groups = np.repeat(
            np.arange(start=1, stop=nsites + 1), int(nsubjects/nsites))
    elif siteDistribution == "UNEQUAL" :
        # only works for 2 sites  and nsubjects must be divisible by 4
        Groups = np.repeat([1,1,1,2], int(nsubjects/ 4))

    elif siteDistribution == "RANDOM" :
        # For unequal assignment sample randomly from uniform
        Groups = rng.choice(nsites, size= nsubjects, replace=True)
    else :
        print("Specified site distribution is not recognized") 

        # Simulate site effects (will rescale to desired contribution later)
    if random_BS :
        Bs = np.matrix(np.random.normal(0,  1, nsites)).T
    else :
        # Make site effects fixed
        Bs = np.arange(nsites, dtype = "float64")
        Bs -= Bs.mean()
        Bs=  np.reshape(Bs, (nsites, 1))
    if nsites == 1:
        Bs = np.array([0])

    # Make a dummy matrix
    Groups_dum = np.matrix(pd.get_dummies(Groups))
    
    return pd.DataFrame({"abcd_site" : Groups,
                         "Site_contrib" : np.array(Groups_dum * Bs).flatten()})

