#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:08:24 2023

@author: christian
"""
import numpy as np

def sim_pop_alleles(rng, theta_alleles = [0.8, 0.2], nclusts=1, nSNPs = 1000, shared = 0.5) :
    """
    Simulate the allele frequencies for a common ancestor and for all genetic clusters taking into account if the causal snps are shared or not.

    Parameters
    ----------
    theta_alleles : list of floats, optional
        between 0, 1, parameter controling the similarity between genetic clusters. near 0- similar clusters, near 1- distinct clusters. 
        First value is for nonshared regions, second is for shared regions
        The default is [0.8, 0.2].
    nclusts : int, optional
        number of genetic clusters. The default is 1.
    nSNPs : int, optional
        Number of snps to simulate for the genome. The default is 1000.
    shared : floa64
        specify the proportion of SNPs for which the allele frequency will be resampled using the second item of theta_alleles.
        The rest will be sampled with the second item in theta_alleles
    Returns
    -------
    tuple with (nSNPs x 1) numpy array where the first column corresponds to the common ancestral frequencies,
    (nSNPs x (nclusts +1)) numpy array columns correspond to each subclusters allele frequencies.
    indices of shared snps
    incides of causal snps

    """


    rng = np.random.default_rng(rng)
    theta = theta_alleles
    # simulate ancestral frequency of each SNP, dimensions = (SNPs,)
    ancest_freqs = rng.uniform(low=0.1, high=0.9, size=nSNPs)

    shared_indices = range(round(nSNPs * shared))
    
    # sample shared cluster allele frequencies
    cluster_freqs_shared = rng.beta(ancest_freqs[shared_indices] * (1- theta[1]) / theta[1],
                                    (1-ancest_freqs[shared_indices]) *
                                    (1-theta[1])/theta[1],
                                    size=(nclusts, len(shared_indices)))
    cluster_freqs_nonShared = rng.beta(ancest_freqs[~np.isin(np.arange(nSNPs), shared_indices)] * (1- theta[0]) / theta[0],
                                        (1-ancest_freqs[~np.isin(np.arange(nSNPs), shared_indices)]) *
                                        (1-theta[0])/theta[0],
                                        size=(nclusts, nSNPs - len(shared_indices)))
    # Concatenate shared and nonShared so that they have same columns and add to the number of rows
    cluster_freqs = np.concatenate((cluster_freqs_shared, cluster_freqs_nonShared), axis = 1)

    return ancest_freqs, cluster_freqs 


def assign_clusters(df, rng, nclusts=1, nsites = 1):
    """
    Simulate the genetic cluster that each subject will be observed from 

    Parameters
    ----------
    df : pandas datafrmae containing site assignments
        containing site assignments in one column.
    nclusts : int, optional
        number of genetic clusters. The default is 1.
    nsites : int, optional
        number of sites. The default is 1.
    site_comp : str, optional
        Specifying the balance of genetic clusters within each site. One of ["IID", "EQUAL", "RAND", "HET"]. The default is "IID".
    dominance : float, optional
        if site_comp is HET, this specifies the change in the sampling exponent associated with the most prevelant genetic cluster.
        The default is 2.
    eq_sites : bool, optional
        If true, forces all sites to have equal balance with all genetic clusters. The default is False.

    Returns
    -------
    (nsubjects x 1) numpy array specifying the cluster that each subject will be sampled from

    """
    nsubjects= df.shape[0]
    # Make sure each cluster is equally represented i neach site
    subj_ancestries = np.tile(np.arange(nclusts), int(
        nsubjects  / nclusts))

    return subj_ancestries

    # Checked samples came from ancester with the following
    # plt.scatter(sim1.ancest_freqs, sim1.pop_freqs.mean(axis = 0))
