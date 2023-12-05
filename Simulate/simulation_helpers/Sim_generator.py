#im_/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate phenotypes
Created on Thu Oct 13 09:25:44 2022

@author: christian
"""
import logging
import numpy as np
import pandas as pd
from pandas_plink import read_plink
from Estimate.data_input.load_data import ReadGRMBin
from Simulate.simulation_helpers.sites import sim_sites
from Simulate.simulation_helpers.clusters import sim_pop_alleles, assign_clusters
from Simulate.simulation_helpers.genos import sim_genos
from Simulate.simulation_helpers.pheno import sim_pheno
from Simulate.simulation_helpers.plink_pheno import sim_plink_pheno


class pheno_simulator():
    def __init__(self, rng= None, nsubjects=1000, nSNPs=1000, plink_prefix=None, grmPrefix= None, covarFile=  None, subjAncestries = None):
        self.rng = np.random.default_rng(rng)
        self.plink_prefix = plink_prefix
        
        if plink_prefix == None :
            self.nsubjects = nsubjects
            self.nSNPs = nSNPs

            # rng the simulated dataframe
            self.df = pd.DataFrame({"FID": np.arange(start=1, stop=nsubjects + 1), 
                                    "IID": np.arange(start=1, stop=nsubjects + 1)})


        else :
            if grmPrefix == None :
                # break the function
                logging.error("GRM prefix not specified")
                return
            else : 
                self.GRM = ReadGRMBin(grmPrefix)
            self.plink_prefix= plink_prefix
            
            (bim, fam, bed) = read_plink(plink_prefix)
            (self.nSNPs, self.nsubjects) = bed.shape
            self.df = pd.DataFrame({"FID": fam.fid,
                                    "IID": fam.iid})

            if covarFile is not None :
                covars = pd.read_csv(covarFile, sep='\s+')
                self.df = pd.merge(self.df, covars, on = ["FID", "IID"])
            # bed is (nSNPs x nsubjects) so transpose it
            self.genotypes = bed.T
            self.rng = np.random.default_rng()
            # Include subject ancestries
            if subjAncestries is None :
                self.df["subj_ancestries"] = 1  
            else : 
                temp = pd.read_csv(subjAncestries)
                self.df["subj_ancestries"]  = temp.iloc[:,2]


    def sim_sites(self, nsites=1, siteDistribution = "EQUAL", random_BS = True):
        self.nsites = nsites
        self.siteDistribution = siteDistribution
        self.df[["abcd_site", "Site_contrib"]] = sim_sites(rng = self.rng, nsubjects = self.nsubjects, 
                                                           nsites=nsites, siteDistribution = self.siteDistribution, 
                                                           random_BS = random_BS)

    def sim_pops(self, theta_alleles = 0.5, nclusts=1, shared = 0.5):
        self.theta_alleles = theta_alleles
        # Clusts are distinct ancestries
        self.nclusts = nclusts
        self.shared = shared
        

        # Simualte allele frequencies for common ancestor and for genetic clusters
        self.ancest_freqs, self.cluster_frequencies = sim_pop_alleles(
            rng = self.rng,
            theta_alleles = self.theta_alleles,
            nclusts=self.nclusts, nSNPs = self.nSNPs,
            shared = shared)

        # Sample ancesrties for each individual Dim = (nsubjects,)
        self.df["subj_ancestries"] = assign_clusters(
            df = self.df,
            rng = self.rng,
            nclusts=self.nclusts,
            nsites = self.nsites)

    def sim_genos(self):
        (self.genotypes, self.GRM, pcs) = sim_genos(rng = self.rng,
                                                    cluster_frequencies = self.cluster_frequencies, 
                                                    subject_ancestries = self.df["subj_ancestries"]) 
        # update the number of SNPs
        self.nSNPS_post_filter = self.genotypes.shape[1]
        self.df = pd.concat([self.df, pcs], axis=1)

    def sim_pheno(self, h2Hom=0.5, h2Het=[0], alpha = -1, phenobasename = "Y", nphenos = 2, prop_causal = [0.1, 0.1], shared = 0.5):
        self.h2Hom = h2Hom
        self.h2Het = h2Het
        self.alpha = alpha
        self.prop_causal = prop_causal
        self.shared = shared
        (self.df, self.causals, self.homo_eff, self.het_eff)  = sim_pheno(rng = self.rng,
                                                                          genotypes = self.genotypes,
                                                                          df = self.df,
                                                                          h2Hom = self.h2Hom,
                                                                          h2Het = self.h2Het,
                                                                          alpha = self.alpha,
                                                                          prop_causal = prop_causal,
                                                                          shared = shared,
                                                                          phenoname = phenobasename + "0")
        for i in range(1, nphenos): 
            (temp, __1, __2, __3)  = sim_pheno(rng = self.rng,
                                               genotypes = self.genotypes,
                                               df = self.df,
                                               h2Hom = self.h2Hom,
                                               h2Het = self.h2Het,
                                               alpha = self.alpha,
                                               prop_causal = [0.1, 0.1],
                                               shared = 0.5,
                                               phenoname = phenobasename + str(i))
            self.df[phenobasename + str(i)] = temp[phenobasename + str(i)]

    def full_sim(self, h2 = 0.5,
                 nsites=30, nclusts=5,
                 nsubjects=1000,
                 nnpc=0, phens = 2,
                 alpha=-1,
                 random_BS = True, npcs = 0,  maf_filter= 0.1):
        
        # Only simulate genes if plink file not specified
        if self.plink_prefix == None: 
  
            # Run through full simulation and estimation
            self.sim_sites(nsites= nsites, random_BS = random_BS)
            
            self.sim_pops(nclusts=nclusts)
            self.sim_covars()
            
            for i in range(phens) :
                self.sim_pheno(h2 = h2, phenoname = "Y" + str(i), alpha = alpha)
        
        else : 
            for i in range(phens) :
                phenoname = "Y" + str(i)
                self.df[phenoname] = sim_plink_pheno(rng = self.rng, bed = self.genotypes, h2= h2, npcs=npcs)

    def summary(self) :
        """
        Summarize all of the input parameters attributes of the simulation object

        Returns
        -------
        None.

        """
        logging.info("Simulated data summary")
        logging.info("Number of subjects: ", self.nsubjects)
        logging.info("Number of SNPs: ", self.nSNPs)
        logging.info("Number of clusters: ", self.nclusts)
        logging.info("Proportion of causal SNPs: ", self.prop_causal)
