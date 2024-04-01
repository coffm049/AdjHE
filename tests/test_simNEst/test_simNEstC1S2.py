from Estimate.estimators.GCTA_wrapper import GCTA, gcta
from Estimate.estimators.all_estimators import h2Estimation
from Simulate.simulation_helpers.Sim_generator import pheno_simulator
import numpy as np
import pytest

#%% Basic testing for single site and cluster

@pytest.fixture
def C1S1() :
    rng = np.random.default_rng(123)
    sim = pheno_simulator(rng = rng, nsubjects= 1000)
    sim.sim_sites(nsites =25, siteDistribution="UNEQUAL")
    sim.sim_pops(nclusts= 1)
    sim.sim_genos()
    sim.sim_pheno(h2Hom = 0.5, h2Het= [0], alpha = 0)
    est = h2Estimation()
    est.GRM = sim.GRM
    est.df = sim.df
    return est 

@pytest.mark.C1S1
def test_simNGCTA(C1S1) :
    est = C1S1
    result = est.estimate(mpheno = ["Y0"], npc = [0,1], Method = "GCTA", fixed_effects= ["Xc", "abcd_site"])
    assert result["h2"][0] == pytest.approx(0.5, abs = 0.05) 

@pytest.mark.C1S1
def test_simAdjHE(C1S1):
    est = C1S1
    result = est.estimate(mpheno = ["Y0"], npc = [0], Method = "AdjHE", fixed_effects= ["Xc", "abcd_site"])
    print(result)
    result = est.estimate(mpheno = ["Y0"], npc = [0], Method = "AdjHE", fixed_effects= ["Xc"], random_groups = "abcd_site")
    print(result)
    assert result["h2"][0] == pytest.approx(0.5, abs = 0.05) 

#%%
@pytest.mark.C1S1_thorough
def test_thoroughC1S1(): 
    h2s = np.zeros((10,2))
    for i in range(10) : 
        sim = pheno_simulator(rng = rng, nsubjects= 1000)
        sim.sim_sites(nsites =25)
        sim.sim_pops(nclusts= 1)
        sim.sim_genos()
        sim.sim_pheno(h2Hom = 0.5, h2Het= [0], alpha = 0)
        est = h2Estimation()
        est.GRM = sim.GRM
        est.df = sim.df
        h2s[i,0]= est.estimate(mpheno = ["Y0"], npc = [0], Method = "AdjHE", random_groups="abcd_site", fixed_effects= ["Xc"], PC_effect = "fixed")["h2"][0]
        h2s[i,1]= est.estimate(mpheno = ["Y0"], npc = [0], Method = "AdjHE", fixed_effects= ["Xc", "abcd_site"], PC_effect = "mixed")["h2"][0]

