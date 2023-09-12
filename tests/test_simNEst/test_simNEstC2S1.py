import numpy as np
import pytest
from Estimate.estimators.GCTA_wrapper import GCTA, gcta
from Estimate.estimators.all_estimators import Basu_estimation
from Simulate.simulation_helpers.Sim_generator import pheno_simulator

#%% Basic testing for single site and cluster

@pytest.fixture
def C2S1() :
    rng = np.random.default_rng(123)
    sim = pheno_simulator(rng = rng, nsubjects= 1000)
    sim.sim_sites(nsites =1)
    sim.sim_pops(nclusts= 2)
    sim.sim_genos()
    sim.sim_pheno(h2Hom = 0.01, h2Het= [0.25, 0.25])
    est = Basu_estimation()
    est.GRM = sim.GRM
    est.df = sim.df
    return est 

@pytest.mark.C2S1
def test_simNGCTA(C2S1) :
    est = C2S1
    result = est.estimate(mpheno = ["Y0"], npc = [0,1], Method = "GCTA", fixed_effects= ["Xc"])
    assert result["h2"][0] == pytest.approx(0.5, abs = 0.05) 

@pytest.mark.C2S1
def test_simAdjHE(C2S1):
    est = C2S1
    result = est.estimate(mpheno = ["Y0"], npc = [0,1,2], Method = "AdjHE", fixed_effects= ["Xc"], PC_effect = "fixed")
    print(result) 
    result = est.estimate(mpheno = ["Y0"], npc = [0,1,2], Method = "AdjHE", fixed_effects= ["Xc"], PC_effect = "mixed")
    print(result)
    result = est.estimate(mpheno = ["Y0"], npc = [0,1,2], Method = "AdjHE", fixed_effects= ["Xc"], PC_effect = "random")
    print(result)
    assert result["h2"][0] == pytest.approx(0.5, abs = 0.05) 

#%%
@pytest.mark.C2S1_thorough
def test_thoroughC1S1(C2S1): 
    est= C2S1
    df1= est.df.query("subj_ancestries==0")
    G1 = est.GRM[df1.index, :][:, df1.index]
    df1 = df1.reset_index()
    df2 = est.df.query("subj_ancestries==1")
    G2 = est.GRM[df2.index, :][:, df2.index]
    df2= df2.reset_index()
    
    est1 = Basu_estimation()
    est1.GRM = G1
    est1.df = df1
    
    est2 = Basu_estimation()
    est2.GRM = G2
    est2.df = df2

    result1 = est1.estimate(mpheno = ["Y0"], npc = [0,1], Method = "GCTA", fixed_effects= ["Xc"])
    result2 = est2.estimate(mpheno = ["Y0"], npc = [0,1], Method = "GCTA", fixed_effects= ["Xc"])
    result1A = est1.estimate(mpheno = ["Y0"], npc = [0,1,2,3], Method = "AdjHE", fixed_effects= ["Xc"])
    result2A = est2.estimate(mpheno = ["Y0"], npc = [0,1,2,3], Method = "AdjHE", fixed_effects= ["Xc"])



@pytest.mark.C5S1_thorough
def test_thoroughC1S1():
    h2s = []
    for i in range(10) : 
        rng = np.random.default_rng(i)
        sim = pheno_simulator(rng = rng, nsubjects= 1000)
        sim.sim_sites(nsites =1)
        sim.sim_pops(nclusts= 2)
        sim.sim_genos()
        sim.sim_pheno(h2Hom = 0.1, h2Het= [0.5, 0.5])
        esti = Basu_estimation()
        esti.GRM = sim.GRM
        esti.df = sim.df
        #resulti = esti.estimate(mpheno = ["Y0"], npc = [0,1], Method = "AdjHE", fixed_effects= ["Xc"], PC_effect = "fixed")
        #print(resulti)
        #resulti = esti.estimate(mpheno = ["Y0"], npc = [0,1], Method = "AdjHE", fixed_effects= ["Xc"], PC_effect = "mixed")
        #print(resulti)
        resulti = esti.estimate(mpheno = ["Y0"], npc = [0,1], Method = "AdjHE", fixed_effects= ["Xc"], PC_effect = "random")["h2"].values
        h2s.append(resulti[resulti-0.5 == min(resulti - 0.5)][0])



