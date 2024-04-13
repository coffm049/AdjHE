import os
import pandas as pd
import numpy as np
import timeit

#%%


def ReadGRMBin(prefix, sub_ids = None):
    """
    Read GCTA style binary GRM file sets into memory.

    Parameters
    ----------
    prefix : string
        filepath common to all files of the GRM that is to be read.
    sub_ids : str
        OPTIONAL: filepath to FIDs/IIDS of subset of the sample
    Returns
    -------
    ids : pandas dataframe
        Dataframe containing the FID and IID of subjects in the same order as the GRM
    GRM : numpy array
        GRM as an (nxn) array.

    """
    print("Reading GRM: ", prefix)
    
    # Specify information about binary GRM format
    dt = np.dtype('f4') # Relatedness is stored as a float of size 4 in the binary file

    # Read IDs
    ids = pd.read_table(prefix + ".grm.id", names = ["FID", "IID"], dtype = str)
    n = ids.shape[0]

    ## Read GRM from binary 
    grm = np.fromfile(prefix + ".grm.bin", dtype = dt)
    # seed empty grm
    GRM = np.zeros((n, n), dtype = dt)
    # make a lower triangle matrix
    l = np.tril_indices(n)
    GRM[l] = grm
    # Make the rest of the symmetric matrix
    GRM = GRM + GRM.T - np.diag(np.diag(GRM))
    
    if sub_ids != None :
        ids2 = pd.read_table(sub_ids, names= ["FID", "IID"], dtype = str)
        # keep the ids that overlap with the additionally specified ids
        ids = ids.reset_index().merge(ids2, on = ["FID", "IID"]).set_index("index")
        # Subset the GRM too
        GRM = GRM[ids.index, :][:, ids.index]        

    return ids, GRM


# To check if a file is missing a header
def check_header(filename):
        with open(filename) as f:
            first = f.read(1)
        return first not in '.-0123456789'


def data_loader(file) :
    # check if it's the pc or covar or pheno file  (pc file won't have a header)
    if check_header(file) :
        df = pd.read_table(file, sep = "\s+", header = 0)

    # if not it's the PC file
    else:
        df = pd.read_table(file, sep= "\s+", header=None)
        df.columns = ["FID", "IID"] + ["pc_" + str(s) for s in range(1, df.shape[1]-1)]
        df.FID = df.FID.astype("Int64")

    # make sure FID and IID are objects to join with the ids from the GRM
    df["FID"] = df.FID.astype(str)
    df["IID"] = df.IID.astype(str)
    return df

def load_tables(ids, list_of_files) :
    # load the rest of the data
    for file in list_of_files :
        if file != None:
            if file.endswith(".npy") :
                # NEED TO LOAD IDS, and match them to FID/ IID
                newIds = pd.read_csv("/panfs/jay/groups/31/rando149/coffm049/ABCD/Workflow/02_Phenotypes/FCsTopo/FC.files", header = None)
                newIds.columns = ["IID"]
                newdf = pd.DataFrame(np.load(file))
                newdf.columns = newdf.columns.astype(str)
                newdf["IID"] = newIds['IID'].str[4:]
                newdf["IID"] = newdf["IID"].str[0:4] + "_" + newdf["IID"].str[4:]
                # this works becauses the phenofile is the first in the list
                newdf = pd.merge(ids, newdf, on = ["IID"], how = "inner")
                newdf = newdf[['FID', 'IID'] + [col for col in newdf.columns if col not in ['FID', 'IID']]]
            else :
                newdf = data_loader(file)
            # merge always using the left keys such that it always aligns with the GRM
            ids = pd.merge(ids, newdf, on = ["FID", "IID"], how = "left")
    return ids




# %% Read GRM
def load_everything(prefix, pheno_file, cov_file=None, PC_file=None, k=0, ids = None):
    """
    Load all covariates, phenotypes, and the GRM

    Parameters
    ----------
    prefix : string
        path to grm files.
    pheno_file : string
        path to phenotype file.
    cov_file : string, optional
        Path to covariate file. The default is None.
    PC_file : string, optional
        path to PC's file. The default is None.

    Returns
    -------
    a tuple of the full dataframe, GRM without missing vlaues where the order of the FIDS and IIDs match between the df and GRM

    """
    
    print("Reading GRM: ", prefix)
    
    # Time reading the GRM and other data
    start_read = timeit.default_timer()
    
    # Read in grm
    ids, GRM = ReadGRMBin(prefix, ids)
    list_of_files = [pheno_file, PC_file, cov_file]
    df = load_tables(ids, list_of_files)

    end_read = timeit.default_timer()
    read_time = end_read - start_read
    
    print("It took " + str(read_time) + " (s) to read GRM, covariates, and phenotypes")
    print("Phenos + Covars:", df.columns.tolist())
   
    # Get the phenotype names
    if pheno_file.endswith(".npy") :
        phenotypes = pd.DataFrame(np.load(pheno_file)).columns.astype(str)

    else : 
        phenotypes = pd.read_table(pheno_file, sep = "\s+", header = 0, nrows= 0).columns.astype(str).tolist()
        phenotypes.remove("FID")
        phenotypes.remove("IID")
    return df, GRM, phenotypes 

