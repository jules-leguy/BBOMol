from abc import ABC, abstractmethod
from io import StringIO

from evomol.evaluation_dft import rdkit_mmff94_xyz, obabel_mmff94_xyz
from evomol.evaluation_entropy import extract_shingles
from sklearn.base import TransformerMixin, BaseEstimator
from dscribe.descriptors import SOAP, CoulombMatrix, MBTR
from joblib import Parallel, delayed, Memory
from rdkit.Chem.rdchem import GetPeriodicTable
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFromSmiles
import numpy as np
from ase.io import read as read_ase


class Descriptor(TransformerMixin, BaseEstimator, ABC):

    def __init__(self, cache_location=None, n_jobs=1, batch_size='auto', pre_dispatch='2 * n_jobs', MM_program="obabel",
                 MM_program_parameters=None):
        """
        :param cache_location: path of the joblib.Memory data
        :param n_jobs: number of jobs used for parallel computation of the descriptors
        :param MM_program: "obabel" to compute MM with obabel or "rdkit" to compute MM using RDKit
        :param MM_program_parameters: parameters to be given to the MM programm function
        """

        if MM_program == "obabel":
            self.geometry_function = obabel_mmff94_xyz
        elif MM_program == "rdkit":
            self.geometry_function = rdkit_mmff94_xyz
        elif callable(MM_program):
            self.geometry_function = MM_program

        self.cache_geom_fun = Memory(cache_location,
                                     verbose=0).cache(self.geometry_function)
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.pre_dispatch = pre_dispatch

        if MM_program_parameters is None and MM_program == "rdkit":
            geometry_function_parameters = {"max_iterations": 500}
        elif MM_program_parameters is None:
            geometry_function_parameters = {}
        else:
            geometry_function_parameters = MM_program_parameters

        self.geometry_function_parameters = geometry_function_parameters

    def fit(self, X, y=None):
        return self

    @abstractmethod
    def get_row_size(self):
        """
        Retuning the size of the vector describing one molecule
        :return: 
        """
        pass

    @abstractmethod
    def min_row_size(self):
        """
        Returning the minimum possible row size depending on the Descriptor instance
        :return:
        """
        pass

    @abstractmethod
    def transform_row(self, smiles):
        """
        Transforming the given SMILES.
        :param smiles: SMILES to be transformed
        :return: tuple (transformed descriptor, success of transformation)
        """
        pass

    def descriptors_shape(self, n_mol):

        return n_mol, self.get_row_size()

    def transform(self, X):
        # Performing a parallel computation of the descriptor
        results_parallel = Parallel(n_jobs=self.n_jobs, batch_size=self.batch_size, pre_dispatch=self.pre_dispatch)(
            delayed(self.transform_row)(X[i]) for i in range(len(X)))

        desc_comput = []
        successes_comput = []

        # Retrieving all parallel results
        for desc_row, success in results_parallel:
            # Saving descriptor
            desc_comput.append(desc_row)
            successes_comput.append(success)

        # Reshaping arrays
        desc_comput = np.array(desc_comput).reshape(self.descriptors_shape(len(X)))
        successes_comput = np.array(successes_comput).reshape((len(X),))

        return desc_comput, successes_comput

    def compute_geometry(self, smiles):
        """
        Return an ASE.Atoms object set with the geometry represented as a XYZ string obtained using the
        self.geometry_function
        :param smiles:
        :return: (ase.Atoms, success status)
        """

        # Making sure the SMILES is in RDKit canonical order
        smiles = MolToSmiles(MolFromSmiles(smiles))

        try:

            xyz_str, success = self.cache_geom_fun(smiles, **self.geometry_function_parameters)

            if success:
                f = StringIO(xyz_str)
                ase_atoms = read_ase(f, format='xyz')

            else:
                ase_atoms = None

        except Exception as e:
            print("Error while computing geometry : " + str(e) + "; smi : " + smiles)
            ase_atoms = None
            success = False

        return ase_atoms, success


def _CoulombMatrixDesc_compute_from_ASE(cm_builder, ase_mol, smiles, n_atoms_max, geom_function_name):
    try:

        cm_desc = cm_builder.create(ase_mol).reshape((-1,))
        success = True

    except Exception as e:
        print("CM failing for " + smiles)
        cm_desc = np.zeros((cm_builder.get_number_of_features()))
        success = False

    return cm_desc, success


class CoulombMatrixDesc(Descriptor):

    def __init__(self, cache_location=None, n_atoms_max=100, n_jobs=1, batch_size='auto', pre_dispatch='2 * n_jobs',
                 MM_program="obabel", MM_program_parameters=None):
        """

        Lauri Himanen et al., « DScribe: Library of Descriptors for Machine Learning in Materials Science »,
        Computer Physics Communications 247 (1 février 2020): 106949, https://doi.org/10.1016/j.cpc.2019.106949.

        :param n_atoms_max:
        :param n_jobs:
        """
        super().__init__(cache_location=cache_location, n_jobs=n_jobs, batch_size=batch_size, pre_dispatch=pre_dispatch,
                         MM_program=MM_program, MM_program_parameters=MM_program_parameters)

        # Parameters
        self.n_atoms_max = n_atoms_max

        # Setting up the CM descriptor
        self.cm = CoulombMatrix(
            n_atoms_max=self.n_atoms_max,
        )

        # Setting up the cache object
        self.cache_desc_fun = Memory(location=cache_location,
                                     verbose=0).cache(_CoulombMatrixDesc_compute_from_ASE,
                                                      ignore=["cm_builder", "ase_mol"])

    def get_row_size(self):
        return self.cm.get_number_of_features()

    def min_row_size(self):
        return self.get_row_size()

    def transform_row(self, smiles):

        # Computing MM and converting to ase.Atoms object
        ase_mol, ase_success = self.compute_geometry(smiles)

        if ase_success:

            # Computing CM descriptor
            cm_desc, cm_success = self.cache_desc_fun(self.cm, ase_mol, smiles, self.n_atoms_max,
                                                      self.geometry_function.__name__)

        else:

            cm_desc = np.zeros((self.get_row_size()))
            cm_success = False

        return cm_desc, ase_success and cm_success


def _SOAPDesc_compute_from_ASE(soap_builder, ase_mol, smiles, rcut, nmax, lmax, species, average, n_max_atoms,
                               geom_function_name):
    try:

        soap_desc = soap_builder.create(ase_mol).reshape((-1,))
        success = True

    except Exception as e:
        print("SOAP failing for " + smiles)
        soap_desc = np.zeros((soap_builder.get_number_of_features()))
        success = False

    return soap_desc, success


class SOAPDesc(Descriptor):

    def __init__(self, cache_location=None, rcut=6.0, nmax=8, lmax=6, species="default", average="inner", n_jobs=1,
                 batch_size='auto', pre_dispatch='2 * n_jobs', n_max_atoms=None, MM_program="obabel",
                 MM_program_parameters=None):
        """
        SOAP descriptor

        Albert P. Bartók, Risi Kondor, et Gábor Csányi, « On Representing Chemical Environments », Physical Review B 87,
        nᵒ 18 (28 mai 2013), https://doi.org/10.1103/PhysRevB.87.184115.

        Lauri Himanen et al., « DScribe: Library of Descriptors for Machine Learning in Materials Science »,
        Computer Physics Communications 247 (1 février 2020): 106949, https://doi.org/10.1016/j.cpc.2019.106949.

        :param rcut: Cutoff for local regions (see DScribe)
        :param nmax: Number of RBF (see DScribe)
        :param lmax: Maximum degree of spherical harmonics (see DScribe)
        :param species: List of atomic symbols that can be encoded
        :param average: Whether to perform an averaging of the environment to represent global structure ("inner",
        "outer", "off")
        :param n_jobs: Maximum number of threads for parallel computation
        """
        super().__init__(cache_location=cache_location, n_jobs=n_jobs, batch_size=batch_size, pre_dispatch=pre_dispatch,
                         MM_program=MM_program, MM_program_parameters=MM_program_parameters)
        self.rcut = rcut
        self.nmax = nmax
        self.lmax = lmax
        self.species = species
        self.average = average

        if average == "off" and n_max_atoms is None:
            self.n_max_atoms = 120
        else:
            self.n_max_atoms = n_max_atoms

        # Setting up the SOAP descriptor
        self.soap = SOAP(
            species=["H", "C", "O", "N", "F", "P", "S", "Cl", "Br"] if self.species == "default" else self.species,
            periodic=False,
            rcut=self.rcut,
            nmax=self.nmax,
            lmax=self.lmax,
            average=self.average
        )

        # Setting up the cache object
        self.cache_desc_fun = Memory(location=cache_location,
                                     verbose=0).cache(_SOAPDesc_compute_from_ASE,
                                                      ignore=["soap_builder", "ase_mol"])

    def get_row_size(self):
        if self.average == "off":
            return self.soap.get_number_of_features() * self.n_max_atoms
        else:
            return self.soap.get_number_of_features()

    def min_row_size(self):
        return self.get_row_size()

    def transform_row(self, smiles):

        # Computing MM and converting to ase.Atoms object
        ase_mol, ase_success = self.compute_geometry(smiles)

        if ase_success:

            complete_desc = np.zeros((self.get_row_size()))

            # Computing descriptor
            soap_desc, soap_success = self.cache_desc_fun(self.soap, ase_mol, smiles, rcut=self.rcut, nmax=self.nmax,
                                                          lmax=self.lmax, species=self.species, average=self.average,
                                                          n_max_atoms=self.n_max_atoms,
                                                          geom_function_name=self.geometry_function.__name__)

            complete_desc[:len(soap_desc)] = soap_desc
            soap_desc = complete_desc

        else:

            soap_desc = np.zeros((self.get_row_size()))
            soap_success = False

        return soap_desc, ase_success and soap_success


def _MBTRDesc_compute_from_ASE(mbtr_builder, ase_mol, smiles, species, atomic_numbers_n, inverse_distances_n,
                               cosine_angles_n, normalization, geom_function_name):
    try:

        cm_desc = mbtr_builder.create(ase_mol).reshape((-1,))
        success = True

    except Exception as e:
        print("MBTR failing for " + smiles)
        cm_desc = np.zeros((mbtr_builder.get_number_of_features()))
        success = False

    return cm_desc, success


class MBTRDesc(Descriptor):

    def __init__(self, cache_location=None, species="default", n_jobs=1, batch_size='auto', pre_dispatch='2 * n_jobs',
                 atomic_numbers_n=100, inverse_distances_n=100, cosine_angles_n=100, MM_program="obabel",
                 MM_program_parameters=None):
        """
        MBTR descriptor

        Haoyan Huo et Matthias Rupp, « Unified Representation of Molecules and Crystals for Machine Learning »,
        arXiv:1704.06439 [cond-mat, physics:physics], 2 janvier 2018, http://arxiv.org/abs/1704.06439.

        Lauri Himanen et al., « DScribe: Library of Descriptors for Machine Learning in Materials Science »,
        Computer Physics Communications 247 (1 février 2020): 106949, https://doi.org/10.1016/j.cpc.2019.106949.

        :param species: List of atomic symbols that can be encoded
        :param n_jobs: Max number of threads for parallel computation of descriptors
        :param atomic_numbers_n: Number of samples to encode atomic numbers in MBTR (see DScribe)
        :param inverse_distances_n: Number of samples to encode inverse distances in MBTR (see DScribe)
        :param cosine_angles_n: Number of samples to encode angles in MBTR (see DScribe)
        """
        super().__init__(cache_location=cache_location, n_jobs=n_jobs, batch_size=batch_size, pre_dispatch=pre_dispatch,
                         MM_program=MM_program, MM_program_parameters=MM_program_parameters)

        self.species = species
        self.atomic_numbers_n = atomic_numbers_n
        self.inverse_distances_n = inverse_distances_n
        self.cosine_angles_n = cosine_angles_n
        self.normalization = "l2_each"

        # Computing atomic numbers
        atomic_numbers = [GetPeriodicTable().GetAtomicNumber(symb) for symb in (
            ["H", "C", "O", "N", "F", "P", "S", "Cl", "Br"] if self.species == "default" else self.species)]

        self.mbtr = MBTR(
            species=["H", "C", "O", "N", "F", "P", "S", "Cl", "Br"] if self.species == "default" else self.species,
            k1={
                "geometry": {"function": "atomic_number"},
                "grid": {"min": 1, "max": max(atomic_numbers), "n": self.atomic_numbers_n, "sigma": 0.1},
            },
            k2={
                "geometry": {"function": "inverse_distance"},
                "grid": {"min": 0.25, "max": 1.25, "n": self.inverse_distances_n, "sigma": 0.1},
                "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
            },
            k3={
                "geometry": {"function": "cosine"},
                "grid": {"min": -1, "max": 1, "n": self.cosine_angles_n, "sigma": 0.1},
                "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
            },
            periodic=False,
            normalization=self.normalization,
        )

        # Setting up the cache object
        self.cache_desc_fun = Memory(location=cache_location,
                                     verbose=0).cache(_MBTRDesc_compute_from_ASE,
                                                      ignore=["mbtr_builder", "ase_mol"])

    def get_row_size(self):
        return self.mbtr.get_number_of_features()

    def min_row_size(self):
        return self.get_row_size()

    def transform_row(self, smiles):

        # Computing MM and converting to ase.Atoms object
        ase_mol, ase_success = self.compute_geometry(smiles)

        if ase_success:

            # Computing descriptor
            mbtr_desc, mbtr_success = self.cache_desc_fun(self.mbtr, ase_mol, smiles, species=self.species,
                                                          atomic_numbers_n=self.atomic_numbers_n,
                                                          inverse_distances_n=self.inverse_distances_n,
                                                          cosine_angles_n=self.cosine_angles_n,
                                                          normalization=self.normalization,
                                                          geom_function_name=self.geometry_function.__name__)
        else:

            mbtr_desc = np.zeros((self.get_row_size()))
            mbtr_success = False

        return mbtr_desc, ase_success and mbtr_success


class ShinglesVectDesc(Descriptor):

    def __init__(self, cache_location=None, lvl=1, vect_size=4000, count=False):
        """
        Shingles vector descriptor. Representing the molecule in the form of a boolean vector (or a count vector) of
        shingles of radius 1 to lvl.

        Daniel Probst et Jean-Louis Reymond, « A Probabilistic Molecular Fingerprint for Big Data Settings »,
        Journal of Cheminformatics 10, nᵒ 1 (décembre 2018), https://doi.org/10.1186/s13321-018-0321-8.

        Due to the fact that the mapping between shingles and identifiers depends on the order of submitted molecules,
        no parallel computation is proposed.
        For the same reason, this object records data in a temporary cache that won't be shared between
        executions.

        :param lvl: diameter of shingles
        :param vect_size: size of the output vector. Limits the maximum number of shingles that can be processed
        :param count: whether to count the number of shingles or to indicate their boolean presence
        """
        super().__init__(cache_location=cache_location)

        self.lvl = lvl
        self.vect_size = vect_size
        self.next_id = 0
        self.desc_id_dict = {}
        self.count = count

        # Setting up the cache object
        self.cache_desc_fun = Memory(location=cache_location,
                                     verbose=0).cache(extract_shingles)

    def get_row_size(self):
        return self.vect_size

    def min_row_size(self):
        return self.next_id

    def transform_row(self, smiles):
        pass

    def get_desc_id(self, desc):

        if desc not in self.desc_id_dict:
            self.desc_id_dict[desc] = self.next_id
            self.next_id += 1

        return self.desc_id_dict[desc]

    def transform(self, X):

        desc = np.zeros((len(X), self.vect_size))

        for i, smi in enumerate(X):

            found_shingles = self.cache_desc_fun(smi, self.lvl, as_list=self.count)
            curr_shg_vect = np.zeros((self.vect_size,))
            for shg in found_shingles:
                curr_shg_vect[self.get_desc_id(shg)] += 1

            # Setting current descriptor in matrix
            desc[i] = curr_shg_vect

        return desc, np.full((len(X),), True)
