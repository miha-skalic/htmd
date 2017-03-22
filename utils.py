import numpy as np
import sys
import os
sys.path = ['/shared/miha/htmd/'] + sys.path

from htmd.molecule.molecule import Molecule
from htmd.molecule.voxeldescriptors import _getAtomtypePropertiesPDBQT, _getGridDescriptors, _getRadii, _getGridCenters
from htmd import getCurrentViewer

from htmd.molecule.util import uniformRandomRotation, writeVoxels
import random


random.seed(0)
np.random.seed(0)


def drawIsoSurface(values3d, mincoor, maxcoor, rescoor, viewer=None):
    from htmd.util import tempname
    if viewer is None:
        viewer = getCurrentViewer()

    outf = tempname(suffix='.cube')
    writeVoxels(values3d, outf, mincoor, maxcoor, rescoor)
    # drawCube(mincoor, maxcoor)
    viewer.send('mol new {} type cube first 0 last -1 step 1 waitfor 1 volsets {{0 }}'.format(outf))
    viewer.send('mol modstyle 0 top Isosurface 0.75 0 2 0 1 1')


def get_number(ret_vals=1, radius=1.):
    """
    returns you random location shift.
    :param ret_vals: number of random values you want generated
    :return: array of shape (ret_vals, 3)
    """
    phi = np.random.uniform(0, 2 * np.pi, ret_vals)

    costheta = np.random.uniform(-1, 1, ret_vals)
    u = np.random.uniform(0, 1, ret_vals)

    theta = np.arccos(costheta)

    r = radius * np.power(u, 1/3.)
    ret_vals = np.empty((ret_vals, 3))
    ret_vals[:, 0] = r * np.sin(theta) * np.cos(phi)
    ret_vals[:, 1] = r * np.sin(theta) * np.sin(phi)
    ret_vals[:, 2] = r * np.cos(theta)

    return ret_vals


class PocketPicture:
    def __init__(self, protein_file, ligand_file,
                 order = ('hydrophobic', 'aromatic', 'hbond_acceptor',
                          'hbond_donor', 'positive_ionizable',
                          'negative_ionizable', 'metal')):

        self.order = order
        self.protein = Molecule(protein_file)
        self.ligand = Molecule(ligand_file)

        self.center = np.mean(self.ligand.coords, axis=0)

        self.properties_protein = _getAtomtypePropertiesPDBQT(self.protein)
        self.properties_ligand = _getAtomtypePropertiesPDBQT(self.ligand)

        sigmas_protein = _getRadii(self.protein)
        sigmas_ligand = _getRadii(self.ligand)

        self.ligamultisigmas = np.zeros([self.ligand.numAtoms, len(order) + 1])
        self.protmultisigmas = np.zeros([self.protein.numAtoms, len(order) + 1])

        for i, p in enumerate(order):
            self.ligamultisigmas[self.properties_ligand[p], i] = sigmas_ligand[self.properties_ligand[p]]
            self.protmultisigmas[self.properties_protein[p], i] = sigmas_protein[self.properties_protein[p]]

        p = 'occupancies'
        self.ligamultisigmas[self.properties_ligand[p], len(order)] = sigmas_ligand[self.properties_ligand[p]]
        self.protmultisigmas[self.properties_protein[p], len(order)] = sigmas_protein[self.properties_protein[p]]

    def generate_box(self, size=16, center=None, shift_center=False):
        return 0
        # # depricated
        # if center is None:
        #     center = np.mean(self.ligand.coords, axis=0)
        # if shift_center:
        #     center += get_number(radius=shift_center).T
        #
        # N = [size] * 3
        # bbm = (center - float(size / 2)).squeeze()
        #
        # centers = _getGridCenters(bbm, N, 1.)
        # centers2D = centers.reshape(np.prod(N), 3)
        #
        # desc_ligand = _getGridDescriptors(self.ligand, centers2D, self.ligamultisigmas)
        # desc_ligand = desc_ligand.transpose((3,0,1,2))
        #
        # desc_protein = _getGridDescriptors(self.protein, centers2D, self.protmultisigmas)
        # desc_protein = desc_protein.transpose((3,0,1,2))
        #
        # return desc_ligand, desc_protein

    def generate_protein_box(self, size=16, center=None, shift_center=False, resolution=1.):
        return 0
        # # depricated
        # if center is None:
        #     center = np.mean(self.ligand.coords, axis=0)
        # if shift_center:
        #     center += get_number(radius=shift_center).T
        #
        # N = [size] * 3
        # bbm = (center - float(size / 2)).squeeze()
        #
        # centers = _getGridCenters(bbm, N, resolution)
        # centers2D = centers.reshape(np.prod(N), 3)
        #
        # desc_protein = _getGridDescriptors(self.protein, centers2D, self.protmultisigmas)
        # return desc_protein.transpose((3,0,1,2))


    def generate_box_pair(self, size=16, center=None, shift_center=False, resolution=1., sigmascale=1.):
        """
        Generates protein/ligand box pair
        :param size:
        :param center:
        :param shift_center:
        :param resolution:
        :return:
        """

        if center is None:
            center = np.mean(self.ligand.coords, axis=0)
        if shift_center:
            center += get_number(radius=shift_center).T

        N = [size] * 3
        bbm = (center - float(size * resolution / 2)).squeeze()

        centers = _getGridCenters(bbm, N, resolution)
        centers2D = centers.reshape(np.prod(N), 3)

        desc_protein = _getGridDescriptors(self.protein, centers2D, self.protmultisigmas * sigmascale)
        desc_protein = desc_protein.reshape((N[0], N[1], N[2], self.protmultisigmas.shape[1]))
        desc_lig = _getGridDescriptors(self.ligand, centers2D, self.ligamultisigmas * sigmascale)
        desc_lig = desc_lig.reshape((N[0], N[1], N[2], self.ligamultisigmas.shape[1]))

        return desc_protein.transpose((3,0,1,2)), desc_lig.transpose((3,0,1,2))


    def rotate_system(self):  # , center=None):
        """
        Randomly rotate the protein and the ligand around the center
        :return: None
        """
        # if center is None:
        #      center = self.center
        rotation = uniformRandomRotation()
        self.protein.rotateBy(rotation)
        self.ligand.rotateBy(rotation)


if __name__ == '__main__':
    all_proteins = os.listdir('/shared/miha/ws/autoencoders/db/')
    for i, prot in enumerate(all_proteins):
        ligand = ('../../autoencoders/db/' + prot + '/ligand.pdbqt')
        protein = ('../../autoencoders/db/' + prot + '/protein.pdbqt')

        picture = PocketPicture(protein, ligand)
