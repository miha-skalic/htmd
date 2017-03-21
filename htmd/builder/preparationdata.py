# (c) 2015-2017 Acellera Ltd http://www.acellera.com
# All Rights Reserved
# Distributed under HTMD Software License Agreement
# No redistribution in whole or part
#
import logging

import numpy as np
import pandas as pd

from htmd.molecule.molecule import Molecule

logger = logging.getLogger(__name__)


def prettyPrintResidue(r):
    rs = "{:4s} {:4d}{:1s} {:1s}".format(r.resname, r.resid, r.insertion, r.chain)
    return rs


# Define a type for holding information on residues decisions
class PreparationData:
    """Results of the system preparation and optimization steps.

    Contains the results of an optimization operation, notably, for each residue name, id, and chain, the
    corresponding pKa and protonation state.

    The most important properties are accessible via the .data property, a pandas DataFrame. As such, they
    can be subset, converted, and saved in several ways (see examples below).

    Examples
    --------
    >>> tryp = Molecule("3PTB")
    >>> tryp_op, ri = proteinPrepare(tryp, returnDetails=True)
    >>> ri                                  # doctest: +NORMALIZE_WHITESPACE
    PreparationData object about 290 residues.
    Unparametrized residue names: CA, BEN
    Please find the full info in the .data property, e.g.:
      resname  resid insertion chain       pKa protonation flipped     buried
    0     ILE     16               A       NaN         ILE     NaN        NaN
    1     VAL     17               A       NaN         VAL     NaN        NaN
    2     GLY     18               A       NaN         GLY     NaN        NaN
    3     GLY     19               A       NaN         GLY     NaN        NaN
    4     TYR     20               A  9.590845         TYR     NaN  14.642857
     . . .
    >>> "%.2f" % ri.data.pKa[ri.data.resid==189]
    '4.95'
    >>> ri.data.to_csv("/tmp/report.csv")

    Attributes
    ----------
    data : :class:`DataFrame <pandas.core.frame.DataFrame>` object
        A pandas DataFrame with these columns: resname "Residue name, as per the original PDB", resid "Residue ID",
        insertion "Insertion code (resid suffix)", chain "Chain", pKa "pKa value computed by propKa",
        "protonation" Forcefield-independent protonation code, flipped "Whether the residue was flipped during the
        optimization", buried "Fraction of residue which is buried", membraneExposed "Whether residue is exposed to
        membrane", etc.
    missedLigands : str
        List of ligands residue names which were not optimized
    header : str
        Messages and warnings from PDB2PQR
    propkaContainer : propka.molecular_container.Molecular_container
        Detailed information returned by propKa 3.1.
    """

    # Important- all must be listed or "set_value" will silently ignore them
    _columns = ['resname', 'resid', 'insertion', 'chain',
                'pKa', 'protonation', 'flipped', 'patches',
                'buried', 'z', 'membraneExposed', 'forced_protonation', 'default_protonation',
                'pka_group_id',
                'pka_residue_type', 'pka_type', 'pka_charge',
                'pka_atom_name', 'pka_atom_sybyl_type']

    # Columns printed by the __str__ method
    _printColumns = ['resname', 'resid', 'insertion', 'chain',
                     'pKa', 'protonation', 'flipped', 'buried']

    def __init__(self):
        self.propkaContainer = None
        self.thickness = None
        self.missedLigands = []

        self.data = pd.DataFrame(columns=self._columns)
        self.data.resid = self.data.resid.astype(int)
        self.data.pKa = self.data.pKa.astype(float)
        self.data.buried = self.data.buried.astype(float)
        self.data.z = self.data.z.astype(float)
        self.data.pka_group_id = self.data.pka_group_id.astype(float)  # should be int, but NaN not allowed
        # self.data.flipped = self.data.flipped.astype(float)             #  should be bool, but NaN not allowed

    def __str__(self):
        r = "PreparationData object about {:d} residues.\n".format(len(self.data))
        if len(self.missedLigands) > 0:
            r += "Unparametrized residue names: " + ", ".join(self.missedLigands) + "\n"
        r += "Please find the full info in the .data property, e.g.: \n".format(len(self.data))
        r += str(self.data[self._printColumns].head())
        r += "\n . . ."
        return r

    def __repr__(self):
        return self.__str__()

    def _findRes(self, a_resname, a_resid, a_icode, a_chain):
        icode_pad = "{:1.1s}".format(a_icode)  # Pad and truncate to 1 char
        chain_pad = "{:1.1s}".format(a_chain)
        # Identity check should ignore residue name (self.data.resname == a_resname).
        # The N+ and C- resnames are indeed duplicated - distinguishing them with icode T.
        # Ditto for ligands, icode L.
        mask = (self.data.chain == chain_pad) & (self.data.resid == a_resid) & \
               (self.data.insertion == icode_pad)
        if sum(mask) == 0:
            self.data = self.data.append({
                                          'resname': a_resname,
                                          'resid': a_resid,
                                          'insertion': icode_pad,
                                          'chain': chain_pad, #     'patches': []
                                          }, ignore_index=True)
            pos = len(self.data) - 1
        elif sum(mask) == 1:
            pos = np.argwhere(mask)
            pos = int(pos)
        else:
            assert False, "More than one resid matched: either duplicated chain-residue-icode, or internal error (please report if the latter)."
        return pos

    # Generic setter in the pandas table. Maybe one should use actual indices instead.
    def _set(self, residue, key, val):
        pos = self._findRes(residue.name, residue.resSeq, residue.iCode, residue.chainID)
        self.data.set_value(pos, key, val)

    # residue is e.g. pdb2pqr.src.aa.ILE
    def _setProtonationState(self, residue, state):
        # logger.debug("_setProtonationState %s %s" % (residue, state))
        self._set(residue, 'protonation', state)

    def _setFlipped(self, residue, state):
        logger.debug("_setFlipped %s %s" % (residue, state))
        self._set(residue, 'flipped', state)

    def _appendPatches(self, residue, patch):
        # logger.debug("_appendPatches %s %s" % (residue, patch))
        pos = self._findRes(residue.name, residue.resSeq, residue.iCode, residue.chainID)
        self.data.patches[pos].append(patch)

    def _importPKAs(self, pkaCont):
        logger.debug("Called _importPKAs")
        self.propkaContainer = pkaCont
        for i, grp in enumerate(self.propkaContainer.conformations['AVR'].groups):
            # This is the key
            # Other places for the resname: grp.type  -  grp.atom.resName  grp.residue_type
            resname = grp.atom.resName
            resid = grp.atom.resNumb
            chain = grp.atom.chainID
            icode = grp.atom.icode
            if grp.residue_type in ['N+', 'C-']:  # Separate info about termini. See _findRes
                resname = grp.residue_type
                icode = "T"
                # forceAppend = True
            elif grp.atom.sybyl_assigned:  # A ligand - a hack to allow multiple groups overriding key. See _findRes
                icode = "L"
                # forceAppend = True
            pos = self._findRes(resname, resid, icode, chain)
            self.data.set_value(pos, 'pKa', grp.pka_value)
            self.data.set_value(pos, 'buried', grp.buried * 100.0)
            self.data.set_value(pos, 'z', grp.atom.z)
            self.data.set_value(pos, 'pka_group_id', i)
            self.data.set_value(pos, 'pka_residue_type', grp.residue_type)
            self.data.set_value(pos, 'pka_type', grp.type)
            self.data.set_value(pos, 'pka_charge', grp.charge)
            self.data.set_value(pos, 'pka_atom_name', grp.atom.name)
            self.data.set_value(pos, 'pka_atom_sybyl_type', grp.atom.sybyl_type)

    def _setMembraneExposureAndWarn(self, thickness, maxBuried=75.0):
        self.thickness = thickness
        ht = thickness / 2.0
        inSlab = (self.data.z > -ht) & (self.data.z < ht)
        notBuried = self.data.buried < maxBuried
        inSlabNotBuried = inSlab & notBuried
        self.data.membraneExposed = inSlabNotBuried
        if np.any(inSlabNotBuried):
            # dl = self._prettyPrintResidues(inSlabNotBuried)
            logger.warning(
                ("Predictions for {:d} residues may be incorrect because they are " +
                 "exposed to the membrane ({:.1f}<z<{:.2f} and buried<{:.1f}%).").format(
                    sum(inSlabNotBuried), -ht, ht, maxBuried))

    def _listNonStandardResidues(self):
        changed = self.data.resname != self.data.protonation
        cl = []
        for i, cr in self.data[changed].iterrows():
            if cr.resname in ['N+', 'C-'] or \
                            cr.protonation in ['WAT'] or \
                            type(cr.protonation) == float:
                continue
            cl.append("{:s} ({:s})".format(prettyPrintResidue(cr), cr.protonation))
        if cl:
            logger.info("The following residues are in a non-standard state: " + ", ".join(cl))

    def _warnIfpKCloseTopH(self, ph, tol=1.0):
        # Looks like NaN < 5 is False today
        dubious = abs(self.data.pKa - ph) < tol
        nd = sum(dubious)
        if nd > 1:
            logger.warning(
                "Dubious protonation state: the pKa of {:d} residues is within {:.1f} units of pH {:.1f}."
                    .format(nd, tol, ph))
            for i, dr in self.data[dubious].iterrows():
                drs = prettyPrintResidue(dr)
                logger.warning("Dubious protonation state:    {:s} (pKa={:5.2f})".format(drs, dr.pKa))


    def _get_pka_plot(self, pH=7.4, figSizeX=10, dpk=1.0, font_size = 12):
        """Internal function to build the protonation diagram"""

        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patheffects as PathEffects

        # Shading
        Xe = np.array([[1, 0], [1, 0]])

        # Shading colors http://matplotlib.org/examples/pylab_examples/custom_cmap.html
        neutral_grey = (.7, .7, .7)
        my_red = (.98, .41, .29)
        my_blue = (.42, .68, .84)
        grey_red = LinearSegmentedColormap.from_list("grey_red", [neutral_grey, my_red])
        grey_blue = LinearSegmentedColormap.from_list("grey_blue", [neutral_grey, my_blue])
        eps = .01  # Tiny overprint to avoid very thin white lines
        outline = [PathEffects.withStroke(linewidth=2,foreground="w")]

        # Color for pk values
        pkcolor = "black"
        pkfontsize = 8
        dtxt = 0  # Displacement

        # Or we could change the figure size, which scales axes
        # http://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
        plt.rc('font', family="Open Sans")
        plt.rc('font', size=font_size)  # controls default text sizes
        plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
        plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
        plt.rc('legend', fontsize=font_size)  # legend fontsize
        plt.rc('figure', titlesize=font_size)  # fontsize of the figure title

        # Constants
        acidicResidues = ['ASP', 'GLU', 'TYR']
        basicResidues = ['HIS', 'LYS', 'ARG']

        # titr =  (~ pd.isnull(d.pKa)) & d.pKa < 99
        d = self.data.copy()
        titr = d.pKa < 99  # Automatically excludes NaN
        N = sum(titr)

        # Dubious residues
        d['dubious'] = abs(d.pKa - pH) < dpk

        # Format residue labels
        labels = ["{:s} {:s}:{:d}{:s}- {:s}".format("(!)" if x.dubious else "",
                                                    x.chain,
                                                    x.resid,
                                                    x.insertion,
                                                    x.resname)
                  for i, x in d.loc[titr].iterrows()]
        pKas = d.pKa.loc[titr]
        restypes = ["neg" if x.resname in acidicResidues else "pos" for i, x in d.loc[titr].iterrows()]

        xmin, xmax = xlim = 0, 14
        ymin, ymax = ylim = -1, N

        width = .8  # Of each band

        # So, arbitrarily, 40 residues are square
        sizePerBand = figSizeX * (N / 40)
        figsize = (figSizeX, sizePerBand)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, xlim=xlim, ylim=ylim,
                             autoscale_on=False)

        ax.xaxis.tick_top()
        ax.set_xlabel("pKa")
        ax.xaxis.set_label_position('top')

        ax.yaxis.set_ticks(range(N))
        ax.yaxis.set_ticklabels(labels)
        ax.invert_yaxis()

        for i in range(N):
            left = xmin
            right = xmax
            top = i + width / 2
            bottom = i - width / 2
            pk = pKas.iloc[i]
            restype = restypes[i]

            if restype == "neg":
                ax.imshow(Xe * 0, interpolation="none",
                          cmap=grey_blue, vmin=0, vmax=1,
                          extent=(left, pk - dpk, bottom, top), alpha=1)
                ax.imshow(np.fliplr(Xe), interpolation="bicubic",
                          cmap=grey_blue, vmin=0, vmax=1,
                          extent=(pk - dpk - eps, pk + dpk, bottom, top), alpha=1)
                ax.imshow(1 + Xe * 0, interpolation="none",
                          cmap=grey_blue, vmin=0, vmax=1,
                          extent=(pk + dpk - eps, right, bottom, top), alpha=1)
                ax.text(pk - dtxt, i, " {:.2f} ".format(pk), color=pkcolor,
                        fontsize=pkfontsize, horizontalalignment="right", zorder=30,
                        path_effects=outline, weight="bold")
            else:
                ax.imshow(1 + Xe * 0, interpolation="none",
                          cmap=grey_red, vmin=0, vmax=1,
                          extent=(left, pk - dpk, bottom, top), alpha=1)
                ax.imshow(Xe, interpolation="bicubic",
                          cmap=grey_red, vmin=0, vmax=1,
                          extent=(pk - dpk - eps, pk + dpk, bottom, top), alpha=1)
                ax.imshow(Xe * 0, interpolation="none",
                          cmap=grey_red, vmin=0, vmax=1,
                          extent=(pk + dpk - eps, right, bottom, top), alpha=1)
                ax.text(pk + dtxt, i, " {:.2f} ".format(pk), color=pkcolor,
                        fontsize=pkfontsize, horizontalalignment="left", zorder=30,
                        path_effects=outline,  weight="bold")
            ax.add_line(Line2D([pk, pk], [bottom, top], linewidth=3, color='white', zorder=2))

            # ax.add_line(Line2D([pk,pk], [bottom,top], linewidth=3, color='blue'))

        ## Shaded vertical band at pH
        ax.axvline(x=pH - dpk, linewidth=2, color="black", alpha=.2, linestyle="dashed")
        ax.axvline(x=pH + dpk, linewidth=2, color="black", alpha=.2, linestyle="dashed")
        ax.axvline(x=pH, linewidth=3, color="black", alpha=.5)
        ax.text(pH - dpk, ymax, " 90% protonated", rotation=90,
                horizontalalignment="right", verticalalignment="bottom",
                style="italic", path_effects=outline)
        ax.text(pH + dpk, ymax, " 10% protonated", rotation=90,
                horizontalalignment="left", verticalalignment="bottom",
                style="italic", path_effects=outline)

        ax.set_aspect('auto')

        # show()   # for interactive use
        from io import StringIO
        imgdata = StringIO()
        fig.savefig(imgdata, format="svg", bbox_inches='tight', )
        ret_img = imgdata.getvalue()

        # fig.savefig("out.svg")
        # fig.savefig("out.png")

        # Png render may be a bit better -
        # http://stackoverflow.com/questions/14824522/dynamically-serving-a-matplotlib-image-to-the-web-using-python
        ##
        # from io import StringIO
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
        # buf.close()

        plt.close(fig)
        return ret_img


    def reprepare(self):
        """Repeat the system preparation, after the user edited the .data table.

        You should only modify the value of the .data.forced_protonation column on the basis of the values
        in the .data.resid, .data.insertion, .data.chain attributes. Any other change will be ignored.

        Returns
        -------
        mol_out : Molecule
            the molecule titrated and optimized. The molecule object contains an additional attribute,
        resData : ResidueData
            a table of residues with the corresponding protonation states, pKas, and other information

        Examples
        --------
        mol, prepData = proteinPrepare(Molecule("3PTB"), returnDetails=True)
        d = prepData.data
        d.loc[d.resid == 40, 'forced_protonation'] = 'HIP'
        mHIP40, pHIP40 = prepData.reprepare()

        """

        from pdb2pqr.src.hydrogens import hydrogenRoutines
        from pdb2pqr.src.forcefield import Forcefield
        from pdb2pqr.src.definitions import Definition
        from htmd.builder.preparation import _buildResAndMol

        d = self.data
        routines = self.pdb2pqr_routines
        p = routines.protein

        keep_pka_columns = ('forced_protonation', 'buried', 'z', 'membraneExposed',
                            'pKa', 'pka_group_id', 'pka_residue_type', 'pka_type',
                            'pka_charge', 'pka_atom_name', 'pka_atom_sybyl_type')

        copy_of_resname = d['resname']
        copy_of_protonation = d['protonation']
        copy_of_default_protonation = d['default_protonation']
        list_of_forced_protonations = ~ pd.isnull(d['forced_protonation'])

        neutraln = neutralc = False
        assign_only = clean = False
        debump = opt = True

        # Code lifted from resinter.py
        routines.removeHydrogens()
        for index, oldResidue in enumerate(p.getResidues()):
            chain = p.chainmap[oldResidue.chainID]
            chainIndex = chain.residues.index(oldResidue)

            d_idx = d.pdb2pqr_idx == index
            if sum(d_idx) != 1:
                logger.warning("Residue {:s} appears {:d} times in data table".format(str(oldResidue), sum(d_idx)))
                continue

            newResidueName = d.forced_protonation[d_idx].iloc[0]
            if pd.isnull(newResidueName):
                # newResidueName = d.protonation[d_idx].iloc[0]
                continue

            logger.debug("Replacing {} with {}".format(oldResidue, newResidueName))

            # Create the replacement residue
            residueAtoms = oldResidue.atoms
            newResidue = routines.protein.createResidue(residueAtoms, newResidueName)
            # Make sure our names are cleaned up for output.
            newResidue.renameResidue(newResidueName)
            # Drop it in
            p.residues[index] = newResidue
            chain.residues[chainIndex] = newResidue
            # Run the meaty bits of PDB2PQR
        routines.setTermini(neutraln, neutralc)
        routines.updateBonds()

        if not clean and not assign_only:
            routines.updateSSbridges()
            if debump:
                routines.debumpProtein()
            routines.addHydrogens()
            hydRoutines = hydrogenRoutines(routines)
            if debump:
                routines.debumpProtein()
            if opt:
                hydRoutines.setOptimizeableHydrogens()
                hydRoutines.initializeFullOptimization()
                hydRoutines.optimizeHydrogens()
            else:
                hydRoutines.initializeWaterOptimization()
                hydRoutines.optimizeHydrogens()
            # Special for GLH/ASH, since both conformations were added
            hydRoutines.cleanup()

        ff = "parse"
        ffout = "amber"
        usernames = userff = None

        routines.setStates()  # ?
        mydef = Definition()  # ?
        myForcefield = Forcefield(ff, mydef, userff, usernames)
        hitlist, misslist = routines.applyForcefield(myForcefield)
        # reslist, charge = routines.getCharge() # <--- ?

        # Copied from runPDB2PQR = ?
        if not ffout is None:
            if ffout != ff:
                myNameScheme = Forcefield(ffout, mydef, userff)
            else:
                myNameScheme = myForcefield
                routines.applyNameScheme(myNameScheme)

        newMol, newResData = _buildResAndMol(p)
        # Assume that the number and order of residues does not change

        # Carry over old pka and other useful info
        newResData.data['resname'] = copy_of_resname
        newResData.data['protonation'] = copy_of_protonation
        newResData.data['default_protonation'] = copy_of_default_protonation
        newResData.data.ix[list_of_forced_protonations, 'protonation'] = \
            d.ix[list_of_forced_protonations, 'forced_protonation']
        for cn in keep_pka_columns:
            newResData.data[cn] = d[cn]

        newResData.pdb2pqr_routines = routines
        newResData.pdb2pqr_protein = routines.protein
        newResData.missedLigands = self.missedLigands

        return newMol, newResData


if __name__ == "__main__":
    from htmd.builder.preparation import proteinPrepare

    import doctest

    doctest.testmod()
