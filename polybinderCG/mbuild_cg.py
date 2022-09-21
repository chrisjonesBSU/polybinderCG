from cmeutils.gsd_utils import get_molecule_cluster
from polybinderCG.writers import write_snapshot
import gsd
import gsd.hoomd
import freud
import json
import numpy as np
from polybinderCG.compounds import COMPOUND_DIR

from math import atan2


class System:
    """
    """
    def __init__(
            self,
            molecule=None,
            atoms_per_monomer=None,
            mb_compound=None,
    ):
        self.mb_compound = mb_compound
        self.contains_H = self._check_for_Hs()
        self.molecule = molecule 
        self.all_particles = [p for p in self.mb_compound.particles()]
        if self.compound != None:
            try:
                f = open(f"{COMPOUND_DIR}/{self.compound}.json")
            except FileNotFoundError:
                raise ValueError(
                    f"No file was found in {COMPOUND_DIR} for {self.compound}"
                )
            self.comp_dict = json.load(f) 
            f.close()
            if self.contains_H:
                self.atoms_per_monomer = self.comp_dict[
                        "atoms_per_monomer_AA"
                        ]
            elif not self.contains_H:
                self.atoms_per_monomer = self.comp_dict[
                        "atoms_per_monomer_UA"
                ]
        elif self.compound == None:
            self.atoms_per_monomer = atoms_per_monomer

        self.clusters = get_molecule_cluster(snap=self.snap)
        self.clusters = [0 for i in range(self.mb_compound.n_particles)]
        self.molecule_ids = set(self.clusters)
        self.n_molecules = 1 
        self.n_atoms = len(self.clusters)
        self.molecules = [Molecule(self, i) for i in self.molecule_ids] 

    def monomers(self):
        """Generate all of the monomers from each molecule in System.molecules.

        Yields:
        -------
        Monomer
     
        """
        for molecule in self.molecules:
            for monomer in molecule.monomers:
                yield monomer

    def segments(self):
        """Generate all of the segments from each molecule in System.

        Yields:
        -------
        Segment

        """
        for molecule in self.molecules:
            for segment in molecule.segments:
                yield segment

    def components(self):
        """Generate all of the components from each molecule in System.

        Yields:
        -------
        Component

        """
        for monomer in self.monomers():
            for component in monomer.components:
                yield component

    def _check_for_Hs(self):
        """Returns True if the gsd snapshot contains hydrogen type atoms"""
        hydrogen_types = ["ha", "h", "ho", "h4", "opls_146", "opls_204"]
        return "H" in [p.name for p in self.mb_compound.particles()]


class Structure:
    """Base class for the Molecule(), Segment(), and Monomer() classes.

    Parameters:
    -----------
    system : 'System()', required
        The system object initially created from the input .gsd file.
    atom_indices : np.ndarray(n, 3), optional, default=None
        The atom indices in the system that belong to this specific structure.
    molecule_id : int, optional, default=None
        The ID number of the specific molecule from system.molecule_ids.

    Attributes:
    -----------
    system : 'cmeutils.polymers.System'
        The system that this structure belong to. Contains needed information
        about the box, and gsd snapshot which are used elsewhere.
    atom_indices : np.ndarray(n_atoms, 3)
        The atom indices in the system that belong to this specific structure
    n_atoms : int
        The number of atoms that belong to this specific structure
    atom_positions : np.narray(n_atoms, 3)
        The x, y, z coordinates of the atoms belonging to this structure.
        The positions are wrapped inside the system's box.
    center_of_mass : np.1darray(1, 3)
        The x, y, z coordinates of the structure's center of mass.

    """
    def __init__(
            self,
            system,
            atom_indices=None,
            name=None,
            parent=None,
            molecule_id=None
    ):
        self.system = system
        self.name = name
        self.parent = parent
        if molecule_id != None:
            self.atom_indices = np.where(self.system.clusters == molecule_id)[0]
            self.molecule_id = molecule_id
        else:
            self.atom_indices = atom_indices
        self.n_atoms = len(self.atom_indices)

    def generate_monomers(self):
        if isinstance(self, Monomer):
            return self
        if self.system.contains_H == False:
            structure_length = int(self.n_atoms / self.system.atoms_per_monomer)
            monomer_indices = np.array_split(self.atom_indices, structure_length)
            assert len(monomer_indices) == structure_length
            return [Monomer(self, i) for i in monomer_indices]
        elif self.system.contains_H == True:
            _head_indices = list(range(0, self.system.atoms_per_monomer - 2))
            _tail_indices = [-i for i in range(3, self.system.atoms_per_monomer+1)]
            _head_indices.append(-2)
            _tail_indices.append(-1)
            head_indices = self.atom_indices[_head_indices]
            tail_indices = self.atom_indices[_tail_indices]
            tail_indices.sort()
            structure_length = int((self.n_atoms-(len(_head_indices)*2)) 
                    / (self.system.atoms_per_monomer - 2)
            )
            start = self.system.atoms_per_monomer-2 
            stop = -self.system.atoms_per_monomer
            monomer_indices = np.array_split(
                    self.atom_indices[start:stop],
                    structure_length
            )
            assert len(monomer_indices) == structure_length

            monomers = [Monomer(self, head_indices)]
            monomers.extend([Monomer(self, i) for i in monomer_indices])
            monomers.append(Monomer(self, tail_indices))
            return monomers 

    @property
    def atom_positions(self):
        """The coordinates of every particle in the structure

        Returns:
        --------
        numpy.ndarray, shape=(n, 3), dtype=float

        """
        return np.array([p.xyz for p in self.system.all_particles[self.atom_indices]])
        #return self.system.snap.particles.position[self.atom_indices]

    @property
    def atom_masses(self):
        """The mass of every particle in the structure

        Returns:
        --------
        numpy.ndarray, shape=(n, 1), dtype=float

        """
        return np.array([p.mass for p in self.system.all_particles[self.atom_indices]])
    
    @property
    def mass(self):
        """The mass of the structure"""
        return sum(self.atom_masses)
        #return sum(self.system.snap.particles.mass[self.atom_indices])

    @property
    def center(self):
        """The (x,y,z) position of the center of the structure
        
        Returns:
        --------
        numpy.ndarray, shape=(3,), dtype=float

        """
        com = sum([
                xyz*mass for xyz, mass in zip(
                        self.atom_positions, self.atom_masses
                    )
                ]) / self.mass 
        return com 


class Molecule(Structure):
    """The Structure object containing information about the entire molecule.

    Parameters:
    -----------
    system : 'System()', required
        The system object initially created from the input .gsd file.
    molecule_id : int, optional, default=None
        The ID number of the specific molecule from system.molecule_ids.

    Attributes:
    -----------
    system : 'System()'
        The system that this structure belong to. Contains needed information
        about the box, and gsd snapshot which are used elsewhere.
    monomers : List of Monomer() objects.
        List of Monomer objects contained only within this molecule.
    segments : List of Segment() objects.
        List of Segment objects contained only within this molecule.
    components : List of Component() objects.
        List of Component objects contained only within this molecule.
    sequence : str
        The monomer type sequence specific to this molecule.

    Methods:
    --------
    assign_types : Assigns the type names to each child monomer bead
        Requires that the Molecule.sequence attribute is defined before hand.
    generate_segments : Creates Structure() objects for child segments.

    """
    def __init__(self, system, molecule_id):
        super(Molecule, self).__init__(
                system=system,
                molecule_id=molecule_id
        )
        self.monomers = self.generate_monomers() 
        self.n_monomers = len(self.monomers)
        self.segments = [] 
        self.components = []
        self.sequence = None

    def assign_types(
            self,
            use_monomers=True,
            use_segments=False,
            use_components=False
    ):
        """Assigns the type names to each child monomer bead.
        Requires that self.sequence attribute is defined behond hand.
        If assigning types for segments or components, they must
        be generated first.
        
        Parameters:
        -----------
        use_monomers, use_segments, use_components : boolean
            Specifies the type of substructure to assign types to.

        """
        if self.sequence is None:
            raise ValueError(
                    "The sequence for each molecule must be set "
                    "before the bead types can be assigned. "
                    "See the `Molecule.sequence attribute."
            )
        if use_monomers:
            n = self.n_monomers // len(self.sequence)
            monomer_sequence = self.sequence * n
            monomer_sequence += self.sequence[:(self.n_monomers
                - len(monomer_sequence))]
            for i, name in enumerate(list(monomer_sequence)):
                self.monomers[i].name = name

        elif use_segments:
            n = len(self.segments) // len(self.sequence)
            segment_sequence = self.sequence * n
            segment_sequence += self.sequence[:(len(self.segments) - 
                len(segment_sequence))]
            for i, name in enumerate(list(segment_sequence)):
                self.segments[i].name = name

        elif use_components:
            n = len(self.components) // len(self.sequence)
            comp_sequence = self.sequence * n
            comp_sequence += self.sequence[:(len(self.components) - 
                len(comp_sequence))]
            for i, name in enumerate(list(comp_sequence)):
                self.components[i].name = name

    def generate_segments(self, monomers_per_segment):
        """Creates a `Segment` that contains a subset of its `Molecule` atoms.

        Segments are defined as containing a certain number of monomers.
        For example, if you want 3 subsequent monomers contained in a single
        Segment instance, use `monomers_per_segment = 3`.
        The segments are accessible in the `Molecule.segments` attribute.

        Parameters:
        -----------
        monomers_per_segment : int, required
            Define the number of consecutive monomers that belong to
            each segment.

        """
        segments_per_molecule = int(self.n_monomers / monomers_per_segment)
        segment_indices = np.array_split(
                self.atom_indices,
                segments_per_molecule
        )
        self.segments.extend([Segment(self, i) for i in segment_indices])
    
    def bond_vectors(
            self,
            use_monomers=False,
            use_segments=False,
            use_components=False,
            normalize=False,
            pair=None,
            exclude_ends=False
    ):
        """Generates a list of the vectors connecting subsequent monomer 
        or segment units.

        Uses the monomer or segment average center coordinates.
        In order to return the bond vectors between segments, the 
        Segment objects need to be created; see the `generate_segments`
        method in the `Molecule` class.

        Parameters:
        -----------
        use_monomers : bool, optional, default=True
            Set to True to return bond vectors between the Molecule's monomers.
        use_segments : bool, optional, default=False
            Set to True to return bond vectors between the Molecule's segments.
        use_components : bool, optional, default=False
            Set to True to return bond vectors between the Molecule's components.
        normalize : bool, optional, default=False
            Set to True to normalize each vector by its magnitude.

        Returns:
        --------
        list of numpy.ndarray, shape=(3,), dtype=float

        """
        sub_structures = self._sub_structures(
                use_monomers,
                use_segments,
                use_components
        )

        vectors = []
        for idx, s in enumerate(sub_structures):
            if exclude_ends:
                if idx == 0 or idx == len(sub_structures) - 1:
                    continue
            try:
                s2 = sub_structures[idx+1]
                if pair:
                    if sorted(pair) == sorted([s.name, s2.name]):
                        pass
                    else:
                        continue
                vector = (s2.unwrapped_center - s.unwrapped_center)
                if normalize:
                    vector /= np.linalg.norm(vector)
                vectors.append(vector)
            except IndexError:
                pass
        return vectors

    def bond_angles(
            self,
            use_monomers=False,
            use_segments=False,
            use_components=False,
            group=None
    ):
        """Generates a list of the angles between subsequent monomer 
        or segment bond vectors.

        Uses the output returned by the `bond_vectors` method
        in the `Molecule` class.
        In order to return the angles between segments, the 
        Segment objects first need to be created; see the
        `generate_segments` method in the `Molecule` class.

        Parameters:
        -----------
        use_monomers : bool, optional, default=True
            Set to True to return angles between the Molecule's monomers
        use_segments : bool, optional, default=False
            Set to True to return angles between the Molecule's segments
        use_components : bool, optional, default=False
            Set to True to reutrn the angles between the Molecule's components

        Returns:
        --------
        list of numpy.ndarray, shape=(3,), dtype=float
        The angles are given in radians

        """
        sub_structures = self._sub_structures(
                use_monomers,
                use_segments,
                use_components
        )

        angles = []
        for idx, s in enumerate(sub_structures):
            try:
                s2 = sub_structures[idx+1]
                s3 = sub_structures[idx+2]
                if group is not None:
                    if list(group) == [s.name, s2.name, s3.name]:
                        pass
                    else:
                        continue
                v1 = (s.unwrapped_center - s2.unwrapped_center)
                v2 = (s3.unwrapped_center - s2.unwrapped_center)
                cos_angle = (
                        np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                        )
                angles.append(np.arccos(cos_angle))
            except IndexError:
                pass
        return angles

    def bond_dihedrals(
            self,
            use_monomers=False,
            use_segments=False,
            use_components=False,
            group=None
    ):
        """Generates a list of the dihedrals between subsequent bond vectors.
        
        Parameters:
        -----------
        use_monomers : bool, optional, default=True
            Set to True to return angles between the Molecule's monomers
        use_segments : bool, optional, default=False
            Set to True to return angles between the Molecule's segments
        use_components : bool, optional, default=False
            Set to True to reutrn the angles between the Molecule's components

        Returns:
        --------
        list of numpy.ndarray, shape=(3,), dtype=float
        The dihedral angles are given in radians

        """
        bonds = self.bond_vectors(use_monomers, use_segments, use_components)
        dihedrals = [] 
        
        for idx, a1 in enumerate(bonds):
            try:
                a2 = bonds[idx+1]
                a3 = bonds[idx+2]
                v1 = np.cross(a1, a2)
                v1 = v1 / (v1 * v1).sum(-1)**0.5
                v2 = np.cross(a2, a3)
                v2 = v2 / (v2 * v2).sum(-1)**0.5
                porm = np.sign((v1 * a3).sum(-1))
                rad = np.arccos((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5)
                if porm != 0:
                    rad = rad * porm
                dihedrals.append(rad)
            except IndexError:
                pass
        return dihedrals

    def end_to_end_distance(self, squared=False):
        """Retruns the magnitude of the vector connecting the first and
        last monomer in Molecule.monomers. Uses each monomer's center
        coordinates.

        Parameters:
        -----------
        squared : bool, optional default=False
            Set to True if you want the squared end-to-end distance

        Returns:
        --------
        numpy.ndarray, shape=(1,), dtype=float

        """
        head = self.monomers[0]
        tail = self.monomers[-1]
        distance = np.linalg.norm(
                tail.unwrapped_center - head.unwrapped_center
        )
        if squared:
            distance = distance**2
        return distance

    def radius_of_gyration(
            self,
            use_monomers=False,
            use_segments=False,
            use_components=False,
            group=None
    ):
        """Finds the squared radius of gyrtation (Rg) 

        Parameters:
        -----------
        use_monomers : bool, optional, default=True
            Set to True to use the Molecule's monomers when finding Rg.
        use_segments : bool, optional, default=False
            Set to True to use the Molecule's segments when finding Rg.
        use_components : bool, optional, default=False
            Set to True to use the Molecule's components when finding Rg.

        Returns:
        --------
        float : Radius of gyration of the molecule

        """
        sub_structures = self._sub_structures(
                use_monomers,
                use_segments,
                use_components
        )
        struc_pos = np.array([s.unwrapped_center for s in sub_structures])
        mol_center = self.unwrapped_center
        radius_of_gyration = (
                np.sum([(i - mol_center)**2 for i in struc_pos])
            ) / len(struc_pos)
        return radius_of_gyration

    def persistence_length(self):
        ""
        ""
        pass

    def _sub_structures(self, monomers, segments, components):
        args = [monomers, segments, components]
        if args.count(True) > 1:
            raise ValueError(
                    "Only one of `monomers`, `segments`, and `components` "
                    "can be set to `True`"
            )
        if not any(args):
            raise ValueError(
                    "Set one of `monomers`, `segments`, `components` to "
                    "`True` depending on which structure bond vectors "
                    "you want returned."
            )
        if monomers:
            sub_structures = self.monomers
        elif segments:
            if self.segments == None:
                raise ValueError(
                        "The segments for this molecule have not been "
                        "created. See the `generate_segments()` method for "
                        "the `Molecule` class."
                )
            sub_structures = self.segments
        elif components:
            if self.components == None:
                raise ValueError(
                        "The components for this molecule have not been "
                        "created. See the `generate_components()` method for "
                        "the `Monomer` class."
                )
            sub_structures = self.components
        return sub_structures


class Monomer(Structure):
    """
    """
    def __init__(self, parent, atom_indices):
        super(Monomer, self).__init__(
                system=parent.system,
                parent=parent,
                atom_indices=atom_indices
        )
        self.components = [] 
        
    def generate_components(self, index_mapping):
        """
        """
        if self.components:
            raise ValueError("Components have already been generated")
        if isinstance(index_mapping, str):
            index_mapping = self.system.comp_dict[
                    "component_mappings"][0][index_mapping]
        elif isinstance(index_mapping, dict):
            pass
        else:
            raise ValueError("Index mapping should be a dictionary of "
                   "bead_name: bead_indices."
                   "Or,` a label for one of the component mappings defined "
                   "in a compound JSON file."
            )

        components = []
        for name, indices in index_mapping.items():
            if all([isinstance(i, list) for i in indices]):
                for i in indices:
                    component = Component(
                            monomer=self,
                            name=name,
                            atom_indices = self.atom_indices[i]
                    )
                    components.append(component)
            else:
                component = Component(
                        monomer=self,
                        name=name,
                        atom_indices = self.atom_indices[indices]
                )
                components.append(component)
        self.components.extend(components)
        self.parent.components.extend(components)


class Segment(Structure):
    """
    """
    def __init__(self, molecule, atom_indices):
        super(Segment, self).__init__(
                system=molecule.system,
                atom_indices=atom_indices,
                parent = molecule
        )
        self.monomers = self.generate_monomers()
        assert len(self.monomers) ==  int(
                self.n_atoms / self.system.atoms_per_monomer
        )


class Component(Structure):
    def __init__(self, monomer, atom_indices, name):
        super(Component, self).__init__(
                system=monomer.system,
                parent=monomer.parent,
                atom_indices=atom_indices,
                name=name
        )
        self.monomer = monomer
