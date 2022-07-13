from cmeutils.gsd_utils import get_molecule_cluster
from polybinderCG.writers import write_snapshot
import gsd
import gsd.hoomd
import freud
import json
import numpy as np
from polybinderCG.compounds import COMPOUND_DIR

class System:
    """
    """
    def __init__(
            self,
            compound=None,
            atoms_per_monomer=None,
            gsd_file=None,
            gsd_frame=-1
    ):
        self.gsd_file = gsd_file
        self.update_frame(gsd_frame) # Sets self.frame, self.snap, self.box
        self.contains_H = self._check_for_Hs()
        self.compound = compound
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
        self.molecule_ids = set(self.clusters)
        self.n_molecules = len(self.molecule_ids)
        self.n_atoms = len(self.clusters)
        self.molecules = [Molecule(self, i) for i in self.molecule_ids] 

    def coarse_grain_trajectory(
            self,
            file_path,
            use_monomers=False,
            use_segments=False,
            use_components=False,
            first_frame = 0,
            last_frame = -1
    ):
        """Creates a new GSD file of the coarse-grained
        representaiton of the system.

        Parameters
        ----------
        use_monomers : bool, optional, default=True
            Set to True to use the molecule's Monomers as the CG beads.
        use_segments : bool, optional, default=False
            Set to True to use the molecule's Components as the CG beads.
        use_components : bool, optional, default=False
            Set to True to use the molecule's Components as the CG beads.

        """
        args = [use_monomers, use_segments, use_components]
        if args.count(True) > 1:
            raise ValueError("You can only choose one of monomers, "
                    "segments or components."
            )
        if not any(args):
            raise ValueError("Select one of monomers, segments, "
                    "or components as the coarse-grained beads."
            )
        current_frame = self.frame
        if first_frame < 0:
            first_frame = self.n_frames + first_frame
        if last_frame < 0:
            last_frame = self.n_frames + last_frame + 1
        with gsd.hoomd.open(file_path, mode="wb") as f:
            for i in range(first_frame, last_frame):
                self.update_frame(frame=i)
                snap = self.coarse_grain_snap(
                        use_monomers=use_monomers,
                        use_segments=use_segments,
                        use_components=use_components
                )
                f.append(snap)
        self.update_frame(frame=current_frame)

    def coarse_grain_snap(
            self, use_monomers=False, use_segments=False, use_components=False
    ):
        """Creates a gsd.hoomd.snapshot of a coarse-grained representation.

        Parameters
        ----------
        use_monomers : bool, optional, default=True
            Set to True to use the molecule's Monomers as the CG beads.
        use_segments : bool, optional, default=False
            Set to True to use the molecule's Components as the CG beads.
        use_components : bool, optional, default=False
            Set to True to use the molecule's Components as the CG beads.

        Returns
        -------
        gsd.hoomd.snapshot
            A snapshot of the coarse-grained representation.

        """
        if use_monomers:
            structures = [i for i in self.monomers()]
        elif use_segments:
            if len(self.molecules[0].segments) == 0:
                raise ValueError("Segments have not been created. "
                        "See the generate_segments method in "
                        "the Molecule class."
                )
            structures = [i for i in self.segments()]
        elif use_components:
            if len(self.molecules[0].components) == 0:
                raise ValueError("Components have not been generated. "
                        "See the generate_components method in "
                        "the Monomer class. "
                )
            structures = [i for i in self.components()]
        return write_snapshot(structures)

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

    def end_to_end_distances(self, squared=True):
        """Returns the end-to-end distances of each molecule in the system.

        Parameters
        ----------
        squared : bool, optional, default=False
            Set to True if you want the mean squared average
            end-to-end distance.

        Returns
        -------
        numpy.ndarray, shape=(1,self.n_molecules), dtype=float
            The average end-to-end distance averaged over all of the
            molecules in System.molecules

        """
        distances = [mol.end_to_end_distance(squared) for mol in self.molecules]
        return distances 

    def radii_of_gyration(
            self,
            use_monomers=False,
            use_segments=False,
            use_components=False,
    ):
        """Returns the squared radius of gyration for each molecule
        in the system.

        """
        radii_gyration = [mol.radius_of_gyration(
                use_monomers, use_segments, use_components
            ) for mol in self.molecules]
        return radii_gyration

    def persistence_lengths(self):
        """
        """
        pass

    def bond_lengths(
            self,
            normalize=False,
            use_monomers=False,
            use_segments=False,
            use_components=False,
            pair=None,
            exclude_ends=False
    ):
        """
        """
        bond_lengths = []
        for mol in self.molecules:
            bond_lengths.extend(
                    [np.linalg.norm(vec) for vec in mol.bond_vectors(
                            use_monomers=use_monomers,
                            use_segments=use_segments,
                            use_components=use_components,
                            normalize=normalize,
                            pair=pair,
                            exclude_ends=exclude_ends
                        )
                    ]
            )
        return bond_lengths

    def bond_angles(
            self,
            use_monomers=False,
            use_segments=False,
            use_components=False,
            group=None,
    ):
        """
        """
        bond_angles = []
        for mol in self.molecules:
            bond_angles.extend(mol.bond_angles(
                use_monomers=use_monomers,
                use_segments=use_segments,
                use_components=use_components,
                group=group
            )
        )
        return bond_angles

    def bond_dihedrals(
            self,
            use_monomers=False,
            use_segments=False,
            use_components=False,
            group=None,
    ):
        """
        """
        dihedrals = []
        for mol in self.molecules:
            dihedrals.extend(mol.bond_dihedrals(
                use_monomers=use_monomers,
                use_segments=use_segments,
                use_components=use_components,
                group=group
            )
        )
        return dihedrals

    def update_frame(self, frame):
        """Change the frame of the atomistic trajectory."""
        self.frame = frame
        with gsd.hoomd.open(self.gsd_file, mode="rb") as f:
            self.snap = f[frame]
            self.box = self.snap.configuration.box
            self.n_frames = len(f) 

    def _check_for_Hs(self):
        """Returns True if the gsd snapshot contains hydrogen type atoms"""
        hydrogen_types = ["ha", "h", "ho", "h4"]
        if any([h in list(self.snap.particles.types) for h in hydrogen_types]):
            return True
        else:
            return False


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
            if self.molecule_id == 1:
                print(self.atom_indices)
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
            start = head_indices[-2]+1
            start = self.system.atoms_per_monomer-2 
            stop = tail_indices[0]
            stop = -self.system.atoms_per_monomer
            if self.molecule_id == 1:
                print(start, stop)
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
        """The wrapped coordinates of every particle in the structure
        as they exist in the periodic box.

        Returns:
        --------
        numpy.ndarray, shape=(n, 3), dtype=float

        """
        return self.system.snap.particles.position[self.atom_indices]
    
    @property
    def mass(self):
        """The mass of the structure"""
        return sum(self.system.snap.particles.mass[self.atom_indices])

    @property
    def center(self):
        """The (x,y,z) position of the center of the structure accounting
        for the periodic boundaries in the system.
        
        Returns:
        --------
        numpy.ndarray, shape=(3,), dtype=float

        """
        freud_box = freud.Box(
                Lx = self.system.box[0],
                Ly = self.system.box[1],
                Lz = self.system.box[2]
        )
        return freud_box.center_of_mass(self.atom_positions)

    @property 
    def unwrapped_atom_positions(self):
        """The unwrapped coordiantes of every particle in the structure.
        The positions are unwrapped using the images for each particle
        stored in the hoomd snapshot.

        Returns:
        --------
        numpy.ndarray, shape=(n, 3), dtype=float

        """
        images = self.system.snap.particles.image[self.atom_indices]
        return self.atom_positions + (images * self.system.box[:3]) 

    @property
    def unwrapped_center(self):
        """The (x,y,z) position of the center using the structure's
        unwrapped coordiantes.

        Returns:
        --------
        numpy.ndarray, shape=(3,), dtype=float

        """
        x_mean = np.mean(self.unwrapped_atom_positions[:,0])
        y_mean = np.mean(self.unwrapped_atom_positions[:,1])
        z_mean = np.mean(self.unwrapped_atom_positions[:,2])
        return np.array([x_mean, y_mean, z_mean])


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
        for idx, vec in enumerate(bonds):
            try:
                vec2 = bonds[idx+1]
                vec3 = bonds[idx+2]
                num = np.dot(np.cross(vec, vec2), np.cross(vec2, vec3))
                denom = (np.linalg.norm(
                    np.cross(vec, vec2)))*(
                            np.linalg.norm(np.cross(vec2, vec3)))
                dihedrals.append(np.arccos(num/denom))
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
