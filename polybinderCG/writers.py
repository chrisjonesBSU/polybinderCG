import gsd
import gsd.hoomd
import freud
import numpy as np
import re

def write_snapshot(beads, rewrap=True, box_expand=None):
    """Creates a gsd.hoomd.snapshot of a coarse-grained mapping.

    Parameters
    ----------
    beads : iterable, required
        An iterable of any of the structure classes in coarse_grain.py
        For example, if you want to coarse-grain based on monomers,
        pass in a list of System's monomer objects.
    rewrap : bool, defualt=True
        Set to True to rewrap the beads into the system accounting
        for periodic boundaries
    box_expand = int, default=None
        If not rewrapping the beads, set a factor in which to increase
        the box size.

    Returns
    -------
    gsd.hoomd.snapshot
        A snapshot of the coarse-grained representation.

    """
    all_types = []
    all_pairs = []
    pair_groups = []
    all_angles = []
    angle_groups = []
    all_dihedrals = []
    dihedral_groups = []
    all_pos = []
    masses = []
    box = beads[0].system.box 

    for idx, bead in enumerate(beads):
        all_types.append(bead.name)
        all_pos.append(bead.unwrapped_center)
        masses.append(bead.mass)

        try:
            # Add bond type and group indices
            if bead.parent == beads[idx+1].parent:
                pair = sorted([bead.name, beads[idx+1].name])
                pair_type = "-".join((pair[0], pair[1]))
                all_pairs.append(pair_type)
                pair_groups.append([idx, idx+1])
                # Add angle types and group indices
                if bead.parent == beads[idx+2].parent:
                    b1, b2, b3 = bead.name, beads[idx+1].name, beads[idx+2].name
                    b1, b3 = sorted([b1, b3], key=_natural_sort)
                    angle_type = "-".join((b1, b2, b3))
                    all_angles.append(angle_type)
                    angle_groups.append([idx, idx+1, idx+2])
                # Add dihedral types and group indices 
                if bead.parent == beads[idx+3].parent:
                    b1, b2, b3, b4 = [
                            bead.name,
                            beads[idx+1].name,
                            beads[idx+2].name,
                            beads[idx+3].name
                    ]
                    _b1, _b4 = sorted(
                            [b1, b4], key=natural_sort
                    )
                    _b2 = b2
                    _b3 = b3
                    
                    if [_b2, _b3] == sorted([_b2, _b3], key=natural_sort):
                        dihedral_type = "-".join((_b1, _b2, _b3, _b4))
                    else:
                        dihedral_type = "-".join((_b4, _b3, _b2, _b1))
                    all_dihedrals.append(dihedral_type)
                    dihedral_groups.append([idx, idx+1, idx+2, idx+3])
        except IndexError:
            pass

    types = list(set(all_types)) 
    pairs = list(set(all_pairs)) 
    angles = list(set(all_angles))
    dihedrals = list(set(all_dihedrals))
    type_ids = [np.where(np.array(types)==i)[0][0] for i in all_types]
    pair_ids = [np.where(np.array(pairs)==i)[0][0] for i in all_pairs]
    angle_ids = [np.where(np.array(angles)==i)[0][0] for i in all_angles]
    dihedral_ids = [np.where(np.array(dihedrals)==i)[0][0] for i in all_dihedrals]

    #Wrap the particle positions
    if rewrap:
        fbox = freud.box.Box(*box)
        w_positions = fbox.wrap(all_pos)
        w_images = fbox.get_images(all_pos)
    else:
        box *= box_expand 
        w_positions = all_pos
        w_images = np.array([[0,0,0] for i in all_pos])

    s = gsd.hoomd.Snapshot()
    #Particles
    s.particles.N = len(all_types)
    s.particles.types = types 
    s.particles.typeid = np.array(type_ids) 
    s.particles.position = w_positions 
    s.particles.mass = masses
    s.particles.image = w_images
    #Bonds
    s.bonds.N = len(all_pairs)
    s.bonds.M = 2
    s.bonds.types = pairs
    s.bonds.typeid = np.array(pair_ids)
    s.bonds.group = np.vstack(pair_groups)
    #Angles
    s.angles.N = len(all_angles)
    s.angles.M = 3
    s.angles.types = angles
    s.angles.typeid = np.array(angle_ids)
    s.angles.group = np.vstack(angle_groups)
    #Dihedrals
    s.dihedrals.N = len(all_dihedrals)
    s.dihedrals.M = 4
    s.dihedrals.types = dihedrals
    s.dihedrals.typeid = np.array(dihedral_ids)
    s.dihedrals.group = np.vstack(dihedral_groups)
    s.configuration.box = box 
    return s

def _atoi(text):
    return int(text) if text.isdigit() else text

def _natural_sort(text):
    """Break apart a string containing letters and digits."""
    return [_atoi(a) for a in re.split(r"(\d+)", text)]

