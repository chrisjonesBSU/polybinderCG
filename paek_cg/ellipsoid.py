import mbuild as mb
import gsd
import gsd.hoomd
import numpy as np

class Ellipsoid(mb.Compound):
    def __init__(
            self,
            name,
            major_length,
            minor_length,
            mass,
            major_axis=[1,0,0],
            minor_axis[0,1,0],
            )
    super(Ellipsoid, self).__init__()
    self.name = name
    self.major_axis = np.array(major_axis)
    self.minor_axis = np.array(minor_axis)
    self.major_length = major_length
    self.minor_length = minor_length

    center = mb.Compound(
            pos=[0,0,0],
            name=self.name,
            mass=mass
            )
    head = mb.Compound(
            pos=major_axis*major_length/2,
            name=f"_{self.name}_head"
            )
    tail = mb.Compound(
            pos=-major_axis*major_length/2,
            name=f"_{self.name}_tail"
            )
    right = mb.Compound(
            pos=minor_axis*minor_length/2,
            name=f"_{self.name}_right"
            )
    left = mb.Compound(
            pos=-minor_axis*minor_length/2,
            name=f"_{self.name}_left"
            )

    self.add([center, head, tail, right, left])


