# Model source

The G1 body and Inspire DFQ hand URDF/mesh assets in this directory were
copied from Unitree Robotics' official `unitree_ros` repository:

```text
https://github.com/unitreerobotics/unitree_ros
robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf
commit f3772ce54c56ef2d34c6aee8100bc768896c7d19
```

Local changes:

- Renamed the URDF to `g1_29dof_inspire_DFQ.urdf`.
- Removed the redundant `meshdir="meshes"` compiler attribute because the
  URDF mesh filenames already start with `meshes/`.
- Copied only the mesh files referenced by this URDF.

The original Unitree BSD 3-Clause license is included as `LICENSE.unitree`.
