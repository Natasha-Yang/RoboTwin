"""Round peg into a round bore, middle rung: 3.0 mm clearance per side (46 mm bore, 40 mm peg).

The cylindrical counterpart of `insert_peg_socket_med`. Everything else is deliberately held
equal -- same 120 mm peg length, same 40 mm width across the fit, same 0.10 x 0.10 x 0.05
socket block, same 0.038 m bore depth, same 6 mm chamfer lead-in, same spawn ranges, same
expert, same grasp band -- so the cross-section of the fit is the only difference between the
two tasks and a difference in outcome is attributable to it.

Two things follow from the peg being a solid of revolution, and both make this the EASIER of
the pair at equal clearance:

  * **No yaw to get right.** The square task has to rotate the peg onto one of the bore's four
    symmetry axes before it can descend; here every yaw seats, so `place_constrain = "free"`
    imposes none at all (see `_PegInsertionBase._place_kwargs`). One whole error source is
    gone, and with it the `get_align_matrix` near-antiparallel discontinuity.
  * **The clearance is the same in every direction.** A square peg in a square bore has
    3.0 mm at the flats but 4.2 mm on the diagonal, so it can wedge on two corners; a round
    fit has 3.0 mm everywhere, which is the classical peg-in-hole geometry.

The collision bore is a 32-gon circumscribing the nominal circle, so the narrowest point is
exactly the nominal clearance and the faceting adds at most 0.11 mm (script/gen_peg_socket_asset.py).
"""

from ._peg_insertion_base import (ROUND_PEG_MODELNAME, ROUND_SOCKET_MODELNAME, _PegInsertionBase)


class insert_peg_socket_round_med(_PegInsertionBase):
    socket_modelname = ROUND_SOCKET_MODELNAME
    socket_model_id = 1
    peg_modelname = ROUND_PEG_MODELNAME
    peg_description = f"{ROUND_PEG_MODELNAME}/base0"
    place_constrain = "free"
