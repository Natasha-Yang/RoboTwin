"""Peg insertion, middle rung: 3.0 mm clearance per side (46 mm bore, 40 mm peg).

See envs/insert_peg_socket_loose.py for the ladder. Half the clearance of `loose`, which is
around where curobo's tracking error starts to matter and the peg begins to ride the bore
rather than drop cleanly through it.
"""

from ._peg_insertion_base import _PegInsertionBase


class insert_peg_socket_med(_PegInsertionBase):
    socket_model_id = 1
