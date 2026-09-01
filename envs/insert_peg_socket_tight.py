"""Peg insertion, hardest rung: 1.5 mm clearance per side (43 mm bore, 40 mm peg).

See envs/insert_peg_socket_loose.py for the ladder. At this clearance the 12 mm chamfer is
doing most of the work: the expert's own placement error is comparable to the fit, so the
peg is expected to make contact with the bore on the way in rather than fall through it.
That contact is the point -- it is what a wrench-conditioned critic has to read.
"""

from ._peg_insertion_base import _PegInsertionBase


class insert_peg_socket_tight(_PegInsertionBase):
    socket_model_id = 2
