"""Peg insertion, easiest rung: 6.0 mm clearance per side (52 mm bore, 40 mm peg).

The ladder is `insert_peg_socket_{loose,med,tight}` -- identical scenes and expert, differing
only in the socket's bore. Start here: if the expert cannot solve this one, the problem is
the motion plan rather than the tolerance.
"""

from ._peg_insertion_base import _PegInsertionBase


class insert_peg_socket_loose(_PegInsertionBase):
    socket_model_id = 0
