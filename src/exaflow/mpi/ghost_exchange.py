from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..config.boundaries import Boundaries, collect_faces
from ..fields import FlowState
from .subdomain import Subdomain

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm, Request


class GhostExchange:
    """
    Exchanges the outermost real layer with every neighboring rank, for every field array in a state.

    Build one per run and call `start` then `complete` on each stage. The plan and the message buffers are built here once, so a stage allocates nothing. Between `start` and `complete` the caller may update cells whose stencil stays inside the real block; every cell whose stencil reaches a ghost layer must wait.

    Only the interior transverse extent is exchanged, so the corner region of a ghost plane is left alone. No stencil in this solver reads a diagonal ghost cell; a stencil that did would need a second pass, one axis at a time.

    A face whose neighbor is another rank needs a communicator, and the constructor raises when one is missing, so `_messages` is empty whenever `comm` is None.

    A face with no neighbor is skipped. A face whose neighbor is this rank itself, which is what a periodic axis carrying a single rank gives, is served by a local copy from the opposite face instead of a message, so a serial periodic run needs no communicator.

    Every message carries the tag of the face it leaves from, and a receive asks for the tag of the opposite face. Without the tag, a periodic axis carrying exactly two ranks makes each rank post two sends and two receives to the same peer, and MPI matches them in order, which fills the low ghost plane with the high one. Within one face the arrays are matched by MPI non-overtaking, so they are always walked in the order `state.collect_arrays()` gives them.
    """

    def __init__(
        self,
        subdomain: Subdomain,
        boundaries: Boundaries,
        comm: Intracomm | None,
        num_arrays: int | None = None,
    ) -> None:
        self._comm: Intracomm | None = comm
        self._requests: list[Request] = []
        self._copies: list[tuple[int, tuple[slice, ...], tuple[slice, ...]]] = []
        self._messages: list[tuple[int, int, tuple[slice, ...], tuple[slice, ...], int, int, np.ndarray, np.ndarray]] = []

        pad = subdomain.grid.num_ghost_layers
        transverse = subdomain.interior
        if num_arrays is None:
            num_arrays = subdomain.grid.dimension + 1

        for face in collect_faces(subdomain.grid.dimension):
            neighbor = subdomain.find_neighbor_rank(face, boundaries)
            if neighbor is None:
                continue
            axis = face.axis
            if face.is_low:
                real_span, ghost_span, opposite_span = slice(pad, 2 * pad), slice(0, pad), slice(-2 * pad, -pad)
            else:
                real_span, ghost_span, opposite_span = slice(-2 * pad, -pad), slice(-pad, None), slice(pad, 2 * pad)
            source = tuple(real_span if i == axis else transverse[i] for i in range(len(transverse)))
            destination = tuple(ghost_span if i == axis else transverse[i] for i in range(len(transverse)))

            if neighbor == subdomain.rank:
                opposite = tuple(opposite_span if i == axis else transverse[i] for i in range(len(transverse)))
                for index in range(num_arrays):
                    self._copies.append((index, destination, opposite))
                continue

            if comm is None:
                raise ValueError(f"{face.name} has neighbor rank {neighbor} but no communicator was given.")

            shape = tuple(
                pad if i == axis else subdomain.shape[i]
                for i in range(subdomain.grid.dimension)
            )
            for index in range(num_arrays):
                self._messages.append((
                    index,
                    neighbor,
                    source,
                    destination,
                    2 * axis + (0 if face.is_low else 1),
                    2 * axis + (1 if face.is_low else 0),
                    np.empty(shape, dtype=float),
                    np.empty(shape, dtype=float),
                ))

    def start(self, state: FlowState) -> None:
        """
        Post every send and receive, and take every local copy. The caller must call `complete` before it reads any ghost cell.
        """

        self.start_arrays(state.collect_arrays())

    def start_arrays(self, arrays: Sequence[np.ndarray]) -> None:
        for index, destination, opposite in self._copies:
            arrays[index][destination] = arrays[index][opposite]
        if self._comm is None:
            return
        for index, neighbor, source, _, send_tag, recv_tag, outgoing, incoming in self._messages:
            np.copyto(outgoing, arrays[index][source])
            self._requests.append(self._comm.Isend(outgoing, dest=neighbor, tag=send_tag))
            self._requests.append(self._comm.Irecv(incoming, source=neighbor, tag=recv_tag))

    def complete(self, state: FlowState) -> None:
        """
        Wait for every posted message, then copy each received plane into its ghost layer.
        """

        self.complete_arrays(state.collect_arrays())

    def complete_arrays(self, arrays: Sequence[np.ndarray]) -> None:
        for request in self._requests:
            request.Wait()
        self._requests.clear()
        for index, _, _, destination, _, _, _, incoming in self._messages:
            arrays[index][destination] = incoming
