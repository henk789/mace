import torch as th
from collections import namedtuple
import ase

System = namedtuple("System", ("R", "Z", "cell"))
UnfoldedSystem = namedtuple(
    "System", ("R", "Z", "cell", "mask", "replica_idx", "padding_mask", "updated")
)
Unfolding = namedtuple(
    "Unfolding",
    (
        "replica_idx",  # [M] ; original atom of which this is a replica
        "replica_offsets",  # [M, 3] ; which cell offsets to apply
        "wrap_offsets",  # [N, 3] ; offsets to apply to return positions to cell
        "padding_mask",  # [M] ; if False this is a fake position
        "reference_positions",  # [N, 3] ; positions at last update
        "reference_cell",  # [3, 3] ; cell at last update
        "updated",  # if True, this unfolding has just been updated
    ),
)


def to_frac(cell, R):
    return th.einsum("Aa,ia->iA", th.inverse(cell), R)


def get_wrap_offsets(positions, cell):
    """
    Get the wrap offsets for each atom in the cell.

    Args:
        positions (torch.Tensor): The positions of the atoms.
        cell (torch.Tensor): The cell vectors.

    Returns:
        torch.Tensor: The wrap offsets for each atom.
    """
    frac = to_frac(cell, positions)
    offsets = -1.0 * (frac // 1.0)

    return offsets


def wrap(positions, cell, offsets):
    """
    Wrap the positions of the atoms.

    Args:
        positions (torch.Tensor): The positions of the atoms.
        cell (torch.Tensor): The cell vectors.
        offsets (torch.Tensor): The wrap offsets for each atom.

    Returns:
        torch.Tensor: The wrapped positions of the atoms.
    """
    return positions + th.einsum("aA,iA->ia", cell, offsets)


def get_normals(cell):
    # surface normals of cell boundaries
    # (i.e. normalised lattice vectors of reciprocal lattice)
    # convention: indexed by the lattice vector they're not orthogonal to
    inv = th.inverse(cell)  # rows: inverse lattice vectors
    normals = inv / th.linalg.norm(inv, axis=1)[:, None]
    return normals


def project_on(normals, R):
    return th.einsum("Aa,ia->iA", normals, R)


def get_heights(cell):
    normals = get_normals(cell)
    return th.diag(project_on(normals, cell.T))


def project_on_normals(cell, R):
    return project_on(get_normals(cell), R)


def collision(X, heights, cutoff):
    x_lo = X[0] <= cutoff
    x_hi = X[0] >= heights[0] - cutoff

    y_lo = X[1] <= cutoff
    y_hi = X[1] >= heights[1] - cutoff

    z_lo = X[2] <= cutoff
    z_hi = X[2] >= heights[2] - cutoff

    return th.tensor([x_lo, x_hi, y_lo, y_hi, z_lo, z_hi], dtype=bool, device=X.device)


def collision_to_replica(collision):
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = collision

    out = th.zeros((3, 3, 3), dtype=bool, device=collision.device)

    # 6 faces

    out[+1, 0, 0] = x_lo
    out[-1, 0, 0] = x_hi

    out[0, +1, 0] = y_lo
    out[0, -1, 0] = y_hi

    out[0, 0, +1] = z_lo
    out[0, 0, -1] = z_hi

    # 12 edges

    out[+1, +1, 0] = x_lo & y_lo
    out[+1, -1, 0] = x_lo & y_hi

    out[-1, +1, 0] = x_hi & y_lo
    out[-1, -1, 0] = x_hi & y_hi

    out[0, +1, +1] = y_lo & z_lo
    out[0, +1, -1] = y_lo & z_hi

    out[0, -1, +1] = y_hi & z_lo
    out[0, -1, -1] = y_hi & z_hi

    out[+1, 0, +1] = x_lo & z_lo
    out[+1, 0, -1] = x_lo & z_hi

    out[-1, 0, +1] = x_hi & z_lo
    out[-1, 0, -1] = x_hi & z_hi

    # 8 corners

    out[+1, +1, +1] = x_lo & y_lo & z_lo
    out[+1, +1, -1] = x_lo & y_lo & z_hi

    out[-1, +1, -1] = x_hi & y_lo & z_hi
    out[-1, +1, +1] = x_hi & y_lo & z_lo

    out[+1, -1, +1] = x_lo & y_hi & z_lo
    out[-1, -1, +1] = x_hi & y_hi & z_lo

    out[-1, -1, -1] = x_hi & y_hi & z_hi
    out[+1, -1, -1] = x_lo & y_hi & z_hi

    return out


def vmap(fn):
    def _fn(*args):
        return th.stack([fn(*x) for x in zip(*args)])

    return _fn


def get_all_replicas(positions, cell, cutoff):
    heights = get_heights(cell)

    # [N, 3] ; positions projected onto normals
    projections = project_on_normals(cell, positions)

    # [N, 6] ; is within cutoff of left/right boundary
    collisions = vmap(lambda X: collision(X, heights, cutoff))(projections)

    # [N, 3, 3, 3] ; for each position, which of the 27 possible replicas is needed
    replicas = vmap(collision_to_replica)(collisions)

    return replicas


def get_unfolding(replicas, size):
    padded_replicas = th.argwhere(replicas)

    replica_idx = padded_replicas[:, 0]
    replica_offsets = th.tensor([0, 1, -1], dtype=int)[padded_replicas[:, 1:]]

    padding_mask = replica_idx != -1

    total = th.sum(replicas)
    overflow = padded_replicas.shape[0] < total

    if overflow:
        raise ValueError(
            f"warning: unfolding is only possible up to {size} replicas but {total} are needed"
        )

    return replica_idx, replica_offsets, padding_mask


def cell_too_small(cell, cutoff, skin):
    min_height = th.min(get_heights(cell))
    return (cutoff + skin) > min_height


def unfold(system, unfolding):
    wrapped = wrap(system.R, system.cell, unfolding.wrap_offsets)
    unfolded = replicate(
        wrapped[unfolding.replica_idx], system.cell, unfolding.replica_offsets
    )

    # avoid spurious gradients to positions[-1] (TODO: needed in torch?)
    unfolded = unfolded * unfolding.padding_mask[:, None]

    return wrapped, unfolded


def replicate(positions, cell, offsets):
    return positions + th.einsum("aA,iA->ia", cell, offsets.to(cell.dtype))


def wrap(positions, cell, offsets):
    return positions + th.einsum("aA,iA->ia", cell, offsets)


def unfold_system(system, unfolding):

    N = system.R.shape[0]

    wrapped, unfolded = unfold(system, unfolding)
    all_R = th.concatenate((wrapped, unfolded), axis=0)
    all_idx = th.concatenate((th.arange(N), unfolding.replica_idx), axis=0)
    all_Z = system.Z[all_idx]

    mask = th.arange(all_R.shape[0]) < N
    padding_mask = th.concatenate((th.ones(N, dtype=bool), unfolding.padding_mask))

    return UnfoldedSystem(
        all_R, all_Z, None, mask, all_idx, padding_mask, unfolding.updated
    )


def atoms_to_system(atoms):
    R = th.tensor(atoms.get_positions())
    Z = th.tensor(atoms.get_atomic_numbers())
    cell = th.tensor(atoms.get_cell())
    return System(R, Z, cell)


def system_to_atoms(system, atoms=None):
    new_atoms = ase.Atoms(
        symbols=system.Z,
        positions=system.R,
        cell=system.cell,
        pbc=False,
    )

    if atoms is not None:
        new_atoms.set_cell(atoms.get_cell())
    elif system.cell is None:
        import warnings

        warnings.warn("No cell information available")

    return new_atoms


def unfolder(system, cutoff, skin):
    """
    Unfold the positions of the atoms.

    Args:
        positions (torch.Tensor): The positions of the atoms.
        cell (torch.Tensor): The cell vectors.
        cutoff (float): The cutoff distance.
        skin (float): The skin distance.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The replica index, replica offsets, padding mask, and overflow.
    """
    positions = system.R.detach()
    cell = system.cell.detach()

    if cell_too_small(cell, cutoff, skin):
        min_height = th.min(get_heights(cell))
        raise ValueError(
            f"warning: unfolding is only possible up to {min_height:.1f} Å but total cutoff is {cutoff:.1f}+{skin:.1f}={skin+cutoff:.1f} Å"
        )

    wrap_offsets = get_wrap_offsets(positions, cell)
    wrapped_positions = wrap(positions, cell, wrap_offsets)

    replicas = get_all_replicas(wrapped_positions, cell, cutoff + skin)

    count = th.sum(replicas)
    size = int(count)

    replica_idx, replica_offsets, padding_mask = get_unfolding(replicas, size)

    return Unfolding(
        replica_idx,
        replica_offsets,
        wrap_offsets,
        padding_mask,
        positions,
        cell,
        th.tensor(False),
    )
