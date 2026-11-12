#!/usr/bin/env python3
"""Build the hero's brain-surface point clouds.

Reads   tools/shell-voxels.bin  (1 mm outer-surface voxels of the MNI152 mask: int8 x,y,z in MNI mm + uint8 T1 intensity)
        brain/slices-hd.webp    (0.5 mm T1 slices, z = -14..82 mm)
Writes  brain/shell-hd.bin, brain/shell-lite.bin

Two surfaces are joined at z = -13 mm:
  above  the folded cortical surface, an isosurface of the T1 volume at the grey matter / CSF boundary, so gyri and sulci are real;
  below  (inferior temporal lobes, cerebellum, brainstem, where there are no slices) the smoothed mask surface.
Points start on a regular lattice (the surface voxels of each volume), which gives the ordered, matrix look,
and are then slid along the gradient onto the smooth isosurface so there are no voxel terraces.
Each carries an outward normal from the smoothed volume gradient and a luminance that falls off into the sulci.

File layout (little endian):  uint32 N
                              int16[N*3]  position, MNI mm * 100
                              int8[N*3]   outward unit normal * 127
                              uint8[N]    luminance (255 = exposed gyral crown ... 0 = deep in a sulcus)

Needs numpy, scipy, pillow.   python3 tools/build_shell.py
"""
import os
import numpy as np
from scipy import ndimage as ni
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "shell-voxels.bin")
OUT = os.path.join(HERE, "..", "brain")
TARGETS = {"shell-hd.bin": 0.8, "shell-lite.bin": 1.25}  # lattice spacing, mm
MAX_DEPTH = 9.0

raw = np.fromfile(SRC, dtype=np.uint8).reshape(-1, 4)
vox = raw[:, :3].view(np.int8).astype(np.int32)

SEAM = -13.0
CORTEX_ISO = 0.58                                   # T1 (0..1) at the grey matter / CSF boundary
NX, NY, NZ, TX, TY, COLS = 294, 366, 97, 296, 368, 10
X0, Y0, Z0 = -73.0, -108.0, -14.0

# ---- mask hull from the 1 mm surface voxels (closed shell, so the interior can be flood-filled)
PAD = 6
org = vox.min(0) - PAD
dims = vox.max(0) - vox.min(0) + 2 * PAD + 1
shell = np.zeros(dims, bool)
q = vox - org
shell[q[:, 0], q[:, 1], q[:, 2]] = True
lab, _ = ni.label(~shell)                           # 6-connected: cannot leak through the shell
solid = (lab != lab[0, 0, 0]).astype(np.float32)    # shell + interior
depth_grid = ni.distance_transform_edt(solid).astype(np.float32)   # mm below the hull
wide = ni.gaussian_filter(solid, 3.2)


hull_field = ni.gaussian_filter(solid, 0.85)
hull_grad = np.stack(np.gradient(ni.gaussian_filter(solid, 1.4)), -1)


def at(vol, idx):
    return ni.map_coordinates(vol, idx.T, order=1, mode="nearest")


def unit(g, idx):
    v = np.stack([at(g[..., i], idx) for i in range(3)], -1)
    return -v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def settle(field, grad, idx, level, limit):
    """Slide lattice points along the gradient onto the smooth isosurface, so the voxel terraces disappear
    while the points keep their lattice ordering. limit = furthest a point may move, in voxels of this grid."""
    p = idx.astype(np.float32).copy()
    for _ in range(5):
        g = np.stack([at(grad[..., i], p) for i in range(3)], -1)
        step = (level - at(field, p))[:, None] * g / np.maximum((g * g).sum(1, keepdims=True), 1e-4)
        p += 0.8 * step
        off = p - idx
        far = np.linalg.norm(off, axis=1, keepdims=True)
        p = idx + off * np.minimum(1.0, limit / np.maximum(far, 1e-6))
    return p


def lattice_surface(inside):
    """Voxels that are inside and have a 6-neighbour outside: a one-voxel-thick surface on the lattice."""
    core = ni.binary_erosion(inside, ni.generate_binary_structure(3, 1), border_value=1)
    return inside & ~core


# ---- cortex from the T1 slices, resampled to 0.5 mm isotropic; index order is (z, y, x)
Image.MAX_IMAGE_PIXELS = None
atlas = np.asarray(Image.open(os.path.join(OUT, "slices-hd.webp")))[..., 0]
vol = np.zeros((NZ, NY, NX), np.float32)
for k in range(NZ):
    ox, oy = (k % COLS) * TX + 1, (k // COLS) * TY + 1
    vol[k] = atlas[oy:oy + NY, ox:ox + NX] / 255.0
CP = 6
t1 = np.pad(ni.zoom(vol, (2, 1, 1), order=1), ((0, CP), (CP, CP), (CP, CP)))
cx_field = ni.gaussian_filter(t1, 1.1)
cx_grad = np.stack(np.gradient(ni.gaussian_filter(t1, 1.6)), -1)


def build(g):
    # a regular lattice of points g mm apart (integer mm when g = 1, so layers line up with the slices)
    xs = np.arange(np.ceil(-76 / g), 76 / g) * g
    ys = np.arange(np.ceil(-112 / g), 78 / g) * g
    zs = np.arange(np.ceil(-76 / g), 86 / g) * g
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    mm = np.stack([X, Y, Z], -1).reshape(-1, 3)
    hi = mm - org                                                   # index into the 1 mm hull grids (x, y, z)
    ci = np.stack([(mm[:, 2] - Z0) / 0.5, (mm[:, 1] - Y0) / 0.5 + CP, (mm[:, 0] - X0) / 0.5 + CP], 1)   # T1 grid (z, y, x)
    shape = X.shape
    in_hull = (at(hull_field, hi) > 0.5).reshape(shape)
    in_cx = ((at(cx_field, ci) > CORTEX_ISO) & (mm[:, 2] >= Z0) & (mm[:, 2] <= 82)).reshape(shape)
    s_hull = lattice_surface(in_hull).ravel() & (mm[:, 2] < SEAM)
    s_cx = lattice_surface(in_cx).ravel() & (mm[:, 2] >= SEAM)
    s_cx &= at(depth_grid, hi) < MAX_DEPTH                          # skip ventricles and the deepest sulcal walls
    ci[s_cx] = settle(cx_field, cx_grad, ci[s_cx], CORTEX_ISO, 0.9 * g / 0.5)
    hi[s_hull] = settle(hull_field, hull_grad, hi[s_hull], 0.5, 0.9 * g)
    mm[s_cx] = np.stack([(ci[s_cx, 2] - CP) * 0.5 + X0, (ci[s_cx, 1] - CP) * 0.5 + Y0, ci[s_cx, 0] * 0.5 + Z0], 1)
    mm[s_hull] = hi[s_hull] + org
    hi[s_cx] = mm[s_cx] - org
    d = at(depth_grid, hi)
    n = np.zeros((len(mm), 3), np.float32)
    n[s_cx] = unit(cx_grad, ci[s_cx])[:, ::-1]
    n[s_hull] = unit(hull_grad, hi[s_hull])
    lum = np.zeros(len(mm), np.float32)
    lum[s_cx] = np.exp(-np.maximum(d[s_cx] - 1.0, 0) / 3.5)
    cav = np.clip((at(wide, hi[s_hull]) - 0.30) / 0.40, 0, 1)
    lum[s_hull] = 0.78 * (1 - 0.7 * cav)
    keep = s_cx | s_hull
    return mm[keep], n[keep], lum[keep]


for name, spacing in TARGETS.items():
    mm, n, lum = build(spacing)
    with open(os.path.join(OUT, name), "wb") as fh:
        fh.write(np.uint32(len(mm)).tobytes())
        fh.write(np.round(mm * 100).astype("<i2").tobytes())
        fh.write(np.round(n * 127).astype(np.int8).tobytes())
        fh.write(np.round(np.clip(lum, 0, 1) * 255).astype(np.uint8).tobytes())
    print(f"{name}: {len(mm)} points, {4 + 10 * len(mm)} bytes")
