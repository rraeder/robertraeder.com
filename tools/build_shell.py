#!/usr/bin/env python3
"""Build the hero's brain-surface point clouds.

Reads   tools/shell-voxels.bin  (1 mm outer-surface voxels of the MNI152 mask: int8 x,y,z in MNI mm + uint8 T1 intensity)
        brain/slices-hd.webp    (0.5 mm T1 slices, z = -14..82 mm)
Writes  brain/shell-hd.bin, brain/shell-lite.bin

Two surfaces are joined at z = -13 mm:
  above  the folded cortical surface, an isosurface of the T1 volume at the grey matter / CSF boundary, so gyri and sulci are real;
  below  (inferior temporal lobes, cerebellum, brainstem, where there are no slices) the smoothed mask surface.
Both are re-surfaced with marching cubes, which gives a sub-voxel surface and a true outward normal at every point.
Points are spread evenly over the surface (no voxel grid, so no moire), thinned with depth below the brain's outer hull,
and shuffled.

File layout (little endian):  uint32 N
                              int16[N*3]  position, MNI mm * 100
                              int8[N*3]   outward unit normal * 127
                              uint8[N]    luminance (255 = exposed gyral crown ... 0 = deep in a sulcus)

Needs numpy, scipy, scikit-image, pillow.   python3 tools/build_shell.py
"""
import os
import numpy as np
from scipy import ndimage as ni
from skimage import measure
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "shell-voxels.bin")
OUT = os.path.join(HERE, "..", "brain")
TARGETS = {"shell-hd.bin": 0.95, "shell-lite.bin": 1.35}  # mean point spacing, mm
rng = np.random.default_rng(7)

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


def surface(field, level, grad_sigma):
    verts, faces, _, _ = measure.marching_cubes(field, level)
    g = np.stack(np.gradient(ni.gaussian_filter(field, grad_sigma)), -1)
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return tri, area, g


def sample(tri, area, spacing_vox):
    """Even-ish sampling: oversample by area, then keep one point per grid cell."""
    n = int(area.sum() / spacing_vox ** 2 * 6)
    f = rng.choice(len(tri), n, p=area / area.sum())
    a, b = rng.random(n), rng.random(n)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    p = tri[f, 0] + a[:, None] * (tri[f, 1] - tri[f, 0]) + b[:, None] * (tri[f, 2] - tri[f, 0])
    key = np.floor(p / (spacing_vox * 0.86)).astype(np.int64)
    key = key[:, 0] * 4000003 + key[:, 1] * 2003 + key[:, 2]
    _, keep = np.unique(key, return_index=True)
    return p[keep]


def normals(g, p):
    v = np.stack([ni.map_coordinates(g[..., i], p.T, order=1) for i in range(3)], -1)
    return -v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


hull_tri, hull_area, hull_grad = surface(ni.gaussian_filter(solid, 0.85), 0.5, 1.2)

# ---- cortex from the T1 slices, resampled to 0.5 mm isotropic; index order is (z, y, x)
Image.MAX_IMAGE_PIXELS = None
atlas = np.asarray(Image.open(os.path.join(OUT, "slices-hd.webp")))[..., 0]
vol = np.zeros((NZ, NY, NX), np.float32)
for k in range(NZ):
    ox, oy = (k % COLS) * TX + 1, (k // COLS) * TY + 1
    vol[k] = atlas[oy:oy + NY, ox:ox + NX] / 255.0
CP = 6
t1 = np.pad(ni.zoom(vol, (2, 1, 1), order=1), ((0, CP), (CP, CP), (CP, CP)))
cx_tri, cx_area, cx_grad = surface(ni.gaussian_filter(t1, 1.1), CORTEX_ISO, 1.0)


def build(spacing):
    # cortex
    p = sample(cx_tri, cx_area, spacing / 0.5)
    n = normals(cx_grad, p)[:, ::-1]
    mm = np.stack([(p[:, 2] - CP) * 0.5 + X0, (p[:, 1] - CP) * 0.5 + Y0, p[:, 0] * 0.5 + Z0], 1)
    d = ni.map_coordinates(depth_grid, (mm - org).T, order=1)
    keep = (mm[:, 2] >= SEAM) & (rng.random(len(mm)) < np.exp(-np.maximum(d - 2.5, 0) / 4.0)) & (d < 16)
    mm, n, d = mm[keep], n[keep], d[keep]
    lum = np.exp(-np.maximum(d - 1.0, 0) / 3.5)
    # mask hull below the seam
    hp = sample(hull_tri, hull_area, spacing)
    hmm = hp + org
    low = hmm[:, 2] < SEAM
    hp, hmm = hp[low], hmm[low]
    hn = normals(hull_grad, hp)
    cav = np.clip((ni.map_coordinates(wide, hp.T, order=1) - 0.30) / 0.40, 0, 1)
    hl = 0.78 * (1 - 0.7 * cav)
    mm, n, lum = np.concatenate([mm, hmm]), np.concatenate([n, hn]), np.concatenate([lum, hl])
    o = rng.permutation(len(mm))
    return mm[o], n[o], lum[o]


for name, spacing in TARGETS.items():
    mm, n, lum = build(spacing)
    with open(os.path.join(OUT, name), "wb") as fh:
        fh.write(np.uint32(len(mm)).tobytes())
        fh.write(np.round(mm * 100).astype("<i2").tobytes())
        fh.write(np.round(n * 127).astype(np.int8).tobytes())
        fh.write(np.round(np.clip(lum, 0, 1) * 255).astype(np.uint8).tobytes())
    print(f"{name}: {len(mm)} points, {4 + 10 * len(mm)} bytes")
