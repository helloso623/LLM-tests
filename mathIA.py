import trimesh as tm
import numpy as np

OBJ = r"C:\Bureau\IA.obj"

# --- inputs you control ---
TARGET_X_CM = 11.0        # width in X after scaling (cm)
TARGET_Z_CM = 11.5        # length in Z after scaling (cm)
H_BREAD_M   = 0.03        # toast thickness/height (m)

# 1) Load mesh
m = tm.load(OBJ, force="mesh")
m.remove_unreferenced_vertices()

# 2) Scale mesh so its bounding box matches your measured X and Z (in cm)
minb, maxb = m.bounding_box.bounds
dx, dy, dz = (maxb - minb)

S = np.eye(4)
S[0, 0] = TARGET_X_CM / dx   # sx
S[2, 2] = TARGET_Z_CM / dz   # sz
# keep Y scale unchanged:
# S[1,1] = 1.0

m.apply_transform(S)

# 3) Projected area on ZX plane (one-sided, along +Y)
#    This equals sum(face_area * max(0, n_y)) and is in cm^2 (because we scaled in cm)
areas = m.area_faces
ny = m.face_normals[:, 1]
A_zx_cm2 = float(np.sum(areas * np.maximum(0.0, ny)))

# 4) Convert area to m^2
A_zx_m2 = A_zx_cm2 * 1e-4  # 1 cm^2 = 1e-4 m^2

# 5) Volume = area * height
V_bread_m3 = A_zx_m2 * H_BREAD_M

print(f"Projected ZX area (one-sided) = {A_zx_cm2:.4f} cm^2")
print(f"Projected ZX area (one-sided) = {A_zx_m2:.8f} m^2")
print(f"Bread height H_bread          = {H_BREAD_M:.4f} m")
print(f"Bread volume V_bread          = {V_bread_m3:.8e} m^3")
