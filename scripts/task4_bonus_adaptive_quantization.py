"""
Bonus Task: Rotation & Translation Invariance + Adaptive Quantization
Option 2 implementation for Mixar assignment.

"""
import os
import numpy as np
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.neighbors import NearestNeighbors

# Configuration
BINS_UNIFORM = 1024
ADAPTIVE_MIN = 256
ADAPTIVE_MAX = 4096
N_TRANSFORMS = 6   # number of random rotations/translations to generate
K_NEIGHBORS = 8    # for local density estimation
OUTPUT_DIR = "outputs/bonus_adaptive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mesh(path):
    mesh = trimesh.load(path, force='mesh')
    if mesh.vertices is None:
        raise ValueError("Mesh has no vertices")
    return mesh

def unit_sphere_normalize(vertices):
    v = np.asarray(vertices)
    centroid = v.mean(axis=0)
    centered = v - centroid
    max_dist = np.linalg.norm(centered, axis=1).max()
    scale = max_dist if max_dist != 0 else 1.0
    normalized = centered / scale
    meta = {"centroid": centroid.tolist(), "scale": float(scale)}
    return normalized, meta

def unit_sphere_denormalize(normalized, meta):
    centroid = np.array(meta["centroid"])
    scale = float(meta["scale"])
    return np.asarray(normalized) * scale + centroid

def apply_random_transform(vertices):
    # Random rotation (uniform using axis-angle)
    theta = np.random.uniform(0, 2*np.pi)
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    ux, uy, uz = axis
    cos_t = np.cos(theta); sin_t = np.sin(theta)
    R = np.array([
        [cos_t + ux*ux*(1-cos_t),    ux*uy*(1-cos_t)-uz*sin_t, ux*uz*(1-cos_t)+uy*sin_t],
        [uy*ux*(1-cos_t)+uz*sin_t,   cos_t+uy*uy*(1-cos_t),    uy*uz*(1-cos_t)-ux*sin_t],
        [uz*ux*(1-cos_t)-uy*sin_t,   uz*uy*(1-cos_t)+ux*sin_t, cos_t+uz*uz*(1-cos_t)]
    ])
    rotated = vertices.dot(R.T)
    # Random translation
    t = np.random.normal(scale=0.5, size=3)
    translated = rotated + t
    return translated, {"R": R.tolist(), "t": t.tolist()}

def estimate_local_density(vertices, k=K_NEIGHBORS):
    # Use average distance to k nearest neighbors as local scale; density ~ 1/avg_dist
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(vertices)), algorithm='auto').fit(vertices)
    distances, _ = nbrs.kneighbors(vertices)
    # distances[:,0] is zero (self), so take from 1..k
    avg = distances[:, 1:].mean(axis=1)
    # density avoid division by zero
    density = 1.0 / (avg + 1e-12)
    return density, avg

def adaptive_bin_assignment(density, bins_min=ADAPTIVE_MIN, bins_max=ADAPTIVE_MAX, n_clusters=8):
    # Map density to bin sizes: higher density -> more bins (finer quantization)
    # We'll cluster the density values into n_clusters via simple quantile-based buckets
    q = np.quantile(density, np.linspace(0,1,n_clusters+1))
    labels = np.digitize(density, q[1:-1], right=True)  # labels in 0..n_clusters-1
    # Map labels to bin sizes linearly between bins_min..bins_max (higher label -> higher density -> more bins)
    bin_sizes = np.linspace(bins_min, bins_max, n_clusters).astype(int)
    per_vertex_bins = bin_sizes[labels]
    return per_vertex_bins, labels

def quantize_uniform(normalized, bins=BINS_UNIFORM):
    # normalized assumed in [-1,1] or [0,1]; map to [0,1] first
    mn, mx = normalized.min(), normalized.max()
    mapped = (normalized - mn) / (mx - mn) if mx>mn else normalized - mn
    q = np.floor(mapped * (bins - 1)).astype(int)
    mapping_meta = {"mapped": True, "orig_min": float(mn), "orig_max": float(mx)}
    return q, mapping_meta

def dequantize_uniform(q, mapping_meta, bins=BINS_UNIFORM):
    deq = q.astype(float) / (bins - 1)
    if mapping_meta.get("mapped"):
        mn = mapping_meta["orig_min"]; mx = mapping_meta["orig_max"]
        return deq * (mx - mn) + mn
    return deq

def quantize_adaptive(normalized, per_vertex_bins):
    # normalized -> map global to [0,1] based on global min/max for stability, then quantize each vertex with its bins
    mn, mx = normalized.min(), normalized.max()
    mapped = (normalized - mn) / (mx - mn) if mx>mn else normalized - mn
    q = np.zeros_like(mapped, dtype=int)
    # per_vertex_bins is length V; quantize each vertex differently
    for i in range(mapped.shape[0]):
        b = per_vertex_bins[i]
        q[i] = np.floor(mapped[i] * (b - 1)).astype(int)
        q[i] = np.clip(q[i], 0, b - 1)
    mapping_meta = {"mapped": True, "orig_min": float(mn), "orig_max": float(mx), "per_vertex_bins": per_vertex_bins.tolist()}
    return q, mapping_meta

def dequantize_adaptive(q, mapping_meta):
    mn = mapping_meta["orig_min"]; mx = mapping_meta["orig_max"]
    per_vertex_bins = np.array(mapping_meta["per_vertex_bins"], dtype=int)
    deq = np.zeros_like(q, dtype=float)
    for i in range(q.shape[0]):
        b = per_vertex_bins[i]
        deq[i] = q[i].astype(float) / (b - 1)
    # map back to original normalized range
    return deq * (mx - mn) + mn

def compute_mse(orig, recon):
    return float(np.mean((orig - recon)**2))

def process_with_transforms(mesh_path):
    mesh = load_mesh(mesh_path)
    V = mesh.vertices.copy()
    name = Path(mesh_path).stem
    results = {"name": name, "transforms": []}

    # Prepare original normalization baseline (no transform)
    baseline_norm, baseline_meta = unit_sphere_normalize(V)

    for i in range(N_TRANSFORMS):
        Vt, trans_meta = apply_random_transform(V)
        norm, meta = unit_sphere_normalize(Vt)  # unit-sphere is translation/rotation invariant in behavior
        # Estimate density on the normalized vertices space
        density, avgd = estimate_local_density(norm, k=K_NEIGHBORS)
        per_vertex_bins, labels = adaptive_bin_assignment(density, bins_min=ADAPTIVE_MIN, bins_max=ADAPTIVE_MAX, n_clusters=8)

        # Uniform quantization path
        q_uniform, mapping_u = quantize_uniform(norm, bins=BINS_UNIFORM)
        deq_u = dequantize_uniform(q_uniform, mapping_u, bins=BINS_UNIFORM)
        recon_u = unit_sphere_denormalize(deq_u, meta)
        mse_u = compute_mse(Vt, recon_u)

        # Adaptive quantization path
        q_adapt, mapping_a = quantize_adaptive(norm, per_vertex_bins)
        deq_a = dequantize_adaptive(q_adapt, mapping_a)
        recon_a = unit_sphere_denormalize(deq_a, meta)
        mse_a = compute_mse(Vt, recon_a)

        results["transforms"].append({
            "transform_index": i,
            "transform_meta": trans_meta,
            "unit_sphere_meta": meta,
            "density_mean": float(density.mean()),
            "density_std": float(density.std()),
            "adaptive_bins_stats": {
                "min": int(per_vertex_bins.min()), "max": int(per_vertex_bins.max()), "mean": float(per_vertex_bins.mean())
            },
            "mse_uniform": mse_u,
            "mse_adaptive": mse_a
        })
        print(f"[Transform {i}] MSE uniform: {mse_u:.6g}, adaptive: {mse_a:.6g}")

    # Save summary JSON
    outp = Path(OUTPUT_DIR) / f"{name}_adaptive_summary.json"
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    # Plot comparison across transforms
    mses_u = [t["mse_uniform"] for t in results["transforms"]]
    mses_a = [t["mse_adaptive"] for t in results["transforms"]]
    idx = np.arange(len(mses_u))
    plt.figure(figsize=(8,4))
    plt.plot(idx, mses_u, marker='o', label='Uniform (1024 bins)')
    plt.plot(idx, mses_a, marker='o', label='Adaptive (256-4096)')
    plt.xlabel('Transform index')
    plt.ylabel('MSE')
    plt.title(f'Uniform vs Adaptive Quantization MSE for {name} across transforms')
    plt.legend()
    plotp = Path(OUTPUT_DIR) / f"{name}_adaptive_mse_plot.png"
    plt.savefig(plotp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary to {outp} and plot to {plotp}")
    return results

if __name__ == "__main__":
    import sys
    mesh_files = list(Path("meshes").glob("*.obj"))
    if len(mesh_files) == 0:
        print("Place .obj files in meshes/ and run again.")
        sys.exit(0)
    # For demo, process first mesh
    res = process_with_transforms(str(mesh_files[0]))
