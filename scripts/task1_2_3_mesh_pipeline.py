import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
import json

BINS = 1024

def load_mesh(path):
    return trimesh.load(path, force='mesh')

def min_max_normalize(vertices):
    v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
    normalized = (vertices - v_min) / (v_max - v_min)
    return normalized, v_min, v_max

def unit_sphere_normalize(vertices):
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    scale = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / scale
    return normalized, centroid, scale

def quantize(normalized, bins=BINS):
    return np.floor(normalized * (bins - 1)).astype(int)

def dequantize(q, bins=BINS):
    return q / (bins - 1)

def process_mesh(path):
    mesh = load_mesh(path)
    vertices = mesh.vertices
    name = Path(path).stem
    os.makedirs('outputs', exist_ok=True)

    # MinMax
    norm, vmin, vmax = min_max_normalize(vertices)
    q = quantize(norm)
    dq = dequantize(q)
    recon = dq * (vmax - vmin) + vmin
    mse = np.mean((vertices - recon)**2)
    print(f"{name} MinMax MSE: {mse}")

    # UnitSphere
    norm2, c, s = unit_sphere_normalize(vertices)
    q2 = quantize((norm2 - norm2.min())/(norm2.max()-norm2.min()))
    dq2 = dequantize(q2)
    recon2 = dq2 * s + c
    mse2 = np.mean((vertices - recon2)**2)
    print(f"{name} UnitSphere MSE: {mse2}")

    plt.bar(['MinMax','UnitSphere'], [mse,mse2])
    plt.ylabel('MSE')
    plt.title(f'Error Comparison for {name}')
    plt.savefig(f'outputs/{name}_error.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    for f in Path('meshes').glob('*.obj'):
        process_mesh(f)
