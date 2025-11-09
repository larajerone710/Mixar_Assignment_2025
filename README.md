# Mesh Normalization, Quantization, and Error Analysis
*Prepared for Mixar SeamGPT data-processing assignment*

This Folder implements the assignment: **Mesh Normalization, Quantization, and Error Analysis** (Min–Max and Unit Sphere normalization; 1024-bin quantization; dequantize + denormalize; error metrics and plots).

## Contents
- `notebooks/Mesh_Normalize_Quantize_Error_Analysis.ipynb` — main Jupyter notebook with step-by-step cells, visualizations, and outputs.
- `scripts/mesh_preprocess.py` — command-line script to run end-to-end processing on a folder of `.obj` meshes.
- `meshes/` — put your `.obj` files here.
- `outputs/` — generated .ply reconstructions, error plots, and JSON summaries will be saved here.
- `requirements.txt` — Python dependencies.

## How to run (quick)
1. Create Python environment:
```bash
python -m venv venv
venv\Scripts\activate    
pip install -r requirements.txt
```

2. Place your `.obj` meshes in `meshes/`.

3. Run the processing script:
```bash
python scripts/mesh_preprocess.py
```

4. Check `outputs/` for:
- `*.recon_minmax.ply` and `*.recon_unitsphere.ply` (reconstructed meshes),
- `*.error_plot.png` (per-axis MSE/MAE),
- `*.summary.json` (metrics and metadata).

## References
- Mesh quantization examples and resources: tsherif/mesh-quantization-example and mesh-compression-examples.
- meshoptimizer (practical mesh processing tools).
- Quantized mesh / Cesium encoder examples.
- SeamGPT research (for assignment context).

## Bonus Task Implementation
- `scripts/bonus_adaptive_quant.py` implements Option 2 (Rotation/Translation invariance + Adaptive Quantization).
- Requires scikit-learn. Run: `pip install scikit-learn`.
- Execute: `python scripts/bonus_adaptive_quant.py` (with meshes/ populated).

## Author
Lara Jerone J
RA2211004010365
SRMIST

## contact
personal mail : larajerone710@gmail.com
college mail : lj8061@srmist.edu.in

## Mixar Assignment 2025 ##
