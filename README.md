# 🧠 Mesh Normalization, Quantization, and Error Analysis  
**Prepared for Mixar SeamGPT Data-Processing Assignment (2025)**  
**Author:** Lara Jerone J (RA2211004010365)  
**Institution:** SRM Institute of Science and Technology (SRMIST)

---

## 📘 Overview

This repository implements the **Mixar SeamGPT data-processing assignment**, focusing on the **normalization, quantization, and error analysis of 3D mesh data**.  
It prepares raw `.obj` meshes for AI-based mesh understanding systems like **SeamGPT** by applying preprocessing steps that ensure data consistency and minimize reconstruction errors.

---

## 🎯 Objective

Before a 3D mesh can be processed by an AI model, its vertex data must be:
1. **Normalized** — scaled into a common coordinate range.  
2. **Quantized** — discretized into bins for compression and consistency.  
3. **Reconstructed & evaluated** — compared against the original mesh to measure precision loss.  
4. **(Bonus)** Made **rotation- and translation-invariant** with adaptive quantization for improved accuracy.

---

## 📂 Repository Structure

```
MIXAR-MESH-PREPROCESS-UPDATED/
│
├── scripts/
│   ├── task1_2_3_mesh_pipeline.py           # Implements Tasks 1–3
│   └── task4_bonus_adaptive_quantization.py # Implements Bonus Task (Option 2)
│
├── meshes/        # Place your input .obj meshes here
├── outputs/       # All generated results (plots, .ply files, JSON)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧩 Task Breakdown

### 🧠 **Task 1: Load and Inspect the Mesh**
**Goal:** Understand the 3D mesh structure and extract vertex information.

**Process:**
- Load `.obj` files using the **Trimesh** library.
- Extract vertex coordinates (`x, y, z`).
- Print mesh statistics:
  - Number of vertices
  - Min, Max, Mean, and Standard Deviation per axis

**Expected Output Example:**
```
Vertices: 3450
Min: [-1.0, -0.5, -0.9]
Max: [1.2, 0.6, 0.9]
Mean: [0.02, -0.01, 0.05]
Std: [0.34, 0.29, 0.27]
```

---

### ⚖️ **Task 2: Normalize and Quantize the Mesh**
**Goal:** Bring all vertex coordinates into a consistent numerical range and discretize them.

**Methods Implemented:**
1. **Min–Max Normalization:**  
   Maps coordinates to `[0, 1]` using:  
   `x' = (x – min) / (max – min)`
2. **Unit Sphere Normalization:**  
   Centers the mesh and scales it so all vertices fit within a sphere of radius 1.

**Quantization:**
- Each normalized coordinate is quantized into **1024 bins**.
- Formula: `q = floor(x' × (bins - 1))`

**Outputs:**
- Normalized and quantized meshes (`.ply` or `.obj`)
- Visualization plots for both normalization methods

---

### 🔄 **Task 3: Dequantize, Denormalize, and Error Analysis**
**Goal:** Measure accuracy loss after quantization and normalization.

**Process:**
- Dequantize: `x'' = q / (bins - 1)`
- Denormalize: Convert back to original coordinate scale.
- Compute reconstruction errors using:
  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)**

**Visualizations:**
- Error bar plots comparing **Min–Max** vs **Unit Sphere** normalization.
- Reconstructed meshes to visually verify structure preservation.

**Expected Deliverables:**
- `.error_plot.png`
- `.summary.json`
- Reconstructed `.ply` files

---

### 💎 **Bonus Task (Option 2): Rotation & Translation Invariance + Adaptive Quantization**
**Goal:**  
Develop a preprocessing method that is robust to 3D transformations and uses **adaptive quantization** based on mesh density.

**Implemented In:**  
`scripts/task4_bonus_adaptive_quantization.py`

**Key Features:**
- Generates multiple **randomly rotated and translated** mesh versions.
- Applies **Unit Sphere normalization** (removes translation and rotation effects).
- Estimates local vertex density using **k-nearest neighbors (k=8)**.
- Assigns **adaptive bin sizes** (from 256 to 4096) — smaller bins for dense areas, larger for sparse.
- Compares reconstruction errors between:
  - **Uniform Quantization (1024 bins)**
  - **Adaptive Quantization (256–4096 bins)**

**Outputs:**
- `*_adaptive_summary.json` — JSON results per transform.
- `*_adaptive_mse_plot.png` — MSE comparison between uniform vs adaptive quantization.

**Results Observation:**
Adaptive quantization produces **lower reconstruction error** and maintains mesh detail better in dense vertex regions.

---

## ⚙️ How to Run

### Step 1: Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Place Input Meshes
Copy your `.obj` files into the `meshes/` folder.

### Step 3: Run Tasks 1–3
```bash
python scripts/task1_2_3_mesh_pipeline.py
```

### Step 4: Run Bonus Task
```bash
python scripts/task4_bonus_adaptive_quantization.py
```

### Step 5: View Results
- Outputs will appear in the `outputs/` folder:
  - `.error_plot.png`
  - `.summary.json`
  - `.ply` reconstructed meshes
  - `bonus_adaptive/*.png` plots for adaptive quantization comparison

---

## 🧰 Libraries Used
| Library | Purpose |
|----------|----------|
| **NumPy** | Array operations and numerical calculations |
| **Trimesh** | 3D mesh loading and vertex manipulation |
| **Matplotlib** | Generating visual plots (error comparisons) |
| **Open3D** | Optional: 3D visualization of meshes |
| **scikit-learn** | Nearest neighbor density estimation for adaptive quantization |

Install them using:
```bash
pip install -r requirements.txt
```

---

## 📊 Output Example (Expected)
| File | Description |
|------|--------------|
| `cube.error_plot.png` | MSE comparison between Min–Max and Unit Sphere |
| `cube.summary.json` | Contains MSE/MAE values |
| `cube.recon_minmax.ply` | Reconstructed mesh (Min–Max method) |
| `bonus_adaptive/cube_adaptive_mse_plot.png` | Comparison of uniform vs adaptive MSE |

---

## 🧾 References
- [tsherif/mesh-quantization-example](https://github.com/tsherif/mesh-quantization-example)
- [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer)
- [Cesium Quantized Mesh Encoder](https://github.com/CesiumGS/quantized-mesh)
- **SeamGPT research paper** — for context on mesh understanding and preprocessing pipelines.

---

## 👩🏻‍💻 Author Information
**Name:** Lara Jerone J  
**Register Number:** RA2211004010365  
**College:** SRM Institute of Science and Technology (SRMIST), Kattankulathur  
**Course:** B.Tech – Electronics and Communication Engineering  
**Batch:** 2022–2026  

📧 **Email:**  
- Personal – [larajerone710@gmail.com](mailto:larajerone710@gmail.com)  
- College – [lj8061@srmist.edu.in](mailto:lj8061@srmist.edu.in)  

🔗 **Profiles:**  
- [GitHub](https://github.com/larajerone710)  
- [LinkedIn](https://www.linkedin.com/in/lara-jerone-j-62604124b)

---

## 🏁 Conclusion
This repository demonstrates a **complete 3D mesh preprocessing pipeline** designed for AI-based systems like SeamGPT.  
All tasks — **Loading, Normalization, Quantization, Error Analysis**, and the **Bonus Adaptive Quantization** — are implemented, tested, and documented for submission.

✅ Tasks 1–3 Completed  
✅ Bonus Task (Option 2) Completed  
✅ Output files generated  
✅ Code tested on multiple `.obj` meshes  

