# Rib Reconstruction Pipeline

This repository contains a comprehensive pipeline for reconstructing 3D volumetric rib models from biplanar medical segmentations. The pipeline processes coronal and sagittal view segmentations to extract 2D midlines, create parametric curve representations, reconstruct 3D midlines, and generate detailed volumetric meshes.

## Repository Structure

```
├── segmentations/
│   ├── coronal/                       # Contains coronal view segmentation files
│   └── sagittal/                      # Contains sagittal view segmentation files
├── src/
│   ├── process_ribs.py                # Main execution script
│   ├── rib_midline_extractor.py       # 2D midline extraction and 3D reconstruction
│   ├── rib_midline_parametrization.py # Parametric curve generation
│   ├── rib_volumetric_reconstructor.py # Volumetric modeling and mesh generation
│   └── rib_visualizer.py              # Comprehensive visualization methods
├── LICENSE
├── README.md
└── requirements.txt                    # Package dependencies
```

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies:
- **numpy**: For numerical computations and array operations
- **scipy**: For scientific computing and spline interpolation
- **matplotlib**: For visualization and plotting
- **nibabel**: For loading NIfTI medical image files
- **scikit-image**: For image processing and morphological operations
- **pyvista**: For advanced 3D visualization (optional but recommended)

## Features

### Midline Extraction
The `rib_midline_extractor.py` module provides:
- Loading and preprocessing of biplanar NIfTI segmentations
- Automatic rotation and orientation correction
- Morphological operations for improved connectivity
- Skeletonization to extract centerlines
- 3D reconstruction using orthogonal projection geometry

### Parametric Curve Generation
The `rib_midline_parametrization.py` module enables:
- Conversion of skeletal midlines to ordered point sequences
- Spline-based parametric curve fitting with adaptive smoothing
- Robust handling of sparse or noisy midline data
- Consistent parameterization for biplanar correspondence

### Volumetric Reconstruction
The `rib_volumetric_reconstructor.py` module supports:
- Cross-sectional dimension measurement from biplanar views
- Anatomically oriented elliptical cross-section generation
- Triangular mesh creation with anterior tapering
- Comprehensive volumetric model generation

### Visualization Suite
The `rib_visualizer.py` module includes:
- 2D midline visualization on segmentation backgrounds
- Parametric curve visualization with control points
- Interactive 3D midline and mesh visualization
- Cross-sectional measurement methodology display

## Usage

### Running the Complete Pipeline

1. Update the file paths in `process_ribs.py` to point to your segmentation data:
   ```python
   coronal_seg_path = "path/to/your/coronal_segmentation.nii.gz"
   sagittal_seg_path = "path/to/your/sagittal_segmentation.nii.gz"
   ```

2. Run the complete pipeline:
   ```bash
   python process_ribs.py
   ```

The pipeline will:
1. Extract 2D midlines from biplanar segmentations
2. Create parametric curve representations
3. Reconstruct 3D midlines using orthogonal geometry
4. Generate volumetric models with cross-sections and meshes
5. Create comprehensive visualizations

## Notes

- The pipeline expects NIfTI format (.nii.gz) binary segmentations
- Input segmentations should have matching rib indices between coronal and sagittal views
- PyVista is optional but recommended for 3D visualizations
- The pipeline assumes orthogonal biplanar geometry

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.