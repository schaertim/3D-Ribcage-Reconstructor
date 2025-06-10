"""
Main script for rib reconstruction pipeline
"""
from rib_midline_extractor import RibMidlineExtractor
from rib_volumetric_reconstructor import RibVolumetricReconstructor
from rib_visualizer import RibVisualizer
import matplotlib

matplotlib.use("TkAgg")

# Enable debug output
debug_mode = True


def main():
    """
    Main function to process rib segmentations, extract 3D rib midlines,
    and generate volumetric reconstructions
    """
    # Paths to your segmentation nifti files
    coronal_seg_path = "S:/rib_segmentation/reconstruction/data/segmentations/coronal/eos_drr_1.nii.gz"
    sagittal_seg_path = "S:/rib_segmentation/reconstruction/data/segmentations/sagittal/eos_drr_1.nii.gz"

    # Step 1: Create rib midline extractor and extract raw midlines
    print("\nCreating rib midline extractor...")
    extractor = RibMidlineExtractor(coronal_seg_path, sagittal_seg_path)

    print("Extracting midlines...")
    extractor.extract_midlines()

    # Step 2: Reconstruct midlines in 3D
    print("\nReconstructing midlines in 3D...")
    extractor.reconstruct_midlines_in_3d()

    # Step 3: Create volumetric reconstructor
    print("\nCreating volumetric reconstructor...")
    reconstructor = RibVolumetricReconstructor(extractor)

    # Step 4: Process individual ribs (for detailed visualization)
    sample_ribs = [(7, 'left')]
    for rib_idx, side in sample_ribs:
        if (rib_idx, side) in extractor.midlines_3d:
            print(f"\nProcessing rib {rib_idx}, {side} side for detailed visualization...")
            reconstructor.generate_all_cross_sections(rib_idx, side)
            reconstructor.generate_mesh(rib_idx, side)

    # Step 5: Process all ribs
    print("\nProcessing all ribs...")
    reconstructor.process_all_ribs()

    # Step 6: Create visualizations
    print("\nCreating visualizations...")
    visualizer = RibVisualizer(extractor, reconstructor)

    # Create all visualizations
    visualizer.visualize_midlines()
    visualizer.visualize_all_rib_parametrizations()
    visualizer.visualize_3d_midlines()
    visualizer.visualize_all_meshes_pyvista()


if __name__ == "__main__":
    main()