"""
Rib midline extraction from biplanar segmentations
"""
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage import measure
from skimage.morphology import skeletonize, binary_closing, disk


class RibMidlineExtractor:
    """
    Extract rib midlines from biplanar segmentations and reconstruct them in 3D
    """

    def __init__(self, coronal_segmentation_path, sagittal_segmentation_path):
        """
        Initialize the rib midline extractor with biplanar segmentations

        Args:
            coronal_segmentation_path: Path to the coronal view segmentation (nifti file)
            sagittal_segmentation_path: Path to the sagittal view segmentation (nifti file)
        """
        # Load segmentations
        coronal_data = nib.load(coronal_segmentation_path).get_fdata()
        sagittal_data = nib.load(sagittal_segmentation_path).get_fdata()

        # Ensure segmentations are binary
        coronal_binary = (coronal_data > 0).astype(np.uint8)
        sagittal_binary = (sagittal_data > 0).astype(np.uint8)

        # Rotate both views by 90 degrees (clockwise)
        print("Rotating coronal and sagittal views by 90 degrees clockwise...")
        rotated_coronal = np.zeros((coronal_binary.shape[1], coronal_binary.shape[0], coronal_binary.shape[2]),
                                   dtype=coronal_binary.dtype)
        rotated_sagittal = np.zeros((sagittal_binary.shape[1], sagittal_binary.shape[0], sagittal_binary.shape[2]),
                                    dtype=sagittal_binary.dtype)

        # Rotate each channel separately
        for i in range(coronal_binary.shape[2]):
            rotated_coronal[:, :, i] = np.rot90(coronal_binary[:, :, i], k=3)
            rotated_sagittal[:, :, i] = np.rot90(sagittal_binary[:, :, i], k=3)

        # Store the rotated arrays
        self.coronal_seg = rotated_coronal
        self.sagittal_seg = rotated_sagittal

        print(f"Coronal shape: {self.coronal_seg.shape}")
        print(f"Sagittal shape: {self.sagittal_seg.shape}")
        self.num_ribs = self.coronal_seg.shape[2]

        # Storage for midlines
        self.midlines = {}
        self.midlines_3d = {}

    def extract_midlines(self):
        """
        Extract midlines for all rib pairs in both views
        """
        for rib_idx in range(0, self.num_ribs):
            print(f"Processing rib pair {rib_idx + 1}...")

            # Extract midlines for coronal view
            coronal_left, coronal_right = self.extract_midlines_for_view(rib_idx, view='coronal')

            # Extract midlines for sagittal view
            sagittal_left, sagittal_right = self.extract_midlines_for_view(rib_idx, view='sagittal')

            # Store midlines
            self.midlines[(rib_idx, 'coronal', 'left')] = coronal_left
            self.midlines[(rib_idx, 'coronal', 'right')] = coronal_right
            self.midlines[(rib_idx, 'sagittal', 'left')] = sagittal_left
            self.midlines[(rib_idx, 'sagittal', 'right')] = sagittal_right

    def extract_midlines_for_view(self, rib_idx, view='coronal'):
        """
        Extract midlines for a rib pair in the specified view

        Args:
            rib_idx: Index of the rib pair
            view: 'coronal' or 'sagittal'

        Returns:
            left_midline, right_midline: Arrays of points representing the midlines
        """
        # Get the segmentation mask for this rib
        seg = self.coronal_seg[:, :, rib_idx] if view == 'coronal' else self.sagittal_seg[:, :, rib_idx]
        seg_binary = (seg > 0).astype(np.uint8)

        # Use connected components to separate ribs
        labeled = measure.label(seg_binary)
        regions = measure.regionprops(labeled)

        # Create masks for left and right sides
        left_mask = np.zeros_like(seg)
        right_mask = np.zeros_like(seg)

        if len(regions) >= 2:
            # Sort regions by their centroid's x-coordinate
            regions.sort(key=lambda r: r.centroid[1])
            if len(regions) >= 1:
                left_mask[labeled == regions[0].label] = 1  # Leftmost region to left mask
            if len(regions) >= 2:
                right_mask[labeled == regions[1].label] = 1  # Rightmost region to right mask

            left_mask = (left_mask > 0).astype(np.uint8)
            right_mask = (right_mask > 0).astype(np.uint8)
        elif len(regions) == 1:
            # If we only have one component in the sagittal view
            if view == 'sagittal':
                print(f"Found only one component in sagittal view for rib {rib_idx}, using it for both sides")
                seg_binary = (labeled == regions[0].label)
                left_mask = seg_binary
                right_mask = seg_binary
            else:
                left_mask[labeled == regions[0].label] = 1
                right_mask[labeled == regions[0].label] = 1

        # Apply morphological closing operation to fill gaps
        selem = disk(2)
        left_mask_closed = binary_closing(left_mask, selem)
        right_mask_closed = binary_closing(right_mask, selem)

        print(f"  Left mask before closing: {np.sum(left_mask)} pixels")
        print(f"  Left mask after closing: {np.sum(left_mask_closed)} pixels")
        print(f"  Right mask before closing: {np.sum(right_mask)} pixels")
        print(f"  Right mask after closing: {np.sum(right_mask_closed)} pixels")

        # Skeletonize
        left_mask_binary = left_mask_closed.astype(bool)
        right_mask_binary = right_mask_closed.astype(bool)

        print(f"Rib {rib_idx}, {view} view:")
        print(f"  Left mask: shape={left_mask_binary.shape}, non-zero pixels={np.sum(left_mask_binary)}")
        print(f"  Right mask: shape={right_mask_binary.shape}, non-zero pixels={np.sum(right_mask_binary)}")

        left_skeleton = skeletonize(left_mask_binary, method='lee')
        print(f"  Left skeleton: non-zero pixels={np.sum(left_skeleton)}")
        left_midline = left_skeleton

        right_skeleton = skeletonize(right_mask_binary, method='lee')
        print(f"  Right skeleton: non-zero pixels={np.sum(right_skeleton)}")
        right_midline = right_skeleton

        if view == 'sagittal' and len(regions) == 1:
            print(f"  Using horizontally flipped mask for right rib in sagittal view")

        return left_midline, right_midline

    def reconstruct_midlines_in_3d(self):
        """
        Reconstruct the 3D rib midlines from biplanar parametric curves
        """
        from rib_midline_parametrization import RibMidlineParametrization
        parametrizer = RibMidlineParametrization(self)
        parametrizer.process_all_midlines()

        print(f"After processing, found {len(parametrizer.parametric_midlines)} parametric midlines")

        # Count midlines
        coronal_left_count = sum(
            1 for k in parametrizer.parametric_midlines.keys() if k[1] == 'coronal' and k[2] == 'left')
        coronal_right_count = sum(
            1 for k in parametrizer.parametric_midlines.keys() if k[1] == 'coronal' and k[2] == 'right')
        sagittal_left_count = sum(
            1 for k in parametrizer.parametric_midlines.keys() if k[1] == 'sagittal' and k[2] == 'left')
        sagittal_right_count = sum(
            1 for k in parametrizer.parametric_midlines.keys() if k[1] == 'sagittal' and k[2] == 'right')

        print(f"Midline counts - Coronal left: {coronal_left_count}, Coronal right: {coronal_right_count}, "
              f"Sagittal left: {sagittal_left_count}, Sagittal right: {sagittal_right_count}")

        for rib_idx in range(0, self.num_ribs):
            print(f"Reconstructing 3D midline for rib pair {rib_idx + 1}...")

            # Get parametric midlines from both views
            coronal_left_key = (rib_idx, 'coronal', 'left')
            coronal_right_key = (rib_idx, 'coronal', 'right')
            sagittal_left_key = (rib_idx, 'sagittal', 'left')
            sagittal_right_key = (rib_idx, 'sagittal', 'right')

            coronal_left_param = parametrizer.parametric_midlines.get(coronal_left_key)
            coronal_right_param = parametrizer.parametric_midlines.get(coronal_right_key)
            sagittal_left_param = parametrizer.parametric_midlines.get(sagittal_left_key)
            sagittal_right_param = parametrizer.parametric_midlines.get(sagittal_right_key)

            # Try to reconstruct left rib
            if coronal_left_param is not None and sagittal_left_param is not None:
                print(f"  Attempting to reconstruct left rib {rib_idx + 1}")
                left_3d = self._direct_orthogonal_reconstruction(coronal_left_param, sagittal_left_param)

                if left_3d is not None and len(left_3d) > 0:
                    self.midlines_3d[(rib_idx, 'left')] = left_3d
                    print(f"  Stored left rib with {len(left_3d)} points")
                else:
                    print("  Failed to reconstruct left rib")
            else:
                print(f"  Skipping left rib {rib_idx + 1} due to missing parametric midlines")

            # Right rib reconstruction
            if coronal_right_param is not None and sagittal_right_param is not None:
                print(f"  Attempting to reconstruct right rib {rib_idx + 1}")
                right_3d = self._direct_orthogonal_reconstruction(coronal_right_param, sagittal_right_param)

                if right_3d is not None and len(right_3d) > 0:
                    self.midlines_3d[(rib_idx, 'right')] = right_3d
                    print(f"  Stored right rib with {len(right_3d)} points")
                else:
                    print("  Failed to reconstruct right rib")
            else:
                print(f"  Skipping right rib {rib_idx + 1} due to missing parametric midlines")

    def _direct_orthogonal_reconstruction(self, coronal_param, sagittal_param):
        """
        Reconstruct a single rib using orthogonal projection
        """
        if coronal_param is None or sagittal_param is None:
            print("  One of the parametric midlines is None")
            return None

        # Get sampled points from both parametric curves
        coronal_points = coronal_param.get('sampled_points')
        sagittal_points = sagittal_param.get('sampled_points')

        if coronal_points is None or sagittal_points is None:
            print("  Missing sampled points in parametric midlines")
            return None

        # Resample both curves with the same number of points
        num_points = 100
        coronal_t = np.linspace(0, 1, len(coronal_points))
        sagittal_t = np.linspace(0, 1, len(sagittal_points))

        new_t = np.linspace(0, 1, num_points)

        coronal_resampled = np.zeros((num_points, 2))
        sagittal_resampled = np.zeros((num_points, 2))

        for i in range(2):
            coronal_resampled[:, i] = np.interp(new_t, coronal_t, coronal_points[:, i])
            sagittal_resampled[:, i] = np.interp(new_t, sagittal_t, sagittal_points[:, i])

        # Create 3D points
        points_3d = np.zeros((num_points, 3))

        for i in range(num_points):
            points_3d[i, 0] = coronal_resampled[i, 1]  # x from coronal view's x
            points_3d[i, 1] = coronal_resampled[i, 0]  # y from coronal view's y
            points_3d[i, 2] = sagittal_resampled[i, 1]  # z from sagittal view's x

        return points_3d