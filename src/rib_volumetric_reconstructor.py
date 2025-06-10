"""
Rib volumetric reconstruction from 3D midlines
"""
import numpy as np
import os
from rib_midline_parametrization import RibMidlineParametrization


class RibVolumetricReconstructor:
    """
    Generate volumetric rib models from 3D midlines
    """

    def __init__(self, rib_extractor):
        """
        Initialize the volumetric rib reconstructor

        Args:
            rib_extractor: RibMidlineExtractor instance with 3D midlines
        """
        self.rib_extractor = rib_extractor

        # Ensure midlines have been reconstructed
        if not hasattr(rib_extractor, 'midlines_3d') or not rib_extractor.midlines_3d:
            raise ValueError("No 3D midlines available. Run reconstruct_midlines_in_3d() first.")

        # Storage for volumetric data
        self.rib_cross_sections = {}
        self.rib_meshes = {}

        # Processing parameters
        self.num_midline_samples = 16
        self.num_cross_section_points = 8

        # Create output directory
        self.output_dir = "volumetric_visualization"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load original segmentation data
        self.coronal_seg = self.rib_extractor.coronal_seg
        self.sagittal_seg = self.rib_extractor.sagittal_seg

    def process_all_ribs(self):
        """
        Process all ribs: generate cross-sections and meshes for all available ribs
        """
        available_ribs = set()
        for key in self.rib_extractor.midlines_3d.keys():
            available_ribs.add(key)

        # Generate meshes for all ribs
        for rib_idx, side in available_ribs:
            print(f"Processing rib {rib_idx}, {side} side...")

            # Generate cross-sections
            print(f"  Generating cross-sections...")
            cross_sections = self.generate_all_cross_sections(rib_idx, side)
            if cross_sections is None:
                print(f"  Failed to generate cross-sections")
                continue

            # Generate mesh
            print(f"  Generating mesh...")
            vertices, faces = self.generate_mesh(rib_idx, side)
            if vertices is None or faces is None:
                print(f"  Failed to generate mesh")
                continue

        print("Processing complete")

    def generate_all_cross_sections(self, rib_idx, side):
        """
        Generate all cross-sections for a rib using both coronal and sagittal view measurements
        """
        # Measure cross-section dimensions
        cross_section_data = self.measure_cross_section_elliptical(rib_idx, side)
        if cross_section_data is None:
            return None

        # Generate cross-sections at each point along the midline
        cross_sections = []
        midline_points = cross_section_data['midline_points']

        # Global reference directions
        global_vertical = np.array([0, 1, 0])  # Y axis is vertical
        global_forward = np.array([0, 0, 1])   # Z axis is forward

        # Calculate tangents for all points
        tangents = np.zeros((self.num_midline_samples, 3))

        for i in range(1, self.num_midline_samples - 1):
            tangents[i] = midline_points[i + 1] - midline_points[i - 1]
            norm = np.linalg.norm(tangents[i])
            if norm > 1e-10:
                tangents[i] /= norm

        # Handle endpoints
        if self.num_midline_samples > 1:
            tangents[0] = midline_points[1] - midline_points[0]
            norm = np.linalg.norm(tangents[0])
            if norm > 1e-10:
                tangents[0] /= norm

            tangents[-1] = midline_points[-1] - midline_points[-2]
            norm = np.linalg.norm(tangents[-1])
            if norm > 1e-10:
                tangents[-1] /= norm

        for i in range(self.num_midline_samples):
            midline_point = midline_points[i]
            tangent = tangents[i]

            # Calculate anatomically oriented cross-section directions
            vertical_proj = global_vertical - np.dot(global_vertical, tangent) * tangent
            height_dir_norm = np.linalg.norm(vertical_proj)

            if height_dir_norm > 1e-4:
                height_dir = vertical_proj / height_dir_norm
                width_dir = np.cross(tangent, height_dir)
                width_dir = width_dir / np.linalg.norm(width_dir)
            else:
                forward_proj = global_forward - np.dot(global_forward, tangent) * tangent
                forward_norm = np.linalg.norm(forward_proj)

                if forward_norm > 1e-4:
                    width_dir = forward_proj / forward_norm
                    height_dir = np.cross(width_dir, tangent)
                    height_dir = height_dir / np.linalg.norm(height_dir)
                else:
                    if abs(tangent[0]) < abs(tangent[1]):
                        width_dir = np.array([1, 0, 0])
                    else:
                        width_dir = np.array([0, 1, 0])

                    width_dir = width_dir - np.dot(width_dir, tangent) * tangent
                    width_dir = width_dir / np.linalg.norm(width_dir)
                    height_dir = np.cross(width_dir, tangent)
                    height_dir = height_dir / np.linalg.norm(height_dir)

            # Get width and height measurements
            width = cross_section_data['total_widths'][i]
            height = cross_section_data['total_heights'][i]

            # Generate elliptical cross-section
            cross_section = self.generate_axis_aligned_elliptical_cross_section(
                midline_point, width_dir, height_dir, width, height
            )

            cross_sections.append(cross_section)

        # Store the cross-section data
        self.rib_cross_sections[(rib_idx, side)] = {
            'cross_section_data': cross_section_data,
            'cross_sections': cross_sections
        }

        return cross_sections

    def measure_cross_section_elliptical(self, rib_idx, side):
        """
        Measure cross-section dimensions from both coronal and sagittal views
        """
        # Get parametric midlines from both views
        parametrizer = RibMidlineParametrization(self.rib_extractor)
        parametrizer.process_all_midlines()

        coronal_midline_key = (rib_idx, 'coronal', side)
        sagittal_midline_key = (rib_idx, 'sagittal', side)

        coronal_param = parametrizer.parametric_midlines.get(coronal_midline_key)
        sagittal_param = parametrizer.parametric_midlines.get(sagittal_midline_key)

        if coronal_param is None or sagittal_param is None:
            print(f"Missing parametric midline for rib {rib_idx}, {side} side")
            return None

        # Get sampled points from parametric curves
        coronal_points = coronal_param.get('sampled_points')
        sagittal_points = sagittal_param.get('sampled_points')

        if coronal_points is None or sagittal_points is None:
            print(f"Missing sampled points for rib {rib_idx}, {side} side")
            return None

        # Resample to get evenly spaced points
        coronal_resampled = np.zeros((self.num_midline_samples, 2))
        sagittal_resampled = np.zeros((self.num_midline_samples, 2))

        coronal_t = np.linspace(0, 1, len(coronal_points))
        sagittal_t = np.linspace(0, 1, len(sagittal_points))
        new_t = np.linspace(0, 1, self.num_midline_samples)

        for i in range(2):
            coronal_resampled[:, i] = np.interp(new_t, coronal_t, coronal_points[:, i])
            sagittal_resampled[:, i] = np.interp(new_t, sagittal_t, sagittal_points[:, i])

        # Get segmentation masks
        coronal_seg = self.coronal_seg[:, :, rib_idx]
        sagittal_seg = self.sagittal_seg[:, :, rib_idx]

        # Ensure binary
        coronal_binary = (coronal_seg > 0).astype(np.uint8)
        sagittal_binary = (sagittal_seg > 0).astype(np.uint8)

        # Calculate 3D midline points
        midline_3d = np.zeros((self.num_midline_samples, 3))
        for i in range(self.num_midline_samples):
            midline_3d[i, 0] = coronal_resampled[i, 1]  # x from coronal view's x
            midline_3d[i, 1] = coronal_resampled[i, 0]  # y from coronal view's y
            midline_3d[i, 2] = sagittal_resampled[i, 1] # z from sagittal view's x

        # Measure widths and heights
        total_widths = self._measure_dimensions_from_view(coronal_resampled, coronal_binary)
        individual_heights = self._measure_dimensions_from_view(sagittal_resampled, sagittal_binary)

        # Apply scaling
        pixel_size_mm = 0.5
        scale_correction = 1.6

        total_widths *= pixel_size_mm * scale_correction
        individual_heights *= pixel_size_mm * scale_correction

        # Calculate average height
        valid_heights = individual_heights[individual_heights > 0]
        avg_height = np.mean(valid_heights) if len(valid_heights) > 0 else 1.0 * pixel_size_mm * scale_correction
        total_heights = np.full(self.num_midline_samples, avg_height)

        return {
            'midline_points': midline_3d,
            'total_widths': total_widths,
            'total_heights': total_heights,
            'avg_height': avg_height
        }

    def _measure_dimensions_from_view(self, resampled_points, binary_image):
        """
        Measure dimensions from a single view
        """
        dimensions = np.zeros(self.num_midline_samples)
        max_search = 50

        # Calculate tangents and normals
        tangents = np.zeros((self.num_midline_samples, 2))
        for i in range(1, self.num_midline_samples - 1):
            tangents[i] = resampled_points[i + 1] - resampled_points[i - 1]
            norm = np.linalg.norm(tangents[i])
            if norm > 1e-10:
                tangents[i] /= norm

        # Handle endpoints
        if self.num_midline_samples > 1:
            tangents[0] = resampled_points[1] - resampled_points[0]
            norm = np.linalg.norm(tangents[0])
            if norm > 1e-10:
                tangents[0] /= norm

            tangents[-1] = resampled_points[-1] - resampled_points[-2]
            norm = np.linalg.norm(tangents[-1])
            if norm > 1e-10:
                tangents[-1] /= norm

        # Calculate normals
        normals = np.zeros((self.num_midline_samples, 2))
        for i in range(self.num_midline_samples):
            normals[i] = np.array([-tangents[i, 1], tangents[i, 0]])

        # Measure dimensions
        for i in range(self.num_midline_samples):
            y, x = int(round(resampled_points[i, 0])), int(round(resampled_points[i, 1]))
            normal = normals[i]

            if 0 <= y < binary_image.shape[0] and 0 <= x < binary_image.shape[1]:
                # Measure outward
                outer_dist = 0
                for d in range(1, max_search):
                    ny = int(round(y + d * normal[0]))
                    nx = int(round(x + d * normal[1]))

                    if (ny < 0 or ny >= binary_image.shape[0] or
                            nx < 0 or nx >= binary_image.shape[1] or
                            binary_image[ny, nx] == 0):
                        outer_dist = d - 1
                        break

                # Measure inward
                inner_dist = 0
                for d in range(1, max_search):
                    ny = int(round(y - d * normal[0]))
                    nx = int(round(x - d * normal[1]))

                    if (ny < 0 or ny >= binary_image.shape[0] or
                            nx < 0 or nx >= binary_image.shape[1] or
                            binary_image[ny, nx] == 0):
                        inner_dist = d - 1
                        break

                dimensions[i] = inner_dist + outer_dist

        return dimensions

    def generate_axis_aligned_elliptical_cross_section(self, midline_point, width_direction, height_direction,
                                                       width, height):
        """
        Generate points for an elliptical cross-section
        """
        # Minimum dimensions
        min_size = 1.0
        width = max(width, min_size)
        height = max(height, min_size)

        # Create elliptical cross-section
        theta = np.linspace(0, 2 * np.pi, self.num_cross_section_points, endpoint=False)

        # Semi-axes
        a = width / 2
        b = height / 2

        # Generate ellipse points
        cross_section_points = np.zeros((self.num_cross_section_points, 3))

        for i, angle in enumerate(theta):
            x_offset = a * np.cos(angle)
            z_offset = b * np.sin(angle)

            cross_section_points[i] = midline_point + x_offset * width_direction + z_offset * height_direction

        return cross_section_points

    def generate_mesh(self, rib_idx, side):
        """
        Generate a triangular mesh for a rib from its cross-sections
        """
        # Get or generate cross-sections
        if (rib_idx, side) not in self.rib_cross_sections:
            cross_sections = self.generate_all_cross_sections(rib_idx, side)
            if cross_sections is None:
                return None, None
        else:
            cross_sections = self.rib_cross_sections[(rib_idx, side)]['cross_sections']

        n_sections = len(cross_sections)
        n_points = self.num_cross_section_points

        if n_sections < 2:
            print(f"Not enough cross-sections to generate mesh for rib {rib_idx}, {side} side")
            return None, None

        # Create tapered end for anterior (front) end
        taper_sections = 2
        if n_sections > taper_sections:
            tapered_cross_sections = [cs.copy() for cs in cross_sections]
            last_center = np.mean(cross_sections[-1], axis=0)

            for i in range(taper_sections):
                reverse_i = n_sections - i - 1
                taper_factor = 0.4 + (0.6 * i / taper_sections)
                for j in range(n_points):
                    vector = cross_sections[reverse_i][j] - last_center
                    tapered_cross_sections[reverse_i][j] = last_center + vector * taper_factor

            cross_sections = tapered_cross_sections

        # Total vertices
        n_vertices = n_sections * n_points + 2

        # Initialize vertex array
        vertices = np.zeros((n_vertices, 3))

        # Add cross-section points
        for i in range(n_sections):
            for j in range(n_points):
                vertex_idx = i * n_points + j
                vertices[vertex_idx] = cross_sections[i][j]

        # Add end cap centers
        vertices[-2] = np.mean(cross_sections[0], axis=0)   # First end cap center
        vertices[-1] = np.mean(cross_sections[-1], axis=0)  # Last end cap center

        # Create triangular faces
        faces = []

        # Faces between adjacent cross-sections
        for i in range(n_sections - 1):
            for j in range(n_points):
                p00 = i * n_points + j
                p01 = i * n_points + ((j + 1) % n_points)
                p10 = (i + 1) * n_points + j
                p11 = (i + 1) * n_points + ((j + 1) % n_points)

                faces.append([p00, p01, p11])
                faces.append([p00, p11, p10])

        # Add end cap faces
        for j in range(n_points):
            # First end cap
            p0 = j
            p1 = (j + 1) % n_points
            faces.append([p0, p1, n_vertices - 2])

            # Last end cap
            p0 = (n_sections - 1) * n_points + j
            p1 = (n_sections - 1) * n_points + ((j + 1) % n_points)
            faces.append([p0, n_vertices - 1, p1])

        faces = np.array(faces, dtype=int)

        # Store the mesh
        self.rib_meshes[(rib_idx, side)] = {
            'vertices': vertices,
            'faces': faces
        }

        return vertices, faces