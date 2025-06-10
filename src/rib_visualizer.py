"""
Rib visualization methods
"""
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from rib_midline_parametrization import RibMidlineParametrization


class RibVisualizer:
    """
    Visualization methods for rib reconstruction results
    """

    def __init__(self, rib_extractor, volumetric_reconstructor=None):
        """
        Initialize visualizer

        Args:
            rib_extractor: RibMidlineExtractor instance
            volumetric_reconstructor: Optional RibVolumetricReconstructor instance
        """
        self.rib_extractor = rib_extractor
        self.volumetric_reconstructor = volumetric_reconstructor

    def visualize_midlines(self, rib_indices=None):
        """
        Visualize the extracted midlines with two views: coronal and sagittal
        """
        if not self.rib_extractor.midlines:
            print("No midlines extracted yet. Run extract_midlines() first.")
            return

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')

        # If no indices specified, use all ribs
        if rib_indices is None:
            rib_indices = range(0, self.rib_extractor.num_ribs)

        # Colors for different ribs
        colors = plt.cm.plasma(np.linspace(0, 1, self.rib_extractor.num_ribs))

        # Show segmentation as background
        coronal_bg = np.sum(self.rib_extractor.coronal_seg, axis=2)
        coronal_bg_display = np.where(coronal_bg == 0, 1.0, 0.3 - (coronal_bg / np.max(coronal_bg) * 0.3))
        axes[0].imshow(coronal_bg_display, cmap='gray', vmin=0, vmax=1)

        sagittal_bg = np.sum(self.rib_extractor.sagittal_seg, axis=2)
        sagittal_bg_display = np.where(sagittal_bg == 0, 1.0, 0.3 - (sagittal_bg / np.max(sagittal_bg) * 0.3))
        axes[1].imshow(sagittal_bg_display, cmap='gray', vmin=0, vmax=1)

        # Plot midlines for each rib in reverse order
        for rib_idx in reversed(list(rib_indices)):
            # Coronal view midlines
            coronal_left = self.rib_extractor.midlines.get((rib_idx, 'coronal', 'left'))
            coronal_right = self.rib_extractor.midlines.get((rib_idx, 'coronal', 'right'))

            # Sagittal view midlines
            sagittal_left = self.rib_extractor.midlines.get((rib_idx, 'sagittal', 'left'))
            sagittal_right = self.rib_extractor.midlines.get((rib_idx, 'sagittal', 'right'))

            # Plot coronal view midlines
            if coronal_left is not None:
                y_coords, x_coords = np.where(coronal_left)
                axes[0].scatter(x_coords, y_coords, s=1, color=colors[rib_idx])
            if coronal_right is not None:
                y_coords, x_coords = np.where(coronal_right)
                axes[0].scatter(x_coords, y_coords, s=1, color=colors[rib_idx], marker='x')

            # Plot sagittal view midlines
            if sagittal_left is not None:
                y_coords, x_coords = np.where(sagittal_left)
                axes[1].scatter(x_coords, y_coords, s=1, color=colors[rib_idx])

            if sagittal_right is not None:
                y_coords, x_coords = np.where(sagittal_right)
                axes[1].scatter(x_coords, y_coords, s=1, color=colors[rib_idx], marker='x')

        # Set titles and labels
        axes[0].set_title("Coronal View", color='black')
        axes[0].set_xlabel("X", color='black')
        axes[0].set_ylabel("Y", color='black')
        axes[0].tick_params(colors='black')

        axes[1].set_title("Sagittal View", color='black')
        axes[1].set_xlabel("Z", color='black')
        axes[1].set_ylabel("Y", color='black')
        axes[1].tick_params(colors='black')

        # Add legend
        legend_handles = []
        legend_labels = []
        for rib_idx in rib_indices:
            has_midlines = any([
                self.rib_extractor.midlines.get((rib_idx, 'coronal', 'left')) is not None,
                self.rib_extractor.midlines.get((rib_idx, 'coronal', 'right')) is not None,
                self.rib_extractor.midlines.get((rib_idx, 'sagittal', 'left')) is not None,
                self.rib_extractor.midlines.get((rib_idx, 'sagittal', 'right')) is not None
            ])
            if has_midlines:
                handle = plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=colors[rib_idx], markersize=8)
                legend_handles.append(handle)
                legend_labels.append(f"Rib {rib_idx + 1}")

        if legend_handles:
            leg = fig.legend(legend_handles, legend_labels, loc='upper center',
                             bbox_to_anchor=(0.5, 0), ncol=min(6, len(legend_handles)))
            for text in leg.get_texts():
                text.set_color('black')

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def visualize_all_rib_parametrizations(self):
        """
        Visualize all rib parametrizations showing matching points between coronal and sagittal views
        """
        # Create parametrizer
        parametrizer = RibMidlineParametrization(self.rib_extractor)
        parametrizer.process_all_midlines()

        # Get all available ribs
        available_ribs = set()
        for key in parametrizer.parametric_midlines.keys():
            available_ribs.add(key[0])

        available_ribs = sorted(list(available_ribs))
        if not available_ribs:
            print("No ribs available for visualization")
            return

        print(f"Available ribs: {available_ribs}")

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')

        # Show segmentation backgrounds
        coronal_bg = np.sum(self.rib_extractor.coronal_seg, axis=2)
        coronal_bg_display = np.where(coronal_bg == 0, 1.0, 0.3 - (coronal_bg / np.max(coronal_bg) * 0.3))
        axes[0].imshow(coronal_bg_display, cmap='gray', vmin=0, vmax=1)

        sagittal_bg = np.sum(self.rib_extractor.sagittal_seg, axis=2)
        sagittal_bg_display = np.where(sagittal_bg == 0, 1.0, 0.3 - (sagittal_bg / np.max(sagittal_bg) * 0.3))
        axes[1].imshow(sagittal_bg_display, cmap='gray', vmin=0, vmax=1)

        # Colors for different ribs
        colors = plt.cm.plasma(np.linspace(0, 1, len(available_ribs)))

        # Plot all ribs
        legend_elements = []

        for i, rib_idx in enumerate(available_ribs):
            rib_color = colors[i]

            # Process both left and right sides
            for side in ['left', 'right']:
                # Get parametric midlines
                coronal_key = (rib_idx, 'coronal', side)
                sagittal_key = (rib_idx, 'sagittal', side)

                coronal_param = parametrizer.parametric_midlines.get(coronal_key)
                sagittal_param = parametrizer.parametric_midlines.get(sagittal_key)

                if coronal_param is None or sagittal_param is None:
                    continue

                # Get sampled points
                coronal_points = coronal_param.get('sampled_points')
                sagittal_points = sagittal_param.get('sampled_points')

                if coronal_points is None or sagittal_points is None:
                    continue

                # Use different marker styles for left and right
                marker = 'o' if side == 'left' else 'x'
                marker_size = 2
                alpha = 0.7

                # Plot the midlines
                axes[0].scatter(coronal_points[:, 1], coronal_points[:, 0], s=marker_size, color=rib_color,
                                marker=marker, alpha=alpha)
                axes[1].scatter(sagittal_points[:, 1], sagittal_points[:, 0], s=marker_size, color=rib_color,
                                marker=marker, alpha=alpha)

                # Mark start points
                axes[0].scatter(coronal_points[0, 1], coronal_points[0, 0], s=50, color=rib_color,
                                marker='o', edgecolor='black', linewidth=1)
                axes[1].scatter(sagittal_points[0, 1], sagittal_points[0, 0], s=50, color=rib_color,
                                marker='o', edgecolor='black', linewidth=1)

                # Mark end points
                axes[0].scatter(coronal_points[-1, 1], coronal_points[-1, 0], s=50, color=rib_color,
                                marker='o', edgecolor='black', linewidth=1)
                axes[1].scatter(sagittal_points[-1, 1], sagittal_points[-1, 0], s=50, color=rib_color,
                                marker='o', edgecolor='black', linewidth=1)

                # Add to legend only once
                if side == 'left':
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                      markerfacecolor=rib_color, markersize=8,
                                                      label=f"Rib {rib_idx + 1}"))

        # Add legend for start and end points
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                 markeredgecolor='black', markersize=8, label='Start Point')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                               markeredgecolor='black', markersize=8, label='End Point')
        left_side = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                               markersize=6, label='Left Rib')
        right_side = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='gray',
                                markersize=6, label='Right Rib')

        legend_elements.append(start_point)
        legend_elements.append(end_point)
        legend_elements.append(left_side)
        legend_elements.append(right_side)

        # Set titles and labels
        axes[0].set_title("Coronal View", color='black')
        axes[0].set_xlabel("X", color='black')
        axes[0].set_ylabel("Y", color='black')
        axes[0].tick_params(colors='black')

        axes[1].set_title("Sagittal View", color='black')
        axes[1].set_xlabel("Z", color='black')
        axes[1].set_ylabel("Y", color='black')
        axes[1].tick_params(colors='black')

        # Add legend
        leg = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0),
                         ncol=min(6, len(legend_elements)), frameon=False)
        for text in leg.get_texts():
            text.set_color('black')

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def visualize_3d_midlines(self, rib_indices=None):
        """
        Visualize 3D reconstructed rib midlines using PyVista
        """
        if not PYVISTA_AVAILABLE:
            print("PyVista not available. Cannot create 3D visualization.")
            return

        if not self.rib_extractor.midlines_3d:
            print("No 3D midlines reconstructed yet. Run reconstruct_midlines_in_3d() first.")
            return

        # If no indices specified, use all ribs
        if rib_indices is None:
            rib_indices = sorted(set([idx for idx, _ in self.rib_extractor.midlines_3d.keys()]))

        # Create PyVista plotter
        plotter = pv.Plotter()

        # Colors for different ribs
        colors = plt.cm.plasma(np.linspace(0, 1, self.rib_extractor.num_ribs))

        # Process each rib
        for rib_idx in rib_indices:
            left_3d = self.rib_extractor.midlines_3d.get((rib_idx, 'left'))
            right_3d = self.rib_extractor.midlines_3d.get((rib_idx, 'right'))

            rib_color = colors[rib_idx][:3]  # RGB values

            # Process left side
            if left_3d is not None and len(left_3d) > 0:
                if len(left_3d) > 1:
                    line_mesh = pv.lines_from_points(left_3d)
                    plotter.add_mesh(line_mesh, color=rib_color, line_width=3,
                                     label=f"Rib {rib_idx + 1} Left")

                    # Add start and end point markers
                    start_point = pv.PolyData(left_3d[0])
                    end_point = pv.PolyData(left_3d[-1])

                    plotter.add_mesh(start_point, color=rib_color, point_size=8,
                                     render_points_as_spheres=True)
                    plotter.add_mesh(end_point, color=rib_color, point_size=8,
                                     render_points_as_spheres=True)

            # Process right side
            if right_3d is not None and len(right_3d) > 0:
                if len(right_3d) > 1:
                    line_mesh = pv.lines_from_points(right_3d)
                    plotter.add_mesh(line_mesh, color=rib_color, line_width=3,
                                     label=f"Rib {rib_idx + 1} Right")

                    # Add start and end point markers
                    start_point = pv.PolyData(right_3d[0])
                    end_point = pv.PolyData(right_3d[-1])

                    plotter.add_mesh(start_point, color=rib_color, point_size=8,
                                     render_points_as_spheres=True)
                    plotter.add_mesh(end_point, color=rib_color, point_size=8,
                                     render_points_as_spheres=True)

        # Set camera position and show
        plotter.camera_position = 'iso'
        plotter.show()

    def visualize_all_meshes_pyvista(self):
        """
        Visualize all reconstructed rib meshes using PyVista
        """
        if not PYVISTA_AVAILABLE:
            print("PyVista not available. Cannot create 3D mesh visualization.")
            return

        if not self.volumetric_reconstructor:
            print("No volumetric reconstructor available")
            return

        if not self.volumetric_reconstructor.rib_meshes:
            print("No meshes available for visualization")
            return

        # Create PyVista plotter
        plotter = pv.Plotter(window_size=[1200, 1000])

        # Colors for ribs
        colors = plt.cm.plasma(np.linspace(0, 1, self.rib_extractor.num_ribs))

        # Plot each mesh
        for rib_idx in range(self.rib_extractor.num_ribs):
            for side in ['left', 'right']:
                mesh_data = self.volumetric_reconstructor.rib_meshes.get((rib_idx, side))

                if mesh_data:
                    vertices = mesh_data['vertices']
                    faces = mesh_data['faces']

                    # Create PyVista mesh
                    pv_faces = np.column_stack((np.full(len(faces), 3), faces))
                    mesh = pv.PolyData(vertices, pv_faces)

                    # Set color and transparency
                    rib_color = colors[rib_idx]
                    rgba_color = (rib_color[0], rib_color[1], rib_color[2],
                                  0.5 if side == 'right' else 0.7)

                    plotter.add_mesh(mesh, color=rgba_color, show_edges=True,
                                     line_width=0.1, edge_color='black')

                    # Plot the midline for reference
                    cross_section_data = self.volumetric_reconstructor.rib_cross_sections.get((rib_idx, side))
                    if cross_section_data:
                        midline_points = cross_section_data['cross_section_data']['midline_points']
                        midline_line = pv.lines_from_points(midline_points)
                        plotter.add_mesh(midline_line, color=tuple(rib_color[:3]),
                                         line_width=2, render_lines_as_tubes=True)

        # Add coordinate axes
        plotter.add_axes(interactive=True)

        # Set camera position
        plotter.camera_position = 'iso'

        # Show the plot
        plotter.show()

    def visualize_width_and_height_measurements(self, rib_idx, side=None, sample_indices=None):
        """
        Visualize cross-section measurement methodology
        """
        if not self.volumetric_reconstructor:
            print("No volumetric reconstructor available")
            return

        # Determine which sides to process
        if side is None:
            sides_to_process = ['left', 'right']
            show_both_sides = True
        else:
            sides_to_process = [side]
            show_both_sides = False

        # Get parametric midlines
        parametrizer = RibMidlineParametrization(self.rib_extractor)
        parametrizer.process_all_midlines()

        # Store data for each side
        side_data = {}

        for current_side in sides_to_process:
            # Get cross-section data
            if (rib_idx, current_side) not in self.volumetric_reconstructor.rib_cross_sections:
                print(f"Generating cross-sections for rib {rib_idx}, {current_side} side first...")
                cross_sections = self.volumetric_reconstructor.generate_all_cross_sections(rib_idx, current_side)
                if cross_sections is None:
                    print(f"Failed to generate cross-sections for rib {rib_idx}, {current_side} side")
                    continue

            cross_section_data = self.volumetric_reconstructor.rib_cross_sections[(rib_idx, current_side)][
                'cross_section_data']

            coronal_midline_key = (rib_idx, 'coronal', current_side)
            sagittal_midline_key = (rib_idx, 'sagittal', current_side)

            coronal_param = parametrizer.parametric_midlines.get(coronal_midline_key)
            sagittal_param = parametrizer.parametric_midlines.get(sagittal_midline_key)

            if coronal_param is None or sagittal_param is None:
                print(f"Missing parametric midline for rib {rib_idx}, {current_side} side")
                continue

            # Store all the data for this side
            side_data[current_side] = {
                'cross_section_data': cross_section_data,
                'coronal_param': coronal_param,
                'sagittal_param': sagittal_param
            }

        # If no valid data was collected, return
        if not side_data:
            print(f"No valid data found for rib {rib_idx}")
            return

        # Get segmentation images
        coronal_seg = self.rib_extractor.coronal_seg[:, :, rib_idx]
        sagittal_seg = self.rib_extractor.sagittal_seg[:, :, rib_idx]

        coronal_binary = (coronal_seg > 0).astype(np.uint8)
        sagittal_binary = (sagittal_seg > 0).astype(np.uint8)

        # If no sample indices provided, choose evenly spaced points
        if sample_indices is None:
            num_samples = 6
            sample_indices = np.linspace(0, self.volumetric_reconstructor.num_midline_samples - 1,
                                         num_samples, dtype=int)

        # Create the visualization
        if show_both_sides:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
            coronal_ax = axes[0]
            sagittal_ax = axes[1]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
            coronal_ax = axes[0]
            sagittal_ax = axes[1]

        for ax in axes:
            ax.set_facecolor('white')

        # Show segmentation as background
        coronal_bg_display = np.where(coronal_binary == 0, 1.0, 0.3)
        sagittal_bg_display = np.where(sagittal_binary == 0, 1.0, 0.3)

        coronal_ax.imshow(coronal_bg_display, cmap='gray', vmin=0, vmax=1)
        sagittal_ax.imshow(sagittal_bg_display, cmap='gray', vmin=0, vmax=1)

        # Set titles
        if show_both_sides:
            coronal_ax.set_title(f"Coronal View, Width Measurements", color='black', fontsize=16)
            avg_heights = [data['cross_section_data']['avg_height'] for data in side_data.values()]
            overall_avg_height = np.mean(avg_heights)
            sagittal_ax.set_title(f"Sagittal View, Height Measurement (Avg Height: {overall_avg_height:.2f}mm)",
                                  color='black', fontsize=16)
        else:
            current_side = list(side_data.keys())[0]
            coronal_ax.set_title(f"Coronal View - Rib {rib_idx}, {current_side.capitalize()} Side (Width Measurements)",
                                 color='black', fontsize=16)
            avg_height = side_data[current_side]['cross_section_data']['avg_height']
            sagittal_ax.set_title(
                f"Sagittal View - Rib {rib_idx}, {current_side.capitalize()} Side (Avg Height: {avg_height:.2f}mm)",
                color='black', fontsize=16)

        # Use fixed color
        rib_color = '#4a1079'
        side_colors = {'left': rib_color, 'right': rib_color}
        sample_colors = plt.cm.rainbow(np.linspace(0, 1, len(sample_indices)))

        # Plot measurements for each side
        for side_name, data in side_data.items():
            coronal_points = data['coronal_param']['sampled_points']
            sagittal_points = data['sagittal_param']['sampled_points']
            cross_section_data = data['cross_section_data']

            # Resample points
            num_midline_samples = self.volumetric_reconstructor.num_midline_samples
            coronal_resampled = np.zeros((num_midline_samples, 2))
            sagittal_resampled = np.zeros((num_midline_samples, 2))

            coronal_t = np.linspace(0, 1, len(coronal_points))
            sagittal_t = np.linspace(0, 1, len(sagittal_points))
            new_t = np.linspace(0, 1, num_midline_samples)

            for i in range(2):
                coronal_resampled[:, i] = np.interp(new_t, coronal_t, coronal_points[:, i])
                sagittal_resampled[:, i] = np.interp(new_t, sagittal_t, sagittal_points[:, i])

            # Use different colors for different sides
            if show_both_sides:
                midline_color = side_colors[side_name]
                alpha = 0.8
            else:
                midline_color = rib_color
                alpha = 1.0

            # Plot the entire parametrized midline
            coronal_ax.plot(coronal_resampled[:, 1], coronal_resampled[:, 0], '-',
                            color=midline_color, linewidth=2, alpha=alpha)
            sagittal_ax.plot(sagittal_resampled[:, 1], sagittal_resampled[:, 0], '-',
                             color=midline_color, linewidth=2, alpha=alpha)

        # Set axis labels
        coronal_ax.set_xlabel('X', color='black')
        coronal_ax.set_ylabel('Y', color='black')
        coronal_ax.tick_params(colors='black')

        sagittal_ax.set_xlabel('Z', color='black')
        sagittal_ax.set_ylabel('Y', color='black')
        sagittal_ax.tick_params(colors='black')

        # Display the title
        if show_both_sides:
            title = f"Rib {rib_idx} - Width and Height Measurements"
        else:
            side_name = list(side_data.keys())[0]
            title = f"Rib {rib_idx}, {side_name.capitalize()} Side - Width and Height Measurements"

        fig.suptitle(title, fontsize=18, color='black')
        plt.tight_layout()
        plt.show()