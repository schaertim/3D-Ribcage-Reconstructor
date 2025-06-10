"""
Rib midline parametrization using spline curves
"""
import numpy as np
from scipy import interpolate
from skimage.graph import route_through_array


class RibMidlineParametrization:
    """
    Create parametric representations of rib midlines
    """

    def __init__(self, rib_extractor):
        """
        Initialize the rib midline parametrization

        Args:
            rib_extractor: RibMidlineExtractor instance with extracted midlines
        """
        self.rib_extractor = rib_extractor
        self.parametric_midlines = {}

    def process_all_midlines(self):
        """
        Process all midlines to create parametric representations
        """
        print(f"Processing all midlines, found {len(self.rib_extractor.midlines)} midlines")

        for (rib_idx, view, side), midline in self.rib_extractor.midlines.items():
            if midline is not None:
                print(f"Processing midline for rib {rib_idx}, {view} view, {side} side")
                print(f"  Midline has {np.sum(midline)} non-zero pixels")

                # Get ordered points from the midline
                ordered_points = self.get_ordered_points(midline)

                if ordered_points is not None and len(ordered_points) > 0:
                    print(f"  Successfully ordered {len(ordered_points)} points")
                    self.create_parametric_midline(rib_idx, view, side, ordered_points)
                    print(f"  Created parametric midline for rib {rib_idx}, {view} view, {side} side")
                else:
                    print(f"  Failed to order points for rib {rib_idx}, {view} view, {side} side")
            else:
                print(f"Midline for rib {rib_idx}, {view} view, {side} side is None")

    def get_ordered_points(self, binary_midline):
        """
        Extract ordered points from a binary midline image

        Args:
            binary_midline: Binary image of the midline

        Returns:
            Array of ordered (y, x) points along the midline
        """
        if binary_midline is None:
            print("  Binary midline is None")
            return None

        # Find all points in the midline
        y_coords, x_coords = np.where(binary_midline)
        points = np.column_stack([y_coords, x_coords])
        print(f"  Found {len(points)} points in binary midline")

        if len(points) == 0:
            print("  No points found in binary midline")
            return None

        # Find endpoints (pixels with only one neighbor)
        endpoints = []
        for i, (y, x) in enumerate(points):
            # Count neighbors in 8-connected neighborhood
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < binary_midline.shape[0] and
                            0 <= nx < binary_midline.shape[1] and
                            binary_midline[ny, nx]):
                        neighbors += 1

            if neighbors == 1:
                endpoints.append(i)

        # Find the two endpoints that are farthest apart
        max_dist = 0
        start_idx, end_idx = 0, 0

        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                p1 = points[endpoints[i]]
                p2 = points[endpoints[j]]
                dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                if dist > max_dist:
                    max_dist = dist
                    start_idx = endpoints[i]
                    end_idx = endpoints[j]

        start_point = (points[start_idx][0], points[start_idx][1])
        end_point = (points[end_idx][0], points[end_idx][1])

        print(f"  Using endpoints at {start_point} and {end_point} (distance: {max_dist:.2f})")

        # Use route_through_array to find the ordered path between endpoints
        cost = 1 - binary_midline

        try:
            path_indices, _ = route_through_array(
                cost,
                start_point,
                end_point,
                fully_connected=True
            )
            ordered_points = np.array(path_indices)
            print(f"  Successfully found path with {len(ordered_points)} points")
            return ordered_points
        except Exception as e:
            print(f"  Path finding failed: {str(e)}")
            print("  Could not determine ordered points")
            return None

    def create_parametric_midline(self, rib_idx, view, side, ordered_points):
        """
        Create a parametric representation of a midline

        Args:
            rib_idx: Index of the rib
            view: 'coronal' or 'sagittal'
            side: 'left' or 'right'
            ordered_points: Array of ordered (y, x) points along the midline
        """
        print(
            f"Creating parametric midline for rib {rib_idx}, {view} view, {side} side with {len(ordered_points)} points")

        if len(ordered_points) < 2:
            print(f"  Not enough points to create parametric midline (need at least 2, have {len(ordered_points)})")
            return

        # If we only have 2 or 3 points, add intermediate points
        if len(ordered_points) < 4:
            print(f"  Only {len(ordered_points)} points available, creating intermediate points")
            augmented_points = []

            for i in range(len(ordered_points) - 1):
                augmented_points.append(ordered_points[i])
                # Add 2 intermediate points between each original point
                for j in range(1, 3):
                    factor = j / 3.0
                    interp_point = ordered_points[i] * (1 - factor) + ordered_points[i + 1] * factor
                    augmented_points.append(interp_point)

            augmented_points.append(ordered_points[-1])
            ordered_points = np.array(augmented_points)
            print(f"  Augmented to {len(ordered_points)} points")

        # Create a parameter along the curve (chord length)
        t = np.zeros(len(ordered_points))
        for i in range(1, len(ordered_points)):
            t[i] = t[i - 1] + np.linalg.norm(ordered_points[i] - ordered_points[i - 1])

        # Normalize parameter to [0, 1]
        if t[-1] > 0:
            t = t / t[-1]

        # Determine the appropriate spline degree
        k = min(3, len(ordered_points) - 1)
        smoothing = max(0.5, len(ordered_points) * 7.0)  # Increased smoothing

        print(f"  Using smoothing parameter of {smoothing} with {len(ordered_points)} points")

        try:
            # Fit splines for each coordinate
            spline_y = interpolate.UnivariateSpline(t, ordered_points[:, 0], k=k, s=smoothing)
            spline_x = interpolate.UnivariateSpline(t, ordered_points[:, 1], k=k, s=smoothing)

            # Sample the spline at regular intervals
            t_params = np.linspace(0, 1, 100)
            sampled_points = np.column_stack([
                spline_y(t_params),
                spline_x(t_params)
            ])

            # Store the parametric representation
            self.parametric_midlines[(rib_idx, view, side)] = {
                'spline_y': spline_y,
                'spline_x': spline_x,
                't_params': t_params,
                'sampled_points': sampled_points,
                'ordered_points': ordered_points
            }
        except Exception as e:
            print(f"  Error creating parametric midline: {e}")
            print(f"  Falling back to linear interpolation")
            t_params = np.linspace(0, 1, 100)

            # Linear interpolation for y and x coordinates
            interp_y = np.interp(t_params, t, ordered_points[:, 0])
            interp_x = np.interp(t_params, t, ordered_points[:, 1])

            sampled_points = np.column_stack([interp_y, interp_x])

            # Store the parametric representation
            self.parametric_midlines[(rib_idx, view, side)] = {
                'spline_y': None,
                'spline_x': None,
                't_params': t_params,
                'sampled_points': sampled_points,
                'ordered_points': ordered_points
            }