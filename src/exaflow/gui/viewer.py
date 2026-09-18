from __future__ import annotations

import os
import math
import traceback
from PySide6 import QtCore, QtWidgets
from typing import Optional, Any

import pyvista as pv
from pyvistaqt import QtInteractor  # type: ignore

import numpy as np
import vtk  # type: ignore[import-untyped]

from .csv_loader import load_total_csv_to_imagedata


class PyVistaViewer(QtWidgets.QFrame):
    render_failed = QtCore.Signal(str)
    """
    Carries the traceback of a step that drew nothing. The viewer has no log of its own, so the window that owns it connects this and reports what the user cannot see on the canvas.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        # Embedded PyVista plotter
        self._plotter = QtInteractor(self)  # type: ignore
        self._plotter.set_background("slategray")  # type: ignore[arg-type]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plotter)

        self._main_mesh_actor: Any = None  # main mesh actor
        self._scalar_bar: Any = None  # not strictly needed; kept for API symmetry

        # Cached dataset (PyVista dataset such as UniformGrid/RectilinearGrid)
        self._simulation_data: Any = None

        # Orientation widget (axes) and overlays
        self._coordinate_axes_actor: Any = None

        # Outline and cube axes
        self._domain_outline_actor: Any = None
        self._cube_axes: Any = None  # flag/actor depending on PyVista version

        # Vector glyph pipeline
        self._velocity_arrows_actor: Any = None

        self._slice_plane_actor: Any = None
        self._slice_plane_mesh: Any = None
        self._scalar_name: Optional[str] = None
        self._preferred_scalar_name: str = "speed"
        self._scalar_range: Any = None
        self._held_scalar_ranges: dict[str, tuple[int, float, float]] = {}
        self._is_index_space: bool = False
        self._is_showing_flat_dataset: bool = False

        # Toggles and settings
        self._show_coordinate_axes: bool = True
        self._show_domain_outline: bool = True
        self._show_cube_axes: bool = True
        self._show_velocity_vectors: bool = False
        self._show_slice_plane: bool = False
        self._slice_axis: int = 2
        self._slice_position: Optional[float] = None
        self._vector_sampling_stride: int = 4  # sampling ratio for vectors

        # Color map to mimic blue->red
        self._color_map: str = "coolwarm"

        # Build orientation widget (disabled/enabled via toggle)
        if self._show_coordinate_axes:
            self._coordinate_axes_actor = self._plotter.add_axes()  # type: ignore[call-arg]

    # ------------------ Public API ------------------
    def clear(self) -> None:
        if self._main_mesh_actor is not None:
            self._plotter.remove_actor(self._main_mesh_actor)
            self._main_mesh_actor = None
        for title in list(self._plotter.scalar_bars.keys()):
            self._plotter.remove_scalar_bar(title)

        # Remove outline
        if self._domain_outline_actor is not None:
            self._plotter.remove_actor(self._domain_outline_actor)
            self._domain_outline_actor = None

        # Remove cube axes
        self._plotter.remove_bounds_axes()  # type: ignore[call-arg]
        self._cube_axes = None

        # Remove vector glyphs
        if self._velocity_arrows_actor is not None:
            self._plotter.remove_actor(self._velocity_arrows_actor)
            self._velocity_arrows_actor = None

        if self._slice_plane_actor is not None:
            self._plotter.remove_actor(self._slice_plane_actor)
            self._slice_plane_actor = None
        self._slice_plane_mesh = None
        self._scalar_name = None
        self._scalar_range = None

        # Orientation marker visibility per toggle
        if self._show_coordinate_axes and self._coordinate_axes_actor is None:
            self._coordinate_axes_actor = self._plotter.add_axes()  # type: ignore[call-arg]
        if (not self._show_coordinate_axes) and (self._coordinate_axes_actor is not None):
            self._plotter.remove_actor(self._coordinate_axes_actor)
            self._coordinate_axes_actor = None

        self._simulation_data = None
        self._plotter.render()

    def load_vtr(self, path: str) -> None:
        abspath = os.path.abspath(path)
        mesh = pv.read(abspath)
        self.load_mesh(mesh)

    def load_csv(self, path: str) -> None:
        vtk_image_data = load_total_csv_to_imagedata(path)
        mesh = pv.wrap(vtk_image_data)
        self.load_mesh(mesh, is_index_space=True)

    def load_mesh(self, mesh: pv.DataSet, *, is_index_space: bool = False) -> None:
        """
        Show `mesh` and drop whatever was on screen. Set `is_index_space` for a mesh built from a CSV file, which is indexed in cells; leave it false for a `.vtr` file and for a streamed dataset, which carry metres. The slice position label reads this to name its unit.
        """

        self.clear()
        self._is_index_space = is_index_space
        self._load_mesh(mesh)

    def describe_time_level(self) -> str:
        """
        Where the run had reached when it wrote the loaded file, as ` (step 400, t = 0.8 s)`, or an empty string unless the file states both the step and the time. Every file ExaFlow writes states both; a file from another tool need not.
        """

        if self._simulation_data is None:
            return ""
        field_data = self._simulation_data.GetFieldData()
        step = field_data.GetArray("StepIndex")
        moment = field_data.GetArray("TimeValue")
        if step is None or moment is None:
            return ""
        return f" (step {int(step.GetTuple1(0))}, t = {float(moment.GetTuple1(0)):.6g} s)"

    def _load_mesh(self, mesh: pv.DataSet) -> None:
        self._simulation_data = mesh

        if "velocity" in mesh.point_data and "speed" not in mesh.point_data:
            mesh.point_data["speed"] = np.linalg.norm(mesh.point_data["velocity"], axis=1)

        # Choose scalar if available
        scalar_name: Optional[str] = None
        if self._preferred_scalar_name in mesh.point_data:
            scalar_name = self._preferred_scalar_name
        elif mesh.point_data and mesh.active_scalars_name:
            scalar_name = mesh.active_scalars_name
        elif "pressure" in mesh.point_data:
            scalar_name = "pressure"
        elif len(mesh.point_data.keys()) > 0:
            scalar_name = list(mesh.point_data.keys())[0]
        self._scalar_name = scalar_name

        # Scalar bar formatting
        scalar_bar_format = "%.3g"
        self._scalar_range = None
        if scalar_name is not None:
            low, high = (float(bound) for bound in mesh.get_data_range(scalar_name, "point"))
            step = mesh.field_data["StepIndex"] if "StepIndex" in mesh.field_data else None
            held = self._held_scalar_ranges.get(scalar_name)
            if step is not None and held is not None:
                if int(step[0]) >= held[0]:
                    low, high = min(low, held[1]), max(high, held[2])
                else:
                    self._held_scalar_ranges.clear()
            if high - low <= abs(high) * 1e-12:
                padding = abs(high) * 0.05 if high != 0.0 else 1.0
                low, high = high - padding, high + padding
                self._held_scalar_ranges.pop(scalar_name, None)
            else:
                tick = 10.0 ** math.floor(math.log10(high - low)) / 5.0
                low, high = round(math.floor(low / tick) * tick, 12), round(math.ceil(high / tick) * tick, 12)
                if step is None:
                    self._held_scalar_ranges.pop(scalar_name, None)
                else:
                    self._held_scalar_ranges[scalar_name] = (int(step[0]), low, high)
            self._scalar_range = (low, high)
            max_magnitude = max(abs(low), abs(high))
            if max_magnitude >= 1e4 or (0 < max_magnitude <= 1e-3):
                scalar_bar_format = "%.2e"

        # Scalar bar geometry
        bar_height = 0.34
        bar_width = 0.038
        margin_left = 0.045
        margin_top = 0.05
        bar_position_x = margin_left
        bar_position_y = 1.0 - bar_height - margin_top

        scalar_bar_args = dict(
            title=(scalar_name or "").capitalize(),
            n_labels=6,
            fmt=scalar_bar_format,
            position_x=bar_position_x,
            position_y=bar_position_y,
            width=bar_width,
            height=bar_height,
            vertical=True,
            label_font_size=11,
            title_font_size=13,
        )

        surface_mesh = mesh.extract_surface()

        self._main_mesh_actor = self._plotter.add_mesh(
            surface_mesh,
            scalars=scalar_name,
            clim=self._scalar_range,  # type: ignore[arg-type]
            cmap=self._color_map,  # type: ignore[arg-type]
            scalar_bar_args=scalar_bar_args,  # type: ignore[arg-type]
        )

        # Cache and tweak scalar bar appearance
        self._scalar_bar = None
        scalar_bars = getattr(self._plotter, "scalar_bars", {})
        if scalar_name and scalar_name in scalar_bars:
            self._scalar_bar = scalar_bars[scalar_name]
        elif getattr(self._plotter, "scalar_bar", None) is not None:
            self._scalar_bar = self._plotter.scalar_bar

        if self._scalar_bar is not None:
            text_color = (1.0, 1.0, 1.0)
            if hasattr(vtk.vtkScalarBarActor, "SetDrawTickMarks"):
                self._scalar_bar.SetDrawTickMarks(True)
                self._scalar_bar.SetDrawTickLabels(True)
            else:
                if hasattr(self._scalar_bar, "DrawTickMarksOn"):
                    self._scalar_bar.DrawTickMarksOn()
                if hasattr(self._scalar_bar, "DrawTickLabelsOn"):
                    self._scalar_bar.DrawTickLabelsOn()
            if hasattr(self._scalar_bar, "SetTickLength"):
                self._scalar_bar.SetTickLength(0.03)
            if hasattr(self._scalar_bar, "SetLabelTextPad"):
                self._scalar_bar.SetLabelTextPad(12)

            label_text_property = self._scalar_bar.GetLabelTextProperty()
            label_text_property.SetColor(*text_color)
            if hasattr(label_text_property, "SetJustificationToCentered"):
                label_text_property.SetJustificationToCentered()
            title_text_property = self._scalar_bar.GetTitleTextProperty()
            title_text_property.SetColor(*text_color)
            if hasattr(title_text_property, "SetJustificationToCentered"):
                title_text_property.SetJustificationToCentered()

        # Update overlays and vectors
        self._update_slice()
        self._update_outline()
        self._update_cube_axes()
        self._update_vectors()
        self._plotter.reset_camera()  # type: ignore[call-arg]
        flat_axis = self._find_flat_axis()
        if flat_axis is not None:
            self._face_plane(flat_axis)
        elif self._is_showing_flat_dataset and not self._show_slice_plane:
            self.view_iso()
        self._is_showing_flat_dataset = flat_axis is not None
        self._plotter.render()


    # ------------------ Overlays and vectors ------------------
    def _find_flat_axis(self) -> Optional[int]:
        """
        The axis the loaded dataset has no thickness on, or None when all three have a span. A 1D or 2D result is flat, and pyvista cuts nothing out of a flat axis.
        """

        if self._simulation_data is None:
            return None
        bounds = self._simulation_data.bounds
        for axis in range(3):
            if bounds[2 * axis + 1] <= bounds[2 * axis]:
                return axis
        return None

    def _face_plane(self, axis: int) -> None:
        self._plotter.enable_parallel_projection()  # type: ignore[call-arg]
        self._plotter.enable_image_style()  # type: ignore[call-arg]
        (self._plotter.view_yz, self._plotter.view_zx, self._plotter.view_xy)[axis]()  # type: ignore[call-arg]

    def _update_slice(self) -> None:
        if self._slice_plane_actor is not None:
            self._plotter.remove_actor(self._slice_plane_actor)
            self._slice_plane_actor = None
        self._slice_plane_mesh = None
        if self._main_mesh_actor is not None:
            self._main_mesh_actor.SetVisibility(not self._show_slice_plane)

        if (not self._show_slice_plane) or self._simulation_data is None:
            if self._find_flat_axis() is None:
                self._plotter.disable_parallel_projection()  # type: ignore[call-arg]
                self._plotter.enable_trackball_style()  # type: ignore[call-arg]
            return

        bounds = self._simulation_data.bounds
        low = float(bounds[2 * self._slice_axis])
        high = float(bounds[2 * self._slice_axis + 1])
        if self._slice_position is None:
            position = (low + high) / 2.0
        else:
            position = min(max(self._slice_position, low), high)

        origin = list(self._read_dataset_center() or (0.0, 0.0, 0.0))
        origin[self._slice_axis] = position
        normal = [0.0, 0.0, 0.0]
        normal[self._slice_axis] = 1.0
        self._slice_plane_mesh = self._simulation_data.slice(normal=normal, origin=origin)

        self._slice_plane_actor = self._plotter.add_mesh(
            self._slice_plane_mesh,
            scalars=self._scalar_name,
            clim=self._scalar_range,  # type: ignore[arg-type]
            cmap=self._color_map,  # type: ignore[arg-type]
            show_scalar_bar=False,
        )
        self._face_plane(self._slice_axis)

    def _update_outline(self) -> None:
        if self._simulation_data is None:
            return
        if self._show_domain_outline and not self._show_slice_plane:
            if self._domain_outline_actor is None:
                domain_outline = self._simulation_data.outline()
                self._domain_outline_actor = self._plotter.add_mesh(
                    domain_outline,
                    color="black",
                    line_width=1,
                    style="wireframe",
                    show_scalar_bar=False,
                )
        else:
            if self._domain_outline_actor is not None:
                self._plotter.remove_actor(self._domain_outline_actor)
                self._domain_outline_actor = None

    def _update_cube_axes(self) -> None:
        if self._simulation_data is None:
            return
        if self._show_cube_axes:
            if self._cube_axes is None:
                # Show cube axes with white labels
                self._plotter.show_bounds(  # type: ignore[call-arg]
                    bounds=self._simulation_data.bounds,
                    location="outer",
                    color="white",
                    show_xaxis=True,
                    show_yaxis=True,
                    show_zaxis=True,
                    grid=False,
                )
                self._cube_axes = True
            else:
                # Refresh bounds if dataset changed
                self._plotter.remove_bounds_axes()  # type: ignore[call-arg]
                self._plotter.show_bounds(  # type: ignore[call-arg]
                    bounds=self._simulation_data.bounds,
                    location="outer",
                    color="white",
                    show_xaxis=True,
                    show_yaxis=True,
                    show_zaxis=True,
                    grid=False,
                )
                self._cube_axes = True
        else:
            self._plotter.remove_bounds_axes()  # type: ignore[call-arg]
            self._cube_axes = None

    def _find_vector_array_name(self) -> Optional[str]:
        point_data = self._simulation_data.point_data
        if point_data is None:
            return None
        # Prefer explicitly named vectors
        for name in ("velocity", "Velocity", "U", "VEL", "vec"):
            if name in point_data:
                vector_array = point_data[name]
                if vector_array.ndim == 2 and vector_array.shape[1] == 3:
                    return name
        # Fallback: search any 3-comp array
        for name in point_data.keys():
            vector_array = point_data[name]
            if vector_array.ndim == 2 and vector_array.shape[1] == 3:
                return name
        return None

    def _update_vectors(self) -> None:
        # Remove if disabled or no dataset
        if (not self._show_velocity_vectors) or (self._simulation_data is None):
            if self._velocity_arrows_actor is not None:
                self._plotter.remove_actor(self._velocity_arrows_actor)
                self._velocity_arrows_actor = None
            return

        vector_field_name = self._find_vector_array_name()
        if not vector_field_name:
            # No vectors available
            if self._velocity_arrows_actor is not None:
                self._plotter.remove_actor(self._velocity_arrows_actor)
                self._velocity_arrows_actor = None
            return

        try:
            mesh = self._simulation_data
            if self._show_slice_plane and self._slice_plane_mesh is not None:
                mesh = self._slice_plane_mesh
            mesh_points = np.asarray(mesh.points)
            velocity_vectors = np.asarray(mesh.point_data[vector_field_name])
            num_points = int(mesh_points.shape[0])

            sampling_stride = max(1, int(self._vector_sampling_stride))
            sampled_indices = np.arange(0, num_points, sampling_stride, dtype=int)
            if sampled_indices.size == 0:
                sampled_indices = np.array([0], dtype=int)

            arrow_centers = mesh_points[sampled_indices]
            arrow_directions = velocity_vectors[sampled_indices]

            # Scale arrows relative to dataset size
            mesh_bounds = mesh.bounds
            diagonal_length = math.sqrt((mesh_bounds[1]-mesh_bounds[0])**2 + (mesh_bounds[3]-mesh_bounds[2])**2 + (mesh_bounds[5]-mesh_bounds[4])**2)
            arrow_scale_factor = max(diagonal_length, 1e-6) / 50.0

            # Build glyphs using PolyData.glyph so we can color by scalars
            arrow_polydata = pv.PolyData(arrow_centers)
            arrow_polydata.point_data[vector_field_name] = arrow_directions
            # Optional coloring by pressure
            pressure_scalar_name: Optional[str] = None
            if "pressure" in mesh.point_data:
                arrow_polydata.point_data["pressure"] = np.asarray(mesh.point_data["pressure"])[sampled_indices]
                pressure_scalar_name = "pressure"

            arrow_glyphs = arrow_polydata.glyph(orient=vector_field_name, scale=False, factor=arrow_scale_factor)

            # Remove old glyph actor
            if self._velocity_arrows_actor is not None:
                self._plotter.remove_actor(self._velocity_arrows_actor)
                self._velocity_arrows_actor = None

            if pressure_scalar_name is not None:
                self._velocity_arrows_actor = self._plotter.add_mesh(
                    arrow_glyphs,
                    scalars=pressure_scalar_name,
                    cmap=self._color_map,  # type: ignore[arg-type]
                    show_scalar_bar=False,
                )
            else:
                self._velocity_arrows_actor = self._plotter.add_mesh(
                    arrow_glyphs,
                    color="white",
                    show_scalar_bar=False,
                )
        except Exception:
            # On any failure, remove glyphs
            if self._velocity_arrows_actor is not None:
                self._plotter.remove_actor(self._velocity_arrows_actor)
                self._velocity_arrows_actor = None
            self.render_failed.emit(f"The velocity arrows were dropped:\n{traceback.format_exc()}")

    # ------------------ Public toggles ------------------
    def set_show_axes(self, show: bool) -> None:
        self._show_coordinate_axes = show
        if self._show_coordinate_axes:
            if self._coordinate_axes_actor is None:
                self._coordinate_axes_actor = self._plotter.add_axes()  # type: ignore[call-arg]
        else:
            if self._coordinate_axes_actor is not None:
                self._plotter.remove_actor(self._coordinate_axes_actor)
                self._coordinate_axes_actor = None
        self._plotter.render()

    def set_show_outline(self, show: bool) -> None:
        self._show_domain_outline = show
        self._update_outline()
        self._plotter.render()

    def set_show_cube_axes(self, show: bool) -> None:
        self._show_cube_axes = show
        self._update_cube_axes()
        self._plotter.render()

    def set_scalar_name(self, name: str) -> None:
        self._preferred_scalar_name = name
        if self._simulation_data is not None:
            self.load_mesh(self._simulation_data, is_index_space=self._is_index_space)

    def clear_held_scalar_ranges(self) -> None:
        self._held_scalar_ranges.clear()

    def set_show_vectors(self, show: bool) -> None:
        self._show_velocity_vectors = show
        self._update_vectors()
        self._plotter.render()

    def set_vector_stride(self, stride: int) -> None:
        self._vector_sampling_stride = max(1, int(stride))
        if self._show_velocity_vectors:
            self._update_vectors()
            self._plotter.render()

    def can_slice(self) -> bool:
        """
        Whether the loaded dataset has three axes of non-zero span. A 1D or 2D result already is one plane and a cut of it returns no points, so the caller disables the control rather than show an empty view.
        """

        return self._simulation_data is not None and self._find_flat_axis() is None

    def read_slice_extent(self, axis: int) -> Optional[tuple[float, float, str]]:
        """
        The low bound, the high bound and the unit word of `axis` (0 for x, 1 for y, 2 for z) in the loaded dataset, such as (0.0, 2.0, "m"). The unit is "cell" for a dataset read from a CSV file, which carries indices and no physical extent. None when no dataset is loaded.
        """

        if self._simulation_data is None:
            return None
        bounds = self._simulation_data.bounds
        return float(bounds[2 * axis]), float(bounds[2 * axis + 1]), "cell" if self._is_index_space else "m"

    def set_show_slice(self, show: bool) -> None:
        """
        Turn the cross-section on or off. On draws one plane and locks the camera to face it; off restores the volume and free rotation.
        """

        self._show_slice_plane = show
        self._update_slice()
        self._update_outline()
        self._update_vectors()
        self._plotter.render()

    def set_slice_axis(self, axis: int) -> None:
        """
        Set the axis the plane cuts across, 0 for x, 1 for y, 2 for z. The plane moves back to the centre of the new axis.
        """

        self._slice_axis = axis
        self._slice_position = None
        if self._show_slice_plane:
            self._update_slice()
            self._update_vectors()
            self._plotter.render()

    def set_slice_position(self, position: float) -> None:
        """
        Move the plane to `position` along the slice axis, in the coordinates of the loaded dataset. A value outside the bounds is clamped to them.
        """

        self._slice_position = position
        if self._show_slice_plane:
            self._update_slice()
            self._update_vectors()
            self._plotter.render()

    # ------------------ Camera presets ------------------
    def _read_dataset_center(self) -> Optional[tuple[float, float, float]]:
        if self._simulation_data is None:
            return None
        bounds = self._simulation_data.bounds
        return ((bounds[0]+bounds[1])/2.0, (bounds[2]+bounds[3])/2.0, (bounds[4]+bounds[5])/2.0)

    def _read_dataset_size(self) -> float:
        bounds = self._simulation_data.bounds
        return math.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)

    def _set_camera(self, position: tuple[float, float, float], up_vector: tuple[float, float, float]) -> None:
        center = self._read_dataset_center()
        if center is None:
            return
        self._plotter.camera_position = [position, center, up_vector]
        self._plotter.reset_camera()  # type: ignore[call-arg]
        self._plotter.render()

    def view_pos_x(self) -> None:
        center = self._read_dataset_center()
        if center is None:
            return
        distance = self._read_dataset_size()
        self._set_camera((center[0] + distance, center[1], center[2]), (0.0, 0.0, 1.0))

    def view_neg_x(self) -> None:
        center = self._read_dataset_center()
        if center is None:
            return
        distance = self._read_dataset_size()
        self._set_camera((center[0] - distance, center[1], center[2]), (0.0, 0.0, 1.0))

    def view_pos_y(self) -> None:
        center = self._read_dataset_center()
        if center is None:
            return
        distance = self._read_dataset_size()
        self._set_camera((center[0], center[1] + distance, center[2]), (0.0, 0.0, 1.0))

    def view_neg_y(self) -> None:
        center = self._read_dataset_center()
        if center is None:
            return
        distance = self._read_dataset_size()
        self._set_camera((center[0], center[1] - distance, center[2]), (0.0, 0.0, 1.0))

    def view_pos_z(self) -> None:
        center = self._read_dataset_center()
        if center is None:
            return
        distance = self._read_dataset_size()
        self._set_camera((center[0], center[1], center[2] + distance), (0.0, 1.0, 0.0))

    def view_neg_z(self) -> None:
        center = self._read_dataset_center()
        if center is None:
            return
        distance = self._read_dataset_size()
        self._set_camera((center[0], center[1], center[2] - distance), (0.0, 1.0, 0.0))

    def view_iso(self) -> None:
        center = self._read_dataset_center()
        if center is None:
            return
        distance = self._read_dataset_size()
        offset = distance / math.sqrt(3.0)
        self._set_camera((center[0] + offset, center[1] + offset, center[2] + offset), (0.0, 0.0, 1.0))

