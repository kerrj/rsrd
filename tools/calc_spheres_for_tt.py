import tyro
from pathlib import Path
import viser
import trimesh
import time

def main(
    tt_path: Path,
    # Path to the tooltip stl file.
    vis_scale: float = 10,
):
    """Visualize the tooltip mesh, and add spheres to it."""
    tooltip = trimesh.load(tt_path, force="mesh")
    assert isinstance(tooltip, trimesh.Trimesh)
    tooltip.vertices *= vis_scale

    server = viser.ViserServer()
    server.add_mesh_trimesh("tooltip", tooltip)
    add_sphere_button = server.add_gui_button("Add Sphere")

    sphere_list = []  # (tf_handle, radius_handle, center, radius)

    @add_sphere_button.on_click
    def _(_):
        idx = len(sphere_list)
        center, radius = (0, 0, 0), 0.005
        tf_handle = server.add_transform_controls(
            f"sphere_{idx}",
            scale=0.1,
            position=center,
            wxyz=(0, 0, 0, 1),
            disable_rotations=True
        )
        mesh = trimesh.creation.icosphere(radius=radius * vis_scale)
        server.add_mesh_trimesh(f"sphere_{idx}/mesh", mesh)

        radius_handle = server.add_gui_slider(
            f"sphere_{idx}/radius",
            min=0.001,
            max=0.01,
            initial_value=radius,
            step=0.001
        )
        @radius_handle.on_update
        def _(_):
            value = radius_handle.value
            mesh = trimesh.creation.icosphere(radius=value * vis_scale)
            server.add_mesh_trimesh(f"sphere_{idx}/mesh", mesh)

        sphere_list.append((tf_handle, radius_handle))

    print_spheres_button = server.add_gui_button("Print Spheres")
    @print_spheres_button.on_click
    def _(_):
        for i, (tf_handle, radius_handle) in enumerate(sphere_list):
            print(f"Sphere {i}:")
            print(f"  Center: {[p / vis_scale for p in tf_handle.position]}")
            print(f"  Radius: {radius_handle.value}")

    while True:
        time.sleep(1000)
    

if __name__ == "__main__":
    tyro.cli(main)