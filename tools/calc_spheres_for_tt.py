import tyro
from pathlib import Path
import viser
import trimesh
import time

# Sphere 0:
#   Center: (0.003256095359052578, 0.16387875565351473, 0.17640238288129317)
# Sphere 1:
#   Center: (-0.02609974188713418, -0.06, 0.5490503601557893)
# Sphere 2:
#   Center: (0.005322841698408194, 0.07656488772025778, 0.228071051252434)
# Sphere 3:
#   Center: (-0.19701014139677747, 0.19, 0.1429758200453164)
# Sphere 4:
#   Center: (-0.17381168707913364, 0.081640117860203, 0.2897407534400887)
# Sphere 5:
#   Center: (0.014881624094412191, 0.01085787261386285, 0.4670406054264251)
# Sphere 6:
#   Center: (-0.16748886496556728, -0.0002538366501695505, 0.3863996411377595)
# Sphere 7:
#   Center: (-0.08247907260087828, -0.00699999988079071, 0.10014157629491019)
# Sphere 8:
#   Center: (-0.08407754954809918, 0.16160459711643588, 0.5428761400764458)
# Sphere 9:
#   Center: (-0.04868174142767175, 0.04528601073894346, 0.5856662328951279)
# Sphere 10:
#   Center: (0.011309107626565562, -0.04153388931740474, 0.3777276899154161)
# Sphere 11:
#   Center: (-0.018386021415558246, -0.06, 0.24926465572423326)
# Sphere 12:
#   Center: (-0.1642704182421847, -0.06, 0.2518167272414147)
# Sphere 13:
#   Center: (0.0026304692339604615, -0.03089479903660937, 0.16076172963928814)
# Sphere 14:
#   Center: (0.016935414436184362, 0.154011148627727, 0.5183853627404235)
# Sphere 15:
#   Center: (-0.1812966327524798, -0.00699999988079071, 0.10576831852725257)
# Sphere 16:
#   Center: (0.009109960817054605, 0.18781332999455747, 0.32274902025812335)
# Sphere 17:
#   Center: (-0.19835541065008302, 0.1609768169387449, 0.23846828619925917)
# Sphere 18:
#   Center: (-0.13041523517999037, 0.19, 0.44264793461973845)
# Sphere 19:
#   Center: (-0.12264942047298444, -0.049189242676637296, 0.48436713768187317)


def main(
    tt_path: Path,
    # Path to the tooltip stl file.
    vis_scale: float = 10,
):
    """Visualize the tooltip mesh, and add spheres to it."""
    # tooltip = trimesh.load(tt_path, force="mesh")
    tooltip = trimesh.load('../yumi_rayfin_v3_flexybit.stl')
    tooltip.vertices /= 1000
    tooltip.vertices[:, 2] += 0.002
    from curobo.geom.sphere_fit import fit_spheres_to_mesh, SphereFitType
    _tooltip = tooltip.copy()
    # _tooltip.vertices *= 0.9
    n_pts, n_centers = fit_spheres_to_mesh(_tooltip, 50, 0.01, SphereFitType.SAMPLE_SURFACE)

    assert isinstance(tooltip, trimesh.Trimesh)
    tooltip.vertices *= vis_scale

    server = viser.ViserServer()
    server.add_mesh_trimesh("tooltip", tooltip)
    add_sphere_button = server.add_gui_button("Add Sphere")

    sphere_list = []  # (tf_handle, radius_handle, center, radius)

    # Sphere 0:
    # Center: [-0.0033858122339594267, 0.0012429177250910502, 0.013481134568233615]
    # Radius: 0.007
    # Sphere 1:
    # Center: [-0.003951992900603203, 0.01340199295585955, 0.013481134568233615]
    # Radius: 0.007
    # Sphere 2:
    # Center: [-0.015297208519018496, 0.011951175487664278, 0.01710286689940423]
    # Radius: 0.009
    # Sphere 3:
    # Center: [-0.014282789651221836, 0.0010132878398739118, 0.016982657589114353]
    # Radius: 0.009



    # import numpy as np
    # mesh = trimesh.creation.icosphere(radius=0.007 * vis_scale)
    # mesh.vertices += np.array([-0.0033858122339594267, 0.0012429177250910502, 0.013481134568233615])*vis_scale
    # server.add_mesh_trimesh("sphere__0/mesh", mesh)
    # mesh = trimesh.creation.icosphere(radius=0.007 * vis_scale)
    # mesh.vertices += np.array([-0.003951992900603203, 0.01340199295585955, 0.013481134568233615])*vis_scale
    # server.add_mesh_trimesh("sphere__1/mesh", mesh)
    # mesh = trimesh.creation.icosphere(radius=0.009 * vis_scale)
    # mesh.vertices += np.array([-0.015297208519018496, 0.011951175487664278, 0.01710286689940423])*vis_scale
    # server.add_mesh_trimesh("sphere__2/mesh", mesh)
    # mesh = trimesh.creation.icosphere(radius=0.009 * vis_scale)
    # mesh.vertices += np.array([-0.014282789651221836, 0.0010132878398739118, 0.016982657589114353])*vis_scale
    # server.add_mesh_trimesh("sphere__3/mesh", mesh)

    # def curry(i):
    #     tf_handle = server.add_gui_vector3(f"sphere_{i}", n_pts[i], step=0.01)
    #     mesh = trimesh.creation.icosphere(radius=n_centers[i] * vis_scale)
    #     frame = server.add_frame(
    #         f'sphere_{i}',
    #         show_axes=False,
    #         position=n_pts[i] * vis_scale,
    #     )
    #     @tf_handle.on_update
    #     def _(_):
    #         frame.position = tf_handle.value*vis_scale
    #         # mesh = trimesh.creation.icosphere(radius=n_centers[i] * vis_scale)
    #         #
    #     server.add_mesh_trimesh(f"sphere_{i}/mesh", mesh)
    #     sphere_list.append((tf_handle, center))

    # for i, center in enumerate(n_centers):
    #     # tf_handle = server.add_transform_controls(
    #     #     f"sphere_{i}",
    #     #     scale=0.15,
    #     #     position=n_pts[i]*vis_scale,
    #     # )
    #     curry(i)



    @add_sphere_button.on_click
    def _(_):
        idx = len(sphere_list)
        center, radius = (0, 0, 0), 0.005
        tf_handle = server.add_transform_controls(
            f"sphere_{idx}",
            scale=0.20,
            position=center,
            wxyz=(0, 0, 0, 1),
            disable_rotations=True
        )
        mesh = trimesh.creation.icosphere(radius=radius * vis_scale)
        server.add_mesh_trimesh(f"sphere_{idx}/mesh", mesh)

        radius_handle = server.add_gui_slider(
            f"sphere_{idx}/radius",
            min=0.001,
            max=0.02,
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
            # print(f"Sphere {i}:")
            # print(f" - \"center\": [{tf_handle.value[0]}, {tf_handle.value[1]}, {tf_handle.value[2]}]")
            # print(f"   \"radius\": 0.01")
            print(f"  Center: {[p / vis_scale for p in tf_handle.position]}")
            print(f"  Radius: {radius_handle.value}")

    while True:
        time.sleep(1000)
    

if __name__ == "__main__":
    tyro.cli(main)