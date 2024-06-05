import cv2
import time
import numpy as np
from autolab_core import CameraIntrinsics,PointCloud,RigidTransform,Point
import matplotlib.pyplot as plt
import viser.transforms as vtf
from toad.zed import Zed

def find_corners(img,sx,sy,SB=True):
    '''
    sx and sy are the number of internal corners in the chessboard
    '''
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((sx * sy, 3), np.float32)
    objp[:, :2] = np.mgrid[0:sx, 0:sy].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # create images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    if SB:
        ret, corners = cv2.findChessboardCornersSB(gray, (sx, sy), None)
    else:
        ret, corners = cv2.findChessboardCorners(gray, (sx, sy), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        if corners is not None:
            return corners.squeeze()
    return None

def register_zed():
    N = 6
    sx = 6
    sy = 4
    zed = Zed()
    cb_trans = [0.025*20, 0, 0.003-0.005-0.001] # 20 * 25 mm, and 
    R = vtf.SO3.from_z_radians(np.pi).as_matrix()
    H_chess_world = RigidTransform(R,cb_trans,from_frame='cb',to_frame='world')
    while True:
        img_left, img_right, _ = zed.get_frame(depth=False)
        img_left,img_right = img_left.cpu().numpy(),img_right.cpu().numpy()
        l_corners = find_corners(img_left, sx, sy,True)
        r_corners = find_corners(img_right, sx, sy,True)
        print(f"Found corners: {l_corners is not None}, {r_corners is not None}")
        if l_corners is None or r_corners is None:
            _,axs=plt.subplots(1,2)
            axs[0].imshow(img_left)
            if l_corners is not None:
                axs[0].scatter(l_corners[:, 0], l_corners[:, 1], s=10)
            axs[1].imshow(img_right)
            if r_corners is not None:
                axs[1].scatter(r_corners[:, 0], r_corners[:, 1], s=10)
            print("Please reposition so corners are all in view")
            plt.show()
            continue
        break
    K = zed.get_K()
    H_ster = zed.get_stereo_transform()
    camera_intr = CameraIntrinsics('zed', K[0,0], K[1,1], K[0,2], K[1,2],height=img_left.shape[0],width=img_left.shape[1])
    cam_sep = abs(H_ster[0,3])  # meters
    print(camera_intr,cam_sep)
    Pl = np.zeros((3, 4))
    Pl[0:3, 0:3] = np.eye(3)
    Pl = camera_intr.proj_matrix @ Pl
    Pr = np.zeros((3, 4))
    Pr[0:3, 0:3] = np.eye(3)
    Pr[0, 3] = -cam_sep
    Pr = camera_intr.proj_matrix @ Pr
    zed_corners_3d = cv2.triangulatePoints(Pl, Pr, l_corners.T, r_corners.T)
    zed_corners_3d = zed_corners_3d[0:3, :] / zed_corners_3d[3, :]
    points_3d_plane=PointCloud(zed_corners_3d,'zed')
    X = np.c_[
        points_3d_plane.x_coords,
        points_3d_plane.y_coords,
        np.ones(points_3d_plane.num_points),
    ]
    y = points_3d_plane.z_coords
    A = X.T.dot(X)
    b = X.T.dot(y)
    w = np.linalg.inv(A).dot(b)
    n = np.array([w[0], w[1], -1])
    n = n / np.linalg.norm(n)
    mean_point_plane = points_3d_plane.mean()

    # find x-axis of the chessboard coordinates on the fitted plane
    T_camera_table = RigidTransform(
        translation=-points_3d_plane.mean().data,
        from_frame=points_3d_plane.frame,
        to_frame="table",
    )
    
    points_3d_centered = T_camera_table * points_3d_plane

    # get points along y
    coord_pos_x = int(np.floor(sx * sy / 2.0))
    coord_neg_x = int(np.ceil(sx * sy / 2.0))

    points_pos_x = points_3d_centered[coord_pos_x:]
    points_neg_x = points_3d_centered[:coord_neg_x]
    x_axis = np.mean(points_pos_x.data, axis=1) - np.mean(
        points_neg_x.data, axis=1
    )
    x_axis = x_axis - np.vdot(x_axis, n) * n
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)

    # produce translation and rotation from plane center and chessboard
    # basis
    rotation_cb_camera = RigidTransform.rotation_from_axes(
        x_axis, y_axis, n
    )
    translation_cb_camera = mean_point_plane.data
    T_cb_camera = RigidTransform(
        rotation=rotation_cb_camera,
        translation=translation_cb_camera,
        from_frame="cb",
        to_frame='zed',
    )
    print(T_cb_camera)
    # UNCOMMENT THE BELOW TO SHOW THE AXES DURING CALIBRATION
    # display image with axes overlayed
    cb_center_im = camera_intr.project(
        Point(T_cb_camera.translation, frame='zed')
    )
    scale=.05
    cb_x_im = camera_intr.project(
        Point(
            T_cb_camera.translation
            + T_cb_camera.x_axis * scale,
            frame='zed',
        )
    )
    cb_y_im = camera_intr.project(
        Point(
            T_cb_camera.translation
            + T_cb_camera.y_axis * scale,
            frame='zed',
        )
    )
    cb_z_im = camera_intr.project(
        Point(
            T_cb_camera.translation
            + T_cb_camera.z_axis * scale,
            frame='zed',
        )
    )
    x_line = np.array([cb_center_im.data, cb_x_im.data])
    y_line = np.array([cb_center_im.data, cb_y_im.data])
    z_line = np.array([cb_center_im.data, cb_z_im.data])

    plt.figure(figsize=(10,10))
    plt.imshow(img_left.data)
    plt.scatter(cb_center_im.data[0], cb_center_im.data[1])
    plt.plot(x_line[:, 0], x_line[:, 1], c="r", linewidth=3)
    plt.plot(y_line[:, 0], y_line[:, 1], c="g", linewidth=3)
    plt.plot(z_line[:, 0], z_line[:, 1], c="b", linewidth=3)
    plt.axis("off")
    plt.title("Chessboard frame in camera %s" % ('zed'))
    plt.show()
    print("Found T_cb_camera")
    print(T_cb_camera)
    print("Using t_cb_world")
    print(H_chess_world)
    T_camera_world = H_chess_world*T_cb_camera.inverse()
    print("Computed T_camera_world")
    print(T_camera_world)
    T_camera_world.save("data/zed_to_world.tf")

if __name__=='__main__':
    register_zed()