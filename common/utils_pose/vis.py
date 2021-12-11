import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from config import cfg
from matplotlib import animation 
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
def vis_keypoints(img, kps, kps_lines, joints_name, kp_thresh=0.4, alpha=1):


    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    check = [0 for _ in range(len(joints_name))]
    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0] # point num
        i2 = kps_lines[l][1] # point num
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if check[i1] == 0:
                # cv2.putText(kp_mask, joints_name[i1], p1, 1, fontScale=0.5, color=colors[l], thickness=1)
                cv2.putText(kp_mask, str(i1), p1, 1, fontScale=1, color=colors[l], thickness=1)
                check[i1] = 1
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if check[i2] == 0:
                # cv2.putText(kp_mask, joints_name[i2], p2, 1, fontScale=0.5, color=colors[l], thickness=1)
                cv2.putText(kp_mask, str(i2), p2, 1, fontScale=1, color=colors[l], thickness=1)
                check[i2] = 1

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None, out_dir=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    root_point = kpt_3d[14]
    x_max = root_point[0] + 1000
    x_min = root_point[0] - 1000
    y_max = root_point[1] + 2000
    y_min = 0
    z_max = root_point[2] + 1000
    z_min = root_point[2] - 1000
    

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlim(x_min, x_max)
    ax.set_zlim(y_min, y_max)
    ax.set_ylim(z_min, z_max)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    #ax.legend()
    
    plt.savefig(out_dir + filename)
    plt.cla()
    plt.close(fig)
    

def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None,  out_dir=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])
            ax.set_ylim
            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)


    ax.set_xlim(-3000, 3000)
    ax.set_zlim(-3000, 3000)
    ax.set_ylim(0, 2000)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    #ax.legend()
    
    plt.savefig(out_dir + filename)
    



def vis_3d_multiple_skeleton_gif(kpt_3d, kpt_3d_vis, kps_lines, filename=None,  out_dir=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()
    
    plt.savefig(out_dir + filename)
    

# fig, axs = plt.subplots(ncols=2, figsize=(10, 5), subplot_kw={"projection":"3d"})

# fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}

# def init():
#     for ax, data in zip(axs, [data1, data2]):
#         ydata = "Y1" if ax == axs[0] else "Y2"
#         ax.set_xlabel("X", fontdict=fontlabel, labelpad=16)
#         ax.set_ylabel(ydata, fontdict=fontlabel, labelpad=16)
#         ax.set_title("Z", fontdict=fontlabel)
    
#         ax.scatter(data["X"], data[ydata], data["Z"], 
#                    c=data["Z"], cmap="inferno", s=5, alpha=0.5)
    
#     return fig,

# def animate(i):
#     axs[0].view_init(elev=30., azim=i)
#     axs[1].view_init(elev=30., azim=i)
#     return fig,

# # Animate
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=360, interval=20, blit=True)
# # Save
# anim.save('mpl3d_scatter.gif', fps=30)

