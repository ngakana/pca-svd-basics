import logging
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from loguru import logger

import generate_plane_pointcloud as gppc
#######################################################################################
#                        FUNCTION DEFINITIONS
#######################################################################################

def compute_z(x: np.array, y:np.array, u:np.array, v:np.array, p:np.array):
    """
    computes the third coordinate values for points on a plane defined by vector u and v
    :param x: array of x coordinates
    :param y: array of y coordinates
    :param u: any vector on the plane not parallel to v
    :param v: any vector on the plane not parallel to u
    :return: array of z coordinates on the plane
    """

    # compute components of the normal vector: u x v
    i = u[2]*v[1] - u[1]*v[2]
    j = u[0]*v[2] - u[2]*v[0]
    k = u[1]*v[0] - u[0]*v[1]

    z = (-1/k)*(i*(x - p[0]) + j*(y - p[1])) + p[2]

    return z

def plot_mosaic_subplots(xs, ys, zs):
    """
    plots a set of axes on a grid layout within one figure
    :return: a figure and its axes
    """
    centroid = np.mean(np.array(list(zip(xs, ys, zs))), axis=0)

    mosaic = """
    xxxxABC
    xxxxDEF
    xxxxGHI
    """
    # mosaic = [['x,x,x,x,A,B,C'],
    #           ['x,x,x,x,D,E,F'],
    #           ['x,x,x,x,G,H,I']]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(12, 5), constrained_layout=True)
    axd['A'].scatter(xs, xs, s=1, c='orange')
    axd['A'].scatter(centroid[0], centroid[0], c='purple')
    axd['A'].set_title('X vs X')

    axd['B'].scatter(xs, ys, s=1, c='orange')
    axd['B'].scatter(centroid[0], centroid[1], c='purple')
    axd['B'].set_title('X vs Y')

    axd['C'].scatter(xs, zs, s=1, c='orange')
    axd['C'].scatter(centroid[0], centroid[2], c='purple')
    axd['C'].set_title('X vs Z')

    axd['D'].scatter(ys, xs, s=1, c='green')
    axd['D'].scatter(centroid[0], centroid[1], c='purple')
    axd['D'].set_title('Y vs X')

    axd['E'].scatter(ys, ys, s=1, c='green')
    axd['E'].scatter(centroid[1], centroid[1], c='purple')
    axd['E'].set_title('Y vs Y')

    axd['F'].scatter(ys, zs, s=1, c='green')
    axd['F'].scatter(centroid[1], centroid[2], c='purple')
    axd['F'].set_title('Y vs Z')

    axd['G'].scatter(zs, xs, s=1, c='red')
    axd['G'].scatter(centroid[2], centroid[0], c='purple')
    axd['G'].set_title('Z vs X')

    axd['H'].scatter(zs, ys, s=1, c='red')
    axd['H'].scatter(centroid[2], centroid[1], c='purple')
    axd['H'].set_title('Z vs Y')

    axd['I'].scatter(zs, zs, s=1, c='red')
    axd['I'].scatter(centroid[2], centroid[2], c='purple')
    axd['I'].set_title('Z vs Z')

    return fig, axd

#######################################################################################

def plt_points():
    # Data point cloud
    n = 500  # create 1,000 points
    logger.info('Generating x and y coordinates')
    ydata = np.random.randn(n)
    xdata = np.random.randn(n)
    logger.info(f'Finished. {len(xdata)} x coords and {len(ydata)} y coords generated.')
    z = []
    logger.info('Computing z values')
    for i in range(n):
        z_value = (xdata[i] + 4 * ydata[i] + 60) / 30 \
            if random.randint(0, 10) > 6 else \
            (xdata[i] + 4 * ydata[i] + 60 + random.randint(-3, 3)) / 30
        z.append(z_value)
    logger.info(f'Done ✅. {len(z)} z values computed.')
    zdata = np.array(z)

    # PCA
    coords = np.array(list(zip(xdata, ydata, zdata)))
    centroid = np.mean(coords, axis=0)  # compute row-wise mean
    B = coords - centroid  # gaussian(zero-mean) distribution matrix
    C = np.dot(B.T, B)  # covariance matrix - variance and orientation(proportionality relationships)
    U, S, Vt = np.linalg.svd(C)

    # vectors spanning the best-fit plane and its normal vector
    v1 = centroid - U[:, 0]*S[0]
    v2 = centroid - U[:, 1]*S[1]
    n = centroid - np.cross(v1, v2)
    n_svd = centroid - U[:, 2]*S[2]

    # best fit plane
    xcoords, ycoords = np.meshgrid(np.linspace(-5, 5, num=200), np.linspace(-5, 5, num=200))
    zcoords = compute_z(xcoords, ycoords, v1, v2, centroid)

    fig, axd = plot_mosaic_subplots(xdata, ydata, zdata)  # create subplot mosaic

    ax = fig.add_subplot(121, projection='3d')
    ax.scatter3D(xcoords, ycoords, zcoords, s=1, alpha=0.1)  # best fit plane
    ax.scatter3D(xdata, ydata, zdata, c='Green', alpha=0.2, s=2)  # point cloud
    ax.scatter3D(centroid[0], centroid[1], centroid[2], color='Red') # centroid
    ax.quiver3D(centroid[0], centroid[1], centroid[2], v1[0], v1[1], v1[2], normalize=True, color='red', alpha=1)
    ax.quiver3D(centroid[0], centroid[1], centroid[2], v2[0], v2[1], v2[2], normalize=True, color='orange', alpha=1)
    ax.quiver3D(centroid[0], centroid[1], centroid[2], n[0], n[1], n[2], normalize=True, color='k')
    ax.quiver3D(centroid[0], centroid[1], centroid[2], n_svd[0], n_svd[1], n_svd[2], normalize=True, color='green')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    plt_points()
