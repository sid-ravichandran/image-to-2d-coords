import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simple_edge_detection(image, plot_flag):
    edges_detected = cv2.Canny(image, 100, 200)
    images = [image, edges_detected]

    if plot_flag:
        location = [121, 122]
        for loc, edge_image in zip(location, images):
            plt.subplot(loc)
            plt.imshow(edge_image, cmap='gray')

    return edges_detected

def edges_to_coordinates(edges,Lx,Ly,write_flag,filename,reduce):
    # filename = string
    # reduce = int (factor by which to reduce the number of points in edges

    filename = filename + '.csv'
    # Create grid of points
    nx = edges.shape[0]
    ny = edges.shape[1]
    x = np.linspace(0,Lx,nx,endpoint=True)
    y = np.linspace(0,Ly,ny,endpoint=True)

    x_p = []
    y_p = []
    for j in range(ny):
        for i in range(nx):
            if edges[i,j] > 0:
                x_p = np.append(x_p,x[i])
                y_p = np.append(y_p,y[j])

    z_p = np.zeros([x_p.shape[0],])
    edge_points = np.vstack([x_p,y_p,z_p])
    df = pd.DataFrame(edge_points.T, columns='x y z'.split())

    df_arranged = arrange_points(df)
    df_arranged_reduced = reduce_points(df_arranged, reduce)

    if write_flag:
        df_arranged_reduced.to_csv(filename,index=False)

    # To ensure that the points when imported into F360, the start and end points are the same
    n = df_arranged_reduced.iloc[:,1].shape[0]
    for i in range(len(df_arranged_reduced.columns)):
        df_arranged_reduced.iloc[n-1,i] = df_arranged_reduced.iloc[0,i]

    return df_arranged_reduced

def arrange_points(df):
    # Function to arrange the points in the coordinate data frame based on distance

    coords = df.to_numpy()
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    n = len(x)

    x_arr, y_arr = [x[0]], [y[0]]
    x, y = np.delete(x,0), np.delete(y,0)
    dist_vec = np.vectorize(dist)

    for i in range(n-1):
        xp, yp = x_arr[i], y_arr[i]
        dist_p = dist_vec(xp,yp,x,y)
        min_p = np.argmin(dist_p)
        x_arr, y_arr = np.append(x_arr,[x[min_p]]), np.append(y_arr,[y[min_p]])
        x, y = np.delete(x,min_p), np.delete(y,min_p)

    arranged_points_array = np.vstack([x_arr,y_arr,z])
    df_arranged = pd.DataFrame(arranged_points_array.T,columns='x y z'.split())
    return df_arranged

def reduce_points(df,reduce):
    # Function to reduce the number of points in coordinate data frame

    coords = df.to_numpy()
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    n = len(x)

    points_array = np.vstack([x,y,z])
    reduced_points_array = points_array[:,np.arange(0,n,reduce)]
    df_reduced = pd.DataFrame(reduced_points_array.T,columns='x y z'.split())

    return df_reduced

def dist(x1,y1,x2,y2):
    # Simple distance between 2 points

    dx = x1-x2
    dy = y1-y2
    return np.sqrt(dx**2 + dy**2)

