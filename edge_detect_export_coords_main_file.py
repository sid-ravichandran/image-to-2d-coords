import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import edge_detector as edge

img = cv2.imread('snowflake3.jpg',0)
edges = edge.simple_edge_detection(img, plot_flag=1)
points = edge.edges_to_coordinates(edges,Lx=100,Ly=100,write_flag=True,filename='snowflake1_pts',reduce=10)