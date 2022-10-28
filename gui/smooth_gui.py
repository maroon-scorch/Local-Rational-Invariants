import itertools
from tkinter import *
from tkinter import ttk
import os

sys.path.insert(1, '../code')
from data.point import *
from main import run
from util import closest_grid_point, visualize

import numpy as np

win=Tk()

point_list = []

screen_width = 400
screen_height = 400
ratio = 5

# Setup of the window
win.geometry("400x500")
win.resizable(False, False)
win.title('GUI')

prev_point = [None, None]

def draw_line(event):
   x1=event.x
   y1=event.y
   x2=event.x
   y2=event.y
   
   # Draw an oval in the given co-ordinates
   canvas.create_oval(x1,y1,x2,y2,fill="black", width=10)
   
   if prev_point != [None, None]:
       canvas.create_line(prev_point[0], prev_point[1], x1, y1, fill="black", width=5)

   prev_point[0] = x1
   prev_point[1] = y1
   cartX = x1 - screen_width/2
   cartY = screen_height/2 - y1
   cartPoint = closest_grid_point(Point(cartX/ratio, cartY/ratio))
   
   print(cartPoint)
   
   point_list.append(cartPoint)

canvas=Canvas(win, width=screen_width, height=screen_height, background="white")
canvas.grid(row=0, column=0)
canvas.bind('<Button-1>', draw_line)
click_num=0

def clickClearButton():
    point_list.clear()
    prev_point[0] = None
    prev_point[1] = None
    canvas.delete('all')
    
def refine(points):
    refined_list = []
    for idx, point in enumerate(points):
        if idx != len(points) - 1:
            next_point = points[idx + 1]
            mid_pt = midpoint_of(point, next_point)
            refined_list.append(point)
            refined_list.append(mid_pt)
        else:
            refined_list.append(point)
    return refined_list

def project(idx, points):
    prev_point = points[idx - 1]
    curr_point = points[idx]
    next_point = points[idx + 1]
    
    v1 = curr_point.vec - prev_point.vec
    v2 = next_point.vec - prev_point.vec
    
    proj_vec = np.dot(v1, v2)/np.dot(v2, v2) * v2 + prev_point.vec
    proj_pt = Point(proj_vec[0], proj_vec[1])
    
    return proj_pt
    
    
def smoothify(points):
    result = []
    # if closed curve
    for idx, _ in enumerate(points):
        if idx != len(points) - 1:
            projected_point = project(idx, points)
            result.append(projected_point)
        else:
            result.append(result[0])
    
    return result

def approx(points, scale):
    result = []
    for pt in points:
        current_x = int(round(scale*pt.x))
        current_y = int(round(scale*pt.y))
        result.append(Point(current_x, current_y))
    return result
        
    
def clickRunButton():
    print(point_list)
    
    filename = "output.txt"
    file = open(filename,'wt')
    file.write("2\n")
    
    for point in point_list:
        px = point.x
        py = point.y
        line = str(px) + " 1 " + str(py) + " 1\n"
        file.write(line)
    
    result = point_list
    if result[0] != result[-1]:
        result.append(result[0])
    iter = 4
    while iter > 0:
        result = refine(result)
        result = smoothify(result)
        iter = iter - 1
    result = approx(result, 10)
    result = [k for k, g in itertools.groupby(result)]
    # visualize(result, "Title", True)
    
    win.destroy()
    run(result, 2, True)

clearButton = Button(text="Clear", command= clickClearButton)
clearButton.place(x=0, y=450)

runButton = Button(text="Run", command= clickRunButton)
runButton.place(x=200, y=450)

win.mainloop()