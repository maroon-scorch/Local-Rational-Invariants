
from tkinter import *
from tkinter import ttk
import os

sys.path.insert(1, '../code')
from point import Point
from main import run, closest_grid_point

win=Tk()

point_list = []

screen_width = 400
screen_height = 400
ratio = 20

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

   prev_point[0] = x1;
   prev_point[1] = y1;
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
    canvas.delete('all')
    
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
    
    win.destroy()
    run(point_list, 2)

clearButton = Button(text="Clear", command= clickClearButton)
clearButton.place(x=0, y=450)

runButton = Button(text="Run", command= clickRunButton)
runButton.place(x=200, y=450)

win.mainloop()