import tkinter as tk
import PIL.Image, PIL.ImageTk
import search


class Picker:
    def __init__(self, dem, occupancy_grid, vals, image2world):
        (h, w) = dem.shape[:2]
        self.h, self.w = h, w

        self.window = tk.Tk()
        self.window.bind('<Escape>', self.quit)

        leftFrame = tk.Frame(self.window, width=w, height=h)
        leftFrame.pack(side='left', anchor='n', fill=tk.Y, expand=False)
        rightFrame = tk.Frame(self.window, width=100, height=h)
        rightFrame.pack(side='right', anchor='n', fill=tk.Y, expand=False)

        self.canvas = tk.Canvas(leftFrame, relief='raised', borderwidth=1, width=w, height=h)
        self.canvas.bind('<Button-1>', self.pick)
        self.canvas.bind('<Motion>', self.update_pos)
        self.canvas.pack(side='left', fill=tk.X, expand=True)

        self.info = {'Start': (0, 0), 'Goal': (0, 0)}
        self.label = tk.Label(rightFrame, width=25, height=4, text='', bg='white')
        self.label.pack(side='bottom')
        self.pos = tk.Label(rightFrame, width=25, height=2, text='mouse pos', bg='white')
        self.pos.pack(side='bottom')

        self.to_show = tk.StringVar()
        self.to_show.set('Show occupancy')

        button1 = tk.Button(rightFrame, command=self.quit, width=25, text='Close', bg='white')
        button2 = tk.Button(rightFrame, command=self.toggle_image, width=25, textvariable=self.to_show, bg='white')
        button3 = tk.Button(rightFrame, command=lambda: self.change_mode('Start'), width=25, text='Pick Start', bg= 'white')
        button4 = tk.Button(rightFrame, command=lambda: self.change_mode('Goal'), width=25, text='Pick Goal', bg='white')
        button5 = tk.Button(rightFrame, command=self.compute, width=25, text='Find Path', bg='white')
        button1.pack(side='top')
        button2.pack(side='top')
        button3.pack(side='top')
        button4.pack(side='top')
        button5.pack(side='top')
        self.buttons = {'Toggle': button2, 'Start': button3, 'Goal': button4, 'Find_path': button5}

        self.mode = ''
        self.start = None
        self.goal = None
        self.occupancy_grid = occupancy_grid
        self.dem = dem
        self.vals = vals
        self.path = []
        self.path_graph = []
        self.image2world = image2world

        picture = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(dem))
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=picture)

        self.window.mainloop()

    def toggle_image(self):
        global picture
        if self.to_show.get() == 'Show occupancy':
            self.to_show.set('Show elevation')
            picture = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.occupancy_grid == 0))  # invert it
        else:
            self.to_show.set('Show occupancy')
            picture = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.dem))
        self.canvas.itemconfig(self.canvas_image, image=picture)


    def update_pos(self, event):
        col, row = event.x, event.y
        if 0 <= col < self.w and 0 <= row < self.h:
            x, y = self.image2world(row, col)
            self.pos['text'] = 'Mouse at [%.2f, %.2f, %.2f]' % (x, y, self.vals[row-1, col-1])

    def change_mode(self, mode):
        colors = {'Start': 'red', 'Goal': 'blue'}
        self.mode = mode
        for i in self.buttons:
            self.buttons[i].config(bg='white')
        self.buttons[mode].config(bg=colors[mode])

    def compute(self):
        if not (self.goal and self.start):
            print('No goal or start node selected')
        else:
            start = self.info['Start']
            goal = self.info['Goal']
            self.path = search.astar_search(start, goal, self.occupancy_grid)
            if self.path is None:
                print('No path found')
            self.build_path()

    def build_path(self):
        for i in self.path:
            row, col = i
            self.path_graph.append(self.draw_rect(col, row, 1, 1, 'green'))

    def destroy_path(self):
        for i in self.path_graph:
            self.canvas.delete(i)
        self.path_graph = []

    def pick(self, event):
        col, row = event.x, event.y
        if self.occupancy_grid[row, col] == 1 and self.mode != '':
            print('Grid at %d, %d is occupied' % (row, col))
            return
        if self.mode == 'Start':
            self.destroy_path()
            if not self.start:
                self.start = self.draw_rect(col-1, row-1, 3, 3, 'red')
            self.canvas.moveto(self.start, col-1, row-1)
            self.buttons['Start'].config(bg='white')
            self.info['Start'] = (row, col)

        elif self.mode == 'Goal':
            self.destroy_path()
            if not self.goal:
                self.goal = self.draw_rect(col-1, row-1, 3, 3, 'blue')
            self.canvas.moveto(self.goal, col-1, row-1)
            self.buttons['Goal'].config(bg='white')
            self.info['Goal'] = (row, col)
        self.mode = ''
        self.make_info()

    def quit(self, event=''):
        self.window.quit()

    def draw_rect(self, x, y, w, h, clr):
        return self.canvas.create_rectangle(x, y, x+w, y+h, fill=clr, width=0)

    def make_info(self):
        info = ''
        if self.start:
            row, col = self.info['Start']
            x, y = self.image2world(row, col)
            info += ('Start is at [%.2f, %.2f, %.2f]\n' % (x, y, self.vals[row, col]))
        if self.goal:
            row, col = self.info['Goal']
            x, y = self.image2world(row, col)
            info += ('Goal is at [%.2f, %.2f, %.2f]\n' % (x, y, self.vals[row, col]))
        self.label['text'] = info
