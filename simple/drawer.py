from matplotlib import pyplot as plt


class LineBuilder(object):
    def __init__(self, fig, ax):
        self.xs = []
        self.ys = []
        self.ax = ax
        self.fig = fig
        self.released = 0

    def mouse_click(self, event):
        print('click')
        if not event.inaxes:
            return
        # left click
        if event.button == 1:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            # add a line to plot if it has 2 points

    def mouse_release(self, event):
        print('released')
        if not event.inaxes:
            return
        # left click
        if event.button == 1:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.released += 1

    def mouse_move(self, event):
        if not event.inaxes:
            return
        # dtaw a temporary line from a single point to the mouse position
        # delete the temporary line when mouse move to another position
        if len(self.xs) % 2 == 1:
            line, = self.ax.plot([self.xs[-1], event.xdata], [self.ys[-1], event.ydata], 'g')
            if self.released > 1:
                line.figure.canvas.draw()
            self.ax.lines.pop()


if __name__ == '__main__':
    fig, ax = plt.subplots()

    '''
    'button_press_event'	MouseEvent	mouse button is pressed
    'button_release_event'	MouseEvent	mouse button is released
    'close_event'	CloseEvent	a figure is closed
    'draw_event'	DrawEvent	canvas draw (but before screen update)
    'key_press_event'	KeyEvent	key is pressed
    'key_release_event'	KeyEvent	key is released
    'motion_notify_event'	MouseEvent	mouse motion
    'pick_event'	PickEvent	an object in the canvas is selected
    'resize_event'	ResizeEvent	figure canvas is resized
    'scroll_event'	MouseEvent	mouse scroll wheel is rolled
    'figure_enter_event'	LocationEvent	mouse enters a new figure
    'figure_leave_event'	LocationEvent	mouse leaves a figure
    'axes_enter_event'	LocationEvent	mouse enters a new axes
    'axes_leave_event'	LocationEvent	mouse leaves an axes
    '''

    draw_line = LineBuilder(fig, ax)
    fig.canvas.mpl_connect('button_press_event', draw_line.mouse_click)
    fig.canvas.mpl_connect('motion_notify_event', draw_line.mouse_move)
    fig.canvas.mpl_connect('button_release_event', draw_line.mouse_release)
    ax.set_title('Draw a Line to find its density fucntion')

    plt.gcf().autofmt_xdate()
    plt.show()
