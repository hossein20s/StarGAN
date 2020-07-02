import time

from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
import pandas_datareader.data as web

class LineBuilder(object):
    def __init__(self, fig, ax):
        self.xs = []
        self.ys = []
        self.ax = ax
        self.fig = fig

    def mouse_click(self, event):
        print('click', event)
        if not event.inaxes:
            return
        #left click
        if event.button == 1:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            #add a line to plot if it has 2 points
            if len(self.xs) % 2 == 0:
                line, = self.ax.plot([self.xs[-2], self.xs[-1]], [self.ys[-2], self.ys[-1]], 'r')
                line.figure.canvas.draw()

        #right click
        if event.button == 3:
            if len(self.xs) > 0:
                self.xs.pop()
                self.ys.pop()
            #delete last line drawn if the line is missing a point,
            #never delete the original stock plot
            if len(self.xs) % 2 == 1 and len(self.ax.lines) > 1:
                self.ax.lines.pop()
            #refresh plot
            self.fig.canvas.draw()

    def mouse_move(self, event):
        if not event.inaxes:
            return
        #dtaw a temporary line from a single point to the mouse position
        #delete the temporary line when mouse move to another position
        if len(self.xs) % 2 == 1:
            line, =self.ax.plot([self.xs[-1], event.xdata], [self.ys[-1], event.ydata], 'r')
            line.figure.canvas.draw()
            self.ax.lines.pop()

if __name__ == '__main__':
    start = dt.datetime(2014, 1, 1)
    end = dt.datetime(2019, 10, 30)
    ticker = 'BNS'

    df = web.DataReader(ticker, 'yahoo', start, end)
    x = df['Adj Close'].index
    y = df['Adj Close'].values
    #number index is need for the cursor position, xaxis is mapped with integer first and converted to date later
    x_num_index = np.arange(0, len(x), 1)

    fig, ax = plt.subplots()
    ax.plot(x_num_index, y)

    draw_line = LineBuilder(fig, ax)
    fig.canvas.mpl_connect('button_press_event', draw_line.mouse_click)
    fig.canvas.mpl_connect('motion_notify_event', draw_line.mouse_move)

    #get default # of lables on x axis
    n = len(ax.get_xticklabels())
    print(n)

    x_format = x.strftime('%b %d %Y')
    xlabels = [x_format[int(i*len(x)/n)] for i in range(0, n)]
    ax.set_xticklabels(xlabels)

    ax.set_ylim([np.min(y), np.max(y)])
    ax.set_title('Closing Price of %s Click to Draw Lines' %(ticker))

    plt.gcf().autofmt_xdate()
    plt.show()
    time.sleep(1000)
