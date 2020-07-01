import numpy
from matplotlib import animation, pyplot


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

# Fixing random state for reproducibility
numpy.random.seed(19680801)

if __name__ == '__main__':
    fig = pyplot.figure()


    def f(x, y):
        return numpy.sin(x) + numpy.cos(y)


    x = numpy.linspace(0, 2 * numpy.pi, 120)
    y = numpy.linspace(0, 2 * numpy.pi, 100).reshape(-1, 1)

    im = pyplot.imshow(f(x, y), animated=True)


    def updatefig(*args):
        global x, y
        x += numpy.pi / 15.
        y += numpy.pi / 20.
        im.set_array(f(x, y))
        return im,


    ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
    pyplot.show()
    
    
def xxx():

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    x = numpy.random.rand(10)
    y = numpy.random.rand(10)

    # animation line plot example

    fig = pyplot.figure(4)
    ax = pyplot.axes(xlim=(0, 1), ylim=(0, 1))
    line, = ax.plot([], [], lw=2)


    def init():
        line.set_data([], [])
        return line,


    def animate(i,xxx,yyy):
        print(i)
        line.set_data(x[:i], y[:i])
        return line,

    xxx = 2
    yyy=4
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x) + 1, fargs=(xxx,yyy,),
                                   interval=200, blit=False)
    anim.save('/opt/host/Downloads/1.mp4', writer=writer)

def xx():
    # Set up formatting for the movie files

    fig1 = pyplot.figure()

    data = numpy.random.rand(2, 25)
    l, = pyplot.plot([], [], 'r-')
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)
    pyplot.xlabel('x')
    pyplot.title('test')
    line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                       interval=50, blit=True)
    line_ani.save('lines.mp4', writer=writer)

    fig2 = pyplot.figure()

    x = numpy.arange(-9, 10)
    y = numpy.arange(-9, 10).reshape(-1, 1)
    base = numpy.hypot(x, y)
    ims = []
    for add in numpy.arange(15):
        ims.append((pyplot.pcolor(x, y, base + add, norm=pyplot.Normalize(0, 30)),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                       blit=True)
    im_ani.save('/opt/host/Downloads/im.mp4', writer=writer)

