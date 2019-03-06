import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

def livegraphs(i):
    data = open('result.txt', 'r').read()
    lines = data.split("\n")
    frame = []
    count = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(",")
            frame.append(int(x))
            count.append(int(y))
    ax1.clear()
    ax1.plot(frame, count)
    plt.suptitle('Dem so xe may')
    plt.xlabel('frame')
    plt.ylabel('count')
    plt.ylim(0, 50)

ani = animation.FuncAnimation(fig, livegraphs, interval=1000)

plt.show()
