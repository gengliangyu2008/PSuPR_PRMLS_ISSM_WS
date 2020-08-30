import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm

# Setting up the font manager, so that
# it can show japanese characters correctly
fpath = os.path.join(os.getcwd(), "ipam.ttf")
prop = fm.FontProperties(fname=fpath)

# Set up 'ggplot' style
# if want to use the default style, set 'classic'
plt.style.use('ggplot')
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family'] = 'Arial'


# Create a function do plot gray easily
def grayplt(img, title=''):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(img[:, :, 0], cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title(title, fontproperties=prop)
    plt.show()


def plotword(item, data, labels):
    clsname = ['お O', 'き Ki', 'す Su', 'つ Tsu', 'な Na', 'は Ha', 'ま Ma', 'や Ya', 'れ Re', 'を Wo']

    if np.size(labels.shape) == 2:
        lbl = np.argmax(labels[item])
    else:
        lbl = labels[item]

    txt = 'Class ' + str(lbl) + ': ' + clsname[lbl]

    grayplt(data[item], title=txt)


def plt_model_csv(modelname):
    records = pd.read_csv(modelname + '.csv')
    plt.figure()
    plt.subplot(211)
    plt.plot(records['val_loss'])
    plt.yticks([0.00, 0.10, 0.20, 0.30])
    plt.title('Loss value', fontsize=12)

    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(212)
    plt.plot(records['val_acc'])
    plt.yticks([0.93, 0.95, 0.97, 0.99])
    plt.title('Accuracy', fontsize=12)
    plt.show()
