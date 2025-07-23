"""Test GL volume tool with MRI data."""

from pyqtgraph.Qt import QtGui, QtWidgets, QtCore
import pyqtgraph.opengl as gl
import numpy as np
from nibabel import load

FILENAME = r"/home/bolgoff/braintumor/MR_Gd.nii.gz"
RENDER_TYPE = "translucent"
THR_MIN = 1
THR_MAX = 2000

# =============================================================================
# Get MRI data
nii = load(FILENAME)
data = np.squeeze(nii.get_fdata())

data[data == 0] = THR_MIN
data[data < THR_MIN] = THR_MIN
data[data >= THR_MAX] = THR_MAX
data -= THR_MIN
data /= THR_MAX - THR_MIN

# (optional) Reorient data
data = data[:, ::-1, :]

# Prepare data for visualization
# Cropping code implemented here. Can specify how much to crop and along what axis
# Prepare data for visualization

half_x = data.shape[0] // 2
cropped_data = data[:half_x, :, :]
d2 = np.zeros(cropped_data.shape + (4,))
d2[..., 3] = cropped_data**1 * 255  # alpha
d2[..., 0] = d2[..., 3]  # red
d2[..., 1] = d2[..., 3]  # green
d2[..., 2] = d2[..., 3]  # blue

# (optional) RGB orientation lines
d2[:40, 0, 0] = [255, 0, 0, 255]
d2[0, :40, 0] = [0, 255, 0, 255]
d2[0, 0, :40] = [0, 0, 255, 255]
d2 = d2.astype(np.ubyte)

# =============================================================================
# Create qtgui
app = QtWidgets.QApplication([])
w = gl.GLViewWidget()
w.setGeometry(0, 0, int(1080/2), int(1920/2))
w.setCameraPosition(distance=120, elevation=0, azimuth=0)
w.pan(0, 0, 10)
w.setWindowTitle(FILENAME)
w.show()

# glOptions are 'opaque', 'translucent' and 'additive'
v = gl.GLVolumeItem(d2, sliceDensity=6, smooth=False, glOptions=RENDER_TYPE)
v.translate(dx=-d2.shape[0]/2, dy=-d2.shape[1]/2, dz=-d2.shape[2]/3)
w.addItem(v)


# =============================================================================
def update_orbit():
    """Rotate camera orbit."""
    global counter
    counter += 1
    w.orbit(1, 0)  # degree


def stop_and_exit():
    """Stop and exit program."""
    app.quit()
    print("Finished")


# =============================================================================
if __name__ == '__main__':
    # Initiate timer
    timer1 = QtCore.QTimer()
    timer2 = QtCore.QTimer()
    counter = 0
    # Connect stuff
    timer1.timeout.connect(update_orbit)
    timer2.timeout.connect(stop_and_exit)

    # Start timer (everytime this time is up connects are excuted)
    NR_FRAMES = 360
    FRAMERATE = 1000 // 2  # ms, NOTE: keep it high to guarantee all frames  # ms, NOTE: keep it high to guarantee all frames
    timer1.start(FRAMERATE)
    timer2.start((NR_FRAMES * FRAMERATE) + 2000)

    # Start program
    QtWidgets.QApplication.instance().exec()