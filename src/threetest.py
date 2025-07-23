import nibabel as nib
import numpy as np
import plotly.graph_objects as go

nii_file = "/home/bolgoff/braintumor/gg.nii"
img = nib.load(nii_file)
data = img.get_fdata()

data = (data - np.min(data)) / (np.max(data) - np.min(data))

fig = go.Figure(data=go.Volume(
    x=np.arange(data.shape[0]),
    y=np.arange(data.shape[1]),
    z=np.arange(data.shape[2]),
    value=data.flatten(),
    opacity=0.1,
    surface_count=20
))

fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    title="3D отображение NIfTI изображения"
)

fig.show()
print("Done")