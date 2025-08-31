import mujoco
import numpy as np

# Load a model from XML
xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 0">
      <geom size="0.1 0.1 0.1" type="box"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Step the simulation
for i in range(10):
    mujoco.mj_step(model, data)
    print("Step:", i, "Box position z:", data.qpos)
