import numpy as np
from scipy.linalg import expm
from scipy.signal import cont2discrete
from contact_ctrl_cpp import ContactControllerParameters, ContactControllerCpp


model_path = "/dex_playground/mujoco_playground/contact_model/test_CQDC_model/models/sdf/allegro_3d_4finger.sdf"

params = ContactControllerParameters()
controller = ContactControllerCpp(model_path, params)

q_init = np.array([
    0.2, 0.95, 1.0, 1.0,                 # index finger
    0.0, 0.6, 1.0, 1.0,                 # middle finger
    -0.2, 0.95, 1.0, 1.0,                # ring finger
    0.5, 1.85, 1.0, 1.0,                 # thumb
    0.0     # ball
])

v_init = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0
])

controller.step(q_init, v_init)
import pdb; pdb.set_trace()

exit(0)

h = 0.001
    
A_c, B_c = np.random.rand(44, 44), np.random.rand(44, 16)
C_c = np.zeros((1, A_c.shape[1]))
D_c = np.zeros((1, B_c.shape[1]))

# use scipy.signal.cont2discrete
A_d1, B_d1, _, _, _ = cont2discrete(
    (A_c, B_c, C_c, D_c), h, method='zoh'
)

# use matrix exponential
A_d2 = expm(A_c * h)
B_d2 = np.linalg.inv(A_c) @ (A_d2 - np.eye(A_d2.shape[0])) @ B_c

# use matrix exponential (for invertible A_c)
AB = np.zeros((44+16, 44+16))
AB[:44, :44] = A_c
AB[:44, 44:] = B_c
AB_exp = expm(AB * h)

print(np.abs(A_d1 - A_d2).max())
print(np.abs(B_d1 - B_d2).max())
