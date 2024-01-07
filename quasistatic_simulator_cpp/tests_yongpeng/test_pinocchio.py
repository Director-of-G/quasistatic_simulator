import numpy as np

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


Identity_SE3 = pin.SE3.Identity()

## load combined model
# urdf_model_path = \
#         "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/robot_obj_combined/allegro_hand_description_right_ball_scene.urdf"

# mesh_dir = \
#         "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/meshes"

# model, collision_model, visual_model = pin.buildModelsFromUrdf(
#     urdf_model_path, mesh_dir
# )

## load models and add the manipuland
robot_urdf_path = "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/robot_single/allegro_hand_description_right.urdf"
object_urdf_path = "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/object_single/sphere_r0.06m.urdf"
robot_srdf_path = "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/srdf/allegro_hand_right_userdefine.srdf"
mesh_dir = "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/meshes"

## planar pushing case
# robot_urdf_path = "/home/yongpeng/research/projects/contact_rich/idto/dex_playground/mujoco_playground/contact_model/test_CQDC_model/models/urdf/dex_2d_robot.urdf"
# object_urdf_path = "/home/yongpeng/research/projects/contact_rich/idto/dex_playground/mujoco_playground/contact_model/test_CQDC_model/models/urdf/dex_2d_obj.urdf"
# mesh_dir = "/home/yongpeng/research/projects/contact_rich/CQDC_model/quasistatic_simulator/models/yongpeng/allegro_hand_description/meshes"

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    robot_urdf_path, mesh_dir
)
object_model, object_collision_model, object_visual_model = pin.buildModelsFromUrdf(
    object_urdf_path, mesh_dir
)

import pdb; pdb.set_trace()

# append manipuland to robot model
object_joint_idx = model.addJoint(0, pin.JointModelFreeFlyer(), Identity_SE3, "object_root_joint")
model.addJointFrame(object_joint_idx)

model.appendBodyToJoint(object_joint_idx, object_model.inertias[-1], Identity_SE3)
object_frame_idx = model.addBodyFrame("object_link", object_joint_idx, Identity_SE3, -1)

object_collision_geom = object_collision_model.geometryObjects[-1]
object_collision_geom.parentFrame = object_frame_idx
object_collision_geom.parentJoint = object_joint_idx

collision_model.addGeometryObject(object_collision_geom)

object_visual_geom = object_visual_model.geometryObjects[-1]
object_visual_geom.parentFrame = object_frame_idx
object_visual_geom.parentJoint = object_joint_idx

visual_model.addGeometryObject(object_visual_geom)

import pdb; pdb.set_trace()

## Display model

viz = MeshcatVisualizer(model, collision_model, visual_model)

try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
 
# Load the robot in the viewer.
viz.loadViewerModel()
 
# Display a robot configuration.
q0 = np.array(
    [
        0.03501504, 0.75276565, 0.74146232, 0.83261002,
        0.63256269, 1.02378254, 0.64089555, 0.82444782,
        -0.1438725, 0.74696812, 0.61908827, 0.70064279,
        -0.06922541, 0.78533142, 0.82942863, 0.90415436,
        0.016, 0.001, 0.071,
        1, 0, 0, 0
    ]
)

viz.display(q0)
viz.displayCollisions(True)
viz.displayVisuals(False)

import pdb; pdb.set_trace()

## Collision detection
collision_model.addAllCollisionPairs()

# remove allegro hand self-collision pairs
pin.removeCollisionPairs(model, collision_model, robot_srdf_path)

data = model.createData()
geom_data = pin.GeometryData(collision_model)
pin.computeCollisions(model, data, collision_model, geom_data, q0, False)

print("------ collision pairs report ------")
for k in range(len(collision_model.collisionPairs)): 
  cr = geom_data.collisionResults[k]
  cp = collision_model.collisionPairs[k]
  import pdb; pdb.set_trace()
#   if cp.first == 0 or cp.second == 0:
  if True:
    print("collision pair: ", cp.first, ", " , cp.second, \
            "distance: ", cr.distance_lower_bound, \
            "- collision:", "Yes" if cr.isCollision() else "No")

import pdb; pdb.set_trace()
