import pybullet as p
import math
from collections import namedtuple
import sys
import os 
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

class RobotBase(object):
    """
    The base class for robots
    """

    def __init__(self, pos, ori):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (that is, the first `arm_num_dofs` of controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.angles = []

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        
    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo', 
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
    
    def __post_load__(self):
        pass

    def reset(self, ee_pose, ee_yaw):
        self.reset_arm(ee_pose, ee_yaw)
        self.reset_gripper()
        for i in range(100):
            p.stepSimulation()

    def reset_arm(self, ee_pose, ee_yaw):
        """
        reset to rest poses
        
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)
        """
        p.resetBasePositionAndOrientation(self.id, self.base_pos, self.base_ori)
        self.move_ee([ee_pose[0], ee_pose[1], ee_pose[2]+0.3], ee_yaw)
        for i in range(10):
            p.stepSimulation()
        self.move_ee([ee_pose[0], ee_pose[1], ee_pose[2]], ee_yaw)
        self.move_gripper(0.085)

        # Wait for a few steps
        for _ in range(100):
            p.stepSimulation()


    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def move_ee(self, translation, yaw_rotation):
        x, y, z = translation
        roll, pitch, yaw = [0, 1.571, yaw_rotation]
        pos = (x, y, z)
        orn = p.getQuaternionFromEuler((roll, pitch, yaw))
        joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn, self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses, maxNumIterations=1000)
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def getRobotID(self):
        return self.id

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_ee_obs(self):
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return ee_pos

class UR5Robotiq85(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        #with suppress_stdout(): 
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
    
    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}
    
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=100.)
        for i in range(100):
            p.stepSimulation()

        angle = p.getJointState(self.id, self.mimic_parent_id)[0]
        self.angles.append(angle)
        if(min(self.angles) > 0.1):
            print("Excite! I grabbed a block.")
        
        

class UR5Robotiq140(UR5Robotiq85):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                               -1.5707970583733368, 0.0009377758247187636]
        #with suppress_stdout():
        self.id = p.loadURDF('./urdf/ur5_robotiq_140.urdf', self.base_pos, self.base_ori,
                                useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.gripper_range = [0, 0.085]
        # TODO: It's weird to use the same range and the same formula to calculate open_angle as Robotiq85.

    def __post_load__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': -1,
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)
