"""
DynamicalSystem Module

This module provides a class for handling robot dynamics calculations using MuJoCo and CasADi.
It implements forward kinematics, dynamics calculations, and state/input management for robotic systems.

Author: Premchand
"""

import numpy as np
import casadi as ca
import mujoco_py as mp

from spatial.jcalc import jcalc
from spatial.plnr import plnr
from spatial.rotz import rotz
from spatial.xlt import xlt
from spatial.pluho import pluho
from spatial.skew import skew

from dynamics.continuous_dynamics.ContinuousDynamics import ContinuousDynamics


class DynamicalSystem:
    """
    A class for managing robot dynamics calculations and state transformations.

    This class handles robot dynamics including state management, kinematics calculations,
    and transformation computations using MuJoCo and CasADi.

    Attributes:
        Model: Robot structure model
        Name (str): Name of the robot
        States (dict): Dictionary containing state variables
        Inputs (dict): Dictionary containing input variables
        Gravity: Gravity vector
        InputMap: Input mapping matrix
        HTransforms (tuple): Homogeneous transformations
        BodyPositions (list): List of body positions
        SitePositions (list): List of site positions
        BodyVelocities (list): List of body velocities
        Dynamics: Continuous dynamics object
    """

    def __init__(self, model):
        """
        Initialize the DynamicalSystem with a robot model.

        Args:
            model: Robot structure model containing the robot's parameters and configuration
        """
        self.Model = model
        self.Name = model.name
        del self.Model.name

        # Initialize system components
        self._initialize_states()
        self._initialize_inputs()
        self._initialize_system_parameters(model)

        # Store initial conditions
        self.qpos0 = model.params.data.qpos
        self.qvel0 = model.params.data.qvel

        # Compute transformations and kinematics
        self._compute_all_kinematics()

    def _initialize_states(self):
        """
        Initialize state variables for positions, velocities, and accelerations.
        """
        self.States = {
            'q': self.Model.q,
            'dq': self.Model.qd,
            'ddq': self.Model.qdd
        }

        # Add state identifiers
        self.States['q'].ID = 'pos'
        self.States['dq'].ID = 'vel'
        self.States['ddq'].ID = 'acc'

        # Cleanup model attributes
        del self.Model.q, self.Model.qd, self.Model.qdd

    def _initialize_inputs(self):
        """
        Initialize control input variables.
        """
        self.Inputs = {'u': self.Model.u}
        self.Inputs['u'].ID = 'input'
        del self.Model.u

    def _initialize_system_parameters(self, model):
        """
        Initialize system-wide parameters.

        Args:
            model: Robot structure model
        """
        self.Gravity = model.gravity
        self.InputMap = model.B
        del self.Model.gravity, self.Model.B

        # Initialize transformation dictionaries
        self.HT = {}
        self.Bpos = {}

    def _compute_all_kinematics(self):
        """
        Compute all kinematic quantities including transforms and positions.
        """
        self.HTransforms = self._compute_homogeneous_transforms()
        self.BodyPositions = self._compute_body_positions()
        self.SitePositions = self._compute_site_positions()
        self.BodyVelocities = self._compute_body_velocities()
        self.Dynamics = ContinuousDynamics(self)

    def _compute_homogeneous_transforms(self):
        """
        Calculate homogeneous transformations for the robot structure.

        Returns:
            tuple: (T, Tsite) containing body and site transformations
        """
        model = self.Model
        q = self.qpos0
        Xtree = model.Xtree

        # Initialize transformation dictionaries
        Xa = {}

        # Xup = {}
        Xup0, _ = jcalc(model.jtype[0], q[:7])
        Xup = {'trunk': Xup0}
        T = []
        Tsite = []
        jnt_id = {}
        parent_child_dict = {}
        # Xa = {0: np.eye(6)}

        print("--------------------------------------------------")
        print("---------Xa------------------------------------")
        print("--------------------------------------------------")

        for j, body_name in enumerate(model.body_names):
            parent_child_dict[body_name] = model.params.mj_model.body_id2name(
                model.params.mj_model.body_parentid[j])

        for i, jnt_name in enumerate(model.joint_names):
            jnt_id[jnt_name] = i
            jnt_parent = model.params.mj_model.body_id2name(
                model.params.mj_model.jnt_bodyid[i])
            # body_id = model.params.mj_model.jnt_bodyid[i]
            # parent_id = model.parent[body_id]
            XJ, _ = jcalc(model.jtype[i], q[model.params.jnt_qposaddr[i]                          :model.params.jnt_qposaddr[i]+model.params.jnt_dim[i]])
            Xa[jnt_name] = XJ @ Xtree[jnt_parent]

            if parent_child_dict[jnt_parent] != 'world':
                Xup[jnt_parent] = Xa[jnt_name] @ Xup[parent_child_dict[jnt_parent]]

            T.append(pluho(Xup[jnt_parent]))

            print(i, jnt_name)
            print(Xup[jnt_parent])

        # for i in range(model.params.nj):
        #     body_name = model.body_names[i]
        #     body_id = model.params.body_id[body_name]
        #     XJ, _ = jcalc(model.jtype[i], q[model.params.jnt_qposaddr[i]:model.params.jnt_qposaddr[i]+model.params.jnt_dim[i]])
        #     Xa[body_id] = XJ @ Xtree[i]

        #     if model.parent[body_id] != 0:
        #         Xa[body_id] = Xa[body_id] @ Xa[model.parent[body_id]]
            # T.append(pluho(Xa[body_id]))
        # Handle free joint case
        # if model.jtype[0] == 'free':
        #     XJ, _ = jcalc(model.jtype[0], q[:7])
        #     Xup[1] = XJ @ Xtree[0]
        #     T.append(pluho(Xup[1]))

        # Compute transforms for each body
        # self._compute_body_transforms(model, q, Xtree, Xa, Xup, T)

        # Compute site transforms
        Tsite = self._compute_site_transforms(model, Xup, Xa, Xtree)

        return T, Tsite

    # def _compute_body_transforms(self, model, q, Xtree, Xa, Xup, T):
    #     """
    #     Compute transformations for each body in the robot.

    #     Args:
    #         model: Robot model
    #         q: Joint positions
    #         Xtree: Tree structure transforms
    #         Xa: Link-to-link transforms
    #         Xup: World-to-link transforms
    #         T: List to store transformations
    #     """
    #     for i in range(1, model.params.nb):
    #         body_name = model.body_names[i]
    #         XJ, _ = jcalc(model.jtype[i], q[i+5])
    #         Xa[i] = XJ @ Xtree[i]

    #         if model.parent[i] != 0:
    #             Xup[i] = Xa[i] @ Xup[model.parent[i]]

    #         T.append(pluho(Xup[i]))

    def _compute_site_transforms(self, model, Xup, Xa, Xtree):
        """
        Compute transformations for each site in the robot.

        Args:
            model: Robot model
            Xup: World-to-link transforms
            Xa: Link-to-link transforms

        Returns:
            list: Site transformations
        """
        Tsite = []
        R = np.zeros((9,))

        for i in range(model.params.nsite):
            body_name = model.params.site_parent[i]
            site_name = model.site_names[i]

            # Compute rotation matrix
            mp.functions.mju_quat2Mat(R, model.params.site_quat[i])
            R_site = R.reshape(3, 3)

            # Compute transforms
            Xtree[site_name] = np.linalg.inv(np.block([
                [R_site, np.zeros((3, 3))],
                [skew(model.params.site_pos[i]) @ R_site, R_site]
            ]))
            pname = model.params.mj_model.body_id2name(
                model.params.mj_model.site_bodyid[i])
            Xa[site_name] = Xtree[site_name] @ Xup[pname]

            Tsite.append(pluho(Xa[site_name]))

        return Tsite

    def _compute_body_positions(self):
        """
        Calculate positions for all bodies in the robot.

        Returns:
            list: Body positions
        """
        model = self.Model
        T, _ = self.HTransforms
        pos_body = []

        print("--------------------------------------------------")
        print("---------Body Positions---------------------------")
        print("--------------------------------------------------")

        for i in range(model.params.nv-5):
            T_i = T[i]
            R_i = T_i[:3, :3].T
            p_i = -R_i @ T_i[:3, 3]
            pos_body.append(p_i)

            print(model.body_names[i], p_i)
            print(model.body_names[i], model.params.data.body_xpos[i+1])

        return pos_body

    def _compute_site_positions(self):
        """
        Calculate positions for all sites in the robot.

        Returns:
            list: Site positions
        """
        model = self.Model
        _, T = self.HTransforms
        pos_site = []

        print("--------------------------------------------------")
        print("---------Site Positions---------------------------")
        print("--------------------------------------------------")

        for i in range(model.params.nsite):
            T_i = T[i]
            R_i = T_i[:3, :3].T
            p_i = -R_i @ T_i[:3, 3]
            pos_site.append(p_i)

            print(model.site_names[i], p_i)
            print(model.site_names[i], model.params.data.site_xpos[i])

        return pos_site

    def _compute_body_velocities(self):
        """
        Calculate velocities for all bodies in the robot.

        Returns:
            list: Body velocities
        """
        model = self.Model
        q = self.States['q']
        dq = self.States['dq']
        S = self.w2velTransform()
        vel_body = []

        if model.params.nq == model.params.nv:
            return self._compute_equal_dim_velocities(model, q, dq)
        else:
            return self._compute_different_dim_velocities(model, q, dq, S)

    def _compute_equal_dim_velocities(self, model, q, dq):
        """
        Compute velocities when number of positions equals number of velocities.

        Args:
            model: Robot model
            q: Joint positions
            dq: Joint velocities

        Returns:
            list: Body velocities
        """
        vel_body = []
        for i in range(model.params.nv):
            Jac_1 = ca.jacobian(self.BodyPositions[i][0], q)
            Jac_2 = ca.jacobian(self.BodyPositions[i][1], q)
            vel_1 = Jac_1 @ dq
            vel_2 = Jac_2 @ dq
            vel_body.append((vel_1, vel_2))
        return vel_body

    def _compute_different_dim_velocities(self, model, q, dq, S):
        """
        Compute velocities when number of positions differs from number of velocities.

        Args:
            model: Robot model
            q: Joint positions
            dq: Joint velocities
            S: Velocity transformation matrix

        Returns:
            list: Body velocities
        """
        vel_body = []
        for i in range(model.params.nv-5):
            Jac_1 = ca.jacobian(self.BodyPositions[i], q) @ S
            vel_1 = Jac_1 @ dq
            vel_body.append(vel_1)
        return vel_body

    def w2velTransform(self):
        """
        Compute the transformation matrix from angular velocities to position vector derivatives.

        Returns:
            ca.SX: Transformation matrix
        """
        model = self.Model
        S = ca.SX.zeros(model.params.nq, model.params.nv)
        Sfree = ca.SX.zeros(7, 6)
        Sfree[:3, :3] = ca.SX.eye(3)

        # Get quaternion components
        q = self.States['q']
        p0, p1, p2, p3 = q[3:7]

        # Compute quaternion transformation matrix
        self._compute_quaternion_transform(Sfree, p0, p1, p2, p3)

        # Assemble final transformation matrix
        S[:7, :6] = Sfree
        for i in range(6, model.params.nv):
            S[i+1, i] = ca.SX(1)

        return S

    @staticmethod
    def _compute_quaternion_transform(Sfree, p0, p1, p2, p3):
        """
        Compute the quaternion transformation matrix components.

        Args:
            Sfree: Free joint transformation matrix
            p0, p1, p2, p3: Quaternion components
        """
        # First row
        Sfree[3, 3:6] = [-p1/2, -p2/2, -p3/2]

        # Second row
        Sfree[4, 3:6] = [p0/2, -p3/2, p2/2]

        # Third row
        Sfree[5, 3:6] = [p3/2, p0/2, -p1/2]

        # Fourth row
        Sfree[6, 3:6] = [-p2/2, p1/2, p0/2]
