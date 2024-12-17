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
    def __init__(self, model):
        """
        Initialize the DynamicalSystem class.

        Parameters:
            model: A function handle that returns the robot structure.
        """
        robot_structure = model

        self.Model = robot_structure
        self.Name = robot_structure.name
        del self.Model.name

        self.add_states()
        self.add_inputs()

        self.Gravity = robot_structure.gravity
        del self.Model.gravity

        self.InputMap = robot_structure.B
        del self.Model.B

        # testing
        self.HT = {}
        self.Bpos = {}

        self.qpos0 = model.qpos0
        self.qvel0 = model.qvel0

        self.HTransforms = self.homogeneous_transforms()
        self.BodyPositions = self.get_body_positions()
        self.SitePositions = self.get_site_positions()
        self.BodyVelocities = self.get_body_velocities()
        self.Dynamics = ContinuousDynamics(self)

    def add_states(self):
        """
        Add states to the DynamicalSystem object.
        """
        states = {}

        states['q'] = self.Model.q
        states['dq'] = self.Model.qd
        states['ddq'] = self.Model.qdd

        states['q'].ID = 'pos'
        states['dq'].ID = 'vel'
        states['ddq'].ID = 'acc'

        self.States = states

        del self.Model.q
        del self.Model.qd
        del self.Model.qdd

    def add_inputs(self):
        """
        Add inputs to the DynamicalSystem object.
        """
        control = {}

        control['u'] = self.Model.u
        control['u'].ID = 'input'

        self.Inputs = control

        del self.Model.u

    def homogeneous_transforms(self):
        """
        Calculate homogeneous transforms.

        Parameters:
            self: Instance of DynamicalSystem

        Returns:
            T: List of homogeneous transforms
        """
        model = self.Model
        q = self.qpos0

        Xtree = model.Xtree

        Xa = {}
        Xup = {}
        T = []
        Tsite = []

        Xup['world'] = ca.SX.eye(6)
        if model.jtype[0] == 'free':
            XJ, _ = jcalc(model.jtype[0], q[:7])
            Xup[model.body_names[1]] = XJ @ Xtree[0]
            T.append(pluho(Xup[model.body_names[1]]))

        for i in range(2, model.params.nv-4):
            body_name = model.body_names[i]
            XJ, _ = jcalc(model.jtype[i-1], q[i+5])
            Xa[i] = XJ @ Xtree[i]

            if model.parent[body_name] != 'world':
                Xup[body_name] = Xa[i] @ Xup[model.parent[body_name]]

            T.append(pluho(Xup[body_name]))

        R = np.zeros((9,))
        for i in range(model.params.nsite):
            body_name = model.params.site_parent[i]
            site_name = model.site_names[i]
            mp.functions.mju_quat2Mat(R, model.params.site_quat[i])
            R_site = R.reshape(3, 3)
            Xtree[site_name] = np.linalg.inv(np.block([[R_site, np.zeros((3, 3))], [
                skew(model.params.site_pos[i]) @ R_site, R_site]]))
            Xa[site_name] = Xtree[site_name] @ Xup[body_name]

            Tsite.append(pluho(Xa[site_name]))

        return T, Tsite

    def get_body_positions(self):
        """
        Calculate the body positions.

        Parameters:
            self: Instance of DynamicalSystem

        Returns:
            pos_body: List of body positions
        """
        model = self.Model
        nd = model.params.nv

        T, _ = self.HTransforms
        pos_body = []

        for i in range(model.params.nv-5):
            T_i = T[i]
            R_i = T_i[:3, :3].T
            p_i = -R_i @ T_i[:3, 3]

            pos_body.append(p_i)

        return pos_body

    def get_site_positions(self):
        """
        Calculate the site positions.

        Parameters:
            self: Instance of DynamicalSystem

        Returns:
            pos_site: List of site positions
        """
        _, T = self.HTransforms
        pos_site = []

        model = self.Model

        for i in range(model.params.nsite):
            T_i = T[i]
            R_i = T_i[:3, :3].T
            p_i = -R_i @ T_i[:3, 3]

            pos_site.append(p_i)

        return pos_site

    def get_body_velocities(self):
        """
        Calculate the body velocities.

        Parameters:
            self: Instance of DynamicalSystem

        Returns:
            vel_body: List of body velocities
        """
        model = self.Model
        nd = model.params.nv

        q = self.States['q']
        dq = self.States['dq']

        vel_body = []
        S = self.w2velTransform()

        if (model.params.nq == model.params.nv):
            for i in range(nd):
                Jac_1 = ca.jacobian(self.BodyPositions[i][0], q)
                Jac_2 = ca.jacobian(self.BodyPositions[i][1], q)

                vel_1 = Jac_1 @ dq
                vel_2 = Jac_2 @ dq

                vel_body.append((vel_1, vel_2))

            return vel_body
        else:
            for i in range(model.params.nv-5):
                Jac_1 = ca.jacobian(self.BodyPositions[i], q) @ S

                vel_1 = Jac_1 @ dq

                vel_body.append(vel_1)

            return vel_body

    def w2velTransform(self):
        """
        This mapping transforms the qd to the derivatives to position vectors.

        Returns:
            S: Transformation matrix
        """
        model = self.Model
        S = ca.SX.zeros(model.params.nq, model.params.nv)
        Sfree = ca.SX.zeros(7, 6)
        Sfree[:3, :3] = ca.SX.eye(3)
        q = self.States['q']
        p0, p1, p2, p3 = q[3], q[4], q[5], q[6]

        Sfree[3, 3] = -p1/2
        Sfree[3, 4] = -p2/2
        Sfree[3, 5] = -p3/2

        Sfree[4, 3] = p0/2
        Sfree[4, 4] = -p3/2
        Sfree[4, 5] = p2/2

        Sfree[5, 3] = p3/2
        Sfree[5, 4] = p0/2
        Sfree[5, 5] = -p1/2

        Sfree[6, 3] = -p2/2
        Sfree[6, 4] = p1/2
        Sfree[6, 5] = p0/2

        S[:7, :6] = Sfree

        for i in range(6, model.params.nv):
            S[i+1, i] = ca.SX(1)

        return S
