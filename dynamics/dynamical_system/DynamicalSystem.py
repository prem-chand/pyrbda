import numpy as np
import casadi as ca

from spatial.jcalc import jcalc
from spatial.plnr import plnr
from spatial.rotz import rotz
from spatial.xlt import xlt
from spatial.pluho import pluho


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

        self.HTransforms = self.homogeneous_transforms()
        self.BodyPositions = self.get_body_positions()
        self.BodyVelocities = self.get_body_velocities()
        self.Dynamics = self.continuous_dynamics()

    def add_states(self):
        # Implementation for adding states
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
        # Implementation for adding inputs
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
        q = self.States['q']

        # print(self.States['q'])

        Xtree = model.Xtree

        # return

        Xa = {}
        T = []

        # if first joint is free joint (6-DOF)
        if model.jtype[0] == 'free':
            # pass # TODO
            XJ, _ = jcalc(model.jtype[0], q[:7])
            Xa[0] = XJ @ Xtree[0]
            T.append(pluho(Xa[0]))
            for i in range(1, model.params.nv-6):
                print(model.jtype[i], q[i+6])
                XJ, _ = jcalc(model.jtype[i], q[i+6])
                Xa[i] = XJ @ Xtree[i]
                if model.parent[i] != 0:
                    Xa[i] = Xa[i] @ Xa[model.parent[i]]

                if Xa[i].shape[0] == 3:  # Xa[i] is a planar coordinate transform
                    theta, r = plnr(Xa[i])
                    X = rotz(theta) @ xlt(np.append(r, 0))
                    T.append(pluho(X))
                else:
                    T.append(pluho(Xa[i]))

        else:
            for i in range(model.params.nv):
                print(model.jtype[i], q[i])
                XJ, _ = jcalc(model.jtype[i], q[i])
                Xa[i] = XJ @ Xtree[i]
                if model.parent[i] != 0:
                    Xa[i] = Xa[i] @ Xa[model.parent[i]]

                if Xa[i].shape[0] == 3:  # Xa[i] is a planar coordinate transform
                    theta, r = plnr(Xa[i])
                    X = rotz(theta) @ xlt(np.append(r, 0))
                    T.append(pluho(X))
                else:
                    T.append(pluho(Xa[i]))

        return T

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

        T = self.HTransforms
        pos_body = []

        for i in range(nd-6):
            T_i = T[i]
            R_i = T_i[:3, :3].T
            p_i = -R_i @ T_i[:3, 3]

            # T_next = np.vstack(
            #     (np.hstack((R_i, p_i.reshape(-1, 1))), [0, 0, 0, 1]))
            # p_next = T_next @ np.append(model['body_length']
            #                             [i] * model['body_axis'][i], 1)

            # pos_body.append((p_i, p_next[:3]))
            pos_body.append(p_i)

        return pos_body

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
                Jac_1 = ca.jacobian(self.BodyPositions[i][0] @ S, q)
                Jac_2 = ca.jacobian(self.BodyPositions[i][1] @ S, q)

                vel_1 = Jac_1 @ dq
                vel_2 = Jac_2 @ dq

                vel_body.append((vel_1, vel_2))

            return vel_body
        else:
            # TODO: Implement this
            return vel_body

    def continuous_dynamics(self):
        # Implementation for continuous dynamics
        pass

    def w2velTransform(self):
        """This mapping transforms the qd to the derivatives to position vectors.

        Returns:
            _type_: _description_
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
