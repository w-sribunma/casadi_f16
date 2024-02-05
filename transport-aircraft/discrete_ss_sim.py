import control
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from transport import *


class DiscreteStateSpace:
  """
  Use this class to implement any controller you need.
  It takes a continuous time transfer function.
  """
  
  def __init__(self, H, dt):
    sys = control.tf2ss(control.c2d(H, dt))
    self.x = np.zeros((sys.A.shape[0], 1))
    self.A = sys.A
    self.B = sys.B
    self.C = sys.C
    self.D = sys.D
    self.dt = sys.dt

  def update(self, u):
    self.x = self.A.dot(self.x) + self.B.dot(u)
    return self.C.dot(self.x) + self.D.dot(u)
 
  def __repr__(self):
    return repr(self.__dict__)
  

def simulate(control_data, operating_point):
    ''' 
    Simulate Discrete Time State Space problem
    Accepts input of control transfer function (H) and linearized operating point
    '''
    op = operating_point
    x_id = op['x_id']
    y_id = op['y_id']
    u_id = op['u_id']
    H = control_data['H']
    x_sym = ca.SX.sym('x', 6, 1)
    u_sym = ca.SX.sym('u', 4, 1)
    dt = 0.01
    controller = DiscreteStateSpace(H, dt)

    # Construct a Function that integrates over 4s
    x = op['x0']
    print('x0', op['x0'])
    print('u0', op['u0'])
    h = []
    data = {
        'VT': [],
        'alpha': [],
        'theta': [],
        'q': [],
        'h': [],
        'd': [],
        't': [],
    }
    t = 0
    tf = 30 # set final time
    h_desired = 750 #desired h 200
    v_desired = 250 #desired h 500
    while t < tf:
        h_error = h_desired - x[x_id['h']]
        v_error = v_desired - x[y_id['VT']]
        u = np.array(op['u0'])
        u[u_id['thrtl']] += 1*h_error #or h error
        u[u_id['thrtl']] += controller.update(h_error) #or h_error
        F = ca.integrator('F','cvodes',{
            'x': x_sym, 'ode': longitudinal(x_sym, u)},{'tf': dt})
        res = F(x0=x)
        x = res['xf']
        data['VT'].append(x[x_id['VT']])
        data['alpha'].append(x[x_id['alpha']])
        data['theta'].append(x[x_id['theta']])
        data['q'].append(x[x_id['q']])
        data['h'].append(x[x_id['h']])
        data['d'].append(x[x_id['d']])
        data['t'].append(t)
        t += dt
    for k in data.keys():
        data[k] = np.array(data[k])
    print(x)
    return data