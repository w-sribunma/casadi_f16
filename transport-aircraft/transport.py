import numpy as np
import casadi as ca
import control
import dataclasses
import scipy


class CasadiDataClass:
    """
    A base class for dataclasses with casadi.
    """

    def __post_init__(self):
        self.__name_to_index = {}
        self.__index_to_name = {}
        for i, field in enumerate(self.fields()):
            self.__name_to_index[field.name] = i
            self.__index_to_name[i] = field.name

    @classmethod
    def fields(cls):
        return dataclasses.fields(cls)

    def to_casadi(self):
        return ca.vertcat(*self.to_tuple())

    def to_tuple(self):
        return dataclasses.astuple(self)

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_casadi(cls, v):
        return cls(*[v[i] for i in range(v.shape[0])])

    @classmethod
    def sym(cls, name):
        v = ca.MX.sym(name, len(cls.fields()))
        return cls(*[v[i] for i in range(v.shape[0])])

    def name_to_index(self, name):
        return self.__name_to_index[name]

    def index_to_name(self, index):
        return self.__index_to_name[index]


@dataclasses.dataclass
class State_TP(CasadiDataClass):
    '''Vehicles State for Transport Aircraft'''
    VT: float = 0  # true velocity, (ft/s)
    alpha: float = 0  # angle of attack, (rad)
    beta: float = 0  # sideslip angle, (rad)
    phi: float = 0  # B321 roll angle, (rad)
    theta: float = 0  # B321 pitch angle, (rad)
    psi: float = 0  # B321 yaw angle, (rad)
    P: float = 0  # body roll rate, (rad/s)
    Q: float = 0  # body pitch rate, (rad/s)
    R: float = 0  # body yaw rate, (rad/s)
    p_N: float = 0  # north position, (m)
    p_E: float = 0  # east position, (m)
    alt: float = 0  # altitude, (m)
    power: float = 0  # power, (0-1)
    ail_deg: float = 0  # aileron position, (deg)
    elv_deg: float = 0  # elevator position, (deg)
    rdr_deg: float = 0  # rudder position, (deg)
    d: float = 0 # Glide-path deviation

@dataclasses.dataclass
class Parameters_TP(CasadiDataClass):
    s: float = 2170.0  # reference area, ft^2
    b: float = 140.0  # wing span, ft
    cbar: float = 17.5  # mean chord, ft
    xcgr: float = 0.35  # reference cg, %chord
    xcg: float = 0.35  # actual cg, %chord
    hx: float = 160.0
    g: float = 32.17  # acceleration of gravity, ft/s^2

    # weight: float = 162000  # weight, slugs
    # axx: float = 9496.0  # moment of inertia about x
    # ayy: float = 55814.0  # moment of inertia about y
    # azz: float = 63100.0  # moment of inertia about z
    # axz: float = 982.0  # xz moment of inertia


class Control_TP(CasadiDataClass):
    '''Control input for Transport Aircraft'''
    thtl: float = 0  # throttle (0-1)
    elv_cmd_deg: float = 0  # elevator command, (deg)
    xcg: float = 0  # actual cg location wrt chord
    land: float =0 # Vehicle configuration: 0 = Clean ; 1 = Landing w/ flaps + gears

def longitudinal(x: State_TP, u: Control_TP):
    ''' 
    Calculate state derivative from state vector and control vector

    Longitudinal Dynamic for medium sized transport aircraft
    pg 182
    Accepts
    States, x: [VT, alpha, theta, Q, Alt, Pos]
    Control, u: [Thtl, elev_deg, xcg, land]
    "xcg" set x-axis position of cg
    "Land configuration" land = clean (0) // landing flaps+gear (1)
    '''
    # Parameters and Constants
    s = 2170
    cbar = 17.5
    mass = 5.0e3
    iyy = 4.1e6
    tstat = 6.0e4
    dtdv = -38.0
    ze = 2.0
    cdcls = 0.042
    cla = 0.085 # per degree
    cma = -0.022 # per degree
    cmde = -0.016 # per degree
    cmq = -16.0  #per radian
    cmadot = -6.0 #per radian
    cladot = 0.0 #per radian
    rtod = 57.29578 #rad to degree
    gd = 32.17 #graviational acceleration ft/s^2
    
    thtl = u[0]
    elev_deg = u[1]
    xcg = u[2]
    land = u[3]
    gam_r = u[4]
    
    vt = x[0]  # True airspeed velocity, ft/s
    alpha = x[1] # AoA, rad
    alpha_deg = rtod*alpha  # angle of attack, deg
    theta = x[2]  # pitch angle, rad
    Q = x[3]  # pitch rate, rad/s
    alt = x[4]  # altitude, ft
    d = x[5] # deviation from glide path (ft)
    
    # Calculate Air data
    r0 = 2.377e-3
    tfac = 1.0 - 0.703e-5*alt
    temperature = ca.if_else(alt > 35000, 390.0, 519.0*tfac)
    rho = r0*(tfac**4.14)
    mach = vt/ca.sqrt(1.4*1716.3*temperature)
    qbar = 0.5*rho*vt**2
    
    qs = qbar*s
    salp = ca.sin(alpha)
    calp = ca.cos(alpha)
    gam = theta - alpha
    sgam = ca.sin(gam)
    cgam = ca.cos(gam)
    
    # Set Landing flaps and gear
    aero = ca.if_else(land,
        (1.0, 0.08, -0.20, 0.02, -0.05),
        (0.2, 0.016, 0.05, 0.0, 0.0))
    cl0 = aero[0]
    cd0 = aero[1]
    cm0 = aero[2] 
    dcdg = aero[3]
    dcmg = aero[4]
    
    thr = (tstat + vt*dtdv)*ca.fmax(thtl, 0) # Thrust
    cl = cl0 + cla*alpha_deg # Nondim lift
    cm = dcmg + cm0 + cma*alpha_deg + cmde*elev_deg + cl*(xcg - 0.25) #Moment
    cd = dcdg + cd0 + cdcls*cl**2 #Drag Polar
    
    # State equations derivatives
    x_dot = ca.SX.zeros(6)
    x_dot[0] = (thr*calp - qs*cd)/mass - gd*sgam
    x_dot[1] = (-thr*salp - qs*cl + mass*(vt*Q + gd*cgam))/(mass*vt + qs*cladot)
    x_dot[2] = Q
    D = 0.5*cbar*(cmq*Q + cmadot*x_dot[1])/vt
    x_dot[3] = (qs*cbar*(cm + D) + thr*ze)/iyy
    x_dot[4] = vt*sgam
    x_dot[5] = vt * ca.sin(gam-gam_r)
    return x_dot

def constrain(s, vt, h, q, gamma,land, gam_r):
    
    # s is our design vector:
    # s = [thtl, elev_deg, alpha]
    thtl = s[0]
    elev_deg = s[1]
    alpha = s[2]
    
    xcg = 0.25  # we assume xcg at 1/4 chord
    theta = alpha + gamma
    d=0 #deviation from glide path

    # vt, alpha, theta, q, h, pos
    x = ca.vertcat(vt, alpha, theta, q, h, d)
    
    # thtl, elev_deg, xcg, land
    u = ca.vertcat(thtl, elev_deg, xcg, land, gam_r)
    return x, u

def objective(s, vt, h, q, gamma,land, gam_r):
    x, u = constrain(s, vt, h, q, gamma,land, gam_r)
    x_dot = longitudinal(x,u)
    return x_dot[0]**2 + x_dot[1]**2+x_dot[2]**2 +x_dot[3]**2

def trim(vt, h, q, gamma, land, gam_r, s0=np.zeros(3)):
    s = ca.SX.sym('s', 3)
    nlp = {'x': s,
           'f': objective(s, vt=vt, h=h, q=q, gamma=gamma,land=land, gam_r=gam_r)}
    S = ca.nlpsol('S', 'ipopt', nlp, {
        'print_time': 0,
        'ipopt': {
            'sb': 'yes',
            'print_level': 0,
            }
        })
    # s = [thtl, elev_deg, alpha]
    res = S(x0=s0,
            lbx=[0, -45, -np.deg2rad(5)],
            ubx=[1, 45, np.deg2rad(25)])
    stats = S.stats()
    if not stats['success']:
        raise ValueError('Trim failed to converge', stats['return_status'])
    s_opt = res['x']
    x0, u0 = constrain(s_opt, vt, h, q, gamma,land, gam_r)
    return {
        'x0': np.array(x0).reshape(-1),
        'u0': np.array(u0).reshape(-1),
        's': np.array(s_opt).reshape(-1),
    }


def linearize(trim):
    x0 = trim['x0']
    u0 = trim['u0']
    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 5)
    y = x
    A = ca.jacobian(longitudinal(x, u), x)
    B = ca.jacobian(longitudinal(x, u), u)
    C = ca.jacobian(y, x)
    D = ca.jacobian(y, u)
    f_ss = ca.Function('ss', [x, u], [A, B, C, D])
    ss = control.ss(*f_ss(x0, u0))

    x_id = {'VT': 0, 'alpha': 1, 'theta': 2, 'q': 3, 'h': 4, 'd': 5}
    y_id = {'VT': 0, 'alpha': 1, 'theta': 2, 'q': 3, 'h': 4, 'd': 5}
    u_id = {'thrtl': 0, 'elev_deg': 1, 'xcg': 2, 'land': 3, 'gam_r':4}
    return {
        'x0': x0,
        'u0': u0,
        'ss': ss,
        'x_id': x_id,
        'y_id': y_id,
        'u_id': u_id,
    }

def get_tf(A, B, C, D, id_in, id_out):
    nums, den = scipy.signal.ss2tf(A, B, C, D, input=id_in)
    num = nums[id_out]
    num[np.abs(num) < 1e-5] = 0
    den[np.abs(den) < 1e-5] = 0
    return control.tf(num, den)

