import numpy as np
from numpy import *
from scipy.io import loadmat
from scipy.io import savemat
from math import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import sign

# rudder derivatives


def kcs_rudder_derivatives():
    d_em = 10.8 / scale
    A_R = 0.0182 * L * d_em
    w = 1 - 0.645   # wake fraction
    n = 18.15   # propeller speed
    Dp = 7.9 / scale    # propeller diameter
    lambdaa = 2.164
    eta = 0.7979
    epsilon = 0.956
    kappa = 0.633
    u = 0.26 * sqrt(g * L)
    J1 = u * (1 - w) / (n * Dp)
    K_T = polyval(pt, J1)
    uR = u * epsilon * (1 - w) * sqrt(eta * (1 + kappa * (sqrt(1 + 8 * K_T / pi / J1 * 2) - 1)) * 2 + (1 - eta))
    f_alpha = 6.13 * lambdaa / (2.25 + lambdaa)
    FN = 1 / 2 * rho * f_alpha * A_R * uR ** 2
    xH = -0.436
    aH = 0.361
    xR = -0.5
    Yd = - (1 + aH) * FN
    Nd = -(xR + aH * xH) * L * FN
    return Yd, Nd


def kcs_controller_gains():
    k1 = 0.3
    t1 = 1.28
    wn = 0.48
    z = 1.9
    Kp = 2
    Kd = 4
    print("kp, kd", Kp, Kd)
    return Kp, Kd


def saturate(x, a):
    if abs(x) > a:
        x = sign(x) * a
    return x


def kcs_hull_force(up, vp, rp):
    b = math.atan2(-vp, up)

    # Surge Hydrodynamic Derivatives in non-dimensional form
    X0 = -0.0167
    Xbb = -0.0549
    Xbr_minus_my = -0.1084
    Xrr = -0.0120
    Xbbbb = -0.0417

    # Sway Hydrodynamic Derivatives in non-dimensional form
    Yb = 0.2252
    Yr_minus_mx = 0.0398
    Ybbb = 1.7179
    Ybbr = -0.4832
    Ybrr = 0.8341
    Yrrr = -0.0050

    # Yaw Hydrodynamic Derivatives in non-dimensional form
    Nb = 0.1111
    Nr = -0.0465
    Nbbb = 0.1752
    Nbbr = -0.6168
    Nbrr = 0.0512
    Nrrr = -0.0387

    # Non-dimensional Surge Hull Hydrodynamic Force
    Xp_H = X0 * (up * 2) + Xbb * (b * 2) + Xbr_minus_my * b * rp + Xrr * (rp * 2) + Xbbbb * (b * 4)

    # Non-dimensional Sway Hull Hydrodynamic Force
    Yp_H = Yb * b + Yr_minus_mx * rp + Ybbb * (b * 3) + Ybbr * (b * 2) * rp + Ybrr * b * (rp * 2) + Yrrr * (rp * 3)

    # Non-dimensional Yaw Hull Hydrodynamic Moment
    Np_H = Nb * b + Nr * rp + Nbbb * (b * 3) + Nbbr * (b * 2) * rp + Nbrr * b * (rp * 2) + Nrrr * (rp * 3)

    # Non-dimensional force vector
    tau_H = [Xp_H, Yp_H, Np_H]
    return tau_H


def kp505_model(u, n_prop):
    # Effective Wake Fraction of the Propeller
    wp1 = 1 - 0.645             # ratio of the speed difference to the observed speed ((V-Va)/V)

    # Thrust Deduction Factor
    tp = 1 - 0.793

    # Propeller diameter
    Dp1 = 7.9 / 75.5

    # Advance Coefficient
    J1 = u * (1 - wp1) / (n_prop * Dp1)

    # Thrust Coefficient
    Kt1 = polyval(pt, J1)

    # Thrust
    T1 = (1 - tp) * rho * Kt1 * Dp1 * 4 * n_prop * 2

    # Propeller force vector (dimensional)
    tau_P = [T1, 0, 0]
    return tau_P


def kcs_rudder_force(u, v, r, delta, n_prop):
    # Effective Wake Fraction of the Propeller
    wp1 = 1 - 0.645

    # Propeller diameter
    Dp1 = 7.9 / 75.5

    # Advance Coefficient
    #print("u is", u)
    J1 = u * (1 - wp1) / (n_prop * Dp1)

    # Thrust Coefficient
    Kt = polyval(pt, J1)

    # Non-dimensionalize velocities
    up = u / U
    vp = v / U
    rp = r * L / U

    # Rudder Force Calculation
    A_R = L * d_em / 54.86
    Lamdaa = 2.164
    f_alp = 6.13 * Lamdaa / (2.25 + Lamdaa)

    eps = 0.956
    eta = 0.7979
    kappa = 0.633
    xp_P = -0.4565  # Assuming propeller location is 10 m ahead of AP (Rudder Location)
    xp_R = -0.5

    b = math.atan2(-v, u)
    b_p = b - xp_P * rp

    if b_p > 0:
        gamma_R = 0.492
    else:
        gamma_R = 0.338

    lp_R = -0.755

    # Non-dimensional flow velocities at the rudder
    up_R = eps * (1 - wp1) * up * sqrt(eta * (1 + kappa * (sqrt(1 + 8 * Kt / (pi * (J1 * 2))) - 1)) * 2 + (1 - eta))
    vp_R = gamma_R * (vp + rp * lp_R)

    Up_R = sqrt(up_R * 2 + vp_R * 2)
    alpha_R = delta - math.atan2(-vp_R, up_R)

    F_N = A_R / (L * d_em) * f_alp * (Up_R ** 2) * sin(alpha_R)

    tR = 1 - 0.742
    aH = 0.361
    xp_H = -0.436

    # Non-dimensional forces and moments
    Xp_R = - (1 - tR) * F_N * sin(delta)
    Yp_R = - (1 + aH) * F_N * cos(delta)
    Np_R = - (xp_R + aH * xp_H) * F_N * cos(delta)

    #   Non-dimensional rudder force vector
    tau_R = [Xp_R, Yp_R, Np_R]
    return tau_R


def kcs_guidance_waypoint(t, x):
    global wp_flag, psid, ypd_int
    #print("wp flag", wp_flag)
    #print("wp1", wp_flag[0])
    # look-ahead distance
    if all(wp_flag):
        psid = 0
        ypd_int = 0
    else:
        d_LA = 2 * L
        # Tolerance of each waypoint. When the ship is closer than this value to
        # the destination waypoint, you may switch to the next waypoint.
        R_tol = 3*L
        # Design Parameter for ILOS
        k = 0.05

        xn = x[3]
        yn = x[4]
        yp_int = x[8]  # For implementing ILOS
        # waypoint switching
        print("wpflag",wp_flag)
        j = 0
        while wp_flag[j] == 1:
            j = j+1
            #print(j)
            #print("inside while")
            print("j is", j)
            #print("loop i ", i)

        i = j-1

        #print("i is", i)

        kp = 1/d_LA
        ki = kp*k
        pi_p = math.atan2(wp[i+1][1]-wp[i][1], wp[i+1][0]-wp[i][0])         # path tangential angle
        ype = -sin(pi_p)(xn-wp[i][0]) + cos(pi_p)(yn-wp[i][1])            # cross track error
        xpe = cos(pi_p)(xn-wp[i][0]) + sin(pi_p)(yn-wp[i][1])            # along track error
        psid = (pi_p - arctan(kp*ype+ki*yp_int))                              # desired heading angle
        ypd_int = (d_LA*ype)/(d_LA*2 + (ype + k*yp_int)*2)                # rate of change of cross track error

        d_wn = sqrt((wp[i+1][1] - wp[i][1])*2 + (wp[i+1][0]-wp[i][0])*2)

        ype_array.append(ype)
        abs_ype_array.append(abs(ype))
        xpe_array.append(xpe)
        abs_xpe_array.append(abs(xpe))
        psid_array.append(ssa(psid)*180/pi)
        t_array.append(t)


        if (d_wn - xpe) <= R_tol:
            print("If called", d_wn-abs(xpe), R_tol)
            wp_flag[i+1] = 1

        else:
            wp_flag[i+1] = 0



    #print("ypdint is", ypd_int)
    if wp_flag[len(wp)-1] == 1:
        terminate_flag = 1

    return psid, ypd_int


def ssa(x):
    if x > pi:
        return x - 2*pi
    elif x < -pi:
        return 2*pi + x
    else:
        return x


def kcs_control(t, psid, x):
    n_c =18.15             #16.3                  commanded propeller speed
    psi = x[5]
    r = x[2]
    Kp, Kd = kcs_controller_gains()
    e = ssa(psi - psid)               # error
    #print("e, r is", e, r)
    delta_c = -Kp * e - Kd * r              # commanded rudder angle
    return saturate(delta_c, 35*pi/180), n_c


def kcs_ode(t, x):
    #print("x is", x)

    u = x[0]
    v = x[1]
    r = x[2]
    xn = x[3]
    yn = x[4]
    psi = x[5]
    delta = x[6]
    n_prop = x[7]

    # non-dimensional velocities
    up = u / U
    vp = v / U
    rp = r * L / U
    #print("U L dem dsp", U, L, d_em, Dsp)

    # saturate rudder angle
    delta = saturate(delta, delta_max)

    # non-dimensional hull force vector
    tau_H = kcs_hull_force(up, vp, rp)
    #print("tau h isssssssssssssssssssssssss ", tau_H)

    # Dimensionalize hull force vector
    tau_H[0] = tau_H[0] * (1 / 2 * rho * U ** 2 * L * d_em)
    tau_H[1] = tau_H[1] * (1 / 2 * rho * U ** 2 * L * d_em)
    tau_H[2] = tau_H[2] * (1 / 2 * rho * U * 2 * L * 2 * d_em)

    # Dimensional propeller force vector
    tau_P = kp505_model(u, n_prop)

    #print("tau p isss", tau_P)
    # Non-dimensional rudder force vector
    tau_R = kcs_rudder_force(u, v, r, delta, n_prop)
    #print("tau rrrrr", tau_R)
    # Dimensionalize rudder force vector
    tau_R[0] = tau_R[0] * (1 / 2 * rho * U ** 2 * L * d_em)
    tau_R[1] = tau_R[1] * (1 / 2 * rho * U ** 2 * L * d_em)
    tau_R[2] = tau_R[2] * (1 / 2 * rho * U * 2 * L * 2 * d_em)

    # Coriolis terms
    mp = Dsp / (0.5 * d_em * (L ** 2))
    xGp = rG[0] / L

    # Non-dimensional Coriolis forces and moments
    Xp_C = mp * vp * rp + mp * xGp * (rp ** 2)
    Yp_C = -mp * up * rp
    Np_C = -mp * xGp * up * rp

    # Non-dimensional Coriolis force vector
    tau_C = [Xp_C, Yp_C, Np_C]

    # Dimensionalize Coriolis force vector
    tau_C[0] = tau_C[0] * (1 / 2 * rho * U ** 2 * L * d_em)
    tau_C[1] = tau_C[1] * (1 / 2 * rho * U ** 2 * L * d_em)
    tau_C[2] = tau_C[2] * (1 / 2 * rho * U * 2 * L * 2 * d_em)
    # wind forces
    wind_f = 13.74 #* (1 / 2 * rho * U ** 2 * L * d_em)
    x_wind = wind_f
    y_wind = wind_f
    r_wind = 0
    tau_wind = [x_wind,y_wind,r_wind]
    print(wind_f)
    # Dimensional total force vector
    tau = zeros(3)
    tau[0] = tau_H[0] + tau_P[0] + tau_R[0] + tau_C[0] + tau_wind[0]
    tau[1] = tau_H[1] + tau_P[1] + tau_R[1] + tau_C[1] + tau_wind[1]
    tau[2] = tau_H[2] + tau_P[2] + tau_R[2] + tau_C[2] +  tau_wind[2]
    tau = transpose(tau)
    print('tau is ', tau)

    #print("tau isssssssssssssssssssssss ", tau)

    # Dimensional total mass matrix
    M = M_RB + M_A
    #print("M is", M)

    # Derivative of velocities
    vel_der = linalg.solve(M, tau)
    vel_der = transpose(vel_der)
    #print("velder", vel_der)

    # Derivative of state vector
    xd = zeros(9)

    xd[0] = vel_der[0]
    xd[1] = vel_der[1]
    xd[2] = vel_der[2]
    xd[3] = u * cos(psi) - v * sin(psi)
    xd[4] = u * sin(psi) + v * cos(psi)
    xd[5] = r
    #print("xd is ", xd[0:5])
    # Desired trajectory from guidance system
    psid, ypd_int = kcs_guidance_waypoint(t, x)
    xd[8] = ypd_int

    # Commanded rudder angle from control system
    delta_c, n_c = kcs_control(t, psid, x)

    # Rudder rate
    deltad = saturate((delta_c - delta) / rudderTC, deltad_max)
    xd[6] = deltad

    # Propeller rate
    nd = (n_c - n_prop) / propTC
    xd[7] = nd
    #print("xd is", xd)
    #print("wp flag is", wp_flag)
    #print("t iss", t)
    delta_array.append(delta)
    t_ode.append(t)

    psi_array.append(ssa(psi)*180/pi)
    return xd


scale = 75.5  # scale
rho = 1000  # density
g = 9.80665  # acc due to gravity
L = 230 / scale  # length
B = 32.2 / scale  # breadth
T = 10.8 / scale  # draft
Cb = 0.651  # block coefficient
Fn = 0.26  # Froude number
d_em = 10.8 / scale
A_R = 0.0182 * L * d_em
w = 1 - 0.645   # wake fraction
n_prop = 18.15  # propeller speed (rps)
Dp = 7.9 / scale    # propeller diameter
lambdaa = 2.164
eta = 0.7979
epsilon = 0.956
kappa = 0.633

# design speed
U = Fn * sqrt(g * L)

# displacement
Dsp = Cb * L * B * T

# mass
m = rho * Dsp

# center of gravity
xG = -1.48 * L / 100
yG = 0
zG = -3.552 / 75.5
rG = [xG, yG, zG]

# yaw mass moment of inertia
Iz = m * ((0.25 * L) * 2 + xG * 2)

# mass matrix
M_RB = [[m, 0, 0], [0, m, -m * rG[0]], [0, -m * rG[0], Iz]]

# added mass matrix
file2 = loadmat(r"C:\Users\anand\Downloads\KCS_Endsem (3)\KCS_hydra.mat")

#print(file2)
M_A_6 = file2['AM']
M_A = zeros([3, 3])
M_A[0][0] = M_A_6[1][0][0] / scale ** 3
M_A[0][1] = M_A_6[1][0][1] / scale ** 3
M_A[0][2] = M_A_6[1][0][5] / scale ** 4
M_A[1][0] = M_A_6[1][1][0] / scale ** 3
M_A[1][1] = M_A_6[1][1][1] / scale ** 3
M_A[1][2] = M_A_6[1][1][5] / scale ** 4
M_A[2][0] = M_A_6[1][5][0] / scale ** 4
M_A[2][1] = M_A_6[1][5][1] / scale ** 4
M_A[2][2] = M_A_6[1][5][5] / scale ** 5
#print("M A issss", M_A)

# linear damping matrix
D_L = zeros([3, 3])
D_L[1][1] = 0.5 * rho * L * T * U * (-0.2252)
D_L[2][2] = 0.5 * rho * L ** 3 * T * U * (-0.0465)
D_L[1][2] = 0.5 * rho * L ** 2 * T * U * 0.0398
D_L[2][1] = 0.5 * rho * L ** 2 * T * U * (-0.1111)

# propeller parameters
pot_data = array([
    [0.000, 0.5327, 0.7517],
    [0.100, 0.4937, 0.7058],
    [0.150, 0.4719, 0.6813],
    [0.200, 0.4469, 0.6538],
    [0.250, 0.4208, 0.6232],
    [0.300, 0.3922, 0.5895],
    [0.350, 0.3657, 0.5589],
    [0.400, 0.3425, 0.5314],
    [0.450, 0.3143, 0.5008],
    [0.500, 0.2895, 0.4702],
    [0.550, 0.2647, 0.4396],
    [0.600, 0.2407, 0.4090],
    [0.650, 0.2162, 0.3784],
    [0.700, 0.1931, 0.3478],
    [0.700, 0.1943, 0.3478],
    [0.750, 0.1688, 0.3172],
    [0.800, 0.1414, 0.2805],
    [0.850, 0.1148, 0.2468],
    [0.900, 0.0870, 0.2132],
    [0.950, 0.0581, 0.1704],
    [1.000, 0.0293, 0.1275],
    [1.050, -0.0033, 0.0786]])

#print(shape(pot_data))

#print(pot_data)
#print(pot_data[:, 1])
(m, n11) = (shape(pot_data))
#print(m)
J = zeros([22, 1])
J[:, 0] = pot_data[:, 0]
#print("J is", J)
kt = zeros([22, 1])
kq = zeros([22, 1])
kt[:, 0] = pot_data[:, 1]
kq[:, 0] = pot_data[:, 2] / 10
#print("kt is", kt)
#print("kq is", kq)
pt = polyfit(J[:, 0], kt[:, 0], 2)
pq = polyfit(J[:, 0], kq[:, 0], 2)
#print("pt is ", pt, "pq is ", pq)

# Rudder Hydrodynamic Derivatives for state estimation model
kcs_rudder_derivatives()
Yd, Nd = kcs_rudder_derivatives()
#print(Yd, "yd and nd ", Nd)

# Maximum rudder and rudder rate
delta_max = 35 * pi / 180
deltad_max = 35 * pi / 180

# Rudder time constant
rudderTC = 0.25

# Propeller time constant
propTC = 0.1

# Measurement noise covariance
xn_noise_sig = 1
yn_noise_sig = 1
psi_noise_sig = 5 * pi / 180

# Waypoints
# Waypoints
# wp1 = [-12,-16]
# wp2 = [0,20]
# wp3 = [12,-16]
# wp4 = [-19,6]
# wp5 = [19,6]
# wp = array([wp1,wp2,wp3,wp4,wp5,wp1
#             ])
# main wp
wp = array([
    [-35, -35],
    [-20, -20],
    [20, -20],
    [20, 20],
    [-20, 20],
    # [-15,5],
    [-20, -20],
])
#star traj
wp0 = [-35, -35]
wp1 = [0,25]
wp2 = [24,8]
wp3 = [15,-20]
wp4 = [-15,-20]
wp5 = [-24,8]


#
# wp = np.array([[0,10], [5.878, -8.09], [-9.511,3.09],[9.511, 3.09],[-5.878, -8.09], [0,10]])*3.05
# wp = array([wp0, wp4, wp1, wp3, wp5, wp2, wp4
#             ])



# Cross track error
ype_array = []
xpe_array = []
abs_ype_array = []
abs_xpe_array = []
psid_array = []
psi_array = []
t_array = []
delta_array = []
t_ode = []
wp_flag = zeros(len(wp))
terminate_flag = 0

Tmax = 300
dt = 0.01

# Note: t = 0 to Tmax
# initial condition and uncertainty
x0 = transpose([U, 0, 0, -35, -35, pi/4, 0, 1, 0]) # u v r xn yn psi delta n_prop yp_int


# simulate and estimate state in a time loop
# options
t = linspace(0, 500, 50000)


def way_points_reached(t, x):
    return sum(wp_flag) - len(wp_flag)


way_points_reached.terminal = True
sol = solve_ivp(kcs_ode, (0, 500), x0, events=way_points_reached)

# print(ss)
y = sol['y']
print("y is ", sol['y'])

t_points = sol['t']
t_called_psi = linspace(0,max(t_points),size(psi_array))
t_called_ype = linspace(0,max(t_points),size(ype_array))
# Outputs
u = y[0][:]
v = y[1][:]
r = y[2][:]
xn = y[3][:]
yn = y[4][:]
psi = y[5][:]
delta = y[6][:]
n_prop = y[7][:]
yint = y[8][:]
print("u is ", u)
# Desired outputs for comparison
xn_des = wp[:,0]
yn_des = wp[:,1]



avg = average(abs_ype_array)
maxi = max(abs_ype_array)
mini = min(abs_ype_array)


Kp, Kd = kcs_controller_gains()

# # plt.figure()
# # plt.plot(xn_des, yn_des, label="desired path",linewidth=2.5)
# # plt.plot(xn, yn, label="actual path",linewidth=2.5)
# # plt.xlabel('x in m',fontsize=25)
# # plt.ylabel('y in m',fontsize=25)
# # plt.tick_params(axis='x', labelsize=15)
# # plt.tick_params(axis='y', labelsize=15)
# # #plt.axes.set_aspect(equal())
# # #plt.text(-36, 10, "Kp:     " + str(Kp) + "\nKd:     " + str(Kd) + "\nype avg:    " + str(avg) + "\nype max:    " + str(maxi), fontsize=10)
# # plt.gca().set_aspect('equal')
# # plt.title('waypoint tracking',fontsize=25)
# # plt.legend(fontsize=25)
# # plt.show()

savemat("x_y_position.mat", {'xn': xn, 'yn': yn})
plt.figure()
plt.plot(xn, yn, linewidth=2.5, color='red')    # label="Actual Path"
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.grid(linewidth =1, color='gray',linestyle='dotted')
plt.gca().set_aspect('equal')
plt.title('Waypoint Tracking PD',fontsize=25)
plt.xlabel('x in m', fontsize=25)
plt.ylabel('y in m', fontsize=25)
# plt.xlim(-60,10)
# plt.ylim(-30,40)

x_ship = array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0, 0]) *5
y_ship = 16.1 / 230 * array([-1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1]) *5

m1 = 6
for i in range(m1):
    m_indx = i * (len(xn) // m1)
    psi_new = psi[m_indx]
    x_new_ship = xn[m_indx] + x_ship * cos(psi_new) - y_ship * sin(psi_new)
    y_new_ship = yn[m_indx] + x_ship * sin(psi_new) + y_ship * cos(psi_new)

    plt.plot(x_new_ship,y_new_ship,'k', linewidth=2.5)
plt.plot(wp[:,0],wp[:,1], 'b*', markersize=15) # ,label='Waypoints'
plt.plot(xn_des, yn_des, label="desired path")
plt.legend(fontsize=20,loc='upper right')
plt.show()

#
# # plt.figure()
# # plt.plot(t[0:size(r)], r,linewidth=2.5)
# # plt.xlabel('Time in s',fontsize=25)
# # plt.ylabel('Yaw Rate in rad/s',fontsize=25)
# # plt.title('Yaw Rate vs Time',fontsize=25)
# # plt.tick_params(axis='x', labelsize=15)
# # plt.tick_params(axis='y', labelsize=15)
# # plt.show()
#
# savemat("delta_vs_t.mat", {'t_points': t_points, 'delta': delta*180/pi})
# plt.figure()
# plt.plot(t_points, -delta*180/pi,linewidth=2.5)
# plt.grid(linewidth =1, color='gray',linestyle='dotted')
# plt.xlabel('Time in s',fontsize=25)
# plt.ylabel('Rudder angle in degrees',fontsize=25)
# plt.title('Rudder angle vs Time',fontsize=25)
# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=15)
# plt.show()
#
# savemat("u_vs_t.mat", {'t_points': t_points, 'u': u})
# plt.figure()
# plt.plot(t_points, u,linewidth=2.5)
# plt.grid(linewidth =1, color='gray',linestyle='dotted')
# plt.xlabel('Time in s',fontsize=25)
# plt.ylabel('Surge velocity (m/s)',fontsize=25)
# plt.title('u vs Time',fontsize=25)
# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=15)
# plt.show()
#
# savemat("v_vs_t.mat", {'t_points': t_points, 'v': v})
# plt.figure()
# plt.plot(t_points, v,linewidth=2.5)
# plt.grid(linewidth =1, color='gray',linestyle='dotted')
# plt.xlabel('time in s',fontsize=25)
# plt.ylabel('sway velocity (m/s)',fontsize=25)
# plt.title('v vs time',fontsize=25)
# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=15)
# plt.show()
#
# ssa_psi = np.vectorize(ssa)
# savemat("psi_vs_t.mat", {'t_points': t_points, 'psi': ssa_psi(psi)*180/pi})
# savemat("psid_vs_t.mat", {'t_array': t_array[1:], 'psi_desired': psid_array[1:]})
#
# plt.figure()
# plt.plot(t_points,ssa_psi(psi)*180/pi,linewidth=2.5, label="Actual Heading Angle")
# plt.plot(t_array[1:],psid_array[1:],linewidth=2.5, label="Desired Heading Angle")
# plt.grid(linewidth =1, color='gray',linestyle='dotted')
# plt.xlabel('Time in s',fontsize=25)
# plt.ylabel('Heading angle in degrees',fontsize=25)
# plt.title('Heading angle (Actual vs Desired)',fontsize=25)
# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=15)
# plt.legend(fontsize=20, loc='upper right')
# plt.show()
#
# savemat("ype_vs_t.mat", {'t_array': t_array[1:], 'ype': ype_array[1:]})
# plt.figure()
# plt.plot(t_array[1:], ype_array[1:],linewidth=2.5)
# plt.grid(linewidth =1, color='gray',linestyle='dotted')
# plt.xlabel('Time in s',fontsize=25)
# plt.ylabel('Cross Track Error in m',fontsize=25)
# plt.title('Cross Track Error vs Time -PD',fontsize=25)
# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=15)
# plt.show()





# k2 = 0
# print("1st", ype_array[0])
# for k1 in ype_array:
#     if ype_array[k2] == maxi:
#         print("pos i ", k2, xn[k2], yn[k2])
#     k2 += 1

print(average(abs_ype_array))
print("avg ype pd",average(abs_ype_array))

#print(ype_array)
# print("ype max",max(ype_array[1:]))
# print(U)
# print(t[-1])
# print("delta", delta*180/pi)


import math
y_actual = zeros(size(ype_array))
y_predicted = ype_array

MSE = square(y_predicted).mean()

RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)
#
plt.figure()
plt.plot(t_points, r, label='r')
plt.plot(t_points,delta,label='delta')
plt.title('r vs t')
plt.legend()
plt.show()

# print('r_tol',3*L)
# plt.figure()
# plt.plot(t_ode, delta_array)
# plt.title('delta vs t')
# plt.show()
print(max(psid_array))