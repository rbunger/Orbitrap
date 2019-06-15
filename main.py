from numpy import zeros, complex
from numpy.fft import fft
from math import atan2, sin, cos, log, sqrt, pi
import matplotlib.pyplot as plt
from vpython import vec, mag, sphere, rate, color, arrow, quad, vertex, canvas

scene = canvas(width=1024, height=768, background=color.gray(0.5))

# Orbitrap

R_1 = 12.0e-3 / 2
R_2 = 30.0e-3 / 2
R_m = 1.1 * R_2 * sqrt(2.0)
U_r = 3.5e3
k = 2 * U_r / (R_m ** 2 * log(R_2 / R_1) - (R_2 ** 2 - R_1 ** 2) / 2)

# particle

e = 1.602e-19
u = 1.66053892e-27
q = e
m = 12 * u
R = (3 * R_1 + R_2) / 4
v_phi = sqrt((k / 2) * (q / m) * (R_m ** 2 - R ** 2))
v = vec(-v_phi, 0, 0)
r = vec(0, R, 12e-3)

# Orbitrap plot


def vrtx(x, y, z):
    return vertex(pos=vec(x, y, z), color=color.blue, normal=vec(x**2, y**2, 0) / sqrt(x**2 + y**2))


def U(r, z):
    return (k / 2) * (z ** 2 - (r ** 2 - R_1 ** 2) / 2 + R_m ** 2 * log(r / R_1)) - U_r


def dU(r, z):
    return (k / 2) * (R_m ** 2 / r - r)


def r_Newton(r, z, U_12):
    while True:
        r_new = r - (U(r, z) - U_12) / dU(r, z)
        if (abs(r_new - r) > 1e-10):
            r = r_new
        else:
            return r_new


def orbitrap_plot(r_start, U_12):
    N_z = 25
    N_phi = 50
    z_max = 2 * R_2
    r = r_start
    dz = z_max / (N_z - 1)
    dphi = 2 * pi / (N_phi - 1)
    rr = zeros(N_z, dtype=float)
    zz = zeros(N_z, dtype=float)
    if (int(U_12) == 0):
        N_phi = int(N_phi / 2)
        E_0 = 10 * mag(E(vec(0, R_2, 0))) / (R_2 - R_1)
    for i in range(0, N_z):
        z = i * dz
        r = r_Newton(r, z, U_12)
        rr[i] = r
        zz[i] = z
        if (abs(U_12) == 0):
            p = vec(0, r, z)
            arrow(pos=p, axis=E(p) / E_0, color=color.red)
        if (i > 0):
            for j in range(0, N_phi):
                phi = pi / 2 + j * dphi
                quad(vs=[vrtx(rr[i - 1] * cos(phi), rr[i - 1] * sin(phi),  zz[i - 1]),
                         vrtx(rr[i] * cos(phi), rr[i] * sin(phi),  zz[i]),
                         vrtx(rr[i] * cos(phi + dphi), rr[i] * sin(phi + dphi),  zz[i]),
                         vrtx(rr[i - 1] * cos(phi + dphi), rr[i - 1] * sin(phi + dphi),  zz[i - 1])])
                quad(vs=[vrtx(rr[i] * cos(phi), rr[i] * sin(phi), -zz[i]),
                         vrtx(rr[i - 1] * cos(phi), rr[i - 1] * sin(phi), -zz[i - 1]),
                         vrtx(rr[i - 1] * cos(phi + dphi), rr[i - 1] * sin(phi + dphi), -zz[i - 1]),
                         vrtx(rr[i] * cos(phi + dphi), rr[i] * sin(phi + dphi), -zz[i])])

# E-field


def E(p):
    r = sqrt(p.x ** 2 + p.y ** 2)
    phi = atan2(p.y, p.x)
    E_r = -(k / 2) * (R_m ** 2 / r - r)
    return vec(E_r * cos(phi), E_r * sin(phi), -k * p.z)

# initializations


orbitrap_plot(R_1, -U_r)
orbitrap_plot(R_2, 0.0)
w = sqrt((q / m) * k)
f = w / (2 * pi)
dt = 1 / f / 100

print(f / 1e6, "MHz axial frequency")
print(v_phi, "m/s tangential velocity")

# LF

N_t = 2 ** 13
ii = zeros(N_t, dtype=complex)
particle = sphere(pos=r, radius=0.5e-3, color=color.green, make_trail=True, retain=100)
v -= dt * (q / m) * E(particle.pos) / 2
for i in range(N_t):
    v += dt * (q / m) * E(particle.pos)
    particle.pos += dt * v
    ii[i] = complex(v.z / v_phi, 0.0)
    rate(100)

# perform FFT
# w = sqrt((q/m)k)
# w^2 = (q/m)k
# m/q = k / w^2
# m/z = ke / w^2

II = fft(ii)
mz = zeros(int(N_t / 2))
II_mag = zeros(int(N_t / 2))
for i in range(1, int(N_t / 2)):
    w = 2 * pi * i / float(N_t * dt)
    mz[i] = k * e / w ** 2
    II_mag[i] = abs(II[i])

plt.figure("Test Ion: Carbon")
plt.xlabel("m/z / u")
plt.ylabel("1 / Hz")
plt.xlim(0, 100)
plt.ylim(0, 4000)
plt.plot(mz / u, II_mag)
plt.show()