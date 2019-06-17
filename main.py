from math import atan2, sin, cos, log, sqrt, pi
from numpy import zeros, max
from numpy.fft import fft
import scipy.constants
import matplotlib.pyplot as plt
from vpython import vec, mag, sphere, rate, color, arrow, quad, vertex, canvas, cross

scene = canvas(title="right mouse=rotate, wheel=zoom, left=resize scene; Fourier analysis follows after time integration has terminated",
               width=1280, height=800, background=color.gray(0.5))

# Orbitrap settings

R_1 = 12.0e-3 / 2
R_2 = 30.0e-3 / 2
R_m = 1.1 * R_2 * sqrt(2.0)
U_r = 3.5e3
k = 2 * U_r / (R_m ** 2 * log(R_2 / R_1) - (R_2 ** 2 - R_1 ** 2) / 2)
N_t = 2 ** 13
kappa = 100

# particle settings

u = scipy.constants.physical_constants["atomic mass constant"][0]
q = 1 * scipy.constants.e
m = 12 * u
R = (3 * R_1 + R_2) / 4
v_phi = sqrt((k / 2) * (q / m) * (R_m ** 2 - R ** 2))
v = vec(-v_phi, 0, 0)
r = vec(0, R, 12e-3)


def U(r, z):
    return (k / 2) * (z ** 2 - (r ** 2 - R_1 ** 2) / 2 + R_m ** 2 * log(r / R_1)) - U_r


def dU(r):
    return (k / 2) * (R_m ** 2 / r - r)


def E(p):
    r = sqrt(p.x ** 2 + p.y ** 2)
    phi = atan2(p.y, p.x)
    E_r = -(k / 2) * (R_m ** 2 / r - r)
    return vec(E_r * cos(phi), E_r * sin(phi), -k * p.z)


class Orbitrap_mesh():

    def __init__(self):
        self.N_z = 25
        self.N_phi = 50
        self.z_max = 2 * R_2
        self.dz = self.z_max / (self.N_z - 1)
        self.dphi = 2 * pi / (self.N_phi - 1)

    def r_Newton(self, z):
        while True:
            r = self.r - (U(self.r, z) - self.U_12) / dU(self.r)
            err = abs((r - self.r) / r)
            self.r = r
            if (err < 1e-12):
                return r

    def point(self, i, phi, mirror=False):
        z = i * self.dz
        r = self.r_Newton(z)
        if (mirror):
            z *= -1
        return vec(r * cos(phi), r * sin(phi), z)

    def vertex(self, p, n):
        return vertex(pos=p, color=color.blue, normal=n)
    
    def quad(self, i_1, phi_1, i_2, phi_2, i_3, phi_3, i_4, phi_4, mirror=False):
        p_1 = self.point(i_1, phi_1, mirror)
        p_2 = self.point(i_2, phi_2, mirror)
        p_3 = self.point(i_3, phi_3, mirror)
        p_4 = self.point(i_4, phi_4, mirror)
        n = cross(p_2 - p_1, p_3 - p_1)
        n /= mag(n)
        if (mirror):
            n *= -1
        return quad(vs=[self.vertex(p_1, n), self.vertex(p_2, n), self.vertex(p_3, n), self.vertex(p_4, n)])
    
    def plot(self, U_12, inner_outer):
        if (inner_outer == "inner"):
            self.r = R_1
        else:
            self.r = R_2
        self.U_12 = U_12
        if (inner_outer == "outer"):
            N_phi = self.N_phi // 2
            E_0 = 10 * mag(E(vec(0, R_2, 0))) / (R_2 - R_1)
        else:
            N_phi = self.N_phi
        for i in range(0, self.N_z):
            if (inner_outer == "outer"):
                p = self.point(i, pi / 2)
                arrow(pos=p, axis=E(p) / E_0, color=color.red)
            for j in range(0, N_phi):
                phi = pi / 2 + j * self.dphi
                self.quad(i - 1, phi, i, phi, i, phi + self.dphi, i - 1, phi + self.dphi)
                self.quad(i - 1, phi, i, phi, i, phi + self.dphi, i - 1, phi + self.dphi, True)


# mesh for the orbitrap

mesh = Orbitrap_mesh()
mesh.plot(-U_r, "inner")
mesh.plot(0.0, "outer")

# Leap-Frog time integration

w = sqrt((q / m) * k)  # omega
f = w / (2 * pi)
dt = 1 / f / kappa
I = zeros(N_t)
particle = sphere(pos=r, radius=0.5e-3, color=color.green, make_trail=True, retain=100)
v -= dt * (q / m) * E(particle.pos) / 2
for i in range(N_t):
    v += dt * (q / m) * E(particle.pos)
    particle.pos += dt * v
    I[i] = v.z / v_phi
    rate(50)

# spectral analysis

I_spec = fft(I)
m = zeros(N_t)
for i in range(1, N_t // 2):
    w = 2 * pi * i / (N_t * dt)
    m[i] = k * q / w ** 2

# plot

plt.figure("Test Ion: Carbon")
plt.xlabel("m / u")
plt.ylabel("normalized density")
plt.xlim(0, 50)
plt.ylim(0, 1.1)
plt.xticks(list(plt.xticks()[0]) + [12])
plt.plot(m / u, abs(I_spec) / max(abs(I_spec)))
plt.show()
