import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time
from matplotlib.colors import LogNorm


class ParticleField(object):

    """ParticleField class for lattice-boltzmann simulation"""

    def __init__(self, height, width, viscosity, flow=(0, 0)):
        super(ParticleField, self).__init__()
        self.height = height
        self.width = width
        self.viscosity = viscosity
        self.flow = flow
        self.barrier = None
        self.densities = None
        self.directions = [
            (0, 0), (1, 0), (0, 1),
            (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1),
            (1, -1)]
        self.twoDweights = [4. / 9., 1. / 9., 1. / 9., 1. /
                            9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36.]
        ones = np.ones((height, width))

    def getVelocityProbability(self, v=None):
        """
        Solve for velocity factors in equation (7)
        """
        if v is None:
            v = self.flow
        weights = []
        vsquared = (v[0]**2 + v[1]**2)
        for direction in self.directions:
            dotProd = (v[0] * direction[0] +
                       v[1] * direction[1])
            weights.append(
                (3 * dotProd + 4.5 * dotProd ** 2 - 1.5 * vsquared))

        return weights

    def getDensities(self):
        """
        Solve for particle densities in all 9 directions
        """
        weights = self.getVelocityProbability()
        densities = OrderedDict()
        ones = np.ones((self.height, self.width), np.float64)
        for direction, vweight, rweight in zip(self.directions, weights,
                                               self.twoDweights):
            densities[direction] = rweight * (ones + vweight)
        return densities

    def getMacroscopicDensity(self):
        """
        Solve for Macroscopic Density
        """
        if self.densities is None:
            densities = self.getDensities()
        else:
            densities = self.densities
        rho = np.zeros(densities[(0, 0)].shape)
        for direction in self.directions:
            rho += densities[direction]
        return rho

    def getMacroscopicVelocity(self):
        """
        Solve for Macroscopic velocity (ux, uy)
        """
        rho = self.getMacroscopicDensity()
        if self.densities is None:
            densities = self.getDensities()
        else:
            densities = self.densities
        ux = np.zeros((self.height, self.width))
        uy = np.zeros((self.height, self.width))
        for direction in self.directions:
            density = densities[direction]
            ux = np.add(ux, (density / rho) * direction[0])
            uy = np.add(uy, (density / rho) * direction[1])
        return (ux, uy)

    def addBarrier(self, type='line'):
        barrier = np.zeros((self.height, self.width), bool)
        # vertical bars
        barrier[
            (self.height / 2) - 16:(self.height / 2) + 16, self.width / 2 - 16:self.width / 2 + 16] = True
  #       barrier[
  #           (self.height / 2) - 16:(self.height / 2) + 16, 5*self.width / 8] = True
        # horizontal bars
  #       barrier[
  #           self.height / 2 - 16, (self.width / 2):5 * self.width / 8] = True
  #       barrier[
  #           self.height / 2 + 16, (self.width / 2):5 * self.width / 8] = True
#        barrier[
#            self.height/4 + 8, (self.height / 2):5*self.height / 8] = True
        self.barrier = barrier

    def stream(self, doRand=False):
        """
        move all particles by one step along directions of motion
        """
        # if we don't have densities to iterate, get them now.
        if doRand:
            self.flow = (
                0.02 + 0.01 * np.random.randn(), 0.02 + 0.01 * np.random.randn())
        if self.densities is None:
            self.densities = self.getDensities()
        for direction in self.directions:
            # move particles
            self.densities[direction] = np.roll(
                self.densities[direction],
                direction[0], axis=1)
            self.densities[direction] = np.roll(
                self.densities[direction],
                direction[1], axis=0)
            # self.densities[direction][0, 1:] = np.zeros(
            #     np.size(self.densities[direction][0, 1:]))
            # self.densities[
            #     direction][-1, 1:] = np.zeros(np.size(self.densities[direction][-1, 1:]))
            # self.densities[direction][
            #     :, -1] = np.zeros(np.size(self.densities[direction][:, -1]))
        if self.barrier is None:
            return
        else:
            for direction in self.directions:
                # if there's a barrier make them bounce off.
                temp = np.roll(self.barrier, direction[0], axis=1)
                temp = np.roll(temp, direction[1], axis=0)
                revDir = (-1 * direction[0], -1 * direction[1])
                self.densities[direction][
                    temp] = self.densities[revDir][self.barrier]

    def collide(self):
        """ make particles collide """
        rho = self.getMacroscopicDensity()
        u = self.getMacroscopicVelocity()
        omega = 1 / (3. * self.viscosity + 0.5)
        macroscopic_vweights = self.getVelocityProbability(u)
        weights = self.getVelocityProbability()
        if self.densities is None:
            self.densities = self.getDensities()
        # collide with macroscopic velocities
        for (direction, vweight, rweight) in zip(self.directions, macroscopic_vweights, self.twoDweights):
            d = self.densities[direction]
            d = (1 - omega) * d + omega * rweight * rho * (1 + vweight)
            self.densities[direction] = d
        # at edges just use general flow...
        for (direction, vweight, rweight) in zip(self.directions, weights, self.twoDweights):
            self.densities[direction][:, 0] = rweight * (1 + vweight)

    def curl(self):
        u = self.getMacroscopicVelocity()
        ux = u[0]
        uy = u[1]
        c = np.roll(uy, -1, axis=1)\
            - np.roll(uy, 1, axis=1)\
            - np.roll(ux, -1, axis=0)\
            + np.roll(ux, 1, axis=0)
        return c


def createPlot(PF, fig):
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title('Density Plot')
    im1 = ax1.imshow(np.log10(PF.getMacroscopicDensity() * np.abs(PF.barrier * 1 - 1)), origin='lower',
                     norm=plt.Normalize(np.log10(0.9), np.log10(1.01)),
                     cmap=plt.get_cmap('viridis'))
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('Curl Plot')
    im2 = ax2.imshow(PF.curl(), origin='lower',
                     norm=plt.Normalize(-.02, .02),
                     cmap=plt.get_cmap('RdBu'))
    return [im1, im2]

# position of test mas
x0 = 200
y0 = 80

# number of frames
NF = 70 * 10
xs = np.arange(1 - x0, 401 - x0)
ys = np.arange(1 - y0, 161 - y0)
Xs, Ys = np.meshgrid(xs, ys)
PF = ParticleField(80 * 2, 200 * 2, 0.00136, flow=(0.02, 0.02))
PF.addBarrier()
im = []
cmin = np.unique(
    min(PF.getMacroscopicDensity().reshape((1, PF.height * PF.width))))[0]
cmax = np.unique(
    max(PF.getMacroscopicDensity().reshape((1, PF.height * PF.width))))[0]
fig = plt.figure(figsize=(8, 8))
axt = np.ones((NF, 1))
ayt = np.ones((NF, 1))
# burn in...
print 'burning in...'
for kk in range(2000):
    PF.stream(doRand=True)
    PF.collide()
print "done burning in...let's do it"
for ii in range(NF):
    for jj in range(5):
        PF.stream(doRand=True)
        PF.collide()
    cmin2 = np.unique(
        min(PF.getMacroscopicDensity().reshape((1, PF.height * PF.width))))[0]
    cmax2 = np.unique(
        max(PF.getMacroscopicDensity().reshape((1, PF.height * PF.width))))[0]
    if cmin2 < cmin:
        cmin = cmin2
    if cmax2 > cmax:
        cmax = cmax2
    dens = PF.getMacroscopicDensity()
    dens = dens / (np.mean(dens[:]))
    ax = np.zeros(np.shape(dens))
    ay = np.zeros(np.shape(dens))
    # integrate over z direction. assume test mass is on ground, atmosphere is uniform 
    # up to 100m...not a good approxmation, but whatever.
    for kk in range(100):
        ax += (dens - np.mean(dens[:])) * np.abs(PF.barrier * 1 - 1) *  Xs / \
            (Xs**2 + Ys**2 + (ii + 1)**2) ** (1.5)
        ay += (dens - np.mean(dens[:])) * np.abs(PF.barrier * 1 - 1) * Ys / \
            (Xs**2 + Ys**2 + (ii + 1)**2) ** (1.5)
    ax = sum(sum(ax))
    ay = sum(sum(ay))
    # density of atmosphere * G * ax
    # from Jan's paper equation 146...
    axt[ii] = 1.225 * 6.67e-11 * ax
    ayt[ii] = 1.225 * 6.67e-11 * ay
    pics = createPlot(PF, fig)
    im.append(pics)

plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'
writer = ani.FFMpegWriter(fps=70)
animate = ani.ArtistAnimation(fig, im)
animate.save('./movie-density.mp4', writer=writer)
plt.close()

axf = np.squeeze(np.abs(np.fft.fft(axt)))
ayf = np.squeeze(np.abs(np.fft.fft(ayt)))
freqs = np.squeeze(np.fft.fftfreq(len(axt),d=(5./350)))
N = len(freqs[freqs > 0])

ts = np.arange(1, 21) * (1. / 25)
fig2 = plt.figure()
plt.plot((axt**2 + ayt**2)**(0.5), c='r')
ax2 = plt.gca()
ax2.set_yscale('log')
plt.savefig('timeseries')
plt.close()


print freqs[1:N]
print freqs[1]
fig3 = plt.figure()
plt.plot(freqs[1:N], ((axf[1:N]**2 + ayf[1:N]**2)**(0.5)/(2*np.pi*freqs[1:N])**2)/(4e3))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Strain / rtHz')
ax3 = plt.gca()
ax3.set_yscale('log')
plt.savefig('amplitude_spectrum')

