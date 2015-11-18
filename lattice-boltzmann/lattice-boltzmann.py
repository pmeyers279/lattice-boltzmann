import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time


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
            dotProd = v[0] * direction[0] + \
                v[1] * direction[1]
            weights.append(
                (1 + 3 * dotProd + 4.5 * dotProd ** 2 - 1.5 * vsquared))

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
        rho = 0
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
        barrier[
            (self.height / 2) - 8:(self.height / 2) + 8, self.height / 2] = True
        self.barrier = barrier

    def stream(self):
        """
        move all particles by one step along directions of motion
        """
        # if we don't have densities to iterate, get them now.
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
            d = (1 - omega) * d + omega * rweight * rho * vweight
            self.densities[direction] = d
        # at edges just use general flow...
        for (direction, vweight, rweight) in zip(self.directions, weights, self.twoDweights):
            self.densities[direction][:, 0] = rweight * vweight

    def curl(self):
        u = self.getMacroscopicVelocity()
        ux = u[0]
        uy = u[1]
        c = np.roll(uy, -1, axis=1)\
            - np.roll(uy, 1, axis=1)\
            - np.roll(ux, -1, axis=0)\
            + np.roll(ux, 1, axis=0)
        return c

PF = ParticleField(80, 200, 0.02, flow=(0.1, 0))
FIG = plt.figure(figsize=(8, 3))
# add a barrier!
PF.addBarrier()
fluidImage = plt.imshow(PF.curl(), origin='lower',
                        norm=plt.Normalize(-.1, .1),
                        cmap=plt.get_cmap('jet'),
                        interpolation='none')
bImageArray = np.zeros((PF.height, PF.width, 4), np.uint8)
bImageArray[PF.barrier, 3] = 255
barrierImage = plt.imshow(bImageArray, origin='lower',
                          interpolation='none')
startTime = time.clock()


def nextFrame(arg):
    global startTime
    if arg % 100 == 0 and arg > 0:
        endTime = time.clock()
        print "%1.1f frames per second" % (100 / endTime - startTime)
    for step in range(20):
        PF.stream()
        PF.collide()
    fluidImage.set_array(PF.curl())
    return (fluidImage, barrierImage)

animate = ani.FuncAnimation(FIG, nextFrame, interval=1, blit=False)
plt.show()
