import math
import numpy as np

class bullet_lidar:

    def __init__(self, startDeg, endDeg, resolusion, maxLen, minLen=0.0, deg_offset=0):
        self.startDeg = startDeg
        self.endDeg = endDeg
        self.resolusion = resolusion
        self.maxLen = maxLen
        self.minLen = minLen
        self.rad_offset = deg_offset*(math.pi/180.0)
        self._shape = np.arange(self.startDeg, self.endDeg, self.resolusion).shape

    @property
    def shape(self):
        return self._shape

    def getScanFromTo(self, rad_offset, height):
        rads = np.arange(self.startDeg, self.endDeg, self.resolusion)*(math.pi/180.0) + rad_offset + self.rad_offset
        cos = np.cos(rads).reshape((-1,1))
        sin = np.sin(rads).reshape((-1,1))
        heights = np.full(sin.shape, height, dtype=float)
        scanFrom2d = np.concatenate([self.minLen*cos, self.minLen*sin],1)
        scanFromLocal = np.concatenate([scanFrom2d, heights],1)
        scanTo2d = np.concatenate([self.maxLen*cos, self.maxLen*sin],1)
        scanToLocal = np.concatenate([scanTo2d, heights],1)
        return scanFromLocal, scanToLocal

    def scanPos(self, phisicsClient, lidarPos, lidarTheta, height):
        scanFrom, scanTo = self.getScanFromTo(lidarTheta, height)
        results = phisicsClient.rayTestBatch(lidarPos+scanFrom, lidarPos+scanTo, numThreads=0)
        pos = np.array([r[3] for r in results])
        return pos # 3d

    def scanPosLocal(self, phisicsClient, lidarPos, lidarTheta, height, maxFlag=True):
        pos = self.scanPos(phisicsClient, lidarPos, lidarTheta, height)
        posLocal = [p - lidarPos if p[0]!=0. or p[1]!=0. else (0,0,lidarPos[2]) for p in pos]
        cos = np.cos(-lidarTheta)
        sin = np.sin(-lidarTheta)
        posLocal = np.array([[cos*p[0]-sin*p[1], sin*p[0]+cos*p[1], p[2]] for p in posLocal])

        if maxFlag:
            dist = np.array([np.sqrt(p[0]**2+p[1]**2) for p in posLocal])
            dist = np.where(dist > 1e-3, dist, self.maxLen)
            rads = np.arange(self.startDeg, self.endDeg, self.resolusion)*(math.pi/180.0) + lidarTheta + self.rad_offset
            cos = np.cos(rads)
            sin = np.sin(rads)
            posLocal = np.array([[l*c, l*s, 0] for l, c, s in zip(dist, cos, sin)])

        return posLocal # 3d

    def scanDistance(self, phisicsClient, lidarPos, lidarTheta, height, maxFlag=True):
        posLocal = self.scanPosLocal(phisicsClient, lidarPos, lidarTheta, height, maxFlag=False)
        dist = np.array([np.sqrt(p[0]**2+p[1]**2) for p in posLocal])
        if maxFlag:
            dist = np.where(dist > 1e-3, dist, self.maxLen)
        return dist # 2d distance       
