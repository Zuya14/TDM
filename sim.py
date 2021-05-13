import cv2
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

import math

import lidar_util

robot_name = "urdf/pla-robot.urdf"

def onRect(pos, rec_s, rec_e):
    return pos[0] >= rec_s[0] and pos[0] <= rec_e[0] and pos[1] >= rec_s[1] and pos[1] <= rec_e[1]

class sim:

    def __init__(self, _id=0, mode=p.DIRECT, sec=0.01):
        self._id = _id
        self.mode = mode
        self.phisicsClient = bc.BulletClient(connection_mode=mode)
        self.reset(sec=sec)

    def reset(self, x=1.0, y=1.0, theta=0.0, vx=0.0, vy=0.0, w=0.0, sec=0.01, action=None, clientReset=False):
        if clientReset:
            self.phisicsClient = bc.BulletClient(connection_mode=self.mode)

        self.steps = 0
        self.sec = sec

        self.vx = vx
        self.vy = vy
        self.w = w

        self.phisicsClient.resetSimulation()
        self.robotUniqueId = 0 
        self.bodyUniqueIds = []
        self.phisicsClient.setTimeStep(sec)

        self.action = action if action is not None else [0.0, 0.0, 0.0]
        
        self.done = False

        self.loadBodys(x, y, theta)

        x, y = self.getState()[:2]
        self.distance = math.sqrt((x - 10.0)**2 + (y - 10.0)**2)
        self.old_distance = self.distance

    def getId(self):
        return self._id

    def loadBodys(self, x, y, theta):
        self.robotPos = (x,y,0)
        self.robotOri = p.getQuaternionFromEuler([0, 0, theta])

        self.robotUniqueId = self.phisicsClient.loadURDF(
            robot_name,
            basePosition=self.robotPos,
            baseOrientation = self.robotOri
            )
        
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(-1, -1, 0))]
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(-1, 11, 0))]
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(11, 11, 0))]
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(11, -1, 0))]

        for i in range(11):
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( i, 11, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( i, -1, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(-1,  i, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(11,  i, 0))]

        for i in range(11):
            if i not in [2, 9]:
                self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( i, 5, 0))]
            if i not in [1, 5, 8]:
                if i < 5:
                    self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( 5, i, 0))]
                else:
                    self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( 6, i, 0))]

    def step(self, action):

        self.old_distance = self.distance

        if not self.done:
            self.action = action

            l = math.sqrt(action[1]**2 + action[2]**2)
            cos = action[1] / l
            sin = action[2] / l

            v  = (self.action[0] + 1.0) * 0.5

            self.vx = v * cos
            self.vy = v * sin

            self.w = 0

            self.updateRobotInfo()

            self.phisicsClient.stepSimulation()

            if self.isContacts():
                self.done = True

            if self.isArrive():
                self.done = True

        else:
            self.vx = 0
            self.vy = 0
            self.w = 0

        x, y = self.getState()[:2]
        self.distance = math.sqrt((x - 10.0)**2 + (y - 10.0)**2)
        self.steps += 1

        return self.done

    def observe(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        scanDist = bullet_lidar.scanDistance(self.phisicsClient, pos, yaw, height=0.1)
        self.scanDist = scanDist / bullet_lidar.maxLen
        self.scanDist = self.scanDist.astype(np.float32)

        x, y = self.getState()[:2]

        dx = 10.0 - x
        dy = 10.0 - y

        obs = self.scanDist
        obs = np.append(obs, [dx, dy, self.vx, self.vy]).astype(np.float32)

        return obs

    def render(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        scanDist = bullet_lidar.scanDistance(self.phisicsClient, pos, yaw, height=0.1)
        self.scanDist = scanDist / bullet_lidar.maxLen
        self.scanDist = self.scanDist.astype(np.float32)

        img = lidar_util.imshowLocalDistance("render"+str(self.phisicsClient), 800, 800, bullet_lidar, self.scanDist, maxLen=1.0, show=False, line=True)
        # print(self.scanDist.shape)

        return img

    def updateRobotInfo(self):

        self.phisicsClient.resetBaseVelocity(
            self.robotUniqueId,
            linearVelocity=[self.vx, self.vy, 0],
            angularVelocity=[0, 0, self.w]
            )

        self.robotPos, self.robotOri = self.phisicsClient.getBasePositionAndOrientation(self.robotUniqueId)

    def getRobotPosInfo(self):
        return self.robotPos, self.robotOri

    def getState(self):
        pos, ori = self.getRobotPosInfo()
        return pos[0], pos[1], p.getEulerFromQuaternion(ori)[2], self.vx, self.vy, self.w 

    def close(self):
        self.phisicsClient.disconnect()

    def contacts(self):
        contactList = []
        for i in self.bodyUniqueIds[0:]: # 接触判定
            contactList += self.phisicsClient.getContactPoints(self.robotUniqueId, i)
        return contactList 

    def isContacts(self):
        return len(self.contacts()) > 0

    def isArrive(self):
        # return self.distance < 0.5
        return self.distance < 0.1

    def isDone(self):
        return self.done

class sim_maze3(sim):

    def loadBodys(self, x, y, theta):
        self.robotPos = (x,y,0)
        self.robotOri = p.getQuaternionFromEuler([0, 0, theta])

        self.robotUniqueId = self.phisicsClient.loadURDF(
            robot_name,
            basePosition=self.robotPos,
            baseOrientation = self.robotOri
            )

        for i in range(10):
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(i-1,  9, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  i, -1, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( -1,i-1, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  9,  i, 0))]

        for i in range(6):
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  3,  i, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  4,  i, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  5,  i, 0))]

    def reset(self, sec, tgtpos=[7.0, 1.5]):
        # init_pos = np.array([1.5, 1.5])
        init_pos = np.random.rand(2) * 8.5
        init_pos = init_pos + 0.25 

        while onRect(init_pos, [3.0-0.25, 0.0-0.25], [6.0+0.25, 6.0+0.25]):
            init_pos = np.random.rand(2) * 8.5
            init_pos = init_pos + 0.25 

        super().reset(x=init_pos[0], y=init_pos[1], sec=sec)

        tgt_pos = np.random.rand(2) * 8.5
        tgt_pos = tgt_pos + 0.25 

        while onRect(tgt_pos, [3.0-0.25, 0.0-0.25], [6.0+0.25, 6.0+0.25]) or math.sqrt((init_pos[0] - tgt_pos[0])**2 + (init_pos[1] - tgt_pos[1])**2) < 1.0:
            tgt_pos = np.random.rand(2) * 8.5
            tgt_pos = tgt_pos + 0.25 

        self.tgt_pos = tgt_pos

        # self.tgt_pos = np.array(tgtpos)

        x, y = self.getState()[:2]
        # self.distance = math.sqrt((x - self.tgt_pos[0])**2 + (y - self.tgt_pos[1])**2)
        # self.old_distance = self.distance

    def test_reset(self, sec, tgtpos=[7.0, 1.5]):
        init_pos = np.array([1.5, 1.5])

        super().reset(x=init_pos[0], y=init_pos[1], sec=sec)

        self.tgt_pos = np.array(tgtpos)

        x, y = self.getState()[:2]
        # self.distance = math.sqrt((x - self.tgt_pos[0])**2 + (y - self.tgt_pos[1])**2)
        # self.old_distance = self.distance

    def step(self, action):

        # self.old_distance = self.distance

        if not self.done:
            self.action = action

            l = math.sqrt(action[1]**2 + action[2]**2)
            cos = action[1] / l
            sin = action[2] / l

            v  = (self.action[0] + 1.0) * 0.5

            self.vx = v * cos
            self.vy = v * sin

            self.w = 0

            self.updateRobotInfo()

            self.phisicsClient.stepSimulation()

            if self.isContacts():
                self.done = True

            if self.isArrive():
                self.done = True

        else:
            self.vx = 0
            self.vy = 0
            self.w = 0

        x, y = self.getState()[:2]
        # self.distance = math.sqrt((x - self.tgt_pos[0])**2 + (y - self.tgt_pos[1])**2)

        return self.done

    def observe(self):
        pos, _ = self.getRobotPosInfo()
        obs = np.array(pos[:2])

        return obs

    def render(self, tgtpos=[7.5, 1.5]):
        w = 800
        h = 800
        img = np.zeros((w, h, 3), np.uint8)
        center = (w//2, h//2)

        points = [
            [0.0, 0.0],
            [0.0, 9.0],
            [9.0, 0.0],
            [9.0, 9.0],
            [3.0, 0.0],
            [3.0, 6.0],
            [6.0, 6.0],
            [6.0, 0.0],
        ]

        points = np.array(points)
        points += np.array([-4.5, -4.5])

        pts = np.zeros((points.shape[0], 2))

        k = center[0] if w<h else center[1]
        # k = 10
        maxLen = 5

        for a, b in zip(pts, points):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        pts += center
        pts = pts.astype(np.int)

        cv2.line(img, tuple(pts[0]), tuple(pts[1]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[0]), tuple(pts[2]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[1]), tuple(pts[3]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[2]), tuple(pts[3]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)

        cv2.line(img, tuple(pts[4]), tuple(pts[5]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[5]), tuple(pts[6]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[6]), tuple(pts[7]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)

        # print([self.observe(), tgtpos])
        points2 = np.array([self.observe(), tgtpos])
        points2 += np.array([-4.5, -4.5])
        pts2 = np.zeros((2, 2))

        for a, b in zip(pts2, points2):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        pts2 += center
        pts2 = pts2.astype(np.int)

        cv2.circle(img, tuple(pts2[0]), radius=20, color=(255,0,0), thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.circle(img, tuple(pts2[1]), radius=20, color=(0,0,255), thickness=2, lineType=cv2.LINE_8, shift=0)

        return img

class sim_square3(sim_maze3):

    def loadBodys(self, x, y, theta):
        self.robotPos = (x,y,0)
        self.robotOri = p.getQuaternionFromEuler([0, 0, theta])

        self.robotUniqueId = self.phisicsClient.loadURDF(
            robot_name,
            basePosition=self.robotPos,
            baseOrientation = self.robotOri
            )

        for i in range(10):
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(i-1,  9, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  i, -1, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=( -1,i-1, 0))]
            self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall.urdf", basePosition=(  9,  i, 0))]

    def render(self, tgtpos=[7.5, 1.5]):
        w = 800
        h = 800
        img = np.zeros((w, h, 3), np.uint8)
        center = (w//2, h//2)

        points = [
            [0.0, 0.0],
            [0.0, 9.0],
            [9.0, 0.0],
            [9.0, 9.0],
            [3.0, 0.0],
            [3.0, 6.0],
            [6.0, 6.0],
            [6.0, 0.0],
        ]

        points = np.array(points)
        points += np.array([-4.5, -4.5])

        pts = np.zeros((points.shape[0], 2))

        k = center[0] if w<h else center[1]
        # k = 10
        maxLen = 5

        for a, b in zip(pts, points):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        pts += center
        pts = pts.astype(np.int)

        cv2.line(img, tuple(pts[0]), tuple(pts[1]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[0]), tuple(pts[2]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[1]), tuple(pts[3]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, tuple(pts[2]), tuple(pts[3]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)

        # print([self.observe(), tgtpos])
        points2 = np.array([self.observe(), tgtpos])
        points2 += np.array([-4.5, -4.5])
        pts2 = np.zeros((2, 2))

        for a, b in zip(pts2, points2):
            a[0] =  b[0] * (k/maxLen)
            a[1] = -b[1] * (k/maxLen)

        pts2 += center
        pts2 = pts2.astype(np.int)

        cv2.circle(img, tuple(pts2[0]), radius=20, color=(255,0,0), thickness=-1, lineType=cv2.LINE_8, shift=0)
        cv2.circle(img, tuple(pts2[1]), radius=20, color=(0,0,255), thickness=2, lineType=cv2.LINE_8, shift=0)

        return img

if __name__ == '__main__':

    # sim = sim(0, mode=p.GUI, sec=0.001)
    # sim = sim_cross(0, mode=p.GUI, sec=0.001)
    # sim = sim(0, mode=p.DIRECT)
    # sim = sim_cross(0, mode=p.DIRECT, sec=0.001)
    # sim = sim_maze3(0, mode=p.GUI, sec=0.001)
    # sim = sim_square(0, mode=p.GUI, sec=0.001)
    # sim = sim_maze3(0, mode=p.DIRECT, sec=0.001)
    sim = sim_square3(0, mode=p.DIRECT, sec=0.001)

    from bullet_lidar import bullet_lidar

    resolusion = 12
    deg_offset = 90.
    rad_offset = deg_offset*(math.pi/180.0)
    startDeg = -180. + deg_offset
    endDeg = 180. + deg_offset
    maxLen = 20.
    minLen = 0.
    lidar = bullet_lidar(startDeg, endDeg, resolusion, maxLen, minLen)

    while True:
        # action = np.array([0.5, 0.5, 1])
        action = np.random.rand(3)

        # sim.step(action=action)

        # print(np.concatenate([action, sim.action]))

        # cv2.imshow("sim", sim.render(lidar))
        cv2.imshow("sim", sim.render())
        if cv2.waitKey(1) >= 0:
            break