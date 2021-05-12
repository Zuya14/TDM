import numpy as np
import cv2
import math

def predict(points, v, theta_world, w, dt):
    
    if type(v) is np.ndarray:
        next_points = np.empty((v.shape[0], points.shape[0], 3))
        i = 0
        for _v, _t, _w in zip(v, theta_world, w):
            dv = _v*dt
            dd = np.array([dv*np.cos(_t), dv*np.sin(_t), 0])
            # w=0 ########################
            cos = np.cos(-_w*dt)
            sin = np.sin(-_w*dt)
            next_points[i] = np.array([[cos*p[0]-sin*p[1], sin*p[0]+cos*p[1], p[2]] for p in points - dd])
            i += 1
    else:
        dv = v*dt
        dd = np.array([dv*np.cos(theta_world), dv*np.sin(theta_world), 0])
        cos = np.cos(-w*dt)
        sin = np.sin(-w*dt)
        next_points = np.array([[cos*p[0]-sin*p[1], sin*p[0]+cos*p[1], p[2]] for p in points - dd])
    # next_points = points - dd

    return next_points

def imshowLocalDistance(name, h, w, lidar, distances, maxLen, point_color=(0,0,255), line_color=(255,0,0), preimg=None, show=True, line=False):

    rads = np.arange(lidar.startDeg, lidar.endDeg, lidar.resolusion)*(math.pi/180.0)
    cos = np.cos(rads)
    sin = np.sin(rads)

    points = np.array([[l*c, l*s] for l, c, s in zip(distances, cos, sin)])

    return imshowLocal(name, h, w, points, maxLen, point_color, line_color, preimg, show, line)

def imshowLocal(name, h, w, points, maxLen, point_color=(0,0,255), line_color=(255,0,0), preimg=None, show=True, line=False):

    img = np.zeros((h, w, 3), np.uint8) if preimg is None else preimg
    center = (w//2, h//2)

    if preimg is None:
        cv2.line(img, (0, center[1]), (w, center[1]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, (center[0], 0), (center[0], h), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)

    pts = np.zeros((points.shape[0], 2))

    k = center[0] if w<h else center[1]

    for a, b in zip(pts, points):
        a[0] =  b[0] * (k/maxLen)
        a[1] = -b[1] * (k/maxLen)

    pts += center

    if line:
        for p in pts.astype(np.int64):
            cv2.line(img, center, tuple(p), color=line_color, thickness=1, lineType=cv2.LINE_8, shift=0)

    for p in pts.astype(np.int64):
        cv2.circle(img, tuple(p), radius=1, color=point_color, thickness=-2, lineType=cv2.LINE_8, shift=0)

    if show:
        cv2.imshow(name, img)

    return img

def imshowTgt(name, h, w, points, maxLen, point_color=(0, 255, 0), preimg=None, show=True):

    img = np.zeros((h, w, 3), np.uint8) if preimg is None else preimg
    center = (w//2, h//2)

    if preimg is None:
        cv2.line(img, (0, center[1]), (w, center[1]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, (center[0], 0), (center[0], h), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)

    pts = np.zeros((points.shape[0], 2))

    k = center[0] if w<h else center[1]

    for a, b in zip(pts, points):
        a[0] =  b[0] * (k/maxLen)
        a[1] = -b[1] * (k/maxLen)

    pts += center

    for p in pts.astype(np.int64):
        cv2.circle(img, tuple(p), radius=1, color=point_color, thickness=-2, lineType=cv2.LINE_8, shift=0)

    if show:
        cv2.imshow(name, img)

    return img

def imshowCircle(name, h, w, l, maxLen, color=(0,0,255), preimg=None, show=True):

    img = np.zeros((h, w, 3), np.uint8) if preimg is None else preimg
    center = (w//2, h//2)

    if preimg is None:
        cv2.line(img, (0, center[1]), (w, center[1]), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)
        cv2.line(img, (center[0], 0), (center[0], h), color=(255,255,255), thickness=1, lineType=cv2.LINE_8, shift=0)

    k = center[0] if w<h else center[1]
    # print(int(l*k//maxLen))

    cv2.circle(img, center, radius=int(l*k//maxLen), color=color, thickness=1, lineType=cv2.LINE_8, shift=0)

    if show:
        cv2.imshow(name, img)

    return img


if __name__ == '__main__':

    import sys
    import os
    import math
    from pathlib import Path

    assert len(sys.argv) == 2

    p = Path(sys.argv[1])

    if Path.is_dir(p):
        files = list(p.glob("*.csv"))
    elif Path.is_file(p):
        files = [p]
    else:
        print("no file")
        exit()

    print([f.stem for f in files])

    w = 800
    h = 800

    for f in files:
        with open(f, newline='') as log_file:

            stem = f.stem

            # maxLen = 8.
            maxLen = 1.

            print("maxLen:"+str(maxLen))

            datas = np.loadtxt(log_file, delimiter=',')
            datas = np.clip(datas, None, maxLen)

            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
            video = cv2.VideoWriter("{}/{}.mp4".format(f.parent, stem), fourcc, 10, (w,h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

            deg_offset = 45.
            rad_offset = deg_offset*(math.pi/180.0)
            startDeg = -135. + deg_offset
            endDeg = 135. + deg_offset
            resolusion = 0.25

            size = datas.shape[-1]
            print('shape:', datas.shape)

            # for data in datas[::10]:
            for data in datas:
                # img = drawLidar(name="data", h=h, w=w, data=data, maxLen=maxLen, deg_range=180)

                rads = np.arange(startDeg, endDeg, resolusion)*(math.pi/180.0) + rad_offset
                cos = np.cos(rads)[:size]
                sin = np.sin(rads)[:size]
                posLocal = np.array([[l*c, l*s, 0] for l, c, s in zip(data, cos, sin)])
                
                img = imshowLocal(name="data", h=h, w=w, points=posLocal, maxLen=maxLen, show=False, line=True)
                
                video.write(img)

                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break
            print("save video:{}".format(stem))
