#!/home/neolux/dev_ws/2023_e/.venv/bin/python

import cv2 as cv
import os
import time
import numpy as np

from serial import Serial
import config as C

ser = Serial(C.ser, 115200)


def find_dot(img):
    x, y, r = -1, -1, -1
    img = cv.GaussianBlur(img, (5, 5), 0)
    mask = cv.inRange(img, np.array([0, 0, 150]), np.array([100, 100, 255]))
    # cv.imshow('mask', mask)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_cnt = None
    max_area = 0
    for cnt in cnts:
        if cv.contourArea(cnt) > max_area:
            max_cnt = cnt
            max_area = cv.contourArea(cnt)
    if max_cnt is not None:
        cv.drawContours(img, [max_cnt], -1, (0, 255, 0), 3)
        (x, y), r = cv.minEnclosingCircle(max_cnt)
    else:
        x, y, r = -1, -1, -1
    return x, y, r


def find_green(img):
    x, y, r = -1, -1, -1
    img = cv.GaussianBlur(img, (5, 5), 0)
    mask = cv.inRange(img, np.array([0, 160, 0]), np.array([160, 255, 160]))
    # cv.imshow("mask", mask)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_cnt = None
    max_area = 0
    for cnt in cnts:
        if cv.contourArea(cnt) > max_area:
            max_cnt = cnt
            max_area = cv.contourArea(cnt)
    if max_cnt is not None:
        cv.drawContours(img, [max_cnt], -1, (0, 255, 0), 3)
        (x, y), r = cv.minEnclosingCircle(max_cnt)
    else:
        x, y, r = -1, -1, -1
    return x, y, r


def send(*args):
    global ser
    sent = [0xFF]
    for arg in args:
        sent.append(arg // 256)
        sent.append(arg % 256)
    sent.append(0xFE)
    sent.pop(1)
    sent = bytearray(sent)
    ser.write(sent)
    print(sent)


def send_vertex(vts):
    global ser
    lg = vts.shape[0]
    if not (lg == 4 or lg == 8 or lg == 10):
        return
    for idx in range(lg - 1):
        sent = [0xFF, 0x02]
        sent.append(vts[idx][0] // 256)
        sent.append(vts[idx][0] % 256)
        sent.append(vts[idx][1] // 256)
        sent.append(vts[idx][1] % 256)
        sent.append(0x01)
        sent.append(0xFE)
        sent = bytearray(sent)
        ser.write(sent)
        print(idx, vts[idx], sent)
    # idx +=1
    sent = [0xFF, 0x02]
    sent.append(vts[lg - 1][0] // 256)
    sent.append(vts[lg - 1][0] % 256)
    sent.append(vts[lg - 1][1] // 256)
    sent.append(vts[lg - 1][1] % 256)
    sent.append(0x02)
    sent.append(0xFE)
    sent = bytearray(sent)
    ser.write(sent)
    print(lg - 1, vts[lg - 1], sent)


def find_corners(out_cnt, in_cnt):
    pnts = [[], []]
    eps = 0.03 * cv.arcLength(out_cnt, True)
    approx = cv.approxPolyDP(out_cnt, eps, True)
    for idx, pnt in enumerate(approx):
        x, y = pnt[0]
        pnts[0].append([x, y])
    eps = 0.03 * cv.arcLength(in_cnt, True)
    approx = cv.approxPolyDP(in_cnt, eps, True)
    for idx, pnt in enumerate(approx):
        x, y = pnt[0]
        pnts[1].append([x, y])
    try:
        return np.array(pnts)
    except:
        return None


def pair(pnts):
    def vec2ctr(points, center):
        return points - center

    def vec_angle(vec1, vec2):
        multip = vec1.dot(vec2)
        m = np.sqrt(np.sum(vec1**2)) * np.sqrt(np.sum(vec2**2))
        ang = np.arccos(multip / m) * 180 / np.pi
        return ang

    ctr = np.mean(pnts, axis=1)
    ctr = np.mean(ctr, axis=0)
    vec_out = vec2ctr(pnts[0], ctr)
    vec_in = vec2ctr(pnts[1], ctr)
    ret = [[], []]
    outer_paired, inner_paired = [], []
    # angs = []
    for oidx, ovec in enumerate(vec_out):
        for iidx, ivec in enumerate(vec_in):
            if oidx not in outer_paired and iidx not in inner_paired:
                ang = vec_angle(ovec, ivec)
                if ang < 10:
                    ret[0].append(pnts[0][oidx])
                    ret[1].append(pnts[1][iidx])
                    outer_paired.append(oidx)
                    inner_paired.append(iidx)
    try:
        return np.array(ret, np.uint16)
    except:
        return None


def merge(pnts):
    ret = np.mean(pnts, axis=0)
    return ret.astype(np.uint16)


def find_shape(img):
    blur = cv.GaussianBlur(img, (5, 5), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    _, thres = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
    # cv.imshow("th", gray)
    cnts, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        out_cnt = max(cnts, key=cv.contourArea)
        mask = np.zeros_like(gray)
        mask = cv.drawContours(mask, [out_cnt], 0, 255, -1)
        mask = cv.bitwise_and(mask, gray)
        _, thres = cv.threshold(mask, 50, 255, cv.THRESH_BINARY)

        # cv.imshow("in_th", thres)
        cnts, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            in_cnt = max(cnts, key=cv.contourArea)
            # cv.drawContours(img, [in_cnt], 0, (0, 0, 255), 2)
            # cv.drawContours(img, [out_cnt], 0, (0, 0, 255), 2)
            pnts = find_corners(out_cnt, in_cnt)
            if pnts is not None:
                pnts = pair(pnts)
                pnts = merge(pnts)
                for idx, pnt in enumerate(pnts):
                    cv.putText(
                        img,
                        f"{idx}",
                        pnt,
                        cv.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (0, 0, 255),
                        1,
                    )
                    cv.circle(img, pnt, 2, (0, 255, 0), -1)
                return pnts
    return None


def main(width=640, height=640):
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv.CAP_PROP_EXPOSURE, 180)

    print(f"Image size: {width}x{height}")

    while True:
        send(0x10)
        if ser.in_waiting > 0:
            read = ser.read_all()
            print(read)
        else:
            read = ""
        if read == b"ok":
            print("get it!")
            break

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame[9:359, 182:570], (width, height))

        gx, gy, gr = find_green(frame)
        if gx != -1 and gy != -1:
            frame = cv.circle(frame, (int(gx), int(gy)), 2, (255, 0, 0), -1)
            gx = int(gx)
            gy = int(gy)
            print(f"x: {gx}, y: {gy}")
            send(0x01, gx, gy)
        rx, ry, rr = find_dot(frame)
        if rx != -1 and ry != -1:
            frame = cv.circle(frame, (int(rx), int(ry)), 2, (0, 255, 0), -1)
            rx = int(rx)
            ry = int(ry)
            print(f"x: {rx}, y: {ry}")
            send(0x03, rx, ry)

        verteces = find_shape(frame)
        if verteces is not None:
            send_vertex(verteces)

        # if ser.in_waiting > 0:
        #     cmd = ser.read_all()
        #     print(cmd)
        #     if cmd == b'\xaa\x01\xbb':
        #         verteces = find_shape(frame)
        #         send_vertex(verteces)

        if C.pc_mode:
            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    ser.close()


if __name__ == "__main__":
    main()
