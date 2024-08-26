import numpy as np
import cv2
import glob
import pickle

# 체스보드 패턴의 내부 코너 개수
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 각 셀의 크기(mm 단위로 설정)
square_size = 15  # mm

# 3D 점의 벡터 준비
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# 객체 포인트와 이미지 포인트 배열 저장
objpoints = []
imgpoints = []

# 이미지 파일 경로
images = glob.glob('/home/sineunji/open_study/YOLOcode/1234/data/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너 그리기 및 보여주기
        cv2.drawChessboardCorners(img,CHECKERBOARD, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# objpoints와 imgpoints를 파일로 저장
with open('calibration_data.pkl', 'wb') as f:
    pickle.dump((objpoints, imgpoints), f)
