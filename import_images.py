import os
import cv2  # OpenCV


def import_images():
    sudoku = []
    sudokuBgr = []
    soccer = []
    road = []
    roadBgr = []
    pcb = []
    building = []
    buildingBgr = []

    sudokuFolder = 'image_database/sudoku'
    soccerFolder = 'image_database/soccer'
    roadFolder = 'image_database/road'
    pcbFolder = 'image_database/pcb'
    buildingFolder = 'image_database/building'

    sudokuNames = sorted([img for img in os.listdir(sudokuFolder) if
                          img.endswith(
                              ".png")])  # ref : https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/
    for name in sudokuNames:
        sudoku.append(
            cv2.imread(sudokuFolder + '/' + name, cv2.IMREAD_GRAYSCALE))
        sudokuBgr.append(cv2.imread(sudokuFolder + '/' + name))

    soccerNames = sorted([img for img in os.listdir(soccerFolder) if
                          img.endswith(
                              ".png")])  # ref : https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/
    for name in soccerNames:
        soccer.append(cv2.imread(soccerFolder + '/' + name))

    roadNames = sorted([img for img in os.listdir(roadFolder) if img.endswith(
        ".png")])  # ref : https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/
    for name in roadNames:
        road.append(cv2.imread(roadFolder + '/' + name, cv2.IMREAD_GRAYSCALE))
        roadBgr.append(cv2.imread(roadFolder + '/' + name))

    buildingNames = sorted([img for img in os.listdir(buildingFolder) if
                            img.endswith(
                                ".png")])  # ref : https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/
    for name in buildingNames:
        building.append(
            cv2.imread(buildingFolder + '/' + name, cv2.IMREAD_GRAYSCALE))
        buildingBgr.append(cv2.imread(buildingFolder + '/' + name))

    pcbNames = sorted([img for img in os.listdir(pcbFolder) if img.endswith(
        ".png")])  # ref : https://www.geeksforgeeks.org/python-create-video-using-multiple-images-using-opencv/
    for name in pcbNames:
        pcb.append(cv2.imread(pcbFolder + '/' + name, cv2.IMREAD_GRAYSCALE))

    return sudoku, sudokuBgr, soccer, road, roadBgr, pcb, building, buildingBgr, sudokuNames, sudokuNames, soccerNames, roadNames, pcbNames, buildingNames
