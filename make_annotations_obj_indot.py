import cv2
import numpy as np
import os
import json
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import math
import random
import time

autoAnnotate = False
useBorders = True

saveName = "yolo_seg_fast_bare"
saveName = "yolo_seg_fast"

saveName = "yolo_seg"

saveName = "yolo_seg_obj"
# saveName = "yolo_seg_shoulderLane"

# saveName = "yolo_seg_roadseg"
"""
checking 20230612_062805.mp4
frame 1107 PTO 1 wing 0.997 0.999 shoulder 1000 road 576 side all
frame 1108 PTO 1 wing 0.999 0.999 shoulder 1000 road 906 side left
frame 1109 PTO 1 wi
"""


# videosPath = "/media/nathan/AE8D-87D2/indot/converted_videos" # dont use
# videosPath = "/media/nathans/EB37-4BA6/indot/converted_videos/converted_again"

# videosPath = "/media/nathans/EB37-4BA6/indot/epsilon" # checked classes Dec 7
# videosPath = "/media/nathans/EB37-4BA6/indot/eta" # checked classes Dec 7
# videosPath = "/media/nathans/EB37-4BA6/indot/zeta"  # checked classes Dec 7 # need to add regression
# videosPath = "/media/nathans/EB37-4BA6/indot/beta/5-25-thru-5-25" # checked classes Dec 7
# videosPath = "/media/nathans/EB37-4BA6/indot/beta/6-16-thru-6-21" # checked classes Dec 7
# videosPath = "/home/nathan/Desktop/CV_lab/lab/non_lab/beeVideos"
# videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/ear/unreal/202309112151_rgb"
# videosPath = "/home/nathan/Downloads/segment_dataset"
# videosPath = "/home/nathan/Desktop/oscar_ml/dataset"
# videosPath = "/home/nathan/Desktop/ear_stalk_detection/datasets/stalk/stalk_testing_videos/lateharvest/f75/gopro_f75"



# videosPath = "/media/nathans/EB37-4BA6/indot/alpha/6-16-thru-6-20" # checked classes Dec 7
# videosPath = "/media/nathans/EB37-4BA6/indot/alpha/5-25-thru-6-16" # checked classes Dec 7
# videosPath = "/media/nathans/EB37-4BA6/indot/delta" # checked classes Dec 7


videosPath = "/media/evanslab/EB37-4BA6/indot/2024/tractor04"

# videosPath = "/home/nathan/Desktop/old_work/oscar_ml/random_driving_dataset"


class Annotator:
    def __init__(self, videosPath, saveName, useBorders=False, model=None):

        self.windowScale = 1#0.85

        self.realImgScale = 1#2.5

        if "converted_again" in videosPath:
            self.realImgScale = 2.5

        self.showPreds = False


        # self.modelName="/home/nathan/Desktop/ear_stalk/stalk_rgb_training_videos/stalk_rgb_training_videos/runs/segment/train2/weights/best.pt "
        # self.modelName = "/home/nathan/Desktop/ear_stalk/stalk_rgb_training_videos/stalk_rgb_training_videos/runs/segment/train4/weights/best.pt"
        # self.modelName = "/media/nathan/AE8D-87D2/annotate_tools/road_etc_nov21.pt"
        self.modelName = "/home/nathan/Desktop/indot/indot_seg_nov23.pt"
        self.modelName = "/media/nathan/AE8D-87D2/indot/yolo_seg_indo_nov27.pt"
        self.modelName = "/media/nathan/AE8D-87D2/annotate_tools/road_etc_nov30.pt"
        self.modelName = "/media/nathan/AE8D-87D2/annotate_tools/road_etc_dec5.pt"
        self.modelName = "/media/nathan/AE8D-87D2/annotate_tools/runs/segment/train4/weights/road_etc_dec8_640.pt"
        self.modelName = "/media/nathans/EB37-4BA6/annotate_export_train/road_etc_dec14_640.pt"
        self.modelName = "road_etc_july11_640.pt"
        # self.modelName = "/media/nathan/AE8D-87D2/annotate_tools/road_etc_jan4.pt"
        # self.modelName = "/home/nathan/Desktop/ear_stalk_detection/datasets/stalk/stalk_rgb_training_videos_numbered/runs/segment/train15/weights/best.pt"
        self.classifierName = "classifier_june6.pt"
        self.classifierName = "classifier_dec7.pt"
        self.classifierName = "classifier_july_old.pt"
        self.classifierName = "/media/nathan/EB37-4BA6/indot_code/classifier_july19.pt"
        self.classifierName = "classifier_july21_2.pt"
        # self.modelName = "/home/nathan/Desktop/ear_stalk_detection/datasets/stalk/stalk_rgb_training_videos_numbered/runs/segment/train12/weights/best.pt"


        self.classifierModel = None
        self.model = None
 
        self.regressioning = False

        self.videosPath = videosPath
        self.saveName = saveName

        self.useBorders=useBorders

        self.currentPoly = [] # format [[x,y], [x,y], ...]
        self.firstCallback = True
        self.falseKey = -1

        self.merge = False

        self.defaultButtons = {"Clear": -2, "Select": 9, "Zoom": -3, "Fast": 102, "Next": 108, "Prev": 106, "Next Vid": 47, "Prev Vid": 46, "Next Annot": 93, "Prev Annot": 91, "Last Annot": 92, "classify": -14, "Auto Annot":-4, "Exit": 27} #"dont save": -7, "toggle view":-13
        
        self.selectButtons = {"Cancel": 27, "Delete": 8, "Merge": -9, "Class++":-11, "Reduce Poly":-12, "Delete pt":8}

        self.labelNames = {-1: "erase", 0: "Road", 1: "grass", 2: "mower", 3: "sign", 17: "Mailbox", 4: "building", 6: "vehicle", 
                        7: "person", 14: "trash",  18: "branches", 19: "cable barrier", 9:"guardrail", 16: "cone", 10: "shoulder", 15: "roadkill", 11: "drain", 13: "post", 
                        12: "bridgeBarrels",  5: "tbd", 8: "gravel"}

        # self.labelNames = {-1: "erase", 0: "shoulderLine", 1: "otherLine", 2: "roadEdge"}


        # self.currentState = { "State": 0, "location": 0,  "leftWing": False, "rightWing": False, 
        # "onRoadSlight": False, "onRoadAlot": False, "Post": False, "Mailbox": False, "Drain":False, 
        # "Bridge":False, "Sign": False, "Turn around": False, "Roadkill": False, "Branches": False, 
        # "Car": False, "Intersection": False, "Person": False, "Guardrail":False}

        self.fullImgClasses = [{
                            "Interstate": {
                                        "Normal mow": {},
                                        "Manuever": {
                                            "Sign": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Post": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Roadkill": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Trash": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Cone": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Turnaround": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Intersection": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Guardrail": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Drain": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                            "Telephone Pole": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        },
                                        "Transit": {}
                            },
                            "Country":  {
                                    "Normal mow": {},
                                    "Manuever": {
                                        "Sign": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Post": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Mailbox": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Branches": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Telephone Pole": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Intersection":{},
                                        "Trash": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Drain": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Cone": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Guardrail": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                        "Roadkill": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                    },
                                    "Transit": {}
                            },
                            "Town": {
                                "Normal mow": {},
                                "Sign": {"hit": {"destroy": {}, "Knock Over": {}}, "not hit": {}},
                                "Transit": {}
                                },
                            "Unusual": {},
                            "Maintenance": {}
                            },

                            {"Right Wing": {}},

                            {"Left Wing": {}},
                            
                            {"On Shoulder": {}},
                            {"On Road": {}},
                            {"Obj Invis": {}},
                            # {"Stick": {}},
                            # {"Cable Barrier": {}},
                            # {"Sign Post": {}},
                            # {"Reflector": {}},
                            # {"Street Num": {}},
                            # {"lollipop": {}},
                            # {"Square utility": {}},
                            # {"tube utility": {}},
                            # {"PVC tube": {}},
                            # {"Concrete": {}},
                            # {"Gnd Flag": {}},
                            # {"Street Light": {}}

                            ]

        self.classifying = False
        self.confirmedClasses = False
        self.imgClasses = {}
        self.lastImgClass = []

        self.openButtons = self.defaultButtons

        self.annotateMode = 0 # 0=normal annotate, 1=selected poly

        self.zoomRect = []
        self.zooming = False

        self.selectingPoly = False

        self.lastPan = []
        self.startPan = []

        self.selectedInd = -1
        self.selectedPt = -1
        self.movingPoint = True

        self.justMadePoly = False

        # this script technically supports multiple classes
        self.labelNum = 0
        self.lastLabelNum = 0

        self.mouseDown = False

        self.unsavedChanges = False

        self.grouping = False

        self.lastState = []

        self.sam_rect = []
        self.sam_model = None

        self.groupsMade=0
        self.frameStartGroups = 0
        self.totalGroups = 0

        self.vidGroups = 0
        self.startVidGroups = 0

        self.crosshair = False
        self.mouseX, self.mouseY = 0, 0

        self.classView = -1

        self.maskWeight = 1

        self.openImgFrame = -2


        self.fastmode = False
        self.fastmode_frames = 0
        self.fastmode_interest_frame = -1

        self.regPoints = []

        self.allLabelColors = [
            (255, 0, 0),     # Bright Blue
            (0, 255, 0),     # Bright Green 
            (255, 0, 255),   # Magenta
            (0, 0, 255),     # Bright Red
            (255, 255, 0),   # Cyan
            (0, 255, 255),   # Yellow
            (0, 165, 255),   # Orange
            (128, 0, 128),   # Purple
            (203, 192, 255), # Pink
            (128, 128, 0),   # Teal
            (0, 0, 128),     # Maroon
            (0, 128, 0),     # Dark Green
            (128, 0, 0),     # Navy Blue
            (0, 128, 128),   # Olive
            (128, 128, 128)  # Grey
        ]

        self.last_suggestions = []

        self.base_epsilon = 0.0001
        self.reduce_epsilon = self.base_epsilon





        # Note: borderlabel depreciated
        self.allPolys = {} # format {"Frame #": [ [[x,y], [x,y], ... label # [borderlabel, borderlabel, borderlabel (optional)]] ], ...}

    def manageKeyResponses(self):
        change = 0
        while change == 0:
            k = cv2.waitKey(10)

            if self.falseKey != -1: # treat button presses as key presses
                k = self.falseKey
                self.falseKey = -1


            if self.fastmode and k == -1:
                # print("fast")
                if self.frameNum + 20 > self.maxFrameNum:
                    self.fastmode = False
                    print("near max frame")
                elif self.frameNum-self.fastmode_frames > 500:
                    self.fastmode = False
                    print("went 500 frames")
                else:
                    k=-10000
                    print("fast")


            elif self.fastmode:
                print("fast off")
                self.fastmode = False



            if k == -1: # no response from keyboard
                continue


            elif k == -2: # clear all annotations in frame
                self.lastState = self.allPolys[str(self.frameNum)]
                self.unsavedChanges = True
                self.allPolys[str(self.frameNum)] = []
                self.dispImg()

            elif k == -3: # begin zoom
                self.zoomRect = []

                newButtons = {}
                for buttonName in self.openButtons:
                    if buttonName == "Zoom":
                        newButtons["Unzoom"] = self.openButtons["Zoom"]
                        self.zooming = True
                    elif buttonName == "Unzoom":
                        newButtons["Zoom"] = self.openButtons["Unzoom"]
                        self.zooming = False
                    else:
                        newButtons[buttonName] = self.openButtons[buttonName]

                self.openButtons = newButtons
                self.dispImg()

            elif k == -4:
                # self.detectPossible()
                self.detectPossible(self.openImg)
                # self.makePoly(addCurrentPoly=False)
                self.dispImg()
            elif k == -5:
                if self.labelNum >= 0:
                    self.lastLabelNum = self.labelNum
                if self.labelNum == -2:
                    self.labelNum = -3
                    self.currentPoly = []
                elif self.labelNum == -3:
                    self.currentPoly = []
                    self.labelNum = self.lastLabelNum
                else:
                    self.lastLabelNum = self.labelNum
                    self.labelNum = -2
                    self.currentPoly = []

                bTypes = ["Mod Border", "Rm Border", "Add Border"]
                newButtons = {}
                for buttonName in self.openButtons:
                    if buttonName in bTypes:
                        newName = bTypes[(bTypes.index(buttonName) + 1 ) % len(bTypes)]
                        newButtons[newName] = -5
                        
                    else:
                        newButtons[buttonName] = self.openButtons[buttonName]

                self.openButtons = newButtons

                self.dispImg()
            elif k == -6:

                if not self.grouping:
                    maxGroup = -1
                    for i in self.allPolys[str(self.frameNum)]:

                        if type(i[-1]) == list:
                            g = i[-2]
                        else:
                            g = i[-1]
                        maxGroup = max(g, maxGroup)
                        if maxGroup > 50:
                            break

                    if maxGroup < 50:
                        self.lastState = self.allPolys[str(self.frameNum)][:]
                        print("auto grouping")
                        stalks = []
                        for i in self.allPolys[str(self.frameNum)]:
                            e=-1
                            if type(i[-1]) == list:
                                e = -2

                            xs = [sublist[0] for sublist in i[0:e]]
                            maxX = max(xs)
                            minX = min(xs)
                            found = False
                            for s in stalks:
                                if s[0] < maxX and s[1] > minX:
                                    i[e] = s[2]
                                    found = True
                                    break
                            if not found:
                                random.seed(time.time())
                                i[e] = random.randint(0,1000) * 10
                                stalks += [[minX, maxX, i[e]]]

                        self.dispImg()
                        continue




                print("grouping")
                bTypes = ["Group", "Cancel"]
                newButtons = {}
                for buttonName in self.openButtons:
                    if buttonName in bTypes:
                        newName = bTypes[(bTypes.index(buttonName) + 1 ) % len(bTypes)]
                        newButtons[newName] = -6
                        
                    else:
                        newButtons[buttonName] = self.openButtons[buttonName]



                self.grouping = not self.grouping
                self.openButtons = newButtons
                self.dispImg()

            elif k == -7:
                self.unsavedChanges = False

            elif k == -9:
                print("Merging Polygons")

            elif k == 9:
                # print("Selecting poly")
                if self.selectingPoly:
                    self.selectingPoly = False
                    btnName = "Select"
                else:
                    self.selectingPoly = True
                    btnName = "Unselect"
                newButtons = {}
                for btn in self.openButtons:
                    if btn in ["Select", "Unselect"]:
                        newButtons[btnName] = 9
                    else:
                        newButtons[btn] = self.openButtons[btn]
                self.openButtons = newButtons
                self.defaultButtons = newButtons
                self.currentPoly = []
                self.dispImg()

            elif k == -12: # simplify poly
                if self.selectedInd != -1:

                    if self.reduce_epsilon == self.base_epsilon:
                        self.original_poly = self.allPolys[str(self.frameNum)][self.selectedInd][0:-1]
                        self.original_poly = [pt[:] for pt in self.original_poly]
                    else:
                        self.allPolys[str(self.frameNum)][self.selectedInd][0:-1] = self.original_poly
                    
                    if self.reduce_epsilon > 0.01:
                        self.reduce_epsilon += 0.01
                        if self.reduce_epsilon > 0.05:
                            self.reduce_epsilon += 0.01
                    else:
                        self.reduce_epsilon += 0.002
                    if self.reduce_epsilon > 0.04:
                        self.reduce_epsilon = self.base_epsilon


                    cnt = self.allPolys[str(self.frameNum)][self.selectedInd][0:-1]
                    cnt = np.array(cnt)
                    label = self.allPolys[str(self.frameNum)][self.selectedInd][-1]

                    
                    try:
                        epsilon = self.reduce_epsilon * cv2.arcLength(cnt, True)
                    except:
                        print("error reducing polygon. Maybe crossing lines?")
                        self.reduce_epsilon == self.base_epsilon
                        continue

                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    newPoly = []
                    for pt in approx:
                        newPoly.append([int(pt[0][0]), int(pt[0][1])])
                    # print(newPoly)
                    self.allPolys[str(self.frameNum)][self.selectedInd] = newPoly + [label]
                    self.unsavedChanges = True
                    self.dispImg()

            elif k == -13: # toggle view
                classesUsed = set()
                for poly in self.allPolys[str(self.frameNum)]:
                    classesUsed.add(poly[-1])

                classesUsed = list(classesUsed)
                classesUsed.sort()

                ind = -1
                if self.classView in classesUsed:
                    self.classView = classesUsed.index(self.classView)+1
                    if self.classView >= len(classesUsed):
                        self.classView = -1
                elif self.classView == -1 and len(classesUsed) > 0:
                    self.classView = classesUsed[0]
                self.dispImg()

            elif k == -14: # classify:
                self.classifying = not self.classifying
                self.dispImg()



            elif k == 119: # w
                if len(self.zoomRect) > 0:
                    changeY = max(-self.zoomRect[0][1], -5)
                    self.zoomRect[0][1] += changeY
                    self.zoomRect[1][1] += changeY
                    self.dispImg()

            elif k == 97: # a
                if len(self.zoomRect) > 0:
                    changeX = max(-self.zoomRect[0][0], -5)
                    self.zoomRect[0][0] += changeX
                    self.zoomRect[1][0] += changeX
                    self.dispImg()

            elif k == 115: # s
                if len(self.zoomRect) > 0:
                    changeY =  min(5, self.openImg.shape[0]-self.zoomRect[1][1])
                    self.zoomRect[0][1] += changeY
                    self.zoomRect[1][1] += changeY
                    self.dispImg()

            elif k == 100: # d
                if len(self.zoomRect) > 0:
                    changeX = min(5, self.openImg.shape[1]-self.zoomRect[1][0])
                    self.zoomRect[0][0] += changeX
                    self.zoomRect[1][0] += changeX
                    self.dispImg()



            elif k == 27: # escape - either stop making polygon or exit
                if len(self.currentPoly) > 0:
                    self.currentPoly = []
                    self.dispImg(drawAmount=1)
                else:
                    self.saveJSON()
                    if self.vidName[-4::].lower() not in [".jpg", ".png"]:
                        self.cap.release() # close current video
                    exit()

            elif k == 13: # enter or middle mouse button - finish making 
                self.justMadePoly = True


                

                if len(self.currentPoly) > 3: # need at least 3 points to make polygon

                    if self.grouping:
                        self.group()

                    elif self.labelNum == -2 or self.labelNum == -3:
                        self.toggleBorders() 
                    else:
                        self.makePoly()

                    self.dispImg()

            

                self.currentPoly = []

                self.dispImg(drawAmount=0)

            elif k == 47: # / - go to next video
                change = self.maxFrameNum + 10

            elif k == 46: # . - go to previous video
                change = -self.frameNum - 10

            elif k == 93: # ] - next annotation
                frameJump = self.maxFrameNum+10
                for frame in self.allPolys:
                    # break
                    if int(frame) > self.frameNum and len(self.allPolys[frame]) > 0:
                        frameJump = min(int(frame), frameJump)
                        # print(self.allPolys[frame])
                        # print("jump")
                        # print("looking for person")
                        # for i in self.allPolys[frame]:
                        #     if i[-1] == 7:
                        #         frameJump = min(int(frame), frameJump)
                        #         print("person found")

                # frameJump = self.maxFrameNum+10
                if self.classifying:
                    for frame in self.imgClasses:
                        if int(frame) > self.frameNum and len(self.imgClasses[frame]) > 0:
                        #and "Country" in self.imgClasses[frame] and "Left Wing" not in self.imgClasses[frame] and "Right Wing" in self.imgClasses[frame] and "On Shoulder" in self.imgClasses[frame] and "Manuever" not in self.imgClasses[frame]: #and "Town" in self.imgClasses[frame]:
                            frameJump = min(int(frame), frameJump)
                            # print("jump")
                if frameJump < self.maxFrameNum+10:
                    change = frameJump - self.frameNum
    
            elif k == 91: # [ - next annotation
                frameJump = -1
                for frame in self.allPolys:
                    if int(frame) < self.frameNum and len(self.allPolys[frame]) > 0:
                        frameJump = max(int(frame), frameJump)

                # frameJump = -1
                if self.classifying:
                    for frame in self.imgClasses:
                        if int(frame) < self.frameNum and len(self.imgClasses[frame]) > 0:# and "Town" in self.imgClasses[frame]:
                            frameJump = max(int(frame), frameJump)
                if frameJump != -1:
                    change = frameJump - self.frameNum

            elif k == 92: # \ - last annotation
                frameJump = -1
                for frame in self.allPolys:
                    if len(self.allPolys[frame]) > 0:
                        frameJump = max(int(frame), frameJump)
                if self.classifying:
                    for frame in self.imgClasses:
                        if len(self.imgClasses[frame]) > 0:
                            frameJump = max(int(frame), frameJump)
                if frameJump != -1:
                    change = frameJump - self.frameNum
                print("skipping", change, "frames")

            elif k == 8: # delete - begin making erase polygon or stop making erase polygon
                self.lastState = self.allPolys[str(self.frameNum)]
                if self.annotateMode == 0:
                    newButtons = {}
                    for buttonName in self.openButtons:
                        if buttonName == "Erase":
                            newButtons["Draw"] = self.openButtons["Erase"]
                        elif buttonName == "Draw":
                            newButtons["Erase"] = self.openButtons["Draw"]
                        else:
                            newButtons[buttonName] = self.openButtons[buttonName]

                    self.openButtons = newButtons

                    if self.labelNum < 0:
                        if self.lastLabelNum < 0:
                            self.labelNum = 0
                        else:
                            self.labelNum = self.lastLabelNum
                    else:
                        self.lastLabelNum = self.labelNum
                        self.labelNum = -1
                    # cool way to do it in 1 line, but it only works if there is 1 class
                    # self.labelNum = (min(self.labelNum, 1) + 2) % 2 -1 # make the label number either -1 or 0
                elif self.annotateMode == 1: # selected a polygon
                    self.unsavedChanges = True
                    if self.selectedPt != -1: # delete point
                        poly = self.allPolys[str(self.frameNum)][self.selectedInd][0:-1]
                        if len(poly) > 2:
                            labelNum = self.allPolys[str(self.frameNum)][self.selectedInd][-1]
                            newPoly = []
                            for i, pt in enumerate(poly):
                                if i != self.selectedPt:
                                    newPoly.append(pt)
                            newPoly.append(labelNum)
                            self.allPolys[str(self.frameNum)][self.selectedInd] = newPoly
                            self.selectedPt = -1
                    else: # delete polygon
                        self.allPolys[str(self.frameNum)].remove(self.allPolys[str(self.frameNum)][self.selectedInd])

                        self.selectedInd = -1
                        self.annotateMode = 0
                        self.openButtons = self.defaultButtons



                self.dispImg()

            elif k == 106: # j - go back 5 frames
                change = - 5

            elif k == 108: # l - go forward 5 frames
                change = 10

            elif 58 > k >= 49:
                self.labelNum = k-49
                self.dispImg()

            elif k == 122: # z - undo
                self.allPolys[str(self.frameNum)] = self.lastState[:]
                print("undoing")
                self.dispImg()
            elif k == 98: # b - change mask weight
                self.maskWeight = self.maskWeight-0.1
                if self.maskWeight < 0.1:
                    self.maskWeight = 1
                # if self.maskWeight == 1:
                #     self.maskWeight = 0.5
                # else:
                #     self.maskWeight = 1
                self.dispImg()
            elif k == 101: # e - go back 50 frames
                change = -50
            elif k == 114: # r - skip 50 frames
                change = 50
            elif k == 116: # t
                self.vidIndex -= 10
                change = -self.frameNum - 10

            elif k == 121: # y
                self.vidIndex += 10
                change = self.maxFrameNum + 10

            elif k == 102: # f

                print("fastmode on")
                self.fastmode = True
                self.fastmode_frames = self.frameNum

            elif k == 109:
                self.merge = not self.merge
                print("merge:", self.merge)

            else: # any other key pressed - go forward 1 key
                change = 1

        return change

    def mouseEvent(self, event, x, y, flags, param):
        if event == 1:
            self.mouseDown = True
        elif event == 4:
            self.mouseDown = False

        if event == 0: # moved mouse after making a polygon. This is to avoid making a second polygon accidently, which happens sometimes
            self.justMadePoly = False

        if 10 < x/self.windowScale < 200: # cursor in region with buttons
            if event == 4: # mouse up
                for i, buttonName in enumerate(self.openButtons):
                    if 5+(i+1)*55 > y/self.windowScale > 10+i*55:
                        self.falseKey = self.openButtons[buttonName]
                        break
            return

        elif self.baseImg.shape[1]-self.buttonPad+10 < x < self.baseImg.shape[1]-10: # cursor in right region with buttons

            if event == 4: # mouse up
                if self.classifying:
                    buttonsShown = []
                    for bs in self.fullImgClasses:
                        for c in self.imgClasses[str(self.frameNum)]:
                            if type(c) == list:
                                continue
                            if c in bs:
                                buttonsShown.append(c)
                                bs = bs[c]
                        buttonsShown += list(bs.keys())
                    # print(buttonsShown)
                    buttonsShown.append("last")
                    buttonsShown.append("clear")
                    buttonsShown.append("Regression")
                    for i, buttonName in enumerate(buttonsShown):
                        if 5+(i+1)*55 > y/self.windowScale > 10+i*55:
                            if buttonName == "clear":
                                self.imgClasses[str(self.frameNum)] = []
                            elif buttonName == "last":
                                self.imgClasses[str(self.frameNum)] = self.lastImgClass[:]
                            elif buttonName == "Regression":
                                self.regressioning = not self.regressioning
                                r = False
                                for c in self.imgClasses[str(self.frameNum)]:
                                    if type(c) == list:
                                        r = True
                                        break
                                if not r:
                                    self.imgClasses[str(self.frameNum)].append([-1, [-1, -1], [-1, -1], [-1, -1]])

                            elif buttonName in self.imgClasses[str(self.frameNum)]:
                                # self.imgClasses[str(self.frameNum)] = []
                                
                                for checking in self.fullImgClasses:
                                    found = True
                                    removing = False
                                    while len(list(checking.keys())) > 0 and found:
                                        found = False
                                        for c in checking:
                                            if c in self.imgClasses[str(self.frameNum)]:
                                                if removing or c == buttonName:
                                                    if c == "Country":
                                                        self.imgClasses[str(self.frameNum)][self.imgClasses[str(self.frameNum)].index(c)] = "Interstate"
                                                    elif c == "Interstate":
                                                        self.imgClasses[str(self.frameNum)][self.imgClasses[str(self.frameNum)].index(c)] = "Country"
                                                    else:
                                                        self.imgClasses[str(self.frameNum)].remove(c)
                                                        removing = True
                                                    checking = checking[c]
                                                    found = True
                                                    self.lastImgClass = self.imgClasses[str(self.frameNum)][:]
                                                    break
                                                else:
                                                    checking = checking[c]
                                                    found = True

                                                    break
                                            


                            else:
                                self.imgClasses[str(self.frameNum)].append(buttonName)
                                self.lastImgClass = self.imgClasses[str(self.frameNum)][:]
                            self.confirmedClasses = True
                            
                            self.unsavedChanges = True

                            self.dispImg()
                            return

                else:
                    for i, v in enumerate(self.labelNames):
                        buttonSize = 50
                        if 5+(i+1)*buttonSize > y/self.windowScale > 10+i*buttonSize:
                            if self.selectedInd != -1:
                                if v >= 0:
                                    self.allPolys[str(self.frameNum)][self.selectedInd][-1] = v

                                else:
                                    self.allPolys[str(self.frameNum)].remove(self.allPolys[str(self.frameNum)][self.selectedInd])
                                    self.selectedPt = -1
                                    self.selectedInd = -1
                                    self.annotateMode = 0
                                self.unsavedChanges = True
                            else:
                                self.labelNum = v
                                self.selectedPt = -1
                                self.selectedInd = -1
                                self.annotateMode = 0
                                if self.selectingPoly:
                                    self.falseKey = 9


                            self.dispImg()



                            break
            return

        else:

            if self.crosshair:
                self.mouseX, self.mouseY = x, y

            # offset if zoomed in on image
            minX = 0
            minY = 0
            if len(self.zoomRect) > 0 and not self.zooming:
                minX = self.zoomRect[0][0]
                minY = self.zoomRect[0][1]

            # convert x,y mouse position to x, y in image scale
            # convert x,y mouse position to x, y in image scale
            ogX = x
            ogY = y
            

            # x = int((x - self.xPad - self.buttonPad) / self.scale) + minX
            # y = int((y - self.yPad) / self.scale) + minY

            # if self.labelNum != -2 and self.labelNum != -3:
            #     x = max(min(x, self.baseImg.shape[1]-self.buttonPad-self.xPad), self.buttonPad+self.xPad)
            #     y = max(min(y, self.baseImg.shape[0]-self.yPad), self.yPad)


            ogX = x
            ogY = y
            x = int((x - self.xPad - self.buttonPad) / self.scale) + minX
            y = int((y - self.yPad) / self.scale) + minY

            if self.labelNum != -2 and self.labelNum != -3:
                x = max(min(x, self.openImg.shape[1]*self.realImgScale), 0)
                y = max(min(y, self.openImg.shape[0]*self.realImgScale), 0)

            # deal with making zoom rectangle
            if self.zooming:
                if event == 1: # mouse clicked, begin making zoom rectangle
                    self.zoomRect = [[x,y], [x,y]]
                    self.dispImg(drawAmount=1)
                elif event == 0 and len(self.zoomRect) > 0: # mouse moved while making zoom rectangle
                    self.zoomRect[1] = (x,y)
                    self.dispImg(drawAmount=1)
                elif event == 4: # let go when making zoom rectangle, begin zooming
                    self.zoomRect = [[max(min(self.zoomRect[0][0], self.zoomRect[1][0], self.openImg.shape[1]), 0),
                                    max(min(self.zoomRect[0][1], self.zoomRect[1][1], self.openImg.shape[0]), 0)],
                                    [min(max(self.zoomRect[0][0], self.zoomRect[1][0], 0), self.openImg.shape[1]),
                                    min(max(self.zoomRect[0][1], self.zoomRect[1][1], 0), self.openImg.shape[1])]
                                    ]
                    if self.zoomRect[1][0]-self.zoomRect[0][0] <= 0 or self.zoomRect[1][1]-self.zoomRect[0][1] <= 0:
                        self.zoomRect = []

                    self.zooming = False

                    self.dispImg(drawAmount=0)

            elif self.regressioning:
            
                if event == 1: # mouse down

                    found = False
                    for i, c in enumerate(self.imgClasses[str(self.frameNum)]):
                        if type(c) == list:
                            r = c
                            found = True
                            break
                    if not found:
                        self.imgClasses[str(self.frameNum)].append([-1, [-1, -1], [-1, -1], [-1, -1]])
                        r = [-1, [-1, -1], [-1, -1], [-1, -1]]
                        i = len(self.imgClasses[str(self.frameNum)])-1
                    
                    self.imgClasses[str(self.frameNum)][i][0] = y

                    reset = True
                    for j, a in enumerate(r[1::]):
                        if a[0] == -1:
                            self.imgClasses[str(self.frameNum)][i][j+1] = [x, x]
                            reset = False

                            break

                    if reset:
                        self.imgClasses[str(self.frameNum)][i] = [y, [x, x], [-1, -1], [-1, -1]]
                    self.unsavedChanges = True

                    self.dispImg(drawAmount=1)

                elif self.mouseDown and event == 0: # mouse move
                    found = False
                    for i, c in enumerate(self.imgClasses[str(self.frameNum)]):
                        if type(c) == list:
                            r = c
                            found = True
                            break
                    if not found:
                        return
                    self.imgClasses[str(self.frameNum)][i][0] = y

                    mod = False
                    for j, a in enumerate(r[1::]):
                        if a[0] == -1:
                            self.imgClasses[str(self.frameNum)][i][j][1] = x
                            mod = True
                            break
                    if not mod:
                        self.imgClasses[str(self.frameNum)][i][3][1] = x
                    self.dispImg(drawAmount=1)



             

            elif self.selectingPoly:#self.annotateMode == 1 and self.selectedInd >= 0: # selected a polygon
                # print("yo")

                """order of checking:
                1. have you selected anything yet? check if clicking anything
                2. are you dragging the point?
                3. are you clicking on an existing point?
                4. are you clicking on a line?
                5. did you click outside the polygon?



                """
                if event == 4 and self.selectedInd == -1:
                    self.annotateMode = 0
                    bestAr = 99999999999
                    for ind, poly in enumerate(self.allPolys[str(self.frameNum)]):
                        a = np.array(poly[0:-1])
                        ar = (np.max(a[:,0])-np.min(a[:,0])) * (np.max(a[:,1])-np.min(a[:,1]))
                        if bestAr > ar:
                            res = self.is_point_in_polygon((x,y), poly[0:-1])
                            
                            if res:

                                self.selectedInd = ind
                                self.openButtons = self.selectButtons
                                self.annotateMode = 1
                                self.reduce_epsilon = self.base_epsilon

                                bestAr = ar
                                

                    if self.annotateMode == 0:
                        self.openButtons = self.defaultButtons
                        # self.selectingPoly = False
                        self.selectedInd=-1

                if event == 0 and self.selectedPt >= 0 and self.movingPoint:
                    self.allPolys[str(self.frameNum)][self.selectedInd][self.selectedPt] = [x,y]
                    self.unsavedChanges = True

                elif event == 4:
                    # self.selectedPt = -1
                    self.movingPoint = False

                elif event == 1: # mouse down
                    self.movingPoint = False
                    
                    self.selectedPt = -1
                    if self.selectedInd != -1:
                        closest = 50/self.scale
                        for ind, pt in enumerate(self.allPolys[str(self.frameNum)][self.selectedInd][0:-1]): # check if selecting point
                            d = (x-pt[0]) ** 2 + (y-pt[1]) ** 2
                            if d < closest:
                                self.selectedPt = ind
                                closest = d
                                self.movingPoint = True

                        if self.selectedPt < 0: # did not select a point. Check if selecting a line
                            poly = self.allPolys[str(self.frameNum)][self.selectedInd][0:-1]
                            polyViewed = poly + [poly[0]] # add another point back to get the first and last points joined
                            label = self.allPolys[str(self.frameNum)][self.selectedInd][-1]
                            closest = 25/self.scale
                            self.selectedPt = -1
                            for i in range(len(polyViewed)-1):
                                d = self.closest_point_on_segment((polyViewed[i], polyViewed[i+1]), (x,y))
                                d = (x-d[0])**2 + (y-d[1])**2
                                if d < closest:
                                    closest=d
                                    self.selectedPt = i
                            if self.selectedPt >= 0:
                                newPoly = []
                                for i, pt in enumerate(poly):
                                    newPoly.append(pt)
                                    if i == self.selectedPt:
                                        newPoly.append([x,y])
                                self.allPolys[str(self.frameNum)][self.selectedInd] = newPoly + [label]
                                self.selectedPt += 1
                                self.movingPoint = True


                    if self.selectedPt == -1: # did not select a point or anywhere on a line. Check if selected in the object at all
                        self.annotateMode = 0
                        bestAr = 99999999999
                        for ind, poly in enumerate(self.allPolys[str(self.frameNum)]):
                            a = np.array(poly[0:-1])
                            ar = (np.max(a[:,0])-np.min(a[:,0])) * (np.max(a[:,1])-np.min(a[:,1]))
                            if bestAr > ar:
                                res = self.is_point_in_polygon((x,y), poly[0:-1])
                                
                                if res:
                                    self.reduce_epsilon = self.base_epsilon
                                    self.selectedInd = ind
                                    self.openButtons = self.selectButtons
                                    self.annotateMode = 1
                                    bestAr = ar

                        if self.annotateMode == 0:
                            self.openButtons = self.defaultButtons
                            # self.selectingPoly = False
                            self.selectedInd=-1

                elif event == 0 and len(self.sam_rect) > 0:
                    self.sam_rect[2::] = [x,y]
                    self.dispImg(drawAmount=1)
                elif event == 6: # middle mouse button up
                    if self.selectedInd >= 0:
                        self.falseKey = -12

                    elif len(self.startPan)>0 and abs(ogX-self.startPan[0]) + abs(ogY-self.startPan[1]) < 10:
                        self.falseKey = 13
                        
                    elif len(self.sam_rect) > 0:

                        self.sam_rect = [min(self.sam_rect[0], self.sam_rect[2]), min(self.sam_rect[1], self.sam_rect[3]), max(self.sam_rect[0], self.sam_rect[2]), max(self.sam_rect[1], self.sam_rect[3])]
                        sr = [int(max(0, r)) for r in self.sam_rect]
                        sr[2::] = [min(sr[2], self.openImg.shape[1]), min(sr[3], self.openImg.shape[0])]
                        if sr[2]-sr[0] > 10 and sr[3]-sr[1] > 10:
                            print("making sam rect")
                            self.make_sam_annot(self.openImg, sr)
    
                        self.dispImg()
                        self.unsavedChanges =True
                    self.sam_rect = []
                    self.lastPan = []
                    self.startPan = []


                elif event == 3: # middle mouse button down

                    if len(self.zoomRect) == 0:
                        self.sam_rect = [x, y, x, y]
                        
                    else:
                        self.lastPan = [ogX, ogY]
                        self.startPan = [ogX, ogY]
                
                else:
                    return


                self.dispImg()





            # elif self.selectingPoly: 
            #     if event == 4: # mouse up
            #         pass
                    # self.allPolys = {} # format {"Frame #": [ [[x,y], [x,y], ... label # ] ], ...}




            else:# make polygon

                if event == 0 and len(self.lastPan) > 0 and len(self.zoomRect) > 0:
                    xChange = self.lastPan[0]- ogX
                    yChange = self.lastPan[1]- ogY
                    ogZoomW = self.zoomRect[1][0]-self.zoomRect[0][0]
                    ogZoomH = self.zoomRect[1][1]-self.zoomRect[0][1]


                    self.zoomRect[0][0] += xChange
                    self.zoomRect[1][0] += xChange
                    self.zoomRect[0][1] += yChange
                    self.zoomRect[1][1] += yChange

                    if self.zoomRect[0][0] < 0:
                        self.zoomRect[0][0] = 0
                        self.zoomRect[1][0] = ogZoomW
                    if self.zoomRect[0][1] < 0:
                        self.zoomRect[0][1] = 0
                        self.zoomRect[1][1] = ogZoomH
                    if self.zoomRect[0][0] > self.openImg.shape[1]:
                        self.zoomRect[0][0] = self.openImg.shape[1]-1
                        self.zoomRect[1][0] = self.openImg.shape[1]-ogZoomW-1
                    if self.zoomRect[0][1] > self.openImg.shape[0]:
                        self.zoomRect[0][1] = self.openImg.shape[0]-1
                        self.zoomRect[1][1] = self.openImg.shape[0]-ogZoomH-1

                    self.lastPan = [ogX,ogY]

                    self.dispImg()
                    return

                elif event == 0 and len(self.currentPoly) > 0: # mouse moved and making polygon
                    self.currentPoly[-1] = [x, y]
                    self.dispImg(drawAmount = 1)
                    return

                elif event == 1: # mouse down, make next point of polygon
                    if self.justMadePoly:
                        return

                    if len(self.currentPoly) == 0:
                        self.currentPoly = [[x, y], [x, y]]

                    else:
                        self.currentPoly.append([x, y])
                        self.dispImg(drawAmount = 1)
                    return

                elif event == 6: # middle mouse button up
                    if abs(ogX-self.startPan[0]) + abs(ogY-self.startPan[1]) < 10:
                        self.falseKey = 13
                    self.lastPan = []
                    self.startPan = []

                elif event == 3: # middle mouse button down
                    self.lastPan = [ogX, ogY]
                    self.startPan = [ogX, ogY]
                elif self.crosshair:
                    self.dispImg(drawAmount=1)


    def dispImg(self, drawAmount=0):
        self.xPad = int(50*self.windowScale)
        self.yPad = int(50*self.windowScale)
        self.buttonPad = int(200*self.windowScale)
        if self.selectedInd == -1:
            self.annotateMode = 0

        # get offset of image if you are zoomed in
        minX = 0
        minY = 0
        if len(self.zoomRect) > 0 and not self.zooming:
            minX = self.zoomRect[0][0]
            minY = self.zoomRect[0][1]


        if drawAmount < 1: # redraw everything - sometimes this is not necessary, so it can keep the previous image, which is much faster
            if len(self.zoomRect) > 0 and not self.zooming:
                openImg = self.openImg[self.zoomRect[0][1]:self.zoomRect[1][1], self.zoomRect[0][0]:self.zoomRect[1][0]]
            else:
                openImg = self.openImg.copy()


            # resize the image to take up as much of the window as possible
            self.scale = 1
            h, w = openImg.shape[0:2]
            if h/w > 850/1280:
                self.scale = 850*self.windowScale/h
                openImg = cv2.resize(openImg, (int(self.scale * w), int(self.scale*h)), interpolation = cv2.INTER_AREA)
            else:
                self.scale = 1280*self.windowScale/w
                openImg = cv2.resize(openImg, (int(self.scale * w), int(self.scale*h)), interpolation = cv2.INTER_AREA)
            self.scale/=self.realImgScale

            # make blank image to show
            drawImg = np.zeros((max(openImg.shape[0]+self.yPad*2, int(950*self.windowScale)), max(openImg.shape[1]+self.xPad*2+self.buttonPad*2, int(1300*self.windowScale)), 3), np.uint8)
            drawImg[:, self.buttonPad:-self.buttonPad] = (200,200,0)
            # draw buttons


            if self.annotateMode == 0:
                for i, buttonName in enumerate(self.openButtons):
                    cv2.rectangle(drawImg, (int(5*self.windowScale), int(self.windowScale*(10+i*55))), (self.buttonPad-2, int(self.windowScale*(5+(i+1)*55))), (255,255,0), -1)
                    cv2.rectangle(drawImg, (int(10*self.windowScale), int(self.windowScale*(15+i*55))), (self.buttonPad-int(self.windowScale*5)-2, int(self.windowScale*((i+1)*55))), (255,255,255), -1)
                    cv2.putText(drawImg, buttonName, (15, int(((i+1)*55-20)*self.windowScale)), cv2.FONT_HERSHEY_SIMPLEX, 1*self.windowScale, (255,0,0), max(int(2*self.windowScale),1), cv2.LINE_AA)
            elif self.annotateMode == 1:

                for i, buttonName in enumerate(self.selectButtons):
                    cv2.rectangle(drawImg, (int(5*self.windowScale), int(self.windowScale*(10+i*55))), (self.buttonPad-2, int(self.windowScale*(5+(i+1)*55))), (255,255,0), -1)
                    cv2.rectangle(drawImg, (int(10*self.windowScale), int(self.windowScale*(15+i*55))), (self.buttonPad-int(self.windowScale*5)-2, int(self.windowScale*((i+1)*55))), (255,255,255), -1)
                    cv2.putText(drawImg, buttonName, (15, int(((i+1)*55-20)*self.windowScale)), cv2.FONT_HERSHEY_SIMPLEX, 1*self.windowScale, (255,0,0), max(int(2*self.windowScale),1), cv2.LINE_AA)

            if self.classifying:
                buttonsShown = []
                for bs in self.fullImgClasses:
                    for c in self.imgClasses[str(self.frameNum)]:
                        if type(c) == list:
                            continue
                        if c in bs:
                            buttonsShown.append(c)
                            bs = bs[c]
                    buttonsShown += list(bs.keys())
                # print(buttonsShown)
                buttonsShown.append("last")
                buttonsShown.append("clear")
                buttonsShown.append("Regression")

                for i, buttonName in enumerate(buttonsShown):

                    color = (0,255,255)
                    if buttonName in self.imgClasses[str(self.frameNum)]:
                        color = (0,200,0)
                        
                    cv2.rectangle(drawImg, (drawImg.shape[1]-int(5*self.windowScale), int(self.windowScale*(10+i*55))), (drawImg.shape[1]-(self.buttonPad-2), int(self.windowScale*(5+(i+1)*55))), color, -1)
                    
                    if self.confirmedClasses:
                        cv2.rectangle(drawImg, (drawImg.shape[1]-int(10*self.windowScale), int(self.windowScale*(15+i*55))), (drawImg.shape[1]-(self.buttonPad-int(self.windowScale*5)-2), int(self.windowScale*((i+1)*55))), (255,255,255), -1)
                    else:
                        cv2.rectangle(drawImg, (drawImg.shape[1]-int(10*self.windowScale), int(self.windowScale*(15+i*55))), (drawImg.shape[1]-(self.buttonPad-int(self.windowScale*5)-2), int(self.windowScale*((i+1)*55))), (100,255,255), -1)
                    cv2.putText(drawImg, buttonName, (drawImg.shape[1]-(self.buttonPad-15), int(((i+1)*55-20)*self.windowScale)), cv2.FONT_HERSHEY_SIMPLEX, 1*self.windowScale, (255,0,0), max(int(2*self.windowScale),1), cv2.LINE_AA)

                    
            else:
                for i, v in enumerate(self.labelNames):
                    buttonName = self.labelNames[v]

                    if v == self.labelNum:
                        color1 = (0,255,0)
                    else:
                        color1 = (255,255,0)
                    if self.selectedInd >= 0 and self.allPolys[str(self.frameNum)][self.selectedInd][-1] == v:
                        color2 = (150, 255, 150)
                        self.allPolys[str(self.frameNum)][self.selectedInd][-1]
                    else:
                        color2 = (255,255,255)

                    color3 = (0,0,0)
                    if v != -1:
                        color3 = self.allLabelColors[v%len(self.allLabelColors)]
          
                    buttonSize = 50

                    cv2.rectangle(drawImg, (drawImg.shape[1]-int(5*self.windowScale), int(self.windowScale*(10+i*buttonSize))), (drawImg.shape[1]-(self.buttonPad-2), int(self.windowScale*(5+(i+1)*buttonSize))), color1, -1)
                    
                    cv2.rectangle(drawImg, (drawImg.shape[1]-int(10*self.windowScale), int(self.windowScale*(15+i*buttonSize))), (drawImg.shape[1]-(self.buttonPad-int(self.windowScale*5)-2), int(self.windowScale*((i+1)*buttonSize))), color2, -1)
                    cv2.putText(drawImg, buttonName, (drawImg.shape[1]-(self.buttonPad-15), int(((i+1)*buttonSize-15)*self.windowScale)), cv2.FONT_HERSHEY_SIMPLEX, 1*self.windowScale, color3, max(int(2*self.windowScale),1), cv2.LINE_AA)



            # draw info text

            classText = ""
            for i in self.imgClasses[str(self.frameNum)]:
                if type(i) == list:
                    continue
                classText += i + " "
            if len(classText) > 0:
                cv2.putText(drawImg, classText, (int(drawImg.shape[1]/4), self.yPad-10), cv2.FONT_HERSHEY_SIMPLEX, 1*self.windowScale, (0,0,255), max(int(2*self.windowScale),1), cv2.LINE_AA)



            txtVals = ["Label Type: " + str(self.labelNum),  "# labeled: " + str(self.numVidAnnotations) + " (" + str(self.totalAnnotations) + ")", "Vid: " + self.vidName, "Vid #: " + str(self.vidIndex) + "/" + str(len(self.videos)), "Frame: " + str(self.frameNum) + "/" + str(self.maxFrameNum)]
            if self.labelNum in self.labelNames:
                txtVals += [self.labelNames[self.labelNum]]
            for i, val in enumerate(txtVals):
                color = (255,255,255)
                if i==0:

                    if self.labelNum >= 0:
                        colors =[(255, 0, 0),  (0, 255, 0), (0, 0, 255),(255, 255, 0),(255, 0, 255), (0, 255, 255), (0, 165, 255),(0, 128, 128), (128, 128, 128)]

                        color = colors[self.labelNum%len(colors)]

                    # if self.labelNum == 0: color=(255,0,0)
                    if self.labelNum == -1: color=(0,0,255)
                    if self.labelNum == -2: color =(255,255,0)
                    if self.labelNum == -3: color =(0,255,255)
                cv2.putText(drawImg, val, (15, int(self.windowScale*(840+i*20))), cv2.FONT_HERSHEY_SIMPLEX, 0.5*self.windowScale, color,  max(int(1*self.windowScale),1), cv2.LINE_AA)

            # groupColors = [(0,255,255), (255,255,0), (100,100,100)]

            # sort groups by x position
            # groups = {}
            # for poly in self.allPolys[str(self.frameNum)]:
            #     mnX = 100000
            #     mxX = -100000
            #     if type(poly[-1]) == int:
            #         if poly[-1] < 10:
            #             continue
            #     elif type(poly[-2]) == int:
            #         if poly[-2] < 10:
            #             continue
            #     for pt in poly:

            #         if type(pt) == int:
                        
            #             if pt in groups:
            #                 groups[pt][0] = min(groups[pt][0], mnX)
            #                 groups[pt][1] = max(groups[pt][1], mxX)
            #             else:
            #                 groups[pt] = [mnX, mxX]
            #             break
            #         else:
            #             mnX = min(pt[0], mnX)
            #             mxX = max(pt[0], mxX)
            # sorted_groups = sorted(groups, key=lambda k: (groups[k][0]+groups[k][1])/2)


            # draw annotation masks
            self.maskImg = np.zeros((openImg.shape[0], openImg.shape[1], 3), np.uint8) 
            for poly in self.allPolys[str(self.frameNum)]:
                scaledPoly = []
                borderLabels = []
                labelNum = poly[-1]
                if self.classView != -1 and labelNum != self.classView:
                    continue
                if type(poly[-1]) != int:
                    borderLabels = poly[-1]
                    poly = poly[0:-1]

                for pt in poly:
                    if type(pt) != int: # last element is label number
                        scaledPoly.append([int((pt[0]-minX)*self.scale), int((pt[1]-minY)*self.scale)])
                    else:
                        # if pt >= 10:
                        #     labelColor=groupColors[sorted_groups.index(pt)%len(groupColors)]
                        # else:
                        labelColor = self.allLabelColors[pt%len(self.allLabelColors)]
                
                cv2.fillPoly(self.maskImg, pts = [np.array(scaledPoly)], color=labelColor)

                
                
                


            # combine mask and image to get the bluish tint around annotations
            # cv2.imshow("mi", self.maskImg)
            openImg = cv2.addWeighted(self.maskImg, self.maskWeight, openImg, 1, 0, openImg)

            for ind, poly in enumerate(self.allPolys[str(self.frameNum)]):
                scaledPoly = []
                borderLabels = []
                if type(poly[-1]) != int:
                    borderLabels = poly[-1]
                    poly = poly[0:-1]
                for pt in poly:
                    if type(pt) != int: # last element is label number
                        scaledPoly.append([int((pt[0]-minX)*self.scale), int((pt[1]-minY)*self.scale)])
                    else:
                        labelColor = self.allLabelColors[pt%len(self.allLabelColors)]
                        if pt >= 10:
                            random.seed(pt)
                            labelColor = (max(min(labelColor[0] + random.randint(-100,100), 200),0), max(min(labelColor[1] + random.randint(-100,100), 200),0), max(min(labelColor[2] + random.randint(-100,100), 200),0))


                if self.useBorders and self.labelNum in [-2, -3]:
                    if len(borderLabels) == 0:
                        borderLabels = [0]*len(scaledPoly)
                    lastPt = scaledPoly[-1]
                    for i, pt in enumerate(scaledPoly):
                        if borderLabels[i] == 0 and (lastPt[0]!=0 or pt[0]!=0) and (lastPt[1]!=0 or pt[1]!=0) and (lastPt[0]!=openImg.shape[1] or pt[0]!=openImg.shape[1]) and (lastPt[1]!=openImg.shape[0] or pt[1]!=openImg.shape[0]):

                            labelColor = (0,0,255)
                            openImg = cv2.line(openImg, pt, lastPt, labelColor, 4)
                        # else:
                        #   openImg = cv2.line(openImg, pt, lastPt, (0,0,0), 1)
                        lastPt = pt
                        # cv2.rectangle(openImg, pt, pt, (0,0,0), 15)
                        # cv2.putText(openImg, str(i), (pt[0]-random.randint(0,100), pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                else:
                    if ind == self.selectedInd:

                        cv2.polylines(openImg, [np.array(scaledPoly)], True, (0,0,0), 3)
                        for i, pt in enumerate(scaledPoly):
                            if i == self.selectedPt:
                                cv2.rectangle(openImg, pt, pt, (0,0,255), 10)
                            else:
                                cv2.rectangle(openImg, pt, pt, (255,0,0,), 5)
                    else:
                        

                        cv2.polylines(openImg, [np.array(scaledPoly)], True, labelColor, 1)

            # add the image to the overall canvas
            drawImg[self.yPad:openImg.shape[0]+self.yPad, self.xPad+self.buttonPad:openImg.shape[1]+self.xPad+self.buttonPad] = openImg

            # draw the line borders of the annotations
            for poly in self.allPolys[str(self.frameNum)]:
                scaledPoly = []
                for pt in poly:
                    if type(pt) == int:
                        labelColor = self.allLabelColors[pt%len(self.allLabelColors)]
                    else:
                        try:
                            scaledPoly.append([int((pt[0]-minX)*self.scale), int((pt[1]-minY)*self.scale)])
                        except:
                            print("Critical error with polygon", poly)
                            self.allPolys[str(self.frameNum)] = []
                            return
                cv2.fillPoly(self.maskImg, pts=[np.array(scaledPoly)], color=(255,0,0))

            if self.classifying:
                color = (255,0,0)
                if "hit" in self.imgClasses[str(self.frameNum)]:
                    color = (0,0,255)

                thickness = 1
                if self.confirmedClasses:
                    thickness = 4
                wingColor = (0,255,255)
                if "Interstate" in self.imgClasses[str(self.frameNum)]:
                    wingColor = (255, 0, 0)
                elif "Country" in self.imgClasses[str(self.frameNum)]:
                    wingColor = (0, 200, 0)


                if "Left Wing" in self.imgClasses[str(self.frameNum)]:

                    if "On Road" in self.imgClasses[str(self.frameNum)]:
                        cv2.line(drawImg, (drawImg.shape[1]//2, drawImg.shape[0]-220), (drawImg.shape[1]//2-150, drawImg.shape[0]), (255, 100, 0), thickness)

                    if "On Shoulder" in self.imgClasses[str(self.frameNum)]:
                        cv2.line(drawImg, (drawImg.shape[1]//2, drawImg.shape[0]-220), (drawImg.shape[1]//2-200, drawImg.shape[0]), (100, 255, 0), thickness)


                    cv2.line(drawImg, (drawImg.shape[1]//2, drawImg.shape[0]-50), (drawImg.shape[1]//2-200, drawImg.shape[0]-200), wingColor, thickness)
                    if "Manuever" in self.imgClasses[str(self.frameNum)]:
                        cv2.line(drawImg, (drawImg.shape[1]//2-220, drawImg.shape[0]-50), (drawImg.shape[1]//2-220, drawImg.shape[0]-220), color, 10)
                else:
                    cv2.line(drawImg, (drawImg.shape[1]//2, drawImg.shape[0]-50), (drawImg.shape[1]//2-200, drawImg.shape[0]-50), wingColor, thickness)
                if "Right Wing" in self.imgClasses[str(self.frameNum)]:

                    if "On Road" in self.imgClasses[str(self.frameNum)]:
                        cv2.line(drawImg, (drawImg.shape[1]//2, drawImg.shape[0]-220), (drawImg.shape[1]//2+150, drawImg.shape[0]), (255, 100, 0), thickness)

                    if "On Shoulder" in self.imgClasses[str(self.frameNum)]:
                        cv2.line(drawImg, (drawImg.shape[1]//2, drawImg.shape[0]-220), (drawImg.shape[1]//2+200, drawImg.shape[0]), (100, 255, 0), thickness)


                    cv2.line(drawImg, (drawImg.shape[1]//2, drawImg.shape[0]-50), (drawImg.shape[1]//2+200, drawImg.shape[0]-200), wingColor, thickness)
                    if "Manuever" in self.imgClasses[str(self.frameNum)]:
                        cv2.line(drawImg, (drawImg.shape[1]//2+220, drawImg.shape[0]-50), (drawImg.shape[1]//2+220, drawImg.shape[0]-220), color, 10)
                else:
                    cv2.line(drawImg, (drawImg.shape[1]//2, drawImg.shape[0]-50), (drawImg.shape[1]//2+200, drawImg.shape[0]-50), wingColor, thickness)


            self.baseImg = drawImg.copy()

        else:

            drawImg = self.baseImg.copy()


        if self.regressioning:
            cs = [(100, 0,100), (20, 20, 150), (20,255, 20)]
            for r in self.imgClasses[str(self.frameNum)]:
                if type(r) == list:
                    y = r[0]

                    if y > 0:
                        y = int((y-minY)*self.scale + self.yPad)
                        for j, m in enumerate(r[1::]):
                            if m[0]>0:
                                
                                color = cs[j]
                                x1 = int((m[0]-minX)*self.scale + self.buttonPad + self.xPad)
                                x2 = int((m[1]-minX)*self.scale + self.buttonPad + self.xPad)
                                cv2.line(drawImg, (x1, y-10*j), (x2, y-10*j), color, 5)

        if self.classifying:
            for r in self.imgClasses[str(self.frameNum)]:
                if type(r) == list:
                    mower = r[1]
                    shoulder = r[2]
                    road = r[3]
                    if shoulder[0]-shoulder[1] != 0:
                        x1, x2 = sorted(mower)
                        x3, x4 = sorted(shoulder)
                        start = max(x1, x3)
                        end = min(x2, x4)
                        overlap = int(100*max(0, end - start) / abs(shoulder[0]-shoulder[1]))

                        cv2.putText(drawImg, str(overlap), (drawImg.shape[0]-50, drawImg.shape[1]//2-100), cv2.FONT_HERSHEY_SIMPLEX, 1*self.windowScale, (0,0,255), max(int(2*self.windowScale),1), cv2.LINE_AA)
                    if road[0]-road[1] != 0:
                        x1, x2 = sorted(mower)
                        x3, x4 = sorted(road)
                        start = max(x1, x3)
                        end = min(x2, x4)
                        overlap = int(100*max(0, end - start) / abs(road[0]-road[1]))
                        cv2.putText(drawImg, str(overlap), (drawImg.shape[0]+50, drawImg.shape[1]//2-100), cv2.FONT_HERSHEY_SIMPLEX, 1*self.windowScale, (255,0,0), max(int(2*self.windowScale),1), cv2.LINE_AA)



        # draw the polygon currently being made
        if len(self.currentPoly) > 0:
            scaledPoly = []
            for pt in self.currentPoly:
                scaledPoly.append([int((pt[0]-minX)*self.scale + self.buttonPad + self.xPad), int((pt[1]-minY)*self.scale + self.yPad)])
            cv2.polylines(drawImg, [np.array(scaledPoly)], True, (0,0,0), 1)

        # draw the zoom rectangle
        if len(self.zoomRect) > 0 and self.zooming:
            cv2.rectangle(drawImg, (int(self.zoomRect[0][0]*self.scale+self.buttonPad+self.xPad), int(self.zoomRect[0][1]*self.scale+self.yPad)), 
                                    (int(self.zoomRect[1][0]*self.scale+self.buttonPad+self.xPad), int(self.zoomRect[1][1]*self.scale+self.yPad)), 
                                    (0,0,0), 2)

        if len(self.sam_rect) == 4:
            cv2.rectangle(drawImg, (int(self.sam_rect[0]*self.scale+self.buttonPad+self.xPad), int(self.sam_rect[1]*self.scale+self.yPad)), 
                                    (int(self.sam_rect[2]*self.scale+self.buttonPad+self.xPad), int(self.sam_rect[3]*self.scale+self.yPad)), 
                                    (0,0,0), 2)

        if self.crosshair and self.mouseX > self.buttonPad:
            cv2.line(drawImg, (self.mouseX, self.yPad), (self.mouseX, drawImg.shape[0]-self.yPad), (0,0,0),1)
            cv2.line(drawImg, (self.xPad + self.buttonPad, self.mouseY), ( drawImg.shape[1]-self.yPad, self.mouseY), (0,0,0),1)

        cv2.imshow("img", drawImg)


        if self.firstCallback: # allow mouse events
            self.firstCallback = False
            cv2.setMouseCallback("img", self.mouseEvent)



    def runThrough(self):
        allFiles = os.listdir(self.videosPath)

        # get the videos from the files
        self.videos = []
        
        priorities = ["oct", "sep", "Aug", "july", "avi", "jun"]
        groups = []
        for i in range(len(priorities)+1):
            groups += [[]]
        # print(groups)

        allFiles = sorted(allFiles)
        for i in allFiles:
            if i[-4::].lower() in [".mov", ".mp4", ".avi", ".mkv", ".jpg", ".png"]:
                added = False
                for n, j in enumerate(priorities):
                    if j in i:
                        groups[n].append(i)
                        added = True
                        break
                if not added:
                    groups[-1].append(i)


        for g in groups:
            self.videos += g

        
        print(self.videos)

        self.vidGroups = 0
        self.infoJsonName = "info_" + self.saveName + ".json"
        if self.infoJsonName in os.listdir(self.videosPath):
            with open(os.path.join(videosPath, self.infoJsonName)) as json_file:
                self.vidInfo = json.load(json_file)
                
                self.totalAnnotations = 0
                for vidName in self.vidInfo:
                    self.totalAnnotations += self.vidInfo[vidName][0]
                    if len(self.vidInfo[vidName]) > 1:
                        self.totalGroups += self.vidInfo[vidName][1]
        else:
            self.totalAnnotations = 0
            self.vidInfo = {}
            



        # go through all the videos
        self.vidIndex = 0
        while self.vidIndex < len(self.videos):
            self.vidName = self.videos[self.vidIndex]
            newVid = True

            self.startVidGroups = 0
            if self.vidName in self.vidInfo:
                if len(self.vidInfo[self.vidName]) > 1:
                    self.startVidGroups = self.vidInfo[self.vidName][1]
            self.vidGroups = self.startVidGroups
            print("start", self.startVidGroups)

            # load the annotations
            self.allPolys = {}
            self.jsonName = self.vidName[0:-4] + self.saveName + ".json"
            if self.jsonName in os.listdir(self.videosPath) and True:
                print("found json", self.jsonName)
                with open(os.path.join(videosPath, self.jsonName)) as json_file:
                    self.allPolys = json.load(json_file)
                    if "classInfo" in self.allPolys:
                        self.imgClasses = self.allPolys["classInfo"]
                        del self.allPolys["classInfo"]
                    else:
                        self.imgClasses = {}
                    self.numVidAnnotations = len(list(self.allPolys.keys()))

                    for fr in self.imgClasses:
                        # print(fr)
                        if "Unusual" in self.imgClasses or "Maintenance" in self.imgClasses:
                            print("unusual at", fr)
            else:

                self.imgClasses = {}
                self.numVidAnnotations = 0

            # begin video
            if self.vidName[-4::].lower() in [".jpg", ".png"]:
                self.maxFrameNum = 1
            else:
                self.cap = cv2.VideoCapture(os.path.join(self.videosPath, self.vidName))
                self.maxFrameNum = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.frameNum = 0

            if self.numVidAnnotations > 0:
                print("annotated 1 in", self.maxFrameNum / self.numVidAnnotations, "frames")


            

            # go through all the frames of the video
            while 0 <= self.frameNum < self.maxFrameNum:
                if self.openImgFrame != self.frameNum or newVid:
                    newVid=False
                    if self.vidName[-4::].lower() in [".jpg", ".png"]:
                        self.openImg = cv2.imread(os.path.join(self.videosPath, self.vidName))
                    else:
                        if self.openImgFrame != self.frameNum-1 :
                            self.cap.set(1, self.frameNum)
                        ret, self.openImg = self.cap.read()
                        # print(self.openImg.shape)
                        self.openImgFrame = self.frameNum
                        if not ret:
                            self.frameNum = self.maxFrameNum + 100
                            print("Error reading video. Saving in 10 seconds   ")
                            time.sleep(10)
                            continue
                if str(self.frameNum) not in self.imgClasses:
                    self.imgClasses[str(self.frameNum)] = []

                if self.classifying and len(self.imgClasses[str(self.frameNum)]) == 0:
                    self.makeClassifierPred()
                    self.confirmedClasses = False
                else:
                    self.confirmedClasses = True

                if self.showPreds:
                    self.makePredExample()

            
                # print("keys", list(self.allPolys.keys()))
                if self.numVidAnnotations != len(list(self.allPolys.keys())):
                    self.numVidAnnotations = len(list(self.allPolys.keys()))
                    self.totalAnnotations = 0
                    for vidName in self.vidInfo:
                        if vidName != self.vidName:
                            self.totalAnnotations += self.vidInfo[vidName][0]
                            
                    self.totalAnnotations += self.numVidAnnotations

                
                if str(self.frameNum) not in self.allPolys: # get the annotation
                    self.allPolys[str(self.frameNum)] = []

                self.lastState = self.allPolys[str(self.frameNum)]
                

                self.dispImg(drawAmount=0) # show the annotation thing
                change = self.manageKeyResponses() # get key responses


                if not self.confirmedClasses:
                    self.imgClasses[str(self.frameNum)] = []

                if len(self.imgClasses[str(self.frameNum)]) == 0:
                    del self.imgClasses[str(self.frameNum)]



                self.selectedPt = -1
                self.selectedInd = -1
                self.annotateMode = 0

                

                # print("groups total", self.totalGroups, "groups made", self.groupsMade, "vid groups", self.vidGroups)


                if len(self.allPolys[str(self.frameNum)]) == 0: # remove annotations from main dict if there are no annotations
                    del(self.allPolys[str(self.frameNum)])
                self.frameNum += change

            self.saveJSON() # save the data

            # go to the appropriate video
            if self.frameNum < 0:
                self.vidIndex = max(0, self.vidIndex-1)
            elif self.frameNum > 0:
                self.vidIndex = min(len(self.videos)+1, self.vidIndex+1)
            else:
                print("zero framenum")
                self.vidIndex = min(len(self.videos)+1, self.vidIndex+1)

            self.currentPoly = []
        print("done with all videos")

    def makePoly(self, addCurrentPoly=True):
        """
        Add the polygon the user just made. It may be erasing the other polygons
        """

        

        self.lastState = self.allPolys[str(self.frameNum)][:]
        

        self.unsavedChanges = True

        if addCurrentPoly:
            # messy way to check if the polygon is valid
            poly = Polygon(self.currentPoly)
            poly2 = Polygon([(0, 0), (0, 1), (1, 1)])
            try:
                unary_union([poly,poly2])
            except:
                print("error: polygon likely has overlapping lines, which is not allowed" )
                self.currentPoly = []
                return
            self.allPolys[str(self.frameNum)].append(self.currentPoly[0:-1] + [self.labelNum])

            if not self.merge:
                return
        try:
            # merge any polygons with the created polygon

            # make shapely polygons out of all of the polygons here
            polys = {}
            
            hold = self.allPolys[str(self.frameNum)][:]

            unused = []

            for i, p in enumerate(self.allPolys[str(self.frameNum)]):
                used = False
                i2 = 1
                while i2 < len(p):
                    pt = p[i2]
                    lastPt = p[i2-1]
                    if type(pt) == int:
                        break
                    i3=0
                    while i3 < len(self.currentPoly):
                        pt2 = self.currentPoly[i3]
                        lastPt2 = self.currentPoly[i3-1]
                        intersecting = self.intersect(pt, lastPt, pt2, lastPt2)
                        if intersecting is not None:
                            used = True
                            break
                        i3 += 1


                    if used:
                        break
                    
                    i2+=1
                pt = p[0]
                if self.is_point_in_polygon(pt, self.currentPoly):
                    used = True

                if used:
                    if type(p[-1]) != int:
                        p = p[0:-1]
                    if len(p) > 3:
                        label = p[-1]
                        # if label > 10:
                        #     label = label%10

                        if str(label) in polys:
                            polys[str(label)] += [Polygon(p[0:-1])]
                        else:
                            polys[str(label)] = [Polygon(p[0:-1])]
                else:
                    unused += [p]

            # wipe all polygons, will add them back once merged
            self.allPolys[str(self.frameNum)] = unused


            if self.labelNum >= 0: # making a new polygon - merge it
                for label in polys:
                    try:
                        mergedPolys = unary_union(polys[label])
                    except:
                        self.allPolys[str(self.frameNum)] = []#hold
                        print("error something went wrong merging the polygons Sorry! - location 0")
                        return
                    
                    if isinstance(mergedPolys, MultiPolygon): # if there are multiple polygons, shapely makes it a multipoly, which is different
                        polygon_coords = [list(poly.exterior.coords) for poly in list(mergedPolys.geoms)]
                    else:
                        polygon_coords = [list(mergedPolys.exterior.coords)]

                    for poly in polygon_coords:
                        intPoly = [[int(x[0]), int(x[1])] for x in poly]
                        self.allPolys[str(self.frameNum)] += [intPoly + [int(label)]]

            elif self.labelNum == -1: # erasing a polygon
                for label in polys:
                    if label == "-1":
                        continue
                    try:
                        for poly in polys[label]:

                            try:
                                diffPoly = poly.difference(polys["-1"][0])
                            except:
                                self.allPolys[str(self.frameNum)] = []
                                print("error something went wrong merging the polygons. Sorry! location 1")
                                continue

                            if isinstance(diffPoly, MultiPolygon): # if there are multiple polygons, shapely makes it a multipoly, which is different
                                polygon_coords = [list(poly.exterior.coords) for poly in list(diffPoly.geoms)]
                            else:
                                polygon_coords = [list(diffPoly.exterior.coords)]

                            for poly in polygon_coords:
                                intPoly = [[int(x[0]), int(x[1])] for x in poly]
                                if len(intPoly)>0:
                                    self.allPolys[str(self.frameNum)] += [intPoly[0:-1] + [int(label)]]
                    except:
                        print("error: something was wrong with the polygon made. Sorry! location 2")
                        self.allPolys[str(self.frameNum)] = []
        except:
            print("Critical error. Reason unknown")


    def group(self):
        random.seed(time.time())
        idNum = random.randint(0,1000) * 10
        groupPolys = []
        for i, p in enumerate(self.allPolys[str(self.frameNum)]):
            used = False
            i2 = 1
            while i2 < len(p):
                pt = p[i2]
                lastPt = p[i2-1]
                if type(pt) == int:
                    break
                i3=0
                if self.is_point_in_polygon(pt, self.currentPoly):
                    used = True
                    break

                while i3 < len(self.currentPoly):
                    pt2 = self.currentPoly[i3]
                    lastPt2 = self.currentPoly[i3-1]
                    intersecting = self.intersect(pt, lastPt, pt2, lastPt2)


                    if intersecting is not None:
                        used = True
                        break

                    i3 += 1

                if used:
                    break
                
                i2+=1

            if used:
                ind = -2
                if type(p[-1]) == int:
                    ind=-1
                self.unsavedChanges = True
                self.allPolys[str(self.frameNum)][i][ind] = self.allPolys[str(self.frameNum)][i][ind]%10 + idNum



    def toggleBorders(self):
        """
        basically makes/removes borders around the stalk.
        It adds a point to the polygon at the location where the user selected polygon intersects the polygon

        """
        self.unsavedChanges = True
        for pNum, p in enumerate(self.allPolys[str(self.frameNum)]):
            if type(p[-1]) == int:
                borders = [0]*(len(p)-1)
                labelNum = p[-1]
                p += [borders]
            else:
                labelNum = p[-2]
                borders = p[-1]


            changedPts = []
            ptsToAdd = []
            i=1
            while i < len(p) - 2:
                i = max(1, i)
                pt = p[i]
                lastPt = p[i-1]

                lastPt2 = self.currentPoly[-1]
                for i2, pt2 in enumerate(self.currentPoly):

                    check = self.intersect(pt2, lastPt2, pt, lastPt)


                    if check is not None and not math.isnan(check[0]) and not math.isnan(check[1]):
                        new = True
                        # for a in p[0:-2]: # uncomment this loop to make it not recreate a point even if there is already a point present
                            # if abs(a[0] - int(check[0])) < 2 and abs(a[1] - int(check[1])) < 2:
                            #   new = False
                            #   break

                        dist = (check[0]-pt[0])**2 + (check[1]-pt[1])**2
                        ptsToAdd += [[int(check[0]), int(check[1]), i, dist, new]]
                
                    lastPt2 = pt2

                i+=1
                
            # ptsToAdd = sorted(ptsToAdd, key=lambda val: val[2])
            ptGroups = {}
            for pt in ptsToAdd:
                if str(pt[2]) in ptGroups:
                    ptGroups[str(pt[2])] += [pt]
                else:
                    ptGroups[str(pt[2])] = [pt]
            newP = []

            if self.labelNum == -2:
                val = 1
            else:
                val = 0
    
            oldBorders = borders[:]
            borders = []
            inPoly = self.is_point_in_polygon(p[0], self.currentPoly)
            for i, pt in enumerate(p[0:-2]):
                
                if str(i) in ptGroups:
                    s = sorted(ptGroups[str(i)], key=lambda val: -val[3])
                    for j in s:
                        if j[4]:
                            newP += [[j[0], j[1]]]
                            if inPoly:
                                borders += [val]
                                inPoly = False
                            else:
                                borders += [oldBorders[i]]
                                inPoly = True
                        else:
                            inPoly = not inPoly
                        

                newP += [pt]
                if inPoly:
                    borders += [val]
                else:
                    borders += [oldBorders[i]]
            p = newP + [labelNum, []]



            # print("changed pt", changedPts)
            # for i in changedPts:
            #   if vals > 0:
            #       borders[i] = 0
            #   else:
            #       borders[i] = 1

            # borders = [0]*(len(p)-1)
            p[-2] = labelNum
            p[-1] = borders
            self.allPolys[str(self.frameNum)][pNum] = p



    def saveJSON(self):
        """
        Save the annotations
        """

        polyNames = list(self.allPolys.keys())
        for i in polyNames:
            if len(self.allPolys[i]) == 0:
                del(self.allPolys[i])

        if self.unsavedChanges:
            a = self.allPolys
            a["classInfo"] = self.imgClasses
            jsonStr = json.dumps(a)
            with open(os.path.join(self.videosPath, self.jsonName), 'w') as fileHandle:
                fileHandle.write(str(jsonStr))
                fileHandle.close()

            print("saved json")
        else:
            print("no changes made")


        self.numVidAnnotations = len(list(self.allPolys.keys()))
        updateVidInfo = False
        if (self.vidName in self.vidInfo and self.vidInfo[self.vidName][0] == self.numVidAnnotations):# and self.vidGroups == self.startVidGroups:
            pass
        else:
            self.vidInfo[self.vidName] = [self.numVidAnnotations]
            jsonStr = json.dumps(self.vidInfo)
            with open(os.path.join(self.videosPath, self.infoJsonName), 'w') as fileHandle:
                print("saving number of videos")
                fileHandle.write(str(jsonStr))
                fileHandle.close()


        self.unsavedChanges = False


    def is_point_in_polygon(self, point, polygon):
        num_vertices = len(polygon)
        x, y = point
        inside = False

        p1x, p1y = polygon[0]
        for i in range(num_vertices + 1):
            p2x, p2y = polygon[i % num_vertices]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside


    def ccw(self, A, B, C):
        """
        Checks if three points A, B, and C are in counterclockwise order.
        Returns True if counterclockwise, False if clockwise or collinear.
        """
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


    def intersect(self, p1, p2, p3, p4):
        x1,y1 = p1
        x2,y2 = p2
        x3,y3 = p3
        x4,y4 = p4
        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if denom == 0: # parallel
            return None
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        if ua < 0 or ua > 1: # out of range
            return None
        ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
        if ub < 0 or ub > 1: # out of range
            return None
        x = x1 + ua * (x2-x1)
        y = y1 + ua * (y2-y1)
        return (x,y)

    def intersect2(self, A, B, C, D):
        """
        Finds the intersection point of two line segments AB and CD (if any).
        Returns the intersection point (as a tuple (x, y)) if they intersect,
        otherwise returns None.
        """
        if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):
            # The line segments intersect; calculate the intersection point
            slope_AB = (B[1] - A[1]) / (B[0] - A[0]) if B[0] != A[0] else float('inf')
            slope_CD = (D[1] - C[1]) / (D[0] - C[0]) if D[0] != C[0] else float('inf')

            if slope_AB == slope_CD:
                # The line segments are collinear but may overlap
                # Check for overlapping points (common endpoints)
                if A == C or A == D:
                    return A
                elif B == C or B == D:
                    return B
                elif A[0] <= C[0] <= B[0] or B[0] <= C[0] <= A[0]:
                    return C
                elif A[0] <= D[0] <= B[0] or B[0] <= D[0] <= A[0]:
                    return D
                else:
                    return None
            else:
                # Calculate the intersection point using the lines' equations
                x = (C[1] - A[1] + slope_AB * A[0] - slope_CD * C[0]) / (slope_AB - slope_CD)
                y = A[1] + slope_AB * (x - A[0])
                return (x, y)
        else:
            # The line segments do not intersect
            return None



    def autoBorder(self):

        # for i in self.allPolys
        
        if self.segger is None:
            # import run_stalk_seg_mri as ss
            import run_stalk_seg as ss
            self.segger = ss
            
        for rect in rects:
            img = self.openImg[rect[1]:rect[3], rect[0]:rect[2]]
            stalkSegImg, edgeSegImg = self.segger.runSeg(cv2.resize(img, (128,128*2), interpolation=cv2.INTER_AREA))

            self.addAutoSegAnnot(stalkSegImg, edgeSegImg, rect)

        self.dispImg()


    def makeClassifierPred(self):
        if self.classifierModel is None:
            import run_classifier
            self.classifierModel = run_classifier
            self.classifierModel.load_model(self.classifierName)
            

        if len(self.last_suggestions) > 0 and self.fastmode and self.frameNum%3!=0:
            self.imgClasses[str(self.frameNum)] = self.last_suggestions
            return

        classes, obstacle = self.classifierModel.run_model(self.openImg)
        rs = []
        for r in classes:
            if len(r[0]) > 1:
                a=[]
                # print(r, len(r[0]))
                for l in r[0]:

                    a.append(float(l))
                rs.append(a)
            else:
                rs.append(float(r[0]))
        # print(rs)
        if rs[0] > 0.5:
            self.imgClasses[str(self.frameNum)].append("Left Wing")

        if rs[1] > 0.5:
            self.imgClasses[str(self.frameNum)].append("Right Wing")
     

        locations = ["Interstate", "Country"]

        if "Country" in self.lastImgClass:
            ind = 1
        elif "Interstate" in self.lastImgClass:
            ind = 0
        else:
            ind = np.argmax(np.array(rs[2]))

        # else:
        #     ind = self.last_location 

        # if rs[2] > 0.5:
        #     ind=1
        # else:
        #     ind=0
        # ind = 1

        if rs[5] > 0.0:#0.2:
            self.imgClasses[str(self.frameNum)].append("On Shoulder")
        if rs[6] > 0.08:#0.2:
            self.imgClasses[str(self.frameNum)].append("On Road")

        print(rs[5],rs[6])

        self.imgClasses[str(self.frameNum)].append(locations[ind])

        actions = ["Normal mow", "Manuever", "Transit"]
        ind = np.argmax(np.array(rs[3]))
        if ind == 1 and rs[0]<0.5 and rs[1]<0.5:
            ind=0
        if ind == 1 and rs[0]>0.5 and rs[1]>0.5 and (rs[5] > 0.2 or rs[6] > 0.2):
            ind=2

        self.imgClasses[str(self.frameNum)].append(actions[ind])

        if "Manuever" in self.imgClasses[str(self.frameNum)]:
            self.fastmode=False

        if ind == 1:
            # print("maneuvering")
            ind = obstacle#np.argmax(np.array(rs[4]))
            if ind != 10:
                maneuverTypes = ["Sign", "Post", "Roadkill", "Trash", "Cone", "Guardrail", "Drain", "Mailbox", "Branches", "Telephone Pole"] 
                # print(maneuverTypes[ind], ind)
                self.imgClasses[str(self.frameNum)].append(maneuverTypes[ind])
            # else:
            #     print("ind na")

        # print("shoulder", rs[5], "road", rs[6])
        self.last_suggestions = self.imgClasses[str(self.frameNum)]
        




    def addAutoSegAnnot(self, stalkSegImg, edgeSegImg, rect, threshold=180):
    
        stalkSegImg[stalkSegImg < threshold] = 0
        stalkSegImg[stalkSegImg > 0] = 255
        gray = stalkSegImg
    
        contours, hierarchy = cv2.findContours(gray, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cnts = []
        h, w = rect[3]-rect[1], rect[2]-rect[0]

        c = random.randint(100, 255**3)*10
        for cnt in contours:
            poly = []
            edged = []
            for pt in cnt:
                x, y = pt[0][0], pt[0][1]
                if edgeSegImg[y,x] > 10:
                    edged += [0]
                else:
                    edged += [1]
                poly += [[int(x*w/128+rect[0]), int(y*h/256+rect[1])]]

            if len(poly) > 5:
                res = poly + [c] + [edged]
            
                self.allPolys[str(self.frameNum)] += [res]


    def make_sam_annot(self, img, bbox):

        if self.sam_model is None:
            print("loading sam")
            from ultralytics import SAM
            self.sam_model = SAM("sam2_b.pt")
            print("loaded sam")
        h, w = img.shape[0:2]
        results = self.sam_model.predict(img, bboxes=bbox)#, conf=0.2)
        res = []

        for result in results:
            if result.masks is None:
                continue
            for j, i in enumerate(result.masks.xy):
                cnt = i
                epsilon = 0.001 * cv2.arcLength(cnt, True)

                approx = cv2.approxPolyDP(cnt, epsilon, True)
                newPoly = []
                for pt in approx:
                    newPoly.append([int(pt[0][0]), int(pt[0][1])])

                if len(newPoly) > 4:
                    newPoly.append(self.labelNum)
                    self.allPolys[str(self.frameNum)].append(newPoly)


    def detectPossible(self, img):
        self.lastState = self.allPolys[str(self.frameNum)]
        print("detecting")
        if self.model is None:
            from ultralytics import YOLO
            print("model", self.modelName)

            self.model = YOLO(self.modelName)

        # resize the image to be square and get predictions. You don't need to make the image a square, but I do it because I don't want to appropriately crop the image.
        h, w = img.shape[0:2]
        imageSize = 640#416
        results = self.model.predict(source=cv2.resize(img, (imageSize, imageSize), interpolation = cv2.INTER_AREA), save=False)#, conf=0.2)

        res = []

        # iterate over each bounding box and its associated confidence
        for result in results:

            # sloppy way to check if there are masks from model prediction
            segd = False
            try:
                x = result.masks.xy
                segd = True
            except:
                print("no mask")
                return

            polysMade = []
            # add the masks to the list
            for j, i in enumerate(result.masks.xy):
                l = []
                # print(i)
                cc= 0 
                cc = int(result.boxes.cls[j])
                # if cc != 2:
                #     continue
                lastPt = [-1,-1]

                minX, maxX = imageSize, -1
                for n in range(i.shape[0]):
                    minX, maxX = min(minX, i[n, 0]), max(maxX, i[n, 0])
                for n in range(i.shape[0]):


                    # sometimes the annotation doesnt reach all the way to the border
                    pt = [int(i[n, 0]*w/imageSize*self.realImgScale), int(i[n, 1]*h/imageSize*self.realImgScale)]
                    if pt[0] == lastPt[0] and pt[1] != lastPt[1] and i[n, 0] < 20 and i[n, 0] == minX:
                        pt[0] = 0
                        l[len(l)-1][0] = 0
                    if pt[0] == lastPt[0] and pt[1] != lastPt[1] and i[n, 0] > imageSize-20 and i[n, 0] == maxX:
                        pt[0] = int(w*self.realImgScale)
                        l[len(l)-1][0] = int(w*self.realImgScale)

                    l.append(pt)
                    lastPt = pt
                if len(l) < 2:
                    continue
                cnt = np.array(l)
                epsilon = 0.001 * cv2.arcLength(cnt, True)

                approx = cv2.approxPolyDP(cnt, epsilon, True)
                newPoly = []
                for pt in approx:
                    newPoly.append([int(pt[0][0]), int(pt[0][1])])


                good = True
                for ind, pol in enumerate(polysMade):
                    try:
                        iou = self.calculate_iou_poly(pol, newPoly)
                    except:
                        iou = 1
                    if iou > 0.5:
                        good=False
                        break
                if good:
                    polysMade.append(newPoly)
                else: # too much overlap. Dont use polygon
                    continue

                # classesUsed = [0,1,2, 3, 4, 6, 8]
                if cc >= 5:
                    cc += 1
                if cc >= 8:
                    cc += 1
                l = newPoly + [cc]

                if len(l) > 5:
                    res += [l]




                
                self.unsavedChanges = True

                self.dispImg()


        self.allPolys[str(self.frameNum)] = res


    def makePredExample(self):

        imageSize = 640#416
        img = cv2.resize(self.openImg, (imageSize, imageSize))
        if self.model is None:
            from ultralytics import YOLO
            print("model", self.modelName)

            self.model = YOLO(self.modelName)

        results = self.model.predict(source=img, save=False)#, conf=0.2)

        res = []

        maskImg = np.zeros((imageSize, imageSize, 3), np.uint8) 

        for result in results:

            # sloppy way to check if there are masks from model prediction
            segd = False
            try:
                x = result.masks.xy
                segd = True
            except:
                print("no mask")
                return

            polysMade = []
            # add the masks to the list
            for j, i in enumerate(result.masks.xy):
                cc= 0 
                cc = int(result.boxes.cls[j])
                if cc >= 5:
                    cc += 1
                if cc >= 8:
                    cc += 1
                
                # for n in range(i.shape[0]):


                    # sometimes the annotation doesnt reach all the way to the border
                labelColor = self.allLabelColors[cc%len(self.allLabelColors)]

                # print(i)
                pts = []
                for n in range(i.shape[0]):
                    pt = [int(i[n, 0]), int(i[n, 1])]
                    pts.append(pt)
                if len(pts) > 3:
                    cv2.fillPoly(maskImg, pts = [np.array(pts)], color=labelColor)

        img = cv2.addWeighted(maskImg, 0.8, img, 1, 0, img)
        cv2.imshow("pred", img)



    def closest_point_on_segment(self, segment, point):
        """
        Finds the closest point on a line segment to a given point.
        
        :param segment: A tuple of tuples ((x1, y1), (x2, y2)) defining the line segment.
        :param point: A tuple (x, y) defining the third point.
        :return: The closest point on the segment.
        """
        (x1, y1), (x2, y2) = segment
        (x3, y3) = point
        
        # Vector from (x1, y1) to (x2, y2)
        segment_vector = np.array([x2 - x1, y2 - y1])
        # Vector from (x1, y1) to (x3, y3)
        point_vector = np.array([x3 - x1, y3 - y1])

        # Calculate the projection of point_vector onto segment_vector
        projection = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)
        
        # Ensure projection is within the bounds of the line segment
        projection = max(0, min(1, projection))
        
        # Calculate the closest point
        closest_point = (x1 + projection * segment_vector[0], y1 + projection * segment_vector[1])
        
        return closest_point

    def distance_to_segment(self, segment, point):
        """
        Calculates the distance from a point to the closest point on a segment.
        
        :param segment: A tuple of tuples ((x1, y1), (x2, y2)) defining the line segment.
        :param point: A tuple (x, y) defining the third point.
        :return: The distance to the closest point on the segment.
        """
        closest_point = self.closest_point_on_segment(segment, point)
        return np.linalg.norm(np.array(point) - np.array(closest_point))

    def calculate_iou_poly(self, poly1_coords, poly2_coords):
        # Create Polygon objects from coordinates
        poly1 = Polygon(poly1_coords)
        poly2 = Polygon(poly2_coords)

        # Calculate intersection and union
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area

        # Calculate IOU
        iou = intersection_area / union_area

        return iou


if __name__ == "__main__":
    A = Annotator(videosPath, saveName, useBorders)
    A.runThrough()