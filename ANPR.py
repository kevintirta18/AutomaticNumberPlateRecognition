import cv2
import tkinter
import PIL
from PIL import Image
from PIL import ImageTk
import datetime
import numpy as np
import pytesseract
import time


class liveBox:
    def __init__(self, top):
        #Main window Tkinter UI
        self.top = top
        self.top.title = ("Camera") # Window Title
        self.top.geometry("600x600")# Window Size

        # open the camera
        # Loop to find camera index to open
        index = -2
        while (True):
            self.camera = cv2.VideoCapture(index)
            ret, frame = self.camera.read()
            if ret == False :
                index+=1
            else:
                break

        # Configure Tkinter Frame to place widgets
        self.box = tkinter.Frame(top, width=500, height=500)
        self.box.place(x=10, y=10)
        # Configure label to place video image
        self.panel = tkinter.Label(self.box)
        self.panel.place(x=10, y=10)

        # Button for capturing frame
        btnCapture = tkinter.Button(self.top, text="Capture", command=self.captureImage)
        btnCapture.place(x=280, y=530)

        # Screen Update
        self.delay = 15
        self.update()# Call update Function to show Video
        self.top.mainloop()

    def update(self):
        # Get image from camera
        ret, frame = self.camera.read()
        frame = cv2.resize(frame, (500, 500))# resize the image from camera
        # Put Frame in Panel
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) #convert from BGR to RGB for tkinter image
        # Convert image from opencv to image object in tkinter
        image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=image)
        # put image in panel
        self.panel.imgtk = imgtk
        self.panel.configure(image=imgtk)
        self.box.after(self.delay, self.update)# update the box every delay to continuously show the frame

    def captureImage(self):
        # Capture and save the frame
        timeStamp = datetime.datetime.now()# get the current time and date
        filename = "{}.jpg".format(timeStamp.strftime("%Y-%m-%d_%H-%M-%S"))# use current time and date as file name
        return_value, image = self.camera.read()# get image from camera
        #Process the image using imgProcess function
        imgPrc = self.imgProcess(image)
        #Get the character from the image using imgtoStr function
        possiblePlateNumber = self.imgtoStr(imgPrc)
        #put the character in the image and save the file
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(possiblePlateNumber) == 0:
            possiblePlateNumber = "Not Detected"
        cv2.putText(image, possiblePlateNumber, (200, 200), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(filename , image)
        # return image

    def imgProcess(self,image):
        #Process the image
        self.image = image
        imgGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) #Convert to Grayscale
        #Create array of one with the size of imgGray
        structuringElement = np.ones_like(imgGray)
        lenRow = len(structuringElement)# get the total Row
        lenColumn = len(structuringElement[0])# get the total Column

        #Create Masking for to remove unnecessary component
        j = 0
        for i in range(int(lenRow / 3)):# 1/3 of the image height is removed
            for j in range(lenColumn):
                structuringElement[i][j] = 0
        for i in range(int(lenRow / 3 ), lenRow):
            for j in range(lenColumn):
                if (j < (lenColumn / 5)) or (j > (lenColumn / 5 * 4)):# image to be shown only from 1/5 to 4/5 of width
                    try:
                        structuringElement[i][j] = 0
                    except:
                        break
        imgGray = cv2.bitwise_and(imgGray, imgGray, mask=structuringElement)

        # Uncomment to show imgGray after Masking
        # cv2.imshow('struct', imgGray)

        # Morphological transformation
        # First Opening to remove noise
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 15))
        imgProcessed = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, structuringElement)
        imgProcessed = cv2.subtract(imgGray, imgProcessed)

        # Image Conditioning using bilateral filtered to remove noise and maintain edges
        imgBlurred = cv2.bilateralFilter(imgProcessed, 11, 17, 17)

        # Image contouring using Canny Edge
        edges = cv2.Canny(imgBlurred, 100, 200)
        contour, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Get the height and shape of the image
        height, width = edges.shape

        j = 0

        # CHAR CANDIDATE LOCALIZATION
        # Array to save coordinates of possilbe char contour
        possibleCharLoc = np.zeros([20, 2], dtype=(np.uint16, 2))
        # array for masking to show only the contour with rectangle
        boundingRectangles = np.zeros([height, width], dtype=np.uint8)
        # Selection of possible Contour for a char
        for i in range(len(contour)):
            # Save the coordinates, ratio, and area of the contour
            x, y, w, h = cv2.boundingRect(contour[i])
            ratio = h / w
            CONTOUR_AREA = w * h
            # Condtion to select possible contour
            if (CONTOUR_AREA > 200) & (CONTOUR_AREA < 5000) & (ratio > 1.3):
                # Save the possible contour coordinates of TopLeft Corner and BottomRight Corner in array
                possibleCharLoc[j][0] = (x, y)
                possibleCharLoc[j][1] = (x + w, y + h)
                cv2.rectangle(boundingRectangles, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)
                j = j + 1

        # Uncomment to show possible char that is detected
        # test = np.zeros_like(imgGray)
        # test = cv2.bitwise_and(imgGray, imgGray, mask=boundingRectangles)
        # cv2.imshow('test', test)

        # All possible contour number is saved
        numCharCandidates = j

        # Polling region of interest
        # Divide image to 5 regions based on height
        # Find if topLeft corner of charLoc is inside the region
        # +1 for each topLeft corner
        regionY = np.array([0, 0, 0, 0, 0], dtype=np.uint8)
        for i in range(j):
            if (possibleCharLoc[i][0][1] < (height / 5)):
                regionY[0] = regionY[0] + 1
            elif (possibleCharLoc[i][0][1] < (2 * height / 5)):
                regionY[1] = regionY[1] + 1
            elif (possibleCharLoc[i][0][1] < (3 * height / 5)):
                regionY[2] = regionY[2] + 1
            elif (possibleCharLoc[i][0][1] < (4 * height / 5)):
                regionY[3] = regionY[3] + 1
            else:
                regionY[4] = regionY[4] + 1

        # Divide image to  regions based on width
        # Find if topLeft corner of charLoc is inside the region
        # +1 for each topLeft corner

        # Find ROI with most topLeft corner of possible char
        maxRegionY = np.where(regionY == np.amax(regionY))

        if (maxRegionY[0] == 0):
            base = 1
        elif (maxRegionY[0] == 1):
            base = 2
        elif (maxRegionY[0] == 2):
            base = 3
        elif (maxRegionY[0] == 3):
            base = 4
        elif (maxRegionY[0] == 4):
            base = 5

        # Find extreme topLeft and BottomRight corner of ROI
        extremeTopLeft = np.zeros([2], dtype=np.uint16)
        extremeTopLeft[0] = 50000
        extremeTopLeft[1] = 50000
        extremeBottomRight = np.zeros([2], dtype=np.uint16)

        for i in range(numCharCandidates):
            if (possibleCharLoc[i][0][1] < (base / 5 * height)) & (
                    possibleCharLoc[i][0][1] > ((base - 1) / 5 * height)):
                if (extremeTopLeft[0] > possibleCharLoc[i][0][0]):
                    extremeTopLeft[0] = possibleCharLoc[i][0][0]
                    if (extremeTopLeft[1] > possibleCharLoc[i][0][1]):
                        extremeTopLeft[1] = possibleCharLoc[i][0][1]
                if (extremeBottomRight[0] < possibleCharLoc[i][1][0]):
                    extremeBottomRight[0] = possibleCharLoc[i][1][0]
                    if (extremeBottomRight[1] < possibleCharLoc[i][1][1]):
                        extremeBottomRight[1] = possibleCharLoc[i][1][1]

        # Make bounding rectangle based on the extreme coordinates for masking
        plateBounds = np.zeros_like(imgGray)
        cv2.rectangle(plateBounds, (extremeTopLeft[0], extremeTopLeft[1]),
                      (extremeBottomRight[0], extremeBottomRight[1]), (255, 255, 255), thickness=cv2.FILLED)

        # Uncomment to show the plate
        # cv2.imshow('Plate', plateBounds)

        # Masking plate candidates from original image with plateBounds to show only the plate number
        plate = np.zeros_like(imgGray)
        plate = cv2.bitwise_and(imgGray, imgGray, mask=plateBounds)
        plate = cv2.bitwise_not(plate)
        # Process the plate number again
        plate = cv2.bilateralFilter(plate, 11, 17, 17)
        plate = cv2.adaptiveThreshold(plate, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 13)

        # Uncomment to show the image
        # cv2.imshow("result", plate)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # return value is image of plate
        return plate


    def imgtoStr(self,plate):
        self.plate = plate
        # OCR Each Candidates
        # String Selection
        buffer = ""
        possiblePlateNumber = pytesseract.image_to_string(self.plate)  # save the string from image
        possiblePlateNumber = possiblePlateNumber.replace('\n', '')  # remove new line that is detected

        # Loop to select the possible char for plate number ( Capital English Alphabet and 0-9 Number )
        for q in range(len(possiblePlateNumber)):
            value = ord(possiblePlateNumber[q])

            if ((value >= 48) and ((value <= 57) or (value >= 65)) and (value <= 90)):
                buffer = buffer + possiblePlateNumber[q]  # buffer diisi char yang sesuai dengan plat nomor

        # rewrite variable possilbePlateNumber
        possiblePlateNumber = buffer
        return possiblePlateNumber
        # buffer = ""
        # print(possiblePlateNumber)


# Call the liveBox class
Test1 = liveBox(tkinter.Tk())
