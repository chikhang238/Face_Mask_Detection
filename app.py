import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import time
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#from detect_mask_video import detect_and_predict_mask
import detect_mask_image
import argparse
from imutils.video import VideoStream


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		print(preds)
		print("")

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def mask_image():
    global RGB_img
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector.model")

    # load the input image from disk and grab the image spatial
    # dimensions
    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_image()

@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)

@st.cache(allow_output_mutation=True)
def get_cap_vid(location):
    print("Loading in function", str(location))
    video_stream = cv2.VideoCapture(str(location))

    # Check if camera opened successfully
    if (video_stream.isOpened() == False):
        print("Error opening video  file")
    return video_stream


def mask_detection():
    st.title("Face mask detection")
    activities = ["Image", "Webcam", "Video"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    choice = st.sidebar.selectbox("Mask Detection on?", activities)

    if choice == 'Image':
        st.subheader("Detection on image")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            im = our_image.save('./images/out.jpg')
            saved_image = st.image(image_file, caption='image uploaded successfully', use_column_width=True)
            if st.button('Process'):
                st.image(RGB_img, use_column_width=True)

    if choice == 'Webcam':
        st.subheader("Detection on webcam")
        # vs = VideoStream(src=0).start()
        # time.sleep(2.0)
        
        cap = get_cap()

        frameST = st.empty()
        param=st.sidebar.slider('choose your value')

        ap = argparse.ArgumentParser()
        ap.add_argument("-f", "--face", type=str,
            default="face_detector",
            help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str,
            default="mask_detector.model",
            help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())

        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([args["face"],
            "res10_300x300_ssd_iter_140000.caffemodel"])
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        maskNet = load_model(args["model"])

        while True:
            ret, frame = cap.read()
            #print(frame.shape)
            # Stop the program if reached end of video
            
            print("Done processing !!!")
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.waitKey(1)
            # Release device
            #  cap.release()
            #  break

            frameST.image(frame, channels="BGR")

    if choice == 'Video':
        st.subheader("Detection on Video")
        st.title("Play Uploaded File")

        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
        temporary_location = False

        if uploaded_file is not None:
            g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
            temporary_location = "testout_simple.mp4"

            with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file

            # close file
            out.close()
        # vs = VideoStream(src=0).start()
        # time.sleep(2.0)
        scaling_factorx = 0.7
        scaling_factory = 0.7
        image_placeholder = st.empty()

        if temporary_location:
            while True:
                cap = get_cap_vid(temporary_location)

                frameST = st.empty()
                param=st.sidebar.slider('choose your value')

                ap = argparse.ArgumentParser()
                ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="path to face detector model directory")
                ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to trained face mask detector model")
                ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
                args = vars(ap.parse_args())

                # load our serialized face detector model from disk
                print("[INFO] loading face detector model...")
                prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
                weightsPath = os.path.sep.join([args["face"],
                    "res10_300x300_ssd_iter_140000.caffemodel"])
                faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

                # load the face mask detector model from disk
                print("[INFO] loading face mask detector model...")
                maskNet = load_model(args["model"])

                while True:
                    ret, frame = cap.read()
                    #print(frame.shape)
                    # Stop the program if reached end of video
                    frame = cv2.resize(frame, None, fx=scaling_factorx, fy=scaling_factory, interpolation=cv2.INTER_AREA)
                    print("Done processing !!!")
                    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                    # loop over the detected face locations and their corresponding
                    # locations
                    for (box, pred) in zip(locs, preds):
                        # unpack the bounding box and predictions
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred

                        # determine the class label and color we'll use to draw
                        # the bounding box and text
                        label = "Mask" if mask > withoutMask else "No Mask"
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                        # include the probability in the label
                        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                        # display the label and bounding box rectangle on the output
                        # frame
                        cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.waitKey(1)
                    # Release device
                    #  cap.release()
                    #  break

                    frameST.image(frame, channels="BGR")


mask_detection()
