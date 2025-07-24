import cv2

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
    return frameOpencvDnn, faceBoxes

def Extract_Face_Features(image):
    padding=20
    # Initialize models
    faceProto = "models/opencv_face_detector.pbtxt"
    faceModel = "models/opencv_face_detector_uint8.pb"
    ageProto = "models/age_deploy.prototxt"
    ageModel = "models/age_net.caffemodel"
    genderProto = "models/gender_deploy.prototxt"
    genderModel = "models/gender_net.caffemodel"
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    
    frame = image
    
    # Detect faces
    _, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes:
        print("No face detected")
        return None
    
    
    faceBox = faceBoxes[0]
        
    # Extract face ROI
    face = frame[max(0, faceBox[1]-padding): 
                min(faceBox[3]+padding, frame.shape[0]-1), 
                max(0, faceBox[0]-padding): 
                min(faceBox[2]+padding, frame.shape[1]-1)]
        
    # Prepare blob
    try:
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    except:
        return None
    
    # Get gender features from fc7 (penultimate layer)
    genderNet.setInput(blob)
    gender_features = genderNet.forward('fc7').flatten()
    
    # Get age features from fc7 (penultimate layer)
    ageNet.setInput(blob)
    age_features = ageNet.forward('fc7').flatten()
        
    feature_vectors={
        'face_box': faceBox,
        'gender_features': gender_features,
        'age_features': age_features,
    }
    
    return feature_vectors