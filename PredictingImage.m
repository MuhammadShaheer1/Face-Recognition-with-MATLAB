function PredictingImage(faceClassifier)

faceDetector = vision.CascadeObjectDetector();

pointTracker = vision.PointTracker('MaxBidirectionalError',2);

if exist('cam') == 0
    cam = webcam('Integrated Webcam');
end

videoFrame = snapshot(cam);
frameSize = size(videoFrame);

videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

%Detection and Tracking
runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop && frameCount < 1000
    
    videoFrame = snapshot(cam);
    
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    if numPts < 10
        
        bbox = faceDetector.step(videoFrameGray);
        
        if ~isempty(bbox)
            
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1,:));
            
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
            
            oldPoints = xyPoints;
            
            bboxPoints = bbox2points(bbox(1,:));
            
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end
        
    else
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        
        numPts = size(visiblePoints, 1);
        
        if numPts >= 10
            
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
            
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
            position1 = min(bboxPolygon(2), bboxPolygon(4));
            position2 = max(bboxPolygon(6), bboxPolygon(8));
            position3 = min(bboxPolygon(1), bboxPolygon(7));
            position4 = max(bboxPolygon(3), bboxPolygon(5));
   
            warning('off')
            %if there is any image captured
            if position2<640 && position4<640 && position1>0 && position2>0
                %Cropping face from the image and saving it in getimage
                getimage = videoFrameGray(position1:position2, position3:position4, :);
                %Resizing the image to a standard size to make processing easy
                getimage = imresize(getimage, [300 300]);
                %Extracting HOG features from the image
                queryFeatures = extractHOGFeatures(getimage);
                %This returns the predicted response values of the linear
                %regression model to the points in queryFeatures
                [personLabel, PostProbs] = predict(faceClassifier, queryFeatures);
                %Selecting the label with the maximum probability
                maxpro = max(abs(PostProbs(1)), abs(PostProbs(2)));
                %Creating a green box in which the person name will be
                %labelled
                position = [position3 position2];
                box_color = {'green'};
                string = strcat(personLabel, num2str(maxpro));
                videoFrame = insertText(videoFrame, position, string, 'FontSize', 18, 'BoxColor',...
                    box_color, 'BoxOpacity', 0.4, 'TextColor', 'white');
            end
        end
        
    end
    
    step(videoPlayer, videoFrame);
    
    runLoop = isOpen(videoPlayer);
    
end

clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);

        
end
