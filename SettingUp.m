clc
clear all
close all
%It will create a detector to detect people's faces using Viola Jones
%Algorithm
faceDetector = vision.CascadeObjectDetector();
%This object will track a set of points in a video with maximum
%bidirectional error of 2
pointTracker = vision.PointTracker('MaxBidirectionalError',2);
%Creates a webcame object
if exist('cam') == 0
    cam = webcam('Integrated Webcam');
end
%It returns a still image from webcam
videoFrame = snapshot(cam);
frameSize = size(videoFrame);
%It will open a video player and setting its left, bottom, width and height
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

%Detection and Tracking
runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop && frameCount < 100
    
    videoFrame = snapshot(cam);
    %Converting rgb image to gray
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    
    if numPts < 10
        %The step function returns a bounding box value that contains [x,
        %y, Height, Width] of the face
        bbox = faceDetector.step(videoFrameGray);
        
        if ~isempty(bbox)
            %The detectMinEigenFeatures function uses the minimum eigen
            %value algorithm and return an object which contains the
            %information about feature points detected in the image.
            %The 'ROI' tells the function about the region of interest and
            %this region of interest is given as the next argument as [x,
            %y, width, height]
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1,:));
            %Returning the (x,y) coordinates of all points detected as
            %features
            xyPoints = points.Location;
            %Returning the total number of points detected
            numPts = size(xyPoints,1);
            %The release function allows the object values to be changed
            release(pointTracker);
            %This function will set the xypoints to be tracked in the video
            %using point Tracker
            initialize(pointTracker, xyPoints, videoFrameGray);
            
            oldPoints = xyPoints;
            %It will convert the bounding box to an M by 2 matrix where
            %each row will be x and y coordinates
            bboxPoints = bbox2points(bbox(1,:));
            %Converting the M by 2 matrix to 1 by M row vector
            bboxPolygon = reshape(bboxPoints', 1, []);
            %This function will insert a polygon in the video at points
            %bboxPolygon and the line width will be 3
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            %This function will insert + signs in the video of white color
            %at xyPoints
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end
        
    else
        %Tracking Mode
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
        
        numPts = size(visiblePoints, 1);
        
        if numPts >= 10
            %This function will estimate geometric transformation between
            %old points and new points with a transform of similarity. The
            %maxDistance of 4 means that the two points can differ at most
            %by 3
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            %It applies the forward transformation of xform to bboxPoints
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
            
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
            
        end
        
    end
    
    step(videoPlayer, videoFrame);
    %It will return true if the video player is on otherwise false
    runLoop = isOpen(videoPlayer);
    
end

clear cam
release(videoPlayer);
release(pointTracker);
release(faceDetector);
%Calling the function savingImage
savingImage(bboxPolygon, videoFrameGray);
            
        