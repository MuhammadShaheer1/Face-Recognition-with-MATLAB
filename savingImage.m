%%%%%%%%%%%%%%%%%%%%%%
%This function will save the image captured from video player in our
%database
%INPUT ARGUMENTS:
%bboxPolygon: This vector is tracking the face on videoPlayer
%videoFrameGray: A single snapshot of video from videoplayer
%OUTPUT ARGUMENTS:
%getimage: Image of the face saved in the database
%%%%%%%%%%%%%%%%%%%%%%
function getimage = savingImage(bboxPolygon, videoFrameGray)
   %Detecting the positions of the four corners of our face
   position1 = min(bboxPolygon(2), bboxPolygon(4));
   position2 = max(bboxPolygon(6), bboxPolygon(8));
   position3 = min(bboxPolygon(1), bboxPolygon(7));
   position4 = max(bboxPolygon(3), bboxPolygon(5));
   %It will turn off the warning messages
   warning('off')
   %Getting the image from the snapshot taken from videoPlayer
   getimage = videoFrameGray(position1:position2, position3:position4, :);
   imshow(getimage);
   %Resizing the image to a standard size to make processing easy
   getimage = imresize(getimage, [300 300]);
   %Saving the image in the database
   imwrite(getimage, 'database/Shaheer/2.jpg');
end