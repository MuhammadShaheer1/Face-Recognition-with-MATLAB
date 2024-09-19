%It will return a vector of image sets found through a recursive search
%starting from database folder
faceDatabase = imageSet('database', 'recursive');

%[training, test] = partition(faceDatabase, [0.8 0.2]);\
%Setting the database to training dataset
training = faceDatabase;
%person = 2;
%[hogFeature, visualization] = extractHOGFeatures(read(training(person),1));
% figure;
% subplot(2,1,1);
% imshow(read(training(person),1));
% title('Input Face');
% 
% subplot(2,1,2);
% plot(visualization);
%title('HOG Feature');
%Creating training features
trainingFeatures = zeros(training(1).Count + training(2).Count, 46656);
featureCount = 1;

for i = 1 : size(training,2)
    for j = 1 : training(i).Count
        %It returns an object containing information about SURF features
        %detected in input image. It implements SURF algorithm to detect
        %features
        points = detectSURFFeatures(read(training(i),j));
        %This function returns features which encode local shape
        %information from regions within an image. This information can be
        %used for classification, detection and tracking.
        trainingFeatures(featureCount, :) = extractHOGFeatures(read(training(i),j));
        %It will return the folder name in which the image is stored
        trainingLabel{featureCount} = training(i).Description;
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end
%It returns a trained ECOC model using the predictors X and the class
%labels Y
faceClassifier = fitcecoc(trainingFeatures, trainingLabel);