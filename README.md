# annotations-indot
This program lets you rapidly annotate long videos of mowing operations.

It has the following features:

## Automatic Saving
Saves the annotations when you press the escape key, the "Exit" button, or when moving to another video. It saves the annotations as a JSON file in the same folder as the video, with the name <video_name> + <annotation_name> + ".json"
It loads the annotations with the save name

## Classifier:
Click "Classify" and it will give the following classes:
- Country/Interstate road
- Maneuver/Normal mowing/Transit/Unusual
- Obstacle type, obstacle collision, obstacle visibility
- On road/On shoulder/Off road
It will automatically load the classifier model and give predictions that can be confirmed by modifying the classes.

## Road Regression
Click on the "Regression" button within the classifying mode to label the location of the mower, shoulder, and road. This information can be used to better train the classifier.

## Fast watching
Press the "f" key to automatically play the video. The video will automatically pause when the mower lifts a wing to maneuver around an obstacle. The video can manually be paused by pressing any button.

## Manual Segmentation
Clicking on various points will create a polygon. To confirm the polygon creation, press the "return" key or the scroll wheel button. You can select the class of the annotated object by clicking on the classes in the boxes on the right side of the annotation window.

## Auto Segmentation
Press the "Auto Annot" Button to automatically annotate the image with the model

## SAM (Segment Anything Model):
in select mode, drag a rectangle with the scroll wheel down. Let go, and it will load the SAM and perform segmentation in the area

## Polygon Point Reduction
Select a polygon and click the scroll wheel to reduce the number of points in a polygon.
