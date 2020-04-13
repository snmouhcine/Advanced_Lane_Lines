## Writeup 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"



### Camera Calibration



Every camera comes with a little bit of distortion in the images it puts out. Since each camera's distortion can be different, a calibration must be done in order to correct the image and make it appear as in the real-world, undistorted. Luckily, OpenCV provides an easy way to do so using chessboard images taken by the camera. I pulled in some of Udacity's provided chessboard images from their camera to begin with.

Next, based on the number of inner corners on the chessboard, I prepare the "object points", which are the (x, y, z) coordinates of where the chessboard corners are in the real world. This particular instance will have me assume z=0, as the chessboard is fixed on a flat (x, y) plane so that each calibration image I use will have the same object points. Next, I am going to grab my "image points", which are where the chessboard corners are in the image. I can do this with cv2.findChessboardCorners(), a useful function for finding just where those inner corners lie in the image. Note that I have appended the object points and image points to lists to use in the next function.

Now that I have my object points and image points, I can calibrate my camera and return the distortion coefficients. Using at least 10 images, cv2.calibrateCamera() can calculate the camera matrix and distortion coefficients (along with a few other pieces of information). I then use cv2.undistort() to see an undistorted chessboard.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Now that we can undistort the images from the camera, I'll apply this to a road image, it is not quite as obvious of a change as the chessboard is, but the same undistortion is being done to it.

Next up, I looked at various color and gradient thresholds to choose how I want to perform my thresholding and generate a binary image. 

The code for this step is in the pipeline() function in my code.


### Magnitude threshold using combined Sobelx and Sobely

This uses the square root of the combined squares of Sobelx and Sobely, which check for horizontal and vertical gradients (shifts in color, or in the case of our images, lighter vs. darker gray after conversion), respectively.

See the results on Notebook.

I did not use this one because it does not do a great job at detecting the left yellow line, especially over the lighter portion of the road.

#### Sobelx threshold

I already explained this one a bit above, and it is in the final product, so I'll just show the resulting image. I used a threshold of between 10 and 100 here (from between 0-255 in a 256 color space).

This one detects the yellow well on the lighter portion of the image, and white is also clear

### RGB color thresolds
I next checked the different RGB color thresholds. The end result only uses the R treshold, but the below code snippet can get you any of them. Note that you must set cmap='gray'.

R = image[:,:,0]
G = image[:,:,1]
B = image[:,:,2]


The R color channel definitely appears to see the lines the best, so I'll use this.

### HLS color thresholds
The last thresholds I checked were in the HLS color space. My final product only uses S, but here's how to pull out all the HLS channels.

hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]



The S color channel looks the best here, so I'll continue on with that.

### Limiting the thresholds
I also did some limiting of how much of the color space of each threshold I wanted to try to narrow it down to just the lane lines. I have shown what I found to be the optimal thresholds for the binary images in the R & S spaces below. Note that I used 200-255 for the R threshold, and 150-255 for the S threshold in these images.


I came up with a final pipeline where I actually used a combination of Sobelx, S and R. see the notebook, if any two of the three are activated, then I want the binary image to be activated. If only one is activated then it does not get activated. Note that due to this approach, I expanded the thresholds on S from the above images to 125-255, as it improved the final returned binary image by a little bit (R and Sobelx stayed the same).

### Perspective transformation

Next up is perspective transformation. This will make the image look like a bird's eye view of the road. In fact, that's exactly how I named the function that does this in notebook. After undistorting the image, I define source ("src") points of where I want the image to be transformed out from - these are essentially the bottom points of the left and right lane lines (based on when the car was traveling on a straight road), and the top of the lines, slightly down from the horizon to account for the blurriness that begins to appear further out in the image. From there, I also chose destination ("dst") points which are where I want the source points to end up in the transformed image. The code containing these points and a chart is shown below:

src = np.float32([[690,450],[1110,img_size[1]],[175,img_size[1]],[595,450]])

offset = 300 # offset for dst points
dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
    [offset, img_size[1]],[offset, 0]])

This resulted in the following source and destination points:

Source	Destination
690, 450	980, 0
1110, 720	980, 720
175, 720	300, 720
595, 450	300, 0


To verify that these points worked, I drew the lines (using cv2.line() with the source points on the original image and the destination points onto the transformed image) back onto the images to check that they were mostly parallel. Note that this was done on one of the provided images of a straight road, as opposed to some of the curves I have been using as example images so far.

## Finding and Fitting the Lines


At this point, I have some nice, perspective transformed, binary images. The next step was to plot a histogram based on where the binary activations occur across the x-axis, as the high points in a histogram are the most likely locations of the lane lines.

This histogram follows pretty close to what I expected from the last binary warped image above.

Now that we have a decent histogram, we can search based off the midpoint of the histogram for two different peak areas. Once the function 'first_lines()' has its peaks, it will use sliding windows (the size of which can be changed within the function) to determine where the line most likely goes from the bottom to the top of the image.

Note that the final version of my code does not stop to spit out an image anymore like the below one. At the end of the included code for first_lines(), adding back the below code would spit out the image of the sliding windows.


# Generate x and y values for plotting

fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(fit_leftx, fity, color='yellow')
plt.plot(fit_rightx, fity, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()


The calculated line is not perfectly parallel, but it still does a decent job.

Note that I ended up saving important information about the lines into separate classes - I do not end up using it for much in my current final version, but a previous iteration in which I took on the challenge videos (further discussed in the "Discussion" section at the end) included various checks utilizing this information. The "try" and "except" portions are based on various errors I ran into working on the challenge videos.

Now that I have found the lines the first time, the full sliding windows calculation is no longer needed. Instead, using the old lines as a basis, the program will search within a certain margin of the first lines detected (in the draw_lines() function). I also added in a counter, where if 5 frames in a row fail to detect a line, first_lines() will be run again and the sliding windows will again be used. Note that the counter gets reset if the line is detected again before reaching five.

## Radius of curvature and position of the vehicle

Two important pieces of information (which can also be used to determine the reasonableness of the returned lines) about the image are what the curvature of the road is, and where the vehicle is in the lane compared to center. If the radius of the curvature were too low, it is probably unlikely, unless it is an extreme curve. A high radius of curvature would seem odd unless it is on a straight road. Also, if the car were very far from the center, perhaps the car is calculating a line for a different lane, or off the road.

I calculated these within the draw_lines() function. For the lane curvature, I first have to convert the space from pixels to real world dimensions. Then, I calculate the polynomial fit in that space. I used the average of the two lines as my lane curvature.

For the car's position vs. center, I calculated where the lines began at the bottom of the picture (using second_ord_poly() with the image's y-dimension size plugged in). I then compared this to where the middle of the image was (assuming the car camera is in the center of the car), after converting the image's x-dimension to meters. 

## The Result

See video : Final_Result.mp4

# Discussion

## Challenges

This project took me longer likely due to my lack of overall experience in computer vision. 

The best approach I eventually found with the thresholds was to try to attack the hardest test images first. This led to me focusing on the images with changes in the color of the lane, as well as large shadows. By doing so, it actually made for a better ending pipeline. This took a significant amount of time, but I think definitely improved the end product (which only produces some slight shakiness when the road changes color from dark to light and back again in the video).

As I mentioned above, I did take a stab at the challenge videos, with a little (but unfortunately not complete) success. 

## Potential improvements

A big part of the code that is now slightly unnecessary (although included because I hope to iterate directly on it in the future, and I still believe it improves the end product) was my inclusion of classes for each line. These stemmed from my effort to remove the remaining slight shakiness of the line when the road changes color, as well as taking a crack at the challenge videos. 


Another potential improvement I thought of was to potentially use deep learning, in one of two ways. First, I could try to enhance the images by training a deep neural network using original images followed by the result being manually drawn on lane lines in more vibrant colors. The neural network would therefore be trained to spit out these improved images. This could then be fed into the above process using more restricted thresholds (as the lane lines could be put down to very specific color spaces after being manually drawn). This of course could take massive amounts of manual image processing.

The second potential option with deep learning would be to jump the process above almost entirely, and instead teach the deep neural network not to generate a new image, but to calculate out the resulting polynomial function - essentially, two regression problems to each of the two lane lines. The lines could then be drawn on using only the very end of the above process (or potentially some new process). This would probably be a bit easier, as the lines could probably be fit with software that calculates polynomials on a perspective transform image for training purposes, but the neural network could potentialy learn to skip this step and figure it out just based on the normal image.

## Conclusion

Overall, it was a very challenging project, but I definitely have a much better feel for computer vision and how to detect lane lines after spending extra time on it. I hope to eventually tackle the challenge videos as well, but for now I think I have a pretty solid pipeline!

