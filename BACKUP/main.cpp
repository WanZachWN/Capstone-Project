#include <iostream>
#include <fstream>
#include "Yolo.hpp"

//opencv libraries
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

/********************************************************************************************************************
* Global Variables
********************************************************************************************************************/
const float SOCIAL_DISTANCE = 6;	//Constant Distance in feet (6ft)

/********************************************************************************************************************
*
********************************************************************************************************************/
string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) + ", height=(int)" +
           to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + to_string(flip_method) + " ! video/x-raw, width=(int)" + to_string(display_width) + ", height=(int)" +
           to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}



/********************************************************************************************************************
*
********************************************************************************************************************/
//draw detecting boxes for frame function
void DrawDetectingBox(Mat &cap, int x, int y)
{	
	circle( cap, Point( x, y ), 3, Scalar( 0, 255, 0), 2, 8 );
}

/********************************************************************************************************************
*
********************************************************************************************************************/
//main function
int main(int argc, const char *argv[])
{
	//if user enter for help
	if (argc != 2 || strcmp(argv[1], "webcam"))
	{
		cout << "how to use the program: " << endl <<
			"For webcam: ./yolo webcam" << endl;
		return 0;
	}
	
	int capture_width = 1280;
    	int capture_height = 720;
    	int display_width = 1280;
    	int display_height = 720;
    	int framerate = 60;
    	int flip_method = 0;

    	std::string pipeline = gstreamer_pipeline(capture_width,
		capture_height,
		display_width,
		display_height,
		framerate,
		flip_method);
    	std::cout << "Using pipeline: \n\t" << pipeline << "\n";

	char fileType[5];	//file type jpg, jpeg, etc is saved here
	VideoCapture cap(pipeline, CAP_GSTREAMER);	//for webcam use
	Mat frame;	//current frame
	TickMeter tm;
	char c;
	
	//try to open image or webcam
	try{
		// Open the webcam if fail, throw error
		if(strcmp(argv[1], "webcam") == 0)
		{
			//cap.open(0);
			if(!cap.isOpened())
			{
				throw("Error!");
			}
		}
    	}
	//catch errors thrown when attempting to open image or webcam
	catch(...)
	{
		cout << "Could not open webcam" << endl;
		return (-1);
	}
	//YOLO files required to do object detection
	string configYolo = "yolov3-tiny.cfg";	//YOLO configuration file
	string weightsYolo = "yolov3-tiny.weights";	//YOLO weights
	string coconamesYolo = "coco.names";	//YOLO classes names
	
	//YOLO class objects
	vector<yoloClass> objects;
	
	//connect to YOLO network
	YoloNetwork yolo = YoloNetwork(configYolo, weightsYolo, coconamesYolo);
	
	//Create a window with the name "Person Detection"
	String windowName = "Person Detection";
	namedWindow(windowName, WINDOW_NORMAL);
	float fps;
   	while(1)
    	{
		//Start the time
		tm.start();

		//read current frame of webcam
		if(!cap.read(frame))
		{
		    cout << "end of video" << endl;
		    break;
		}
		//current frame being processed
		yolo.CurrentFrame(frame);
		
		//get objects in the frame
		objects = yolo.outputObject();
		
		//print detected frame and draw predicting boxes around the
		//person on each frame
		for(int i = 0; i < objects.size(); i++)
		{
		    DrawDetectingBox(frame, objects[i].x, objects[i].y);
		}
		//Stop the time
		tm.stop();
		
		//place average fps top left corner of image
		string label = format("%.2f fps", tm.getFPS());
		putText(frame, label, Point(0, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		/*fps = tm.getFPS();
		if(fps != tm.getFPS())
		{
			cout << fps <<endl;
		}*/
		//show window of the frame
		imshow(windowName, frame);
		c = (char)waitKey(1);
		if(c == 27)  // exit using 'esc'
		{
		    break;
		}
		objects.clear();
    	}
	cap.release();
	destroyAllWindows();
	return 0;
}
