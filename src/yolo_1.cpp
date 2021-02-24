#include "Yolo.hpp"
#include "Coords.hpp"

#define RED     
#define GREEN   
//Requirements: OpenCV and Darknet

//To compile, paste in terminal:
//g++ Yolo.cpp -o output `pkg-config --cflags --libs opencv`

string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) + ", height=(int)" +
           to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + to_string(flip_method) + " ! video/x-raw, width=(int)" + to_string(display_width) + ", height=(int)" +
           to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


//draw detecting boxes for frame function
void DrawDetectingBox(Mat &cap, const string coconame, float confidence, Rect box)
{
	int top = box.y - box.height/2;
	int left = box.x - box.width/2;
	
	rectangle(cap, Rect(left, top, box.width, box.height), Scalar(0,255,0),2);
	
	String label_text = format("%.2f", confidence);
	label_text = coconame + ":" + label_text;
	
	int baseline;
	Size label_size = getTextSize(label_text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
	top = max(top, label_size.height);
	rectangle(cap, Point(left,top - label_size.height), Point(left + label_size.width, 
	top + baseline), Scalar(0,0,0), FILLED);
	putText(cap, label_text, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5,
	Scalar(255,255,255));
}
//main function
int main(int argc, const char *argv[])
{
	//if user enter for help
	if (strcmp(argv[1], "webcam"))
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
    int flip_method = 180;

    std::string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

	char fileType[5];	//file type jpg, jpeg, etc is saved here
	string str, outputFile;	//str to save image file entered by user
				//outputFile is to save final image name ex: person_output.jpg
	VideoCapture cap(pipeline, CAP_GSTREAMER});	//for webcam use
	Mat frame;	//current frame
	TickMeter tm;
    char c;
	
	//try to open image or webcam
	try{
		// Open the webcam if fail, throw error
		if(strcmp(argv[1], "webcam") == 0)
		{
			cap.open(0);
			if(!cap.isOpened())
			{
				throw("Error!");
			}
		}
    }
	//catch errors thrown when attempting to open image or webcam
	catch(...)
	{
		cout << "Could not open image or webcam" << endl;
		return (-1);
	}
	//YOLO files required to do object detection
	string configYolo = "cfg/yolov3-tiny.cfg";	//YOLO configuration file
	string weightsYolo = "yolov3-tiny.weights";	//YOLO weights
	string coconamesYolo = "coco.names";	//YOLO classes names
	
	//YOLO class objects
	vector<yoloClass> objects;
	
	//connect to YOLO network
	YoloNetwork yolo = YoloNetwork(configYolo, weightsYolo, coconamesYolo);
	
	//Create a window with the name "Person Detection"
	String windowName = "Person Detection";
	namedWindow(windowName, WINDOW_NORMAL);
	
    while(1)
    {
        //Start the time
        tm.start();
        //read current frame of webcam
                
        //if reach end of frame
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
            //cout << objects[i].classID << ": " << objects[i].confidence << endl;
            if(!objects[i].classID.compare("person"))
            {
                DrawDetectingBox(frame, objects[i].classID, 
                objects[i].confidence, objects[i].boundBox);
            }
        }
        //Stop the time
        tm.stop();
        
        //place average fps top left corner of image
        string label = format("%.2f fps", tm.getFPS());
        putText(frame, label, Point(0, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        //show window of the frame
        imshow(windowName, frame);
        c = (char)waitKey(1);
        if(c == 27)  // exit using 'esc'
        {
            break;
        }
    }
	
	return 0;
}
