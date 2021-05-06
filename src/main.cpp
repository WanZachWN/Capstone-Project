/********************************************************************************************************************
* COVID-19 SDR | Written by Wan Fatimah Wan Nawawi
* University of Missouri Senior Capstone II - Group 2
* Kenyon Shutt | Juan Martinez | Gillian Schulte | Wan Fatimah Wan Nawawi
*
* Requirements to compile:
*	- OpenCV
*	- Yolov3 tiny weight file 		- yolov3-tiny.weights 
*		To get the weight file in terminal: wget https://pjreddie.com/media/files/yolov3-tiny.weights
*	- Yolov3 tiny configuration file 	- yolov3-tiny.cfg
*	- Coco class identification files 	- coco.names
*
* Note**: Although this code is written for CUDA GPU processing. It will work for CPU processing. If 
*	  OpenCV does not detect any CUDA Present, it will default to CPU processing.
*
* To compile, paste in terminal:
*	g++ main.cpp -o output `pkg-config --cflags --libs opencv`
*	or
*	type: make
*
* To run the program: ./main webcam
*
* To run i2c Camera: python3 i2c_camera.py
*
* To check i2c pins: i2cdetect -y -r 1
*
********************************************************************************************************************/

/********************************************************************************************************************
* Yolo Header File
********************************************************************************************************************/
#include "Yolo.hpp"

/********************************************************************************************************************
* GStreamer Pipeline Function
********************************************************************************************************************/
string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {

    /*
      Using GStreamer Pipeline to capture and display current frame captured by the camera
    */
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) + ", height=(int)" +
           to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + to_string(flip_method) + " ! video/x-raw, width=(int)" + to_string(display_width) + ", height=(int)" +
           to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}



/********************************************************************************************************************
* Draw Indicatior Function
********************************************************************************************************************/
void DrawDetector(Mat &cap, struct yoloClass person)
{	
	//compliance is a flag whether the person is within social distance compliance
	//If the flag is true, Indicate with a red dot
	/*if(person.noncompliance == true)
	{
		circle( cap, Point( person.x, person.y ), 3, Scalar(0, 0, 255), 2, 8 );
	}
	//If the flag is false, Indicate with a green dot
	else
	{
		circle( cap, Point( person.x, person.y ), 3, Scalar( 0, 255, 0), 2, 8 );
	}*/
	int top = person.y - person.height/2;
	int left = person.x - person.width/2;
	
	if(person.noncompliance == true)
	{
		rectangle(cap, Rect(left, top, person.width, person.height), Scalar(0,255,0),2);
	}
	else
	{
		rectangle(cap, Rect(left, top, person.width, person.height), Scalar(0,0, 255),2);
	}
}

/********************************************************************************************************************
*
********************************************************************************************************************/
//main function
int main(int argc, const char *argv[])
{
	//if user enter for help or invalid number of arguments when starting of program
	if (argc != 2 || strcmp(argv[1], "webcam"))
	{
		cout << "how to use the program: " << endl <<
			"For webcam: ./main webcam" << endl;
		return 0;
	}
	
	//Set GStreamer Pipeline values
	int capture_width = 1280;
    	int capture_height = 720;
    	int display_width = 1280;
    	int display_height = 720;
    	int framerate = 60;
    	int flip_method = 0;

	//Pass data for GStreamer Pipeline with the values set
    	std::string pipeline = gstreamer_pipeline(capture_width,
		capture_height,
		display_width,
		display_height,
		framerate,
		flip_method);
    	std::cout << "Using pipeline: \n\t" << pipeline << "\n";
    
    	//Webcam use declaration
	VideoCapture cap(pipeline, CAP_GSTREAMER);	//Capture video
	Mat frame;					//n-dimensional dense array class to store current frame
	TickMeter tm;					//Tickmeter to calculate frame per second(FPS) when processing each frame
	char c;						//Character type to check if user ends the program
	fstream ptrFile("number_people.txt");				//File pointer
	char* num_people = new char[1];
	fstream ptrFile2("compliance.txt");
	
	//try to open webcam
	try{
		// Open the webcam if fail, throw error
		if(strcmp(argv[1], "webcam") == 0)
		{
			if(!cap.isOpened())
			{
				throw(-1);
			}
		}
		if(!ptrFile.is_open())
		{
			throw("opening file: number_people.txt");
		}
		else
		{
			cout << "Opening file" << endl;
		}
		ptrFile.close();
    	}
	//catch errors thrown when attempting to open webcam
	catch(const char* error)
	{
		cout << "Error " << error << endl;
		return (-2);
	}
	catch(...)
	{
		cout << "Could not open webcam" << endl;
		return (-1);
	}
	
	//YOLO files required to do object detection
	//string configYolo = "yolov3.cfg";	//YOLO configuration file
	//string weightsYolo = "yolov3.weights";	//YOLO weights
	string configYolo = "yolov3-tiny.cfg";	//YOLO configuration file
	string weightsYolo = "yolov3-tiny.weights";	//YOLO weights
	string coconamesYolo = "coco.names";	//YOLO classes names
	int compliance = 0;
	
	//YOLO class objects
	vector<yoloClass> objects;
	
	//connect to YOLO network by passing file names
	YoloNetwork yolo = YoloNetwork(configYolo, weightsYolo, coconamesYolo);
	
	//Create a window with the name "Person Detection"
	String windowName = "Person Detection";
	namedWindow(windowName, WINDOW_NORMAL);
	
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
		objects = yolo.getOutputObject();
		
		/*
		  print detected frame and draw predicting dots around the
		  person on each frame
		*/
		for(int i = 0; i < objects.size(); i++)
		{
		    DrawDetector(frame, objects[i]);
		}
		
		//Stop the time
		tm.stop();
		
		//place average fps top left corner of window frame 
		string label = format("%.2f fps", tm.getFPS());
		putText(frame, label, Point(0, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		
		//show window of the frame
		imshow(windowName, frame);
		
		
		ptrFile.open("number_people.txt", ifstream::out);
		ptrFile2.open("compliance.txt", ifstream::out);
		//send number of people
		//cout << objects.size() << endl; 
		//num_people = objects.size();
		snprintf(num_people, sizeof(num_people), "%zu", objects.size());
		//cout << num_people << endl;
		//ptrFile.write(num_people, sizeof(num_people));
		compliance = yolo.GetCompliance();
		ptrFile2 << compliance;
		ptrFile2.close();
		/*if(compliance == false)
		{
			num_people = num_people + 64;
		}*/
		ptrFile << num_people;
		ptrFile.close();
		
		//exit using 'esc'
		c = (char)waitKey(1);
		if(c == 27)  
		{
		    	//ptrFile.open("number_people.txt", ifstream::out);
			//send number of people
			//ptrFile.write(num_people, sizeof(num_people));
			//ptrFile << 0;
			//ptrFile.close();
			break;
		}
		
		//clear vector after each frame as it doesn't continue expanding
		objects.clear();
    	}
    	//Release capture and destroy all windows created
	cap.release();
	destroyAllWindows();
	
	 
	return 0;
}
