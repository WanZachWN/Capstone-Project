#include "Yolo.hpp"

//Requirements: OpenCV and Darknet

//To compile, paste in terminal:
//g++ Yolo.cpp -o output `pkg-config --cflags --libs opencv`

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
	if (!strcmp(argv[1], "help"))
	{
		cout << "how to use the program: " << endl <<
			"For images: ./yolo image.jpg" << endl <<
			"For webcam: ./yolo webcam" << endl;
		return 0;
	}
	
	char fileType[5];	//file type jpg, jpeg, etc is saved here
	string str, outputFile;	//str to save image file entered by user
				//outputFile is to save final image name ex: person_output.jpg
	VideoCapture cap;	//for webcam use
	Mat frame;	//current frame
	bool image = false;
	TickMeter tm;
	
	//try to open image or webcam
	try{
		// Open the webcam if fail, throw error
		if(strcmp(argv[1], "webcam") == 0)
		{
			cap.open(0);
			if(!cap.isOpened())
			{
				throw("error");
			}
		}
		//if user enters two arguments
		else if(argc == 2)
		{
			//try to read image
			frame = imread(argv[1]);
			//if image is empty or unable to read image, throw error
			if (frame.empty())
			{
				throw("error");
			}
			//set image flag to true if successful
			image = true;
			/*
			   Copy image file name to str and find the "." to get file type.
			   Copy file type of original image after position "."
			   Replace at "." location with _output to save the detected image
			   as a new file.
			   Append original filetype to the new name string.
			   Example:
			   	user entered: ./yolo person.jpg
			   	str = person.jpg
			   	find_dot at position 6
			   	outputfile-> str.replace = person_output
			   	outputfile.append = person_output.jpg
			*/
			str = argv[1];
			int find_dot = str.find(".");
			str.copy(fileType, 5, find_dot);
			outputFile = str.replace(find_dot, 7, "_output");
			outputFile.append(fileType);
		}
	}
	//catch errors thrown when attempting to open image or webcam
	catch(...)
	{
		cout << "Could not open image or webcam" << endl;
		return 0;
	}
	//YOLO files required to do object detection
	string configYolo = "yolov3.cfg";	//YOLO configuration file
	string weightsYolo = "yolov3.weights";	//YOLO weights
	string coconamesYolo = "coco.names";	//YOLO classes names
	
	//YOLO class objects
	vector<yoloClass> objects;
	
	//connect to YOLO network
	YoloNetwork yolo = YoloNetwork(configYolo, weightsYolo, coconamesYolo);
	
	//Create a window with the name "Person Detection"
	String windowName = "Person Detection";
	namedWindow(windowName, WINDOW_NORMAL);
	if(image == true)
	{
		//Start the time
		tm.start();
		//current frame being processed
		yolo.CurrentFrame(frame);
		
		//get objects in the frame
		objects = yolo.outputObject();
		
		//print detected frame and draw predicting boxes around the person on each frame
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
		waitKey(0);

		Mat detectedFrame;
        	frame.convertTo(detectedFrame, CV_8U);
		imwrite(outputFile, detectedFrame);		
	}
	else
	{
		while(1)
		{
			//Start the time
			tm.start();
			//read current frame of webcam
			cap.read(frame);
			
			//if reach end of frame
			if(frame.empty())
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
			char c = (char)waitKey(1);
			if(c == 27)  // exit using 'esc'
			{
				break;
			}
		}
	}
	
	return 0;
}
