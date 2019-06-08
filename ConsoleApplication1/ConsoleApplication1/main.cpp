/*
* Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
* Released to public domain under terms of the BSD Simplified license.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*   * Neither the name of the organization nor the names of its contributors
*     may be used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*   See <http://www.opensource.org/licenses/bsd-license>
*/
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#define SAMPLE_NUMBER 4
#define FACE_NUMBER_MAX 200
#define FACE_NUMBER_TEST 5
#define RATIO 0.7
#define FILE_ROUTE "csv.ext"
using namespace cv;
using namespace std;
string User_Name[FACE_NUMBER_MAX];
int User_Num;
ofstream User_File;
ofstream Name_File;
void Get_Names();
void Get_Photos();
int face_recognition();
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') 
{
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) 
	{
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if(!path.empty() && !classlabel.empty())
		{
			images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}
int main()
{
	char Mode_Char;
	cout<<"Enter 'P' into photo input or 'R' into face recognition\r\n";
	cin>>Mode_Char;
	if(Mode_Char=='P')
	{
		Get_Names();
		Get_Photos();
		cout<<"Now enter face recognition mode, enter ESC to exit program\r\n";
		face_recognition();
		return 0;
	}
	if(Mode_Char=='R')
	{
		int i=0;
		fstream file("name.ext");
		vector<string>  words; //创建一个vector<string>对象
		string      line; //保存读入的每一行
		while(!file.eof())
		{
			getline(file,line);
			line=line.substr(0,line.length()-1);
			User_Name[i++]=line;
		}
		file.close();
		cout<<"Now enter face recognition mode, enter ESC to exit program\r\n";
		face_recognition();
		return 0;
	}

}

void Get_Names()
{
	cout<<"Please enter the number of people, not more than 200\r\n";
	cin>>User_Num;
	for(int i=0;i<User_Num;i++)
	{
		cout<<"Please enter the name of User "+to_string(i+1)+"\r\n";
		cin>>User_Name[i];
	}
	User_File.open(FILE_ROUTE);
	Name_File.open("name.ext");
	for(int i=0;i<User_Num;i++)
	{
		for(int j=0;j<10;j++)
		{
			User_File<<"photos\\"+User_Name[i]+to_string(j)+".jpg;"+to_string(i)+"\r\n";
		}
	}
	for(int i=0;i<User_Num;i++)
	{
		Name_File<<User_Name[i]+"\r\n";
	}
	User_File.close();
	Name_File.close();
}

void Get_Photos()
{
	cout<<"Now enter the photo input mode and click Enter to save the photos. Each person needs to save 10 photos.\r\n";
	int Photo_Num=0;
	string fn_haar = "haarcascade_frontalface_alt2.xml";
	CascadeClassifier haar_cascade;
	VideoCapture cap(0);
	vector< Rect_<int> > faces;
	Rect face;
	Mat frame,frame_gray,frame_init;
	fstream file(FILE_ROUTE);
	vector<string>  words; //创建一个vector<string>对象
	string      line; //保存读入的每一行
	haar_cascade.load(fn_haar);
	if(!cap.isOpened())
	{
		cout<<"Cannot turn on the camera\r\n";
	}
	while(Photo_Num<User_Num*10)
	{
		cap >> frame_init;
		flip(frame_init,frame_init,1);
		bilateralFilter(frame_init, frame, 5, 10, 3);
		cvtColor(frame,frame_gray,CV_RGB2GRAY);
		haar_cascade.detectMultiScale(frame_gray, faces,1.2,3,CV_HAAR_DO_CANNY_PRUNING);
		faces[0].x+=faces[0].width*((1-RATIO)/2);
		faces[0].y+=faces[0].height*((1-RATIO)/2);
		faces[0].width*=RATIO;
		faces[0].height*=RATIO;
		rectangle(frame,faces[0],CV_RGB(0,255,0));
		imshow("origin",frame);
		Mat roi=frame(faces[0]);
		resize(roi,roi,Size(70,70));
		//cvtColor(roi,roi,CV_RGB2GRAY);
		imshow("face",roi);
		int key_number=waitKey(10);
		//cout<<key_number;
		if (key_number==13)
		{
			cout<<User_Name[Photo_Num/10]+" have saved "+to_string(Photo_Num%10+1)+" photos.\r\n";
			Photo_Num++;
			getline(file,line);
			line=line.substr(0,line.length()-3);
			imwrite(line,roi);
		}
	}
	destroyWindow("origin");
	destroyWindow("face");
	cap.release();
}

int face_recognition()
{
	//printf("学习中…………");
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	//if (argc != 4) {
	//    cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
	//    cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
	//    cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
	//    cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
	//    exit(1);
	//}
	//// Get the path to your CSV:
	//string fn_haar = string(argv[1]);
	//string fn_csv = string(argv[2]);
	//int deviceId = atoi(argv[3]);
	//// Get the path to your CSV:
	// please set the correct path based on your folder
	int prediction[FACE_NUMBER_MAX]; 
	double prediction_sum[FACE_NUMBER_MAX];
	double sample_decision[FACE_NUMBER_MAX][SAMPLE_NUMBER+1];
	string fn_haar = "haarcascade_frontalface_alt2.xml";
	string fn_csv = "csv.ext";
	int deviceId = 0;			// here is my webcam Id. 
	// These vectors hold the images and corresponding labels:
	vector<Mat> images;
	vector<int> labels;
	int label_out;
	double confidence[FACE_NUMBER_MAX],confidence_sum[FACE_NUMBER_MAX],confidence_filted[FACE_NUMBER_MAX];
	static int threshold=200;
	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try 
	{
		read_csv(fn_csv, images, labels);
	} catch (cv::Exception& e) 
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size AND we need to reshape incoming faces to this size:
	int im_width = images[0].cols;
	int im_height = images[0].rows;
	// Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer(0);
	model->train(images, labels);
	// That's it for learning the Face Recognition model. You now
	// need to create the classifier for the task of Face Detection.
	// We are going to use the haar cascade you have specified in the
	// command line arguments:
	//
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Get a handle to the Video device:
	VideoCapture cap(deviceId);
	// Check if we can use this device at all:
	if(!cap.isOpened())
	{
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		cout<<"Cannot turn on the camera\r\n";
	}
	// Holds the current frame from the Video device:
	Mat frame,frame_init;
	for(;;) 
	{
		static int decision_time;
		cap >> frame_init;

		flip(frame_init,frame_init,1);
		bilateralFilter(frame_init, frame, 5, 10, 3);
		// Clone the current frame:
		Mat original = frame.clone();
		// Convert the current frame to grayscale:
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces,1.2,5,CV_HAAR_DO_CANNY_PRUNING);
		// At this point you have the position of the faces in
		// faces. Now we'll get the faces, make a prediction and
		// annotate it in the video. Cool or what?
		for(int i = 0; i < faces.size(); i++) 
		{
			// Process face by face:
			Rect face_i = faces[i];
			face_i.x+=face_i.width*((1-RATIO)/2);
			face_i.y+=face_i.height*((1-RATIO)/2);
			face_i.width*=RATIO;
			face_i.height*=RATIO;
			// Crop the face from the image. So simple with OpenCV C++:
			Mat face = gray(face_i);
			// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
			// verify this, by reading through the face recognition tutorial coming with OpenCV.
			// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
			// input data really depends on the algorithm used.
			//
			// I strongly encourage you to play around with the algorithms. See which work best
			// in your scenario, LBPH should always be a contender for robust face recognition.
			//
			// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
			// face you have just found:
			Mat face_resized;
			cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			//imshow("2",face_resized);
			//cvtColor(face_resized,face_resized,CV_RGB2GRAY);
			// Now perform the prediction, see how easy that is:
			model->predict(face_resized,prediction[i],confidence[i]);
			//if(confidence[i]>threshold)
			//{
			//	prediction[i]=-1;
			//}
			// And finally write all we've found out to the original image!
			// First of all draw a green rectangle around the detected face:
			rectangle(original, face_i, CV_RGB(0, 255,0), 1);
			// Create the text we will annotate the box with:
			string box_text;
			box_text = format( "Prediction = " );
			// Get stringname
			if ( prediction[i] >= 0 && prediction[i] <=SAMPLE_NUMBER-1 )
			{
				box_text.append( User_Name[prediction[i]] );
			}
			else box_text.append( "Unknown" );
			// Calculate the position for annotated text (make sure we don't
			// put illegal values in there):
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
			//if(prediction[i]!=-1)
			//printf("匹配：%d  不确定度：%.f  阈值：%d\r\n",prediction[i],confidence[i],threshold);
		}
		imshow("face_recognizer", original);
		/*******平均数*******/
		//decision_time++;
		//for(int i=0;i<faces.size();i++)
		//{
		//	confidence_sum[i]+=confidence[i]/10.0;
		//}
		//if(decision_time>=10)
		//{
		//	decision_time=0;
		//	printf("平均数据输出：\r\n");
		//	for(int i=0;i<faces.size();i++)
		//	{
		//		printf("%.2f\r\n",confidence_sum[i]);
		//		confidence_sum[i]=0;
		//	}
		//}
		// And display it:
		/*******卡尔曼滤波***/
		//printf("滤波器输出：\r\n");
		//for(int i=0;i<faces.size();i++)
		//{
		//	confidence_filted[i]=Kalman_Making(confidence[i]);
		//	printf("%.2f\r\n",confidence_filted[i]);
		//}
		char key = (char) waitKey(20);
		// Exit this loop on escape:
		switch(key)
		{
		case 27: cap.release();return 0;
		}
		//system("cls");
		//printf("%d\r\n",key);
	}
}