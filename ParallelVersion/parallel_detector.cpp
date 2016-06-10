#include "opencv2/objdetect.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <string>
#include <pthread.h>

#define	MAX_DETECTED_FACES	5

// Custom types
typedef struct
{
	cv::Rect	face;			// Current face in the coords of the full image
	cv::Rect	eyes[2];
	cv::Rect	nose;
	cv::Rect	mouth;
	int			thread_id;
	
} feature_set, *feature_set_ptr;

// Prototypes
void* find_features(void* data);
void* find_eyes(void* data);
void* find_nose(void* data);
void* find_mouth(void* data);

// Classifiers for thread use
cv::CascadeClassifier cclass_face; 
cv::CascadeClassifier cclass_eyes; 
cv::CascadeClassifier cclass_nose;
cv::CascadeClassifier cclass_mouth;

// Other global vars
size_t		max_faces;
cv::Mat		image;
cv::Mat		image_gray;

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <img_filename1> [<img_filename2>...]" 
					<< std::endl;
		std::exit(1);
	}
	else if (argc > 2)
	{
		std::cout << "Multiple image processing is unsupported at this time. "
					<< "Processing the first image only." << std::endl;
	}
	
	int ret;
	
	// Read the image file
	image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!image.data)
    {
        std::cout << "*** ERROR: failed to read image file." << std::endl;
        exit(2);
    }
	// Convert it to grayscale
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
	// And apply equalize histogram
    cv::equalizeHist(image_gray, image_gray);
	
	// Haar cascade files
	std::string haar_casc_face	= "haarcascade_frontalface_alt.xml";
	std::string haar_casc_eyes	= "haarcascade_eye_tree_eyeglasses.xml";
	std::string haar_casc_nose	= "haarcascade_mcs_nose.xml";
	std::string haar_casc_mouth	= "haarcascade_mcs_mouth.xml";
	
	// Initialize the Classifiers
    if (!cclass_face.load(haar_casc_face))
    {
    	std::cout << "*** ERROR: failed to load the face cascade" << std::endl;
    	exit(2);
    }
    if (!cclass_eyes.load(haar_casc_eyes))
    {
    	std::cout << "*** ERROR: failed to load the eyes cascade" << std::endl;
    	exit(2);
    }
    if (!cclass_nose.load(haar_casc_nose))
    {
    	std::cout << "*** ERROR: failed to load the nose cascade" << std::endl;
    	exit(2);
    }
    if (!cclass_mouth.load(haar_casc_mouth))
    {
    	std::cout << "*** ERROR: failed to load the mouth cascade" << std::endl;
    	exit(2);
    }
	
	// Data storage
	std::vector<cv::Rect>	faces;
	// Arrays may be larger than necessary, but it's easy to ignore the extras
	feature_set				face_features[MAX_DETECTED_FACES];
	pthread_t 				face_threads[MAX_DETECTED_FACES];
	
	// Find faces in photo
    cclass_face.detectMultiScale(image_gray, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    std::cout << "Detected " << faces.size() << " faces." << std::endl;
	
	// Process as many faces as were detected, up to the pre-defined max
	max_faces = (faces.size() >= MAX_DETECTED_FACES) ? MAX_DETECTED_FACES : faces.size();
	std::cout << "max_faces: " << max_faces << std::endl;
	
	// Create threads to process each face
	for (size_t i = 0; i < max_faces; i++)
	{
		// Build the thread data structure
		face_features[i].face		= faces[i];
		face_features[i].thread_id	= i;
		std::cout << "Thread " << i << " face: " << faces[i] << std::endl;
		
		// Create the face threads
		pthread_create(&face_threads[i], NULL, find_features, (void*)(&face_features[i]));
		std::cout << "Created thread " << i << std::endl;
	}
	// And wait for the threads to complete
	for (size_t j = 0; j < max_faces; j++)
	{
		pthread_join(face_threads[j], NULL);
	}
	
	// Call out the found features
	for (size_t k = 0; k < max_faces; k++)
	{
		// Faces
		cv::rectangle(image, cv::Point(faces[k].x, faces[k].y), cv::Point(faces[k].x + faces[k].width, faces[k].y + faces[k].height), cv::Scalar(255, 0, 255));

		// TODO: Eyes
		std::cout << "Found eyes: " << face_features[k].eyes[0] << ", " << face_features[k].eyes[1] << std::endl;
		cv::rectangle(image, cv::Point(face_features[k].eyes[0].x, face_features[k].eyes[0].y), cv::Point(face_features[k].eyes[0].x + face_features[k].eyes[0].width, face_features[k].eyes[0].y + face_features[k].eyes[0].height), cv::Scalar(0,255,0));
		cv::rectangle(image, cv::Point(face_features[k].eyes[1].x, face_features[k].eyes[1].y), cv::Point(face_features[k].eyes[1].x + face_features[k].eyes[1].width, face_features[k].eyes[1].y + face_features[k].eyes[1].height), cv::Scalar(0,255,0));
		
		
		// Nose
		std::cout << "Found nose: " << face_features[k].nose << std::endl;
		cv::rectangle(image, cv::Point(face_features[k].nose.x, face_features[k].nose.y), cv::Point(face_features[k].nose.x + face_features[k].nose.width, face_features[k].nose.y + face_features[k].nose.height), cv::Scalar(0,255,0));
		
		// Mouth
		std::cout << "Found mouth: " << face_features[k].mouth << std::endl;
		rectangle(image, cv::Point(face_features[k].mouth.x, face_features[k].mouth.y), cv::Point(face_features[k].mouth.x + face_features[k].mouth.width, face_features[k].mouth.y + face_features[k].mouth.height), cv::Scalar(255,255,255));
	}
	
	// Display original photo with features highlighted
    cv::imshow("Detected Face", image);
    std::cout << "Press any key to close window and exit!" << std::endl;
    cv::waitKey(0);
	
	return 0;
}

void* find_features(void* data)
{
	// Create thread to find eyes
	pthread_t eyes_thread;
	pthread_create(&eyes_thread, NULL, find_eyes, data);
	
	// Create thread to find nose
	pthread_t nose_thread;
	pthread_create(&nose_thread, NULL, find_nose, data);
	
	// Create thread to find mouth
	pthread_t mouth_thread;
	pthread_create(&mouth_thread, NULL, find_mouth, data);
	
	// Wait for all threads to complete
	pthread_join(eyes_thread, NULL);
	pthread_join(nose_thread, NULL);
	pthread_join(mouth_thread, NULL);

	return NULL;
}

void* find_eyes(void* data)
{
	feature_set_ptr features = (feature_set_ptr)data;

	// Eyes are typically found in the top horizontal half of an image
	cv::Rect eyes_roi;
	eyes_roi.x			= features->face.x;
	eyes_roi.y			= features->face.y;
	eyes_roi.width		= features->face.width;
	eyes_roi.height		= 0.5 * features->face.height;
	std::cout << "Thread " << features->thread_id << " eyes_roi: " << eyes_roi << std::endl;
	
	cv::Mat eyes_subimg	= image_gray(eyes_roi);
	
	std::vector<cv::Rect> eyes;
	cclass_eyes.detectMultiScale(eyes_subimg, eyes, 1.1, 2, 0 |cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	std::cout << "Detected " << eyes.size() << " eyes in the face for thread " << features->thread_id << std::endl;
	for (size_t i = 0; i < eyes.size(); i++)
		std::cout << "Found eye " << eyes[i] << " for thread " << features->thread_id << std::endl;
	
	if (eyes.size() > 1)
	{
		// Convert the found feature Rect back to full image coordinate space
		features->eyes[0].x			= eyes_roi.x + eyes[0].x;
		features->eyes[0].y			= eyes_roi.y + eyes[0].y;
		features->eyes[0].width		= eyes[0].width;
		features->eyes[0].height	= eyes[0].height;

		features->eyes[1].x			= eyes_roi.x + eyes[0].x;
		features->eyes[1].y			= eyes_roi.y + eyes[0].y;
		features->eyes[1].width		= eyes[0].width;
		features->eyes[1].height	= eyes[0].height;
		std::cout << "Returning found eyes: " << features->eyes[0] << ", " << features->eyes[1] << std::endl;
	}
	
	return NULL;
}

void* find_nose(void* data)
{
	feature_set_ptr features = (feature_set_ptr)data;
	
	// Noses are typically found in the middle vertical third of an image
	cv::Rect nose_roi;
	nose_roi.x			= features->face.x + (0.33 * features->face.width);
	nose_roi.y			= features->face.y;
	nose_roi.width		= 0.33 * features->face.width;
	nose_roi.height		= features->face.height;
	std::cout << "Thread " << features->thread_id << " nose_roi: " << nose_roi << std::endl;
	
	cv::Mat nose_subimg	= image_gray(nose_roi);
	
	std::vector<cv::Rect> nose;
	cclass_nose.detectMultiScale(nose_subimg, nose, 1.1, 2, 0 |cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	std::cout << "Detected " << nose.size() << " nose(s) in the face for thread " << features->thread_id << std::endl;
	for (size_t i = 0; i < nose.size(); i++)
		std::cout << "Found nose " << nose[i] << " for thread " << features->thread_id << std::endl;
	
	if (nose.size() > 0)
	{
		// Convert the found feature Rect back to full image coordinate space
		features->nose.x		= nose_roi.x + nose[0].x;
		features->nose.y		= nose_roi.y + nose[0].y;
		features->nose.width	= nose[0].width;
		features->nose.height	= nose[0].height;
		std::cout << "Returning found nose: " << features->nose << std::endl;
	}
	
	return NULL;
}

void* find_mouth(void* data)
{
	feature_set_ptr features = (feature_set_ptr)data;
	
	// Mouths are typically found in the bottom horizontal half of an image
	cv::Rect mouth_roi;
	mouth_roi.x				= features->face.x;
	mouth_roi.y				= features->face.y + 0.5 * features->face.height;
	mouth_roi.width			= features->face.width;
	mouth_roi.height		= 0.5 * features->face.height;
	std::cout << "Thread " << features->thread_id << " mouth_roi: " << mouth_roi << std::endl;
	
	cv::Mat mouth_subimg	= image_gray(mouth_roi);
	
	std::vector<cv::Rect> mouth;
	cclass_mouth.detectMultiScale(mouth_subimg, mouth, 1.1, 2, 0 |cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	std::cout << "Detected " << mouth.size() << " mouth(s) in the face for thread " << features->thread_id << std::endl;
	for (size_t i = 0; i < mouth.size(); i++)
		std::cout << "Found mouth " << mouth[i] << " for thread " << features->thread_id << std::endl;
	
	if (mouth.size() > 0)
	{
		// Convert the found feature Rect back to full image coordinate space
		features->mouth.x		= mouth_roi.x + mouth[0].x;
		features->mouth.y		= mouth_roi.y + mouth[0].y;
		features->mouth.width	= mouth[0].width;
		features->mouth.height	= mouth[0].height;
		std::cout << "Returning found mouth: " << features->mouth << std::endl;
	}
	
	return NULL;
}



/*
using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay( Mat image );

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
String nose_cascade_name = "haarcascade_mcs_nose.xml";
String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
CascadeClassifier face_cascade; //creating instance of CascadeClassifier class
CascadeClassifier eyes_cascade; //creating instance of CascadeClassifier class
CascadeClassifier nose_cascade;
CascadeClassifier mouth_cascade;


// @function main
int main( int argc, char** argv )
{
     if(argc!=2)
    {
        cout<<" usage : ffv Image_file_name"<<endl;
        return -1;
    }
    
    Mat image;
    //-- 1. Read the image
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if(!image.data)
    {
        cout<<"Image file not found\n";
        return -1;
    }
    
    //-- 2. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ cout<<"--(!)Error loading face cascade\n"; return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ cout<<"--(!)Error loading eyes cascade\n"; return -1; };
    if( !nose_cascade.load( nose_cascade_name ) ){ cout<<"--(!)Error loading nose cascade\n"; return -1; };
    if( !mouth_cascade.load( mouth_cascade_name ) ){ cout<<"--(!)Error loading mouth cascade\n"; return -1; };


    
    //-- 3. Apply the classifier to the image
    detectAndDisplay( image );
    
    return 0;
}

// @function detectAndDisplay
void detectAndDisplay( Mat image )
{
    std::vector<Rect> faces;
    Mat image_gray;

    cvtColor( image, image_gray, COLOR_BGR2GRAY ); //convert color 
    equalizeHist( image_gray, image_gray );

    //-- faces detection
    face_cascade.detectMultiScale( image_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    cout<<"Number of faces="<<faces.size()<<endl;
 
    for ( size_t i = 0; i < faces.size(); i++ )
    {
      
        rectangle(image,Point(faces[i].x,faces[i].y),Point(faces[i].x + faces[i].width,faces[i].y + faces[i].height),Scalar(255,0,255));
        Mat faceROI = image_gray(faces[i]);
        
        
        
        
        //eyes detection
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        

        for ( size_t j = 0; j < eyes.size(); j++ ) //eyes.size() always equal to 2 for one face picture
        {
            
            rectangle(image,Point(faces[i].x + eyes[j].x,faces[i].y + eyes[j].y),Point(faces[i].x + eyes[j].x + eyes[j].width,faces[i].y + eyes[j].y + eyes[j].height),Scalar(255,255,0));
           
        }

        //nose detection
        std::vector<Rect> nose;
        nose_cascade.detectMultiScale( faceROI, nose, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        
        if(nose.size()>0) 
        {
         rectangle(image,Point(faces[i].x + nose[0].x,faces[i].y + nose[0].y),Point(faces[i].x + nose[0].x + nose[0].width,faces[i].y + nose[0].y + nose[0].height),Scalar(0,255,0));
        }
        
        //mouth detection
        std::vector<Rect> mouth;
        mouth_cascade.detectMultiScale( faceROI, mouth, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        if(mouth.size()>0)
        {
            
            rectangle(image,Point(faces[i].x + mouth[0].x,faces[i].y + mouth[0].y),Point(faces[i].x + mouth[0].x + mouth[0].width,faces[i].y + mouth[0].y + mouth[0].height),Scalar(255,255,255));
            
        }
        
    }
    
    //-- Show what you got
    imshow( "Detected Face", image );
    cout<<"press any key to close window and exit!\n";
    waitKey(0);
    
}
*/