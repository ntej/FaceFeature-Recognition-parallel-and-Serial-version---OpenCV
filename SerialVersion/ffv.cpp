#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat image );

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
String nose_cascade_name = "haarcascade_mcs_nose.xml";
String mouth_cascade_name = "haarcascade_mcs_mouth.xml";
CascadeClassifier face_cascade; //creating instance of CascadeClassifier class
CascadeClassifier eyes_cascade; //creating instance of CascadeClassifier class
CascadeClassifier nose_cascade;
CascadeClassifier mouth_cascade;


/** @function main */
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

/** @function detectAndDisplay */
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
