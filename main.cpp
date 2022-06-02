/*

            The purpose of this program is to clone an Apple feature called Center Stage.
            This project is for personal and educational purposes
            Me: https://www.sloandavis.com

*/

#include "main.h"


int main(int argc, const char **argv){
    //get path to Haar cascade xml file
    //the file trained for frontal face detection can be found on the OpenCV website
    std::string faces_path = "../../haarcascade_frontalface_alt.xml";

    //open the cascade classifier
    CascadeClassifier fClassifier;
    fClassifier.load(faces_path);
    if(fClassifier.empty()){
        cout << "Unable to find cascade\n";
        return -2;
    }
    cout << "Found Haar-Cascade Classifier Model\n";

    //run face detection and image cropping
    detectFaceAndCrop(1, fClassifier);
    
    cout << "Program Complete\n";
    return 0;
}

void detectFaceAndCrop(bool isDebug, CascadeClassifier &fCascade){
    //create object to store video frame and object for video capture    
    Mat frame;
    VideoCapture cap;

    //open video
    cap.open(0);

    if(!cap.isOpened()){
        //check if video loaded
        throw std::runtime_error("Could Not Open Camera");
    }

    //read in each frame until exit or error
    while(cap.read(frame)){
        bool doCalculations = true;
        if(frame.empty()){
            throw std::runtime_error("Dropped Frame");
        }

        //preproccess frame
        Mat frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGRA2GRAY);
        cv::equalizeHist(frameGray, frameGray);

        //detect faces into a vector
        std::vector<Rect> faceVec;
        fCascade.detectMultiScale(frameGray, faceVec);

        //SINGLE FACE SUPPORT CHECK
        if(faceVec.size() > 1){
            doCalculations = false;
        }

        //draw rectangle around face if debug enabled and there is a face found
        if(isDebug && !faceVec.empty() && doCalculations){
            cv::rectangle(frame, Point(faceVec[0].x, faceVec[0].y), Point(faceVec[0].x + faceVec[0].width, faceVec[0].y + faceVec[0].height), Scalar(255, 20, 114), 5);
        }

        cv::imshow("Webcam", frame);

        //check for key press to exit program
        if(waitKey(1) >= 1){
            break;
        }

    }
    
}