#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

using std::cout;

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

using cv::Mat;
using cv::VideoCapture;
using cv::CascadeClassifier;
using cv::Rect;
using cv::Point;
using cv::Scalar;
using cv::waitKey;

void detectFaceAndCrop(bool isDebug, CascadeClassifier &fCascade);

