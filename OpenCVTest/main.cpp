#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>



#include <iostream>



using namespace cv;



using namespace std;

void markSeam(Mat& image, Mat& index, bool horizontal = false){
    for (int i=0;i<image.rows;i++){
        int column = index.at<int32_t>(i,0);
        image.at<Vec3b>(i,column)[0] = 0;
        image.at<Vec3b>(i,column)[1] = 0;
        image.at<Vec3b>(i,column)[2] = 255;
    }

}
void removeVerticalSeam(Mat &image,Mat&newImage, Mat &index){
    int oldrows = image.rows;
    int oldcols = image.cols;
    newImage = Mat(oldrows, oldcols-1, image.type());
    cout <<"New Image dims:" <<newImage.rows <<"cols:"<<newImage.cols<<endl;
    //copy image to new destination
    for (int i=0;i<oldrows;i++){
        int idx = index.at<int32_t>(i,0);
        bool skipped = false;
        for (int j=0;j<oldcols;j++){
            // check for curr pixel to skip
            if (j==idx){
                skipped = true;
            }
            else{
                //normal
                if (!skipped){
                    Vec3b oldColor = image.at<cv::Vec3b>(i,j);
                    newImage.at<Vec3b>(i,j) = oldColor;
                }
                else{
                    Vec3b oldColor = image.at<cv::Vec3b>(i,j);
                    newImage.at<Vec3b>(i,j-1) = oldColor;
                }
            }
        }
    }
}
void verticalSeamDP(Mat &image, Mat &M, Mat &dpIndex){
    
    //get dimensions
    unsigned int m,n;
    n = image.rows;
    m = image.cols;
    M.create(Size(m,n),CV_32S);
    dpIndex.create(Size(1,n),CV_32S);
    
    cout <<"Image:" << image.size()<<endl;
    cout <<"M:"<<M.size()<<endl;
    cout <<"n,m"<<n<<","<<m<<endl;
    // init first line
    for (int i=0;i<n;i++)
        M.at<int32_t>(0,i) = (int) image.at<uchar>(0,i);
    
    for ( unsigned int i=1;i<n;i++){
        for (unsigned int j=0;j<m;j++){
           // M.at<int32_t>(j,i) = (int) image.at<uchar>(j,i);
            int currVal = (int) image.at<uchar>(i,j);
            //left border
            if ( j == 0){
                //compute min DP value
                int top = M.at<int32_t>(i-1,j);
                int topRight = M.at<int32_t>(i-1,j+1);
                
                int minPreVal = std::min(top,topRight);
                M.at<int32_t>(i,j) = currVal + minPreVal;
            }
            // right border
            else if (j == m-1){
                int top = M.at<int32_t>(i-1,j);
                int topLeft = M.at<int32_t>(i-1,j-1);
                int minPreVal = std::min(top,topLeft);
                M.at<int32_t>(i,j) = currVal + minPreVal;
            }
            // all the rest
            else {
                int top = M.at<int32_t>(i-1,j);
                int topLeft = M.at<int32_t>(i-1,j-1);
                int topRight = M.at<int32_t>(i-1,j+1);
                int minPreVal = std::min(std::min(top,topLeft),topRight);
                M.at<int32_t>(i,j) = currVal + minPreVal;
            }
        }
    }
    // find seam
    
    // lowest value in last row
    //M.at<int32_t>(19,0) = 900;
    cv::Point minIdx,maxIdx;
    double minVal,maxVal;
    cv::minMaxLoc(M.row(n-1), &minVal,&maxVal,&minIdx,&maxIdx);
  //  std::cout <<"MinVal:"<<minVal<<" at: " <<minIdx<<endl;
   // std::cout <<"MaxVal:"<<maxVal<<" at: " <<maxIdx<<endl;
    int index = minIdx.x;
    dpIndex.at<int32_t>(n-1,0) = index;
    for (int i=n-2;i>=0;i--){
        int leftIdx = max(index-1,0);
        int topIdx = index;
        int rightIdx = min(index+1,(int) m-1);
        int leftVal = M.at<int32_t>(i,leftIdx);
        int rightVal = M.at<int32_t>(i,rightIdx);
        int topVal = M.at<int32_t>(i,topIdx);
        int smallestVal = leftVal;
        index = leftIdx;
        if (topVal<=smallestVal){
            smallestVal = topVal;
            index = topIdx;
        }
        if (rightVal<smallestVal){
            smallestVal = rightVal;
            index = rightIdx;
        }
        dpIndex.at<int32_t>(i,0) = index;
        //cout <<"At row "<<i<<" val:"<<smallestVal<<" at "<<index<<endl;
    }
}
void computeGradient(Mat &image, Mat &grad)
{
    Mat imageblur,image_gray, grad_x,grad_y,abs_grad_x,abs_grad_y;
    GaussianBlur(image, imageblur, Size(5,5), 0,0,BORDER_DEFAULT);
    
    // convert2gray
    cvtColor(imageblur, image_gray, CV_BGR2GRAY);
    
    // gradient
    Sobel(image_gray, grad_x, CV_16S,1, 0, 3,1,0,BORDER_DEFAULT);
    Sobel(image_gray, grad_y, CV_16S,0, 1, 3,1,0,BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
}
int main( int argc, char** argv ) {
    
    
    Mat image, image_gray;
    Mat grad;
    Mat grad_x, grad_y,abs_grad_x,abs_grad_y;
    Mat imageOut;
    Mat frame, frameOut, frameColor;
    string windowName = "Window";
    /*
    VideoCapture capture;
    capture.open(0);
    namedWindow("Camera", WINDOW_AUTOSIZE);
    namedWindow("Blur",WINDOW_AUTOSIZE);
    bool stopIt = false;
    while (capture.isOpened() && !stopIt) {
        
        capture >> frameColor;
        cvtColor(frameColor, frame, COLOR_BGR2GRAY);
        //GaussianBlur(frame, frameOut, Size(25,25), 0);
        medianBlur(frame, frameOut, 5);
        imshow("Camera",frame);
        imshow("Blur",frameOut);
    }
    */
    
   image = imread("clubb .png", IMREAD_COLOR); // Read the file
    
    
    
    
    
    if(!image.data ) {
        
        
        cout << "Could not open or find the image" << std::endl ;
        
        
        return -1;
        
        
    }
    
    // gaussianBlur
    
    namedWindow(windowName, WINDOW_AUTOSIZE ); // Create a window for display.
    namedWindow("newImageWithSeam", WINDOW_AUTOSIZE ); // Create a window for display.
    namedWindow("newImage",WINDOW_AUTOSIZE);
    namedWindow("debug",WINDOW_AUTOSIZE);
    namedWindow("grad",WINDOW_AUTOSIZE);
    namedWindow("dpmImage",WINDOW_AUTOSIZE);
    Mat dpMImage;
    //compute Seam
    Mat dpM,dpIndex,image2;
    Mat origImage;
    origImage = image.clone();
    for (int i=0;i<300 ;i++){
        computeGradient(image, grad);
        verticalSeamDP(grad, dpM, dpIndex);
        //cout << dpM<<endl;
        //cout << dpIndex<<endl;
        removeVerticalSeam(image, image2, dpIndex);
        //debug show every steop
        Mat imageSeam = image.clone();
        markSeam(imageSeam,dpIndex);
        image2.copyTo(image);
        convertScaleAbs(dpM, dpMImage);
        imshow("debug",imageSeam  );
        imshow("grad", grad);
        imshow("dpm",dpMImage);
        cvWaitKey(0);
    }
    namedWindow("small",WINDOW_AUTOSIZE);
    namedWindow("orig",WINDOW_AUTOSIZE);
    imshow("orig",origImage);
    imshow("small", image);
    cvWaitKey(0);
    exit(0);
   
    computeGradient(image, grad);
    
    verticalSeamDP(grad, dpM,dpIndex);
    //paint image
    for (int i=0;i<image.rows;i++){
        int column = dpIndex.at<int32_t>(i,0);
        image.at<Vec3b>(i,column)[0] = 0;
        image.at<Vec3b>(i,column)[1] = 0;
        image.at<Vec3b>(i,column)[2] = 255;
    }
        
   // cout <<"dpM:"<<endl<<dpM<<endl;
   // cout <<"grad:"<<endl<<grad<<endl;
   // cout <<"dpIndex:"<<endl<<dpIndex<<endl;
    imshow(windowName, image ); // Show our image inside it.
    // imshow("M",dpM);
    Mat newImage,newImage2;
    removeVerticalSeam(image, newImage, dpIndex);
    //imshow("newImage", newImage);
    verticalSeamDP(newImage, dpM, dpIndex);
    for (int i=0;i<newImage.rows;i++){
        int column = dpIndex.at<int32_t>(i,0);
        newImage.at<Vec3b>(i,column)[0] = 0;
        newImage.at<Vec3b>(i,column)[1] = 0;
        newImage.at<Vec3b>(i,column)[2] = 255;
    }
    namedWindow("seam2",WINDOW_AUTOSIZE);
    imshow("seam2", newImage);
    removeVerticalSeam(newImage, newImage2, dpIndex);
    cvWaitKey(0);
    //waitKey(0); // Wait for a keystroke in the window
    
   // try more removals
    Mat testImage, testImage2;

    
    return 0;
    
    
}