#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <sstream>  
#include <fstream> 

namespace fs = std::filesystem;
using namespace cv;
using namespace std;


//calcular o LBP de uma imagem em escala de cinza
Mat computeLBP(const Mat& img) {
    Mat lbpImage = Mat::zeros(img.rows - 2, img.cols - 2, CV_8UC1);

    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            uchar center = img.at<uchar>(i, j);
            uchar lbpCode = 0;

            lbpCode |= (img.at<uchar>(i-1, j-1) >= center) << 7;
            lbpCode |= (img.at<uchar>(i-1, j  ) >= center) << 6;
            lbpCode |= (img.at<uchar>(i-1, j+1) >= center) << 5;
            lbpCode |= (img.at<uchar>(i  , j+1) >= center) << 4;
            lbpCode |= (img.at<uchar>(i+1, j+1) >= center) << 3;
            lbpCode |= (img.at<uchar>(i+1, j  ) >= center) << 2;
            lbpCode |= (img.at<uchar>(i+1, j-1) >= center) << 1;
            lbpCode |= (img.at<uchar>(i  , j-1) >= center) << 0;

            lbpImage.at<uchar>(i-1, j-1) = lbpCode;
        }
    }
    return lbpImage;
}

//calcular o histograma LBP e armazená-lo em um vetor de características
vector<float> computeLBPHistogram(const Mat& lbpImage) {
    int histSize = 256;
    float range[] = {0, 256}; // considerando que temos 256 valores possíveis para o LBP	
    const float* histRange = {range};

    Mat hist;
    calcHist(&lbpImage, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
    // normalize(hist, hist, 1, 0, NORM_L1);
    vector<float> featureVector(histSize);
    for (int i = 0; i < histSize; i++) {
        featureVector[i] = hist.at<float>(i);
    }

    return featureVector;
}

void processDirectory(const std::string& basePath, const std::string& csvFile) {
    ofstream file(csvFile, ios::app);
    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo CSV!" << endl;
        return;
    }

    for (const auto& entry : fs::recursive_directory_iterator(basePath)) {
        if (entry.path().extension() == ".jpg") {
            std::string imgPath = entry.path().string();
            Mat img = imread(imgPath, IMREAD_GRAYSCALE);
            if (img.empty()) {
                cout << "Erro ao carregar a imagem: " << imgPath << endl;
                continue;
            }

            Mat lbpImage = computeLBP(img);
            vector<float> featureVector = computeLBPHistogram(lbpImage);

            ostringstream featureStream;
            for (float value : featureVector) {
                featureStream << value << ";";
            }

            featureStream << (imgPath.find("Empty") != string::npos ? "0" : "1");
            file << featureStream.str() << endl;
        }
    }

    file.close(); 
}

int main() {

    std::string basePath = "treinamentoLBP/base/PKLot/PKLotSegmented";
    std::string csvFile = "treinamentoLBP/base/output.csv";
    processDirectory(basePath, csvFile);

    return 0;
}