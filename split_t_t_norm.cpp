#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <map>
#include <limits>
#include <algorithm>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

//armazenar os valores mínimo e máximo do histograma
struct MinMax {
    vector<float> minValues;
    vector<float> maxValues;
};

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

//calcular o histograma LBP e armazenar em um vetor de características
vector<float> computeLBPHistogram(const Mat& lbpImage) {
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;
    calcHist(&lbpImage, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
    vector<float> featureVector(histSize);
    for (int i = 0; i < histSize; i++) {
        featureVector[i] = hist.at<float>(i);
    }
    return featureVector;
}

//normalizar um vetor de características com Min-Max Scaling
vector<float> normalizeFeatures(const vector<float>& features, const MinMax& minMax) {
    vector<float> normalized(features.size());
    for (size_t i = 0; i < features.size(); i++) {
        if (minMax.maxValues[i] > minMax.minValues[i]) {
            normalized[i] = (features[i] - minMax.minValues[i]) / (minMax.maxValues[i] - minMax.minValues[i]);
        } else {
            normalized[i] = 0.0f;
        }
    }
    return normalized;
}

void processDirectory(const std::string& basePath, const std::string& trainFile, const std::string& testFile) {
    map<string, bool> dateAssignment; //mapeia a pasta de data para treino (true) ou teste (false)
    MinMax minMax;
    minMax.minValues.resize(256, numeric_limits<float>::max());
    minMax.maxValues.resize(256, numeric_limits<float>::lowest());
    
    vector<string> dateFolders; 
    vector<pair<string, vector<float>>> allFeatures;
    
    //encontrar todas as pastas de datas
    for (const auto& entry : fs::recursive_directory_iterator(basePath)) {
        if (fs::is_directory(entry.path())) {
            string folderName = entry.path().filename().string();
            if (folderName.find("-") != string::npos) {
                if (find(dateFolders.begin(), dateFolders.end(), folderName) == dateFolders.end()) {
                    dateFolders.push_back(folderName);
                }
            }
        }
    }
    
    //ordenar as pastas de datas e atribuir alternadamente para treino e teste
    sort(dateFolders.begin(), dateFolders.end());
    bool toggle = true; //switch entre treino (true) e teste (false)
    for (const auto& date : dateFolders) {
        dateAssignment[date] = toggle;
        toggle = !toggle; 
    }

    //processar todas as imagens
    for (const auto& entry : fs::recursive_directory_iterator(basePath)) {
        if (entry.path().extension() == ".jpg") {
            std::string imgPath = entry.path().string();
            std::string dateFolder = entry.path().parent_path().parent_path().filename().string();
            
            if (dateAssignment.find(dateFolder) == dateAssignment.end()) {
                cout << "Erro: Pasta de data não encontrada no mapeamento: " << dateFolder << endl;
                continue;
            }
            
            //carregar a imagem em escala de cinza
            Mat img = imread(imgPath, IMREAD_GRAYSCALE);
            if (img.empty()) {
                cout << "Erro ao carregar a imagem: " << imgPath << endl;
                continue;
            }
            
            Mat lbpImage = computeLBP(img);
            vector<float> featureVector = computeLBPHistogram(lbpImage);
            
            //compara e armazena mínimos e máximos para normalização
            for (int i = 0; i < 256; i++) {
                minMax.minValues[i] = min(minMax.minValues[i], featureVector[i]);
                minMax.maxValues[i] = max(minMax.maxValues[i], featureVector[i]);
            }
            
            allFeatures.emplace_back(imgPath, featureVector);
        }
    }
    
    //salvar os dados normalizados nos arquivos de treino e teste
    ofstream train(trainFile), test(testFile);
    for (const auto& [imgPath, featureVector] : allFeatures) {
        std::string dateFolder = fs::path(imgPath).parent_path().parent_path().filename().string();
        
        //verificar se a imagem deve ser atribuída ao treino ou teste
        bool isTrain = dateAssignment[dateFolder];
        vector<float> normalized = normalizeFeatures(featureVector, minMax);
        string normFeatureString;
        for (float value : normalized) {
            normFeatureString += to_string(value) + ";";
        }
        normFeatureString += (imgPath.find("Empty") != string::npos ? "0" : "1");
        
        //escrever no arquivo de treino ou teste
        (isTrain ? train : test) << normFeatureString << endl;
    }
    train.close();
    test.close();

    int trainCount = 0, testCount = 0;
    for (const auto& [date, isTrain] : dateAssignment) {
        if (isTrain) trainCount++;
        else testCount++;
    }
    cout << "Datas no Treino: " << trainCount << ", Datas no Teste: " << testCount << endl;
}

int main() {
    std::string basePath = "treinamentoLBP/base/PKLot/PKLotSegmented";
    std::string trainFile = "treinamentoLBP/base/train_norm.csv";
    std::string testFile = "treinamentoLBP/base/test_norm.csv";
    
    processDirectory(basePath, trainFile, testFile);
    return 0;
}
