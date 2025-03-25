#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <iomanip> 

// Função para ler o arquivo CSV e separar características e rótulos
void read_csv(std::string filename, std::vector<std::vector<float>> &features, std::vector<int> &labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Arquivo invalido: " << filename << std::endl;
        return;
    }

    std::string line;
    std::vector<std::string> lines;

    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();

    features.resize(lines.size());
    labels.resize(lines.size());

    #pragma omp parallel for
    for (size_t i = 0; i < lines.size(); i++) {
        std::vector<float> row;
        std::stringstream ss(lines[i]);
        std::string cell;

        while (std::getline(ss, cell, ';')) {
            row.push_back(std::stof(cell));
        }

        labels[i] = static_cast<int>(row.back());
        row.pop_back(); //remove a categoria do vetor de características
        features[i] = row;
    }
}

//calcular a distância Euclidiana entre dois pontos
float euclidean_distance(const std::vector<float> &a, const std::vector<float> &b) {
    float dist = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(dist);
}

int knn(const std::vector<std::vector<float>> &train_features, const std::vector<int> &train_labels,
        const std::vector<float> &query, int k) {
    std::vector<std::pair<float, int>> distances(train_features.size());

    //paraleliza o cálculo das distâncias
    #pragma omp parallel for
    for (size_t i = 0; i < train_features.size(); i++) {
        float dist = euclidean_distance(train_features[i], query);
        distances[i] = std::make_pair(dist, i);
    }

    std::sort(distances.begin(), distances.end());

    //conta a frequência dos rótulos dos K vizinhos mais próximos
    std::vector<int> label_counts(2, 0);
    for (int i = 0; i < k; i++) {
        int label = train_labels[distances[i].second];
        label_counts[label]++;
    }

    return (label_counts[1] > label_counts[0]) ? 1 : 0;
}

//calcular a acurácia
float calculate_accuracy(const std::vector<int> &predictions, const std::vector<int> &labels) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.size();
}

int main() {
    std::vector<std::vector<float>> train_features, test_features;
    std::vector<int> train_labels, test_labels;

    read_csv("treinamentoLBP/base/train_norm.csv", train_features, train_labels);
    std::cout << "Dados de treinamento lidos: " << train_features.size() << std::endl;

    read_csv("treinamentoLBP/base/test_norm.csv", test_features, test_labels);
    std::cout << "Dados de teste lidos: " << test_features.size() << std::endl;

    int k = 3;
    std::vector<int> predictions(test_features.size());

    //classifica cada ponto de teste em paralelo
    #pragma omp parallel for
    for (size_t i = 0; i < test_features.size(); i++) {
        predictions[i] = knn(train_features, train_labels, test_features[i], k);
        std::cout << i << std::endl;
    }

    std::cout << "Fim do processamento" << std::endl;

    //calcula a acurácia
    float accuracy = calculate_accuracy(predictions, test_labels);
    std::cout << "Acurácia do modelo: " << accuracy * 100 << "%" << std::endl;

    return 0;
}