#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <libxml/parser.h>
#include <libxml/tree.h>

namespace fs = std::filesystem;
using namespace cv;

void processXML(const std::string& xmlPath, const std::string& imgPath, const std::string& outputBasePath) {
    // Carrega o XML
    xmlDocPtr doc = xmlReadFile(xmlPath.c_str(), NULL, 0);
    if (!doc) {
        std::cerr << "Erro ao ler o XML: " << xmlPath << std::endl;
        return;
    }

    // Carrega a imagem 
    Mat image = imread(imgPath);
    if (image.empty()) {
        std::cerr << "Erro ao carregar a imagem: " << imgPath << std::endl;
        xmlFreeDoc(doc);
        return;
    }

    xmlNodePtr root = xmlDocGetRootElement(doc);
    xmlNodePtr spaceNode = root->children;

    // Processa cada <space>
    while (spaceNode) {
        if (spaceNode->type == XML_ELEMENT_NODE && xmlStrcmp(spaceNode->name, BAD_CAST "space") == 0) {
            xmlChar* idAttr = xmlGetProp(spaceNode, BAD_CAST "id");
            xmlChar* occupiedAttr = xmlGetProp(spaceNode, BAD_CAST "occupied");
            std::string status = (xmlStrcmp(occupiedAttr, BAD_CAST "1") == 0) ? "Occupied" : "Empty";

            // Extrai dados do rotatedRect
            int x = 0, y = 0, w = 0, h = 0;
            float angle = 0.0;
            xmlNodePtr rectNode = spaceNode->children;
            while (rectNode) {
                if (rectNode->type == XML_ELEMENT_NODE && xmlStrcmp(rectNode->name, BAD_CAST "rotatedRect") == 0) {
                    xmlNodePtr rectChild = rectNode->children;
                    while (rectChild) {
                        if (xmlStrcmp(rectChild->name, BAD_CAST "center") == 0) {
                            x = std::stoi((char*)xmlGetProp(rectChild, BAD_CAST "x"));
                            y = std::stoi((char*)xmlGetProp(rectChild, BAD_CAST "y"));
                        } else if (xmlStrcmp(rectChild->name, BAD_CAST "size") == 0) {
                            w = std::stoi((char*)xmlGetProp(rectChild, BAD_CAST "w"));
                            h = std::stoi((char*)xmlGetProp(rectChild, BAD_CAST "h"));
                        } else if (xmlStrcmp(rectChild->name, BAD_CAST "angle") == 0) {
                            angle = std::stof((char*)xmlGetProp(rectChild, BAD_CAST "d"));
                        }
                        rectChild = rectChild->next;
                    }
                }
                rectNode = rectNode->next;
            }

            // Verifica se os dados são válidos
            if (w <= 0 || h <= 0) {
                std::cerr << "Dados inválidos para o espaço ID " << (char*)idAttr << std::endl;
                continue;
            }

            // Cria a matriz de rotação
            Mat rotationMatrix = getRotationMatrix2D(Point2f(x, y), angle, 1.0);

            // Aplica a rotação na imagem
            Mat rotatedImage;
            warpAffine(image, rotatedImage, rotationMatrix, image.size());

            // Recorta a ROI da imagem rotacionada
            Rect roi(x - w / 2, y - h / 2, w, h);
            Mat cropped = rotatedImage(roi);

            // Salva a imagem
            std::string outputPath = outputBasePath + "/" + status + "/" + fs::path(imgPath).stem().string() + "#" + (char*)idAttr + ".jpg";
            fs::create_directories(fs::path(outputPath).parent_path());
            imwrite(outputPath, cropped);

            xmlFree(idAttr);
            xmlFree(occupiedAttr);
        }
        spaceNode = spaceNode->next;
    }

    xmlFreeDoc(doc);
}

void processDirectory(const std::string& basePath, const std::string& outputBasePath) {
    for (const auto& entry : fs::recursive_directory_iterator(basePath)) {
        if (entry.path().extension() == ".xml") {
            std::string xmlPath = entry.path().string();
            std::string imgPath = std::filesystem::path(entry.path()).replace_extension(".jpg").string();
            if (fs::exists(imgPath)) {
                std::string relativePath = fs::relative(entry.path().parent_path(), basePath).string();
                std::string outputPath = outputBasePath + "/" + relativePath;
                processXML(xmlPath, imgPath, outputPath);
            }
        }
    }
}

int main() {
    std::string basePath = "treinamentoLBP/base/PKLot/PKLot";
    std::string outputBasePath = "treinamentoLBP/base/PKLot_Segmented";
    processDirectory(basePath, outputBasePath);
    return 0;
}