#include <iostream>
#include <fstream>
#include <string>
#include <json/json.h>

void LoadData(float data[32][32][1][2], int shape[4], string fileName, string itemName)
{   
    shape[0] = 32;
    shape[1] = 32;
    shape[2] = 1;
    shape[3] = 2;
    Json::Value values;
    std::ifstream value_file("data/metr_epoch_33_2.8.json", std::ifstream::binary);
    value_file >> values;

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 1; k++) {
                for (int l = 0; l < 2; l++) {
                    data[i][j][k][l] = values["filter_convs.1.weight"]["__ndarray__"][i][j][k][l].asFloat();
                    std::cout << data[i][j][k][l] << " ";
                }
            }
        }
    }
}