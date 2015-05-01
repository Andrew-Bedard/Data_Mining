#include <fstream>
#include <iostream>
#include <random>


int main()
{
    std::ifstream infile("../data/training_set_VU_DM_2014.csv");
    std::ofstream outfile("../data/decimated_training_set.csv");

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<> distReal(0, 1);
    
    std::string line;
    int counter = 0;
    std::getline(infile, line);
    outfile << line << std::endl;


    while (std::getline(infile, line)) {
        if (distReal(engine) < 0.01) {
            outfile << line << std::endl;
            counter++;
        }
    }
    std::cout << "number of lines: " << counter << std::endl;
    return 0;
}
