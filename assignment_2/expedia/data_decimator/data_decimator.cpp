#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdlib.h>    

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

    std::set<int> accepted_search_ids;
    std::set<int> rejected_search_ids;

    while (std::getline(infile, line)) {
        int searchid = atoi(line.substr(0, line.find(",")).c_str());

        if (accepted_search_ids.count(searchid) == 0 && rejected_search_ids.count(searchid) == 0) {
            if (distReal(engine) < 0.01) {
                outfile << line << std::endl;
                counter++;
                accepted_search_ids.insert(searchid);
            } else {
                rejected_search_ids.insert(searchid);
            }
        } else if (accepted_search_ids.count(searchid) > 0) {
                outfile << line << std::endl;
                counter++;
        }

    }
    std::cout << "number of lines: " << counter << std::endl;
    return 0;
}
