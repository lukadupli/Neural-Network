#include "pch.h"
#include "csv_read.h"

std::vector<std::vector<std::string>> csv_read(const char* path) {
    std::ifstream file(path);

    std::vector<std::vector<std::string>> ret;
    std::string line;

    while (getline(file, line)) {
        ret.push_back({});

        line.push_back(',');
        std::stringstream line_s(line);

        std::string val;

        while (getline(line_s, val, ',')) ret.back().push_back(val);
    }

    return ret;
}

