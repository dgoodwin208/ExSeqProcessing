#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>

namespace utils {

std::string getCommonPrefix(const std::string& str1, const std::string& str2) {

    std::string common_prefix;
    for (size_t i = 0; i < std::min(str1.size(), str2.size()); i++) {
        if (str1[i] == str2[i]) {
            common_prefix += str1[i];
        } else {
            break;
        }
    }

    return common_prefix;
}


}

#endif // __UTILS_H__

