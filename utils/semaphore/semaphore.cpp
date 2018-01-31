#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include <semaphore.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>


void print_help() {
    std::cout << "semaphore SEMAPHORE_NAMES COMMAND [ARGUMENTS].." << std::endl;
    std::cout << "    SEMAPHORE_NAMES : a list of semaphore names with comma separators" << std::endl;
    std::cout << "    COMMAND         : open, wait, post, close, unlink" << std::endl;
    std::cout << "    ARGUMENTS       : arguments for 'open' command" << std::endl;
}

std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> terms;
    std::string term;
    std::istringstream iss(str);

    while (std::getline(iss, term, delim)) {
        terms.push_back(term);
    }

    return terms;
}

int main(int argc, char **argv) {

    if (argc < 3) {
        print_help();
        return 0;
    }

    std::string sem_names_str(argv[1]);
    std::string command(argv[2]);
    std::vector<std::string> sem_names = split(sem_names_str, ',');

    std::cout << "command=" << command << std::endl;

    unsigned int sem_value = 0;
    if (command == "open") {
        if (argc != 4) {
            std::cout << "'open' needs an initial value for semaphore." << std::endl;
            return 1;
        }

        sem_value = static_cast<unsigned int>(std::atoi(argv[3]));
        std::cout << "initial value=" << sem_value << std::endl;
    }

    sem_t *sem;
    int total_ret = 0;
    int ret;
    int e;
    int val;
    for (std::string sem_name : sem_names) {
        std::cout << std::setw(15) << std::left << "[" + sem_name + "] ";

        if (command == "unlink") {
            ret = sem_unlink(sem_name.c_str());
            if (ret == -1) {
                e = errno;
                std::cout << "ERR=" << e << "; failed to unlink semaphore." << std::endl;
                total_ret = 1;
            } else {
                std::cout << "OK" << std::endl;
            }
            continue;
        } else if (command == "open") {
            mode_t old_umask = umask(0);
            sem = sem_open(sem_name.c_str(), O_CREAT|O_RDWR, 0777, sem_value);
            umask(old_umask);
            if (sem == SEM_FAILED) {
                e = errno;
                std::cout << "ERR=" << e << "; failed to create semaphore." << std::endl;
                total_ret = 1;
            } else {
                std::cout << "OK" << std::endl;
            }
            continue;
        }

        sem = sem_open(sem_name.c_str(), O_RDWR);
        e = errno;
        if (sem == SEM_FAILED) {
            std::cout << "ERR=" << e << "; failed to open semaphore." << std::endl;
            total_ret = 1;
            continue;
        }

        if (command == "wait") {
            ret = sem_wait(sem);
            if (ret == -1) {
                e = errno;
                std::cout << "ERR=" << e << "; failed to wait semaphore." << std::endl;
                total_ret = 1;
            } else {
                std::cout << "OK" << std::endl;
            }
        } else if (command == "trywait") {
            ret = sem_trywait(sem);
            if (ret == -1) {
                e = errno;
                std::cout << "ERR=" << e << "; failed to trywait semaphore." << std::endl;
                total_ret = 1;
            } else {
                std::cout << "OK" << std::endl;
            }
        } else if (command == "post") {
            ret = sem_post(sem);
            if (ret == -1) {
                e = errno;
                std::cout << "ERR=" << e << "; failed to post semaphore." << std::endl;
                total_ret = 1;
            } else {
                std::cout << "OK" << std::endl;
            }
        } else if (command == "getvalue") {
            ret = sem_getvalue(sem, &val);
            if (ret == -1) {
                e = errno;
                std::cout << "ERR=" << e << "; failed to getvalue." << std::endl;
                total_ret = 1;
            } else {
                std::cout << "OK value=" << val << std::endl;
            }
        } else if (command == "close") {
            ret = sem_close(sem);
            if (ret == -1) {
                e = errno;
                std::cout << "ERR=" << e << "; failed to close semaphore." << std::endl;
                total_ret = 1;
            } else {
                std::cout << "OK" << std::endl;
            }
        } else {
            std::cout << "unknown command" << std::endl;
            total_ret = 1;
        }
    }

    return total_ret;
}

