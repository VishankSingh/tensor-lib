/**
 * @file hello.cpp
 * @brief 
 * @author Vishank Singh, https://github.com/VishankSingh
 */

#include <thread>
#include <iostream>
#include <vector>

void helloWorld(int threads) {
    auto task = [](int id) {
        std::cout << "Hello, World from CPU! Thread ID: " << id << std::endl;
    };

    std::vector<std::thread> threadPool;
    for (int i = 0; i < threads; ++i) {
        threadPool.emplace_back(task, i);
    }

    for (auto& t : threadPool) {
        t.join();
    }

    threadPool.clear();

}