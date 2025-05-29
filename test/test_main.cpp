/**
 * @file test_main.cpp
 * @brief Main file for running tests.
 * @author Vishank Singh
 * Github: https://github.com/VishankSingh
 */

#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}