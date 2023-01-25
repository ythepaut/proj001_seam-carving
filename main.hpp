#pragma once
#ifndef MAIN_HPP_INCLUDED
#define MAIN_HPP_INCLUDED

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <argparse/argparse.hpp>

struct Seam
{
    std::vector<int> cols;
    int col_min{};
    int col_max{};
};

/**
 * Returns a seam from the cumulative energy matrix
 */
Seam backtrack(const cv::Mat &minCumulativeEnergy, int startCol);

/**
 * Returns the cumulative energy matrix M from the energy matrix (e.g. gradient)
 */
cv::Mat cumulativeEnergy(const cv::Mat &energy);

#endif
