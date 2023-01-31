#pragma once
#ifndef MAIN_HPP_INCLUDED
#define MAIN_HPP_INCLUDED

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <argparse/argparse.hpp>
#include <cstdint>

using namespace std;
using namespace cv;

typedef std::vector<int> Seam;

/**
 *
 */
Mat differentialFilterX(const Mat &input, float delta);

/**
 *
 */
Mat differentialFilterY(const Mat &input, float delta);

/**
 *
 */
Mat channelGradient(const Mat &input);


/**
 * Returns the image's energy matrix (gradient)
 */
Mat energyMatrix(const Mat &input, float (*norm)(float, float, float));

/**
 * Returns the matrix showing energy paths from all starting points
 */
Mat energyPaths(const Mat3f &input);

/**
 * Returns the cumulative energy matrix M from the energy matrix (e.g. gradient)
 */
tuple<Mat1f, vector<int>> getCumulativeEnergy(const Mat1f &energy);

/**
 * Returns the seam from the cumulative energy matrix that starts at startCol
 */
Seam backtrack(const Mat1f &minCumulativeEnergy, int startCol);

/**
 * Removes a seam from the image by shifting the pixels to the left
 */
template<typename T>
Mat resize(const Mat &input, const Seam &seam);

/**
 * Creates a matrix of size input.rows x input.cols
 * such that for all M[i, j] = j
 */
Mat1i generateIndexMatrix(const Mat &input);

/**
 * Returns an image showing all the seams from the index matrix
 */
Mat3f getImageWithSeams(const Mat3f &input, const Mat1i &index);

/**
 *
 */
Mat3f seamCarve(const Mat3f &input, int newWidth, int newHeight, bool exportSeams);


#endif
