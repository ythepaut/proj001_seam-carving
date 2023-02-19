#pragma once
#ifndef MAIN_HPP_INCLUDED
#define MAIN_HPP_INCLUDED

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <argparse/argparse.hpp>
#include <cstdint>

using std::vector;
using std::set;
using std::tuple;
using std::string;

using cv::Mat_;
using cv::Mat3f;
using cv::Mat1f;
using cv::Mat1i;
using cv::Vec3f;

typedef vector<int> Seam;

/**
 * Differentiate image on x axis
 * @param input Image to be processed
 * @param delta Delta to apply, default is 0
 * @returns Image differentiated on x axis
 */
Mat1f differentialFilterX(const Mat1f &input, float delta);

/**
 * Differentiate image on y axis
 * @param input Image to be processed
 * @param delta Delta to apply, default is 0
 * @returns Image differentiated on y axis
 */
Mat1f differentialFilterY(const Mat1f &input, float delta);

/**
 * Calculate the gradient of an image using x and y differentiation
 * @param input Image to be processed
 * @returns Image gradient
 */
Mat1f channelGradient(const Mat1f &input);

/**
 * Calculates the gradient's norm (vector's norm)
 * @param a First component
 * @param b Second component
 * @param c Third component
 * @returns Euclidean distance
 */
float euclideanNorm(float a, float b, float c);

/**
 * Calculates the gradient's "norm" (sum of components)
 * @param a First component
 * @param b Second component
 * @param c Third component
 * @return Sum of all components
 */
float sumNorm(float a, float b, float c);

/**
 * @param input Image to be processed
 * @param norm Norm applied, default is euclidean
 * @returns Image's energy matrix (gradient)
 */
Mat1f energyMatrix(const Mat3f &input, float (*norm)(float, float, float));

/**
 * @param input Image to be processed
 * @returns Matrix showing energy paths from all starting points
 */
Mat3f energyPaths(const Mat3f &input);

/**
 * @param energy Energy matrix
 * @returns Cumulative energy matrix M from the energy matrix (e.g. gradient)
 *          and the beginning of all path starting at the last row
 */
tuple<Mat1f, vector<int>> getCumulativeEnergy(const Mat1f &energy);

/**
 * @param minCumulativeEnergy Minimum cumulative energy
 * @param startCol Starting column
 * @returns Seam from the cumulative energy matrix that starts at startCol
 */
Seam backtrack(const Mat1f &minCumulativeEnergy, int startCol);

/**
 * Removes a seam from the image by shifting the pixels to the left
 * @param input Image to be processed
 * @param seam Seam to be removed
 * @returns Image with seam removed
 */
template<typename T>
Mat_<T> resize(const Mat_<T> &input, const set<Seam> &seams);

/**
 * Creates a matrix of size input.rows x input.cols
 * such that for all M[i, j] = j
 * @param input Image to be processed
 * @returns Index matrix
 */
Mat1i generateIndexMatrix(const Mat3f &input);

/**
 * @param input Original input image
 * @param index Index matrix
 * @êeturns Image showing all the seams from the index matrix
 */
Mat3f getImageWithSeams(const Mat3f &input, const Mat1i &index);

/**
 * Carve an image
 * @param input Original image
 * @param newWidth New width of the image
 * @param newHeight New height of the image
 * @param exportSeams Whether to export seams or not
 */
Mat3f seamCarve(const Mat3f &input, int newWidth, int newHeight, bool exportSeams);

int main(int argc, char *argv[]);

#endif
