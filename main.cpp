#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <argparse/argparse.hpp>

#include "main.hpp"

using namespace cv;
using namespace std;
using namespace argparse;

Mat differentialFilterX(const Mat &input, float delta = 0.f)
{
    Mat output;
    Mat kernel = (Mat_<float>(3, 3) << 1 / 4., 0, -1 / 4., 2 / 4., 0, -2 / 4., 1 / 4., 0, -1 / 4.);
    filter2D(input, output, -1, kernel, Point(-1, -1), delta, BORDER_DEFAULT);
    return output;
}

Mat differentialFilterY(const Mat &input, float delta = 0.f)
{
    Mat output;
    Mat kernel = (Mat_<float>(3, 3) << -1 / 4., -1 / 2., -1 / 4., 0, 0, 0, 1 / 4., 1 / 2., 1 / 4.);
    filter2D(input, output, -1, kernel, Point(-1, -1), delta, BORDER_DEFAULT);
    return output;
}

Mat channelGradient(const Mat &input)
{
    Mat output(input.rows, input.cols, CV_32FC1);
    Mat dx = differentialFilterX(input);
    Mat dy = differentialFilterY(input);
    for (int row = 0; row < input.rows; ++row)
        for (int col = 0; col < input.cols; ++col)
            output.at<float>(row, col) = (
                    dx.at<float>(row, col) * dx.at<float>(row, col) +
                    dy.at<float>(row, col) * dy.at<float>(row, col)
            );
    return output;
}

float euclideanNorm(float a, float b, float c)
{
    return sqrt(a * a + b * b + c * c);
}

/**
 * Returns the image's energy matrix (gradient)
 */
Mat energyMatrix(const Mat &input, float (*norm)(float, float, float))
{
    vector<Mat> channels;
    split(input, channels);

    vector<Mat> gradients;
    for (const Mat &channel: channels)
        gradients.push_back(channelGradient(channel));

    Mat energy(input.rows, input.cols, CV_32FC1);
    for (int row = 0; row < input.rows; ++row)
        for (int col = 0; col < input.cols; ++col)
            energy.at<float>(row, col) = norm(
                    gradients[0].at<float>(row, col),
                    gradients[1].at<float>(row, col),
                    gradients[2].at<float>(row, col)
            );

    return energy;
}

void exportAllEnergyPath(const Mat &input, float (*norm)(float, float, float))
{
    Mat output(input.rows, input.cols, CV_32FC1);
    Mat energy = energyMatrix(input, norm);
    for (int startCol = 0; startCol < input.cols; ++startCol)
    {
        int col = startCol;
        for (int row = 0; row < input.rows; ++row)
        {
            ++output.at<float>(row, col);
            float minEnergy = INFINITY;
            int minOffset = 0;
            for (int offset = -1; offset <= 1; ++offset)
            {
                if (col + offset < 0 || col + offset >= energy.cols)
                    continue;

                if (minEnergy < energy.at<float>(row - 1, col + offset))
                {
                    minEnergy = energy.at<float>(row - 1, col + offset);
                    minOffset = offset;
                }
            }
            col += minOffset;
        }
        ++output.at<float>(output.rows - 1, col);
    }

    float maxi = 0;
    for (int col = 0; col < input.cols; ++col)
        if (output.at<float>(output.rows - 1, col) > maxi)
            maxi = output.at<float>(output.rows - 1, col);

    output /= maxi;
    imwrite("../out/energyPaths.png", output * 256);
}

Mat cumulativeEnergy(const Mat &input)
{
    Mat energy = energyMatrix(input, euclideanNorm);
    Mat minCumulativeEnergy(energy.rows, energy.cols, CV_32FC1);

    for (int col = 0; col < minCumulativeEnergy.cols; ++col)
        minCumulativeEnergy.at<float>(0, col) = energy.at<float>(0, col);

    for (int row = 1; row < minCumulativeEnergy.rows; ++row)
    {
        for (int col = 0; col < minCumulativeEnergy.cols; ++col)
        {
            float minEnergy = INFINITY;
            for (int offset = -1; offset <= 1; ++offset)
            {
                if (col + offset < 0 || col + offset >= minCumulativeEnergy.cols)
                    continue;

                minEnergy = MIN(minEnergy, minCumulativeEnergy.at<float>(row - 1, col + offset));
            }

            minCumulativeEnergy.at<float>(row, col) = energy.at<float>(row, col) + minEnergy;
        }
    }

    return minCumulativeEnergy;
}

vector<int> backtrack(const Mat &input)
{
    vector<int> seam;
    Mat minCumulativeEnergy = cumulativeEnergy(input);

    int startCol = 0;
    for (int col = 1; col < minCumulativeEnergy.cols; ++col)
        if (minCumulativeEnergy.at<float>(minCumulativeEnergy.rows - 1, col) <
            minCumulativeEnergy.at<float>(minCumulativeEnergy.rows - 1, startCol))
            startCol = col;

    int currentCol = startCol;
    for (int row = minCumulativeEnergy.rows - 1; row > 0; --row)
    {
        seam.push_back(currentCol);
        int minCol = currentCol;
        for (int offset = -1; offset <= 1; ++offset)
        {
            if (currentCol + offset < 0 || currentCol + offset >= minCumulativeEnergy.cols)
                continue;
            if (minCumulativeEnergy.at<float>(row, currentCol + offset) < minCumulativeEnergy.at<float>(row, minCol))
                minCol = currentCol + offset;
        }
        currentCol = minCol;
    }
    seam.push_back(currentCol);
    return seam;
}

Mat resize(const Mat3f &input, const vector<int> &seam)
{
    Mat3f output(input.rows, input.cols - 1, CV_32FC3);
    for (int row = 0; row < input.rows; ++row)
    {
        int found = 0;
        int seamCol = seam[row];
        for (int col = 0; col < input.cols; ++col) {
            if (col == seamCol)
                found = 1;
            else
                output.at<Vec3f>(row, col - found) = input.at<Vec3f>(row, col);
        }
    }
    return output;
}

void exportSeam(const Mat3f &input, vector<int> seam)
{
    Mat3f output = input.clone();
    for (int i = 0 ; i < seam.size() ; ++i)
        output.at<Vec3f>(i, seam[i]) = Vec3f(0, 0, 1);
    imwrite("../out/seam.png", output * 256);
}

Mat seamCarve(const Mat3f &input, int newWidth, int newHeight)
{
    Mat3f output = input.clone();

    for (int i = 0; i < 250; ++i)
    {
        cout << i << endl;
        vector<int> seam = backtrack(output);
        //exportSeam(output, seam);
        output = resize(output, seam);
    }

    return output;
}

int main(int argc, char *argv[])
{
    ArgumentParser program("seam_carve");

    program.add_argument("image")
            .help("input image path");
    program.add_argument("-o", "--output")
            .help("output directory path");
    program.add_argument("-w", "--width")
            .help("output image width");
    program.add_argument("-H", "--height")
            .help("output image height");


    try
    {
        program.parse_args(argc, argv);

        auto const &inputPath = program.get<string>("image");
        auto const &outputPath = program.get<string>("-o");

        Mat input = imread(inputPath);
        input.convertTo(input, CV_32FC3, 1 / 256.f);

        exportAllEnergyPath(input, euclideanNorm);

        // Mat3f output = seamCarve(input, 100, 100);
        // imwrite(outputPath, output * 256);
    }
    catch (const runtime_error &err)
    {
        cerr << err.what() << std::endl;
        cerr << program;
        return 1;
    }
    return 0;
}
