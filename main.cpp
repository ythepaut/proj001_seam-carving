#include <iostream>
#include <set>

#include "main.hpp"

using std::cout;
using std::cerr;
using std::endl;

Mat1f differentialFilterX(const Mat1f &input, float delta = 0.f)
{
    Mat1f output;
    Mat1f kernel = (Mat1f(3, 3) << 1 / 4., 0, -1 / 4., 2 / 4., 0, -2 / 4., 1 / 4., 0, -1 / 4.);
    filter2D(input, output, -1, kernel, cv::Point(-1, -1), delta);
    return output;
}

Mat1f differentialFilterY(const Mat1f &input, float delta = 0.f)
{
    Mat1f output;
    Mat1f kernel = (Mat1f(3, 3) << -1 / 4., -1 / 2., -1 / 4., 0, 0, 0, 1 / 4., 1 / 2., 1 / 4.);
    filter2D(input, output, -1, kernel, cv::Point(-1, -1), delta);
    return output;
}

Mat1f channelGradient(const Mat1f &input)
{
    Mat1f output(input.rows, input.cols);
    Mat1f dx = differentialFilterX(input);
    Mat1f dy = differentialFilterY(input);
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
    return std::sqrt(a * a + b * b + c * c);
}

Mat1f energyMatrix(const Mat3f &input, float (*norm)(float, float, float) = euclideanNorm)
{
    // Splits the channels for the gradient computation
    vector<Mat1f> channels(3);
    split(input, channels);

    // Computes and stores the gradient for each channel
    vector<Mat1f> gradients;
    for (const Mat1f &channel: channels)
        gradients.push_back(channelGradient(channel));

    // Computes the energy matrix from the norm
    Mat1f energy(input.rows, input.cols);
    energy.at<float>(0, 0) = 1.0;
    for (int row = 0; row < input.rows; ++row)
        for (int col = 0; col < input.cols; ++col)
            energy.at<float>(row, col) = norm(
                    gradients[0].at<float>(row, col),
                    gradients[1].at<float>(row, col),
                    gradients[2].at<float>(row, col)
            );

    return energy;
}

Mat3f energyPaths(const Mat3f &input)
{
    Mat1f output(input.rows, input.cols, CV_32FC1);
    Mat1f energy = energyMatrix(input);
    auto [cumulativeEnergy, startColumns] = getCumulativeEnergy(energy);
    for (int startCol = 0; startCol < output.cols; ++startCol)
    {
        Seam seam = backtrack(cumulativeEnergy, startCol);
        for (int row = 0; row < output.rows; ++row)
            ++output.at<float>(row, seam[row]);
    }

    float maxi = 0;
    for (int col = 0; col < output.cols; ++col)
        if (output.at<float>(output.rows - 1, col) > maxi)
            maxi = output.at<float>(output.rows - 1, col);

    output += 1.f;

    for (int col = 0; col < output.cols; ++col)
        for (int row = 0; row < output.rows; ++row)
            output.at<float>(row, col) = std::log(output.at<float>(row, col));

    output /= std::log(maxi + 1.f);

    output.convertTo(output, CV_32FC3);
    return output;
}

tuple<Mat1f, vector<int>> getCumulativeEnergy(const Mat1f &energy)
{
    Mat1f minCumulativeEnergy(energy.rows, energy.cols);
    vector<int> start(energy.cols);

    for (int col = 0; col < minCumulativeEnergy.cols; ++col)
    {
        minCumulativeEnergy.at<float>(0, col) = energy.at<float>(0, col);
        start.at(col) = col;
    }

    for (int row = 1; row < minCumulativeEnergy.rows; ++row)
    {
        for (int col = 0; col < minCumulativeEnergy.cols; ++col)
        {
            float minEnergy = minCumulativeEnergy.at<float>(row - 1, col);
            int minOffset = 0;
            for (int offset = -1; offset <= 1; offset += 2)
            {
                if (col + offset < 0 || col + offset >= minCumulativeEnergy.cols)
                    continue;

                float offsetEnergy = minCumulativeEnergy.at<float>(row - 1, col + offset);
                if (offsetEnergy < minEnergy)
                {
                    minEnergy = offsetEnergy;
                    minOffset = offset;
                }
            }

            minCumulativeEnergy.at<float>(row, col) = energy.at<float>(row, col) + minEnergy;
            start.at(col) = col + minOffset;
        }
    }

    return make_tuple(minCumulativeEnergy, start);
}

Seam backtrack(const Mat1f &minCumulativeEnergy, int startCol)
{
    vector<int> seamCols;

    int currentCol = startCol;
    for (int row = minCumulativeEnergy.rows - 1; row > 0; --row)
    {
        seamCols.push_back(currentCol);
        int minCol = currentCol;
        for (int offset = -1; offset <= 1; offset += 2)
        {
            if (currentCol + offset < 0 || currentCol + offset >= minCumulativeEnergy.cols)
                continue;
            if (minCumulativeEnergy.at<float>(row, currentCol + offset) < minCumulativeEnergy.at<float>(row, minCol))
                minCol = currentCol + offset;
        }
        currentCol = minCol;
    }
    seamCols.push_back(currentCol);
    return seamCols;
}

template<typename T>
Mat_<T> resize(const Mat_<T> &input, const Seam &seam)
{
    // TODO Allow multiple seam deletion
    cv::Mat_<T> output = input.clone();
    // Shifts all the pixels on the right of the seam
    for (int row = 0; row < output.rows; ++row)
        for (int col = seam[row]; col < output.cols - 1; ++col)
            output.template at<T>(row, col) = output.template at<T>(row, col + 1);
    // Crop the output image
    return output(cv::Rect(0, 0, output.cols - 1, output.rows));
}

Mat1i generateIndexMatrix(const Mat3f &input)
{
    Mat1i output(input.rows, input.cols);
    for (int row = 0; row < input.rows; ++row)
        for (int col = 0; col < input.cols; ++col)
            output.at<int>(row, col) = col;
    return output;
}

Mat3f getImageWithSeams(const Mat3f &input, const Mat1i &index)
{
    // Fills the image with red pixels
    Mat3f output(input.rows, input.cols);
    for (int row = 0; row < output.rows; ++row)
        for (int col = 0; col < output.cols; ++col)
            output.at<Vec3f>(row, col) = Vec3f(0., 0., 1.);

    // Adds the input image's pixels at their original position
    for (int row = 0; row < output.rows; ++row)
    {
        for (int col = 0; col < index.cols; ++col)
        {
            int newCol = index.at<int>(row, col);
            output.at<Vec3f>(row, newCol) = input.at<Vec3f>(row, newCol);
        }
    }

    return output;
}

Mat3f seamCarve(const Mat3f &input, int newWidth, int newHeight, bool exportSeams)
{
    Mat3f output = input.clone();

    // Index matrix for seams visualisation
    Mat1i index;
    if (exportSeams)
        index = generateIndexMatrix(input);

    // TODO: Use newWidth instead
    int dw = 500;
    int removedSeams = 0;

    while (removedSeams < dw)
    {
        // Get all candidate seams
        auto [cumulativeEnergy, startColumns] = getCumulativeEnergy(energyMatrix(output));

        // Stores the index of the columns with the least energy on the last row
        vector<int> sortedEnergiesIndex(cumulativeEnergy.cols);
        for (int i = 0; i < cumulativeEnergy.cols; ++i)
            sortedEnergiesIndex.push_back(i);

        // Sort energies
        sort(
                sortedEnergiesIndex.begin(),
                sortedEnergiesIndex.end(),
                [cumulativeEnergy](int i, int j)
                {
                    const int lastRow = cumulativeEnergy.rows - 1;
                    const float energyI = cumulativeEnergy.at<float>(lastRow, i);
                    const float energyJ = cumulativeEnergy.at<float>(lastRow, j);
                    return energyI < energyJ;
                }
        );

        // Last positions (stores the last position for each seam to check collisions)
        set<int> lastCols;

        // Finds the best seams that do not intersect
        set<Seam> seams;
        int i = 0;

        float minimumEnergy = cumulativeEnergy.at<float>(cumulativeEnergy.rows - 1, sortedEnergiesIndex[0]);

        while (
                i < (int) sortedEnergiesIndex.size() &&
                cumulativeEnergy.at<float>(cumulativeEnergy.rows - 1, sortedEnergiesIndex[i]) <= 2 * minimumEnergy &&
                (int) seams.size() < output.cols / 10 &&
                removedSeams + (int) seams.size() < dw
                )
        {
            // If seam does not intersect any other seam
            // FIXME: Store seam last column instead of computing backtrack each time
            const Seam seam = backtrack(cumulativeEnergy, sortedEnergiesIndex[i]);
            int seamLastCol = seam[seam.size() - 1];
            if (lastCols.find(seamLastCol) == lastCols.end())
            {
                // Add seam to seams
                //const Seam seam = backtrack(cumulativeEnergy, sortedEnergiesIndex[i]);
                seams.insert(seam);
                // Add seam last column to last positions
                lastCols.insert(seamLastCol);
            }
            ++i;
        }

        // Remove seams from the image
        for (const Seam &seam: seams)
        {
            output = resize<Vec3f>(output, seam);
            if (exportSeams)
                index = resize<int>(index, seam);
        }
        removedSeams += (int) seams.size();

        cout << removedSeams << "/" << dw << " (-" << seams.size() << ")\r";
        cout << endl;
    }

    return exportSeams ? getImageWithSeams(input, index) : output;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("seam_carve");

    program.add_argument("image")
            .help("input image path");
    program.add_argument("-o", "--output")
            .help("output directory path");
    program.add_argument("-w", "--width")
            .help("output image width");
    program.add_argument("-H", "--height")
            .help("output image height");
    program.add_argument("--energy-paths")
            .help("export energy paths from the input image")
            .default_value(false)
            .implicit_value(true);
    program.add_argument("--export-seams")
            .help("export the input image with the seams")
            .default_value(false)
            .implicit_value(true);

    try
    {
        program.parse_args(argc, argv);

        auto const &inputPath = program.get<string>("image");
        auto const &outputPath = program.get<string>("-o");

        Mat3f input = cv::imread(inputPath);
        input.convertTo(input, CV_32FC3, 1 / 256.f);

        Mat3f output;
        if (program.get<bool>("--energy-paths"))
            output = energyPaths(input);
        else
            output = seamCarve(input, 100, 100, program.get<bool>("--export-seams"));

        imwrite(outputPath, output * 256);
    }
    catch (const std::runtime_error &err)
    {
        cerr << err.what() << std::endl;
        cerr << program;
        return 1;
    }
    return 0;
}
