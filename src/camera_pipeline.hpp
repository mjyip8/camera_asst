#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include "camera_pipeline_interface.hpp"
#include "image.hpp"
#include "pixel.hpp"

#ifdef __USE_HALIDE__
#include "Halide.h"
#include "halide_utils.hpp"
#endif

typedef std::vector<std::unique_ptr<Image<Float3Pixel>>> ImgVec;
typedef std::vector<std::unique_ptr<Image<FloatPixel>>> WeightVec;

class CameraPipeline : public CameraPipelineInterface {
 public:
    
  explicit CameraPipeline(CameraSensor* sensor)
    : CameraPipelineInterface(sensor) {}
    
 private:
  using T = typename CameraSensor::T;
  using CameraPipelineInterface::sensor_;

  std::unique_ptr<Image<RgbPixel>> ProcessShot() const override;

  // BEGIN: CS348K STUDENTS MODIFY THIS CODE

  //             MAIN PARTS 1 and 2
  std::unique_ptr<Image<RgbPixel>> BasicCameraRAWPipeline(std::unique_ptr<CameraSensorData<CameraSensor::T>>& raw_data) const;
  void ExposureFusion(std::unique_ptr<Image<RgbPixel>>& image) const;
  std::unique_ptr<Image<RgbPixel>> AlignAndMerge(std::unique_ptr<CameraSensorData<CameraSensor::T>>& raw_data, std::vector<std::unique_ptr<CameraSensorData<CameraSensor::T>>>& burst_raw_data) const;

  //             PART 1 HELPERS 
  bool OverThreshold(float threshold, float mean, float val) const;
  float CalculateNeighborMean(float neighbors, int row, int col, std::unique_ptr<CameraSensorData<CameraSensor::T>>& old_data, int step_size) const;
  float CornerShapeMean(std::unique_ptr<CameraSensorData<CameraSensor::T>>& old_data, int row, int col, int step_x, int step_y) const;
  void FixDefectivePixel(float& pixel, int row, int col, std::unique_ptr<CameraSensorData<CameraSensor::T>>& old_data) const;
  void RemoveDefectivePixels(std::unique_ptr<CameraSensorData<CameraSensor::T>>& raw_data) const;

  void MedianFilter(std::unique_ptr<Image<RgbPixel>>& image) const;
  void AddBlackOffset(std::unique_ptr<Image<RgbPixel>>& image_black, std::unique_ptr<Image<RgbPixel>>& image) const;
  std::unique_ptr<Image<RgbPixel>> ConvertToBayerFilterPattern(std::unique_ptr<CameraSensorData<CameraSensor::T>>& raw_data) const;
  void LinearDemosaik(std::unique_ptr<Image<RgbPixel>>& image) const;
  void ConvertToYCbCr(std::unique_ptr<Image<RgbPixel>>& image) const;
  void LowPassCbCr(std::unique_ptr<Image<RgbPixel>>& image) const;
  void ConvertToRGB(std::unique_ptr<Image<RgbPixel>>& image) const;

  //             PART 2 HELPERS 
  void CreateExposureBrackets(std::unique_ptr<Image<YuvPixel>>& dark, std::unique_ptr<Image<YuvPixel>>& bright) const;
  void ConvertToLuma(std::unique_ptr<Image<YuvPixel>>& dark, std::unique_ptr<Image<YuvPixel>>& bright) const;
  void ComputeWeights(std::unique_ptr<Image<Float3Pixel>>& dark_weights, std::unique_ptr<Image<Float3Pixel>>& bright_weights) const;

  // Pyramid helpers
  std::unique_ptr<Image<RgbPixel>> Downsample(std::unique_ptr<Image<RgbPixel>>& image) const;
  std::unique_ptr<Image<RgbPixel>> Upsample(std::unique_ptr<Image<RgbPixel>>& image, int width, int height) const;
  void Subtract(std::unique_ptr<Image<RgbPixel>>& left, std::unique_ptr<Image<RgbPixel>>& right) const;
  void Lerp(std::unique_ptr<Image<RgbPixel>>& left, std::unique_ptr<Image<RgbPixel>>& right, 
                                        std::unique_ptr<Image<RgbPixel>>& left_weights, std::unique_ptr<Image<RgbPixel>>& right_weights) const;
  std::unique_ptr<Image<RgbPixel>> Add(std::unique_ptr<Image<RgbPixel>>& left, std::unique_ptr<Image<RgbPixel>>& right) const;

  std::unique_ptr<Image<RgbPixel>> GaussianBlur(std::unique_ptr<Image<RgbPixel>>& image) const;
  std::unique_ptr<Image<Float3Pixel>> GaussianPyramidStep(std::unique_ptr<Image<Float3Pixel>>& layer) const;

   //             PART 3 HELPERS 
  std::unique_ptr<Image<RgbPixel>> AvgBayerGrid(std::unique_ptr<Image<RgbPixel>>& bayer_grid) const;
  void ConvertToGrayScale(std::unique_ptr<Image<RgbPixel>>& image) const; 
  std::vector<std::vector<int>> SplitIntoTiles(std::unique_ptr<Image<RgbPixel>>& image, int& n_tiles_in_col, int& n_tiles_in_row) const;
  float L1Residual(std::unique_ptr<Image<RgbPixel>>& ref_image, int start_row, int start_col, std::unique_ptr<Image<RgbPixel>>& burst_gaussian, int row, int col) const;
  float L2Residual(std::unique_ptr<Image<RgbPixel>>& ref_image, int start_row, int start_col, std::unique_ptr<Image<RgbPixel>>& burst_gaussian, int row, int col) const;
  ImgVec ConstructTiles(std::unique_ptr<Image<RgbPixel>>& image, std::vector<std::vector<int>> tile_idxs) const;
  float GetPixelCosineWeight(int row, int col) const;
  void GetRaisedCosine(std::unique_ptr<Image<FloatPixel>>& tile_weight, int tile_row, int tile_col, int ref_height_tiles, int ref_width_tiles) const;

  // END: CS348K STUDENTS MODIFY THIS CODE  
};
