#include <iostream>

#ifdef __USE_HALIDE__
#include "Halide.h"
#include "halide_utils.hpp"
#endif

#include "camera_pipeline.hpp"
#include <omp.h>
#include <limits>
#include <vector>

typedef std::vector<std::vector<std::vector<int>>> BurstPyramidTileDisplaceVec;
typedef std::vector<std::vector<std::vector<int>>> PyramidTilesVec;

static constexpr int n_threads = 4;
static constexpr int depth = 5;


// *********************************************************************************
//                              PART 1 HELPERS
// *********************************************************************************

bool CameraPipeline::OverThreshold(float threshold, float mean, float val) const {
  float max = mean * (1. + threshold);
  float min = mean * (1. - threshold);
  return (val > max || val < min);
}

float CameraPipeline::CalculateNeighborMean(float neighbors, int row, int col, std::unique_ptr<CameraSensorData<CameraSensor::T>>& old_data, int step_size) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  // compute neighborhood size
  int row_min = std::max(row - step_size, 0);
  int row_max = std::min(row + step_size, height - 1);
  int col_min = std::max(col - step_size, 0);
  int col_max = std::min(col + step_size, width - 1);

  float sum = 0.;
  #pragma omp parallel for collapse(2) reduction(+ : sum)
  for (int j = row_min; j < row_max; j += 2) {
    for (int i = col_min; i < col_max; i += 2) {
      if (j != row || i != col) sum += old_data->data(j, i);
    }
  }
  return sum * (1. / neighbors);
}

float CameraPipeline::CornerShapeMean(std::unique_ptr<CameraSensorData<CameraSensor::T>>& old_data, int row, int col, int step_x, int step_y) const {
  float corner_sum = 0.;
  if (std::abs(step_x) == 2) {
    corner_sum = old_data->data(row + step_y, col + step_x);
    corner_sum += old_data->data(row + step_y, col);
    corner_sum += old_data->data(row, col + step_x); 
  } else {
    corner_sum = old_data->data(row + step_y, col + step_x);
    corner_sum += old_data->data(row + step_y, col - step_x);
    corner_sum += old_data->data(row - step_y, col + step_x); 
  }

    return corner_sum * 1./3.;
}


void CameraPipeline::FixDefectivePixel(float& pixel, int row, int col, std::unique_ptr<CameraSensorData<CameraSensor::T>>& old_data) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  int step_size = 2;
  int row_min = std::max(row - step_size, 0);
  int row_max = std::min(row + step_size, height - 1);
  int col_min = std::max(col - step_size, 0);
  int col_max = std::min(col + step_size, width - 1);

  float threshold = 0.1;
  float neighbors = 8.;
  if (row_max - row_min == step_size && col_max - col_min == step_size) { // 3 pixel neighbors; 1 shape
    neighbors = (step_size == 1.) ? 1 : 3.;
  } else if (row_max - row_min == step_size || col_max - col_min == step_size || step_size == 1) { // more than 3 pixel neighbors; 4 shapes
    if (step_size == 2) {
      threshold *= 2.;
      neighbors = 5.;     
    } else {
      neighbors = 2.;     
    }
  } else { // more than 6 pixel neighbors; 12 shapes
    if (step_size == 2) {
      threshold *= 3.;
      neighbors = 8.;     
    } else {
      neighbors = 4.;     
    }
  }
  float mean = CalculateNeighborMean(neighbors, row, col, old_data, step_size);
  auto& old_pixel = old_data->data(row, col);

  if (neighbors > 1. && neighbors != 3.) {
    if (neighbors == 8. || neighbors == 4. || row_max - row_min == step_size) {
      // horizontal lines
      for (int j = row_min; j <= row_max; j += 2) {
        float row_sum = 0.;
        float count = 0.;
        for (int i = col_min; i <= col_max; i += 2) {
          row_sum += old_data->data(j, i);
          count++;
        }
        if (!OverThreshold(threshold, row_sum * (1. / count), old_pixel)) {
          return;
        }
        
      }
    }
    if (neighbors == 8. || neighbors == 4. || col_max - col_min == step_size) {
      // vertical lines
      for (int i = col_min; i <= col_max; i += 2) {
        float col_sum = 0.;
        float count = 0.;
        for (int j = row_min; j <= row_max; j += 2) {
          col_sum += old_data->data(j, i);
          count++;
        }
        if (!OverThreshold(threshold, col_sum * (1. / count), old_pixel)) {
          return;
        }
      }
    }
  }

  if (threshold == 0.3) { 
    // diagonals
    float down_diag_sum = 0.;
    float up_diag_sum = 0.;

    for (int step = 0; step <= 2 * step_size; step += step_size) {
      down_diag_sum += old_data->data(row_min + step, col_min + step);
      up_diag_sum += old_data->data(row_min + step, col_max - step);
    }
    if (!OverThreshold(threshold, down_diag_sum * (1. / 3.), old_pixel) || 
        !OverThreshold(threshold, up_diag_sum * (1. / 3.), old_pixel)) {
      return;
    }

    // 4 corners
    for (int step_y = -step_size; step_y <= step_size; step_y += (2 * step_size)) {
      for (int step_x = -step_size; step_x <= step_size; step_x += (2 * step_size)) {
        float corner_mean = CornerShapeMean(old_data, row, col, step_x, step_y);
        if (!OverThreshold(threshold, corner_mean, old_pixel)) {
          return;
        }
      }
    }
  } else if (threshold == 0.2) { 
    // two corners
    if (row_max - row_min == step_size) {
      // horizontal
      int step_y = ((row == row_min) ? step_size : -step_size);
      for (int step_x = -step_size; step_x <= step_size; step_x += (2 * step_size)) {
        float corner_mean = CornerShapeMean(old_data, row, col, step_x, step_y);
        if (!OverThreshold(threshold, corner_mean, old_pixel)) {
          return;
        }
      }
    } else {
      // vertical
      int step_x = ((col == col_min) ? step_size : -step_size);
      for (int step_y = -step_size; step_y <= step_size; step_y += (2 * step_size)) {
        float corner_mean= CornerShapeMean(old_data, row, col, step_x, step_y);
        if (!OverThreshold(threshold, corner_mean, old_pixel)) {
          return;
        }
      }
    }
  } else { 
    // one corner
    int step_y = (row_min == row) ? 2 : -2;
    int step_x = (col_min == col) ? 2 : -2;
    float corner_mean = (neighbors == 3.) ? CornerShapeMean(old_data, row, col, step_x, step_y) : 
                                              old_data->data(row + step_y, col + step_x);
    if (!OverThreshold(threshold, corner_mean, old_pixel)) {
      return;
    }
  }

  pixel = mean;
}


void CameraPipeline::RemoveDefectivePixels(std::unique_ptr<CameraSensorData<CameraSensor::T>>& raw_data) const {

  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight(); 
  std::unique_ptr<CameraSensorData<CameraSensor::T>> old_data = raw_data->Clone();

  # pragma omp parallel num_threads(n_threads)
  {
    // Setting the green value
    #pragma omp for collapse(2) nowait 
    for (int row = 0; row < height; row += 2) {
      for (int col = 0; col < width; col += 2) {
        auto& pixel = raw_data->data(row, col);
        if (pixel > .95) FixDefectivePixel(pixel, row, col, raw_data);
      }
    }
    #pragma omp for collapse(2) nowait 
    for (int row = 1; row < height; row += 2) {
      for (int col = 1; col < width; col += 2) {
        auto& pixel = raw_data->data(row, col);
        if (pixel > .95) FixDefectivePixel(pixel, row, col, raw_data);
      }
    }

    // Setting the red value
    #pragma omp for collapse(2) nowait
    for (int row = 0; row < height; row += 2) {
      for (int col = 1; col < width; col += 2) {
        auto& pixel = raw_data->data(row, col);
        if (pixel > .95) FixDefectivePixel(pixel, row, col, raw_data);
      }
    }  
    // Setting the blue value
    #pragma omp for collapse(2) nowait
    for (int row = 1; row < height; row += 2) {
      for (int col = 0; col < width; col += 2) {
        auto& pixel = raw_data->data(row, col);
        if (pixel > .95) FixDefectivePixel(pixel, row, col, raw_data);               
      }
    }   
  }
}

void CameraPipeline::MedianFilter(std::unique_ptr<Image<RgbPixel>>& image) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  auto old = image->Clone();

  int window_size = 3;
  int step = window_size / 2;

  struct {
      bool operator()(RgbPixel a, RgbPixel b) const { return a.r + a.g + a.b < b.r + b.g + b.b; }
  } customLess;

  # pragma omp parallel for collapse(2)
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      auto& pixel = (*image)(row, col);
      int min_row = std::max(row - step , 0); 
      int min_col = std::max(col - step , 0); 

      int max_row = std::min(row + step, height - 1);
      int max_col = std::min(col + step, width - 1); 

      std::vector<RgbPixel> window;

      for (int j = min_row; j <= max_row; j++) {
        for (int i = min_col ; i <= max_col; i++) {
          window.push_back((*image)(j, i));
        }
      }

      std::sort(window.begin(), window.end(), customLess);
      pixel = window[window.size() / 2];   
    }    
  }
}

void CameraPipeline::AddBlackOffset(std::unique_ptr<Image<RgbPixel>>& image_black, std::unique_ptr<Image<RgbPixel>>& image) const {
  Float3Pixel avg = Float3Pixel();
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();

  #pragma omp parallel for collapse(2) reduction(+ : avg)
    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        avg +=(*image_black)(j, i);
      }
    }
    avg.r = avg.r / ((float) width * height);
    avg.g = avg.g / ((float) width * height);
    avg.b = avg.b / ((float) width * height);

  #pragma omp parallel for collapse(2) 
    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        auto& pixel = (*image)(j, i);
        pixel -= avg;
      }
    }
}

std::unique_ptr<Image<RgbPixel>> CameraPipeline::ConvertToBayerFilterPattern(std::unique_ptr<CameraSensorData<CameraSensor::T>>& raw_data) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight(); 
  std::unique_ptr<Image<RgbPixel>> image(new Image<RgbPixel>(width, height));

  # pragma omp parallel num_threads(n_threads)
  {
    // Setting the green value
    #pragma omp for collapse(2) nowait 
    for (int row = 0; row < height; row += 2) {
      for (int col = 0; col < width; col += 2) {
        const auto val = raw_data->data(row, col);
        auto& pixel = (*image)(row, col);
        pixel.g = val;
      }
    }
    #pragma omp for collapse(2) nowait 
    for (int row = 1; row < height; row += 2) {
      for (int col = 1; col < width; col += 2) {
        const auto val = raw_data->data(row, col);
        auto& pixel = (*image)(row, col);
        pixel.g = val;
      }
    }

    // Setting the red value
    #pragma omp for collapse(2) nowait
    for (int row = 0; row < height; row += 2) {
      for (int col = 1; col < width; col += 2) {
        const auto val = raw_data->data(row, col);
        auto& pixel = (*image)(row, col);
        pixel.r = val;
      }
    }  
    // Setting the blue value
    #pragma omp for collapse(2) nowait
    for (int row = 1; row < height; row += 2) {
      for (int col = 0; col < width; col += 2) {
        const auto val = raw_data->data(row, col);
        auto& pixel = (*image)(row, col);
        pixel.b = val;
      }
    }   
  }

  return image;
}

void CameraPipeline::LinearDemosaik(std::unique_ptr<Image<RgbPixel>>& image) const {
  // TODO: Debug Boundary demosaik border issue
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight(); 
  auto old_image = image->Clone();

  # pragma omp parallel num_threads(n_threads)
  {

    // green
    #pragma omp for collapse(2) nowait
    for (int row = 0; row < height; row += 2) {
      for (int col = 1; col < width; col += 2) {
        float count = 0.;
        float sum = 0.;
        #pragma omp parallel for reduction(+:sum, count)
        for (int step = -1; step <= 1; step += 2) {

          count += ((((col + step) < 0) || ((col + step) > width - 1)) ? 0 : 1);
          count += ((((row + step) < 0) || ((row + step) > height - 1))  ? 0 : 1);

          sum += ((((col + step) < 0) || ((col + step) > width - 1)) ? 0. : (*old_image)(row, col + step).g);
          sum += ((((row + step) < 0) || ((row + step) > height - 1)) ? 0. : (*old_image)(row + step, col).g);
        }

        auto& pixel = (*image)(row, col);
        pixel.g = (sum / count);
      }
    }

    #pragma omp for collapse(2) nowait
    for (int row = 1; row < height; row += 2) {
      for (int col = 0; col < width; col += 2) {
        float count = 0.;
        float sum = 0.;
        #pragma omp parallel for reduction(+:sum, count)
        for (int step = -1; step <= 1; step += 2) {

          count += ((((col + step) < 0) || ((col + step) > width - 1)) ? 0 : 1);
          count += ((((row + step) < 0) || ((row + step) > height - 1))  ? 0 : 1);

          sum += ((((col + step) < 0) || ((col + step) > width - 1)) ? 0. : (*old_image)(row, col + step).g);
          sum += ((((row + step) < 0) || ((row + step) > height - 1)) ? 0. : (*old_image)(row + step, col).g);
        }

        auto& pixel = (*image)(row, col);
        pixel.g = (sum / count);
      }
    }

    // red even rows
    #pragma omp for collapse(2) nowait 
    for (int row = 0; row < height; row += 2) {
      for (int col = 0; col < width; col += 2) {
        float count = 0.;
        float sum = 0.;
        for (int step = -1; step <= 1; step += 2) {
          count += (((col + step) < 0 || (col + step) > width - 1) ? 0 : 1);
          sum += (((col + step) < 0 || (col + step) > width - 1) ? 0. : (*old_image)(row, col + step).r);          
        }
        auto& pixel = (*image)(row, col);
        pixel.r = sum / count;
      }
    }
    // red odd rows
    #pragma omp for collapse(2) nowait 
    for (int row = 1; row < height; row += 2) {
      for (int col = 0; col < width; col++) {
        float count = 0.;
        float sum = 0.;
        if (col % 2 == 1) {
          // green looks at top and bottom
          for (int step = -1; step <= 1; step += 2) {
            count += (((row + step) < 0 || (row + step) > height - 1) ? 0 : 1);
            sum += (((row + step) < 0 || (row + step) > height - 1) ? 0. : (*old_image)(row + step, col).r);  
          }
        } else {
          // blue looks at diagonals
          for (int step_y = -1; step_y <= 1; step_y += 2) {
            for (int step_x = -1; step_x <= 1; step_x += 2) {
              count += (((col + step_x) < 0 || (col + step_x) > width - 1 || (row + step_y) < 0 || (row + step_y) > height - 1) ? 0. : 1.);
              sum += (((col + step_x) < 0 || (col + step_x) > width - 1 || (row + step_y) < 0 || (row + step_y) > height - 1) ? 0. : (*old_image)(row + step_y, col + step_x).r); 
            }        
          }
        }
        auto& pixel = (*image)(row, col);
        pixel.r = sum / count;
      }
    }

    // blue odd rows
    #pragma omp for collapse(2) nowait 
    for (int row = 1; row < height; row += 2) {
      for (int col = 1; col < width; col += 2) {
        float count = 0.;
        float sum = 0.;
        for (int step = -1; step <= 1; step += 2) {
          count += (((col + step) < 0 || (col + step) > width - 1) ? 0 : 1);
          sum += (((col + step) < 0 || (col + step) > width - 1) ? 0. : (*old_image)(row, col + step).b);          
        }
        auto& pixel = (*image)(row, col);
        pixel.b = sum / count;
      }
    }
    // blue even rows
    #pragma omp for collapse(2) nowait 
    for (int row = 0; row < height; row += 2) {
      for (int col = 0; col < width; col++) {
        float count = 0.;
        float sum = 0.;

        if (col % 2 == 0) {
          // green looks at top and bottom
          for (int step = -1; step <= 1; step += 2) {
            count += (((row + step) < 0 || (row + step) > height - 1) ? 0 : 1);
            sum += (((row + step) < 0 || (row + step) > height - 1) ? 0. : (*old_image)(row + step, col).b);  
          }
        } else {
          // blue looks at diagonals
          for (int step_y = -1; step_y <= 1; step_y += 2) {
            for (int step_x = -1; step_x <= 1; step_x += 2) {
              count += (((col + step_x) < 0 || (col + step_x) > width - 1 || (row + step_y) < 0 || (row + step_y) > height - 1) ? 0. : 1.);
              sum += (((col + step_x) < 0 || (col + step_x) > width - 1 || (row + step_y) < 0 || (row + step_y) > height - 1) ? 0. : (*old_image)(row + step_y, col + step_x).b); 
            }        
          }
        }
        auto& pixel = (*image)(row, col);
        pixel.b = sum / count;
      }
    }
  }
}

void CameraPipeline::ConvertToYCbCr(std::unique_ptr<Image<RgbPixel>>& image) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  #pragma omp parallel for collapse(2)
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      auto& pixel = (*image)(row, col);
      pixel = Float3Pixel::RgbToYuv((*image)(row, col));
    }
  }
}


void CameraPipeline::LowPassCbCr(std::unique_ptr<Image<YuvPixel>>& image) const {
  int window_size = 5;
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  std::unique_ptr<Image<YuvPixel>> old = image->Clone();

  #pragma omp parallel for collapse(2)
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      auto& pixel = (*image)(row, col);

      int step = window_size / 2;
      int min_col = std::clamp(col - step, 0, col - step);
      int max_col = std::clamp(col + step, col + step, width - 1);
      int min_row = std::clamp(row - step, 0, row - step);
      int max_row = std::clamp(row + step, row + step, height - 1);
      float count = 0.;
      float sum_u = 0.;
      float sum_v = 0.;

      for (int j = min_row; j <= max_row; j++) {
        for (int i = min_col; i <= max_col; i++) {
          count += 1.;
          sum_u += (*old)(j, i).u;
          sum_v += (*old)(j, i).v;
        }      
      }
      pixel.u = sum_u / count;
      pixel.v = sum_v / count;
    }
  }
}

void CameraPipeline::ConvertToRGB(std::unique_ptr<Image<RgbPixel>>& image) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  #pragma omp parallel for collapse(2)
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      auto& pixel = (*image)(row, col);
      pixel = Float3Pixel::YuvToRgb((*image)(row, col));
    }
  }
}

// *********************************************************************************
//                              PART 2 HELPERS
// *********************************************************************************
static constexpr float sigma = 0.3;
static constexpr float offset = 0.75;
static constexpr float bright_gain = 2.3;
static constexpr float dark_gain = 1.01;

static inline float WellExposedness(float y) {
  return std::exp(-std::pow((y - offset), 2) / (2. * std::pow(sigma, 2)) );
}

void CameraPipeline::CreateExposureBrackets(std::unique_ptr<Image<YuvPixel>>& dark, std::unique_ptr<Image<YuvPixel>>& bright) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  #pragma omp parallel for collapse(2)
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      auto& bright_pixel = (*bright)(row, col);
      bright_pixel.y *= bright_gain;
      auto& dark_pixel = (*dark)(row, col);
      dark_pixel.y *= dark_gain;
    }   
  }
}

void CameraPipeline::ConvertToLuma(std::unique_ptr<Image<YuvPixel>>& dark, std::unique_ptr<Image<YuvPixel>>& bright) const {
  ConvertToRGB(dark);
  dark->GammaCorrect(0.55);
  MedianFilter(dark);
  ConvertToYCbCr(dark);
  // LowPassCbCr(dark);

  ConvertToRGB(bright);
  bright->GammaCorrect(0.45);
  MedianFilter(bright);
  ConvertToYCbCr(bright);

  #pragma omp parallel for collapse(2)
  for (int row = 0; row < bright->height(); row++) {
    for (int col = 0; col < bright->width(); col++) {
      auto& pixel = (*bright)(row, col);
      pixel.u = (*dark)(row, col).u;
      pixel.v = (*dark)(row, col).v;
    }
  }
}

void CameraPipeline::ComputeWeights(std::unique_ptr<Image<Float3Pixel>>& dark_weights, std::unique_ptr<Image<Float3Pixel>>& bright_weights) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();

  #pragma omp parallel for collapse(2)
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      float dark_weight = WellExposedness((*dark_weights)(row, col).y);
      float bright_weight = WellExposedness((*bright_weights)(row, col).y); 
      dark_weight /= (dark_weight + bright_weight);
      bright_weight = 1. - dark_weight;

      (*dark_weights)(row, col) = Float3Pixel(dark_weight, dark_weight, dark_weight);
      (*bright_weights)(row, col) = Float3Pixel(bright_weight, bright_weight, bright_weight);
    }   
  }
}

std::unique_ptr<Image<RgbPixel>> CameraPipeline::Downsample(std::unique_ptr<Image<RgbPixel>>& image) const {
  auto width = image->width();
  auto height = image->height();
  std::unique_ptr<Image<RgbPixel>> output_image(new Image<RgbPixel>((int)std::ceil((float) width/2), (int)std::ceil((float) height/2)));

  # pragma omp parallel for collapse(2)
  for (int row = 0; row < image->height(); row += 2) {
    for (int col = 0; col < image->width(); col += 2) {
      (*output_image)(row/2, col/2) = (*image)(row, col);
    }    
  }
  return output_image;
}

std::unique_ptr<Image<RgbPixel>> CameraPipeline::Upsample(std::unique_ptr<Image<RgbPixel>>& image, int orig_width, int orig_height) const {
  std::unique_ptr<Image<RgbPixel>> output_image(new Image<RgbPixel>(orig_width, orig_height));

  # pragma omp parallel for collapse(2)
  for (int row = 0; row < image->height(); row++) {
    for (int col = 0; col < image->width(); col++) {
      (*output_image)(row * 2, col * 2) = (*image)(row, col);
    }    
  }
   # pragma omp parallel for collapse(2)
  for (int row = 0; row < orig_height; row+=2) {
    for (int col = 1; col < orig_width - 1; col+=2) {
      (*output_image)(row, col) = ((*output_image)(row, col - 1) * .5) + ((*output_image)(row, col + 1) * .5);
    }    
  }
   # pragma omp parallel for collapse(2)
  for (int row = 1; row < orig_height - 1; row+=2) {
    for (int col = 0; col < orig_width; col+=2) {
      (*output_image)(row, col) = ((*output_image)(row - 1, col) * .5) + ((*output_image)(row + 1, col) * .5);
    }    
  }
  # pragma omp parallel for collapse(2)
  for (int row = 1; row < orig_height - 1; row+=2) {
    for (int col = 1; col < orig_width - 1; col+=2) {
      (*output_image)(row, col) = ((*output_image)(row - 1, col) * .25) + ((*output_image)(row + 1, col) * .25f);
      (*output_image)(row, col) += ((*output_image)(row, col - 1) * .25) + ((*output_image)(row, col + 1) * .25f);
    }    
  }
  if (orig_height % 2 == 0) {
    # pragma omp parallel for
    for (int col = 1; col < orig_width - 1; col++) {
        (*output_image)(orig_height - 1, col) = (*output_image)(orig_height - 2, col); 
    }  
    (*output_image)(orig_height - 1, 0) = ((*output_image)(orig_height - 2, 0) * .5) + ((*output_image)(orig_height - 1, 1) * .5);
    (*output_image)(orig_height - 1, orig_width - 1) = ((*output_image)(orig_height - 2, orig_width - 1) * .5) + ((*output_image)(orig_height - 1, orig_width - 2) * .5);
  }
  if (orig_width % 2 == 0) {
    # pragma omp parallel for 
    for (int row = 1; row < orig_height - 1; row++) {
        (*output_image)(row, orig_width - 1) = (*output_image)(row, orig_width - 2); 
    } 
    (*output_image)(orig_height - 1, 0) = ((*output_image)(orig_height - 2, 0) * .5) + ((*output_image)(orig_height - 1, 1) * .5);
    (*output_image)(orig_height - 1, orig_width - 1) = ((*output_image)(orig_height - 2, orig_width - 1) * .5) + ((*output_image)(orig_height - 1, orig_width - 2) * .5);
  }  
  return output_image;
}

void CameraPipeline::Subtract(std::unique_ptr<Image<RgbPixel>>& left, std::unique_ptr<Image<RgbPixel>>& right) const {
  # pragma omp parallel for collapse(2)
  for (int row = 0; row < left->height(); row ++) {
    for (int col = 0; col < left->width(); col ++) {
      auto& pixel = (*right)(row, col);
      pixel = (*left)(row, col) - (*right)(row, col);
    }    
  }
}

std::unique_ptr<Image<RgbPixel>> CameraPipeline::Add(std::unique_ptr<Image<RgbPixel>>& left, std::unique_ptr<Image<RgbPixel>>& right) const {
  std::unique_ptr<Image<RgbPixel>> output(new Image<RgbPixel>(left->width(), right->height()));
  # pragma omp parallel for collapse(2)
  for (int row = 0; row < left->height(); row ++) {
    for (int col = 0; col < left->width(); col ++) {
      auto& pixel = (*output)(row, col);
      pixel = (*left)(row, col) + (*right)(row, col);
    }    
  }
  return output;
}

void CameraPipeline::Lerp(std::unique_ptr<Image<RgbPixel>>& left, std::unique_ptr<Image<RgbPixel>>& right,
                                                      std::unique_ptr<Image<RgbPixel>>& left_weights, std::unique_ptr<Image<RgbPixel>>& right_weights) const {
  auto width = left->width();
  auto height = left->height();

  # pragma omp parallel for collapse(2)
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      auto& left_pixel = (*left)(row, col);
      auto& left_pixel_w = (*left_weights)(row, col);
      auto& right_pixel = (*right)(row, col);
      auto& right_pixel_w = (*right_weights)(row, col);
      auto& output_pixel = (*left)(row, col);
      output_pixel.r = (left_pixel.r * left_pixel_w.r) + (right_pixel.r * right_pixel_w.r);
      output_pixel.g = (left_pixel.g * left_pixel_w.g) + (right_pixel.g * right_pixel_w.g);
      output_pixel.b = (left_pixel.b * left_pixel_w.b) + (right_pixel.b * right_pixel_w.b);
    }    
  }
}

std::unique_ptr<Image<RgbPixel>> CameraPipeline::GaussianBlur(std::unique_ptr<Image<RgbPixel>>& image) const {
  auto width = image->width();
  auto height = image->height();
  std::unique_ptr<Image<RgbPixel>> output_image = image->Clone();

  int window_size = 5;
  const float k = 1./256;
  const float weights [5][5] = {  {static_cast<float>(k*1.),   static_cast<float>(k*4.),  static_cast<float>(k*6.),   static_cast<float>(k* 4.), static_cast<float>(k* 1.)},
                                  {static_cast<float>(k*4.),  static_cast<float>(k*16.), static_cast<float>(k*24.),  static_cast<float>(k* 16.), static_cast<float>(k* 4.)},
                                  {static_cast<float>(k*6.),  static_cast<float>(k*24.), static_cast<float>(k*36.),  static_cast<float>(k* 24.), static_cast<float>(k* 6.)},
                                  {static_cast<float>(k*4.),  static_cast<float>(k*16.), static_cast<float>(k*24.),  static_cast<float>(k* 16.), static_cast<float>(k* 4.)},
                                  {static_cast<float>(k*1.),   static_cast<float>(k*4.),  static_cast<float>(k*6.),   static_cast<float>(k* 4.), static_cast<float>(k* 1.)}
                                };

  # pragma omp parallel for collapse(2)
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int half = window_size / 2;
      int min_row = std::max(row - half , 0); 
      int min_col = std::max(col - half , 0); 
      int max_row = std::min(row + half, height - 1);
      int max_col = std::min(col + half, width - 1); 

      auto& pixel = (*output_image)(row, col);
      RgbPixel sum = RgbPixel(0.f, 0.f, 0.f);

      if (max_col - min_row == window_size - 1 && max_col - min_row == window_size - 1) {
        for (int j = -half; j < half; j++) {
          for (int i = -half; i < half; i++) {
            sum += ((*image)(row + j, col + i) * weights[j + half][i + half]);
          }
        }  
      } else {
        float count = 0.;
        for (int j = min_row; j <= max_row; ++j) {
          for (int i = min_col; i <= max_col; ++i) {
              count++;
              sum += (*image)(j, i);
          }
          pixel = sum * (1.f/count);
        }
      }    
    }
  }
  return output_image;
}

std::unique_ptr<Image<Float3Pixel>> CameraPipeline::GaussianPyramidStep(std::unique_ptr<Image<Float3Pixel>>& layer) const {
  std::unique_ptr<Image<Float3Pixel>> tmp = GaussianBlur(layer);
  return Downsample(tmp);
}

// *********************************************************************************
//                              PART 3 HELPERS
// *********************************************************************************

static constexpr int align_pyramid_depth = 4;
static constexpr int displace_radius = 2;
static constexpr int max_displacement = 64;
static constexpr int tile_size = 16;
static constexpr int stride = 8;
static constexpr float lower_bound = 0.2;
static constexpr float upper_bound = 6.;

std::unique_ptr<Image<RgbPixel>> CameraPipeline::AvgBayerGrid(std::unique_ptr<Image<RgbPixel>>& bayer_grid) const {
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  std::unique_ptr<Image<RgbPixel>> output_image(new Image<RgbPixel>(width / 2, height / 2));
 
  # pragma omp parallel for collapse(2)
  for (int row = 0; row < height - 1; row += 2) {
    for (int col = 0; col < width - 1; col += 2) {
      RgbPixel val = RgbPixel(0.f, 0.f, 0.f);
      val.r = (*bayer_grid)(row, col + 1).r; 
      val.g = ((*bayer_grid)(row, col) * 0.5).g + ((*bayer_grid)(row, col) * 0.5).g; 
      val.b = (*bayer_grid)(row + 1, col).b; 
      val = Float3Pixel::RgbToYuv(val);

      auto& output_pixel = (*output_image)(row/2, col/2); 
      output_pixel = RgbPixel(val.y, val.y, val.y);
    }    
  }
  return output_image;
}

void CameraPipeline::ConvertToGrayScale(std::unique_ptr<Image<RgbPixel>>& image) const {
  ConvertToYCbCr(image);
  for (int row = 0; row < image->height(); row++) {
    for (int col = 0; col < image->width(); col++) {
      auto& pixel = (*image)(row, col);
      pixel.u = pixel.y;
      pixel.v = pixel.y;
    }
  }
}

std::vector<std::vector<int>> CameraPipeline::SplitIntoTiles(std::unique_ptr<Image<RgbPixel>>& image, int& height, int& width) const {
  std::vector<std::vector<int>> tiles;
  for (int row = 0; row < image->height(); row+=stride) {
    for (int col = 0; col < image->width(); col+=stride) {
      if (row == 0) width++;

      std::vector<int> bounds(4);
      bounds[0] = row;
      bounds[1] = col;
      bounds[2] = std::min(row*tile_size, image->height());
      bounds[3] = std::min(col*tile_size, image->width());
      tiles.push_back(bounds);
    }
    height++;
  }
  return tiles;
}

ImgVec CameraPipeline::ConstructTiles(std::unique_ptr<Image<RgbPixel>>& image, std::vector<std::vector<int>> tile_idxs) const {
  ImgVec merged_tiles(tile_idxs.size());

  # pragma omp parallel for 
  for (int tile_num = 0; tile_num < tile_idxs.size(); tile_num++) {
    std::unique_ptr<Image<RgbPixel>> tile(new Image<RgbPixel>(tile_size, tile_size));
    merged_tiles[tile_num] = std::move(tile);
    for (int j = 0; j < tile_size; j++) {
      for (int i = 0; i < tile_size; i++) {
        int row = tile_idxs[tile_num][0] + j;
        int col = tile_idxs[tile_num][1] + i;
        if (row < image->height() && col < image->width()) {
          (*merged_tiles[tile_num])(j, i) = (*image)(row, col);
        } else {
          (*merged_tiles[tile_num])(j, i) = RgbPixel(0.f, 0.f, 0.f); 
        }
      }
    }
  } 
  return merged_tiles;
}

float CameraPipeline::L1Residual(std::unique_ptr<Image<RgbPixel>>& ref_image, int start_row, int start_col, 
                std::unique_ptr<Image<RgbPixel>>& burst_gaussian, int row, int col) const {
  float residual = 0.;
  for (int j = 0; j < tile_size; j++) {
    for (int i = 0; i < tile_size; i++) {
      if (start_row + j < ref_image->height() && start_col + i < ref_image->width() && 
          row + j < burst_gaussian->height() && col + i < burst_gaussian->width()) {
        residual += std::abs((*ref_image)(start_row + j, start_col + i).r - (*burst_gaussian)(row + j, col + i).r);
        residual += std::abs((*ref_image)(start_row + j, start_col + i).g - (*burst_gaussian)(row + j, col + i).g);
        residual += std::abs((*ref_image)(start_row + j, start_col + i).b - (*burst_gaussian)(row + j, col + i).b);
      }
    }
  }
  return residual;
}

float CameraPipeline::L2Residual(std::unique_ptr<Image<RgbPixel>>& ref_image, int start_row, int start_col, 
                std::unique_ptr<Image<RgbPixel>>& burst_gaussian, int row, int col) const {
  float residual = 0.;
  for (int j = 0; j < tile_size; j++) {
    for (int i = 0; i < tile_size; i++) {
      if (start_row + j < ref_image->height() && start_col + i < ref_image->width() && 
          row + j < burst_gaussian->height() && col + i < burst_gaussian->width()) {
        residual += std::pow(std::abs((*ref_image)(start_row + j, start_col + i).r - (*burst_gaussian)(row + j, col + i).r), 2);
        residual += std::pow(std::abs((*ref_image)(start_row + j, start_col + i).g - (*burst_gaussian)(row + j, col + i).g), 2);
        residual += std::pow(std::abs((*ref_image)(start_row + j, start_col + i).b - (*burst_gaussian)(row + j, col + i).b), 2);
      }
    }
  }
  return residual;
}

static inline float RaisedCosine(float num) {
  return 0.5 + (0.5 * cos(2 * M_PI * num / (float) tile_size));
}

float CameraPipeline::GetPixelCosineWeight(int row, int col) const {
  float num = std::sqrt(std::pow((float) std::abs(row - tile_size / 2), 2) + std::pow((float) std::abs(col - tile_size / 2), 2));
  return RaisedCosine(num);
}

void CameraPipeline::GetRaisedCosine(std::unique_ptr<Image<FloatPixel>>& tile_weight,
                                      int tile_row, int tile_col, int ref_height_tiles, int ref_width_tiles) const {
  if (tile_row > 0) {
    // just do the tops
    if (tile_col > 0) {
      for (int row = 0; row < tile_size / 2; row++) {
        for (int col = 0; col < tile_size / 2; col++) {
          (*tile_weight)(row, col) = FloatPixel(GetPixelCosineWeight(row, col));
        }            
      }
    }

    if (tile_col < ref_width_tiles - 1) {
      for (int row = 0; row < tile_size / 2; row++) {
        for (int col = tile_size / 2; col < tile_size; col++) {
          (*tile_weight)(row, col) = FloatPixel(GetPixelCosineWeight(row, col));
        }            
      }  
    } 
  }
  if (tile_row < ref_height_tiles - 1) {
    // just do the bottoms first 
    if (tile_col > 0) {
      for (int row = tile_size / 2; row < tile_size; row++) {
        for (int col = 0; col < tile_size / 2; col++) {
          (*tile_weight)(row, col) = FloatPixel(GetPixelCosineWeight(row, col));
        }            
      } 
    }
    if (tile_col < ref_width_tiles - 1) {
      for (int row = tile_size / 2; row < tile_size; row++) {
        for (int col = tile_size / 2; col < tile_size; col++) {
          (*tile_weight)(row, col) = FloatPixel(GetPixelCosineWeight(row, col));
        }            
      } 
    }   
  }
}

// *********************************************************************************
//                                PIPELINE CODE
// *********************************************************************************

//        PART 1 (30 points): Basic Camera RAW Pipeline
std::unique_ptr<Image<RgbPixel>> CameraPipeline::BasicCameraRAWPipeline(std::unique_ptr<CameraSensorData<CameraSensor::T>>& raw_data) const {
  RemoveDefectivePixels(raw_data); 
  std::unique_ptr<Image<RgbPixel>> image = ConvertToBayerFilterPattern(raw_data);
  LinearDemosaik(image);
  return image;
}

//        PART 2 (70 points): Local Tone Mapping + Burst Mode Alignment for Denoising
void CameraPipeline::ExposureFusion(std::unique_ptr<Image<RgbPixel>>& image) const {
  // step 1
  ConvertToYCbCr(image);
  auto dark = image->Clone();
  auto bright = image->Clone();

  // step 2
  CreateExposureBrackets(dark, bright);

  // step 3: section 6 using luma
  std::unique_ptr<Image<Float3Pixel>> dark_weights = dark->Clone();
  std::unique_ptr<Image<Float3Pixel>> bright_weights = bright->Clone();
  ConvertToLuma(dark, bright);
  ComputeWeights(dark_weights, bright_weights);
  ConvertToRGB(dark);
  ConvertToRGB(bright);

  // step 4: construct pyramids
  ImgVec dark_img_laplacian(depth);
  ImgVec bright_img_laplacian(depth);
  ImgVec dark_img_gaussian(depth + 1);
  ImgVec bright_img_gaussian(depth + 1);
  ImgVec dark_w_gaussian(depth + 1);
  ImgVec bright_w_gaussian(depth + 1);

  dark_w_gaussian[0] = std::move(dark_weights);
  bright_w_gaussian[0] = std::move(bright_weights);
  dark_img_gaussian[0] = std::move(dark);
  bright_img_gaussian[0] = std::move(bright);

  for (int n = 0; n < depth; n++) {
    // Gaussian step
    dark_w_gaussian[n+1] = GaussianPyramidStep(dark_w_gaussian[n]);
    bright_w_gaussian[n+1] = GaussianPyramidStep(bright_w_gaussian[n]);

    bright_img_gaussian[n+1] = GaussianPyramidStep(bright_img_gaussian[n]);
    std::unique_ptr<Image<RgbPixel>> bright_upsample = Upsample(bright_img_gaussian[n+1], bright_img_gaussian[n]->width(), bright_img_gaussian[n]->height());
    Subtract(bright_img_gaussian[n], bright_upsample);
    bright_img_laplacian[n] = std::move(bright_upsample);

    dark_img_gaussian[n+1] = GaussianPyramidStep(dark_img_gaussian[n]);
    std::unique_ptr<Image<RgbPixel>> dark_upsample = Upsample(dark_img_gaussian[n+1], dark_img_gaussian[n]->width(), dark_img_gaussian[n]->height());
    Subtract(dark_img_gaussian[n], dark_upsample);
    dark_img_laplacian[n] = std::move(dark_upsample);
  }

  // step 5: section 3.2
  ImgVec blended_laplacian(depth);
  ImgVec blended_gaussian(depth);
  ImgVec blended_image(depth);

  # pragma omp parallel for
  for (int n = 0; n < depth; n++) {
    std::unique_ptr<Image<RgbPixel>> blendLevel = std::move(dark_img_laplacian[n]); 
    Lerp(blendLevel, bright_img_laplacian[n], dark_w_gaussian[n], bright_w_gaussian[n]);
    blended_laplacian[n] = std::move(blendLevel);

    std::unique_ptr<Image<RgbPixel>> blendGaussian = std::move(dark_img_gaussian[n]); 
    Lerp(blendGaussian, bright_img_gaussian[n], dark_w_gaussian[n], bright_w_gaussian[n]);
    blended_gaussian[n] = std::move(blendGaussian);
  }

  // Step 6
  # pragma omp parallel for
  for (int n = depth - 1; n > 0; n--) {
    std::unique_ptr<Image<RgbPixel>> upsampleLapl = Upsample(blended_laplacian[n], blended_laplacian[n-1]->width(), blended_laplacian[n-1]->height());
    std::unique_ptr<Image<RgbPixel>> upsampleGauss = Upsample(blended_gaussian[n], blended_gaussian[n-1]->width(), blended_gaussian[n-1]->height());
   
    blended_image[n-1] = Add(upsampleLapl, upsampleGauss);
  }
  image = std::move(blended_image[0]);
}

std::unique_ptr<Image<RgbPixel>> CameraPipeline::AlignAndMerge(std::unique_ptr<CameraSensorData<CameraSensor::T>>& raw_data, std::vector<std::unique_ptr<CameraSensorData<CameraSensor::T>>>& burst_raw_data) const {

  ImgVec burst_images_gray(burst_raw_data.size());
  // ALIGN STEP
  // step 1: bayer processing
  RemoveDefectivePixels(raw_data);
  std::unique_ptr<Image<RgbPixel>> ref_bayer = ConvertToBayerFilterPattern(raw_data);
  std::unique_ptr<Image<RgbPixel>> avg_bayer = AvgBayerGrid(ref_bayer);

  ImgVec burst_bayer(burst_raw_data.size());
  # pragma omp parallel for
  for (int n = 0; n < burst_raw_data.size(); n++) {
    RemoveDefectivePixels(burst_raw_data[n]);
    burst_bayer[n] = ConvertToBayerFilterPattern(burst_raw_data[n]);
    std::unique_ptr<Image<RgbPixel>> avg_bayer = AvgBayerGrid(burst_bayer[n]);
    burst_images_gray[n] = std::move(avg_bayer);
  }

  // step 2
  // reference image gaussian pyramid
  ImgVec ref_image_gaussian_pyramid(align_pyramid_depth + 1);
  ref_image_gaussian_pyramid[0] = std::move(avg_bayer);
  for (int n = 0; n < align_pyramid_depth - 1; n++) {
    ref_image_gaussian_pyramid[n+1] = GaussianPyramidStep(ref_image_gaussian_pyramid[n]);
  }

  // burst images gaussian pyramids
  std::vector<ImgVec> burst_gaussian_pyramids(burst_raw_data.size());
  # pragma omp parallel for
  for (int n = 0; n < burst_raw_data.size(); n++) {
    ImgVec gaussian(align_pyramid_depth);
    gaussian[0] = std::move(burst_images_gray[n]);
    for (int m = 0; m < align_pyramid_depth - 1; m++) {
      gaussian[m+1] = GaussianPyramidStep(gaussian[m]);
    }
    burst_gaussian_pyramids[n] = std::move(gaussian);
  }

  // figure out tile locations beforehand across all burst images

  PyramidTilesVec pyramid_tiles(align_pyramid_depth);
  BurstPyramidTileDisplaceVec burst_displace_u(burst_raw_data.size(), std::vector<std::vector<int>>(align_pyramid_depth));
  BurstPyramidTileDisplaceVec burst_displace_v(burst_raw_data.size(), std::vector<std::vector<int>>(align_pyramid_depth));
  std::vector<int> width_tiles(align_pyramid_depth, 0);
  std::vector<int> height_tiles(align_pyramid_depth, 0);
  BurstPyramidTileDisplaceVec burst_final_displace_u(burst_raw_data.size(), std::vector<std::vector<int>>(align_pyramid_depth));
  BurstPyramidTileDisplaceVec burst_final_displace_v(burst_raw_data.size(), std::vector<std::vector<int>>(align_pyramid_depth));


  for (int n = 0; n < align_pyramid_depth; n++) {
    std::vector<std::vector<int>> tiles = SplitIntoTiles(ref_image_gaussian_pyramid[n], height_tiles[n], width_tiles[n]);
    for (int m = 0; m < burst_raw_data.size(); m++) {
      burst_displace_u[m][n] = std::vector<int>(tiles.size());
      burst_displace_v[m][n] = std::vector<int>(tiles.size());
      burst_final_displace_u[m][n] = std::vector<int>(tiles.size());
      burst_final_displace_v[m][n] = std::vector<int>(tiles.size());
    }
    pyramid_tiles[n] = std::move(tiles);
  }

  // step 3 (4 - 6)
  std::cout << "Sorry I take a long time to perform alignment even though I am multi-threaded." << std::endl;
  # pragma omp parallel num_threads(burst_raw_data.size())
  {
    for (int n = 0; n < burst_raw_data.size(); n++) {
      // go from smallest levels in pyramid back up
      for (int m = align_pyramid_depth - 1; m >= 0; m--) {
        // divide into tiles
        int height = ref_image_gaussian_pyramid[m]->height();
        int width = ref_image_gaussian_pyramid[m]->width();

        # pragma omp parallel for
        for (int tile_y = 0; tile_y < height_tiles[m]; tile_y++) {
          for (int tile_x = 0; tile_x < width_tiles[m]; tile_x++) {
            auto tile = pyramid_tiles[m][tile_y * width_tiles[m] + tile_x];
            int start_row = tile[0];
            int start_col = tile[1];
            int coarser_tile_y = tile_y / 2;
            int coarser_tile_x = tile_x / 2;

            int coarser_displace_u = (m == align_pyramid_depth - 1)? 0 : burst_displace_u[n][m+1][(coarser_tile_y * width_tiles[m+1]) + coarser_tile_x];
            int coarser_displace_v = (m == align_pyramid_depth - 1)? 0 : burst_displace_v[n][m+1][(coarser_tile_y * width_tiles[m+1]) + coarser_tile_x];
            int search_row_start = std::max(start_row - displace_radius + (coarser_displace_u * 2), 0);
            int search_col_start = std::max(start_col - displace_radius + (coarser_displace_v * 2), 0);
       
            float best_distance = std::numeric_limits<float>::max();
            int displace_row = 0;
            int displace_col = 0;

            std::mutex distance_mutex;

            # pragma omp parallel for shared(distance_mutex) collapse(2) 
            for (int row = search_row_start; row <= search_row_start + displace_radius; row++) {
              for (int col = search_col_start; col <= search_col_start + displace_radius; col++) {
                if (row < height && col < width) {
                  float distance = (m == 0)? L1Residual(ref_image_gaussian_pyramid[m], start_row, start_col, burst_gaussian_pyramids[n][m], row, col) : 
                                            L2Residual(ref_image_gaussian_pyramid[m], start_row, start_col, burst_gaussian_pyramids[n][m], row, col);
                  distance_mutex.lock();
                  if (distance < best_distance) {
                    best_distance = distance;
                    displace_row = row - start_row;
                    displace_col = col - start_col;
                  }
                  distance_mutex.unlock();
                }
              }
            }
            burst_displace_u[n][m][(tile_y * width_tiles[m]) + tile_x] = displace_row;
            burst_displace_v[n][m][(tile_y * width_tiles[m]) + tile_x] = displace_col;    
          }
        }
      }
    }
  }

  // MERGE STEP 
  std::cout << "Finished aligning. Now merging. Thanks for your patience :) " << std::endl;
  int ref_width_tiles = 0;
  int ref_height_tiles = 0;
  std::vector<std::vector<int>> ref_tiles = SplitIntoTiles(ref_bayer, ref_height_tiles, ref_width_tiles);
  ImgVec merged_tiles = ConstructTiles(ref_bayer, ref_tiles);
  WeightVec weight_tiles(ref_tiles.size());

  for (int tile_y = 0; tile_y < ref_height_tiles; tile_y++) {
    for (int tile_x = 0; tile_x < ref_width_tiles; tile_x++) {
      int tile_idx = tile_y * ref_width_tiles + tile_x;
      auto& tile = merged_tiles[tile_idx];
      for (int n = 0; n < burst_raw_data.size(); n++) {
        int start_row = ref_tiles[tile_idx][0];
        int start_col = ref_tiles[tile_idx][1];
        int coarser_tile_y = tile_y / 2;
        int coarser_tile_x = tile_x / 2;

        // step 2 get alignment offset
        int offset_y =  2. * burst_displace_u[n][0][(coarser_tile_y * width_tiles[0]) + coarser_tile_x];
        int offset_x = 2. * burst_displace_v[n][0][(coarser_tile_y * width_tiles[0]) + coarser_tile_x];

        // step 3 compute weight
        float burst_weight = 0.;
        if (std::abs(offset_x) < max_displacement && std::abs(offset_y) < max_displacement) {
          float initial_weight = L1Residual(tile, 0, 0, burst_bayer[n], start_row + offset_y, start_col + offset_x);
          burst_weight = std::clamp(initial_weight, lower_bound, upper_bound);
          burst_weight = 1. - ((burst_weight - lower_bound) / (upper_bound - lower_bound));
        }

        // step 4 merge into tile
        for (int j = 0; j < tile_size; j++) {
          for (int i = 0; i < tile_size; i++) {
            if (start_row + j < ref_bayer->height() && start_col + i < ref_bayer->width() && 
                start_row + offset_y + j < ref_bayer->height() && start_col + offset_x + i < ref_bayer->width() &&
                start_row + offset_y + j >= 0 && start_col + offset_x + i >= 0) {
              auto& ref_pixel = (*tile)(j, i);
              auto burst_pixel = (*burst_bayer[n])(start_row + offset_y + j, start_col + offset_x + i);
              ref_pixel = (ref_pixel * (1. - burst_weight)) + (burst_pixel * burst_weight);
            }
          }
        }
      }
      std::unique_ptr<Image<FloatPixel>> cos_weights(new Image<FloatPixel>(tile_size, tile_size));
      GetRaisedCosine(cos_weights, tile_y, tile_x, ref_height_tiles, ref_width_tiles);
      weight_tiles[tile_idx] = std::move(cos_weights);
    }
  }

  // blend together overlapping tiles
  std::unique_ptr<Image<RgbPixel>> output(new Image<RgbPixel>(ref_bayer->width(), ref_bayer->height()));
  std::unique_ptr<Image<FloatPixel>> weights(new Image<FloatPixel>(ref_bayer->width(), ref_bayer->height()));

  # pragma omp parallel for collapse(2)
  for (int row = 0; row < output->height(); row++) {
    for (int col = 0; col < output->width(); col++) {
      (*output)(row, col) = RgbPixel(0.f, 0.f, 0.f);
    }
  }
  // add weights to normalize
  int tile_x = 0;
  int tile_y = 0;
  for (int row = 0; row < output->height(); row += tile_size / 2) {
    for (int col = 0; col < output->width(); col += tile_size / 2) {

      for (int j = 0; j < tile_size; j++) {
        for (int i = 0; i < tile_size; i++) {
          int tile_idx = (tile_y * ref_width_tiles) + tile_x;

          if (row + j < output->height() && col + i < output->width()) {
            (*weights)(row + j, col + i) += ((*weight_tiles[tile_idx])(j, i).i);
          }
        }
      }
      tile_x++;
    }
    tile_y++;
    tile_x = 0;
  }
  tile_x = 0;
  tile_y = 0;
  for (int row = 0; row < output->height(); row += tile_size / 2) {
    for (int col = 0; col < output->width(); col += tile_size /2) {

      for (int j = 0; j < tile_size; j++) {
        for (int i = 0; i < tile_size; i++) {
          int tile_idx = (tile_y * ref_width_tiles) + tile_x;
          if (row + j < output->height() && col + i < output->width()) {
            if ((*weights)(row + j, col + i).i == 0.) {
              (*output)(row + j, col + i) = (*ref_bayer)(row + j, col + i);
            } else {
              (*output)(row + j, col + i) += (*merged_tiles[tile_idx])(j, i) * (((*weight_tiles[tile_idx])(j, i).i ) / (*weights)(row + j, col + i).i);
            }
          }
        }
      }
      tile_x++;
    }
    tile_y++;
    tile_x = 0;
  }

  LinearDemosaik(output);
  return output;
}

std::unique_ptr<Image<RgbPixel>> CameraPipeline::ProcessShot() const {
  // In this function you should implement your full RAW image processing pipeline.
  //   (1) Demosaicing
  //   (2) Address sensing defects such as bad pixels and image noise.
  //   (3) Apply local tone mapping based on the local laplacian filter or exposure fusion.
  //   (4) gamma correction
    
  // The starter code copies the raw data from the sensor to all rgb
  // channels. This results in a gray image that is just a
  // visualization of the sensor's contents.

  // BEGIN: CS348K STUDENTS MODIFY THIS CODE

  // put the lens cap on if you'd like to measure a "dark frame"
  const int width = sensor_->GetSensorWidth();
  const int height = sensor_->GetSensorHeight();
  // PART 1: Basic Camera RAW Pipeline
  sensor_->SetLensCap(false);
  auto raw_data = sensor_->GetSensorData(0, 0, width, height);
  auto burst_raw_data = sensor_->GetBurstSensorData(0, 0, width, height);

#ifdef __USE_HALIDE__
  std::cout << "Using Halide pipeline" << std::endl;

  Halide::Buffer<float> input =
    sensorDataToHalide(raw_data.get(), width, height);

  // A stub camera pipeline that copies
  // the input to all output color channels
  Halide::Var x, y, c;
  Halide::Func cameraPipeline;
  cameraPipeline(x, y, c) =
    input(x, y) * 255.0f;

  Halide::Buffer<float> output =
    cameraPipeline.realize(input.width(), input.height(), 3);

  std::unique_ptr<Image<RgbPixel>> image = rgbImageFromHalide(output);

  return image;

#else
    
  std::cout << "Using C++ pipeline made of Maddy's tears :<" << std::endl;
  // PART 1 (30 points): Basic Camera RAW Pipeline
  std::unique_ptr<Image<RgbPixel>> image = AlignAndMerge(raw_data, burst_raw_data);
  // PART 2 Local Tone Mapping 
  ExposureFusion(image);

  // PART 3 Align and merge

  // FINAL STEP: convert to 8 bit
  for (int row = 0; row < image->height(); row++) {
    for (int col = 0; col < image->width(); col++) {
      auto& pixel = (*image)(row, col);
      pixel = pixel * 255.f;
      pixel.r = std::clamp(pixel.r, 0.f, 255.f);
      pixel.g = std::clamp(pixel.g, 0.f, 255.f);
      pixel.b = std::clamp(pixel.b, 0.f, 255.f);
    }
  }

  // return processed image output
  return image;
#endif

  // END: CS348K STUDENTS MODIFY THIS CODE  
}
