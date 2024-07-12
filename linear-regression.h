#ifndef _LINEAR_REGRESSION_H_
#define _LINEAR_REGRESSION_H_

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

template <std::size_t N>
class LinearRegression {
public:
  template <typename... Args>
    requires(sizeof...(Args) == N + 1 && (std::convertible_to<Args, double> && ...))
  void push(Args&&... args) {
    variables.emplace_back(std::array<double, N + 1>{static_cast<double>(args)...});
  }

  virtual std::array<double, N + 1> solve() = 0;

  std::size_t size(){
    return variables.size();
  }

  const std::vector<std::array<double, N + 1>>& get(){
    return variables; 
  }

protected:
  // last element is dependant variable, or y, rest is independent variable
  std::vector<std::array<double, N + 1>> variables;
};

class TwoVariableLinearRegression : public LinearRegression<2> {
public:
  std::array<double, 3> solve() {
    // https://www.youtube.com/watch?v=m-k84cCves8&t=91s&ab_channel=MathsandStats
    double y_sum       = 0.0;
    double x1_sum      = 0.0;
    double x2_sum      = 0.0;
    double x1_sqr      = 0.0;
    double x2_sqr      = 0.0;
    double x1_cross    = 0.0;
    double x2_cross    = 0.0;
    double x1_x2_cross = 0.0;
    for (int i = 0; i < variables.size(); i++) {
      x1_sum += variables[i][0];
      x2_sum += variables[i][1];
      y_sum += variables[i][2];
      x1_sqr += variables[i][0] * variables[i][0];
      x2_sqr += variables[i][1] * variables[i][1];
      x1_cross += variables[i][0] * variables[i][2];
      x2_cross += variables[i][1] * variables[i][2];
      x1_x2_cross += variables[i][0] * variables[i][1];
    }

    double x1_sqr_sum      = x1_sqr - (x1_sum * (x1_sum / (double)variables.size()));
    double x2_sqr_sum      = x2_sqr - (x2_sum * (x2_sum / (double)variables.size()));
    double x1_y_cross_sum  = x1_cross - (x1_sum * (y_sum / (double)variables.size()));
    double x2_y_cross_sum  = x2_cross - (x2_sum * (y_sum / (double)variables.size()));
    double x1_x2_cross_sum = x1_x2_cross - (x1_sum * (x2_sum / (double)variables.size()));

    double b1 = (x2_sqr_sum * x1_y_cross_sum - x1_x2_cross_sum * x2_y_cross_sum) / (x1_sqr_sum * x2_sqr_sum - x1_x2_cross_sum * x1_x2_cross_sum);
    double b2 = (x1_sqr_sum * x2_y_cross_sum - x1_x2_cross_sum * x1_y_cross_sum) / (x1_sqr_sum * x2_sqr_sum - x1_x2_cross_sum * x1_x2_cross_sum);
    double b0 = (y_sum / (double)variables.size()) - (b1 * (x1_sum / (double)variables.size())) - (b2 * (x2_sum / (double)variables.size()));

    return std::array<double, 3>{b0, b1, b2};
  }

private:
};

class SimpleLinearRegression : public LinearRegression<1> {
public:
  std::array<double, 2> solve() {
    float a_inv = 0;
    float b_inv = 0;
    float c_inv = 0;
    float d_inv = 0;

    a_inv = variables.size();
    for (int i = 0; i < variables.size(); i++) {
      b_inv += variables[i][0];
    }
    c_inv = b_inv;
    for (int i = 0; i < variables.size(); i++) {
      d_inv += variables[i][0] * variables[i][0];
    }

    float det = a_inv * d_inv - b_inv * c_inv;

    a_inv = (1.0f / det) * a_inv;
    b_inv = (1.0f / det) * -b_inv;
    c_inv = (1.0f / det) * -c_inv;
    d_inv = (1.0f / det) * d_inv;

    float y0 = 0;
    float y1 = 0;
    for (int i = 0; i < variables.size(); i++) {
      y0 += variables[i][1];
      y1 += variables[i][0] * variables[i][1];
    }

    double y_intercept = d_inv * y0 + b_inv * y1;
    double sloap       = c_inv * y0 + a_inv * y1;

    return std::array<double, 2>{y_intercept, sloap};
  }

private:
};

#endif // _LINEAR_REGRESSION_H_