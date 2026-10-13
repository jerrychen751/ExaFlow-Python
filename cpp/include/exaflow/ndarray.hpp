#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace exaflow {

class NDArray {
public:
    NDArray() = default; // Forces generation of default constructor

    // Uses explicit to prevent implicit conversions during object construction
    // std::vector<std::size_t> dims{2, 3};
    // NDArray a(dims);
    explicit NDArray(std::span<const std::size_t> shape, double fill_value = 0.0)
        : shape_(shape.begin(), shape.end()) {
        if (shape_.empty()) {
            throw std::invalid_argument("NDArray must have at least one dimension");
        }

        // If any dim is 0, return early w/ successful construction (no data)
        if (std::find(shape_.begin(), shape_.end(), 0) != shape_.end()) {
            return;
        }

        // Compute number of values the array needs and allocate it
        std::size_t size = 1;
        for (std::size_t extent : shape_) {
            if (extent > values_.max_size() / size) {
                throw std::length_error("NDArray is too large");
            }
            size *= extent;
        }
        values_.resize(size, fill_value);
    }

    // NDArray a({2, 3})
    NDArray(std::initializer_list<std::size_t> shape, double fill_value = 0.0)
        : NDArray(std::span<const std::size_t>(shape.begin(), shape.size()), fill_value) {}

    // NDArray a({2, 3}, {1, 2, 3, 4, 5, 6}) -> [[1, 2, 3], [4, 5, 6]]
    NDArray(std::initializer_list<std::size_t> shape, std::initializer_list<double> values)
        : NDArray(shape) {
        if (values.size() != values_.size()) {
            throw std::invalid_argument("NDArray value count must match its shape");
        }
        std::copy(values.begin(), values.end(), values_.begin());
    }

    // Copy constructor; new object gets its own buffers w/ same contents
    // Suppose we've already allocated NDArray a({2, 3})
    // We can say NDArray b(a); this performs a copy action instead of sharing the same underlying shape_ and values_
    // Note: the default copy constructor shallow copies raw ptr members
    NDArray(const NDArray&) = default;

    // Uses noexcept to guarantee that the function will not throw
    // Move constructor; new object gets transferred ownership of member data and the original will not be used again
    // When calling this, use std::move() to turn an object into an rvalue reference
    // Suppose we've already allocated NDArray a({2, 3})
    // We can say NDArray b(std::move(a));
    NDArray(NDArray&& other) noexcept {
        shape_.swap(other.shape_);
        values_.swap(other.values_);
    }

    // Overloads the "=" operator (argument is what's on the right side of equals sign)
    // If the argument is an lvalue, we use copy constructor
    // If the argument is an rvalue, we use move constructor
    NDArray& operator=(NDArray other) noexcept {
        shape_.swap(other.shape_);
        values_.swap(other.values_);
        return *this;
    }

    // Operator overload for indexing
    // Suppose we've already allocated NDArray a({2, 3})
    // Then we can perform a[1, 2] to access the 2nd row, 3rd col element
    // This requires the two types of operator below (each duplicated for const version)
    // When we do a[1, 2] it goes to the (Indices... indices) version which calls a helper to validate the indices and return an array instead of separate individual int arguments
    // Then it calls the operator which takes in std::span to return the element at a particular index
    double& operator[](std::span<const std::size_t> indices) {
        return values_.at(get_index(indices));
    }

    const double& operator[](std::span<const std::size_t> indices) const {
        return values_.at(get_index(indices));
    }

    template<std::integral... Indices>
    double& operator[](Indices... indices) {
        const auto coordinates = get_coordinates(indices...);
        return (*this)[std::span<const std::size_t>(coordinates)];
    }

    template<std::integral... Indices>
    const double& operator[](Indices... indices) const {
        const auto coordinates = get_coordinates(indices...);
        return (*this)[std::span<const std::size_t>(coordinates)];
    }

    std::size_t get_dimension() const {
        return shape_.size();
    }

    std::span<const std::size_t> get_shape() const {
        return shape_;
    }

    std::size_t get_size() const {
        return values_.size();
    }

    std::span<double> get_values() {
        return values_;
    }

    std::span<const double> get_values() const {
        return values_;
    }

private:
    // Helper for obtaining index position in flat values_ buffer given some indexed dimensions which are valid for our shape_
    std::size_t get_index(std::span<const std::size_t> indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("NDArray index count must match its dimension");
        }

        std::size_t index = 0;
        for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
            if (indices[axis] >= shape_[axis]) {
                throw std::out_of_range("NDArray index is out of range");
            }
            // Multiplying by the next extent keeps the last axis contiguous.
            index = index * shape_[axis] + indices[axis];
        }
        return index;
    }

    // Helper for validating a series of integer indices and returning them as an array
    template<std::integral... Indices> // compile-time requirement that we get zero or more integers
    std::array<std::size_t, sizeof...(Indices)> get_coordinates(Indices... indices) const {
        if (!(std::in_range<std::size_t>(+indices) && ...)) {
            throw std::out_of_range("NDArray index is out of range");
        }
        return {static_cast<std::size_t>(indices)...};
    }

    std::vector<std::size_t> shape_;
    std::vector<double> values_;
};

}
