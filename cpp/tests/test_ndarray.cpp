#include "exaflow/kernels.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <limits>
#include <utility>

TEST(NDArray, StoresOneDimensionalValues) {
    exaflow::NDArray array({3});
    array[1] = 7.0;
    EXPECT_EQ(array.get_dimension(), 1);
    EXPECT_EQ(array.get_size(), 3);
    EXPECT_EQ(array.get_shape()[0], 3);
    EXPECT_DOUBLE_EQ(array[0], 0.0);
    EXPECT_DOUBLE_EQ(array.get_values()[1], 7.0);
}

TEST(NDArray, StoresTwoDimensionalValuesInRowOrder) {
    exaflow::NDArray array({2, 3});
    array[1, 2] = 7.0;
    EXPECT_EQ(array.get_dimension(), 2);
    EXPECT_EQ(array.get_size(), 6);
    EXPECT_DOUBLE_EQ(array.get_values()[5], 7.0);
    array.get_values()[3] = 8.0;
    EXPECT_DOUBLE_EQ((array[1, 0]), 8.0);
}

TEST(NDArray, StoresThreeDimensionalValuesWithLastAxisContiguous) {
    exaflow::NDArray array({2, 3, 4});
    std::size_t index = 0;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                array[i, j, k] = static_cast<double>(index++);
            }
        }
    }
    EXPECT_EQ(array.get_dimension(), 3);
    EXPECT_EQ(array.get_size(), 24);
    for (std::size_t i = 0; i < array.get_size(); ++i) {
        EXPECT_DOUBLE_EQ(array.get_values()[i], static_cast<double>(i));
    }
    EXPECT_DOUBLE_EQ((array[1, 2, 3]), 23.0);
}

TEST(NDArray, SupportsFourDimensionsAndAnInitialValue) {
    exaflow::NDArray array({2, 2, 3, 4}, 2.5);
    array[1, 0, 2, 3] = 9.0;
    EXPECT_EQ(array.get_dimension(), 4);
    EXPECT_EQ(array.get_size(), 48);
    EXPECT_DOUBLE_EQ(array.get_values()[35], 9.0);
    EXPECT_DOUBLE_EQ((array[0, 1, 2, 3]), 2.5);
}

TEST(NDArray, TakesInitialValuesInStorageOrder) {
    exaflow::NDArray array({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    EXPECT_EQ(array.get_dimension(), 2);
    EXPECT_DOUBLE_EQ((array[0, 2]), 3.0);
    EXPECT_DOUBLE_EQ((array[1, 0]), 4.0);
    EXPECT_THROW((exaflow::NDArray({2, 3}, {1.0, 2.0})), std::invalid_argument);
}

TEST(NDArray, AcceptsRuntimeShapeAndCoordinateSpans) {
    std::array<std::size_t, 3> shape{2, 3, 4};
    exaflow::NDArray array{std::span<const std::size_t>(shape)};
    shape[0] = 99;
    const std::array<std::size_t, 3> coordinates{1, 2, 3};
    array[std::span<const std::size_t>(coordinates)] = 7.0;
    const auto& read_only = array;
    EXPECT_EQ(array.get_shape()[0], 2);
    EXPECT_DOUBLE_EQ(read_only[std::span<const std::size_t>(coordinates)], 7.0);
    EXPECT_DOUBLE_EQ((read_only[1, 2, 3]), 7.0);
    static_assert(std::same_as<decltype(read_only[1, 2, 3]), const double&>);
    static_assert(std::same_as<decltype(read_only.get_shape()), std::span<const std::size_t>>);
    static_assert(std::same_as<decltype(read_only.get_values()), std::span<const double>>);
}

TEST(NDArray, CopiesOwnTheirShapeAndValues) {
    exaflow::NDArray original({2, 3}, 2.5);
    auto copy = original;
    copy[1, 2] = 9.0;
    EXPECT_DOUBLE_EQ((original[1, 2]), 2.5);
    exaflow::NDArray assigned({1});
    assigned = original;
    original[1, 2] = 8.0;
    EXPECT_EQ(assigned.get_dimension(), 2);
    EXPECT_DOUBLE_EQ((assigned[1, 2]), 2.5);
}

TEST(NDArray, MovesTransferStorageAndLeaveAnEmptySource) {
    exaflow::NDArray original({2, 3, 4}, 2.5);
    exaflow::NDArray moved(std::move(original));
    EXPECT_EQ(original.get_dimension(), 0);
    EXPECT_EQ(original.get_size(), 0);
    EXPECT_DOUBLE_EQ((moved[1, 2, 3]), 2.5);
    exaflow::NDArray assigned({1});
    assigned = std::move(moved);
    EXPECT_EQ(moved.get_dimension(), 0);
    EXPECT_EQ(moved.get_size(), 0);
    EXPECT_EQ(assigned.get_size(), 24);
    EXPECT_DOUBLE_EQ((assigned[1, 2, 3]), 2.5);
}

TEST(NDArray, RejectsTheWrongNumberOfIndices) {
    exaflow::NDArray array({2, 3, 4});
    EXPECT_THROW((array[1, 2]), std::invalid_argument);
    EXPECT_THROW((array[0, 0, 0, 0]), std::invalid_argument);
    const auto& read_only = array;
    EXPECT_THROW((read_only[0]), std::invalid_argument);
}

TEST(NDArray, RejectsNegativeAndOutOfRangeCoordinates) {
    exaflow::NDArray array({2, 3, 4});
    EXPECT_THROW((array[-1, 0, 0]), std::out_of_range);
    EXPECT_THROW((array[2, 0, 0]), std::out_of_range);
    EXPECT_THROW((array[0, 3, 0]), std::out_of_range);
    EXPECT_THROW((array[0, 0, 4]), std::out_of_range);
    const auto& read_only = array;
    EXPECT_THROW((read_only[0, 0, -1]), std::out_of_range);
}

TEST(NDArray, SupportsZeroSizedAxesWithoutAllocatingValues) {
    exaflow::NDArray array({std::numeric_limits<std::size_t>::max(), 0, 3});
    EXPECT_EQ(array.get_dimension(), 3);
    EXPECT_EQ(array.get_size(), 0);
    EXPECT_TRUE(array.get_values().empty());
    EXPECT_THROW((array[0, 0, 0]), std::out_of_range);
}

TEST(NDArray, RejectsOversizedShapesBeforeAllocation) {
    EXPECT_THROW(
        (exaflow::NDArray({std::numeric_limits<std::size_t>::max(), 2})),
        std::length_error
    );
    EXPECT_THROW(
        (exaflow::NDArray({2, std::numeric_limits<std::size_t>::max()})),
        std::length_error
    );
}

TEST(NDArray, DefaultsToEmptyAndRejectsAnExplicitEmptyShape) {
    exaflow::NDArray array;
    EXPECT_EQ(array.get_dimension(), 0);
    EXPECT_EQ(array.get_size(), 0);
    EXPECT_TRUE(array.get_shape().empty());
    EXPECT_THROW((array[0]), std::invalid_argument);
    EXPECT_THROW((exaflow::NDArray(std::initializer_list<std::size_t>{})), std::invalid_argument);
}
