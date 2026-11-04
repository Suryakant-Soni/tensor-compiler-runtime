#include <catch2/catch_test_macros.hpp>

#include "tcr/ir/dtype.hpp"
#include "tcr/ir/shape.hpp"
#include "tcr/runtime/storage.hpp"
#include "tcr/runtime/tensor.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using tcr::ir::Dtype;
using tcr::ir::Shape;
using tcr::runtime::RuntimeStorage;
using tcr::runtime::Tensor;

// -----------------------------------------------------------------------------
// Construction and metadata
// -----------------------------------------------------------------------------

TEST_CASE("runtime Tensor stores vector shape dtype and data") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{1.0F, 2.0F, 3.0F},
  };

  REQUIRE(tensor.shape() == Shape{3});
  REQUIRE(tensor.shape().rank() == 1);
  REQUIRE(tensor.shape().num_elements() == 3);
  REQUIRE(tensor.dtype() == Dtype::F32);

  REQUIRE(tensor.data() == std::vector<float>{
                               1.0F,
                               2.0F,
                               3.0F,
                           });
}

TEST_CASE("runtime Tensor stores matrix shape dtype and data") {
  const Tensor tensor{
      Shape{2, 3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
          5.0F,
          6.0F,
      },
  };

  REQUIRE(tensor.shape() == Shape{2, 3});
  REQUIRE(tensor.shape().rank() == 2);
  REQUIRE(tensor.shape().num_elements() == 6);
  REQUIRE(tensor.dtype() == Dtype::F32);

  REQUIRE(tensor.data() == std::vector<float>{
                               1.0F,
                               2.0F,
                               3.0F,
                               4.0F,
                               5.0F,
                               6.0F,
                           });
}

TEST_CASE("runtime Tensor supports higher rank shape") {
  const Tensor tensor{
      Shape{2, 2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
          5.0F,
          6.0F,
          7.0F,
          8.0F,
      },
  };

  REQUIRE(tensor.shape() == Shape{2, 2, 2});
  REQUIRE(tensor.shape().rank() == 3);
  REQUIRE(tensor.shape().num_elements() == 8);
  REQUIRE(tensor.dtype() == Dtype::F32);

  REQUIRE(tensor.data() == std::vector<float>{
                               1.0F,
                               2.0F,
                               3.0F,
                               4.0F,
                               5.0F,
                               6.0F,
                               7.0F,
                               8.0F,
                           });
}

TEST_CASE("runtime Tensor supports scalar shape") {
  const Tensor tensor{
      Shape{},
      Dtype::F32,
      std::vector<float>{42.0F},
  };

  REQUIRE(tensor.shape() == Shape{});
  REQUIRE(tensor.shape().rank() == 0);
  REQUIRE(tensor.shape().num_elements() == 1);
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() == std::vector<float>{42.0F});
}

TEST_CASE("runtime Tensor supports single-element vector") {
  const Tensor tensor{
      Shape{1},
      Dtype::F32,
      std::vector<float>{7.0F},
  };

  REQUIRE(tensor.shape() == Shape{1});
  REQUIRE(tensor.shape().rank() == 1);
  REQUIRE(tensor.shape().num_elements() == 1);
  REQUIRE(tensor.dtype() == Dtype::F32);
  REQUIRE(tensor.data() == std::vector<float>{7.0F});
}

// -----------------------------------------------------------------------------
// Stored values
// -----------------------------------------------------------------------------

TEST_CASE("runtime Tensor preserves zero values") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{0.0F, 0.0F, 0.0F},
  };

  REQUIRE(tensor.data() == std::vector<float>{
                               0.0F,
                               0.0F,
                               0.0F,
                           });
}

TEST_CASE("runtime Tensor preserves negative values") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          -1.0F,
          -2.0F,
          -3.0F,
      },
  };

  REQUIRE(tensor.data() == std::vector<float>{
                               -1.0F,
                               -2.0F,
                               -3.0F,
                           });
}

TEST_CASE("runtime Tensor preserves mixed floating point values") {
  const Tensor tensor{
      Shape{5},
      Dtype::F32,
      std::vector<float>{
          -2.5F,
          0.0F,
          0.25F,
          3.75F,
          100.5F,
      },
  };

  REQUIRE(tensor.data() == std::vector<float>{
                               -2.5F,
                               0.0F,
                               0.25F,
                               3.75F,
                               100.5F,
                           });
}

// -----------------------------------------------------------------------------
// Construction validation
// -----------------------------------------------------------------------------

TEST_CASE("runtime Tensor rejects too little data for vector shape") {
  REQUIRE_THROWS_AS((Tensor{
                        Shape{3},
                        Dtype::F32,
                        std::vector<float>{
                            1.0F,
                            2.0F,
                        },
                    }),
                    std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects too much data for vector shape") {
  REQUIRE_THROWS_AS((Tensor{
                        Shape{2},
                        Dtype::F32,
                        std::vector<float>{
                            1.0F,
                            2.0F,
                            3.0F,
                        },
                    }),
                    std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects empty data for scalar") {
  REQUIRE_THROWS_AS((Tensor{
                        Shape{},
                        Dtype::F32,
                        std::vector<float>{},
                    }),
                    std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects multiple values for scalar") {
  REQUIRE_THROWS_AS((Tensor{
                        Shape{},
                        Dtype::F32,
                        std::vector<float>{
                            1.0F,
                            2.0F,
                        },
                    }),
                    std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects too little matrix data") {
  REQUIRE_THROWS_AS((Tensor{
                        Shape{2, 3},
                        Dtype::F32,
                        std::vector<float>{
                            1.0F,
                            2.0F,
                            3.0F,
                            4.0F,
                            5.0F,
                        },
                    }),
                    std::invalid_argument);
}

TEST_CASE("runtime Tensor rejects too much matrix data") {
  REQUIRE_THROWS_AS((Tensor{
                        Shape{2, 2},
                        Dtype::F32,
                        std::vector<float>{
                            1.0F,
                            2.0F,
                            3.0F,
                            4.0F,
                            5.0F,
                        },
                    }),
                    std::invalid_argument);
}

// -----------------------------------------------------------------------------
// Flat load
// -----------------------------------------------------------------------------

TEST_CASE("runtime Tensor load reads first flat element") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          10.0F,
          20.0F,
          30.0F,
      },
  };

  REQUIRE(tensor.load(0) == 10.0F);
}

TEST_CASE("runtime Tensor load reads middle flat element") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          10.0F,
          20.0F,
          30.0F,
      },
  };

  REQUIRE(tensor.load(1) == 20.0F);
}

TEST_CASE("runtime Tensor load reads last flat element") {
  const Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          10.0F,
          20.0F,
          30.0F,
      },
  };

  REQUIRE(tensor.load(2) == 30.0F);
}

TEST_CASE("runtime Tensor load uses row-major flat matrix storage") {
  const Tensor tensor{
      Shape{2, 3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
          5.0F,
          6.0F,
      },
  };

  REQUIRE(tensor.load(0) == 1.0F);
  REQUIRE(tensor.load(1) == 2.0F);
  REQUIRE(tensor.load(2) == 3.0F);
  REQUIRE(tensor.load(3) == 4.0F);
  REQUIRE(tensor.load(4) == 5.0F);
  REQUIRE(tensor.load(5) == 6.0F);
}

TEST_CASE("runtime Tensor load reads scalar value at offset zero") {
  const Tensor tensor{
      Shape{},
      Dtype::F32,
      std::vector<float>{42.0F},
  };

  REQUIRE(tensor.load(0) == 42.0F);
}

// -----------------------------------------------------------------------------
// Flat store
// -----------------------------------------------------------------------------

TEST_CASE("runtime Tensor store updates first element") {
  Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
      },
  };

  tensor.store(0, 10.0F);

  REQUIRE(tensor.load(0) == 10.0F);
  REQUIRE(tensor.load(1) == 2.0F);
  REQUIRE(tensor.load(2) == 3.0F);
}

TEST_CASE("runtime Tensor store updates middle element") {
  Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
      },
  };

  tensor.store(1, 20.0F);

  REQUIRE(tensor.load(0) == 1.0F);
  REQUIRE(tensor.load(1) == 20.0F);
  REQUIRE(tensor.load(2) == 3.0F);
}

TEST_CASE("runtime Tensor store updates last element") {
  Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
      },
  };

  tensor.store(2, 30.0F);

  REQUIRE(tensor.load(0) == 1.0F);
  REQUIRE(tensor.load(1) == 2.0F);
  REQUIRE(tensor.load(2) == 30.0F);
}

TEST_CASE("runtime Tensor store updates scalar") {
  Tensor tensor{
      Shape{},
      Dtype::F32,
      std::vector<float>{0.0F},
  };

  tensor.store(0, 42.0F);

  REQUIRE(tensor.load(0) == 42.0F);
  REQUIRE(tensor.data() == std::vector<float>{42.0F});
}

TEST_CASE("runtime Tensor store updates flat matrix element") {
  Tensor tensor{
      Shape{2, 3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
          5.0F,
          6.0F,
      },
  };

  tensor.store(4, 50.0F);

  REQUIRE(tensor.data() == std::vector<float>{
                               1.0F,
                               2.0F,
                               3.0F,
                               4.0F,
                               50.0F,
                               6.0F,
                           });
}

TEST_CASE("runtime Tensor supports repeated stores") {
  Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          0.0F,
          0.0F,
          0.0F,
      },
  };

  tensor.store(0, 1.0F);
  tensor.store(1, 2.0F);
  tensor.store(2, 3.0F);
  tensor.store(1, 20.0F);

  REQUIRE(tensor.data() == std::vector<float>{
                               1.0F,
                               20.0F,
                               3.0F,
                           });
}

// -----------------------------------------------------------------------------
// Shared-storage copy semantics
// -----------------------------------------------------------------------------

TEST_CASE("runtime Tensor copy preserves metadata and data") {
  const Tensor original{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
      },
  };

  const Tensor copy = original;

  REQUIRE(copy.shape() == original.shape());
  REQUIRE(copy.dtype() == original.dtype());
  REQUIRE(copy.data() == original.data());
}

TEST_CASE("runtime Tensor copy shares underlying storage") {
  Tensor original{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
      },
  };

  Tensor copy = original;

  copy.store(1, 99.0F);

  REQUIRE(copy.load(1) == 99.0F);
  REQUIRE(original.load(1) == 99.0F);
}

TEST_CASE("runtime Tensor mutation through original is visible through copy") {
  Tensor original{
      Shape{2},
      Dtype::F32,
      std::vector<float>{
          5.0F,
          10.0F,
      },
  };

  Tensor copy = original;

  original.store(0, 50.0F);

  REQUIRE(original.load(0) == 50.0F);
  REQUIRE(copy.load(0) == 50.0F);
}

TEST_CASE("multiple runtime Tensor copies share underlying storage") {
  Tensor first{
      Shape{2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
      },
  };

  Tensor second = first;
  Tensor third = second;

  third.store(1, 100.0F);

  REQUIRE(first.load(1) == 100.0F);
  REQUIRE(second.load(1) == 100.0F);
  REQUIRE(third.load(1) == 100.0F);
}

TEST_CASE("runtime Tensor copy keeps shape and dtype independent objects") {
  Tensor first{
      Shape{2, 2},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
          4.0F,
      },
  };

  Tensor second = first;

  REQUIRE(first.shape() == Shape{2, 2});
  REQUIRE(second.shape() == Shape{2, 2});
  REQUIRE(first.dtype() == Dtype::F32);
  REQUIRE(second.dtype() == Dtype::F32);
}

// -----------------------------------------------------------------------------
// data() consistency
// -----------------------------------------------------------------------------

TEST_CASE("runtime Tensor load agrees with data accessor") {
  const Tensor tensor{
      Shape{4},
      Dtype::F32,
      std::vector<float>{
          4.0F,
          8.0F,
          12.0F,
          16.0F,
      },
  };

  for (std::size_t i = 0; i < tensor.data().size(); ++i) {
    REQUIRE(tensor.load(i) == tensor.data()[i]);
  }
}

TEST_CASE("runtime Tensor store is reflected by const data accessor") {
  Tensor tensor{
      Shape{3},
      Dtype::F32,
      std::vector<float>{
          1.0F,
          2.0F,
          3.0F,
      },
  };

  tensor.store(2, 30.0F);

  const Tensor &const_tensor = tensor;

  REQUIRE(const_tensor.data()[2] == 30.0F);
  REQUIRE(const_tensor.load(2) == 30.0F);
}

// -----------------------------------------------------------------------------
// Shared physical storage / logical tensor bounds
// -----------------------------------------------------------------------------

TEST_CASE("runtime Tensor can wrap existing RuntimeStorage without copying") {
  auto storage = std::make_shared<RuntimeStorage>();
  storage->bytes.resize(2 * sizeof(float));
  storage->store_f32(0, 3.0F);
  storage->store_f32(1, 7.0F);

  Tensor tensor{Shape{2}, Dtype::F32, storage};

  REQUIRE(tensor.storage() == storage);
  REQUIRE(tensor.data() == std::vector<float>{3.0F, 7.0F});

  storage->store_f32(1, 11.0F);
  REQUIRE(tensor.load(1) == 11.0F);
}

TEST_CASE("runtime Tensor allows physical storage larger than logical tensor") {
  auto storage = std::make_shared<RuntimeStorage>();
  storage->bytes.resize(4 * sizeof(float));
  storage->store_f32(0, 5.0F);
  storage->store_f32(1, 99.0F);
  storage->store_f32(2, 88.0F);
  storage->store_f32(3, 77.0F);

  const Tensor tensor{Shape{1}, Dtype::F32, storage};

  REQUIRE(tensor.shape().num_elements() == 1);
  REQUIRE(tensor.data() == std::vector<float>{5.0F});
  REQUIRE_THROWS_AS(tensor.load(1), std::out_of_range);
}

TEST_CASE("runtime Tensor store cannot write beyond logical shape even when "
          "storage is larger") {
  auto storage = std::make_shared<RuntimeStorage>();
  storage->bytes.resize(4 * sizeof(float));

  Tensor tensor{Shape{1}, Dtype::F32, storage};

  tensor.store(0, 12.0F);
  REQUIRE(tensor.load(0) == 12.0F);
  REQUIRE_THROWS_AS(tensor.store(1, 13.0F), std::out_of_range);
}

TEST_CASE(
    "runtime Tensor rejects wrapped storage smaller than logical tensor") {
  auto storage = std::make_shared<RuntimeStorage>();
  storage->bytes.resize(sizeof(float));

  REQUIRE_THROWS_AS((Tensor{Shape{2}, Dtype::F32, storage}),
                    std::invalid_argument);
}
