defmodule VectorBasicsTest do
  use ExUnit.Case, async: true

  alias Matrex.Vector

  test "test new" do
    vec1 = Vector.new([1, 2, 3])
    vec2 = Vector.from_list([1, 2, 3])
    assert vec1 == Matrex.new(" 1 2 3")
    assert vec2 == Matrex.new(" 1 2 3")
  end

  test "test zeros" do
    vec1 = Vector.zeros(3)
    assert vec1 == Matrex.new("0 0 0")
  end

  test "test fill" do
    vec1 = Vector.fill(2, 3.5)
    assert vec1 == Matrex.new("3.5 3.5")
  end

  test "test size" do
    vec1 = Vector.fill(3, 3.5)
    assert Vector.size(vec1) == 3
  end

  test "test set_submatrix slice" do
    m = Vector.new("7 2 3")
    slice = Vector.set_slice(m, 2..3, Vector.new("1 0"))
    assert slice == Vector.new("7 1 0")
  end

  test "test is_vector?" do
    v = Matrex.ones(1, 3)
    m = Matrex.ones(3, 3)

    assert Vector.is_vector?(v) == true
    assert Vector.is_vector?(m) == false
    assert Vector.is_vector?(3) == false
  end

  test "test assert_vector" do
    v = Matrex.ones(1, 3)
    m = Matrex.ones(3, 3)

    assert Vector.assert_vector!(v) == true

    assert_raise ArgumentError, fn ->
      assert Vector.assert_vector!(m) == true
    end
  end

end
