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

end
