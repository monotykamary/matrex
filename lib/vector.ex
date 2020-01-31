defmodule Matrex.Vector do
  import Matrex.Guards

  @moduledoc """
  Vector wrappers for Matrex.
  ```


  """

  @doc """
  Creates new 1-column matrix (aka vector) from the given list.

  ## Examples

      iex> [1,2,3] |> Vector.new()
      #Matrex[1×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      └                         ┘

  """
  def new(lst) when is_list(lst) do
    # Matrex.new( for l <- lst, into: [] do [l] end )
    Matrex.new( [lst] )
  end

  def new(text) when is_binary(text) do
    res = Matrex.new( text )

  end

  @doc """
  Is given Matrex a vector?

  ## Examples

      iex> [1,2,3] |> Vector.new()
      #Matrex[1×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      └                         ┘

  """
  def is_vector?( vector_data(_len, _data) = m ), do: true
  def is_vector?( _m ), do: false

  def assert_vector!( vector_data(_len, _data) = m ), do: true
  def assert_vector!( matrex_data(c, r, _data) = m ), do: throw %ArgumentError{message: "Expecting a Vector[1xN], got Matrex[#{c}, #{r}]"}
  def assert_vector!( _m ), do: throw %ArgumentError{message: "Expecting a Matrex Vector"}

  @doc """
  Creates new 1-column matrix (aka vector) from the given list.

  ## Examples

      iex> [1,2,3] |> Vector.from_list()
      #Matrex[1×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      └                         ┘

  """
  def from_list(lst) when is_list(lst) do
    # Matrex.new( for l <- lst, into: [] do [l] end )
    Matrex.new( [lst] )
  end

  @doc """
  Create matrix filled with given value. NIF.

  ## Example

      iex> Vector.fill(3, 55)
      #Matrex[1×3]
      ┌                         ┐
      │    55.0    55.0    55.0 │
      └                         ┘
  """
  @spec fill(Matrex.index(), Matrex.element()) :: Matrex.t()
  def fill(len, value)
      when (is_integer(len) and is_number(value)) or is_atom(value),
      do: Matrex.fill(1, len, value)

  @doc """
  Create matrix filled with ones.

  ## Example

      iex> Vector.ones(3)
      #Matrex[1×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      └                         ┘
  """
  @spec ones(Matrex.index()) :: Matrex.t()
  def ones(len) when is_integer(len), do: Matrex.fill(1, len, 1)

  @doc """
  Create matrix of zeros of the specified size. NIF, using `memset()`.

  Faster, than `fill(rows, cols, 0)`.

  ## Example

      iex> Matrex.zeros(3)
      #Matrex[4×3]
      ┌                         ┐
      │     0.0     0.0     0.0 │
      └                         ┘
  """
  @spec zeros(Matrex.index()) :: Matrex.t()
  def zeros(len) when is_integer(len), do: Matrex.zeros(1,len)

  @doc """
  Set element of matrix at the specified position (one-based) to new value.

  ## Example

      iex> m = Vector.ones(3)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      └                         ┘

      iex> m = Vector.set(m, 2, 0.0)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     0.0     1.0 │
      └                         ┘

      iex> m = Vector.set(m, 2, :neg_inf)
      #Matrex[3×3]
      ┌                         ┐
      │     1.0     -∞      1.0 │
      └                         ┘
  """
  @spec set(Matrex.t(), Matrex.index(), Matrex.element()) :: Matrex.t()
  def set(vector_data(len, _rest, _data) = m, idx, value)
      when (is_number(value) or value in [:nan, :inf, :neg_inf]) and idx > 0 and idx <= len,
      do: Matrex.set(m, 1, idx, value)

  @doc """
  Return size of matrix as `{rows, cols}`

  ## Example

      iex> m = Vector.random(3)
      #Matrex[2×3]
      ┌                         ┐
      │ 0.63423 0.29651 0.22844 │
      └                         ┘
      iex> Vector.size(m)
      3
  """
  @spec size(Matrex.t()) :: Matrex.index()
  def size(vector_data(len, _)), do: len

  @doc """
  Set submatrix for a given matrix. NIF.

  Rows and columns ranges are inclusive and one-based.

  ## Example

      iex> m = Vector.new("7 2 3")
      #Matrex[3×3]
      ┌                         ┐
      │     7.0     2.0     3.0 │
      └                         ┘

      iex> Vector.set_slice(m, 2..3, Vector.new("1 0"))
      #Matrex[3×3]
      ┌                         ┐
      │     7.0     0.0     1.0 │
      └                         ┘
  """
  @spec set_slice(Matrex.t(), Range.t(), Matrex.t()) :: Matrex.t()
  def set_slice(vector_data(len, _rest, _data) = v,
                    idx_from..idx_to,
                    vector_data(sub_len, _sub_rest, _subdata) = subslice)
      when idx_from in 1..len and
           idx_to in idx_from..len and
           sub_len == (idx_to-idx_from+1),
      do: Matrex.set_submatrix(v, 1..1, idx_from..idx_to, subslice)

end
