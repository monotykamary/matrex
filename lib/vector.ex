defmodule Matrex.Vector do
  @moduledoc """
  Vector wrappers for Matrex.
  ```

  """

  @doc """
  Creates new 1-column matrix (aka vector) from the given list.

  ## Examples

      iex> [1,2,3] |> Matrex.from_list()
      #Matrex[1×3]
      ┌                         ┐
      │     1.0     2.0     3.0 │
      └                         ┘

  """
  def from_list(lst) when is_list(lst) do
    Matrex.new( for l <- lst, into: [] do [l] end )
  end

  @doc """
  Create matrix filled with given value. NIF.

  ## Example

      iex> Matrex.fill(3, 55)
      #Matrex[4×3]
      ┌                         ┐
      │    55.0    55.0    55.0 │
      │    55.0    55.0    55.0 │
      │    55.0    55.0    55.0 │
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

      iex> Matrex.ones(2, 3)
      #Matrex[2×3]
      ┌                         ┐
      │     1.0     1.0     1.0 │
      │     1.0     1.0     1.0 │
      └                         ┘
  """
  @spec ones(Matrex.index()) :: Matrex.t()
  def ones(len) when is_integer(len), do: fill(len, 1)

  @doc """
  Create matrix of zeros of the specified size. NIF, using `memset()`.

  Faster, than `fill(rows, cols, 0)`.

  ## Example

      iex> Matrex.zeros(4,3)
      #Matrex[4×3]
      ┌                         ┐
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      │     0.0     0.0     0.0 │
      └                         ┘
  """
  @spec zeros(Matrex.index()) :: Matrex.t()
  def zeros(len) when is_integer(len), do: Matrex.zeros(1,len)

end
