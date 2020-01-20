# Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
#
#
# (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
#
# Permission is granted for anyone to copy, use, or modify these
# programs and accompanying documents for purposes of research or
# education, provided this copyright notice is retained, and note is
# made of any changes that have been made.
#
# These programs and documents are distributed without any warranty,
# express or implied.  As the programs were written for research
# purposes only, they have not been tested to the degree that would be
# advisable in any important application.  All use of these programs is
# entirely at the user's own risk.
#
# [ml-class] Changes Made:
# 1) Function name and argument specifications
# 2) Output display
#
# [versilov] Changes:
# 1) Ported to Elixir

defmodule Matrex.Algorithms do
  alias Matrex.Algorithms

  @doc """
  Logistic regression cost and gradient function with regularization from Andrew Ng's course (ex3).

  Computes the cost of using `theta` as the parameter for regularized logistic regression and the
  gradient of the cost w.r.t. to the parameters.

  Compatible with `fmincg/4` algorithm from this module.

  `theta`  — parameters, to compute cost for

  `X`  — training data input.

  `y`  — training data output.

  `lambda`  — regularization parameter.

  """
  @spec lr_cost_fun(Matrex.t(), {Matrex.t(), Matrex.t(), number, non_neg_integer}, pos_integer) ::
          {float, Matrex.t()}
  def lr_cost_fun(
        %Matrex{} = theta,
        {%Matrex{} = x, %Matrex{} = y, lambda, digit} = _params,
        iteration \\ 0
      )
      when is_number(lambda) do
    m = y[:rows]

    h = Matrex.dot_and_apply(x, theta, :sigmoid)
    l = Matrex.ones(theta[:rows], theta[:cols]) |> Matrex.set(1, 1, 0)

    regularization =
      Matrex.dot_tn(l, Matrex.square(theta))
      |> Matrex.scalar()
      |> Kernel.*(lambda / (2 * m))

    j =
      y
      |> Matrex.dot_tn(Matrex.apply(h, :log), -1)
      |> Matrex.subtract(
        Matrex.dot_tn(
          Matrex.subtract(1, y),
          Matrex.apply(Matrex.subtract(1, h), :log)
        )
      )
      |> Matrex.scalar()
      |> (fn
            :nan -> :nan
            :inf -> :inf
            x -> x / m + regularization
          end).()

    grad =
      x
      |> Matrex.dot_tn(Matrex.subtract(h, y))
      |> Matrex.add(Matrex.multiply(theta, l), 1.0, lambda)
      |> Matrex.divide(m)

    if theta[:rows] == 785 do
      r = div(digit - 1, 3) * 17 + 1
      c = rem(digit - 1, 3) * 30 + 1

      j_str = j |> Matrex.element_to_string() |> String.pad_leading(20)
      iter_str = iteration |> Integer.to_string() |> String.pad_leading(3)

      IO.puts("#{IO.ANSI.home()}")

      theta[2..785]
      |> Matrex.reshape(28, 28)
      |> Dashboard.heatmap(
        digit,
        :mono256,
        at: {r, c},
        title: "[#{IO.ANSI.bright()}#{rem(digit, 10)}#{IO.ANSI.normal()} | #{iter_str} #{j_str}]"
      )
    end

    {j, grad}
  end

  @doc """
  The same cost function, implemented with  operators from `Matrex.Operators` module.

  Works 2 times slower, than standard implementation. But it's a way more readable.
  """
  def lr_cost_fun_ops(
        %Matrex{} = theta,
        {%Matrex{} = x, %Matrex{} = y, lambda} = _params,
        _iteration \\ 0
      )
      when is_number(lambda) do
    # Turn off original operators
    import Kernel, except: [-: 1, +: 2, -: 2, *: 2, /: 2, <|>: 2]
    import Matrex
    import Matrex.Operators

    m = y[:rows]

    h = sigmoid(x * theta)
    l = ones(size(theta)) |> set(1, 1, 0.0)

    j = (-t(y) * log(h) - t(1 - y) * log(1 - h) + lambda / 2 * t(l) * pow2(theta)) / m

    grad = (t(x) * (h - y) + (theta <|> l) * lambda) / m

    {scalar(j), grad}
  end

  @doc """
  Run logistic regression one-vs-all MNIST digits recognition in parallel.
  """
  def run_lr(iterations \\ 56, concurrency \\ 1) do
    start_timestamp = :os.timestamp()

    {x, y} =
      case Mix.env() do
        :test ->
          {Matrex.load("test/data/X.mtx.gz"), Matrex.load("test/data/Y.mtx")}

        _ ->
          Dashboard.start()
          {Matrex.load("test/data/Xtest.mtx.gz"), Matrex.load("test/data/Ytest.mtx")}
      end

    x = Matrex.concat(Matrex.ones(x[:rows], 1), x)
    theta = Matrex.zeros(x[:cols], 1)

    lambda = 0.3

    solutions =
      1..10
      |> Task.async_stream(
        fn digit ->
          y3 = Matrex.apply(y, fn val -> if(val == digit, do: 1.0, else: 0.0) end)

          {sX, fX, _i} = Algorithms.GradientDescent.fmincg(&lr_cost_fun/3, theta, {x, y3, lambda, digit}, iterations)

          {digit, List.last(fX), sX}
        end,
        max_concurrency: concurrency,
        timeout: 100_000
      )
      |> Enum.map(fn {:ok, {_d, _l, theta}} -> Matrex.to_list(theta) end)
      |> Matrex.new()

    # Visualize solutions
    # solutions
    # |> Matrex.to_list_of_lists()
    # |> Enum.each(&(Matrex.reshape(tl(&1), 28, 28) |> Matrex.heatmap()))

    # x_test = Matrex.load("test/data/Xtest.mtx.gz") |> Matrex.normalize()
    # x_test = Matrex.concat(Matrex.ones(x_test[:rows], 1), x_test)

    predictions =
      x
      |> Matrex.dot_nt(solutions)
      |> Matrex.apply(:sigmoid)

    # |> IO.inspect(label: "Predictions")

    # y_test = Matrex.load("test/data/Ytest.mtx")

    accuracy =
      1..predictions[:rows]
      |> Enum.reduce(0, fn row, acc ->
        if y[row] == predictions[row][:argmax] do
          acc + 1
        else
          # Show wrongful predictions
          # x[row][2..785] |> Matrex.reshape(28, 28) |> Matrex.heatmap()
          # IO.puts("#{y[row]} != #{predictions[row][:argmax]}")
          acc
        end
      end)
      |> Kernel./(predictions[:rows])
      |> Kernel.*(100)
      |> IO.inspect(label: "\rTraining set accuracy")

    time_elapsed = :timer.now_diff(:os.timestamp(), start_timestamp)
    IO.puts("Time elapsed: #{time_elapsed / 1_000_000} sec.")

    accuracy
  end

  @doc """
  Linear regression cost and gradient function, with no normalization.

  Computes the cost of using `theta` as the parameter for regularized linear regression and the
  gradient of the cost w.r.t. to the parameters.

  Compatible with `fmincg/4` algorithm from thise module.

  `theta`  — parameters, to compute cost for

  `X`  — training data input.

  `y`  — training data output.

  `lambda`  — regularization parameter.

  """
  @spec linear_cost_fun(Matrex.t(), {Matrex.t(), Matrex.t(), number, non_neg_integer}, pos_integer) ::
          {float, Matrex.t()}
  def linear_cost_fun(
        %Matrex{} = theta,
        {%Matrex{} = x, %Matrex{} = y, lambda} = _params,
        _iteration \\ 0
      ) when is_number(lambda) do
    n = Enum.count(y)
    h! = x |> Matrex.dot(theta)

    h_sub_y = Matrex.subtract(h!,y)

    j = 1.0/(2*n) *
      (Matrex.dot( h_sub_y |> Matrex.transpose, h_sub_y) |> Matrex.scalar())

    grad = x |> Matrex.transpose |> Matrex.dot(h_sub_y) |> Matrex.apply(& &1 * lambda / n)

    {j, grad}
  end

  @doc """
  Fit polynomial function based on given `x` and `y` using gradient descent (`fmincg/4`) using
  `linear_cost_fun/4`.

  This approach produces decent results for many datasets. It isn't as efficient as using
  least squares, but is still useful and provides a goode example of how to optimize general
  functions.

  Note that gradient descent won't always converge well for polynomials with linear cost function.
  If this happens for your dataset try adjusting `opts` parameters. Uses the `fmincg/4` algorithm
  from this module.

  `x`  — training data input.

  `y`  — training data output.

  `opts` - algorithm parameters
    `lambda`  — regularization parameter.
    `iterations`  — regularization parameter.

  """
  @spec fit_poly(Matrex.t(), Matrex.t(), pos_integer, keyword() ) ::
      %{ coefs: keyword(), error: float(), fun: (Matrex.t() -> Matrex.t()) }
  def fit_poly(x, y, degree, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 100)
    lambda  = Keyword.get(opts, :lambda, 1.0)

    {m, n} = Matrex.size(y)
    unless m >= n, do: raise %ArgumentError{message: "y shape (m,n) must have m > n"}

    xx =
      for i <- 0..degree, into: [] do
        x |> Matrex.apply(&:math.pow(&1, i))
      end |> Matrex.concat()

    theta = Matrex.zeros(degree + 1, 1)

    {sX, fX, _i} =  Algorithms.GradientDescent.fmincg(&linear_cost_fun/3, theta, {xx, y, lambda}, iterations)

    coefs = sX |> Enum.to_list() |> Enum.with_index(0) |> Enum.map(fn {x,y} -> {y,x} end)
    %{coefs: coefs, fun: &poly_func(&1, coefs), error: fX |> Enum.at(-1)}
  end

  @doc """
  Fit polynomial function based on given `x` and `y` using gradient descent (`fmincg/4`) using
  `linear_cost_fun/4`.

  This approach produces decent results for many datasets. It isn't as efficient as using
  least squares, but is still useful and provides a goode example of how to optimize general
  functions.

  Note that gradient descent won't always converge well for polynomials with linear cost function.
  If this happens for your dataset try adjusting `opts` parameters. Uses the `fmincg/4` algorithm
  from this module.

  `x`  — training data input.

  `y`  — training data output.

  `opts` - algorithm parameters
    `lambda`  — regularization parameter.
    `iterations`  — regularization parameter.

  """
  @spec fit_func(Matrex.t(), Matrex.t(), pos_integer, keyword() ) ::
      %{ coefs: keyword(), error: float(), fun: (Matrex.t() -> Matrex.t()) }
  def fit_func(x, y, fun_arrays, opts \\ []) do
    iterations = Keyword.get(opts, :iterations, 100)
    lambda = Keyword.get(opts, :lambda, 1.0)
    thetas = Keyword.get(opts, :thetas, Matrex.zeros(Enum.count(fun_arrays), 1))

    {m, n} = Matrex.size(y)
    unless m >= n, do: raise %ArgumentError{message: "y shape (m,n) must have m > n"}

    xx =
      for ith_fun <- fun_arrays, into: [] do
        x |> Matrex.apply(ith_fun)
      end |> Matrex.concat()

    {sX, fX, _i} =  Algorithms.GradientDescent.fmincg(&linear_cost_fun/3, thetas, {xx, y, lambda}, iterations)

    coefs = sX |> Enum.to_list() |> Enum.with_index(0) |> Enum.map(fn {x,y} -> {y,x} end)
    %{coefs: coefs, fun: &array_func(&1, fun_arrays, coefs), error: fX |> Enum.at(-1)}
  end

  def fit_linear(x, y, opts \\ []) do
    fit_poly(x, y, 1, opts)
  end

  def poly_func(x, coefs) when is_list(coefs) do
    x |> Enum.map(fn x ->
      Enum.reduce(coefs, 0.0, fn {i, c}, acc ->
        acc + c * :math.pow(x, i)
      end)
    end)
  end

  def array_func(x, fun_arrays, coefs) when is_list(coefs) do
    x |> Enum.map(fn x ->
      Enum.reduce(Enum.zip(coefs, fun_arrays), 0.0, fn {{_i, c}, fun}, acc ->
        acc + c * fun.(x)
      end)
    end)
  end

  @doc """
  Function of a surface with two hills.
  """
  @spec twin_peaks(float, float) :: float
  def twin_peaks(x, y) do
    x = (x - 40) / 4
    y = (y - 40) / 4

    :math.exp(-:math.pow(:math.pow(x - 4, 2) + :math.pow(y - 4, 2), 2) / 1000) +
      :math.exp(-:math.pow(:math.pow(x + 4, 2) + :math.pow(y + 4, 2), 2) / 1000) +
      0.1 * :math.exp(-:math.pow(:math.pow(x + 4, 2) + :math.pow(y + 4, 2), 2)) +
      0.1 * :math.exp(-:math.pow(:math.pow(x - 4, 2) + :math.pow(y - 4, 2), 2))
  end

  @doc """
  Computes sigmoid gradinet for the given matrix.


      g = sigmoid(X) * (1 - sigmoid(X))
  """
  @spec sigmoid_gradient(Matrex.t()) :: Matrex.t()
  def sigmoid_gradient(%Matrex{} = z) do
    s = Matrex.apply(z, :sigmoid)

    Matrex.multiply(s, Matrex.subtract(1, s))
  end

  @doc """
  Cost function for neural network with one hidden layer.

  Does delta computation in parallel.

  Ported from Andrew Ng's course, ex4.
  """
  @spec nn_cost_fun(
          Matrex.t(),
          {pos_integer, pos_integer, pos_integer, Matrex.t(), Matrex.t(), number},
          pos_integer
        ) :: {number, Matrex.t()}
  def nn_cost_fun(
        %Matrex{} = theta,
        {input_layer_size, hidden_layer_size, num_labels, x, y, lambda} = _params,
        _iteration \\ 0
      ) do
    alias Matrex, as: M

    theta1 =
      theta[1..(hidden_layer_size * (input_layer_size + 1))]
      |> M.reshape(hidden_layer_size, input_layer_size + 1)

    theta2 =
      theta[(hidden_layer_size * (input_layer_size + 1) + 1)..theta[:rows]]
      |> M.reshape(num_labels, hidden_layer_size + 1)

    # IO.write(IO.ANSI.home())
    #
    # data_side_size = trunc(:math.sqrt(theta1[:cols]))
    #
    # theta1
    # |> Matrex.submatrix(1..theta1[:rows], 2..theta1[:cols])
    # |> visual_net({5, 5}, {data_side_size, data_side_size})
    # |> Matrex.heatmap()

    m = x[:rows]

    x = M.concat(M.ones(m, 1), x)
    a2 = M.dot_nt(theta1, x) |> M.apply(:sigmoid)
    a2 = M.concat(M.ones(1, m), a2, :rows)
    a3 = M.dot_and_apply(theta2, a2, :sigmoid)

    y_b = M.zeros(num_labels, m)
    y_b = Enum.reduce(1..m, y_b, fn i, y_b -> M.set(y_b, trunc(y[i]), i, 1) end)

    c =
      M.neg(y_b)
      |> M.multiply(M.apply(a3, :log))
      |> M.subtract(M.multiply(M.subtract(1, y_b), M.apply(M.subtract(1, a3), :log)))

    theta1_sum =
      theta1
      |> M.submatrix(1..theta1[:rows], 2..theta1[:columns])
      |> M.square()
      |> M.sum()

    theta2_sum =
      theta2
      |> M.submatrix(1..theta2[:rows], 2..theta2[:columns])
      |> M.square()
      |> M.sum()

    reg = lambda / (2 * m) * (theta1_sum + theta2_sum)

    sum_c = M.sum(c)

    # Check for special sum_C value
    sum_c =
      if sum_c == :inf or sum_c == :nan do
        IO.inspect(sum_c, label: "Bad sum from a matrix")
        IO.inspect(c)
        1_000_000_000
      else
        sum_c
      end

    j = sum_c / m + reg

    # Compute gradients
    classes = M.reshape(1..num_labels, num_labels, 1)

    delta1_init = M.zeros(M.size(theta1))
    delta2_init = M.zeros(M.size(theta2))

    n_chunks = 5
    chunk_size = trunc(m / n_chunks)

    {delta1, delta2} =
      1..n_chunks
      |> Task.async_stream(fn n ->
        ((n - 1) * chunk_size + 1)..(n * chunk_size)
        |> Enum.reduce({delta1_init, delta2_init}, fn t, {delta1, delta2} ->
          a1 = M.transpose(x[t])
          z2 = M.dot(theta1, a1)
          a2 = M.concat(M.new([[1]]), M.apply(z2, :sigmoid), :rows)

          a3 = M.dot_and_apply(theta2, a2, :sigmoid)

          sigma3 = M.subtract(a3, M.apply(classes, &if(&1 == y[t], do: 1.0, else: 0.0)))

          sigma2 =
            theta2
            |> M.submatrix(1..theta2[:rows], 2..theta2[:cols])
            |> M.dot_tn(sigma3)
            |> M.multiply(sigmoid_gradient(z2))

          delta2 = M.add(delta2, M.dot_nt(sigma3, a2))
          delta1 = M.add(delta1, M.dot_nt(sigma2, a1))
          {delta1, delta2}
        end)
      end)
      |> Enum.reduce({delta1_init, delta2_init}, fn {:ok, {delta1, delta2}},
                                                    {delta1_result, delta2_result} ->
        {M.add(delta1_result, delta1), M.add(delta2_result, delta2)}
      end)

    theta1 = M.set_column(theta1, 1, M.zeros(hidden_layer_size, 1))
    theta2 = M.set_column(theta2, 1, M.zeros(num_labels, 1))
    theta1_grad = M.divide(delta1, m) |> M.add(M.multiply(lambda / m, theta1))
    theta2_grad = M.divide(delta2, m) |> M.add(M.multiply(lambda / m, theta2))

    theta = M.concat(M.to_row(theta1_grad), M.to_row(theta2_grad)) |> M.transpose()

    {j, theta}
  end

  @doc """
  Predict labels for the featurex with pre-trained neuron coefficients theta1 and theta2.
  """
  @spec nn_predict(Matrex.t(), Matrex.t(), Matrex.t()) :: Matrex.t()
  def nn_predict(theta1, theta2, x) do
    m = x[:rows]

    h1 =
      Matrex.concat(Matrex.ones(m, 1), x)
      |> Matrex.dot_nt(theta1)
      |> Matrex.apply(:sigmoid)

    Matrex.concat(Matrex.ones(m, 1), h1)
    |> Matrex.dot_nt(theta2)
    |> Matrex.apply(:sigmoid)
  end

  # Reshape each row of theta into a n_rows x n_cols matrix
  # Group these matrices into a rows x cols big matrix for visualization
  defp visual_net(theta, {rows, cols} = _visu_size, {n_rows, n_cols} = _neuron_size) do
    1..theta[:rows]
    |> Enum.map(&(theta[&1] |> Matrex.reshape(n_rows, n_cols)))
    |> Matrex.reshape(rows, cols)
  end

  @sample_side_size 20
  @input_layer_size @sample_side_size * @sample_side_size
  @hidden_layer_size 25
  @num_labels 10

  @doc """
  Run neural network with one hidden layer.


  """
  def run_nn(epsilon \\ 0.12, iterations \\ 100, lambdas \\ [0.1, 5, 50]) do
    start_timestamp = :os.timestamp()

    x = Matrex.load("test/data/X.mtx.gz")
    y = Matrex.load("test/data/Y.mtx")

    # {x_train, y_train, x_test, y_test} = split_data(x, y)

    {x_train, y_train, x_test, y_test} = {x, y, x, y}

    lambdas
    |> Task.async_stream(
      fn lambda ->
        initial_theta1 =
          random_weights(@input_layer_size, @hidden_layer_size, epsilon) |> Matrex.to_row()

        initial_theta2 =
          random_weights(@hidden_layer_size, @num_labels, epsilon) |> Matrex.to_row()

        initial_nn_params = Matrex.concat(initial_theta1, initial_theta2) |> Matrex.transpose()

        {sX, fX, _i} =
          Algorithms.GradientDescent.fmincg(
            &nn_cost_fun/3,
            initial_nn_params,
            {@input_layer_size, @hidden_layer_size, @num_labels, x_train, y_train, lambda},
            iterations
          )

        {lambda, List.last(fX), sX}
      end,
      timeout: 600_000,
      max_concurrency: 8
    )
    |> Enum.each(fn
      {:ok, {lambda, cost, sX}} ->
        # Unpack thetas from the found solution
        theta1 =
          sX[1..(@hidden_layer_size * (@input_layer_size + 1))]
          |> Matrex.reshape(@hidden_layer_size, @input_layer_size + 1)

        theta2 =
          sX[(@hidden_layer_size * (@input_layer_size + 1) + 1)..sX[:rows]]
          |> Matrex.reshape(@num_labels, @hidden_layer_size + 1)

        predictions = Matrex.Algorithms.nn_predict(theta1, theta2, x_test)

        1..predictions[:rows]
        |> Enum.reduce(0, fn row, acc ->
          if y_test[row] == predictions[row][:argmax] do
            acc + 1
          else
            # Show wrongful predictions
            # x[row][2..785] |> Matrex.reshape(28, 28) |> Matrex.heatmap()
            # IO.puts("#{y[row]} != #{predictions[row][:argmax]}")
            acc
          end
        end)
        |> Kernel./(predictions[:rows])
        |> Kernel.*(100)
        |> IO.inspect(label: "\rTraining set accuracy with lambda #{lambda} and cost #{cost}")

        theta1
        |> Matrex.submatrix(1..theta1[:rows], 2..theta1[:cols])
        |> visual_net({5, 5}, {@sample_side_size, @sample_side_size})
        |> Matrex.heatmap()

      _ ->
        :noop
    end)

    time_elapsed = :timer.now_diff(:os.timestamp(), start_timestamp)
    IO.puts("Time elapsed: #{time_elapsed / 1_000_000} sec.")
  end

  defp random_weights(l_in, l_out, epsilon) do
    Matrex.random(l_out, 1 + l_in)
    |> Matrex.multiply(2 * epsilon)
    |> Matrex.subtract(epsilon)
  end
end
