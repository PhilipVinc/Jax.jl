using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated

using Jax

#
N = 5000

# Classify MNIST digits with a simple multi-layer-perceptron
imgs = MNIST.images()[1:N]
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> tojax

labels = MNIST.labels()[1:N]
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9) |>collect |> tojax

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> tojax

loss(m, x, y) = crossentropy(m(x), y)

dataset = repeated((X, Y), 200)
opt = ADAM()

#

vals, funs = Jax.flatten(m)
loss(w::Tuple, x, y) = loss(Jax.unflatten(funs, w), x, y)
grad(vals, X, Y) = jax.grad(loss)(vals, X, Y)

function Flux.update!(opt, xs::Tuple, gs::Tuple)
  for (x,g) in zip(xs,gs)
    Flux.update!(opt, x, g)
  end
end

function do_update(vals, X, Y)
  g = grad(vals, X, Y)
  Flux.update!(opt, vals, g)
  return vals
end

do_update(vals, X, Y)

j_du = jax.jit(do_update)

for data=dataset
  global vals = j_du(vals, data...)
  show(loss(vals, data...))
end
