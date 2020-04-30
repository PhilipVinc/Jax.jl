using NNlib

Jax.@jaxfunc NNlib.leakyrelu(x) = Core.jax.nn.leaky_relu(x)
Jax.@jaxfunc NNlib.celu(x) = Core.jax.nn.celu(x)
Jax.@jaxfunc NNlib.relu(x) = Core.jax.nn.relu(x)
Jax.@jaxfunc NNlib.selu(x) = Core.jax.nn.selu(x)
Jax.@jaxfunc NNlib.gelu(x) = Core.jax.nn.gelu(x)
Jax.@jaxfunc NNlib.elu(x) = Core.jax.nn.elu(x)
Jax.@jaxfunc NNlib.sigmoid(x) = Core.jax.nn.sigmoid(x)
Jax.@jaxfunc NNlib.logsigmoid(x) = Core.jax.nn.log_sigmoid(x)
Jax.@jaxfunc NNlib.swish(x) = Core.jax.nn.swish(x)
Jax.@jaxfunc NNlib.softplus(x) = Core.jax.nn.softplus(x)
Jax.@jaxfunc NNlib.softsign(x) = Core.jax.nn.soft_sign(x)
Jax.@jaxfunc NNlib.hardtanh(x) = Core.jax.nn.hard_tanh(x)

NNlib.softmax(x::AbstractJaxArray; dims=1) =
  jax.nn.softmax(x, axis=_convert_dims(x, dims))

NNlib.logsoftmax(x::AbstractJaxArray; dims=1) =
    jax.nn.log_softmax(x, axis=_convert_dims(x, dims))

NNlib.batched_mul(a::AbstractJaxArray, b::AbstractJaxArray) =
  Core.jax.lax.batch_matmul(b, a)
