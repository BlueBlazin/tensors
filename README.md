# Tensors in Rust

Basic tensor / ndarray library from scratch in Rust.

Build stuff to learn and understand 🙂

## Indexing

Because the actual `data` stored is a flat vector, an index like `x[1, 3, 0, 1]` needs to be converted to a single `data_index` which maps to the correct position in `data`.

This is where strides come in. Each stride tells you how many places to jump for each index. For a 2d array of dims `rows x cols` and index `(row, col)`, the formula is well known: `data_index` = `row * cols + col` or `row * cols + col * 1` . The strides in the simple 2d case are just `[cols, 1]`.

Playing around with this idea you can convince yourself that for a tensor with shape `[l, m, n]` the strides are `[m * n, n, 1]`. Or for one with shape `[u, v, w, x]` they are `[v * w * x, w * x, x, 1]` and so on.

Then an index like `[0, 1, 2, 3]` gets mapped to `(v * w * x) * 0 + (w * x) * 1 + (x) * 2 + (1) * 3`.

### Reshape

Implementing `reshape` is easy. As long as the product of the new shape matches the size of the flat data vector, we can just recompute the `strides` and modify the `shape`.

### Permute

To implement `permute` we just a permute the `shape` and `strides` according to the permutation provided.


## Slices

Slicing a tensor should not copy the underlying data. Only the shape and strides should change to match the expectations of the slice. The key insight for implementing slicing is realizing that strides do not need to be modified at all.

Consider the example of `y = x[2:, 3, :, 1]`. Indexing `y` like this:
```py
assert x.shape == (6, 6, 4, 4)
assert y.shape == (4, 4)
y[1, 0]
```
is the same as indexing `x` like:
```py
x[2 + 1, 3, 0 + 0, 1]
```
This works for slices of slices too:
```py
z = y[1:, :4]
assert z.shape == (3, 4)
z[0, 1]
```
Is the same as:
```py
y[1 + 0, 0 + 1]
```
which is the same as:
```py
x[2 + (1 + 0), 3, 0 + (0 + 1), 1]
```

So the problem of slicing is actually just the problem of mapping a slice's index to the original tensor's index. But we don't want to unwrap the whole chain of slices to find the correct `data_index`. This is where the concept of an `offset` comes into play.

First let's categorize the two types of arguments you can pass in for each axis when creating a slice:
1. A number, e.g. `0`, `1`, `7`, etc.
2. A range, e.g. `a:b`, `a:`, `:b`, `:`.

### Number Arguments

These collapse an axis of the original tensor being sliced. Ok, what does having a collapsed axis imply? Take an index `[x, y, 7]` for a tensor with strides `[a, b, c]`. Here `x` and `y` are variables and `7`, `a`, `b`, and `c` are constants. The `data_index` will be calculated as described in the Indexing section:
```py
x * a + y * b + 7 * c
```
And because only `x` and `y` can change and the third axis has collapsed for this slice, this calculation will always be:
```py
x * a + y * b + offset
```

So, on each `Tensor` we can store an `offset` and every time we want to index we add this offset to the calculation. For an unsliced / original `Tensor`, `offset` will of course always be `0`.

### Range Arguments

Ranges don't collapse an axis but rather just limit the number of values available and potentially add a per-axis offset when a start is present (`3:`, `2:5`, etc.). If the end of the range is present, that is taken care of by the new shape. But as for the start offset, the next key insight is that this can be merged into the global `offset` from before.

Consider again `[x, y, 7]` with strides `[a, b, c]` for a slice such as `[2:, :, 7]`. Now `data_index` will be:
```py
(2 + x) * a + y * b + 7 * c
```
But this reduces to:
```py
x * a + 2 * a + y * b + 7 * c
= (x * a + y * b) + (2 * a + 7 * c)
= x * a + y * b + offset
```
The local offset of `2` for axis 0 is just a `2 * a` contribution to the global `offset`. This way, using just one offset per `Tensor`, we can handle slices.

## Broadcasting

Two tensors can be broadcasted to the same shape if, starting from the rightmost axis:
1. The sizes match.
2. One of the sizes is 1.

In this case we can broadcast the axes with size 1 to the required size. For example:
```py
x.shape ==    [2, 1, 2]
y.shape == [3, 1, 4, 2]

result.shape == [3, 2, 4, 2]
```
Missing axes (after right aligning) are treated as having size 1. The implementation of this turned out to be surprisingly easy. We just need to set the corresponding strides to 0. In the above example:
```py
x.strides == [2, 2, 1]

broadcast_strides(x.strides) == [0, 2, 0, 1]
```

## Log

**2025-12-24**
- Implement elementwise sub, mul, div.
- Implement scalar multiplication.

**2025-12-23**
- After coming with the very complex approach for broadcasting I gave up and asked the LLMs. I did have the right idea and split-splat correctly does what broadcasting should, but it's too explicit. Turns out broadcasting is extremely easy: just use a stride of 0 for those axes which need to be broadcast.
- Implement an `add` method on `Tensor`.

**2025-12-22**
- Tried implementing broadcasting by myself.
- I came up with an algorithm I'm calling split-splat-merge which creates a list of ranges whose lengths sum to the length of the flat data of the bigger tensor.

**2025-12-21**
- Tried implementing slicing with a new `Stride { start, value }` struct and a Tensor level offset for fixed indices.
- Realized that adding start on Stride is awkward.
- Had the even greater realization that slicing was entirely independent of modifying strides and only needs index modifying.
- Finally realized why only a Tensor level offset is required and per-stride (or per-index) start offsets are not (commutativity).
- Implement slicing.
- Implement better `Display` for `Tensor` with LLM help (this is the only LLM generated code because it was a very laborious task).


**2025-12-20**
- Moved to a simple `Storage` enum that can eventually support multiple devices (cpu, gpu, mps, etc.)
- Got rid of `Index` and `IndexMut` traits. Now indexing into and modifying the Tensor is via methods.

**Pre 2025-12-20**
- Implemented indexing using the `Index` trait.
- Tried to use Arc and RwLock for the data.
- Many changes upon changes as I kept realizing problems after going down a path and running into a wall.
- Used const generics to specify number of dims (rank) on `Tensor`.
- Tried to implement IndexMut and realized it's not going to be possible to do cleanly with the IndexMut trait.