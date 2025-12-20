# Tensors in Rust

Basic tensor / ndarray library from scratch in Rust.

Build stuff to learn and understand 🙂

## Log

**Pre 2025-12-20**
- Implemented indexing using the `Index` trait.
- Tried to use Arc and RwLock for the data.
- Many changes upon changes as I kept realizing problems after going down a path and running into a wall.
- Used const generics to specify number of dims (rank) on `Tensor`.
- Tried to implement IndexMut and realized it's not going to be possible to do cleanly with the IndexMut trait.