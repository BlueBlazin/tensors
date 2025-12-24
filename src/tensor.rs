use itertools::Itertools;
use rand::Rng;
use rand_distr::StandardNormal;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Range, RangeFrom, RangeFull, RangeTo, Sub};
use std::rc::Rc;

pub trait TensorElement:
    Copy
    + Default
    + Debug
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + Sum
{
    fn zero() -> Self;
    fn one() -> Self;
}

impl TensorElement for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

impl TensorElement for i32 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

fn suffix_prod(data: &[usize]) -> Vec<usize> {
    data.iter()
        .rev()
        .scan(1, |prod, &x| {
            let value = *prod;
            *prod *= x;
            Some(value)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
}

impl Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
        }
    }
}

#[derive(Debug)]
pub enum Storage<T: TensorElement> {
    Cpu(RefCell<Vec<T>>),
}

impl<T: TensorElement> Storage<T> {
    pub fn new(data: Vec<T>, device: Device) -> Self {
        match device {
            Device::Cpu => Storage::Cpu(RefCell::new(data)),
        }
    }

    pub fn get(&self, index: usize) -> T {
        match self {
            Self::Cpu(data) => data.borrow()[index],
        }
    }

    pub fn set(&self, index: usize, value: T) {
        match self {
            Self::Cpu(data) => {
                data.borrow_mut()[index] = value;
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(data) => data.borrow().len(),
        }
    }

    pub fn clone_storage(&self) -> Self {
        match self {
            Self::Cpu(data) => Self::Cpu(data.clone()),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Ax {
    Range {
        start: Option<usize>,
        end: Option<usize>,
    },
    Idx(usize),
    All,
}

impl From<usize> for Ax {
    fn from(value: usize) -> Self {
        Self::Idx(value)
    }
}
impl From<RangeFull> for Ax {
    fn from(_: RangeFull) -> Self {
        Self::All
    }
}
impl From<Range<usize>> for Ax {
    fn from(range: Range<usize>) -> Self {
        Self::Range {
            start: Some(range.start),
            end: Some(range.end),
        }
    }
}
impl From<RangeFrom<usize>> for Ax {
    fn from(range: RangeFrom<usize>) -> Self {
        Self::Range {
            start: Some(range.start),
            end: None,
        }
    }
}
impl From<RangeTo<usize>> for Ax {
    fn from(range: RangeTo<usize>) -> Self {
        Self::Range {
            start: None,
            end: Some(range.end),
        }
    }
}

fn broadcast_shape(shape: &[usize], other: &[usize]) -> Result<Vec<usize>, &'static str> {
    let m = shape.len();
    let n = other.len();
    let length = m.max(n);

    let mut new_shape = Vec::with_capacity(length);

    for i in 0..length {
        let x = m.checked_sub(length - i).map(|j| shape[j]).unwrap_or(1);
        let y = n.checked_sub(length - i).map(|j| other[j]).unwrap_or(1);

        if x == y || x == 1 || y == 1 {
            new_shape.push(x.max(y));
        } else {
            return Err("Operands could not be broadcast together.");
        }
    }

    Ok(new_shape)
}

fn broadcast_strides(strides: &[usize], shape: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let m = shape.len();
    let n = target_shape.len();
    let diff = m.abs_diff(n);

    let mut new_strides: Vec<usize> = shape
        .iter()
        .enumerate()
        .rev()
        .map(|(i, &x)| if x == 1 { 0 } else { strides[i] })
        .collect();

    if m > n {
        new_strides.extend_from_slice(&strides[0..diff]);
    } else {
        new_strides.extend_from_slice(&vec![0; diff]);
    }

    new_strides.reverse();
    new_strides
}

#[macro_export]
macro_rules! dims {
    ($($e:expr),* $(,)?) => {
        &[$(Ax::from($e)),*]
    };
}

macro_rules! binary_op {
    ($name:ident, $op:tt) => {
        pub fn $name(&self, other: &Tensor<T>) -> Tensor<T> {
            assert_eq!(
                self.data.device(),
                other.data.device(),
                "Devices do not match. {} != {}",
                self.data.device(),
                other.data.device(),
            );

            let shape = broadcast_shape(&self.shape, &other.shape).unwrap();

            let strides1 = broadcast_strides(&self.strides, &self.shape, &shape);
            let strides2 = broadcast_strides(&other.strides, &other.shape, &shape);

            let data: Vec<_> = shape
                .iter()
                .map(|&x| 0..x)
                .multi_cartesian_product()
                .map(|index| self.get(&index, &strides1) $op other.get(&index, &strides2))
                .collect();

            Tensor::from_data(&data, &shape)
        }
    };
}

pub enum DType {
    Float32,
    Int32,
}

#[derive(Debug)]
pub struct Tensor<T: TensorElement> {
    data: Rc<Storage<T>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    is_contiguous: bool,
    offset: usize,
}

impl<T: TensorElement> Tensor<T> {
    pub fn fill(value: T, shape: &[usize]) -> Self {
        // assert_eq!(shape.len(), N);
        let length = shape.iter().product();
        let strides = suffix_prod(shape);

        Self {
            data: Rc::new(Storage::new(vec![value; length], Device::Cpu)),
            shape: shape.to_vec(),
            strides,
            is_contiguous: true,
            offset: 0,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::fill(T::zero(), shape)
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::fill(T::one(), shape)
    }

    pub fn randn(shape: &[usize]) -> Tensor<f32> {
        let n = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..n)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect();

        Tensor::from_data(&data, shape)
    }

    pub fn from_data(data: &[T], shape: &[usize]) -> Self {
        let strides = suffix_prod(shape);

        Self {
            data: Rc::new(Storage::new(data.to_vec(), Device::Cpu)),
            shape: shape.to_vec(),
            strides,
            is_contiguous: true,
            offset: 0,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn permute(&self, dims: &[usize]) -> Self {
        let shape = dims.iter().map(|&i| self.shape[i]).collect();
        let strides = dims.iter().map(|&i| self.strides[i]).collect();

        Self {
            data: Rc::clone(&self.data),
            shape,
            strides,
            is_contiguous: false,
            offset: self.offset,
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor<T> {
        assert_eq!(
            shape.iter().product::<usize>(),
            self.data.len(),
            "Invalid shape"
        );
        let strides = suffix_prod(shape);

        let data = if self.is_contiguous {
            Rc::clone(&self.data)
        } else {
            // TODO: Instead of just cloning the data, we should clone and make it contiguous.
            // Current approach is a BUG.
            Rc::new(self.data.clone_storage())
        };

        Tensor {
            data,
            shape: shape.to_vec(),
            strides,
            is_contiguous: true,
            offset: self.offset,
        }
    }

    #[inline]
    pub fn i(&self, index: &[usize]) -> T {
        assert!(
            index.iter().enumerate().all(|(i, n)| n < &self.shape[i]),
            "Index out of range. Invalid index {:?} for Tensor with shape {:?}.",
            &index,
            &self.shape
        );

        self.get(index, &self.strides)
    }

    #[inline(always)]
    fn get(&self, index: &[usize], strides: &[usize]) -> T {
        self.data.get(self.broadcast_data_index(index, strides))
    }

    #[inline]
    pub fn set(&mut self, index: &[usize], value: T) {
        assert!(
            index.iter().enumerate().all(|(i, n)| n < &self.shape[i]),
            "Index out of range. Invalid index {:?} for Tensor with shape {:?}.",
            &index,
            &self.shape
        );

        self.data.set(self.data_index(index), value);
    }

    #[inline(always)]
    fn data_index(&self, index: &[usize]) -> usize {
        self.broadcast_data_index(index, &self.strides)
    }

    #[inline(always)]
    fn broadcast_data_index(&self, index: &[usize], strides: &[usize]) -> usize {
        self.offset
            + index
                .iter()
                .enumerate()
                .map(|(i, n)| n * strides[i])
                .sum::<usize>()
    }

    pub fn slice(&self, axes: &[Ax]) -> Tensor<T> {
        let mut shape = vec![];
        let mut offset = self.offset;
        let mut strides = vec![];

        for (idx, &x) in axes.iter().enumerate() {
            match x {
                Ax::Idx(n) => {
                    offset += n * self.strides[idx];
                }
                Ax::Range { start, end } => {
                    let stride = self.strides[idx];
                    let start = start.unwrap_or(0);
                    let end = end.unwrap_or(self.shape[idx] - 1);
                    let size = end - start + 1;

                    offset += start * stride;
                    strides.push(stride);
                    shape.push(size);
                }
                Ax::All => {
                    let stride = self.strides[idx];
                    let start = 0;
                    let end = self.shape[idx] - 1;
                    let size = end - start + 1;

                    strides.push(stride);
                    shape.push(size);
                }
            }
        }

        Tensor {
            data: Rc::clone(&self.data),
            shape,
            strides,
            is_contiguous: self.is_contiguous,
            offset,
        }
    }

    binary_op!(add, +);
    binary_op!(sub, -);
    binary_op!(mul, *);
    binary_op!(div, /);

    pub fn scalar_mul(&self, scalar: T) -> Tensor<T> {
        let data: Vec<_> = self
            .shape
            .iter()
            .map(|&x| 0..x)
            .multi_cartesian_product()
            .map(|index| self.i(&index) * scalar)
            .collect();

        Tensor::from_data(&data, &self.shape)
    }

    pub fn sum(&self) -> T {
        self.shape
            .iter()
            .map(|&x| 0..x)
            .multi_cartesian_product()
            .map(|index| self.i(&index))
            .sum()
    }
}

impl<T: TensorElement> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn write_indent(f: &mut std::fmt::Formatter<'_>, indent: usize) -> std::fmt::Result {
            for _ in 0..indent {
                write!(f, "  ")?;
            }
            Ok(())
        }

        fn elem_str<T: TensorElement>(t: &Tensor<T>, flat_idx: usize, data_len: usize) -> String {
            if flat_idx < data_len {
                format!("{:?}", t.data.get(flat_idx))
            } else {
                "<?>".to_string()
            }
        }

        fn max_width_axis<T: TensorElement>(
            t: &Tensor<T>,
            dim: usize,
            base: usize,
            data_len: usize,
        ) -> usize {
            let rank = t.shape.len();

            if dim >= rank {
                return elem_str(t, base, data_len).len();
            }

            let len = t.shape[dim];
            if len == 0 {
                return 0;
            }

            let mut best = 0usize;
            for i in 0..len {
                let idx = base + i * t.strides[dim];
                let w = max_width_axis(t, dim + 1, idx, data_len);
                if w > best {
                    best = w;
                }
            }
            best
        }

        fn fmt_axis<T: TensorElement>(
            t: &Tensor<T>,
            dim: usize,
            base: usize,
            indent: usize,
            data_len: usize,
            col_width: usize,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            let rank = t.shape.len();

            if dim >= rank {
                let s = elem_str(t, base, data_len);
                write!(f, "{:>width$}", s, width = col_width)?;
                return Ok(());
            }

            let len = t.shape[dim];
            if len == 0 {
                write!(f, "[]")?;
                return Ok(());
            }

            if dim == rank - 1 {
                write!(f, "[")?;
                for i in 0..len {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    let idx = base + i * t.strides[dim];
                    let s = elem_str(t, idx, data_len);
                    write!(f, "{:>width$}", s, width = col_width)?;
                }
                write!(f, "]")?;
                return Ok(());
            }

            write!(f, "[")?;
            for i in 0..len {
                if i == 0 {
                    write!(f, "\n")?;
                } else {
                    if dim < rank - 2 {
                        write!(f, ",\n\n")?;
                    } else {
                        write!(f, ",\n")?;
                    }
                }

                write_indent(f, indent + 1)?;
                let idx = base + i * t.strides[dim];
                fmt_axis(t, dim + 1, idx, indent + 1, data_len, col_width, f)?;
            }

            write!(f, "\n")?;
            write_indent(f, indent)?;
            write!(f, "]")?;
            Ok(())
        }

        let rank = self.shape.len();
        if rank != self.strides.len() {
            write!(
                f,
                "tensor(<invalid: shape={:?}, strides={:?}, offset={}>)",
                self.shape, self.strides, self.offset
            )?;
            return Ok(());
        }

        let data_len = self.data.len();
        let col_width = max_width_axis(self, 0, self.offset, data_len).max(1);

        write!(f, "tensor(")?;
        fmt_axis(self, 0, self.offset, 0, data_len, col_width, f)?;
        write!(f, ")")?;
        Ok(())
    }
}

/**
 * We want to impl all binary ops for these 4 combinations:
 *
 * lhs: Tensor, rhs: Tensor
 * lhs: Tensor, rhs: &Tensor
 * lhs: &Tensor, rhs: Tensor
 * lhs: &Tensor, rhs: &Tensor
 *
 */
macro_rules! impl_binary_op {
    ($trait:ident, $op:tt, $method:ident) => {
        impl<T: TensorElement> $trait<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                Tensor::$method(&self, rhs)
            }
        }

        impl<T: TensorElement> $trait<&Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                Tensor::$method(self, rhs)
            }
        }

        impl<T: TensorElement> $trait<Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                Tensor::$method(&self, &rhs)
            }
        }

        impl<T: TensorElement> $trait<Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                Tensor::$method(self, &rhs)
            }
        }
    };
}

impl_binary_op!(Add, +, add);
impl_binary_op!(Sub, -, sub);
impl_binary_op!(Mul, *, mul);
impl_binary_op!(Div, /, div);
