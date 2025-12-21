use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
use std::rc::Rc;

pub trait TensorElement: Copy + Default + Debug {
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

pub enum Device {
    Cpu,
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

#[macro_export]
macro_rules! dims {
    ($($e:expr),* $(,)?) => {
        &[$(Ax::from($e)),*]
    };
}

#[derive(Debug)]
pub struct Tensor<T: TensorElement, const N: usize> {
    data: Rc<Storage<T>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    is_contiguous: bool,
    offset: usize,
}

impl<T: TensorElement, const N: usize> Tensor<T, N> {
    pub fn fill(value: T, shape: &[usize; N]) -> Self {
        assert_eq!(shape.len(), N);
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

    pub fn zeros(shape: &[usize; N]) -> Self {
        Self::fill(T::zero(), shape)
    }

    pub fn ones(shape: &[usize; N]) -> Self {
        Self::fill(T::one(), shape)
    }

    pub fn from_data(data: &[T], shape: &[usize; N]) -> Self {
        let strides = suffix_prod(shape);

        Self {
            data: Rc::new(Storage::new(data.to_vec(), Device::Cpu)),
            shape: shape.to_vec(),
            strides,
            is_contiguous: true,
            offset: 0,
        }
    }

    pub fn permute(&self, dims: &[usize; N]) -> Self {
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

    pub fn reshape<const M: usize>(&self, shape: &[usize; M]) -> Tensor<T, M> {
        assert_eq!(
            shape.iter().product::<usize>(),
            self.data.len(),
            "Invalid shape"
        );
        let strides = suffix_prod(shape);

        let data = if self.is_contiguous {
            Rc::clone(&self.data)
        } else {
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

    pub fn i(&self, index: &[usize; N]) -> T {
        assert!(
            index.iter().enumerate().all(|(i, n)| n < &self.shape[i]),
            "Index out of range. Invalid index {:?} for Tensor with shape {:?}.",
            &index,
            &self.shape
        );

        self.data.get(self.data_index(index))
    }

    pub fn set(&mut self, index: &[usize; N], value: T) {
        assert!(
            index.iter().enumerate().all(|(i, n)| n < &self.shape[i]),
            "Index out of range. Invalid index {:?} for Tensor with shape {:?}.",
            &index,
            &self.shape
        );

        self.data.set(self.data_index(index), value);
    }

    #[inline(always)]
    fn data_index(&self, index: &[usize; N]) -> usize {
        self.offset
            + index
                .iter()
                .enumerate()
                .map(|(i, n)| n * self.strides[i])
                .sum::<usize>()
    }

    pub fn slice<const M: usize>(&self, axes: &[Ax; N]) -> Tensor<T, M> {
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
}

impl<T: TensorElement, const N: usize> Display for Tensor<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn write_indent(f: &mut std::fmt::Formatter<'_>, indent: usize) -> std::fmt::Result {
            for _ in 0..indent {
                write!(f, "  ")?;
            }
            Ok(())
        }

        fn elem_str<T: TensorElement, const N: usize>(
            t: &Tensor<T, N>,
            flat_idx: usize,
            data_len: usize,
        ) -> String {
            if flat_idx < data_len {
                format!("{:?}", t.data.get(flat_idx))
            } else {
                "<?>".to_string()
            }
        }

        fn max_width_axis<T: TensorElement, const N: usize>(
            t: &Tensor<T, N>,
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

        fn fmt_axis<T: TensorElement, const N: usize>(
            t: &Tensor<T, N>,
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

// /**
//  * We want to impl all binary ops for these 4 combinations:
//  *
//  * lhs: Tensor, rhs: Tensor
//  * lhs: Tensor, rhs: &Tensor
//  * lhs: &Tensor, rhs: Tensor
//  * lhs: &Tensor, rhs: &Tensor
//  *
//  */
// macro_rules! impl_binary_op {
//     ($trait:ident, $op:tt, $method:ident) => {
//         impl<T: TensorElement> $trait for Tensor<T> {
//             type Output = Self;

//             fn $method(self, rhs: Tensor<T>) -> Self {
//                 assert!(
//                     ((self.shape.len() == rhs.shape.len())
//                         && self
//                             .shape
//                             .iter()
//                             .zip(other.shape.iter())
//                             .all(|(x, y)| x == y)),
//                     "Shapes don't match. {} != {}",
//                     self.shape,
//                     rhs.shape
//                 )

//             }
//         }
//     };
// }

// impl_binary_op!(Add, +, add);
