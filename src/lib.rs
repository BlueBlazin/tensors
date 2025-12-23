use std::cell::RefCell;
use std::fmt::{Debug, Display};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Broadcast {
    Split(usize),
    Splat(usize),
}

fn broadcast_shape(shape: &[usize], other: &[usize]) -> Vec<usize> {
    let m = shape.len();
    let n = other.len();
    let diff = m.abs_diff(n);

    let mut new_shape: Vec<usize> = shape
        .iter()
        .rev()
        .zip(other.iter().rev())
        .map(|(&x, &y)| x.max(y))
        .collect();

    if m > n {
        new_shape.extend_from_slice(&shape[0..diff]);
    } else {
        new_shape.extend_from_slice(&other[0..diff]);
    }

    new_shape.reverse();
    new_shape
}

fn broadcast_actions(shape: &[usize], other: &[usize]) -> Result<Vec<Broadcast>, &'static str> {
    let m = shape.len();
    let n = other.len();

    let mut actions = vec![];

    let mut i: isize = m as isize - 1;
    let mut j: isize = n as isize - 1;
    while i >= 0 && j >= 0 {
        if shape[i as usize] == other[j as usize] || other[j as usize] == 1 {
            actions.push(Broadcast::Split(shape[i as usize]));
        } else if shape[i as usize] == 1 {
            actions.push(Broadcast::Splat(other[j as usize]));
        } else {
            return Err("Operands could not be broadcast together.");
        }

        i -= 1;
        j -= 1;
    }

    while i >= 0 {
        actions.push(Broadcast::Split(shape[i as usize]));
        i -= 1;
    }

    while j >= 0 {
        actions.push(Broadcast::Splat(other[j as usize]));
        j -= 1;
    }

    actions.reverse();
    Ok(actions)
}

/// Uses the splits and splats from a previous step to generate ranges/segments
/// which broadcast the correct parts from the flat data array. The sum of the lengths
/// of these ranges equals the max flat data length of both Tensors being broadcast.
fn generate_segments(actions: &[Broadcast], length: usize) -> Vec<(usize, usize)> {
    let mut segments = vec![(0, length - 1)];
    for &action in actions {
        let next_segments = segments
            .iter()
            .flat_map(|(start, end)| match action {
                Broadcast::Split(n) => {
                    let size = (end - start + 1) / n;
                    (0..n)
                        .map(|i| (start + i * size, start + i * size + size - 1))
                        .collect::<Vec<_>>()
                }
                Broadcast::Splat(n) => std::iter::repeat((*start, *end)).take(n).collect(),
            })
            .collect();

        segments = next_segments;
    }

    segments
}

#[macro_export]
macro_rules! dims {
    ($($e:expr),* $(,)?) => {
        &[$(Ax::from($e)),*]
    };
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

    pub fn i(&self, index: &[usize]) -> T {
        assert!(
            index.iter().enumerate().all(|(i, n)| n < &self.shape[i]),
            "Index out of range. Invalid index {:?} for Tensor with shape {:?}.",
            &index,
            &self.shape
        );

        self.data.get(self.data_index(index))
    }

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
        self.offset
            + index
                .iter()
                .enumerate()
                .map(|(i, n)| n * self.strides[i])
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

    fn broadcast(&self, other_shape: &[usize]) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let m = self.data.len();
        let n = other_shape.iter().product();

        let self_actions = broadcast_actions(&self.shape, other_shape).unwrap();
        let mut self_segments = generate_segments(&self_actions, m);

        let other_actions = broadcast_actions(other_shape, &self.shape).unwrap();
        let mut other_segments = generate_segments(&other_actions, n);

        let mut self_chunks = vec![];
        let mut other_chunks = vec![];

        let mut start = 0;
        let end = m.max(n);

        let mut i = 0;
        let mut j = 0;

        assert_eq!(self_segments[0].0, 0);
        assert_eq!(other_segments[0].0, 0);

        while start < end {
            let (left1, right1) = self_segments[i];
            let (left2, right2) = other_segments[j];

            let length1 = right1 - left1 + 1;
            let length2 = right2 - left2 + 1;

            if length1 < length2 {
                self_chunks.push((left1, right1));
                other_chunks.push((left2, left2 + length1 - 1));
                other_segments[j].0 = left2 + length1 - 1;
                i += 1;
                start += length1;
            } else if length1 > length2 {
                other_chunks.push((left2, right2));
                self_chunks.push((left1, left1 + length2 - 1));
                self_segments[i].0 = left1 + length2 - 1;
                j += 1;
                start += length2;
            } else {
                self_chunks.push((left1, right1));
                other_chunks.push((left2, right2));
                i += 1;
                j += 1;
                start += length1;
            }
        }

        (self_chunks, other_chunks)
    }

    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        assert_eq!(
            self.data.device(),
            other.data.device(),
            "Devices do not match. {} != {}",
            self.data.device(),
            other.data.device(),
        );

        let (chunks1, chunks2) = self.broadcast(&other.shape);

        let data: Vec<_> = chunks1
            .into_iter()
            .zip(chunks2.into_iter())
            .flat_map(|((start1, end1), (start2, end2))| {
                (start1..end1 + 1)
                    .zip(start2..end2 + 1)
                    .map(|(i, j)| self.data.get(i) + other.data.get(j))
            })
            .collect();

        let shape = broadcast_shape(&self.shape, &other.shape);

        Tensor::from_data(&data, &shape)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_actions() {
        let actions = broadcast_actions(&[2], &[3, 3, 2]);
        let expected = vec![
            Broadcast::Splat(3),
            Broadcast::Splat(3),
            Broadcast::Split(2),
        ];

        assert_eq!(Ok(expected), actions);
    }
}
