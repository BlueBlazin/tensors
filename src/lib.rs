use std::cell::RefCell;
use std::fmt::{Debug, Display};
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

#[derive(Debug)]
pub struct Tensor<T: TensorElement, const DIMS: usize> {
    data: Rc<Storage<T>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    is_contiguous: bool,
}

impl<T: TensorElement, const DIMS: usize> Tensor<T, DIMS> {
    pub fn fill(value: T, shape: &[usize]) -> Self {
        assert_eq!(shape.len(), DIMS);
        let length = shape.iter().product();
        let strides = suffix_prod(shape);

        Self {
            data: Rc::new(Storage::new(vec![value; length], Device::Cpu)),
            shape: shape.to_vec(),
            strides,
            is_contiguous: true,
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
        }
    }

    pub fn permute(&self, dims: &[usize]) -> Self {
        let shape = dims.iter().map(|&i| self.shape[i]).collect();
        let strides = dims.iter().map(|&i| self.strides[i]).collect();

        Self {
            data: Rc::clone(&self.data),
            shape,
            strides,
            is_contiguous: false,
        }
    }

    pub fn reshape<const NEW_DIMS: usize>(&self, shape: &[usize; NEW_DIMS]) -> Tensor<T, NEW_DIMS> {
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
        }
    }

    pub fn i(&self, index: &[usize; DIMS]) -> T {
        assert!(
            index.iter().enumerate().all(|(i, n)| n < &self.shape[i]),
            "Index out of range. Invalid index {:?} for Tensor with shape {:?}.",
            &index,
            &self.shape
        );

        let data_index: usize = index
            .iter()
            .enumerate()
            .map(|(i, n)| n * self.strides[i])
            .sum();

        self.data.get(data_index)
    }

    pub fn set(&mut self, index: &[usize; DIMS], value: T) {
        assert!(
            index.iter().enumerate().all(|(i, n)| n < &self.shape[i]),
            "Index out of range. Invalid index {:?} for Tensor with shape {:?}.",
            &index,
            &self.shape
        );

        let data_index: usize = index
            .iter()
            .enumerate()
            .map(|(i, n)| n * self.strides[i])
            .sum();

        // self.data.set(data_index, value);
        self.data.set(data_index, value);
    }

    // pub fn add(&self, rhs: &Tensor<T>) -> Tensor<T> {
    //     assert!(
    //         self.shape == rhs.shape,
    //         "Shapes don't match. {:?} != {:?}",
    //         self.shape,
    //         rhs.shape
    //     );
    // }
}

impl<T: TensorElement, const DIMS: usize> Display for Tensor<T, DIMS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.data)
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
