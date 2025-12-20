use std::fmt::{Debug, Display};
use std::ops::IndexMut;
use std::ops::{Add, Index};
use std::sync::{Arc, RwLock};

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

#[derive(Debug)]
pub struct Tensor<T: TensorElement, const DIMS: usize> {
    data: Arc<RwLock<Vec<T>>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T: TensorElement, const DIMS: usize> Tensor<T, DIMS> {
    pub fn fill(value: T, shape: &[usize]) -> Self {
        assert_eq!(shape.len(), DIMS);
        let length = shape.iter().product();
        let strides = suffix_prod(shape);

        Self {
            data: Arc::new(RwLock::new(vec![value; length])),
            shape: shape.to_vec(),
            strides,
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
            data: Arc::new(RwLock::new(data.to_vec())),
            shape: shape.to_vec(),
            strides,
        }
    }

    pub fn permute(&self, dims: &[usize]) -> Self {
        let shape = dims.iter().map(|&i| self.shape[i]).collect();
        let strides = dims.iter().map(|&i| self.strides[i]).collect();

        Self {
            data: Arc::clone(&self.data),
            shape,
            strides,
        }
    }

    pub fn reshape<const M: usize>(&self, shape: &[usize; M]) -> Tensor<T, M> {
        assert_eq!(
            shape.iter().product::<usize>(),
            self.data.read().unwrap().len(),
            "Invalid shape"
        );
        let strides = suffix_prod(shape);

        Tensor {
            data: Arc::clone(&self.data),
            shape: shape.to_vec(),
            strides,
        }
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

impl<T: TensorElement, const DIMS: usize> Index<[usize; DIMS]> for Tensor<T, DIMS> {
    type Output = T;

    fn index(&self, index: [usize; DIMS]) -> &Self::Output {
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

        &self.data.get_mut().unwrap()[data_index]
    }
}

impl<T: TensorElement> Index<usize> for Tensor<T, 1> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: TensorElement, const DIMS: usize> IndexMut<[usize; DIMS]> for Tensor<T, DIMS> {
    fn index_mut(&mut self, index: [usize; DIMS]) -> &mut Self::Output {
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

        &mut Arc::get_mut(&mut self.data).unwrap()[data_index]
    }
}

impl<T: TensorElement> IndexMut<usize> for Tensor<T, 1> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut Arc::get_mut(&mut self.data).unwrap()[index]
    }
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
