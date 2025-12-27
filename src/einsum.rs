use crate::{Tensor, TensorElement};
use std::collections::HashMap;

pub fn parse(equation: &str) -> Result<(Vec<Vec<char>>, Vec<char>), &'static str> {
    let equation = equation.split_whitespace().collect::<String>();
    let mut parts = equation.split("->");
    let mut inputs = parts.next().ok_or("Invalid equation.")?.split(",");
    let first: Option<Vec<char>> = inputs.next().map(|s| s.chars().collect());
    let second: Option<Vec<char>> = inputs.next().map(|s| s.chars().collect());
    let output = parts
        .next()
        .map(|s| s.chars().collect())
        .unwrap_or_else(|| {
            let mut counts: HashMap<char, usize> = HashMap::new();
            if let Some(labels) = &first {
                for &label in labels.iter() {
                    *counts.entry(label).or_insert(0) += 1;
                }
            }
            if let Some(labels) = &second {
                for &label in labels.iter() {
                    *counts.entry(label).or_insert(0) += 1;
                }
            }

            let mut labels: Vec<_> = counts
                .iter()
                .filter(|&(_, &count)| count == 1)
                .map(|(&label, _)| label)
                .collect();

            labels.sort_unstable();

            labels
        });

    match (first, second) {
        (Some(x_labels), Some(y_labels)) => Ok((vec![x_labels, y_labels], output)),
        (Some(x_labels), None) => Ok((vec![x_labels], output)),
        (None, _) => Err("Invalid equation."),
    }
}

struct Alignment {
    new_shape: Vec<usize>,
    permutation: Vec<usize>,
    new_labels: Vec<char>,
}

pub fn einsum2<T: TensorElement>(
    x: &Tensor<T>,
    x_labels: &[char],
    y: &Tensor<T>,
    y_labels: &[char],
    out_labels: &[char],
) -> Tensor<T> {
    // Diagonalize `x` and `y` to handle self-contraction dims.
    let (x, x_labels) = diagonalize(x, x_labels);
    let (y, y_labels) = diagonalize(y, y_labels);

    // Collect all input and output labels in sorted order.
    let mut labels: Vec<char> = x_labels
        .to_vec()
        .into_iter()
        .chain(y_labels.to_vec())
        .collect();
    labels.sort_unstable();
    labels.dedup();

    let x_shape = x.shape();
    let y_shape = y.shape();

    let x_alignment = align(&x_labels, x_shape, &y_labels);
    let y_alignment = align(&y_labels, y_shape, &x_labels);

    let x = x
        .reshape(&x_alignment.new_shape)
        .permute(&x_alignment.permutation);

    let y = y
        .reshape(&y_alignment.new_shape)
        .permute(&y_alignment.permutation);

    // We only use x_alignment because new_labels is identical for both x and y at this point.
    let contraction_labels: Vec<&char> = x_alignment
        .new_labels
        .iter()
        .filter(|&label| !out_labels.contains(label))
        .collect();

    let contraction_dims: Vec<usize> = x_alignment
        .new_labels
        .iter()
        .enumerate()
        .filter(|(_, label)| contraction_labels.contains(label))
        .map(|(dim, _)| dim)
        .collect();

    let keep_dims: Vec<usize> = x_alignment
        .new_labels
        .iter()
        .enumerate()
        .filter(|(_, label)| !contraction_labels.contains(label))
        .map(|(dim, _)| dim)
        .collect();

    let keep_labels: Vec<char> = keep_dims
        .iter()
        .map(|&dim| x_alignment.new_labels[dim])
        .collect();

    let result = (x * y).sum_dims(&contraction_dims);

    // Find permutation order to match out_labels order again.
    let mut reorder_permutation: Vec<usize> = (0..keep_labels.len()).collect();
    reorder_permutation
        .sort_by_key(|&i| out_labels.iter().position(|label| label == &keep_labels[i]));

    result.permute(&reorder_permutation)
}

pub fn diagonalize<T: TensorElement>(
    tensor: &Tensor<T>,
    labels: &[char],
) -> (Tensor<T>, Vec<char>) {
    let shape = tensor.shape();
    let strides = tensor.strides();

    let mut label_to_dims: HashMap<char, Vec<usize>> = HashMap::new();

    for (dim, &label) in labels.iter().enumerate() {
        label_to_dims.entry(label).or_default().push(dim);
    }

    for (label, dims) in &label_to_dims {
        assert!(
            dims.iter().all(|&dim| shape[dim] == shape[dims[0]]),
            "All dimensions for label '{}' don't have the same size.",
            label
        );
    }

    let mut new_labels = Vec::new();
    let mut new_shape = Vec::new();
    let mut new_strides = Vec::new();

    for (label, size, stride) in label_to_dims.into_iter().map(|(label, dims)| {
        (
            label,
            shape[dims[0]],
            dims.iter().map(|&dim| strides[dim]).sum::<usize>(),
        )
    }) {
        new_labels.push(label);
        new_shape.push(size);
        new_strides.push(stride);
    }

    (tensor.restride(&new_strides, &new_shape), new_labels)
}

/// Aligns `x_labels` with `y_labels` and returns an `Alignment`` containing the new shape and permutation
/// required to achieve the alignment.
fn align(x_labels: &[char], x_shape: &[usize], y_labels: &[char]) -> Alignment {
    let x_label_counts = count(x_labels);
    let y_label_counts = count(y_labels);

    let mut new_labels = x_labels.to_vec();
    let mut new_shape = x_shape.to_vec();

    for (label, y_count) in y_label_counts {
        let x_count = *x_label_counts.get(&label).unwrap_or(&0);

        if y_count > x_count {
            let diff = y_count - x_count;
            new_labels.extend_from_slice(&vec![label; diff]);
            new_shape.extend_from_slice(&vec![1; diff]);
        }
    }

    let mut permutation: Vec<usize> = (0..new_shape.len()).collect();
    permutation.sort_by_key(|&i| new_labels[i]);

    new_labels.sort_unstable();

    Alignment {
        new_shape,
        permutation,
        new_labels,
    }
}

fn count(labels: &[char]) -> HashMap<char, usize> {
    let mut counts = HashMap::new();

    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }

    counts
}

// pub fn einsum2<T: TensorElement>(
//     x: &Tensor<T>,
//     x_labels: &[char],
//     y: &Tensor<T>,
//     y_labels: &[char],
//     out_labels: &[char],
// ) -> Tensor<T> {
//     let x_shape = x.shape();
//     let y_shape = y.shape();

//     let label_to_size: HashMap<char, usize> = x_labels
//         .iter()
//         .enumerate()
//         .map(|(i, &label)| (label, x_shape[i]))
//         .chain(
//             y_labels
//                 .iter()
//                 .enumerate()
//                 .map(|(j, &label)| (label, y_shape[j])),
//         )
//         .collect();

//     let shape: Vec<_> = out_labels
//         .iter()
//         .map(|label| label_to_size[label])
//         .collect();

//     let out_label_to_idx: HashMap<char, usize> = out_labels
//         .iter()
//         .enumerate()
//         .map(|(i, &label)| (label, i))
//         .collect();

//     let data: Vec<_> = IndexGen::new(
//         out_labels
//             .iter()
//             .map(|label| label_to_size[label])
//             .collect(),
//     )
//     .map_iter(|index| {
//         let x_index: Vec<_> = x_labels
//             .iter()
//             .map(|label| {
//                 if out_label_to_idx.contains_key(label) {
//                     Ax::Idx(index[out_label_to_idx[label]])
//                 } else {
//                     Ax::Range {
//                         start: Some(0),
//                         end: Some(label_to_size[label] - 1),
//                     }
//                 }
//             })
//             .collect();

//         let y_index: Vec<_> = y_labels
//             .iter()
//             .map(|label| {
//                 if out_label_to_idx.contains_key(label) {
//                     Ax::Idx(out_label_to_idx[label])
//                 } else {
//                     Ax::Range {
//                         start: Some(0),
//                         end: Some(label_to_size[label] - 1),
//                     }
//                 }
//             })
//             .collect();

//         let x_slice = x.slice(&x_index);
//         let y_slice = y.slice(&y_index);

//         let x_slice_labels: Vec<_> = x_labels
//             .iter()
//             .filter(|&label| !out_label_to_idx.contains_key(label))
//             .collect();
//         let y_slice_labels: Vec<_> = y_labels
//             .iter()
//             .filter(|&label| !out_label_to_idx.contains_key(label))
//             .collect();

//         let mut contractions: Vec<_> = x_slice_labels
//             .iter()
//             .chain(y_slice_labels.iter())
//             .map(|&&c| c)
//             .collect();

//         contractions.sort_unstable();
//         contractions.dedup();

//         let label_to_contraction_idx: HashMap<char, usize> = contractions
//             .iter()
//             .enumerate()
//             .map(|(i, &label)| (label, i))
//             .collect();

//         IndexGen::new(
//             contractions
//                 .iter()
//                 .map(|label| label_to_size[label])
//                 .collect(),
//         )
//         .map_iter(|contraction_index| {
//             let x_slice_index: Vec<_> = x_slice_labels
//                 .iter()
//                 .map(|&label| contraction_index[label_to_contraction_idx[label]])
//                 .collect();
//             let y_slice_index: Vec<_> = y_slice_labels
//                 .iter()
//                 .map(|&label| contraction_index[label_to_contraction_idx[label]])
//                 .collect();

//             x_slice.i(&x_slice_index) * y_slice.i(&y_slice_index)
//         })
//         .sum()
//     })
//     .collect();

//     Tensor::from_data(&data, &shape)
// }

#[macro_export]
macro_rules! einsum {
    ($equation:literal, $x:expr, $y:expr) => {{
        let (input_labels, out_labels) = parse($equation).unwrap();
        einsum2($x, &input_labels[0], $y, &input_labels[1], &out_labels)
    }};
    ($equation:literal, $x:expr) => {
        // einsum1($equation, $x)
        unimplemented!()
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let eq = parse("ij,jk");
        println!("{:?}", eq);
        let eq = parse("ii");
        println!("{:?}", eq);
        let eq = parse("aii,bii");
        println!("{:?}", eq);
    }
}
