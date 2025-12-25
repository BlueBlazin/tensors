use crate::{Ax, Tensor, TensorElement};
use itertools::Itertools;
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

            counts
                .iter()
                .filter(|&(_, &count)| count == 1)
                .map(|(&label, _)| label)
                .sorted()
                .collect()
        });

    match (first, second) {
        (Some(x_labels), Some(y_labels)) => Ok((vec![x_labels, y_labels], output)),
        (Some(x_labels), None) => Ok((vec![x_labels], output)),
        (None, _) => Err("Invalid equation."),
    }
}

pub fn einsum2<T: TensorElement>(
    x: &Tensor<T>,
    x_labels: &[char],
    y: &Tensor<T>,
    y_labels: &[char],
    out_labels: &[char],
) -> Tensor<T> {
    let x_shape = x.shape();
    let y_shape = y.shape();

    let label_to_size: HashMap<char, usize> = x_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, x_shape[i]))
        .chain(
            y_labels
                .iter()
                .enumerate()
                .map(|(j, &label)| (label, y_shape[j])),
        )
        .collect();

    let shape: Vec<_> = out_labels
        .iter()
        .map(|label| label_to_size[label])
        .collect();

    let out_label_to_idx: HashMap<char, usize> = out_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    let data: Vec<_> = out_labels
        .iter()
        .map(|label| 0..label_to_size[label])
        .multi_cartesian_product()
        .map(|index| {
            let x_index: Vec<_> = x_labels
                .iter()
                .map(|label| {
                    if out_label_to_idx.contains_key(label) {
                        Ax::Idx(index[out_label_to_idx[label]])
                    } else {
                        Ax::Range {
                            start: Some(0),
                            end: Some(label_to_size[label] - 1),
                        }
                    }
                })
                .collect();

            let y_index: Vec<_> = y_labels
                .iter()
                .map(|label| {
                    if out_label_to_idx.contains_key(label) {
                        Ax::Idx(out_label_to_idx[label])
                    } else {
                        Ax::Range {
                            start: Some(0),
                            end: Some(label_to_size[label] - 1),
                        }
                    }
                })
                .collect();

            let x_slice = x.slice(&x_index);
            let y_slice = y.slice(&y_index);

            let x_slice_labels: Vec<_> = x_labels
                .iter()
                .filter(|&label| !out_label_to_idx.contains_key(label))
                .collect();
            let y_slice_labels: Vec<_> = y_labels
                .iter()
                .filter(|&label| !out_label_to_idx.contains_key(label))
                .collect();

            let contractions: Vec<_> = x_slice_labels
                .iter()
                .chain(y_slice_labels.iter())
                .map(|&&c| c)
                .sorted()
                .dedup()
                .collect();

            let label_to_contraction_idx: HashMap<char, usize> = contractions
                .iter()
                .enumerate()
                .map(|(i, &label)| (label, i))
                .collect();

            contractions
                .iter()
                .map(|label| 0..label_to_size[label])
                .multi_cartesian_product()
                .map(|contraction_index| {
                    let x_slice_index: Vec<_> = x_slice_labels
                        .iter()
                        .map(|&label| contraction_index[label_to_contraction_idx[label]])
                        .collect();
                    let y_slice_index: Vec<_> = y_slice_labels
                        .iter()
                        .map(|&label| contraction_index[label_to_contraction_idx[label]])
                        .collect();

                    x_slice.i(&x_slice_index) * y_slice.i(&y_slice_index)
                })
                .sum()
        })
        .collect();

    Tensor::from_data(&data, &shape)
}

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
