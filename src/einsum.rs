use itertools::Itertools;

use crate::{Ax, Tensor, TensorElement};
use std::{collections::HashMap, ops::Index};

#[derive(Debug)]
struct Equation {
    args: Vec<Vec<char>>,
    result: Vec<char>,
}

fn parse(equation: &str) -> Result<Equation, &'static str> {
    let mut args = vec![[].to_vec()];
    let mut result = vec![];
    let mut chars = equation.chars();
    let mut has_result = false;

    while let Some(c) = chars.next() {
        match c {
            ',' => {
                args.push(Vec::new());
            }
            '-' => {
                if Some('>') != chars.next() {
                    return Err("Invalid equation.");
                }
                has_result = true;
            }
            c if c.is_whitespace() => (),
            c => {
                if has_result {
                    result.push(c);
                } else {
                    args.last_mut().unwrap().push(c);
                }
            }
        }
    }

    args.iter_mut().for_each(|arg| arg.sort_unstable());

    if result.is_empty() {
        let mut counts: HashMap<char, usize> = HashMap::new();
        for indices in &args {
            for &c in indices {
                *counts.entry(c).or_insert(0) += 1;
            }
        }

        for (c, count) in counts.into_iter() {
            if count == 1 {
                result.push(c);
            }
        }

        result.sort_unstable();
    }

    Ok(Equation { args, result })
}

#[derive(Debug)]
pub enum EinsumResult<T: TensorElement> {
    Tensor(Tensor<T>),
    Scalar(T),
}

pub fn einsum2<T: TensorElement>(equation: &str, x: &Tensor<T>, y: &Tensor<T>) -> EinsumResult<T> {
    let eq = parse(equation).unwrap();
    assert_eq!(
        eq.args.len(),
        2,
        "Too many subscripts in equation. Expected 2 found {}",
        eq.args.len()
    );

    let shape1 = x.shape();
    let shape2 = y.shape();

    let indices1 = &eq.args[0];
    let indices2 = &eq.args[1];
    assert_eq!(indices1.len(), shape1.len(), "Invalid equation.");
    assert_eq!(indices2.len(), shape2.len(), "Invalid equation.");

    let ctoi1: HashMap<char, usize> = indices1.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let ctoi2: HashMap<char, usize> = indices2.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    let shape: Vec<usize> = eq
        .result
        .iter()
        .map(|&c| {
            ctoi1
                .get(&c)
                .map(|&i| shape1[i])
                .or_else(|| ctoi2.get(&c).map(|&i| shape2[i]))
                .unwrap()
        })
        .collect();

    let data: Vec<T> = shape
        .iter()
        .map(|&n| 0..n)
        .multi_cartesian_product()
        .map(|index| {
            let mut dims1 = vec![Ax::All; shape1.len()];
            let mut dims2 = vec![Ax::All; shape2.len()];

            for (i, &value) in index.iter().enumerate() {
                let c = &eq.result[i];
                if ctoi1.contains_key(c) {
                    dims1[ctoi1[c]] = Ax::Idx(value);
                }
                if ctoi2.contains_key(c) {
                    dims2[ctoi2[c]] = Ax::Idx(value);
                }
            }

            let slice1 = x.slice(&dims1);
            let slice2 = y.slice(&dims2);

            slice1.mul(&slice2).sum()
        })
        .collect();

    if data.len() == 1 {
        EinsumResult::Scalar(data[0])
    } else {
        EinsumResult::Tensor(Tensor::from_data(&data, &shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let eq = parse("ij,jk");
        println!("{:?}", eq);
    }
}
