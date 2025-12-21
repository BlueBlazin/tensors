use std::{error::Error, fs};
use tensors::{Ax, Tensor, dims};

fn main() -> Result<(), Box<dyn Error>> {
    let data: Vec<f32> = fs::read_to_string("src/data.txt")?
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<Vec<_>, _>>()?;

    let x: Tensor<f32, 3> = Tensor::from_data(&data, &[3, 3, 2]);
    let y: Tensor<f32, 3> = x.slice(dims![1.., 1.., ..]);
    println!("{}", y);
    let mut z: Tensor<f32, 2> = y.slice(dims![1.., 1.., 0]);
    z.set(&[0, 0], 42.0);
    println!("{}", y);

    Ok(())
}
