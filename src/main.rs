use std::{error::Error, fs};
use tensors::{Ax, Tensor, dims};

fn main() -> Result<(), Box<dyn Error>> {
    let data: Vec<f32> = fs::read_to_string("src/data.txt")?
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<Vec<_>, _>>()?;

    let x: Tensor<f32> = Tensor::from_data(&data, &[3, 3, 2]);

    let y: Tensor<f32> = Tensor::from_data(&vec![100.0, 200.0], &[2]);

    let z = x.div(&y);

    println!(
        "x.shape = {:?}, y.shape = {:?}, z.shape = {:?}",
        x.shape(),
        y.shape(),
        z.shape()
    );

    println!("{}", z);

    Ok(())
}
