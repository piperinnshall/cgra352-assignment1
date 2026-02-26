use std::error::Error;

use opencv::{
    core::{self, Mat, Vector},
    imgcodecs,
};


fn main() -> Result<(), Box<dyn Error>> {
    let source_img = imgcodecs::imread("car.png", imgcodecs::IMREAD_UNCHANGED)?;

    // Flipping image horizontally
    let mut destination_arr = Mat::default();
    core::flip(&source_img, &mut destination_arr, 1)?;

    // Creating an output image
    let arguments: Vector<i32> = Vector::new();
    imgcodecs::imwrite("final-output.png", &destination_arr, &arguments)?;
    Ok(())
}
