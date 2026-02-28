use opencv::{
    core::{AlgorithmHint, Mat, MatTraitConst, Vector},
    imgcodecs, imgproc,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let params = Vector::new();
    let flower_img = imgcodecs::imread("assets/Flower.jpg", imgcodecs::IMREAD_UNCHANGED)?;

    let hsv_img = convert_to_hsv(&flower_img)?;
    let width = hsv_img.cols() * 3;

    imgcodecs::imwrite("hsv.jpg", &hsv_img, &params)?;

    Ok(())
}

// fn split_bgr(mat: &Mat

fn convert_to_hsv(material: &Mat) -> Result<Mat, Box<dyn Error>> {
    let mut dest_material = Mat::default();
    imgproc::cvt_color(
        material,
        &mut dest_material,
        imgproc::COLOR_BGR2HSV,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(dest_material)
}
