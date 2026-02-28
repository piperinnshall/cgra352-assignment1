use opencv::{
    core::{AlgorithmHint, Mat, MatTraitConst, Vector},
    imgcodecs, imgproc,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let params = Vector::<i32>::new();
    let flower_mat = imgcodecs::imread("assets/Flower.jpg", imgcodecs::IMREAD_UNCHANGED)?;
    let flower_mat_hsv = convert_to_hsv(&flower_mat)?;

    let width = flower_mat_hsv.cols() * 3;

    // imgcodecs::imwrite("hsv.jpg", &flower_mat_hsv, &params)?;
    Ok(())
}

fn convert_to_hsv(src: &Mat) -> Result<Mat, Box<dyn Error>> {
    let mut dst = Mat::default();
    imgproc::cvt_color(
        src,
        &mut dst,
        imgproc::COLOR_BGR2HSV,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(dst)
}

fn split_bgr(src: &Mat) -> Result<(Mat, Mat, Mat), Box<dyn Error>> {
    let mut channels: Vector<Mat> = vec![Mat::default(); src.channels() as usize].into();
    opencv::core::split(src, &mut channels)?;
    Ok((channels.get(0)?, channels.get(1)?, channels.get(2)?))
}
