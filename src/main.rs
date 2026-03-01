use opencv::{
    core::{
        AlgorithmHint, Mat, MatExprTraitConst, MatTraitConst, Rect, Vector, VectorToVec,
    },
    imgcodecs, imgproc,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let params = Vector::new();
    core(params)?;
    Ok(())
}

fn core(params: Vector<i32>) -> Result<(), Box<dyn Error>> {
    let flower_bgr = imgcodecs::imread("assets/Flower.jpg", imgcodecs::IMREAD_UNCHANGED)?;
    let flower_hsv = convert_to_hsv(&flower_bgr)?;

    let mut images = Vec::new();
    images.append(&mut split_channels(&flower_bgr)?);
    images.append(&mut split_channels(&flower_hsv)?);

    let large_image_height = 2;
    let large_image_width = 3;
    let large_image = create_large_image(&images, &large_image_height, &large_image_width)?;

    imgcodecs::imwrite("assets/Core1.jpg", &large_image, &params)?;

    Ok(())
}

/** 
Converts a BGR image Matrix to an HSV image Matrix
*/
fn convert_to_hsv(m: &Mat) -> Result<Mat, Box<dyn Error>> {
    let mut dst = Mat::default();
    let dst_cn = 0;
    let code = imgproc::COLOR_BGR2HSV;
    let algo = AlgorithmHint::ALGO_HINT_DEFAULT;
    imgproc::cvt_color(m, &mut dst, code, dst_cn, algo)?;
    Ok(dst)
}

/** 
Divides a multi-channel Matrix into several single-channel Matrices
*/
fn split_channels(m: &Mat) -> Result<Vec<Mat>, Box<dyn Error>> {
    let mut mv: Vector<Mat> = vec![Mat::default(); m.channels() as usize].into();
    opencv::core::split(m, &mut mv)?;
    Ok(mv.to_vec())
}

/** 
Creates a large image from of a grid of small same-sized images and a large base image
*/
fn create_large_image(
    images: &Vec<Mat>,
    large_image_height: &i32,
    large_image_width: &i32,
) -> Result<Mat, Box<dyn Error>> {
    let small_image = images.get(0).ok_or("Error: Images vector is empty")?;
    let height = small_image.rows() * large_image_height;
    let width = small_image.cols() * large_image_width;

    // Creates a zero-initialized single channel (CV_8UC1) Matrix 
    let mut large_image = Mat::zeros(height, width, opencv::core::CV_8UC1)?.to_mat()?;

    for (idx, m) in images.iter().enumerate() {
        let width = m.cols();
        let height = m.rows();

        let row = idx as i32 / large_image_width;
        let col = idx as i32 % large_image_width;

        let y = row * height;
        let x = col * width;

        // Creates a region of interest that a small image can be copied to
        let mut roi = Mat::roi_mut(&mut large_image, Rect::new(x, y, width, height))?;
        m.copy_to(&mut roi)?;
    }
    Ok(large_image)
}
