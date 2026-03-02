use anyhow::{anyhow, Result};
use opencv::{
    core::{
        AlgorithmHint, Mat, MatExprTraitConst, MatTrait, MatTraitConst, Rect, Scalar, Vec3b,
        Vector, VectorToVec,
    },
    imgproc,
};

/**
Converts an image Matrix to an image Matrix of different color space.
*/
pub fn convert_matrix_color_space(m: &Mat, code: i32) -> Result<Mat> {
    let mut dst = Mat::default();
    let dst_cn = 0;
    let algo = AlgorithmHint::ALGO_HINT_DEFAULT;
    imgproc::cvt_color(m, &mut dst, code, dst_cn, algo)?;
    Ok(dst)
}

/**
Divides a multi-channel Matrix into several single-channel ones.
*/
pub fn split_channels(m: &Mat) -> Result<Vec<Mat>> {
    let mut mv: Vector<Mat> = vec![Mat::default(); m.channels() as usize].into();
    opencv::core::split(m, &mut mv)?;
    Ok(mv.to_vec())
}

/**
Multiplies a multi-channel Matrix by a scalar.
*/
fn multiply(src: &Mat, scalar: Scalar) -> Result<Mat> {
    let mut dst = Mat::default();
    opencv::core::multiply_def(src, &scalar, &mut dst)?;
    Ok(dst)
}

/**
Multiplies an HSV Matrix at each of its channels and returns a Vec for each channel.
*/
pub fn multiply_channel(m: &Mat) -> Result<(Vec<Mat>, Vec<Mat>, Vec<Mat>)> {
    let v = multiply_channel_triplet(m)?;
    Ok(unzip3(v))
}

/**
Multiplies an HSV Matrix by 0.0, 0.2, 0.4, 0.6, and 0.8 at each of its channels.
*/
fn multiply_channel_triplet(m: &Mat) -> Result<Vec<(Mat, Mat, Mat)>> {
    [0.0, 0.2, 0.4, 0.6, 0.8]
        .iter()
        .map(|&scalar| {
            let h = multiply(m, Scalar::new(scalar, 1.0, 1.0, 1.0))?;
            let s = multiply(m, Scalar::new(1.0, scalar, 1.0, 1.0))?;
            let v = multiply(m, Scalar::new(1.0, 1.0, scalar, 1.0))?;
            Ok((h, s, v))
        })
        .collect()
}

/**
Unzips a `Vec<(Mat, Mat, Mat)>` into three separate `Vec<Mat>`
*/
fn unzip3(v: Vec<(Mat, Mat, Mat)>) -> (Vec<Mat>, Vec<Mat>, Vec<Mat>) {
    let mut v1 = Vec::with_capacity(v.len());
    let mut v2 = Vec::with_capacity(v.len());
    let mut v3 = Vec::with_capacity(v.len());

    for (a, b, c) in v {
        v1.push(a);
        v2.push(b);
        v3.push(c);
    }

    (v1, v2, v3)
}

/**
Creates a large image from of a grid of small same-sized images and a large base image.
*/
pub fn create_large_image(
    images: &Vec<Mat>,
    large_image_height: i32,
    large_image_width: i32,
    typ: i32,
) -> Result<Mat> {
    let small_image = images
        .get(0)
        .ok_or_else(|| anyhow!("Err: Images is empty"))?;

    let height = small_image.rows() * large_image_height;
    let width = small_image.cols() * large_image_width;

    // Create a zero-initialized Matrix
    let mut large_image = Mat::zeros(height, width, typ)?.to_mat()?;

    for (idx, m) in images.iter().enumerate() {
        let height = m.rows();
        let width = m.cols();

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

/**
Converts an image Matrix to an image mask.
*/
pub fn euclidean_mask(m: &Mat) -> Result<Mat> {
    // Create a zero-initialized greyscale Matrix
    let mut dst = Mat::zeros(m.rows(), m.cols(), opencv::core::CV_8UC1)?.to_mat()?;

    let px_eighty = m.at_2d::<Vec3b>(80, 80)?;

    for y in 0..m.rows() {
        for x in 0..m.cols() {
            let px = m.at_2d::<Vec3b>(y, x)?;
            let norm = opencv::core::norm2(
                px,
                px_eighty,
                opencv::core::NORM_L2,
                &opencv::core::no_array(),
            )?;

            let dst_px = dst.at_2d_mut::<u8>(y, x)?;
            *dst_px = if norm < 100.0 { 255 } else { 0 }
        }
    }
    Ok(dst)
}

