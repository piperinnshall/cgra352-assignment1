use anyhow::{anyhow, Result};
use opencv::{
    core::{
        AlgorithmHint, Mat, MatExprTraitConst, MatTraitConst, Rect, Scalar, Vector, VectorToVec,
    },
    imgcodecs, imgproc,
};

fn main() -> Result<()> {
    let params = Vector::new();
    core(params)?;
    Ok(())
}

fn core(params: Vector<i32>) -> Result<()> {
    let flower_bgr = imgcodecs::imread("assets/Flower.jpg", imgcodecs::IMREAD_UNCHANGED)?;
    let flower_hsv = convert_matrix(&flower_bgr, imgproc::COLOR_BGR2HSV)?;

    // Core 1
    let mut core_1 = Vec::new();
    core_1.append(&mut split_channels(&flower_bgr)?);
    core_1.append(&mut split_channels(&flower_hsv)?);

    let core_1_large = create_large_image(&core_1, 2, 3, opencv::core::CV_8UC1)?;
    imgcodecs::imwrite("assets/Core1.jpg", &core_1_large, &params)?;

    // Core 2
    let mut core_2 = Vec::new();
    let mut flower_hsv_split = unzip3(multiply_channel(&flower_hsv)?);
    core_2.append(&mut flower_hsv_split.0);
    core_2.append(&mut flower_hsv_split.1);
    core_2.append(&mut flower_hsv_split.2);

    let core_2_large = create_large_image(&core_2, 3, 5, opencv::core::CV_8UC3)?;
    imgcodecs::imwrite("assets/Core2.png", &convert_matrix(&core_2_large, imgproc::COLOR_HSV2BGR)?, &params)?;

    Ok(())
}

/**
Converts an image Matrix to an image Matrix of different type.
*/
fn convert_matrix(m: &Mat, code: i32) -> Result<Mat> {
    let mut dst = Mat::default();
    let dst_cn = 0;
    let algo = AlgorithmHint::ALGO_HINT_DEFAULT;
    imgproc::cvt_color(m, &mut dst, code, dst_cn, algo)?;
    Ok(dst)
}

/**
Divides a multi-channel Matrix into several single-channel ones.
*/
fn split_channels(m: &Mat) -> Result<Vec<Mat>> {
    let mut mv: Vector<Mat> = vec![Mat::default(); m.channels() as usize].into();
    opencv::core::split(m, &mut mv)?;
    Ok(mv.to_vec())
}

/**
Multiplies a multi-channel Matrix by a scalar.
*/
pub fn multiply(src: &Mat, scalar: Scalar) -> Result<Mat> {
    let mut dst = Mat::default();
    opencv::core::multiply_def(src, &scalar, &mut dst)?;
    Ok(dst)
}

/**
Multiplies an HSV Matrix by 0.0, 0.2, 0.4, 0.6, and 0.8 at each of its channels.
*/
fn multiply_channel(m: &Mat) -> Result<Vec<(Mat, Mat, Mat)>> {
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
fn create_large_image(
    images: &Vec<Mat>,
    large_image_height: i32,
    large_image_width: i32,
    typ: i32
) -> Result<Mat> {
    let small_image = images
        .get(0)
        .ok_or_else(|| anyhow!("Error: Images vector is empty"))?;
    let height = small_image.rows() * large_image_height;
    let width = small_image.cols() * large_image_width;

    // Creates a zero-initialized Matrix
    let mut large_image = Mat::zeros(height, width, typ)?.to_mat()?;

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
