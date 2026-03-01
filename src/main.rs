use anyhow::{anyhow, Result};
use opencv::{
    core::{
        AlgorithmHint, Mat, MatExprTraitConst, MatTraitConst, Rect, Vector, VectorToVec,
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
    let flower_hsv = convert_to_hsv(&flower_bgr)?;

    let flower_bgr_split = split_channels(&flower_bgr)?;
    let flower_hsv_split = split_channels(&flower_hsv)?;

    // Core 1
    let mut core_1 = Vec::new();
    core_1.extend(flower_bgr_split.clone());
    core_1.extend(flower_hsv_split.clone());

    let large = create_large_image(&core_1, 2, 3)?;
    imgcodecs::imwrite("assets/Core1.jpg", &large, &params)?;

    // Core 2
    // let mut core_2 = Vec::new();
    
    // merge_channels(Vector::from_slice());
    // core_2.extend(multiply_channel(&flower_hsv_split[0].clone())?); // Hue
    // core_2.extend(multiply_channel(&flower_hsv_split[1].clone())?); // Saturation
    // core_2.extend(multiply_channel(&flower_hsv_split[2].clone())?); // Value

    imgcodecs::imwrite("assets/Core2.jpg", &core_2[0], &params)?;

    Ok(())
}

/**
Converts a BGR image Matrix to an HSV image Matrix.
*/
fn convert_to_hsv(m: &Mat) -> Result<Mat> {
    let mut dst = Mat::default();
    let dst_cn = 0;
    let code = imgproc::COLOR_BGR2HSV;
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
 * 
Creates one multi-channel Matrix out of several single-channel ones.
*/
fn merge_channels(mv: &Vector<Mat>) -> Result<Mat> {
    let mut m = Mat::default();
    opencv::core::merge(mv, &mut m);
    Ok(m)
}

/**
Multiplies a single-channel Matrix by 0.0, 0.2, 0.4, 0.6, and 0.8.
*/
fn multiply_channel(m: &Mat) -> Result<Vec<Mat>> {
    [0.0, 0.2, 0.4, 0.6, 0.8]
        .iter()
        .map(|&mult| {
            (m * mult)
                .into_result()?
                .to_mat()
                .map_err(|e| anyhow!(e.to_string()))
        })
        .collect()
}

/**
Creates a large image from of a grid of small same-sized images and a large base image.
*/
fn create_large_image(
    images: &Vec<Mat>,
    large_image_height: i32,
    large_image_width: i32,
) -> Result<Mat> {
    let small_image = images
        .get(0)
        .ok_or_else(|| anyhow!("Error: Images vector is empty"))?;
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
