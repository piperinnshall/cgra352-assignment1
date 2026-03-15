use anyhow::Result;
use opencv::core::{Mat, MatTrait, MatTraitConst, MatTraitConstManual, MatTraitManual, Scalar};

/**
Converts an image Matrix to an image Matrix with a border.
*/
fn border(src: &Mat) -> Result<Mat> {
    let top = 1;
    let bottom = 1;
    let left = 1;
    let right = 1;

    // Convert src to f32
    let mut src_f32 = Mat::default();
    src.convert_to(&mut src_f32, opencv::core::CV_32F, 1.0, 0.0)?;

    // Create output matrix by OpenCV
    let mut dst = Mat::default();
    opencv::core::copy_make_border(
        &src_f32,
        &mut dst,
        top,
        bottom,
        left,
        right,
        opencv::core::BORDER_REFLECT_101,
        Scalar::default(),
    )?;

    Ok(dst)
}

/**
Computes the neighborhood of a given center pixel.
*/
fn neighborhood(m: &Mat, cy: i32, cx: i32) -> Result<Vec<f32>> {
    let mut neighb = Vec::with_capacity(9);
    for i in -1..=1 {
        for j in -1..=1 {
            let val = *m.at_2d::<f32>(cy + i, cx + j)?;
            neighb.push(val);
        }
    }
    Ok(neighb)
}

/**
Computes the derivative of a given pixel's neighborhood.
*/
fn derive(kernel: [[f32; 3]; 3], neighborhood: &Vec<f32>) -> f32 {
    kernel
        .iter()
        .flat_map(|row| row.iter())
        .zip(neighborhood.iter())
        .map(|(k, &pix)| k * pix)
        .sum()
}

/**
Applies a 3x3 kernel to a matrix
*/
fn apply_kernel(m: &Mat, kernel: [[f32; 3]; 3]) -> Result<Mat> {
    let bordered = border(m)?;
    let mut dst = Mat::new_rows_cols_with_default(
        m.rows(),
        m.cols(),
        opencv::core::CV_32FC1,
        Scalar::default(),
    )?;

    for y in 0..m.rows() {
        for x in 0..m.cols() {
            let neighb = neighborhood(&bordered, y + 1, x + 1)?;
            *dst.at_2d_mut::<f32>(y, x)? = derive(kernel, &neighb);
        }
    }

    Ok(dst)
}

/**
Normalizes a CV_32F matrix to 0..255 CV_8U
*/
fn normalize(src: &Mat) -> Result<Mat> {
    let mut min_val = 0.0;
    let mut max_val = 0.0;
    opencv::core::min_max_loc(
        src,
        Some(&mut min_val),
        Some(&mut max_val),
        None,
        None,
        &Mat::default(),
    )?;
    let min_val = min_val as f32;
    let max_val = max_val as f32;
    // Allocate CV_8U output
    let mut dst = Mat::new_rows_cols_with_default(
        src.rows(),
        src.cols(),
        opencv::core::CV_8UC1,
        Scalar::default(),
    )?;
    if max_val - min_val > 0.0 {
        let src_slice = src.data_typed::<f32>()?;
        let dst_slice = dst.data_typed_mut::<u8>()?;

        // Iterate over src/dst slices and normalize pixels
        src_slice
            .iter()
            .zip(dst_slice.iter_mut())
            .for_each(|(&src_pixel, dst_pixel)| {
                // Current min-max approach
                // *dst_pixel =
                //     (((src_pixel - min_val) / (max_val - min_val) * 255.0).clamp(0.0, 255.0)) as u8;
                // Zero-centred approach 
                let abs_max = min_val.abs().max(max_val.abs());
                *dst_pixel = ((src_pixel / abs_max) * 127.0 + 127.0).clamp(0.0, 255.0) as u8;
            });
    }
    Ok(dst)
}

/**
  Main edge detection function
  */
pub fn edge_detection(m: &Mat) -> Result<(Mat, Mat, Mat)> {
    let laplacian = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]];
    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    let dst_laplacian = normalize(&apply_kernel(m, laplacian)?)?;
    let dst_sobel_x = normalize(&apply_kernel(m, sobel_x)?)?;
    let dst_sobel_y = normalize(&apply_kernel(m, sobel_y)?)?;

    Ok((dst_laplacian, dst_sobel_x, dst_sobel_y))
}
