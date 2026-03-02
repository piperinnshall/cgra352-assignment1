use anyhow::Result;
use opencv::core::{Mat, MatExprTraitConst, MatTrait, MatTraitConst, MatTraitConstManual, MatTraitManual, Scalar};

/**
Converts an image Matrix to an image Matrix with a border.
*/
fn border(src: &Mat) -> Result<Mat> {
    let top = 1;
    let bottom = 1;
    let left = 1;
    let right = 1;

    // Create a new matrix with extra border and type CV_32F
    let mut dst = Mat::zeros(
        src.rows() + top + bottom,
        src.cols() + left + right,
        opencv::core::CV_32FC1,
    )?
    .to_mat()?;

    // Convert source to f32 if it's not already
    let mut src_f32 = Mat::default();
    src.convert_to(&mut src_f32, opencv::core::CV_32F, 1.0, 0.0)?;

    // Add border using reflection
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

    // Iterate over 3x3 neighborhood
    for i in -1..=1 {
        for j in -1..=1 {
            // Get pixel value as f32
            let val = *m.at_2d::<f32>(cy + i, cx + j)?;
            neighb.push(val);
        }
    }

    Ok(neighb)
}

/**
Computes the derivative of a given pixels neighborhood.
*/
fn derive(direction_matrix: [[f32; 3]; 3], neighborhood: &Vec<f32>) -> f32 {
    direction_matrix
        .iter()
        .flat_map(|row| row.iter())
        .zip(neighborhood.iter())
        .map(|(k, &pix)| k * pix)
        .sum()
}

fn normalize_to_u8(src: &Mat) -> Result<Mat> {
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

    // Create the output CV_8U matrix
    let mut dst = Mat::zeros(src.rows(), src.cols(), opencv::core::CV_8UC1)?.to_mat()?;
    
    if max_val - min_val > 0.0 {
        let src_slice = src.data_typed::<f32>()?;
        let dst_slice = dst.data_typed_mut::<u8>()?;

        // Iterate over src/dst slices and normalize pixels
        src_slice
            .iter()
            .zip(dst_slice.iter_mut())
            .for_each(|(&src_pixel, dst_pixel)| {
                *dst_pixel = (((src_pixel - min_val) / (max_val - min_val) * 255.0)
                    .clamp(0.0, 255.0)) as u8;
            });
    }

    Ok(dst)
}


/**
Converts an image Matrix to edge filter responses in the x and y directions.
*/
pub fn edge_detection(m: &Mat) -> Result<(Mat, Mat, Mat)> {
    let laplacian = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]];
    let sobel_horizontal = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let sobel_vertical = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    // Create bordered image (CV_32FC1)
    let bordered = border(m)?;

    // Output images same size as original
    let mut dst_laplacian = Mat::zeros(m.rows(), m.cols(), opencv::core::CV_32FC1)?.to_mat()?;
    let mut dst_sobel_x = Mat::zeros(m.rows(), m.cols(), opencv::core::CV_32FC1)?.to_mat()?;
    let mut dst_sobel_y = Mat::zeros(m.rows(), m.cols(), opencv::core::CV_32FC1)?.to_mat()?;

    for y in 0..m.rows() {
        for x in 0..m.cols() {
            // Compute neighborhood
            let neighb = neighborhood(&bordered, y + 1, x + 1)?;

            // Compute derivatives
            let laplacian_der = derive(laplacian, &neighb);
            let sobel_x_der = derive(sobel_horizontal, &neighb);
            let sobel_y_der = derive(sobel_vertical, &neighb);

            *dst_laplacian.at_2d_mut::<f32>(y, x)? = laplacian_der;
            *dst_sobel_x.at_2d_mut::<f32>(y, x)? = sobel_x_der;
            *dst_sobel_y.at_2d_mut::<f32>(y, x)? = sobel_y_der;
        }
    }

    // Normalize all outputs to 0..255 and convert to CV_8U
    let dst_laplacian_u8 = normalize_to_u8(&dst_laplacian)?;
    let dst_sobel_x_u8 = normalize_to_u8(&dst_sobel_x)?;
    let dst_sobel_y_u8 = normalize_to_u8(&dst_sobel_y)?;

    Ok((dst_laplacian_u8, dst_sobel_x_u8, dst_sobel_y_u8))
}
