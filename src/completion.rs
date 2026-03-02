use anyhow::Result;
use opencv::core::{Mat, MatExprTraitConst, MatTrait, MatTraitConst, Scalar};

/**
Converts an image Matrix to an image Matrix with a border.
*/
fn border(src: &Mat) -> Result<Mat> {
    let top = 1;
    let bottom = 1;
    let left = 1;
    let right = 1;

    let mut dst = Mat::zeros(
        src.rows() + top + bottom,
        src.cols() + left + right,
        opencv::core::CV_8UC1,
    )?
    .to_mat()?;

    opencv::core::copy_make_border(
        src,
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
fn neighborhood(m: &Mat, cy: i32, cx: i32) -> Result<Vec<i32>> {
    let mut neighb = Vec::with_capacity(9);
    for i in -1..=1 {
        for j in -1..=1 {
            neighb.push(*m.at_2d::<u8>(cy + i, cx + j)? as i32);
        }
    }
    Ok(neighb)
}

/**
Computes the derivative of a given pixels neighborhood.
*/
fn derive(direction_matrix: [[i32; 3]; 3], neighborhood: &Vec<i32>) -> i32 {
    direction_matrix
        .iter()
        .flat_map(|row| row.iter())
        .zip(neighborhood.iter())
        .map(|(k, &pix)| k * pix)
        .sum()
}

/**
Converts an image Matrix to edge filter responses in the x and y directions.
*/
pub fn edge_detection(m: &Mat) -> Result<(Mat, Mat, Mat)> {
    let laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]];
    let sobel_horizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    let sobel_vertical = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    // Create bordered image (CV_8UC1)
    let bordered = border(m)?;

    // Output images same size as original
    let mut dst_laplacian = Mat::zeros(m.rows(), m.cols(), opencv::core::CV_8UC1)?.to_mat()?;
    let mut dst_sobel_x = Mat::zeros(m.rows(), m.cols(), opencv::core::CV_8UC1)?.to_mat()?;
    let mut dst_sobel_y = Mat::zeros(m.rows(), m.cols(), opencv::core::CV_8UC1)?.to_mat()?;

    for y in 0..m.rows() {
        for x in 0..m.cols() {
            let neighb = neighborhood(&bordered, y + 1, x + 1)?;

            let laplacian_der = derive(laplacian, &neighb);
            let sobel_x_der = derive(sobel_horizontal, &neighb);
            let sobel_y_der = derive(sobel_vertical, &neighb);

            let laplacian_u8 = (127 + laplacian_der).clamp(0, 255) as u8;
            let sobel_x_u8 = (127 + sobel_x_der).clamp(0, 255) as u8;
            let sobel_y_u8 = (127 + sobel_y_der).clamp(0, 255) as u8;

            *dst_laplacian.at_2d_mut::<u8>(y, x)? = laplacian_u8;
            *dst_sobel_x.at_2d_mut::<u8>(y, x)? = sobel_x_u8;
            *dst_sobel_y.at_2d_mut::<u8>(y, x)? = sobel_y_u8;
        }
    }

    Ok((dst_laplacian, dst_sobel_x, dst_sobel_y))
}
