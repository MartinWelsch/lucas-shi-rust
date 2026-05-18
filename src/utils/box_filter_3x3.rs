use image::{ImageBuffer, Luma};

pub(crate) fn box_filter_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    // Apply the separable 3x3 box filter: first across rows, then across columns
    box_filter_horizontal_3x3_in_place(image);
    box_filter_vertical_3x3_in_place(image);
}

fn box_filter_horizontal_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = (image.width(), image.height());

    for y in 0..height {
        // Initialize the sum for the first pixel in the row
        let mut sum = image.get_pixel(0, y)[0] as i32 + image.get_pixel(1, y)[0] as i32;
        let mut count = 2;

        // Process the first pixel in the row
        if width > 2 {
            sum += image.get_pixel(2, y)[0] as i32;
            count += 1;
        }
        image.put_pixel(0, y, Luma([(sum / count) as i16]));

        // Sliding window for the remaining pixels
        for x in 1..width {
            // Remove the left pixel from the sum
            if x > 1 {
                sum -= image.get_pixel(x - 2, y)[0] as i32;
                count -= 1;
            }

            // Add the right pixel to the sum
            if x + 1 < width {
                sum += image.get_pixel(x + 1, y)[0] as i32;
                count += 1;
            }

            // Store the result in the current pixel
            image.put_pixel(x, y, Luma([(sum / count) as i16]));
        }
    }
}

fn box_filter_vertical_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = (image.width(), image.height());

    for x in 0..width {
        // Initialize the sum for the first pixel in the column
        let mut sum = image.get_pixel(x, 0)[0] as i32 + image.get_pixel(x, 1)[0] as i32;
        let mut count = 2;

        // Process the first pixel in the column
        if height > 2 {
            sum += image.get_pixel(x, 2)[0] as i32;
            count += 1;
        }
        image.put_pixel(x, 0, Luma([(sum / count) as i16]));

        // Sliding window for the remaining pixels
        for y in 1..height {
            // Remove the top pixel from the sum
            if y > 1 {
                sum -= image.get_pixel(x, y - 2)[0] as i32;
                count -= 1;
            }

            // Add the bottom pixel to the sum
            if y + 1 < height {
                sum += image.get_pixel(x, y + 1)[0] as i32;
                count += 1;
            }

            // Store the result in the current pixel
            image.put_pixel(x, y, Luma([(sum / count) as i16]));
        }
    }
}
