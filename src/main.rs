use minifb::{Window, WindowOptions};

fn main() {
    let width: usize = 1800;
    let height: usize = 1300;
    let center_x: f64 = -0.5;
    let center_y: f64 = 0.0;
    let zoom: f64 = 300.0;
    let max_iterations: u32 = 256;

    let mut buffer: Vec<u32> = vec![0; width * height];

    for y in 0..height {
        for x in 0..width {
            let cx = (x as f64 - width as f64 / 2.0) / zoom + center_x;
            let cy = (y as f64 - height as f64 / 2.0) / zoom + center_y;

            let mut zx: f64 = 0.0;
            let mut zy: f64 = 0.0;
            let mut iter: u32 = 0;

            while zx * zx + zy * zy <= 4.0 && iter < max_iterations {
                let xtemp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = xtemp;
                iter += 1;
            }

            let gray = 255 - (iter * 255 / max_iterations) as u32;
            let color: u32 = (gray << 16) | (gray << 8) | gray;
            buffer[y * width + x] = color;
        }
    }

    let mut window = Window::new("Mandelbrot", width, height, WindowOptions::default()).unwrap();

    while window.is_open() {
        window.update_with_buffer(&buffer, width, height).unwrap();
    }
}
