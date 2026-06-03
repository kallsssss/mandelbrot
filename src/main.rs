use minifb::{Key, Window, WindowOptions};

fn main() {
    let width: usize = 1200;
    let height: usize = 800;
    let mut center_x: f64 = -0.5;
    let mut center_y: f64 = 0.0;
    let mut zoom: f64 = 300.0;
    let max_iterations: u32 = 256;

    let mut buffer: Vec<u32> = vec![0; width * height];
    let mut window = Window::new("Mandelbrot", width, height, WindowOptions::default()).unwrap();

    let mut needs_render = true;

    while window.is_open() {
        //key presses
        if let Some((mx, my)) = window.get_scroll_wheel() {
            let mouse_cx = (mx as f64 - width as f64 / 2.0) / zoom + center_x;
            let mouse_cy = (my as f64 - height as f64 / 2.0) / zoom + center_y;

            zoom *= 1.1;

            let new_cx = (mx as f64 - width as f64 / 2.0) / zoom + center_x;
            let new_cy = (my as f64 - height as f64 / 2.0) / zoom + center_y;

            center_x += mouse_cx - new_cx;
            center_y += mouse_cy - new_cy;

            needs_render = true;
        }
        let pan_speed = 50.0 / zoom; // pan slower when zoomed in

        if window.is_key_down(Key::Left) {
            center_x -= pan_speed;
            needs_render = true;
        }
        if window.is_key_down(Key::Right) {
            center_x += pan_speed;
            needs_render = true;
        }
        if window.is_key_down(Key::Up) {
            center_y -= pan_speed;
            needs_render = true;
        }
        if window.is_key_down(Key::Down) {
            center_y += pan_speed;
            needs_render = true;
        }

        // window draw loop
        if needs_render {
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

                    let color = if iter == max_iterations {
                        0x00000000 // inside the set = black
                    } else {
                        let t = iter as f64 / max_iterations as f64;
                        let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u32;
                        let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u32;
                        let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u32;
                        (r << 16) | (g << 8) | b
                    };
                    buffer[y * width + x] = color;
                }
            }
            needs_render = false;
        }

        //draw window
        window.update_with_buffer(&buffer, width, height).unwrap();
    }
}
