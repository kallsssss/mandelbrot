// Mandelbrot explorer — compute shader edition (deep zoom · resizable · WASD)
// GPU: AMD RX 570 (or any Vulkan/DX12/Metal device) via wgpu

use bytemuck::{Pod, Zeroable};
use minifb::{Key, MouseMode, Window, WindowOptions};
use wgpu::util::DeviceExt;

// ── initial window size ───────────────────────────────────────────────────────

const INIT_WIDTH:  usize = 1200;
const INIT_HEIGHT: usize = 800;

// ── GPU parameter block ───────────────────────────────────────────────────────

/// Center stored as double-float (hi + lo, each f32).  True value ≈ hi + lo.
/// Layout is byte-identical to the WGSL Params struct.  32 bytes, no padding.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    cx_hi:    f32,   // re(center) — high word
    cx_lo:    f32,   // re(center) — low word
    cy_hi:    f32,   // im(center) — high word
    cy_lo:    f32,   // im(center) — low word
    zoom:     f32,   // pixels per world unit
    width:    u32,
    height:   u32,
    max_iter: u32,
}

/// Dekker decomposition: split an f64 into two f32 so that hi + lo == x
/// (to f64 precision).  The pair has effectively ~48 bits of mantissa.
#[inline]
fn split_df(x: f64) -> (f32, f32) {
    let hi = x as f32;
    let lo = (x - hi as f64) as f32;
    (hi, lo)
}

// ── WGSL compute shader ───────────────────────────────────────────────────────

const SHADER: &str = r#"
struct Params {
    cx_hi    : f32,
    cx_lo    : f32,
    cy_hi    : f32,
    cy_lo    : f32,
    zoom     : f32,
    width    : u32,
    height   : u32,
    max_iter : u32,
}

@group(0) @binding(0) var<uniform>             params : Params;
@group(0) @binding(1) var<storage, read_write> pixels : array<u32>;

// ── Double-float (df) helpers ─────────────────────────────────────────────────
//
// A df value is vec2<f32>(hi, lo); the exact real is hi + lo.
// Algorithms: Knuth two-sum (exact error-free addition) and
// Veltkamp/Dekker two-product (exact error-free multiplication).
//
// Veltkamp split constant for 24-bit f32 mantissa: 2^12 + 1 = 4097.

fn df_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    return vec2<f32>(s, (a - (s - v)) + (b - v));
}

fn df_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p  = a * b;
    let ca = 4097.0 * a;  let ah = ca - (ca - a);  let al = a - ah;
    let cb = 4097.0 * b;  let bh = cb - (cb - b);  let bl = b - bh;
    return vec2<f32>(p, ((ah * bh - p) + ah * bl) + (al * bh + al * bl));
}

fn df_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = df_two_sum(a.x, b.x);
    return df_two_sum(s.x, s.y + a.y + b.y);
}

fn df_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return df_add(a, vec2<f32>(-b.x, -b.y));
}

fn df_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let t = df_two_prod(a.x, b.x);
    return df_two_sum(t.x, t.y + a.x * b.y + a.y * b.x);
}

// Add a plain f32 scalar into a df value
fn df_add_f(a: vec2<f32>, b: f32) -> vec2<f32> {
    let s = df_two_sum(a.x, b);
    return vec2<f32>(s.x, s.y + a.y);
}

// ── Main kernel ───────────────────────────────────────────────────────────────

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }

    // Pixel offset from image centre — plain f32 is fine (range ≈ ±max(w,h)/2)
    let px = f32(gid.x) - f32(params.width)  * 0.5;
    let py = f32(gid.y) - f32(params.height) * 0.5;

    // C for this pixel = centre + offset/zoom, computed in double-float so deep
    // zoom locations remain sharp even when px/zoom is tiny.
    let cx: vec2<f32> = df_add_f(vec2<f32>(params.cx_hi, params.cx_lo), px / params.zoom);
    let cy: vec2<f32> = df_add_f(vec2<f32>(params.cy_hi, params.cy_lo), py / params.zoom);

    // Mandelbrot iteration in full double-float (zx, zy are df values)
    var zx = vec2<f32>(0.0, 0.0);
    var zy = vec2<f32>(0.0, 0.0);
    var n  = 0u;

    while n < params.max_iter {
        // Escape test on hi components only — fast and accurate enough
        if zx.x * zx.x + zy.x * zy.x > 4.0 { break; }

        // z → z² + C  in double-float:
        //   new_zx = zx² - zy² + cx
        //   new_zy = 2·zx·zy  + cy
        let zx2    = df_mul(zx, zx);
        let zy2    = df_mul(zy, zy);
        let cross  = df_mul(zx, zy);
        let zx_new = df_add(df_sub(zx2, zy2), cx);
        let zy_new = df_add(df_add(cross, cross), cy);  // df_add(cross,cross) = 2·cross
        zx = zx_new;
        zy = zy_new;
        n += 1u;
    }

    var col: u32;
    if n == params.max_iter {
        col = 0u;   // inside the set → black
    } else {
        // Bernstein-polynomial smooth-colouring palette
        let t = f32(n) / f32(params.max_iter);
        let r = u32(9.0  * (1.0 - t) * t * t * t             * 255.0);
        let g = u32(15.0 * (1.0 - t) * (1.0 - t) * t * t     * 255.0);
        let b = u32(8.5  * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0);
        col = (r << 16u) | (g << 8u) | b;
    }
    pixels[gid.y * params.width + gid.x] = col;
}
"#;

// ── GPU wrapper ───────────────────────────────────────────────────────────────

struct Gpu {
    device:      wgpu::Device,
    queue:       wgpu::Queue,
    pipeline:    wgpu::ComputePipeline,
    bgl:         wgpu::BindGroupLayout,  // kept so resize() can rebuild the bind group
    uniform_buf: wgpu::Buffer,           // UNIFORM | COPY_DST   — written each frame
    output_buf:  wgpu::Buffer,           // STORAGE | COPY_SRC   — filled by shader
    staging_buf: wgpu::Buffer,           // MAP_READ | COPY_DST  — CPU readback
    bind_group:  wgpu::BindGroup,
}

impl Gpu {
    fn new(width: usize, height: usize) -> Self {
        pollster::block_on(Self::init(width, height))
    }

    async fn init(width: usize, height: usize) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            })
            .await
            .expect(
                "No GPU adapter found.\n\
                 On Arch: ensure mesa / vulkan-radeon / amdvlk is installed,\n\
                 or run with WGPU_BACKEND=gl for the software fallback.",
            );

        let info = adapter.get_info();
        println!("Adapter : {} ({:?})", info.name, info.backend);
        println!("Driver  : {}", info.driver_info);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to open wgpu device");

        // ── shader & pipeline ─────────────────────────────────────────────

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("mandelbrot"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:  Some("mandelbrot"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("pl"),
                bind_group_layouts:   &[&bgl],
                push_constant_ranges: &[],
            })),
            module:              &shader,
            entry_point:         "cs_main",
            compilation_options: Default::default(),
            cache:               None,
        });

        // ── uniform buffer ────────────────────────────────────────────────

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("uniform"),
            contents: bytemuck::bytes_of(&GpuParams {
                cx_hi: -0.5, cx_lo: 0.0,
                cy_hi:  0.0, cy_lo: 0.0,
                zoom:   300.0,
                width:  width  as u32,
                height: height as u32,
                max_iter: 256,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── size-dependent buffers ─────────────────────────────────────────

        let (output_buf, staging_buf) = Self::make_pixel_bufs(&device, width, height);

        let bind_group = Self::make_bind_group(&device, &bgl, &uniform_buf, &output_buf);

        Gpu { device, queue, pipeline, bgl, uniform_buf, output_buf, staging_buf, bind_group }
    }

    fn make_pixel_bufs(
        device: &wgpu::Device,
        width: usize,
        height: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let n = (width * height * 4) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("output"),
            size:               n,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("staging"),
            size:               n,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (output, staging)
    }

    fn make_bind_group(
        device:      &wgpu::Device,
        bgl:         &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        output_buf:  &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("bg"),
            layout:  bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buf.as_entire_binding()  },
            ],
        })
    }

    /// Recreate size-dependent buffers and bind group after a window resize.
    fn resize(&mut self, width: usize, height: usize) {
        let (output_buf, staging_buf) = Self::make_pixel_bufs(&self.device, width, height);
        self.bind_group = Self::make_bind_group(
            &self.device, &self.bgl, &self.uniform_buf, &output_buf,
        );
        self.output_buf  = output_buf;
        self.staging_buf = staging_buf;
    }

    /// Upload params → dispatch shader → synchronous GPU→CPU readback.
    fn render(&self, cpu_buf: &mut [u32], p: &GpuParams, width: usize, height: usize) {
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(p));

        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("frame") },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("mandelbrot"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // 16×16 workgroups, rounded up to cover the full image
            pass.dispatch_workgroups(
                (width  as u32 + 15) / 16,
                (height as u32 + 15) / 16,
                1,
            );
        }
        enc.copy_buffer_to_buffer(
            &self.output_buf, 0,
            &self.staging_buf, 0,
            (width * height * 4) as u64,
        );
        self.queue.submit(Some(enc.finish()));

        // Block until the GPU is done, then copy to CPU memory
        let slice = self.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("GPU buffer map failed");
        {
            let view = slice.get_mapped_range();
            cpu_buf.copy_from_slice(bytemuck::cast_slice(&view));
        }
        self.staging_buf.unmap();
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    // Keep the view state in f64 on the CPU to avoid rounding drift during pan/zoom.
    // Only cast to f32 (or df pair) when uploading to the GPU.
    let mut cx:   f64 = -0.5;
    let mut cy:   f64 =  0.0;
    let mut zoom: f64 =  300.0;

    let mut width  = INIT_WIDTH;
    let mut height = INIT_HEIGHT;

    let mut gpu   = Gpu::new(width, height);
    let mut buf   = vec![0u32; width * height];
    let mut dirty = true;

    // resize: true  →  the user (and Hyprland) can drag the window corner.
    let mut window = Window::new(
        "Mandelbrot — deep zoom",
        width,
        height,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .expect("Window creation failed");

    while window.is_open() && !window.is_key_down(Key::Escape) {

        // ── window resize ──────────────────────────────────────────────────
        let (nw, nh) = window.get_size();
        if (nw, nh) != (width, height) && nw > 0 && nh > 0 {
            width  = nw;
            height = nh;
            gpu.resize(width, height);
            buf.resize(width * height, 0);
            dirty = true;
        }

        // ── scroll-to-zoom ─────────────────────────────────────────────────
        if let Some((_sx, sy)) = window.get_scroll_wheel() {
            if sy != 0.0 {
                if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
                    let mx = mx as f64;
                    let my = my as f64;
                    // World-space point under the cursor before zooming
                    let wx = (mx - width  as f64 * 0.5) / zoom + cx;
                    let wy = (my - height as f64 * 0.5) / zoom + cy;

                    zoom *= if sy > 0.0 { 1.15 } else { 1.0 / 1.15 };

                    // Re-anchor: same world point stays under the cursor after zoom
                    cx = wx - (mx - width  as f64 * 0.5) / zoom;
                    cy = wy - (my - height as f64 * 0.5) / zoom;
                    dirty = true;
                }
            }
        }

        // ── pan: WASD + arrow keys  (10 pixels per frame, zoom-independent) ─
        let pan = 10.0 / zoom;
        if window.is_key_down(Key::Left)  || window.is_key_down(Key::A) { cx -= pan; dirty = true; }
        if window.is_key_down(Key::Right) || window.is_key_down(Key::D) { cx += pan; dirty = true; }
        if window.is_key_down(Key::Up)    || window.is_key_down(Key::W) { cy -= pan; dirty = true; }
        if window.is_key_down(Key::Down)  || window.is_key_down(Key::S) { cy += pan; dirty = true; }

        // ── GPU render (only when the view has changed) ────────────────────
        if dirty {
            // Scale iteration depth with zoom so detail never washes out.
            // log10(300) ≈ 2.5  →  min-clamped to 256
            // log10(1e12) = 12  →  ~700 iterations
            let max_iter = ((100.0 + zoom.log10() * 50.0) as u32).clamp(256, 1536);

            let (cx_hi, cx_lo) = split_df(cx);
            let (cy_hi, cy_lo) = split_df(cy);

            gpu.render(
                &mut buf,
                &GpuParams {
                    cx_hi, cx_lo,
                    cy_hi, cy_lo,
                    zoom:     zoom as f32,
                    width:    width  as u32,
                    height:   height as u32,
                    max_iter,
                },
                width,
                height,
            );
            dirty = false;

            window.set_title(&format!(
                "Mandelbrot  x={:+.9e}  y={:+.9e}  zoom={:.3e}  iter={}  \
                 [WASD/↑↓←→ pan | scroll zoom | Esc quit]",
                cx, cy, zoom, max_iter,
            ));
        }

        window.update_with_buffer(&buf, width, height).unwrap();
    }
}
