use crate::{PointResult, StringResult};
use rayon::prelude::*;
use std::arch::x86_64::{
    __m256i, _mm256_loadu_si256, _mm256_set_epi8, _mm256_shuffle_epi8, _mm256_storeu_si256,
};
use std::error::Error;
use std::ffi::{c_char, c_void};
use std::pin::Pin;
use std::ptr;
use windows_capture::capture::{CaptureControl, Context, GraphicsCaptureApiHandler};
use windows_capture::encoder::{
    AudioSettingsBuilder, ContainerSettingsBuilder, VideoEncoder, VideoSettingsBuilder,
};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
use windows_capture::monitor::Monitor;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};

type CaptureError = Box<dyn Error + Send + Sync>;

/// 帧回调函数, 注意: windows 使用的帧数据通常是 bottom-to-top(像素顺序从下到上)
/// data 为视频帧的字节, 长度为 4 * width * height
/// width 为宽度
/// height 为高度
pub type OnFrame = extern "C" fn(*mut u8, i32, i32);

/// 枚举显示器列表
#[repr(C)]
pub struct MonitorItem {
    pub index: usize,
    pub width: u32,
    pub height: u32,
    pub name: StringResult,
}

/// 枚举显示器列表的打包结构体
#[repr(C)]
pub struct MonitorItemList {
    pub ptr: *const MonitorItem,
    pub len: usize,
}

/// 配置, 宽度高度仅作为, 使用屏幕尺寸
/// `index` 通过枚举返回的 index 字段
/// `audio` 是否记录声音, `true` 记录
/// `path` 录屏保存路径
/// `cursor_capture` 是否捕获光标, `true` 为捕获
/// `border_draw` 是否显示屏幕边框, `true` 为显示
/// `color_format_rgb8` 是否使用 rgba 色彩(windows 默认是 bgra), `true` 为 rgba
/// `trans_color` 回调数据是否处理反转颜色数据中的 r / g 通道 (注意, 此处理消耗大量cpu性能), `true` 反转
/// `vertical_flip` 回调数据是否反转Y轴(注意, 反转需要 cpu 操作所以尽量使 canvas 绘制操作Y轴反转) `true` 反转
/// `shared_memory` 是否开启共享内存, `true` 开启
#[repr(C)]
pub struct CaptureConfigWithoutCallback {
    pub index: usize,
    pub audio: bool,
    pub path: *const c_char,
    pub width: u32,
    pub height: u32,
    pub cursor_capture: bool,
    pub border_draw: bool,
    pub color_format_rgb8: bool,
    pub trans_color: bool,
    pub vertical_flip: bool,
    pub shared_memory: bool,
}

/// 跟上面一样, 只是多了 on_frame
/// `on_frame` 回调函数
#[repr(C)]
pub struct CaptureConfigWithCallback {
    pub index: usize,
    pub audio: bool,
    pub path: *const c_char,
    pub width: u32,
    pub height: u32,
    pub cursor_capture: bool,
    pub border_draw: bool,
    pub color_format_rgb8: bool,
    pub trans_color: bool,
    pub vertical_flip: bool,
    pub shared_memory: bool,
    pub on_frame: OnFrame,
}

/// 配置, 宽度高度仅作为, 使用屏幕尺寸
/// `cursor_capture` 是否捕获光标
/// `border_draw` 是否显示屏幕边框
/// `color_format_rgb8` 是否使用 rgba 色彩(windows 默认是 bgra)
/// `trans_color` 回调数据是否处理反转颜色数据中的 r / g 通道 (注意, 此处理消耗大量cpu性能)
/// `vertical_flip` 回调数据是否反转Y轴(注意, 反转需要 cpu 操作所以尽量使 canvas 绘制操作Y轴反转)
/// `shared_memory` 是否开启共享内存
/// `on_frame` 回调函数, 可以是空, 类型为 `OnFrame`
pub struct CaptureConfig {
    pub audio: bool,
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub cursor_capture: bool,
    pub border_draw: bool,
    pub color_format_rgb8: bool,
    pub trans_color: bool,
    pub vertical_flip: bool,
    pub shared_memory: bool,
    pub on_frame: Option<OnFrame>,
}

impl From<CaptureConfigWithCallback> for CaptureConfig {
    fn from(config: CaptureConfigWithCallback) -> Self {
        Self {
            audio: config.audio,
            path: c_str_to_string(config.path),
            width: config.width,
            height: config.height,
            cursor_capture: config.cursor_capture,
            border_draw: config.border_draw,
            color_format_rgb8: config.color_format_rgb8,
            trans_color: config.trans_color,
            vertical_flip: config.vertical_flip,
            shared_memory: config.shared_memory,
            on_frame: Some(config.on_frame),
        }
    }
}

impl From<CaptureConfigWithoutCallback> for CaptureConfig {
    fn from(config: CaptureConfigWithoutCallback) -> Self {
        Self {
            audio: config.audio,
            path: c_str_to_string(config.path),
            width: config.width,
            height: config.height,
            cursor_capture: config.cursor_capture,
            border_draw: config.border_draw,
            color_format_rgb8: config.color_format_rgb8,
            trans_color: config.trans_color,
            vertical_flip: config.vertical_flip,
            shared_memory: config.shared_memory,
            on_frame: None,
        }
    }
}

fn c_str_to_string(c_str: *const c_char) -> String {
    if c_str.is_null() {
        return "".to_string();
    }
    unsafe {
        let c_str = std::ffi::CStr::from_ptr(c_str);
        c_str.to_string_lossy().to_string()
    }
}

#[repr(C)]
pub struct Capture {
    pub width: u32,
    pub height: u32,
    pub trans_color: bool,
    pub vertical_flip: bool,
    pub video_encoder: Option<VideoEncoder>,
    pub on_frame: Option<OnFrame>,
    pub cache: Option<Vec<u8>>,
}

unsafe impl Send for Capture {}
unsafe impl Sync for Capture {}

impl Capture {
    pub fn on_frame(&mut self, frame: &mut Frame) -> Result<(), CaptureError> {
        if self.on_frame.is_none() && self.cache.is_none() {
            return Ok(());
        }
        let mut buffer = frame.buffer()?;
        let data = buffer.as_raw_buffer();
        if self.trans_color {
            trans_color_avx2(data)
        }
        if self.vertical_flip {
            flip_vertical(data, self.width as usize, self.height as usize);
        }

        if let Some(cache) = &mut self.cache {
            let original_ptr = data.as_ptr();
            let cache_ptr = cache.as_mut_ptr();
            unsafe { ptr::copy_nonoverlapping(original_ptr, cache_ptr, data.len()) }
        }

        if let Some(on_frame) = self.on_frame {
            on_frame(data.as_mut_ptr(), self.width as i32, self.height as i32);
        }
        Ok(())
    }
}
impl GraphicsCaptureApiHandler for Capture {
    type Flags = CaptureConfig;
    type Error = CaptureError;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        let width = ctx.flags.width;
        let height = ctx.flags.height;
        let audio = ctx.flags.audio;
        let path = ctx.flags.path;
        let video_encoder = VideoEncoder::new(
            VideoSettingsBuilder::new(width, height),
            AudioSettingsBuilder::default().disabled(!audio),
            ContainerSettingsBuilder::default(),
            path,
        )?;
        let video_encoder = Some(video_encoder);
        let trans_color = ctx.flags.trans_color;
        let vertical_flip = ctx.flags.vertical_flip;
        let on_frame = ctx.flags.on_frame;
        let cache = if ctx.flags.shared_memory {
            Some(vec![0; width as usize * height as usize * 4])
        } else {
            None
        };

        Ok(Self {
            width,
            height,
            trans_color,
            video_encoder,
            vertical_flip,
            on_frame,
            cache,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        if let Some(video_encoder) = self.video_encoder.as_mut() {
            video_encoder.send_frame(frame)?;
        }
        self.on_frame(frame)?;
        Ok(())
    }
}

/// 枚举显示器列表
#[unsafe(no_mangle)]
pub extern "C" fn enumerate_monitor() -> MonitorItemList {
    if let Ok(monitors) = Monitor::enumerate() {
        let mut list = vec![];
        for m in &monitors {
            let width = m.width().unwrap_or_default();
            let height = m.height().unwrap_or_default();
            match (m.index(), m.device_name()) {
                (Ok(index), Ok(name)) => list.push(MonitorItem {
                    index,
                    width,
                    height,
                    name: StringResult::new(name),
                }),
                _ => continue,
            };
        }
        let list = list.into_boxed_slice();
        let ptr = list.as_ptr();
        let len = list.len();
        std::mem::forget(list);
        MonitorItemList { ptr, len }
    } else {
        MonitorItemList {
            ptr: ptr::null(),
            len: 0,
        }
    }
}

/// cpp 那边记得释放
#[unsafe(no_mangle)]
pub extern "C" fn free_enumerate_monitor(list: MonitorItemList) {
    if !list.ptr.is_null() {
        unsafe {
            let _ = std::slice::from_raw_parts(list.ptr, list.len);
        }
    }
}

/// 开始采集, 使用回调函数
#[unsafe(no_mangle)]
pub extern "C" fn start_capture_with_callback(config: CaptureConfigWithCallback) -> PointResult {
    let monitor_index = config.index;
    start_capture(config.into(), monitor_index)
}

/// 开始采集, **不使用** 回调函数
#[unsafe(no_mangle)]
pub extern "C" fn start_capture_without_callback(
    config: CaptureConfigWithoutCallback,
) -> PointResult {
    let monitor_index = config.index;
    start_capture(config.into(), monitor_index)
}

fn start_capture(mut config: CaptureConfig, monitor_index: usize) -> PointResult {
    let monitor = if let Ok(mut monitors) = Monitor::enumerate() {
        let n = monitors
            .iter()
            .position(|m| matches!(m.index(), Ok(v) if v == monitor_index));
        if let Some(n) = n {
            monitors.remove(n)
        } else {
            return PointResult::error("no monitor found");
        }
    } else {
        return PointResult::error("no monitor found");
    };
    let width = monitor.width().unwrap_or_default();
    let height = monitor.height().unwrap_or_default();
    config.width = width;
    config.height = height;

    let enable_cursor = if config.cursor_capture {
        CursorCaptureSettings::WithCursor
    } else {
        CursorCaptureSettings::WithoutCursor
    };

    let enable_border = if config.border_draw {
        DrawBorderSettings::WithBorder
    } else {
        DrawBorderSettings::WithoutBorder
    };

    let enable_rgba = if config.color_format_rgb8 {
        ColorFormat::Rgba8
    } else {
        ColorFormat::Bgra8
    };
    let settings = Settings::new(
        monitor,
        enable_cursor,
        enable_border,
        SecondaryWindowSettings::Default,
        MinimumUpdateIntervalSettings::Default,
        DirtyRegionSettings::Default,
        enable_rgba,
        config.into(),
    );
    match Capture::start_free_threaded(settings) {
        Ok(data) => PointResult::create(data),
        Err(err) => PointResult::error(err.to_string()),
    }
}
type SelfCaptureControl = CaptureControl<Capture, CaptureError>;

/// 获取共享内存指针
#[unsafe(no_mangle)]
pub extern "C" fn get_shared_memory_ptr(ptr: *mut c_void) -> PointResult {
    let capture = match get_capture_handle(ptr) {
        Some(capture) => capture,
        None => return PointResult::error("point error"),
    };

    let capture = capture.callback();
    let obj = capture.lock();
    if let Some(ref cache) = obj.cache {
        let pinned: Pin<&Vec<u8>> = Pin::new(cache);
        PointResult::create_ptr(pinned.as_ptr() as *mut c_void)
    } else {
        PointResult::null()
    }
}

/// 记得释放控制器
#[unsafe(no_mangle)]
pub extern "C" fn stop_and_free_capture(ptr: *mut c_void) {
    let ptr = ptr as *mut SelfCaptureControl;
    let capture_control = unsafe { Box::from_raw(ptr) };
    let capture = capture_control.callback();
    let mut capture_mutex = capture.lock();
    if let Some(video_encoder) = capture_mutex.video_encoder.take() {
        _ = video_encoder.finish();
    }
    if !capture_control.is_finished() {
        _ = capture_control.stop();
    }
}

#[inline(always)]
fn get_capture_handle(ptr: *mut c_void) -> Option<&'static mut SelfCaptureControl> {
    unsafe { (ptr as *mut SelfCaptureControl).as_mut() }
}
/*
pub fn trans_color(data: &mut [u8]) {
    // 转换 rgba <-> bgra
    for chunk in data.chunks_exact_mut(4) {
        chunk.swap(0, 2); // R <-> B
    }
}
*/

/// 颜色通道转换（RGBA <-> BGRA），使用 AVX2 + 多线程
pub fn trans_color_avx2(data: &mut [u8]) {
    let chunk_size = 32 * 1024; // 每线程处理 32KB
    data.par_chunks_mut(chunk_size)
        .for_each(|chunk| unsafe { trans_color_avx2_step(chunk) });
}

/// AVX2 实现每像素 R/B 通道交换
#[target_feature(enable = "avx2")]
fn trans_color_avx2_step(data: &mut [u8]) {
    let len = data.len();
    let ptr = data.as_mut_ptr();

    unsafe {
        let mut i = 0;
        while i + 31 < len {
            let p = ptr.add(i) as *mut __m256i;
            let pixels = _mm256_loadu_si256(p);
            let swapped = swap_r_b_avx2(pixels);
            _mm256_storeu_si256(p, swapped);
            i += 32;
        }

        // 处理剩余未对齐部分
        while i + 3 < len {
            let r = ptr.add(i);
            let b = ptr.add(i + 2);
            ptr::swap(r, b);
            i += 4;
        }
    }
}

#[target_feature(enable = "avx2")]
fn swap_r_b_avx2(vec: __m256i) -> __m256i {
    unsafe {
        let mask = _mm256_set_epi8(
            31, 28, 29, 30, // Pixel 7
            27, 24, 25, 26, // Pixel 6
            23, 20, 21, 22, // Pixel 5
            19, 16, 17, 18, // Pixel 4
            15, 12, 13, 14, // Pixel 3
            11, 8, 9, 10, // Pixel 2
            7, 4, 5, 6, // Pixel 1
            3, 0, 1, 2, // Pixel 0
        );
        _mm256_shuffle_epi8(vec, mask)
    }
}

pub fn flip_vertical(data: &mut [u8], width: usize, height: usize) {
    let stride = width * 4;
    let half = height / 2;
    let (top_rows, bottom_rows) = data.split_at_mut(half * stride);

    bottom_rows
        .par_chunks_mut(stride)
        .rev()
        .zip(top_rows.par_chunks_mut(stride))
        .for_each(|(bottom_row, top_row)| {
            top_row.swap_with_slice(bottom_row);
        });
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    static A: AtomicU64 = AtomicU64::new(0);

    fn get_time() -> u64 {
        let now = std::time::SystemTime::now();
        let millis = now.duration_since(std::time::UNIX_EPOCH).unwrap();
        millis.as_millis() as u64
    }

    fn get_diff() -> u64 {
        let time = get_time();
        let before = A.swap(time, Ordering::Relaxed);
        time - before
    }

    use std::ptr;
    use std::time::Instant;

    use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
    use windows_capture::encoder::{
        AudioSettingsBuilder, ContainerSettingsBuilder, VideoEncoder, VideoSettingsBuilder,
    };
    use windows_capture::frame::Frame;
    use windows_capture::graphics_capture_api::InternalCaptureControl;
    use windows_capture::monitor::Monitor;
    use windows_capture::settings::{
        ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
        MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
    };

    // 处理捕获事件的结构体
    struct Capture {
        // 用于编码帧的视频编码器
        encoder: Option<VideoEncoder>,
        cache: Vec<u8>,
        // 用于测量捕获运行的时间
        start: Instant,
    }

    impl GraphicsCaptureApiHandler for Capture {
        // 从设置中获取值的标志类型
        type Flags = String;

        // `CaptureControl` 和 `start` 函数可能返回的错误类型
        type Error = Box<dyn std::error::Error + Send + Sync>;

        // 创建新实例的函数。标志可以从设置中传递。
        fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
            println!("Created with Flags: {}", ctx.flags);
            let encoder = VideoEncoder::new(
                VideoSettingsBuilder::new(1920, 1080),
                AudioSettingsBuilder::default().disabled(true),
                ContainerSettingsBuilder::default(),
                "video.mp4",
            )?;

            Ok(Self {
                encoder: Some(encoder),
                cache: vec![0; 1920 * 1080 * 4],
                start: Instant::now(),
            })
        }

        // 每当有新帧到达时调用
        fn on_frame_arrived(
            &mut self,
            frame: &mut Frame,
            capture_control: InternalCaptureControl,
        ) -> Result<(), Self::Error> {
            let n = get_diff();
            let time = frame.timestamp();
            println!("{n}: {}", time.Duration);

            /**/
            let mut buffer = frame.buffer()?;
            let _w = buffer.width() as usize;
            let _h = buffer.height() as usize;
            let buf = buffer.as_raw_buffer();
            // super::trans_color_avx2(buf);
            // super::flip_vertical(buf, w, h);
            unsafe { ptr::copy_nonoverlapping(buf.as_ptr(), self.cache.as_mut_ptr(), buf.len()) }
            // 将帧发送到视频编码器
            self.encoder
                .as_mut()
                .unwrap()
                .send_frame_buffer(buf, time.Duration)?;

            // 停止
            if self.start.elapsed().as_secs() >= 10 {
                // 完成编码器并保存视频
                self.encoder.take().unwrap().finish()?;

                capture_control.stop();
            }

            Ok(())
        }

        // 可选的处理器，在捕获项（通常是窗口）关闭时调用
        fn on_closed(&mut self) -> Result<(), Self::Error> {
            println!("Capture session ended");

            Ok(())
        }
    }

    #[test]
    fn it_works() {
        let primary_monitor = if let Ok(mut monitors) = Monitor::enumerate() {
            for m in &monitors {
                let (index, name) = match (m.index(), m.device_name()) {
                    (Ok(index), Ok(name)) => (index, name),
                    _ => continue,
                };
                println!(">> {index}: {name}");
            }
            if monitors.is_empty() {
                return;
            }
            monitors.remove(0)
        } else {
            return;
        };

        let settings = Settings::new(
            // 要捕获的项
            primary_monitor,
            // 光标捕获设置
            CursorCaptureSettings::Default,
            // 绘制边框设置
            DrawBorderSettings::WithoutBorder,
            // 次要窗口设置，如果你想在捕获中包含次要窗口
            SecondaryWindowSettings::Default,
            // 最小更新间隔，如果你想更改帧率限制（默认为60 FPS或16.67毫秒）
            MinimumUpdateIntervalSettings::Default,
            // 脏区域设置
            DirtyRegionSettings::Default,
            // 捕获帧所需的颜色格式
            ColorFormat::Bgra8,
            // 捕获设置的附加标志，将传递给用户定义的 `new` 函数
            "capture".to_string(),
        );

        // 开始捕获并控制当前线程
        // 处理程序 trait 中的错误将在此处结束
        if let Err(e) = Capture::start(settings) {
            println!("{e}");
        }
    }
}
