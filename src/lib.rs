mod capture;

use std::ffi::{CString, c_char, c_void};
use std::ptr::{null, null_mut};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use windows_capture::capture::{Context, GraphicsCaptureApiError, GraphicsCaptureApiHandler};
use windows_capture::encoder::{
    AudioSettingsBuilder, ContainerSettingsBuilder, VideoEncoder, VideoSettingsBuilder,
};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::{Error, InternalCaptureControl};
use windows_capture::monitor::Monitor;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};

/// 封装的字符串结果
/// `code`: 返回 code 协议: 0 正常, -1 null, -2 error(带信息), -3 trouble(无法获取异常信息)
/// `data`: 可能为空的字符串指针, 如果 `code == -2` 则 data 为异常信息
#[repr(C)]
pub struct CaptureStringResult {
    pub code: i32,
    pub data: *const c_char,
}

#[unsafe(no_mangle)]
pub extern "C" fn free_string_result(s: CaptureStringResult) {
    free_string(s.data)
}

#[unsafe(no_mangle)]
pub extern "C" fn free_string(s: *const c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s as *mut c_char);
    }
}

impl CaptureStringResult {
    pub fn new<T: Into<Vec<u8>>>(data: T) -> Self {
        let (data, code): (*const c_char, i32) = match CString::new(data) {
            Ok(s) => (s.into_raw(), 0),
            Err(_) => (null(), -3),
        };
        Self { data, code }
    }
    pub fn null() -> Self {
        Self {
            data: null(),
            code: -1,
        }
    }
    pub fn throw<T: Into<Vec<u8>>>(data: T) -> Self {
        let (data, code): (*const c_char, i32) = match CString::new(data) {
            Ok(s) => (s.into_raw(), -1),
            Err(_) => (null(), -3),
        };
        Self { data, code }
    }
}

/// 封装的返回任意类型指针的结果
/// `ptr`: 指针, 具体类型根据调用方法区分, 可能为空
/// `code`: 返回 code 协议: 0 正常, -1 null, -2 error(带信息), -3 trouble(无法获取异常信息)
/// `error`: 异常信息指针, 可能为空
#[repr(C)]
pub struct CapturePointResult {
    pub ptr: *mut c_void,
    pub code: i32,
    pub error: *const c_char,
}

impl CapturePointResult {
    pub fn null() -> Self {
        Self {
            ptr: null_mut(),
            code: -1,
            error: null(),
        }
    }

    pub fn error<T: Into<Vec<u8>>>(message: T) -> Self {
        let (err_str, status): (*const c_char, i32) = match CString::new(message) {
            Ok(s) => (s.into_raw(), -2),
            Err(_) => (null(), -3),
        };
        Self {
            ptr: null_mut(),
            code: status,
            error: err_str,
        }
    }

    pub fn create<T>(any: T) -> Self {
        let result = Box::into_raw(Box::new(any));
        Self {
            ptr: result as *mut c_void,
            code: 0,
            error: null(),
        }
    }

    pub fn create_ptr(ptr: *mut c_void) -> Self {
        Self {
            ptr,
            code: 0,
            error: null(),
        }
    }
}

struct TestCapture {
    encoder: Option<VideoEncoder>,
    start: Instant,
}

impl GraphicsCaptureApiHandler for TestCapture {
    type Flags = (u32, u32);
    type Error = ();

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        let encoder = VideoEncoder::new(
            VideoSettingsBuilder::new(ctx.flags.0, ctx.flags.1),
            AudioSettingsBuilder::default().disabled(true),
            ContainerSettingsBuilder::default(),
            "video.mp4",
        )
        .unwrap();
        Ok(Self {
            encoder: Some(encoder),
            start: Instant::now(),
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        self.encoder.as_mut().unwrap().send_frame(frame).unwrap();
        if self.start.elapsed().as_secs() >= 10 {
            // 完成编码器并保存视频
            self.encoder.take().unwrap().finish().unwrap();
            capture_control.stop();
        }
        Ok(())
    }
}

fn main() {
    static UNSUPPORT_CURSOR:AtomicBool = AtomicBool::new(false);
    static UNSUPPORT_BORDER:AtomicBool = AtomicBool::new(false);

    let mut list = match Monitor::enumerate() {
        Ok(list) => list,
        Err(e) => panic!("{}", e),
    };
    if list.is_empty() {
        panic!("No monitors found.");
    };
    let first_monitor = list.remove(0);
    let w = first_monitor.width().unwrap_or(0);
    let h = first_monitor.height().unwrap_or(0);
    loop {
        let cursor_conf = if UNSUPPORT_CURSOR.load(std::sync::atomic::Ordering::SeqCst) {
            CursorCaptureSettings::Default
        } else {
            CursorCaptureSettings::WithoutCursor
        };

        let border_conf = if UNSUPPORT_BORDER.load(std::sync::atomic::Ordering::SeqCst) {
            DrawBorderSettings::Default
        } else {
            DrawBorderSettings::WithoutBorder
        };
        let settings = Settings::new(
            // 要捕获的项
            first_monitor,
            // 光标捕获设置
            cursor_conf,
            // 绘制边框设置
            border_conf,
            // 次要窗口设置，如果你想在捕获中包含次要窗口
            SecondaryWindowSettings::Default,
            // 最小更新间隔，如果你想更改帧率限制（默认为60 FPS或16.67毫秒）
            MinimumUpdateIntervalSettings::Default,
            // 脏区域设置
            DirtyRegionSettings::Default,
            // 捕获帧所需的颜色格式
            ColorFormat::Bgra8,
            // 捕获设置的附加标志，将传递给用户定义的 `new` 函数
            (w, h),
        );

        let start = TestCapture::start(settings);
        if start.is_ok() {
            break;
        }

        match start {

            Err(GraphicsCaptureApiError::GraphicsCaptureApiError(e)) => match e {
                Error::CursorConfigUnsupported => {
                    UNSUPPORT_CURSOR.store(true, std::sync::atomic::Ordering::SeqCst)
                }
                Error::BorderConfigUnsupported => {
                    UNSUPPORT_BORDER.store(true, std::sync::atomic::Ordering::SeqCst)
                }
                _ => {
                    panic!("{}", e);
                }
            }
            Err(e) => panic!("{:?}", e),
            _ => return,
        }
    }
}
