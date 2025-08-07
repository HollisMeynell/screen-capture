mod capture;

use std::ffi::{CString, c_char, c_void};
use std::ptr::{null, null_mut};

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
