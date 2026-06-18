use jni::objects::JClass;
use jni::sys::jstring;
use jni::JNIEnv;

use crate::jni_base::{ensure_jvm, jstr_from_str};

#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_com_superglue_kt_SuperglueNativeJni_version(
    mut env: JNIEnv,
    _cl: JClass,
) -> jstring {
    let _jvm = ensure_jvm(&mut env);
    jstr_from_str(&mut env, superglue::version())
}
