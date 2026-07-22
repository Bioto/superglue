//! Direct `serde_json::Value` ↔ Python object conversion (no `json.loads`/`dumps`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};
use pyo3::IntoPyObject;
use serde_json::{Map, Number, Value};

pub(crate) fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Null => Ok(py.None().into_bound(py).into_any()),
        Value::Bool(b) => Ok(PyBool::new(py, *b).to_owned().into_any()),
        Value::Number(n) => number_to_py(py, n),
        Value::String(s) => {
            let py_str = s.into_pyobject(py)?;
            Ok(py_str.into_any())
        }
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

fn number_to_py<'py>(py: Python<'py>, n: &Number) -> PyResult<Bound<'py, PyAny>> {
    if let Some(i) = n.as_i64() {
        let py_int = i.into_pyobject(py)?;
        return Ok(py_int.into_any());
    }
    if let Some(u) = n.as_u64() {
        let py_int = u.into_pyobject(py)?;
        return Ok(py_int.into_any());
    }
    if let Some(f) = n.as_f64() {
        let py_float = f.into_pyobject(py)?;
        return Ok(py_float.into_any());
    }
    Ok(py.None().into_bound(py).into_any())
}

pub(crate) fn py_to_json(py: Python<'_>, obj: Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(obj.extract()?));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::Number(u.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(
            Number::from_f64(f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
        );
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(py_to_json(py, item)?);
        }
        return Ok(Value::Array(arr));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(py, v)?);
        }
        return Ok(Value::Object(map));
    }
    Err(PyValueError::new_err(format!(
        "tool return value must be JSON-serializable (got {})",
        obj.get_type().name()?
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    #[test]
    fn roundtrip_dict() {
        Python::attach(|py| {
            let value = serde_json::json!({"a": 1, "b": ["x", null]});
            let py_obj = json_to_py(py, &value).unwrap();
            let back = py_to_json(py, py_obj).unwrap();
            assert_eq!(back, value);
        });
    }

    #[test]
    fn tool_callback_dict() {
        Python::attach(|py| {
            let code = c"def echo(args): return {'ok': True, 'n': args['n']}";
            let module = PyModule::from_code(py, code, c"test", c"test").unwrap();
            let echo = module.getattr("echo").unwrap();
            let args = {
                let d = PyDict::new(py);
                d.set_item("n", 42).unwrap();
                d.into_any()
            };
            let result = echo.call1((args,)).unwrap();
            let json = py_to_json(py, result).unwrap();
            assert_eq!(json, serde_json::json!({"ok": true, "n": 42}));
        });
    }
}
