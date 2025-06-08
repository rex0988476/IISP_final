import onnxruntime

# 創建所需要的全域變數
all_faces = None
log_level = 'error'
cpu_cores = None
gpu_threads = None
gpu_vendor = None
providers = onnxruntime.get_available_providers()

# 檢查 'TensorrtExecutionProvider' 是否在提供者列表中。TensorRT 是 NVIDIA 提供的高性能深度學習推理加速庫。
if 'TensorrtExecutionProvider' in providers:
    # 如果 'TensorrtExecutionProvider' 存在於提供者列表中，則從列表中刪除它。這樣做是為了確保 ONNX Runtime 不使用 TensorRT 作為運行時提供者。
    providers.remove('TensorrtExecutionProvider')
