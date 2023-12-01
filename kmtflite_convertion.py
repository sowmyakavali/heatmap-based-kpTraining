import tensorflow as tf

def default_convertion(input_model):
    model = tf.keras.models.load_model(input_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(input_model.replace(".h5", "_default.tflite"), "wb").write(tflite_model)
    print("Default convertion Done....✌")

def dynamic_quantizaton(input_model):
    model = tf.keras.models.load_model(input_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(input_model.replace(".h5", "_dynamicQuant.tflite"), "wb").write(tflite_model)
    print("Dynamic Quantization Done.......✌")

def float16_quantizaton(input_model):
    model = tf.keras.models.load_model(input_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    open(input_model.replace(".h5", "_float16Quant.tflite"), "wb").write(tflite_model)
    print("Float16 Quantization Done.......✌")

def onnx_convertion(input_model):
    return

if __name__=="__main__":
    
    RESIZE = 96
    channels = 3

    input_model = r"D:\Hand\KPV8_19042023\FinalDataset_96x96\RGB_croppedwristkp_24042023_tf2.10.1.h5"
    
    default_convertion(input_model)
    float16_quantizaton(input_model)