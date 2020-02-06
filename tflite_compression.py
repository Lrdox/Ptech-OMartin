import tensorflow

converter = tensorflow.lite.TFLiteConverter.from_keras_model_file('saved_models/simplenet/pruned_sgd_custom/sparsity90/pruned_simplenet.h5')
converter.optimizations = [tensorflow.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
tflite_model_quant_file = open("test.tflite",'wb')
#tflite_model_quant_file = "./test.tflite"
tflite_model_quant_file.write(tflite_model)