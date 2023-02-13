#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

#include <emscripten/emscripten.h>

#ifdef __cplusplus
extern "C" {
#endif
    EMSCRIPTEN_KEEPALIVE
    float test_linear(char* modelbuf, size_t bufsize, float in)
    {
        std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer(modelbuf, bufsize, nullptr);
        if(!model){
            printf("Failed to mmap model\n");
            return 0;
        }
        tflite::ops::builtin::BuiltinOpResolver resolver;
        std::unique_ptr<tflite::Interpreter> interpreter;
        tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
        interpreter->AllocateTensors();

        float* input = interpreter->typed_input_tensor<float>(0);
        *input = in;
        interpreter->Invoke();

        float* output = interpreter->typed_output_tensor<float>(0);
        printf("Result is: %f\n", *output);
        return (*output);

    }

#ifdef __cplusplus
}
#endif
